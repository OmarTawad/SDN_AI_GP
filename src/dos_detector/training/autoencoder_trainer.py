"""Training loop for the LSTM autoencoder."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import SequenceDataset, collate_fn
from ..data.dataset import filter_normal_sequences
from ..models.autoencoder import LSTMAutoencoder
from ..utils.io import (
    ensure_dir,
    load_dataframe,
    load_json,
    save_joblib,
    save_json,
)
from ..utils.logging import configure_logging, get_logger
from ..utils.seed import seed_everything


class AutoencoderTrainer:
    """Train the sequence autoencoder on normal data only."""

    def __init__(self, config: Config) -> None:
        self.config = config
        configure_logging()
        seed_everything(config.seed)
        self.logger = get_logger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ensure_dir(config.paths.models_dir)
        self.manifest = load_json(config.paths.manifest_path)
        self.feature_columns: Sequence[str] = self.manifest.get("feature_columns", [])
        if not self.feature_columns:
            raise ValueError("Feature manifest is empty. Run extract-features first.")

    def _resolve_files(self, split: str) -> List[str]:
        entries = [entry["pcap"] for entry in self.manifest.get("frames", [])]
        configured = getattr(self.config.data, f"{split}_files")
        if configured:
            return list(configured)
        return entries

    def _load_split(self, files: Sequence[str]) -> List[pd.DataFrame]:
        frames: List[pd.DataFrame] = []
        for name in files:
            path = self.config.paths.processed_dir / f"{Path(name).stem}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing processed features for {name}")
            frames.append(load_dataframe(path))
        return frames

    def _fit_scaler(self, frames: Sequence[pd.DataFrame]) -> StandardScaler:
        scaler = StandardScaler()
        data = np.concatenate([frame[self.feature_columns].to_numpy(dtype=np.float32) for frame in frames], axis=0)
        scaler.fit(data)
        save_joblib(self.config.paths.ae_scaler_path, scaler)
        return scaler

    def _transform(self, frames: Sequence[pd.DataFrame], scaler: StandardScaler) -> List[pd.DataFrame]:
        transformed: List[pd.DataFrame] = []
        for frame in frames:
            frame = frame.copy()
            frame[self.feature_columns] = scaler.transform(frame[self.feature_columns])
            transformed.append(frame)
        return transformed

    def train(self) -> Dict[str, float]:
        train_frames = self._load_split(self._resolve_files("train"))
        val_frames = self._load_split(self._resolve_files("val"))
        scaler = self._fit_scaler(train_frames)
        train_frames = self._transform(train_frames, scaler)
        val_frames = self._transform(val_frames, scaler)

        train_dataset = SequenceDataset(train_frames, self.feature_columns, self.config.labels.family_mapping, self.config.windowing)
        val_dataset = SequenceDataset(val_frames, self.feature_columns, self.config.labels.family_mapping, self.config.windowing)
        train_dataset.samples = filter_normal_sequences(train_dataset.samples)
        val_dataset.samples = filter_normal_sequences(val_dataset.samples)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.autoencoder.batch_size,
            shuffle=True,
            num_workers=self.config.training.autoencoder.num_workers,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.autoencoder.batch_size,
            shuffle=False,
            num_workers=self.config.training.autoencoder.num_workers,
            collate_fn=collate_fn,
        )

        model = LSTMAutoencoder(len(self.feature_columns), self.config.model.autoencoder).to(self.device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.training.autoencoder.learning_rate,
            weight_decay=self.config.training.autoencoder.weight_decay,
        )
        best_loss = float("inf")
        best_state: Dict[str, torch.Tensor] | None = None
        patience = self.config.training.autoencoder.early_stopping_patience
        epochs_without_improvement = 0
        history: List[Dict[str, float]] = []

        for epoch in range(1, self.config.training.autoencoder.max_epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer)
            val_loss, stats = self._evaluate(model, val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **stats})
            self.logger.info(
                "ae_epoch_end",
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    self.logger.info("ae_early_stop", epoch=epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), self.config.paths.ae_model_path)
        self._persist_metrics(history)
        return history[-1] if history else {}

    def _train_epoch(
        self,
        model: LSTMAutoencoder,
        loader: DataLoader,
        optimizer: optim.Optimizer,
    ) -> float:
        model.train()
        total_loss = 0.0
        max_batches = self.config.training.autoencoder.max_train_batches
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            features = batch["features"].to(self.device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(features)
            loss = F.mse_loss(reconstruction, features)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.autoencoder.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        batches = max(1, min(len(loader), (max_batches or len(loader))))
        return total_loss / batches

    def _evaluate(self, model: LSTMAutoencoder, loader: DataLoader) -> tuple[float, Dict[str, float]]:
        model.eval()
        losses: List[float] = []
        errors: List[float] = []
        with torch.no_grad():
            for step, batch in enumerate(loader):
                if self.config.training.autoencoder.max_val_batches is not None and step >= self.config.training.autoencoder.max_val_batches:
                    break
                features = batch["features"].to(self.device)
                reconstruction, _ = model(features)
                loss = F.mse_loss(reconstruction, features)
                losses.append(float(loss.cpu()))
                window_errors = LSTMAutoencoder.reconstruction_error(features, reconstruction)
                errors.extend(window_errors.cpu().numpy().reshape(-1).tolist())
        if not losses:
            return 0.0, {"ae_error_mean": 0.0, "ae_error_std": 0.0, "ae_error_p95": 0.0}
        error_array = np.array(errors, dtype=float)
        stats = {
            "ae_error_mean": float(error_array.mean()),
            "ae_error_std": float(error_array.std()),
            "ae_error_p95": float(np.percentile(error_array, 95)),
        }
        return float(np.mean(losses)), stats

    def _persist_metrics(self, history: List[Dict[str, float]]) -> None:
        metrics_path = self.config.paths.metrics_path
        metrics = load_json(metrics_path) if metrics_path.exists() else {}
        metrics["autoencoder_history"] = history
        save_json(metrics_path, metrics)


__all__ = ["AutoencoderTrainer"]
