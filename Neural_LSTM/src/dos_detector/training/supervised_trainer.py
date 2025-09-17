"""Training loop for the supervised sequence detector."""
from __future__ import annotations

import os
import torch
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

from ..utils.progress import progress

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import SequenceDataset, collate_fn
from ..evaluation.metrics import compute_file_metrics, compute_window_metrics
from ..models.supervised import SequenceClassifier
from ..utils.io import ensure_dir, load_dataframe, load_json, save_joblib, save_json
from ..utils.logging import configure_logging, get_logger
from ..utils.seed import seed_everything


class SupervisedTrainer:
    """Train and evaluate the supervised detector."""

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
        self.num_types = len(config.labels.family_mapping)

    def _load_split(self, files: Sequence[str]) -> List[pd.DataFrame]:
        data_frames: List[pd.DataFrame] = []
        for name in files:
            path = self.config.paths.processed_dir / f"{Path(name).stem}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing processed features for {name}")
            data_frames.append(load_dataframe(path))
        return data_frames

    def _resolve_files(self, split: str) -> List[str]:
        entries = [entry["pcap"] for entry in self.manifest.get("frames", [])]
        configured = getattr(self.config.data, f"{split}_files")
        if configured:
            return list(configured)
        if split == "train":
            return entries
        if split == "val":
            return entries
        return []

    def _fit_scaler(self, frames: Sequence[pd.DataFrame]) -> StandardScaler:
        scaler = StandardScaler()
        data = np.concatenate([frame[self.feature_columns].to_numpy(dtype=np.float32) for frame in frames], axis=0)
        scaler.fit(data)
        save_joblib(self.config.paths.scaler_path, scaler)
        return scaler

    def _transform(self, frames: Sequence[pd.DataFrame], scaler: StandardScaler) -> List[pd.DataFrame]:
        transformed: List[pd.DataFrame] = []
        for frame in frames:
            frame = frame.copy()
            frame[self.feature_columns] = scaler.transform(frame[self.feature_columns])
            transformed.append(frame)
        return transformed

    def _class_weights(self, frames: Sequence[pd.DataFrame]) -> torch.Tensor:
        counts = np.zeros(self.num_types, dtype=float)
        for frame in frames:
            values = frame["family_index"].value_counts()
            for index, count in values.items():
                counts[int(index)] += float(count)
        counts[counts == 0] = 1.0
        weights = counts.sum() / counts
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def train(self) -> Dict[str, float]:
        train_files = self._resolve_files("train")
        val_files = self._resolve_files("val")
        train_frames = self._load_split(train_files)
        val_frames = self._load_split(val_files)
        scaler = self._fit_scaler(train_frames)
        train_frames = self._transform(train_frames, scaler)
        val_frames = self._transform(val_frames, scaler)

        train_dataset = SequenceDataset(train_frames, self.feature_columns, self.config.labels.family_mapping, self.config.windowing)
        val_dataset = SequenceDataset(val_frames, self.feature_columns, self.config.labels.family_mapping, self.config.windowing)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.supervised.batch_size,
            shuffle=True,
            num_workers=self.config.training.supervised.num_workers,
            collate_fn=collate_fn,
            pin_memory=True, prefetch_factor=4, persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.supervised.batch_size,
            shuffle=False,
            num_workers=self.config.training.supervised.num_workers,
            collate_fn=collate_fn,
            pin_memory=True, prefetch_factor=4, persistent_workers=True,
        )

        model = SequenceClassifier(
            input_size=len(self.feature_columns),
            num_attack_types=self.num_types,
            config=self.config.model.supervised,
        ).to(self.device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.training.supervised.learning_rate,
            weight_decay=self.config.training.supervised.weight_decay,
        )
        pos_weight = torch.tensor(
            [self.config.training.supervised.bce_pos_weight],
            dtype=torch.float32,
            device=self.device,
        )
        class_weights = self._class_weights(train_frames)

        best_auc = -float("inf")
        best_state: Dict[str, torch.Tensor] | None = None
        patience = self.config.training.supervised.early_stopping_patience
        epochs_without_improvement = 0
        history: List[Dict[str, float]] = []

        for epoch in progress(range(1, self.config.training.supervised.max_epochs + 1), desc="Supervised epochs", unit="ep"):
            train_loss = self._train_epoch(model, train_loader, optimizer, pos_weight, class_weights)
            metrics = self._evaluate(model, val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
            self.logger.info(
                "epoch_end",
                epoch=epoch,
                train_loss=train_loss,
                val_auc_pr=metrics["val_auc_pr"],
                val_auc_roc=metrics["val_auc_roc"],
            )
            if metrics["val_auc_pr"] > best_auc:
                best_auc = metrics["val_auc_pr"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    self.logger.info("early_stop", epoch=epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), self.config.paths.supervised_model_path)
        save_json(self.config.paths.metrics_path, {"supervised_history": history})
        return history[-1] if history else {}

    def _train_epoch(
        self,
        model: SequenceClassifier,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        pos_weight: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> float:
        model.train()
        total_loss = 0.0
        max_batches = self.config.training.supervised.max_train_batches
        for step, batch in enumerate(progress(loader, desc="Batches (supervised)", unit="batch", leave=False)):
            if max_batches is not None and step >= max_batches:
                break
            features = batch["features"].to(self.device)
            binary_labels = batch["binary_labels"].to(self.device)
            family_labels = batch["family_labels"].to(self.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            binary_loss = F.binary_cross_entropy_with_logits(outputs.window_logits, binary_labels, pos_weight=pos_weight)
            type_loss = self._type_loss(outputs.type_logits, family_labels, class_weights)
            loss = binary_loss + self.config.training.supervised.type_loss_weight * type_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config.training.supervised.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        batches = max(1, min(len(loader), (max_batches or len(loader))))
        return total_loss / batches

    def _type_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        flat_logits = logits.view(-1, self.num_types)
        flat_labels = labels.view(-1)
        ce = F.cross_entropy(flat_logits, flat_labels, weight=class_weights, reduction="none")
        gamma = self.config.training.supervised.focal_gamma
        if gamma > 0:
            pt = torch.exp(-ce)
            ce = ((1 - pt) ** gamma) * ce
        return ce.mean()

    def _evaluate(self, model: SequenceClassifier, loader: DataLoader) -> Dict[str, float]:
        model.eval()
        window_scores: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        window_labels: Dict[Tuple[str, int], int] = {}
        file_labels: Dict[str, int] = defaultdict(int)
        with torch.no_grad():
            for step, batch in enumerate(progress(loader, desc="Batches (supervised)", unit="batch", leave=False)):
                if self.config.training.supervised.max_val_batches is not None and step >= self.config.training.supervised.max_val_batches:
                    break
                features = batch["features"].to(self.device)
                binary_labels = batch["binary_labels"].to(self.device)
                outputs = model(features)
                probs = torch.sigmoid(outputs.window_logits).cpu().numpy()
                labels = binary_labels.cpu().numpy()
                for i, meta in enumerate(batch["metadata"]):
                    pcap = meta["pcap"]
                    start_index = meta["start_index"]
                    end_index = meta["end_index"]
                    for offset, window_index in enumerate(range(start_index, end_index + 1)):
                        key = (pcap, window_index)
                        window_scores[key].append(float(probs[i, offset]))
                        window_labels[key] = int(labels[i, offset])
                    file_labels[pcap] = max(file_labels[pcap], int(labels[i].max()))
        if not window_scores:
            return {"val_auc_pr": 0.0, "val_auc_roc": 0.0}
        sorted_keys = sorted(window_scores.keys(), key=lambda item: (item[0], item[1]))
        scores = [float(np.mean(window_scores[key])) for key in sorted_keys]
        labels = [window_labels[key] for key in sorted_keys]
        window_metrics = compute_window_metrics(labels, scores)
        file_scores: Dict[str, List[float]] = defaultdict(list)
        for (pcap, _), score in zip(sorted_keys, scores):
            file_scores[pcap].append(score)
        file_preds = {pcap: int(max(scores) >= self.config.postprocessing.tau_file) for pcap, scores in file_scores.items()}
        files_sorted = sorted(file_scores.keys())
        file_label_list = [file_labels.get(pcap, 0) for pcap in files_sorted]
        file_pred_list = [file_preds.get(pcap, 0) for pcap in files_sorted]
        file_metrics = compute_file_metrics(file_label_list, file_pred_list)
        return {
            "val_auc_pr": window_metrics.auc_pr,
            "val_auc_roc": window_metrics.auc_roc,
            "val_precision": window_metrics.precision,
            "val_recall": window_metrics.recall,
            "val_f1": window_metrics.f1,
            "val_file_precision": file_metrics.precision,
            "val_file_recall": file_metrics.recall,
            "val_file_f1": file_metrics.f1,
        }


__all__ = ["SupervisedTrainer"]
