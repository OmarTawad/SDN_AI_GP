"""Training loop for the supervised sequence detector."""
from __future__ import annotations

import gc
import os
import time
from collections import defaultdict, deque
from contextlib import suppress
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import StreamingSequenceDataset, collate_fn
from ..evaluation.metrics import compute_file_metrics, compute_window_metrics
from ..models.supervised import SequenceClassifier
from ..utils import (
    DEFAULT_NUMPY_DTYPE,
    DEFAULT_TORCH_DTYPE,
    configure_cpu_environment,
    resolve_device,
    resolve_precision_mode,
    resolve_torch_dtype,
    resolve_project_root,
    sanitize_numpy,
)
from ..utils.io import (
    ensure_dir,
    load_json,
    resolve_processed_frame,
    save_compressed_array,
    save_joblib,
    save_json,
    stream_dataframe,
)
from ..utils.logging import configure_logging, get_logger
from ..utils.progress import progress
from ..utils.seed import seed_everything

DEVICE = resolve_device()
if DEVICE.type == "cpu":
    configure_cpu_environment()
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

PROJECT_ROOT = resolve_project_root()
EVAL_DIR = PROJECT_ROOT / "eval"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SCALER_CHUNK_SIZE = 50_000
MAX_CHECKPOINTS = 3
LOGIT_CLIP = 20.0


class SupervisedTrainer:
    """Train and evaluate the supervised detector."""

    def __init__(self, config: Config) -> None:
        self.config = config
        ensure_dir(Path(self.config.paths.models_dir))
        ensure_dir(EVAL_DIR)
        ensure_dir(CHECKPOINT_DIR)
        configure_logging(log_file=EVAL_DIR / "log.txt")
        self.device = DEVICE
        seed_everything(config.seed, deterministic=self.device.type == "cpu")
        self.logger = get_logger(__name__)
        self.manifest = load_json(config.paths.manifest_path)
        self.feature_columns: Sequence[str] = self.manifest.get("feature_columns", [])
        if not self.feature_columns:
            raise ValueError("Feature manifest is empty. Run extract-features first.")
        self.chunk_size = getattr(config.data, "chunk_size", SCALER_CHUNK_SIZE)
        self.proc = psutil.Process(os.getpid())
        self.eval_dir = EVAL_DIR
        self.checkpoint_dir = CHECKPOINT_DIR
        self.checkpoints: deque[Path] = deque()
        self.processed_dir = Path(self.config.paths.processed_dir)
        self.precision_mode = (getattr(self.config.training.supervised, "precision_mode", "") or "").lower()
        self.torch_dtype, self.numpy_dtype = resolve_precision_mode(self.precision_mode)
        self.torch_dtype = resolve_torch_dtype(self.device, self.torch_dtype)
        self.use_amp = self.precision_mode == "autocast" and self.device.type == "cuda"
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _resolve_files(self, split: str) -> List[str]:
        entries = [entry["pcap"] for entry in self.manifest.get("frames", [])]
        configured = getattr(self.config.data, f"{split}_files")
        if configured:
            return list(configured)
        if split in {"train", "val"}:
            return entries
        return []

    def _iter_feature_chunks(self, files: Sequence[str]) -> Iterator[np.ndarray]:
        for name in files:
            path = resolve_processed_frame(self.processed_dir, name)
            for chunk in stream_dataframe(path, columns=self.feature_columns, chunk_size=self.chunk_size):
                if chunk.empty:
                    continue
                block = chunk.to_numpy(dtype=self.numpy_dtype or DEFAULT_NUMPY_DTYPE, copy=True)
                block = sanitize_numpy(block)
                yield block
                del chunk
                gc.collect()

    def _fit_scaler(self, files: Sequence[str]) -> StandardScaler:
        scaler = StandardScaler()
        for block in self._iter_feature_chunks(files):
            scaler.partial_fit(block)
        save_joblib(self.config.paths.scaler_path, scaler)
        return scaler

    def _build_loader(
        self,
        files: Sequence[str],
        scaler: StandardScaler,
        shuffle_files: bool,
    ) -> DataLoader:
        dataset = StreamingSequenceDataset(
            files=files,
            processed_dir=self.processed_dir,
            feature_columns=self.feature_columns,
            family_mapping=self.config.labels.family_mapping,
            windowing=self.config.windowing,
            chunk_size=self.chunk_size,
            scaler=scaler,
            shuffle_files=shuffle_files,
            seed=self.config.seed,
        )
        workers = 2
        return DataLoader(
            dataset,
            batch_size=self.config.training.supervised.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=collate_fn,
        )

    def train(self) -> Dict[str, float]:
        train_files = self._resolve_files("train")
        val_files = self._resolve_files("val")
        if not train_files or not val_files:
            raise RuntimeError("Training/validation splits are empty. Check configs/config.yaml.")
        scaler = self._fit_scaler(train_files)
        train_loader = self._build_loader(train_files, scaler, shuffle_files=True)
        val_loader = self._build_loader(val_files, scaler, shuffle_files=False)

        model = SequenceClassifier(
            input_size=len(self.feature_columns),
            num_attack_types=len(self.config.labels.family_mapping),
            config=self.config.model.supervised,
        ).to(device=self.device, dtype=self.torch_dtype or DEFAULT_TORCH_DTYPE)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.training.supervised.learning_rate,
            weight_decay=self.config.training.supervised.weight_decay,
        )
        pos_weight = torch.tensor(
            [self.config.training.supervised.bce_pos_weight],
            dtype=self.torch_dtype or DEFAULT_TORCH_DTYPE,
            device=self.device,
        )

        best_auc = -float("inf")
        best_state: Dict[str, torch.Tensor] | None = None
        patience = self.config.training.supervised.early_stopping_patience
        epochs_without_improvement = 0
        history: List[Dict[str, float]] = []

        for epoch in progress(range(1, self.config.training.supervised.max_epochs + 1), desc="Supervised epochs", unit="ep"):
            epoch_start = time.time()
            self.proc.cpu_percent(interval=None)  # reset CPU stats window
            train_loss = self._train_epoch(model, train_loader, optimizer, pos_weight)
            metrics, _ = self._evaluate(model, val_loader, collect_confusion=False)
            elapsed = time.time() - epoch_start
            cpu_usage = self.proc.cpu_percent(interval=None)
            mem_gb = self.proc.memory_info().rss / (1024**3)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                **metrics,
                "epoch_seconds": elapsed,
                "cpu_percent": cpu_usage,
                "memory_gb": mem_gb,
            }
            history.append(record)
            self.logger.info("epoch_end", **record)
            if metrics["val_auc_pr"] > best_auc:
                best_auc = metrics["val_auc_pr"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                self.logger.info("early_stop", epoch=epoch)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._flush_model(Path(self.config.paths.supervised_model_path), model.state_dict())
        save_json(self.config.paths.metrics_path, {"supervised_history": history})

        final_metrics, cm = self._evaluate(model, val_loader, collect_confusion=True)
        final_metrics["checkpoint"] = str(self.config.paths.supervised_model_path)
        final_metrics["split"] = "val"
        self._persist_eval_outputs(final_metrics, cm)
        return final_metrics

    def _train_epoch(
        self,
        model: SequenceClassifier,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        pos_weight: torch.Tensor,
    ) -> float:
        model.train()
        total_loss = 0.0
        max_batches = self.config.training.supervised.max_train_batches
        steps = 0
        for step, batch in enumerate(progress(loader, desc="Batches (supervised)", unit="batch", leave=False)):
            if max_batches is not None and step >= max_batches:
                break
            steps += 1
            features = batch["features"].to(self.device, dtype=self.torch_dtype or DEFAULT_TORCH_DTYPE, non_blocking=True)
            binary_labels = batch["binary_labels"].to(self.device, dtype=self.torch_dtype or DEFAULT_TORCH_DTYPE, non_blocking=True)
            if not torch.isfinite(features).all():
                self.logger.warning("non_finite_features_detected", step=step)
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = model(features)
                logits = torch.nan_to_num(outputs.window_logits, nan=0.0, posinf=LOGIT_CLIP, neginf=-LOGIT_CLIP)
                logits = torch.clamp(logits, min=-LOGIT_CLIP, max=LOGIT_CLIP)
                loss_dtype = torch.float32 if self.use_amp else (self.torch_dtype or DEFAULT_TORCH_DTYPE)
                loss = F.binary_cross_entropy_with_logits(
                    logits.to(loss_dtype),
                    binary_labels.to(loss_dtype),
                    pos_weight=pos_weight.to(loss_dtype),
                )
            if not torch.isfinite(loss):
                self.logger.warning("non_finite_loss_detected", step=step)
                continue
            if self.grad_scaler.is_enabled():
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.config.training.supervised.grad_clip)
                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.config.training.supervised.grad_clip)
                optimizer.step()
            total_loss += float(loss.detach().cpu())
        steps = max(1, steps)
        return total_loss / steps

    def _evaluate(
        self,
        model: SequenceClassifier,
        loader: DataLoader,
        collect_confusion: bool = False,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        model.eval()
        window_scores: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        window_labels: Dict[Tuple[str, int], int] = {}
        file_labels: Dict[str, int] = defaultdict(int)
        confusion = np.zeros((2, 2), dtype=np.int64)
        with torch.no_grad():
            for step, batch in enumerate(progress(loader, desc="Batches (supervised)", unit="batch", leave=False)):
                if self.config.training.supervised.max_val_batches is not None and step >= self.config.training.supervised.max_val_batches:
                    break
                features = batch["features"].to(self.device, dtype=self.torch_dtype or DEFAULT_TORCH_DTYPE, non_blocking=True)
                binary_labels = batch["binary_labels"].to(self.device, dtype=self.torch_dtype or DEFAULT_TORCH_DTYPE, non_blocking=True)
                if not torch.isfinite(features).all():
                    self.logger.warning("non_finite_features_detected", step=step)
                    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = model(features)
                    logits = torch.nan_to_num(outputs.window_logits, nan=0.0, posinf=LOGIT_CLIP, neginf=-LOGIT_CLIP)
                    logits = torch.clamp(logits, min=-LOGIT_CLIP, max=LOGIT_CLIP)
                probs = torch.sigmoid(logits).cpu().numpy()
                if not np.isfinite(probs).all():
                    self.logger.warning("non_finite_probs_detected", step=step)
                    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
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
            metrics = {"val_auc_pr": 0.0, "val_auc_roc": 0.0}
            return metrics, confusion
        sorted_keys = sorted(window_scores.keys(), key=lambda item: (item[0], item[1]))
        scores_list: List[float] = []
        for key in sorted_keys:
            values = np.array(window_scores[key], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                scores_list.append(0.0)
            else:
                scores_list.append(float(values.mean()))
        scores = np.array(scores_list, dtype=self.numpy_dtype or DEFAULT_NUMPY_DTYPE)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        scores = np.clip(scores, 0.0, 1.0)
        labels = np.array([window_labels[key] for key in sorted_keys], dtype=int)
        window_metrics = compute_window_metrics(labels.tolist(), scores.tolist())
        file_scores: Dict[str, List[float]] = defaultdict(list)
        for (pcap, _), score in zip(sorted_keys, scores):
            file_scores[pcap].append(score)
        file_preds = {
            pcap: int(max(values) >= self.config.postprocessing.tau_file) for pcap, values in file_scores.items()
        }
        files_sorted = sorted(file_scores.keys())
        file_label_list = [file_labels.get(pcap, 0) for pcap in files_sorted]
        file_pred_list = [file_preds.get(pcap, 0) for pcap in files_sorted]
        file_metrics = compute_file_metrics(file_label_list, file_pred_list)
        if collect_confusion:
            threshold = self.config.postprocessing.tau_window
            predictions = (scores >= threshold).astype(int)
            confusion = confusion_matrix(labels, predictions, labels=[0, 1])
        metrics = {
            "val_auc_pr": window_metrics.auc_pr,
            "val_auc_roc": window_metrics.auc_roc,
            "val_precision": window_metrics.precision,
            "val_recall": window_metrics.recall,
            "val_f1": window_metrics.f1,
            "val_file_precision": file_metrics.precision,
            "val_file_recall": file_metrics.recall,
            "val_file_f1": file_metrics.f1,
        }
        return metrics, confusion

    def _persist_eval_outputs(self, metrics: Dict[str, float], confusion: np.ndarray) -> None:
        ensure_dir(self.eval_dir)
        metrics_path = self.eval_dir / "metrics.json"
        save_json(metrics_path, metrics)
        np.save(self.eval_dir / "confusion_matrix.npy", confusion)
        save_compressed_array(self.eval_dir / "confusion_matrix_compressed.npz", confusion=confusion)
        self.logger.info("artifacts_saved", metrics=str(metrics_path))

    def _save_checkpoint(self, state: Dict[str, torch.Tensor], epoch: int, tag: str) -> Path:
        ensure_dir(self.checkpoint_dir)
        checkpoint_path = self.checkpoint_dir / f"supervised_epoch{epoch:03d}_{tag}.pt"
        with checkpoint_path.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        self.checkpoints.append(checkpoint_path)
        while len(self.checkpoints) > MAX_CHECKPOINTS:
            old = self.checkpoints.popleft()
            with suppress(FileNotFoundError):
                old.unlink()
        return checkpoint_path

    def _flush_model(self, path: Path, state: Dict[str, torch.Tensor]) -> None:
        ensure_dir(path.parent)
        with path.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())


__all__ = ["SupervisedTrainer"]
