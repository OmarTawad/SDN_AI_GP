#!/usr/bin/env python3
"""Evaluate the LSTM-based DoS detector on the cached feature windows."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import psutil
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dos_detector.config import load_config
from dos_detector.data.dataset import StreamingSequenceDataset, collate_fn
from dos_detector.models.supervised import SequenceClassifier
from dos_detector.utils import (
    DEFAULT_TORCH_DTYPE,
    configure_cpu_environment,
    resolve_device,
    resolve_precision_mode,
    resolve_project_root,
    resolve_torch_dtype,
    safe_cast_tensor,
)
from dos_detector.utils.io import ensure_dir, load_joblib, load_json, save_compressed_array
from dos_detector.utils.logging import configure_logging, get_logger

ROOT = resolve_project_root()
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
DEFAULT_EVAL_DIR = ROOT / "eval"


def _resolve_files(manifest: dict, config, split: str) -> List[str]:
    configured: Sequence[str] = getattr(config.data, f"{split}_files", []) or []
    if configured:
        return sorted(set(str(path) for path in configured))
    entries = [entry.get("pcap") for entry in manifest.get("frames", []) if entry.get("pcap")]
    if not entries:
        raise FileNotFoundError(f"Manifest has no frames → {config.paths.manifest_path}")
    if split == "test":
        excluded = set(config.data.train_files or []) | set(config.data.val_files or [])
        remaining = [name for name in entries if name not in excluded]
        return sorted(remaining or entries)
    return sorted(entries)


def _build_loader(
    files: Sequence[str],
    config,
    feature_columns: Sequence[str],
    batch_size: int,
    scaler,
    device: torch.device,
) -> DataLoader:
    dataset = StreamingSequenceDataset(
        files=files,
        processed_dir=Path(config.paths.processed_dir),
        feature_columns=feature_columns,
        family_mapping=config.labels.family_mapping,
        windowing=config.windowing,
        chunk_size=50_000,
        scaler=scaler,
        shuffle_files=False,
        seed=config.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    return loader


def _find_checkpoint(config, override: str | None) -> Path:
    def _normalize(path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else (ROOT / candidate)

    candidates: List[Path] = []
    if override:
        candidates.append(_normalize(override))
    candidates.append(Path(config.paths.supervised_model_path))

    search_dirs = {Path(config.paths.models_dir), Path(config.paths.models_dir) / "checkpoints", ROOT / "models", ROOT / "checkpoints", ROOT / "release"}
    preferred_names = ("model_best.pt", "supervised.pt", "best.pt")
    for directory in sorted(d for d in search_dirs if d.exists()):
        for name in preferred_names:
            path = directory / name
            if path.is_file():
                candidates.append(path)
        pt_files: List[Path] = []
        for pattern in ("*.pt", "*.pth"):
            pt_files.extend(directory.glob(pattern))
        pt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(pt_files)

    seen = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    raise FileNotFoundError("Unable to locate a supervised checkpoint under models/ or checkpoints/.")


def _load_model(
    config,
    checkpoint: Path,
    feature_columns: Sequence[str],
    device: torch.device,
    torch_dtype: torch.dtype,
) -> SequenceClassifier:
    model = SequenceClassifier(
        input_size=len(feature_columns),
        num_attack_types=len(config.labels.family_mapping),
        config=config.model.supervised,
    )
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
    model.load_state_dict(state)
    return model.to(device=device, dtype=torch_dtype or DEFAULT_TORCH_DTYPE).eval()


def _evaluate(
    model: SequenceClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    torch_dtype: torch.dtype,
) -> tuple[dict, np.ndarray, str]:
    y_true: List[np.ndarray] = []
    y_prob: List[np.ndarray] = []
    sequences = 0
    with torch.inference_mode():
        for batch in loader:
            features = safe_cast_tensor(batch["features"], torch_dtype or DEFAULT_TORCH_DTYPE).to(device, non_blocking=True)
            labels = safe_cast_tensor(batch["binary_labels"], torch_dtype or DEFAULT_TORCH_DTYPE).to(device, non_blocking=True)
            outputs = model(features)
            probs = torch.sigmoid(outputs.window_logits).cpu().numpy()
            y_prob.append(probs.reshape(-1))
            y_true.append(labels.cpu().numpy().reshape(-1))
            sequences += features.shape[0]
            del batch, features, labels, outputs, probs
            gc.collect()
    if not y_true:
        raise RuntimeError("No evaluation batches produced predictions.")
    y_true_flat = np.concatenate(y_true).astype(int)
    y_prob_flat = np.concatenate(y_prob)
    y_pred = (y_prob_flat >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true_flat, y_pred)),
        "precision": float(precision_score(y_true_flat, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_flat, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_flat, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "num_windows": int(len(y_true_flat)),
        "num_sequences": int(sequences),
        "positives": int(y_true_flat.sum()),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_flat, y_prob_flat))
    except ValueError:
        metrics["roc_auc"] = None
    cm = confusion_matrix(y_true_flat, y_pred, labels=[0, 1])
    report = classification_report(
        y_true_flat,
        y_pred,
        labels=[0, 1],
        target_names=["normal", "attack"],
        zero_division=0,
    )
    return metrics, cm, report


def main() -> None:
    device = resolve_device()
    if device.type == "cpu":
        configure_cpu_environment(threads=2, interop_threads=2)
    parser = argparse.ArgumentParser(description="Evaluate the Neural LSTM DoS detector.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR, help="Output directory for metrics/logs")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Window-level decision threshold")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Dataset split to evaluate")
    args = parser.parse_args()

    ensure_dir(args.eval_dir)
    configure_logging(log_file=args.eval_dir / "log.txt")
    logger = get_logger(__name__)

    config = load_config(args.config)
    manifest = load_json(config.paths.manifest_path)
    feature_columns = manifest.get("feature_columns", [])
    if not feature_columns:
        raise ValueError(f"No feature columns found in manifest → {config.paths.manifest_path}")
    files = _resolve_files(manifest, config, args.split)
    if not files:
        raise ValueError(f"No files resolved for split='{args.split}'. Check configs/config.yaml.")

    scaler = load_joblib(config.paths.scaler_path)
    loader = _build_loader(files, config, feature_columns, args.batch_size, scaler, device)

    checkpoint = _find_checkpoint(config, args.checkpoint)
    torch_dtype, _ = resolve_precision_mode(getattr(config.training.supervised, "precision_mode", None))
    torch_dtype = resolve_torch_dtype(device, torch_dtype)
    model = _load_model(config, checkpoint, feature_columns, device, torch_dtype)

    process = psutil.Process()
    process.cpu_percent(interval=None)
    start = time.time()
    metrics, cm, report = _evaluate(model, loader, device, args.threshold, torch_dtype)
    elapsed = time.time() - start
    cpu_usage = process.cpu_percent(interval=None)
    mem_gb = process.memory_info().rss / (1024**3)
    metrics.update(
        {
            "split": args.split,
            "checkpoint": str(checkpoint),
            "elapsed_seconds": elapsed,
            "cpu_percent": cpu_usage,
            "memory_gb": mem_gb,
        }
    )

    metrics_path = args.eval_dir / "metrics.json"
    np.save(args.eval_dir / "confusion_matrix.npy", cm)
    save_compressed_array(args.eval_dir / "confusion_matrix_compressed.npz", confusion=cm)
    report_path = args.eval_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("evaluation_complete", **metrics)
    logger.info("classification_report", report=report)

    print(json.dumps(metrics, indent=2))
    print(f"Classification report → {report_path}")
    print(f"✅ Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
