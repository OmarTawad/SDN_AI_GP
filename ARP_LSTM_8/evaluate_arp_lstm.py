#!/usr/bin/env python3
"""Evaluate the LSTM-based ARP spoofing detector on cached sequences."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from arp_detector.config import load_config  # noqa: E402
from arp_detector.data.dataset import SequenceDataset, collate_fn  # noqa: E402
from arp_detector.models.quantization import convert_module_to_int8, Int8Quantizer  # noqa: E402
from arp_detector.models.supervised import SequenceClassifier  # noqa: E402
from arp_detector.utils.io import ensure_dir, load_dataframe, load_joblib, load_json  # noqa: E402

DEFAULT_CONFIG, DEFAULT_EVAL_DIR = ROOT / "configs" / "config.yaml", ROOT / "eval"

def _resolve_split_files(config, manifest: Dict[str, object], split: str) -> List[str]:
    configured: Sequence[str] = getattr(config.data, f"{split}_files", []) or []
    if configured:
        return list(configured)
    entries = [entry.get("pcap") for entry in manifest.get("frames", []) if entry.get("pcap")]
    if not entries:
        raise ValueError("Manifest is missing PCAP entries. Run feature extraction first.")
    if split != "test":
        return entries
    excluded = {Path(name).name for name in (config.data.train_files or [])}
    excluded |= {Path(name).name for name in (config.data.val_files or [])}
    remaining = [entry for entry in entries if Path(entry).name not in excluded]
    return remaining or entries


def _build_loader(config, files: Sequence[str], feature_columns: Sequence[str], scaler, batch_size: int) -> DataLoader:
    if not files:
        raise ValueError("No files resolved for evaluation. Populate data.*_files in the config.")
    frames = []
    for name in files:
        parquet_path = config.paths.processed_dir / f"{Path(name).stem}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing processed features → {parquet_path}")
        frame = load_dataframe(parquet_path)
        if not frame.empty:
            frame[feature_columns] = scaler.transform(frame[feature_columns])
        frames.append(frame)
    dataset = SequenceDataset(frames, feature_columns, config.labels.family_mapping, config.windowing)
    if len(dataset) == 0:
        raise RuntimeError("Sequence dataset is empty. Adjust the split or window configuration.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.training.supervised.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def _find_checkpoint(config, override: str | None) -> Path:
    def expand(path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()

    candidates: List[Path] = []
    if override:
        candidates.append(expand(override))
    candidates.append(config.paths.supervised_model_path)
    search_dirs = [
        config.paths.models_dir,
        config.paths.models_dir / "checkpoints",
        ROOT / "models",
        ROOT / "checkpoints",
    ]
    for directory in search_dirs:
        if not directory.exists():
            continue
        for name in ("supervised.pt", "model_best.pt", "best.pt"):
            path = directory / name
            if path.is_file():
                candidates.append(path)
        for pattern in ("*.pt", "*.pth"):
            candidates.extend(sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError("Unable to locate a trained checkpoint under models/ or checkpoints/.")


def _load_model(config, checkpoint: Path, feature_columns: Sequence[str], device: torch.device) -> SequenceClassifier:
    model = SequenceClassifier(
        input_size=len(feature_columns),
        num_attack_types=len(config.labels.family_mapping),
        config=config.model.supervised,
    )
    convert_module_to_int8(model, Int8Quantizer())
    state = torch.load(checkpoint, map_location="cpu")
    for key in ("state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    model.load_state_dict(state)
    return model.to(device).eval()


def _evaluate(model: SequenceClassifier, loader: DataLoader, device: torch.device, threshold: float):
    y_true, y_prob = [], []
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["binary_labels"].to(device, non_blocking=True)
            probs = torch.sigmoid(model(features).window_logits).cpu().numpy()
            y_prob.append(probs.reshape(-1))
            y_true.append(labels.cpu().numpy().reshape(-1))
    if not y_true:
        raise RuntimeError("No predictions were produced. Confirm the dataset split is non-empty.")
    truth = np.concatenate(y_true).astype(int)
    prob = np.concatenate(y_prob)
    pred = (prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(truth, pred)),
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "threshold": float(threshold),
        "samples": int(len(truth)),
        "positives": int(truth.sum()),
    }
    try:
        roc_val = float(roc_auc_score(truth, prob))
        metrics["roc_auc"] = None if math.isnan(roc_val) else roc_val
    except ValueError:
        metrics["roc_auc"] = None
    matrix = confusion_matrix(truth, pred, labels=[0, 1])
    report = classification_report(
        truth,
        pred,
        labels=[0, 1],
        target_names=["normal", "attack"],
        zero_division=0,
    )
    return metrics, matrix, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ARP LSTM detector.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override (.pt / .pth)")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR, help="Directory for evaluation artifacts")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold applied to window probabilities")
    args = parser.parse_args()
    ensure_dir(args.eval_dir)
    config = load_config(args.config)
    manifest = load_json(config.paths.manifest_path)
    feature_columns: Sequence[str] = manifest.get("feature_columns", [])
    if not feature_columns:
        raise ValueError(f"No feature columns found in manifest → {config.paths.manifest_path}")
    files = _resolve_split_files(config, manifest, args.split)
    scaler = load_joblib(config.paths.scaler_path)
    loader = _build_loader(config, files, feature_columns, scaler, args.batch_size)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for int8-quantized evaluation.")
    device = torch.device("cuda")
    checkpoint = _find_checkpoint(config, args.checkpoint)
    model = _load_model(config, checkpoint, feature_columns, device)

    start = time.time()
    metrics, matrix, report = _evaluate(model, loader, device, args.threshold)
    pretty_files = sorted({Path(name).name for name in files})
    metrics.update({
        "split": args.split,
        "checkpoint": str(checkpoint),
        "files_evaluated": pretty_files,
        "num_sequences": int(len(loader.dataset)),
        "elapsed_seconds": float(time.time() - start),
        "device": str(device),
    })
    np.save(args.eval_dir / "confusion_matrix.npy", matrix)
    (args.eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log_path = args.eval_dir / "log.txt"
    log_path.write_text(
        f"Split: {args.split}\nCheckpoint: {checkpoint}\nFiles: {', '.join(pretty_files)}\n\nClassification report:\n{report}",
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print(f"\nPer-class report written to {log_path}")
    print("✅ “arp_lstm evaluation complete – metrics saved under arp_lstm/eval/metrics.json”")


if __name__ == "__main__":
    main()
