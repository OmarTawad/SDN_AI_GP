#!/usr/bin/env python3
"""Evaluate the autoencoder anomaly detector on labelled Parquet windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dae.config import Config, load_config  # noqa: E402
from dae.model import build_model  # noqa: E402
from dae.preprocess import apply_log_transform, load_feature_list, load_scaler, replace_invalid  # noqa: E402
from dae.threshold import load_threshold  # noqa: E402
from dae.utils_io import read_parquet_batches  # noqa: E402

try:  # optional dependency for safer Parquet streaming on older CPUs
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None

DEFAULT_EVAL_DIR = ROOT / "eval"
LABEL_CANDIDATES = ("label", "attack", "is_attack", "is_anomaly", "y", "target")
MODEL_FILES = ("model.pt", "model_best.pt", "best_model.pt")


def _resolve(path: str | Path) -> Path: return Path(path) if Path(path).is_absolute() else ROOT / Path(path)


def _prepare_features(df, feature_names: Sequence[str], log_features: Sequence[str], clip_bounds):
    work = replace_invalid(df.copy()).reindex(columns=feature_names, fill_value=0.0)
    work = apply_log_transform(work, log_features)
    lower = {k: float(v) for k, v in clip_bounds.get("lower", {}).items()}
    upper = {k: float(v) for k, v in clip_bounds.get("upper", {}).items()}
    if lower: work = work.clip(lower=lower, axis=1)
    if upper: work = work.clip(upper=upper, axis=1)
    return work.values.astype(np.float32)


def _score_batches(model: torch.nn.Module, array: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    if array.size == 0: return np.empty(0, dtype=np.float32)
    errs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(array), batch_size):
            tensor = torch.from_numpy(array[start : start + batch_size]).to(device)
            recon = model(tensor)
            mse = torch.mean((recon - tensor) ** 2, dim=1).cpu().numpy()
            errs.append(mse.astype(np.float32))
    return np.concatenate(errs) if errs else np.empty(0, dtype=np.float32)


def _artifact_dir(config: Config, override: str | None) -> Path:
    raw = [override, config.get("paths", "artifacts_dir"), ROOT / "data" / "artifacts", ROOT / "artifacts", ROOT / "models", ROOT / "checkpoints"]
    for entry in raw:
        if not entry: continue
        path = _resolve(entry)
        target = path if path.is_dir() else path.parent
        if any((target / name).exists() for name in MODEL_FILES):
            return target
    raise FileNotFoundError("Unable to locate model.pt/model_best.pt in artifacts/checkpoints.")


def _load_artifacts(config: Config, override: str | None, device_str: str):
    device = torch.device(device_str); art_dir = _artifact_dir(config, override)
    meta = json.loads((art_dir / "model_config.json").read_text(encoding="utf-8"))
    model = build_model(meta.get("model", {}), input_dim=int(meta["input_dim"]))
    model_path = next((art_dir / name for name in MODEL_FILES if (art_dir / name).exists()), art_dir / "model.pt")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if state_dict is None: state_dict = checkpoint
    model.load_state_dict(state_dict); model.to(device).eval()
    clip_path = art_dir / "clip_bounds.json"
    clip_bounds = json.loads(clip_path.read_text(encoding="utf-8")) if clip_path.exists() else {"lower": {}, "upper": {}}
    return {
        "dir": art_dir,
        "model": model,
        "device": device,
        "scaler": load_scaler(art_dir / "scaler.pkl"),
        "feature_names": load_feature_list(art_dir / "feature_list.json"),
        "log_features": list(meta.get("log_features", [])),
        "clip_bounds": clip_bounds,
        "threshold": load_threshold(art_dir / "threshold.json"),
    }


def _collect_windows(arg_path: str | None, config: Config) -> List[Path]:
    candidates: List[Path] = []
    targets = [_resolve(arg_path)] if arg_path else []
    windows_dir = config.get("paths", "windows_dir")
    if not targets and windows_dir: targets.append(_resolve(windows_dir))
    for target in targets:
        if target.is_file() and target.suffix == ".parquet":
            candidates.append(target)
        elif target.is_dir():
            candidates.extend(sorted(target.glob("*.parquet")))
    if not candidates: raise FileNotFoundError("No parquet windows found; pass --windows.")
    tokens = ("test", "eval", "mixed", "attack", "validation", "val")
    return sorted(dict.fromkeys(candidates), key=lambda p: (next((i for i, tok in enumerate(tokens) if tok in p.stem.lower()), len(tokens)), p.name))


def _iter_windows(path: Path, rows_per_batch: int):
    if duckdb is not None:
        rel = duckdb.read_parquet(str(path))
        offset = 0
        while True:
            chunk = rel.limit(rows_per_batch, offset=offset).df()
            if chunk.empty:
                break
            offset += len(chunk)
            yield chunk
        return
    yield from read_parquet_batches(path, rows_per_batch)


def _evaluate_dataset(paths: Sequence[Path], artifacts, label_hint: str | None, rows_per_batch: int, batch_size: int, threshold_override: float | None):
    label_name = label_hint
    y_true, scores = [], []
    for path in paths:
        for chunk in _iter_windows(path, rows_per_batch):
            if chunk.empty: continue
            if label_name is None:
                order = ([label_hint] if label_hint else []) + list(LABEL_CANDIDATES)
                label_name = next((col for col in order if col and col in chunk.columns), None)
            if not label_name or label_name not in chunk.columns: continue
            mask = chunk[label_name].notna()
            if not mask.any(): continue
            subset = chunk.loc[mask].reset_index(drop=True); labels = subset[label_name].astype(int).to_numpy()
            features = subset.drop(columns=[label_name], errors="ignore")
            matrix = _prepare_features(features, artifacts["feature_names"], artifacts["log_features"], artifacts["clip_bounds"])
            scaled = artifacts["scaler"].transform(matrix).astype(np.float32)
            errors = _score_batches(artifacts["model"], scaled, artifacts["device"], batch_size)
            if errors.size != labels.size: raise RuntimeError(f"Mismatch between scores and labels in {path}")
            y_true.append(labels)
            scores.append(errors)
    if not y_true: raise RuntimeError("Evaluation found zero labelled samples.")
    y = np.concatenate(y_true).astype(int); err = np.concatenate(scores).astype(np.float32)
    threshold = float(threshold_override if threshold_override is not None else artifacts["threshold"].threshold)
    preds = (err > threshold).astype(int)
    metrics = {
        "samples": int(len(y)), "positives": int(y.sum()), "threshold": threshold,
        "accuracy": float(accuracy_score(y, preds)), "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)), "f1": float(f1_score(y, preds, zero_division=0)),
    }
    metrics["roc_auc"] = float(roc_auc_score(y, err)) if len(np.unique(y)) > 1 else None
    err_stats = {
        "mean": float(np.mean(err)), "std": float(np.std(err)), "median": float(np.median(err)),
        "p95": float(np.quantile(err, 0.95)), "max": float(np.max(err)),
    }
    metrics["errors"] = err_stats
    cm = confusion_matrix(y, preds, labels=[0, 1])
    return metrics, cm, classification_report(y, preds, labels=[0, 1], target_names=["normal", "anomaly"], zero_division=0), err_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the autoencoder on labelled window datasets.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--windows", help="Parquet file or directory with labelled windows")
    parser.add_argument("--artifacts", help="Override artifacts directory (defaults to config paths)")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR), help="Directory to store evaluation artifacts")
    parser.add_argument("--label-column", help="Explicit label column name to use")
    parser.add_argument("--device", default="cpu", help="Torch device for inference")
    parser.add_argument("--batch-size", type=int, default=4096, help="Inference batch size")
    parser.add_argument("--rows-per-batch", type=int, default=50000, help="Parquet rows per streaming batch")
    parser.add_argument("--threshold", type=float, help="Override anomaly threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(_resolve(args.config))
    artifacts = _load_artifacts(config, args.artifacts, args.device)
    windows = _collect_windows(args.windows, config)
    eval_dir = _resolve(args.eval_dir); eval_dir.mkdir(parents=True, exist_ok=True)
    metrics, cm, report, err_stats = _evaluate_dataset(
        windows,
        artifacts,
        args.label_column,
        args.rows_per_batch,
        args.batch_size,
        args.threshold,
    )
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.save(eval_dir / "confusion_matrix.npy", cm)
    log_path = eval_dir / "log.txt"
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Artifacts: {artifacts['dir']}\nThreshold: {metrics['threshold']}\n\nClassification report:\n")
        fh.write(report)
        fh.write("\n\nReconstruction error stats:\n")
        for key, val in err_stats.items():
            fh.write(f"{key}: {val:.6f}\n")
    print(json.dumps(metrics, indent=2)); print(f"Per-class report written to {log_path}")
    print('✅ “autoencoder evaluation complete – metrics saved under autoencoder/eval/metrics.json”')


if __name__ == "__main__":
    main()
