#!/usr/bin/env python3
"""Evaluate the CNN-based DoS detector on the cached test split."""

from __future__ import annotations

import argparse
import math
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from torch.utils.data import DataLoader

from features.feature_slimming import StaticSlimmer
from features.scaler import RobustScaler
from models.dws_cnn import FastDetector
from train import CachedDataset, collate, load_manifest, _read_parquet

ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_DIR = ROOT / "eval"


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def _test_paths(cfg: dict) -> List[str]:
    cache_dir = _resolve_path(cfg["paths"]["cache_dir"])
    manifest = load_manifest(str(cache_dir))
    shard_rel = [entry["path"] for entry in manifest.get("files", [])]
    if not shard_rel:
        raise FileNotFoundError("No parquet shards listed in cache/manifest.json.")

    stats = []
    for rel in shard_rel:
        path = _resolve_path(rel)
        pos = int(_read_parquet(path, columns=["y"])["y"].sum())
        stats.append({"path": str(path), "pos": pos})
    stats.sort(key=lambda s: (-s["pos"], s["path"]))

    f_train, f_val, _ = cfg["split"]["train_val_test"]
    n = len(stats)
    n_train = max(1, int(round(n * f_train)))
    n_val = max(1, int(round(n * f_val)))

    buckets = [[], [], []]
    for shard in stats:
        idx = 0 if len(buckets[0]) < n_train else 1 if len(buckets[1]) < n_val else 2
        buckets[idx].append(shard)
    if not buckets[2]:
        buckets[2] = [buckets[0][-1]]

    if buckets[1] and sum(s["pos"] for s in buckets[1]) == 0:
        donor = next((s for s in buckets[0] + buckets[2] if s["pos"] > 0), None)
        if donor:
            pool = buckets[0] if donor in buckets[0] else buckets[2]
            pool[pool.index(donor)] = buckets[1][0]
            buckets[1][0] = donor

    return [s["path"] for s in buckets[2]]


def _resolve_artifacts_dir(arg: str | None, cfg: dict) -> Path:
    candidates = [
        arg,
        cfg.get("paths", {}).get("artifacts_dir"),
        "artifacts",
        "models",
        "checkpoints",
    ]
    for cand in candidates:
        if not cand:
            continue
        path = _resolve_path(cand)
        if path.is_dir() and (path / "model_best.pt").exists():
            return path
    raise FileNotFoundError("Unable to locate model_best.pt in artifacts/models/checkpoints.")


def _load_artifacts(art_dir: Path, cfg: dict):
    scaler = RobustScaler.load(str(art_dir))
    slimmer = StaticSlimmer()
    slimmer.load(str(art_dir))
    with open(art_dir / "feature_model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = FastDetector(
        seq_in_dim=int(meta["seq_in_dim"]),
        static_dim=int(meta["static_dim"]),
        channels=tuple(meta.get("channels", cfg["training"]["channels"])),
        k=int(meta.get("kernel_size", cfg["training"]["kernel_size"])),
        drop=float(meta.get("dropout", cfg["training"]["dropout"])),
        mlp_hidden=tuple(meta.get("mlp_hidden", cfg["training"]["mlp_hidden"])),
    )
    checkpoint = torch.load(art_dir / "model_best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    calib_path = art_dir / "calibration.json"
    if calib_path.exists():
        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
    else:
        calib = {"temperature": 1.0, "threshold": 0.5}
    return model, scaler, slimmer, calib


def _build_loader(cfg: dict, batch_size: int) -> DataLoader:
    dataset = CachedDataset(_test_paths(cfg), normal_subsample_rate=1.0)
    num_workers = max(0, cfg["training"].get("dataloader_workers", 0))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)


def _evaluate(
    model: FastDetector,
    loader: DataLoader,
    scaler: RobustScaler,
    slimmer: StaticSlimmer,
    calib: Dict[str, float],
    device: torch.device,
) -> Tuple[Dict[str, float | None], np.ndarray, str]:
    model.to(device).eval()
    temperature = float(calib.get("temperature", 1.0)) or 1.0
    threshold = float(calib.get("threshold", 0.5))

    probs, preds, trues = [], [], []
    with torch.inference_mode():
        for seq, static, y, *_ in loader:
            seq = seq.to(device, non_blocking=True)
            stat_np = static.numpy()
            stat_scaled = scaler.transform(stat_np)
            stat_slim = slimmer.transform(stat_scaled).astype(np.float32)
            static_t = torch.from_numpy(stat_slim).to(device, non_blocking=True)
            logits = model(seq, static_t)["logits"].squeeze(-1) / temperature
            batch_prob = torch.sigmoid(logits).cpu().numpy()
            probs.append(batch_prob)
            preds.append((batch_prob >= threshold).astype(int))
            trues.append(y.numpy().ravel().astype(int))

    y_true = np.concatenate(trues).astype(int)
    y_prob = np.concatenate(probs)
    y_pred = np.concatenate(preds)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        roc = float(roc_auc_score(y_true, y_prob))
        metrics["roc_auc"] = None if math.isnan(roc) else roc
    except ValueError:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["normal", "attack"], zero_division=0)
    metrics.update({"threshold": threshold, "temperature": temperature, "samples": int(len(y_true)), "positives": int(y_true.sum())})
    return metrics, cm, report


def main():
    parser = argparse.ArgumentParser(description="Evaluate the DoS CNN on cached test data.")
    parser.add_argument("--config", default="config.yaml", help="YAML config path.")
    parser.add_argument("--artifacts", default=None, help="Override artifacts directory.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR), help="Where to write eval outputs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(_resolve_path(args.config), "r", encoding="utf-8"))
    artifacts_dir = _resolve_artifacts_dir(args.artifacts, cfg)
    eval_dir = _resolve_path(args.eval_dir)
    os.makedirs(eval_dir, exist_ok=True)

    loader = _build_loader(cfg, batch_size=args.batch_size)
    model, scaler, slimmer, calib = _load_artifacts(artifacts_dir, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics, cm, report = _evaluate(model, loader, scaler, slimmer, calib, device)

    with open(eval_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.save(eval_dir / "confusion_matrix.npy", cm)
    with open(eval_dir / "log.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(json.dumps(metrics, indent=2))
    print(f"\nPer-class report written to {eval_dir / 'log.txt'}")
    print("✅ dosdet evaluation complete – metrics saved under dosdet/eval/metrics.json")


if __name__ == "__main__":
    main()
