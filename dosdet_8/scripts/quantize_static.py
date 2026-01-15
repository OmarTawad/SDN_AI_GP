#!/usr/bin/env python3
"""Post-training static quantization (PTQ) for the CNN DoS detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.feature_slimming import StaticSlimmer
from features.scaler import RobustScaler
from models.dws_cnn import FastDetector
from models.quantization import apply_static_quantization_fx, set_quantized_engine, unpack_checkpoint
from train import CachedDataset, collate, load_manifest


def _resolve_checkpoint(cfg: dict, checkpoint: str | None) -> Path:
    if checkpoint:
        return Path(checkpoint).expanduser().resolve()
    artifacts_dir = cfg.get("paths", {}).get("artifacts_dir", "artifacts")
    return (Path(artifacts_dir) / "model_best.pt").expanduser().resolve()


def _resolve_output(cfg: dict, output: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    cfg_quant = cfg.get("quantization", {}) or {}
    configured = cfg_quant.get("checkpoint_path")
    if configured:
        return Path(configured).expanduser().resolve()
    artifacts_dir = cfg.get("paths", {}).get("artifacts_dir", "artifacts")
    return (Path(artifacts_dir) / "model_best_int8_static.pt").expanduser().resolve()


def _load_meta(artifacts_dir: Path) -> dict:
    meta_path = artifacts_dir / "feature_model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing feature_model_meta.json at {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _build_calib_loader(cfg: dict, batch_size: int) -> DataLoader:
    cache_dir = cfg.get("paths", {}).get("cache_dir", "cache")
    manifest = load_manifest(cache_dir)
    shard_paths = [entry["path"] for entry in manifest.get("files", [])]
    if not shard_paths:
        raise FileNotFoundError("No parquet shards listed in cache/manifest.json.")
    dataset = CachedDataset(shard_paths, normal_subsample_rate=1.0)
    num_workers = max(0, int(cfg.get("training", {}).get("dataloader_workers", 0)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)


def _collect_calib_batches(
    loader: DataLoader,
    scaler: RobustScaler,
    slimmer: StaticSlimmer,
    max_batches: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    feature_names = getattr(slimmer, "src_names", None) or scaler.feature_names_
    for step, batch in enumerate(loader):
        if step >= max_batches:
            break
        seq, static, *_ = batch
        stat_np = static.numpy()
        if feature_names is None or len(feature_names) != stat_np.shape[1]:
            feature_names = [f"f_{i}" for i in range(stat_np.shape[1])]
        try:
            stat_scaled = scaler.transform(stat_np, feature_names)
        except Exception:
            continue
        stat_slim = slimmer.transform(stat_scaled).astype(np.float32)
        seq_t = seq.to(dtype=torch.float32)
        static_t = torch.from_numpy(stat_slim).to(dtype=torch.float32)
        batches.append((seq_t, static_t))
    if not batches:
        raise RuntimeError("No calibration batches were collected.")
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description="PTQ (static int8) for the CNN detector.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Float checkpoint (defaults to artifacts/model_best.pt)")
    parser.add_argument("--output", default=None, help="Output path for int8 checkpoint")
    parser.add_argument("--backend", default=None, help="Quantized backend engine (fbgemm or qnnpack)")
    parser.add_argument("--calib-batches", type=int, default=32, help="Calibration batches to run")
    parser.add_argument("--batch-size", type=int, default=128, help="Calibration batch size")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    artifacts_dir = Path(cfg.get("paths", {}).get("artifacts_dir", "artifacts")).expanduser().resolve()
    meta = _load_meta(artifacts_dir)

    checkpoint = _resolve_checkpoint(cfg, args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    backend = set_quantized_engine(args.backend or (cfg.get("quantization", {}) or {}).get("backend"))

    model = FastDetector(
        seq_in_dim=int(meta["seq_in_dim"]),
        static_dim=int(meta["static_dim"]),
        channels=tuple(meta.get("channels", cfg["training"]["channels"])),
        k=int(meta.get("kernel_size", cfg["training"]["kernel_size"])),
        drop=float(meta.get("dropout", cfg["training"]["dropout"])),
        mlp_hidden=tuple(meta.get("mlp_hidden", cfg["training"]["mlp_hidden"])),
    ).to(device="cpu", dtype=torch.float32)

    state = torch.load(checkpoint, map_location="cpu")
    state_dict, _ = unpack_checkpoint(state)
    model.load_state_dict(state_dict)
    model.eval()

    scaler = RobustScaler.load(str(artifacts_dir))
    slimmer = StaticSlimmer()
    slimmer.load(str(artifacts_dir))
    loader = _build_calib_loader(cfg, args.batch_size)
    calib_batches = _collect_calib_batches(loader, scaler, slimmer, args.calib_batches)

    example_inputs = calib_batches[0]
    quantized = apply_static_quantization_fx(model, example_inputs, calib_batches, backend)

    output_path = _resolve_output(cfg, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": quantized.state_dict(),
        "quantized": True,
        "quantization": {"mode": "static", "dtype": "qint8", "backend": backend},
        "source_checkpoint": str(checkpoint),
        "calibration_batches": int(args.calib_batches),
    }
    torch.save(payload, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved static int8 checkpoint → {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
