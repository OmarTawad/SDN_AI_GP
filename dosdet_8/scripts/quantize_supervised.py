#!/usr/bin/env python3
"""Create a dynamic int8 checkpoint for the CNN DoS detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.dws_cnn import FastDetector
from models.quantization import apply_dynamic_quantization, set_quantized_engine, unpack_checkpoint


def _resolve_output(cfg: dict, output: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    cfg_quant = cfg.get("quantization", {}) or {}
    configured = cfg_quant.get("checkpoint_path")
    if configured:
        return Path(configured).expanduser().resolve()
    artifacts_dir = cfg.get("paths", {}).get("artifacts_dir", "artifacts")
    return (Path(artifacts_dir) / "model_best_int8_dynamic.pt").expanduser().resolve()


def _resolve_checkpoint(cfg: dict, checkpoint: str | None) -> Path:
    if checkpoint:
        return Path(checkpoint).expanduser().resolve()
    artifacts_dir = cfg.get("paths", {}).get("artifacts_dir", "artifacts")
    return (Path(artifacts_dir) / "model_best.pt").expanduser().resolve()


def _load_meta(artifacts_dir: Path) -> dict:
    meta_path = artifacts_dir / "feature_model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing feature_model_meta.json at {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize the CNN checkpoint to dynamic int8.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Float checkpoint (defaults to artifacts/model_best.pt)")
    parser.add_argument("--output", default=None, help="Output path for int8 checkpoint")
    parser.add_argument("--backend", default=None, help="Quantized backend engine (fbgemm or qnnpack)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    artifacts_dir = Path(cfg.get("paths", {}).get("artifacts_dir", "artifacts")).expanduser().resolve()
    meta = _load_meta(artifacts_dir)

    checkpoint = _resolve_checkpoint(cfg, args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    engine = set_quantized_engine(args.backend or (cfg.get("quantization", {}) or {}).get("backend"))

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

    quantized = apply_dynamic_quantization(model, dtype=torch.qint8)

    output_path = _resolve_output(cfg, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": quantized.state_dict(),
        "quantized": True,
        "quantization": {"mode": "dynamic", "dtype": "qint8", "backend": engine},
        "source_checkpoint": str(checkpoint),
    }
    torch.save(payload, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved int8 checkpoint → {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
