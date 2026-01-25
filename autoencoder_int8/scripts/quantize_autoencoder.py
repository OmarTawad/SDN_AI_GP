#!/usr/bin/env python3
"""Create a dynamic int8 checkpoint for the autoencoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dae.config import load_config  # noqa: E402
from dae.model import build_model  # noqa: E402
from dae.quantization import (  # noqa: E402
    apply_dynamic_quantization,
    ensure_quantized_modules,
    set_quantized_engine,
    unpack_checkpoint,
)


def _resolve_output(config, output: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    quant_cfg = config.get("quantization", default={}) or {}
    configured = quant_cfg.get("checkpoint_path")
    if configured:
        return Path(configured).expanduser().resolve()
    artifacts_dir = Path(config.get("paths", "artifacts_dir"))
    return artifacts_dir / "model_int8_dynamic.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize the autoencoder checkpoint to dynamic int8.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Float checkpoint (defaults to artifacts/model.pt)")
    parser.add_argument("--output", type=str, default=None, help="Output path for int8 checkpoint")
    parser.add_argument("--backend", type=str, default=None, help="Quantized backend engine (fbgemm or qnnpack)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(cfg.get("paths", "artifacts_dir"))
    meta_path = artifacts_dir / "model_config.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Model config not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    checkpoint = args.checkpoint or (artifacts_dir / "model.pt")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    quant_cfg = cfg.get("quantization", default={}) or {}
    engine = set_quantized_engine(args.backend or quant_cfg.get("backend"))

    model = build_model(meta.get("model", {}), input_dim=int(meta["input_dim"]))
    model.to(device="cpu", dtype=torch.float32)
    state = torch.load(checkpoint, map_location="cpu")
    state_dict, _ = unpack_checkpoint(state)
    model.load_state_dict(state_dict)
    model.eval()

    quantized = apply_dynamic_quantization(model, dtype=torch.qint8)
    ensure_quantized_modules(quantized)

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
    print(f"Saved int8 checkpoint -> {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
