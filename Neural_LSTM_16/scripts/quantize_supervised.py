#!/usr/bin/env python3
"""Create a dynamic int8 checkpoint for the supervised LSTM detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dos_detector.config import load_config
from dos_detector.models.supervised import SequenceClassifier
from dos_detector.utils.io import ensure_dir, load_json
from dos_detector.utils.quantization import apply_dynamic_quantization, set_quantized_engine, unpack_checkpoint


def _resolve_output(cfg, output: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    configured = getattr(cfg.quantization, "checkpoint_path", None)
    if configured:
        return Path(configured)
    return Path(cfg.paths.models_dir) / "supervised_int8_dynamic.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize the supervised LSTM checkpoint to dynamic int8.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Float checkpoint (defaults to config paths)")
    parser.add_argument("--output", type=str, default=None, help="Output path for int8 checkpoint")
    parser.add_argument("--backend", type=str, default=None, help="Quantized backend engine (fbgemm or qnnpack)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = load_json(cfg.paths.manifest_path)
    feature_columns = manifest.get("feature_columns", [])
    if not feature_columns:
        raise ValueError(f"No feature columns found in manifest → {cfg.paths.manifest_path}")

    checkpoint = args.checkpoint or Path(cfg.paths.supervised_model_path)
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    engine = set_quantized_engine(args.backend or getattr(cfg.quantization, "backend", None))

    model = SequenceClassifier(
        input_size=len(feature_columns),
        num_attack_types=len(cfg.labels.family_mapping),
        config=cfg.model.supervised,
    ).to(device="cpu", dtype=torch.float32)
    state = torch.load(checkpoint, map_location="cpu")
    state_dict, _ = unpack_checkpoint(state)
    model.load_state_dict(state_dict)
    model.eval()

    quantized = apply_dynamic_quantization(model, dtype=torch.qint8)

    output_path = _resolve_output(cfg, args.output)
    ensure_dir(output_path.parent)
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
