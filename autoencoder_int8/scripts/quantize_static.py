#!/usr/bin/env python3
"""Post-training static quantization (PTQ) for the autoencoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dae.config import load_config  # noqa: E402
from dae.model import build_model  # noqa: E402
from dae.preprocess import apply_log_transform, replace_invalid, load_feature_list, load_scaler, select_features  # noqa: E402
from dae.quantization import apply_static_quantization_fx, set_quantized_engine, unpack_checkpoint  # noqa: E402

try:  # optional dependency for safer Parquet streaming on older CPUs
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None


def _resolve_checkpoint(config, checkpoint: str | None) -> Path:
    if checkpoint:
        return Path(checkpoint).expanduser().resolve()
    artifacts_dir = Path(config.get("paths", "artifacts_dir"))
    return (artifacts_dir / "model.pt").expanduser().resolve()


def _resolve_output(config, output: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    quant_cfg = config.get("quantization", default={}) or {}
    configured = quant_cfg.get("checkpoint_path")
    if configured:
        return Path(configured).expanduser().resolve()
    artifacts_dir = Path(config.get("paths", "artifacts_dir"))
    return (artifacts_dir / "model_int8_static.pt").expanduser().resolve()


def _resolve_windows(config, windows: str | None) -> list[Path]:
    if windows:
        path = Path(windows).expanduser().resolve()
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(path.glob("*.parquet"))
        raise FileNotFoundError(f"Windows path not found: {path}")
    windows_dir = config.get("paths", "windows_dir")
    if not windows_dir:
        raise FileNotFoundError("No windows path provided and paths.windows_dir is empty.")
    path = Path(windows_dir).expanduser().resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.parquet"))
    raise FileNotFoundError(f"Windows path not found: {path}")


def _iter_calib_batches(
    windows_paths: Iterable[Path],
    feature_names: list[str],
    log_features: list[str],
    scaler,
    clip_bounds: dict,
    batch_size: int,
    max_batches: int,
) -> Iterator[tuple[torch.Tensor]]:
    lower = {k: float(v) for k, v in clip_bounds.get("lower", {}).items()}
    upper = {k: float(v) for k, v in clip_bounds.get("upper", {}).items()}
    emitted = 0
    for path in windows_paths:
        for df in _iter_parquet_batches(path, batch_size):
            if df.empty:
                continue
            missing = [col for col in feature_names if col not in df.columns]
            if len(missing) == len(feature_names):
                continue
            for col in missing:
                df[col] = 0.0
            features = select_features(df, feature_names)
            features = apply_log_transform(features, log_features)
            if lower:
                features = features.clip(lower=lower, axis=1)
            if upper:
                features = features.clip(upper=upper, axis=1)
            features = replace_invalid(features)
            scaled = scaler.transform(features.values).astype(np.float32, copy=False)
            yield (torch.from_numpy(scaled),)
            emitted += 1
            if emitted >= max_batches:
                return


def _iter_parquet_batches(path: Path, rows_per_batch: int) -> Iterator:
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
    from dae.utils_io import read_parquet_batches

    yield from read_parquet_batches(path, rows_per_batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="PTQ (static int8) for the autoencoder.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Float checkpoint (defaults to artifacts/model.pt)")
    parser.add_argument("--output", default=None, help="Output path for int8 checkpoint")
    parser.add_argument("--backend", default=None, help="Quantized backend engine (fbgemm or qnnpack)")
    parser.add_argument("--windows", default=None, help="Parquet file or directory for calibration windows")
    parser.add_argument("--calib-batches", type=int, default=32, help="Calibration batches to run")
    parser.add_argument("--batch-size", type=int, default=8192, help="Calibration batch size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(cfg.get("paths", "artifacts_dir"))
    meta_path = artifacts_dir / "model_config.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Model config not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    checkpoint = _resolve_checkpoint(cfg, args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    backend = set_quantized_engine(args.backend or (cfg.get("quantization", default={}) or {}).get("backend"))

    model = build_model(meta.get("model", {}), input_dim=int(meta["input_dim"]))
    model.to(device="cpu", dtype=torch.float32)
    state = torch.load(checkpoint, map_location="cpu")
    state_dict, _ = unpack_checkpoint(state)
    model.load_state_dict(state_dict)
    model.eval()

    feature_names = load_feature_list(artifacts_dir / "feature_list.json")
    log_features = list(meta.get("log_features", []))
    scaler = load_scaler(artifacts_dir / "scaler.pkl")
    clip_path = artifacts_dir / "clip_bounds.json"
    clip_bounds = json.loads(clip_path.read_text(encoding="utf-8")) if clip_path.exists() else {"lower": {}, "upper": {}}

    windows_paths = _resolve_windows(cfg, args.windows)
    calib_batches = list(
        _iter_calib_batches(
            windows_paths,
            feature_names,
            log_features,
            scaler,
            clip_bounds,
            args.batch_size,
            args.calib_batches,
        )
    )
    if not calib_batches:
        raise RuntimeError("No calibration batches were collected; check --windows points to autoencoder parquet files.")

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
    print(f"Saved static int8 checkpoint -> {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
