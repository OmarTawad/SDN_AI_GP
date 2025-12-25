#!/usr/bin/env python3
"""Verify dynamic int8 quantization for the supervised LSTM model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from dos_detector.config import load_config
from dos_detector.data.dataset import StreamingSequenceDataset, collate_fn
from dos_detector.models.supervised import SequenceClassifier
from dos_detector.utils import configure_cpu_environment, safe_cast_tensor
from dos_detector.utils.io import load_joblib, load_json
from dos_detector.utils.quantization import apply_dynamic_quantization, set_quantized_engine, unpack_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def _resolve_files(manifest: dict, config, split: str) -> Sequence[str]:
    configured = getattr(config.data, f"{split}_files", []) or []
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


def _build_loader(files: Sequence[str], config, feature_columns: Sequence[str], batch_size: int, scaler) -> DataLoader:
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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )


def _load_model(cfg, feature_columns: Sequence[str], checkpoint: Path, quantized: bool, backend: str | None) -> nn.Module:
    model = SequenceClassifier(
        input_size=len(feature_columns),
        num_attack_types=len(cfg.labels.family_mapping),
        config=cfg.model.supervised,
    )
    if not quantized:
        state = torch.load(checkpoint, map_location="cpu")
        state_dict, _ = unpack_checkpoint(state)
        model.load_state_dict(state_dict)
        return model.to(device="cpu", dtype=torch.float32).eval()

    set_quantized_engine(backend)
    model = model.to(device="cpu", dtype=torch.float32)
    state = torch.load(checkpoint, map_location="cpu")
    state_dict, is_quantized = unpack_checkpoint(state)
    if is_quantized:
        quantized_model = apply_dynamic_quantization(model)
        quantized_model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
        quantized_model = apply_dynamic_quantization(model)
    return quantized_model.eval()


def _quantized_classes() -> tuple[type, ...]:
    qdynamic = getattr(torch.nn, "quantized", None)
    qdynamic = getattr(qdynamic, "dynamic", None) if qdynamic else None
    candidates: Iterable[type] = []
    if qdynamic is not None:
        candidates = [getattr(qdynamic, "Linear", None), getattr(qdynamic, "LSTM", None)]
    return tuple(cls for cls in candidates if cls is not None)


def _summarize_modules(label: str, model: nn.Module) -> None:
    qtypes = _quantized_classes()
    lines = []
    quantized_hits = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.LSTM, nn.Linear)) or (qtypes and isinstance(module, qtypes)):
            lines.append(f"{name or '<root>'}: {type(module).__name__}")
        if qtypes and isinstance(module, qtypes):
            quantized_hits.append(name or "<root>")
        elif "Quantized" in type(module).__name__:
            quantized_hits.append(name or "<root>")
    print(f"\n[{label}] module types")
    for line in lines:
        print(f"  - {line}")
    if quantized_hits:
        print(f"[{label}] quantized modules detected: {len(quantized_hits)}")
    else:
        print(f"[{label}] quantized modules detected: 0")


def _measure_latency(model: nn.Module, features: torch.Tensor, runs: int, warmup: int) -> float:
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = model(features)
        start = time.perf_counter()
        for _ in range(max(1, runs)):
            _ = model(features)
        elapsed = time.perf_counter() - start
    return elapsed / max(1, runs)


def _run_eval(config_path: Path, checkpoint: Path, eval_dir: Path, args, quantized: bool, quantized_checkpoint: Path | None) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "evaluate_lstm.py"),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--eval-dir",
        str(eval_dir),
        "--batch-size",
        str(args.batch_size),
        "--threshold",
        str(args.threshold),
        "--split",
        args.split,
    ]
    if quantized:
        cmd.append("--quantized")
    if quantized_checkpoint is not None:
        cmd.extend(["--quantized-checkpoint", str(quantized_checkpoint)])
    if args.backend:
        cmd.extend(["--quant-backend", args.backend])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)
    metrics_path = eval_dir / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dynamic int8 quantization.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    parser.add_argument("--float-checkpoint", type=Path, default=None, help="Float checkpoint (defaults to config)")
    parser.add_argument("--int8-checkpoint", type=Path, default=None, help="Int8 checkpoint (defaults to config)")
    parser.add_argument("--backend", type=str, default=None, help="Quantized backend engine (fbgemm or qnnpack)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Window-level decision threshold")
    parser.add_argument("--latency-runs", type=int, default=50, help="Number of latency runs")
    parser.add_argument("--latency-warmup", type=int, default=5, help="Warmup iterations before timing")
    parser.add_argument("--eval-dir", type=Path, default=Path("eval/quantization"), help="Output directory for eval runs")
    args = parser.parse_args()

    configure_cpu_environment(threads=2, interop_threads=2)
    cfg = load_config(args.config)
    manifest = load_json(cfg.paths.manifest_path)
    feature_columns = manifest.get("feature_columns", [])
    if not feature_columns:
        raise ValueError(f"No feature columns found in manifest → {cfg.paths.manifest_path}")

    float_ckpt = args.float_checkpoint or Path(cfg.paths.supervised_model_path)
    float_ckpt = Path(float_ckpt).expanduser().resolve()
    int8_ckpt = args.int8_checkpoint or getattr(cfg.quantization, "checkpoint_path", None)
    if int8_ckpt is None:
        raise FileNotFoundError("No int8 checkpoint specified. Run scripts/quantize_supervised.py first.")
    int8_ckpt = Path(int8_ckpt).expanduser().resolve()
    if not float_ckpt.is_file():
        raise FileNotFoundError(f"Float checkpoint not found: {float_ckpt}")
    if not int8_ckpt.is_file():
        raise FileNotFoundError(f"Int8 checkpoint not found: {int8_ckpt}")

    files = _resolve_files(manifest, cfg, args.split)
    scaler = load_joblib(cfg.paths.scaler_path)
    loader = _build_loader(files, cfg, feature_columns, args.batch_size, scaler)
    batch = next(iter(loader))
    features = safe_cast_tensor(batch["features"], torch.float32)

    float_model = _load_model(cfg, feature_columns, float_ckpt, quantized=False, backend=args.backend)
    int8_model = _load_model(cfg, feature_columns, int8_ckpt, quantized=True, backend=args.backend)

    _summarize_modules("float", float_model)
    _summarize_modules("int8", int8_model)

    float_latency = _measure_latency(float_model, features, args.latency_runs, args.latency_warmup)
    int8_latency = _measure_latency(int8_model, features, args.latency_runs, args.latency_warmup)

    float_size_mb = float_ckpt.stat().st_size / (1024 * 1024)
    int8_size_mb = int8_ckpt.stat().st_size / (1024 * 1024)

    eval_root = args.eval_dir.resolve()
    float_eval_dir = eval_root / "float"
    int8_eval_dir = eval_root / "int8"
    float_eval_dir.mkdir(parents=True, exist_ok=True)
    int8_eval_dir.mkdir(parents=True, exist_ok=True)

    float_metrics = _run_eval(args.config, float_ckpt, float_eval_dir, args, quantized=False, quantized_checkpoint=None)
    int8_metrics = _run_eval(args.config, float_ckpt, int8_eval_dir, args, quantized=True, quantized_checkpoint=int8_ckpt)

    print("\n[checkpoint sizes]")
    print(f"  float: {float_size_mb:.2f} MB → {float_ckpt}")
    print(f"  int8 : {int8_size_mb:.2f} MB → {int8_ckpt}")

    print("\n[latency]")
    print(f"  float avg: {float_latency * 1000:.3f} ms")
    print(f"  int8  avg: {int8_latency * 1000:.3f} ms")

    print("\n[accuracy/F1 via evaluate_lstm.py]")
    print(f"  float accuracy={float_metrics.get('accuracy')} f1={float_metrics.get('f1')}")
    print(f"  int8  accuracy={int8_metrics.get('accuracy')} f1={int8_metrics.get('f1')}")


if __name__ == "__main__":
    main()
