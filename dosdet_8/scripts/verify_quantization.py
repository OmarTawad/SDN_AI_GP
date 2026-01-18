#!/usr/bin/env python3
"""Verify dynamic int8 quantization for the CNN DoS detector."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.feature_slimming import StaticSlimmer
from features.scaler import RobustScaler
from models.dws_cnn import FastDetector
from models.quantization import apply_dynamic_quantization, set_quantized_engine, unpack_checkpoint
from train import CachedDataset, collate, load_manifest, _read_parquet


def _resolve_path(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root / p)


def _test_paths(cfg: dict, root: Path) -> list[str]:
    cache_dir = _resolve_path(root, cfg["paths"]["cache_dir"])
    manifest = load_manifest(str(cache_dir))
    shard_rel = [entry["path"] for entry in manifest.get("files", [])]
    if not shard_rel:
        raise FileNotFoundError("No parquet shards listed in cache/manifest.json.")

    stats = []
    for rel in shard_rel:
        path = _resolve_path(root, rel)
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


def _build_loader(cfg: dict, root: Path, batch_size: int) -> DataLoader:
    dataset = CachedDataset(_test_paths(cfg, root), normal_subsample_rate=1.0)
    num_workers = max(0, cfg["training"].get("dataloader_workers", 0))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)


def _load_meta(artifacts_dir: Path) -> dict:
    meta_path = artifacts_dir / "feature_model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing feature_model_meta.json at {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_model(cfg: dict, meta: dict, checkpoint: Path, quantized: bool, backend: str | None) -> nn.Module:
    model = FastDetector(
        seq_in_dim=int(meta["seq_in_dim"]),
        static_dim=int(meta["static_dim"]),
        channels=tuple(meta.get("channels", cfg["training"]["channels"])),
        k=int(meta.get("kernel_size", cfg["training"]["kernel_size"])),
        drop=float(meta.get("dropout", cfg["training"]["dropout"])),
        mlp_hidden=tuple(meta.get("mlp_hidden", cfg["training"]["mlp_hidden"])),
    ).to(device="cpu", dtype=torch.float32)

    state = torch.load(checkpoint, map_location="cpu")
    state_dict, is_quantized = unpack_checkpoint(state)
    if not quantized:
        model.load_state_dict(state_dict)
        return model.eval()

    set_quantized_engine(backend)
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
        candidates = [getattr(qdynamic, "Linear", None)]
    return tuple(cls for cls in candidates if cls is not None)


def _summarize_modules(label: str, model: nn.Module) -> None:
    qtypes = _quantized_classes()
    lines = []
    quantized_hits = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)) or (qtypes and isinstance(module, qtypes)):
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


def _measure_latency(model: nn.Module, seq: torch.Tensor, static: torch.Tensor, runs: int, warmup: int) -> float:
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = model(seq, static)
        start = time.perf_counter()
        for _ in range(max(1, runs)):
            _ = model(seq, static)
        elapsed = time.perf_counter() - start
    return elapsed / max(1, runs)


def _run_eval(
    config_path: Path,
    artifacts_dir: Path,
    eval_dir: Path,
    args,
    quantized: bool,
    quantized_checkpoint: Path | None,
) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "evaluate_dosdet.py"),
        "--config",
        str(config_path),
        "--artifacts",
        str(artifacts_dir),
        "--eval-dir",
        str(eval_dir),
        "--batch-size",
        str(args.batch_size),
    ]
    if quantized:
        cmd.append("--quantized")
    else:
        cmd.append("--no-quantized")
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
    parser = argparse.ArgumentParser(description="Verify dynamic int8 quantization (CNN).")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--float-checkpoint", default=None, help="Float checkpoint (defaults to artifacts/model_best.pt)")
    parser.add_argument("--int8-checkpoint", default=None, help="Int8 checkpoint (defaults to config)")
    parser.add_argument("--backend", type=str, default=None, help="Quantized backend engine (fbgemm or qnnpack)")
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--latency-runs", type=int, default=50, help="Number of latency runs")
    parser.add_argument("--latency-warmup", type=int, default=5, help="Warmup iterations before timing")
    parser.add_argument("--eval-dir", default="eval/quantization", help="Output directory for eval runs")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    artifacts_dir = Path(cfg.get("paths", {}).get("artifacts_dir", "artifacts")).expanduser().resolve()
    meta = _load_meta(artifacts_dir)

    float_ckpt = Path(args.float_checkpoint).expanduser().resolve() if args.float_checkpoint else (artifacts_dir / "model_best.pt").resolve()
    int8_ckpt = args.int8_checkpoint or (cfg.get("quantization", {}) or {}).get("checkpoint_path")
    if int8_ckpt is None:
        raise FileNotFoundError("No int8 checkpoint specified. Run scripts/quantize_supervised.py first.")
    int8_ckpt = Path(int8_ckpt).expanduser().resolve()
    if not float_ckpt.is_file():
        raise FileNotFoundError(f"Float checkpoint not found: {float_ckpt}")
    if not int8_ckpt.is_file():
        raise FileNotFoundError(f"Int8 checkpoint not found: {int8_ckpt}")

    root = ROOT
    scaler = RobustScaler.load(str(artifacts_dir))
    slimmer = StaticSlimmer()
    slimmer.load(str(artifacts_dir))
    loader = _build_loader(cfg, root, args.batch_size)
    batch = next(iter(loader))
    seq, static, *_ = batch
    stat_np = static.numpy()
    stat_scaled = scaler.transform(stat_np)
    stat_slim = slimmer.transform(stat_scaled).astype(np.float32)
    seq_t = seq.to(dtype=torch.float32)
    static_t = torch.from_numpy(stat_slim).to(dtype=torch.float32)

    float_model = _load_model(cfg, meta, float_ckpt, quantized=False, backend=args.backend)
    int8_model = _load_model(cfg, meta, int8_ckpt, quantized=True, backend=args.backend)

    _summarize_modules("float", float_model)
    _summarize_modules("int8", int8_model)

    float_latency = _measure_latency(float_model, seq_t, static_t, args.latency_runs, args.latency_warmup)
    int8_latency = _measure_latency(int8_model, seq_t, static_t, args.latency_runs, args.latency_warmup)

    float_size_mb = float_ckpt.stat().st_size / (1024 * 1024)
    int8_size_mb = int8_ckpt.stat().st_size / (1024 * 1024)

    eval_root = Path(args.eval_dir).expanduser().resolve()
    float_eval_dir = eval_root / "float"
    int8_eval_dir = eval_root / "int8"
    float_eval_dir.mkdir(parents=True, exist_ok=True)
    int8_eval_dir.mkdir(parents=True, exist_ok=True)

    float_metrics = _run_eval(Path(args.config).resolve(), artifacts_dir, float_eval_dir, args, quantized=False, quantized_checkpoint=None)
    int8_metrics = _run_eval(Path(args.config).resolve(), artifacts_dir, int8_eval_dir, args, quantized=True, quantized_checkpoint=int8_ckpt)

    print("\n[checkpoint sizes]")
    print(f"  float: {float_size_mb:.2f} MB → {float_ckpt}")
    print(f"  int8 : {int8_size_mb:.2f} MB → {int8_ckpt}")

    print("\n[latency]")
    print(f"  float avg: {float_latency * 1000:.3f} ms")
    print(f"  int8  avg: {int8_latency * 1000:.3f} ms")

    print("\n[accuracy/F1 via evaluate_dosdet.py]")
    print(f"  float accuracy={float_metrics.get('accuracy')} f1={float_metrics.get('f1')}")
    print(f"  int8  accuracy={int8_metrics.get('accuracy')} f1={int8_metrics.get('f1')}")


if __name__ == "__main__":
    main()
