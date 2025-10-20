"""Precompute MoE window features from PCAP captures into cached tensors."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from gateway.data_pipeline import (
    CACHE_ROOT,
    MoEDataset,
    PcapInfo,
    discover_pcaps,
    tasks_slug,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute MoE window features from PCAP files.")
    parser.add_argument("--tasks", type=str, default="dos,arp", help="Comma-separated list of tasks (dos, arp).")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size while streaming windows.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for deterministic shuffling (default: 17).")
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=0,
        help="Cap the number of windows extracted per PCAP (0 disables the cap).",
    )
    parser.add_argument(
        "--max-total-windows",
        type=int,
        default=0,
        help="Cap the total number of windows extracted per PCAP (0 disables the cap).",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=0.0,
        help="Skip PCAP files larger than this size in MB (0 disables the cap).",
    )
    parser.add_argument(
        "--max-packets-per-file",
        type=int,
        default=0,
        help="Skip after processing this many packets per PCAP (0 disables the cap).",
    )
    parser.add_argument(
        "--max-packets-per-window",
        type=int,
        default=0,
        help="Limit stored packets per window (0 disables the cap).",
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=0.0,
        help="Abort processing a PCAP after this many seconds (0 disables the timeout).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to store cached tensors (default: gateway/cache/<tasks_slug>).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild cached tensors even if they already exist.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file streaming logs.",
    )
    default_threads = max(1, min(2, os.cpu_count() or 1))
    parser.add_argument(
        "--num-threads",
        type=int,
        default=default_threads,
        help="Torch CPU thread pool size (default: min(2, available cores)).",
    )
    return parser.parse_args()


def _prepare_dataset(
    info: PcapInfo,
    tasks: Sequence[str],
    batch_size: int,
    seed: int,
    max_windows_per_file: Optional[int],
    max_total_windows: Optional[int],
    max_file_size: Optional[int],
    max_packets_per_file: Optional[int],
    file_timeout: Optional[float],
    max_packets_per_window: Optional[int],
) -> MoEDataset:
    dataset = MoEDataset(
        files=[info],
        tasks=tasks,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        max_windows_per_file=max_windows_per_file,
        max_total_windows=max_total_windows,
        status_interval=None,
        max_file_size=max_file_size,
        max_packets_per_file=max_packets_per_file,
        file_timeout=file_timeout,
        max_packets_per_window=max_packets_per_window,
    )
    dataset.set_window_budget(None)
    return dataset


def _stack_buffers(buffers: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: torch.cat([tensor.cpu() for tensor in values], dim=0) for key, values in buffers.items()}


def preprocess(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, min(args.num_threads, os.cpu_count() or 1)))

    selected_tasks = [task.strip().lower() for task in args.tasks.split(",") if task.strip()]
    if not selected_tasks:
        raise ValueError("No tasks specified. Choose from 'dos' and 'arp'.")
    valid_tasks = {"dos", "arp"}
    for task in selected_tasks:
        if task not in valid_tasks:
            raise ValueError(f"Unsupported task '{task}'. Valid options: dos, arp.")

    files = discover_pcaps(selected_tasks)
    if not files:
        raise RuntimeError("No eligible PCAP files found in 'samples/'. Populate the directory with .pcap files.")

    max_windows_per_file = args.max_windows_per_file if args.max_windows_per_file > 0 else None
    max_total_windows = args.max_total_windows if args.max_total_windows > 0 else None
    max_file_size = int(args.max_file_size_mb * 1024 * 1024) if args.max_file_size_mb and args.max_file_size_mb > 0 else None
    max_packets_per_file = args.max_packets_per_file if args.max_packets_per_file and args.max_packets_per_file > 0 else None
    file_timeout = args.file_timeout if args.file_timeout and args.file_timeout > 0 else None

    cache_base = Path(args.cache_dir).resolve() if args.cache_dir else CACHE_ROOT
    slug = tasks_slug(selected_tasks)
    cache_dir = cache_base / slug
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_files: List[Dict[str, object]] = []
    manifest_skipped: List[Dict[str, object]] = []

    progress = tqdm(files, desc="Preprocessing PCAPs", unit="pcap")
    for info in progress:
        cache_path = cache_dir / f"{info.path.name}.pt"
        progress.set_postfix_str(info.path.name)
        if cache_path.exists() and not args.overwrite:
            progress.write(f"[Skip] {info.path.name}: cache already exists (use --overwrite to rebuild).")
            manifest_files.append(
                {
                    "source": str(info.path),
                    "cache": str(cache_path),
                    "status": "existing",
                }
            )
            continue

        dataset = _prepare_dataset(
            info=info,
            tasks=selected_tasks,
            batch_size=args.batch_size,
            seed=args.seed,
            max_windows_per_file=max_windows_per_file,
            max_total_windows=max_total_windows,
            max_file_size=max_file_size,
            max_packets_per_file=max_packets_per_file,
            file_timeout=file_timeout,
            max_packets_per_window=args.max_packets_per_window if args.max_packets_per_window > 0 else None,
        )
        window_total = max_windows_per_file or max_total_windows
        window_bar = tqdm(
            total=window_total,
            unit="window",
            desc=f"{info.path.name}",
            leave=False,
        )
        if not args.quiet:
            dataset.set_log_fn(window_bar.write)
        else:
            dataset.set_log_fn(None)

        def _window_hook(_info: PcapInfo, increment: int) -> None:
            window_bar.update(increment)

        dataset.set_window_callback(_window_hook)
        dataset.set_epoch(0)

        feature_buffers: Dict[str, List[torch.Tensor]] = defaultdict(list)
        label_buffers: Dict[str, List[float]] = {task: [] for task in selected_tasks}

        for batch_features, batch_labels in dataset:
            for key, tensor in batch_features.items():
                feature_buffers[key].append(tensor.detach().cpu())
            for task, tensor in batch_labels.items():
                label_buffers[task].extend([float(value) for value in tensor.detach().cpu().tolist()])

        stats = dataset.stats.get(info.path, {})
        skip_reason = stats.get("skipped_reason") or dataset.skipped.get(info.path)
        window_bar.close()
        if skip_reason:
            progress.write(f"[Skip] {info.path.name}: {skip_reason}")
            manifest_skipped.append({"source": str(info.path), "reason": skip_reason})
            if cache_path.exists():
                cache_path.unlink()
            continue

        window_count = int(stats.get("windows", 0))
        if window_count <= 0 or not feature_buffers:
            progress.write(f"[Skip] {info.path.name}: no windows extracted.")
            manifest_skipped.append({"source": str(info.path), "reason": "no windows extracted"})
            if cache_path.exists():
                cache_path.unlink()
            continue

        features = _stack_buffers(feature_buffers)
        labels = {
            task: torch.tensor(values, dtype=torch.float32)
            for task, values in label_buffers.items()
        }

        if any(tensor.shape[0] != window_count for tensor in features.values()):
            raise RuntimeError(f"Feature tensor length mismatch while processing {info.path.name}.")

        for task, tensor in labels.items():
            if tensor.shape[0] != window_count:
                raise RuntimeError(f"Label tensor length mismatch for task '{task}' in {info.path.name}.")

        meta = {
            "source_path": str(info.path),
            "source_size": info.path.stat().st_size if info.path.exists() else 0,
            "windows": window_count,
            "batches": int(stats.get("batches", 0)),
            "labels": stats.get("labels", {}),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "params": {
                "tasks": list(selected_tasks),
                "max_windows_per_file": max_windows_per_file,
                "max_total_windows": max_total_windows,
                "max_file_size": max_file_size,
                "max_packets_per_file": max_packets_per_file,
                "file_timeout": file_timeout,
            },
        }

        cache_entry = {
            "tasks": list(selected_tasks),
            "features": features,
            "labels": labels,
            "meta": meta,
        }

        torch.save(cache_entry, cache_path)
        manifest_files.append(
            {
                "source": str(info.path),
                "cache": str(cache_path),
                "windows": window_count,
                "labels": meta["labels"],
                "status": "generated",
            }
        )
        progress.write(f"[Cache] Wrote {window_count} windows for {info.path.name} -> {cache_path}")

    manifest = {
        "tasks": selected_tasks,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": manifest_files,
        "skipped": manifest_skipped,
        "params": {
            "batch_size": args.batch_size,
            "max_windows_per_file": max_windows_per_file,
            "max_total_windows": max_total_windows,
            "max_file_size": max_file_size,
            "max_packets_per_file": max_packets_per_file,
            "file_timeout": file_timeout,
        },
    }
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    preprocess(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
