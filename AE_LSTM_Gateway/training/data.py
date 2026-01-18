"""Dataset orchestration helpers for MoE training."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from gateway.core import class_id_to_name
from gateway.data.datasets.cache import load_cache_entries, tasks_slug
from gateway.data.datasets.cached_dataset import CachedMoEDataset
from gateway.data.datasets.streaming_dataset import MoEDataset
from gateway.data.structures.pcap import PcapInfo


def build_dataset(
    config,
    files: Sequence[PcapInfo],
) -> Tuple[IterableDataset[Tuple[Dict[str, Tensor], Tensor]], bool]:
    entries, missing, mismatched = load_cache_entries(config.cache_base, files, config.tasks)
    slug = tasks_slug(config.tasks)
    cache_dir = config.cache_base / slug

    if config.use_cache != "off" and entries and not missing and not mismatched:
        print(f"[Cache] Using cached tensors from {cache_dir}")
        payloads: List[Dict[str, object]] = []
        for entry in entries:
            data = entry["data"]
            features = {key: tensor.detach().to(torch.float32) for key, tensor in data.get("features", {}).items()}
            payload: Dict[str, object] = {"features": features}
            targets = data.get("targets")
            if targets is not None:
                payload["targets"] = torch.as_tensor(targets).to(torch.long)
            labels = data.get("labels")
            if isinstance(labels, dict):
                payload["labels"] = {key: torch.as_tensor(value).to(torch.float32) for key, value in labels.items()}
            meta = dict(data.get("meta", {}))
            meta.setdefault("source_path", str(entry["path"]))
            meta.setdefault("cache_path", str(entry["cache_path"]))
            payload["meta"] = meta
            payloads.append(payload)
        dataset = CachedMoEDataset(
            cache_entries=payloads,
            tasks=config.tasks,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed,
            max_windows_per_file=config.max_windows_per_file,
            max_total_windows=config.max_total_windows,
        )
        return dataset, True

    if config.use_cache == "on":
        problems: List[str] = []
        if not entries:
            problems.append("no cache files were found")
        if missing:
            problems.append("missing caches for: " + ", ".join(path.name for path in missing))
        if mismatched:
            problems.append(
                "invalid caches: " + ", ".join(f"{path.name} ({reason})" for path, reason in mismatched)
            )
        raise RuntimeError("Cache usage forced with --use-cache on, but " + "; ".join(problems))

    if entries and (missing or mismatched):
        if missing:
            print(f"[Cache] Missing caches for {len(missing)} PCAPs; falling back to raw streaming.")
        if mismatched:
            print(f"[Cache] Ignoring {len(mismatched)} cache files due to mismatches.")

    dataset = MoEDataset(
        files=files,
        tasks=config.tasks,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        max_windows_per_file=config.max_windows_per_file,
        max_total_windows=config.max_total_windows,
        status_interval=config.status_interval,
        max_file_size=config.file_size_bytes,
        max_packets_per_file=config.max_packets_per_file,
        file_timeout=config.file_timeout,
        max_packets_per_window=config.max_packets_per_window,
    )
    return dataset, False


def set_dataset_budget(dataset: IterableDataset, window_budget: Optional[int]) -> None:
    if hasattr(dataset, "set_window_budget"):
        dataset.set_window_budget(window_budget)


def set_dataset_logger(dataset: IterableDataset, logger) -> None:
    if hasattr(dataset, "set_log_fn"):
        dataset.set_log_fn(logger)


def set_dataset_epoch(dataset: IterableDataset, epoch: int) -> None:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


__all__ = [
    "build_dataset",
    "set_dataset_budget",
    "set_dataset_logger",
    "set_dataset_epoch",
]
