"""Dataset that streams windows from cached tensor files.


"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from gateway.core import class_id_to_name
from gateway.data.datasets.labels import coerce_targets_from_cache, infer_label_from_metadata


def _prepare_entry(entry: Dict[str, Any], idx: int) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
    processed = dict(entry)
    meta = processed.get("meta", {})
    source_path = Path(meta.get("source_path", f"cache_{idx}.pcap"))
    label = infer_label_from_metadata(meta)
    if label is None:
        label = 0
    label = int(label)
    processed["class_label"] = label
    processed["path"] = source_path

    stats_entry = {
        "windows": 0,
        "batches": 0,
        "label": label,
        "label_name": class_id_to_name(label),
        "truncated_windows": 0,
    }

    features = processed.get("features", {})
    if not features:
        return processed, stats_entry, "missing features"
    feature_lengths = {key: tensor.shape[0] for key, tensor in features.items()}
    if not feature_lengths:
        return processed, stats_entry, "empty feature tensors"

    window_count = min(feature_lengths.values())
    processed["num_windows"] = window_count
    raw_targets = processed.get("targets", processed.get("labels"))
    processed["targets"] = coerce_targets_from_cache(raw_targets, label, window_count)
    return processed, stats_entry, None


def _build_window_index(
    entries: Sequence[Dict[str, Any]],
    max_windows_per_file: Optional[int],
    max_total_windows: Optional[int],
) -> List[Tuple[int, int]]:
    indices: List[Tuple[int, int]] = []
    total_added = 0
    per_file_limit = max_windows_per_file if max_windows_per_file and max_windows_per_file > 0 else None
    total_limit = max_total_windows if max_total_windows and max_total_windows > 0 else None

    for idx, entry in enumerate(entries):
        num_windows = entry.get("num_windows", 0)
        if num_windows <= 0:
            continue
        cap = per_file_limit if per_file_limit is not None else num_windows
        cap = min(cap, num_windows) if per_file_limit is not None else num_windows
        for window_idx in range(cap):
            if total_limit is not None and total_added >= total_limit:
                break
            indices.append((idx, window_idx))
            total_added += 1
        if total_limit is not None and total_added >= total_limit:
            break
    return indices


class CachedMoEDataset(IterableDataset[Tuple[Dict[str, Tensor], Tensor]]):
    """Iterable dataset backed by cached tensors."""

    def __init__(
        self,
        cache_entries: Sequence[Dict[str, Any]],
        tasks: Sequence[str],
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 17,
        max_windows_per_file: Optional[int] = None,
        max_total_windows: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tasks = tuple(tasks)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self.log_fn: Callable[[str], None] = print
        self._window_budget: Optional[int] = max_total_windows

        processed_entries: List[Dict[str, Any]] = []
        self.stats: Dict[Path, Dict[str, Any]] = {}
        self.skipped: Dict[Path, str] = {}
        for idx, entry in enumerate(cache_entries):
            processed, stats_entry, reason = _prepare_entry(entry, idx)
            path = processed["path"]
            self.stats[path] = stats_entry
            if reason:
                self.skipped[path] = reason
                processed["num_windows"] = 0
            processed_entries.append(processed)
        self.cache_entries = processed_entries
        self._window_indices = _build_window_index(processed_entries, max_windows_per_file, max_total_windows)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def set_log_fn(self, fn: Optional[Callable[[str], None]]) -> None:
        self.log_fn = fn or (lambda _: None)

    def set_window_budget(self, budget: Optional[int]) -> None:
        self._window_budget = budget

    def _stack_batch(
        self,
        batch_features: Dict[str, List[Tensor]],
        batch_targets: Sequence[int],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        features = {key: torch.stack(values, dim=0) for key, values in batch_features.items()}
        targets = torch.tensor(list(batch_targets), dtype=torch.long)
        return features, targets

    def __iter__(self) -> Iterator[Tuple[Dict[str, Tensor], Tensor]]:
        if not self._window_indices:
            return iter([])
        rng = random.Random(self.seed + self._epoch)
        order = list(range(len(self._window_indices)))
        if self.shuffle and len(order) > 1:
            rng.shuffle(order)

        batch_features: Dict[str, List[Tensor]] = defaultdict(list)
        batch_targets: List[int] = []
        batch_sources: Set[Path] = set()

        for position in order:
            file_idx, window_idx = self._window_indices[position]
            entry = self.cache_entries[file_idx]
            path: Path = entry["path"]
            features = entry["features"]
            targets_tensor: Tensor = entry["targets"]

            window_features: Dict[str, Tensor] = {key: tensor[window_idx] for key, tensor in features.items()}
            truncated_flag = window_features.pop("_truncated", None)
            try:
                if "gating_input" not in window_features:
                    from gateway.data.datasets.gating import build_unified_gating

                    window_features["gating_input"] = build_unified_gating(window_features)
            except Exception as exc:  # pragma: no cover - defensive
                self.log_fn(f"[SkipWindow] Failed to assemble gating input from cache {path.name}: {exc}")
                continue
            if any(not isinstance(value, Tensor) for value in window_features.values()):
                self.log_fn(f"[SkipWindow] Cache {path.name}: missing tensor data; dropping window.")
                continue
            for key, value in window_features.items():
                batch_features[key].append(value)
            batch_targets.append(int(targets_tensor[window_idx].item()))

            self.stats[path]["windows"] += 1
            if truncated_flag is not None:
                self.stats[path]["truncated_windows"] += 1
            batch_sources.add(path)

            if len(batch_targets) >= self.batch_size:
                features_tensor, targets_tensor_out = self._stack_batch(batch_features, batch_targets)
                yield features_tensor, targets_tensor_out
                for source in batch_sources:
                    self.stats[source]["batches"] += 1
                batch_features = defaultdict(list)
                batch_targets = []
                batch_sources = set()

        if batch_targets:
            features_tensor, targets_tensor_out = self._stack_batch(batch_features, batch_targets)
            yield features_tensor, targets_tensor_out
            for source in batch_sources:
                self.stats[source]["batches"] += 1


__all__ = ["CachedMoEDataset"]
