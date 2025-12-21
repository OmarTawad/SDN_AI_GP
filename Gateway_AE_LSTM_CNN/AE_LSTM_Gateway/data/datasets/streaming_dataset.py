"""Streaming MoE dataset for processing PCAP files into feature windows."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from gateway.core import class_id_to_name
from gateway.data.datasets.streaming_worker import stream_file
from gateway.data.structures.pcap import PcapInfo


class MoEDataset(IterableDataset[Tuple[Dict[str, Tensor], Tensor]]):
    """Iterable dataset that streams PCAP files and yields windowed tensors."""

    def __init__(
        self,
        files: Sequence[PcapInfo],
        tasks: Sequence[str],
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 17,
        max_windows_per_file: Optional[int] = None,
        max_total_windows: Optional[int] = None,
        status_interval: Optional[int] = None,
        max_file_size: Optional[int] = None,
        max_packets_per_file: Optional[int] = None,
        file_timeout: Optional[float] = None,
        max_packets_per_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.tasks = tuple(tasks)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self.max_windows_per_file = max_windows_per_file
        self.max_total_windows = max_total_windows
        self.max_file_size = max_file_size
        self.max_packets_per_file = max_packets_per_file
        self.file_timeout = file_timeout
        self.max_packets_per_window = max_packets_per_window
        self.total_windows_processed = 0
        self.log_every = status_interval if status_interval and status_interval > 0 else None
        self.log_fn: Callable[[str], None] = self._default_log
        self._next_log = self.log_every
        self._window_budget = max_total_windows
        self._window_cap_announced = False
        self._file_cap_announced: Set[Path] = set()
        self._seen_files: Set[Path] = set()
        self.skipped: Dict[Path, str] = {}
        self.stats: Dict[Path, Dict[str, Any]] = {}
        self.window_callback: Optional[Callable[[PcapInfo, int], None]] = None
        for info in self.files:
            self.stats[info.path] = {
                "windows": 0,
                "batches": 0,
                "label": int(info.label),
                "label_name": class_id_to_name(int(info.label)),
                "truncated_windows": 0,
            }

    @staticmethod
    def _default_log(message: str) -> None:
        print(message)

    def set_log_fn(self, fn: Optional[Callable[[str], None]]) -> None:
        self.log_fn = fn or self._default_log

    def set_window_budget(self, budget: Optional[int]) -> None:
        self._window_budget = budget

    def set_window_callback(self, fn: Optional[Callable[[PcapInfo, int]], None]) -> None:
        self.window_callback = fn

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self.total_windows_processed = 0
        self._window_cap_announced = False
        self._file_cap_announced.clear()
        self._seen_files.clear()
        self._next_log = self.log_every if self.log_every is not None else None
        self.skipped.clear()
        for stats in self.stats.values():
            stats["windows"] = 0
            stats["batches"] = 0
            stats.pop("skipped_reason", None)

    # ------------------------------------------------------------------ Logging helpers
    def log_file_start(self, info: PcapInfo) -> None:
        if info.path in self._seen_files:
            return
        self.log_fn(f"[Stream] {info.path.name}: label={class_id_to_name(int(info.label))}")
        self._seen_files.add(info.path)

    def log_progress(self, info: PcapInfo, windows_for_file: int) -> None:
        if self.log_every is None:
            return
        if self._next_log is None:
            self._next_log = self.log_every
        while self._next_log is not None and self.total_windows_processed >= self._next_log:
            message = f"[WindowProgress] total={self.total_windows_processed}"
            if self._window_budget:
                pct = min(100.0, (self.total_windows_processed / self._window_budget) * 100.0)
                message += f"/{self._window_budget} ({pct:4.1f}%)"
            message += f" file={info.path.name} file_windows={windows_for_file}"
            self.log_fn(message)
            self._next_log += self.log_every

    def note_file_cap(self, info: PcapInfo, windows_for_file: int) -> None:
        if (
            self.max_windows_per_file is None
            or windows_for_file < self.max_windows_per_file
            or info.path in self._file_cap_announced
        ):
            return
        self._file_cap_announced.add(info.path)
        self.log_fn(
            f"[WindowBudget] Reached per-file cap ({self.max_windows_per_file} windows) on {info.path.name}; switching captures."
        )

    def note_window_cap(self, info: PcapInfo) -> None:
        if self.max_total_windows is None or self._window_cap_announced:
            return
        if self.total_windows_processed >= self.max_total_windows:
            self._window_cap_announced = True
            self.log_fn(
                f"[WindowBudget] Global window cap ({self.max_total_windows}) reached while reading {info.path.name}; halting stream."
            )

    def mark_skip(self, info: PcapInfo, reason: str) -> None:
        record = self.stats.get(info.path)
        if record is None or record.get("skipped_reason"):
            return
        record["skipped_reason"] = reason
        self.skipped[info.path] = reason
        self.log_fn(f"[Skip] {info.path.name}: {reason}")

    def notify_windows(self, info: PcapInfo, increment: int) -> None:
        if increment <= 0 or self.window_callback is None:
            return
        try:
            self.window_callback(info, increment)
        except Exception:
            pass

    def stack_batch(
        self,
        batch_features: Dict[str, List[Tensor]],
        batch_targets: Sequence[int],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        features = {key: torch.stack(values, dim=0) for key, values in batch_features.items()}
        targets = torch.tensor(list(batch_targets), dtype=torch.long)
        return features, targets

    def __iter__(self) -> Iterator[Tuple[Dict[str, Tensor], Tensor]]:
        rng = random.Random(self.seed + self._epoch)
        indices = list(range(len(self.files)))
        if self.shuffle and len(indices) > 1:
            rng.shuffle(indices)
        for idx in indices:
            yield from stream_file(self, self.files[idx])

__all__ = ["MoEDataset"]
