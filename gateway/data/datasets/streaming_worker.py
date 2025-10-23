"""Worker routines that drive the streaming MoE dataset.


"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import torch
from scapy.utils import PcapReader
from torch import Tensor

from gateway.data.datasets.assembly import assemble_window_features
from gateway.data.datasets.gating import build_unified_gating
from gateway.data.datasets.packet import packet_to_auto
from gateway.data.extractors.features import (
    SSDP_MULTICAST_V4,
    SSDP_MULTICAST_V6,
    ARP_LSTM_SEQUENCE_LENGTH,
    ARP_MICRO_BINS,
    DOS_LSTM_SEQUENCE_LENGTH,
    DOS_MICRO_BINS,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    arp_scapy_pkt_to_row,
    dos_scapy_pkt_to_row,
)
from gateway.data.structures.pcap import PcapInfo
from gateway.data.structures.windowing import SequenceState, StreamingWindowManager

if TYPE_CHECKING:  # pragma: no cover
    from gateway.data.datasets.streaming_dataset import MoEDataset


def _fallback_arp_row(pkt) -> Dict[str, object]:
    return {
        "ts": float(getattr(pkt, "time", 0.0)),
        "len": int(len(pkt)),
        "is_arp": 0,
        "arp_opcode": 0,
        "arp_sender_ip": None,
        "arp_sender_mac": None,
        "arp_target_ip": None,
        "arp_target_mac": None,
        "arp_is_gratuitous": 0,
    }


def stream_file(dataset: "MoEDataset", info: PcapInfo) -> Iterator[Tuple[Dict[str, Tensor], Tensor]]:
    if not info.path.exists():
        return
    if dataset.max_total_windows is not None and dataset.total_windows_processed >= dataset.max_total_windows:
        return

    if dataset.max_file_size is not None:
        try:
            file_size = info.path.stat().st_size
        except FileNotFoundError:
            return
        if file_size > dataset.max_file_size:
            size_mb = file_size / (1024 * 1024)
            limit_mb = dataset.max_file_size / (1024 * 1024)
            dataset.mark_skip(info, f"size {size_mb:.1f}MB exceeds limit {limit_mb:.1f}MB")
            return

    dataset.log_file_start(info)

    micro_bins: Dict[str, int] = {}
    if "dos" in dataset.tasks:
        micro_bins["dos"] = DOS_MICRO_BINS
    if "arp" in dataset.tasks:
        micro_bins["arp"] = ARP_MICRO_BINS

    manager = StreamingWindowManager(WINDOW_SIZE, WINDOW_STRIDE, micro_bins, dataset.max_packets_per_window)
    dos_state = SequenceState(sequence_length=DOS_LSTM_SEQUENCE_LENGTH) if "dos" in dataset.tasks else None
    arp_state = SequenceState(sequence_length=ARP_LSTM_SEQUENCE_LENGTH) if "arp" in dataset.tasks else None

    batch_features: Dict[str, List[Tensor]] = defaultdict(list)
    batch_targets: List[int] = []
    batch_sources: Set[Path] = set()
    windows_for_file = 0
    drop_reason: Optional[str] = None
    packets_seen = 0
    start_time = time.monotonic()

    def flush_batch() -> Iterator[Tuple[Dict[str, Tensor], Tensor]]:
        nonlocal batch_features, batch_targets, batch_sources
        if not batch_targets:
            return
        yield dataset.stack_batch(batch_features, batch_targets)
        dataset.stats[info.path]["batches"] += 1
        batch_features = defaultdict(list)
        batch_targets = []
        batch_sources = set()

    def push_window(window_features: Dict[str, Tensor]) -> Iterator[Tuple[Dict[str, Tensor], Tensor]]:
        nonlocal windows_for_file
        truncated_flag = window_features.pop("_truncated", None)
        try:
            window_features.setdefault("gating_input", build_unified_gating(window_features))
        except Exception as exc:  # pragma: no cover - defensive
            dataset.log_fn(f"[SkipWindow] {info.path.name}: gating assembly failed ({exc}).")
            return iter(())
        for key, value in window_features.items():
            if not isinstance(value, Tensor):
                dataset.log_fn(f"[SkipWindow] {info.path.name}: feature '{key}' missing tensor data; dropping window.")
                return iter(())
            batch_features[key].append(value)
        batch_targets.append(int(info.label))
        dataset.stats[info.path]["windows"] += 1
        if truncated_flag is not None:
            dataset.stats[info.path]["truncated_windows"] += 1
        windows_for_file += 1
        dataset.total_windows_processed += 1
        batch_sources.add(info.path)
        dataset.notify_windows(info, 1)
        dataset.log_progress(info, windows_for_file)
        if len(batch_targets) >= dataset.batch_size:
            return flush_batch()
        return iter(())

    with PcapReader(str(info.path)) as reader:
        for pkt in reader:
            if dataset.file_timeout is not None and (time.monotonic() - start_time) >= dataset.file_timeout:
                drop_reason = f"time budget {dataset.file_timeout:.1f}s exceeded"
                break
            if dataset.max_packets_per_file is not None and packets_seen >= dataset.max_packets_per_file:
                drop_reason = f"packet cap {dataset.max_packets_per_file} reached"
                break
            packets_seen += 1
            if dataset.max_total_windows is not None and dataset.total_windows_processed >= dataset.max_total_windows:
                break

            dos_row = None
            if "dos" in dataset.tasks:
                try:
                    dos_row = dos_scapy_pkt_to_row(pkt, SSDP_MULTICAST_V4, SSDP_MULTICAST_V6)
                except Exception:
                    dos_row = None
            arp_row = arp_scapy_pkt_to_row(pkt) if "arp" in dataset.tasks else None
            fallback_arp_row = arp_row or _fallback_arp_row(pkt)
            timestamp, tcp_flags, auto_row = packet_to_auto(pkt, fallback_arp_row)
            length = float(len(pkt))
            rows: Dict[str, Dict[str, object]] = {}
            if dos_row is not None:
                rows["dos"] = dos_row
            if arp_row is not None:
                rows["arp"] = arp_row
            completed = manager.add_packet(rows, auto_row, timestamp, length, tcp_flags)
            for buffer in completed:
                features = assemble_window_features(buffer, dataset.tasks, dos_state, arp_state)
                if features is None:
                    continue
                for batch in push_window(features):
                    yield batch
                if (
                    dataset.max_windows_per_file is not None
                    and windows_for_file >= dataset.max_windows_per_file
                ):
                    dataset.note_file_cap(info, windows_for_file)
                    break
                if (
                    dataset.max_total_windows is not None
                    and dataset.total_windows_processed >= dataset.max_total_windows
                ):
                    dataset.note_window_cap(info)
                    break
            else:
                continue
            break

    for buffer in manager.flush():
        features = assemble_window_features(buffer, dataset.tasks, dos_state, arp_state)
        if features is None:
            continue
        for batch in push_window(features):
            yield batch

    if drop_reason is not None:
        batch_features.clear()
        batch_targets.clear()
        if dataset.stats[info.path]["batches"] == 0:
            dataset.total_windows_processed = max(0, dataset.total_windows_processed - windows_for_file)
            dataset.stats[info.path]["windows"] = 0
        dataset.mark_skip(info, drop_reason)
        return

    for batch in flush_batch():
        yield batch
