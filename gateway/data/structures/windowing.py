"""Windowing utilities and sequence state trackers for MoE datasets.

----
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor


def _entropy(counter: Counter) -> float:
    """Compute entropy for a multiset represented by a counter."""

    total = float(sum(counter.values()))
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        probability = float(count) / total
        value -= probability * math.log2(probability)
    return value


@dataclass
class AutoFeatureAccumulator:
    """Streaming accumulator that computes aggregate autoencoder statistics."""

    duration: float
    count: int = 0
    byte_count: float = 0.0
    mean_len: float = 0.0
    m2_len: float = 0.0
    min_len: float = field(default_factory=lambda: float("inf"))
    max_len: float = 0.0
    last_ts: Optional[float] = None
    iat_count: int = 0
    iat_mean: float = 0.0
    iat_m2: float = 0.0
    proto_counts: Counter = field(default_factory=Counter)
    tcp_flag_counts: Counter = field(default_factory=Counter)
    src_ips: Counter = field(default_factory=Counter)
    dst_ips: Counter = field(default_factory=Counter)
    src_ports: Counter = field(default_factory=Counter)
    dst_ports: Counter = field(default_factory=Counter)

    def add(
        self,
        row: Dict[str, object],
        length: float,
        timestamp: float,
        tcp_flags: Optional[int],
    ) -> None:
        """Update accumulators with a single packet."""

        self.count += 1
        self.byte_count += float(length)
        delta = length - self.mean_len
        self.mean_len += delta / self.count
        self.m2_len += delta * (length - self.mean_len)
        self.min_len = min(self.min_len, length)
        self.max_len = max(self.max_len, length)

        if self.last_ts is not None:
            inter_arrival = timestamp - self.last_ts
            self.iat_count += 1
            delta_iat = inter_arrival - self.iat_mean
            self.iat_mean += delta_iat / self.iat_count
            self.iat_m2 += delta_iat * (inter_arrival - self.iat_mean)
        self.last_ts = timestamp

        proto = row.get("protocol")
        if isinstance(proto, str):
            proto = proto.upper()
        if proto in {"TCP", "UDP", "ICMP"}:
            self.proto_counts.update([proto])

        if tcp_flags is not None:
            if tcp_flags & 0x02:
                self.tcp_flag_counts.update(["SYN"])
            if tcp_flags & 0x10:
                self.tcp_flag_counts.update(["ACK"])
            if tcp_flags & 0x04:
                self.tcp_flag_counts.update(["RST"])
            if tcp_flags & 0x01:
                self.tcp_flag_counts.update(["FIN"])

        src_ip = row.get("src_ip")
        dst_ip = row.get("dst_ip")
        src_port = row.get("src_port")
        dst_port = row.get("dst_port")

        if src_ip:
            self.src_ips.update([str(src_ip)])
        if dst_ip:
            self.dst_ips.update([str(dst_ip)])
        if src_port is not None:
            self.src_ports.update([int(src_port)])
        if dst_port is not None:
            self.dst_ports.update([int(dst_port)])

    def finalize(self) -> Optional[Dict[str, float]]:
        """Materialise the statistics accumulated over the window."""

        if self.count == 0:
            return None
        mean_iat = self.iat_mean if self.iat_count else 0.0
        std_iat = (
            math.sqrt(self.iat_m2 / max(self.iat_count - 1, 1)) if self.iat_count > 1 else 0.0
        )
        std_len = math.sqrt(self.m2_len / max(self.count - 1, 1)) if self.count > 1 else 0.0
        pkt_count = float(self.count)
        duration = max(self.duration, 1e-6)
        tcp_count = float(self.proto_counts.get("TCP", 0))
        udp_count = float(self.proto_counts.get("UDP", 0))
        icmp_count = float(self.proto_counts.get("ICMP", 0))

        return {
            "pkt_count": pkt_count,
            "byte_count": float(self.byte_count),
            "pps": pkt_count / duration,
            "bps": (8.0 * float(self.byte_count)) / duration,
            "mean_pkt_len": float(self.mean_len),
            "std_pkt_len": std_len,
            "min_pkt_len": 0.0 if math.isinf(self.min_len) else float(self.min_len),
            "max_pkt_len": float(self.max_len),
            "mean_iat": mean_iat,
            "std_iat": std_iat,
            "tcp_count": tcp_count,
            "udp_count": udp_count,
            "icmp_count": icmp_count,
            "tcp_syn": float(self.tcp_flag_counts.get("SYN", 0)),
            "tcp_ack": float(self.tcp_flag_counts.get("ACK", 0)),
            "tcp_rst": float(self.tcp_flag_counts.get("RST", 0)),
            "tcp_fin": float(self.tcp_flag_counts.get("FIN", 0)),
            "src_ip_entropy": _entropy(self.src_ips),
            "dst_ip_entropy": _entropy(self.dst_ips),
            "src_port_entropy": _entropy(self.src_ports),
            "dst_port_entropy": _entropy(self.dst_ports),
            "unique_src_ips": float(len(self.src_ips)),
            "unique_dst_ips": float(len(self.dst_ips)),
            "unique_src_ports": float(len(self.src_ports)),
            "unique_dst_ports": float(len(self.dst_ports)),
            "tcp_ratio": tcp_count / max(pkt_count, 1.0),
            "udp_ratio": udp_count / max(pkt_count, 1.0),
            "icmp_ratio": icmp_count / max(pkt_count, 1.0),
        }


@dataclass
class WindowBuffer:
    """Buffer collecting packets for a sliding window."""

    start: float
    end: float
    index: int
    duration: float
    micro_bins: Dict[str, int]
    max_packets: Optional[int] = None
    auto_acc: AutoFeatureAccumulator = field(init=False)
    task_rows: Dict[str, List[Dict[str, object]]] = field(init=False)
    bin_indices: Dict[str, List[int]] = field(init=False)
    packet_count: int = field(default=0, init=False)
    truncated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.auto_acc = AutoFeatureAccumulator(duration=self.duration)
        self.task_rows = {name: [] for name in self.micro_bins}
        self.bin_indices = {name: [] for name in self.micro_bins}

    def add(
        self,
        rows: Dict[str, Dict[str, object]],
        auto_row: Dict[str, object],
        timestamp: float,
        length: float,
        tcp_flags: Optional[int],
        bin_index_map: Dict[str, int],
    ) -> None:
        """Insert packet details into the buffer."""

        if self.max_packets is not None and self.packet_count >= self.max_packets:
            self.truncated = True
            return
        self.auto_acc.add(auto_row, length, timestamp, tcp_flags)
        self.packet_count += 1
        for key, indices in self.bin_indices.items():
            row = rows.get(key)
            self.task_rows[key].append(row or {})
            indices.append(bin_index_map.get(key, 0))


class StreamingWindowManager:
    """Sliding window manager that segments packets into fixed-duration buffers."""

    def __init__(
        self,
        window_size: float,
        stride: float,
        micro_bins: Dict[str, int],
        max_packets_per_window: Optional[int],
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.micro_bins = {name: int(count) for name, count in micro_bins.items()}
        self.bin_widths = {
            name: (window_size / max(count, 1)) if count > 0 else window_size
            for name, count in self.micro_bins.items()
        }
        self.next_window_start: Optional[float] = None
        self.active: Deque[WindowBuffer] = deque()
        self.index = 0
        self.max_packets_per_window = max_packets_per_window

    def _open_window(self, start: float) -> None:
        buffer = WindowBuffer(
            start=start,
            end=start + self.window_size,
            index=self.index,
            duration=self.window_size,
            micro_bins=self.micro_bins,
            max_packets=self.max_packets_per_window,
        )
        self.index += 1
        self.active.append(buffer)

    def add_packet(
        self,
        rows: Dict[str, Dict[str, object]],
        auto_row: Dict[str, object],
        timestamp: float,
        length: float,
        tcp_flags: Optional[int],
    ) -> List[WindowBuffer]:
        """Feed a packet to active windows, returning completed buffers."""

        completed: List[WindowBuffer] = []
        if self.next_window_start is None:
            stride = self.stride if self.stride > 0 else self.window_size
            self.next_window_start = math.floor(timestamp / stride) * stride

        while self.next_window_start is not None and self.next_window_start <= timestamp:
            self._open_window(self.next_window_start)
            self.next_window_start += self.stride

        while self.active and self.active[0].end <= timestamp:
            completed.append(self.active.popleft())

        for window in self.active:
            if window.start <= timestamp < window.end:
                offset = timestamp - window.start
                bin_index_map: Dict[str, int] = {}
                for name, count in self.micro_bins.items():
                    width = self.bin_widths.get(name, self.window_size)
                    if count <= 0 or width <= 0:
                        idx = 0
                    else:
                        idx = max(0, min(count - 1, int(offset / width)))
                    bin_index_map[name] = idx
                window.add(rows, auto_row, timestamp, length, tcp_flags, bin_index_map)
        return completed

    def flush(self) -> List[WindowBuffer]:
        """Return remaining buffers and clear internal state."""

        remaining = list(self.active)
        self.active.clear()
        return remaining


@dataclass
class SequenceState:
    """Fixed-length sequence container used by LSTM feature builders."""

    sequence_length: int
    feature_names: Optional[List[str]] = None
    buffer: Deque[Tensor] = field(init=False)

    def __post_init__(self) -> None:
        self.buffer = deque(maxlen=self.sequence_length)

    def update(self, features: Dict[str, float], scaler) -> Tensor:
        """Append new feature vector to the sequence, returning a Tensor window."""

        if self.feature_names is None:
            self.feature_names = list(features.keys())
        vector = np.array(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float32,
        )
        scaled = scaler.transform(vector.reshape(1, -1)).astype(np.float32)
        tensor = torch.from_numpy(scaled.squeeze(0))
        self.buffer.append(tensor)
        if len(self.buffer) < self.sequence_length:
            pad_count = self.sequence_length - len(self.buffer)
            pad_tensor = torch.zeros_like(tensor)
            tensors = [pad_tensor for _ in range(pad_count)] + list(self.buffer)
        else:
            tensors = list(self.buffer)
        return torch.stack(tensors, dim=0)


__all__ = [
    "AutoFeatureAccumulator",
    "SequenceState",
    "StreamingWindowManager",
    "WindowBuffer",
]

