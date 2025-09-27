from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Generator, Iterable, List, Optional


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if value < self.min_value:
            self.min_value = value
        if value > self.max_value:
            self.max_value = value

    def mean_value(self) -> float:
        return self.mean if self.count > 0 else 0.0

    def std_value(self) -> float:
        if self.count < 2:
            return 0.0
        variance = self.m2 / (self.count - 1)
        return math.sqrt(max(variance, 0.0))

    def min(self) -> float:
        return self.min_value if self.count > 0 else 0.0

    def max(self) -> float:
        return self.max_value if self.count > 0 else 0.0


@dataclass
class PacketSummary:
    timestamp: float
    length: int
    protocol: str
    src_ip: Optional[str]
    dst_ip: Optional[str]
    src_port: Optional[int]
    dst_port: Optional[int]
    tcp_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class WindowStats:
    start: float
    end: float
    index: int
    duration: float
    packet_count: int = 0
    byte_count: int = 0
    length_stats: RunningStats = field(default_factory=RunningStats)
    iat_stats: RunningStats = field(default_factory=RunningStats)
    last_packet_ts: Optional[float] = None
    proto_counts: Counter = field(default_factory=Counter)
    tcp_flag_counts: Counter = field(default_factory=Counter)
    src_ips: Counter = field(default_factory=Counter)
    dst_ips: Counter = field(default_factory=Counter)
    src_ports: Counter = field(default_factory=Counter)
    dst_ports: Counter = field(default_factory=Counter)

    def add_packet(self, packet: PacketSummary) -> None:
        self.packet_count += 1
        self.byte_count += packet.length
        self.length_stats.update(packet.length)

        if self.last_packet_ts is not None:
            self.iat_stats.update(packet.timestamp - self.last_packet_ts)
        self.last_packet_ts = packet.timestamp

        self.proto_counts.update([packet.protocol])

        if packet.protocol == "TCP":
            for flag, present in packet.tcp_flags.items():
                if present:
                    self.tcp_flag_counts.update([flag])

        if packet.src_ip:
            self.src_ips.update([packet.src_ip])
        if packet.dst_ip:
            self.dst_ips.update([packet.dst_ip])
        if packet.src_port is not None:
            self.src_ports.update([packet.src_port])
        if packet.dst_port is not None:
            self.dst_ports.update([packet.dst_port])


class SlidingWindowManager:
    def __init__(self, window_seconds: float, stride_seconds: float) -> None:
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.active: Deque[WindowStats] = deque()
        self.next_window_start: Optional[float] = None
        self.window_index: int = 0

    def _align_timestamp(self, timestamp: float) -> float:
        stride = self.stride_seconds
        if stride <= 0:
            return timestamp
        return math.floor(timestamp / stride) * stride

    def add_packet(self, packet: PacketSummary) -> Generator[WindowStats, None, None]:
        ts = packet.timestamp
        if self.next_window_start is None:
            self.next_window_start = self._align_timestamp(ts)

        # Create windows that should start before or at current timestamp
        while self.next_window_start is not None and self.next_window_start <= ts:
            start = self.next_window_start
            end = start + self.window_seconds
            self.active.append(
                WindowStats(
                    start=start,
                    end=end,
                    index=self.window_index,
                    duration=self.window_seconds,
                )
            )
            self.window_index += 1
            self.next_window_start += self.stride_seconds

        # Flush windows that ended before current timestamp
        while self.active and self.active[0].end <= ts:
            yield self.active.popleft()

        # Update active windows that include the packet
        for window in self.active:
            if window.start <= ts < window.end:
                window.add_packet(packet)

    def finalize(self) -> Iterable[WindowStats]:
        while self.active:
            yield self.active.popleft()
