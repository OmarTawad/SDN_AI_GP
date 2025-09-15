"""Window construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .structures import PacketRecord, Window


@dataclass
class WindowingParams:
    window_size: float
    hop_size: float
    max_windows: int | None = None


class WindowBuilder:
    """Create fixed-duration windows from packets."""

    def __init__(self, params: WindowingParams) -> None:
        self.params = params

    def build(self, packets: Sequence[PacketRecord]) -> List[Window]:
        if not packets:
            return []
        sorted_packets = sorted(packets, key=lambda pkt: pkt.timestamp)
        first_ts = sorted_packets[0].timestamp
        last_ts = sorted_packets[-1].timestamp
        windows: List[Window] = []
        index = 0
        window_start = first_ts
        window_size = self.params.window_size
        hop = self.params.hop_size
        max_windows = self.params.max_windows

        while window_start <= last_ts:
            window_end = window_start + window_size
            bucket = [
                pkt
                for pkt in sorted_packets
                if window_start <= pkt.timestamp < window_end
            ]
            windows.append(
                Window(
                    index=index,
                    start_time=window_start,
                    end_time=window_end,
                    packets=bucket,
                )
            )
            index += 1
            if max_windows is not None and index >= max_windows:
                break
            window_start += hop
        return windows
