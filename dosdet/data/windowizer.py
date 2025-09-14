from __future__ import annotations
from typing import Iterable, Dict, Tuple, List, Iterator
import math

def iter_windows(
    rows: Iterable[Dict],
    window_sec: float,
    stride_sec: float,
    micro_bins: int,
) -> Iterator[Tuple[float, float, List[Dict], List[int]]]:
    """
    Consume a time-ordered iterable of packet rows and yield sliding windows.
    Returns (t_start, t_end, rows_in_window, bin_index_per_row).
    - Window size W, stride S
    - Micro-bins M across each window; bin width = W/M
    Edge rule: include packet with ts in [t_start, t_end), i.e., end-exclusive.
    """
    buffer: List[Dict] = []
    t0: float | None = None
    W = float(window_sec)
    S = float(stride_sec)
    bw = W / micro_bins

    for r in rows:
        ts = float(r["ts"])
        if t0 is None:
            t0 = ts
        # drop old packets < current window start
        # we maintain a moving window start `ws` aligned to first window
        # but since we don't know future, we push rows and pop later on boundary
        buffer.append(r)
        # While we can emit windows:
        while t0 is not None and ts - t0 >= W:
            # Emit window [t0, t0+W)
            win_rows = [x for x in buffer if t0 <= float(x["ts"]) < t0 + W]
            bin_idx = []
            for x in win_rows:
                dt = float(x["ts"]) - t0
                b = int(math.floor(dt / bw))
                if b >= micro_bins:
                    b = micro_bins - 1
                if b < 0:
                    b = 0
                bin_idx.append(b)
            yield t0, t0 + W, win_rows, bin_idx
            # advance start by stride; evict older than new start
            t0 += S
            buffer = [x for x in buffer if float(x["ts"]) >= t0]

    # Flush remaining windows after input ends
    if t0 is not None and buffer:
        last_ts = float(buffer[-1]["ts"])
        # continue emitting until the start surpasses last_ts
        while last_ts - t0 >= 0:
            win_rows = [x for x in buffer if t0 <= float(x["ts"]) < t0 + W]
            if not win_rows:
                break
            bin_idx = []
            bw = W / micro_bins
            for x in win_rows:
                dt = float(x["ts"]) - t0
                b = int(math.floor(dt / bw))
                if b >= micro_bins:
                    b = micro_bins - 1
                if b < 0:
                    b = 0
                bin_idx.append(b)
            yield t0, t0 + W, win_rows, bin_idx
            t0 += S
            buffer = [x for x in buffer if float(x["ts"]) >= t0]
