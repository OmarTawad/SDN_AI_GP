#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Iterable, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
from data.pcap_reader import iter_rows_from_pcap
from data.windowizer import iter_windows


def _take(n: int | None, it: Iterable[dict]) -> Iterable[dict]:
    if n is None:
        yield from it
        return
    for i, row in enumerate(it):
        if i >= n:
            break
        yield row


def _verify_window(t0: float, t1: float, rows: List[dict], bins: List[int], W: float, M: int) -> List[str]:
    errors: List[str] = []
    if not rows:
        errors.append("window has no packets")
        return errors
    if not math.isclose(t1 - t0, W, rel_tol=1e-9, abs_tol=1e-12):
        errors.append(f"window length {t1 - t0:.6f} != W {W:.6f}")
    if len(rows) != len(bins):
        errors.append(f"bins length {len(bins)} != rows length {len(rows)}")
    bw = W / M
    for idx, (row, b) in enumerate(zip(rows, bins)):
        ts = row.get("ts")
        if ts is None:
            errors.append(f"row {idx} missing ts")
            continue
        if not (t0 <= ts < t1):
            errors.append(f"row {idx} ts {ts:.6f} out of window [{t0:.6f}, {t1:.6f})")
        expected = int(math.floor((ts - t0) / bw))
        if expected < 0:
            expected = 0
        if expected >= M:
            expected = M - 1
        if b != expected:
            errors.append(f"row {idx} ts {ts:.6f} bin {b} != expected {expected}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="TC-2 window segmentation check for a single pcap")
    ap.add_argument("--pcap", required=True, help="Path to .pcap file to verify")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"), help="Path to config.yaml")
    ap.add_argument("--window-index", type=int, default=0, help="Which window to verify (0-based)")
    ap.add_argument("--max-packets", type=int, default=None, help="Limit packets read from pcap")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    W = float(cfg["windowing"]["window_sec"])
    S = float(cfg["windowing"]["stride_sec"])
    M = int(cfg["windowing"]["micro_bins"])
    ssdp_v4 = cfg["features"]["ssdp_multicast_ipv4"]
    ssdp_v6 = cfg["features"]["ssdp_multicast_ipv6"]

    rows = _take(args.max_packets, iter_rows_from_pcap(args.pcap, ssdp_v4, ssdp_v6))
    windows = iter_windows(rows, W, S, M)

    target = None
    next_win = None
    for i, win in enumerate(windows):
        if i == args.window_index:
            target = win
        elif i == args.window_index + 1:
            next_win = win
            break
        if i > args.window_index + 1:
            break

    if target is None:
        print("TC-2 FAIL: no windows produced (check pcap or window settings)")
        return 1

    t0, t1, win_rows, bins = target
    errors = _verify_window(t0, t1, win_rows, bins, W, M)
    if next_win is not None:
        n0, n1, _, _ = next_win
        if not math.isclose(n0 - t0, S, rel_tol=1e-9, abs_tol=1e-12):
            errors.append(f"next window start {n0:.6f} != t0+S {t0 + S:.6f}")
        if not math.isclose(n1 - t1, S, rel_tol=1e-9, abs_tol=1e-12):
            errors.append(f"next window end {n1:.6f} != t1+S {t1 + S:.6f}")

    ts_preview = [f"{r['ts']:.6f}" for r in win_rows[:10]]
    print("TC-2 WINDOW CHECK")
    print(f"pcap: {args.pcap}")
    print(f"window_index: {args.window_index}")
    print(f"window: [{t0:.6f}, {t1:.6f}) W={W} S={S} M={M}")
    print(f"packets: {len(win_rows)}")
    print(f"ts_preview: {', '.join(ts_preview)}")
    print(f"bins_preview: {bins[:10]}")

    if errors:
        print("TC-2 FAIL:")
        for e in errors:
            print(f"- {e}")
        return 1

    print("TC-2 PASS: window segmentation, timestamped packets, correct bin assignment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
