#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.pcap_reader import iter_rows_from_pcap
from data.windowizer import iter_windows
from features.seq_features import compute_sequence_features
from features.static_features import compute_static_features


def _take(n: int | None, it: Iterable[dict]) -> Iterable[dict]:
    if n is None:
        yield from it
        return
    for i, row in enumerate(it):
        if i >= n:
            break
        yield row


def _nonzero(x: float) -> bool:
    return abs(float(x)) > 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="TC-4 feature extraction check for a DoS pcap")
    ap.add_argument("--pcap", required=True, help="Path to .pcap file to verify")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"), help="Path to config.yaml")
    ap.add_argument("--window-index", type=int, default=0, help="Which window to verify (0-based)")
    ap.add_argument("--max-packets", type=int, default=None, help="Limit packets read from pcap")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    W = float(cfg["windowing"]["window_sec"])
    S = float(cfg["windowing"]["stride_sec"])
    M = int(cfg["windowing"]["micro_bins"])
    top_ports = list(cfg["data"]["top_k_udp_ports"])
    ssdp_v4 = cfg["features"]["ssdp_multicast_ipv4"]
    ssdp_v6 = cfg["features"]["ssdp_multicast_ipv6"]

    rows = _take(args.max_packets, iter_rows_from_pcap(args.pcap, ssdp_v4, ssdp_v6))
    windows = iter_windows(rows, W, S, M)

    target = None
    for i, win in enumerate(windows):
        if i == args.window_index:
            target = win
            break

    if target is None:
        print("TC-4 FAIL: no windows produced (check pcap or window settings)")
        return 1

    t0, t1, win_rows, bins = target
    if not win_rows:
        print("TC-4 FAIL: selected window has no packets")
        return 1

    seq_np, extras = compute_sequence_features(win_rows, bins, M, top_ports)
    static_vec, static_names, _ = compute_static_features(
        win_rows, M, extras["per_bin_total_pkts"], top_ports, W
    )
    static_map = dict(zip(static_names, static_vec))

    # Volumetric checks
    pkts_per_s = float(static_map.get("pkts_per_s", 0.0))
    bytes_per_s = float(static_map.get("bytes_per_s", 0.0))
    udp_per_s = float(static_map.get("udp_per_s", 0.0))
    tcp_per_s = float(static_map.get("tcp_per_s", 0.0))
    icmp_per_s = float(static_map.get("icmp_per_s", 0.0))
    max_bin_pkts = float(static_map.get("max_bin_pkts", 0.0))

    per_bin_pkts = extras["per_bin_total_pkts"]
    per_bin_bytes = extras["per_bin_total_bytes"]
    seq_total_pkts = float(np.sum(per_bin_pkts))
    seq_total_bytes = float(np.sum(per_bin_bytes))

    errors: List[str] = []
    if not _nonzero(pkts_per_s):
        errors.append("pkts_per_s is zero")
    if not _nonzero(bytes_per_s):
        errors.append("bytes_per_s is zero")
    if not (_nonzero(udp_per_s) or _nonzero(tcp_per_s) or _nonzero(icmp_per_s)):
        errors.append("all protocol rates are zero")
    if not _nonzero(max_bin_pkts):
        errors.append("max_bin_pkts is zero")
    if not _nonzero(seq_total_pkts):
        errors.append("sequence per-bin total pkts sum is zero")
    if not _nonzero(seq_total_bytes):
        errors.append("sequence per-bin total bytes sum is zero")

    print("TC-4 FEATURE CHECK")
    print(f"pcap: {args.pcap}")
    print(f"window_index: {args.window_index}")
    print(f"window: [{t0:.6f}, {t1:.6f}) W={W} S={S} M={M}")
    print(f"packets: {len(win_rows)}")
    print(
        "volumetric_static: "
        f"pkts_per_s={pkts_per_s:.3f}, bytes_per_s={bytes_per_s:.3f}, "
        f"udp_per_s={udp_per_s:.3f}, tcp_per_s={tcp_per_s:.3f}, icmp_per_s={icmp_per_s:.3f}, "
        f"max_bin_pkts={max_bin_pkts:.3f}"
    )
    print(f"seq_totals: per_bin_pkts_sum={seq_total_pkts:.1f}, per_bin_bytes_sum={seq_total_bytes:.1f}")

    if errors:
        print("TC-4 FAIL:")
        for e in errors:
            print(f"- {e}")
        return 1

    print("TC-4 PASS: feature extraction with non-zero volumetric features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
