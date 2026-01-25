#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import resource
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


def _rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def main() -> int:
    ap = argparse.ArgumentParser(description="TC-7 scalability check for large pcap")
    ap.add_argument("--pcap", required=True, help="Path to .pcap file to verify")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"), help="Path to config.yaml")
    ap.add_argument("--max-packets", type=int, default=None, help="Limit packets read from pcap")
    ap.add_argument("--max-windows", type=int, default=0, help="Limit windows checked (0 = no limit)")
    ap.add_argument("--progress-every", type=int, default=200, help="Print progress every N windows")
    ap.add_argument("--max-rss-mb", type=float, default=None, help="Fail if RSS exceeds this MB")
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

    windows_processed = 0
    packets_processed = 0
    max_rss = 0.0

    for win_idx, (t0, t1, win_rows, bins) in enumerate(windows):
        if not win_rows:
            continue
        packets_processed += len(win_rows)
        seq_np, extras = compute_sequence_features(win_rows, bins, M, top_ports)
        compute_static_features(win_rows, M, extras["per_bin_total_pkts"], top_ports, W)

        windows_processed += 1
        rss = _rss_mb()
        if rss > max_rss:
            max_rss = rss

        if args.max_rss_mb is not None and rss > args.max_rss_mb:
            print(f"TC-7 FAIL: RSS {rss:.1f} MB exceeded limit {args.max_rss_mb:.1f} MB")
            return 1

        if args.progress_every > 0 and windows_processed % args.progress_every == 0:
            print(
                f"progress windows={windows_processed} packets={packets_processed} "
                f"rss_mb={rss:.1f} max_rss_mb={max_rss:.1f}"
            )

        if args.max_windows and windows_processed >= args.max_windows:
            break

    if windows_processed == 0:
        print("TC-7 FAIL: no windows with packets were produced")
        return 1

    print("TC-7 SCALABILITY CHECK")
    print(f"pcap: {args.pcap}")
    print(f"windows: {windows_processed}")
    print(f"packets: {packets_processed}")
    print(f"max_rss_mb: {max_rss:.1f}")
    print("TC-7 PASS: processed without memory overflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
