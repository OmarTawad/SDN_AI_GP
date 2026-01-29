#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List

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


def main() -> int:
    ap = argparse.ArgumentParser(description="TC-6 vector consistency check for mixed traffic pcap")
    ap.add_argument("--pcap", required=True, help="Path to .pcap file to verify")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"), help="Path to config.yaml")
    ap.add_argument("--max-packets", type=int, default=None, help="Limit packets read from pcap")
    ap.add_argument("--max-windows", type=int, default=10, help="Limit windows checked")
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

    base_k_seq = None
    base_k_static = None
    checked = 0
    errors: List[str] = []

    for win_idx, (t0, t1, win_rows, bins) in enumerate(windows):
        if not win_rows:
            continue
        seq_np, extras = compute_sequence_features(win_rows, bins, M, top_ports)
        static_vec, static_names, _ = compute_static_features(
            win_rows, M, extras["per_bin_total_pkts"], top_ports, W
        )

        if seq_np.shape[0] != M:
            errors.append(f"window {win_idx}: seq rows {seq_np.shape[0]} != M {M}")
        k_seq = seq_np.shape[1]
        k_static = int(static_vec.size)

        if base_k_seq is None:
            base_k_seq = k_seq
            base_k_static = k_static
        else:
            if k_seq != base_k_seq:
                errors.append(f"window {win_idx}: K_seq {k_seq} != {base_k_seq}")
            if k_static != base_k_static:
                errors.append(f"window {win_idx}: K_static {k_static} != {base_k_static}")

        checked += 1
        if checked >= args.max_windows:
            break

    if checked == 0:
        print("TC-6 FAIL: no windows with packets were produced")
        return 1

    print("TC-6 VECTOR CONSISTENCY CHECK")
    print(f"pcap: {args.pcap}")
    print(f"checked_windows: {checked}")
    print(f"expected: M={M}, K_seq={base_k_seq}, K_static={base_k_static}")

    if errors:
        print("TC-6 FAIL:")
        for e in errors:
            print(f"- {e}")
        return 1

    print("TC-6 PASS: fixed-length feature vectors across windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
