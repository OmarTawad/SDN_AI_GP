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


def _nonzero(x: float) -> bool:
    return abs(float(x)) > 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="TC-5 ARP feature extraction check (structural features)")
    ap.add_argument("--pcap", required=True, help="Path to ARP spoofing .pcap")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"), help="Path to config.yaml")
    ap.add_argument("--window-index", type=int, default=0, help="Which window to verify (0-based)")
    ap.add_argument("--max-packets", type=int, default=None, help="Limit packets read from pcap")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    W = float(cfg["windowing"]["window_sec"])
    S = float(cfg["windowing"]["stride_sec"])
    M = int(cfg["windowing"]["micro_bins"])

    rows = _take(args.max_packets, iter_rows_from_pcap(args.pcap))
    windows = iter_windows(rows, W, S, M)

    target = None
    for i, win in enumerate(windows):
        if i == args.window_index:
            target = win
            break

    if target is None:
        print("TC-5 FAIL: no windows produced (check pcap or window settings)")
        return 1

    t0, t1, win_rows, bins = target
    if not win_rows:
        print("TC-5 FAIL: selected window has no packets")
        return 1

    seq_np, extras = compute_sequence_features(win_rows, bins, M)
    static_vec, static_names, _ = compute_static_features(win_rows, M, extras, W)
    static_map = dict(zip(static_names, static_vec))

    arp_pkts_per_s = float(static_map.get("arp_pkts_per_s", 0.0))
    arp_fraction = float(static_map.get("arp_fraction", 0.0))
    arp_request_rate = float(static_map.get("arp_request_rate", 0.0))
    arp_reply_rate = float(static_map.get("arp_reply_rate", 0.0))

    structural_keys = [
        "unique_sender_ips",
        "unique_sender_macs",
        "unique_target_ips",
        "unique_target_macs",
        "unique_sender_target_pairs",
        "max_claims_per_ip",
        "max_ips_per_mac",
        "sender_conflict_ratio",
        "conflict_ip_ratio",
        "reply_conflict_ratio",
        "target_reply_conflict_ratio",
    ]
    structural_vals = {k: float(static_map.get(k, 0.0)) for k in structural_keys}
    any_structural = any(_nonzero(v) for v in structural_vals.values())

    errors: List[str] = []
    if not _nonzero(arp_pkts_per_s):
        errors.append("arp_pkts_per_s is zero")
    if not any_structural:
        errors.append("structural ARP features are all zero")

    print("TC-5 ARP FEATURE CHECK")
    print(f"pcap: {args.pcap}")
    print(f"window_index: {args.window_index}")
    print(f"window: [{t0:.6f}, {t1:.6f}) W={W} S={S} M={M}")
    print(f"packets: {len(win_rows)}")
    print(
        "rates: "
        f"arp_pkts_per_s={arp_pkts_per_s:.3f}, arp_fraction={arp_fraction:.3f}, "
        f"arp_request_rate={arp_request_rate:.3f}, arp_reply_rate={arp_reply_rate:.3f}"
    )
    print("structural_features:")
    for k in structural_keys:
        print(f"- {k}: {structural_vals[k]:.3f}")

    if errors:
        print("TC-5 FAIL:")
        for e in errors:
            print(f"- {e}")
        return 1

    print("TC-5 PASS: structural ARP features populated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
