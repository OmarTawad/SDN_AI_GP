from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

def _safe_div(num: float, den: float) -> float:
    if den in (0.0, 0, None):
        return 0.0
    return float(num) / float(den)

def _arp_interarrival_stats(timestamps: List[float]) -> Tuple[float, float]:
    if len(timestamps) < 2:
        return 0.0, 0.0
    arr = np.diff(np.sort(np.asarray(timestamps, dtype=float)))
    return float(np.mean(arr)), float(np.std(arr))

def compute_static_features(
    win_rows: List[Dict],
    micro_bins: int,
    extras: Dict[str, np.ndarray],
    window_sec: float,
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """
    Compute per-window ARP-centric static features.
    """
    window_width = float(window_sec) if window_sec > 0 else 1.0
    total_packets = len(win_rows)
    total_bytes = 0.0

    arp_packets = 0
    arp_requests = 0
    arp_replies = 0
    gratuitous = 0
    broadcast_requests = 0

    sender_ips: set[str] = set()
    sender_macs: set[str] = set()
    target_ips: set[str] = set()
    target_macs: set[str] = set()
    sender_target_pairs: set[Tuple[str, str]] = set()
    arp_timestamps: List[float] = []

    mac_to_ips: defaultdict[str, set[str]] = defaultdict(set)
    ip_to_macs: defaultdict[str, set[str]] = defaultdict(set)
    reply_mac_to_ips: defaultdict[str, set[str]] = defaultdict(set)
    target_ip_reply_macs: defaultdict[str, set[str]] = defaultdict(set)

    for row in win_rows:
        length = float(row.get("len") or 0.0)
        total_bytes += length

        if int(row.get("is_arp", 0)) != 1:
            continue

        arp_packets += 1
        opcode = int(row.get("arp_opcode") or 0)
        arp_requests += int(opcode == 1)
        arp_replies += int(opcode == 2)
        gratuitous += int(row.get("arp_is_gratuitous", 0) == 1)

        sender_ip = row.get("arp_sender_ip")
        sender_mac = row.get("arp_sender_mac")
        target_ip = row.get("arp_target_ip")
        target_mac = row.get("arp_target_mac")

        if isinstance(row.get("ts"), (int, float)):
            arp_timestamps.append(float(row["ts"]))

        if sender_ip:
            sender_ips.add(sender_ip)
        if sender_mac:
            sender_macs.add(sender_mac)
        if target_ip:
            target_ips.add(target_ip)
        if target_mac:
            target_macs.add(target_mac)

        pair = (sender_ip or "", target_ip or "")
        sender_target_pairs.add(pair)

        if sender_mac and sender_ip:
            mac_to_ips[sender_mac].add(sender_ip)
        if target_ip and sender_mac:
            ip_to_macs[target_ip].add(sender_mac)
        if opcode == 2 and sender_mac and sender_ip:
            reply_mac_to_ips[sender_mac].add(sender_ip)
        if opcode == 2 and target_ip and sender_mac:
            target_ip_reply_macs[target_ip].add(sender_mac)

        if opcode == 1:
            if (target_mac or "").lower() in {"00:00:00:00:00:00", "ff:ff:ff:ff:ff:ff"}:
                broadcast_requests += 1

    pkts_per_s = _safe_div(total_packets, window_width)
    bytes_per_s = _safe_div(total_bytes, window_width)
    arp_pkts_per_s = _safe_div(arp_packets, window_width)
    arp_bytes_per_s = _safe_div(total_bytes if arp_packets == total_packets else total_bytes * (arp_packets / max(total_packets, 1)), window_width)
    arp_request_rate = _safe_div(arp_requests, window_width)
    arp_reply_rate = _safe_div(arp_replies, window_width)
    arp_gratuitous_rate = _safe_div(gratuitous, window_width)
    arp_fraction = _safe_div(arp_packets, total_packets)
    gratuitous_fraction = _safe_div(gratuitous, arp_packets)
    broadcast_fraction = _safe_div(broadcast_requests, max(arp_requests, 1))

    reply_to_request_ratio = _safe_div(arp_replies, max(arp_requests, 1))
    request_to_reply_ratio = _safe_div(arp_requests, max(arp_replies, 1))
    arp_iat_mean, arp_iat_std = _arp_interarrival_stats(arp_timestamps)

    claims_per_ip = [len(macs) for macs in ip_to_macs.values()]
    max_claims_per_ip = max(claims_per_ip, default=0)
    mean_claims_per_ip = float(np.mean(claims_per_ip)) if claims_per_ip else 0.0
    conflict_ip_ratio = _safe_div(sum(1 for c in claims_per_ip if c > 1), len(claims_per_ip))

    ips_per_mac = [len(ips) for ips in mac_to_ips.values()]
    max_ips_per_mac = max(ips_per_mac, default=0)
    mean_ips_per_mac = float(np.mean(ips_per_mac)) if ips_per_mac else 0.0
    sender_conflict_ratio = _safe_div(sum(1 for c in ips_per_mac if c > 1), len(ips_per_mac))

    reply_conflict_ratio = _safe_div(sum(1 for ips in reply_mac_to_ips.values() if len(ips) > 1), len(reply_mac_to_ips))
    target_reply_conflicts = _safe_div(sum(1 for macs in target_ip_reply_macs.values() if len(macs) > 1), len(target_ip_reply_macs))

    per_bin_total = np.asarray(extras.get("per_bin_total_pkts", np.zeros(micro_bins)), dtype=float)
    per_bin_conf_sender = np.asarray(extras.get("per_bin_conflicting_sender_claims", np.zeros(micro_bins)), dtype=float)
    per_bin_conf_target = np.asarray(extras.get("per_bin_conflicting_target_claims", np.zeros(micro_bins)), dtype=float)

    max_bin_pkts = float(per_bin_total.max()) if per_bin_total.size else 0.0
    median_bin_pkts = float(np.median(per_bin_total)) if per_bin_total.size else 0.0
    max_conflicting_senders_bin = float(per_bin_conf_sender.max()) if per_bin_conf_sender.size else 0.0
    max_conflicting_targets_bin = float(per_bin_conf_target.max()) if per_bin_conf_target.size else 0.0

    features = [
        ("pkts_per_s", pkts_per_s),
        ("bytes_per_s", bytes_per_s),
        ("arp_pkts_per_s", arp_pkts_per_s),
        ("arp_bytes_per_s", arp_bytes_per_s),
        ("arp_fraction", arp_fraction),
        ("arp_request_rate", arp_request_rate),
        ("arp_reply_rate", arp_reply_rate),
        ("arp_gratuitous_rate", arp_gratuitous_rate),
        ("arp_gratuitous_fraction", gratuitous_fraction),
        ("arp_broadcast_fraction", broadcast_fraction),
        ("arp_reply_to_request_ratio", reply_to_request_ratio),
        ("arp_request_to_reply_ratio", request_to_reply_ratio),
        ("arp_iat_mean", arp_iat_mean),
        ("arp_iat_std", arp_iat_std),
        ("unique_sender_ips", float(len(sender_ips))),
        ("unique_sender_macs", float(len(sender_macs))),
        ("unique_target_ips", float(len(target_ips))),
        ("unique_target_macs", float(len(target_macs))),
        ("unique_sender_target_pairs", float(len(sender_target_pairs))),
        ("max_claims_per_ip", float(max_claims_per_ip)),
        ("mean_claims_per_ip", float(mean_claims_per_ip)),
        ("conflict_ip_ratio", float(conflict_ip_ratio)),
        ("max_ips_per_mac", float(max_ips_per_mac)),
        ("mean_ips_per_mac", float(mean_ips_per_mac)),
        ("sender_conflict_ratio", float(sender_conflict_ratio)),
        ("reply_conflict_ratio", float(reply_conflict_ratio)),
        ("target_reply_conflict_ratio", float(target_reply_conflicts)),
        ("max_bin_pkts", float(max_bin_pkts)),
        ("median_bin_pkts", float(median_bin_pkts)),
        ("max_conflicting_senders_bin", float(max_conflicting_senders_bin)),
        ("max_conflicting_targets_bin", float(max_conflicting_targets_bin)),
    ]

    vec = np.array([value for _, value in features], dtype=np.float32)
    names = [name for name, _ in features]
    snaps = {
        "pkts": float(total_packets),
        "arp_packets": float(arp_packets),
        "arp_requests": float(arp_requests),
        "arp_replies": float(arp_replies),
        "sender_conflict_ratio": float(sender_conflict_ratio),
        "conflict_ip_ratio": float(conflict_ip_ratio),
        "reply_conflict_ratio": float(reply_conflict_ratio),
        "max_claims_per_ip": float(max_claims_per_ip),
        "max_ips_per_mac": float(max_ips_per_mac),
        "max_bin_pkts": float(max_bin_pkts),
        "median_bin_pkts": float(median_bin_pkts),
        "broadcast_requests": float(broadcast_requests),
        "unique_sender_macs": float(len(sender_macs)),
        "unique_target_ips": float(len(target_ips)),
        "arp_reply_rate": float(arp_reply_rate),
        "arp_request_rate": float(arp_request_rate),
        "broadcast_fraction": float(broadcast_fraction),
    }
    return vec, names, snaps
