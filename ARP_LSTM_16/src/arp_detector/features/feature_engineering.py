"""Feature engineering tailored for ARP spoofing detection."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config.types import FeatureConfig
from ..data.structures import PacketRecord, Window


def _safe_div(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _arp_interarrival_stats(timestamps: Sequence[float]) -> Tuple[float, float]:
    if len(timestamps) < 2:
        return 0.0, 0.0
    diffs = np.diff(np.sort(np.asarray(timestamps, dtype=float)))
    return float(np.mean(diffs)), float(np.std(diffs))


def compute_static_features(
    win_rows: List[Dict[str, object]],
    window_sec: float,
) -> tuple[np.ndarray, List[str], Dict[str, float]]:
    """Compute ARP-centric per-window features."""

    pkts = len(win_rows)
    window_width = float(window_sec) if window_sec > 0 else 1.0

    total_bytes = 0.0
    arp_bytes = 0.0
    arp_packets = 0
    arp_requests = 0
    arp_replies = 0
    gratuitous = 0

    sender_ips: set[str] = set()
    sender_macs: set[str] = set()
    target_ips: set[str] = set()
    target_macs: set[str] = set()
    sender_target_pairs: set[Tuple[str, str]] = set()
    arp_timestamps: List[float] = []

    ip_to_macs: defaultdict[str, set[str]] = defaultdict(set)
    mac_to_ips: defaultdict[str, set[str]] = defaultdict(set)
    target_ip_to_macs: defaultdict[str, set[str]] = defaultdict(set)

    for row in win_rows:
        length = float(row.get("len") or 0.0)
        total_bytes += length
        if row.get("is_arp", 0):
            arp_packets += 1
            arp_bytes += length
            opcode = int(row.get("arp_opcode") or 0)
            arp_requests += int(opcode == 1)
            arp_replies += int(opcode == 2)
            gratuitous += int(bool(row.get("arp_is_gratuitous")))
            sender_ip = row.get("arp_sender_ip")
            sender_mac = row.get("arp_sender_mac")
            target_ip = row.get("arp_target_ip")
            target_mac = row.get("arp_target_mac")
            ts_val = row.get("ts")
            if isinstance(ts_val, (int, float)):
                arp_timestamps.append(float(ts_val))
            if isinstance(sender_ip, str):
                sender_ips.add(sender_ip)
                if isinstance(sender_mac, str):
                    ip_to_macs[sender_ip].add(sender_mac)
                    sender_target_pairs.add((sender_ip, target_ip or ""))
            if isinstance(sender_mac, str):
                sender_macs.add(sender_mac)
                if isinstance(sender_ip, str):
                    mac_to_ips[sender_mac].add(sender_ip)
            if isinstance(target_ip, str):
                target_ips.add(target_ip)
                if isinstance(sender_mac, str):
                    target_ip_to_macs[target_ip].add(sender_mac)
            if isinstance(target_mac, str):
                target_macs.add(target_mac)

    pkts_per_s = _safe_div(pkts, window_width)
    bytes_per_s = _safe_div(total_bytes, window_width)
    arp_pkts_per_s = _safe_div(arp_packets, window_width)
    arp_bytes_per_s = _safe_div(arp_bytes, window_width)
    arp_request_rate = _safe_div(arp_requests, window_width)
    arp_reply_rate = _safe_div(arp_replies, window_width)
    arp_gratuitous_rate = _safe_div(gratuitous, window_width)
    arp_fraction = _safe_div(arp_packets, pkts)
    gratuitous_fraction = _safe_div(gratuitous, arp_packets)
    reply_to_request_ratio = _safe_div(arp_replies, arp_requests)
    request_to_reply_ratio = _safe_div(arp_requests, arp_replies)
    arp_iat_mean, arp_iat_std = _arp_interarrival_stats(arp_timestamps)

    claim_cardinality = [len(macs) for macs in ip_to_macs.values()]
    max_claims_per_ip = max(claim_cardinality, default=0)
    mean_claims_per_ip = float(np.mean(claim_cardinality)) if claim_cardinality else 0.0
    conflict_ips = sum(1 for value in claim_cardinality if value > 1)
    conflict_ip_ratio = _safe_div(conflict_ips, len(claim_cardinality))

    mac_cardinality = [len(ips) for ips in mac_to_ips.values()]
    max_ips_per_mac = max(mac_cardinality, default=0)
    mean_ips_per_mac = float(np.mean(mac_cardinality)) if mac_cardinality else 0.0
    conflict_macs = sum(1 for value in mac_cardinality if value > 1)
    sender_conflict_ratio = _safe_div(conflict_macs, len(mac_cardinality))

    target_conflicts = [len(macs) for macs in target_ip_to_macs.values()]
    target_conflict_ratio = _safe_div(sum(1 for value in target_conflicts if value > 1), len(target_conflicts))

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
        ("target_conflict_ratio", float(target_conflict_ratio)),
    ]

    vector = np.array([value for _, value in features], dtype=np.float32)
    names = [name for name, _ in features]

    snapshots = {
        "pkts": float(pkts),
        "arp_packets": float(arp_packets),
        "arp_requests": float(arp_requests),
        "arp_replies": float(arp_replies),
        "conflict_ip_ratio": float(conflict_ip_ratio),
        "max_claims_per_ip": float(max_claims_per_ip),
        "sender_conflict_ratio": float(sender_conflict_ratio),
    }
    return vector, names, snapshots


def _packet_to_row(packet: PacketRecord) -> Dict[str, object]:
    return {
        "ts": float(packet.timestamp),
        "len": packet.length,
        "is_arp": 1 if packet.protocol.lower() == "arp" else 0,
        "arp_opcode": packet.arp_opcode,
        "arp_sender_ip": packet.arp_sender_ip,
        "arp_sender_mac": packet.arp_sender_mac,
        "arp_target_ip": packet.arp_target_ip,
        "arp_target_mac": packet.arp_target_mac,
        "arp_is_gratuitous": packet.arp_is_gratuitous,
    }


class FeatureExtractor:
    """Compute per-window ARP spoofing features."""

    def __init__(self, config: FeatureConfig, window_size: float) -> None:
        self.window_size = float(window_size)
        self._feature_names: List[str] | None = None

    def extract(self, windows: Sequence[Window]) -> pd.DataFrame:
        records: List[Dict[str, float]] = []
        for window in windows:
            features = self._features_for_window(window)
            features["window_index"] = int(window.index)
            features["window_start"] = float(window.start_time)
            features["window_end"] = float(window.end_time)
            records.append(features)

        if not records:
            return pd.DataFrame(columns=["window_index", "window_start", "window_end"] + (self._feature_names or []))

        frame = pd.DataFrame(records)
        ordered = ["window_index", "window_start", "window_end"] + (self._feature_names or [])
        return frame[ordered]

    def _features_for_window(self, window: Window) -> Dict[str, float]:
        win_rows = [_packet_to_row(packet) for packet in window.packets]
        vector, names, _ = compute_static_features(win_rows, self.window_size)
        if self._feature_names is None:
            self._feature_names = names
        elif self._feature_names != names:
            raise ValueError("Static feature order mismatch across windows.")
        return {name: float(value) for name, value in zip(names, vector.tolist())}


__all__ = ["FeatureExtractor", "compute_static_features"]
