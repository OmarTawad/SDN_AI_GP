"""Feature engineering aligned with the CNN detector."""

from __future__ import annotations

from collections import Counter
from math import log2
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..config.types import FeatureConfig
from ..data.structures import PacketRecord, Window


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator in (0, 0.0, None):
        return 0.0
    return float(numerator) / float(denominator)


def _entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        probability = value / total
        entropy -= probability * log2(probability)
    return float(entropy)


def _gini_from_vector(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    sorted_vals = np.sort(arr)
    cumulative = np.cumsum(sorted_vals, dtype=float)
    n = sorted_vals.size
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


def compute_static_features(
    win_rows: List[Dict],
    micro_bins: int,
    per_bin_total_pkts: np.ndarray,
    top_k_udp_ports: Sequence[int],
    window_sec: float,
) -> tuple[np.ndarray, List[str], Dict[str, float]]:
    """Replicate the per-window static features used by the CNN detector."""

    names: List[str] = []
    values: List[float] = []

    timestamps = np.array([float(row["ts"]) for row in win_rows], dtype=float)
    sizes = np.array([float(row.get("len") or 0.0) for row in win_rows], dtype=float)
    prot_udp = [row.get("is_udp", 0) == 1 for row in win_rows]
    prot_tcp = [row.get("is_tcp", 0) == 1 for row in win_rows]
    prot_icmp = [row.get("is_icmp", 0) == 1 for row in win_rows]

    pkts = len(win_rows)
    total_bytes = float(sizes.sum())
    W = float(window_sec)

    rate_features = {
        "pkts_per_s": pkts / W if W > 0 else 0.0,
        "bytes_per_s": total_bytes / W if W > 0 else 0.0,
        "tcp_per_s": (sum(prot_tcp)) / W if W > 0 else 0.0,
        "udp_per_s": (sum(prot_udp)) / W if W > 0 else 0.0,
        "icmp_per_s": (sum(prot_icmp)) / W if W > 0 else 0.0,
    }
    for key in ["pkts_per_s", "bytes_per_s", "tcp_per_s", "udp_per_s", "icmp_per_s"]:
        names.append(key)
        values.append(rate_features[key])

    for port in top_k_udp_ports:
        count = sum(
            1
            for row in win_rows
            if row.get("is_udp", 0) == 1 and row.get("dst_port") == port
        )
        names.append(f"udp_dst_{port}_per_s")
        values.append(count / W if W > 0 else 0.0)

    inter_arrivals = np.diff(np.sort(timestamps)) if timestamps.size >= 2 else np.array([])

    def _stat_or_zero(array: np.ndarray, func, default: float = 0.0) -> float:
        if array.size == 0:
            return default
        try:
            return float(func(array))
        except Exception:
            return default

    iat_mean = _stat_or_zero(inter_arrivals, np.mean)
    iat_std = _stat_or_zero(inter_arrivals, np.std)
    iat_cv = (iat_std / iat_mean) if iat_mean > 0 else 0.0
    iat_median = _stat_or_zero(inter_arrivals, np.median)
    iat_p95 = _stat_or_zero(inter_arrivals, lambda data: np.percentile(data, 95))
    iat_p99 = _stat_or_zero(inter_arrivals, lambda data: np.percentile(data, 99))
    for key, value in [
        ("iat_mean", iat_mean),
        ("iat_std", iat_std),
        ("iat_cv", iat_cv),
        ("iat_median", iat_median),
        ("iat_p95", iat_p95),
        ("iat_p99", iat_p99),
    ]:
        names.append(key)
        values.append(value)

    max_bin_pkts = float(np.max(per_bin_total_pkts)) if per_bin_total_pkts.size > 0 else 0.0
    gini = _gini_from_vector(per_bin_total_pkts)
    for key, value in [("max_bin_pkts", max_bin_pkts), ("gini_pkts", gini)]:
        names.append(key)
        values.append(value)

    src_ips = Counter([row.get("src_ip") for row in win_rows if row.get("src_ip")])
    dst_ips = Counter([row.get("dst_ip") for row in win_rows if row.get("dst_ip")])
    src_macs = Counter([row.get("src_mac") for row in win_rows if row.get("src_mac")])
    dst_ports = Counter([row.get("dst_port") for row in win_rows if row.get("dst_port") is not None])
    ttls = Counter([row.get("ttl") for row in win_rows if row.get("ttl") is not None])

    uniq_features = {
        "n_unique_src_ip": float(len(src_ips)),
        "n_unique_dst_ip": float(len(dst_ips)),
        "n_unique_src_mac": float(len(src_macs)),
        "H_src_ip": _entropy_from_counts(src_ips),
        "H_dst_port": _entropy_from_counts(dst_ports),
        "H_ttl": _entropy_from_counts(ttls),
    }
    for key in [
        "n_unique_src_ip",
        "n_unique_dst_ip",
        "n_unique_src_mac",
        "H_src_ip",
        "H_dst_port",
        "H_ttl",
    ]:
        names.append(key)
        values.append(uniq_features[key])

    tcp_syn = sum(int(row.get("tcp_syn", 0)) for row in win_rows)
    tcp_synack = sum(int(row.get("tcp_synack", 0)) for row in win_rows)
    tcp_ratio = _safe_div(tcp_syn, tcp_synack)

    ssdp_req = sum(
        1
        for row in win_rows
        if row.get("is_udp", 0) == 1 and row.get("ssdp_method") in {"M-SEARCH", "NOTIFY"}
    )
    ssdp_resp = sum(
        1
        for row in win_rows
        if row.get("is_udp", 0) == 1 and row.get("ssdp_method") == "200-OK"
    )
    ssdp_ratio = _safe_div(ssdp_req, ssdp_resp)

    dns_req = sum(
        1
        for row in win_rows
        if row.get("is_udp", 0) == 1 and row.get("dst_port") == 53
    )
    ntp_req = sum(
        1
        for row in win_rows
        if row.get("is_udp", 0) == 1 and row.get("dst_port") == 123
    )
    dns_resp = 0
    ntp_resp = 0
    dns_ratio = _safe_div(dns_req, dns_resp)
    ntp_ratio = _safe_div(ntp_req, ntp_resp)

    for key, value in [
        ("tcp_syn_over_synack", tcp_ratio),
        ("udp_ssdp_req_over_resp", ssdp_ratio),
        ("udp_dns_req_over_resp", dns_ratio),
        ("udp_ntp_req_over_resp", ntp_ratio),
    ]:
        names.append(key)
        values.append(value)

    size_mean = float(sizes.mean()) if sizes.size > 0 else 0.0
    size_std = float(sizes.std()) if sizes.size > 0 else 0.0
    if sizes.size > 0:
        bins = min(20, max(5, int(np.sqrt(sizes.size))))
        hist, edges = np.histogram(sizes, bins=bins)
        mode = float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2.0)
    else:
        mode = 0.0
    for key, value in [
        ("pkt_len_mean", size_mean),
        ("pkt_len_std", size_std),
        ("pkt_len_mode", mode),
    ]:
        names.append(key)
        values.append(value)

    udp_1900 = sum(
        1
        for row in win_rows
        if row.get("is_udp", 0) == 1 and row.get("dst_port") == 1900
    )
    udp_total = sum(1 for row in win_rows if row.get("is_udp", 0) == 1)
    udp_1900_frac = _safe_div(udp_1900, udp_total)

    ssdp_msearch = sum(1 for row in win_rows if row.get("ssdp_method") == "M-SEARCH")
    ssdp_notify = sum(1 for row in win_rows if row.get("ssdp_method") == "NOTIFY")
    ssdp_ok = sum(1 for row in win_rows if row.get("ssdp_method") == "200-OK")

    st_counts = Counter([row.get("ssdp_st") for row in win_rows if row.get("ssdp_st")])
    man_counts = Counter([row.get("ssdp_man") for row in win_rows if row.get("ssdp_man")])
    ua_counts = Counter(
        [row.get("ssdp_user_agent") for row in win_rows if row.get("ssdp_user_agent")]
    )

    multicast_hits = sum(
        1
        for row in win_rows
        if row.get("is_udp", 0) == 1 and row.get("dst_port") == 1900
    )
    multicast_rate = _safe_div(multicast_hits, max(1, udp_total))

    ssdp_features = {
        "udp_1900_fraction": udp_1900_frac,
        "ssdp_msearch": float(ssdp_msearch),
        "ssdp_notify": float(ssdp_notify),
        "ssdp_200ok": float(ssdp_ok),
        "H_ST": _entropy_from_counts(st_counts),
        "H_MAN": _entropy_from_counts(man_counts),
        "H_USER_AGENT": _entropy_from_counts(ua_counts),
        "ssdp_multicast_hit_rate": float(multicast_rate),
    }
    for key in [
        "udp_1900_fraction",
        "ssdp_msearch",
        "ssdp_notify",
        "ssdp_200ok",
        "H_ST",
        "H_MAN",
        "H_USER_AGENT",
        "ssdp_multicast_hit_rate",
    ]:
        names.append(key)
        values.append(ssdp_features[key])

    total_mac = sum(src_macs.values())
    mac_dom = max(src_macs.values()) / total_mac if total_mac > 0 else 0.0
    names.append("src_mac_dominance")
    values.append(float(mac_dom))

    vector = np.array(values, dtype=np.float32)
    snaps = {
        "pkts": float(pkts),
        "bytes": float(total_bytes),
        "max_bin_pkts": float(max_bin_pkts),
        "udp_1900_fraction": float(udp_1900_frac),
        "tcp_syn_over_synack": float(tcp_ratio),
        "udp_ssdp_req_over_resp": float(ssdp_ratio),
    }
    return vector, names, snaps


def _packet_to_row(packet: PacketRecord) -> Dict[str, object]:
    """Convert a PacketRecord into the flat dict expected by compute_static_features."""

    protocol = (packet.protocol or "").lower()
    tcp_flags = packet.tcp_flags or 0
    tcp_syn = 1 if tcp_flags & 0x02 else 0
    tcp_synack = 1 if tcp_flags & 0x12 == 0x12 else 0
    is_udp = 1 if protocol == "udp" else 0
    is_tcp = 1 if protocol == "tcp" else 0
    is_icmp = 1 if protocol == "icmp" else 0

    ssdp_method = str(packet.info.get("ssdp_method") or "NONE").upper()
    ssdp_st = packet.info.get("ssdp_st")
    ssdp_man = packet.info.get("ssdp_man")
    ssdp_user_agent = packet.info.get("ssdp_user_agent")

    return {
        "ts": float(packet.timestamp),
        "src_mac": packet.src_mac,
        "dst_mac": packet.dst_mac,
        "src_ip": packet.src_ip,
        "dst_ip": packet.dst_ip,
        "len": int(packet.length) if packet.length is not None else 0,
        "ttl": packet.ttl,
        "src_port": packet.src_port,
        "dst_port": packet.dst_port,
        "tcp_syn": tcp_syn,
        "tcp_synack": tcp_synack,
        "is_udp": is_udp,
        "is_tcp": is_tcp,
        "is_icmp": is_icmp,
        "udp_len": packet.payload_len if is_udp else None,
        "ssdp_method": ssdp_method,
        "ssdp_st": ssdp_st,
        "ssdp_man": ssdp_man,
        "ssdp_user_agent": ssdp_user_agent,
    }


def _per_bin_packet_counts(
    win_rows: Sequence[Dict],
    start_time: float,
    end_time: float,
    micro_bins: int,
) -> np.ndarray:
    if micro_bins <= 0:
        return np.zeros(0, dtype=float)
    counts = np.zeros(micro_bins, dtype=float)
    duration = max(end_time - start_time, 0.0)
    bin_width = duration / micro_bins if micro_bins > 0 else 0.0
    for row in win_rows:
        ts = float(row["ts"])
        if duration <= 0 or bin_width <= 0:
            index = 0
        else:
            index = int((ts - start_time) / bin_width)
            if index >= micro_bins:
                index = micro_bins - 1
            if index < 0:
                index = 0
        counts[index] += 1.0
    return counts


class FeatureExtractor:
    """Compute per-window static features compatible with the CNN detector."""

    def __init__(self, config: FeatureConfig, window_size: float) -> None:
        self.micro_bins = int(config.micro_bins)
        self.top_udp_ports = [int(port) for port in config.top_udp_ports]
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
        ordered_columns = ["window_index", "window_start", "window_end"] + (self._feature_names or [])
        return frame[ordered_columns]

    def _features_for_window(self, window: Window) -> Dict[str, float]:
        win_rows = [_packet_to_row(packet) for packet in window.packets]
        per_bin = _per_bin_packet_counts(
            win_rows,
            float(window.start_time),
            float(window.end_time),
            self.micro_bins,
        )
        vector, names, _ = compute_static_features(
            win_rows,
            self.micro_bins,
            per_bin,
            self.top_udp_ports,
            self.window_size,
        )
        if self._feature_names is None:
            self._feature_names = names
        elif self._feature_names != names:
            raise ValueError("Static feature order mismatch across windows.")
        return {name: float(value) for name, value in zip(names, vector.tolist())}


__all__ = ["FeatureExtractor", "compute_static_features"]
