<<<<<<< HEAD
#src/dos_detector/features/feature_engineering.py
"""Feature engineering for PCAP windows."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, MutableMapping, Sequence
=======
"""Feature engineering aligned with the CNN detector."""

from __future__ import annotations

from collections import Counter
from math import log2
from typing import Dict, List, Sequence
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7

import numpy as np
import pandas as pd

from ..config.types import FeatureConfig
<<<<<<< HEAD
from ..data.structures import Window


@dataclass
class HostHistory:
    """Maintains rolling statistics for a host."""

    counts: Deque[float]
    maxlen: int

    def update(self, value: float) -> float:
        if len(self.counts) >= self.maxlen:
            self.counts.popleft()
        self.counts.append(value)
        return value

    def zscore(self, value: float) -> float:
        if not self.counts:
            return 0.0
        arr = np.fromiter(self.counts, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-9:
            return 0.0
        return (value - mean) / std


def _entropy_from_counts(counts: Iterable[int], base: float, eps: float) -> float:
    arr = np.fromiter((c for c in counts if c > 0), dtype=float)
    if arr.size == 0:
        return 0.0
    probs = arr / arr.sum()
    logs = np.log(probs + eps) / np.log(base)
    return float(-np.sum(probs * logs))


def _gini_from_counts(counts: Iterable[int]) -> float:
    arr = np.sort(np.fromiter((c for c in counts if c > 0), dtype=float))
    if arr.size == 0:
        return 0.0
    n = arr.size
    cumulative = np.cumsum(arr)
=======
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
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


<<<<<<< HEAD
def _top_k_share(counts: Iterable[int], k: int) -> float:
    arr = np.sort(np.fromiter((c for c in counts if c > 0), dtype=float))
    if arr.size == 0:
        return 0.0
    total = arr.sum()
    top = arr[-k:].sum()
    return float(top / total)


def _coefficient_of_variation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = float(values.mean())
    if mean == 0:
        return 0.0
    return float(values.std(ddof=0) / mean)


def _fano_factor(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = float(values.mean())
    if mean == 0:
        return 0.0
    return float(values.var(ddof=0) / mean)


def _multi_scale_variance(timestamps: List[float], start: float, end: float, bins: Sequence[int]) -> float:
    if not timestamps:
        return 0.0
    duration = end - start
    values: List[float] = []
    for bin_count in bins:
        if bin_count <= 0:
            continue
        counts = np.zeros(bin_count, dtype=float)
        for ts in timestamps:
            ratio = (ts - start) / duration if duration > 0 else 0
            index = min(bin_count - 1, max(0, int(ratio * bin_count)))
            counts[index] += 1
        if counts.mean() > 0:
            values.append(float(counts.var(ddof=0) / (counts.mean() ** 2)))
    if not values:
        return 0.0
    return float(np.mean(values))


def _herfindahl_index(counts: Iterable[int]) -> float:
    arr = np.fromiter((c for c in counts if c > 0), dtype=float)
    if arr.size == 0:
        return 0.0
    shares = arr / arr.sum()
    return float(np.sum(shares**2))


def _percentiles(values: np.ndarray, percentiles: Sequence[float]) -> Dict[str, float]:
    if values.size == 0:
        return {f"p{int(p)}": 0.0 for p in percentiles}
    results = np.percentile(values, percentiles)
    return {f"p{int(p)}": float(val) for p, val in zip(percentiles, results)}


class FeatureExtractor:
    """Compute features for each window."""

    def __init__(self, config: FeatureConfig, window_size: float) -> None:
        self.config = config
        self.window_size = window_size

    def extract(self, windows: Sequence[Window]) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        prev_features: Dict[str, float] | None = None
        global_rate_history: Deque[float] = deque(maxlen=self.config.rolling_zscore_window)
        host_histories: Dict[str, HostHistory] = defaultdict(
            lambda: HostHistory(deque(maxlen=self.config.host_history), self.config.host_history)
        )

        for window in windows:
            features = self._features_for_window(
                window=window,
                prev_features=prev_features,
                global_rate_history=global_rate_history,
                host_histories=host_histories,
            )
            rows.append(features)
            prev_features = features
            global_rate_history.append(features["packet_rate"])
        frame = pd.DataFrame(rows)
        frame.insert(0, "window_index", [w.index for w in windows])
        frame.insert(1, "window_start", [w.start_time for w in windows])
        frame.insert(2, "window_end", [w.end_time for w in windows])
        return frame

    def _features_for_window(
        self,
        window: Window,
        prev_features: Dict[str, float] | None,
        global_rate_history: Deque[float],
        host_histories: MutableMapping[str, HostHistory],
    ) -> Dict[str, float]:
        packets = window.packets
        packet_count = float(len(packets))
        byte_count = float(sum(pkt.length for pkt in packets))
        packet_rate = packet_count / max(self.window_size, self.config.eps)
        byte_rate = byte_count / max(self.window_size, self.config.eps)

        timestamps = np.array([pkt.timestamp for pkt in packets], dtype=float)
        inter_arrivals = np.diff(np.sort(timestamps)) if timestamps.size > 1 else np.array([], dtype=float)
        inter_stats = {
            "inter_mean": float(inter_arrivals.mean()) if inter_arrivals.size else 0.0,
            "inter_var": float(inter_arrivals.var()) if inter_arrivals.size else 0.0,
            "inter_cv": _coefficient_of_variation(inter_arrivals),
        }
        inter_stats.update(_percentiles(inter_arrivals, [10, 50, 90]))

        src_ip_counts = Counter(pkt.src_ip for pkt in packets if pkt.src_ip)
        src_mac_counts = Counter(pkt.src_mac for pkt in packets if pkt.src_mac)
        dst_ip_counts = Counter(pkt.dst_ip for pkt in packets if pkt.dst_ip)
        dst_port_counts = Counter(pkt.dst_port for pkt in packets if pkt.dst_port is not None)
        per_sender_counts = Counter(pkt.src_ip or pkt.src_mac for pkt in packets if pkt.src_ip or pkt.src_mac)

        per_sender_array = np.fromiter(per_sender_counts.values(), dtype=float)
        per_sender_rate = per_sender_array / max(self.window_size, self.config.eps) if per_sender_array.size else np.array([], dtype=float)

        features: Dict[str, float] = {
            "packet_count": packet_count,
            "byte_count": byte_count,
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "unique_src_ips": float(len(src_ip_counts)),
            "unique_dst_ips": float(len(dst_ip_counts)),
            "unique_dst_ports": float(len(dst_port_counts)),
            "unique_src_macs": float(len(src_mac_counts)),
            "src_ip_entropy": _entropy_from_counts(src_ip_counts.values(), self.config.entropy_base, self.config.eps),
            "src_mac_entropy": _entropy_from_counts(src_mac_counts.values(), self.config.entropy_base, self.config.eps),
            "gini_src": _gini_from_counts(per_sender_counts.values()),
            "fano_factor": _fano_factor(per_sender_array) if per_sender_array.size else 0.0,
            "multi_scale_var": _multi_scale_variance(list(timestamps), window.start_time, window.end_time, self.config.multi_scale_bins),
            "top1_src_share": _top_k_share(per_sender_counts.values(), 1),
            "top3_src_share": _top_k_share(per_sender_counts.values(), 3),
            "top5_src_share": _top_k_share(per_sender_counts.values(), 5),
            "per_sender_rate_mean": float(per_sender_rate.mean()) if per_sender_rate.size else 0.0,
            "per_sender_rate_max": float(per_sender_rate.max()) if per_sender_rate.size else 0.0,
        }
        features.update(inter_stats)

        protocol_features = self._protocol_features(window)
        features.update(protocol_features)

        if self.config.enable_change_features and prev_features is not None:
            features.update(self._change_features(features, prev_features))
        else:
            features.update({
                "delta_packet_rate": 0.0,
                "delta_byte_rate": 0.0,
                "delta_unique_src_ips": 0.0,
            })

        zscores = self._host_zscores(per_sender_counts, host_histories)
        features.update(zscores)

        global_z = self._global_rate_zscore(packet_rate, global_rate_history)
        features["global_packet_rate_zscore"] = global_z
        return features

    def _change_features(
        self,
        current: Dict[str, float],
        previous: Dict[str, float],
    ) -> Dict[str, float]:
        return {
            "delta_packet_rate": current["packet_rate"] - previous.get("packet_rate", 0.0),
            "delta_byte_rate": current["byte_rate"] - previous.get("byte_rate", 0.0),
            "delta_unique_src_ips": current["unique_src_ips"] - previous.get("unique_src_ips", 0.0),
        }

    def _host_zscores(
        self,
        per_sender_counts: Counter[str],
        host_histories: MutableMapping[str, HostHistory],
    ) -> Dict[str, float]:
        if not self.config.enable_host_zscores:
            return {
                "max_host_packet_rate_zscore": 0.0,
                "mean_host_packet_rate_zscore": 0.0,
            }
        zscores: List[float] = []
        for host, count in per_sender_counts.items():
            history = host_histories[host]
            z = history.zscore(float(count))
            history.update(float(count))
            zscores.append(z)
        if not zscores:
            return {
                "max_host_packet_rate_zscore": 0.0,
                "mean_host_packet_rate_zscore": 0.0,
            }
        arr = np.array(zscores, dtype=float)
        return {
            "max_host_packet_rate_zscore": float(arr.max()),
            "mean_host_packet_rate_zscore": float(arr.mean()),
        }

    def _global_rate_zscore(self, value: float, history: Deque[float]) -> float:
        if len(history) < 2:
            return 0.0
        arr = np.fromiter(history, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-9:
            return 0.0
        return (value - mean) / std

    def _protocol_features(self, window: Window) -> Dict[str, float]:
        packets = window.packets
        if not self.config.enable_protocol_features:
            return {
                "udp_1900_rate": 0.0,
                "ssdp_share": 0.0,
                "ssdp_multicast_fraction": 0.0,
                "ssdp_ipv6_multicast_fraction": 0.0,
                "ssdp_notify_count": 0.0,
                "ssdp_msearch_count": 0.0,
                "tcp_syn_rate": 0.0,
                "tcp_syn_ack_ratio": 0.0,
                "tcp_half_open_surplus": 0.0,
                "tcp_retransmit_proxy": 0.0,
                "icmp_echo_rate": 0.0,
                "icmp_size_cv": 0.0,
                "icmp_ttl_variance": 0.0,
                "udp_unique_ports": 0.0,
                "udp_port_herfindahl": 0.0,
                "udp_top_port_share": 0.0,
                "udp_payload_size_variance": 0.0,
            }

        ssdp_packets = [
            pkt for pkt in packets if pkt.protocol == "udp" and (pkt.dst_port == 1900 or pkt.src_port == 1900)
        ]
        udp_packets = [pkt for pkt in packets if pkt.protocol == "udp"]
        tcp_packets = [pkt for pkt in packets if pkt.protocol == "tcp"]
        icmp_packets = [pkt for pkt in packets if pkt.protocol == "icmp"]

        ssdp_multicast = [
            pkt
            for pkt in ssdp_packets
            if pkt.dst_ip in {"239.255.255.250", "ff02::c"} or pkt.src_ip in {"239.255.255.250", "ff02::c"}
        ]
        ssdp_share = float(len(ssdp_packets)) / max(len(packets), 1)
        ssdp_sender_counts = Counter(pkt.src_ip or pkt.src_mac for pkt in ssdp_packets if pkt.src_ip or pkt.src_mac)
        ssdp_rates = np.fromiter(ssdp_sender_counts.values(), dtype=float) / max(self.window_size, self.config.eps) if ssdp_sender_counts else np.array([], dtype=float)

        tcp_syn_packets = [pkt for pkt in tcp_packets if pkt.tcp_flags is not None and pkt.tcp_flags & 0x02]
        tcp_ack_packets = [pkt for pkt in tcp_packets if pkt.tcp_flags is not None and pkt.tcp_flags & 0x10]
        half_open_surplus = float(len(tcp_syn_packets) - len(tcp_ack_packets))
        retransmit_proxy = 0.0
        if tcp_syn_packets:
            syn_pairs = Counter((pkt.src_ip, pkt.dst_ip, pkt.dst_port) for pkt in tcp_syn_packets)
            retransmit_proxy = float(sum(count - 1 for count in syn_pairs.values() if count > 1))

        icmp_echo = [pkt for pkt in icmp_packets if pkt.info.get("icmp_type") in {"8", "128"}]
        icmp_sizes = np.fromiter((pkt.length for pkt in icmp_echo), dtype=float)
        icmp_ttls = np.fromiter((pkt.ttl for pkt in icmp_echo if pkt.ttl is not None), dtype=float)

        udp_port_counts = Counter(pkt.dst_port for pkt in udp_packets if pkt.dst_port is not None)
        udp_payloads = np.fromiter((pkt.payload_len for pkt in udp_packets), dtype=float)

        return {
            "udp_1900_rate": float(len(ssdp_packets)) / max(self.window_size, self.config.eps),
            "ssdp_share": ssdp_share,
            "ssdp_multicast_fraction": float(len(ssdp_multicast)) / max(len(ssdp_packets), 1),
            "ssdp_ipv6_multicast_fraction": float(
                len([pkt for pkt in ssdp_packets if pkt.dst_ip == "ff02::c" or pkt.src_ip == "ff02::c"])
            )
            / max(len(ssdp_packets), 1),
            "ssdp_notify_count": float(sum(1 for pkt in ssdp_packets if pkt.info.get("ssdp_method") == "NOTIFY")),
            "ssdp_msearch_count": float(sum(1 for pkt in ssdp_packets if pkt.info.get("ssdp_method") == "M-SEARCH")),
            "ssdp_sender_rate_mean": float(ssdp_rates.mean()) if ssdp_rates.size else 0.0,
            "ssdp_sender_rate_max": float(ssdp_rates.max()) if ssdp_rates.size else 0.0,
            "ssdp_top_sender_share": _top_k_share(ssdp_sender_counts.values(), 1) if ssdp_sender_counts else 0.0,
            "tcp_syn_rate": float(len(tcp_syn_packets)) / max(self.window_size, self.config.eps),
            "tcp_syn_ack_ratio": float(len(tcp_syn_packets)) / max(len(tcp_ack_packets), 1),
            "tcp_half_open_surplus": half_open_surplus,
            "tcp_retransmit_proxy": retransmit_proxy,
            "icmp_echo_rate": float(len(icmp_echo)) / max(self.window_size, self.config.eps),
            "icmp_size_cv": _coefficient_of_variation(icmp_sizes) if icmp_sizes.size else 0.0,
            "icmp_ttl_variance": float(icmp_ttls.var()) if icmp_ttls.size else 0.0,
            "udp_unique_ports": float(len(udp_port_counts)),
            "udp_port_herfindahl": _herfindahl_index(udp_port_counts.values()),
            "udp_top_port_share": _top_k_share(udp_port_counts.values(), 1) if udp_port_counts else 0.0,
            "udp_payload_size_variance": float(udp_payloads.var()) if udp_payloads.size else 0.0,
        }


__all__ = ["FeatureExtractor"]
=======
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
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7
