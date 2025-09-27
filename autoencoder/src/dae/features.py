from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy.stats import entropy as scipy_entropy

from .window import WindowStats


FEATURES_DEFAULT = (
    "pkt_count",
    "byte_count",
    "pps",
    "bps",
    "mean_pkt_len",
    "std_pkt_len",
    "min_pkt_len",
    "max_pkt_len",
    "mean_iat",
    "std_iat",
    "tcp_count",
    "udp_count",
    "icmp_count",
    "tcp_syn",
    "tcp_ack",
    "tcp_rst",
    "tcp_fin",
    "src_ip_entropy",
    "dst_ip_entropy",
    "src_port_entropy",
    "dst_port_entropy",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_src_ports",
    "unique_dst_ports",
)


def _compute_entropy(counter) -> float:
    if not counter:
        return 0.0
    counts = np.array(list(counter.values()), dtype=float)
    if counts.sum() == 0:
        return 0.0
    return float(scipy_entropy(counts, base=2))


class FeatureExtractor:
    def __init__(self, include: Iterable[str] = FEATURES_DEFAULT, ratios: bool = True) -> None:
        self.include = tuple(include)
        self.ratios = ratios

    def build_row(self, window: WindowStats) -> Dict[str, float]:
        pkt_count = window.packet_count
        duration = window.duration or 1.0
        byte_count = window.byte_count

        base: Dict[str, float] = {
            "window_idx": float(window.index),
            "start_ts": window.start,
            "end_ts": window.end,
            "pkt_count": float(pkt_count),
            "byte_count": float(byte_count),
            "pps": float(pkt_count) / duration if duration > 0 else 0.0,
            "bps": (8.0 * float(byte_count)) / duration if duration > 0 else 0.0,
            "mean_pkt_len": window.length_stats.mean_value(),
            "std_pkt_len": window.length_stats.std_value(),
            "min_pkt_len": window.length_stats.min(),
            "max_pkt_len": window.length_stats.max(),
            "mean_iat": window.iat_stats.mean_value(),
            "std_iat": window.iat_stats.std_value(),
            "tcp_count": float(window.proto_counts.get("TCP", 0)),
            "udp_count": float(window.proto_counts.get("UDP", 0)),
            "icmp_count": float(window.proto_counts.get("ICMP", 0)),
            "tcp_syn": float(window.tcp_flag_counts.get("SYN", 0)),
            "tcp_ack": float(window.tcp_flag_counts.get("ACK", 0)),
            "tcp_rst": float(window.tcp_flag_counts.get("RST", 0)),
            "tcp_fin": float(window.tcp_flag_counts.get("FIN", 0)),
            "src_ip_entropy": _compute_entropy(window.src_ips),
            "dst_ip_entropy": _compute_entropy(window.dst_ips),
            "src_port_entropy": _compute_entropy(window.src_ports),
            "dst_port_entropy": _compute_entropy(window.dst_ports),
            "unique_src_ips": float(len(window.src_ips)),
            "unique_dst_ips": float(len(window.dst_ips)),
            "unique_src_ports": float(len(window.src_ports)),
            "unique_dst_ports": float(len(window.dst_ports)),
        }

        row = {key: base.get(key, 0.0) for key in self.include if key in base}

        if self.ratios:
            denom = float(pkt_count) if pkt_count > 0 else 1.0
            row.update(
                {
                    "tcp_ratio": base["tcp_count"] / denom,
                    "udp_ratio": base["udp_count"] / denom,
                    "icmp_ratio": base["icmp_count"] / denom,
                }
            )

        # Always provide timestamps and index for downstream use
        row.update({
            "window_idx": float(window.index),
            "start_ts": window.start,
            "end_ts": window.end,
        })
        return row
