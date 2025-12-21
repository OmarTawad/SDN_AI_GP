from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

def compute_sequence_features(
    win_rows: List[Dict],
    bin_idx: List[int],
    micro_bins: int,
    top_k_udp_ports: List[int],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      seq_feats: shape [M, K_seq]
      extras: dict for explainability (e.g., per-bin totals)
    Per-bin features:
      - total_pkts, total_bytes
      - bin_tcp_pkts, bin_udp_pkts, bin_icmp_pkts
      - bin_tcp_syn, bin_tcp_synack
      - bin_udp_pkts_dst_<p> for p in top_k_udp_ports
      - bin_ssdp_msearch, bin_ssdp_response
    """
    M = micro_bins
    P = len(top_k_udp_ports)
    # feature order definition
    names = []
    # indices to fill quickly
    idx = {}

    # base 2 + 3 + 2
    base = [
        "bin_total_pkts", "bin_total_bytes",
        "bin_tcp_pkts", "bin_udp_pkts", "bin_icmp_pkts",
        "bin_tcp_syn", "bin_tcp_synack",
    ]
    names.extend(base)
    # per-port UDP dst counts
    for p in top_k_udp_ports:
        names.append(f"bin_udp_pkts_dst_{p}")
    # SSDP
    names.extend(["bin_ssdp_msearch", "bin_ssdp_response"])

    K = len(names)
    X = np.zeros((M, K), dtype=np.float32)

    # Build a map from name to column index
    col = {n:i for i,n in enumerate(names)}

    for r, b in zip(win_rows, bin_idx):
        X[b, col["bin_total_pkts"]] += 1.0
        length = float(r.get("len") or 0)
        X[b, col["bin_total_bytes"]] += length

        if r.get("is_tcp", 0):
            X[b, col["bin_tcp_pkts"]] += 1.0
            if r.get("tcp_syn", 0):
                X[b, col["bin_tcp_syn"]] += 1.0
            if r.get("tcp_synack", 0):
                X[b, col["bin_tcp_synack"]] += 1.0
        elif r.get("is_udp", 0):
            X[b, col["bin_udp_pkts"]] += 1.0
            dp = r.get("dst_port")
            if dp in top_k_udp_ports:
                X[b, col[f"bin_udp_pkts_dst_{dp}"]] += 1.0
            meth = (r.get("ssdp_method") or "").upper()
            if meth == "M-SEARCH":
                X[b, col["bin_ssdp_msearch"]] += 1.0
            elif meth == "200-OK":
                X[b, col["bin_ssdp_response"]] += 1.0
        elif r.get("is_icmp", 0):
            X[b, col["bin_icmp_pkts"]] += 1.0

    extras = {
        "per_bin_total_pkts": X[:, col["bin_total_pkts"]].copy(),
        "per_bin_total_bytes": X[:, col["bin_total_bytes"]].copy(),
        "feature_names": np.array(names),
    }
    return X, extras
