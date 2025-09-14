from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter
from math import log2

def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0, None) else 0.0

def _entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * log2(p)
    return ent

def _gini_from_vector(x: np.ndarray) -> float:
    """Gini of non-negative vector; 0 if degenerate."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    s = np.sort(x)
    n = s.size
    cum = np.cumsum(s, dtype=float)
    g = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(g)

def compute_static_features(
    win_rows: List[Dict],
    micro_bins: int,
    per_bin_total_pkts: np.ndarray,
    top_k_udp_ports: List[int],
    window_sec: float,
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """
    Compute per-window static features:
      rates: pkts/s, bytes/s, per protocol, per top-ports
      inter-arrival stats: mean/std/CV/median/p95/p99
      burstiness: max per-bin pkts; Gini over per-bin pkts
      uniqueness: #unique src/dst IPs, src MACs; entropies of src IP, dst port, TTL
      asymmetry: TCP SYN/SYN-ACK; UDP req/resp ratios (SSDP/DNS/NTP heuristics)
      size distribution: mean/std/mode of packet length
      SSDP specifics: fraction UDP/1900; counts of M-SEARCH/NOTIFY/200 OK; entropies of ST/MAN/USER-AGENT; multicast hit rate
      L2 hints: per-src-MAC dominance; (optional OUI diversity not implemented here)
    Returns (vector, names, dict_snapshots)
    """
    names: List[str] = []
    values: List[float] = []

    # timestamps, sizes
    ts = np.array([float(r["ts"]) for r in win_rows], dtype=float)
    sizes = np.array([float(r.get("len") or 0) for r in win_rows], dtype=float)
    prot_udp = [r.get("is_udp", 0) == 1 for r in win_rows]
    prot_tcp = [r.get("is_tcp", 0) == 1 for r in win_rows]
    prot_icmp = [r.get("is_icmp", 0) == 1 for r in win_rows]

    pkts = len(win_rows)
    bytes_ = float(sizes.sum())
    W = float(window_sec)

    # rates
    rate_feats = {
        "pkts_per_s": pkts / W if W > 0 else 0.0,
        "bytes_per_s": bytes_ / W if W > 0 else 0.0,
        "tcp_per_s": (sum(prot_tcp)) / W if W > 0 else 0.0,
        "udp_per_s": (sum(prot_udp)) / W if W > 0 else 0.0,
        "icmp_per_s": (sum(prot_icmp)) / W if W > 0 else 0.0,
    }
    for k in ["pkts_per_s","bytes_per_s","tcp_per_s","udp_per_s","icmp_per_s"]:
        names.append(k); values.append(rate_feats[k])

    # top ports rates (dst)
    for p in top_k_udp_ports:
        cnt = sum(1 for r in win_rows if r.get("is_udp",0)==1 and r.get("dst_port")==p)
        names.append(f"udp_dst_{p}_per_s"); values.append(cnt / W if W>0 else 0.0)

    # inter-arrival stats
    iat = np.diff(np.sort(ts)) if ts.size >= 2 else np.array([])
    def stat_or0(x, fn, default=0.0):
        try:
            return float(fn(x)) if x.size>0 else default
        except Exception:
            return default
    iat_mean = stat_or0(iat, np.mean)
    iat_std = stat_or0(iat, np.std)
    iat_cv = iat_std / iat_mean if iat_mean>0 else 0.0
    iat_median = stat_or0(iat, np.median)
    iat_p95 = stat_or0(iat, lambda z: np.percentile(z,95))
    iat_p99 = stat_or0(iat, lambda z: np.percentile(z,99))
    for k,v in [("iat_mean",iat_mean),("iat_std",iat_std),("iat_cv",iat_cv),
                ("iat_median",iat_median),("iat_p95",iat_p95),("iat_p99",iat_p99)]:
        names.append(k); values.append(v)

    # burstiness
    max_bin_pkts = float(np.max(per_bin_total_pkts)) if per_bin_total_pkts.size>0 else 0.0
    gini = _gini_from_vector(per_bin_total_pkts)
    for k,v in [("max_bin_pkts",max_bin_pkts),("gini_pkts",gini)]:
        names.append(k); values.append(v)

    # uniqueness & entropies
    src_ips = Counter([r.get("src_ip") for r in win_rows if r.get("src_ip")])
    dst_ips = Counter([r.get("dst_ip") for r in win_rows if r.get("dst_ip")])
    src_macs = Counter([r.get("src_mac") for r in win_rows if r.get("src_mac")])
    dst_ports = Counter([r.get("dst_port") for r in win_rows if r.get("dst_port") is not None])
    ttls = Counter([r.get("ttl") for r in win_rows if r.get("ttl") is not None])

    uniq_feats = {
        "n_unique_src_ip": float(len(src_ips)),
        "n_unique_dst_ip": float(len(dst_ips)),
        "n_unique_src_mac": float(len(src_macs)),
        "H_src_ip": _entropy_from_counts(src_ips),
        "H_dst_port": _entropy_from_counts(dst_ports),
        "H_ttl": _entropy_from_counts(ttls),
    }
    for k in ["n_unique_src_ip","n_unique_dst_ip","n_unique_src_mac","H_src_ip","H_dst_port","H_ttl"]:
        names.append(k); values.append(uniq_feats[k])

    # asymmetries
    tcp_syn = sum(int(r.get("tcp_syn",0)) for r in win_rows)
    tcp_synack = sum(int(r.get("tcp_synack",0)) for r in win_rows)
    tcp_ratio = _safe_div(tcp_syn, tcp_synack)

    # SSDP req/resp
    ssdp_req = sum(1 for r in win_rows if r.get("is_udp",0)==1 and (r.get("ssdp_method") in ("M-SEARCH","NOTIFY")))
    ssdp_resp = sum(1 for r in win_rows if r.get("is_udp",0)==1 and (r.get("ssdp_method") == "200-OK"))
    ssdp_ratio = _safe_div(ssdp_req, ssdp_resp)

    # (placeholders for DNS/NTP heuristics – counted via dst ports)
    dns_req = sum(1 for r in win_rows if r.get("is_udp",0)==1 and r.get("dst_port")==53)
    ntp_req = sum(1 for r in win_rows if r.get("is_udp",0)==1 and r.get("dst_port")==123)
    # responses are harder in statics without DPI; keep ratios conservative
    dns_resp = 0
    ntp_resp = 0
    dns_ratio = _safe_div(dns_req, dns_resp)
    ntp_ratio = _safe_div(ntp_req, ntp_resp)

    for k,v in [("tcp_syn_over_synack",tcp_ratio),
                ("udp_ssdp_req_over_resp",ssdp_ratio),
                ("udp_dns_req_over_resp",dns_ratio),
                ("udp_ntp_req_over_resp",ntp_ratio)]:
        names.append(k); values.append(v)

    # size distribution
    sz_mean = float(sizes.mean()) if sizes.size>0 else 0.0
    sz_std = float(sizes.std()) if sizes.size>0 else 0.0
    # approximate mode via small histogram
    if sizes.size>0:
        hist, edges = np.histogram(sizes, bins=min(20, max(5, int(np.sqrt(sizes.size)))))
        sz_mode = float((edges[np.argmax(hist)] + edges[np.argmax(hist)+1]) / 2.0)
    else:
        sz_mode = 0.0
    for k,v in [("pkt_len_mean",sz_mean),("pkt_len_std",sz_std),("pkt_len_mode",sz_mode)]:
        names.append(k); values.append(v)

    # SSDP specifics
    udp_1900 = sum(1 for r in win_rows if r.get("is_udp",0)==1 and r.get("dst_port")==1900)
    udp_total = sum(1 for r in win_rows if r.get("is_udp",0)==1)
    udp_1900_frac = _safe_div(udp_1900, udp_total)

    msearch = sum(1 for r in win_rows if r.get("ssdp_method")=="M-SEARCH")
    notify = sum(1 for r in win_rows if r.get("ssdp_method")=="NOTIFY")
    ok200 = sum(1 for r in win_rows if r.get("ssdp_method")=="200-OK")

    ST = Counter([r.get("ssdp_st") for r in win_rows if r.get("ssdp_st")])
    MAN = Counter([r.get("ssdp_man") for r in win_rows if r.get("ssdp_man")])
    UA = Counter([r.get("ssdp_user_agent") for r in win_rows if r.get("ssdp_user_agent")])

    # multicast hit rate: approximate via dst_port==1900 (since we don't retain explicit dst==multicast in this function)
    # If desired, add a boolean to rows when dst is multicast and compute precisely.
    multicast_hits = sum(1 for r in win_rows if r.get("is_udp",0)==1 and r.get("dst_port")==1900)
    multicast_rate = _safe_div(multicast_hits, max(1, udp_total))

    ssdp_feats = {
        "udp_1900_fraction": udp_1900_frac,
        "ssdp_msearch": float(msearch),
        "ssdp_notify": float(notify),
        "ssdp_200ok": float(ok200),
        "H_ST": _entropy_from_counts(ST),
        "H_MAN": _entropy_from_counts(MAN),
        "H_USER_AGENT": _entropy_from_counts(UA),
        "ssdp_multicast_hit_rate": float(multicast_rate),
    }
    for k in ["udp_1900_fraction","ssdp_msearch","ssdp_notify","ssdp_200ok","H_ST","H_MAN","H_USER_AGENT","ssdp_multicast_hit_rate"]:
        names.append(k); values.append(ssdp_feats[k])

    # L2 hints (weak evidence)
    src_mac_counts = src_macs
    total_mac = sum(src_mac_counts.values())
    mac_dom = max(src_mac_counts.values())/total_mac if total_mac>0 else 0.0
    names.append("src_mac_dominance"); values.append(float(mac_dom))

    vec = np.array(values, dtype=np.float32)
    snaps = {
        "pkts": float(pkts),
        "bytes": float(bytes_),
        "max_bin_pkts": float(max_bin_pkts),
        "udp_1900_fraction": float(udp_1900_frac),
        "tcp_syn_over_synack": float(tcp_ratio),
        "udp_ssdp_req_over_resp": float(ssdp_ratio),
    }
    return vec, names, snaps
