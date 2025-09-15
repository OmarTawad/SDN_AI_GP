# decision.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import statistics
import numpy as np


@dataclass
class WindowObs:
    prob: float
    snaps: Dict[str, float]
    t0: float
    t1: float


@dataclass
class DecisionConfig:
    # Hysteresis
    tau_high: float = 0.70
    tau_low: float = 0.55
    # Temporal consistency
    min_attack_windows: int = 3
    consecutive_required: int = 2
    cooldown_windows: int = 1
    # Plausibility gate
    enable_gate: bool = True
    warmup_windows: int = 60
    # Generic caps
    abs_pkts_per_s_cap: float = 1500.0
    burstiness_multiple: float = 3.0
    # TCP SYN completion threshold (≤ this is bad)
    syn_completion_max: float = 0.10
    # High-confidence override (disabled by default for safety)
    high_confidence_override: bool = False
    high_conf_tau: float = 0.93
    high_conf_windows: int = 3
    # Optional safety: avoid override at vanishing traffic
    min_pkts_per_s_for_override: float = 3.0


@dataclass
class FileDecision:
    file: str
    decision: str                   # "attack" | "normal"
    first_attack_timestamp: float | None
    num_attack_windows: int
    max_prob: float
    gate_reasons: List[str]


# ---------- Robust baselines over the quiet half of the file ----------

def _robust_stats(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values]) if len(values) > 1 else 0.0
    sigma = 1.4826 * mad if mad > 0 else 1.0
    return med, sigma


def _p10(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, 10))


def _build_quiet_baselines(windows: List[WindowObs]) -> Dict[str, float]:
    if not windows:
        return {
            "pkts_med": 0.0, "pkts_sigma": 1.0,
            "H_src_ip_p10": 0.0, "H_dst_port_p10": 0.0, "H_ttl_p10": 0.0,
            "median_bin_pkts_med": 1.0,
        }
    pkts = [float(w.snaps.get("pkts_per_s", 0.0)) for w in windows]
    order = np.argsort(pkts)
    n = len(order)
    k = max(100, int(0.60 * n))
    idx = order[:min(k, n)]

    quiet_pkts = [float(windows[i].snaps.get("pkts_per_s", 0.0)) for i in idx]
    quiet_H_src  = [float(windows[i].snaps.get("H_src_ip", 8.0)) for i in idx]
    quiet_H_dstp = [float(windows[i].snaps.get("H_dst_port", 8.0)) for i in idx]
    quiet_H_ttl  = [float(windows[i].snaps.get("H_ttl", 8.0)) for i in idx]
    quiet_medbin = [float(windows[i].snaps.get("median_bin_pkts", 1.0)) for i in idx]

    pkts_med, pkts_sigma = _robust_stats(quiet_pkts)
    return {
        "pkts_med": pkts_med,
        "pkts_sigma": pkts_sigma,
        "H_src_ip_p10": _p10(quiet_H_src),
        "H_dst_port_p10": _p10(quiet_H_dstp),
        "H_ttl_p10": _p10(quiet_H_ttl),
        "median_bin_pkts_med": statistics.median(quiet_medbin) if quiet_medbin else 1.0,
    }


# ------------------------ Plausibility gate (low-rate hardened) ------------------------

def _gate_generic(snaps: Dict[str, float], base: Dict[str, float], cfg: DecisionConfig):
    reasons, count = [], 0
    pkts = float(snaps.get("pkts_per_s", 0.0))
    total_pkts = float(snaps.get("total_pkts", 0.0))
    max_bin = float(snaps.get("max_bin_pkts", 0.0))
    med_bin = float(snaps.get("median_bin_pkts", base.get("median_bin_pkts_med", 1.0)))
    if med_bin <= 0.0:
        med_bin = max(1.0, float(base.get("median_bin_pkts_med", 1.0)))

    # Rate spike (stricter when pkts < 50/s)
    sigma = float(base.get("pkts_sigma", 1.0) or 1.0)
    med   = float(base.get("pkts_med", 0.0))
    spike_rel = (pkts >= med + 6.0 * sigma)
    if pkts < 50.0:
        spike_rel = (pkts >= med + 8.0 * sigma) and (pkts >= 25.0)
    rate_spike = spike_rel or (pkts >= cfg.abs_pkts_per_s_cap)
    if rate_spike:
        reasons.append("rate spike"); count += 1

    # Asymmetry (higher min events at low rate)
    udp_rr = float(snaps.get("udp_ssdp_req_over_resp", 0.0))
    tcp_rr = float(snaps.get("tcp_syn_over_synack", 0.0))
    gen_rr = float(snaps.get("generic_req_resp_ratio", 0.0))
    syn_ct = float(snaps.get("tcp_syn_count", 0.0))
    min_events = 60.0 if pkts < 50.0 else 40.0
    asym_udp_ok = (udp_rr >= 10.0) and (total_pkts >= min_events or pkts >= 60.0)
    asym_tcp_ok = (tcp_rr >= 10.0) and (syn_ct      >= min_events or pkts >= 60.0)
    asym_gen_ok = (gen_rr >= 10.0) and (total_pkts >= min_events or pkts >= 60.0)
    if asym_udp_ok or asym_tcp_ok or asym_gen_ok:
        reasons.append("asymmetry"); count += 1

    # Entropy collapse needs solid mass
    if total_pkts >= 100.0:
        H_src_ip  = float(snaps.get("H_src_ip", 1e9))
        H_dst_port= float(snaps.get("H_dst_port", 1e9))
        H_ttl     = float(snaps.get("H_ttl", 1e9))
        collapse = (
            (H_src_ip  <= float(base.get("H_src_ip_p10",  -1.0))) or
            (H_dst_port<= float(base.get("H_dst_port_p10",-1.0))) or
            (H_ttl     <= float(base.get("H_ttl_p10",     -1.0)))
        )
        if collapse:
            reasons.append("entropy collapse"); count += 1

    # Burstiness:
    # Normal rule needs solid mass (>=120 pkts in the window).
    burst_ok = (total_pkts >= 120.0) and (max_bin >= cfg.burstiness_multiple * med_bin)

    # Fallback for mid-low rate attacks (e.g., ~48 pkts/s):
    # If rate is decent (>=40/s) and the micro-burst is very sharp,
    # accept it even without 120+ packets. Use stricter ratio (+1.0).
    if not burst_ok:
        if (pkts >= 40.0) and (med_bin >= 2.0) and (max_bin >= (cfg.burstiness_multiple + 1.0) * med_bin):
            burst_ok = True

    if burst_ok:
        reasons.append("burstiness"); count += 1

    return reasons, count


def _gate_ssdp(snaps: Dict[str, float]):
    reasons: List[str] = []
    frac = float(snaps.get("udp_1900_fraction", 0.0))
    rr = float(snaps.get("udp_ssdp_req_over_resp", 0.0))
    multicast_hit = bool(snaps.get("ssdp_multicast_hit", 0.0))
    header_rep = bool(snaps.get("ssdp_header_repetition", 0.0))
    notify_ct = float(snaps.get("ssdp_notify_count", 0.0))
    msearch_ct = float(snaps.get("ssdp_msearch_count", 0.0))
    ok200_ct = float(snaps.get("ssdp_200ok_count", 0.0))
    pkts = float(snaps.get("pkts_per_s", 0.0))

    ok = False
    if frac >= 0.5:
        reasons.append("SSDP dominance"); ok = True
    if multicast_hit and (notify_ct + msearch_ct) >= 3:
        reasons.append("SSDP multicast"); ok = True
    if rr >= 10.0 and (notify_ct + msearch_ct) >= 5:
        reasons.append("SSDP asymmetry"); ok = True
    if (notify_ct >= 8 and ok200_ct <= 1 and frac >= 0.3) or (pkts >= 80.0 and ok200_ct == 0 and msearch_ct <= 2 and frac >= 0.3):
        reasons.append("SSDP notify-dominant"); ok = True
    if header_rep and (frac >= 0.3 or multicast_hit):
        reasons.append("SSDP header repetition"); ok = True

    return reasons, ok


def _gate_syn(snaps: Dict[str, float], cfg: DecisionConfig):
    """
    TCP SYN-based plausibility: fires for clear SYN floods or very low SYN→SYN/ACK completion.
    Returns (reasons: List[str], ok: bool)
    """
    reasons: List[str] = []
    syn_rate = float(snaps.get("tcp_syn_rate", 0.0))
    completion = snaps.get("tcp_synack_completion", None)
    syn_over_ack = float(snaps.get("tcp_syn_over_synack", 0.0))
    syn_cnt = float(snaps.get("tcp_syn_count", 0.0))
    ok = False

    # High SYN rate or strong SYN>>SYN/ACK asymmetry with enough events
    if (syn_rate >= 200.0) or (syn_over_ack >= 10.0 and syn_cnt >= 100.0):
        reasons.append("TCP SYN surge")
        ok = True

    # Very low completion with enough SYNs observed
    if completion is not None and syn_cnt >= 100.0 and float(completion) <= cfg.syn_completion_max:
        reasons.append("Low SYN-ACK completion")
        ok = True

    return reasons, ok


def plausibility_gate(snaps: Dict[str, float], base: Dict[str, float], cfg: DecisionConfig):
    gen_reasons, gen_count = _gate_generic(snaps, base, cfg)
    ssdp_reasons, ssdp_ok = _gate_ssdp(snaps)
    syn_reasons, syn_ok = _gate_syn(snaps, cfg)

    pkts = float(snaps.get("pkts_per_s", 0.0))
    proto_ok = ssdp_ok or syn_ok

    # Low-rate safety: protocol hints alone never suffice;
    # need ≥2 generic reasons, and 'burstiness' only counted if mass (above or fallback).
    if pkts < 50.0:
        return (gen_count >= 2), (gen_reasons if gen_count >= 2 else [])

    if gen_count >= 2:
        return True, gen_reasons
    if proto_ok and gen_count >= 1:
        return True, (ssdp_reasons if ssdp_ok else syn_reasons) + gen_reasons
    return False, []


# ------------------------ State machine decision ------------------------

def decide_file(
    file_path: str,
    windows: Iterable[WindowObs],
    cfg: DecisionConfig = DecisionConfig(),
) -> FileDecision:
    win_list = list(windows)
    if not win_list:
        return FileDecision(file=file_path, decision="normal", first_attack_timestamp=None,
                            num_attack_windows=0, max_prob=0.0, gate_reasons=[])

    base = _build_quiet_baselines(win_list)

    state = "NORMAL"
    max_prob = 0.0
    first_attack_ts: float | None = None
    flagged_total = 0
    gate_reasons_final: List[str] = []

    consec_high = 0
    consec_low = 0
    cluster_size = 0
    cooldown_ctr = 0

    # High-confidence override bookkeeping
    hi_conf_streak = 0
    hi_conf_cluster = 0
    hi_conf_triggered = False

    for w in win_list:
        max_prob = max(max_prob, w.prob)
        prob = w.prob
        pkts_ps = float(w.snaps.get("pkts_per_s", 0.0))

        # Update high-confidence streaks
        if cfg.high_confidence_override and prob >= cfg.high_conf_tau and pkts_ps >= cfg.min_pkts_per_s_for_override:
            hi_conf_streak += 1
            hi_conf_cluster += 1
        else:
            hi_conf_streak = 0
            hi_conf_cluster = 0

        if state == "NORMAL":
            if cooldown_ctr > 0:
                cooldown_ctr -= 1

            gate_ok, gate_reasons = (True, []) if not cfg.enable_gate else plausibility_gate(w.snaps, base, cfg)

            # Normal (gate-respecting) path
            if (prob >= cfg.tau_high) and gate_ok and cooldown_ctr == 0:
                consec_high += 1
                cluster_size += 1
            else:
                consec_high = 0
                cluster_size = 0

            # Promote via gate path
            if (consec_high >= cfg.consecutive_required) and (cluster_size >= cfg.min_attack_windows):
                state = "ATTACK"
                flagged_total += cluster_size
                if first_attack_ts is None:
                    first_attack_ts = w.t0
                    gate_reasons_final = gate_reasons[:4]
                consec_low = 0
                continue

            # Promote via high-confidence override (no gate)
            if cfg.high_confidence_override and (hi_conf_streak >= cfg.consecutive_required) and (hi_conf_cluster >= cfg.min_attack_windows):
                state = "ATTACK"
                hi_conf_triggered = True
                flagged_total += hi_conf_cluster
                if first_attack_ts is None:
                    first_attack_ts = w.t0
                    gate_reasons_final = ["model-high-confidence"]
                consec_low = 0
                continue

        else:  # ATTACK
            gate_ok, _ = (True, []) if not cfg.enable_gate else plausibility_gate(w.snaps, base, cfg)
            if prob >= cfg.tau_high and (gate_ok or hi_conf_triggered):
                flagged_total += 1
            if prob <= cfg.tau_low:
                consec_low += 1
            else:
                consec_low = 0
            if consec_low >= 2:
                state = "NORMAL"
                cooldown_ctr = cfg.cooldown_windows
                consec_high = 0
                cluster_size = 0
                consec_low = 0
                hi_conf_streak = 0
                hi_conf_cluster = 0
                hi_conf_triggered = False

    decision = "attack" if first_attack_ts is not None else "normal"
    return FileDecision(
        file=file_path,
        decision=decision,
        first_attack_timestamp=first_attack_ts,
        num_attack_windows=flagged_total,
        max_prob=max_prob,
        gate_reasons=gate_reasons_final[:4],
    )


# === DEBUG helper mirroring the current gate ===

def gate_checks(snaps: Dict[str, float], base: Dict[str, float], cfg: DecisionConfig):
    out = {}
    pkts = float(snaps.get("pkts_per_s", 0.0))
    total_pkts = float(snaps.get("total_pkts", 0.0))
    total_udp  = float(snaps.get("total_udp", 0.0))

    # Rate spike (same low-rate rule)
    sigma = float(base.get("pkts_sigma", 1.0) or 1.0)
    med   = float(base.get("pkts_med", 0.0))
    spike_rel = (pkts >= med + 6.0 * sigma)
    if pkts < 50.0:
        spike_rel = (pkts >= med + 8.0 * sigma) and (pkts >= 25.0)
    rate_spike = spike_rel or (pkts >= cfg.abs_pkts_per_s_cap)
    out["rate_spike"] = bool(rate_spike)
    out["pkts_per_s"] = pkts
    out["base_pkts_med"] = med
    out["base_pkts_sigma"] = sigma

    # Asymmetry (min events stricter at low rate)
    udp_rr = float(snaps.get("udp_ssdp_req_over_resp", 0.0))
    tcp_rr = float(snaps.get("tcp_syn_over_synack", 0.0))
    gen_rr = float(snaps.get("generic_req_resp_ratio", 0.0))
    syn_ct = float(snaps.get("tcp_syn_count", 0.0))
    min_events = 60.0 if pkts < 50.0 else 40.0
    asym_udp_ok = (udp_rr >= 10.0) and (total_udp >= min_events or pkts >= 60.0)
    asym_tcp_ok = (tcp_rr >= 10.0) and (syn_ct    >= min_events or pkts >= 60.0)
    asym_gen_ok = (gen_rr >= 10.0) and (total_pkts>= min_events or pkts >= 60.0)
    out["asymmetry"] = bool(asym_udp_ok or asym_tcp_ok or asym_gen_ok)
    out["udp_rr"] = udp_rr; out["tcp_rr"] = tcp_rr; out["gen_rr"] = gen_rr
    out["syn_ct"] = syn_ct; out["total_pkts"] = total_pkts; out["total_udp"] = total_udp

    # Entropy collapse (mass guard)
    ent_ok = False
    if total_pkts >= 100.0:
        H_src_ip = float(snaps.get("H_src_ip", 1e9))
        H_dst_port = float(snaps.get("H_dst_port", 1e9))
        H_ttl = float(snaps.get("H_ttl", 1e9))
        ent_ok = (
            (H_src_ip  <= float(base.get("H_src_ip_p10",  -1.0))) or
            (H_dst_port<= float(base.get("H_dst_port_p10",-1.0))) or
            (H_ttl     <= float(base.get("H_ttl_p10",     -1.0)))
        )
        out["H_src_ip"] = H_src_ip; out["H_dst_port"] = H_dst_port; out["H_ttl"] = H_ttl
        out["H_src_ip_p10"] = float(base.get("H_src_ip_p10", 0.0))
        out["H_dst_port_p10"] = float(base.get("H_dst_port_p10", 0.0))
        out["H_ttl_p10"] = float(base.get("H_ttl_p10", 0.0))
    out["entropy_collapse"] = bool(ent_ok)

    # Burstiness (with mid-low-rate fallback mirrored)
    max_bin = float(snaps.get("max_bin_pkts", 0.0))
    med_bin = float(snaps.get("median_bin_pkts", base.get("median_bin_pkts_med", 1.0)))
    if med_bin <= 0:
        med_bin = max(1.0, float(base.get("median_bin_pkts_med", 1.0)))
    burst_ok = (total_pkts >= 120.0) and (max_bin >= cfg.burstiness_multiple * med_bin)
    if not burst_ok:
        if (pkts >= 40.0) and (med_bin >= 2.0) and (max_bin >= (cfg.burstiness_multiple + 1.0) * med_bin):
            burst_ok = True
    out["burstiness"] = bool(burst_ok)
    out["max_bin_pkts"] = max_bin; out["median_bin_pkts"] = med_bin

    # SSDP quick flags (summarized)
    frac = float(snaps.get("udp_1900_fraction", 0.0))
    rr = float(snaps.get("udp_ssdp_req_over_resp", 0.0))
    multicast_hit = bool(snaps.get("ssdp_multicast_hit", 0.0))
    header_rep = bool(snaps.get("ssdp_header_repetition", 0.0))
    notify_count  = float(snaps.get("ssdp_notify_count", 0.0))
    msearch_count = float(snaps.get("ssdp_msearch_count", 0.0))
    ok200_count   = float(snaps.get("ssdp_200ok_count", 0.0))
    ssdp_ok = False
    if frac >= 0.5: ssdp_ok = True
    if multicast_hit and (notify_count + msearch_count) >= 3: ssdp_ok = True
    if rr >= 10.0 and (notify_count + msearch_count) >= 5: ssdp_ok = True
    if (notify_count >= 8 and ok200_count <= 1 and frac >= 0.3) or (pkts >= 80.0 and ok200_count == 0 and msearch_count <= 2 and frac >= 0.3):
        ssdp_ok = True
    if header_rep and (frac >= 0.3 or multicast_hit): ssdp_ok = True
    out["ssdp_gate"] = bool(ssdp_ok)
    out["udp_1900_fraction"] = frac; out["ssdp_rr"] = rr
    out["ssdp_multicast"] = multicast_hit; out["ssdp_header_rep"] = header_rep
    out["ssdp_notify_count"] = notify_count; out["ssdp_msearch_count"] = msearch_count; out["ssdp_200ok_count"] = ok200_count

    # SYN quick flags
    syn_rate = float(snaps.get("tcp_syn_rate", 0.0))
    completion = snaps.get("tcp_synack_completion", None)
    syn_over_ack = float(snaps.get("tcp_syn_over_synack", 0.0))
    syn_cnt = float(snaps.get("tcp_syn_count", 0.0))
    syn_ok = False
    if (syn_rate >= 200.0) or (syn_over_ack >= 10.0 and syn_cnt >= 100):
        syn_ok = True
    if completion is not None and syn_cnt >= 100 and float(completion) <= cfg.syn_completion_max:
        syn_ok = True
    out["syn_gate"] = bool(syn_ok)
    out["tcp_syn_rate"] = syn_rate; out["tcp_syn_over_synack"] = syn_over_ack
    out["tcp_synack_completion"] = float(completion if completion is not None else -1.0)

    gen_count = int(out["rate_spike"]) + int(out["asymmetry"]) + int(out["entropy_collapse"]) + int(out["burstiness"])
    proto_ok = bool(out["ssdp_gate"] or out["syn_gate"])
    out["generic_count"] = gen_count; out["proto_ok"] = proto_ok

    # Final pass rule (mirror plausibility_gate)
    if pkts < 50.0:
        out["gate_pass"] = bool(gen_count >= 2)
    else:
        out["gate_pass"] = bool((gen_count >= 2) or (proto_ok and gen_count >= 1))
    return out
