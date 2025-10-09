# infer.py
from __future__ import annotations

import os

# Constrain BLAS threads early for 2 vCPU deployments
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import argparse
import glob
import json
import csv
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

torch.set_num_threads(min(2, max(1, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except AttributeError:
    pass

from data.pcap_reader import iter_rows_from_pcap
from data.windowizer import iter_windows
from features.seq_features import compute_sequence_features
from features.static_features import compute_static_features
from features.scaler import RobustScaler

# Robust file-level decision (hysteresis + gate + temporal rules)
from decision import DecisionConfig, WindowObs, decide_file


# ------------------------- load artifacts -------------------------

def _load_artifacts(save_dir: str, cfg: dict):
    """
    Returns:
      model (eval mode), scaler, slimmer, meta:dict, calib:dict{"temperature","threshold"}
    """
    from features.feature_slimming import StaticSlimmer
    from models.dws_cnn import FastDetector

    # scaler (backward-compatible custom saver/loader)
    scaler = RobustScaler.load(save_dir)

    meta_path = os.path.join(save_dir, "feature_model_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    static_dim = int(meta.get("static_dim", scaler.n_features_ or 0))
    slimmer = StaticSlimmer(out_dim=static_dim)
    slimmer.load(save_dir)

    seq_in_dim = int(meta["seq_in_dim"])
    channels = tuple(meta.get("channels", cfg["training"]["channels"]))
    kernel_size = int(meta.get("kernel_size", cfg["training"]["kernel_size"]))
    dropout = float(meta.get("dropout", cfg["training"]["dropout"]))
    mlp_hidden = tuple(meta.get("mlp_hidden", cfg["training"]["mlp_hidden"]))
    model = FastDetector(
        seq_in_dim=seq_in_dim,
        static_dim=static_dim,
        channels=channels,
        k=kernel_size,
        drop=dropout,
        mlp_hidden=mlp_hidden,
    )
    state = torch.load(os.path.join(save_dir, "model_best.pt"), map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    # calibration
    calib_path = os.path.join(save_dir, "calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path, "r") as f:
            calib = json.load(f)
    else:
        calib = {"temperature": 1.0, "threshold": 0.5}

    return model, scaler, slimmer, meta, calib


# ------------------------- helpers -------------------------

def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _build_snapshots(win_rows, W, M, extras, snaps, ssdp_v4, ssdp_v6):
    """
    Robust snapshot builder:
      - IPv4 (239.255.255.250) and IPv6 (ff02::c, ff0e::c) multicast + L2 multicast MACs
      - infers SSDP methods from multiple payload keys when missing
      - computes UDP/1900 fraction from tolerant port key discovery
    """
    out = dict(snaps)

    # --- counts
    total_pkts = float(len(win_rows))
    udp_pkts   = float(sum(int(r.get("is_udp", 0)) or str(r.get("proto","")).upper()=="UDP" for r in win_rows))
    tcp_pkts   = float(sum(int(r.get("is_tcp", 0)) or str(r.get("proto","")).upper()=="TCP" for r in win_rows))
    out["total_pkts"] = total_pkts
    out["total_udp"]  = udp_pkts
    out["total_tcp"]  = tcp_pkts

    # pkts/s
    out["pkts_per_s"] = total_pkts / max(W, 1e-6)

    # per-bin stats
    per_bin = extras.get("per_bin_total_pkts", [])
    if isinstance(per_bin, (list, np.ndarray)) and len(per_bin) > 0:
        arr = np.asarray(per_bin, dtype=float)
        out["median_bin_pkts"] = float(np.median(arr))
        out["max_bin_pkts"]    = float(np.max(arr))
    else:
        out["median_bin_pkts"] = out.get("median_bin_pkts", 1.0)
        out["max_bin_pkts"]    = out.get("max_bin_pkts", total_pkts)

    # --- TCP SYN metrics
    syn_ct    = sum(int(r.get("tcp_syn", 0)) for r in win_rows)
    synack_ct = sum(int(r.get("tcp_synack", 0)) for r in win_rows)
    out["tcp_syn_count"]         = float(syn_ct)
    out["tcp_syn_rate"]          = float(syn_ct) / max(W, 1e-6)
    out["tcp_synack_completion"] = float(synack_ct) / max(syn_ct + 1e-6, 1e-6)
    out.setdefault("tcp_syn_over_synack", snaps.get("tcp_syn_over_synack", 0.0))

    # --- Multicast hit (IPv4 + both IPv6 scopes + L2 multicast MACs)
    dst_ip_lower_set = {"239.255.255.250", ssdp_v6.lower(), "ff02::c", "ff0e::c"}
    def is_multicast_l2(mac: str) -> bool:
        m = (mac or "").lower()
        return m.startswith("33:33:") or m.startswith("01:00:5e")
    ssdp_multicast_hit = any(
        str(r.get("dst_ip","")).lower() in dst_ip_lower_set or is_multicast_l2(str(r.get("dst_mac","")))
        for r in win_rows
    )
    out["ssdp_multicast_hit"] = 1.0 if ssdp_multicast_hit else 0.0

    # --- tolerant UDP 1900 port discovery
    def get_ports(row):
        cand = [
            row.get("udp_dport"), row.get("udp_sport"),
            row.get("udp_dstport"), row.get("udp_srcport"),
            row.get("dport"), row.get("sport"),
            row.get("dst_port"), row.get("src_port"),
            row.get("udp.dstport"), row.get("udp.srcport"),
        ]
        ints = []
        for v in cand:
            try:
                if v is not None:
                    ints.append(int(v))
            except Exception:
                pass
        return ints

    udp1900 = 0
    for r in win_rows:
        is_udp = int(r.get("is_udp", 0)) or str(r.get("proto","")).upper()=="UDP"
        if not is_udp:
            continue
        ports = get_ports(r)
        if any(p == 1900 for p in ports):
            udp1900 += 1
    out["udp_1900_fraction"] = float(udp1900) / max(udp_pkts, 1.0)

    # --- SSDP method inference (very tolerant)
    PAYLOAD_KEYS = ("payload","payload_str","payload_utf8","payload_raw","raw","data","tcp_payload","udp_payload")
    def sniff_method(row):
        m = row.get("ssdp_method")
        if isinstance(m, str) and m:
            u = m.upper()
            if "NOTIFY" in u:   return "NOTIFY"
            if "M-SEARCH" in u: return "M-SEARCH"
            if "200" in u:      return "200-OK"
        for k in PAYLOAD_KEYS:
            val = row.get(k)
            if val is None:
                continue
            if isinstance(val, (bytes, bytearray)):
                try:
                    s = val.decode("latin1", errors="ignore")
                except Exception:
                    continue
            else:
                s = str(val)
            u = s.upper()
            if "NOTIFY * HTTP/1.1" in u or u.startswith("NOTIFY "):    return "NOTIFY"
            if "M-SEARCH * HTTP/1.1" in u or u.startswith("M-SEARCH "): return "M-SEARCH"
            if "HTTP/1.1 200 OK" in u or " 200 OK" in u:                return "200-OK"
        return None

    msearch_ct = ok200_ct = notify_ct = 0
    for r in win_rows:
        is_udp = int(r.get("is_udp", 0)) or str(r.get("proto","")).upper()=="UDP"
        if not is_udp:
            continue
        if 1900 not in get_ports(r):
            continue
        mt = sniff_method(r)
        if mt == "M-SEARCH": msearch_ct += 1
        elif mt == "200-OK": ok200_ct   += 1
        elif mt == "NOTIFY": notify_ct  += 1

    out["ssdp_msearch_count"] = float(msearch_ct)
    out["ssdp_200ok_count"]   = float(ok200_ct)
    out["ssdp_notify_count"]  = float(notify_ct)
    if "udp_ssdp_req_over_resp" not in out:
        req = float(msearch_ct + notify_ct)
        resp = float(ok200_ct)
        out["udp_ssdp_req_over_resp"] = req / max(resp, 1.0)

    # Entropies / generic ratio defaults
    out.setdefault("H_src_ip",  snaps.get("H_src_ip",  8.0))
    out.setdefault("H_dst_port",snaps.get("H_dst_port",8.0))
    out.setdefault("H_ttl",     snaps.get("H_ttl",     8.0))
    out.setdefault("generic_req_resp_ratio", snaps.get("generic_req_resp_ratio", 0.0))
    return out


def _summarize_actor_suspiciousness(
    tracker: Dict[str, Dict[str, float]],
    window_sec: float,
    key_label: str,
) -> List[Dict[str, float]]:
    summary: List[Dict[str, float]] = []
    if not tracker:
        return summary

    for actor, stats in tracker.items():
        total_packets = int(stats.get("total_packets", 0) or 0)
        if total_packets <= 0:
            continue
        prob_weighted = float(stats.get("prob_weighted_packets", 0.0))
        windows = int(stats.get("windows", 0) or 0)
        high_windows = int(stats.get("high_prob_windows", 0) or 0)
        max_prob = float(stats.get("max_prob", 0.0))
        max_packets = int(stats.get("max_packets", 0) or 0)

        mean_prob = prob_weighted / total_packets
        high_frac = high_windows / max(1, windows)
        packets_per_sec_peak = max_packets / max(window_sec, 1e-6)

        score_components = (
            0.6 * mean_prob
            + 0.25 * max_prob
            + 0.15 * min(1.0, high_frac)
        )
        score = score_components * math.log1p(total_packets)
        summary.append(
            {
                key_label: actor,
                "score": round(score, 6),
                "mean_prob": round(mean_prob, 6),
                "max_prob": round(max_prob, 6),
                "windows_with_activity": windows,
                "high_prob_windows": high_windows,
                "total_packets": total_packets,
                "max_packets_in_window": max_packets,
                "peak_packets_per_sec": round(packets_per_sec_peak, 3),
            }
        )

    summary.sort(key=lambda row: row["score"], reverse=True)
    return summary


# ------------------------- main per-file routine -------------------------

def run_on_pcap(
    pcap: str,
    cfg: dict,
    model,
    scaler,
    slimmer,
    meta: dict,
    T: float,
    tau: float,
    out_dir: str,
    decid_cfg: DecisionConfig,
    device: torch.device,
) -> Tuple[Dict, str, str]:
    """
    Streams the pcap, computes features, scores windows, applies robust file-level decision.
    Writes per-window CSV and per-file JSON. Returns (per_file_json, csv_path, json_path).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Track per-MAC and per-IP activity for later suspiciousness ranking
    mac_tracker: Dict[str, Dict[str, float]] = {}
    ip_tracker: Dict[str, Dict[str, float]] = {}
    ip_mac_counter: Dict[str, Counter] = defaultdict(Counter)

    # Windowing & feature config
    W = float(cfg["windowing"]["window_sec"])
    S = float(cfg["windowing"]["stride_sec"])
    M = int(meta.get("micro_bins", cfg["windowing"]["micro_bins"]))
    top_ports = list(meta.get("top_k_udp_ports", cfg["data"]["top_k_udp_ports"]))
    ssdp_v4 = cfg["features"]["ssdp_multicast_ipv4"]
    ssdp_v6 = cfg["features"]["ssdp_multicast_ipv6"]

    # Iterate windows
    rows_iter = iter_rows_from_pcap(pcap, ssdp_v4, ssdp_v6)
    win_iter = iter_windows(rows_iter, W, S, M)

    base = os.path.basename(pcap)
    csv_path = os.path.join(out_dir, f"{os.path.splitext(base)[0]}_windows.csv")
    json_path = os.path.join(out_dir, f"{os.path.splitext(base)[0]}.json")

    fieldnames = [
        "t_start", "t_end", "prob", "attn_peak_bin", "max_bin_pkts",
        "udp_1900_fraction", "ssdp_msearch", "ssdp_200ok", "tcp_syn", "tcp_synack",
        "icmp_pkts", "dst_port_entropy", "src_ip_entropy"
    ]

    window_obs: List[WindowObs] = []
    max_prob = 0.0

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for (t0, t1, win_rows, bins) in tqdm(win_iter, desc=f"Windows {base}", unit="win"):
            if not win_rows:
                continue

            # Sequence + static feature vectors
            seq, extras = compute_sequence_features(win_rows, bins, M, top_ports)
            static_vec, static_names, snaps = compute_static_features(
                win_rows, M, extras.get("per_bin_total_pkts", []), top_ports, W
            )

            # Scale + slim static
            feature_names = getattr(slimmer, "src_names", None) or scaler.feature_names_
            if feature_names is None or len(feature_names) != static_vec.size:
                feature_names = [f"f_{i}" for i in range(static_vec.size)]
            try:
                stat_scaled = scaler.transform(static_vec.reshape(1, -1), feature_names)
            except Exception:
                # Skip this window if schema mismatch; safer than garbage scores
                continue
            stat_slim = slimmer.transform(stat_scaled)

            # Tensors
            seq_t = torch.from_numpy(seq).unsqueeze(0).to(device).float()
            static_t = torch.from_numpy(stat_slim).to(device).float()

            with torch.no_grad():
                out = model(seq_t, static_t)
                logit = float(out["logits"].cpu().numpy().ravel()[0])
                logit_T = logit / max(float(T), 1e-3)
                prob = float(1.0 / (1.0 + np.exp(-logit_T)))
                max_prob = max(max_prob, prob)
                # attention (optional)
                attn = out.get("attn", None)
                if attn is not None:
                    attn = attn.cpu().numpy().ravel()
                    peak_bin = int(np.argmax(attn))
                else:
                    peak_bin = 0

            # Per-actor contribution within this window
            mac_counts: Dict[str, int] = {}
            ip_counts: Dict[str, int] = {}
            for row in win_rows:
                mac_raw = row.get("src_mac") or ""
                ip_raw = row.get("src_ip") or ""

                mac = str(mac_raw).lower() if mac_raw else ""
                if mac and mac not in ("ff:ff:ff:ff:ff:ff", "00:00:00:00:00:00"):
                    mac_counts[mac] = mac_counts.get(mac, 0) + 1

                ip = str(ip_raw).strip()
                if ip:
                    ip_counts[ip] = ip_counts.get(ip, 0) + 1
                    if mac:
                        ip_mac_counter[ip][mac] += 1

            if mac_counts:
                for mac, pkt_count in mac_counts.items():
                    stats = mac_tracker.setdefault(
                        mac,
                        {
                            "total_packets": 0,
                            "prob_weighted_packets": 0.0,
                            "windows": 0,
                            "high_prob_windows": 0,
                            "max_prob": 0.0,
                            "max_packets": 0,
                        },
                    )
                    stats["total_packets"] += pkt_count
                    stats["prob_weighted_packets"] += prob * pkt_count
                    stats["windows"] += 1
                    stats["max_prob"] = max(stats["max_prob"], prob)
                    stats["max_packets"] = max(stats["max_packets"], pkt_count)
                    if prob >= tau and pkt_count >= 3:
                        stats["high_prob_windows"] += 1

            if ip_counts:
                for ip, pkt_count in ip_counts.items():
                    stats = ip_tracker.setdefault(
                        ip,
                        {
                            "total_packets": 0,
                            "prob_weighted_packets": 0.0,
                            "windows": 0,
                            "high_prob_windows": 0,
                            "max_prob": 0.0,
                            "max_packets": 0,
                        },
                    )
                    stats["total_packets"] += pkt_count
                    stats["prob_weighted_packets"] += prob * pkt_count
                    stats["windows"] += 1
                    stats["max_prob"] = max(stats["max_prob"], prob)
                    stats["max_packets"] = max(stats["max_packets"], pkt_count)
                    if prob >= tau and pkt_count >= 3:
                        stats["high_prob_windows"] += 1

            # Build plausibility snapshots (for file decision)
            ssdp_snap = _build_snapshots(win_rows, W, M, extras, snaps, ssdp_v4, ssdp_v6)

            # Record to state machine list
            window_obs.append(WindowObs(prob=prob, snaps=ssdp_snap, t0=float(t0), t1=float(t1)))

            # Write CSV (key visibility fields)
            writer.writerow({
                "t_start": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                "t_end":   datetime.fromtimestamp(t1, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                "prob": f"{prob:.6f}",
                "attn_peak_bin": peak_bin,
                "max_bin_pkts": f"{ssdp_snap.get('max_bin_pkts', 0.0):.6f}",
                "udp_1900_fraction": f"{ssdp_snap.get('udp_1900_fraction', 0.0):.6f}",
                "ssdp_msearch": int(sum(1 for r in win_rows if (r.get('ssdp_method') == 'M-SEARCH'))),
                "ssdp_200ok": int(sum(1 for r in win_rows if (r.get('ssdp_method') in ('200-OK','200 OK')))),
                "tcp_syn": int(sum(int(r.get('tcp_syn', 0)) for r in win_rows)),
                "tcp_synack": int(sum(int(r.get('tcp_synack', 0)) for r in win_rows)),
                "icmp_pkts": int(sum(int(r.get('is_icmp', 0)) for r in win_rows)),
                "dst_port_entropy": f"{ssdp_snap.get('H_dst_port', 0.0):.6f}",
                "src_ip_entropy": f"{ssdp_snap.get('H_src_ip', 0.0):.6f}",
            })

    # -------- Robust file-level decision (state machine + gate + hysteresis) --------
    file_dec = decide_file(
        file_path=os.path.abspath(pcap),
        windows=window_obs,
        cfg=decid_cfg
    )

    mac_summary = _summarize_actor_suspiciousness(mac_tracker, W, "mac")
    ip_summary = _summarize_actor_suspiciousness(ip_tracker, W, "ip")
    mac_summary_map = {row["mac"]: row for row in mac_summary}

    most_suspicious_ip = ip_summary[0] if ip_summary else None
    most_suspicious_mac = None
    if most_suspicious_ip:
        if isinstance(most_suspicious_ip, dict):
            most_suspicious_ip = dict(most_suspicious_ip)
        ip_val = most_suspicious_ip["ip"]
        mac_counter = ip_mac_counter.get(ip_val)
        best_mac = None
        if mac_counter:
            best_mac = mac_counter.most_common(1)[0][0]
        if best_mac:
            mac_row = mac_summary_map.get(best_mac)
            if mac_row:
                most_suspicious_mac = {**mac_row, "associated_ip": ip_val}
            else:
                most_suspicious_mac = {
                    "mac": best_mac,
                    "associated_ip": ip_val,
                    "score": None,
                    "mean_prob": None,
                    "max_prob": None,
                    "windows_with_activity": None,
                    "high_prob_windows": None,
                    "total_packets": ip_mac_counter[ip_val][best_mac],
                    "max_packets_in_window": None,
                    "peak_packets_per_sec": None,
                }
            most_suspicious_ip["top_mac"] = best_mac

    if most_suspicious_mac is None and mac_summary:
        most_suspicious_mac = dict(mac_summary[0])

    if isinstance(most_suspicious_mac, dict) and "associated_ip" not in most_suspicious_mac:
        mac_val = most_suspicious_mac.get("mac")
        if mac_val:
            best_ip = None
            best_count = 0
            for ip_val, counter in ip_mac_counter.items():
                count = counter.get(mac_val, 0)
                if count > best_count:
                    best_count = count
                    best_ip = ip_val
            if best_ip:
                most_suspicious_mac["associated_ip"] = best_ip

    per_file = {
        "file": file_dec.file,
        "decision": file_dec.decision,
        "first_attack_timestamp": _iso(file_dec.first_attack_timestamp),
        "num_attack_windows": int(file_dec.num_attack_windows),
        "max_prob": float(round(file_dec.max_prob, 6)),
        "gate_reasons": file_dec.gate_reasons,
        "most_suspicious_mac": most_suspicious_mac,
        "most_suspicious_ip": most_suspicious_ip,
    }
    if mac_summary:
        per_file["suspicious_macs"] = mac_summary[:5]
    if ip_summary:
        per_file["suspicious_ips"] = ip_summary[:5]
    with open(json_path, "w") as fj:
        json.dump(per_file, fj, indent=2)

    mac_csv_path = None
    if mac_summary:
        mac_csv_path = os.path.join(out_dir, f"{os.path.splitext(base)[0]}_macs.csv")
        with open(mac_csv_path, "w", newline="") as fmac:
            writer = csv.DictWriter(
                fmac,
                fieldnames=[
                    "mac",
                    "score",
                    "mean_prob",
                    "max_prob",
                    "windows_with_activity",
                    "high_prob_windows",
                    "total_packets",
                    "max_packets_in_window",
                    "peak_packets_per_sec",
                ],
            )
            writer.writeheader()
            for row in mac_summary:
                writer.writerow(row)

    if ip_summary:
        ip_csv_path = os.path.join(out_dir, f"{os.path.splitext(base)[0]}_ips.csv")
        with open(ip_csv_path, "w", newline="") as fip:
            writer = csv.DictWriter(
                fip,
                fieldnames=[
                    "ip",
                    "score",
                    "mean_prob",
                    "max_prob",
                    "windows_with_activity",
                    "high_prob_windows",
                    "total_packets",
                    "max_packets_in_window",
                    "peak_packets_per_sec",
                ],
            )
            writer.writeheader()
            for row in ip_summary:
                writer.writerow(row)

    mac_print = most_suspicious_mac.get("mac") if isinstance(most_suspicious_mac, dict) else "n/a"
    ip_print = most_suspicious_ip.get("ip") if isinstance(most_suspicious_ip, dict) else "n/a"
    print(
        f"[{os.path.basename(pcap)}] decision={per_file['decision']} max_prob={per_file['max_prob']} "
        f"num_attack_windows={per_file['num_attack_windows']} top_mac={mac_print} top_ip={ip_print}"
    )
    return per_file, csv_path, json_path


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pcaps", default=None, help='Glob like "samples/*.pcap" or file path; defaults to config preprocess.pcaps_glob')
    ap.add_argument("--out", default=None, help="Reports output directory (defaults to config paths.reports_dir)")
    # Optional overrides for robust file-decision thresholds/rules
    ap.add_argument("--tau-high", type=float, default=None)
    ap.add_argument("--tau-low", type=float, default=None)
    ap.add_argument("--min-attack-windows", type=int, default=None)
    ap.add_argument("--consecutive-required", type=int, default=None)
    ap.add_argument("--cooldown-windows", type=int, default=None)
    ap.add_argument("--disable-gate", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load artifacts
    save_dir = cfg["paths"]["artifacts_dir"]
    model, scaler, slimmer, meta, calib = _load_artifacts(save_dir, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    T = float(calib.get("temperature", 1.0))
    tau = float(calib.get("threshold", 0.5))

    # DecisionConfig defaults sourced from config with sane fallbacks
    dec_section = cfg.get("decision", {})
    dec_cfg = DecisionConfig(
        tau_high=float(dec_section.get("tau_high", 0.70)),
        tau_low=float(dec_section.get("tau_low", 0.55)),
        min_attack_windows=int(dec_section.get("min_attack_windows", 3)),
        consecutive_required=int(dec_section.get("consecutive_required", 2)),
        cooldown_windows=int(dec_section.get("cooldown_windows", 1)),
        enable_gate=bool(dec_section.get("enable_gate", True)),
        warmup_windows=int(dec_section.get("warmup_windows", 60)),
        abs_pkts_per_s_cap=float(dec_section.get("abs_pkts_per_s_cap", 1500.0)),
        burstiness_multiple=float(dec_section.get("burstiness_multiple", 3.0)),
        syn_completion_max=float(dec_section.get("syn_completion_max", 0.10)),
        high_confidence_override=bool(dec_section.get("high_confidence_override", False)),
        high_conf_tau=float(dec_section.get("high_conf_tau", 0.93)),
        high_conf_windows=int(dec_section.get("high_conf_windows", 3)),
        min_pkts_per_s_for_override=float(dec_section.get("min_pkts_per_s_for_override", 3.0)),
    )
    # Allow optional overrides via CLI
    if args.tau_high is not None:               dec_cfg.tau_high = float(args.tau_high)
    if args.tau_low is not None:                dec_cfg.tau_low = float(args.tau_low)
    if args.min_attack_windows is not None:     dec_cfg.min_attack_windows = int(args.min_attack_windows)
    if args.consecutive_required is not None:   dec_cfg.consecutive_required = int(args.consecutive_required)
    if args.cooldown_windows is not None:       dec_cfg.cooldown_windows = int(args.cooldown_windows)
    if args.disable_gate:                       dec_cfg.enable_gate = False

    # Expand pcaps
    pcaps_arg = args.pcaps or cfg["preprocess"]["pcaps_glob"]
    pcaps = sorted(glob.glob(pcaps_arg)) if any(ch in pcaps_arg for ch in "*?[]") else [pcaps_arg]
    assert pcaps, f"No pcaps matched {pcaps_arg}"
    reports_dir = args.out or cfg["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)

    for p in tqdm(pcaps, desc="PCAPs", unit="file"):
        run_on_pcap(p, cfg, model, scaler, slimmer, meta, T, tau, reports_dir, dec_cfg, device)


if __name__ == "__main__":
    main()
