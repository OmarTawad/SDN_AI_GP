# infer.py
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import argparse
import glob
import json
import csv
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

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
from decision import DecisionConfig, WindowObs, decide_file


def _load_artifacts(save_dir: str, cfg: dict):
    from features.feature_slimming import StaticSlimmer
    from models.dws_cnn import FastDetector

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

    calib_path = os.path.join(save_dir, "calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
    else:
        calib = {"temperature": 1.0, "threshold": 0.5}

    return model, scaler, slimmer, meta, calib


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


class ArpClaimTracker:
    """Track ARP reply claims to surface suspicious MAC addresses."""

    def __init__(self) -> None:
        self._stats: Dict[str, Dict[str, object]] = {}

    def observe(self, mac_to_ip_counts: Dict[str, Dict[str, int]], prob: float, window_start: float | None = None) -> None:
        for mac, ip_counts in mac_to_ip_counts.items():
            stats = self._stats.setdefault(
                mac,
                {
                    "total_replies": 0,
                    "windows": 0,
                    "conflict_windows": 0,
                    "prob_weighted_packets": 0.0,
                    "max_prob": 0.0,
                    "claimed_ips": {},
                    "first_conflict_ts": None,
                },
            )
            reply_count = sum(ip_counts.values())
            stats["total_replies"] = int(stats["total_replies"]) + int(reply_count)
            stats["windows"] = int(stats["windows"]) + 1
            stats["prob_weighted_packets"] = float(stats["prob_weighted_packets"]) + float(prob * reply_count)
            stats["max_prob"] = max(float(stats["max_prob"]), float(prob))

            claimed = stats.setdefault("claimed_ips", {})
            nonzero_ips = []
            for ip, cnt in ip_counts.items():
                if cnt > 0:
                    claimed[ip] = int(claimed.get(ip, 0)) + int(cnt)
                    nonzero_ips.append(ip)

            if len(nonzero_ips) > 1:
                stats["conflict_windows"] = int(stats["conflict_windows"]) + 1
                if window_start is not None and stats.get("first_conflict_ts") in (None, float("inf")):
                    stats["first_conflict_ts"] = float(window_start)

    def _build_entry(self, mac: str, stats: Dict[str, object]) -> Tuple[Dict[str, object], float]:
        claimed_raw = stats.get("claimed_ips", {})
        if isinstance(claimed_raw, dict):
            items = claimed_raw.items()
        else:
            items = []
        ranked_claims = sorted(
            ((ip, int(cnt)) for ip, cnt in items if cnt),
            key=lambda item: item[1],
            reverse=True,
        )
        distinct_ips = len(ranked_claims)
        total_replies = int(stats.get("total_replies", 0))
        windows = int(stats.get("windows", 0))
        conflict_windows = int(stats.get("conflict_windows", 0))
        prob_weighted = float(stats.get("prob_weighted_packets", 0.0))
        max_prob = float(stats.get("max_prob", 0.0))

        score = (
            prob_weighted
            + max_prob * 10.0
            + total_replies * 0.01
            + windows * 0.1
            + conflict_windows * 500.0
            + max(0, distinct_ips - 1) * 1000.0
        )

        entry = {
            "mac": mac,
            "distinct_ips": distinct_ips,
            "total_replies": total_replies,
            "windows": windows,
            "conflict_windows": conflict_windows,
            "max_prob": round(max_prob, 6),
            "score": round(score, 6),
            "claimed_ips": [
                {"ip": ip, "replies": cnt}
                for ip, cnt in ranked_claims[:5]
            ],
        }

        first_conflict = stats.get("first_conflict_ts")
        entry["first_conflict_ts"] = float(first_conflict) if isinstance(first_conflict, (float, int)) else None
        return entry, score

    def best(self) -> Dict[str, object] | None:
        best_entry: Dict[str, object] | None = None
        best_score = float("-inf")
        for mac, raw_stats in self._stats.items():
            entry, score = self._build_entry(mac, raw_stats)
            if score > best_score:
                best_score = score
                best_entry = entry
        return best_entry

    def best_conflict(self) -> Dict[str, object] | None:
        best_entry: Dict[str, object] | None = None
        best_score = float("-inf")
        for mac, raw_stats in self._stats.items():
            entry, score = self._build_entry(mac, raw_stats)
            if entry["distinct_ips"] >= 2 and score > best_score:
                best_score = score
                best_entry = entry
        return best_entry


def run_on_pcap(
    pcap: str,
    cfg: dict,
    model,
    scaler,
    slimmer,
    meta: dict,
    temperature: float,
    out_dir: str,
    decid_cfg: DecisionConfig,
    device: torch.device,
) -> Tuple[Dict, str, str]:
    os.makedirs(out_dir, exist_ok=True)

    W = float(cfg["windowing"]["window_sec"])
    S = float(cfg["windowing"]["stride_sec"])
    M = int(meta.get("micro_bins", cfg["windowing"]["micro_bins"]))

    rows_iter = iter_rows_from_pcap(pcap)
    win_iter = iter_windows(rows_iter, W, S, M)

    base = os.path.basename(pcap)
    csv_path = os.path.join(out_dir, f"{os.path.splitext(base)[0]}_windows.csv")
    json_path = os.path.join(out_dir, f"{os.path.splitext(base)[0]}.json")

    fieldnames = [
        "t_start",
        "t_end",
        "prob",
        "arp_packets",
        "arp_requests",
        "arp_replies",
        "sender_conflict_ratio",
        "conflict_ip_ratio",
        "reply_conflict_ratio",
        "max_claims_per_ip",
        "max_ips_per_mac",
        "max_conflicting_senders_bin",
        "max_conflicting_targets_bin",
        "broadcast_fraction",
        "reply_conflict_macs",
    ]

    window_obs: List[WindowObs] = []
    tracker = ArpClaimTracker()
    max_prob_seen = 0.0

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for (t0, t1, win_rows, bins) in tqdm(win_iter, desc=f"Windows {base}", unit="win"):
            if not win_rows:
                continue

            seq, extras = compute_sequence_features(win_rows, bins, M)
            static_vec, _, snaps = compute_static_features(win_rows, M, extras, W)

            feature_names = getattr(scaler, "feature_names_", None) or getattr(slimmer, "src_names", None)
            if feature_names is None or len(feature_names) != static_vec.size:
                feature_names = [f"f_{i}" for i in range(static_vec.size)]

            try:
                stat_scaled = scaler.transform(static_vec.reshape(1, -1), feature_names)
            except Exception:
                continue
            stat_slim = slimmer.transform(stat_scaled)

            seq_t = torch.from_numpy(seq).unsqueeze(0).to(device).float()
            static_t = torch.from_numpy(stat_slim).to(device).float()

            with torch.no_grad():
                out = model(seq_t, static_t)
                logit = float(out["logits"].cpu().numpy().ravel()[0])
                logit_T = logit / max(float(temperature), 1e-3)
                prob = float(1.0 / (1.0 + np.exp(-logit_T)))
                max_prob_seen = max(max_prob_seen, prob)

            window_obs.append(WindowObs(prob=prob, snaps=snaps, t0=float(t0), t1=float(t1)))

            mac_to_ip_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for row in win_rows:
                if int(row.get("is_arp", 0)) != 1:
                    continue
                if int(row.get("arp_opcode") or 0) != 2:
                    continue
                mac = (row.get("arp_sender_mac") or row.get("src_mac") or "").lower()
                ip_addr = row.get("arp_sender_ip") or ""
                if not mac or not ip_addr:
                    continue
                mac_to_ip_counts[mac][ip_addr] = mac_to_ip_counts[mac].get(ip_addr, 0) + 1

            conflicting_macs = 0
            conflict_ip_totals = 0
            for ip_count_map in mac_to_ip_counts.values():
                distinct_ip_in_window = sum(1 for cnt in ip_count_map.values() if cnt > 0)
                if distinct_ip_in_window >= 2:
                    conflicting_macs += 1
                    conflict_ip_totals += distinct_ip_in_window

            snaps["reply_conflict_macs"] = float(conflicting_macs)
            snaps["reply_conflict_ip_count"] = float(conflict_ip_totals)

            tracker.observe(mac_to_ip_counts, prob, float(t0))

            writer.writerow({
                "t_start": _iso(t0),
                "t_end": _iso(t1),
                "prob": f"{prob:.6f}",
                "arp_packets": int(snaps.get("arp_packets", 0.0)),
                "arp_requests": int(snaps.get("arp_requests", 0.0)),
                "arp_replies": int(snaps.get("arp_replies", 0.0)),
                "sender_conflict_ratio": f"{snaps.get('sender_conflict_ratio', 0.0):.6f}",
                "conflict_ip_ratio": f"{snaps.get('conflict_ip_ratio', 0.0):.6f}",
                "reply_conflict_ratio": f"{snaps.get('reply_conflict_ratio', 0.0):.6f}",
                "max_claims_per_ip": f"{snaps.get('max_claims_per_ip', 0.0):.6f}",
                "max_ips_per_mac": f"{snaps.get('max_ips_per_mac', 0.0):.6f}",
                "max_conflicting_senders_bin": f"{snaps.get('max_conflicting_senders_bin', 0.0):.6f}",
                "max_conflicting_targets_bin": f"{snaps.get('max_conflicting_targets_bin', 0.0):.6f}",
                "broadcast_fraction": f"{snaps.get('broadcast_fraction', 0.0):.6f}",
                "reply_conflict_macs": int(conflicting_macs),
            })

    file_decision = decide_file(
        file_path=os.path.abspath(pcap),
        windows=window_obs,
        cfg=decid_cfg,
    )

    conflict_mac = tracker.best_conflict()
    overall_mac = tracker.best()
    suspicious_mac = conflict_mac or overall_mac

    raw_conflict_ts = conflict_mac.get("first_conflict_ts") if conflict_mac else None
    if suspicious_mac is not None:
        fc_ts = suspicious_mac.get("first_conflict_ts")
        if isinstance(fc_ts, (int, float)):
            suspicious_mac["first_conflict_ts"] = _iso(float(fc_ts))
        else:
            suspicious_mac["first_conflict_ts"] = None

    has_conflict = conflict_mac is not None
    gate_reasons = list(file_decision.gate_reasons)

    if has_conflict and "mac_conflict" not in gate_reasons:
        gate_reasons.append("mac_conflict")
    if not has_conflict:
        final_decision = "normal"
        binary_head = 0
        num_attack_windows = 0
        gate_reasons = []
        first_attack_ts = None
    else:
        final_decision = "attack"
        binary_head = 1
        conflict_windows = int(conflict_mac.get("conflict_windows", 0)) if conflict_mac else 0
        num_attack_windows = max(int(file_decision.num_attack_windows), conflict_windows or 1)
        first_attack_ts = raw_conflict_ts if raw_conflict_ts is not None else file_decision.first_attack_timestamp

    result_payload = {
        "file": base,
        "binary_head": binary_head,
        "decision": final_decision,
        "num_attack_windows": int(num_attack_windows),
        "max_probability": round(file_decision.max_prob, 6),
        "gate_reasons": gate_reasons,
        "first_attack_window_ts": _iso(first_attack_ts),
        "mac_conflict_detected": has_conflict,
        "suspicious_mac": suspicious_mac,
    }

    with open(json_path, "w", encoding="utf-8") as fj:
        json.dump(result_payload, fj, indent=2)

    mac_print = suspicious_mac.get("mac") if suspicious_mac else "n/a"
    print(
        f"[{base}] decision={final_decision} max_prob={file_decision.max_prob:.4f} "
        f"num_attack_windows={num_attack_windows} suspicious_mac={mac_print}"
    )
    return result_payload, csv_path, json_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pcaps", default=None, help='Glob like "samples/*.pcap" or single file')
    ap.add_argument("--out", default=None, help="Output directory for reports (defaults to paths.reports_dir)")
    ap.add_argument("--tau-high", type=float, default=None)
    ap.add_argument("--tau-low", type=float, default=None)
    ap.add_argument("--min-attack-windows", type=int, default=None)
    ap.add_argument("--consecutive-required", type=int, default=None)
    ap.add_argument("--cooldown-windows", type=int, default=None)
    ap.add_argument("--disable-gate", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    save_dir = cfg["paths"]["artifacts_dir"]
    model, scaler, slimmer, meta, calib = _load_artifacts(save_dir, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    T = float(calib.get("temperature", 1.0))

    dec_section = cfg.get("decision", {})
    dec_cfg = DecisionConfig(
        tau_high=float(dec_section.get("tau_high", 0.70)),
        tau_low=float(dec_section.get("tau_low", 0.55)),
        min_attack_windows=int(dec_section.get("min_attack_windows", 2)),
        consecutive_required=int(dec_section.get("consecutive_required", 1)),
        cooldown_windows=int(dec_section.get("cooldown_windows", 1)),
        enable_gate=bool(dec_section.get("enable_gate", True)),
        min_arp_replies=float(dec_section.get("min_arp_replies", 2.0)),
        min_conflict_macs=int(dec_section.get("min_conflict_macs", 1)),
    )

    if args.tau_high is not None:
        dec_cfg.tau_high = float(args.tau_high)
    if args.tau_low is not None:
        dec_cfg.tau_low = float(args.tau_low)
    if args.min_attack_windows is not None:
        dec_cfg.min_attack_windows = int(args.min_attack_windows)
    if args.consecutive_required is not None:
        dec_cfg.consecutive_required = int(args.consecutive_required)
    if args.cooldown_windows is not None:
        dec_cfg.cooldown_windows = int(args.cooldown_windows)
    if args.disable_gate:
        dec_cfg.enable_gate = False

    pcaps_arg = args.pcaps or cfg["preprocess"]["pcaps_glob"]
    pcaps = sorted(glob.glob(pcaps_arg)) if any(ch in pcaps_arg for ch in "*?[]") else [pcaps_arg]
    assert pcaps, f"No pcaps matched {pcaps_arg}"

    reports_dir = args.out or cfg["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)

    for p in tqdm(pcaps, desc="PCAPs", unit="file"):
        run_on_pcap(p, cfg, model, scaler, slimmer, meta, T, reports_dir, dec_cfg, device)


if __name__ == "__main__":
    main()
