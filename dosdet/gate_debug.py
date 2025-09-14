# gate_debug.py
from __future__ import annotations
import os, argparse, json
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.pcap_reader import iter_rows_from_pcap
from data.windowizer import iter_windows
from features.seq_features import compute_sequence_features
from features.static_features import compute_static_features
from features.scaler import RobustScaler

from decision import DecisionConfig, WindowObs, gate_checks
from decision import _build_quiet_baselines as build_quiet_baselines


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _load(save_dir: str, cfg: dict):
    from features.feature_slimming import StaticSlimmer
    from models.dws_cnn import FastDetector

    scaler = RobustScaler.load(save_dir)
    slimmer = StaticSlimmer(out_dim=1)
    slimmer.load(save_dir)

    meta = json.load(open(os.path.join(save_dir, "feature_model_meta.json"), "r"))
    seq_in_dim = int(meta["seq_in_dim"])
    static_dim = int(meta["static_dim"])

    trn = cfg["training"]
    model = FastDetector(
        seq_in_dim=seq_in_dim,
        static_dim=static_dim,
        channels=tuple(trn["channels"]),
        k=trn["kernel_size"],
        drop=trn["dropout"],
        mlp_hidden=tuple(trn["mlp_hidden"]),
        aux_family_head=bool(trn.get("aux_family_head", False)),
        n_families=6
    )
    state = torch.load(os.path.join(save_dir, "model_best.pt"), map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    calib_path = os.path.join(save_dir, "calibration.json")
    T = 1.0
    if os.path.exists(calib_path):
        calib = json.load(open(calib_path, "r"))
        T = float(calib.get("temperature", 1.0))

    return model, scaler, slimmer, T


def _build_snapshots(win_rows: List[Dict], W: float, M: int, extras: Dict, snaps: Dict,
                     ssdp_v4: str, ssdp_v6: str) -> Dict:
    out = dict(snaps)
    total_pkts = float(len(win_rows))
    total_tcp  = float(sum(int(r.get("is_tcp", 0)) for r in win_rows))
    total_udp  = float(sum(int(r.get("is_udp", 0)) for r in win_rows))
    out["total_pkts"] = total_pkts
    out["total_tcp"]  = total_tcp
    out["total_udp"]  = total_udp
    out["pkts_per_s"] = total_pkts / max(W, 1e-6)

    per_bin = extras.get("per_bin_total_pkts", [])
    if isinstance(per_bin, (list, np.ndarray)) and len(per_bin) > 0:
        arr = np.asarray(per_bin, dtype=float)
        out["median_bin_pkts"] = float(np.median(arr))
        out["max_bin_pkts"]    = float(np.max(arr))
    else:
        out["median_bin_pkts"] = out.get("median_bin_pkts", 1.0)
        out["max_bin_pkts"]    = out.get("max_bin_pkts", total_pkts)

    syn_ct    = sum(int(r.get("tcp_syn", 0)) for r in win_rows)
    synack_ct = sum(int(r.get("tcp_synack", 0)) for r in win_rows)
    out["tcp_syn_count"]         = float(syn_ct)
    out["tcp_syn_rate"]          = float(syn_ct) / max(W, 1e-6)
    out["tcp_synack_completion"] = float(synack_ct) / max(syn_ct + 1e-6, 1e-6)
    out.setdefault("tcp_syn_over_synack", snaps.get("tcp_syn_over_synack", 0.0))

    dst_ip_lower_set = {"239.255.255.250", str(ssdp_v6).lower(), "ff02::c", "ff0e::c"}
    out["ssdp_multicast_hit"] = 1.0 if any(
        str(r.get("dst_ip", "")).lower() in dst_ip_lower_set for r in win_rows
    ) else 0.0

    udp_total = int(total_udp)
    udp1900 = 0
    msearch_ct = 0
    ok200_ct   = 0
    notify_ct  = 0
    for r in win_rows:
        if not int(r.get("is_udp", 0)): continue
        dport = r.get("dst_port")
        sport = r.get("src_port")
        try:
            if int(dport) == 1900 or int(sport) == 1900:
                udp1900 += 1
                m = (r.get("ssdp_method") or "").upper()
                if m == "M-SEARCH": msearch_ct += 1
                elif m in ("200-OK","200 OK"): ok200_ct   += 1
                elif m == "NOTIFY": notify_ct  += 1
        except Exception:
            pass
    out["udp_1900_fraction"] = float(udp1900) / max(udp_total, 1)
    out["ssdp_msearch_count"] = float(msearch_ct)
    out["ssdp_200ok_count"]   = float(ok200_ct)
    out["ssdp_notify_count"]  = float(notify_ct)
    if "udp_ssdp_req_over_resp" not in out:
        out["udp_ssdp_req_over_resp"] = float(msearch_ct + notify_ct) / max(ok200_ct, 1)

    out.setdefault("H_src_ip",  snaps.get("H_src_ip",  8.0))
    out.setdefault("H_dst_port",snaps.get("H_dst_port",8.0))
    out.setdefault("H_ttl",     snaps.get("H_ttl",     8.0))
    out.setdefault("generic_req_resp_ratio", snaps.get("generic_req_resp_ratio", 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    save_dir = cfg["logging"]["save_dir"]
    model, scaler, slimmer, T = _load(save_dir, cfg)

    W = float(cfg["windowing"]["window_sec"])
    S = float(cfg["windowing"]["stride_sec"])
    M = int(cfg["windowing"]["micro_bins"])
    top_ports = list(cfg["data"]["top_k_udp_ports"])
    ssdp_v4 = cfg["features"]["ssdp_multicast_ipv4"]
    ssdp_v6 = cfg["features"]["ssdp_multicast_ipv6"]

    rows_iter = iter_rows_from_pcap(args.pcap, ssdp_v4, ssdp_v6)
    win_iter = iter_windows(rows_iter, W, S, M)

    windows: List[WindowObs] = []
    for (t0, t1, win_rows, bins) in tqdm(win_iter, desc=f"Windows {os.path.basename(args.pcap)}", unit="win"):
        if not win_rows: continue
        seq, extras = compute_sequence_features(win_rows, bins, M, top_ports)
        static_vec, static_names, snaps = compute_static_features(win_rows, M, extras.get("per_bin_total_pkts", []), top_ports, W)

        names_stub = [f"f_{i}" for i in range(static_vec.size)]
        try:
            stat_scaled = scaler.transform(static_vec.reshape(1, -1), names_stub)
        except Exception:
            continue
        stat_slim = slimmer.transform(stat_scaled)

        with torch.no_grad():
            seq_t = torch.from_numpy(seq).unsqueeze(0).float()
            static_t = torch.from_numpy(stat_slim).float()
            out = model(seq_t, static_t)
            logit = float(out["logits"].cpu().numpy().ravel()[0])
            prob = float(1.0 / (1.0 + np.exp(-logit / max(T, 1e-3))))

        snaps_full = _build_snapshots(win_rows, W, M, extras, snaps, ssdp_v4, ssdp_v6)
        windows.append(WindowObs(prob=prob, snaps=snaps_full, t0=float(t0), t1=float(t1)))

    base = build_quiet_baselines(windows)

    idx = np.argsort([w.prob for w in windows])[::-1][:max(1, args.topk)]
    rows = []
    print("\n=== Top-K windows by model probability ===")
    print("rank  prob    t_start (UTC)                pkts/s  gate  reasons / key-fields")
    print("----  ------  ---------------------------  ------  ----  ---------------------")
    for rank, i in enumerate(idx, 1):
        w = windows[i]
        checks = gate_checks(w.snaps, base, DecisionConfig())
        gate = "PASS" if checks["gate_pass"] else "----"
        reason_bits = []
        if checks["rate_spike"]:        reason_bits.append("rate-spike")
        if checks["asymmetry"]:         reason_bits.append("asym")
        if checks["entropy_collapse"]:  reason_bits.append("entropy")
        if checks["burstiness"]:        reason_bits.append("burst")
        if checks["ssdp_gate"]:         reason_bits.append("SSDP")
        if checks["syn_gate"]:          reason_bits.append("SYN")

        print(f"{rank:>4}  {w.prob:>6.3f}  {_iso(w.t0):<27}  {checks['pkts_per_s']:>6.0f}  {gate:>4}  "
              f"{','.join(reason_bits) or '—'}")
        rows.append({
            "rank": rank,
            "prob": w.prob,
            "t_start": _iso(w.t0),
            "t_end": _iso(w.t1),
            "pkts_per_s": checks["pkts_per_s"],
            "gate_pass": bool(checks["gate_pass"]),
            "reasons": reason_bits,
            "checks": checks,
        })

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"pcap": os.path.abspath(args.pcap), "baselines": base, "topk": rows}, f, indent=2)
        print(f"\nWrote debug JSON -> {args.out_json}")


if __name__ == "__main__":
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    main()
