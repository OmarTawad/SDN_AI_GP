from __future__ import annotations

# Small-window inference helper for ARP detector.
# Loads pretrained artifacts, extracts the first N windows from a PCAP, and measures wall-clock inference time.

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

# Constrain BLAS threads early for 2 vCPU deployments
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import numpy as np
import torch
import yaml

from data.pcap_reader import iter_rows_from_pcap
from data.windowizer import iter_windows
from features.seq_features import compute_sequence_features
from features.static_features import compute_static_features
from infer import _load_artifacts

torch.set_num_threads(min(2, max(1, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except (AttributeError, RuntimeError):
    pass


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _color_label(label: str) -> str:
    if os.environ.get("NO_COLOR"):
        return label
    if label == "attack":
        return f"\033[31m{label}\033[0m"
    if label == "normal":
        return f"\033[32m{label}\033[0m"
    return label


def _reply_conflict_stats(win_rows: list[dict]) -> tuple[int, int]:
    mac_to_ips: dict[str, set[str]] = {}
    for row in win_rows:
        if int(row.get("is_arp", 0)) != 1:
            continue
        if int(row.get("arp_opcode") or 0) != 2:
            continue
        mac = (row.get("arp_sender_mac") or row.get("src_mac") or "").lower()
        ip_addr = row.get("arp_sender_ip") or ""
        if not mac or not ip_addr:
            continue
        mac_to_ips.setdefault(mac, set()).add(ip_addr)
    conflict_macs = 0
    conflict_ip_count = 0
    for ips in mac_to_ips.values():
        if len(ips) >= 2:
            conflict_macs += 1
            conflict_ip_count += len(ips)
    return conflict_macs, conflict_ip_count


def _infer_window(
    window: tuple[float, float, list[dict], list[int]],
    cfg: Dict[str, Any],
    model,
    scaler,
    slimmer,
    meta: dict,
    T: float,
    tau: float,
    window_sec: float,
    micro_bins: int,
    device: torch.device,
) -> Dict[str, Any]:
    t0, t1, win_rows, bin_idx = window
    seq, extras = compute_sequence_features(win_rows, bin_idx, micro_bins)
    static_vec, static_names, snaps = compute_static_features(win_rows, micro_bins, extras, window_sec)

    feature_names = getattr(scaler, "feature_names_", None) or getattr(slimmer, "src_names", None)
    if feature_names is None or len(feature_names) != static_vec.size:
        feature_names = [f"f_{i}" for i in range(static_vec.size)]
    stat_scaled = scaler.transform(static_vec.reshape(1, -1), feature_names)
    stat_slim = slimmer.transform(stat_scaled)

    seq_t = torch.from_numpy(seq).unsqueeze(0).to(device).float()
    static_t = torch.from_numpy(stat_slim).to(device).float()

    with torch.no_grad():
        out = model(seq_t, static_t)
        logit = float(out["logits"].float().cpu().numpy().ravel()[0])
        attn_peak_bin = None
        attn = out.get("attn")
        if attn is not None:
            attn_arr = attn.cpu().numpy().ravel()
            attn_peak_bin = int(np.argmax(attn_arr))

    logit_T = logit / max(float(T), 1e-3)
    logit_T = max(min(logit_T, 60.0), -60.0)
    prob = float(torch.sigmoid(torch.tensor(logit_T)).item())
    label = "attack" if prob >= tau else "normal"

    reply_conflict_macs, reply_conflict_ip_count = _reply_conflict_stats(win_rows)

    return {
        "window_start": float(t0),
        "window_end": float(t1),
        "window_start_iso": _iso(float(t0)),
        "window_end_iso": _iso(float(t1)),
        "packets_in_window": len(win_rows),
        "prob": round(prob, 6),
        "logit": round(logit, 6),
        "label": label,
        "threshold": tau,
        "temperature": T,
        "attn_peak_bin": attn_peak_bin,
        "window_sec": window_sec,
        "micro_bins": micro_bins,
        "arp_packets": int(snaps.get("arp_packets", 0.0)),
        "arp_requests": int(snaps.get("arp_requests", 0.0)),
        "arp_replies": int(snaps.get("arp_replies", 0.0)),
        "sender_conflict_ratio": float(snaps.get("sender_conflict_ratio", 0.0)),
        "conflict_ip_ratio": float(snaps.get("conflict_ip_ratio", 0.0)),
        "reply_conflict_ratio": float(snaps.get("reply_conflict_ratio", 0.0)),
        "broadcast_fraction": float(snaps.get("broadcast_fraction", 0.0)),
        "reply_conflict_macs": int(reply_conflict_macs),
        "reply_conflict_ip_count": int(reply_conflict_ip_count),
    }


def infer_windows(
    pcap_path: str,
    cfg: Dict[str, Any],
    model,
    scaler,
    slimmer,
    meta: dict,
    T: float,
    tau: float,
    window_sec: float,
    micro_bins: int,
    device: torch.device,
    num_windows: int,
) -> list[Dict[str, Any]]:
    rows_iter = iter_rows_from_pcap(pcap_path)
    win_iter = iter_windows(rows_iter, window_sec, window_sec, micro_bins)

    results: list[Dict[str, Any]] = []
    while len(results) < max(1, int(num_windows)):
        win = next(win_iter, None)
        if win is None:
            break
        t0, t1, win_rows, bin_idx = win
        if not win_rows:
            continue
        result = _infer_window(
            (t0, t1, win_rows, bin_idx),
            cfg,
            model,
            scaler,
            slimmer,
            meta,
            T,
            tau,
            window_sec,
            micro_bins,
            device,
        )
        results.append(result)

    if not results:
        raise ValueError(f"No packets found in {pcap_path}")
    return results


def infer_first_window(
    pcap_path: str,
    cfg: Dict[str, Any],
    model,
    scaler,
    slimmer,
    meta: dict,
    T: float,
    tau: float,
    window_sec: float,
    micro_bins: int,
    device: torch.device,
) -> Dict[str, Any]:
    results = infer_windows(
        pcap_path,
        cfg,
        model,
        scaler,
        slimmer,
        meta,
        T,
        tau,
        window_sec,
        micro_bins,
        device,
        num_windows=1,
    )
    return results[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Small-window inference with timing output.")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--pcap", required=True, help="PCAP file to read (first N windows)")
    ap.add_argument("--window-sec", type=float, default=None, help="Window length in seconds (default: config.windowing.window_sec)")
    ap.add_argument("--micro-bins", type=int, default=None, help="Micro-bins across the window (default: model meta or config)")
    ap.add_argument("--num-windows", type=int, default=1, help="Number of windows to infer (default: 1)")
    ap.add_argument("--tau", type=float, default=None, help="Optional decision threshold override (default: calibrated threshold)")
    ap.add_argument("--out", default=None, help="Optional directory to write a JSON result")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pcap_path = args.pcap
    if not os.path.isfile(pcap_path):
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    save_dir = cfg["paths"]["artifacts_dir"]
    quantized = True
    model, scaler, slimmer, meta, calib = _load_artifacts(
        save_dir, cfg, quantized=quantized, quantized_checkpoint=None, quant_backend=None
    )
    device = torch.device("cpu")

    T = float(calib.get("temperature", 1.0))
    tau = float(args.tau if args.tau is not None else calib.get("threshold", 0.5))
    window_sec = float(args.window_sec if args.window_sec is not None else cfg["windowing"]["window_sec"])
    micro_bins = int(args.micro_bins if args.micro_bins is not None else meta.get("micro_bins", cfg["windowing"]["micro_bins"]))

    num_windows = max(1, int(args.num_windows))
    start = time.perf_counter()
    if num_windows == 1:
        result = infer_first_window(
            pcap_path, cfg, model, scaler, slimmer, meta, T, tau, window_sec, micro_bins, device
        )
        result["pcap"] = os.path.abspath(pcap_path)
        result["inference_time_sec"] = round(time.perf_counter() - start, 6)
        colored_label = _color_label(result["label"])
        print(
            f"[{os.path.basename(pcap_path)}] label={colored_label} prob={result['prob']} "
            f"time={result['inference_time_sec']}s packets={result['packets_in_window']} "
            f"arp_replies={result['arp_replies']} reply_conflict_macs={result['reply_conflict_macs']} "
            f"window={result['window_start_iso']} -> {result['window_end_iso']} attn_peak_bin={result['attn_peak_bin']}"
        )
        payload = result
        suffix = "single_window"
    else:
        windows = infer_windows(
            pcap_path, cfg, model, scaler, slimmer, meta, T, tau, window_sec, micro_bins, device, num_windows
        )
        total_time = round(time.perf_counter() - start, 6)
        for idx, win in enumerate(windows, start=1):
            colored_label = _color_label(win["label"])
            print(
                f"[{os.path.basename(pcap_path)}] window={idx}/{len(windows)} label={colored_label} "
                f"prob={win['prob']} packets={win['packets_in_window']} "
                f"arp_replies={win['arp_replies']} reply_conflict_macs={win['reply_conflict_macs']} "
                f"window={win['window_start_iso']} -> {win['window_end_iso']} attn_peak_bin={win['attn_peak_bin']}"
            )
        payload = {
            "pcap": os.path.abspath(pcap_path),
            "num_windows_requested": num_windows,
            "num_windows_inferred": len(windows),
            "window_sec": window_sec,
            "micro_bins": micro_bins,
            "threshold": tau,
            "temperature": T,
            "total_inference_time_sec": total_time,
            "windows": windows,
        }
        suffix = f"first_{num_windows}_windows"

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        base = os.path.splitext(os.path.basename(pcap_path))[0]
        json_path = os.path.join(args.out, f"{base}_{suffix}.json")
        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump(payload, fj, indent=2)
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
