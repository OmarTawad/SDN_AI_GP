from __future__ import annotations

# Single-window inference helper for ARP detector.
# Loads pretrained artifacts, extracts the first window from a PCAP, and measures wall-clock inference time.

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict
from contextlib import nullcontext

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
    use_amp: bool,
) -> Dict[str, Any]:
    use_amp = bool(use_amp and device.type == "cuda")
    rows_iter = iter_rows_from_pcap(pcap_path)
    win_iter = iter_windows(rows_iter, window_sec, window_sec, micro_bins)
    first = next(win_iter, None)
    if first is None:
        raise ValueError(f"No packets found in {pcap_path}")

    t0, t1, win_rows, bin_idx = first
    if not win_rows:
        raise ValueError("First window was empty; try a longer PCAP or larger window.")

    seq, extras = compute_sequence_features(win_rows, bin_idx, micro_bins)
    static_vec, static_names, snaps = compute_static_features(win_rows, micro_bins, extras, window_sec)

    feature_names = getattr(scaler, "feature_names_", None) or getattr(slimmer, "src_names", None)
    if feature_names is None or len(feature_names) != static_vec.size:
        feature_names = [f"f_{i}" for i in range(static_vec.size)]
    stat_scaled = scaler.transform(static_vec.reshape(1, -1), feature_names)
    stat_slim = slimmer.transform(stat_scaled)

    dtype = torch.float16 if use_amp else torch.float32
    seq_t = torch.from_numpy(seq).unsqueeze(0).to(device=device, dtype=dtype)
    static_t = torch.from_numpy(stat_slim).to(device=device, dtype=dtype)

    with torch.no_grad():
        with (torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp else nullcontext()):
            out = model(seq_t, static_t)
        logit = float(out["logits"].detach().float().cpu().numpy().ravel()[0])
        attn_peak_bin = None
        attn = out.get("attn")
        if attn is not None:
            attn_arr = attn.cpu().numpy().ravel()
            attn_peak_bin = int(np.argmax(attn_arr))

    logit_T = logit / max(float(T), 1e-3)
    prob = float(1.0 / (1.0 + np.exp(-logit_T)))
    label = "attack" if prob >= tau else "normal"

    return {
        "pcap": os.path.abspath(pcap_path),
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
        "arp_replies": int(snaps.get("arp_replies", 0.0)),
        "sender_conflict_ratio": float(snaps.get("sender_conflict_ratio", 0.0)),
        "conflict_ip_ratio": float(snaps.get("conflict_ip_ratio", 0.0)),
        "reply_conflict_ratio": float(snaps.get("reply_conflict_ratio", 0.0)),
        "broadcast_fraction": float(snaps.get("broadcast_fraction", 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-window inference with timing output.")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--pcap", required=True, help="PCAP file to read (first window only)")
    ap.add_argument("--window-sec", type=float, default=None, help="Window length in seconds (default: config.windowing.window_sec)")
    ap.add_argument("--micro-bins", type=int, default=None, help="Micro-bins across the window (default: model meta or config)")
    ap.add_argument("--tau", type=float, default=None, help="Optional decision threshold override (default: calibrated threshold)")
    ap.add_argument("--out", default=None, help="Optional directory to write a JSON result")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pcap_path = args.pcap
    if not os.path.isfile(pcap_path):
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    save_dir = cfg["paths"]["artifacts_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("training", {}).get("amp", False) and device.type == "cuda")
    model, scaler, slimmer, meta, calib = _load_artifacts(save_dir, cfg, device, use_amp)

    T = float(calib.get("temperature", 1.0))
    tau = float(args.tau if args.tau is not None else calib.get("threshold", 0.5))
    window_sec = float(args.window_sec if args.window_sec is not None else cfg["windowing"]["window_sec"])
    micro_bins = int(args.micro_bins if args.micro_bins is not None else meta.get("micro_bins", cfg["windowing"]["micro_bins"]))

    start = time.perf_counter()
    result = infer_first_window(pcap_path, cfg, model, scaler, slimmer, meta, T, tau, window_sec, micro_bins, device, use_amp)
    result["inference_time_sec"] = round(time.perf_counter() - start, 6)

    print(
        f"[{os.path.basename(pcap_path)}] label={result['label']} prob={result['prob']} "
        f"time={result['inference_time_sec']}s packets={result['packets_in_window']} "
        f"arp_replies={result['arp_replies']} window={result['window_start_iso']} -> {result['window_end_iso']} "
        f"attn_peak_bin={result['attn_peak_bin']}"
    )

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        base = os.path.splitext(os.path.basename(pcap_path))[0]
        json_path = os.path.join(args.out, f"{base}_single_window.json")
        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump(result, fj, indent=2)
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
