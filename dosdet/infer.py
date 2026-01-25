# infer.py
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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
    model.load_state_dict(state["model"], strict=False)
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

    rows_iter = iter_rows_from_pcap(pcap, cfg["features"]["ssdp_multicast_ipv4"], cfg["features"]["ssdp_multicast_ipv6"])
    win_iter = iter_windows(rows_iter, W, S, M)

    base = os.path.basename(pcap)
    csv_path = os.path.join(out_dir, f"{os.path.splitext(base)[0]}_windows.csv")
    json_path = os.path.join(out_dir, f"{os.path.splitext(base)[0]}.json")

    fieldnames = [
        "t_start",
        "t_end",
        "prob",
    ]

    window_obs: List[WindowObs] = []
    max_prob_seen = 0.0

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for (t0, t1, win_rows, bins) in tqdm(win_iter, desc=f"Windows {base}", unit="win"):
            if not win_rows:
                continue

            seq, extras = compute_sequence_features(win_rows, bins, M, list(meta.get("top_k_udp_ports", cfg["data"]["top_k_udp_ports"])))
            static_vec, _, snaps = compute_static_features(win_rows, M, extras.get("per_bin_total_pkts", []), list(meta.get("top_k_udp_ports", cfg["data"]["top_k_udp_ports"])), W)

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

            writer.writerow({
                "t_start": _iso(t0),
                "t_end": _iso(t1),
                "prob": f"{prob:.6f}",
            })

    file_decision = decide_file(
        file_path=os.path.abspath(pcap),
        windows=window_obs,
        cfg=decid_cfg,
    )

    result_payload = {
        "file": base,
        "decision": file_decision.decision,
        "num_attack_windows": int(file_decision.num_attack_windows),
        "max_probability": round(file_decision.max_prob, 6),
        "first_attack_window_ts": _iso(file_decision.first_attack_timestamp),
    }
    with open(json_path, "w", encoding="utf-8") as fj:
        json.dump(result_payload, fj, indent=2)

    return result_payload, csv_path, json_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run calibrated inference over pcaps.")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--pcaps", required=True, help="PCAP glob or file")
    ap.add_argument("--out", default=None, help="Output directory")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pcaps = glob.glob(args.pcaps)
    if not pcaps and os.path.isfile(args.pcaps):
        pcaps = [args.pcaps]
    if not pcaps:
        raise FileNotFoundError(f"No PCAPs found for {args.pcaps}")

    save_dir = cfg["paths"]["artifacts_dir"]
    model, scaler, slimmer, meta, calib = _load_artifacts(save_dir, cfg)
    device = torch.device(args.device)
    model.to(device)

    temperature = float(calib.get("temperature", 1.0))
    out_dir = args.out or cfg["paths"]["reports_dir"]
    decid_cfg = DecisionConfig(**cfg.get("decision", {}))

    for pcap in pcaps:
        run_on_pcap(pcap, cfg, model, scaler, slimmer, meta, temperature, out_dir, decid_cfg, device)


if __name__ == "__main__":
    main()
