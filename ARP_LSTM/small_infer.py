from __future__ import annotations

# Small-window inference helper for the ARP_LSTM detector.
# Loads pretrained artifacts, extracts the first N windows from a PCAP, and reports wall-clock latency.

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

# Constrain BLAS threads early for small CPU footprints
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Ensure local package is importable when running the script directly
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from arp_detector.config import load_config
from arp_detector.data.pcap_reader import iter_pcap
from arp_detector.data.structures import Window
from arp_detector.data.windowing import WindowBuilder, WindowingParams
from arp_detector.features.feature_engineering import FeatureExtractor
from arp_detector.models.supervised import SequenceClassifier
from arp_detector.utils.io import load_joblib, load_json

DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


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


def _load_model(cfg, feature_columns: Sequence[str], device: torch.device) -> SequenceClassifier:
    model = SequenceClassifier(
        input_size=len(feature_columns),
        num_attack_types=len(cfg.labels.family_mapping),
        config=cfg.model.supervised,
    ).to(device)
    state = torch.load(cfg.paths.supervised_model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _window_packet_counts(window: Window) -> tuple[int, int, int]:
    arp_packets = 0
    arp_requests = 0
    arp_replies = 0
    for pkt in window.packets:
        if (pkt.protocol or "").lower() != "arp":
            continue
        arp_packets += 1
        opcode = int(pkt.arp_opcode or 0)
        arp_requests += int(opcode == 1)
        arp_replies += int(opcode == 2)
    return arp_packets, arp_requests, arp_replies


def _reply_conflict_stats(window: Window) -> tuple[int, int]:
    mac_to_ips: dict[str, set[str]] = {}
    for pkt in window.packets:
        if (pkt.protocol or "").lower() != "arp":
            continue
        if int(pkt.arp_opcode or 0) != 2:
            continue
        mac = (pkt.arp_sender_mac or pkt.src_mac or "").lower()
        ip_addr = pkt.arp_sender_ip or ""
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


def _collect_windows(
    pcap_path: Path,
    window_sec: float,
    hop_sec: float,
    num_windows: int,
) -> list[Window]:
    packets = list(iter_pcap(pcap_path))
    if not packets:
        raise ValueError(f"No packets found in {pcap_path}")
    builder = WindowBuilder(WindowingParams(window_size=window_sec, hop_size=hop_sec))
    windows: list[Window] = []
    for window in builder.build(packets):
        if not window.packets:
            continue
        windows.append(window)
        if len(windows) >= num_windows:
            break
    if not windows:
        raise ValueError("No non-empty windows found; try a longer PCAP or larger window.")
    return windows


def infer_windows(
    pcap_path: Path,
    cfg,
    feature_columns: Sequence[str],
    model: SequenceClassifier,
    scaler,
    device: torch.device,
    num_windows: int,
    window_sec: float,
    hop_sec: float,
    tau: float,
) -> tuple[list[Dict[str, Any]], int | None]:
    windows = _collect_windows(pcap_path, window_sec, hop_sec, num_windows)
    extractor = FeatureExtractor(cfg.feature, window_sec)
    frame = extractor.extract(windows)
    if frame.empty:
        raise ValueError("Feature frame is empty for selected windows.")

    feature_matrix = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    scaled = scaler.transform(feature_matrix).astype(np.float32, copy=False)

    tensor = torch.from_numpy(scaled).to(device)
    tensor = tensor.unsqueeze(0)  # (batch=1, seq=N, features)

    with torch.no_grad():
        outputs = model(tensor)
        logits = outputs.window_logits.detach().cpu().numpy().ravel()
        probs = 1.0 / (1.0 + np.exp(-logits))
        attn_weights = None
        attn_peak = None
        if outputs.attention is not None:
            attn_weights = outputs.attention.detach().cpu().numpy().ravel()
            if attn_weights.size:
                attn_peak = int(np.argmax(attn_weights))

    results: list[Dict[str, Any]] = []
    for idx, window in enumerate(windows):
        logit = float(logits[idx]) if idx < logits.size else 0.0
        prob = float(probs[idx]) if idx < probs.size else 0.0
        label = "attack" if prob >= tau else "normal"
        arp_packets, arp_requests, arp_replies = _window_packet_counts(window)
        reply_conflict_macs, reply_conflict_ip_count = _reply_conflict_stats(window)

        row = frame.iloc[idx].to_dict() if idx < len(frame) else {}
        result = {
            "window_start": float(window.start_time),
            "window_end": float(window.end_time),
            "window_start_iso": _iso(float(window.start_time)),
            "window_end_iso": _iso(float(window.end_time)),
            "packets_in_window": len(window.packets),
            "arp_packets": int(arp_packets),
            "arp_requests": int(arp_requests),
            "arp_replies": int(arp_replies),
            "sender_conflict_ratio": float(row.get("sender_conflict_ratio", 0.0)),
            "conflict_ip_ratio": float(row.get("conflict_ip_ratio", 0.0)),
            "target_conflict_ratio": float(row.get("target_conflict_ratio", 0.0)),
            "prob": round(prob, 6),
            "logit": round(logit, 6),
            "label": label,
            "threshold": tau,
            "window_sec": float(window_sec),
            "reply_conflict_macs": int(reply_conflict_macs),
            "reply_conflict_ip_count": int(reply_conflict_ip_count),
        }
        if attn_weights is not None and idx < attn_weights.size:
            result["attention_weight"] = round(float(attn_weights[idx]), 6)
        results.append(result)

    return results, attn_peak


def infer_first_window(
    pcap_path: Path,
    cfg,
    feature_columns: Sequence[str],
    model: SequenceClassifier,
    scaler,
    device: torch.device,
    window_sec: float,
    hop_sec: float,
    tau: float,
) -> tuple[Dict[str, Any], int | None]:
    results, attn_peak = infer_windows(
        pcap_path,
        cfg,
        feature_columns,
        model,
        scaler,
        device,
        num_windows=1,
        window_sec=window_sec,
        hop_sec=hop_sec,
        tau=tau,
    )
    return results[0], attn_peak


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-window inference (ARP_LSTM).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("--pcap", type=Path, required=True, help="PCAP file to read (first N windows)")
    parser.add_argument("--window-sec", type=float, default=None, help="Window length in seconds (default: config.windowing.window_size)")
    parser.add_argument("--num-windows", type=int, default=1, help="Number of windows to infer (default: 1)")
    parser.add_argument("--tau", type=float, default=None, help="Optional decision threshold override (default: config.postprocessing.tau_window)")
    parser.add_argument("--out", type=Path, default=None, help="Optional directory to write a JSON result")
    args = parser.parse_args()

    if not args.pcap.is_file():
        raise FileNotFoundError(f"PCAP not found: {args.pcap}")

    cfg_path = args.config
    if not cfg_path.exists() and not cfg_path.is_absolute():
        alt = ROOT / cfg_path
        if alt.exists():
            cfg_path = alt
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config} (looked in {cfg_path})")

    cfg = load_config(cfg_path)
    manifest = load_json(cfg.paths.manifest_path)
    feature_columns = manifest.get("feature_columns") or []
    if not feature_columns:
        raise ValueError(f"Manifest missing feature_columns → {cfg.paths.manifest_path}")

    device = torch.device("cpu")
    torch.set_num_threads(min(2, os.cpu_count() or 1))

    scaler = load_joblib(cfg.paths.scaler_path)
    model = _load_model(cfg, feature_columns, device)

    window_sec = float(args.window_sec if args.window_sec is not None else cfg.windowing.window_size)
    hop_sec = float(cfg.windowing.hop_size)
    tau = float(args.tau if args.tau is not None else cfg.postprocessing.tau_window)
    num_windows = max(1, int(args.num_windows))

    start = time.perf_counter()
    if num_windows == 1:
        result, attn_peak = infer_first_window(
            args.pcap, cfg, feature_columns, model, scaler, device, window_sec, hop_sec, tau
        )
        result["pcap"] = str(args.pcap.resolve())
        result["inference_time_sec"] = round(time.perf_counter() - start, 6)
        result["attention_peak_step"] = attn_peak
        colored_label = _color_label(result["label"])
        print(
            f"[{args.pcap.name}] label={colored_label} prob={result['prob']} "
            f"time={result['inference_time_sec']}s packets={result['packets_in_window']} "
            f"arp_packets={result['arp_packets']} window={result['window_start_iso']} -> {result['window_end_iso']} "
            f"attn_step={result['attention_peak_step']}"
        )
        payload: Dict[str, Any] = result
        suffix = "single_window"
    else:
        windows, attn_peak = infer_windows(
            args.pcap, cfg, feature_columns, model, scaler, device, num_windows, window_sec, hop_sec, tau
        )
        total_time = round(time.perf_counter() - start, 6)
        for idx, win in enumerate(windows, start=1):
            colored_label = _color_label(win["label"])
            attn_weight = win.get("attention_weight")
            attn_info = f"{attn_weight}" if attn_weight is not None else "None"
            print(
                f"[{args.pcap.name}] window={idx}/{len(windows)} label={colored_label} "
                f"prob={win['prob']} packets={win['packets_in_window']} "
                f"arp_packets={win['arp_packets']} attn_weight={attn_info} "
                f"window={win['window_start_iso']} -> {win['window_end_iso']}"
            )
        payload = {
            "pcap": str(args.pcap.resolve()),
            "num_windows_requested": num_windows,
            "num_windows_inferred": len(windows),
            "window_sec": window_sec,
            "threshold": tau,
            "attention_peak_step": attn_peak,
            "total_inference_time_sec": total_time,
            "windows": windows,
        }
        suffix = f"first_{num_windows}_windows"

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        base = args.pcap.stem
        json_path = args.out / f"{base}_{suffix}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
