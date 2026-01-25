from __future__ import annotations

# Small-window inference helper for the Neural_LSTM DoS detector.
# Loads pretrained artifacts, extracts the first N windows from a PCAP, and reports wall-clock latency.

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

# Constrain BLAS threads early for small CPU footprints
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

# Ensure local package is importable when running the script directly
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from dos_detector.config import load_config
from dos_detector.data.pcap_reader import iter_pcap
from dos_detector.data.structures import Window
from dos_detector.features.feature_engineering import FeatureExtractor
from dos_detector.models.supervised import SequenceClassifier
from dos_detector.utils import configure_cpu_environment, resolve_project_root
from dos_detector.utils.io import load_joblib, load_json

DEFAULT_CONFIG = resolve_project_root() / "configs" / "config.yaml"


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


def _iter_windows(pcap_path: Path, window_sec: float, hop_sec: float) -> Iterator[Window]:
    if window_sec <= 0:
        raise ValueError("Window size must be positive.")
    if hop_sec <= 0:
        raise ValueError("Window hop size must be positive.")

    gen = iter_pcap(pcap_path)
    try:
        try:
            first = next(gen)
        except StopIteration:
            return

        index = 0
        window_start = float(first.timestamp)
        window_end = window_start + float(window_sec)
        packets = [first]

        for pkt in gen:
            ts = float(pkt.timestamp)
            if ts < window_end:
                packets.append(pkt)
                continue

            yield Window(index=index, start_time=window_start, end_time=window_end, packets=packets)
            index += 1
            window_start += float(hop_sec)
            window_end = window_start + float(window_sec)

            while ts >= window_end:
                index += 1
                window_start += float(hop_sec)
                window_end = window_start + float(window_sec)

            packets = [pkt]

        if packets:
            yield Window(index=index, start_time=window_start, end_time=window_end, packets=packets)
    finally:
        gen.close()


def infer_windows(
    pcap_path: Path,
    cfg,
    feature_columns: Sequence[str],
    model: SequenceClassifier,
    scaler,
    device: torch.device,
    window_sec: float,
    num_windows: int,
) -> list[Dict[str, Any]]:
    hop_sec = float(window_sec)
    extractor = FeatureExtractor(cfg.feature, window_sec)
    windows: list[Window] = []
    for window in _iter_windows(pcap_path, window_sec, hop_sec):
        if not window.packets:
            continue
        windows.append(window)
        if len(windows) >= max(1, int(num_windows)):
            break

    if not windows:
        raise ValueError(f"No packets found in {pcap_path}")

    frame = extractor.extract(windows)
    if frame.empty:
        raise ValueError("Feature frame is empty for inferred windows.")

    rows = frame.to_dict(orient="records")
    if len(rows) < len(windows):
        raise ValueError("Feature extraction returned fewer rows than windows.")

    feature_matrix = np.array(
        [[float(row.get(name, 0.0)) for name in feature_columns] for row in rows[: len(windows)]],
        dtype=np.float32,
    )
    scaled = scaler.transform(feature_matrix).astype(np.float32, copy=False)

    tensor = torch.from_numpy(scaled).to(device)
    tensor = tensor.unsqueeze(0)  # (batch=1, seq=len(windows), features)

    with torch.no_grad():
        outputs = model(tensor)
        logits = outputs.window_logits[0].detach().cpu().numpy().ravel()
        probs = torch.sigmoid(outputs.window_logits)[0].detach().cpu().numpy().ravel()
        attn_peak = None
        if outputs.attention is not None:
            weights = outputs.attention[0].detach().cpu().numpy().ravel()
            attn_peak = int(np.argmax(weights)) if weights.size else None

    threshold = float(cfg.postprocessing.tau_window)
    results: list[Dict[str, Any]] = []
    for idx, window in enumerate(windows):
        prob = float(probs[idx])
        logit = float(logits[idx])
        label = "attack" if prob >= threshold else "normal"
        results.append(
            {
                "window_start": float(window.start_time),
                "window_end": float(window.end_time),
                "window_start_iso": _iso(float(window.start_time)),
                "window_end_iso": _iso(float(window.end_time)),
                "packets_in_window": len(window.packets),
                "prob": round(prob, 6),
                "logit": round(logit, 6),
                "label": label,
                "threshold": threshold,
                "window_sec": float(window_sec),
                "micro_bins": int(cfg.feature.micro_bins),
                "attention_peak_step": attn_peak,
            }
        )

    return results


def infer_first_window(
    pcap_path: Path,
    cfg,
    feature_columns: Sequence[str],
    model: SequenceClassifier,
    scaler,
    device: torch.device,
    window_sec: float,
) -> Dict[str, Any]:
    results = infer_windows(
        pcap_path,
        cfg,
        feature_columns,
        model,
        scaler,
        device,
        window_sec,
        num_windows=1,
    )
    return results[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-window inference (Neural_LSTM).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("--pcap", type=Path, required=True, help="PCAP file to read (first N windows)")
    parser.add_argument(
        "--window-sec",
        type=float,
        default=None,
        help="Window length in seconds (default: config.windowing.window_size)",
    )
    parser.add_argument("--num-windows", type=int, default=1, help="Number of windows to infer (default: 1)")
    parser.add_argument("--out", type=Path, default=None, help="Optional directory to write a JSON result")
    args = parser.parse_args()

    if not args.pcap.is_file():
        raise FileNotFoundError(f"PCAP not found: {args.pcap}")

    cfg = load_config(args.config)
    manifest = load_json(cfg.paths.manifest_path)
    feature_columns = manifest.get("feature_columns") or []
    if not feature_columns:
        raise ValueError(f"Manifest missing feature_columns → {cfg.paths.manifest_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_cpu_environment(threads=2, interop_threads=1)
    scaler = load_joblib(cfg.paths.scaler_path)
    model = _load_model(cfg, feature_columns, device)

    start = time.perf_counter()
    window_sec = float(args.window_sec if args.window_sec is not None else cfg.windowing.window_size)
    num_windows = max(1, int(args.num_windows))
    if num_windows == 1:
        result = infer_first_window(args.pcap, cfg, feature_columns, model, scaler, device, window_sec)
        result["pcap"] = str(args.pcap.resolve())
        result["inference_time_sec"] = round(time.perf_counter() - start, 6)

        colored_label = _color_label(result["label"])
        print(
            f"[{args.pcap.name}] label={colored_label} prob={result['prob']} "
            f"time={result['inference_time_sec']}s packets={result['packets_in_window']} "
            f"window={result['window_start_iso']} -> {result['window_end_iso']} attn_step={result['attention_peak_step']}"
        )
        payload = result
        suffix = "single_window"
    else:
        windows = infer_windows(args.pcap, cfg, feature_columns, model, scaler, device, window_sec, num_windows)
        total_time = round(time.perf_counter() - start, 6)
        for idx, win in enumerate(windows, start=1):
            colored_label = _color_label(win["label"])
            print(
                f"[{args.pcap.name}] window={idx}/{len(windows)} label={colored_label} "
                f"prob={win['prob']} packets={win['packets_in_window']} "
                f"window={win['window_start_iso']} -> {win['window_end_iso']} attn_step={win['attention_peak_step']}"
            )
        payload = {
            "pcap": str(args.pcap.resolve()),
            "num_windows_requested": num_windows,
            "num_windows_inferred": len(windows),
            "window_sec": window_sec,
            "micro_bins": int(cfg.feature.micro_bins),
            "threshold": float(cfg.postprocessing.tau_window),
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
