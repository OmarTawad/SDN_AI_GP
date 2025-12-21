from __future__ import annotations

# Single-window inference helper for the ARP_LSTM detector.
# Loads pretrained artifacts, extracts the first 1s window from a PCAP, and reports wall-clock latency.

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
from arp_detector.features.feature_engineering import FeatureExtractor
from arp_detector.models.quantization import convert_module_to_int8, Int8Quantizer
from arp_detector.models.supervised import SequenceClassifier
from arp_detector.utils.io import load_joblib, load_json

DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _load_model(cfg, feature_columns: Sequence[str], device: torch.device) -> SequenceClassifier:
    model = SequenceClassifier(
        input_size=len(feature_columns),
        num_attack_types=len(cfg.labels.family_mapping),
        config=cfg.model.supervised,
    )
    convert_module_to_int8(model, Int8Quantizer())
    state = torch.load(cfg.paths.supervised_model_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def _first_window(pcap_path: Path, window_sec: float) -> Window:
    gen = iter_pcap(pcap_path)
    try:
        first = next(gen)
    except StopIteration:
        gen.close()
        raise ValueError(f"No packets found in {pcap_path}")
    start_ts = float(first.timestamp)
    end_ts = start_ts + float(window_sec)
    packets = [first]
    try:
        for pkt in gen:
            if float(pkt.timestamp) < end_ts:
                packets.append(pkt)
            else:
                break
    finally:
        gen.close()
    return Window(index=0, start_time=start_ts, end_time=end_ts, packets=packets)


def infer_single_window(
    pcap_path: Path,
    cfg,
    feature_columns: Sequence[str],
    model: SequenceClassifier,
    scaler,
    device: torch.device,
) -> Dict[str, Any]:
    extractor = FeatureExtractor(cfg.feature, cfg.windowing.window_size)
    window = _first_window(pcap_path, cfg.windowing.window_size)
    frame = extractor.extract([window])
    if frame.empty:
        raise ValueError("Feature frame is empty for the first window.")

    row = frame.iloc[0].to_dict()
    feature_vector = np.array([float(row.get(name, 0.0)) for name in feature_columns], dtype=np.float32)
    scaled = scaler.transform(feature_vector.reshape(1, -1)).astype(np.float32, copy=False)

    tensor = torch.from_numpy(scaled).to(device)
    tensor = tensor.unsqueeze(0)  # (batch=1, seq=1, features)

    with torch.no_grad():
        outputs = model(tensor)
        logit = float(outputs.window_logits[0, 0].item())
        prob = float(torch.sigmoid(outputs.window_logits)[0, 0].item())
        attn_peak = None
        if outputs.attention is not None:
            weights = outputs.attention[0].detach().cpu().numpy().ravel()
            attn_peak = int(np.argmax(weights)) if weights.size else None

    threshold = float(cfg.postprocessing.tau_window)
    label = "attack" if prob >= threshold else "normal"
    arp_packets = sum(1 for pkt in window.packets if (pkt.protocol or "").lower() == "arp")

    return {
        "pcap": str(pcap_path.resolve()),
        "window_start": float(window.start_time),
        "window_end": float(window.end_time),
        "window_start_iso": _iso(float(window.start_time)),
        "window_end_iso": _iso(float(window.end_time)),
        "packets_in_window": len(window.packets),
        "arp_packets": int(arp_packets),
        "prob": round(prob, 6),
        "logit": round(logit, 6),
        "label": label,
        "threshold": threshold,
        "window_sec": float(cfg.windowing.window_size),
        "attention_peak_step": attn_peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-window inference (ARP_LSTM).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("--pcap", type=Path, required=True, help="PCAP file to read (first 1s window only)")
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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for int8-quantized inference.")
    device = torch.device("cuda")
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except (AttributeError, RuntimeError):
        pass

    scaler = load_joblib(cfg.paths.scaler_path)
    model = _load_model(cfg, feature_columns, device)

    start = time.perf_counter()
    result = infer_single_window(args.pcap, cfg, feature_columns, model, scaler, device)
    result["inference_time_sec"] = round(time.perf_counter() - start, 6)

    print(
        f"[{args.pcap.name}] label={result['label']} prob={result['prob']} "
        f"time={result['inference_time_sec']}s packets={result['packets_in_window']} "
        f"arp_packets={result['arp_packets']} window={result['window_start_iso']} -> {result['window_end_iso']} "
        f"attn_step={result['attention_peak_step']}"
    )

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        base = args.pcap.stem
        json_path = args.out / f"{base}_single_window.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
