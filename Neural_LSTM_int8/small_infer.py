from __future__ import annotations

# Single-window inference helper for the Neural_LSTM DoS detector.
# Loads pretrained artifacts, yields windows from a PCAP, and reports wall-clock latency.

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence, Generator

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
from dos_detector.utils import (
    DEFAULT_NUMPY_DTYPE,
    DEFAULT_TORCH_DTYPE,
    configure_cpu_environment,
    resolve_device,
    resolve_precision_mode,
    resolve_project_root,
    resolve_torch_dtype,
    safe_cast_tensor,
)
from dos_detector.utils.quantization import apply_dynamic_quantization, set_quantized_engine, unpack_checkpoint
from dos_detector.utils.io import load_joblib, load_json

DEFAULT_CONFIG = resolve_project_root() / "configs" / "config.yaml"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _load_model(
    cfg,
    feature_columns: Sequence[str],
    device: torch.device,
    torch_dtype: torch.dtype,
    quantized: bool,
    state_dict: Dict[str, Any],
    is_quantized_checkpoint: bool,
) -> SequenceClassifier:
    model = SequenceClassifier(
        input_size=len(feature_columns),
        num_attack_types=len(cfg.labels.family_mapping),
        config=cfg.model.supervised,
    )
    if not quantized:
        model = model.to(device=device, dtype=torch_dtype or DEFAULT_TORCH_DTYPE)
        # Using pre-loaded state_dict
        model.load_state_dict(state_dict)
        model.eval()
        return model

    set_quantized_engine(getattr(cfg.quantization, "backend", None))
    # Quantization typically runs on CPU
    model = model.to(device="cpu", dtype=torch.float32)
    
    if is_quantized_checkpoint:
        quantized_model = apply_dynamic_quantization(model)
        quantized_model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
        quantized_model = apply_dynamic_quantization(model)
        
    quantized_model.eval()
    return quantized_model


def _iter_windows(pcap_path: Path, window_sec: float, max_windows: int) -> Generator[Window, None, None]:
    gen = iter_pcap(pcap_path)
    try:
        try:
            first = next(gen)
        except StopIteration:
            return

        start_ts = float(first.timestamp)
        current_window_start = start_ts
        current_window_end = start_ts + window_sec
        current_packets = [first]
        
        count = 0
        
        for pkt in gen:
            ts = float(pkt.timestamp)
            
            # While packet is beyond current window, yield completed windows and advance
            while ts >= current_window_end:
                count += 1
                yield Window(index=count, start_time=current_window_start, end_time=current_window_end, packets=current_packets)
                
                if count >= max_windows:
                    return

                # Prepare next window
                current_window_start = current_window_end
                current_window_end += window_sec
                current_packets = []
            
            # Packet belongs to current window
            current_packets.append(pkt)
            
        # Yield the last incomplete window if we haven't hit limit
        if current_packets and count < max_windows:
             count += 1
             yield Window(index=count, start_time=current_window_start, end_time=current_window_end, packets=current_packets)
             
    finally:
        gen.close()


def infer_window(
    window: Window,
    pcap_path: Path,
    cfg,
    feature_columns: Sequence[str],
    model: SequenceClassifier,
    scaler,
    device: torch.device,
    torch_dtype: torch.dtype,
    numpy_dtype,
    extractor: FeatureExtractor,
) -> Dict[str, Any]:
    
    frame = extractor.extract([window])
    if frame.empty:
         raise ValueError("Feature frame is empty for window.")

    row = frame.iloc[0].to_dict()
    feature_vector = np.array([float(row.get(name, 0.0)) for name in feature_columns], dtype=numpy_dtype or DEFAULT_NUMPY_DTYPE)
    scaled = scaler.transform(feature_vector.reshape(1, -1)).astype(numpy_dtype or DEFAULT_NUMPY_DTYPE, copy=False)

    tensor = safe_cast_tensor(scaled, torch_dtype or DEFAULT_TORCH_DTYPE).to(device)
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

    return {
        "pcap": str(pcap_path.resolve()),
        "window_start": float(window.start_time),
        "window_end": float(window.end_time),
        "window_start_iso": _iso(float(window.start_time)),
        "window_end_iso": _iso(float(window.end_time)),
        "packets_in_window": len(window.packets),
        "prob": round(prob, 6),
        "logit": round(logit, 6),
        "label": label,
        "threshold": threshold,
        "window_sec": float(cfg.windowing.window_size),
        "micro_bins": int(cfg.feature.micro_bins),
        "attention_peak_step": attn_peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-window inference (Neural_LSTM).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("--pcap", type=Path, required=True, help="PCAP file to read")
    parser.add_argument("--out", type=Path, default=None, help="Optional directory to write a JSON result")
    parser.add_argument("--num-windows", type=int, default=1, help="Number of windows to process")
    args = parser.parse_args()

    if not args.pcap.is_file():
        raise FileNotFoundError(f"PCAP not found: {args.pcap}")

    cfg = load_config(args.config)
    manifest = load_json(cfg.paths.manifest_path)
    feature_columns = manifest.get("feature_columns") or []
    if not feature_columns:
        raise ValueError(f"Manifest missing feature_columns → {cfg.paths.manifest_path}")

    quantized = bool(getattr(cfg, "quantization", None) and cfg.quantization.enabled)
    device = resolve_device(prefer_cuda=not quantized)
    if device.type == "cpu":
        configure_cpu_environment(threads=2, interop_threads=1)
    
    scaler = load_joblib(cfg.paths.scaler_path)
    torch_dtype, numpy_dtype = resolve_precision_mode(getattr(cfg.training.supervised, "precision_mode", None))
    torch_dtype = resolve_torch_dtype(device, torch_dtype)
    
    # Load state dict and handle quantization + truncation
    if quantized:
        quant_path = getattr(cfg.quantization, "checkpoint_path", None)
        checkpoint_path = quant_path if quant_path and Path(quant_path).is_file() else cfg.paths.supervised_model_path
        state = torch.load(checkpoint_path, map_location="cpu") # Quantization uses CPU
    else:
        state = torch.load(cfg.paths.supervised_model_path, map_location=device)
        
    state_dict, is_quantized = unpack_checkpoint(state)

    # Heuristic: check first layer weight size to determine expected input size
    # LSTM/GRU weight_ih_l0 shape is (hidden_size * X, input_size)
    if "rnn.weight_ih_l0" in state_dict:
        expected_cols = state_dict["rnn.weight_ih_l0"].shape[1]
        if len(feature_columns) > expected_cols:
            print(f"Warning: extracted {len(feature_columns)} features; truncating to {expected_cols} to match the checkpoint.")
            feature_columns = feature_columns[:expected_cols]
            
            # Truncate scaler to match feature columns
            if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ > expected_cols:
                print(f"Warning: Scaler expects {scaler.n_features_in_} features but got {expected_cols}. Falling back to fitted scaling.")
                scaler.n_features_in_ = expected_cols
                if hasattr(scaler, "mean_") and scaler.mean_ is not None:
                    scaler.mean_ = scaler.mean_[:expected_cols]
                if hasattr(scaler, "scale_") and scaler.scale_ is not None:
                    scaler.scale_ = scaler.scale_[:expected_cols]
                if hasattr(scaler, "var_") and scaler.var_ is not None:
                    scaler.var_ = scaler.var_[:expected_cols]

    model = _load_model(cfg, feature_columns, device, torch_dtype, quantized, state_dict, is_quantized)
    
    extractor = FeatureExtractor(cfg.feature, cfg.windowing.window_size)
    window_gen = _iter_windows(args.pcap, cfg.windowing.window_size, args.num_windows)
    
    results = []

    for i, window in enumerate(window_gen, 1):
        start = time.perf_counter()
        try:
            result = infer_window(window, args.pcap, cfg, feature_columns, model, scaler, device, torch_dtype, numpy_dtype, extractor)
            result["inference_time_sec"] = round(time.perf_counter() - start, 6)
            
            print(
                f"[{args.pcap.name}] window={i}/{args.num_windows} label={result['label']} prob={result['prob']} "
                f"time={result['inference_time_sec']}s packets={result['packets_in_window']} "
                f"window={result['window_start_iso']} -> {result['window_end_iso']} attn_step={result['attention_peak_step']}"
            )
            results.append(result)
        except Exception as e:
            print(f"[{args.pcap.name}] window={i}/{args.num_windows} Error: {e}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        base = args.pcap.stem
        json_path = args.out / f"{base}_inference.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
