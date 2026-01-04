from __future__ import annotations

# Single-window inference helper for the autoencoder.
# Loads pretrained artifacts, scores the first 1s window from a PCAP, and reports wall-clock latency.

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

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

import pandas as pd
import torch
from scapy.utils import RawPcapReader

from dae.config import Config, load_config
from dae.extract import _packet_summary, _packet_timestamp, _parse_packet
from dae.features import FeatureExtractor
from dae.infer import InferenceArtifacts, META_COLUMNS, _prepare_batch, _score_batch, load_inference_artifacts
from dae.window import SlidingWindowManager

DEFAULT_CONFIG = ROOT / "config.yaml"


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _first_window_row(pcap_path: Path, config: Config, feature_extractor: FeatureExtractor) -> Dict[str, Any]:
    window_seconds = float(config.get("extract", "window_seconds", default=1.0))
    stride_seconds = float(config.get("extract", "stride_seconds", default=0.5))
    manager = SlidingWindowManager(window_seconds=window_seconds, stride_seconds=stride_seconds)

    reader = RawPcapReader(str(pcap_path))
    first_window = None
    try:
        for raw_packet, metadata in reader:
            ts = _packet_timestamp(metadata)
            summary = _packet_summary(_parse_packet(raw_packet), ts)
            completed = list(manager.add_packet(summary))
            if completed:
                first_window = completed[0]
                break
    finally:
        reader.close()

    if first_window is None:
        remaining = list(manager.finalize())
        if remaining:
            first_window = remaining[0]
    if first_window is None:
        raise ValueError(f"No windows could be constructed from {pcap_path}")

    row = feature_extractor.build_row(first_window)
    row["source"] = pcap_path.name
    return row


def infer_first_window(
    pcap_path: Path,
    config: Config,
    artifacts: InferenceArtifacts,
) -> Dict[str, Any]:
    include = config.get("features", "include", default=[])
    ratios = bool(config.get("features", "ratios", default=True))
    feature_extractor = FeatureExtractor(include=include, ratios=ratios)

    row = _first_window_row(pcap_path, config, feature_extractor)
    batch_df = pd.DataFrame([row])
    inference_df = batch_df.drop(columns=[col for col in META_COLUMNS if col in batch_df.columns])

    scaled = _prepare_batch(inference_df, artifacts)
    errors = _score_batch(
        artifacts.model,
        scaled,
        artifacts.device,
        batch_size=1,
        torch_dtype=artifacts.dtype,
        fp16_clamp=artifacts.fp16_clamp,
        fp16_max=artifacts.fp16_max,
    )
    if errors.size == 0:
        raise ValueError("No window features computed for inference.")

    error_val = float(errors[0])
    threshold = float(artifacts.threshold.threshold)
    is_anomaly = error_val > threshold

    start_ts = float(batch_df["start_ts"].iloc[0]) if "start_ts" in batch_df else None
    end_ts = float(batch_df["end_ts"].iloc[0]) if "end_ts" in batch_df else None
    window_idx = int(batch_df["window_idx"].iloc[0]) if "window_idx" in batch_df else 0

    return {
        "file": pcap_path.name,
        "window_idx": window_idx,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "start_iso": _iso(start_ts),
        "end_iso": _iso(end_ts),
        "error": round(error_val, 6),
        "threshold": threshold,
        "decision": "attack detected" if is_anomaly else "normal",
        "anomalous": bool(is_anomaly),
        "method": artifacts.threshold.method,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-window inference (autoencoder).")
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

    torch.set_num_threads(min(2, os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except (AttributeError, RuntimeError):
        pass

    config = load_config(cfg_path)
    artifacts = load_inference_artifacts(config)

    start = time.perf_counter()
    result = infer_first_window(args.pcap, config, artifacts)
    result["inference_time_sec"] = round(time.perf_counter() - start, 6)

    print(
        f"[{args.pcap.name}] decision={result['decision']} error={result['error']} "
        f"thr={result['threshold']} time={result['inference_time_sec']}s window={result['start_iso']} -> {result['end_iso']}"
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
