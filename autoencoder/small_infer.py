from __future__ import annotations

# Small-window inference helper for the autoencoder.
# Loads pretrained artifacts, scores the first N windows from a PCAP, and reports per-window latency.

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

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

import pandas as pd
import torch
from scapy.utils import RawPcapReader

from dae.config import Config, load_config
from dae.extract import _packet_summary, _packet_timestamp, _parse_packet
from dae.features import FeatureExtractor
from dae.infer import (
    InferenceArtifacts,
    META_COLUMNS,
    _prepare_batch,
    _score_batch,
    load_inference_artifacts,
)
from dae.window import SlidingWindowManager, WindowStats

DEFAULT_CONFIG = ROOT / "config.yaml"


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _color_decision(decision: str) -> str:
    if os.environ.get("NO_COLOR"):
        return decision
    if decision.lower().startswith("attack"):
        return f"\033[31m{decision}\033[0m"
    if decision.lower() == "normal":
        return f"\033[32m{decision}\033[0m"
    return decision


def _collect_windows(
    pcap_path: Path,
    window_seconds: float,
    stride_seconds: float,
    num_windows: int,
) -> List[WindowStats]:
    manager = SlidingWindowManager(window_seconds=window_seconds, stride_seconds=stride_seconds)
    windows: List[WindowStats] = []
    reader = RawPcapReader(str(pcap_path))
    try:
        for raw_packet, metadata in reader:
            ts = _packet_timestamp(metadata)
            summary = _packet_summary(_parse_packet(raw_packet), ts)
            for window in manager.add_packet(summary):
                if window.packet_count == 0:
                    continue
                windows.append(window)
                if len(windows) >= num_windows:
                    break
            if len(windows) >= num_windows:
                break
    finally:
        reader.close()

    if len(windows) < num_windows:
        for window in manager.finalize():
            if window.packet_count == 0:
                continue
            windows.append(window)
            if len(windows) >= num_windows:
                break

    if not windows:
        raise ValueError(f"No windows could be constructed from {pcap_path}")
    return windows[:num_windows]


def _rows_from_windows(
    windows: Iterable[WindowStats],
    feature_extractor: FeatureExtractor,
    source_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for window in windows:
        row = feature_extractor.build_row(window)
        row["source"] = source_name
        rows.append(row)
    return rows


def _infer_rows(
    rows: List[Dict[str, Any]],
    artifacts: InferenceArtifacts,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    threshold = float(artifacts.threshold.threshold)
    for row in rows:
        batch_df = pd.DataFrame([row])
        inference_df = batch_df.drop(columns=[col for col in META_COLUMNS if col in batch_df.columns])

        start = time.perf_counter()
        scaled = _prepare_batch(inference_df, artifacts)
        errors = _score_batch(artifacts.model, scaled, artifacts.device, batch_size=1)
        elapsed = round(time.perf_counter() - start, 6)

        if errors.size == 0:
            continue
        error_val = float(errors[0])
        is_anomaly = error_val > threshold
        decision = "attack detected" if is_anomaly else "normal"

        start_ts = float(batch_df["start_ts"].iloc[0]) if "start_ts" in batch_df else None
        end_ts = float(batch_df["end_ts"].iloc[0]) if "end_ts" in batch_df else None
        window_idx = int(batch_df["window_idx"].iloc[0]) if "window_idx" in batch_df else 0

        results.append({
            "window_idx": window_idx,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "start_iso": _iso(start_ts),
            "end_iso": _iso(end_ts),
            "error": round(error_val, 6),
            "threshold": threshold,
            "decision": decision,
            "anomalous": bool(is_anomaly),
            "method": artifacts.threshold.method,
            "inference_time_sec": elapsed,
        })
    return results


def infer_windows(
    pcap_path: Path,
    config: Config,
    artifacts: InferenceArtifacts,
    num_windows: int,
    window_seconds: float,
    stride_seconds: float,
) -> List[Dict[str, Any]]:
    include = config.get("features", "include", default=[])
    ratios = bool(config.get("features", "ratios", default=True))
    feature_extractor = FeatureExtractor(include=include, ratios=ratios)

    windows = _collect_windows(pcap_path, window_seconds, stride_seconds, num_windows)
    rows = _rows_from_windows(windows, feature_extractor, pcap_path.name)
    return _infer_rows(rows, artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-window inference (autoencoder).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml")
    parser.add_argument("--pcap", type=Path, required=True, help="PCAP file to read (first N windows)")
    parser.add_argument("--window-sec", type=float, default=None, help="Window length in seconds (default: config.extract.window_seconds)")
    parser.add_argument("--stride-sec", type=float, default=None, help="Stride in seconds (default: config.extract.stride_seconds)")
    parser.add_argument("--num-windows", type=int, default=1, help="Number of windows to infer (default: 1)")
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

    config = load_config(cfg_path)
    artifacts = load_inference_artifacts(config)

    window_seconds = float(args.window_sec if args.window_sec is not None else config.get("extract", "window_seconds", default=1.0))
    stride_seconds = float(args.stride_sec if args.stride_sec is not None else config.get("extract", "stride_seconds", default=0.5))
    num_windows = max(1, int(args.num_windows))

    results = infer_windows(args.pcap, config, artifacts, num_windows, window_seconds, stride_seconds)
    if not results:
        raise ValueError("No window features computed for inference.")

    if num_windows == 1:
        result = results[0]
        result["file"] = args.pcap.name
        colored_decision = _color_decision(result["decision"])
        print(
            f"[{args.pcap.name}] decision={colored_decision} error={result['error']} "
            f"thr={result['threshold']} time={result['inference_time_sec']}s "
            f"window={result['start_iso']} -> {result['end_iso']}"
        )
        payload: Dict[str, Any] = result
        suffix = "single_window"
    else:
        for idx, win in enumerate(results, start=1):
            colored_decision = _color_decision(win["decision"])
            print(
                f"[{args.pcap.name}] window={idx}/{len(results)} decision={colored_decision} "
                f"error={win['error']} time={win['inference_time_sec']}s "
                f"window={win['start_iso']} -> {win['end_iso']}"
            )
        payload = {
            "pcap": str(args.pcap.resolve()),
            "num_windows_requested": num_windows,
            "num_windows_inferred": len(results),
            "window_sec": window_seconds,
            "stride_sec": stride_seconds,
            "windows": results,
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
