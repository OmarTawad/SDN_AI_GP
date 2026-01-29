from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .config import Config
from .extract import process_pcap
from .features import FeatureExtractor
from .logging import get_logger
from .model import build_model
from .preprocess import apply_log_transform, replace_invalid, load_feature_list, load_scaler
from .quantization import (
    apply_dynamic_quantization,
    build_static_fx_model,
    ensure_quantized_modules,
    normalize_checkpoint_path,
    set_quantized_engine,
    unpack_checkpoint,
)
from .threshold import ThresholdResult, load_threshold


@dataclass
class InferenceArtifacts:
    model: torch.nn.Module
    scaler: object
    feature_names: List[str]
    log_features: List[str]
    clip_bounds: Dict[str, Dict[str, float]]
    threshold: ThresholdResult
    device: torch.device
    dtype: torch.dtype
    error_stats: Dict[str, float]
    source_stats: Dict[str, Dict[str, float]]


META_COLUMNS = ["window_idx", "start_ts", "end_ts", "source"]
TOPK_FEATURES = [
    "pkt_count",
    "pps",
    "byte_count",
    "src_ip_entropy",
    "dst_ip_entropy",
    "tcp_ratio",
    "udp_ratio",
]


def _resolve_int8_infer_config(config: Config) -> tuple[torch.device, torch.dtype, dict]:
    train_cfg = config.get("train") or {}
    device_value = str(train_cfg.get("device", "cpu")).lower()
    if device_value != "cpu":
        raise RuntimeError("autoencoder_int8 enforces CPU-only inference; set train.device: cpu")
    quant_cfg = config.get("quantization", default={}) or {}
    if not bool(quant_cfg.get("enabled", True)):
        raise RuntimeError("autoencoder_int8 requires quantization.enabled: true")
    set_quantized_engine(quant_cfg.get("backend"))
    return torch.device("cpu"), torch.float32, quant_cfg


def load_inference_artifacts(config: Config) -> InferenceArtifacts:
    artifacts_dir = Path(config.get("paths", "artifacts_dir"))
    model_config_path = artifacts_dir / "model_config.json"
    model_path = artifacts_dir / "model.pt"
    scaler_path = artifacts_dir / "scaler.pkl"
    feature_path = artifacts_dir / "feature_list.json"
    clip_path = artifacts_dir / "clip_bounds.json"
    threshold_path = artifacts_dir / "threshold.json"
    error_stats_path = artifacts_dir / "error_stats.json"
    source_stats_path = artifacts_dir / "source_stats.json"

    if not model_config_path.exists():
        raise FileNotFoundError("Model config not found. Train the model first.")

    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    feature_names = load_feature_list(feature_path)
    log_features = list(model_config.get("log_features", []))
    scaler = load_scaler(scaler_path)
    clip_bounds = json.loads(clip_path.read_text(encoding="utf-8")) if clip_path.exists() else {"lower": {}, "upper": {}}
    threshold = load_threshold(threshold_path)

    error_stats: Dict[str, float] = {}
    if error_stats_path.exists():
        try:
            payload = json.loads(error_stats_path.read_text(encoding="utf-8"))
            error_stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            error_stats = {}

    source_stats: Dict[str, Dict[str, float]] = {}
    if source_stats_path.exists():
        try:
            source_stats = json.loads(source_stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            source_stats = {}

    model = build_model(model_config.get("model", {}), input_dim=int(model_config["input_dim"]))
    device, torch_dtype, quant_cfg = _resolve_int8_infer_config(config)
    model.to(device=device, dtype=torch_dtype)
    mode = str(quant_cfg.get("mode", "dynamic")).lower()
    quant_path = normalize_checkpoint_path(quant_cfg.get("checkpoint_path"))
    if mode == "static":
        if quant_path is None or not quant_path.is_file():
            raise FileNotFoundError("Static int8 requires quantization.checkpoint_path to exist.")
        state = torch.load(quant_path, map_location=device)
        state_dict, is_quantized = unpack_checkpoint(state)
        if not is_quantized:
            raise RuntimeError("Static quantization requested but checkpoint is not quantized.")
        example_inputs = (torch.zeros((1, int(model_config["input_dim"])), dtype=torch.float32),)
        quantized_model = build_static_fx_model(model, example_inputs, quant_cfg.get("backend"))
        quantized_model.load_state_dict(state_dict)
    else:
        checkpoint = quant_path if (quant_path and quant_path.is_file()) else model_path
        if quant_path and not quant_path.is_file():
            get_logger("infer").warning("quantized_checkpoint_missing", path=str(quant_path))
        state = torch.load(checkpoint, map_location=device)
        state_dict, is_quantized = unpack_checkpoint(state)
        if is_quantized:
            quantized_model = apply_dynamic_quantization(model)
            quantized_model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
            quantized_model = apply_dynamic_quantization(model)
    ensure_quantized_modules(quantized_model)
    quantized_model.eval()

    return InferenceArtifacts(
        model=quantized_model,
        scaler=scaler,
        feature_names=feature_names,
        log_features=log_features,
        clip_bounds=clip_bounds,
        threshold=threshold,
        device=device,
        dtype=torch_dtype,
        error_stats=error_stats,
        source_stats=source_stats,
    )


def _ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = 0.0
    return df


def _apply_clip(df: pd.DataFrame, clip_bounds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    lower_bounds = clip_bounds.get("lower", {})
    upper_bounds = clip_bounds.get("upper", {})
    for col, value in lower_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=float(value))
    for col, value in upper_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=float(value))
    return df


def _prepare_batch(df: pd.DataFrame, artifacts: InferenceArtifacts) -> np.ndarray:
    df = replace_invalid(df)
    df = _ensure_columns(df, artifacts.feature_names)
    df = apply_log_transform(df, artifacts.log_features)
    df = _apply_clip(df, artifacts.clip_bounds)
    features = df[artifacts.feature_names]
    scaled = artifacts.scaler.transform(features.values)
    return scaled.astype(np.float32)


def _score_batch(
    model: torch.nn.Module,
    batch: np.ndarray,
    device: torch.device,
    batch_size: int,
    torch_dtype: torch.dtype,
) -> np.ndarray:
    if batch.size == 0:
        return np.array([], dtype=np.float32)
    model.eval()
    errors: List[float] = []
    with torch.no_grad():
        for start in range(0, len(batch), batch_size):
            end = start + batch_size
            tensor = torch.from_numpy(batch[start:end]).to(device=device, dtype=torch_dtype)
            outputs = model(tensor)
            mse = ((outputs - tensor) ** 2).mean(dim=1)
            errors.extend(mse.detach().cpu().numpy().tolist())
    return np.asarray(errors, dtype=np.float32)


def _postprocess_flags(flags: np.ndarray, min_consecutive: int, cooldown: int) -> np.ndarray:
    n = len(flags)
    result = np.zeros(n, dtype=bool)
    idx = 0
    cooldown_remaining = 0
    while idx < n:
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            idx += 1
            continue
        if flags[idx]:
            start = idx
            while idx < n and flags[idx]:
                idx += 1
            run_length = idx - start
            if run_length >= min_consecutive:
                result[start:idx] = True
                cooldown_remaining = cooldown
        else:
            idx += 1
    return result


def infer_pcap(config: Config, pcap_path: Path) -> Dict[str, object]:
    logger = get_logger("infer")
    artifacts = load_inference_artifacts(config)
    include = config.get("features", "include", default=[])
    ratios = bool(config.get("features", "ratios", default=True))
    feature_extractor = FeatureExtractor(include=include, ratios=ratios)

    threshold_params = artifacts.threshold.params
    min_consecutive = int(config.get("threshold", "min_consecutive", default=threshold_params.get("min_consecutive", 1)))
    cooldown = int(config.get("threshold", "cooldown", default=threshold_params.get("cooldown", 0)))
    min_attack_windows = int(config.get("threshold", "min_attack_windows", default=threshold_params.get("min_attack_windows", 1)))

    batch_errors: List[float] = []
    meta_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    batch_size = int(config.get("infer", "batch_windows", default=8192))

    def on_rows(rows: List[dict]) -> None:
        batch_df = pd.DataFrame(rows)
        if batch_df.empty:
            return
        metas = batch_df[META_COLUMNS].copy()
        inference_df = batch_df.drop(columns=[col for col in META_COLUMNS if col in batch_df.columns])
        scaled = _prepare_batch(inference_df, artifacts)
        errors = _score_batch(
            artifacts.model,
            scaled,
            artifacts.device,
            batch_size,
            artifacts.dtype,
        )
        batch_errors.extend(errors.tolist())

        metas["error"] = errors
        meta_rows.extend(metas.to_dict(orient="records"))

        selected = batch_df[[col for col in TOPK_FEATURES if col in batch_df.columns]].copy()
        selected["window_idx"] = batch_df["window_idx"]
        selected["start_ts"] = batch_df.get("start_ts")
        selected["end_ts"] = batch_df.get("end_ts")
        selected["error"] = errors
        feature_rows.extend(selected.to_dict(orient="records"))

    process_pcap(pcap_path, config, feature_extractor, on_rows)

    errors_array = np.asarray(batch_errors, dtype=np.float32)
    if errors_array.size == 0:
        raise ValueError("No windows extracted from PCAP")

    trained_threshold = float(artifacts.threshold.threshold)
    effective_threshold = trained_threshold

    source_name: str | None = None
    if meta_rows:
        candidates = {row.get("source") for row in meta_rows if row.get("source")}
        if len(candidates) == 1:
            source_name = next(iter(candidates))

    stats_threshold = None
    stats_quantile = config.get("threshold", "infer_stats_quantile")
    stats_scale = float(config.get("threshold", "infer_stats_scale", default=1.0))
    if stats_quantile is not None and artifacts.error_stats:
        try:
            q = float(stats_quantile)
            quantile_value = None
            key_map = {
                0.99: "q99",
                0.995: "q995",
                0.999: "q999",
            }
            key = key_map.get(round(q, 3))
            if key and key in artifacts.error_stats:
                quantile_value = float(artifacts.error_stats[key])
            else:
                keys = sorted(
                    (float(k[1:]) / (1000 if len(k) == 4 else 100), float(v))
                    for k, v in artifacts.error_stats.items()
                    if k.startswith("q")
                )
                if keys:
                    quantiles, values = zip(*keys)
                    quantile_value = float(np.interp(q, quantiles, values))
            if quantile_value is not None:
                stats_threshold = quantile_value * stats_scale
        except (ValueError, TypeError):
            stats_threshold = None

    lower_bound = trained_threshold
    if stats_threshold is not None:
        lower_bound = max(lower_bound, stats_threshold)

    source_threshold = None
    if source_name and artifacts.source_stats:
        src_stats = artifacts.source_stats.get(source_name)
        if src_stats:
            source_quantile = config.get("threshold", "source_quantile")
            source_scale = float(config.get("threshold", "source_scale", default=1.0))
            quantile_value = None
            if source_quantile is not None:
                try:
                    sq = float(source_quantile)
                    key_map = {0.99: "q99", 0.995: "q995", 0.999: "q999"}
                    key = key_map.get(round(sq, 3))
                    if key and key in src_stats:
                        quantile_value = float(src_stats[key])
                    else:
                        available = [
                            (0.99, src_stats.get("q99")),
                            (0.995, src_stats.get("q995")),
                            (0.999, src_stats.get("q999")),
                        ]
                        available = [(q, float(v)) for q, v in available if v is not None]
                        if available:
                            quantiles, values = zip(*sorted(available))
                            quantile_value = float(np.interp(sq, quantiles, values))
                except (ValueError, TypeError):
                    quantile_value = None
            if quantile_value is not None:
                source_threshold = quantile_value * source_scale
                lower_bound = max(lower_bound, source_threshold)

    auto_threshold = None
    if source_threshold is None:
        auto_quantile = config.get("threshold", "auto_calibrate_quantile")
        auto_scale = float(config.get("threshold", "auto_calibrate_scale", default=1.0))
        auto_cap = float(config.get("threshold", "auto_calibrate_cap", default=0.0))
        if auto_quantile is not None and auto_scale > 0:
            try:
                aq = float(auto_quantile)
                base_value = float(np.quantile(errors_array, aq))
                candidate = base_value * auto_scale
                cap_value = None
                if auto_cap > 0 and stats_threshold is not None:
                    cap_value = stats_threshold * auto_cap
                if cap_value is not None:
                    candidate = min(candidate, cap_value)
                if candidate > 0:
                    auto_threshold = candidate
                    lower_bound = max(lower_bound, auto_threshold)
            except (ValueError, TypeError):
                auto_threshold = None

    clamp_quantile = config.get("threshold", "infer_clamp_quantile")
    if clamp_quantile is not None:
        fallback_threshold = float(np.quantile(errors_array, float(clamp_quantile)))
        clamp_floor_ratio = float(config.get("threshold", "infer_clamp_floor", default=0.5))
        min_allowed = lower_bound * clamp_floor_ratio
        effective_threshold = max(min_allowed, min(lower_bound, fallback_threshold))
    else:
        effective_threshold = lower_bound

    flags = errors_array > effective_threshold
    filtered = _postprocess_flags(flags, min_consecutive, cooldown)
    anomalous_count = int(filtered.sum())

    decision = "attack detected" if anomalous_count >= min_attack_windows else "normal"

    topk_pct = float(config.get("infer", "topk_pct", default=0.1))
    topk_n = min(len(feature_rows), max(1, int(len(feature_rows) * topk_pct)))
    sorted_features = sorted(feature_rows, key=lambda row: row["error"], reverse=True)[:topk_n]

    report = {
        "file": pcap_path.name,
        "decision": decision,
        "anomalous_windows": anomalous_count,
        "total_windows": len(errors_array),
        "threshold": effective_threshold,
        "method": artifacts.threshold.method,
        "params": {
            "min_consecutive": min_consecutive,
            "cooldown": cooldown,
            "min_attack_windows": min_attack_windows,
            "trained_threshold": trained_threshold,
            "stats_threshold": stats_threshold,
            "source": source_name,
            "source_threshold": source_threshold,
            "auto_threshold": auto_threshold,
        },
        "topk_windows": [
            {
                "idx": int(row.get("window_idx", -1)),
                "error": float(row["error"]),
                "start": row.get("start_ts"),
                "end": row.get("end_ts"),
                "pkt_count": float(row.get("pkt_count", 0.0)),
                "pps": float(row.get("pps", 0.0)),
                "src_ip_entropy": float(row.get("src_ip_entropy", 0.0)),
                "dst_ip_entropy": float(row.get("dst_ip_entropy", 0.0)),
            }
            for row in sorted_features
        ],
    }

    if meta_rows:
        max_error_row = max(meta_rows, key=lambda row: row["error"])
        report["max_error_window"] = {
            "idx": int(max_error_row.get("window_idx", -1)),
            "start": max_error_row.get("start_ts"),
            "end": max_error_row.get("end_ts"),
            "error": float(max_error_row.get("error", 0.0)),
        }

    logger.info(
        "inference_complete",
        decision=decision,
        anomalous_windows=anomalous_count,
        total_windows=len(errors_array),
    )

    detailed = pd.DataFrame(meta_rows)
    detailed["threshold"] = effective_threshold
    detailed["raw_flag"] = flags
    detailed["flagged"] = filtered

    return {
        "report": report,
        "details": detailed,
        "errors": errors_array,
    }
