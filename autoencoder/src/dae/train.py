from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .logging import get_logger
from .model import build_model, count_parameters
from .preprocess import (
    apply_log_transform,
    replace_invalid,
    save_feature_list,
    save_scaler,
    scaler_from_name,
    select_features,
)
from .threshold import ThresholdResult, save_threshold, select_threshold
from .utils_io import read_parquet_batches


@dataclass
class TrainingArtifacts:
    model_path: Path
    scaler_path: Path
    feature_list_path: Path
    clip_bounds_path: Path
    metrics_path: Path
    threshold_path: Path
    error_stats_path: Path
    source_stats_path: Path


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_windows_dataset(path: Path, batch_rows: int = 50000) -> pd.DataFrame:
    batches = list(read_parquet_batches(path, batch_rows))
    if not batches:
        raise ValueError(f"No data found in {path}")
    return pd.concat(batches, ignore_index=True)


def prepare_features(
    df: pd.DataFrame,
    feature_names: List[str],
    log_features: Sequence[str],
    clip_bounds: Tuple[float, float] | None,
    shuffle: bool,
    seed: int,
    split_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    logger = get_logger("train")
    df = df.copy()
    df = replace_invalid(df)

    subset = select_features(df, feature_names)
    subset = apply_log_transform(subset, log_features)

    if shuffle:
        subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    split_index = int(math.floor(len(subset) * split_ratio))
    if split_index <= 0 or split_index >= len(subset):
        raise ValueError("Invalid train/validation split result")

    train_df = subset.iloc[:split_index].copy()
    val_df = subset.iloc[split_index:].copy()

    clip_mapping: Dict[str, Dict[str, float]] | None = None
    if clip_bounds:
        lower, upper = clip_bounds
        train_lower = train_df.quantile(lower)
        train_upper = train_df.quantile(upper)
        train_df = train_df.clip(lower=train_lower, upper=train_upper, axis=1)
        val_df = val_df.clip(lower=train_lower, upper=train_upper, axis=1)
        clip_mapping = {
            "lower": train_lower.to_dict(),
            "upper": train_upper.to_dict(),
        }

    train_df = replace_invalid(train_df)
    val_df = replace_invalid(val_df)

    if clip_mapping is None:
        clip_mapping = {"lower": {}, "upper": {}}

    return train_df, val_df, train_df.values.astype(np.float32), val_df.values.astype(np.float32), clip_mapping


def fit_and_transform(
    train_array: np.ndarray,
    val_array: np.ndarray,
    scaler_name: str,
) -> Tuple[np.ndarray, np.ndarray, object]:
    scaler = scaler_from_name(scaler_name)
    scaler.fit(train_array)
    train_scaled = scaler.transform(train_array)
    val_scaled = scaler.transform(val_array)
    return train_scaled.astype(np.float32), val_scaled.astype(np.float32), scaler


def create_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(torch.from_numpy(train_data))
    val_dataset = TensorDataset(torch.from_numpy(val_data))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    denoising: bool,
    noise_std: float,
    grad_clip: float,
) -> float:
    model.train()
    epoch_loss = 0.0
    total = 0
    for (batch,) in loader:
        inputs = batch.to(device)
        if denoising and noise_std > 0:
            noise = torch.randn_like(inputs) * noise_std
            inputs_noisy = inputs + noise
        else:
            inputs_noisy = inputs

        optimizer.zero_grad()
        outputs = model(inputs_noisy)
        loss = criterion(outputs, inputs)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = inputs.size(0)
        epoch_loss += loss.item() * batch_size
        total += batch_size

    return epoch_loss / max(total, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    model.eval()
    epoch_loss = 0.0
    total = 0
    errors: List[float] = []
    with torch.no_grad():
        for (batch,) in loader:
            inputs = batch.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            batch_size = inputs.size(0)
            epoch_loss += loss.item() * batch_size
            total += batch_size

            batch_errors = ((outputs - inputs) ** 2).mean(dim=1).detach().cpu().numpy()
            errors.extend(batch_errors.tolist())
    return epoch_loss / max(total, 1), np.asarray(errors, dtype=np.float32)


def train_model(config: Config, windows_path: Path) -> TrainingArtifacts:
    logger = get_logger("train")

    artifacts_dir = Path(config.get("paths", "artifacts_dir"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config.get("train") or {}
    preprocess_cfg = config.get("preprocess") or {}
    model_cfg = config.get("model") or {}
    threshold_cfg = config.get("threshold") or {}
    feature_cfg = config.get("features") or {}

    seed = int(preprocess_cfg.get("seed", 1337))
    set_global_seed(seed)
    torch.set_num_threads(int(train_cfg.get("num_threads", 2)))

    batch_rows = int(config.get("extract", "batch_rows", default=50000))
    df = load_windows_dataset(windows_path, batch_rows)

    ratios_enabled = bool(feature_cfg.get("ratios", True))
    feature_names = list(feature_cfg.get("include", []))
    if ratios_enabled:
        for col in ("tcp_ratio", "udp_ratio", "icmp_ratio"):
            if col not in feature_names:
                feature_names.append(col)

    log_features = [str(f) for f in preprocess_cfg.get("log_features", [])]

    clip_bounds = None
    clip_quantiles = preprocess_cfg.get("clip_quantiles")
    if clip_quantiles and len(clip_quantiles) == 2:
        clip_bounds = (float(clip_quantiles[0]), float(clip_quantiles[1]))

    train_df, val_df, train_array, val_array, clip_mapping = prepare_features(
        df,
        feature_names,
        log_features,
        clip_bounds,
        bool(preprocess_cfg.get("shuffle", True)),
        seed,
        float(preprocess_cfg.get("train_val_split", 0.9)),
    )

    train_scaled, val_scaled, scaler = fit_and_transform(
        train_array,
        val_array,
        str(preprocess_cfg.get("scaler", "RobustScaler")),
    )

    save_scaler(scaler, artifacts_dir / "scaler.pkl")
    save_feature_list(feature_names, artifacts_dir / "feature_list.json")
    clean_clip = {k: {col: float(val) for col, val in mapping.items()} for k, mapping in clip_mapping.items()}
    (artifacts_dir / "clip_bounds.json").write_text(
        json.dumps(clean_clip, indent=2),
        encoding="utf-8",
    )

    train_loader, val_loader = create_dataloaders(
        train_scaled,
        val_scaled,
        int(train_cfg.get("batch_size", 1024)),
    )

    device = torch.device(train_cfg.get("device", "cpu"))
    model = build_model(model_cfg, input_dim=train_scaled.shape[1])
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(model_cfg.get("l2", 0.0)),
    )

    epochs = int(train_cfg.get("epochs", 30))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    grad_clip = float(train_cfg.get("gradient_clip_norm", 1.0))
    denoising = (model_cfg.get("type", "DenoisingAE").lower() == "denoisingae")
    noise_std = float(train_cfg.get("noise_std", 0.0))

    best_val_loss = float("inf")
    best_state = None
    best_errors = None
    patience_counter = 0
    history: List[Dict[str, float]] = []

    logger.info(
        "training_start",
        epochs=epochs,
        params=count_parameters(model),
        batch_size=train_loader.batch_size,
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            denoising,
            noise_std,
            grad_clip,
        )
        val_loss, val_errors = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        logger.info(
            "epoch_complete",
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_errors = val_errors
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("early_stopping", epoch=epoch)
                break

    if best_state is None or best_errors is None:
        raise RuntimeError("Training failed to produce a valid model")

    model.load_state_dict(best_state)

    error_stats = {
        "mean": float(np.mean(best_errors)),
        "std": float(np.std(best_errors)),
        "median": float(np.median(best_errors)),
        "mad": float(np.median(np.abs(best_errors - np.median(best_errors)))),
        "q99": float(np.quantile(best_errors, 0.99)),
        "q995": float(np.quantile(best_errors, 0.995)),
        "q999": float(np.quantile(best_errors, 0.999)),
    }

    trim_quantile = threshold_cfg.get("trim_quantile")
    threshold_errors = best_errors
    if trim_quantile is not None:
        tq = float(trim_quantile)
        if 0.0 < tq < 1.0:
            cutoff = np.quantile(best_errors, tq)
            threshold_errors = best_errors[best_errors <= cutoff]
            if threshold_errors.size == 0:
                threshold_errors = best_errors

    threshold_result = select_threshold(threshold_errors, threshold_cfg)
    save_threshold(threshold_result, artifacts_dir / "threshold.json")
    (artifacts_dir / "error_stats.json").write_text(
        json.dumps({
            "threshold_errors": {
                "count": int(threshold_errors.size),
                "quantile": float(trim_quantile) if trim_quantile is not None else None,
            },
            "stats": error_stats,
        }, indent=2),
        encoding="utf-8",
    )

    source_errors: Dict[str, List[float]] = defaultdict(list)
    lower_bounds_series = pd.Series(clip_mapping.get("lower", {}))
    upper_bounds_series = pd.Series(clip_mapping.get("upper", {}))

    eval_batch_size = int(train_cfg.get("eval_batch_size", 32768))
    total_rows = len(df)
    for start in range(0, total_rows, eval_batch_size):
        end = min(start + eval_batch_size, total_rows)
        chunk = df.iloc[start:end]
        feature_chunk = select_features(chunk, feature_names)
        feature_chunk = apply_log_transform(feature_chunk, log_features)
        if not lower_bounds_series.empty:
            feature_chunk = feature_chunk.clip(lower=lower_bounds_series, axis=1)
        if not upper_bounds_series.empty:
            feature_chunk = feature_chunk.clip(upper=upper_bounds_series, axis=1)
        feature_chunk = replace_invalid(feature_chunk)
        scaled_chunk = scaler.transform(feature_chunk.values).astype(np.float32)
        tensor = torch.from_numpy(scaled_chunk).to(device)
        with torch.no_grad():
            reconstruction = model(tensor)
            chunk_errors = ((reconstruction - tensor) ** 2).mean(dim=1).detach().cpu().numpy()
        sources = chunk.get("source")
        if sources is None:
            continue
        for src, err in zip(sources.tolist(), chunk_errors):
            source_errors.setdefault(str(src), []).append(float(err))

    source_stats = {}
    for src, values in source_errors.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float32)
        source_stats[src] = {
            "count": int(arr.size),
            "q99": float(np.quantile(arr, 0.99)),
            "q995": float(np.quantile(arr, 0.995)),
            "q999": float(np.quantile(arr, 0.999)),
        }

    (artifacts_dir / "source_stats.json").write_text(
        json.dumps(source_stats, indent=2),
        encoding="utf-8",
    )

    model_path = artifacts_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    model_config_path = artifacts_dir / "model_config.json"
    model_config_payload = {
        "input_dim": train_scaled.shape[1],
        "model": model_cfg,
        "feature_names": feature_names,
        "log_features": log_features,
        "config_hash": config.hash(),
    }
    model_config_path.write_text(json.dumps(model_config_payload, indent=2), encoding="utf-8")

    metrics_path = artifacts_dir / "train_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    logger.info(
        "training_complete",
        best_val_loss=best_val_loss,
        threshold=threshold_result.threshold,
    )

    return TrainingArtifacts(
        model_path=model_path,
        scaler_path=artifacts_dir / "scaler.pkl",
        feature_list_path=artifacts_dir / "feature_list.json",
        clip_bounds_path=artifacts_dir / "clip_bounds.json",
        metrics_path=metrics_path,
        threshold_path=artifacts_dir / "threshold.json",
        error_stats_path=artifacts_dir / "error_stats.json",
        source_stats_path=artifacts_dir / "source_stats.json",
    )
