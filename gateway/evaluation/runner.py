"""Execution flow for unified MoE global evaluation."""

from __future__ import annotations

import json
import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from gateway.core import CLASS_NAME_TO_ID
from gateway.data.datasets.cache import discover_pcaps, load_cache_entries
from gateway.data.structures.pcap import PcapInfo
from gateway.evaluation.configuration import EvaluationConfig
from gateway.evaluation.reporting import (
    ensure_output_dir,
    write_classification_report,
    write_confusion_matrix,
    write_metrics_json,
)
from gateway.inference.model_loader import load_model


def _load_labelled_files() -> List[PcapInfo]:
    files = discover_pcaps(tasks=("dos", "arp"))
    labelled = [info for info in files if str(info.meta.get("label_source", "")).lower() == "labels.csv"]
    if not labelled:
        raise RuntimeError(
            "No labels.csv-backed files were discovered under samples/. "
            "Provide samples/labels.csv with filename+label columns."
        )
    return sorted(labelled, key=lambda info: (info.path.name.lower(), str(info.path)))


def _split_files(files: Sequence[PcapInfo], seed: int, train_ratio: float, val_ratio: float) -> Dict[str, List[PcapInfo]]:
    if not files:
        return {"train": [], "val": [], "test": []}

    shuffled = list(files)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))

    buckets: List[List[PcapInfo]] = [[], [], []]
    for info in shuffled:
        idx = 0 if len(buckets[0]) < n_train else 1 if len(buckets[1]) < n_val else 2
        buckets[idx].append(info)

    if not buckets[2] and buckets[0]:
        buckets[2] = [buckets[0][-1]]

    return {"train": buckets[0], "val": buckets[1], "test": buckets[2]}


def _split_stats(files: Sequence[PcapInfo]) -> Tuple[int, int]:
    total = len(files)
    positives = sum(1 for info in files if int(info.label) != CLASS_NAME_TO_ID["normal"])
    return total, positives


def _has_both_classes(files: Sequence[PcapInfo]) -> bool:
    total, positives = _split_stats(files)
    return total > 0 and 0 < positives < total


def _select_split_with_fallback(
    buckets: Dict[str, List[PcapInfo]],
    requested_split: str,
) -> Tuple[List[PcapInfo], str]:
    requested = requested_split.lower()
    if requested not in buckets:
        raise ValueError(f"Unknown split '{requested}'. Expected one of {sorted(buckets)}.")

    if _has_both_classes(buckets[requested]):
        return buckets[requested], requested

    for candidate in ["val", "train", "test"]:
        if candidate == requested:
            continue
        if _has_both_classes(buckets[candidate]):
            total, positives = _split_stats(buckets[requested])
            print(
                f"[Eval] Requested split='{requested}' has single class "
                f"(pos={positives}, total={total}); falling back to '{candidate}'."
            )
            return buckets[candidate], candidate

    total, positives = _split_stats(buckets[requested])
    print(
        f"[Eval] Warning: split='{requested}' has single class (pos={positives}, total={total}) "
        "and no alternative split contains both classes. Metrics may be degenerate."
    )
    return buckets[requested], requested


def _build_dataset(
    config: EvaluationConfig,
    files: Sequence[PcapInfo],
) -> Tuple[IterableDataset[Tuple[Dict[str, Tensor], Tensor]], bool]:
    entries, missing, mismatched = load_cache_entries(config.cache_base, files, config.tasks)

    if config.use_cache != "off" and entries and not missing and not mismatched:
        from gateway.data.datasets.cached_dataset import CachedMoEDataset

        payloads: List[Dict[str, object]] = []
        for entry in entries:
            data = entry["data"]
            features = {
                key: tensor.detach().to(torch.float32)
                for key, tensor in data.get("features", {}).items()
                if isinstance(tensor, Tensor)
            }
            payload: Dict[str, object] = {"features": features}
            targets = data.get("targets")
            if targets is not None:
                payload["targets"] = torch.as_tensor(targets).to(torch.long)
            labels = data.get("labels")
            if isinstance(labels, dict):
                payload["labels"] = {
                    key: torch.as_tensor(value).to(torch.float32)
                    for key, value in labels.items()
                }
            meta = dict(data.get("meta", {}))
            meta.setdefault("source_path", str(entry["path"]))
            meta.setdefault("cache_path", str(entry["cache_path"]))
            payload["meta"] = meta
            payloads.append(payload)
        dataset = CachedMoEDataset(
            cache_entries=payloads,
            tasks=config.tasks,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed,
            max_windows_per_file=config.max_windows_per_file,
            max_total_windows=config.max_total_windows,
        )
        return dataset, True

    if config.use_cache == "on":
        problems: List[str] = []
        if not entries:
            problems.append("no cache files were found")
        if missing:
            problems.append("missing caches for: " + ", ".join(path.name for path in missing))
        if mismatched:
            problems.append(
                "invalid caches: " + ", ".join(f"{path.name} ({reason})" for path, reason in mismatched)
            )
        raise RuntimeError("Cache usage was forced with --use-cache on, but " + "; ".join(problems))

    if entries and (missing or mismatched):
        if missing:
            print(f"[Cache] Missing caches for {len(missing)} files; falling back to streaming.")
        if mismatched:
            print(f"[Cache] Ignoring {len(mismatched)} cache files due to mismatches.")

    from gateway.data.datasets.streaming_dataset import MoEDataset

    dataset = MoEDataset(
        files=files,
        tasks=config.tasks,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        max_windows_per_file=config.max_windows_per_file,
        max_total_windows=config.max_total_windows,
        status_interval=config.status_interval,
        max_file_size=config.file_size_bytes,
        max_packets_per_file=config.max_packets_per_file,
        file_timeout=config.file_timeout,
        max_packets_per_window=config.max_packets_per_window,
    )
    return dataset, False


def _collect_logits_and_targets(
    model: torch.nn.Module,
    config: EvaluationConfig,
    files: Sequence[PcapInfo],
    split_name: str,
) -> Tuple[Tensor, Tensor]:
    dataset, used_cache = _build_dataset(config, files)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    logits_parts: List[Tensor] = []
    target_parts: List[Tensor] = []
    with torch.inference_mode():
        for features, targets in dataloader:
            logits = model(features)
            logits_parts.append(logits.detach().cpu())
            target_parts.append(targets.detach().cpu().to(torch.long))

    if not logits_parts:
        raise RuntimeError(f"No windows were produced for split '{split_name}'.")

    logits = torch.cat(logits_parts, dim=0)
    targets = torch.cat(target_parts, dim=0)
    source = "cache" if used_cache else "stream"
    print(f"[Eval] Collected {targets.numel()} windows from split '{split_name}' via {source}.")
    return logits, targets


def _binary_labels(targets: Tensor, normal_index: int) -> Tensor:
    return (targets != normal_index).to(torch.float32)


def _attack_probabilities(logits: Tensor, temperature: float, normal_index: int) -> Tensor:
    safe_temperature = max(float(temperature), 1e-6)
    scaled = logits / safe_temperature
    probs = torch.softmax(scaled, dim=1)
    return 1.0 - probs[:, normal_index]


def _fit_temperature(logits: Tensor, y_true: Tensor, normal_index: int) -> float:
    if logits.numel() == 0:
        return 1.0
    positives = float(y_true.sum().item())
    total = float(y_true.numel())
    if positives <= 0 or positives >= total:
        print(
            f"[Calib] Warning: validation windows are single-class (pos={int(positives)}, total={int(total)}). "
            "Skipping temperature fit and using T=1.0."
        )
        return 1.0

    log_temp = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.Adam([log_temp], lr=0.05)

    for _ in range(300):
        optimizer.zero_grad()
        temperature = torch.exp(log_temp).clamp(min=1e-3, max=100.0)
        scaled = logits / temperature
        attack_prob = (1.0 - torch.softmax(scaled, dim=1)[:, normal_index]).clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(attack_prob, y_true)
        loss.backward()
        optimizer.step()

    fitted = float(torch.exp(log_temp).clamp(min=1e-3, max=100.0).item())
    return fitted


def _fit_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    positives = int(y_true.sum())
    total = int(y_true.size)
    if positives <= 0 or positives >= total:
        print(
            f"[Calib] Warning: validation windows are single-class (pos={positives}, total={total}). "
            "Using threshold=0.5."
        )
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if thresholds.size == 0:
        return 0.5

    f1_scores = (2.0 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    if np.isnan(f1_scores).all():
        return 0.5
    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx])


def _evaluate_binary(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    temperature: float,
) -> Tuple[Dict[str, float | int], np.ndarray, str]:
    y_pred = (probs >= threshold).astype(int)
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    if np.unique(y_true).size > 1:
        roc_auc = float(roc_auc_score(y_true, probs))
        if math.isnan(roc_auc):
            roc_auc = 0.5
    else:
        roc_auc = 0.5

    metrics: Dict[str, float | int] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "temperature": float(temperature),
        "samples": int(y_true.size),
        "positives": int(y_true.sum()),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["normal", "attack"],
        zero_division=0,
    )
    return metrics, cm, report


def run(config: EvaluationConfig) -> Dict[str, float | int]:
    """Run global unified MoE evaluation and persist report artifacts."""

    if not config.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint}")

    torch.set_num_threads(config.num_threads)
    model = load_model(config.checkpoint, gating_hidden_override=None)

    files = _load_labelled_files()
    buckets = _split_files(files, seed=config.seed, train_ratio=config.train_ratio, val_ratio=config.val_ratio)

    eval_files, eval_split = _select_split_with_fallback(buckets, config.split)
    if not eval_files:
        raise RuntimeError(f"Evaluation split '{eval_split}' is empty.")

    needs_calibration = config.temperature is None or config.threshold is None
    calibration_logits: Tensor | None = None
    calibration_targets: Tensor | None = None
    calibration_split = "val"

    if needs_calibration:
        calibration_files, calibration_split = _select_split_with_fallback(buckets, "val")
        if not calibration_files:
            raise RuntimeError("Calibration split is empty and no fallback split is available.")
        calibration_logits, raw_targets = _collect_logits_and_targets(model, config, calibration_files, calibration_split)
        calibration_targets = _binary_labels(raw_targets, CLASS_NAME_TO_ID["normal"])

    if config.temperature is not None:
        temperature = float(config.temperature)
    else:
        assert calibration_logits is not None and calibration_targets is not None
        temperature = _fit_temperature(calibration_logits, calibration_targets, CLASS_NAME_TO_ID["normal"])
    temperature = max(temperature, 1e-6)

    if config.threshold is not None:
        threshold = float(config.threshold)
    else:
        assert calibration_logits is not None and calibration_targets is not None
        calibration_probs = _attack_probabilities(
            calibration_logits,
            temperature=temperature,
            normal_index=CLASS_NAME_TO_ID["normal"],
        ).numpy()
        threshold = _fit_threshold(calibration_targets.numpy().astype(int), calibration_probs)

    eval_logits, eval_targets_raw = _collect_logits_and_targets(model, config, eval_files, eval_split)
    eval_targets = _binary_labels(eval_targets_raw, CLASS_NAME_TO_ID["normal"]).numpy().astype(int)
    eval_probs = _attack_probabilities(
        eval_logits,
        temperature=temperature,
        normal_index=CLASS_NAME_TO_ID["normal"],
    ).numpy()

    metrics, cm, report = _evaluate_binary(eval_targets, eval_probs, threshold=threshold, temperature=temperature)

    output_dir = ensure_output_dir(config.output_dir)
    write_metrics_json(output_dir, metrics)
    write_confusion_matrix(output_dir, cm)
    write_classification_report(output_dir, report)

    print(f"[Eval] split={eval_split} calibration_split={calibration_split}")
    print(json.dumps(metrics, indent=2))
    return metrics


__all__ = ["run"]
