#src/dos_detector/evaluation/metrics.py

"""Evaluation metrics utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn import metrics


@dataclass
class WindowMetrics:
    auc_roc: float
    auc_pr: float
    precision: float
    recall: float
    f1: float


@dataclass
class FileMetrics:
    precision: float
    recall: float
    f1: float


def compute_window_metrics(labels: Sequence[int], scores: Sequence[float]) -> WindowMetrics:
    y_true = np.array(labels, dtype=int)
    y_scores = np.array(scores, dtype=float)
    if y_true.size == 0:
        return WindowMetrics(auc_roc=0.0, auc_pr=0.0, precision=0.0, recall=0.0, f1=0.0)
    finite_mask = np.isfinite(y_scores)
    if not finite_mask.all():
        y_true = y_true[finite_mask]
        y_scores = y_scores[finite_mask]
    if y_true.size == 0:
        return WindowMetrics(auc_roc=0.0, auc_pr=0.0, precision=0.0, recall=0.0, f1=0.0)
    if np.unique(y_true).size < 2:
        auc_roc = 0.0
        auc_pr = 0.0
    else:
        auc_roc = metrics.roc_auc_score(y_true, y_scores)
        auc_pr = metrics.average_precision_score(y_true, y_scores)
    preds = (y_scores >= 0.5).astype(int)
    precision = metrics.precision_score(y_true, preds, zero_division=0)
    recall = metrics.recall_score(y_true, preds, zero_division=0)
    f1 = metrics.f1_score(y_true, preds, zero_division=0)
    return WindowMetrics(auc_roc=auc_roc, auc_pr=auc_pr, precision=precision, recall=recall, f1=f1)


def compute_file_metrics(labels: Sequence[int], predictions: Sequence[int]) -> FileMetrics:
    y_true = np.array(labels, dtype=int)
    y_pred = np.array(predictions, dtype=int)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    return FileMetrics(precision=precision, recall=recall, f1=f1)


__all__ = ["compute_window_metrics", "compute_file_metrics", "WindowMetrics", "FileMetrics"]
