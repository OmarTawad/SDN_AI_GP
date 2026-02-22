"""Reporting helpers for unified MoE global evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def ensure_output_dir(path: Path) -> Path:
    """Create and return the output directory."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_metrics_json(output_dir: Path, metrics: Dict[str, float | int]) -> Path:
    """Persist metrics JSON payload."""

    path = output_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def write_confusion_matrix(output_dir: Path, matrix: np.ndarray) -> Path:
    """Persist binary confusion matrix as ``.npy``."""

    path = output_dir / "confusion_matrix.npy"
    np.save(path, matrix)
    return path


def write_classification_report(output_dir: Path, report: str) -> Path:
    """Persist textual classification report."""

    path = output_dir / "classification_report.txt"
    path.write_text(report, encoding="utf-8")
    return path


__all__ = [
    "ensure_output_dir",
    "write_classification_report",
    "write_confusion_matrix",
    "write_metrics_json",
]
