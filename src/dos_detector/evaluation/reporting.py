"""Reporting utilities."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict

import pandas as pd

from ..utils.io import ensure_dir
from .metrics import FileMetrics, WindowMetrics


def write_metrics_report(
    out_dir: Path,
    window_metrics: WindowMetrics,
    file_metrics: FileMetrics,
    notes: Dict[str, object] | None = None,
) -> None:
    """Persist metrics to CSV and Markdown."""

    ensure_dir(out_dir)
    window_df = pd.DataFrame([asdict(window_metrics)])
    file_df = pd.DataFrame([asdict(file_metrics)])
    window_df.to_csv(out_dir / "window_metrics.csv", index=False)
    file_df.to_csv(out_dir / "file_metrics.csv", index=False)
    summary_lines = ["# Evaluation Summary", "", "## Window metrics"]
    for key, value in asdict(window_metrics).items():
        summary_lines.append(f"- **{key}**: {value:.4f}")
    summary_lines.append("\n## File metrics")
    for key, value in asdict(file_metrics).items():
        summary_lines.append(f"- **{key}**: {value:.4f}")
    if notes:
        summary_lines.append("\n## Notes")
        for key, value in notes.items():
            summary_lines.append(f"- **{key}**: {value}")
    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


__all__ = ["write_metrics_report"]
