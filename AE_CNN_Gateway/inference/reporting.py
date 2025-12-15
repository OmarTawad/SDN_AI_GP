"""Reporting utilities for MoE inference results.


"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from gateway.core import CLASS_LABELS
from gateway.utils import get_logger
from .aggregation import AggregationSummary, ClassAggregation
from .suspicion import SuspicionSummary

LOGGER = get_logger("inference.reporting")


@dataclass(frozen=True)
class WindowRecord:
    """Record describing a single window prediction."""

    index: int
    prediction: str
    probabilities: Sequence[float]
    gating_weights: Sequence[float]


def write_window_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    """Persist per-window predictions to CSV.

    Args:
        path: Output CSV path.
        headers: Column headers to write as the first row.
        rows: Iterable of string sequences representing window records.
    """

    import csv

    LOGGER.info("Writing window-level CSV to %s", path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def build_json_summary(
    pcap_name: str,
    aggregation: AggregationSummary,
    suspicion: SuspicionSummary,
    attack_threshold: float,
    attack_prob_threshold: float,
    high_conf_threshold: float,
    min_attack_windows: int,
    min_high_conf_windows: int,
) -> dict[str, object]:
    """Construct a serialisable summary dictionary.

    Args:
        pcap_name: Source PCAP filename.
        aggregation: Aggregated class statistics.
        suspicion: Suspicion summary extracted from the PCAP.
        attack_threshold: Fraction of windows required to flag an attack.
        attack_prob_threshold: Probability threshold applied to attack promotion.
        high_conf_threshold: High confidence probability threshold.
        min_attack_windows: Minimum attack windows threshold.
        min_high_conf_windows: Minimum high confidence windows for escalation.

    Returns:
        dict[str, object]: Serialisable payload for JSON export.
    """

    class_payload = {
        label: {
            "count": stats.count,
            "fraction": stats.fraction,
            "mean_probability": stats.mean_probability,
            "high_confidence_windows": stats.high_conf_windows,
        }
        for label, stats in aggregation.classes.items()
    }
    return {
        "pcap": pcap_name,
        "verdict": aggregation.verdict,
        "classes": class_payload,
        "top_suspicious_ips": suspicion.ip_addresses,
        "top_suspicious_macs": suspicion.mac_addresses,
        "thresholds": {
            "attack_fraction": attack_threshold,
            "attack_probability": attack_prob_threshold,
            "high_confidence": high_conf_threshold,
            "min_attack_windows": min_attack_windows,
            "min_high_confidence_windows": min_high_conf_windows,
        },
        "total_windows": aggregation.total_windows,
    }


def write_json_summary(path: Path, payload: dict[str, object]) -> None:
    """Write the inference summary to JSON.

    Args:
        path: Destination JSON path.
        payload: Serialised summary payload.
    """

    LOGGER.info("Writing summary JSON to %s", path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def format_verdict_banner(
    pcap_name: str,
    aggregation: AggregationSummary,
) -> List[str]:
    """Create a verdict banner for console output.

    Args:
        pcap_name: Name of the processed capture.
        aggregation: Aggregated class statistics and verdict.

    Returns:
        List[str]: Formatted banner lines for logging.
    """

    verdict_stats: ClassAggregation = aggregation.classes[aggregation.verdict]
    banner = f"=== Final Verdict for {pcap_name} ==="
    lines = [banner]
    if aggregation.verdict == "normal":
        lines.append(
            f"Traffic appears normal (fraction {verdict_stats.fraction:.2f}, "
            f"mean probability {verdict_stats.mean_probability:.2f})"
        )
    else:
        lines.append(
            f"{aggregation.verdict.upper()} attack detected "
            f"(fraction {verdict_stats.fraction:.2f}, "
            f"mean probability {verdict_stats.mean_probability:.2f}, "
            f"windows {verdict_stats.count})"
        )
    lines.append("===========================================")
    return lines


__all__ = [
    "WindowRecord",
    "build_json_summary",
    "format_verdict_banner",
    "write_json_summary",
    "write_window_csv",
]
