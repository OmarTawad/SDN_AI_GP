"""Label handling utilities."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from ..config.types import LabelsConfig
from .structures import Window, WindowLabels


@dataclass
class AttackInterval:
    """Represents an attack interval for a PCAP."""

    start: float
    end: float
    family: str

    def overlaps(self, start: float, end: float) -> bool:
        return max(self.start, start) < min(self.end, end)


def _parse_time(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value)
    try:
        return float(value)
    except ValueError:
        return dt.datetime.fromisoformat(value).timestamp()


def load_attack_intervals(path: Path, config: LabelsConfig) -> Dict[str, List[AttackInterval]]:
    """Load attack intervals from CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Interval file not found: {path}")
    frame = pd.read_csv(path)
    required = {"pcap", "start", "end"}
    if not required.issubset(frame.columns):
        missing = required - set(frame.columns)
        raise ValueError(f"Missing required columns: {missing}")
    intervals: Dict[str, List[AttackInterval]] = {}
    for _, row in frame.iterrows():
        family = row.get("family", config.default_family)
        interval = AttackInterval(
            start=_parse_time(row["start"]),
            end=_parse_time(row["end"]),
            family=str(family).lower(),
        )
        intervals.setdefault(str(row["pcap"]), []).append(interval)
    return intervals


def label_windows(
    windows: Sequence[Window],
    intervals: Sequence[AttackInterval],
    config: LabelsConfig,
) -> List[WindowLabels]:
    """Assign attack labels to each window."""

    labels: List[WindowLabels] = []
    for window in windows:
        family = config.default_family
        attack = 0
        for interval in intervals:
            if interval.overlaps(window.start_time, window.end_time):
                family = interval.family
                attack = 1
                break
        labels.append(WindowLabels(attack=attack, family=family))
    return labels
