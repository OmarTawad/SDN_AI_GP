#src/dos_detector/data/labels.py

"""Label handling utilities."""

from __future__ import annotations

import datetime as dt
import warnings
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


@dataclass
class FileLabel:
    """Represents a file-level label."""

    attack: int
    family: str


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
        start = _parse_time(row["start"])
        end = _parse_time(row["end"])
        if end < start:
            warnings.warn(
                f"Attack interval end < start for {row.get('pcap')}; swapping values.",
                RuntimeWarning,
                stacklevel=2,
            )
            start, end = end, start
        interval = AttackInterval(start=start, end=end, family=str(family).lower())
        intervals.setdefault(str(row["pcap"]), []).append(interval)
    return intervals


def load_file_labels(path: Path) -> Dict[str, FileLabel]:
    """Load file-level labels from CSV."""

    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    required = {"file", "attack_label"}
    if not required.issubset(frame.columns):
        missing = required - set(frame.columns)
        raise ValueError(f"Missing required columns: {missing}")
    labels: Dict[str, FileLabel] = {}
    for _, row in frame.iterrows():
        name = Path(str(row["file"])).name
        attack = int(row.get("attack_label", 0))
        family = str(row.get("family", "normal")).lower()
        labels[name] = FileLabel(attack=attack, family=family)
    return labels


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
