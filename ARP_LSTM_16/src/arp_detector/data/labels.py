#src/arp_detector/data/labels.py

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
        dt_obj = dt.datetime.fromisoformat(value)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
        return dt_obj.timestamp()


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
        intervals.setdefault(Path(str(row["pcap"])).name, []).append(interval)
    return intervals


def label_windows(
    windows: Sequence[Window],
    intervals: Sequence[AttackInterval],
    config: LabelsConfig,
) -> List[WindowLabels]:
    """Assign attack labels to each window."""

    labels: List[WindowLabels] = []
    
    # DEBUG: Check alignment
    if windows and len(windows) > 0:
        w0 = windows[0]
        print(f"[LABELS_DEBUG] First Window Start: {w0.start_time} ({dt.datetime.fromtimestamp(w0.start_time, tz=dt.timezone.utc)})")
        
        relevant = [i for i in intervals]
        if relevant:
            i0 = relevant[0]
            print(f"[LABELS_DEBUG] First Label Interval: {i0.start} ({dt.datetime.fromtimestamp(i0.start, tz=dt.timezone.utc)})")
            print(f"[LABELS_DEBUG] Diff: {w0.start_time - i0.start} sec")
        else:
            print("[LABELS_DEBUG] No intervals found for this file!")

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
