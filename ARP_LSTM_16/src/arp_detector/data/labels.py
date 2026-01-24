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
    
    # DEBUG: Print intervals and first window time
    if windows:
        first_w = windows[0]
        print(f"[DEBUG_LABELS] Processing {len(windows)} windows.")
        print(f"[DEBUG_LABELS] First Window: {first_w.start_time} - {first_w.end_time}")
        print(f"[DEBUG_LABELS] First Window Date: {dt.datetime.fromtimestamp(first_w.start_time, tz=dt.timezone.utc)}")
        
        relevant_intervals = [i for i in intervals]
        print(f"[DEBUG_LABELS] Loaded {len(relevant_intervals)} intervals.")
        if relevant_intervals:
            first_i = relevant_intervals[0]
            print(f"[DEBUG_LABELS] First Interval: {first_i.start} - {first_i.end}")
            print(f"[DEBUG_LABELS] First Interval Date: {dt.datetime.fromtimestamp(first_i.start, tz=dt.timezone.utc)}")
            
            diff = first_w.start_time - first_i.start
            print(f"[DEBUG_LABELS] Diff (WinStart - IntStart): {diff:.2f} sec ({diff/3600:.4f} hrs)")

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
