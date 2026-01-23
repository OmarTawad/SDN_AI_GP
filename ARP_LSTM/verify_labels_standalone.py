import csv
import datetime as dt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

@dataclass
class AttackInterval:
    start: float
    end: float
    family: str

    def overlaps(self, start: float, end: float) -> bool:
        return max(self.start, start) < min(self.end, end)

@dataclass
class Window:
    start_time: float
    end_time: float

def _parse_time(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value)
    if not value:
        return float('nan')
    try:
        return float(value)
    except ValueError:
        # Replicating labels.py: interpret naive as UTC
        dt_obj = dt.datetime.fromisoformat(value)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
        return dt_obj.timestamp()

def load_intervals(path) -> Dict[str, List[AttackInterval]]:
    intervals = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pcap = row.get("pcap")
            if not pcap: continue
            family = row.get("family", "normal")
            raw_start = row.get("start", "")
            raw_end = row.get("end", "")
            
            # Skip if normal/empty
            if not raw_start or not raw_end:
                continue

            interval = AttackInterval(
                start=_parse_time(raw_start),
                end=_parse_time(raw_end),
                family=str(family).lower(),
            )
            name = Path(pcap).name # Logic from labels.py
            intervals.setdefault(name, []).append(interval)
    return intervals

# Mock Window from pure_attack.pcap
# User said: pure_attack.pcap Frame 297: Jan 7, 2014, 02:56:34.433255
# My CSV has: 2014-01-07T02:56:34.433255
ts_str = "2014-01-07T02:56:34.433255"
t0 = _parse_time(ts_str)
print(f"Testing Window Start (Timestamp): {t0} ({ts_str})")
window = Window(start_time=t0, end_time=t0+1.0)

# Load intervals
mapping = load_intervals("data/arp_attack_intervals.csv")
pcap_key = "pure_attack.pcap"
file_intervals = mapping.get(pcap_key, [])

print(f"Loaded {len(file_intervals)} intervals for {pcap_key}")
attack_found = 0
for iv in file_intervals:
    print(f"  Interval: {iv.start} - {iv.end} ({iv.family})")
    if iv.overlaps(window.start_time, window.end_time):
        print("  --> OVERLAP FOUND!")
        attack_found = 1

if attack_found:
    print("SUCCESS: Labeling logic works for this timestamp.")
else:
    print("FAILURE: No overlap found.")
