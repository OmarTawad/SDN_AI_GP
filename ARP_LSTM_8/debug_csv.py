
from pathlib import Path
from arp_detector.data.labels import load_attack_intervals
from arp_detector.config.types import LabelsConfig

path = Path("data/arp_attack_intervals.csv")
config = LabelsConfig(intervals_csv=path, default_family="normal", family_mapping={}, attack_families=[])

import pandas as pd
print(f"Reading {path.absolute()}")
try:
    df = pd.read_csv(path)
    print(f"Columns: {df.columns}")
    print(f"Shape: {df.shape}")
    print(f"Head:\n{df.head()}")
    
    intervals = load_attack_intervals(path, config)
    print(f"Loaded {len(intervals)} keys: {list(intervals.keys())}")
    for k, v in intervals.items():
        print(f"Key: {k}, Intervals: {len(v)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
