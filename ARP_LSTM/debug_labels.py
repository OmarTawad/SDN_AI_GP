import pandas as pd
from pathlib import Path
import glob

# Ensure we use PyArrow to read parquet
try:
    import pyarrow
except ImportError:
    print("PyArrow not found")

files = glob.glob("data/processed/*.parquet")
print(f"Found {len(files)} parquet files.")

total_pos = 0
for f in sorted(files):
    try:
        df = pd.read_parquet(f)
        pos = df["attack"].sum()
        total = len(df)
        print(f"{Path(f).name}: {pos} positives / {total} total")
        total_pos += pos
    except Exception as e:
        print(f"Error reading {f}: {e}")

print(f"Total Positives: {total_pos}")
