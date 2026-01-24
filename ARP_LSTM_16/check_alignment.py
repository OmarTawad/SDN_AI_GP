
import sys
from pathlib import Path
import pandas as pd
import datetime as dt

def main():
    # 1. Read the Processed Parquet File
    parquet_path = Path("data/processed/attack.parquet")
    if not parquet_path.exists():
        print(f"Error: {parquet_path} does not exist. Did you run preprocessing?")
        return

    print(f"Loading {parquet_path}...")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Failed to read parquet: {e}")
        return

    if df.empty:
        print("Dataframe is empty!")
        return

    # Get time range from windows
    min_time = df["window_start"].min()
    max_time = df["window_end"].max()
    
    print(f"\n[Parquet Data] Window Range:")
    print(f"  Start Timestamp: {min_time}")
    print(f"  Start Date (UTC): {dt.datetime.fromtimestamp(min_time, tz=dt.timezone.utc)}")
    print(f"  End Timestamp:   {max_time}")
    print(f"  End Date (UTC):   {dt.datetime.fromtimestamp(max_time, tz=dt.timezone.utc)}")

    # 2. Read the Label CSV
    csv_path = Path("data/arp_attack_intervals.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    print(f"\n[Label CSV] Intervals for 'attack.pcap':")
    intervals = []
    
    # Simple CSV parse to avoid dependencies/complex logic
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        # assume order: pcap,start,end,family (or find indices)
        try:
            pcap_idx = header.index("pcap")
            start_idx = header.index("start")
            end_idx = header.index("end")
        except ValueError:
            print(f"Error: CSV header missing columns. Found: {header}")
            return

        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3: continue
            
            pcap_name = parts[pcap_idx]
            if "attack.pcap" in pcap_name: # Simple substring match
                start_str = parts[start_idx]
                end_str = parts[end_idx]
                
                # Parse ISO
                try:
                    s_dt = dt.datetime.fromisoformat(start_str)
                    if s_dt.tzinfo is None: s_dt = s_dt.replace(tzinfo=dt.timezone.utc)
                    
                    e_dt = dt.datetime.fromisoformat(end_str)
                    if e_dt.tzinfo is None: e_dt = e_dt.replace(tzinfo=dt.timezone.utc)
                    
                    intervals.append((s_dt.timestamp(), e_dt.timestamp()))
                    print(f"  Interval: {start_str} -> {end_str}")
                    print(f"     TS: {s_dt.timestamp()} -> {e_dt.timestamp()}")
                except ValueError:
                    print(f"  Skipping malformed row: {line}")

    if not intervals:
        print("  No intervals found for attack.pcap!")
        return

    # 3. Calculate Deltas
    # We compare the first window start to the first interval start roughly
    first_interval_start = intervals[0][0]
    
    diff = min_time - first_interval_start
    print(f"\n[Analysis]")
    print(f"  Data Start - Label Start = {diff:.2f} seconds")
    print(f"  Difference in Hours: {diff/3600:.4f} hours")
    
    if abs(diff) > 3600:
        print("\n  -> HUGE GAP DETECTED. You likely need a timezone offset.")
        if diff > 0:
            print("  -> Data is AHEAD of labels. Try subtracting hours from Data.")
        else:
            print("  -> Data is BEHIND labels. Try adding hours to Data.")
            
    # Check overlap
    overlap = False
    for (s, e) in intervals:
        # Check if [min_time, max_time] overlaps [s, e]
        if max(min_time, s) < min(max_time, e):
            overlap = True
            break
            
    if overlap:
        print("\n  -> STATUS: Alignment looks OK (Overlap detected).")
    else:
        print("\n  -> STATUS: NO OVERLAP. The labels will all be 0!")

if __name__ == "__main__":
    main()
