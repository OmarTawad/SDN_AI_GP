import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append("src")
from arp_detector.config import load_config
from arp_detector.utils.io import load_json

# Load Data
df = pd.read_parquet("data/processed/attack.parquet")

# Filter for attack windows
attack_df = df[df["attack"] == 1]
print(f"Attack Windows: {len(attack_df)}")

# Print some key columns (assuming names like 'arp_count', 'arp_request', etc.)
# I'll just print columns that contain 'arp'
arp_cols = [c for c in df.columns if 'arp' in c.lower()]
if not arp_cols:
    print("No columns with 'arp' in name found! Printing first 5 columns.")
    print(attack_df.iloc[:, :5].head())
else:
    print(f"ARP Features (Mean of attack windows):")
    print(attack_df[arp_cols].mean())

# Also print 'packet_count' if it exists
if 'packet_count' in df.columns:
     print(f"Packet Count (Mean): {attack_df['packet_count'].mean()}")

