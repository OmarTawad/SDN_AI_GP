# scripts/list_attack_macs.py
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    attackers = df[df["label"]==1]["sender_mac_int"].unique()
    print("[+] Attacker MACs:")
    for m in attackers:
        print(hex(m))
