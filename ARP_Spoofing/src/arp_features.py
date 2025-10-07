#!/usr/bin/env python3
"""
Enhanced ARP feature extraction:
- Computes raw fields + derived anomaly flags
- Supports PCAP directories and safe CSV output
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from scapy.all import PcapReader, ARP
from collections import defaultdict, deque
import numpy as np

def ip_to_int(ip: str) -> int:
    try:
        parts = [int(p) for p in ip.split('.')]
        return (parts[0]<<24) | (parts[1]<<16) | (parts[2]<<8) | parts[3]
    except:
        return 0

def extract_features(pcap_file, window_sec=5):
    features = []
    mac_last_ip = {}  # MAC -> last seen IP
    ip_mac_map = defaultdict(set)  # IP -> set(MACs)
    mac_reply_times = defaultdict(deque)  # MAC -> deque of timestamps
    ip_mac_history = defaultdict(deque)  # IP -> deque of (timestamp, MAC)

    with PcapReader(pcap_file) as pcap:
        for pkt in tqdm(pcap, desc=f"Processing {os.path.basename(pcap_file)}", unit="pkt"):
            if ARP not in pkt:
                continue

            ts = pkt.time
            op = pkt[ARP].op
            sender_ip = pkt[ARP].psrc
            target_ip = pkt[ARP].pdst
            sender_mac = pkt[ARP].hwsrc
            target_mac = pkt[ARP].hwdst

            # Basic features
            row = {
                "ts": ts,
                "op": op,
                "op_is_request": int(op == 1),
                "op_is_reply": int(op == 2),
                "sender_ip": sender_ip,
                "target_ip": target_ip,
                "sender_mac": sender_mac,
                "target_mac": target_mac,
                "sender_ip_int": ip_to_int(sender_ip),
                "target_ip_int": ip_to_int(target_ip),
                "sender_mac_int": int(sender_mac.replace(":", ""), 16),
                "target_mac_int": int(target_mac.replace(":", ""), 16)
            }

            # ip_change_flag
            last_ip = mac_last_ip.get(sender_mac)
            row["ip_change_flag"] = int(last_ip is not None and last_ip != sender_ip)
            mac_last_ip[sender_mac] = sender_ip

            # mac_impersonation_flag
            ip_mac_map[sender_ip].add(sender_mac)
            row["mac_impersonation_flag"] = int(len(ip_mac_map[sender_ip]) > 1)

            # dup_ip_distinct_macs_5s
            hist = ip_mac_history[sender_ip]
            hist.append((ts, sender_mac))
            # remove entries older than window_sec
            while hist and ts - hist[0][0] > window_sec:
                hist.popleft()
            distinct_macs = {m for t,m in hist}
            row["dup_ip_distinct_macs_5s"] = len(distinct_macs)

            # rate_from (replies only)
            if op == 2:
                times = mac_reply_times[sender_mac]
                times.append(ts)
                while times and ts - times[0] > window_sec:
                    times.popleft()
                row["rate_from"] = len(times)
            else:
                row["rate_from"] = 0

            features.append(row)

    return pd.DataFrame(features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap_dir", default="data/pcaps", help="Directory containing PCAP files")
    parser.add_argument("--out_dir", default="data/features", help="Directory to save CSV feature files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pcap_files = [f for f in os.listdir(args.pcap_dir) if f.endswith(".pcap")]

    for pcap_file in pcap_files:
        pcap_path = os.path.join(args.pcap_dir, pcap_file)
        print(f"[*] Processing {pcap_file}")
        try:
            df = extract_features(pcap_path)
        except Exception as e:
            print(f"[!] Failed: {e}")
            continue
        csv_name = os.path.splitext(pcap_file)[0] + ".csv"
        df.to_csv(os.path.join(args.out_dir, csv_name), index=False)
        print(f"[+] Saved: {csv_name} ({len(df)} rows)")
