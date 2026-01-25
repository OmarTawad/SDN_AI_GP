
import sys
import struct
import datetime as dt
import csv
from pathlib import Path

def parse_pcap_header_timestamp(pcap_path):
    print(f"Reading {pcap_path}...")
    with open(pcap_path, "rb") as f:
        # Global Header: 24 bytes
        global_header = f.read(24)
        if len(global_header) < 24:
            raise ValueError("File too short for global header")
            
        magic = struct.unpack("I", global_header[0:4])[0]
        
        # Determine endianness
        endian = "<"
        if magic == 0xa1b2c3d4: # Big-endian
            endian = ">"
        elif magic == 0xd4c3b2a1: # Little-endian
            endian = "<"
        elif magic == 0x0a0d0d0a: # Pcapng
             print("Warning: PCAPNG format detected. Trying to find first Enhanced Packet Block...")
             # Pcapng is harder. Let's skip simplified logic for now and hope it's legacy pcap.
             # If it acts weird, we'll see.
             endian = "<" 
        
        # Packet Header: 16 bytes
        pkt_header = f.read(16)
        if len(pkt_header) < 16:
            raise ValueError("File too short for packet header")
            
        # struct pcap_pkthdr {
        #     struct timeval ts;  /* time stamp */
        #     bpf_u_int32 caplen; /* length of portion present */
        #     bpf_u_int32 len;    /* length this packet (off wire) */
        # };
        ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f"{endian}IIII", pkt_header)
        
        return float(ts_sec) + float(ts_usec) / 1_000_000.0

def load_label_start(csv_path, pcap_filename):
    print(f"Reading labels from {csv_path}...")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if pcap_filename in row["pcap"]:
                return row["start"]
    return None

def main():
    if len(sys.argv) < 2:
        # Default fallback
        pcap_path_str = "../samples/attack.pcap"
    else:
        pcap_path_str = sys.argv[1]

    pcap_path = Path(pcap_path_str)
    if not pcap_path.exists():
        print(f"Error: File not found: {pcap_path}")
        print("Please provide the correct path to attack.pcap as an argument.")
        print("Example: python3 diagnose_timestamps.py ../samples/attack.pcap")
        return

    csv_path = Path("data/arp_attack_intervals.csv")
    
    try:
        pkt_ts = parse_pcap_header_timestamp(pcap_path)
        print(f"\n[Packet 1] Raw Stamp: {pkt_ts:.6f}")
        print(f"[Packet 1] Date (UTC): {dt.datetime.fromtimestamp(pkt_ts, tz=dt.timezone.utc)}")
        
        label_str = load_label_start(csv_path, pcap_path.name)
        if not label_str:
            print(f"Error: Could not find '{pcap_path.name}' in {csv_path}")
            return
            
        # Parse ISO
        dt_obj = dt.datetime.fromisoformat(label_str)
        if dt_obj.tzinfo is None: dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
        label_ts = dt_obj.timestamp()
        
        print(f"\n[Label Start] Raw Stamp: {label_ts:.6f}")
        print(f"[Label Start] Date (UTC): {dt_obj}")
        
        diff = pkt_ts - label_ts
        print(f"\n[DIFFERENCE] Packet - Label = {diff:.2f} seconds")
        print(f"[DIFFERENCE] In Hours: {diff/3600:.4f} hours")
        
        if abs(diff) > 100:
            print("\n*** DIAGNOSIS ***")
            if diff > 0:
                print(f"Packet is AHEAD of Label by {diff/3600:.1f} hours.")
                print(f"FIX: We need to SUBTRACT {abs(diff):.0f} seconds from packet timestamps.")
            else:
                print(f"Packet is BEHIND Label by {abs(diff)/3600:.1f} hours.")
                print(f"FIX: We need to ADD {abs(diff):.0f} seconds to packet timestamps.")
        else:
            print("\n*** DIAGNOSIS ***\nTimestamps look aligned! (Difference is negligible)")

    except Exception as e:
        print(f"Crash: {e}")

if __name__ == "__main__":
    main()
