
import sys
import struct
import datetime as dt
import csv
from pathlib import Path

def main():
    pcap_path = Path("../samples/attack.pcap")
    if not pcap_path.exists():
        print(f"Error: {pcap_path} not found")
        return

    print(f"Reading first packet from {pcap_path} (manual parse)...")
    
    with open(pcap_path, "rb") as f:
        # Global Header: 24 bytes
        # magic_number (4), version_major (2), version_minor (2), thiszone (4), sigfigs (4), snaplen (4), network (4)
        global_header = f.read(24)
        if len(global_header) < 24:
            print("File too short for global header")
            return
            
        magic = struct.unpack("I", global_header[0:4])[0]
        # Check endianness (0xa1b2c3d4 is standard big-endian, 0xd4c3b2a1 is little-endian)
        # Actually standard pcap magic is 0xa1b2c3d4.
        
        endian = "<" # Default to little-endian (PC intel)
        if magic == 0xa1b2c3d4:
            endian = ">"
        elif magic == 0xd4c3b2a1:
            endian = "<"
        elif magic == 0x0a0d0d0a: # Pcapng
             print("PCAPNG format detected. Simplistic parser might fail but trying...")
             # Pcapng is complex block structure.
        
        # Packet Header: 16 bytes
        # ts_sec (4), ts_usec (4), incl_len (4), orig_len (4)
        pkt_header = f.read(16)
        if len(pkt_header) < 16:
            print("File too short for packet header")
            return
            
        ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f"{endian}IIII", pkt_header)
        
        raw_ts = float(ts_sec) + float(ts_usec) / 1_000_000.0
        
        print(f"Raw Timestamp: {raw_ts}")
        print(f"Raw Date (UTC): {dt.datetime.fromtimestamp(raw_ts, tz=dt.timezone.utc)}")
        
        # Check against CSV
        csv_path = "data/arp_attack_intervals.csv"
        start_str = None
        if Path(csv_path).exists():
            with open(csv_path, "r") as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    if "attack.pcap" in row["pcap"]:
                        start_str = row["start"]
                        break
        
        if start_str:
            dt_obj = dt.datetime.fromisoformat(start_str)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
            label_ts = dt_obj.timestamp()
            
            print(f"Label Timestamp: {label_ts}")
            print(f"Label Date (UTC): {dt_obj}")
            
            diff = raw_ts - label_ts
            print(f"Diff (Packet - Label): {diff:.2f} seconds")
            print(f"Diff in hours: {diff/3600:.4f} hours")
            
            if abs(diff - 14400) < 60:
                print("CONCLUSION: Packet is ~4 hours AHEAD of Label. Needs -4h offset.")
            elif abs(diff + 14400) < 60:
                print("CONCLUSION: Packet is ~4 hours BEHIND of Label. Needs +4h offset.")
            elif abs(diff) < 60:
                print("CONCLUSION: Timestamps MATCH. No offset needed.")
            else:
                print("CONCLUSION: Offset is weird.")
        else:
            print("Label not found.")

if __name__ == "__main__":
    main()
