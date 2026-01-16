#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import yaml
from pathlib import Path

# Ensure root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.preprocess import preprocess

def main():
    parser = argparse.ArgumentParser(description="Preprocess PCAP files with a 50MB limit per file (arpdet).")
    parser.add_argument("pcaps", help="Glob pattern for PCAP files (e.g. 'data/*.pcap')")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--labels", default="labels/labels.csv", help="Path to labels.csv")
    
    args = parser.parse_args()

    # Load configuration
    if not args.config.exists():
         print(f"[ERR] Config not found: {args.config}", file=sys.stderr)
         sys.exit(1)
         
    cfg = yaml.safe_load(open(args.config))
    
    # Enforce 50MB limit
    print(f"[INFO] Enforcing 50MB limit per file via configuration override.")
    if "preprocess" not in cfg:
        cfg["preprocess"] = {}
    cfg["preprocess"]["byte_limit"] = int(50.0 * 1024 * 1024)
    cfg["preprocess"]["limit"] = 0 # No window limit, just byte limit

    # Resolve files
    # The existing preprocess function handles glob resolution but expects a list or string
    # We pass the glob directly.
    
    print(f"[DEBUG] CWD: {os.getcwd()}")
    print(f"[DEBUG] args.labels: '{args.labels}'")
    
    # Force correct labels path if default seems wrong or to be safe
    if args.labels == "data/labels.csv":
        print("[WARN] Correcting labels path from data/labels.csv to labels/labels.csv")
        args.labels = "labels/labels.csv"

    print(f"[DEBUG] Using labels: {args.labels}")
    # Force correct labels path if default seems wrong or to be safe
    if args.labels == "data/labels.csv":
        print("[WARN] Correcting labels path from data/labels.csv to labels/labels.csv")
        args.labels = "labels/labels.csv"

    print(f"[DEBUG] Using labels: {args.labels}")
    preprocess(cfg, args.pcaps, args.labels)
    
    print(f"Done.")

if __name__ == "__main__":
    main()
