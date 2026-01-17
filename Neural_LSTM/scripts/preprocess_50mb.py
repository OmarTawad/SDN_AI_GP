#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Ensure root (src) is in python path if not installed
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from dos_detector.config import load_config
from dos_detector.data.processor import FeaturePipeline

def main():
    parser = argparse.ArgumentParser(description="Preprocess PCAP files with a 50MB limit per file (Neural_LSTM).")
    parser.add_argument("pcaps", nargs="+", help="List of PCAP files to process")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to config.yaml")
    
    args = parser.parse_args()

    # Load configuration
    if not args.config.exists():
         print(f"[ERR] Config not found: {args.config}", file=sys.stderr)
         sys.exit(1)
         
    cfg = load_config(args.config)
    
    # Enforce 50MB limit
    print(f"[INFO] Enforcing 50MB limit per file.")
    limit_mb = 50.0

    # Initialize Pipeline
    pipeline = FeaturePipeline(cfg)
    
    # Process
    pipeline.process_files(
        pcaps=args.pcaps,
        out_dir=cfg.paths.processed_dir,
        limit=0, # No packet/window limit
        limit_mb=limit_mb # Byte limit
    )
    
    print(f"Done.")

if __name__ == "__main__":
    main()
