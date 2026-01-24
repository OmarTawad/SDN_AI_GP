#!/usr/bin/env python3
import argparse
import sys
import glob
from pathlib import Path

# Ensure src is in python path if running locally without install
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from arp_detector.config import load_config
from arp_detector.data.processor import FeaturePipeline
from arp_detector.utils.io import ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Preprocess PCAP files with a 50MB limit per file.")
    parser.add_argument("pcaps", nargs='+', help="Glob patterns for PCAP files")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for processed features")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="Path to configuration file")
    
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Initialize pipeline
    pipeline = FeaturePipeline(config)
    
    # Determine output directory
    target_dir = args.out or config.paths.processed_dir
    ensure_dir(target_dir)
    
    # Resolve files
    paths = []
    for pattern in args.pcaps:
        paths.extend(Path(p) for p in glob.glob(pattern))
    paths = sorted(list(set(paths)))
    if not paths:
        print(f"No PCAPs matched pattern: {args.pcaps}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found {len(paths)} files. Processing with 50MB limit...")
    
    # Process with strict 50MB limit
    pipeline.process_files(paths, target_dir, limit=0, limit_mb=None)
    
    print(f"Done. Processed features saved to {target_dir}")

if __name__ == "__main__":
    main()
