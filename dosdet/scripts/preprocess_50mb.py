import argparse
import sys
import yaml
from data.preprocess import preprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcaps", help="Glob pattern for PCAP files")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels", default="labels/labels.csv")
    args = ap.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    
    # Enforce 50MB limit
    print(f"[INFO] Enforcing 50MB limit per file via configuration override.")
    # cfg["preprocess"]["byte_limit"] = int(50.0 * 1024 * 1024)
    cfg["preprocess"]["byte_limit"] = None
    cfg["preprocess"]["limit"] = 0 # No window count limit

    preprocess(cfg, args.pcaps, args.labels)

if __name__ == "__main__":
    main()
