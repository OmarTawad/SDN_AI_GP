import argparse
import yaml
from data.preprocess import preprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcaps", help="Glob pattern for PCAP files")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    
    # Enforce 50MB limit per file
    print("[INFO] Enforcing 50MB limit per file via configuration override.")
    cfg["preprocess"]["byte_limit"] = int(50.0 * 1024 * 1024)
    cfg["preprocess"]["limit"] = 0

    # arpdet doesn't need labels csv argument in standard path? 
    # Checking existing code: preprocess(cfg, pcaps_glob, labels_csv)
    # arpdet config has labels_csv but preprocess func signature takes it.
    # The default script I wrote passed it.
    
    # Assuming standard signature from arpdet/data/preprocess.py
    labels = cfg["preprocess"].get("labels_csv", "labels/labels.csv")
    preprocess(cfg, args.pcaps, labels)

if __name__ == "__main__":
    main()
