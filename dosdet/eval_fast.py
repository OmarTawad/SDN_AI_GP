from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, classification_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()
    # You can reuse your existing eval.py if you want.
    # This stub is a thin wrapper to avoid recomputing features; it expects infer.py outputs:
    from eval import file_metrics, window_metrics, detection_latency
    os.makedirs(args.reports_dir, exist_ok=True)
    csvs = sorted(glob.glob(os.path.join(args.reports_dir, "*_windows.csv")))
    jsons = sorted(glob.glob(os.path.join(args.reports_dir, "*.json")))
    wm = window_metrics(csvs, args.labels)
    fm = file_metrics(jsons, args.labels)
    dl = detection_latency(jsons, args.labels)
    with open(os.path.join(args.reports_dir, "summary_fast.json"), "w") as f:
        json.dump({"window_metrics": wm, "file_metrics": fm, "detection_latency": dl}, f, indent=2)
    print("Saved summary_fast.json")

if __name__ == "__main__":
    main()
