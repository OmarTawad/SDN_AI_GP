from __future__ import annotations
import argparse, glob, json, os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report
)

def _iso_to_epoch(s: str) -> float:
    if not s or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    return pd.to_datetime(s, utc=True).view("int64") / 1e9

def file_metrics(json_paths: List[str], labels_csv: str) -> Dict:
    labs = pd.read_csv(labels_csv).fillna("")
    labs["base"] = labs["file"].apply(lambda p: os.path.basename(p))
    labs["attack"] = labs["attack_label"].astype(int)
    truth = {row["base"]: row["attack"] for _, row in labs.iterrows()}

    preds, gts = [], []
    for j in json_paths:
        with open(j, "r") as f:
            d = json.load(f)
        base = os.path.basename(d["file"])
        if base not in truth:
            continue
        preds.append(1 if d["decision"] == "attack" else 0)
        gts.append(truth[base])

    cm = confusion_matrix(gts, preds, labels=[0,1])
    report = classification_report(gts, preds, labels=[0,1], target_names=["normal","attack"], output_dict=True)
    return {"confusion": cm.tolist(), "report": report}

def window_metrics(csv_paths: List[str], labels_csv: str) -> Dict:
    labs = pd.read_csv(labels_csv).fillna("")
    labs["base"] = labs["file"].apply(lambda p: os.path.basename(p))
    labs["attack"] = labs["attack_label"].astype(int)
    labs["attack_start_ts"] = labs["attack_start_iso"].apply(_iso_to_epoch)

    y_true, y_score = [], []
    for c in csv_paths:
        base = os.path.basename(c).replace("_windows.csv","") + ".pcap"
        label_row = labs[labs["base"] == base]
        if label_row.empty:
            continue
        is_attack = int(label_row["attack"].values[0])
        atk_ts = label_row["attack_start_ts"].values[0] if "attack_start_ts" in label_row else np.nan

        df = pd.read_csv(c)
        # derive window start epoch from ISO
        ts_start = pd.to_datetime(df["t_start"], utc=True).view("int64")/1e9
        if is_attack == 0 or np.isnan(atk_ts):
            gt = np.zeros(len(df), dtype=int)
        else:
            gt = (ts_start >= atk_ts).astype(int)
        y_true.extend(gt.tolist())
        y_score.extend(df["prob"].astype(float).tolist())

    if len(set(y_true)) <= 1:
        roc = np.nan
        pr = np.nan
    else:
        roc = roc_auc_score(y_true, y_score)
        pr = average_precision_score(y_true, y_score)
    p, r, thr = precision_recall_curve(y_true, y_score)
    # Save PR curve to reports
    return {
        "roc_auc": float(roc) if not np.isnan(roc) else None,
        "pr_auc": float(pr) if not np.isnan(pr) else None,
        "pr_curve": {"precision": p.tolist(), "recall": r.tolist(), "thresholds": thr.tolist()}
    }

def detection_latency(json_paths: List[str], labels_csv: str) -> Dict:
    labs = pd.read_csv(labels_csv).fillna("")
    labs["base"] = labs["file"].apply(lambda p: os.path.basename(p))
    labs["attack"] = labs["attack_label"].astype(int)
    labs["attack_start_ts"] = labs["attack_start_iso"].apply(_iso_to_epoch)

    latencies = []
    for j in json_paths:
        with open(j, "r") as f:
            d = json.load(f)
        base = os.path.basename(d["file"])
        row = labs[labs["base"] == base]
        if row.empty or int(row["attack"].values[0]) == 0:
            continue
        atk_ts = row["attack_start_ts"].values[0]
        first_iso = d.get("first_attack_window_ts")
        if not first_iso:
            continue
        det_ts = _iso_to_epoch(first_iso)
        lat = max(0.0, det_ts - atk_ts)
        latencies.append(lat)

    if latencies:
        return {"count": len(latencies), "mean_sec": float(np.mean(latencies)), "p95_sec": float(np.percentile(latencies,95))}
    else:
        return {"count": 0, "mean_sec": None, "p95_sec": None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml for default paths")
    ap.add_argument("--reports_dir", default=None, help="Directory with infer outputs; defaults to config paths.reports_dir")
    ap.add_argument("--labels", default=None, help="labels.csv path; defaults to config preprocess.labels_csv")
    ap.add_argument("--out", default=None, help="Where to write summary.json and plots; defaults to reports_dir")
    args = ap.parse_args()

    cfg = {}
    if args.config and os.path.exists(args.config):
        cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    reports_dir = args.reports_dir or cfg.get("paths", {}).get("reports_dir", "reports")
    labels_path = args.labels or cfg.get("preprocess", {}).get("labels_csv", "labels/labels.csv")
    out_dir = args.out or reports_dir

    os.makedirs(out_dir, exist_ok=True)
    csvs = sorted(glob.glob(os.path.join(reports_dir, "*_windows.csv")))
    jsons = sorted(glob.glob(os.path.join(reports_dir, "*.json")))

    wm = window_metrics(csvs, labels_path)
    fm = file_metrics(jsons, labels_path)
    dl = detection_latency(jsons, labels_path)

    summary = {"window_metrics": wm, "file_metrics": fm, "detection_latency": dl}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # PR curve plot
    if wm["pr_curve"]["precision"]:
        plt.figure()
        plt.plot(wm["pr_curve"]["recall"], wm["pr_curve"]["precision"])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.savefig(os.path.join(out_dir, "pr_curve.png"), bbox_inches="tight")
        plt.close()

    # Confusion matrix plot
    cm = np.array(fm["confusion"])
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("File-level Confusion Matrix")
    plt.colorbar()
    plt.xticks([0,1], ["normal","attack"])
    plt.yticks([0,1], ["normal","attack"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center", color="black")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    print("Saved evaluation summary to", os.path.join(out_dir, "summary.json"))

if __name__ == "__main__":
    main()
