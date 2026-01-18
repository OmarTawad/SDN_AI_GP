from __future__ import annotations
import argparse, glob, json, os, sys, time
try:
    import resource
except ImportError:  # pragma: no cover - not available on Windows
    resource = None
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

def _iso_to_epoch(s: str) -> float:
    if not s or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    dt = pd.to_datetime(s, utc=True)
    return dt.value / 1e9

def file_metrics(json_paths: List[str], labels_csv: str) -> Dict:
    labs = pd.read_csv(labels_csv).fillna("")
    labs["base"] = labs["file"].apply(lambda p: os.path.basename(p))
    labs["attack"] = labs["attack_label"].astype(int)
    truth = {row["base"]: row["attack"] for _, row in labs.iterrows()}

    preds, gts = [], []
    score_true, score_prob = [], []
    for j in json_paths:
        with open(j, "r") as f:
            d = json.load(f)
        if "file" not in d:
            continue
        base = os.path.basename(d["file"])
        if base not in truth:
            continue
        pred = 1 if d["decision"] == "attack" else 0
        gt = truth[base]
        preds.append(pred)
        gts.append(gt)
        if "max_probability" in d:
            score_true.append(gt)
            score_prob.append(float(d["max_probability"]))

    cm = confusion_matrix(gts, preds, labels=[0, 1])
    if gts:
        report = classification_report(
            gts,
            preds,
            labels=[0, 1],
            target_names=["normal", "attack"],
            output_dict=True,
            zero_division=0,
        )
    else:
        report = {}
    metrics = {
        "accuracy": float(accuracy_score(gts, preds)) if gts else None,
        "precision": float(precision_score(gts, preds, zero_division=0)) if gts else None,
        "recall": float(recall_score(gts, preds, zero_division=0)) if gts else None,
        "f1": float(f1_score(gts, preds, zero_division=0)) if gts else None,
        "samples": int(len(gts)),
        "positives": int(sum(gts)),
    }

    if len(set(score_true)) <= 1:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    elif score_true:
        metrics["roc_auc"] = float(roc_auc_score(score_true, score_prob))
        metrics["pr_auc"] = float(average_precision_score(score_true, score_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    metrics.update({"confusion": cm.tolist(), "report": report})
    return metrics

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
        ts_start = pd.to_datetime(df["t_start"], utc=True, format="mixed", errors="coerce").astype("int64")/1e9
        if is_attack == 0 or np.isnan(atk_ts):
            gt = np.zeros(len(df), dtype=int)
        else:
            gt = (ts_start >= atk_ts).astype(int)
        y_true.extend(gt.tolist())
        y_score.extend(df["prob"].astype(float).tolist())

    if not y_true:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "best_f1": None,
            "best_f1_threshold": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "samples": 0,
            "positives": 0,
            "pr_curve": {"precision": [], "recall": [], "thresholds": []},
        }

    if len(set(y_true)) <= 1:
        roc = np.nan
        pr = np.nan
    else:
        roc = roc_auc_score(y_true, y_score)
        pr = average_precision_score(y_true, y_score)

    p, r, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / (p + r + 1e-9)
    best_idx = int(np.nanargmax(f1)) if len(f1) else 0
    best_tau = thr[max(0, best_idx - 1)] if len(thr) else 0.5
    best_f1 = float(np.nanmax(f1)) if len(f1) else 0.0
    y_pred = (np.array(y_score) >= best_tau).astype(int)

    return {
        "roc_auc": float(roc) if not np.isnan(roc) else None,
        "pr_auc": float(pr) if not np.isnan(pr) else None,
        "best_f1": best_f1,
        "best_f1_threshold": float(best_tau),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "samples": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "pr_curve": {"precision": p.tolist(), "recall": r.tolist(), "thresholds": thr.tolist()},
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
        if "file" not in d:
            continue
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

def _memory_gb() -> float | None:
    if resource is None:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss <= 0:
        return None
    if sys.platform == "darwin":
        rss_bytes = float(rss)
    else:
        rss_bytes = float(rss) * 1024.0
    return rss_bytes / (1024.0 ** 3)

def main():
    start_wall = time.perf_counter()
    start_cpu = time.process_time()

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

    elapsed = time.perf_counter() - start_wall
    cpu_time = time.process_time() - start_cpu
    cpu_percent = (cpu_time / elapsed * 100.0) if elapsed > 0 else None
    checkpoint = None
    artifacts_dir = cfg.get("paths", {}).get("artifacts_dir")
    if artifacts_dir:
        ckpt_path = os.path.join(artifacts_dir, "model_best.pt")
        if os.path.exists(ckpt_path):
            checkpoint = os.path.abspath(ckpt_path)
    flat = {
        "accuracy": wm.get("accuracy"),
        "precision": wm.get("precision"),
        "recall": wm.get("recall"),
        "f1": wm.get("f1"),
        "threshold": wm.get("best_f1_threshold"),
        "num_windows": wm.get("samples"),
        "num_sequences": fm.get("samples"),
        "positives": wm.get("positives"),
        "roc_auc": wm.get("roc_auc"),
        "pr_auc": wm.get("pr_auc"),
        "split": "reports",
        "checkpoint": checkpoint,
        "elapsed_seconds": float(elapsed),
        "cpu_percent": float(cpu_percent) if cpu_percent is not None else None,
        "memory_gb": _memory_gb(),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(flat, f, indent=2)

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
    print("Saved flat metrics to", os.path.join(out_dir, "metrics.json"))

if __name__ == "__main__":
    main()
