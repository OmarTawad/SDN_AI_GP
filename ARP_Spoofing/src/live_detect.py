#!/usr/bin/env python3
"""
live_detect.py — ARP spoofing detection with ML + behavior score.

Improvements:
- Only flags MACs that reply to multiple IPs in at least one window.
- Clear output when no spoofing is detected.
"""

from __future__ import annotations
import argparse
import os
import json
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tqdm import tqdm

DEFAULT_FEATURES = ['op_is_request', 'sender_ip_int', 'target_ip_int', 'sender_mac_int', 'target_mac_int']

# ---------------- helpers ----------------
def int_to_mac(mac_int: int) -> str:
    return ':'.join(f"{(mac_int >> ele) & 0xff:02x}" for ele in range(40, -1, -8))

def ip_to_int_safe(ip: str) -> int:
    try:
        parts = [int(p) for p in str(ip).split('.')]
        if len(parts) != 4:
            return 0
        return (parts[0]<<24) | (parts[1]<<16) | (parts[2]<<8) | parts[3]
    except:
        return 0

def convert_mac_ip(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['sender_mac', 'target_mac']:
        intcol = f"{col}_int"
        if intcol not in df.columns:
            if col in df.columns:
                df[intcol] = df[col].astype(str).fillna("00:00:00:00:00:00").apply(
                    lambda x: int(x.replace(":", "").replace("-", ""), 16) if x else 0
                )
            else:
                df[intcol] = 0
    for col in ['sender_ip', 'target_ip']:
        intcol = f"{col}_int"
        if intcol not in df.columns:
            if col in df.columns:
                df[intcol] = df[col].astype(str).fillna("0.0.0.0").apply(ip_to_int_safe)
            else:
                df[intcol] = 0
    if 'op' in df.columns:
        df['op_is_request'] = (df['op']==1).astype(int)
        df['op_is_reply'] = (df['op']==2).astype(int)
    else:
        df['op_is_request'] = 0
        df['op_is_reply'] = 0
    return df

def align_features_with_scaler(df: pd.DataFrame, scaler) -> Tuple[np.ndarray, List[str]]:
    if hasattr(scaler, "feature_names_in_") and scaler.feature_names_in_ is not None:
        feature_order = list(scaler.feature_names_in_)
    else:
        feature_order = DEFAULT_FEATURES.copy()
    for c in feature_order:
        if c not in df.columns:
            df[c] = 0
    return df[feature_order].values.astype(np.float32), feature_order

def build_rolling_sequences(X2d: np.ndarray, timesteps: int) -> np.ndarray:
    n,f = X2d.shape
    if timesteps <= 1:
        return X2d.reshape((n,1,f))
    Xseq = np.zeros((n,timesteps,f), dtype=X2d.dtype)
    for i in range(n):
        start = max(0,i-timesteps+1)
        seq_len = i-start+1
        Xseq[i,timesteps-seq_len:,:] = X2d[start:i+1,:]
    return Xseq

# ---------------- detection ----------------
def run_detection(args):
    if not os.path.isfile(args.csv):
        raise SystemExit(f"CSV file not found: {args.csv}")
    if not os.path.isfile(args.model_path):
        raise SystemExit(f"Model file not found: {args.model_path}")
    if args.scaler_path and not os.path.isfile(args.scaler_path):
        raise SystemExit(f"Scaler file not found: {args.scaler_path}")

    print(f"[info] Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    scaler = None
    if args.scaler_path:
        try:
            loaded = joblib.load(args.scaler_path)
            if isinstance(loaded, dict) and 'scaler' in loaded:
                scaler = loaded['scaler']
            elif hasattr(loaded, "transform"):
                scaler = loaded
        except Exception:
            print("[warning] Failed to load scaler; using raw features")

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("CSV empty")
    if 'ts' not in df.columns:
        df['ts'] = np.arange(len(df), dtype=np.float64)

    df = convert_mac_ip(df)
    X_raw, _ = align_features_with_scaler(df, scaler if scaler else SimpleIdentityScaler())
    if scaler:
        try:
            X_scaled = scaler.transform(X_raw)
        except:
            X_scaled = X_raw
    else:
        X_scaled = X_raw

    # handle sequences if model expects timesteps
    timesteps = 1
    try:
        inp = model.input_shape
        if isinstance(inp,(list,tuple)) and len(inp)==3:
            timesteps = int(inp[1]) if inp[1] else 1
    except: pass
    X_for_model = build_rolling_sequences(X_scaled, timesteps) if timesteps>1 else X_scaled

    # batch predict
    n = X_scaled.shape[0]
    preds = np.zeros((n,), dtype=np.float32)
    print(f"[info] Running model predictions (batch={args.batch_size})")
    for i in tqdm(range(0,n,args.batch_size), desc="Predicting"):
        xb = X_for_model[i:i+args.batch_size]
        y = model.predict(xb, verbose=0)
        preds[i:i+args.batch_size] = np.squeeze(y)

    # sliding window per MAC
    window_sec = args.window_sec
    times = df['ts'].values.astype(np.float64)
    history = deque()
    start_ts = float(times[0])

    mac_window_scores = defaultdict(lambda: defaultdict(list))
    mac_window_ips = defaultdict(lambda: defaultdict(set))  # track which IPs MAC replied with

    for idx in tqdm(range(n), desc="Sequential scan"):
        row = {
            'ts': float(times[idx]),
            'sender_mac_int': int(df.iloc[idx].get('sender_mac_int',0)),
            'sender_ip_int': int(df.iloc[idx].get('sender_ip_int',0)),
            'op_is_reply': int(df.iloc[idx].get('op_is_reply',0))
        }
        history.append(row)
        while history and (row['ts'] - history[0]['ts']) > window_sec:
            history.popleft()

        mac = row['sender_mac_int']
        if row['op_is_reply']:
            current_window = int((row['ts']-start_ts)//window_sec)
            mac_window_ips[mac][current_window].add(row['sender_ip_int'])
            reply_score = len(mac_window_ips[mac][current_window])
            final_score = args.alpha_model * preds[idx] + args.alpha_behavior * reply_score
            mac_window_scores[mac][current_window].append(final_score)

    # aggregate per MAC
    per_mac_summary = {}
    for mac, wins in mac_window_scores.items():
        window_avgs = []
        win_ip_counts = []
        for wid, scores in wins.items():
            avg = float(np.mean(scores))
            window_avgs.append(avg)
            win_ip_counts.append(len(mac_window_ips[mac][wid]))
        per_mac_summary[mac] = {
            'window_avgs': window_avgs,
            'mean_window_avg': float(np.mean(window_avgs)) if window_avgs else 0.0,
            'suspicious_windows': sum(1 for v in window_avgs if v>=args.window_score_threshold),
            'mean_ip_count_per_window': float(np.mean(win_ip_counts)) if win_ip_counts else 0.0
        }

    # ---------------- FLAG SUSPICIOUS MACS ----------------
    flagged = []
    for mac, summary in per_mac_summary.items():
        if summary['suspicious_windows'] >= args.min_windows and summary['mean_window_avg'] >= args.mac_score_thresh:
            # Only flag MACs that replied to multiple IPs
            if summary['mean_ip_count_per_window'] > 1:
                flagged.append((mac, summary['mean_window_avg'], summary['suspicious_windows'], summary['mean_ip_count_per_window']))

    flagged.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Suspicious MACs ===")
    if not flagged:
        print("No ARP spoofing detected (no MAC met strict thresholds).")
        # show top candidates for debugging if desired
        candidates = sorted(
            [(mac, s['mean_window_avg'], s['suspicious_windows'], s['mean_ip_count_per_window']) for mac,s in per_mac_summary.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        if candidates:
            print("\nTop candidate MACs (not flagged):")
            for mac, avg, sus, ipc in candidates:
                print(f"{int_to_mac(mac)} | avg={avg:.3f} | suspicious_windows={sus} | mean_ip_count={ipc:.1f}")
    else:
        for mac, avg, sus, ipc in flagged:
            print(f"MAC {int_to_mac(mac)} | avg_score={avg:.3f} | suspicious_windows={sus} | mean_ip_count={ipc:.1f}")
        top = flagged[0]
        print(f"\nMost suspicious MAC: {int_to_mac(top[0])} (avg_score={top[1]:.3f}, suspicious_windows={top[2]})")
        os.makedirs("results", exist_ok=True)
        out = [{"mac": int_to_mac(m), "avg_score": a, "suspicious_windows": s, "mean_ip_count_per_window": ipc} for m,a,s,ipc in flagged]
        pd.DataFrame(out).to_csv("results/flagged_macs.csv", index=False)
        with open("results/flagged_macs.json","w") as fh:
            json.dump(out, fh, indent=2)
        print("[+] Saved flagged MACs to results/flagged_macs.csv/.json")

# ---------------- helpers ----------------
class SimpleIdentityScaler:
    def transform(self, X):
        return X

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="ARP spoofing detection")
    p.add_argument("--csv", required=True)
    p.add_argument("--model-path", default="./models/arp_detector.h5")
    p.add_argument("--scaler-path", default=None)
    p.add_argument("--batch-size", type=int, default=20000)
    p.add_argument("--alpha_model", type=float, default=0.6)
    p.add_argument("--alpha_behavior", type=float, default=0.4)
    p.add_argument("--window-sec", type=int, default=5)
    p.add_argument("--min-windows", type=int, default=2)
    p.add_argument("--window-score-threshold", type=float, default=0.6)
    p.add_argument("--mac-score-thresh", type=float, default=0.7)
    return p.parse_args()

def main():
    args = parse_args()
    total = args.alpha_model + args.alpha_behavior
    if total <= 0:
        raise SystemExit("alpha_model + alpha_behavior must be > 0")
    args.alpha_model /= total
    args.alpha_behavior /= total
    run_detection(args)

if __name__=="__main__":
    main()
