#!/usr/bin/env python3
"""
Live detection using PyTorch CNN for ARP spoofing.
"""
import argparse, os, json
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

# ---------- helpers ----------
def convert_mac_ip(df):
    for col in ['sender_mac','target_mac']:
        intcol = f"{col}_int"
        if intcol not in df.columns:
            df[intcol] = df.get(col,"00:00:00:00:00:00").astype(str).fillna("00:00:00:00:00:00").apply(
                lambda x: int(x.replace(":","").replace("-",""),16) if x else 0)
    for col in ['sender_ip','target_ip']:
        intcol = f"{col}_int"
        if intcol not in df.columns:
            df[intcol] = df.get(col,"0.0.0.0").astype(str).fillna("0.0.0.0").apply(
                lambda ip: sum(int(p) << (8*(3-i)) for i,p in enumerate(ip.split('.'))))
    if 'op' in df.columns:
        df['op_is_request'] = (df['op']==1).astype(int)
        df['op_is_reply'] = (df['op']==2).astype(int)
    else:
        df['op_is_request'] = 0
        df['op_is_reply'] = 0
    return df

def build_rolling_sequences(X, timesteps):
    n,f = X.shape
    if timesteps<=1: return X.reshape(n,1,f)
    Xseq = np.zeros((n,timesteps,f), dtype=X.dtype)
    for i in range(n):
        start=max(0,i-timesteps+1)
        seq_len=i-start+1
        Xseq[i,timesteps-seq_len:,:]=X[start:i+1,:]
    return Xseq

# ---------- CNN model wrapper ----------
class CNNClassifier(nn.Module):
    def __init__(self, input_features, timesteps):
        super().__init__()
        self.conv1 = nn.Conv1d(input_features,64,3,padding=1)
        self.conv2 = nn.Conv1d(64,32,3,padding=1)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32*timesteps,64)
        self.fc2 = nn.Linear(64,1)
    def forward(self,x):
        x = x.transpose(1,2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)

# ---------- detection ----------
def run_detection(args):
    if not os.path.isfile(args.csv):
        raise SystemExit(f"CSV file not found: {args.csv}")
    if not os.path.isfile(args.model_path):
        raise SystemExit(f"Model file not found: {args.model_path}")
    scaler = joblib.load(args.scaler_path) if args.scaler_path else None

    df = pd.read_csv(args.csv)
    df = convert_mac_ip(df)
    features = ['op_is_request','sender_ip_int','target_ip_int','sender_mac_int','target_mac_int']
    for c in features:
        if c not in df.columns: df[c]=0
    X_raw = df[features].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw) if scaler else X_raw
    X_seq = build_rolling_sequences(X_scaled, args.seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(X_seq.shape[2],X_seq.shape[1]).to(device)
    model.load_state_dict(torch.load(args.model_path,map_location=device))
    model.eval()

    # batch inference
    n = X_seq.shape[0]
    batch_size = args.batch_size
    preds = np.zeros(n, dtype=np.float32)
    for i in range(0,n,batch_size):
        xb = torch.from_numpy(X_seq[i:i+batch_size]).to(device)
        with torch.no_grad():
            p = model(xb).cpu().numpy()
        preds[i:i+len(p)] = p

    # simple scoring (example: use behavior scoring as in LSTM)
    window_sec=args.window_sec
    times = df.get('ts', pd.Series(range(n))).values
    history = deque()
    mac_scores = defaultdict(float)
    for idx,row in enumerate(df.itertuples()):
        mac = getattr(row,'sender_mac_int',0)
        score = preds[idx]
        mac_scores[mac] += score

    print("\n=== Suspicious MACs ===")
    for mac,score in sorted(mac_scores.items(), key=lambda x:x[1], reverse=True)[:5]:
        print(f"MAC {mac:012x} | score={score:.3f}")

# ---------- CLI ----------
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--model-path",required=True)
    p.add_argument("--scaler-path",required=True)
    p.add_argument("--batch-size",type=int,default=20000)
    p.add_argument("--seq-len",type=int,default=5)
    p.add_argument("--window-sec",type=int,default=5)
    return p.parse_args()

def main():
    args=parse_args()
    run_detection(args)

if __name__=="__main__":
    main()

