#!/usr/bin/env python3
"""
model_train_cnn.py — train a CNN (PyTorch) to detect ARP spoofing.

Usage example:
python src/model_train_cnn.py \
    --normal data/features/normal.csv \
    --attack data/features/attack.csv data/features/pure_attack.csv data/features/pure_attack2.csv \
    --epochs 50 \
    --batch-size 64 \
    --timesteps 5 \
    --model-out models/cnn/cnn_arp_detector.pt \
    --scaler-out models/cnn/cnn_scaler.pkl
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# ---------------- helpers ----------------
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
                def ip_to_int(ip: str) -> int:
                    parts = [int(p) for p in ip.split('.')]
                    return (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]
                df[intcol] = df[col].astype(str).fillna("0.0.0.0").apply(lambda ip: ip_to_int(ip) if ip and ip != "0.0.0.0" else 0)
            else:
                df[intcol] = 0
    if 'op' in df.columns:
        df['op_is_request'] = (df['op'] == 1).astype(int)
        df['op_is_reply'] = (df['op'] == 2).astype(int)
    else:
        df['op_is_request'] = 0
        df['op_is_reply'] = 0
    return df

def build_rolling_sequences(X2d: np.ndarray, timesteps: int) -> np.ndarray:
    n, f = X2d.shape
    if timesteps <= 1:
        return X2d.reshape((n, 1, f))
    Xseq = np.zeros((n, timesteps, f), dtype=X2d.dtype)
    for i in range(n):
        start = max(0, i - timesteps + 1)
        seq_len = i - start + 1
        Xseq[i, timesteps - seq_len:, :] = X2d[start:i + 1, :]
    return Xseq

def load_and_preprocess(csv_files: List[str], label: int) -> pd.DataFrame:
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df = convert_mac_ip(df)
        df['label'] = label
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# ---------------- Dataset ----------------
class SequenceDataset(Dataset):
    """Simple Dataset for (seq_len, features) sequences and binary labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, seq_len, features)
        # y: (N,)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.X[idx]),
            "label": torch.tensor(self.y[idx], dtype=torch.float32),
        }

def collate_batch(batch):
    features = torch.stack([b["features"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"features": features, "label": labels}

# ---------------- Model (CNN) ----------------
class CNNClassifier(nn.Module):
    """1D-conv based classifier inspired by the Keras model in the TF script."""

    def __init__(self, input_channels: int, timesteps: int, dropout: float = 0.2):
        super().__init__()
        # We will treat the feature axis as channels for Conv1d over timesteps: Conv1d(in_channels, out_channels, kernel)
        # To match the Keras Conv1D usage (input_shape=(timesteps, features)), in PyTorch we want (batch, features, timesteps)
        # So we will transpose in forward.
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=0)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=0)
        self.act2 = nn.ReLU()
        # After two valid convs, the temporal dimension shrinks: timesteps' = timesteps - 2*(kernel-1)
        self.flatten = nn.Flatten()
        # We'll compute flattened size dynamically at first forward pass if needed
        self.fc1 = nn.Linear(32 * max(1, timesteps - 4), 64)  # conservative; will be adjusted if timesteps small
        self.act3 = nn.ReLU()
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, timesteps, features)
        # transpose to (batch, features, timesteps)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        # if fc1 input size mismatches, resize layer (lazy workaround)
        if self.fc1.in_features != x.size(1):
            self.fc1 = nn.Linear(x.size(1), 64).to(x.device)
        x = self.fc1(x)
        x = self.act3(x)
        logits = self.fc_out(x).squeeze(-1)
        return logits  # raw logits; apply sigmoid outside if needed

# ---------------- Training utilities ----------------
def train_epoch(model, loader, optimizer, criterion, device, grad_clip=None, max_batches=None):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    batches = max(1, min(len(loader), (max_batches or len(loader))))
    return total_loss / batches

@torch.no_grad()
def eval_epoch(model, loader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []
    criterion = nn.BCEWithLogitsLoss()
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        total_loss += float(loss.detach().cpu())
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)
        trues.append(labels.cpu().numpy())
    if not preds:
        return {"val_loss": 0.0, "val_acc": 0.0}
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    acc = ( (preds >= 0.5).astype(int) == trues.astype(int) ).mean()
    return {"val_loss": float(total_loss / max(1, len(loader))), "val_acc": float(acc)}

# ---------------- main ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Train CNN ARP spoofing detector (PyTorch)")
    p.add_argument("--normal", required=True, help="Normal CSV file")
    p.add_argument("--attack", nargs='+', required=True, help="Attack CSV files")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--timesteps", type=int, default=5)
    p.add_argument("--model-out", required=True)
    p.add_argument("--scaler-out", required=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default=None, help="cpu or cuda (auto if omitted)")
    return p.parse_args()

def main():
    args = parse_args()

    print("[*] Loading normal data...")
    normal_df = load_and_preprocess([args.normal], 0)
    print("[*] Loading attack data...")
    attack_df = load_and_preprocess(args.attack, 1)
    if normal_df.empty and attack_df.empty:
        raise SystemExit("No data loaded.")

    df = pd.concat([normal_df, attack_df], ignore_index=True)
    feature_cols = ['op_is_request','sender_ip_int','target_ip_int','sender_mac_int','target_mac_int']
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.float32)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    os.makedirs(os.path.dirname(args.scaler_out) or ".", exist_ok=True)
    joblib.dump({'scaler': scaler}, args.scaler_out)
    print(f"[+] Saved scaler to {args.scaler_out}")

    # Build sequences
    X_seq = build_rolling_sequences(X_scaled, args.timesteps)
    # labels align to last packet in sequence
    n_seq = X_seq.shape[0]
    y_seq = y[:n_seq]

    if len(y_seq) == 0:
        raise SystemExit("No sequences created (timesteps too large or no data).")

    # Train/validation split (stratify by label)
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq if len(np.unique(y_seq))>1 else None
    )

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model
    input_features = X_seq.shape[2]
    model = CNNClassifier(input_channels=input_features, timesteps=args.timesteps).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print("[*] Training CNN (PyTorch)...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, grad_clip=args.grad_clip)
        metrics = eval_epoch(model, val_loader, device)
        val_loss = metrics["val_loss"]
        val_acc = metrics["val_acc"]
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            # save best immediately
            os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
            torch.save(best_state, args.model_out)
            print(f"[+] Saved best model state to {args.model_out}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

    # finalize: ensure best state saved to model_out (already saved during training loop)
    if best_state is not None:
        torch.save(best_state, args.model_out)
        print(f"[+] Final model saved to {args.model_out}")

if __name__ == "__main__":
    main()

