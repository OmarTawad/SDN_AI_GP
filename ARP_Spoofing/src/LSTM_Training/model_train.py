#!/usr/bin/env python3
"""
train_lstm_pytorch.py

PyTorch LSTM training script for ARP-spoofing sequence detection.
This script mirrors the structure and behavior of the earlier
TensorFlow trainer but in PyTorch and in a style consistent with
the project's `SupervisedTrainer` approach.

Features:
- Loads normal + attack CSVs, labels them
- Encodes categorical fields (MAC/IP) with sklearn LabelEncoder
- Scales numeric features with StandardScaler
- Converts packets into sliding sequences for LSTM
- Trains a PyTorch LSTM classifier with BCEWithLogitsLoss
- Supports early stopping, grad clipping, saving model + scaler/encoders

Note: Designed for ARP spoofing classification (binary).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    """Simple dataset of (seq_len, features) sequences and binary labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (num_seqs, seq_len, n_feat)
        # y: (num_seqs,)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.X[idx]).float(),
            "label": torch.tensor(self.y[idx], dtype=torch.float32),
        }


def collate_batch(batch):
    # batch is list of dicts
    features = torch.stack([b["features"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"features": features, "label": labels}


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_units=(64, 32), dropout=0.3):
        super().__init__()
        self.hidden_units = hidden_units
        self.input_size = input_size
        self.lstm1 = nn.LSTM(input_size, hidden_units[0], batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(dropout)
        if len(hidden_units) > 1:
            self.lstm2 = nn.LSTM(hidden_units[0], hidden_units[1], batch_first=True, bidirectional=False)
            self.dropout2 = nn.Dropout(dropout)
            final_dim = hidden_units[1]
        else:
            self.lstm2 = None
            final_dim = hidden_units[0]
        self.fc = nn.Linear(final_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        if self.lstm2 is not None:
            # take sequence output and pass to second LSTM (expecting return_sequences=False)
            out, _ = self.lstm2(out)
            out = self.dropout2(out)
            # out now (batch, seq_len, final_dim) but second LSTM returns last hidden state at each time;
            # for classification we select the last time-step
        # select last time-step
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits.squeeze(1)  # (batch,)


def preprocess_packet_df(df: pd.DataFrame, feature_cols: list) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    encoders: Dict[str, LabelEncoder] = {}
    df = df.copy()
    for col in feature_cols:
        if df[col].dtype == object or df[col].dtype == "str":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    sequences = []
    labels = []
    for i in range(len(X) - seq_len + 1):
        sequences.append(X[i : i + seq_len])
        labels.append(y[i + seq_len - 1])
    if not sequences:
        return np.zeros((0, seq_len, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


def train_loop(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    batches = max(1, min(len(loader), max_batches or len(loader)))
    return total_loss / batches


@torch.no_grad()
def evaluate_loop(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_labels = []
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        logits = model(features)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    if not all_logits:
        return {"val_loss": 0.0, "val_acc": 0.0}
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)
    acc = (preds == all_labels).mean()
    # compute bce loss for reporting
    bce = -(
        all_labels * np.log(np.clip(probs, 1e-8, 1 - 1e-8))
        + (1 - all_labels) * np.log(np.clip(1 - probs, 1e-8, 1 - 1e-8))
    )
    val_loss = float(bce.mean())
    return {"val_loss": val_loss, "val_acc": float(acc)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--normal", required=True)
    p.add_argument("--attacks", nargs="+", required=True)
    p.add_argument("--model_out", default="models/pytorch_lstm_arp.pt")
    p.add_argument("--scaler_out", default="models/pytorch_scaler_encoders.pkl")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--hidden", nargs="+", type=int, default=[64, 32])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    args = p.parse_args()

    # load CSVs
    normal_df = pd.read_csv(args.normal)
    normal_df["label"] = 0
    attack_frames = []
    for atk in args.attacks:
        df_atk = pd.read_csv(atk)
        df_atk["label"] = 1
        attack_frames.append(df_atk)
    df = pd.concat([normal_df] + attack_frames, ignore_index=True)
    print(f"[*] Combined dataset shape: {df.shape}")

    feature_cols = ["op", "sender_ip", "target_ip", "sender_mac", "target_mac"]
    df, encoders = preprocess_packet_df(df, feature_cols)

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_seq, y_seq = create_sequences(X, y, seq_len=args.seq_len)
    print(f"[*] Sequence data shape: {X_seq.shape}, {y_seq.shape}")

    # train/val split
    split_idx = int(len(X_seq) * (1 - args.val_split))
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_batch)

    model = LSTMClassifier(input_size=X_seq.shape[2], hidden_units=tuple(args.hidden), dropout=args.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    epochs_without_improve = 0
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_loop(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate_loop(model, val_loader, device)
        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["val_acc"]
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # early stopping on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improve = 0
            # save best weights
            torch.save(model.state_dict(), args.model_out)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print("Early stopping")
                break

    # save scaler & encoders
    os.makedirs(os.path.dirname(args.scaler_out) or ".", exist_ok=True)
    joblib.dump({"scaler": scaler, "encoders": encoders}, args.scaler_out)
    print(f"[+] Model (best) saved to {args.model_out}")
    print(f"[+] Scaler & encoders saved to {args.scaler_out}")


if __name__ == "__main__":
    main()

