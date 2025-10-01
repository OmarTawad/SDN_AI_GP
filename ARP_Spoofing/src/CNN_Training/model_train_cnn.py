#!/usr/bin/env python3
"""
model_train_cnn.py — train a CNN to detect ARP spoofing.

Usage example:
python src/model_train_cnn.py \
    --normal data/features/normal.csv \
    --attack data/features/attack.csv data/features/pure_attack.csv data/features/pure_attack2.csv \
    --epochs 50 \
    --batch-size 64 \
    --timesteps 5 \
    --model-out models/cnn/cnn_arp_detector.h5 \
    --scaler-out models/cnn/cnn_scaler.pkl
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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
                df[intcol] = df[col].astype(str).fillna("0.0.0.0").apply(lambda ip: sum(int(p) << (8*(3-i)) for i,p in enumerate(ip.split('.'))))
            else:
                df[intcol] = 0
    if 'op' in df.columns:
        df['op_is_request'] = (df['op']==1).astype(int)
        df['op_is_reply'] = (df['op']==2).astype(int)
    else:
        df['op_is_request'] = 0
        df['op_is_reply'] = 0
    return df

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

def load_and_preprocess(csv_files, label):
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df = convert_mac_ip(df)
        df['label'] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ---------------- CNN model ----------------
def build_cnn(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='valid', input_shape=input_shape),
        Dropout(0.2),
        Conv1D(32, kernel_size=3, activation='relu', padding='valid'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Train CNN ARP spoofing detector")
    parser.add_argument("--normal", required=True, help="Normal CSV file")
    parser.add_argument("--attack", nargs='+', required=True, help="Attack CSV files")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=5)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--scaler-out", required=True)
    args = parser.parse_args()

    # Load data
    print("[*] Loading normal data...")
    normal_df = load_and_preprocess([args.normal], 0)
    print("[*] Loading attack data...")
    attack_df = load_and_preprocess(args.attack, 1)

    df = pd.concat([normal_df, attack_df], ignore_index=True)
    features = ['op_is_request','sender_ip_int','target_ip_int','sender_mac_int','target_mac_int']
    X = df[features].values.astype(np.float32)
    y = df['label'].values.astype(np.float32)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, args.scaler_out)
    print(f"[+] Saved scaler to {args.scaler_out}")

    # Build sequences
    X_seq = build_rolling_sequences(X_scaled, args.timesteps)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y, test_size=0.2, random_state=42, stratify=y)

    # Build model
    model = build_cnn(input_shape=(X_train.shape[1], X_train.shape[2]))
    print(model.summary())

    # Train
    print("[*] Training CNN...")
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=args.epochs, batch_size=args.batch_size, callbacks=[es], verbose=2)

    # Save model
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)
    print(f"[+] Saved model to {args.model_out}")

if __name__ == "__main__":
    main()
