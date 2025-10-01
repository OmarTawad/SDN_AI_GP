import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm_model(input_shape, hidden_units=[64, 32], dropout=0.3, lr=0.001):
    from tensorflow.keras.optimizers import Adam
    model = Sequential()
    # Masking layer in case of padded sequences
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    # First LSTM layer
    model.add(LSTM(hidden_units[0], return_sequences=True))
    model.add(Dropout(dropout))
    # Second LSTM (optional)
    if len(hidden_units) > 1:
        model.add(LSTM(hidden_units[1]))
        model.add(Dropout(dropout))
    # Output
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=lr),
                  metrics=['accuracy'])
    return model

def preprocess_packet_df(df, feature_cols):
    # Encode categorical fields (MAC/IP)
    encoders = {}
    for col in feature_cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders

def create_sequences(X, y, seq_len=10):
    """Convert per-packet data into sequences for LSTM."""
    sequences = []
    labels = []
    for i in range(len(X) - seq_len + 1):
        sequences.append(X[i:i+seq_len])
        labels.append(y[i+seq_len-1])  # label for last packet in sequence
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int32)

def train(args):
    print("[*] Loading CSVs...")
    normal_df = pd.read_csv(args.normal)
    normal_df['label'] = 0

    attack_dfs = []
    for attack_file in args.attacks:
        df = pd.read_csv(attack_file)
        df['label'] = 1
        attack_dfs.append(df)

    df = pd.concat([normal_df] + attack_dfs, ignore_index=True)
    print(f"[*] Combined dataset shape: {df.shape}")

    feature_cols = ['op', 'sender_ip', 'target_ip', 'sender_mac', 'target_mac']
    df, encoders = preprocess_packet_df(df, feature_cols)

    X = df[feature_cols].values.astype(float)
    y = df['label'].values.astype(int)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Create sequences
    X_seq, y_seq = create_sequences(X, y, seq_len=args.seq_len)
    print(f"[*] Sequence data shape: {X_seq.shape}, {y_seq.shape}")

    # Train/val split
    split_idx = int(len(X_seq) * (1 - args.val_split))
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    # Build model
    model = build_lstm_model(
        input_shape=(X_seq.shape[1], X_seq.shape[2]),
        hidden_units=args.hidden,
        dropout=args.dropout,
        lr=args.lr
    )

    callbacks = [EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)]

    print("[*] Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)
    print(f"[+] Model saved to {args.model_out}")

    # Save scaler + encoders
    os.makedirs(os.path.dirname(args.scaler_out), exist_ok=True)
    joblib.dump({'scaler': scaler, 'encoders': encoders}, args.scaler_out)
    print(f"[+] Scaler & encoders saved to {args.scaler_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", required=True, help="Normal CSV path")
    parser.add_argument("--attacks", nargs='+', required=True, help="Attack CSV paths")
    parser.add_argument("--model_out", default="models/lstm_arp_detector.h5")
    parser.add_argument("--scaler_out", default="models/lstm_scaler.pkl")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--hidden", nargs='+', type=int, default=[64,32])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seq_len", type=int, default=10, help="Number of packets per LSTM sequence")

    args = parser.parse_args()
    train(args)
