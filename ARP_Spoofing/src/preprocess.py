import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.preprocessing import StandardScaler

def preprocess(csv_glob, output_dir="data/processed", scaler_path="models/scaler.joblib"):
    os.makedirs(output_dir, exist_ok=True)

    # Get all CSV files
    all_files = glob.glob(csv_glob)
    if not all_files:
        raise FileNotFoundError(f"No CSV files found for pattern: {csv_glob}")

    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        # Ensure label exists
        if "label" not in df.columns:
            if "attack" in f or "pure_attack" in f:
                df["label"] = 1
            else:
                df["label"] = 0
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print(f"[+] Loaded {len(dfs)} CSVs, total rows: {len(data)}")

    # Separate features and labels
    X = data.drop(columns=["label"], errors='ignore')
    y = data["label"].values

    # Fit scaler and save
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)
    print(f"[+] Scaler saved to {scaler_path}")

    # Split train/test 80/20
    split = int(0.8 * len(X_scaled))
    np.savez(os.path.join(output_dir, "train.npz"), X=X_scaled[:split], y=y[:split])
    np.savez(os.path.join(output_dir, "test.npz"), X=X_scaled[split:], y=y[split:])
    print(f"[+] Preprocessed datasets saved in {output_dir}")
    print(f"[+] Train size: {split}, Test size: {len(X_scaled)-split}")

if __name__ == "__main__":
    # Default pattern: all CSVs in features dir
    preprocess("data/features/*.csv")
