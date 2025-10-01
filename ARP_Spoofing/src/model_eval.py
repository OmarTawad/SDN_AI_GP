import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True, help="Directory with feature CSVs")
    args = parser.parse_args()

    csv_files = glob.glob(f"{args.features_dir}/*.csv")
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    X = df.drop(columns=["label", "ts"], errors='ignore')
    y = df["label"].values

    scaler = joblib.load("models/scaler.joblib")
    X_scaled = scaler.transform(X)

    model = load_model("models/lstm_model.h5")
    preds = model.predict(X_scaled, verbose=1)
    preds_class = (preds > 0.5).astype(int)

    df["pred"] = preds_class
    print(df[["ts", "label", "pred"]].head(20))
    print("\nTotal attack packets predicted:", df["pred"].sum())
    print("Total normal packets predicted:", len(df) - df["pred"].sum())
