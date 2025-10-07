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
    scaler = joblib.load("models/scaler.joblib")
    X_scaled = scaler.transform(X)

    model = load_model("models/lstm_model.h5")
    preds = model.predict(X_scaled, verbose=1)
    df["pred"] = (preds > 0.5).astype(int)

    for _, row in df.head(20).iterrows():
        ts = row.get("ts", None)
        label = row["label"]
        pred = row["pred"]
        print(f"ts={ts}, label={label}, pred={pred}")

    print("\nTotal attack packets predicted:", df["pred"].sum())
    print("Total normal packets predicted:", len(df) - df["pred"].sum())
