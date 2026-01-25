import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("models/lstm_model.h5")
scaler = joblib.load("models/scaler.joblib")

# Load CSV
df = pd.read_csv("data/features/attack.csv")

# Keep a copy of sender_mac_int for printing
macs = df["sender_mac_int"].copy()

# Drop only the label column
if "label" in df.columns:
    df = df.drop(columns=["label"])

print(f"Number of features for scaling: {df.shape[1]}")  # should be 12

# Scale
X_scaled = scaler.transform(df.values).reshape((len(df), 1, df.shape[1]))

# Predict
preds = model.predict(X_scaled)

# Show MACs with predictions
for mac_int, prob in zip(macs, preds):
    if prob[0] >= 0.5:  # threshold for suspicious
        mac_hex = ":".join(f"{(mac_int >> ele) & 0xff:02x}" for ele in [40,32,24,16,8,0])
        print(f"Suspicious MAC: {mac_hex} (Prob: {prob[0]:.2f})")
