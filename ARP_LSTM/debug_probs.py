import pandas as pd
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append("src")
from arp_detector.config import load_config
from arp_detector.utils.io import load_joblib, load_json
from arp_detector.models.supervised import SequenceClassifier

# Load Config & Data
config = load_config(Path("./eval_override.yaml"))
df = pd.read_parquet("data/processed/attack.parquet")
manifest = load_json(config.paths.manifest_path)
cols = manifest["feature_columns"]

# Filter for attack windows
attack_df = df[df["attack"] == 1]
print(f"Attack windows found: {len(attack_df)}")

# Load Scaler & Model
scaler = load_joblib(config.paths.scaler_path)
model_path = config.paths.supervised_model_path
state = torch.load(model_path, map_location="cpu")
if "state_dict" in state: state = state["state_dict"]

model = SequenceClassifier(len(cols), 2, config.model.supervised)
model.load_state_dict(state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
model.eval()

# Transform features
features = attack_df[cols].to_numpy()
features = scaler.transform(features)
features = torch.tensor(features, dtype=torch.float32).unsqueeze(1).to(device)

# Inference
with torch.no_grad():
    logits = model(features).window_logits
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

print(f"Probabilities (Min/Max/Mean): {probs.min():.6f} / {probs.max():.6f} / {probs.mean():.6f}")
print("Sample probabilities:", probs[:20])
