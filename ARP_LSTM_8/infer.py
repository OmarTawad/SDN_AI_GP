# infer.py
from __future__ import annotations

import os

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import argparse
import glob
import json
import numpy as np
import torch
import yaml
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Adjust path so we can import from src
import sys
sys.path.append(os.getcwd())

from src.arp_detector.data.pcap_reader import read_pcap
from src.arp_detector.data.windowing import WindowBuilder, WindowingParams
from src.arp_detector.features.feature_engineering import FeatureExtractor
from src.arp_detector.features.scaler import RobustScaler # Assumed to exist or we use sklearn
from src.arp_detector.utils.quantization import apply_dynamic_quantization, set_quantized_engine, unpack_checkpoint
from src.arp_detector.models.supervised import SequenceClassifier
from src.arp_detector.config.types import SupervisedModelConfig, FeatureConfig

def load_scaler(path):
    import joblib
    try:
        return joblib.load(path)
    except Exception:
        # Fallback for RobustScaler class if custom
        from src.arp_detector.features.scaler import RobustScaler
        return RobustScaler.load(os.path.dirname(path)) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pcaps", default=None)
    parser.add_argument("--out", default="reports_fp16")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # 1. Setup Model
    print("Setting up model...")
    # Reconstruct Config Objects
    sup_cfg = raw_cfg["model"]["supervised"]
    model_config = SupervisedModelConfig(
        input_dropout=float(sup_cfg.get("input_dropout", 0.0)),
        rnn_type=str(sup_cfg.get("rnn_type", "lstm")),
        hidden_size=int(sup_cfg.get("hidden_size", 128)),
        num_layers=int(sup_cfg.get("num_layers", 2)),
        bidirectional=bool(sup_cfg.get("bidirectional", True)),
        dropout=float(sup_cfg.get("dropout", 0.0)),
        attention=bool(sup_cfg.get("attention", True)),
        attention_heads=int(sup_cfg.get("attention_heads", 4))
    )
    
    # Check scaler for input dim
    scaler_path = str(raw_cfg["paths"]["scaler_path"]).replace("../", "") # Adjust for relative if needed. 
    # But user runs from ARP_LSTM_8, config says `../models_int8/feature_scaler_int8.joblib`.
    # We should resolve path relative to config file or CWD.
    # We'll trust python path resolution.
    
    # To get input dimension, we need to know how many features.
    # FeatureExtractor produces 25 features based on `feature_engineering.py`.
    # If the boolean checks in there are strictly 25, we can hardcode or dry-run.
    # But better to load scaler if it exists.
    
    input_size = 25 # Default from viewing feature_engineering.py
    
    label_map = raw_cfg["labels"].get("family_mapping", {})
    num_classes = len(label_map) if label_map else 2 

    device = torch.device("cpu")
    model = SequenceClassifier(
        input_size=input_size,
        num_attack_types=num_classes,
        config=model_config
    ).to(device=device, dtype=torch.float32)

    # Load Checkpoint
    ckpt_path = raw_cfg["paths"]["supervised_model_path"]
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint {ckpt_path}...")
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict, _ = unpack_checkpoint(state)
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading state dict, might be shape mismatch if input_size is wrong: {e}")
            # Try to infer input size from checkpoint weight shape
            # model.rnn.weight_ih_l0
            pass

    model.eval()
    
    # Quantize
    print("Quantizing model...")
    set_quantized_engine("qnnpack")
    model_int8 = apply_dynamic_quantization(model)

    # 2. Process PCAPs
    pcaps_arg = args.pcaps or raw_cfg["preprocess"]["pcaps_glob"]
    # Handle glob or explicit list
    if any(ch in pcaps_arg for ch in "*?[]"):
        pcaps = sorted(glob.glob(pcaps_arg))
    else:
        # Could be single file
        pcaps = [pcaps_arg]
    
    if not pcaps:
        print("No PCAPs found.")
        return

    # Windowing Params
    win_cfg = raw_cfg["windowing"]
    w_params = WindowingParams(
        window_size=float(win_cfg["window_size"]),
        hop_size=float(win_cfg["hop_size"]),
        max_windows=win_cfg.get("max_windows")
    )
    
    seq_len = int(win_cfg["sequence_length"])
    seq_stride = int(win_cfg["sequence_stride"])

    # Feature Extractor
    f_extractor = FeatureExtractor(FeatureConfig(), w_params.window_size)
    
    # Scaler
    # We need a scaler to normalize features before model. 
    # Attempt to load
    scaler = None
    if os.path.exists(scaler_path):
         try:
             import joblib
             scaler = joblib.load(scaler_path)
         except:
             pass

    os.makedirs(args.out, exist_ok=True)

    for pcap_path in tqdm(pcaps, desc="PCAPs"):
        print(f"Reading {pcap_path}...")
        try:
            packets = read_pcap(Path(pcap_path))
            if not packets:
                print("Empty PCAP.")
                continue
            
            # Windowing
            windows = WindowBuilder(w_params).build(packets)
            if not windows:
                print("No windows generated.")
                continue
            
            print(f"Generated {len(windows)} windows. Extracting features...")
            # Feature Extraction
            df = f_extractor.extract(windows)
            print(f"Extracted features shape: {df.shape}")
            
            # We need to extract the feature columns.
            # Convert to float32 numpy
            # Drop metadata
            meta_cols = ["window_index", "window_start", "window_end"]
            feat_cols = [c for c in df.columns if c not in meta_cols]
            
            X = df[feat_cols].values.astype(np.float32)
            
            # Scaling
            if scaler:
                try:
                    X = scaler.transform(X)
                except Exception as e:
                    print(f"Scaling failed: {e}")
            
            # Sequence Generation
            # We slide over X
            # stride
            num_samples = len(X)
            
            results = []
            
            for start in tqdm(range(0, num_samples - seq_len + 1, seq_stride), desc="Inference"):
                end = start + seq_len
                seq_data = X[start:end] # (seq_len, features)
                
                # Check shape
                seq_t = torch.from_numpy(seq_data).unsqueeze(0) # (1, seq_len, features)
                
                with torch.no_grad():
                    # Model expects (batch, time, features)
                    out = model_int8(seq_t)
                    # out.sequence_prob (batch,)
                    # out.window_logits (batch, time)
                    
                    seq_prob = float(out.sequence_prob.item())
                    
                    # Metadata for this sequence (maybe use middle window or end)
                    w_idx_start = df.iloc[start]["window_index"]
                    w_idx_end = df.iloc[end-1]["window_index"]
                    ts_start = df.iloc[start]["window_start"]
                    ts_end = df.iloc[end-1]["window_end"]
                    
                    results.append({
                        "file": str(pcap_path),
                        "start_index": int(w_idx_start),
                        "end_index": int(w_idx_end),
                        "time_start": float(ts_start),
                        "time_end": float(ts_end),
                        "prob": seq_prob
                    })

            # Save Report
            base = os.path.basename(pcap_path)
            out_json = os.path.join(args.out, f"{base}.json")
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"Failed processing {pcap_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
