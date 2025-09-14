from __future__ import annotations
import argparse, glob, json, os
from typing import List

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from data.dataset import WindowDataset, LabelProvider
from features.scaler import RobustScaler
from models.head import Detector

def _load_artifacts(save_dir: str):
    with open(os.path.join(save_dir, "feature_model_meta.json"), "r") as f:
        meta = json.load(f)
    with open(os.path.join(save_dir, "calibration.json"), "r") as f:
        calib = json.load(f)
    scaler = RobustScaler()
    scaler.load(save_dir)
    model = Detector(
        seq_in_dim=meta["seq_in_dim"],
        static_dim=meta["static_dim"],
        channels=tuple(meta["channels"]),
        k=meta["kernel_size"],
        drop=meta["dropout"],
        heads=meta["attention_heads"],
        mlp_hidden=tuple(meta["mlp_hidden"]),
        aux_family_head=bool(meta.get("aux_family_head", True)),
        n_families=6,
    )
    state = torch.load(os.path.join(save_dir, "model_best.pt"), map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model, scaler, meta, calib

def permutation_importance(model, scaler, ds: WindowDataset, n_samples: int = 200) -> List[tuple]:
    # sample windows
    idxs = np.random.RandomState(1337).choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    # collect baseline
    Xs, Xstat, y = [], [], []
    for i in idxs:
        seq, static, yy, _, _, _ = ds[i]
        Xs.append(seq.numpy())
        Xstat.append(static.numpy())
        y.append(yy.item())
    Xs = np.stack(Xs, axis=0)
    Xstat = np.stack(Xstat, axis=0)
    y = np.array(y, dtype=float).reshape(-1,1)

    # scale static
    names = [f"f_{i}" for i in range(Xstat.shape[1])]
    Xstat_scaled = scaler.transform(Xstat, names)
    with torch.no_grad():
        out = model(torch.from_numpy(Xs).float(), torch.from_numpy(Xstat_scaled).float())
        base = torch.sigmoid(out["logits"]).numpy()

    importances = []
    for j in range(Xstat.shape[1]):
        Xp = Xstat_scaled.copy()
        np.random.shuffle(Xp[:, j])
        with torch.no_grad():
            out = model(torch.from_numpy(Xs).float(), torch.from_numpy(Xp).float())
            pert = torch.sigmoid(out["logits"]).numpy()
        delta = float(np.mean(np.abs(base - pert)))
        importances.append((j, delta))
    importances.sort(key=lambda t: t[1], reverse=True)
    return importances

def save_attention_heatmaps(model, scaler, ds: WindowDataset, out_dir: str, top_n: int = 10):
    os.makedirs(out_dir, exist_ok=True)
    # score all and pick top-N by prob
    scores = []
    with torch.no_grad():
        for i in range(len(ds)):
            seq, static, _, _, _, base = ds[i]
            stat_scaled = scaler.transform(static.numpy().reshape(1,-1), [f"f_{k}" for k in range(static.numel())])
            out = model(seq.unsqueeze(0), torch.from_numpy(stat_scaled).float())
            prob = float(torch.sigmoid(out["logits"]).item())
            attn = out["attn"].numpy().ravel()
            scores.append((prob, i, attn, base))
    scores.sort(key=lambda t: t[0], reverse=True)
    for rank, (prob, i, attn, base) in enumerate(scores[:top_n], start=1):
        plt.figure()
        plt.bar(np.arange(attn.size), attn)
        plt.xlabel("Micro-bin")
        plt.ylabel("Attention weight")
        plt.title(f"{base} | prob={prob:.3f}")
        path = os.path.join(out_dir, f"attn_{rank:02d}_{os.path.splitext(base)[0]}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pcaps", required=True, help='Glob like "samples/*.pcap"')
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="reports/explain")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model, scaler, meta, calib = _load_artifacts(cfg["logging"]["save_dir"])

    files = sorted(glob.glob(args.pcaps))
    labels = LabelProvider(args.labels)
    ds = WindowDataset(
        files, labels,
        window_sec=cfg["windowing"]["window_sec"],
        stride_sec=cfg["windowing"]["stride_sec"],
        micro_bins=cfg["windowing"]["micro_bins"],
        top_k_udp_ports=cfg["data"]["top_k_udp_ports"],
        augment=None,
        ssdp_multicast_v4=cfg["features"]["ssdp_multicast_ipv4"],
        ssdp_multicast_v6=cfg["features"]["ssdp_multicast_ipv6"],
        for_training=False
    )
    imps = permutation_importance(model, scaler, ds, n_samples=200)
    with open(os.path.join(args.out, "perm_importance.json"), "w") as f:
        json.dump([{"feature_index": i, "delta_prob": float(d)} for i,d in imps], f, indent=2)

    save_attention_heatmaps(model, scaler, ds, args.out, top_n=10)
    print("Saved permutation importances and attention heatmaps to", args.out)

if __name__ == "__main__":
    main()

