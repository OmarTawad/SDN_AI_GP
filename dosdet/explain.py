from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import argparse
import glob
import json
from typing import List, Tuple

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from data.dataset import WindowDataset, LabelProvider
from features.scaler import RobustScaler
from features.feature_slimming import StaticSlimmer
from models.dws_cnn import FastDetector

torch.set_num_threads(min(2, max(1, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except AttributeError:
    pass


def _load_artifacts(cfg: dict) -> Tuple[FastDetector, RobustScaler, StaticSlimmer, dict]:
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    meta_path = os.path.join(artifacts_dir, "feature_model_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    scaler = RobustScaler.load(artifacts_dir)

    slimmer = StaticSlimmer(out_dim=int(meta.get("static_dim", 1)))
    slimmer.load(artifacts_dir)

    trn = cfg["training"]
    model = FastDetector(
        seq_in_dim=int(meta["seq_in_dim"]),
        static_dim=int(meta["static_dim"]),
        channels=tuple(meta.get("channels", trn["channels"])),
        k=int(meta.get("kernel_size", trn["kernel_size"])),
        drop=float(meta.get("dropout", trn["dropout"])),
        mlp_hidden=tuple(meta.get("mlp_hidden", trn["mlp_hidden"])),
        aux_family_head=bool(meta.get("aux_family_head", trn.get("aux_family_head", False))),
        n_families=int(meta.get("n_families", 6)),
    )
    state = torch.load(os.path.join(artifacts_dir, "model_best.pt"), map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model, scaler, slimmer, meta


def _transform_static(scaler: RobustScaler, slimmer: StaticSlimmer, mat: np.ndarray) -> np.ndarray:
    feature_names = getattr(scaler, "feature_names_", None) or getattr(slimmer, "src_names", None)
    if feature_names is None or len(feature_names) != mat.shape[1]:
        feature_names = [f"f_{i}" for i in range(mat.shape[1])]
    scaled = scaler.transform(mat, feature_names)
    slim = slimmer.transform(scaled)
    return slim


def permutation_importance(
    model: FastDetector,
    scaler: RobustScaler,
    slimmer: StaticSlimmer,
    ds: WindowDataset,
    n_samples: int = 200,
) -> List[Tuple[int, float]]:
    device = next(model.parameters()).device
    rng = np.random.default_rng(1337)
    sample_idx = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    seq_list, stat_list, y_list = [], [], []
    for idx in sample_idx:
        seq, static, target, *_ = ds[idx]
        seq_list.append(seq.numpy())
        stat_list.append(static.numpy())
        y_list.append(target.item())

    seq_arr = np.stack(seq_list, axis=0)
    stat_arr = np.stack(stat_list, axis=0)

    stat_slim = _transform_static(scaler, slimmer, stat_arr)
    seq_tensor = torch.from_numpy(seq_arr).to(device).float()
    stat_tensor = torch.from_numpy(stat_slim).to(device).float()

    with torch.no_grad():
        logits = model(seq_tensor, stat_tensor)["logits"].squeeze(1)
        base_probs = torch.sigmoid(logits).cpu().numpy()

    importances: List[Tuple[int, float]] = []
    for j in range(stat_slim.shape[1]):
        perturbed = stat_slim.copy()
        rng.shuffle(perturbed[:, j])
        pert_tensor = torch.from_numpy(perturbed).to(device).float()
        with torch.no_grad():
            pert_logits = model(seq_tensor, pert_tensor)["logits"].squeeze(1)
            pert_probs = torch.sigmoid(pert_logits).cpu().numpy()
        delta = float(np.mean(np.abs(base_probs - pert_probs)))
        importances.append((j, delta))
    importances.sort(key=lambda tup: tup[1], reverse=True)
    return importances


def save_attention_heatmaps(
    model: FastDetector,
    scaler: RobustScaler,
    slimmer: StaticSlimmer,
    ds: WindowDataset,
    out_dir: str,
    top_n: int = 10,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device

    scores = []
    with torch.no_grad():
        for idx in range(len(ds)):
            seq, static, *_ , name = ds[idx]
            stat_np = static.numpy().reshape(1, -1)
            stat_slim = _transform_static(scaler, slimmer, stat_np)
            seq_t = seq.unsqueeze(0).to(device)
            stat_t = torch.from_numpy(stat_slim).to(device).float()
            out = model(seq_t, stat_t)
            prob = float(torch.sigmoid(out["logits"]).item())
            attn = out.get("attn")
            if attn is not None:
                attn_vec = attn.cpu().numpy().ravel()
            else:
                attn_vec = np.zeros(seq.shape[0], dtype=np.float32)
            scores.append((prob, attn_vec, name))

    scores.sort(key=lambda tup: tup[0], reverse=True)
    for rank, (prob, attn, name) in enumerate(scores[:top_n], start=1):
        plt.figure(figsize=(8, 3))
        plt.bar(np.arange(attn.size), attn)
        plt.xlabel("Micro-bin index")
        plt.ylabel("Attention weight")
        plt.title(f"{name} | prob={prob:.3f}")
        fname = os.path.join(out_dir, f"attn_{rank:02d}_{os.path.splitext(name)[0]}.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation importance + attention visualisations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pcaps", default=None, help='Glob like "samples/*.pcap" (defaults to config preprocess.pcaps_glob)')
    parser.add_argument("--labels", default=None, help="labels.csv (defaults to config preprocess.labels_csv)")
    parser.add_argument("--out", default=None, help="Output directory (defaults to paths.reports_dir/explain)")
    parser.add_argument("--n-samples", type=int, default=200, help="Windows to sample for permutation importance")
    parser.add_argument("--top-n", type=int, default=10, help="Top windows for attention plots")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model, scaler, slimmer, meta = _load_artifacts(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pcap_glob = args.pcaps or cfg["preprocess"]["pcaps_glob"]
    files = sorted(glob.glob(pcap_glob)) if any(ch in pcap_glob for ch in "*?[]") else [pcap_glob]
    if not files:
        raise FileNotFoundError(f"No pcaps matched pattern {pcap_glob}")

    labels_path = args.labels or cfg["preprocess"]["labels_csv"]
    labels = LabelProvider(labels_path)
    ds = WindowDataset(
        pcap_paths=files,
        labels=labels,
        window_sec=cfg["windowing"]["window_sec"],
        stride_sec=cfg["windowing"]["stride_sec"],
        micro_bins=meta.get("micro_bins", cfg["windowing"]["micro_bins"]),
        top_k_udp_ports=meta.get("top_k_udp_ports", cfg["data"]["top_k_udp_ports"]),
        augment=None,
        ssdp_multicast_v4=cfg["features"]["ssdp_multicast_ipv4"],
        ssdp_multicast_v6=cfg["features"]["ssdp_multicast_ipv6"],
        for_training=False,
    )

    out_dir = args.out or os.path.join(cfg["paths"]["reports_dir"], "explain")
    os.makedirs(out_dir, exist_ok=True)
    feature_imps = permutation_importance(model, scaler, slimmer, ds, n_samples=args.n_samples)
    with open(os.path.join(out_dir, "perm_importance.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {"component": int(idx), "delta_prob": float(delta)}
                for idx, delta in feature_imps
            ],
            f,
            indent=2,
        )

    save_attention_heatmaps(model, scaler, slimmer, ds, out_dir, top_n=args.top_n)
    print(f"Saved permutation importances and attention heatmaps under {out_dir}")


if __name__ == "__main__":
    main()
