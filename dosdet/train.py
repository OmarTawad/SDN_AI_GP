from __future__ import annotations
import argparse
import glob
import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import yaml

from data.dataset import WindowDataset, LabelProvider, ATTACK_FAMILIES
from features.scaler import RobustScaler
from models.head import Detector
from models.utils import seed_everything, get_device, FocalLoss, EarlyStopping, pr_auc
from models.calibrate import temperature_scale, apply_temperature, find_threshold, save_calibration

def split_by_file(files: List[str], ratios=(0.7, 0.15, 0.15), seed=1337) -> Tuple[List[str], List[str], List[str]]:
    rng = np.random.default_rng(seed)
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = files[:n_train]
    val = files[n_train:n_train+n_val]
    test = files[n_train+n_val:]
    return train, val, test

def build_loaders(cfg, labels_csv: str, all_pcaps_glob: str, for_scaler_fit: bool = True):
    files = sorted(glob.glob(all_pcaps_glob))
    assert files, f"No pcaps matched: {all_pcaps_glob}"

    train_f, val_f, test_f = split_by_file(files, tuple(cfg["split"]["train_val_test"]), seed=cfg["seed"])
    labels = LabelProvider(labels_csv)

    aug = {
        "time_warp_pct": float(cfg["augment"]["time_warp_pct"]),
        "benign_overlay_prob": float(cfg["augment"]["benign_overlay_prob"]),
        "header_randomize_prob": float(cfg["augment"]["header_randomize_prob"]),
    }

    def mk(files, training):
        return WindowDataset(
            pcap_paths=files,
            labels=labels,
            window_sec=cfg["windowing"]["window_sec"],
            stride_sec=cfg["windowing"]["stride_sec"],
            micro_bins=cfg["windowing"]["micro_bins"],
            top_k_udp_ports=cfg["data"]["top_k_udp_ports"],
            augment=aug,
            ssdp_multicast_v4=cfg["features"]["ssdp_multicast_ipv4"],
            ssdp_multicast_v6=cfg["features"]["ssdp_multicast_ipv6"],
            for_training=training,
        )

    ds_train = mk(train_f, True)
    ds_val = mk(val_f, False)
    ds_test = mk(test_f, False)

    # Balanced sampler over windows (approx) using labels from dataset
    y_train = []
    for i in range(len(ds_train)):
        _, _, y, _, _, _ = ds_train[i]
        y_train.append(int(y.item() >= 0.5))
    # avoid zero-division
    n_pos = max(1, sum(y_train))
    n_neg = max(1, len(y_train) - n_pos)
    w = [len(y_train)/(2*n_pos) if yi==1 else len(y_train)/(2*n_neg) for yi in y_train]
    sampler = WeightedRandomSampler(w, num_samples=len(y_train), replacement=True)

    def collate(batch):
        seq = torch.stack([b[0] for b in batch], dim=0)
        static = torch.stack([b[1] for b in batch], dim=0)
        y = torch.stack([b[2] for b in batch], dim=0)
        fam = torch.stack([b[3] for b in batch], dim=0).squeeze(1)
        ts = torch.stack([b[4] for b in batch], dim=0)
        names = [b[5] for b in batch]
        return seq, static, y, fam, ts, names

    train_loader = DataLoader(ds_train, batch_size=cfg["training"]["batch_size"], sampler=sampler, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(ds_val, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(ds_test, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0, collate_fn=collate)

    return train_loader, val_loader, test_loader, ds_train, ds_val, ds_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pcaps", required=True, help='Glob like "samples/*.pcap"')
    ap.add_argument("--labels", required=True, help="Path to labels.csv")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device","auto"))

    # Build datasets and loaders
    train_loader, val_loader, test_loader, ds_train, ds_val, ds_test = build_loaders(cfg, args.labels, args.pcaps)

    # ===== Fit scaler on STATIC features only (robust scaling for numeric features) =====
    # We will gather a pass of static features from train set to fit scaler and then transform on the fly in the loop.
    static_stack = []
    name_ref = None
    for i in range(len(ds_train)):
        _, static, _, _, _, _ = ds_train[i]
        static_stack.append(static.numpy())
        if name_ref is None:
            # recompute names via compute_static_features — stored not returned directly here; we approximate by length check.
            name_ref = [f"f_{j}" for j in range(static.numel())]
    static_stack = np.stack(static_stack, axis=0)
    scaler = RobustScaler()
    scaler.fit(static_stack, name_ref)

    # ===== Build model =====
    seq_example, static_example, _, _, _, _ = ds_train[0]
    M, K_seq = seq_example.shape
    K_static = static_example.numel()
    md = cfg["training"]
    model = Detector(
        seq_in_dim=K_seq,
        static_dim=K_static,
        channels=tuple(md["channels"]),
        k=md["kernel_size"],
        drop=md["dropout"],
        heads=md["attention_heads"],
        mlp_hidden=tuple(md["mlp_hidden"]),
        aux_family_head=bool(md.get("aux_family_head", True)),
        n_families=len(ATTACK_FAMILIES),
    ).to(device)

    # Loss & optimizer
    if cfg["training"]["loss"] == "focal":
        criterion = FocalLoss(gamma=cfg["training"]["focal_gamma"], alpha_pos=cfg["training"]["class_weight_pos"])
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg["training"]["class_weight_pos"]], device=device))

    aux_ce = nn.CrossEntropyLoss(ignore_index=-1)
    opt = torch.optim.AdamW(model.parameters(), lr=md["lr"], weight_decay=md["weight_decay"])
    early = EarlyStopping(patience=md["early_stop_patience"], mode="max")
    grad_clip = float(md["grad_clip"])
    save_dir = cfg["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # ===== Training loop =====
    best_state = None
    for epoch in range(1, md["epochs"] + 1):
        model.train()
        tr_losses = []
        for seq, static, y, fam, ts, names in train_loader:
            seq = seq.to(device)
            # scale static via fitted scaler (on CPU numpy, then back)
            static_np = scaler.transform(static.numpy(), name_ref)
            static_t = torch.from_numpy(static_np).to(device).float()

            y = y.to(device)
            fam = fam.to(device)

            out = model(seq, static_t)
            logits = out["logits"]
            loss = criterion(logits, y)
            if md.get("aux_family_head", True) and "family_logits" in out:
                loss = loss + 0.3 * aux_ce(out["family_logits"], fam.squeeze(1))

            opt.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_losses.append(loss.item())

        # Validation
        model.eval()
        v_logits, v_targets = [], []
        with torch.no_grad():
            for seq, static, y, fam, ts, names in val_loader:
                seq = seq.to(device)
                static_np = scaler.transform(static.numpy(), name_ref)
                static_t = torch.from_numpy(static_np).to(device).float()
                y = y.to(device)
                out = model(seq, static_t)
                v_logits.append(out["logits"].cpu().numpy())
                v_targets.append(y.cpu().numpy())
        v_logits = np.vstack(v_logits).astype(np.float64)
        v_targets = np.vstack(v_targets).astype(np.float64)
        v_probs = 1 / (1 + np.exp(-v_logits))
        pr = pr_auc(v_targets.ravel(), v_probs.ravel())

        print(f"Epoch {epoch:02d} | train_loss={np.mean(tr_losses):.4f} | val_pr_auc={pr:.4f}")

        improved = early.step(pr)
        if improved:
            best_state = {
                "model": model.state_dict(),
                "val_pr_auc": pr,
                "epoch": epoch,
            }
            torch.save(best_state, os.path.join(save_dir, "model_best.pt"))

        if early.stopped:
            print("Early stopping triggered.")
            break

    assert best_state is not None, "Training did not produce a best state"

    # ===== Calibration on validation set =====
    # recompute val logits with best model
    model.load_state_dict(torch.load(os.path.join(save_dir, "model_best.pt"))["model"])
    model.eval()
    v_logits, v_targets = [], []
    with torch.no_grad():
        for seq, static, y, fam, ts, names in val_loader:
            seq = seq.to(device)
            static_np = scaler.transform(static.numpy(), name_ref)
            static_t = torch.from_numpy(static_np).to(device).float()
            y = y.to(device)
            out = model(seq, static_t)
            v_logits.append(out["logits"].cpu())
            v_targets.append(y.cpu())
    v_logits = torch.cat(v_logits, dim=0)
    v_targets = torch.cat(v_targets, dim=0)

    T = temperature_scale(v_logits, v_targets, lr=1e-2, steps=200)
    probs_cal = 1 / (1 + np.exp(- (v_logits.numpy() / max(T,1e-3))))
    tau, f1 = find_threshold(v_targets.numpy().ravel(), probs_cal.ravel())
    print(f"Calibrated temperature T={T:.3f}, threshold τ={tau:.3f}, val-F1={f1:.3f}")

    # ===== Save artifacts =====
    scaler.save(save_dir)
    # Also stash feature dims for inference
    meta = {
        "seq_in_dim": int(K_seq),
        "static_dim": int(K_static),
        "micro_bins": int(cfg["windowing"]["micro_bins"]),
        "top_k_udp_ports": list(cfg["data"]["top_k_udp_ports"]),
        "channels": list(md["channels"]),
        "kernel_size": md["kernel_size"],
        "dropout": md["dropout"],
        "attention_heads": md["attention_heads"],
        "mlp_hidden": list(md["mlp_hidden"]),
        "aux_family_head": bool(md.get("aux_family_head", True)),
    }
    with open(os.path.join(save_dir, "feature_schema.json"), "r") as f:
        schema = json.load(f)
    meta["static_feature_names"] = schema.get("feature_names", [])

    with open(os.path.join(save_dir, "feature_model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    save_calibration(save_dir, T, tau)
    print(f"Saved best model and calibration under: {save_dir}")

if __name__ == "__main__":
    main()
