from __future__ import annotations
import argparse, json, os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from tqdm import tqdm

from models.dws_cnn import FastDetector
from models.utils import seed_everything, pr_auc, EarlyStopping
from models.calibrate import temperature_scale, find_threshold, save_calibration
from features.scaler import RobustScaler
from features.feature_slimming import StaticSlimmer
import os
os.environ.setdefault("TORCH_CPP_LOG_LEVEL","ERROR")

class CachedDataset(Dataset):
    def __init__(self, shard_paths: List[str], normal_subsample_rate: float = 1.0, take_indices: List[int] | None = None):
        self.shards = shard_paths
        self.normal_rate = float(normal_subsample_rate)
        self.rows = []
        for path in shard_paths:
            df = pd.read_parquet(path, columns=["file","t0","t1","y","fam","M","K_seq","K_static","seq","static"])
            if take_indices is not None:
                df = df.iloc[take_indices]
            if self.normal_rate < 0.999:
                mask_norm = (df["y"].values == 0)
                keep = np.ones(len(df), dtype=bool)
                rng = np.random.RandomState(1337)
                drop_idx = np.where(mask_norm)[0]
                drop_mask = rng.rand(drop_idx.shape[0]) > self.normal_rate
                keep[drop_idx[drop_mask]] = False
                df = df[keep]
            self.rows.append(df)
        self.df = pd.concat(self.rows, ignore_index=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        M = int(r["M"]); Kseq = int(r["K_seq"])
        seq = np.array(r["seq"], dtype=np.float32).reshape(M, Kseq)
        stat = np.array(r["static"], dtype=np.float32)
        y = np.array([float(r["y"])], dtype=np.float32)
        fam = np.array([int(r["fam"])], dtype=np.int64)
        t0 = np.array([float(r["t0"])], dtype=np.float64)
        return torch.from_numpy(seq), torch.from_numpy(stat), torch.from_numpy(y), torch.from_numpy(fam), torch.from_numpy(t0), r["file"]

def load_manifest(cache_dir: str):
    with open(os.path.join(cache_dir, "manifest.json"), "r") as f:
        return json.load(f)

def build_datasets(cfg):
    cache_dir = cfg["paths"]["cache_dir"]
    mani = load_manifest(cache_dir)
    shards = [x["path"] for x in mani["files"]]
    assert shards, "No shards found. Run preprocessing first."

    # Read quick y-stats per shard (cheap: only 'y' column)
    shard_stats = []
    for p in shards:
        dfy = pd.read_parquet(p, columns=["y"])
        n = len(dfy)
        pos = int(dfy["y"].sum())
        neg = int(n - pos)
        shard_stats.append({"path": p, "n": n, "pos": pos, "neg": neg})
    # Sort shards to interleave pos/neg better
    shards_sorted = sorted(shard_stats, key=lambda s: (-s["pos"], s["path"]))

    # Target counts
    f_train, f_val, f_test = cfg["split"]["train_val_test"]
    n = len(shards_sorted)
    n_train = max(1, int(n * f_train))
    n_val = max(1, int(n * f_val))
    n_test = max(1, n - n_train - n_val)

    # Greedy fill
    train_sh, val_sh, test_sh = [], [], []
    for s in shards_sorted:
        if len(train_sh) < n_train:
            train_sh.append(s)
        elif len(val_sh) < n_val:
            val_sh.append(s)
        else:
            test_sh.append(s)

    # Ensure positives exist in val; if not, swap one from train/test that has pos
    if sum(s["pos"] for s in val_sh) == 0:
        donor = next((s for s in train_sh if s["pos"] > 0), None) or next((s for s in test_sh if s["pos"] > 0), None)
        if donor:
            if val_sh:
                victim_idx = int(np.argmin([vs["pos"] for vs in val_sh]))
                victim = val_sh[victim_idx]
                val_sh[victim_idx] = donor
                if donor in train_sh:
                    train_sh[train_sh.index(donor)] = victim
                else:
                    test_sh[test_sh.index(donor)] = victim
            else:
                val_sh.append(donor)
                if donor in train_sh: train_sh.remove(donor)
                if donor in test_sh: test_sh.remove(donor)

    # Convert to paths
    train_paths = [s["path"] for s in train_sh]
    val_paths   = [s["path"] for s in val_sh]
    test_paths  = [s["path"] for s in test_sh] if test_sh else [train_paths[-1]]

    ds_train = CachedDataset(train_paths, normal_subsample_rate=cfg["sampling"]["normal_subsample_rate"])
    ds_val_full = CachedDataset(val_paths, normal_subsample_rate=1.0)
    ds_test = CachedDataset(test_paths, normal_subsample_rate=1.0)

    # Make thin-val stratified (keep positives)
    thin_frac = float(cfg["split"]["thin_val_fraction"])
    n_val = len(ds_val_full)
    if n_val == 0:
        # fallback: a slice of train as thin val
        ds_val_thin = CachedDataset(train_paths, normal_subsample_rate=1.0, take_indices=list(range(min(1024, len(ds_train)))))
    else:
        y = ds_val_full.df["y"].values.astype(int)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        target = max(1, int(thin_frac * n_val))
        n_pos = min(len(pos_idx), max(1, target // 2))
        n_neg = min(len(neg_idx), max(1, target - n_pos))
        rng = np.random.default_rng(1337)
        sel = []
        if len(pos_idx) > 0: sel += rng.choice(pos_idx, size=n_pos, replace=False).tolist()
        if len(neg_idx) > 0: sel += rng.choice(neg_idx, size=n_neg, replace=False).tolist()
        if not sel: sel = list(range(min(target, n_val)))
        ds_val_thin = CachedDataset(val_paths, normal_subsample_rate=1.0, take_indices=sel)

    # Debug prints
    print(f"[Split] train shards={len(train_paths)} val shards={len(val_paths)} test shards={len(test_paths)}")
    print(f"[Split] val_full size={len(ds_val_full)} (pos={int(ds_val_full.df['y'].sum())}, neg={len(ds_val_full)-int(ds_val_full.df['y'].sum())})")
    print(f"[Split] val_thin size={len(ds_val_thin)} (pos={int(ds_val_thin.df['y'].sum())}, neg={len(ds_val_thin)-int(ds_val_thin.df['y'].sum())})")

    return ds_train, ds_val_thin, ds_val_full, ds_test

def collate(batch):
    seq = torch.stack([b[0] for b in batch], dim=0)
    static = torch.stack([b[1] for b in batch], dim=0)
    y = torch.stack([b[2] for b in batch], dim=0)
    fam = torch.stack([b[3] for b in batch], dim=0).squeeze(1)
    ts = torch.stack([b[4] for b in batch], dim=0)
    names = [b[5] for b in batch]
    return seq, static, y, fam, ts, names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed_everything(cfg["seed"])
    if cfg["training"].get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # ===== datasets from cache =====
    ds_train, ds_val_thin, ds_val_full, ds_test = build_datasets(cfg)

    # fit scaler & static slimmer on train static features
    def stack_static(ds, limit=20000):
        xs = []
        for i in range(min(limit, len(ds))):
            xs.append(ds[i][1].numpy())
        return np.stack(xs, axis=0)
    Xs_train = stack_static(ds_train)
    scaler = RobustScaler()
    names_stub = [f"f_{j}" for j in range(Xs_train.shape[1])]
    scaler.fit(Xs_train, names_stub)

    slimmer = StaticSlimmer(out_dim=cfg["features"]["static_max_dim"], whiten=cfg["features"]["pca_whiten"])
    slimmer.fit(scaler.transform(Xs_train, names_stub), names_stub)

    # build model
    seq_example, stat_example, *_ = ds_train[0]
    M, Kseq = seq_example.shape
    Kstat_slim = slimmer.out_dim
    model = FastDetector(
        seq_in_dim=Kseq,
        static_dim=Kstat_slim,
        channels=tuple(cfg["training"]["channels"]),
        k=cfg["training"]["kernel_size"],
        drop=cfg["training"]["dropout"],
        mlp_hidden=tuple(cfg["training"]["mlp_hidden"]),
        aux_family_head=bool(cfg["training"]["aux_family_head"]),
        n_families=6
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # criterion & optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg["training"]["class_weight_pos"]], device=device))
    aux_ce = nn.CrossEntropyLoss(ignore_index=-1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=cfg["training"]["epochs"]) if cfg["training"]["cosine_lr"] else None
    scaler_amp = torch.cuda.amp.GradScaler(enabled=bool(cfg["training"]["amp"]))

    # DataLoaders
    def mk_loader(ds, shuffle, workers):
        return DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=shuffle,
                          num_workers=workers, collate_fn=collate, pin_memory=bool(cfg["training"]["pinned_memory"]))
    train_loader = mk_loader(ds_train, True, cfg["training"]["dataloader_workers"])
    val_thin_loader = mk_loader(ds_val_thin, False, max(1, cfg["training"]["dataloader_workers"]//2))
    val_full_loader = mk_loader(ds_val_full, False, max(1, cfg["training"]["dataloader_workers"]//2))

    early = EarlyStopping(patience=cfg["training"]["early_stop_patience"], mode="max")
    best_saved = False
    accum = max(1, int(cfg["training"]["grad_accum_steps"]))

    # ====== TRAIN ======
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        running = []
        opt.zero_grad(set_to_none=True)

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch"), start=1):
            seq, static, y, fam, ts, names = batch
            # transform static on CPU for speed/memory, then move
            stat_np = static.numpy()
            stat_scaled = scaler.transform(stat_np, names_stub)
            stat_slim = slimmer.transform(stat_scaled)
            static_t = torch.from_numpy(stat_slim).to(device).float()

            seq = seq.to(device, non_blocking=True)
            y = y.to(device)
            fam = fam.to(device)

            with torch.cuda.amp.autocast(enabled=bool(cfg["training"]["amp"])):
                out = model(seq, static_t)
                loss = criterion(out["logits"], y)
                if cfg["training"]["aux_family_head"] and "family_logits" in out:
                    loss = loss + 0.2 * aux_ce(out["family_logits"], fam)

            scaler_amp.scale(loss / accum).backward()
            if i % accum == 0:
                scaler_amp.step(opt)
                scaler_amp.update()
                opt.zero_grad(set_to_none=True)
            running.append(loss.item())

        # ===== THIN-VAL =====
        model.eval()
        v_logits, v_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_thin_loader, desc="Thin-val", unit="batch", leave=False):
                seq, static, y, fam, ts, names = batch
                stat_np = static.numpy()
                stat_scaled = scaler.transform(stat_np, names_stub)
                stat_slim = slimmer.transform(stat_scaled)
                static_t = torch.from_numpy(stat_slim).to(device).float()
                seq = seq.to(device)
                logits = model(seq, static_t)["logits"]
                v_logits.append(logits.cpu().numpy())
                v_targets.append(y.cpu().numpy())
        v_logits = np.vstack(v_logits); v_targets = np.vstack(v_targets)
        probs = 1/(1+np.exp(-v_logits))
        pr = pr_auc(v_targets.ravel(), probs.ravel())
        print(f"Epoch {epoch:02d} loss={np.mean(running):.4f} thin-val PR-AUC={pr:.4f}")
        if sched: sched.step()

        improved = early.step(pr)
        if improved:
            os.makedirs(cfg["logging"]["save_dir"], exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "pr": pr}, os.path.join(cfg["logging"]["save_dir"], "model_best.pt"))
            best_saved = True
        # Always save at least once (epoch 1) so downstream steps have a checkpoint
        if epoch == 1 and not best_saved:
            os.makedirs(cfg["logging"]["save_dir"], exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "pr": pr}, os.path.join(cfg["logging"]["save_dir"], "model_best.pt"))
            best_saved = True

        if early.stopped:
            print("Early stop.")
            break

        # ===== Hard Negative Mining (optional) =====
        if cfg["sampling"]["hard_negative_mining"]["enabled"] and (epoch % cfg["sampling"]["hard_negative_mining"]["refresh_every_epochs"] == 0):
            preds = probs.ravel()
            targets = v_targets.ravel()
            neg_idx = np.where(targets == 0)[0]
            if neg_idx.size > 0:
                k = cfg["sampling"]["hard_negative_mining"]["top_k_per_file"]
                sel = neg_idx[np.argsort(preds[neg_idx])[::-1][:k]]
                ds = ds_val_thin.df.iloc[sel]
                ds_train.df = pd.concat([ds_train.df, ds], ignore_index=True)
                train_loader = mk_loader(ds_train, True, cfg["training"]["dataloader_workers"])
                print(f"[HNM] Added {len(ds)} hard negatives. Train size={len(ds_train)}")

    # ===== CALIBRATION on FULL VAL =====
    best_path = os.path.join(cfg["logging"]["save_dir"], "model_best.pt")
    if not os.path.exists(best_path):
        print(f"[Calib] No best model saved at {best_path}. Skipping calibration.")
    else:
        # reload best model
        model.load_state_dict(torch.load(best_path)["model"])
        model.eval()
        v_logits, v_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_full_loader, desc="Full-val (calibration)", unit="batch"):
                seq, static, y, fam, ts, names = batch
                stat_np = static.numpy()
                stat_scaled = scaler.transform(stat_np, names_stub)
                stat_slim = slimmer.transform(stat_scaled)
                static_t = torch.from_numpy(stat_slim).to(device).float()
                seq = seq.to(device)
                out = model(seq, static_t)
                v_logits.append(out["logits"].cpu())
                v_targets.append(y.cpu())
        v_logits = torch.cat(v_logits, dim=0)
        v_targets = torch.cat(v_targets, dim=0)

        y_np = v_targets.numpy().ravel()
        pos_ct, neg_ct = int((y_np == 1).sum()), int((y_np == 0).sum())
        if pos_ct == 0 or neg_ct == 0:
            print(f"[Calib] Warning: full-val has single class (pos={pos_ct}, neg={neg_ct}). "
                  f"Skipping temperature scaling; using τ=0.5.")
            T = 1.0
            tau = 0.5
        else:
            T = temperature_scale(v_logits, v_targets, lr=1e-2, steps=200)
            probs_cal = 1/(1+np.exp(-(v_logits.numpy()/max(T,1e-3))))
            tau, f1 = find_threshold(y_np, probs_cal.ravel())

        save_dir = cfg["logging"]["save_dir"]
        RobustScaler.save(scaler, save_dir) if hasattr(RobustScaler, "save") else scaler.save(save_dir)
        slimmer.save(save_dir)
        with open(os.path.join(save_dir, "feature_model_meta.json"), "w") as f:
            json.dump({"seq_in_dim": int(Kseq), "static_dim": int(slimmer.out_dim), "micro_bins": int(M)}, f, indent=2)
        save_calibration(save_dir, T, tau)
        print(f"Calibrated T={T:.3f}, τ={tau:.3f}. Saved to {save_dir}.")

if __name__ == "__main__":
    main()
