from __future__ import annotations

import os

# Constrain BLAS / Torch threads for 2 vCPU boxes before heavy imports
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import argparse
import json
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm

from features.feature_slimming import StaticSlimmer
from features.scaler import RobustScaler
from models.dws_cnn import FastDetector
from models.calibrate import find_threshold, save_calibration, temperature_scale
from models.utils import EarlyStopping, pr_auc, seed_everything

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

# Honour the low-core environment at runtime as well
_cpu_slots = max(1, os.cpu_count() or 1)
torch.set_num_threads(min(2, _cpu_slots))
try:
    torch.set_num_interop_threads(1)
except AttributeError:
    pass


def _read_parquet(path: str, columns: List[str]) -> pd.DataFrame:
    """Helper that favours single-threaded parquet reads for small VMs."""
    kwargs = {"columns": columns, "engine": "pyarrow"}
    try:
        return pd.read_parquet(path, use_threads=False, **kwargs)
    except TypeError:
        return pd.read_parquet(path, **kwargs)


class CachedDataset(Dataset):
    """Thin wrapper that streams parquet shards produced by data/preprocess.py."""

    def __init__(
        self,
        shard_paths: List[str],
        normal_subsample_rate: float = 1.0,
        take_indices: List[int] | None = None,
        rng_seed: int = 1337,
    ) -> None:
        self.normal_rate = float(normal_subsample_rate)
        self._rng = np.random.default_rng(rng_seed)
        frames: List[pd.DataFrame] = []
        cols = [
            "file",
            "t0",
            "t1",
            "y",
            "fam",
            "M",
            "K_seq",
            "K_static",
            "seq",
            "static",
        ]
        for path in tqdm(shard_paths, desc="Loading shards", unit="shard", leave=False):
            df = _read_parquet(path, cols)
            if take_indices is not None:
                df = df.iloc[take_indices]
            if self.normal_rate < 0.999:
                mask_norm = df["y"].values == 0
                keep = np.ones(len(df), dtype=bool)
                drop_idx = np.where(mask_norm)[0]
                drop_draw = self._rng.random(drop_idx.size) > self.normal_rate
                keep[drop_idx[drop_draw]] = False
                df = df[keep]
            frames.append(df)
        if not frames:
            raise ValueError("No parquet shards found to construct dataset.")
        self.df = pd.concat(frames, ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        M = int(row["M"])
        K_seq = int(row["K_seq"])
        seq = np.array(row["seq"], dtype=np.float32).reshape(M, K_seq)
        static = np.array(row["static"], dtype=np.float32)
        y = np.array([float(row["y"])], dtype=np.float32)
        fam = np.array([int(row["fam"])], dtype=np.int64)
        t0 = np.array([float(row["t0"])], dtype=np.float64)
        return (
            torch.from_numpy(seq),
            torch.from_numpy(static),
            torch.from_numpy(y),
            torch.from_numpy(fam),
            torch.from_numpy(t0),
            row["file"],
        )


def load_manifest(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def build_datasets(cfg: dict):
    cache_dir = cfg["paths"]["cache_dir"]
    mani = load_manifest(cache_dir)
    shards = [entry["path"] for entry in mani["files"]]
    assert shards, "No shards found. Run preprocess first."

    shard_stats = []
    for path in tqdm(shards, desc="Shard stats", unit="shard", leave=False):
        dfy = _read_parquet(path, ["y"])
        n = len(dfy)
        pos = int(dfy["y"].sum())
        shard_stats.append({"path": path, "n": n, "pos": pos})
    shards_sorted = sorted(shard_stats, key=lambda s: (-s["pos"], s["path"]))

    f_train, f_val, f_test = cfg["split"]["train_val_test"]
    n = len(shards_sorted)
    n_train = max(1, int(round(n * f_train)))
    n_val = max(1, int(round(n * f_val)))
    n_test = max(1, n - n_train - n_val)

    train_sh, val_sh, test_sh = [], [], []
    for shard in shards_sorted:
        if len(train_sh) < n_train:
            train_sh.append(shard)
        elif len(val_sh) < n_val:
            val_sh.append(shard)
        else:
            test_sh.append(shard)

    if sum(s["pos"] for s in val_sh) == 0:
        donor = next((s for s in train_sh if s["pos"] > 0), None) or next(
            (s for s in test_sh if s["pos"] > 0),
            None,
        )
        if donor is not None:
            if val_sh:
                victim_idx = int(np.argmin([s["pos"] for s in val_sh]))
                victim = val_sh[victim_idx]
                val_sh[victim_idx] = donor
                if donor in train_sh:
                    train_sh[train_sh.index(donor)] = victim
                else:
                    test_sh[test_sh.index(donor)] = victim
            else:
                val_sh.append(donor)
                if donor in train_sh:
                    train_sh.remove(donor)
                if donor in test_sh:
                    test_sh.remove(donor)

    train_paths = [s["path"] for s in train_sh]
    val_paths = [s["path"] for s in val_sh]
    test_paths = [s["path"] for s in test_sh] if test_sh else [train_paths[-1]]

    ds_train = CachedDataset(
        train_paths,
        normal_subsample_rate=cfg["sampling"]["normal_subsample_rate"],
    )
    ds_val_full = CachedDataset(val_paths, normal_subsample_rate=1.0)

    thin_frac = float(cfg["split"]["thin_val_fraction"])
    n_val = len(ds_val_full)
    if n_val == 0:
        ds_val_thin = CachedDataset(
            train_paths,
            normal_subsample_rate=1.0,
            take_indices=list(range(min(1024, len(ds_train)))),
        )
    else:
        y = ds_val_full.df["y"].values.astype(int)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        target = max(1, int(round(thin_frac * n_val)))
        n_pos = min(len(pos_idx), max(1, target // 2))
        n_neg = min(len(neg_idx), max(1, target - n_pos))
        rng = np.random.default_rng(1337)
        selected: List[int] = []
        if len(pos_idx) > 0:
            selected += rng.choice(pos_idx, size=n_pos, replace=False).tolist()
        if len(neg_idx) > 0:
            selected += rng.choice(neg_idx, size=n_neg, replace=False).tolist()
        if not selected:
            selected = list(range(min(target, n_val)))
        ds_val_thin = CachedDataset(val_paths, normal_subsample_rate=1.0, take_indices=selected)

    print(
        f"[Split] shards train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}"
    )
    print(
        f"[Split] val_full size={len(ds_val_full)} pos={int(ds_val_full.df['y'].sum())}"
    )
    print(
        f"[Split] val_thin size={len(ds_val_thin)} pos={int(ds_val_thin.df['y'].sum())}"
    )

    return ds_train, ds_val_thin, ds_val_full


def collate(batch):
    seq = torch.stack([b[0] for b in batch], dim=0)
    static = torch.stack([b[1] for b in batch], dim=0)
    y = torch.stack([b[2] for b in batch], dim=0)
    # Family labels are still emitted by cached shards for backwards compatibility.
    fam = torch.stack([b[3] for b in batch], dim=0).squeeze(1)
    ts = torch.stack([b[4] for b in batch], dim=0)
    names = [b[5] for b in batch]
    return seq, static, y, fam, ts, names


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN detector from cached parquet shards")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 1337))
    if cfg["training"].get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    ds_train, ds_val_thin, ds_val_full = build_datasets(cfg)

    def stack_static(ds, limit: int = 20000):
        xs = []
        for i in tqdm(range(min(limit, len(ds))), desc="Static sampling", unit="window", leave=False):
            xs.append(ds[i][1].numpy())
        return np.stack(xs, axis=0)

    static_matrix = stack_static(ds_train)
    feature_names = [f"f_{j}" for j in range(static_matrix.shape[1])]
    scaler = RobustScaler().fit(static_matrix, feature_names)
    feature_names = scaler.feature_names_ or feature_names

    slimmer = StaticSlimmer(
        out_dim=cfg["features"]["static_max_dim"],
        whiten=cfg["features"]["pca_whiten"],
    )
    slimmer.fit(scaler.transform(static_matrix, feature_names), feature_names)

    seq_example, _, *_ = ds_train[0]
    M, K_seq = seq_example.shape
    K_static_slim = slimmer.out_dim

    tr_cfg = cfg["training"]
    model = FastDetector(
        seq_in_dim=K_seq,
        static_dim=K_static_slim,
        channels=tuple(tr_cfg["channels"]),
        k=tr_cfg["kernel_size"],
        drop=tr_cfg["dropout"],
        mlp_hidden=tuple(tr_cfg["mlp_hidden"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([tr_cfg["class_weight_pos"]], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tr_cfg["lr"], weight_decay=tr_cfg["weight_decay"]
    )
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=tr_cfg["epochs"])
        if tr_cfg.get("cosine_lr", False)
        else None
    )
    amp_enabled = bool(tr_cfg.get("amp", False) and torch.cuda.is_available())
    scaler_amp = GradScaler(enabled=amp_enabled)
    grad_clip = float(tr_cfg.get("grad_clip", 0.0) or 0.0)
    accum_steps = max(1, int(tr_cfg.get("grad_accum_steps", 1)))

    def make_loader(ds, shuffle: bool, workers: int):
        cpu_slots = max(1, os.cpu_count() or 1)
        worker_cap = min(workers, max(0, cpu_slots - 1))
        pin_mem = bool(tr_cfg.get("pinned_memory", False) and torch.cuda.is_available())
        return DataLoader(
            ds,
            batch_size=tr_cfg["batch_size"],
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=worker_cap,
            pin_memory=pin_mem,
        )

    train_loader = make_loader(ds_train, True, tr_cfg["dataloader_workers"])
    val_thin_loader = make_loader(ds_val_thin, False, max(1, tr_cfg["dataloader_workers"] // 2))
    val_full_loader = make_loader(ds_val_full, False, max(1, tr_cfg["dataloader_workers"] // 2))

    early = EarlyStopping(patience=tr_cfg["early_stop_patience"], mode="max")
    best_saved = False

    for epoch in range(1, tr_cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss: List[float] = []

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch:02d}", unit="batch"),
            start=1,
        ):
            seq, static, y, _, ts, names = batch
            stat_np = static.cpu().numpy()
            stat_scaled = scaler.transform(stat_np, feature_names)
            stat_slim = slimmer.transform(stat_scaled)
            static_t = torch.from_numpy(stat_slim).to(device, non_blocking=True).float()

            seq = seq.to(device, non_blocking=True)
            y = y.to(device)

            cast_ctx = torch.amp.autocast("cuda") if amp_enabled else nullcontext()
            with cast_ctx:
                out = model(seq, static_t)
                loss = criterion(out["logits"], y)

            scaler_amp.scale(loss / accum_steps).backward()
            if step % accum_steps == 0 or step == len(train_loader):
                if grad_clip > 0:
                    scaler_amp.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss.append(loss.item())

        model.eval()
        val_logits, val_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_thin_loader, desc="Thin-val", unit="batch", leave=False):
                seq, static, y, _, ts, names = batch
                stat_np = static.cpu().numpy()
                stat_scaled = scaler.transform(stat_np, feature_names)
                stat_slim = slimmer.transform(stat_scaled)
                static_t = torch.from_numpy(stat_slim).to(device).float()
                seq = seq.to(device)
                with (torch.amp.autocast("cuda") if amp_enabled else nullcontext()):
                    logits = model(seq, static_t)["logits"]
                val_logits.append(logits.cpu().numpy())
                val_targets.append(y.cpu().numpy())
        val_logits_np = np.vstack(val_logits)
        val_targets_np = np.vstack(val_targets)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits_np))
        pr = pr_auc(val_targets_np.ravel(), val_probs.ravel())
        print(f"Epoch {epoch:02d} loss={np.mean(running_loss):.4f} thin-val PR-AUC={pr:.4f}")
        if scheduler:
            scheduler.step()

        improved = early.step(pr)
        artifacts_dir = cfg["paths"]["artifacts_dir"]
        os.makedirs(artifacts_dir, exist_ok=True)
        if improved:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "pr": pr},
                os.path.join(artifacts_dir, "model_best.pt"),
            )
            best_saved = True
        if epoch == 1 and not best_saved:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "pr": pr},
                os.path.join(artifacts_dir, "model_best.pt"),
            )
            best_saved = True
        if early.stopped:
            print("Early stopping triggered.")
            break

        if cfg["sampling"]["hard_negative_mining"]["enabled"] and (
            epoch % cfg["sampling"]["hard_negative_mining"]["refresh_every_epochs"] == 0
        ):
            negatives = np.where(val_targets_np.ravel() == 0)[0]
            if negatives.size > 0:
                k = cfg["sampling"]["hard_negative_mining"]["top_k_per_file"]
                idx = negatives[np.argsort(val_probs.ravel()[negatives])[::-1][:k]]
                ds_hard = ds_val_thin.df.iloc[idx]
                ds_train.df = pd.concat([ds_train.df, ds_hard], ignore_index=True)
                train_loader = make_loader(
                    ds_train,
                    True,
                    tr_cfg["dataloader_workers"],
                )
                print(f"[HNM] Added {len(ds_hard)} hard negatives. New train size={len(ds_train)}")

    artifacts_dir = cfg["paths"]["artifacts_dir"]
    best_path = os.path.join(artifacts_dir, "model_best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError("Training finished but no best model checkpoint was saved.")

    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    model.eval()
    full_logits, full_targets = [], []
    with torch.no_grad():
        for batch in tqdm(val_full_loader, desc="Full-val (calibration)", unit="batch"):
            seq, static, y, _, ts, names = batch
            stat_np = static.cpu().numpy()
            stat_scaled = scaler.transform(stat_np, feature_names)
            stat_slim = slimmer.transform(stat_scaled)
            static_t = torch.from_numpy(stat_slim).to(device).float()
            seq = seq.to(device)
            with (torch.amp.autocast("cuda") if amp_enabled else nullcontext()):
                out = model(seq, static_t)
            full_logits.append(out["logits"].cpu())
            full_targets.append(y.cpu())
    full_logits_t = torch.cat(full_logits, dim=0)
    full_targets_t = torch.cat(full_targets, dim=0)

    y_np = full_targets_t.numpy().ravel()
    pos_ct, neg_ct = int((y_np == 1).sum()), int((y_np == 0).sum())
    if pos_ct == 0 or neg_ct == 0:
        print(
            f"[Calib] Warning: validation split has single class (pos={pos_ct}, neg={neg_ct})."
            " Skipping temperature scaling and using τ=0.5."
        )
        T = 1.0
        tau = 0.5
    else:
        T = temperature_scale(full_logits_t, full_targets_t, lr=1e-2, steps=200)
        probs_cal = 1.0 / (1.0 + np.exp(-(full_logits_t.numpy() / max(T, 1e-3))))
        tau, best_f1 = find_threshold(y_np, probs_cal.ravel())
        print(f"[Calib] Best F1={best_f1:.4f} at τ={tau:.3f} with T={T:.3f}")

    scaler.save(artifacts_dir)
    slimmer.save(artifacts_dir)
    meta = {
        "seq_in_dim": int(K_seq),
        "static_dim": int(K_static_slim),
        "micro_bins": int(M),
        "top_k_udp_ports": list(cfg["data"]["top_k_udp_ports"]),
        "channels": list(tr_cfg["channels"]),
        "kernel_size": tr_cfg["kernel_size"],
        "dropout": tr_cfg["dropout"],
        "mlp_hidden": list(tr_cfg["mlp_hidden"]),
    }
    with open(os.path.join(artifacts_dir, "feature_model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    save_calibration(artifacts_dir, T, tau)
    print(f"Saved artifacts to {artifacts_dir}")


if __name__ == "__main__":
    main()
