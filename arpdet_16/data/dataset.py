from __future__ import annotations
import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.pcap_reader import iter_rows_from_pcap
from data.windowizer import iter_windows
from features.seq_features import compute_sequence_features
from features.static_features import compute_static_features

ATTACK_FAMILIES = ["SSDP", "UDP", "TCP", "ICMP", "HTTP", "DNS", "OTHER"]

@dataclass
class LabelRecord:
    file: str
    attack_label: int
    attack_start_ts: Optional[float]  # epoch seconds if available
    family: Optional[str]

class LabelProvider:
    def __init__(self, labels_csv: str):
        """
        labels.csv columns:
           file,attack_label,attack_start_iso,family
        attack_start_iso may be empty. If provided, it will be parsed to epoch seconds lazily.
        """
        df = pd.read_csv(labels_csv)
        df = df.fillna("")
        self._rows: Dict[str, LabelRecord] = {}
        for _, r in df.iterrows():
            f = str(r["file"])
            lab = int(r["attack_label"])
            fam = str(r["family"]).upper() if str(r["family"]) else None
            iso = str(r["attack_start_iso"]) if "attack_start_iso" in df.columns else ""
            # lazy conversion; we'll convert using pandas to_datetime if present
            ts = None
            if iso and iso.strip():
                try:
                    ts = pd.to_datetime(iso, utc=True).view("int64") / 1e9
                    ts = float(ts)
                except Exception:
                    ts = None
            self._rows[os.path.basename(f)] = LabelRecord(os.path.basename(f), lab, ts, fam)

    def window_label(self, file_basename: str, window_start: float) -> Tuple[int, Optional[int]]:
        """Return (binary_label, family_id) for a window starting at window_start."""
        rec = self._rows.get(file_basename)
        if rec is None:
            # default: normal
            return 0, None
        if rec.attack_label == 0:
            return 0, None
        # attack file
        if rec.attack_start_ts is None or window_start >= rec.attack_start_ts:
            family_id = ATTACK_FAMILIES.index(rec.family) if (rec.family and rec.family in ATTACK_FAMILIES) else None
            return 1, family_id
        return 0, None

def _random_header_remap(rows: List[Dict]) -> List[Dict]:
    """Randomize IP/MAC identities without changing semantics."""
    if not rows:
        return rows
    rnd = random.Random()
    # build mappings
    ips = list({r.get("src_ip") for r in rows if r.get("src_ip")} | {r.get("dst_ip") for r in rows if r.get("dst_ip")})
    macs = list({r.get("src_mac") for r in rows if r.get("src_mac")} | {r.get("dst_mac") for r in rows if r.get("dst_mac")})
    ip_map = {ip: f"10.{rnd.randint(0,255)}.{rnd.randint(0,255)}.{rnd.randint(1,254)}" for ip in ips}
    mac_map = {m: f"{rnd.randint(0,255):02x}:{rnd.randint(0,255):02x}:{rnd.randint(0,255):02x}:{rnd.randint(0,255):02x}:{rnd.randint(0,255):02x}:{rnd.randint(0,255):02x}" for m in macs}
    out = []
    for r in rows:
        rr = dict(r)
        if rr.get("src_ip"): rr["src_ip"] = ip_map[rr["src_ip"]]
        if rr.get("dst_ip"): rr["dst_ip"] = ip_map[rr["dst_ip"]]
        if rr.get("src_mac"): rr["src_mac"] = mac_map[rr["src_mac"]]
        if rr.get("dst_mac"): rr["dst_mac"] = mac_map[rr["dst_mac"]]
        out.append(rr)
    return out

def _time_warp_rows(rows: List[Dict], factor: float) -> List[Dict]:
    """Scale inter-arrivals by a factor around the window start."""
    if not rows:
        return rows
    ts = np.array([float(r["ts"]) for r in rows], dtype=float)
    t0 = float(min(ts))
    out = []
    for r in rows:
        rr = dict(r)
        rr["ts"] = t0 + (float(r["ts"]) - t0) * factor
        out.append(rr)
    return out

class WindowDataset(Dataset):
    """
    Streams pcaps, builds sliding windows, computes features, and yields tensors.
    Each __getitem__ returns:
      seq: [M, K_seq], static: [K_static], y: 0/1, family_id or -1, t_start (float)
    """
    def __init__(
        self,
        pcap_paths: List[str],
        labels: LabelProvider,
        window_sec: float,
        stride_sec: float,
        micro_bins: int,
        augment: Optional[Dict] = None,
        for_training: bool = True,
    ):
        self.files = pcap_paths
        self.labels = labels
        self.W = window_sec
        self.S = stride_sec
        self.M = micro_bins
        self.for_training = for_training
        self.aug = augment or {}

        # index of all windows across files: list of (file, t_start, t_end, offset_in_file)
        self.index: List[Tuple[int, float, float, int]] = []
        self._materialize_index()

    def _materialize_index(self):
        for i, p in enumerate(self.files):
            # We only build window start times; features computed lazily
            rows = list(iter_rows_from_pcap(p))
            for (t0, t1, win_rows, bins) in iter_windows(rows, self.W, self.S, self.M):
                self.index.append((i, t0, t1, 0))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        fi, t0, t1, _ = self.index[idx]
        p = self.files[fi]
        base = os.path.basename(p)

        rows = list(iter_rows_from_pcap(p))
        # isolate current window rows only (cheap filter)
        win_rows = [r for r in rows if t0 <= float(r["ts"]) < t1]
        # augmentation (training only)
        if self.for_training and win_rows:
            # time-warp
            tw = self.aug.get("time_warp_pct", 0.0)
            if tw > 0:
                factor = 1.0 + random.uniform(-tw, tw)
                win_rows = _time_warp_rows(win_rows, factor)
            # header remap
            if random.random() < self.aug.get("header_randomize_prob", 0.0):
                win_rows = _random_header_remap(win_rows)
            # benign overlay: sample another file labeled normal and overlay few rows
            if random.random() < self.aug.get("benign_overlay_prob", 0.0):
                normals = [f for f in self.files if self.labels._rows.get(os.path.basename(f), LabelRecord("",0,None,None)).attack_label == 0]
                if normals:
                    q = random.choice(normals)
                    q_rows = list(iter_rows_from_pcap(q))
                    # pick random slice up to 20% of window rows
                    k = max(1, int(0.2 * len(win_rows)))
                    add = random.sample(q_rows, min(k, len(q_rows)))
                    win_rows = win_rows + add

        # recompute bin assignments (in case of time-warp)
        bins = []
        bw = self.W / self.M
        for r in win_rows:
            dt = float(r["ts"]) - t0
            b = int(np.floor(dt / bw))
            b = max(0, min(self.M - 1, b))
            bins.append(b)

        # features
        seq, extras = compute_sequence_features(win_rows, bins, self.M)
        static_vec, static_names, snaps = compute_static_features(win_rows, self.M, extras, self.W)

        y, fam_id = self.labels.window_label(base, t0)
        fam_id = -1 if fam_id is None else int(fam_id)

        seq = torch.from_numpy(seq).float()
        static = torch.from_numpy(static_vec).float()
        y = torch.tensor([float(y)], dtype=torch.float32)
        fam = torch.tensor([fam_id], dtype=torch.long)
        ts_start = torch.tensor([t0], dtype=torch.float64)
        return seq, static, y, fam, ts_start, base
