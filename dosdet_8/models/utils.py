from __future__ import annotations
import math
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

def seed_everything(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device(pref: str = "auto") -> torch.device:
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if pref == "mps" or (pref == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha_pos: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha_pos = alpha_pos
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        # logits: [B,1], targets: [B,1]
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        at = self.alpha_pos * targets + (1 - targets)
        loss = at * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

@dataclass
class EarlyStopping:
    patience: int = 5
    mode: str = "max"  # 'max' for PR-AUC
    best: float = float("-inf")
    counter: int = 0
    stopped: bool = False

    def step(self, metric: float) -> bool:
        improve = metric > self.best if self.mode == "max" else metric < self.best
        if improve:
            self.best = metric
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
            return False

def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return 0.0

def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p * r / (p + r + 1e-9)
    i = int(np.nanargmax(f1))
    # precision_recall_curve returns thresholds shorter by 1 than p/r
    tau = thr[max(0, i - 1)] if len(thr) else 0.5
    return float(tau), float(np.nanmax(f1))
