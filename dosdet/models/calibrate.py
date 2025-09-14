from __future__ import annotations
import json
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from .utils import best_f1_threshold

class _TempScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.T.clamp_min(1e-3)

def temperature_scale(logits: torch.Tensor, labels: torch.Tensor, lr: float = 1e-2, steps: int = 500) -> float:
    """
    Optimize temperature T to minimize NLL on validation.
    """
    model = _TempScaling().to(logits.device)
    opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=steps)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(model(logits), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(model.T.detach().cpu().item())

def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    logits = logits / max(T, 1e-3)
    return 1 / (1 + np.exp(-logits))

def find_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    return best_f1_threshold(y_true, probs)

def save_calibration(path_dir: str, T: float, tau: float):
    with open(f"{path_dir}/calibration.json", "w") as f:
        json.dump({"temperature": float(T), "threshold": float(tau)}, f, indent=2)
