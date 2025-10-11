#src/arp_detector/utils/seed.py
"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed common libraries for reproducibility."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def set_deterministic(mode: Optional[str]) -> torch.autocast | None:
    """Return an autocast context manager respecting the requested precision mode."""

    if mode is None or mode.lower() == "none":
        return None
    if mode.lower() == "autocast":
        return torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported precision mode: {mode}")