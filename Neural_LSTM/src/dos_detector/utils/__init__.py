<<<<<<< HEAD
#src/dos_detector/utils/__init__.py
=======
"""Utility helpers exposed at the package root."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

CPU_DEVICE = torch.device("cpu")


def _safe_set_threads(setter, value: int, label: str) -> None:
    try:
        setter(value)
    except RuntimeError as exc:
        warnings.warn(
            f"Unable to set torch {label} to {value}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def configure_cpu_environment(threads: int = 2, interop_threads: int = 2) -> torch.device:
    """Clamp PyTorch to the requested CPU threading budget."""

    threads = max(1, threads)
    interop_threads = max(1, interop_threads)
    _safe_set_threads(torch.set_num_threads, threads, "num_threads")
    _safe_set_threads(torch.set_num_interop_threads, interop_threads, "num_interop_threads")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))
    return CPU_DEVICE


def resolve_project_root(anchor: str = "Neural_LSTM") -> Path:
    """Return the repository root by walking up from this file location."""

    path = Path(__file__).resolve()
    for parent in path.parents:
        if parent.name == anchor:
            return parent
    return Path.cwd()


__all__ = ["CPU_DEVICE", "configure_cpu_environment", "resolve_project_root"]
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7
