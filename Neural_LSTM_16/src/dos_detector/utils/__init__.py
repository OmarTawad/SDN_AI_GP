"""Utility helpers exposed at the package root."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

from .dtypes import (
    DEFAULT_NUMPY_DTYPE,
    DEFAULT_TORCH_DTYPE,
    resolve_precision_mode,
    resolve_torch_dtype,
    safe_cast_tensor,
)

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


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Return a CUDA device when supported by this PyTorch build, else CPU."""

    if not prefer_cuda or not torch.cuda.is_available():
        return CPU_DEVICE
    try:
        arch_list = torch.cuda.get_arch_list()
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        if arch_list and arch not in arch_list:
            try:
                # Probe CUDA execution to handle compatible SMs not listed in arch list.
                torch.zeros(1, device="cuda")
            except Exception:
                warnings.warn(
                    f"CUDA device {arch} not supported by this PyTorch build ({', '.join(arch_list)}). "
                    "Falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return CPU_DEVICE
    except Exception:
        return CPU_DEVICE
    return torch.device("cuda")


__all__ = [
    "CPU_DEVICE",
    "configure_cpu_environment",
    "resolve_project_root",
    "resolve_device",
    "DEFAULT_TORCH_DTYPE",
    "DEFAULT_NUMPY_DTYPE",
    "resolve_precision_mode",
    "resolve_torch_dtype",
    "safe_cast_tensor",
]
