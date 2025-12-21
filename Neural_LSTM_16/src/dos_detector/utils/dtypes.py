"""Shared dtype helpers used across training and inference."""

from __future__ import annotations

import numpy as np
import torch

# Default to half precision for torch, but keep numpy in float32 to avoid overflow
# during feature extraction and scaling.
DEFAULT_TORCH_DTYPE: torch.dtype = torch.float16
DEFAULT_NUMPY_DTYPE: np.dtype = np.float32


def resolve_precision_mode(mode: str | None) -> tuple[torch.dtype, np.dtype]:
    """Map a precision mode string to concrete torch / numpy dtypes."""

    normalized = (mode or "").lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16, np.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16, np.float32
    if normalized in {"float32", "fp32", "full", "32"}:
        return torch.float32, np.float32
    if normalized == "autocast":
        if torch.cuda.is_available():
            return torch.float16, np.float32
        return torch.bfloat16, np.float32
    return DEFAULT_TORCH_DTYPE, DEFAULT_NUMPY_DTYPE


def resolve_torch_dtype(device: torch.device, dtype: torch.dtype) -> torch.dtype:
    """Ensure torch dtype is supported on the requested device."""

    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def safe_cast_tensor(value: torch.Tensor | np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Cast to a dtype while clamping to the representable range."""

    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if torch.is_floating_point(tensor) and dtype in (torch.float16, torch.bfloat16):
        tensor = tensor.to(torch.float32)
        info = torch.finfo(dtype)
        tensor = torch.clamp(tensor, min=info.min, max=info.max)
    return tensor.to(dtype=dtype)


__all__ = [
    "DEFAULT_TORCH_DTYPE",
    "DEFAULT_NUMPY_DTYPE",
    "resolve_precision_mode",
    "safe_cast_tensor",
    "resolve_torch_dtype",
]
