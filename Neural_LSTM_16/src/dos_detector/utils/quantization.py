"""Quantization helpers for CPU deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Tuple

import torch
from torch import nn

SUPPORTED_ENGINES = {"fbgemm", "qnnpack"}


def set_quantized_engine(backend: str | None) -> str:
    """Set the quantized backend engine if requested."""

    if not backend:
        return torch.backends.quantized.engine
    normalized = str(backend).lower()
    if normalized not in SUPPORTED_ENGINES:
        raise ValueError(f"Unsupported quantized engine '{backend}'. Use one of {sorted(SUPPORTED_ENGINES)}.")
    torch.backends.quantized.engine = normalized
    return normalized


def state_dict_is_quantized(state_dict: Mapping[str, Any]) -> bool:
    """Best-effort detection of dynamic-quantized checkpoints."""

    for key in state_dict.keys():
        if "._packed_params" in key or key.endswith(".scale") or key.endswith(".zero_point"):
            return True
    return False


def unpack_checkpoint(state: Any) -> Tuple[dict, bool]:
    """Return a state_dict and whether it appears quantized."""

    quantized = False
    if isinstance(state, dict) and "state_dict" in state:
        quantized = bool(state.get("quantized", False))
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        quantized = bool(state.get("quantized", False))
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint did not contain a state_dict.")
    if not quantized:
        quantized = state_dict_is_quantized(state)
    return state, quantized


def apply_dynamic_quantization(model: nn.Module, dtype: torch.dtype = torch.qint8) -> nn.Module:
    """Apply dynamic int8 quantization to LSTM/Linear layers."""

    model.eval()
    return torch.quantization.quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=dtype)


def normalize_checkpoint_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


__all__ = [
    "SUPPORTED_ENGINES",
    "set_quantized_engine",
    "state_dict_is_quantized",
    "unpack_checkpoint",
    "apply_dynamic_quantization",
    "normalize_checkpoint_path",
]
