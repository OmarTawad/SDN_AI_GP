"""Quantization helpers for CPU deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Tuple
import warnings

import torch
from torch import nn
from torch.ao.quantization.quantization_mappings import get_default_dynamic_quant_module_mappings

SUPPORTED_ENGINES = {"fbgemm", "qnnpack"}


class _LinearDynamicNoReduceRange(torch.ao.nn.quantized.dynamic.Linear):
    """Dynamic quantized Linear without reduce_range to silence qnnpack warnings."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self._packed_params.dtype == torch.qint8:
            y = torch.ops.quantized.linear_dynamic(
                x, self._packed_params._packed_params, reduce_range=False
            )
        elif self._packed_params.dtype == torch.float16:
            y = torch.ops.quantized.linear_dynamic_fp16(x, self._packed_params._packed_params)
        else:
            raise RuntimeError("Unsupported dtype on dynamic quantized linear!")
        return y.to(x.dtype)


def set_quantized_engine(backend: str | None) -> str:
    """Set the quantized backend engine if requested."""

    if not backend:
        return torch.backends.quantized.engine
    normalized = str(backend).lower()
    supported = set(torch.backends.quantized.supported_engines or [])
    if normalized not in supported:
        fallback = "qnnpack" if "qnnpack" in supported else (next(iter(supported), None))
        if fallback is None:
            raise RuntimeError("No quantized engine is available in this PyTorch build.")
        warnings.warn(
            f"Quantized engine '{backend}' is not supported; falling back to '{fallback}'.",
            RuntimeWarning,
            stacklevel=2,
        )
        normalized = fallback
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
    mapping = None
    engine = str(torch.backends.quantized.engine).lower()
    if engine == "qnnpack":
        mapping = dict(get_default_dynamic_quant_module_mappings())
        mapping[nn.Linear] = _LinearDynamicNoReduceRange
        if hasattr(nn, "NonDynamicallyQuantizableLinear"):
            mapping[nn.NonDynamicallyQuantizableLinear] = _LinearDynamicNoReduceRange
    return torch.quantization.quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=dtype, mapping=mapping)


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
