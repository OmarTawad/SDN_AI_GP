from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.quantization_mappings import get_default_dynamic_quant_module_mappings


class Int8Quantizer:
    """
    Lightweight fake-quantizer that keeps tensors on CUDA while constraining
    values to an int8 grid. The straight-through estimator keeps gradients
    flowing as if the quantization step were identity.
    """

    def __init__(self, qmin: int = -128, qmax: int = 127, eps: float = 1e-8) -> None:
        self.qmin = int(qmin)
        self.qmax = int(qmax)
        self.eps = float(eps)

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        max_val = torch.max(x.detach().abs())
        scale = max_val / float(self.qmax)
        return torch.clamp(scale, min=self.eps)

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize-dequantize with STE so autograd treats the op as identity.
        """
        if not torch.is_floating_point(x):
            x = x.float()
        scale = self._scale(x)
        q = torch.clamp(torch.round(x / scale), self.qmin, self.qmax)
        deq = q * scale
        return x + (deq - x).detach()


class QuantizedConv1d(nn.Conv1d):
    """
    Conv1d that fake-quantizes inputs, weights, and outputs to int8 ranges.
    """

    def __init__(self, *args, quantizer: Optional[Int8Quantizer] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quantizer = quantizer or Int8Quantizer()

    @classmethod
    def from_conv(cls, conv: nn.Conv1d, quantizer: Optional[Int8Quantizer] = None) -> "QuantizedConv1d":
        qconv = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            quantizer=quantizer,
        )
        qconv.load_state_dict(conv.state_dict())
        return qconv

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x_q = self.quantizer.fake_quant(x)
        w_q = self.quantizer.fake_quant(self.weight)
        out = F.conv1d(
            x_q,
            w_q,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return self.quantizer.fake_quant(out)


class QuantizedLinear(nn.Linear):
    """
    Linear layer with fake int8 quantization on inputs/weights/outputs.
    """

    def __init__(self, *args, quantizer: Optional[Int8Quantizer] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quantizer = quantizer or Int8Quantizer()

    @classmethod
    def from_linear(cls, linear: nn.Linear, quantizer: Optional[Int8Quantizer] = None) -> "QuantizedLinear":
        qlin = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            quantizer=quantizer,
        )
        qlin.load_state_dict(linear.state_dict())
        return qlin

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x_q = self.quantizer.fake_quant(x)
        w_q = self.quantizer.fake_quant(self.weight)
        out = F.linear(x_q, w_q, self.bias)
        return self.quantizer.fake_quant(out)


def convert_module_to_int8(module: nn.Module, quantizer: Optional[Int8Quantizer] = None) -> nn.Module:
    """
    Recursively swap Conv1d/Linear layers with quantized variants.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (QuantizedConv1d, QuantizedLinear)):
            convert_module_to_int8(child, quantizer)
        elif isinstance(child, nn.Conv1d):
            setattr(module, name, QuantizedConv1d.from_conv(child, quantizer))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QuantizedLinear.from_linear(child, quantizer))
        else:
            convert_module_to_int8(child, quantizer)
    return module


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
    """Apply dynamic int8 quantization to supported modules."""

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
