from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
