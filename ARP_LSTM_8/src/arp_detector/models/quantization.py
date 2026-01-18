from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


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


class QuantizedLSTM(nn.LSTM):
    """
    LSTM that fake-quantizes inputs, weights, and outputs to int8 ranges.
    """

    def __init__(self, *args, quantizer: Optional[Int8Quantizer] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quantizer = quantizer or Int8Quantizer()

    @classmethod
    def from_lstm(cls, lstm: nn.LSTM, quantizer: Optional[Int8Quantizer] = None) -> "QuantizedLSTM":
        qlstm = cls(
            lstm.input_size,
            lstm.hidden_size,
            lstm.num_layers,
            bias=lstm.bias,
            batch_first=lstm.batch_first,
            dropout=lstm.dropout,
            bidirectional=lstm.bidirectional,
            proj_size=getattr(lstm, "proj_size", 0),
            quantizer=quantizer,
        )
        qlstm.load_state_dict(lstm.state_dict())
        return qlstm

    def forward(self, input: torch.Tensor, hx: Optional[tuple[torch.Tensor, torch.Tensor]] = None):  # type: ignore[override]
        if not torch.jit.is_scripting():
            if self._weights_have_changed():
                self._init_flat_weights()

        if isinstance(input, PackedSequence):
            raise TypeError("PackedSequence is not supported for int8-quantized LSTM.")

        assert input.dim() in (2, 3), (
            f"LSTM: Expected input to be 2-D or 3-D but received {input.dim()}-D tensor"
        )
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        if not is_batched:
            input = input.unsqueeze(batch_dim)

        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = self.proj_size if getattr(self, "proj_size", 0) > 0 else self.hidden_size
            h_zeros = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                real_hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            c_zeros = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = (h_zeros, c_zeros)
        else:
            if is_batched:
                if hx[0].dim() != 3 or hx[1].dim() != 3:
                    raise RuntimeError(
                        "For batched 3-D input, hx and cx should also be 3-D tensors"
                    )
            else:
                if hx[0].dim() != 2 or hx[1].dim() != 2:
                    raise RuntimeError(
                        "For unbatched 2-D input, hx and cx should also be 2-D tensors"
                    )
                hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))

        hx = self.permute_hidden(hx, None)
        self.check_forward_args(input, hx, None)

        input_q = self.quantizer.fake_quant(input)
        h_q = self.quantizer.fake_quant(hx[0])
        c_q = self.quantizer.fake_quant(hx[1])
        flat_weights_q = [self.quantizer.fake_quant(w) for w in self._flat_weights]

        result = torch._VF.lstm(
            input_q,
            (h_q, c_q),
            flat_weights_q,
            self.bias,
            self.num_layers,
            self.dropout,
            self.training,
            self.bidirectional,
            self.batch_first,
        )
        output = self.quantizer.fake_quant(result[0])
        h_n = self.quantizer.fake_quant(result[1])
        c_n = self.quantizer.fake_quant(result[2])

        if not is_batched:
            output = output.squeeze(batch_dim)
            h_n = h_n.squeeze(1)
            c_n = c_n.squeeze(1)

        return output, self.permute_hidden((h_n, c_n), None)


class QuantizedGRU(nn.GRU):
    """
    GRU that fake-quantizes inputs, weights, and outputs to int8 ranges.
    """

    def __init__(self, *args, quantizer: Optional[Int8Quantizer] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quantizer = quantizer or Int8Quantizer()

    @classmethod
    def from_gru(cls, gru: nn.GRU, quantizer: Optional[Int8Quantizer] = None) -> "QuantizedGRU":
        qgru = cls(
            gru.input_size,
            gru.hidden_size,
            gru.num_layers,
            bias=gru.bias,
            batch_first=gru.batch_first,
            dropout=gru.dropout,
            bidirectional=gru.bidirectional,
            quantizer=quantizer,
        )
        qgru.load_state_dict(gru.state_dict())
        return qgru

    def forward(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None):  # type: ignore[override]
        if not torch.jit.is_scripting():
            if self._weights_have_changed():
                self._init_flat_weights()

        if isinstance(input, PackedSequence):
            raise TypeError("PackedSequence is not supported for int8-quantized GRU.")

        assert input.dim() in (2, 3), (
            f"GRU: Expected input to be 2-D or 3-D but received {input.dim()}-D tensor"
        )
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if hx is not None:
                if hx.dim() != 2:
                    raise RuntimeError("For unbatched 2-D input, hx should also be 2-D tensor")
                hx = hx.unsqueeze(1)
        else:
            if hx is not None and hx.dim() != 3:
                raise RuntimeError("For batched 3-D input, hx should also be 3-D tensor")

        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
        else:
            hx = self.permute_hidden(hx, None)

        self.check_forward_args(input, hx, None)

        input_q = self.quantizer.fake_quant(input)
        hx_q = self.quantizer.fake_quant(hx)
        flat_weights_q = [self.quantizer.fake_quant(w) for w in self._flat_weights]

        result = torch._VF.gru(
            input_q,
            hx_q,
            flat_weights_q,
            self.bias,
            self.num_layers,
            self.dropout,
            self.training,
            self.bidirectional,
            self.batch_first,
        )
        output = self.quantizer.fake_quant(result[0])
        hidden = self.quantizer.fake_quant(result[1])

        if not is_batched:
            output = output.squeeze(batch_dim)
            hidden = hidden.squeeze(1)

        return output, self.permute_hidden(hidden, None)


def convert_module_to_int8(module: nn.Module, quantizer: Optional[Int8Quantizer] = None) -> nn.Module:
    """
    Recursively swap Conv1d/Linear/LSTM/GRU layers with quantized variants.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (QuantizedConv1d, QuantizedLinear, QuantizedLSTM, QuantizedGRU)):
            convert_module_to_int8(child, quantizer)
        elif isinstance(child, nn.Conv1d):
            setattr(module, name, QuantizedConv1d.from_conv(child, quantizer))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QuantizedLinear.from_linear(child, quantizer))
        elif isinstance(child, nn.LSTM):
            setattr(module, name, QuantizedLSTM.from_lstm(child, quantizer))
        elif isinstance(child, nn.GRU):
            setattr(module, name, QuantizedGRU.from_gru(child, quantizer))
        else:
            convert_module_to_int8(child, quantizer)
    return module


__all__ = [
    "Int8Quantizer",
    "QuantizedConv1d",
    "QuantizedLinear",
    "QuantizedLSTM",
    "QuantizedGRU",
    "convert_module_to_int8",
]
