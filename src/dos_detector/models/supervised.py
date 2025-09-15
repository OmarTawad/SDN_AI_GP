"""Supervised sequence detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..config.types import SupervisedModelConfig
from .attention import TemporalAttention


@dataclass
class SequenceClassifierOutput:
    """Outputs produced by :class:`SequenceClassifier`."""

    window_logits: torch.Tensor
    type_logits: torch.Tensor
    file_logits: torch.Tensor
    attention: Optional[torch.Tensor]


class SequenceClassifier(nn.Module):
    """BiLSTM/GRU-based DoS detector."""

    def __init__(
        self,
        input_size: int,
        num_attack_types: int,
        config: SupervisedModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_dropout = nn.Dropout(config.input_dropout)
        rnn_cls = nn.LSTM if config.rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size,
            config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )
        rnn_dim = config.hidden_size * (2 if config.bidirectional else 1)
        self.post_rnn_dropout = nn.Dropout(config.dropout)
        self.attention = TemporalAttention(rnn_dim, config.hidden_size) if config.attention else None
        self.binary_head = nn.Linear(rnn_dim, 1)
        self.type_head = nn.Linear(rnn_dim, num_attack_types)
        self.sequence_head = nn.Linear(rnn_dim, 1)

    def forward(self, features: torch.Tensor) -> SequenceClassifierOutput:
        features = self.input_dropout(features)
        outputs, _ = self.rnn(features)
        outputs = self.post_rnn_dropout(outputs)
        if self.attention is not None:
            context, weights = self.attention(outputs)
        else:
            context = outputs.mean(dim=1)
            weights = None
        window_logits = self.binary_head(outputs).squeeze(-1)
        type_logits = self.type_head(outputs)
        file_logits = self.sequence_head(context).squeeze(-1)
        return SequenceClassifierOutput(
            window_logits=window_logits,
            type_logits=type_logits,
            file_logits=file_logits,
            attention=weights,
        )


__all__ = ["SequenceClassifier", "SequenceClassifierOutput"]
