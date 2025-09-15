"""Attention mechanisms for temporal aggregation."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class TemporalAttention(nn.Module):
    """Additive attention over time."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return context vector and attention weights.

        Args:
            inputs: Tensor of shape (batch, time, features).
        Returns:
            context: (batch, features)
            weights: (batch, time)
        """

        transformed = torch.tanh(self.query(inputs) + self.key(inputs))
        scores = self.score(transformed).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.einsum("btd,bt->bd", inputs, weights)
        return context, weights


__all__ = ["TemporalAttention"]
