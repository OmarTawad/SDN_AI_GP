"""Sequence autoencoder for anomaly detection."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from ..config.types import AutoencoderModelConfig


class LSTMAutoencoder(nn.Module):
    """LSTM encoder-decoder trained on normal traffic."""

    def __init__(self, input_size: int, config: AutoencoderModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.LSTM(
            input_size,
            config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.latent_proj = nn.Linear(config.hidden_size, config.latent_size)
        self.decoder_init = nn.Linear(config.latent_size, config.hidden_size)
        self.decoder = nn.LSTM(
            config.latent_size,
            config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(config.hidden_size, input_size)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (hidden, _cell) = self.encoder(features)
        latent = torch.tanh(self.latent_proj(hidden[-1]))
        seq_len = features.shape[1]
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)
        init_hidden = torch.tanh(self.decoder_init(latent)).unsqueeze(0).repeat(self.config.num_layers, 1, 1)
        init_cell = torch.zeros_like(init_hidden)
        decoded, _ = self.decoder(decoder_input, (init_hidden, init_cell))
        reconstruction = self.output_layer(decoded)
        return reconstruction, latent

    @staticmethod
    def reconstruction_error(inputs: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        """Mean squared error per window."""

        return torch.mean((inputs - reconstruction) ** 2, dim=-1)


__all__ = ["LSTMAutoencoder"]
