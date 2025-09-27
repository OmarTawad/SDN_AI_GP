from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


def activation_factory(name: str):
    name_lower = name.lower()
    if name_lower == "relu":
        return lambda: nn.ReLU()
    if name_lower == "leakyrelu":
        return lambda: nn.LeakyReLU(negative_slope=0.01)
    if name_lower == "elu":
        return lambda: nn.ELU()
    if name_lower == "gelu":
        return lambda: nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class FeedForwardAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Iterable[int],
        latent_dim: int,
        activation: str = "ReLU",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = list(hidden_layers)
        if not hidden:
            raise ValueError("hidden_layers must not be empty")

        layers: List[nn.Module] = []
        prev_dim = input_dim
        activation_ctor = activation_factory(activation)
        for width in hidden:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(activation_ctor())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = width

        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

        decoder_layers: List[nn.Module] = []
        prev_dim = latent_dim
        reversed_hidden = list(hidden)[::-1]
        for width in reversed_hidden:
            decoder_layers.append(nn.Linear(prev_dim, width))
            decoder_layers.append(activation_ctor())
            prev_dim = width

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encoder(x)
        return self.decoder(latent)


@dataclass
class ModelConfig:
    input_dim: int
    hidden: List[int]
    latent_dim: int
    activation: str
    dropout: float


def build_model(model_cfg: dict, input_dim: int) -> FeedForwardAutoencoder:
    hidden = model_cfg.get("hidden", [128, 64, 32])
    latent_dim = int(model_cfg.get("latent_dim", 16))
    activation = model_cfg.get("activation", "ReLU")
    dropout = float(model_cfg.get("dropout", 0.0))

    return FeedForwardAutoencoder(
        input_dim=input_dim,
        hidden_layers=hidden,
        latent_dim=latent_dim,
        activation=activation,
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
