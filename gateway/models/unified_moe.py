"""Unified Mixture-of-Experts architecture definition.


"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from gateway.moe_model import (  # type: ignore[import]
    ARP_CNN_SPEC,
    ARP_LSTM_SPEC,
    ARP_CNN_STATIC_DIM,
    ARP_SEQ_FEATURE_DIM,
    ARP_LSTM_FEATURE_DIM,
    AUTOENCODER_SPEC,
    AUTO_FEATURE_DIM,
    DOS_CNN_SPEC,
    DOS_CNN_STATIC_DIM,
    DOS_LSTM_SPEC,
    DOS_SEQ_FEATURE_DIM,
    DOS_LSTM_FEATURE_DIM,
    FrozenExpert,
    load_frozen_expert,
)

UNIFIED_EXPERT_SPECS: Tuple = (
    AUTOENCODER_SPEC,
    DOS_CNN_SPEC,
    DOS_LSTM_SPEC,
    ARP_CNN_SPEC,
    ARP_LSTM_SPEC,
)
NUM_EXPERTS: int = len(UNIFIED_EXPERT_SPECS)
NUM_CLASSES: int = 3
UNIFIED_GATING_INPUT_DIM: int = (
    AUTO_FEATURE_DIM
    + DOS_CNN_STATIC_DIM
    + DOS_SEQ_FEATURE_DIM
    + DOS_LSTM_FEATURE_DIM
    + ARP_CNN_STATIC_DIM
    + ARP_SEQ_FEATURE_DIM
    + ARP_LSTM_FEATURE_DIM
)
DEFAULT_GATING_HIDDEN: int = 128


class UnifiedGating(nn.Module):
    """Two-layer gating network producing a distribution over frozen experts."""

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        logits = self.fc2(self.act(self.fc1(features)))
        weights = torch.softmax(logits, dim=-1)
        return weights, logits


class UnifiedMoE(nn.Module):
    """Unified Mixture-of-Experts with a single gating network and multiclass head."""

    def __init__(
        self,
        experts: Sequence[FrozenExpert],
        gating: UnifiedGating,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        if not experts:
            raise ValueError("UnifiedMoE requires at least one expert.")
        self.experts = nn.ModuleList(experts)
        self.gating = gating
        self.classifier = nn.Linear(len(experts), num_classes)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare_features(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        prepared: Dict[str, Tensor] = {}
        for key, value in features.items():
            if isinstance(value, Tensor):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        return prepared

    def forward(
        self,
        features: Dict[str, Tensor],
        return_attention: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]] | Tensor:
        if not isinstance(features, dict):
            raise TypeError("UnifiedMoE expects a feature dictionary.")
        device_features = self._prepare_features(features)
        if "gating_input" not in device_features:
            raise KeyError("Feature dict must include 'gating_input' for gating network.")

        gate_weights, gate_logits = self.gating(device_features["gating_input"])
        expert_outputs: List[Tensor] = []
        for expert in self.experts:
            output = expert(device_features)
            if output.dim() > 1:
                output = output.view(output.size(0), -1).mean(dim=1)
            expert_outputs.append(output.to(gate_weights.dtype))
        expert_matrix = torch.stack(expert_outputs, dim=1)
        weighted = expert_matrix * gate_weights
        logits = self.classifier(weighted)
        if return_attention:
            attention = {
                "weights": gate_weights,
                "gate_logits": gate_logits,
                "expert_outputs": expert_matrix,
            }
            return logits, attention
        return logits

    def train(self, mode: bool = True) -> "UnifiedMoE":  # type: ignore[override]
        super().train(mode)
        for expert in self.experts:
            expert.eval()
        return self


def build_unified_moe(
    device: Optional[torch.device] = None,
    gating_hidden_dim: int = DEFAULT_GATING_HIDDEN,
    expert_specs: Sequence = UNIFIED_EXPERT_SPECS,
) -> UnifiedMoE:
    """Construct the unified Mixture-of-Experts model on the requested device."""

    target_device = device or torch.device("cpu")
    experts = [load_frozen_expert(spec, device=target_device) for spec in expert_specs]
    gating = UnifiedGating(
        input_dim=UNIFIED_GATING_INPUT_DIM,
        hidden_dim=gating_hidden_dim,
        num_experts=len(experts),
    ).to(target_device)
    model = UnifiedMoE(experts=experts, gating=gating).to(target_device)
    return model


__all__ = [
    "DEFAULT_GATING_HIDDEN",
    "NUM_CLASSES",
    "NUM_EXPERTS",
    "UNIFIED_EXPERT_SPECS",
    "UNIFIED_GATING_INPUT_DIM",
    "UnifiedGating",
    "UnifiedMoE",
    "build_unified_moe",
]

