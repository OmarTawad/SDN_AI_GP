"""Gating vector construction utilities for the unified MoE datasets.


"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor

from gateway.data.extractors.features import (
    ARP_MICRO_BINS,
    DOS_MICRO_BINS,
)
from gateway.moe_model import (
    ARP_CNN_SEQ_IN_DIM,
    ARP_CNN_STATIC_DIM,
    AUTO_FEATURE_DIM,
    DOS_CNN_SEQ_IN_DIM,
    DOS_CNN_STATIC_DIM,
)

UNIFIED_GATING_COMPONENT_KEYS: Tuple[str, ...] = (
    "auto",
    "dos_cnn_static",
    "dos_cnn_seq",
    "arp_cnn_static",
    "arp_cnn_seq",
)

GATING_COMPONENT_LENGTHS: Dict[str, int] = {
    "auto": AUTO_FEATURE_DIM,
    "dos_cnn_static": DOS_CNN_STATIC_DIM,
    "dos_cnn_seq": DOS_MICRO_BINS * DOS_CNN_SEQ_IN_DIM,
    "arp_cnn_static": ARP_CNN_STATIC_DIM,
    "arp_cnn_seq": ARP_MICRO_BINS * ARP_CNN_SEQ_IN_DIM,
}


def build_unified_gating(features: Dict[str, Tensor]) -> Tensor:
    """Concatenate gating components into the unified vector expected by the model."""

    components: List[Tensor] = []
    for key in UNIFIED_GATING_COMPONENT_KEYS:
        expected = GATING_COMPONENT_LENGTHS[key]
        tensor = features.get(key)
        if tensor is None:
            flat = torch.zeros(expected, dtype=torch.float32)
        else:
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Gating component '{key}' must be a tensor.")
            flat = tensor.reshape(-1).to(torch.float32)
            if flat.numel() != expected:
                raise ValueError(
                    f"Gating component '{key}' has {flat.numel()} elements, expected {expected}."
                )
        components.append(flat)
    if not components:
        raise ValueError("Cannot assemble unified gating vector without feature tensors.")
    return torch.cat(components, dim=0)


__all__ = [
    "GATING_COMPONENT_LENGTHS",
    "UNIFIED_GATING_COMPONENT_KEYS",
    "build_unified_gating",
]
