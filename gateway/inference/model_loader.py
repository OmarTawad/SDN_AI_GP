"""Model loading utilities for inference.


"""

from __future__ import annotations

from pathlib import Path

import torch

from gateway.models.unified_moe import build_unified_moe


def load_model(checkpoint: Path, gating_hidden_override: int | None) -> torch.nn.Module:
    """Load the unified MoE model from the specified checkpoint.

    Args:
        checkpoint: Filesystem path to the checkpoint to be deserialised.
        gating_hidden_override: Optional override for the gating hidden size.

    Returns:
        torch.nn.Module: Loaded unified MoE model in evaluation mode.
    """

    payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        gating_hidden = gating_hidden_override or int(payload.get("gating_hidden", 128))
        state_dict = payload["state_dict"]
    else:
        gating_hidden = gating_hidden_override or 128
        state_dict = payload
    model = build_unified_moe(device=torch.device("cpu"), gating_hidden_dim=gating_hidden)
    model.load_state_dict(state_dict)
    model.eval()
    return model


__all__ = ["load_model"]
