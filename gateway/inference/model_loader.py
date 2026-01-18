"""Model loading utilities for inference.


"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Tuple

import torch

from gateway.models.unified_moe import build_unified_moe


def _extract_state_dict(
    payload: object,
    gating_hidden_override: int | None,
) -> Tuple[int, Mapping[str, torch.Tensor]]:
    """Normalise supported checkpoint payload formats."""

    default_hidden = gating_hidden_override or 128
    if isinstance(payload, Mapping):
        if "state_dict" in payload:
            state_dict = payload["state_dict"]
            if not isinstance(state_dict, Mapping):
                raise RuntimeError("Checkpoint 'state_dict' entry is not a mapping.")
            gating_hidden = gating_hidden_override or int(payload.get("gating_hidden", default_hidden))
            return gating_hidden, state_dict
        if all(isinstance(key, str) for key in payload.keys()):
            gating_hidden = gating_hidden_override or int(payload.get("gating_hidden", default_hidden))
            return gating_hidden, payload
    raise RuntimeError(
        "Unsupported checkpoint format. Expected either a state_dict mapping or a dict "
        "containing a 'state_dict' entry."
    )


def load_model(checkpoint: Path, gating_hidden_override: int | None) -> torch.nn.Module:
    """Load the unified MoE model from the specified checkpoint.

    Args:
        checkpoint: Filesystem path to the checkpoint to be deserialised.
        gating_hidden_override: Optional override for the gating hidden size.

    Returns:
        torch.nn.Module: Loaded unified MoE model in evaluation mode.
    """

    payload = torch.load(checkpoint, map_location="cpu")
    gating_hidden, state_dict = _extract_state_dict(payload, gating_hidden_override)
    model = build_unified_moe(device=torch.device("cpu"), gating_hidden_dim=gating_hidden)
    try:
        model.load_state_dict(state_dict)  # type: ignore[arg-type]
    except RuntimeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Failed to load unified MoE checkpoint. Ensure the frozen experts and gating hidden "
            "size match the saved weights."
        ) from exc
    model.eval()
    return model


__all__ = ["load_model"]
