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


def _remap_expert_axes(state_dict: Mapping[str, torch.Tensor], keep_indices, target_num_experts: int):
    """Remap expert rows/cols when loading legacy checkpoints."""

    remapped = dict(state_dict)
    fc2_w_key = "gating.fc2.weight"
    fc2_b_key = "gating.fc2.bias"
    cls_w_key = "classifier.weight"

    if fc2_w_key in remapped and remapped[fc2_w_key].dim() == 2:
        weights = remapped[fc2_w_key]
        if weights.size(0) > target_num_experts:
            valid_indices = tuple(idx for idx in keep_indices if idx < weights.size(0))
            if len(valid_indices) < target_num_experts:
                valid_indices = tuple(range(target_num_experts))
            remapped[fc2_w_key] = weights[valid_indices[:target_num_experts], :]
    if fc2_b_key in remapped and remapped[fc2_b_key].dim() == 1:
        bias = remapped[fc2_b_key]
        if bias.numel() > target_num_experts:
            valid_indices = tuple(idx for idx in keep_indices if idx < bias.numel())
            if len(valid_indices) < target_num_experts:
                valid_indices = tuple(range(target_num_experts))
            remapped[fc2_b_key] = bias[valid_indices[:target_num_experts]]
    if cls_w_key in remapped and remapped[cls_w_key].dim() == 2:
        cls_w = remapped[cls_w_key]
        if cls_w.size(1) > target_num_experts:
            valid_indices = tuple(idx for idx in keep_indices if idx < cls_w.size(1))
            if len(valid_indices) < target_num_experts:
                valid_indices = tuple(range(target_num_experts))
            remapped[cls_w_key] = cls_w[:, valid_indices[:target_num_experts]]
    return remapped


def _align_state_dict(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    """Trim or filter checkpoint weights to match the CNN+autoencoder architecture."""

    target_state = model.state_dict()
    # Legacy 5-expert ordering: auto, DoS CNN, DoS LSTM, ARP CNN, ARP LSTM
    keep_indices = (0, 1, 3)
    pruned = _remap_expert_axes(
        state_dict,
        keep_indices=keep_indices[: len(model.experts)],
        target_num_experts=len(model.experts),
    )

    aligned = {}
    for key, value in pruned.items():
        if key.startswith("experts."):
            # Rely on the frozen source checkpoints rather than legacy saved weights.
            continue
        target = target_state.get(key)
        if target is None:
            continue
        if target.shape != value.shape:
            continue
        aligned[key] = value
    return aligned


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
        filtered_state = _align_state_dict(model, state_dict)
        model.load_state_dict(filtered_state, strict=False)  # type: ignore[arg-type]
    except RuntimeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Failed to load unified MoE checkpoint. Ensure the frozen experts and gating hidden "
            "size match the saved weights."
        ) from exc
    model.eval()
    return model


__all__ = ["load_model"]
