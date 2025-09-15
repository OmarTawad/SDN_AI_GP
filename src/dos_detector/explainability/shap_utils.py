"""Optional SHAP utilities for sequence models."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

try:  # pragma: no cover - optional dependency
    import shap
except Exception:  # pragma: no cover
    shap = None  # type: ignore


def approximate_sequence_shap(
    model: torch.nn.Module,
    sample: torch.Tensor,
    background: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """Approximate SHAP values for a single sequence.

    Falls back to gradient-based attribution if SHAP is unavailable.
    """

    model.eval()
    sample = sample.detach().clone().requires_grad_(True)
    background = background.detach().clone()

    def _forward(x: torch.Tensor) -> torch.Tensor:
        output = model(x)
        logits = getattr(output, "file_logits", output)
        return torch.sigmoid(logits)

    if shap is not None:
        try:
            explainer = shap.GradientExplainer(lambda x: _forward(x).detach(), background)
            shap_values = explainer.shap_values(sample)
            values = np.array(shap_values)[0]
            return {"values": values, "mean_abs": np.mean(np.abs(values), axis=0)}
        except Exception:  # pragma: no cover
            pass

    output = _forward(sample)
    output.backward(torch.ones_like(output))
    gradients = sample.grad.detach().cpu().numpy()
    return {
        "values": gradients,
        "mean_abs": np.mean(np.abs(gradients), axis=0),
    }


__all__ = ["approximate_sequence_shap"]
