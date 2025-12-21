"""Runtime adapter that exposes a clean Python API for the unified MoE."""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import torch
import yaml

from gateway.core import PATHS, class_id_to_name, CLASS_LABELS
from gateway.inference.model_loader import load_model
from gateway.models.unified_moe import UNIFIED_EXPERT_SPECS
from gateway.utils import get_logger
from gateway.data.datasets.gating import build_unified_gating

LOGGER = get_logger("dynamic_moe.adapter")


@dataclass(frozen=True)
class GatewayAdapterConfig:
    """Configuration payload consumed by :class:`DynamicMoEGateway`."""

    checkpoint: Path
    gating_hidden: int
    attack_thresholds: Dict[str, float]
    device: torch.device


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PATHS.project_root / path
    return path


def _load_config(config_path: str | Path | None) -> GatewayAdapterConfig:
    path = Path(config_path or (PATHS.gateway_root / "config_dynamic.yaml"))
    if not path.is_absolute():
        path = PATHS.project_root / path
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        LOGGER.warning("Dynamic MoE config %s missing; using defaults.", path)
        data = {}
    checkpoint = _resolve_path(data.get("checkpoint", PATHS.gateway_root / "unified_moe.pt"))
    gating_hidden = int(data.get("gating_hidden", 128))
    thresholds_raw: Mapping[str, float] = data.get("attack_thresholds", {})
    threshold_default = float(data.get("default_attack_threshold", 0.6))
    env_override = os.environ.get("DYNAMIC_MOE_DEFAULT_THRESHOLD")
    if env_override is not None:
        threshold_default = float(env_override)
    attack_thresholds: Dict[str, float] = {label: float(thresholds_raw.get(label, threshold_default)) for label in CLASS_LABELS}
    for label in CLASS_LABELS:
        env_key = f"DYNAMIC_MOE_THRESHOLD_{label.upper()}"
        if env_key in os.environ:
            attack_thresholds[label] = float(os.environ[env_key])
    attack_thresholds.setdefault("default", threshold_default)
    device = torch.device(str(data.get("device", "cpu")))
    return GatewayAdapterConfig(
        checkpoint=checkpoint,
        gating_hidden=gating_hidden,
        attack_thresholds=attack_thresholds,
        device=device,
    )


class DynamicMoEGateway:
    """Thread-safe inference wrapper used by the Ryu controller."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config = _load_config(config_path)
        self.logger = LOGGER
        self.model = load_model(self.config.checkpoint, self.config.gating_hidden).to(self.config.device)
        self.expert_names = [spec.name for spec in UNIFIED_EXPERT_SPECS]
        self._lock = threading.Lock()
        self.logger.info(
            "DynamicMoEGateway initialised | checkpoint=%s device=%s",
            self.config.checkpoint,
            self.config.device,
        )

    def _ensure_gating(self, features: MutableMapping[str, torch.Tensor]) -> None:
        if "gating_input" in features:
            return
        gating_vec = build_unified_gating(features)
        features["gating_input"] = gating_vec

    def _batchify(self, features: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}
        for key, value in features.items():
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
            tensor = tensor.to(self.config.device).to(torch.float32)
            batch[key] = tensor.unsqueeze(0)
        return batch

    def predict(self, feature_window: Mapping[str, torch.Tensor]) -> Dict[str, object]:
        """Run a single-window inference call."""

        if not feature_window:
            raise ValueError("Feature dictionary is empty.")
        mutable_features: Dict[str, torch.Tensor] = dict(feature_window)
        self._ensure_gating(mutable_features)
        batch = self._batchify(mutable_features)
        with self._lock, torch.no_grad():
            logits, attention = self.model(batch, return_attention=True)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        prob_map = {label: float(probabilities[idx].item()) for idx, label in enumerate(CLASS_LABELS)}
        prediction_id = int(probabilities.argmax().item())
        label = class_id_to_name(prediction_id)
        attack_prob = 1.0 - prob_map["normal"]
        score = max(attack_prob, float(probabilities[prediction_id].item()))
        threshold = self.config.attack_thresholds.get(label, self.config.attack_thresholds["default"])
        is_attack = label != "normal" and score >= threshold
        expert_weights = attention["weights"].squeeze(0).tolist()
        expert_votes = {name: float(weight) for name, weight in zip(self.expert_names, expert_weights)}
        response = {
            "label": label,
            "attack_type": label if label != "normal" else None,
            "is_attack": is_attack,
            "score": float(score),
            "probabilities": prob_map,
            "expert_votes": expert_votes,
            "raw_logits": logits.squeeze(0).tolist(),
        }
        return response


__all__ = ["DynamicMoEGateway"]
