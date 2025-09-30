"""Helper module exposing ready-to-instantiate models for profiling."""

from __future__ import annotations

import sys
from typing import Tuple
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parents[1]
_NEURAL_LSTM_SRC = _BASE_DIR / "Neural_LSTM" / "src"
if str(_NEURAL_LSTM_SRC) not in sys.path:
    sys.path.append(str(_NEURAL_LSTM_SRC))

import torch

try:  # pragma: no cover - optional dependency handling
    from dos_detector.models.supervised import SequenceClassifier as _SequenceClassifier
    from dos_detector.config.types import SupervisedModelConfig
except Exception as exc:  # type: ignore
    raise ImportError("Failed to import SequenceClassifier from Neural_LSTM project") from exc


class SequenceClassifierDefault(_SequenceClassifier):
    """Sequence classifier with default configuration for profiling purposes."""

    def __init__(self) -> None:
        config = SupervisedModelConfig(
            input_dropout=0.1,
            rnn_type="lstm",
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            attention=True,
            attention_heads=4,
        )
        super().__init__(
            input_size=50,
            num_attack_types=7,
            config=config,
        )


class SequenceClassifierProfile(SequenceClassifierDefault):
    """Wrap the default classifier to expose tensor outputs for profiling."""

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        outputs = super().forward(features)
        return outputs.window_logits, outputs.type_logits, outputs.file_logits, outputs.attention


__all__ = ["SequenceClassifierDefault", "SequenceClassifierProfile"]
