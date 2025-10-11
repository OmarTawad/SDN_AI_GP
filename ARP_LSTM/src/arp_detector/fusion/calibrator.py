"""Score fusion via logistic calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..utils.io import load_joblib, save_joblib


@dataclass
class FusionSample:
    features: Sequence[float]
    label: int


class ScoreFusion:
    """Combine detector scores into a final probability."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = list(feature_names)
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, samples: Sequence[FusionSample]) -> None:
        if not samples:
            raise ValueError("No samples provided for fusion calibration")
        X = np.array([sample.features for sample in samples], dtype=float)
        y = np.array([sample.label for sample in samples], dtype=int)
        self.model.fit(X, y)

    def predict_proba(self, features: Sequence[float]) -> float:
        X = np.array(features, dtype=float).reshape(1, -1)
        return float(self.model.predict_proba(X)[0, 1])

    def save(self, path: Path) -> None:
        save_joblib(path, {"feature_names": self.feature_names, "model": self.model})

    @classmethod
    def load(cls, path: Path) -> "ScoreFusion":
        payload = load_joblib(path)
        fusion = cls(payload["feature_names"])
        fusion.model = payload["model"]
        return fusion

__all__ = ["ScoreFusion", "FusionSample"]