# features/scaler.py
from __future__ import annotations
import os
import pickle
from typing import List, Optional
import numpy as np


class RobustScaler:
    """
    Lightweight robust scaler:
      x_scaled = (x - median) / (IQR + eps)

    Persistence:
      - Instance .save(dir) writes a small dict payload (fmt=dict_v1) to <dir>/scaler.pkl
      - Classmethod .load(dir) loads either:
          * our dict_v1 (new)
          * older pickled RobustScaler instance (ours)
          * scikit-learn RobustScaler (center_/scale_)
          * generic object with similar attributes
    """

    def __init__(self, eps: float = 1e-9):
        self.eps = float(eps)
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None
        self.fitted_: bool = False

    # -------- fit / transform --------
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"RobustScaler.fit expects 2D array, got shape {X.shape}")
        med = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0  # avoid zero division
        self.median_ = med.astype(np.float32)
        self.iqr_ = iqr.astype(np.float32)
        self.n_features_ = X.shape[1]
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"f_{i}" for i in range(self.n_features_)]
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("RobustScaler not fitted.")
        X = np.asarray(X, dtype=np.float32)
        flat = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            flat = True
        if X.shape[-1] != self.n_features_:
            raise ValueError(f"Feature dimension mismatch: got {X.shape[-1]}, expected {self.n_features_}")
        eps = getattr(self, "eps", 1e-9)  # default if missing in legacy pickles
        Xs = (X - self.median_) / (self.iqr_ + eps)
        return Xs.squeeze(0) if flat else Xs

    # -------- persistence --------
    def save(self, dirpath: str):
        """Save to <dirpath>/scaler.pkl as a dict payload."""
        os.makedirs(dirpath, exist_ok=True)
        payload = {
            "fmt": "dict_v1",
            "eps": getattr(self, "eps", 1e-9),
            "median_": self.median_,
            "iqr_": self.iqr_,
            "n_features_": self.n_features_,
            "feature_names_": self.feature_names_,
            "fitted_": self.fitted_,
        }
        with open(os.path.join(dirpath, "scaler.pkl"), "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, dirpath: str) -> "RobustScaler":
        """
        Load from <dirpath>/scaler.pkl.

        Compatible with:
          1) dict payload (new)
          2) older pickled our RobustScaler instance
          3) scikit-learn RobustScaler (center_/scale_)
          4) generic object with similar attrs
        """
        path = os.path.join(dirpath, "scaler.pkl")
        with open(path, "rb") as f:
            payload = pickle.load(f)

        # Case 1: already our instance
        if isinstance(payload, RobustScaler):
            # ensure missing attrs exist
            if not hasattr(payload, "eps"): payload.eps = 1e-9
            if not hasattr(payload, "fitted_"): payload.fitted_ = True
            return payload

        # Case 2: dict payload (new)
        if isinstance(payload, dict) and "median_" in payload and "iqr_" in payload:
            obj = cls()  # avoid passing args for max compat
            obj.eps = float(payload.get("eps", 1e-9))
            obj.median_ = np.asarray(payload["median_"], dtype=np.float32)
            obj.iqr_ = np.asarray(payload["iqr_"], dtype=np.float32)
            obj.n_features_ = int(payload.get("n_features_", obj.median_.shape[0]))
            obj.feature_names_ = payload.get("feature_names_")
            obj.fitted_ = bool(payload.get("fitted_", True))
            return obj

        # Case 3: sklearn RobustScaler
        try:
            from sklearn.preprocessing import RobustScaler as SkRobust  # type: ignore
        except Exception:
            SkRobust = None
        if SkRobust is not None and isinstance(payload, SkRobust):
            obj = cls()
            center = getattr(payload, "center_", None)
            scale = getattr(payload, "scale_", None)
            if center is None or scale is None:
                raise TypeError("sklearn RobustScaler payload missing center_/scale_.")
            center = np.asarray(center, dtype=np.float32)
            scale = np.asarray(scale, dtype=np.float32)
            scale[scale == 0] = 1.0
            obj.median_ = center
            obj.iqr_ = scale
            obj.n_features_ = int(center.shape[0])
            obj.feature_names_ = None
            obj.fitted_ = True
            return obj

        # Case 4: generic object with similar attrs
        try:
            obj = cls()
            obj.eps = float(getattr(payload, "eps", 1e-9))
            med = getattr(payload, "median_", None)
            iqr = getattr(payload, "iqr_", None)
            nfeat = getattr(payload, "n_features_", None)
            if med is None or iqr is None:
                raise ValueError("Payload missing median_/iqr_.")
            med = np.asarray(med, dtype=np.float32)
            iqr = np.asarray(iqr, dtype=np.float32)
            iqr[iqr == 0] = 1.0
            obj.median_ = med
            obj.iqr_ = iqr
            obj.n_features_ = int(nfeat) if nfeat is not None else int(med.shape[0])
            obj.feature_names_ = getattr(payload, "feature_names_", None)
            obj.fitted_ = bool(getattr(payload, "fitted_", True))
            return obj
        except Exception as e:
            raise TypeError(f"Unsupported scaler.pkl format: {type(payload)}") from e

    @staticmethod
    def save_instance(scaler: "RobustScaler", dirpath: str) -> None:
        if not isinstance(scaler, RobustScaler):
            raise TypeError("Expected RobustScaler instance")
        scaler.save(dirpath)
