# features/feature_slimming.py
from __future__ import annotations
import json, os
from typing import List
import numpy as np
from sklearn.decomposition import PCA

class StaticSlimmer:
    """
    PCA-based static feature slimming. Automatically caps n_components to the valid range:
    min(n_samples-1, n_features). Writes a small report on kept dims and variance explained.
    """
    def __init__(self, out_dim: int = 128, whiten: bool = False):
        self.req_out_dim = int(out_dim)
        self.out_dim = int(out_dim)
        self.pca = None
        self.fitted = False
        self.src_names: List[str] = []

    def fit(self, X: np.ndarray, names: List[str]):
        self.src_names = list(names)
        n_samples, n_features = X.shape
        max_dim = max(1, min(n_samples - 1, n_features))
        if self.req_out_dim > max_dim:
            # auto-cap and inform user
            print(f"[StaticSlimmer] Requested {self.req_out_dim} components but max allowed is {max_dim} "
                  f"(n_samples={n_samples}, n_features={n_features}). Using {max_dim}.")
            self.out_dim = max_dim
        else:
            self.out_dim = self.req_out_dim
        self.pca = PCA(n_components=self.out_dim, whiten=False, random_state=1337)
        self.pca.fit(X)
        self.fitted = True
        evr = self.pca.explained_variance_ratio_
        print(f"[StaticSlimmer] Fitted PCA to {self.out_dim} dims. "
              f"Explained variance (first 5): {np.round(evr[:5], 4)}. "
              f"Cumulative @K={self.out_dim}: {evr.cumsum()[-1]:.4f}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.fitted and self.pca is not None, "StaticSlimmer not fitted"
        return self.pca.transform(X)

    def save(self, path_dir: str):
        os.makedirs(path_dir, exist_ok=True)
        meta = {
            "requested_out_dim": self.req_out_dim,
            "out_dim": self.out_dim,
            "src_names": self.src_names,
            "components_shape": self.pca.components_.shape,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
        }
        with open(os.path.join(path_dir, "static_slimmer_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        np.save(os.path.join(path_dir, "pca_components.npy"), self.pca.components_)
        np.save(os.path.join(path_dir, "pca_mean.npy"), self.pca.mean_)
        np.save(os.path.join(path_dir, "pca_var.npy"), self.pca.explained_variance_)

    def load(self, path_dir: str):
        with open(os.path.join(path_dir, "static_slimmer_meta.json"), "r") as f:
            meta = json.load(f)
        self.req_out_dim = int(meta.get("requested_out_dim", meta["out_dim"]))
        self.out_dim = int(meta["out_dim"])
        self.src_names = meta["src_names"]
        comps = np.load(os.path.join(path_dir, "pca_components.npy"))
        mean = np.load(os.path.join(path_dir, "pca_mean.npy"))
        var = np.load(os.path.join(path_dir, "pca_var.npy"))
        self.pca = PCA(n_components=self.out_dim, random_state=1337)
        self.pca.components_ = comps
        self.pca.mean_ = mean
        self.pca.explained_variance_ = var
        self.pca.n_features_in_ = mean.shape[0]
        self.fitted = True
