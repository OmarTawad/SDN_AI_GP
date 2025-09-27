from __future__ import annotations

import numpy as np
import pandas as pd

from dae.preprocess import fit_scaler, load_scaler, save_scaler, transform_with_scaler


def test_scaler_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    scaler, features = fit_scaler(df, "RobustScaler")
    transformed = transform_with_scaler(df, scaler)

    path = tmp_path / "scaler.pkl"
    save_scaler(scaler, path)
    loaded = load_scaler(path)
    transformed_loaded = transform_with_scaler(df, loaded)

    assert np.allclose(transformed, transformed_loaded)
    assert features == ["a", "b"]
