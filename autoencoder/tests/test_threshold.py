from __future__ import annotations

import numpy as np

from dae.threshold import select_threshold


def test_quantile_threshold_selection():
    errors = np.linspace(0.0, 1.0, 1001)
    config = {"method": "quantile", "quantile": 0.9}
    result = select_threshold(errors, config)
    assert result.method == "quantile"
    assert np.isclose(result.threshold, np.quantile(errors, 0.9))


def test_gaussian_threshold_selection():
    rng = np.random.default_rng(42)
    errors = rng.normal(loc=0.1, scale=0.01, size=1000)
    config = {"method": "gaussian_fit", "gaussian_k": 3.0}
    result = select_threshold(errors, config)
    mean = errors.mean()
    std = errors.std()
    assert np.isclose(result.threshold, mean + 3.0 * std, atol=1e-6)
    assert "min_consecutive" in result.params
