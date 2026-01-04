from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import stats


@dataclass
class ThresholdResult:
    method: str
    threshold: float
    params: Dict[str, float]


def quantile_threshold(errors: np.ndarray, quantile: float) -> float:
    return float(np.quantile(errors, quantile))


def gaussian_threshold(errors: np.ndarray, k: float) -> Tuple[float, Dict[str, float]]:
    mean = float(np.mean(errors))
    std = float(np.std(errors))
    threshold = mean + k * std
    return float(threshold), {"mean": mean, "std": std, "k": float(k)}


def evt_threshold(
    errors: np.ndarray,
    tail_quantile: float,
    min_tail_size: int,
) -> Tuple[float, Dict[str, float]]:
    if not 0.0 < tail_quantile < 1.0:
        raise ValueError("tail_quantile must be between 0 and 1")
    threshold_seed = float(np.quantile(errors, tail_quantile))
    tail = errors[errors >= threshold_seed]
    if tail.size < min_tail_size:
        raise ValueError("Not enough tail samples for EVT fit")
    c, loc, scale = stats.genpareto.fit(tail - threshold_seed)
    extreme_threshold = threshold_seed + stats.genpareto.ppf(0.99, c, loc=loc, scale=scale)
    return float(extreme_threshold), {
        "seed_quantile": tail_quantile,
        "seed_threshold": threshold_seed,
        "shape": float(c),
        "loc": float(loc),
        "scale": float(scale),
    }


def select_threshold(errors: np.ndarray, config: Dict[str, float]) -> ThresholdResult:
    method = (config.get("method") or "quantile").lower()
    if method == "quantile":
        quantile = float(config.get("quantile", 0.995))
        threshold = quantile_threshold(errors, quantile)
        params = {"quantile": quantile}
    elif method == "gaussian_fit":
        k = float(config.get("gaussian_k", 3.0))
        threshold, params = gaussian_threshold(errors, k)
    elif method == "evt":
        tail_quantile = float(config.get("evt_tail_quantile", 0.98))
        min_tail_size = int(config.get("evt_min_tail", 50))
        threshold, params = evt_threshold(errors, tail_quantile, min_tail_size)
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    params.setdefault("min_consecutive", int(config.get("min_consecutive", 1)))
    params.setdefault("cooldown", int(config.get("cooldown", 0)))
    params.setdefault("min_attack_windows", int(config.get("min_attack_windows", 1)))

    return ThresholdResult(method=method, threshold=float(threshold), params=params)


def save_threshold(result: ThresholdResult, path: Path | str) -> None:
    Path(path).write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def load_threshold(path: Path | str) -> ThresholdResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ThresholdResult(method=data["method"], threshold=float(data["threshold"]), params=data.get("params", {}))
