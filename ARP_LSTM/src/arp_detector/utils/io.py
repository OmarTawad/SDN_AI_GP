#src/arp_detector/utils/io.py
"""I/O helpers for saving and loading artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


def ensure_dir(path: Path) -> None:
    """Ensure that a directory exists."""

    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON payload with UTF-8 encoding."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Persist a dataframe to Parquet."""

    ensure_dir(path.parent)
    frame.to_parquet(path, index=False)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataframe from Parquet, allowing the engine to be selected via env."""

    pref = os.environ.get("ARP_LSTM_PARQUET_ENGINE", "fastparquet").lower()
    default_order = ("fastparquet", "pyarrow")
    if pref == "auto":
        engines = default_order
    elif pref in ("pyarrow", "fastparquet"):
        engines = (pref,)
    else:
        engines = default_order

    errors: list[str] = []
    for engine in engines:
        try:
            return pd.read_parquet(path, engine=engine)
        except ImportError as exc:
            errors.append(f"{engine}: {exc}")
        except Exception as exc:
            errors.append(f"{engine}: {exc}")
    raise RuntimeError(f"Failed to read parquet shard {path}; tried {engines} → {errors}")


def save_joblib(path: Path, obj: Any) -> None:
    """Save a Python object with joblib."""

    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    """Load a Python object saved with joblib."""

    return joblib.load(path)
