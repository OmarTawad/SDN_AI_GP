#src/dos_detector/utils/io.py
"""I/O helpers for saving and loading artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

import joblib
import numpy as np
import pandas as pd

# try:  # pragma: no cover - optional dependency
#     import pyarrow.parquet as pq
# except ImportError:  # pragma: no cover
#     pq = None  # type: ignore

ALLOW_PARQUET = os.getenv("DOS_ENABLE_PARQUET") == "1"
PARQUET_EXTENSIONS = {".parquet", ".pqt"}
CSV_EXTENSIONS = {".csv"}

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

    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _suffix(path: Path) -> str:
    return path.suffix.lower()


def save_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Persist a dataframe to disk, preferring CSV to avoid SIMD crashes."""

    ensure_dir(path.parent)
    suffix = _suffix(path)
    if suffix in CSV_EXTENSIONS or not suffix:
        frame.to_csv(path if suffix else path.with_suffix(".csv"), index=False)
        return
    if suffix in PARQUET_EXTENSIONS:
        if not ALLOW_PARQUET:
            raise RuntimeError(
                f"Parquet writes are disabled on this platform for stability. Use a '.csv' target instead ({path})."
            )
        frame.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported dataframe format: {path}")


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataframe saved via :func:`save_dataframe`."""

    suffix = _suffix(path)
    if suffix in CSV_EXTENSIONS or not suffix:
        return pd.read_csv(path)
    if suffix in PARQUET_EXTENSIONS:
        if not ALLOW_PARQUET:
            raise RuntimeError(
                f"Parquet reads are disabled. Convert {path} to CSV or set DOS_ENABLE_PARQUET=1 if your CPU supports it."
            )
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataframe format: {path}")


def stream_dataframe(path: Path, columns: Sequence[str] | None = None, chunk_size: int = 50_000) -> Iterator[pd.DataFrame]:
    """Yield dataframe chunks without loading everything into RAM."""

    chunk_size = max(1_000, int(chunk_size))
    suffix = _suffix(path)
    if suffix in CSV_EXTENSIONS or not suffix:
        for chunk in pd.read_csv(path, usecols=columns, chunksize=chunk_size):
            yield chunk
        return
    if suffix in PARQUET_EXTENSIONS:
        if not ALLOW_PARQUET:
            raise RuntimeError(
                f"Parquet streaming is disabled. Convert {path} to CSV or enable DOS_ENABLE_PARQUET."
            )
        try:
            import pyarrow.parquet as pq
        except ImportError:
             raise ImportError("PyArrow is required for Parquet streaming but is not installed.")

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=chunk_size, columns=columns):
            yield batch.to_pandas()
        return
    raise ValueError(f"Unsupported dataframe format: {path}")


def save_joblib(path: Path, obj: Any) -> None:
    """Save a Python object with joblib."""

    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    """Load a Python object saved with joblib."""

    return joblib.load(path)


def save_compressed_array(path: Path, **arrays: np.ndarray) -> None:
    """Persist numpy arrays using np.savez_compressed."""

    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def resolve_processed_frame(processed_dir: Path, name: str) -> Path:
    """Locate a processed feature file (prefers CSV, falls back to Parquet)."""

    processed_dir = Path(processed_dir)
    stem = Path(name).stem
    candidates = [
        processed_dir / f"{stem}.csv",
        processed_dir / f"{stem}.parquet",
        processed_dir / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing processed features for {name} (expected under {processed_dir})")
