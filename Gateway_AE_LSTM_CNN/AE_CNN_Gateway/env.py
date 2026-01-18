"""Project path utilities for the gateway package."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _candidate_paths() -> Iterable[Path]:
    yield PROJECT_ROOT
    yield PROJECT_ROOT / "dosdet"
    yield PROJECT_ROOT / "arpdet"
    yield PROJECT_ROOT / "autoencoder"


def extend_sys_path() -> None:
    """Ensure dependent project folders are importable."""
    for candidate in _candidate_paths():
        candidate_path = candidate.resolve()
        if not candidate_path.exists():
            continue
        candidate_str = str(candidate_path)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


extend_sys_path()

CACHE_ROOT = Path(__file__).resolve().parent / "cache"
SAMPLES_DIR = PROJECT_ROOT / "samples"


__all__ = ["PROJECT_ROOT", "CACHE_ROOT", "SAMPLES_DIR", "extend_sys_path"]
