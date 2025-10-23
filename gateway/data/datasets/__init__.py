"""Dataset helpers for the gateway pipeline."""

from __future__ import annotations

from .cached_dataset import CachedMoEDataset
from .gating import GATING_COMPONENT_LENGTHS, UNIFIED_GATING_COMPONENT_KEYS, build_unified_gating
from .labels import (
    CLASS_ID_TO_NAME,
    CLASS_NAME_TO_ID,
    coerce_targets_from_cache,
    infer_label_from_metadata,
    infer_label_from_tasks,
    resolve_label_id,
)
from .streaming_dataset import MoEDataset

__all__ = [
    "CachedMoEDataset",
    "MoEDataset",
    "GATING_COMPONENT_LENGTHS",
    "UNIFIED_GATING_COMPONENT_KEYS",
    "build_unified_gating",
    "CLASS_ID_TO_NAME",
    "CLASS_NAME_TO_ID",
    "coerce_targets_from_cache",
    "infer_label_from_metadata",
    "infer_label_from_tasks",
    "resolve_label_id",
]

