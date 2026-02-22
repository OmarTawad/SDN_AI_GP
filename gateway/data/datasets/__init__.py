"""Dataset helpers for the gateway pipeline.

The heavy dataset modules are imported lazily to avoid triggering optional runtime
requirements (for example Scapy interface probing) during lightweight imports.
"""

from __future__ import annotations

from .labels import (
    CLASS_ID_TO_NAME,
    CLASS_NAME_TO_ID,
    coerce_targets_from_cache,
    infer_label_from_metadata,
    infer_label_from_tasks,
    resolve_label_id,
)

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


def __getattr__(name: str):
    if name == "CachedMoEDataset":
        from .cached_dataset import CachedMoEDataset

        return CachedMoEDataset
    if name == "MoEDataset":
        from .streaming_dataset import MoEDataset

        return MoEDataset
    if name in {"GATING_COMPONENT_LENGTHS", "UNIFIED_GATING_COMPONENT_KEYS", "build_unified_gating"}:
        from .gating import GATING_COMPONENT_LENGTHS, UNIFIED_GATING_COMPONENT_KEYS, build_unified_gating

        return {
            "GATING_COMPONENT_LENGTHS": GATING_COMPONENT_LENGTHS,
            "UNIFIED_GATING_COMPONENT_KEYS": UNIFIED_GATING_COMPONENT_KEYS,
            "build_unified_gating": build_unified_gating,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
