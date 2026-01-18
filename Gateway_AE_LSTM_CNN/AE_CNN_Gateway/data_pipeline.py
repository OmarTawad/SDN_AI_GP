"""Backwards-compatible facade for dataset utilities.


"""

from __future__ import annotations

from gateway.core import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID, class_id_to_name
from gateway.data.datasets.cache import CACHE_ROOT, discover_pcaps, load_cache_entries, tasks_slug
from gateway.data.datasets.cached_dataset import CachedMoEDataset
from gateway.data.datasets.gating import build_unified_gating
from gateway.data.datasets.labels import (
    coerce_targets_from_cache,
    infer_label_from_metadata,
    infer_label_from_tasks,
    resolve_label_id,
)
from gateway.data.datasets.streaming_dataset import MoEDataset
from gateway.data.structures.pcap import PcapInfo
from gateway.data.structures.windowing import (
    AutoFeatureAccumulator,
    SequenceState,
    StreamingWindowManager,
)

__all__ = [
    "AutoFeatureAccumulator",
    "CachedMoEDataset",
    "CLASS_ID_TO_NAME",
    "CLASS_NAME_TO_ID",
    "CACHE_ROOT",
    "MoEDataset",
    "PcapInfo",
    "SequenceState",
    "StreamingWindowManager",
    "build_unified_gating",
    "class_id_to_name",
    "coerce_targets_from_cache",
    "discover_pcaps",
    "infer_label_from_metadata",
    "infer_label_from_tasks",
    "load_cache_entries",
    "resolve_label_id",
    "tasks_slug",
]

