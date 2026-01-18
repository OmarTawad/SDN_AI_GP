"""Models subpackage initialisation for the gateway project.


"""

from __future__ import annotations

from .unified_moe import (
    DEFAULT_GATING_HIDDEN,
    NUM_CLASSES,
    NUM_EXPERTS,
    UNIFIED_EXPERT_SPECS,
    UNIFIED_GATING_INPUT_DIM,
    UnifiedGating,
    UnifiedMoE,
    build_unified_moe,
)

__all__ = [
    "DEFAULT_GATING_HIDDEN",
    "NUM_CLASSES",
    "NUM_EXPERTS",
    "UNIFIED_EXPERT_SPECS",
    "UNIFIED_GATING_INPUT_DIM",
    "UnifiedGating",
    "UnifiedMoE",
    "build_unified_moe",
]

