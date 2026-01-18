"""Dynamic Mixture-of-Experts SDN integration utilities."""

from __future__ import annotations

from .device_map import DEVICE_MAP
from .topology import build_dynamic_moe_topology

__all__ = [
    "DEVICE_MAP",
    "build_dynamic_moe_topology",
]
