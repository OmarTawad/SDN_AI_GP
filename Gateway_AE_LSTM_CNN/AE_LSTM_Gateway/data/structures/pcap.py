"""PCAP-related dataclasses shared across the gateway project.

----
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class PcapInfo:
    """Metadata describing a PCAP file to be processed by datasets."""

    path: Path
    label: int
    meta: Dict[str, Any] = field(default_factory=dict)


__all__ = ["PcapInfo"]

