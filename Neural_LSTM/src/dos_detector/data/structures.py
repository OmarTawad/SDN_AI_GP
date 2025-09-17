#src/dos_detector/data/structures.py

"""Data structures used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PacketRecord:
    """Normalized packet representation extracted from a PCAP."""

    timestamp: float
    src_mac: Optional[str]
    dst_mac: Optional[str]
    src_ip: Optional[str]
    dst_ip: Optional[str]
    src_port: Optional[int]
    dst_port: Optional[int]
    protocol: str
    length: int
    ttl: Optional[int]
    tcp_flags: Optional[int]
    payload_len: int
    info: Dict[str, Optional[str]]


@dataclass
class Window:
    """A fixed-duration packet window."""

    index: int
    start_time: float
    end_time: float
    packets: List[PacketRecord]


@dataclass
class WindowLabels:
    """Labels associated with a window."""

    attack: int
    family: str


@dataclass
class SequenceSample:
    """A training-ready sequence."""

    features: List[List[float]]
    binary_labels: List[int]
    family_labels: List[int]
    metadata: Dict[str, object]