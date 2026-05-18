"""Streaming feature extraction bridge feeding the unified MoE."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Sequence

import torch
from scapy.packet import Packet

from gateway.data.datasets.assembly import assemble_window_features
from gateway.data.datasets.gating import build_unified_gating
from gateway.data.datasets.packet import packet_to_auto
from gateway.data.extractors.features import (
    ARP_LSTM_SEQUENCE_LENGTH,
    ARP_MICRO_BINS,
    DOS_LSTM_SEQUENCE_LENGTH,
    DOS_MICRO_BINS,
    SSDP_MULTICAST_V4,
    SSDP_MULTICAST_V6,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    arp_scapy_pkt_to_row,
    dos_scapy_pkt_to_row,
)
from gateway.data.structures.windowing import SequenceState, StreamingWindowManager, WindowBuffer

DEFAULT_TASKS: Sequence[str] = ("dos", "arp")


def _fallback_arp_row(pkt: Packet) -> Dict[str, object]:
    return {
        "ts": float(getattr(pkt, "time", 0.0)),
        "len": int(len(pkt)),
        "is_arp": 0,
        "arp_opcode": 0,
        "arp_sender_ip": None,
        "arp_sender_mac": None,
        "arp_target_ip": None,
        "arp_target_mac": None,
        "arp_is_gratuitous": 0,
    }


def _top_value(counter) -> Optional[str]:
    if hasattr(counter, "most_common"):
        try:
            return counter.most_common(1)[0][0]
        except IndexError:
            return None
    return None


@dataclass
class FeatureWindow:
    """Container pairing tensor features with helpful metadata."""

    features: Dict[str, torch.Tensor]
    context: Dict[str, object]
    packet_metadata: Dict[str, object]


class StreamingFeatureExtractor:
    """Stateful helper mirroring the offline feature engineering pipeline."""

    def __init__(
        self,
        tasks: Sequence[str] = DEFAULT_TASKS,
        window_manager: StreamingWindowManager | None = None,
        max_packets_per_window: int | None = None,
    ) -> None:
        self.tasks = tuple(tasks)
        micro_bins: Dict[str, int] = {}
        if "dos" in self.tasks:
            micro_bins["dos"] = DOS_MICRO_BINS
        if "arp" in self.tasks:
            micro_bins["arp"] = ARP_MICRO_BINS
        self.manager = window_manager or StreamingWindowManager(
            window_size=WINDOW_SIZE,
            stride=WINDOW_STRIDE,
            micro_bins=micro_bins,
            max_packets_per_window=max_packets_per_window,
        )
        self.dos_state = SequenceState(sequence_length=DOS_LSTM_SEQUENCE_LENGTH) if "dos" in self.tasks else None
        self.arp_state = SequenceState(sequence_length=ARP_LSTM_SEQUENCE_LENGTH) if "arp" in self.tasks else None
        self._last_packet_meta: Dict[str, object] = {}

    def _assemble(
        self,
        buffer: WindowBuffer,
        packet_meta: Dict[str, object],
    ) -> Optional[FeatureWindow]:
        features = assemble_window_features(buffer, self.tasks, self.dos_state, self.arp_state)
        if features is None:
            return None
        truncated_flag = bool(features.pop("_truncated", torch.tensor([0])))
        if "gating_input" not in features:
            features["gating_input"] = build_unified_gating(features)
        for key, tensor in list(features.items()):
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.to(torch.float32)
        context = {
            "window_index": buffer.index,
            "start_time": buffer.start,
            "end_time": buffer.end,
            "packet_count": buffer.packet_count,
            "truncated": truncated_flag or buffer.truncated,
            "top_src_ip": _top_value(buffer.auto_acc.src_ips),
            "top_dst_ip": _top_value(buffer.auto_acc.dst_ips),
        }
        return FeatureWindow(features=features, context=context, packet_metadata=packet_meta.copy())

    def process_packet(self, pkt: Packet) -> List[FeatureWindow]:
        """Feed a scapy packet into the streaming window pipeline."""

        if pkt is None:
            return []
        if pkt.time is None:
            pkt.time = time.time()
        packet_meta: Dict[str, object] = {}
        dos_row = None
        arp_row = None
        if "dos" in self.tasks:
            try:
                dos_row = dos_scapy_pkt_to_row(pkt, SSDP_MULTICAST_V4, SSDP_MULTICAST_V6)
            except Exception:
                dos_row = None
        if "arp" in self.tasks:
            try:
                arp_row = arp_scapy_pkt_to_row(pkt)
            except Exception:
                arp_row = None
        fallback_row = arp_row or _fallback_arp_row(pkt)
        timestamp, tcp_flags, auto_row = packet_to_auto(pkt, fallback_row)
        packet_meta.update(
            {
                "timestamp": timestamp,
                "src_mac": auto_row.get("src_mac"),
                "dst_mac": auto_row.get("dst_mac"),
                "src_ip": auto_row.get("src_ip"),
                "dst_ip": auto_row.get("dst_ip"),
                "src_port": auto_row.get("src_port"),
                "dst_port": auto_row.get("dst_port"),
                "protocol": auto_row.get("protocol"),
            }
        )
        self._last_packet_meta = packet_meta.copy()
        rows: Dict[str, Dict[str, object]] = {}
        if dos_row is not None:
            rows["dos"] = dos_row
        if arp_row is not None:
            rows["arp"] = arp_row
        completed = self.manager.add_packet(
            rows=rows,
            auto_row=auto_row,
            timestamp=timestamp,
            length=float(len(pkt)),
            tcp_flags=tcp_flags,
        )
        windows: List[FeatureWindow] = []
        for buffer in completed:
            result = self._assemble(buffer, packet_meta)
            if result is not None:
                windows.append(result)
        return windows

    def flush(self) -> List[FeatureWindow]:
        """Force any buffered windows to be emitted."""

        windows: List[FeatureWindow] = []
        packet_meta = self._last_packet_meta or {"timestamp": time.time()}
        for buffer in self.manager.flush():
            result = self._assemble(buffer, packet_meta)
            if result is not None:
                windows.append(result)
        return windows


_GLOBAL_EXTRACTOR: Optional[StreamingFeatureExtractor] = None


def extract_features(pkt: Packet, extractor: StreamingFeatureExtractor | None = None) -> List[FeatureWindow]:
    """Convenience wrapper compatible with the module-level API expected by the controller."""

    global _GLOBAL_EXTRACTOR
    if extractor is None:
        if _GLOBAL_EXTRACTOR is None:
            _GLOBAL_EXTRACTOR = StreamingFeatureExtractor()
        extractor = _GLOBAL_EXTRACTOR
    return extractor.process_packet(pkt)


__all__ = ["FeatureWindow", "StreamingFeatureExtractor", "extract_features"]
