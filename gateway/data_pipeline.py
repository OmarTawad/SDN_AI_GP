"""Streaming and cached dataset utilities for gateway training."""

from __future__ import annotations

import csv
import math
import random
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether
from scapy.utils import PcapReader
from torch import Tensor
from torch.utils.data import IterableDataset

from Neural_LSTM.src.dos_detector.data.structures import PacketRecord as DosPacketRecord, Window as DosWindow
from ARP_LSTM.src.arp_detector.data.structures import PacketRecord as ArpPacketRecord, Window as ArpWindow

from .env import CACHE_ROOT, SAMPLES_DIR
from .features import (
    ARP_LSTM_SCALER,
    ARP_FEATURE_EXTRACTOR,
    ARP_LSTM_SEQUENCE_LENGTH,
    ARP_MICRO_BINS,
    ARP_WINDOW_SIZE,
    ARP_WINDOW_STRIDE,
    DOS_LSTM_SCALER,
    DOS_FEATURE_EXTRACTOR,
    DOS_LSTM_SEQUENCE_LENGTH,
    DOS_MICRO_BINS,
    DOS_WINDOW_SIZE,
    DOS_WINDOW_STRIDE,
    SSDP_MULTICAST_V6,
    SSDP_MULTICAST_V4,
    TOP_UDP_PORTS,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    arp_scapy_pkt_to_row,
    compute_arp_sequence_features,
    compute_arp_static_features,
    compute_dos_sequence_features,
    compute_dos_static_features,
    dos_scapy_pkt_to_row,
    prepare_arp_static,
    prepare_auto_tensor,
    prepare_dos_static,
)
from gateway.moe_model import (
    ARP_CNN_SEQ_IN_DIM,
    ARP_CNN_STATIC_DIM,
    ARP_LSTM_INPUT_DIM,
    AUTO_FEATURE_DIM,
    DOS_CNN_SEQ_IN_DIM,
    DOS_CNN_STATIC_DIM,
    DOS_LSTM_INPUT_DIM,
)


CLASS_NAME_TO_ID: Dict[str, int] = {
    "0": 0,
    "normal": 0,
    "benign": 0,
    "background": 0,
    "1": 1,
    "dos": 1,
    "dos_attack": 1,
    "flood": 1,
    "attack_dos": 1,
    "2": 2,
    "arp": 2,
    "arp_spoof": 2,
    "spoof": 2,
    "poison": 2,
    "attack_arp": 2,
}
CLASS_ID_TO_NAME: Dict[int, str] = {0: "normal", 1: "dos", 2: "arp"}
CLASS_SUBDIRS: Dict[int, str] = {label: name for label, name in CLASS_ID_TO_NAME.items()}

UNIFIED_GATING_COMPONENT_KEYS: Tuple[str, ...] = (
    "auto",
    "dos_cnn_static",
    "dos_cnn_seq",
    "dos_lstm_seq",
    "arp_cnn_static",
    "arp_cnn_seq",
    "arp_lstm_seq",
)

GATING_COMPONENT_LENGTHS: Dict[str, int] = {
    "auto": AUTO_FEATURE_DIM,
    "dos_cnn_static": DOS_CNN_STATIC_DIM,
    "dos_cnn_seq": DOS_MICRO_BINS * DOS_CNN_SEQ_IN_DIM,
    "dos_lstm_seq": DOS_LSTM_SEQUENCE_LENGTH * DOS_LSTM_INPUT_DIM,
    "arp_cnn_static": ARP_CNN_STATIC_DIM,
    "arp_cnn_seq": ARP_MICRO_BINS * ARP_CNN_SEQ_IN_DIM,
    "arp_lstm_seq": ARP_LSTM_SEQUENCE_LENGTH * ARP_LSTM_INPUT_DIM,
}


def resolve_label_id(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        label = int(value)
        if label in CLASS_ID_TO_NAME:
            return label
        raise ValueError(f"Unsupported label id '{label}'.")
    if isinstance(value, (float, np.floating)):
        return resolve_label_id(int(value))
    if isinstance(value, str):
        token = value.strip().lower()
        if token in CLASS_NAME_TO_ID:
            return CLASS_NAME_TO_ID[token]
        try:
            numeric = int(token)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported label name '{value}'.") from exc
        return resolve_label_id(numeric)
    raise TypeError(f"Cannot resolve label from value of type {type(value)}.")


def class_id_to_name(label: int) -> str:
    return CLASS_ID_TO_NAME.get(label, f"class_{label}")


def _flatten_for_gating(tensor: Tensor) -> Tensor:
    if tensor.dim() == 0:
        return tensor.reshape(1)
    return tensor.reshape(-1)


def build_unified_gating(features: Dict[str, Tensor]) -> Tensor:
    components: List[Tensor] = []
    for key in UNIFIED_GATING_COMPONENT_KEYS:
        expected = GATING_COMPONENT_LENGTHS[key]
        tensor = features.get(key)
        if tensor is None:
            flat = torch.zeros(expected, dtype=torch.float32)
        else:
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Gating component '{key}' must be a tensor.")
            flat = _flatten_for_gating(tensor).to(torch.float32)
            if flat.numel() != expected:
                raise ValueError(
                    f"Gating component '{key}' has {flat.numel()} elements, expected {expected}."
                )
        components.append(flat)
    if not components:
        raise ValueError("Cannot assemble unified gating vector without feature tensors.")
    return torch.cat(components, dim=0)


def infer_label_from_tasks(task_labels: Dict[str, Any]) -> Optional[int]:
    if not isinstance(task_labels, dict):
        return None
    try:
        arp_flag = int(round(float(task_labels.get("arp", 0))))
        dos_flag = int(round(float(task_labels.get("dos", 0))))
    except Exception:
        return None
    if arp_flag > 0:
        return 2
    if dos_flag > 0:
        return 1
    return 0


def infer_label_from_metadata(meta: Dict[str, Any]) -> Optional[int]:
    if not isinstance(meta, dict):
        return None
    if "class_label" in meta:
        try:
            return resolve_label_id(meta["class_label"])
        except (TypeError, ValueError):
            pass
    label_name = meta.get("label_name")
    if label_name is not None:
        try:
            return resolve_label_id(label_name)
        except (TypeError, ValueError):
            pass
    task_labels = meta.get("labels")
    if isinstance(task_labels, dict):
        inferred = infer_label_from_tasks(task_labels)
        if inferred is not None:
            return inferred
    return None


def coerce_targets_from_cache(raw_labels: Any, default_label: int, window_count: int) -> Tensor:
    if isinstance(raw_labels, Tensor):
        tensor = raw_labels.detach().clone()
        if tensor.dim() == 0:
            tensor = tensor.reshape(1)
        tensor = tensor.reshape(-1).to(torch.long)
        if tensor.shape[0] < window_count:
            pad = torch.full((window_count - tensor.shape[0],), default_label, dtype=torch.long)
            tensor = torch.cat([tensor, pad], dim=0)
        elif tensor.shape[0] > window_count:
            tensor = tensor[:window_count]
        return tensor
    if isinstance(raw_labels, dict):
        targets = torch.full((window_count,), default_label, dtype=torch.long)
        if "dos" in raw_labels:
            dos_tensor = torch.as_tensor(raw_labels["dos"]).reshape(-1)
            limit = min(window_count, dos_tensor.shape[0])
            mask = dos_tensor[:limit].round().to(torch.long) > 0
            targets[:limit][mask] = 1
        if "arp" in raw_labels:
            arp_tensor = torch.as_tensor(raw_labels["arp"]).reshape(-1)
            limit = min(window_count, arp_tensor.shape[0])
            mask = arp_tensor[:limit].round().to(torch.long) > 0
            targets[:limit][mask] = 2
        return targets
    return torch.full((window_count,), default_label, dtype=torch.long)


def _entropy(counter: Counter) -> float:
    total = float(sum(counter.values()))
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        p = float(count) / total
        value -= p * math.log2(p)
    return value


@dataclass
class AutoFeatureAccumulator:
    duration: float
    count: int = 0
    byte_count: float = 0.0
    mean_len: float = 0.0
    m2_len: float = 0.0
    min_len: float = field(default_factory=lambda: float("inf"))
    max_len: float = 0.0
    last_ts: Optional[float] = None
    iat_count: int = 0
    iat_mean: float = 0.0
    iat_m2: float = 0.0
    proto_counts: Counter = field(default_factory=Counter)
    tcp_flag_counts: Counter = field(default_factory=Counter)
    src_ips: Counter = field(default_factory=Counter)
    dst_ips: Counter = field(default_factory=Counter)
    src_ports: Counter = field(default_factory=Counter)
    dst_ports: Counter = field(default_factory=Counter)

    def add(
        self,
        row: Dict[str, object],
        length: float,
        timestamp: float,
        tcp_flags: Optional[int],
    ) -> None:
        self.count += 1
        self.byte_count += float(length)
        delta = length - self.mean_len
        self.mean_len += delta / self.count
        self.m2_len += delta * (length - self.mean_len)
        self.min_len = min(self.min_len, length)
        self.max_len = max(self.max_len, length)

        if self.last_ts is not None:
            iat = timestamp - self.last_ts
            self.iat_count += 1
            delta_iat = iat - self.iat_mean
            self.iat_mean += delta_iat / self.iat_count
            self.iat_m2 += delta_iat * (iat - self.iat_mean)
        self.last_ts = timestamp

        proto = row.get("protocol")
        if isinstance(proto, str):
            proto = proto.upper()
        if proto in {"TCP", "UDP", "ICMP"}:
            self.proto_counts.update([proto])

        if tcp_flags is not None:
            if tcp_flags & 0x02:
                self.tcp_flag_counts.update(["SYN"])
            if tcp_flags & 0x10:
                self.tcp_flag_counts.update(["ACK"])
            if tcp_flags & 0x04:
                self.tcp_flag_counts.update(["RST"])
            if tcp_flags & 0x01:
                self.tcp_flag_counts.update(["FIN"])

        src_ip = row.get("src_ip")
        dst_ip = row.get("dst_ip")
        src_port = row.get("src_port")
        dst_port = row.get("dst_port")

        if src_ip:
            self.src_ips.update([str(src_ip)])
        if dst_ip:
            self.dst_ips.update([str(dst_ip)])
        if src_port is not None:
            self.src_ports.update([int(src_port)])
        if dst_port is not None:
            self.dst_ports.update([int(dst_port)])

    def finalize(self) -> Optional[Dict[str, float]]:
        if self.count == 0:
            return None
        mean_iat = self.iat_mean if self.iat_count else 0.0
        std_iat = math.sqrt(self.iat_m2 / max(self.iat_count - 1, 1)) if self.iat_count > 1 else 0.0
        std_len = math.sqrt(self.m2_len / max(self.count - 1, 1)) if self.count > 1 else 0.0
        pkt_count = float(self.count)
        duration = max(self.duration, 1e-6)
        tcp_count = float(self.proto_counts.get("TCP", 0))
        udp_count = float(self.proto_counts.get("UDP", 0))
        icmp_count = float(self.proto_counts.get("ICMP", 0))

        stats: Dict[str, float] = {
            "pkt_count": pkt_count,
            "byte_count": float(self.byte_count),
            "pps": pkt_count / duration,
            "bps": (8.0 * float(self.byte_count)) / duration,
            "mean_pkt_len": float(self.mean_len),
            "std_pkt_len": std_len,
            "min_pkt_len": 0.0 if math.isinf(self.min_len) else float(self.min_len),
            "max_pkt_len": float(self.max_len),
            "mean_iat": mean_iat,
            "std_iat": std_iat,
            "tcp_count": tcp_count,
            "udp_count": udp_count,
            "icmp_count": icmp_count,
            "tcp_syn": float(self.tcp_flag_counts.get("SYN", 0)),
            "tcp_ack": float(self.tcp_flag_counts.get("ACK", 0)),
            "tcp_rst": float(self.tcp_flag_counts.get("RST", 0)),
            "tcp_fin": float(self.tcp_flag_counts.get("FIN", 0)),
            "src_ip_entropy": _entropy(self.src_ips),
            "dst_ip_entropy": _entropy(self.dst_ips),
            "src_port_entropy": _entropy(self.src_ports),
            "dst_port_entropy": _entropy(self.dst_ports),
            "unique_src_ips": float(len(self.src_ips)),
            "unique_dst_ips": float(len(self.dst_ips)),
            "unique_src_ports": float(len(self.src_ports)),
            "unique_dst_ports": float(len(self.dst_ports)),
            "tcp_ratio": tcp_count / max(pkt_count, 1.0),
            "udp_ratio": udp_count / max(pkt_count, 1.0),
            "icmp_ratio": icmp_count / max(pkt_count, 1.0),
        }
        return stats


@dataclass
class WindowBuffer:
    start: float
    end: float
    index: int
    duration: float
    micro_bins: Dict[str, int]
    max_packets: Optional[int] = None
    auto_acc: AutoFeatureAccumulator = field(init=False)
    task_rows: Dict[str, List[Dict[str, object]]] = field(init=False)
    bin_indices: Dict[str, List[int]] = field(init=False)
    packet_count: int = field(default=0, init=False)
    truncated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.auto_acc = AutoFeatureAccumulator(duration=self.duration)
        self.task_rows = {name: [] for name in self.micro_bins}
        self.bin_indices = {name: [] for name in self.micro_bins}

    def add(
        self,
        rows: Dict[str, Dict[str, object]],
        auto_row: Dict[str, object],
        timestamp: float,
        length: float,
        tcp_flags: Optional[int],
        bin_index_map: Dict[str, int],
    ) -> None:
        if self.max_packets is not None and self.packet_count >= self.max_packets:
            self.truncated = True
            return
        self.auto_acc.add(auto_row, length, timestamp, tcp_flags)
        self.packet_count += 1
        for key, indices in self.bin_indices.items():
            row = rows.get(key)
            if row is not None:
                self.task_rows[key].append(row)
            else:
                self.task_rows[key].append({})
            indices.append(bin_index_map.get(key, 0))


class StreamingWindowManager:
    def __init__(
        self,
        window_size: float,
        stride: float,
        micro_bins: Dict[str, int],
        max_packets_per_window: Optional[int],
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.micro_bins = {name: int(count) for name, count in micro_bins.items()}
        self.bin_widths = {
            name: (window_size / max(count, 1)) if count > 0 else window_size
            for name, count in self.micro_bins.items()
        }
        self.next_window_start: Optional[float] = None
        self.active: Deque[WindowBuffer] = deque()
        self.index = 0
        self.max_packets_per_window = max_packets_per_window

    def _open_window(self, start: float) -> None:
        buffer = WindowBuffer(
            start=start,
            end=start + self.window_size,
            index=self.index,
            duration=self.window_size,
            micro_bins=self.micro_bins,
            max_packets=self.max_packets_per_window,
        )
        self.index += 1
        self.active.append(buffer)

    def add_packet(
        self,
        rows: Dict[str, Dict[str, object]],
        auto_row: Dict[str, object],
        timestamp: float,
        length: float,
        tcp_flags: Optional[int],
    ) -> List[WindowBuffer]:
        completed: List[WindowBuffer] = []
        if self.next_window_start is None:
            stride = self.stride
            if stride <= 0:
                stride = self.window_size
            self.next_window_start = math.floor(timestamp / stride) * stride

        while self.next_window_start is not None and self.next_window_start <= timestamp:
            self._open_window(self.next_window_start)
            self.next_window_start += self.stride

        while self.active and self.active[0].end <= timestamp:
            completed.append(self.active.popleft())

        for window in self.active:
            if window.start <= timestamp < window.end:
                offset = timestamp - window.start
                bin_index_map: Dict[str, int] = {}
                for name, count in self.micro_bins.items():
                    width = self.bin_widths.get(name, self.window_size)
                    if count <= 0 or width <= 0:
                        idx = 0
                    else:
                        idx = int(offset / width)
                        idx = max(0, min(count - 1, idx))
                    bin_index_map[name] = idx
                window.add(rows, auto_row, timestamp, length, tcp_flags, bin_index_map)
        return completed

    def flush(self) -> List[WindowBuffer]:
        remaining = list(self.active)
        self.active.clear()
        return remaining


def _packet_to_auto(
    pkt,
    arp_row: Dict[str, object],
) -> Tuple[float, int, Dict[str, object]]:
    length = float(len(pkt))
    timestamp = float(pkt.time)

    eth = pkt.getlayer(Ether)
    src_mac = getattr(eth, "src", None) if eth is not None else None
    dst_mac = getattr(eth, "dst", None) if eth is not None else None
    protocol = "other"
    src_ip = dst_ip = None
    ttl = None
    src_port = dst_port = None
    tcp_flags = None
    payload_len = 0
    info: Dict[str, Optional[str]] = {}

    layer_ip = pkt.getlayer(IP) or pkt.getlayer(IPv6)
    if layer_ip is not None:
        src_ip = getattr(layer_ip, "src", None)
        dst_ip = getattr(layer_ip, "dst", None)
        ttl = getattr(layer_ip, "ttl", getattr(layer_ip, "hlim", None))

        if layer_ip.haslayer(TCP):
            tcp = layer_ip.getlayer(TCP)
            src_port = int(getattr(tcp, "sport", 0))
            dst_port = int(getattr(tcp, "dport", 0))
            tcp_flags = int(getattr(tcp, "flags", 0))
            protocol = "tcp"
            payload_len = len(bytes(tcp.payload))
        elif layer_ip.haslayer(UDP):
            udp = layer_ip.getlayer(UDP)
            src_port = int(getattr(udp, "sport", 0))
            dst_port = int(getattr(udp, "dport", 0))
            protocol = "udp"
            payload = bytes(udp.payload)
            payload_len = len(payload)
            if payload_len:
                text = payload.decode(errors="ignore")
                if "M-SEARCH" in text:
                    info["ssdp_method"] = "M-SEARCH"
                elif "NOTIFY" in text:
                    info["ssdp_method"] = "NOTIFY"
        elif layer_ip.haslayer(ICMP):
            icmp = layer_ip.getlayer(ICMP)
            protocol = "icmp"
            info["icmp_type"] = str(getattr(icmp, "type", None))
            payload_len = len(bytes(icmp.payload))
        else:
            protocol = layer_ip.name.lower()
            payload_len = len(bytes(layer_ip.payload))
    else:
        payload_len = len(bytes(pkt.payload))

    arp_opcode = arp_row.get("arp_opcode")
    auto_row: Dict[str, object] = {
        "is_tcp": 1 if protocol == "tcp" else 0,
        "is_udp": 1 if protocol == "udp" else 0,
        "is_icmp": 1 if protocol == "icmp" else 0,
        "tcp_syn": 1 if tcp_flags is not None and (tcp_flags & 0x02) else 0,
        "tcp_synack": 1 if tcp_flags is not None and (tcp_flags & 0x12) == 0x12 else 0,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "src_mac": src_mac,
        "dst_mac": dst_mac,
        "ttl": ttl,
        "payload_len": payload_len,
        "protocol": protocol,
        "arp_opcode": int(arp_row.get("arp_opcode") or 0),
        "arp_sender_ip": str(arp_row.get("arp_sender_ip")) if arp_row.get("arp_sender_ip") else None,
        "arp_sender_mac": str(arp_row.get("arp_sender_mac")) if arp_row.get("arp_sender_mac") else None,
        "arp_target_ip": str(arp_row.get("arp_target_ip")) if arp_row.get("arp_target_ip") else None,
        "arp_target_mac": str(arp_row.get("arp_target_mac")) if arp_row.get("arp_target_mac") else None,
        "arp_is_gratuitous": bool(int(arp_row.get("arp_is_gratuitous", 0) or 0)),
        "ssdp_method": info.get("ssdp_method"),
        "icmp_type": info.get("icmp_type"),
    }
    return timestamp, tcp_flags or 0, auto_row


def _protocol_from_row(row: Dict[str, object]) -> str:
    if row.get("is_tcp"):
        return "tcp"
    if row.get("is_udp"):
        return "udp"
    if row.get("is_icmp"):
        return "icmp"
    if row.get("is_arp"):
        return "arp"
    return "other"


def _build_dos_window(index: int, start: float, end: float, rows: Sequence[Dict[str, object]]) -> DosWindow:
    packets: List[DosPacketRecord] = []
    for row in rows:
        protocol = _protocol_from_row(row)
        flags = 0
        if row.get("tcp_syn"):
            flags |= 0x02
        if row.get("tcp_synack"):
            flags |= 0x12
        info: Dict[str, Optional[str]] = {}
        if row.get("ssdp_method") not in (None, "NONE"):
            info["ssdp_method"] = str(row.get("ssdp_method"))
        packet = DosPacketRecord(
            timestamp=float(row.get("ts", 0.0)),
            src_mac=row.get("src_mac"),
            dst_mac=row.get("dst_mac"),
            src_ip=row.get("src_ip"),
            dst_ip=row.get("dst_ip"),
            src_port=int(row.get("src_port")) if row.get("src_port") is not None else None,
            dst_port=int(row.get("dst_port")) if row.get("dst_port") is not None else None,
            protocol=protocol,
            length=int(row.get("len") or 0),
            ttl=int(row.get("ttl") or 0) if row.get("ttl") is not None else None,
            tcp_flags=int(flags),
            payload_len=int(row.get("udp_len") or row.get("len") or 0),
            info=info,
        )
        packets.append(packet)
    return DosWindow(index=index, start_time=float(start), end_time=float(end), packets=packets)


def _build_arp_window(index: int, start: float, end: float, rows: Sequence[Dict[str, object]]) -> ArpWindow:
    packets: List[ArpPacketRecord] = []
    for row in rows:
        info: Dict[str, Optional[str]] = {}
        packet = ArpPacketRecord(
            timestamp=float(row.get("ts", 0.0)),
            src_mac=row.get("src_mac"),
            dst_mac=row.get("dst_mac"),
            src_ip=row.get("src_ip"),
            dst_ip=row.get("dst_ip"),
            src_port=None,
            dst_port=None,
            protocol="arp" if row.get("is_arp") else _protocol_from_row(row),
            length=int(row.get("len") or 0),
            ttl=None,
            tcp_flags=0,
            payload_len=int(row.get("len") or 0),
            info=info,
            arp_opcode=int(row.get("arp_opcode") or 0),
            arp_sender_ip=row.get("arp_sender_ip"),
            arp_sender_mac=row.get("arp_sender_mac"),
            arp_target_ip=row.get("arp_target_ip"),
            arp_target_mac=row.get("arp_target_mac"),
            arp_is_gratuitous=bool(row.get("arp_is_gratuitous")),
        )
        packets.append(packet)
    return ArpWindow(index=index, start_time=float(start), end_time=float(end), packets=packets)


@dataclass
class SequenceState:
    sequence_length: int
    feature_names: Optional[List[str]] = None
    buffer: Deque[Tensor] = field(init=False)

    def __post_init__(self) -> None:
        self.buffer = deque(maxlen=self.sequence_length)

    def update(self, features: Dict[str, float], scaler) -> Tensor:
        if self.feature_names is None:
            self.feature_names = list(features.keys())
        vector = np.array(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float32,
        )
        scaled = scaler.transform(vector.reshape(1, -1)).astype(np.float32)
        tensor = torch.from_numpy(scaled.squeeze(0))
        self.buffer.append(tensor)
        if len(self.buffer) < self.sequence_length:
            pad_count = self.sequence_length - len(self.buffer)
            pad_tensor = torch.zeros_like(tensor)
            tensors = [pad_tensor for _ in range(pad_count)] + list(self.buffer)
        else:
            tensors = list(self.buffer)
        return torch.stack(tensors, dim=0)


@dataclass
class PcapInfo:
    path: Path
    label: int
    meta: Dict[str, Any] = field(default_factory=dict)


class MoEDataset(IterableDataset[Tuple[Dict[str, Tensor], Tensor]]):
    """Iterable dataset that streams PCAP files and yields windowed tensors."""

    def __init__(
        self,
        files: Sequence[PcapInfo],
        tasks: Sequence[str],
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 17,
        max_windows_per_file: Optional[int] = None,
        max_total_windows: Optional[int] = None,
        status_interval: Optional[int] = None,
        max_file_size: Optional[int] = None,
        max_packets_per_file: Optional[int] = None,
        file_timeout: Optional[float] = None,
        max_packets_per_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.tasks = tuple(tasks)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self.max_windows_per_file = max_windows_per_file
        self.max_total_windows = max_total_windows
        self.max_file_size = max_file_size
        self.max_packets_per_file = max_packets_per_file
        self.file_timeout = file_timeout
        self.max_packets_per_window = max_packets_per_window
        self.total_windows_processed = 0
        self.log_every = status_interval if status_interval and status_interval > 0 else None
        self.log_fn: Callable[[str], None] = self._default_log
        self._next_log = self.log_every
        self._window_budget = max_total_windows
        self._window_cap_announced = False
        self._file_cap_announced: Set[Path] = set()
        self._seen_files: Set[Path] = set()
        self.skipped: Dict[Path, str] = {}
        self.stats: Dict[Path, Dict[str, Any]] = {}
        self.window_callback: Optional[Callable[[PcapInfo, int], None]] = None
        for info in self.files:
            self.stats[info.path] = {
                "windows": 0,
                "batches": 0,
                "label": int(info.label),
                "label_name": class_id_to_name(int(info.label)),
                "truncated_windows": 0,
            }

    @staticmethod
    def _default_log(message: str) -> None:
        print(message)

    def set_log_fn(self, fn: Optional[Callable[[str], None]]) -> None:
        self.log_fn = fn or self._default_log

    def set_window_budget(self, budget: Optional[int]) -> None:
        self._window_budget = budget

    def set_window_callback(self, fn: Optional[Callable[[PcapInfo, int], None]]) -> None:
        self.window_callback = fn

    def _mark_skip(self, info: PcapInfo, reason: str) -> None:
        record = self.stats.get(info.path)
        if record is None:
            return
        if record.get("skipped_reason"):
            return
        record["skipped_reason"] = reason
        self.skipped[info.path] = reason
        self.log_fn(f"[Skip] {info.path.name}: {reason}")

    def _notify_windows(self, info: PcapInfo, increment: int) -> None:
        if increment <= 0 or self.window_callback is None:
            return
        try:
            self.window_callback(info, increment)
        except Exception:
            pass

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self.total_windows_processed = 0
        self._window_cap_announced = False
        self._file_cap_announced.clear()
        self._seen_files.clear()
        self._next_log = self.log_every if self.log_every is not None else None
        self.skipped.clear()
        for stats in self.stats.values():
            stats["windows"] = 0
            stats["batches"] = 0
            stats.pop("skipped_reason", None)

    def _log_file_start(self, info: PcapInfo) -> None:
        if info.path in self._seen_files:
            return
        label_name = class_id_to_name(int(info.label))
        self.log_fn(f"[Stream] {info.path.name}: label={label_name}")
        self._seen_files.add(info.path)

    def _maybe_log_progress(self, info: PcapInfo, windows_for_file: int) -> None:
        if self.log_every is None:
            return
        if self._next_log is None:
            self._next_log = self.log_every
        while self._next_log is not None and self.total_windows_processed >= self._next_log:
            message = f"[WindowProgress] total={self.total_windows_processed}"
            if self._window_budget:
                pct = min(100.0, (self.total_windows_processed / self._window_budget) * 100.0)
                message += f"/{self._window_budget} ({pct:4.1f}%)"
            message += f" file={info.path.name} file_windows={windows_for_file}"
            self.log_fn(message)
            self._next_log += self.log_every

    def _note_file_cap(self, info: PcapInfo, windows_for_file: int) -> None:
        if (
            self.max_windows_per_file is None
            or windows_for_file < self.max_windows_per_file
            or info.path in self._file_cap_announced
        ):
            return
        self._file_cap_announced.add(info.path)
        self.log_fn(
            f"[WindowBudget] Reached per-file cap ({self.max_windows_per_file} windows) on {info.path.name}; switching captures."
        )

    def _note_window_cap(self, info: PcapInfo) -> None:
        if self.max_total_windows is None or self._window_cap_announced:
            return
        if self.total_windows_processed >= self.max_total_windows:
            self._window_cap_announced = True
            self.log_fn(
                f"[WindowBudget] Global window cap ({self.max_total_windows}) reached while reading {info.path.name}; halting stream."
            )

    def _iter_file(self, info: PcapInfo) -> Iterator[Tuple[Dict[str, Tensor], Tensor]]:
        if not info.path.exists():
            return
        if self.max_total_windows is not None and self.total_windows_processed >= self.max_total_windows:
            return

        if self.max_file_size is not None:
            try:
                file_size = info.path.stat().st_size
            except FileNotFoundError:
                return
            if file_size > self.max_file_size:
                size_mb = file_size / (1024 * 1024)
                limit_mb = self.max_file_size / (1024 * 1024)
                self._mark_skip(info, f"size {size_mb:.1f}MB exceeds limit {limit_mb:.1f}MB")
                return

        self._log_file_start(info)

        micro_bins: Dict[str, int] = {}
        if "dos" in self.tasks:
            micro_bins["dos"] = DOS_MICRO_BINS
        if "arp" in self.tasks:
            micro_bins["arp"] = ARP_MICRO_BINS

        manager = StreamingWindowManager(
            WINDOW_SIZE,
            WINDOW_STRIDE,
            micro_bins,
            self.max_packets_per_window,
        )
        dos_state = SequenceState(DOS_LSTM_SEQUENCE_LENGTH) if "dos" in self.tasks else None
        arp_state = SequenceState(ARP_LSTM_SEQUENCE_LENGTH) if "arp" in self.tasks else None

        batch_features: Dict[str, List[Tensor]] = defaultdict(list)
        batch_targets: List[int] = []
        windows_for_file = 0
        stop_file = False
        drop_reason: Optional[str] = None
        packets_seen = 0
        start_time = time.monotonic()

        def _push_window(window_features: Dict[str, Tensor]) -> bool:
            nonlocal windows_for_file
            truncated_flag = window_features.pop("_truncated", None)
            try:
                window_features.setdefault("gating_input", build_unified_gating(window_features))
            except Exception as exc:  # pragma: no cover - defensive
                self.log_fn(f"[SkipWindow] {info.path.name}: gating assembly failed ({exc}).")
                return False
            for key, value in window_features.items():
                if key == "_truncated":
                    continue
                if not isinstance(value, Tensor):
                    self.log_fn(
                        f"[SkipWindow] {info.path.name}: feature '{key}' missing tensor data; dropping window."
                    )
                    return False
                batch_features[key].append(value)
            batch_targets.append(int(info.label))
            self.stats[info.path]["windows"] += 1
            if truncated_flag is not None:
                self.stats[info.path]["truncated_windows"] += 1
            windows_for_file += 1
            self.total_windows_processed += 1
            self._notify_windows(info, 1)
            self._maybe_log_progress(info, windows_for_file)
            return len(batch_targets) >= self.batch_size

        with PcapReader(str(info.path)) as reader:
            for pkt in reader:
                if self.file_timeout is not None and (time.monotonic() - start_time) >= self.file_timeout:
                    drop_reason = f"time budget {self.file_timeout:.1f}s exceeded"
                    break
                if self.max_packets_per_file is not None and packets_seen >= self.max_packets_per_file:
                    drop_reason = f"packet cap {self.max_packets_per_file} reached"
                    break
                packets_seen += 1
                if self.max_total_windows is not None and self.total_windows_processed >= self.max_total_windows:
                    stop_file = True
                    break
                dos_row: Optional[Dict[str, object]] = None
                if "dos" in self.tasks:
                    try:
                        dos_row = dos_scapy_pkt_to_row(pkt, SSDP_MULTICAST_V4, SSDP_MULTICAST_V6)
                    except Exception:
                        continue
                arp_row: Optional[Dict[str, object]] = None
                if "arp" in self.tasks:
                    try:
                        arp_row = arp_scapy_pkt_to_row(pkt)
                    except Exception:
                        arp_row = {
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
                fallback_arp_row = arp_row or {
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
                timestamp, tcp_flags, auto_row = _packet_to_auto(pkt, fallback_arp_row)
                length = float(len(pkt))
                rows: Dict[str, Dict[str, object]] = {}
                if "dos" in self.tasks and dos_row is not None:
                    rows["dos"] = dos_row
                if "arp" in self.tasks and arp_row is not None:
                    rows["arp"] = arp_row
                completed = manager.add_packet(rows, auto_row, timestamp, length, tcp_flags)
                for buffer in completed:
                    features = self._finalize_window(buffer, dos_state, arp_state)
                    if features is None:
                        continue
                    if _push_window(features):
                        yield self._stack_batch(batch_features, batch_targets)
                        batch_features = defaultdict(list)
                        batch_targets = []
                        self.stats[info.path]["batches"] += 1
                    if (
                        self.max_windows_per_file is not None
                        and windows_for_file >= self.max_windows_per_file
                    ):
                        stop_file = True
                        self._note_file_cap(info, windows_for_file)
                        break
                    if (
                        self.max_total_windows is not None
                        and self.total_windows_processed >= self.max_total_windows
                    ):
                        stop_file = True
                        self._note_window_cap(info)
                        break
                if stop_file:
                    break

        if not stop_file and drop_reason is None:
            for buffer in manager.flush():
                features = self._finalize_window(buffer, dos_state, arp_state)
                if features is None:
                    continue
                if _push_window(features):
                    yield self._stack_batch(batch_features, batch_targets)
                    batch_features = defaultdict(list)
                    batch_targets = []
                    self.stats[info.path]["batches"] += 1
                if (
                    self.max_windows_per_file is not None
                    and windows_for_file >= self.max_windows_per_file
                ):
                    self._note_file_cap(info, windows_for_file)
                    break
                if (
                    self.max_total_windows is not None
                    and self.total_windows_processed >= self.max_total_windows
                ):
                    self._note_window_cap(info)
                    break

        if drop_reason is not None:
            batch_features = defaultdict(list)
            batch_targets = []
            if self.stats[info.path]["batches"] == 0:
                self.total_windows_processed = max(0, self.total_windows_processed - windows_for_file)
                self.stats[info.path]["windows"] = 0
            self._mark_skip(info, drop_reason)
            return

        if batch_targets:
            yield self._stack_batch(batch_features, batch_targets)
            self.stats[info.path]["batches"] += 1

    def _stack_batch(
        self,
        batch_features: Dict[str, List[Tensor]],
        batch_targets: Sequence[int],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        features = {key: torch.stack(values, dim=0) for key, values in batch_features.items()}
        targets = torch.tensor(list(batch_targets), dtype=torch.long)
        return features, targets

    def _finalize_window(
        self,
        buffer: WindowBuffer,
        dos_state: Optional[SequenceState],
        arp_state: Optional[SequenceState],
    ) -> Optional[Dict[str, Tensor]]:
        auto_stats = buffer.auto_acc.finalize()
        if auto_stats is None:
            return None
        truncated = False
        auto_tensor = torch.from_numpy(prepare_auto_tensor(auto_stats))
        features: Dict[str, Tensor] = {"auto": auto_tensor}

        if "dos" in self.tasks:
            if dos_state is None:
                raise RuntimeError("DoS task selected but DOS state not initialised.")
            dos_rows = buffer.task_rows.get("dos", [])
            if dos_rows:
                try:
                    dos_seq_np, dos_extras = compute_dos_sequence_features(
                        dos_rows,
                        buffer.bin_indices.get("dos", []),
                        DOS_MICRO_BINS,
                        TOP_UDP_PORTS,
                    )
                    dos_static_vec, dos_static_names, _ = compute_dos_static_features(
                        dos_rows,
                        DOS_MICRO_BINS,
                        dos_extras["per_bin_total_pkts"],
                        TOP_UDP_PORTS,
                        WINDOW_SIZE,
                    )
                    dos_cnn_seq_tensor = torch.from_numpy(dos_seq_np.astype(np.float32))
                    dos_cnn_static_tensor = torch.from_numpy(prepare_dos_static(dos_static_vec.astype(np.float32)))
                    dos_window = _build_dos_window(buffer.index, buffer.start, buffer.end, dos_rows)
                    dos_features = DOS_FEATURE_EXTRACTOR._features_for_window(dos_window)
                    dos_lstm_seq_tensor = dos_state.update(dos_features, DOS_LSTM_SCALER)
                except Exception:
                    dos_cnn_seq_tensor = torch.zeros((DOS_MICRO_BINS, DOS_CNN_SEQ_IN_DIM), dtype=torch.float32)
                    dos_cnn_static_tensor = torch.zeros(DOS_CNN_STATIC_DIM, dtype=torch.float32)
                    dos_lstm_seq_tensor = torch.zeros(
                        (DOS_LSTM_SEQUENCE_LENGTH, DOS_LSTM_INPUT_DIM),
                        dtype=torch.float32,
                    )
            else:
                dos_cnn_seq_tensor = torch.zeros((DOS_MICRO_BINS, DOS_CNN_SEQ_IN_DIM), dtype=torch.float32)
                dos_cnn_static_tensor = torch.zeros(DOS_CNN_STATIC_DIM, dtype=torch.float32)
                dos_lstm_seq_tensor = torch.zeros(
                    (DOS_LSTM_SEQUENCE_LENGTH, DOS_LSTM_INPUT_DIM),
                    dtype=torch.float32,
                )
            truncated = truncated or buffer.truncated
            features.update(
                {
                    "dos_cnn_seq": dos_cnn_seq_tensor,
                    "dos_cnn_static": dos_cnn_static_tensor,
                    "dos_lstm_seq": dos_lstm_seq_tensor,
                    "dos_gating": torch.cat(
                        [
                            auto_tensor.reshape(-1),
                            dos_cnn_static_tensor.reshape(-1),
                            dos_cnn_seq_tensor.reshape(-1),
                            dos_lstm_seq_tensor.reshape(-1),
                        ],
                        dim=0,
                    ),
                }
            )

        if "arp" in self.tasks:
            if arp_state is None:
                raise RuntimeError("ARP task selected but ARP state not initialised.")
            arp_rows = buffer.task_rows.get("arp", [])
            if arp_rows:
                try:
                    arp_seq_np, arp_extras = compute_arp_sequence_features(
                        arp_rows,
                        buffer.bin_indices.get("arp", []),
                        ARP_MICRO_BINS,
                    )
                    arp_static_vec, arp_static_names, _ = compute_arp_static_features(
                        arp_rows,
                        ARP_MICRO_BINS,
                        arp_extras,
                        ARP_WINDOW_SIZE,
                    )
                    arp_cnn_seq_tensor = torch.from_numpy(arp_seq_np.astype(np.float32))
                    arp_cnn_static_tensor = torch.from_numpy(prepare_arp_static(arp_static_vec.astype(np.float32)))
                    arp_window = _build_arp_window(buffer.index, buffer.start, buffer.end, arp_rows)
                    arp_features = ARP_FEATURE_EXTRACTOR._features_for_window(arp_window)
                    arp_lstm_seq_tensor = arp_state.update(arp_features, ARP_LSTM_SCALER)
                except Exception:
                    arp_cnn_seq_tensor = torch.zeros((ARP_MICRO_BINS, ARP_CNN_SEQ_IN_DIM), dtype=torch.float32)
                    arp_cnn_static_tensor = torch.zeros(ARP_CNN_STATIC_DIM, dtype=torch.float32)
                    arp_lstm_seq_tensor = torch.zeros(
                        (ARP_LSTM_SEQUENCE_LENGTH, ARP_LSTM_INPUT_DIM),
                        dtype=torch.float32,
                    )
            else:
                arp_cnn_seq_tensor = torch.zeros((ARP_MICRO_BINS, ARP_CNN_SEQ_IN_DIM), dtype=torch.float32)
                arp_cnn_static_tensor = torch.zeros(ARP_CNN_STATIC_DIM, dtype=torch.float32)
                arp_lstm_seq_tensor = torch.zeros(
                    (ARP_LSTM_SEQUENCE_LENGTH, ARP_LSTM_INPUT_DIM),
                    dtype=torch.float32,
                )
            truncated = truncated or buffer.truncated
            features.update(
                {
                    "arp_cnn_seq": arp_cnn_seq_tensor,
                    "arp_cnn_static": arp_cnn_static_tensor,
                    "arp_lstm_seq": arp_lstm_seq_tensor,
                    "arp_gating": torch.cat(
                        [
                            auto_tensor.reshape(-1),
                            arp_cnn_static_tensor.reshape(-1),
                            arp_cnn_seq_tensor.reshape(-1),
                            arp_lstm_seq_tensor.reshape(-1),
                        ],
                        dim=0,
                    ),
                }
            )

        if len(features) <= 1:  # only auto tensor present
            return None
        truncated = truncated or buffer.truncated
        features["gating_input"] = build_unified_gating(features)
        if truncated:
            features["_truncated"] = torch.tensor([1], dtype=torch.float32)
        return features

    def __iter__(self) -> Iterator[Tuple[Dict[str, Tensor], Tensor]]:
        rng = random.Random(self.seed + self._epoch)
        indices = list(range(len(self.files)))
        if self.shuffle and len(indices) > 1:
            rng.shuffle(indices)
        for idx in indices:
            yield from self._iter_file(self.files[idx])


class CachedMoEDataset(IterableDataset[Tuple[Dict[str, Tensor], Tensor]]):
    """Iterable dataset backed by cached tensors."""

    def __init__(
        self,
        cache_entries: Sequence[Dict[str, Any]],
        tasks: Sequence[str],
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 17,
        max_windows_per_file: Optional[int] = None,
        max_total_windows: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tasks = tuple(tasks)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self.log_fn: Callable[[str], None] = print
        self._window_budget: Optional[int] = max_total_windows
        self.cache_entries: List[Dict[str, Any]] = [dict(entry) for entry in cache_entries]
        self.stats: Dict[Path, Dict[str, Any]] = {}
        self.skipped: Dict[Path, str] = {}
        self._window_indices: List[Tuple[int, int]] = []

        for idx, entry in enumerate(self.cache_entries):
            meta = entry.get("meta", {})
            source_path = Path(meta.get("source_path", f"cache_{idx}.pcap"))
            label = infer_label_from_metadata(meta)
            if label is None:
                label = 0
            label = int(label)
            entry["class_label"] = label
            self.stats[source_path] = {
                "windows": 0,
                "batches": 0,
                "label": label,
                "label_name": class_id_to_name(label),
                "truncated_windows": 0,
            }
            entry["path"] = source_path
            features = entry.get("features", {})
            if not features:
                reason = "missing features"
                self.skipped[source_path] = reason
                self.stats[source_path]["skipped_reason"] = reason
                entry["num_windows"] = 0
                continue
            feature_lengths = {key: tensor.shape[0] for key, tensor in features.items()}
            if not feature_lengths:
                reason = "empty feature tensors"
                self.skipped[source_path] = reason
                self.stats[source_path]["skipped_reason"] = reason
                entry["num_windows"] = 0
                continue
            window_count = min(feature_lengths.values())
            entry["num_windows"] = window_count
            raw_targets = entry.get("targets")
            if raw_targets is None:
                raw_targets = entry.get("labels")
            entry["targets"] = coerce_targets_from_cache(raw_targets, label, window_count)

        max_windows_per_file = max_windows_per_file if max_windows_per_file and max_windows_per_file > 0 else None
        max_total_windows = max_total_windows if max_total_windows and max_total_windows > 0 else None

        total_added = 0
        for idx, entry in enumerate(self.cache_entries):
            num_windows = entry.get("num_windows", 0)
            if num_windows <= 0:
                continue
            per_file_cap = num_windows
            if max_windows_per_file is not None:
                per_file_cap = min(per_file_cap, max_windows_per_file)
            for window_idx in range(per_file_cap):
                if max_total_windows is not None and total_added >= max_total_windows:
                    break
                self._window_indices.append((idx, window_idx))
                total_added += 1
            if max_total_windows is not None and total_added >= max_total_windows:
                break

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def set_log_fn(self, fn: Optional[Callable[[str], None]]) -> None:
        self.log_fn = fn or (lambda _: None)

    def set_window_budget(self, budget: Optional[int]) -> None:
        self._window_budget = budget

    def _stack_batch(
        self,
        batch_features: Dict[str, List[Tensor]],
        batch_targets: Sequence[int],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        features = {key: torch.stack(values, dim=0) for key, values in batch_features.items()}
        targets = torch.tensor(list(batch_targets), dtype=torch.long)
        return features, targets

    def __iter__(self) -> Iterator[Tuple[Dict[str, Tensor], Tensor]]:
        if not self._window_indices:
            return iter([])
        rng = random.Random(self.seed + self._epoch)
        order = list(range(len(self._window_indices)))
        if self.shuffle and len(order) > 1:
            rng.shuffle(order)

        batch_features: Dict[str, List[Tensor]] = defaultdict(list)
        batch_targets: List[int] = []
        batch_sources: Set[Path] = set()

        for position in order:
            file_idx, window_idx = self._window_indices[position]
            entry = self.cache_entries[file_idx]
            path = entry["path"]
            features = entry["features"]
            targets_tensor: Tensor = entry["targets"]

            window_features: Dict[str, Tensor] = {}
            for key, tensor in features.items():
                window_features[key] = tensor[window_idx]
            truncated_flag = window_features.pop("_truncated", None)
            try:
                window_features.setdefault("gating_input", build_unified_gating(window_features))
            except Exception as exc:  # pragma: no cover - defensive
                self.log_fn(f"[SkipWindow] Failed to assemble gating input from cache {path.name}: {exc}")
                continue
            invalid = False
            for key, value in window_features.items():
                if not isinstance(value, Tensor):
                    self.log_fn(
                        f"[SkipWindow] Cache {path.name}: feature '{key}' missing tensor data; dropping window."
                    )
                    invalid = True
                    break
            if invalid:
                continue
            for key, value in window_features.items():
                batch_features[key].append(value)
            target_value = int(targets_tensor[window_idx].item())
            batch_targets.append(target_value)

            self.stats[path]["windows"] += 1
            if truncated_flag is not None:
                self.stats[path]["truncated_windows"] += 1
            batch_sources.add(path)

            if len(batch_targets) >= self.batch_size:
                features_tensor, targets_tensor_out = self._stack_batch(batch_features, batch_targets)
                yield features_tensor, targets_tensor_out
                for source in batch_sources:
                    self.stats[source]["batches"] += 1
                batch_features = defaultdict(list)
                batch_targets = []
                batch_sources = set()

        if batch_targets:
            features_tensor, targets_tensor_out = self._stack_batch(batch_features, batch_targets)
            yield features_tensor, targets_tensor_out
            for source in batch_sources:
                self.stats[source]["batches"] += 1


def _load_labels_from_csv(base_dir: Path) -> Dict[Path, Dict[str, Any]]:
    manifest_path = base_dir / "labels.csv"
    if not manifest_path.exists():
        return {}
    entries: Dict[Path, Dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("labels.csv must include a header row with at least 'filename' and 'label'.")
        name_fields = [field for field in ("filename", "file", "pcap", "path") if field in reader.fieldnames]
        if not name_fields:
            raise ValueError("labels.csv header must include one of: filename, file, pcap, path.")
        label_fields = [field for field in ("label", "class", "target") if field in reader.fieldnames]
        if not label_fields:
            raise ValueError("labels.csv header must include a 'label' (or class/target) column.")
        for row in reader:
            raw_name = next((row.get(field) for field in name_fields if row.get(field)), None)
            if not raw_name:
                continue
            raw_label = next((row.get(field) for field in label_fields if row.get(field) is not None), None)
            if raw_label is None:
                raise ValueError(f"Row for '{raw_name}' missing label column in labels.csv.")
            label_id = resolve_label_id(raw_label)
            candidate = Path(raw_name)
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            if not candidate.exists():
                # Fallback: treat value as filename relative to base directory root.
                candidate = (base_dir / Path(raw_name).name).resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"Manifest entry '{raw_name}' does not match an existing PCAP file.")
            entries[candidate] = {
                "label": label_id,
                "meta": {
                    "label_source": "labels.csv",
                    "label_name": class_id_to_name(label_id),
                    "manifest_path": str(manifest_path),
                },
            }
    return entries


def _load_labels_from_subdirs(base_dir: Path) -> Dict[Path, Dict[str, Any]]:
    entries: Dict[Path, Dict[str, Any]] = {}
    for label_id, folder_name in CLASS_SUBDIRS.items():
        folder = base_dir / folder_name
        if not folder.exists():
            continue
        for pcap in sorted(folder.rglob("*.pcap")):
            path = pcap.resolve()
            entries.setdefault(
                path,
                {
                    "label": label_id,
                    "meta": {
                        "label_source": "folder",
                        "folder": str(folder),
                        "label_name": class_id_to_name(label_id),
                    },
                },
            )
    return entries


def discover_pcaps(tasks: Optional[Sequence[str]] = None) -> List[PcapInfo]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_labels_from_csv(SAMPLES_DIR)
    folder_records = _load_labels_from_subdirs(SAMPLES_DIR)
    for path, payload in folder_records.items():
        records.setdefault(path, payload)

    infos: List[PcapInfo] = []
    for path, payload in records.items():
        label = int(payload["label"])
        meta = dict(payload.get("meta", {}))
        infos.append(PcapInfo(path=path, label=label, meta=meta))

    infos.sort(key=lambda info: (info.path.name.lower(), str(info.path)))
    return infos


def tasks_slug(tasks: Sequence[str]) -> str:
    return "-".join(sorted(tasks))


def load_cache_entries(
    cache_base: Path,
    files: Sequence[PcapInfo],
    tasks: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Path], List[Tuple[Path, str]]]:
    slug = tasks_slug(tasks)
    cache_dir = cache_base / slug
    entries: List[Dict[str, Any]] = []
    missing: List[Path] = []
    mismatched: List[Tuple[Path, str]] = []

    if not cache_dir.exists():
        missing.extend([info.path for info in files])
        return entries, missing, mismatched

    task_signature = tuple(sorted(tasks))
    if not files:
        for cache_path in sorted(cache_dir.glob("*.pt")):
            try:
                data = torch.load(cache_path, map_location="cpu")
            except Exception as exc:  # pragma: no cover - IO failure
                mismatched.append((cache_path, f"failed to load cache ({exc})"))
                continue
            cache_tasks = tuple(sorted(data.get("tasks", [])))
            if cache_tasks and cache_tasks != task_signature:
                mismatched.append((cache_path, "cache tasks mismatch"))
                continue
            features = data.get("features")
            if not isinstance(features, dict) or not features:
                mismatched.append((cache_path, "cache missing feature tensors"))
                continue
            meta = data.get("meta", {})
            source_path = Path(meta.get("source_path", cache_path.stem))
            cache_label = infer_label_from_metadata(meta)
            if cache_label is None:
                cache_label = 0
            info = PcapInfo(path=source_path, label=int(cache_label), meta=dict(meta))
            entries.append({"data": data, "path": source_path, "cache_path": cache_path, "info": info})
        return entries, missing, mismatched

    for info in files:
        cache_path = cache_dir / f"{info.path.name}.pt"
        if not cache_path.exists():
            missing.append(info.path)
            continue
        try:
            data = torch.load(cache_path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - IO failure
            mismatched.append((info.path, f"failed to load cache ({exc})"))
            continue
        cache_tasks = tuple(sorted(data.get("tasks", [])))
        if cache_tasks and cache_tasks != task_signature:
            mismatched.append((info.path, "cache tasks mismatch"))
            continue
        features = data.get("features")
        if not isinstance(features, dict) or not features:
            mismatched.append((info.path, "cache missing feature tensors"))
            continue
        meta = data.get("meta", {})
        cache_label = infer_label_from_metadata(meta)
        if cache_label is not None:
            updated_meta = dict(info.meta)
            updated_meta.update(meta)
            info = PcapInfo(path=info.path, label=int(cache_label), meta=updated_meta)
        entries.append({"data": data, "path": info.path, "cache_path": cache_path, "info": info})
    return entries, missing, mismatched


__all__ = [
    "AutoFeatureAccumulator",
    "CachedMoEDataset",
    "class_id_to_name",
    "MoEDataset",
    "PcapInfo",
    "SequenceState",
    "StreamingWindowManager",
    "discover_pcaps",
    "load_cache_entries",
    "tasks_slug",
    "CACHE_ROOT",
]
