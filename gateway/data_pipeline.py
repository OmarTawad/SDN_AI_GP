"""Streaming and cached dataset utilities for gateway training."""

from __future__ import annotations

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
    labels: Dict[str, int]


class MoEDataset(IterableDataset[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]):
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
            labels_snapshot = {task: int(info.labels.get(task, 0)) for task in self.tasks}
            self.stats[info.path] = {"windows": 0, "batches": 0, "labels": labels_snapshot, "truncated_windows": 0}

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
        label_bits = ", ".join(
            f"{task}={'attack' if info.labels.get(task, 0) else 'normal'}" for task in self.tasks
        )
        if not label_bits:
            label_bits = "n/a"
        self.log_fn(f"[Stream] {info.path.name}: labels={label_bits}")
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

    def _iter_file(self, info: PcapInfo) -> Iterator[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]:
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
        truncated = False
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
        batch_labels: Dict[str, List[float]] = {task: [] for task in self.tasks}
        windows_for_file = 0
        stop_file = False
        drop_reason: Optional[str] = None
        packets_seen = 0
        start_time = time.monotonic()

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
                    truncated_flag = features.pop("_truncated", None)
                    for key, value in features.items():
                        batch_features[key].append(value)
                    for task in batch_labels:
                        batch_labels[task].append(float(info.labels.get(task, 0)))
                    self.stats[info.path]["windows"] += 1
                    if truncated_flag is not None:
                        self.stats[info.path]["truncated_windows"] += 1
                    windows_for_file += 1
                    self.total_windows_processed += 1
                    self._notify_windows(info, 1)
                    self._maybe_log_progress(info, windows_for_file)
                    first_task = next(iter(batch_labels)) if batch_labels else None
                    if first_task and len(batch_labels[first_task]) >= self.batch_size:
                        yield self._stack_batch(batch_features, batch_labels)
                        batch_features = defaultdict(list)
                        batch_labels = {task: [] for task in self.tasks}
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
                for key, value in features.items():
                    if key == "_truncated":
                        continue
                    batch_features[key].append(value)
                for task in batch_labels:
                    batch_labels[task].append(float(info.labels.get(task, 0)))
                self.stats[info.path]["windows"] += 1
                windows_for_file += 1
                self.total_windows_processed += 1
                self._notify_windows(info, 1)
                self._maybe_log_progress(info, windows_for_file)
                first_task = next(iter(batch_labels)) if batch_labels else None
                if first_task and len(batch_labels[first_task]) >= self.batch_size:
                    yield self._stack_batch(batch_features, batch_labels)
                    batch_features = defaultdict(list)
                    batch_labels = {task: [] for task in self.tasks}
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
            batch_labels = {task: [] for task in self.tasks}
            if self.stats[info.path]["batches"] == 0:
                self.total_windows_processed = max(0, self.total_windows_processed - windows_for_file)
                self.stats[info.path]["windows"] = 0
            self._mark_skip(info, drop_reason)
            return

        remaining_task = next(iter(batch_labels)) if batch_labels else None
        if remaining_task and batch_labels[remaining_task]:
            yield self._stack_batch(batch_features, batch_labels)
            self.stats[info.path]["batches"] += 1

    def _stack_batch(
        self,
        batch_features: Dict[str, List[Tensor]],
        batch_labels: Dict[str, List[float]],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        features = {key: torch.stack(values, dim=0) for key, values in batch_features.items()}
        labels = {task: torch.tensor(values, dtype=torch.float32) for task, values in batch_labels.items()}
        return features, labels

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
            if not dos_rows:
                return None
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
            truncated = truncated or buffer.truncated
            dos_gating = torch.cat(
                [
                    auto_tensor,
                    dos_cnn_static_tensor,
                    dos_cnn_seq_tensor.reshape(-1),
                    dos_lstm_seq_tensor.reshape(-1),
                ],
                dim=0,
            )
            features.update(
                {
                    "dos_cnn_seq": dos_cnn_seq_tensor,
                    "dos_cnn_static": dos_cnn_static_tensor,
                    "dos_lstm_seq": dos_lstm_seq_tensor,
                    "dos_gating": dos_gating,
                }
            )

        if "arp" in self.tasks:
            if arp_state is None:
                raise RuntimeError("ARP task selected but ARP state not initialised.")
            arp_rows = buffer.task_rows.get("arp", [])
            if not arp_rows:
                return None
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
            truncated = truncated or buffer.truncated
            arp_gating = torch.cat(
                [
                    auto_tensor,
                    arp_cnn_static_tensor,
                    arp_cnn_seq_tensor.reshape(-1),
                    arp_lstm_seq_tensor.reshape(-1),
                ],
                dim=0,
            )
            features.update(
                {
                    "arp_cnn_seq": arp_cnn_seq_tensor,
                    "arp_cnn_static": arp_cnn_static_tensor,
                    "arp_lstm_seq": arp_lstm_seq_tensor,
                    "arp_gating": arp_gating,
                }
            )

        if len(features) <= 1:  # only auto tensor present
            return None
        truncated = truncated or buffer.truncated
        if truncated:
            features["_truncated"] = torch.tensor([1], dtype=torch.float32)
        return features

    def __iter__(self) -> Iterator[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]:
        rng = random.Random(self.seed + self._epoch)
        indices = list(range(len(self.files)))
        if self.shuffle and len(indices) > 1:
            rng.shuffle(indices)
        for idx in indices:
            yield from self._iter_file(self.files[idx])


class CachedMoEDataset(IterableDataset[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]):
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
            labels_snapshot = {task: int(meta.get("labels", {}).get(task, 0)) for task in self.tasks}
            self.stats[source_path] = {"windows": 0, "batches": 0, "labels": labels_snapshot}
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
            entry["num_windows"] = min(feature_lengths.values())

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
        batch_labels: Dict[str, List[float]],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        features = {key: torch.stack(values, dim=0) for key, values in batch_features.items()}
        labels = {task: torch.tensor(values, dtype=torch.float32) for task, values in batch_labels.items()}
        return features, labels

    def __iter__(self) -> Iterator[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]:
        if not self._window_indices:
            return iter([])
        rng = random.Random(self.seed + self._epoch)
        order = list(range(len(self._window_indices)))
        if self.shuffle and len(order) > 1:
            rng.shuffle(order)

        batch_features: Dict[str, List[Tensor]] = defaultdict(list)
        batch_labels: Dict[str, List[float]] = {task: [] for task in self.tasks}
        batch_sources: Set[Path] = set()

        for position in order:
            file_idx, window_idx = self._window_indices[position]
            entry = self.cache_entries[file_idx]
            path = entry["path"]
            features = entry["features"]
            labels = entry["labels"]

            for key, tensor in features.items():
                batch_features[key].append(tensor[window_idx])
            for task in self.tasks:
                batch_labels[task].append(float(labels[task][window_idx].item()))

            self.stats[path]["windows"] += 1
            batch_sources.add(path)

            first_task = next(iter(batch_labels)) if batch_labels else None
            if first_task and len(batch_labels[first_task]) >= self.batch_size:
                features_tensor, labels_tensor = self._stack_batch(batch_features, batch_labels)
                yield features_tensor, labels_tensor
                for source in batch_sources:
                    self.stats[source]["batches"] += 1
                batch_features = defaultdict(list)
                batch_labels = {task: [] for task in self.tasks}
                batch_sources = set()

        if batch_labels:
            first_task = next(iter(batch_labels)) if batch_labels else None
            if first_task and batch_labels[first_task]:
                features_tensor, labels_tensor = self._stack_batch(batch_features, batch_labels)
                yield features_tensor, labels_tensor
                for source in batch_sources:
                    self.stats[source]["batches"] += 1


def _infer_label_from_name(name: str, attack_tokens: Sequence[str], benign_tokens: Sequence[str]) -> Optional[int]:
    lower = name.lower()
    for token in attack_tokens:
        if token in lower:
            return 1
    for token in benign_tokens:
        if token in lower:
            return 0
    return None


def discover_pcaps(tasks: Sequence[str]) -> List[PcapInfo]:
    tasks = tuple(dict.fromkeys(task.lower() for task in tasks))
    registry: Dict[Path, Dict[str, int]] = {}

    def _ensure_entry(path: Path) -> Dict[str, int]:
        return registry.setdefault(path, {task: 0 for task in tasks})

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    def _sort_key(path: Path) -> Tuple[float, str]:
        try:
            size = float(path.stat().st_size)
        except OSError:
            size = float("inf")
        return (size, path.name.lower())

    for pcap in sorted(SAMPLES_DIR.glob("*.pcap"), key=_sort_key):
        entry = None
        if "dos" in tasks:
            dos_label = _infer_label_from_name(pcap.name, ("mixed", "attack", "dos", "flood"), ("normal",))
            if dos_label is not None:
                entry = _ensure_entry(pcap)
                entry["dos"] = dos_label
        if "arp" in tasks:
            arp_label = _infer_label_from_name(pcap.name, ("attack", "spoof", "poison", "arp"), ("normal",))
            if arp_label is not None:
                entry = _ensure_entry(pcap)
                entry["arp"] = arp_label

    infos = [PcapInfo(path=path, labels=labels) for path, labels in registry.items()]
    infos.sort(key=lambda info: str(info.path))
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
        if cache_tasks != task_signature:
            mismatched.append((info.path, "cache tasks mismatch"))
            continue
        features = data.get("features")
        labels = data.get("labels")
        if not isinstance(features, dict) or not isinstance(labels, dict):
            mismatched.append((info.path, "cache missing feature tensors"))
            continue
        entries.append({"data": data, "path": info.path, "cache_path": cache_path})
    return entries, missing, mismatched


__all__ = [
    "AutoFeatureAccumulator",
    "CachedMoEDataset",
    "MoEDataset",
    "PcapInfo",
    "SequenceState",
    "StreamingWindowManager",
    "discover_pcaps",
    "load_cache_entries",
    "tasks_slug",
    "CACHE_ROOT",
]
