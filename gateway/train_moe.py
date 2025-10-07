"""Train the Mixture-of-Experts gating network on streaming PCAP features."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import sys

import joblib
import numpy as np
import torch
import yaml
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether
from scapy.utils import PcapReader
from torch import Tensor, nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[0]
DOSDET_ROOT = (PROJECT_ROOT / "dosdet").resolve()
if str(DOSDET_ROOT) not in sys.path:
    sys.path.insert(0, str(DOSDET_ROOT))

from dosdet.data.packet_to_frame import scapy_pkt_to_row
from dosdet.features.feature_slimming import StaticSlimmer
from dosdet.features.scaler import RobustScaler
from dosdet.features.seq_features import compute_sequence_features
from dosdet.features.static_features import compute_static_features
from gateway.moe_model import (
    AUTOENCODER_SPEC,
    CNN_MICRO_BINS,
    CNN_SEQ_IN_DIM,
    CNN_STATIC_DIM,
    CNN_SPEC,
    LSTM_CONFIG,
    LSTM_INPUT_DIM,
    LSTM_SPEC,
    build_moe_model,
)
from Neural_LSTM.src.dos_detector.config import load_config
from Neural_LSTM.src.dos_detector.features.feature_engineering import FeatureExtractor, HostHistory
from Neural_LSTM.src.dos_detector.data.structures import PacketRecord, Window

# ---------------------------------------------------------------------------
# Project artefacts and feature metadata
# ---------------------------------------------------------------------------

AUTO_ARTIFACT_DIR = PROJECT_ROOT / "autoencoder" / "data" / "artifacts"
CNN_ARTIFACT_DIR = PROJECT_ROOT / "dosdet" / "artifacts_fast"
LSTM_ARTIFACT_DIR = PROJECT_ROOT / "Neural_LSTM" / "models"
LSTM_CONFIG_PATH = PROJECT_ROOT / "Neural_LSTM" / "configs" / "config.yaml"
DOS_CONFIG_PATH = PROJECT_ROOT / "dosdet" / "config.yaml"

AUTO_MODEL_CONFIG = json.loads((AUTO_ARTIFACT_DIR / "model_config.json").read_text())
AUTO_FEATURE_NAMES: List[str] = list(AUTO_MODEL_CONFIG.get("feature_names", []))
AUTO_LOG_FEATURES: List[str] = list(AUTO_MODEL_CONFIG.get("log_features", []))
AUTO_FEATURE_INDEX = {name: idx for idx, name in enumerate(AUTO_FEATURE_NAMES)}
AUTO_CLIP_BOUNDS = json.loads((AUTO_ARTIFACT_DIR / "clip_bounds.json").read_text())
AUTO_CLIP_LOWER: Dict[str, float] = {k: float(v) for k, v in AUTO_CLIP_BOUNDS.get("lower", {}).items()}
AUTO_CLIP_UPPER: Dict[str, float] = {k: float(v) for k, v in AUTO_CLIP_BOUNDS.get("upper", {}).items()}
AUTO_SCALER = joblib.load(AUTO_ARTIFACT_DIR / "scaler.pkl")

with DOS_CONFIG_PATH.open("r", encoding="utf-8") as cfg_file:
    DOS_CONFIG = yaml.safe_load(cfg_file)
TOP_UDP_PORTS: List[int] = [int(p) for p in DOS_CONFIG["data"]["top_k_udp_ports"]]
SSDP_MULTICAST_V4: str = DOS_CONFIG["features"]["ssdp_multicast_ipv4"]
SSDP_MULTICAST_V6: str = DOS_CONFIG["features"]["ssdp_multicast_ipv6"]

CNN_SCALER = RobustScaler.load(str(CNN_ARTIFACT_DIR))
CNN_SLIMMER = StaticSlimmer()
CNN_SLIMMER.load(str(CNN_ARTIFACT_DIR))

LSTM_SCALER = joblib.load(LSTM_ARTIFACT_DIR / "feature_scaler.joblib")
LSTM_CFG = load_config(LSTM_CONFIG_PATH)
LSTM_FEATURE_EXTRACTOR = FeatureExtractor(LSTM_CFG.feature, LSTM_CFG.windowing.window_size)
LSTM_ROLLING_HISTORY = LSTM_CFG.feature.rolling_zscore_window
LSTM_HOST_HISTORY = LSTM_CFG.feature.host_history
LSTM_SEQUENCE_LENGTH = LSTM_CFG.windowing.sequence_length
WINDOW_SIZE = float(LSTM_CFG.windowing.window_size)
WINDOW_STRIDE = float(LSTM_CFG.windowing.hop_size)
MICRO_BINS = CNN_MICRO_BINS

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


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

    def add(self, row: Dict[str, object], length: float, timestamp: float, tcp_flags: Optional[int]) -> None:
        self.count += 1
        self.byte_count += float(length)
        delta = length - self.mean_len
        self.mean_len += delta / self.count
        self.m2_len += delta * (length - self.mean_len)
        self.min_len = min(self.min_len, float(length))
        self.max_len = max(self.max_len, float(length))

        if self.last_ts is not None:
            iat = max(timestamp - self.last_ts, 0.0)
            self.iat_count += 1
            delta_iat = iat - self.iat_mean
            self.iat_mean += delta_iat / self.iat_count
            self.iat_m2 += delta_iat * (iat - self.iat_mean)
        self.last_ts = timestamp

        if row.get("is_tcp", 0):
            self.proto_counts.update(["TCP"])
        elif row.get("is_udp", 0):
            self.proto_counts.update(["UDP"])
        elif row.get("is_icmp", 0):
            self.proto_counts.update(["ICMP"])
        else:
            self.proto_counts.update(["OTHER"])

        self.tcp_flag_counts.update({"SYN": int(row.get("tcp_syn", 0))})
        self.tcp_flag_counts.update({"SYNACK": int(row.get("tcp_synack", 0))})
        if tcp_flags is not None:
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
    rows: List[Dict[str, object]] = field(default_factory=list)
    bin_indices: List[int] = field(default_factory=list)
    packets: List[PacketRecord] = field(default_factory=list)
    auto_acc: AutoFeatureAccumulator = field(init=False)

    def __post_init__(self) -> None:
        self.auto_acc = AutoFeatureAccumulator(duration=self.duration)

    def add(
        self,
        row: Dict[str, object],
        record: PacketRecord,
        timestamp: float,
        length: float,
        tcp_flags: Optional[int],
        bin_index: int,
    ) -> None:
        self.rows.append(row)
        self.bin_indices.append(bin_index)
        self.packets.append(record)
        self.auto_acc.add(row, length, timestamp, tcp_flags)


class StreamingWindowManager:
    def __init__(self, window_size: float, stride: float, micro_bins: int) -> None:
        self.window_size = window_size
        self.stride = stride
        self.micro_bins = micro_bins
        self.bin_width = window_size / max(micro_bins, 1)
        self.next_window_start: Optional[float] = None
        self.active: Deque[WindowBuffer] = deque()
        self.index = 0

    def _open_window(self, start: float) -> None:
        buffer = WindowBuffer(
            start=start,
            end=start + self.window_size,
            index=self.index,
            duration=self.window_size,
        )
        self.index += 1
        self.active.append(buffer)

    def add_packet(
        self,
        row: Dict[str, object],
        record: PacketRecord,
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
                bin_index = int(offset / self.bin_width) if self.bin_width > 0 else 0
                bin_index = max(0, min(self.micro_bins - 1, bin_index))
                window.add(row, record, timestamp, length, tcp_flags, bin_index)
        return completed

    def flush(self) -> List[WindowBuffer]:
        remaining = list(self.active)
        self.active.clear()
        return remaining


def _prepare_auto_tensor(stats: Dict[str, float]) -> Tensor:
    values = np.zeros(len(AUTO_FEATURE_NAMES), dtype=np.float32)
    for name, idx in AUTO_FEATURE_INDEX.items():
        values[idx] = float(stats.get(name, 0.0))
    for name in AUTO_LOG_FEATURES:
        idx = AUTO_FEATURE_INDEX.get(name)
        if idx is None:
            continue
        values[idx] = math.log1p(max(values[idx], 0.0))
    for name, idx in AUTO_FEATURE_INDEX.items():
        lower = AUTO_CLIP_LOWER.get(name)
        upper = AUTO_CLIP_UPPER.get(name)
        if lower is not None:
            values[idx] = max(values[idx], float(lower))
        if upper is not None:
            values[idx] = min(values[idx], float(upper))
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = AUTO_SCALER.transform(values.reshape(1, -1)).astype(np.float32)
    return torch.from_numpy(scaled.squeeze(0))


def _prepare_cnn_static(static_vec: np.ndarray) -> Tensor:
    names_stub = [f"f_{i}" for i in range(static_vec.shape[0])]
    scaled = CNN_SCALER.transform(static_vec.reshape(1, -1), names_stub)
    slim = CNN_SLIMMER.transform(scaled)
    return torch.from_numpy(slim.astype(np.float32).squeeze(0))


def _packet_to_record(pkt) -> Tuple[PacketRecord, float, int]:
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

    record = PacketRecord(
        timestamp=timestamp,
        src_mac=src_mac,
        dst_mac=dst_mac,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        length=int(length),
        ttl=ttl,
        tcp_flags=tcp_flags,
        payload_len=payload_len,
        info=info,
    )
    return record, timestamp, tcp_flags or 0


@dataclass
class LSTMState:
    prev_features: Optional[Dict[str, float]]
    global_history: Deque[float]
    host_histories: Dict[str, HostHistory]
    feature_names: Optional[List[str]]
    buffer: Deque[Tensor]


class MoEDataset(IterableDataset[Tuple[Dict[str, Tensor], Tensor]]):
    def __init__(
        self,
        files: Sequence[Tuple[Path, int]],
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 17,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self.stats: Dict[Path, Dict[str, float]] = defaultdict(lambda: {"windows": 0, "batches": 0})
        for path, label in self.files:
            self.stats[path]["label"] = float(label)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _iter_file(self, path: Path, label: int) -> Iterator[Tuple[Dict[str, Tensor], Tensor]]:
        if not path.exists():
            return
        manager = StreamingWindowManager(WINDOW_SIZE, WINDOW_STRIDE, MICRO_BINS)
        lstm_state = LSTMState(
            prev_features=None,
            global_history=deque(maxlen=LSTM_ROLLING_HISTORY),
            host_histories=defaultdict(
                lambda: HostHistory(deque(maxlen=LSTM_HOST_HISTORY), LSTM_HOST_HISTORY)
            ),
            feature_names=None,
            buffer=deque(maxlen=LSTM_SEQUENCE_LENGTH),
        )

        batch_features: Dict[str, List[Tensor]] = defaultdict(list)
        batch_labels: List[float] = []

        with PcapReader(str(path)) as reader:
            for pkt in reader:
                try:
                    row = scapy_pkt_to_row(pkt, SSDP_MULTICAST_V4, SSDP_MULTICAST_V6)
                except Exception:
                    continue
                record, timestamp, tcp_flags = _packet_to_record(pkt)
                length = float(len(pkt))
                completed = manager.add_packet(row, record, timestamp, length, tcp_flags)
                for buffer in completed:
                    features = self._finalize_window(buffer, lstm_state)
                    if features is None:
                        continue
                    for key, value in features.items():
                        batch_features[key].append(value)
                    batch_labels.append(float(label))
                    self.stats[path]["windows"] += 1
                    if len(batch_labels) >= self.batch_size:
                        yield self._stack_batch(batch_features, batch_labels)
                        batch_features = defaultdict(list)
                        batch_labels = []
                        self.stats[path]["batches"] += 1

        for buffer in manager.flush():
            features = self._finalize_window(buffer, lstm_state)
            if features is None:
                continue
            for key, value in features.items():
                batch_features[key].append(value)
            batch_labels.append(float(label))
            self.stats[path]["windows"] += 1
            if len(batch_labels) >= self.batch_size:
                yield self._stack_batch(batch_features, batch_labels)
                batch_features = defaultdict(list)
                batch_labels = []
                self.stats[path]["batches"] += 1

        if batch_labels:
            yield self._stack_batch(batch_features, batch_labels)
            self.stats[path]["batches"] += 1

    def _stack_batch(
        self,
        batch_features: Dict[str, List[Tensor]],
        batch_labels: Sequence[float],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        stacked = {key: torch.stack(values, dim=0) for key, values in batch_features.items()}
        labels = torch.tensor(batch_labels, dtype=torch.float32)
        return stacked, labels

    def _finalize_window(
        self,
        buffer: WindowBuffer,
        lstm_state: LSTMState,
    ) -> Optional[Dict[str, Tensor]]:
        if not buffer.rows:
            return None
        auto_stats = buffer.auto_acc.finalize()
        if auto_stats is None:
            return None
        auto_tensor = _prepare_auto_tensor(auto_stats)

        seq_np, extras = compute_sequence_features(buffer.rows, buffer.bin_indices, MICRO_BINS, TOP_UDP_PORTS)
        static_vec, _, _ = compute_static_features(
            buffer.rows,
            MICRO_BINS,
            extras["per_bin_total_pkts"],
            TOP_UDP_PORTS,
            WINDOW_SIZE,
        )
        cnn_seq_tensor = torch.from_numpy(seq_np.astype(np.float32))
        cnn_static_tensor = _prepare_cnn_static(static_vec.astype(np.float32))

        window = Window(
            index=buffer.index,
            start_time=buffer.start,
            end_time=buffer.end,
            packets=buffer.packets,
        )
        features = LSTM_FEATURE_EXTRACTOR._features_for_window(
            window=window,
            prev_features=lstm_state.prev_features,
            global_rate_history=lstm_state.global_history,
            host_histories=lstm_state.host_histories,
        )
        lstm_state.prev_features = features
        lstm_state.global_history.append(features.get("packet_rate", 0.0))
        if lstm_state.feature_names is None:
            lstm_state.feature_names = list(features.keys())
        lstm_vector = np.array(
            [float(features.get(name, 0.0)) for name in lstm_state.feature_names],
            dtype=np.float32,
        )
        lstm_scaled = LSTM_SCALER.transform(lstm_vector.reshape(1, -1)).astype(np.float32)
        lstm_tensor = torch.from_numpy(lstm_scaled.squeeze(0))
        lstm_state.buffer.append(lstm_tensor)
        if len(lstm_state.buffer) < LSTM_SEQUENCE_LENGTH:
            pad_count = LSTM_SEQUENCE_LENGTH - len(lstm_state.buffer)
            pad_tensor = torch.zeros_like(lstm_tensor)
            seq_tensors = [pad_tensor for _ in range(pad_count)] + list(lstm_state.buffer)
        else:
            seq_tensors = list(lstm_state.buffer)
        lstm_seq_tensor = torch.stack(seq_tensors, dim=0)

        gating_components = [
            auto_tensor,
            cnn_static_tensor,
            cnn_seq_tensor.reshape(-1),
            lstm_seq_tensor.reshape(-1),
        ]
        gating_tensor = torch.cat(gating_components, dim=0)

        sample = {
            "gating": gating_tensor,
            "auto": auto_tensor,
            "cnn_seq": cnn_seq_tensor,
            "cnn_static": cnn_static_tensor,
            "lstm_seq": lstm_seq_tensor,
        }
        return sample

    def __iter__(self) -> Iterator[Tuple[Dict[str, Tensor], Tensor]]:
        rng = random.Random(self.seed + self._epoch)
        indices = list(range(len(self.files)))
        if self.shuffle and len(indices) > 1:
            rng.shuffle(indices)
        for idx in indices:
            path, label = self.files[idx]
            yield from self._iter_file(path, label)


def discover_pcaps(paths: Sequence[Path]) -> List[Tuple[Path, int]]:
    labeled: List[Tuple[Path, int]] = []
    for root in paths:
        if not root.exists() or not root.is_dir():
            continue
        for pcap in sorted(root.glob("*.pcap")):
            name = pcap.name.lower()
            if "ssdpflood" in name:
                continue
            if "normal" in name:
                label = 0
            elif "mixed" in name:
                label = 1
            else:
                continue
            labeled.append((pcap, label))
    return labeled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MoE gate on streaming PCAP features")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=17, help="Random seed (default: 17)")
    return parser.parse_args()


def build_dataset(files: Sequence[Tuple[Path, int]], batch_size: int, seed: int) -> MoEDataset:
    return MoEDataset(files=files, batch_size=batch_size, shuffle=True, seed=seed)


def main() -> None:
    args = parse_args()

    candidate_roots = [
        Path("/home/omar/SDN_IoT_Simulated/dosdet/samples"),
        Path("/home/omar"),
    ]
    files = discover_pcaps(candidate_roots)
    if not files:
        raise RuntimeError("No eligible PCAP files found. Check the directories and filenames.")

    dataset = build_dataset(files, batch_size=args.batch_size, seed=args.seed)
    model = build_moe_model(
        specs=[AUTOENCODER_SPEC, CNN_SPEC, LSTM_SPEC],
        gating_input_dim=(
            len(AUTO_FEATURE_NAMES)
            + CNN_STATIC_DIM
            + CNN_SEQ_IN_DIM * MICRO_BINS
            + LSTM_SEQUENCE_LENGTH * LSTM_INPUT_DIM
        ),
        gating_hidden_dim=128,
        device=torch.device("cpu"),
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.gating.parameters(), lr=args.learning_rate)

    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        epoch_loss = 0.0
        batch_count = 0
        printed_shapes = False
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
        for batch_idx, (features, labels) in enumerate(progress, start=1):
            if not printed_shapes:
                print(
                    "[Shapes] gating=%s auto=%s cnn_seq=%s cnn_static=%s lstm_seq=%s"
                    % (
                        tuple(features["gating"].shape),
                        tuple(features["auto"].shape),
                        tuple(features["cnn_seq"].shape),
                        tuple(features["cnn_static"].shape),
                        tuple(features["lstm_seq"].shape),
                    )
                )
                printed_shapes = True

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            epoch_loss += loss_value
            batch_count += 1
            progress.set_postfix(loss=f"{loss_value:.4f}")
            progress.write(f"Epoch {epoch + 1} Batch {batch_idx}: loss={loss_value:.4f}")

        average_loss = epoch_loss / batch_count if batch_count else float("nan")
        print(f"Epoch {epoch + 1} average loss: {average_loss:.4f}")

    torch.save(model.gating.state_dict(), "moe_gate.pt")
    print("Saved gating weights to moe_gate.pt")

    print("\nPCAP summary:")
    label_totals = defaultdict(int)
    for path, info in dataset.stats.items():
        label_value = int(info.get("label", -1))
        windows = int(info.get("windows", 0))
        batches = int(info.get("batches", 0))
        label_name = "mixed" if label_value == 1 else "normal"
        label_totals[label_name] += windows
        print(f" - {path}: label={label_name}, windows={windows}, batches={batches}")

    print("\nWindow counts by label:")
    for label_name, count in label_totals.items():
        print(f" * {label_name}: {count} windows")


if __name__ == "__main__":
    main()
