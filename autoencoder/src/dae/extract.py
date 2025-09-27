from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether
from scapy.utils import RawPcapReader
from tqdm import tqdm

from .config import Config
from .features import FeatureExtractor
from .logging import get_logger
from .utils_io import ParquetBatchWriter
from .window import PacketSummary, SlidingWindowManager, WindowStats


@dataclass
class ExtractionStats:
    packets: int
    windows: int
    files: int


def _packet_timestamp(metadata) -> float:
    """Extract a floating-point timestamp from RawPcapReader metadata."""

    # Common libpcap fields
    if hasattr(metadata, "sec") and hasattr(metadata, "usec"):
        return float(metadata.sec) + float(metadata.usec) / 1_000_000.0

    if hasattr(metadata, "seconds") and hasattr(metadata, "microseconds"):
        return float(metadata.seconds) + float(metadata.microseconds) / 1_000_000.0

    if hasattr(metadata, "tshigh") and hasattr(metadata, "tslow"):
        tsresol = float(getattr(metadata, "tsresol", 1_000_000))
        if tsresol == 0:
            tsresol = 1_000_000
        timestamp_raw = (int(getattr(metadata, "tshigh", 0)) << 32) + int(getattr(metadata, "tslow", 0))
        return timestamp_raw / tsresol

    # Scapy >= 2.5 may expose a precomputed time attribute
    if hasattr(metadata, "time"):
        return float(metadata.time)

    # Some formats expose nanoseconds
    if hasattr(metadata, "nanoseconds"):
        return float(metadata.nanoseconds) / 1_000_000_000.0

    if hasattr(metadata, "tstmp"):
        return float(metadata.tstmp)

    # tuple-like fallback (sec, usec)
    if isinstance(metadata, (tuple, list)) and len(metadata) >= 2:
        return float(metadata[0]) + float(metadata[1]) / 1_000_000.0

    raise AttributeError("Unsupported pcap metadata timestamp format")


def _parse_packet(raw_packet: bytes) -> Ether:
    return Ether(raw_packet)


def _packet_summary(packet: Ether, timestamp: float) -> PacketSummary:
    length = len(packet.original) if hasattr(packet, "original") else len(bytes(packet))
    src_ip = None
    dst_ip = None
    src_port = None
    dst_port = None
    protocol = "OTHER"
    tcp_flags = {"SYN": False, "ACK": False, "RST": False, "FIN": False}

    ip_layer = None
    if IP in packet:
        ip_layer = packet[IP]
    elif IPv6 in packet:
        ip_layer = packet[IPv6]

    if ip_layer is not None:
        src_ip = getattr(ip_layer, "src", None)
        dst_ip = getattr(ip_layer, "dst", None)

        if TCP in packet:
            protocol = "TCP"
            tcp_layer = packet[TCP]
            flags = int(getattr(tcp_layer, "flags", 0))
            tcp_flags = {
                "SYN": bool(flags & 0x02),
                "ACK": bool(flags & 0x10),
                "RST": bool(flags & 0x04),
                "FIN": bool(flags & 0x01),
            }
            src_port = getattr(tcp_layer, "sport", None)
            dst_port = getattr(tcp_layer, "dport", None)
        elif UDP in packet:
            protocol = "UDP"
            udp_layer = packet[UDP]
            src_port = getattr(udp_layer, "sport", None)
            dst_port = getattr(udp_layer, "dport", None)
        elif ICMP in packet:
            protocol = "ICMP"

    return PacketSummary(
        timestamp=timestamp,
        length=length,
        protocol=protocol,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        tcp_flags=tcp_flags,
    )


def _finalize_window_rows(
    windows: Iterable[WindowStats],
    feature_extractor: FeatureExtractor,
    source: str,
) -> List[dict]:
    rows: List[dict] = []
    for window in windows:
        row = feature_extractor.build_row(window)
        row["source"] = source
        rows.append(row)
    return rows


def process_pcap(
    path: Path,
    config: Config,
    feature_extractor: FeatureExtractor,
    on_rows: Callable[[List[dict]], None],
) -> ExtractionStats:
    logger = get_logger("extract")
    window_seconds = float(config.get("extract", "window_seconds", default=1.0))
    stride_seconds = float(config.get("extract", "stride_seconds", default=0.5))
    max_packets = int(config.get("extract", "max_packets_per_file", default=0))
    batch_rows = int(config.get("extract", "batch_rows", default=20000))
    progress_every = max(10000, batch_rows)

    window_manager = SlidingWindowManager(window_seconds=window_seconds, stride_seconds=stride_seconds)

    rows_buffer: List[dict] = []
    packet_counter = 0
    window_counter = 0

    reader = RawPcapReader(str(path))
    for packet_counter, (raw_packet, metadata) in enumerate(reader, start=1):
        if max_packets and packet_counter > max_packets:
            break
        timestamp = _packet_timestamp(metadata)
        ether = _parse_packet(raw_packet)
        summary = _packet_summary(ether, timestamp)

        completed = list(window_manager.add_packet(summary))
        window_counter += len(completed)
        rows_buffer.extend(_finalize_window_rows(completed, feature_extractor, path.name))

        if len(rows_buffer) >= batch_rows:
            on_rows(rows_buffer)
            rows_buffer = []

        if packet_counter % progress_every == 0:
            logger.info(
                "extract_progress",
                file=str(path),
                packets=packet_counter,
                windows=window_counter,
            )

    reader.close()

    remaining = list(window_manager.finalize())
    window_counter += len(remaining)
    rows_buffer.extend(_finalize_window_rows(remaining, feature_extractor, path.name))

    if rows_buffer:
        on_rows(rows_buffer)

    logger.info(
        "pcap_processed",
        file=str(path),
        packets=packet_counter,
        windows=window_counter,
    )

    return ExtractionStats(packets=packet_counter, windows=window_counter, files=1)


def extract_pcaps(
    paths: Sequence[Path],
    config: Config,
    output_path: Path,
) -> ExtractionStats:
    include = config.get("features", "include", default=[])
    ratios = bool(config.get("features", "ratios", default=True))
    feature_extractor = FeatureExtractor(include=include, ratios=ratios)

    total_packets = 0
    total_windows = 0

    if output_path.exists():
        output_path.unlink()

    with ParquetBatchWriter(output_path) as writer:
        for path in tqdm(paths, desc="Extracting", unit="file"):
            def on_rows(rows: List[dict]) -> None:
                writer.write(rows)

            stats = process_pcap(path, config, feature_extractor, on_rows)
            total_packets += stats.packets
            total_windows += stats.windows

    return ExtractionStats(packets=total_packets, windows=total_windows, files=len(paths))


def extract_single_pcap_to_rows(
    path: Path,
    config: Config,
    feature_extractor: FeatureExtractor,
) -> List[dict]:
    rows: List[dict] = []

    def on_rows(batch: List[dict]) -> None:
        rows.extend(batch)

    process_pcap(path, config, feature_extractor, on_rows)
    return rows
