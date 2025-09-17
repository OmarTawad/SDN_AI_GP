"""PCAP reading utilities.""" 
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional
import shutil, subprocess

from scapy.all import ICMP, IP, IPv6, RawPcapReader, TCP, UDP, Ether
from ..utils.progress import progress
from .structures import PacketRecord

@dataclass
class PCAPMetadata:
    path: Path
    packet_count: int
    duration: float
    start_time: float
    end_time: float

def _timestamp_from_meta(meta) -> float:
    # legacy PCAP
    if hasattr(meta, "sec"):
        usec = getattr(meta, "usec", 0)
        return float(meta.sec) + float(usec) / 1_000_000.0
    # PCAPNG
    if hasattr(meta, "tshigh") and hasattr(meta, "tslow"):
        ticks = (int(meta.tshigh) << 32) | int(meta.tslow)
        res = getattr(meta, "tsresol", 1_000_000)
        if isinstance(res, int) and res >= 256:
            denom = float(res)
        elif isinstance(res, int) and res >= 0x80:
            denom = float(2 ** (res & 0x7F))
        elif isinstance(res, int):
            denom = float(10 ** res) if res < 12 else 1_000_000.0
        else:
            denom = 1_000_000.0
        return float(ticks) / denom
    return 0.0

def _decode_packet(data: bytes, timestamp: float) -> Optional[PacketRecord]:
    packet = Ether(data)
    src_mac = getattr(packet, "src", None)
    dst_mac = getattr(packet, "dst", None)

    ip_layer = packet.getlayer(IP) or packet.getlayer(IPv6)
    src_ip = dst_ip = None
    ttl = None
    protocol = "other"
    src_port = dst_port = None
    tcp_flags = None
    payload_len = 0
    info: dict[str, Optional[str]] = {}

    if ip_layer is not None:
        src_ip = getattr(ip_layer, "src", None)
        dst_ip = getattr(ip_layer, "dst", None)
        ttl = getattr(ip_layer, "ttl", getattr(ip_layer, "hlim", None))

        if ip_layer.haslayer(TCP):
            tcp = ip_layer.getlayer(TCP)
            src_port = int(tcp.sport); dst_port = int(tcp.dport)
            tcp_flags = int(tcp.flags); protocol = "tcp"
            payload_len = len(bytes(tcp.payload))
        elif ip_layer.haslayer(UDP):
            udp = ip_layer.getlayer(UDP)
            src_port = int(udp.sport); dst_port = int(udp.dport)
            protocol = "udp"
            payload = bytes(udp.payload)
            payload_len = len(payload)
            if payload_len:
                try:
                    text = payload.decode(errors="ignore")
                    if "M-SEARCH" in text: info["ssdp_method"] = "M-SEARCH"
                    elif "NOTIFY" in text: info["ssdp_method"] = "NOTIFY"
                except Exception:
                    pass
        elif ip_layer.haslayer(ICMP):
            icmp = ip_layer.getlayer(ICMP)
            protocol = "icmp"
            info["icmp_type"] = str(getattr(icmp, "type", None))
            payload_len = len(bytes(icmp.payload))
        else:
            payload_len = len(bytes(ip_layer.payload))
            protocol = ip_layer.name.lower()
    else:
        payload_len = len(bytes(packet.payload))

    return PacketRecord(
        timestamp=timestamp,
        src_mac=src_mac, dst_mac=dst_mac,
        src_ip=src_ip, dst_ip=dst_ip,
        src_port=src_port, dst_port=dst_port,
        protocol=protocol, length=len(packet),
        ttl=ttl, tcp_flags=tcp_flags,
        payload_len=payload_len, info=info,
    )

def _capinfos_count(path: Path) -> Optional[int]:
    exe = shutil.which("capinfos")
    if not exe: return None
    try:
        out = subprocess.check_output([exe, "-c", str(path)], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None
    for line in out.splitlines():
        if "Number of packets" in line:
            digits = "".join(ch for ch in line if ch.isdigit())
            return int(digits) if digits else None
    return None

def read_pcap(path: Path, limit: Optional[int] = None) -> List[PacketRecord]:
    packets: List[PacketRecord] = []
    reader = RawPcapReader(str(path))
    try:
        total = _capinfos_count(path)
        it = enumerate(reader)
        if total:
            it = progress(it, total=total, desc=f"Reading {path.name}", unit="pkt", leave=False)
        for index, (data, meta) in it:
            if limit is not None and index >= limit:
                break
            ts = _timestamp_from_meta(meta)
            rec = _decode_packet(data, ts)
            if rec is not None:
                packets.append(rec)
    finally:
        reader.close()
    return packets

def iter_pcap(path: Path) -> Iterator[PacketRecord]:
    reader = RawPcapReader(str(path))
    try:
        for data, meta in reader:
            ts = _timestamp_from_meta(meta)
            rec = _decode_packet(data, ts)
            if rec is not None:
                yield rec
    finally:
        reader.close()

def summarize_packets(packets: List[PacketRecord], path: Path) -> PCAPMetadata:
    if not packets:
        return PCAPMetadata(path=path, packet_count=0, duration=0.0, start_time=0.0, end_time=0.0)
    start = packets[0].timestamp
    end = packets[-1].timestamp
    return PCAPMetadata(path=path, packet_count=len(packets), duration=end-start, start_time=start, end_time=end)
