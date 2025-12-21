"""Packet conversion helpers for MoE streaming datasets.


"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether

from Neural_LSTM.src.dos_detector.data.structures import PacketRecord as DosPacketRecord, Window as DosWindow
from ARP_LSTM.src.arp_detector.data.structures import PacketRecord as ArpPacketRecord, Window as ArpWindow


def packet_to_auto(pkt, arp_row: Dict[str, object]) -> Tuple[float, int, Dict[str, object]]:
    """Convert a scapy packet into the autoencoder feature row."""

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


def build_dos_window(index: int, start: float, end: float, rows: Sequence[Dict[str, object]]) -> DosWindow:
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


def build_arp_window(index: int, start: float, end: float, rows: Sequence[Dict[str, object]]) -> ArpWindow:
    packets: List[ArpPacketRecord] = []
    for row in rows:
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
            info={},
            arp_opcode=int(row.get("arp_opcode") or 0),
            arp_sender_ip=row.get("arp_sender_ip"),
            arp_sender_mac=row.get("arp_sender_mac"),
            arp_target_ip=row.get("arp_target_ip"),
            arp_target_mac=row.get("arp_target_mac"),
            arp_is_gratuitous=bool(row.get("arp_is_gratuitous")),
        )
        packets.append(packet)
    return ArpWindow(index=index, start_time=float(start), end_time=float(end), packets=packets)


__all__ = [
    "build_arp_window",
    "build_dos_window",
    "packet_to_auto",
]

