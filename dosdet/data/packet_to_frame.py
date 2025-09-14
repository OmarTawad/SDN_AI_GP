from __future__ import annotations
from typing import Dict, Optional, Tuple
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, UDP, TCP, ICMP
from scapy.layers.inet6 import IPv6
from scapy.packet import Raw
import data.pcap_reader as pcap_reader  # for constants

def _tcp_flags(pkt: TCP) -> Tuple[int, int]:
    syn = 1 if pkt.flags & 0x02 else 0
    ack = 1 if pkt.flags & 0x10 else 0
    synack = 1 if syn and ack else 0
    return syn, synack

def scapy_pkt_to_row(pkt, ssdp_multicast_v4: str, ssdp_multicast_v6: str) -> Dict:
    """
    Normalize a Scapy packet into a flat dict row matching feature spec.
    """
    row: Dict = {
        "ts": float(pkt.time),
        "src_mac": None, "dst_mac": None,
        "src_ip": None, "dst_ip": None,
        "ip_proto": None, "ttl": None, "len": None,
        "src_port": None, "dst_port": None,
        "tcp_syn": 0, "tcp_synack": 0,
        "is_udp": 0, "is_tcp": 0, "is_icmp": 0,
        "udp_len": None,
        "ssdp_method": "NONE",
        "ssdp_st": None, "ssdp_man": None, "ssdp_user_agent": None,
        "is_ssdp_candidate": 0,
    }

    # L2
    eth = pkt.getlayer(Ether)
    if eth is not None:
        row["src_mac"] = str(eth.src) if getattr(eth, "src", None) else None
        row["dst_mac"] = str(eth.dst) if getattr(eth, "dst", None) else None

    ip4 = pkt.getlayer(IP)
    ip6 = pkt.getlayer(IPv6)
    if ip4 is not None:
        row["src_ip"] = str(ip4.src)
        row["dst_ip"] = str(ip4.dst)
        row["ttl"] = int(getattr(ip4, "ttl", 0))
        row["len"] = int(getattr(ip4, "len", 0))
    elif ip6 is not None:
        row["src_ip"] = str(ip6.src)
        row["dst_ip"] = str(ip6.dst)
        row["ttl"] = int(getattr(ip6, "hlim", 0))
        row["len"] = int(getattr(ip6, "plen", 0))

    if pkt.haslayer(TCP):
        tcp = pkt.getlayer(TCP)
        row["ip_proto"] = 6
        row["is_tcp"] = 1
        row["src_port"] = int(tcp.sport)
        row["dst_port"] = int(tcp.dport)
        syn, synack = _tcp_flags(tcp)
        row["tcp_syn"] = syn
        row["tcp_synack"] = synack

    elif pkt.haslayer(UDP):
        udp = pkt.getlayer(UDP)
        row["ip_proto"] = 17
        row["is_udp"] = 1
        row["src_port"] = int(udp.sport)
        row["dst_port"] = int(udp.dport)
        row["udp_len"] = int(getattr(udp, "len", 0))

        # SSDP candidate?
        dst_is_1900 = row["dst_port"] == 1900
        dst_is_multicast_v4 = (row["dst_ip"] == ssdp_multicast_v4)
        dst_is_multicast_v6 = (row["dst_ip"] == ssdp_multicast_v6)
        is_ssdp = dst_is_1900 or dst_is_multicast_v4 or dst_is_multicast_v6

        if is_ssdp:
            row["is_ssdp_candidate"] = 1
            raw = pkt.getlayer(Raw)
            payload = raw.load if raw is not None else b""
            method, tokens = pcap_reader.parse_ssdp(payload)  # via reader shim
            row["ssdp_method"] = method
            row["ssdp_st"] = tokens.get("ST")
            row["ssdp_man"] = tokens.get("MAN")
            row["ssdp_user_agent"] = tokens.get("USER-AGENT")

    elif pkt.haslayer(ICMP):
        ic = pkt.getlayer(ICMP)
        row["ip_proto"] = 1
        row["is_icmp"] = 1

    return row
