from __future__ import annotations
from typing import Dict
from scapy.layers.l2 import Ether, ARP

def _safe_str(value) -> str | None:
    if value is None:
        return None
    try:
        text = str(value)
    except Exception:
        return None
    text = text.strip()
    return text or None

def scapy_pkt_to_row(pkt) -> Dict:
    """
    Normalize a Scapy packet into a flat dict row matching the ARP feature spec.
    """
    length = 0
    try:
        length = int(len(pkt))
    except Exception:
        length = 0

    row: Dict = {
        "ts": float(pkt.time),
        "len": length,
        "src_mac": None,
        "dst_mac": None,
        "eth_type": None,
        "is_arp": 0,
        "arp_opcode": 0,
        "arp_sender_ip": None,
        "arp_sender_mac": None,
        "arp_target_ip": None,
        "arp_target_mac": None,
        "arp_is_gratuitous": 0,
    }

    # L2
    eth = pkt.getlayer(Ether)
    if eth is not None:
        row["src_mac"] = _safe_str(getattr(eth, "src", None))
        row["dst_mac"] = _safe_str(getattr(eth, "dst", None))
        try:
            row["eth_type"] = int(getattr(eth, "type", None) or 0)
        except Exception:
            row["eth_type"] = None

    arp = pkt.getlayer(ARP)
    if arp is not None:
        op = int(getattr(arp, "op", 0) or 0)
        sender_ip = _safe_str(getattr(arp, "psrc", None))
        target_ip = _safe_str(getattr(arp, "pdst", None))
        sender_mac = _safe_str(getattr(arp, "hwsrc", None))
        target_mac = _safe_str(getattr(arp, "hwdst", None))

        row.update({
            "is_arp": 1,
            "arp_opcode": op,
            "arp_sender_ip": sender_ip,
            "arp_sender_mac": sender_mac,
            "arp_target_ip": target_ip,
            "arp_target_mac": target_mac,
        })

        # Mark gratuitous ARP: reply where sender advertises itself or broadcast target MAC
        is_gratuitous = False
        if op == 2:
            if sender_ip and target_ip and sender_ip == target_ip:
                is_gratuitous = True
            if target_mac and target_mac.lower() in {"00:00:00:00:00:00", "ff:ff:ff:ff:ff:ff"}:
                is_gratuitous = True
        row["arp_is_gratuitous"] = 1 if is_gratuitous else 0

    return row
