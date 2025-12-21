"""Suspicious entity extraction for inference summaries.


"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from gateway.utils import get_logger

LOGGER = get_logger("inference.suspicion")


class PacketLike(Protocol):
    """Protocol describing the subset of scapy packet API used here."""

    time: float

    def getlayer(self, layer):  # type: ignore[override]
        ...


class PacketReader(Protocol):
    """Protocol for objects that iterate over packets."""

    def __iter__(self) -> Iterable[PacketLike]:  # pragma: no cover - interface
        ...

    def close(self) -> None:  # pragma: no cover - interface
        ...


@dataclass(frozen=True)
class SuspicionSummary:
    """Top suspicious IP and MAC entities reported by inference."""

    ip_addresses: list[str]
    mac_addresses: list[str]


def _load_default_reader(path: Path) -> PacketReader:
    """Return a scapy-backed reader for the supplied PCAP path."""

    try:
        from scapy.utils import PcapReader
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("Scapy is required for PCAP parsing during inference.") from exc
    return PcapReader(str(path))  # type: ignore[return-value]


def extract_suspicion(
    pcap_path: Path,
    confidences: dict[str, float],
    reader: PacketReader | None = None,
) -> SuspicionSummary:
    """Extract suspicious IP and MAC addresses from a PCAP file.

    Args:
        pcap_path: Location of the capture.
        confidences: Per-class confidence dictionary used to weight scores.
        reader: Optional custom reader implementation (useful for tests).

    Returns:
        SuspicionSummary: Ranked suspicious entities.
    """

    ip_counter: Counter[str] = Counter()
    mac_counter: Counter[str] = Counter()
    dos_weight = float(confidences.get("dos", 0.0))
    arp_weight = float(confidences.get("arp", 0.0))

    packet_reader = reader or _load_default_reader(pcap_path)
    try:
        from scapy.layers.inet import IP, TCP  # type: ignore[import]
        from scapy.layers.l2 import ARP  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - import guard
        LOGGER.warning("Scapy layers unavailable: %s", exc)
        return SuspicionSummary(ip_addresses=[], mac_addresses=[])

    mac_to_ips: dict[str, set[str]] = defaultdict(set)
    ip_to_macs: dict[str, set[str]] = defaultdict(set)
    syn_sources: Counter[str] = Counter()
    total_sources: Counter[str] = Counter()

    try:
        for packet in packet_reader:
            ip_layer = packet.getlayer(IP)
            tcp_layer = packet.getlayer(TCP)
            arp_layer = packet.getlayer(ARP)

            src_ip = getattr(ip_layer, "src", None) if ip_layer is not None else None
            if src_ip:
                src_ip_str = str(src_ip)
                total_sources[src_ip_str] += 1
            else:
                src_ip_str = None

            if tcp_layer is not None and src_ip_str:
                flags = int(getattr(tcp_layer, "flags", 0))
                if flags & 0x02 and not (flags & 0x10):  # SYN without ACK
                    syn_sources[src_ip_str] += 1

            if arp_layer is not None:
                sender_ip = str(getattr(arp_layer, "psrc", "") or "")
                sender_mac = str(getattr(arp_layer, "hwsrc", "") or "").lower()
                target_ip = str(getattr(arp_layer, "pdst", "") or "")
                target_mac = str(getattr(arp_layer, "hwdst", "") or "").lower()

                if sender_ip and sender_mac:
                    mac_to_ips[sender_mac].add(sender_ip)
                    ip_to_macs[sender_ip].add(sender_mac)
                if target_ip and target_mac and target_mac not in {
                    "ff:ff:ff:ff:ff:ff",
                    "00:00:00:00:00:00",
                }:
                    mac_to_ips[target_mac].add(target_ip)
                    ip_to_macs[target_ip].add(target_mac)
    finally:
        try:
            packet_reader.close()
        except Exception:  # pragma: no cover - best effort
            pass

    dos_factor = dos_weight if dos_weight > 0 else (0.1 if syn_sources else 0.0)
    for address, count in syn_sources.items():
        ip_counter[address] += count * max(dos_factor, 1.0)
    for address, count in total_sources.items():
        ip_counter[address] += 0.1 * count * dos_weight

    arp_factor = arp_weight if arp_weight > 0 else (0.1 if mac_to_ips or ip_to_macs else 0.0)
    for mac, ips in mac_to_ips.items():
        if mac and len(ips) > 1:
            mac_counter[mac] += float(len(ips)) * max(arp_factor, 1.0)
            for ip in ips:
                if ip:
                    ip_counter[ip] += max(arp_factor, 1.0)
    for ip, macs in ip_to_macs.items():
        if ip and len(macs) > 1:
            ip_counter[ip] += float(len(macs)) * max(arp_factor, 1.0)
            for mac in macs:
                if mac:
                    mac_counter[mac] += 0.5 * max(arp_factor, 1.0)

    for placeholder in ("", None, "ff:ff:ff:ff:ff:ff", "00:00:00:00:00:00"):
        ip_counter.pop(placeholder, None)
        mac_counter.pop(placeholder, None)

    return SuspicionSummary(
        ip_addresses=top_entries(ip_counter),
        mac_addresses=top_entries(mac_counter),
    )


def top_entries(counter: Counter[str], limit: int = 3) -> list[str]:
    """Return the highest-frequency keys in a counter.

    Args:
        counter: Counter containing entity frequencies.
        limit: Maximum number of entries to return.

    Returns:
        list[str]: Most frequent keys sorted by frequency and key name.
    """

    sorted_items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [item[0] for item in sorted_items[:limit]]


__all__ = ["PacketReader", "SuspicionSummary", "extract_suspicion"]
