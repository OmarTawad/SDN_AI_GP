from __future__ import annotations
from typing import Dict, Iterator
from scapy.utils import PcapReader
from .packet_to_frame import scapy_pkt_to_row
from features.ssdp_parser import parse_ssdp_payload

# expose parser to packet_to_frame via this module to avoid circular import issues
parse_ssdp = parse_ssdp_payload

def iter_rows_from_pcap(
    pcap_path: str,
    ssdp_multicast_v4: str = "239.255.255.250",
    ssdp_multicast_v6: str = "ff02::c",
    byte_limit: int | None = None,
) -> Iterator[Dict]:
    """
    Stream-normalize packets from a pcap file into dict rows.
    """
    accumulated_bytes = 0
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if byte_limit is not None:
                accumulated_bytes += len(pkt)
                if accumulated_bytes > byte_limit:
                    break
            try:
                row = scapy_pkt_to_row(pkt, ssdp_multicast_v4, ssdp_multicast_v6)
                yield row
            except Exception:
                # robust to weird frames; skip
                continue
