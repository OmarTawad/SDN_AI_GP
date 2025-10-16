from __future__ import annotations
from typing import Dict, Iterator
from scapy.utils import PcapReader
from .packet_to_frame import scapy_pkt_to_row

def iter_rows_from_pcap(
    pcap_path: str,
) -> Iterator[Dict]:
    """
    Stream-normalize packets from a pcap file into dict rows.
    """
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            try:
                row = scapy_pkt_to_row(pkt)
                yield row
            except Exception:
                # robust to weird frames; skip
                continue
