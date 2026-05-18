"""Generate a safe synthetic PCAP for the NetSentinel E2E demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import ARP, Ether
from scapy.packet import Packet
from scapy.utils import PcapWriter


def _stamp(pkt: Packet, timestamp: float) -> Packet:
    pkt.time = timestamp
    return pkt


def build_packets() -> list[Packet]:
    packets: list[Packet] = []
    ts = 1_700_000_000.0

    for idx in range(18):
        packets.append(
            _stamp(
                Ether(src="00:00:00:00:00:01", dst="00:00:00:00:00:02")
                / IP(src="10.0.0.1", dst="10.0.0.2")
                / TCP(sport=1024 + idx, dport=80, flags="PA")
                / (b"normal" * 8),
                ts + idx * 0.12,
            )
        )

    burst_start = ts + 2.5
    for idx in range(90):
        packets.append(
            _stamp(
                Ether(src="00:00:00:00:00:03", dst="00:00:00:00:00:04")
                / IP(src=f"10.0.1.{(idx % 40) + 1}", dst="10.0.0.10")
                / UDP(sport=10_000 + idx, dport=1900)
                / (b"dos-burst" * 24),
                burst_start + idx * 0.01,
            )
        )

    arp_start = ts + 4.5
    for idx in range(36):
        packets.append(
            _stamp(
                Ether(src="00:00:00:00:00:05", dst="ff:ff:ff:ff:ff:ff")
                / ARP(
                    op=2,
                    hwsrc="00:00:00:00:00:05",
                    psrc="10.0.0.1",
                    hwdst="00:00:00:00:00:06",
                    pdst=f"10.0.0.{(idx % 8) + 2}",
                ),
                arp_start + idx * 0.03,
            )
        )

    return sorted(packets, key=lambda pkt: float(pkt.time))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate NetSentinel synthetic demo traffic.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo/demo_traffic.pcap"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    packets = build_packets()
    with PcapWriter(str(args.output), sync=True) as writer:
        for pkt in packets:
            writer.write(pkt)
    print(f"wrote {len(packets)} packets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
