from __future__ import annotations

from dae.features import FeatureExtractor
from dae.window import PacketSummary, WindowStats


def test_entropy_and_unique_counts():
    window = WindowStats(start=0.0, end=1.0, index=0, duration=1.0)
    packets = [
        PacketSummary(0.1, 100, "TCP", "10.0.0.1", "10.0.0.2", 1234, 80, {"SYN": True, "ACK": False, "RST": False, "FIN": False}),
        PacketSummary(0.2, 100, "TCP", "10.0.0.1", "10.0.0.2", 1234, 80, {"SYN": False, "ACK": True, "RST": False, "FIN": False}),
        PacketSummary(0.3, 100, "UDP", "10.0.0.3", "10.0.0.4", 5000, 53, {}),
        PacketSummary(0.4, 100, "UDP", "10.0.0.3", "10.0.0.5", 5001, 53, {}),
    ]

    for pkt in packets:
        window.add_packet(pkt)

    extractor = FeatureExtractor(include=["src_ip_entropy", "unique_src_ips", "pkt_count"])
    row = extractor.build_row(window)

    assert row["pkt_count"] == float(len(packets))
    assert row["unique_src_ips"] == 2.0
    # Entropy should be >0 because there are two distinct source IPs
    assert row["src_ip_entropy"] > 0.0
