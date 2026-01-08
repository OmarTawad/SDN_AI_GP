from __future__ import annotations

from dae.window import PacketSummary, SlidingWindowManager


def make_packet(ts: float, length: int = 100) -> PacketSummary:
    return PacketSummary(
        timestamp=ts,
        length=length,
        protocol="TCP",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_port=1234,
        dst_port=80,
        tcp_flags={"SYN": True, "ACK": False, "RST": False, "FIN": False},
    )


def test_sliding_window_alignment_and_counts():
    manager = SlidingWindowManager(window_seconds=1.0, stride_seconds=0.5)
    packets = [make_packet(ts) for ts in (0.1, 0.6, 1.1, 1.6, 2.1)]

    completed = []
    for pkt in packets:
        completed.extend(manager.add_packet(pkt))
    completed.extend(manager.finalize())

    # We expect 5 windows given 2 second coverage with 0.5 stride
    assert len(completed) >= 4

    first_window = completed[0]
    assert first_window.start <= 0.1 < first_window.end
    assert first_window.packet_count >= 1

    # Ensure each packet contributed to at least one window
    total_packets = sum(window.packet_count for window in completed)
    assert total_packets >= len(packets)
