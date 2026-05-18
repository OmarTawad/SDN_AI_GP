"""Unit tests for MoE inference reporting helpers."""

from __future__ import annotations

from gateway.inference.reporting import binary_label_for_prediction, format_window_log


def test_binary_label_for_normal_prediction() -> None:
    """Normal predictions should remain normal in the compact log format."""

    assert binary_label_for_prediction("normal") == ("normal", None)


def test_binary_label_for_attack_prediction() -> None:
    """Attack classes should collapse to attack and preserve the subtype."""

    assert binary_label_for_prediction("dos") == ("attack", "dos")


def test_format_window_log_includes_binary_and_attack_labels() -> None:
    """Per-window logs should match the small-infer style summary."""

    line = format_window_log(
        pcap_name="mixed1.pcap",
        window_index=3,
        total_windows=60,
        prediction="arp",
        probabilities=[0.12, 0.08, 0.80],
    )

    assert "[mixed1.pcap]" in line
    assert "window=3/60" in line
    assert "label=attack" in line
    assert "attack_type=arp" in line
    assert "P(normal)=0.12" in line
    assert "P(dos)=0.08" in line
    assert "P(arp)=0.80" in line
