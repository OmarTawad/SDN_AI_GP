"""Tests for host activity aggregation during inference."""

from dos_detector.inference.pipeline import InferencePipeline
from dos_detector.inference.postprocessing import WindowDecision


def test_compute_host_activity_ranks_attack_sources() -> None:
    pipeline = InferencePipeline.__new__(InferencePipeline)

    host_maps = {
        "macs": {
            0: {"aa:aa:aa:aa:aa:aa": 3, "bb:bb:bb:bb:bb:bb": 1},
            1: {"aa:aa:aa:aa:aa:aa": 2, "cc:cc:cc:cc:cc:cc": 2},
        },
        "ips": {
            0: {"10.0.0.1": 3, "10.0.0.2": 1},
            1: {"10.0.0.1": 2, "10.0.0.3": 2},
        },
    }
    window_results = [
        {"index": 0, "fused_score": 0.9},
        {"index": 1, "fused_score": 0.5},
    ]
    decisions = [
        WindowDecision(index=0, score=0.9, family="udp", is_attack=True),
        WindowDecision(index=1, score=0.5, family="udp", is_attack=False),
    ]

    activity = pipeline._compute_host_activity(host_maps, window_results, decisions)

    assert activity["macs"][0]["mac"] == "aa:aa:aa:aa:aa:aa"
    assert activity["macs"][0]["attack_packets"] == 3
    assert activity["macs"][0]["attack_windows"] == 1
    assert [entry["mac"] for entry in activity["macs"]] == [
        "aa:aa:aa:aa:aa:aa",
        "bb:bb:bb:bb:bb:bb",
        "cc:cc:cc:cc:cc:cc",
    ]

    assert activity["ips"][0]["ip"] == "10.0.0.1"
    assert activity["ips"][0]["attack_packets"] == 3
