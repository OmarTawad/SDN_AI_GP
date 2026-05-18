"""End-to-end command tests for the Dynamic MoE demo runner."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from dynamic_moe.policy import PolicyConfig, SdnPolicyEngine


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generated_demo_pcap_processes_end_to_end_from_clean_runtime(tmp_path: Path) -> None:
    pcap = tmp_path / "demo_traffic.pcap"
    runtime_dir = tmp_path / "runtime"
    output_json = runtime_dir / "e2e_summary.json"

    generate = subprocess.run(
        [sys.executable, "tools/generate_demo_pcap.py", "--output", str(pcap)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert generate.returncode == 0, generate.stderr
    assert pcap.exists()
    assert "wrote" in generate.stdout

    command = subprocess.run(
        [
            sys.executable,
            "-m",
            "dynamic_moe.e2e_demo",
            "--pcap",
            str(pcap),
            "--runtime-dir",
            str(runtime_dir),
            "--mode",
            "full",
            "--no-install-flows",
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert command.returncode == 0, command.stderr + command.stdout
    assert output_json.exists()

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["packets_read"] > 0
    assert summary["windows_generated"] > 0
    assert summary["preprocessing"]["window_size_seconds"] == 1.0
    assert summary["preprocessing"]["hop_size_seconds"] == 0.5
    assert summary["preprocessing"]["dos_micro_bins"] == 8
    assert summary["preprocessing"]["arp_micro_bins"] == 8
    assert set(summary["final_prediction"]["probabilities"]) == {"normal", "dos", "arp"}
    assert isinstance(summary["final_prediction"]["confidence"], float)
    assert summary["final_prediction"]["selected_action"] in {
        "monitor",
        "alert",
        "rate_limit",
        "isolate",
        "quarantine",
        "drop",
    }
    assert math.isclose(
        sum(summary["final_prediction"]["expert_weights"].values()),
        1.0,
        rel_tol=1e-5,
    )
    assert math.isclose(float(summary["final_prediction"]["gate_weight_sum"]), 1.0, rel_tol=1e-5)

    first_window = summary["windows"][0]
    assert first_window["feature_tensor_shapes"]["gating_input"] == [2257]
    assert set(first_window["expert_output_shapes"]) == {
        "autoencoder",
        "dos_cnn",
        "dos_lstm",
        "arp_cnn",
        "arp_lstm",
    }
    assert summary["openflow"]["attempted"] is False
    assert summary["openflow"]["succeeded"] is False

    mitigation_path = Path(summary["logs"]["mitigations"])
    assert mitigation_path.exists()
    with mitigation_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {"timestamp", "class", "confidence", "src_mac", "dst_mac", "action", "expiry"} <= set(rows[0])


def test_policy_low_confidence_attack_does_not_install_or_drop() -> None:
    decision = SdnPolicyEngine(PolicyConfig()).decide(
        {"label": "dos", "is_attack": True, "confidence": 0.40}
    )
    assert decision.action == "monitor"
    assert decision.install_flow is False


def test_policy_high_confidence_maps_to_flow_install_action() -> None:
    decision = SdnPolicyEngine(
        PolicyConfig(mitigation_mode="drop", allow_automatic_blocking=True)
    ).decide({"label": "dos", "is_attack": True, "confidence": 0.99})
    assert decision.action == "drop"
    assert decision.install_flow is True
    assert decision.expiry
