"""Smoke-level tests for the Dynamic MoE SDN integration."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

from dynamic_moe.config import RuntimeConfig
from dynamic_moe.feature_extractor import StreamingFeatureExtractor
from dynamic_moe.policy import PolicyConfig, SdnPolicyEngine
from dynamic_moe.runtime import RuntimeLogger
from gateway.dynamic_moe_adapter import DynamicMoEGateway
from gateway.core import CLASS_LABELS
from gateway.data.datasets.gating import UNIFIED_GATING_COMPONENT_KEYS, build_unified_gating
from gateway.data.extractors import features as feature_meta
from gateway.models.unified_moe import UNIFIED_GATING_INPUT_DIM, UnifiedGating, UnifiedMoE
from gateway.moe_model import (
    ARP_CNN_SEQ_IN_DIM,
    ARP_CNN_STATIC_DIM,
    ARP_LSTM_INPUT_DIM,
    AUTO_FEATURE_DIM,
    DOS_CNN_SEQ_IN_DIM,
    DOS_CNN_STATIC_DIM,
    DOS_LSTM_INPUT_DIM,
)


class _ScalarExpert(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.register_buffer("value", torch.tensor([value], dtype=torch.float32))

    def forward(self, features):  # type: ignore[no-untyped-def]
        batch_size = features["gating_input"].shape[0]
        return self.value.expand(batch_size)


def _feature_dict() -> dict[str, torch.Tensor]:
    return {
        "auto": torch.zeros(AUTO_FEATURE_DIM),
        "dos_cnn_static": torch.zeros(DOS_CNN_STATIC_DIM),
        "dos_cnn_seq": torch.zeros(feature_meta.DOS_MICRO_BINS, DOS_CNN_SEQ_IN_DIM),
        "dos_lstm_seq": torch.zeros(feature_meta.DOS_LSTM_SEQUENCE_LENGTH, DOS_LSTM_INPUT_DIM),
        "arp_cnn_static": torch.zeros(ARP_CNN_STATIC_DIM),
        "arp_cnn_seq": torch.zeros(feature_meta.ARP_MICRO_BINS, ARP_CNN_SEQ_IN_DIM),
        "arp_lstm_seq": torch.zeros(feature_meta.ARP_LSTM_SEQUENCE_LENGTH, ARP_LSTM_INPUT_DIM),
    }


def test_preprocessing_contract_and_deterministic_gating_order() -> None:
    assert feature_meta.WINDOW_SIZE == 1.0
    assert feature_meta.WINDOW_STRIDE == 0.5
    assert feature_meta.DOS_MICRO_BINS == 8
    assert feature_meta.ARP_MICRO_BINS == 8
    assert UNIFIED_GATING_COMPONENT_KEYS == (
        "auto",
        "dos_cnn_static",
        "dos_cnn_seq",
        "dos_lstm_seq",
        "arp_cnn_static",
        "arp_cnn_seq",
        "arp_lstm_seq",
    )

    feature_dict = _feature_dict()
    gating = build_unified_gating(feature_dict)
    assert gating.shape == (UNIFIED_GATING_INPUT_DIM,)


def test_streaming_extractor_uses_required_windowing() -> None:
    extractor = StreamingFeatureExtractor()
    assert extractor.manager.window_size == 1.0
    assert extractor.manager.stride == 0.5
    assert extractor.manager.micro_bins == {"dos": 8, "arp": 8}


def test_dense_gate_and_final_classifier_shapes() -> None:
    feature_dict = _feature_dict()
    feature_dict["gating_input"] = build_unified_gating(feature_dict)
    batch = {key: value.unsqueeze(0).float() for key, value in feature_dict.items()}

    experts = [_ScalarExpert(float(idx + 1)) for idx in range(5)]
    gate = UnifiedGating(UNIFIED_GATING_INPUT_DIM, hidden_dim=16, num_experts=len(experts))
    model = UnifiedMoE(experts=experts, gating=gate, num_classes=len(CLASS_LABELS))

    logits, attention = model(batch, return_attention=True)
    assert logits.shape == (1, 3)
    assert attention["weights"].shape == (1, 5)
    assert attention["expert_outputs"].shape == (1, 5)
    assert math.isclose(float(attention["weights"].sum(dim=1).item()), 1.0, rel_tol=1e-6)


def test_gateway_prediction_contract_with_existing_checkpoint() -> None:
    feature_dict = _feature_dict()
    feature_dict["gating_input"] = build_unified_gating(feature_dict)
    result = DynamicMoEGateway().predict(feature_dict)

    assert result["label"] in CLASS_LABELS
    assert isinstance(result["confidence"], float)
    assert set(result["probabilities"]) == set(CLASS_LABELS)
    assert set(result["expert_weights"]) == {
        "autoencoder",
        "dos_cnn",
        "dos_lstm",
        "arp_cnn",
        "arp_lstm",
    }
    assert math.isclose(sum(result["expert_weights"].values()), 1.0, rel_tol=1e-5)


def test_sdn_policy_mapping_confidence_tiers() -> None:
    conservative = SdnPolicyEngine(PolicyConfig())
    blocking = SdnPolicyEngine(
        PolicyConfig(mitigation_mode="drop", allow_automatic_blocking=True)
    )
    isolating = SdnPolicyEngine(
        PolicyConfig(mitigation_mode="isolate", allow_automatic_blocking=True)
    )

    assert conservative.decide({"label": "normal", "score": 0.99}).action == "monitor"
    assert conservative.decide({"label": "dos", "is_attack": True, "score": 0.40}).action == "monitor"
    assert conservative.decide({"label": "dos", "is_attack": True, "score": 0.92}).action == "rate_limit"

    high_dos_safe = conservative.decide({"label": "dos", "is_attack": True, "score": 0.99})
    assert high_dos_safe.action == "alert"
    assert high_dos_safe.install_flow is False

    high_dos_block = blocking.decide({"label": "dos", "is_attack": True, "score": 0.99})
    assert high_dos_block.action == "drop"
    assert high_dos_block.install_flow is True
    assert high_dos_block.expiry

    high_arp = isolating.decide({"label": "arp", "is_attack": True, "score": 0.96})
    assert high_arp.action == "isolate"
    assert high_arp.install_flow is True


def test_mitigation_audit_log_creation(tmp_path: Path) -> None:
    config = RuntimeConfig(
        config_path=tmp_path / "config.yaml",
        runtime_dir=tmp_path,
        alerts_path=tmp_path / "alerts.jsonl",
        flows_path=tmp_path / "flows.csv",
        packets_meta_path=tmp_path / "packets.csv",
        decisions_log_path=tmp_path / "decisions.log",
        mitigations_path=tmp_path / "mitigations.csv",
        attack_pcap_path=None,
        controller_ip="127.0.0.1",
        controller_port=6633,
        default_replay_host="h1",
        mitigation="alert",
        min_alert_confidence=0.70,
        min_rate_limit_confidence=0.90,
        min_block_confidence=0.98,
        arp_isolate_confidence=0.95,
        allow_automatic_blocking=False,
        action_expiry_seconds=300,
    )
    logger = RuntimeLogger(config)
    logger.log_mitigation(
        packet_meta={
            "switch": 1,
            "src_mac": "00:00:00:00:00:01",
            "dst_mac": "00:00:00:00:00:02",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
        },
        inference={"label": "dos", "attack_type": "dos", "score": 0.92},
        action="rate_limit",
        expiry="2026-05-18T00:00:00+00:00",
        reason="medium_confidence_dos",
    )
    logger.close()

    with config.mitigations_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["class"] == "dos"
    assert rows[0]["confidence"] == "0.9200"
    assert rows[0]["src_mac"] == "00:00:00:00:00:01"
    assert rows[0]["action"] == "rate_limit"
    assert rows[0]["expiry"] == "2026-05-18T00:00:00+00:00"
