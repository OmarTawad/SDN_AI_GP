"""Post-processing gates for false-positive control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..config.types import PostProcessingConfig


@dataclass
class WindowDecision:
    index: int
    score: float
    family: str
    is_attack: bool


class DecisionGate:
    """Apply gating logic on per-window scores."""

    def __init__(self, config: PostProcessingConfig) -> None:
        self.config = config

    def apply(self, windows: Sequence[Dict[str, object]]) -> tuple[List[WindowDecision], bool]:
        decisions: List[WindowDecision] = []
        consecutive = 0
        cooldown = 0
        attack_windows = 0
        plausible_attacks = 0
        for window in windows:
            score = float(window["score"])
            family = str(window.get("family", "normal")).lower()
            features = window.get("features", {})
            passed = score >= self.config.tau_window
            if cooldown > 0:
                cooldown -= 1
                passed = False
            if passed:
                consecutive += 1
            else:
                consecutive = 0
            is_attack = False
            if consecutive >= self.config.consecutive_required:
                if self._plausibility(family, features):
                    is_attack = True
                    attack_windows += 1
                    plausible_attacks += 1
                    consecutive = 0
                    cooldown = self.config.cooldown_windows
            decisions.append(
                WindowDecision(
                    index=int(window["index"]),
                    score=score,
                    family=family,
                    is_attack=is_attack,
                )
            )
        file_attack = attack_windows >= self.config.min_attack_windows and plausible_attacks > 0
        file_attack = file_attack or any(dec.is_attack for dec in decisions)
        file_attack = file_attack and max((dec.score for dec in decisions), default=0.0) >= self.config.tau_file
        return decisions, file_attack

    def _plausibility(self, family: str, features: object) -> bool:
        if not isinstance(features, dict):
            return False
        if family == "ssdp":
            share = float(features.get("ssdp_share", 0.0))
            udp_rate = float(features.get("udp_1900_rate", 0.0))
            return share >= self.config.plausibility.ssdp_udp_1900_min_fraction or udp_rate > 0
        if family == "syn":
            syn_rate = float(features.get("tcp_syn_rate", 0.0))
            return syn_rate >= self.config.plausibility.syn_min_syn_rate
        if family == "icmp":
            icmp_rate = float(features.get("icmp_echo_rate", 0.0))
            return icmp_rate >= self.config.plausibility.icmp_min_rate
        if family == "udp":
            udp_rate = float(features.get("packet_rate", 0.0))
            return udp_rate >= self.config.plausibility.udp_min_rate
        if family == "http":
            byte_rate = float(features.get("byte_rate", 0.0))
            return byte_rate >= self.config.plausibility.http_min_rate
        return True


__all__ = ["DecisionGate", "WindowDecision"]
