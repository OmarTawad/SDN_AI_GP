#src/arp_detector/inference/postprocessing.py
"""Post-processing gates for false-positive control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..config.types import PostProcessingConfig


@dataclass
class WindowDecision:
    index: int
    score: float
    is_attack: bool


class DecisionGate:
    """Simple threshold-based gating for the ARP detector."""

    def __init__(self, config: PostProcessingConfig) -> None:
        self.config = config

    def apply(self, windows: Sequence[Dict[str, object]]) -> tuple[List[WindowDecision], bool]:
        decisions: List[WindowDecision] = []
        attack_windows = 0
        max_score = 0.0
        for window in windows:
            score = float(window["score"])
            is_attack = score >= self.config.tau_window
            if is_attack:
                attack_windows += 1
            max_score = max(max_score, score)
            decisions.append(
                WindowDecision(
                    index=int(window["index"]),
                    score=score,
                    is_attack=is_attack,
                )
            )
        file_attack = attack_windows >= self.config.min_attack_windows and max_score >= self.config.tau_file
        return decisions, file_attack


__all__ = ["DecisionGate", "WindowDecision"]
