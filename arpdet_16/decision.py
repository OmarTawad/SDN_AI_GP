# decision.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class WindowObs:
    prob: float
    snaps: Dict[str, float]
    t0: float
    t1: float


@dataclass
class DecisionConfig:
    tau_high: float = 0.70
    tau_low: float = 0.55
    min_attack_windows: int = 2
    consecutive_required: int = 1
    cooldown_windows: int = 1
    enable_gate: bool = True
    min_arp_replies: float = 2.0
    min_conflict_macs: int = 1


@dataclass
class FileDecision:
    file: str
    decision: str                   # "attack" | "normal"
    first_attack_timestamp: float | None
    num_attack_windows: int
    max_prob: float
    gate_reasons: List[str]


def _gate_arp(snaps: Dict[str, float], cfg: DecisionConfig) -> Tuple[bool, List[str]]:
    replies = float(snaps.get("arp_replies", 0.0))
    conflicting_macs = int(float(snaps.get("reply_conflict_macs", 0.0)))

    reasons: List[str] = []
    if replies >= cfg.min_arp_replies:
        reasons.append("arp_replies")
    if conflicting_macs >= cfg.min_conflict_macs:
        reasons.append("mac_conflict")

    gate_ok = ("arp_replies" in reasons) and ("mac_conflict" in reasons)
    return gate_ok, reasons


def decide_file(
    file_path: str,
    windows: Iterable[WindowObs],
    cfg: DecisionConfig = DecisionConfig(),
) -> FileDecision:
    win_list = list(windows)
    if not win_list:
        return FileDecision(
            file=file_path,
            decision="normal",
            first_attack_timestamp=None,
            num_attack_windows=0,
            max_prob=0.0,
            gate_reasons=[],
        )

    state = "NORMAL"
    max_prob = 0.0
    first_attack_ts: float | None = None
    flagged_total = 0
    gate_reasons_final: List[str] = []

    consec_high = 0
    cluster_size = 0
    consec_low = 0
    cooldown_ctr = 0
    cooldown_len = max(1, int(cfg.cooldown_windows))

    for w in win_list:
        prob = float(w.prob)
        max_prob = max(max_prob, prob)
        gate_ok, gate_reasons = (True, []) if not cfg.enable_gate else _gate_arp(w.snaps, cfg)

        if state == "NORMAL":
            if cooldown_ctr > 0:
                cooldown_ctr -= 1

            if prob >= cfg.tau_high and gate_ok and cooldown_ctr == 0:
                consec_high += 1
                cluster_size += 1
            else:
                consec_high = 0
                cluster_size = 0

            if consec_high >= cfg.consecutive_required and cluster_size >= cfg.min_attack_windows:
                state = "ATTACK"
                flagged_total += cluster_size
                if first_attack_ts is None:
                    first_attack_ts = w.t0
                    gate_reasons_final = gate_reasons[:4]
                consec_low = 0
                continue

        else:  # ATTACK state
            if prob >= cfg.tau_high and (gate_ok or not cfg.enable_gate):
                flagged_total += 1
            if prob <= cfg.tau_low:
                consec_low += 1
            else:
                consec_low = 0
            if consec_low >= cooldown_len:
                state = "NORMAL"
                cooldown_ctr = max(0, int(cfg.cooldown_windows))
                consec_high = 0
                cluster_size = 0
                consec_low = 0

    decision = "attack" if first_attack_ts is not None else "normal"
    return FileDecision(
        file=file_path,
        decision=decision,
        first_attack_timestamp=first_attack_ts,
        num_attack_windows=flagged_total,
        max_prob=max_prob,
        gate_reasons=gate_reasons_final[:4],
    )
