"""Safe SDN mitigation policy for Dynamic MoE detections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping, Optional


ATTACK_LABELS = {"dos", "arp"}


@dataclass(frozen=True)
class PolicyConfig:
    """Confidence tiers used to map IDS output to SDN actions."""

    min_alert_confidence: float = 0.70
    min_rate_limit_confidence: float = 0.90
    min_block_confidence: float = 0.98
    arp_isolate_confidence: float = 0.95
    mitigation_mode: str = "alert"
    allow_automatic_blocking: bool = False
    action_expiry_seconds: int = 300


@dataclass(frozen=True)
class MitigationDecision:
    """Controller-facing policy decision."""

    action: str
    install_flow: bool
    reason: str
    expiry: Optional[str] = None
    priority: int = 20


class SdnPolicyEngine:
    """Maps MoE predictions to safe SDN actions.

    The default policy is intentionally conservative: attacks can alert or
    rate-limit in dry-run output, but automatic drop/isolate actions require an
    explicit allow flag and very high confidence.
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    @staticmethod
    def _label(inference: Mapping[str, object]) -> str:
        label = inference.get("attack_type") or inference.get("label") or "normal"
        return str(label).lower()

    @staticmethod
    def _confidence(inference: Mapping[str, object]) -> float:
        if "confidence" in inference:
            return float(inference.get("confidence") or 0.0)
        if "score" in inference:
            return float(inference.get("score") or 0.0)
        probabilities = inference.get("probabilities")
        if isinstance(probabilities, Mapping):
            label = SdnPolicyEngine._label(inference)
            return float(probabilities.get(label, 0.0) or 0.0)
        return 0.0

    def decide(self, inference: Mapping[str, object]) -> MitigationDecision:
        """Return the SDN action for one model prediction."""

        label = self._label(inference)
        confidence = self._confidence(inference)
        is_attack = bool(inference.get("is_attack")) or label in ATTACK_LABELS

        if label == "normal" or not is_attack:
            return MitigationDecision("monitor", False, "normal_or_not_attack")

        if confidence < self.config.min_alert_confidence:
            return MitigationDecision("monitor", False, "below_alert_confidence")

        if label == "dos" and confidence >= self.config.min_rate_limit_confidence:
            if confidence >= self.config.min_block_confidence:
                return self._high_confidence_action("drop", "high_confidence_dos")
            return self._expiring_action("rate_limit", False, "medium_confidence_dos")

        if label == "arp" and confidence >= self.config.arp_isolate_confidence:
            return self._high_confidence_action("isolate", "high_confidence_arp")

        return MitigationDecision("alert", False, "attack_above_alert_confidence")

    def _high_confidence_action(self, action: str, reason: str) -> MitigationDecision:
        if not self.config.allow_automatic_blocking:
            return MitigationDecision("alert", False, f"{reason}_auto_block_disabled")
        if self.config.mitigation_mode not in {"drop", "isolate", "quarantine"}:
            return MitigationDecision("alert", False, f"{reason}_mitigation_mode_{self.config.mitigation_mode}")
        selected = action if self.config.mitigation_mode == "drop" else self.config.mitigation_mode
        return self._expiring_action(selected, True, reason)

    def _expiring_action(self, action: str, install_flow: bool, reason: str) -> MitigationDecision:
        expiry = datetime.now(timezone.utc) + timedelta(seconds=self.config.action_expiry_seconds)
        return MitigationDecision(
            action=action,
            install_flow=install_flow,
            reason=reason,
            expiry=expiry.isoformat(),
        )


__all__ = ["MitigationDecision", "PolicyConfig", "SdnPolicyEngine"]
