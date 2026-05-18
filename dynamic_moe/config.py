"""Runtime configuration helpers for the dynamic MoE pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from gateway.core import PATHS

LOGGER = logging.getLogger("dynamic_moe.config")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")


@dataclass(frozen=True)
class RuntimeConfig:
    """Container describing filesystem paths and controller defaults."""

    config_path: Path
    runtime_dir: Path
    alerts_path: Path
    flows_path: Path
    packets_meta_path: Path
    decisions_log_path: Path
    mitigations_path: Path
    attack_pcap_path: Optional[Path]
    controller_ip: str
    controller_port: int
    default_replay_host: str
    mitigation: str
    min_alert_confidence: float
    min_rate_limit_confidence: float
    min_block_confidence: float
    arp_isolate_confidence: float
    allow_automatic_blocking: bool
    action_expiry_seconds: int


def _resolve_path(value: str | Path, base: Path) -> Path:
    """Normalise a configuration path relative to the repository root."""

    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        LOGGER.warning("Runtime config %s is missing; falling back to defaults.", path)
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Runtime config at {path} must contain a mapping.")
        return loaded


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_runtime_config(path: str | Path | None = None) -> RuntimeConfig:
    """Load the dynamic MoE runtime configuration."""

    env_override = os.environ.get("DYNAMIC_MOE_CONFIG")
    selected = Path(path or env_override or DEFAULT_CONFIG_PATH)
    if not selected.is_absolute():
        selected = PATHS.project_root / selected
    data = _read_yaml(selected)
    runtime_section = data.get("runtime", {})
    controller_section = data.get("controller", {})
    pcap_section = data.get("pcap", {})

    runtime_dir = _resolve_path(runtime_section.get("root_dir", "dynamic_moe_runtime"), PATHS.project_root)
    alerts_path = runtime_dir / runtime_section.get("alerts_file", "alerts.jsonl")
    flows_path = runtime_dir / runtime_section.get("flows_file", "flows.csv")
    packets_meta_path = runtime_dir / runtime_section.get("packets_meta_file", "packets_meta.csv")
    decisions_log_path = runtime_dir / runtime_section.get("decisions_log", "moe_decisions.log")
    mitigations_path = runtime_dir / runtime_section.get("mitigations_file", "mitigations.csv")
    attack_pcap = runtime_section.get("attack_pcap")
    attack_pcap_path = runtime_dir / attack_pcap if attack_pcap else None

    runtime_dir.mkdir(parents=True, exist_ok=True)

    controller_ip = str(controller_section.get("ip", "127.0.0.1"))
    controller_port = int(controller_section.get("port", 6633))
    mitigation = str(controller_section.get("mitigation", "alert"))
    min_alert_confidence = float(controller_section.get("min_alert_confidence", 0.70))
    min_rate_limit_confidence = float(controller_section.get("min_rate_limit_confidence", 0.90))
    min_block_confidence = float(controller_section.get("min_block_confidence", 0.98))
    arp_isolate_confidence = float(controller_section.get("arp_isolate_confidence", 0.95))
    allow_automatic_blocking = _as_bool(controller_section.get("allow_automatic_blocking", False))
    action_expiry_seconds = int(controller_section.get("action_expiry_seconds", 300))
    default_replay_host = str(pcap_section.get("default_replay_host", "h_smartthings"))

    return RuntimeConfig(
        config_path=selected,
        runtime_dir=runtime_dir,
        alerts_path=alerts_path,
        flows_path=flows_path,
        packets_meta_path=packets_meta_path,
        decisions_log_path=decisions_log_path,
        mitigations_path=mitigations_path,
        attack_pcap_path=attack_pcap_path,
        controller_ip=controller_ip,
        controller_port=controller_port,
        default_replay_host=default_replay_host,
        mitigation=mitigation,
        min_alert_confidence=min_alert_confidence,
        min_rate_limit_confidence=min_rate_limit_confidence,
        min_block_confidence=min_block_confidence,
        arp_isolate_confidence=arp_isolate_confidence,
        allow_automatic_blocking=allow_automatic_blocking,
        action_expiry_seconds=action_expiry_seconds,
    )


__all__ = ["RuntimeConfig", "load_runtime_config", "DEFAULT_CONFIG_PATH"]
