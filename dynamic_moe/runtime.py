"""Runtime logging utilities for the dynamic MoE SDN pipeline."""

from __future__ import annotations

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional

from scapy.layers.l2 import Ether
from scapy.utils import PcapWriter

from .config import RuntimeConfig

FLOW_HEADERS = [
    "timestamp",
    "switch",
    "in_port",
    "out_port",
    "src_mac",
    "dst_mac",
    "action",
    "notes",
]

PACKET_HEADERS = [
    "timestamp",
    "switch",
    "window_index",
    "src_mac",
    "dst_mac",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "decision",
    "score",
]

MITIGATION_HEADERS = [
    "timestamp",
    "switch",
    "class",
    "confidence",
    "src_mac",
    "dst_mac",
    "src_ip",
    "dst_ip",
    "action",
    "expiry",
    "reason",
]


class RuntimeLogger:
    """Filesystem-backed logger that mirrors controller + MoE decisions."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.config.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._decision_lock = threading.Lock()
        self._alerts_lock = threading.Lock()
        self._alert_file = self.config.alerts_path.open("a", encoding="utf-8")
        self._decision_file = self.config.decisions_log_path.open("a", encoding="utf-8")
        self._pcap_writer: Optional[PcapWriter] = None
        if self.config.attack_pcap_path is not None:
            self._pcap_writer = PcapWriter(str(self.config.attack_pcap_path), append=True, sync=True)

    @staticmethod
    def _append_csv(path: Path, headers: list[str], row: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_flow_event(
        self,
        switch: int,
        in_port: int,
        out_port: int,
        src_mac: str,
        dst_mac: str,
        action: str,
        notes: str | None = None,
    ) -> None:
        """Record a flow programming decision."""

        row = {
            "timestamp": self._utc_now(),
            "switch": switch,
            "in_port": in_port,
            "out_port": out_port,
            "src_mac": src_mac,
            "dst_mac": dst_mac,
            "action": action,
            "notes": notes or "",
        }
        self._append_csv(self.config.flows_path, FLOW_HEADERS, row)

    def log_packet_metadata(
        self,
        context: Mapping[str, object],
        packet_meta: Mapping[str, object],
        inference: Mapping[str, object],
    ) -> None:
        """Append a short metadata row for every inference window."""

        row = {
            "timestamp": self._utc_now(),
            "switch": packet_meta.get("switch"),
            "window_index": context.get("window_index"),
            "src_mac": packet_meta.get("src_mac"),
            "dst_mac": packet_meta.get("dst_mac"),
            "src_ip": packet_meta.get("src_ip"),
            "dst_ip": packet_meta.get("dst_ip"),
            "src_port": packet_meta.get("src_port"),
            "dst_port": packet_meta.get("dst_port"),
            "protocol": packet_meta.get("protocol", "unknown"),
            "decision": inference.get("attack_type") or "normal",
            "score": f"{float(inference.get('score', 0.0)):.4f}",
        }
        self._append_csv(self.config.packets_meta_path, PACKET_HEADERS, row)

    def log_decision(
        self,
        context: Mapping[str, object],
        packet_meta: Mapping[str, object],
        inference: Mapping[str, object],
    ) -> None:
        """Write a human-readable log line for every MoE decision."""

        message = (
            f"{self._utc_now()} switch={packet_meta.get('switch')} "
            f"window={context.get('window_index')} "
            f"{packet_meta.get('src_mac')}->{packet_meta.get('dst_mac')} "
            f"decision={inference.get('attack_type') or 'normal'} "
            f"score={float(inference.get('score', 0.0)):.3f} "
            f"is_attack={bool(inference.get('is_attack'))}"
        )
        with self._decision_lock:
            self._decision_file.write(message + "\n")
            self._decision_file.flush()

    def log_attack(
        self,
        context: Mapping[str, object],
        packet_meta: Mapping[str, object],
        inference: Mapping[str, object],
        raw_frame: bytes | None = None,
    ) -> None:
        """Persist a JSON alert for high-confidence attacks."""

        payload: Dict[str, object] = {
            "timestamp": self._utc_now(),
            "switch": packet_meta.get("switch"),
            "in_port": packet_meta.get("in_port"),
            "out_port": packet_meta.get("out_port"),
            "src_mac": packet_meta.get("src_mac"),
            "dst_mac": packet_meta.get("dst_mac"),
            "src_ip": packet_meta.get("src_ip"),
            "dst_ip": packet_meta.get("dst_ip"),
            "src_port": packet_meta.get("src_port"),
            "dst_port": packet_meta.get("dst_port"),
            "src_device": packet_meta.get("src_device"),
            "dst_device": packet_meta.get("dst_device"),
            "attack_type": inference.get("attack_type"),
            "score": inference.get("score"),
            "probabilities": inference.get("probabilities"),
            "expert_votes": inference.get("expert_votes"),
            "expert_weights": inference.get("expert_weights"),
            "window_index": context.get("window_index"),
            "window_start": context.get("start_time"),
            "window_end": context.get("end_time"),
            "packet_count": context.get("packet_count"),
        }
        with self._alerts_lock:
            self._alert_file.write(json.dumps(payload) + "\n")
            self._alert_file.flush()
        if self._pcap_writer is not None and raw_frame is not None:
            try:
                pkt = Ether(raw_frame)
                self._pcap_writer.write(pkt)
            except Exception:
                pass

    def log_mitigation(
        self,
        packet_meta: Mapping[str, object],
        inference: Mapping[str, object],
        action: str,
        expiry: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Append an auditable SDN mitigation decision."""

        label = inference.get("attack_type") or inference.get("label") or "normal"
        confidence = float(inference.get("confidence", inference.get("score", 0.0)) or 0.0)
        row = {
            "timestamp": self._utc_now(),
            "switch": packet_meta.get("switch"),
            "class": label,
            "confidence": f"{confidence:.4f}",
            "src_mac": packet_meta.get("src_mac"),
            "dst_mac": packet_meta.get("dst_mac"),
            "src_ip": packet_meta.get("src_ip"),
            "dst_ip": packet_meta.get("dst_ip"),
            "action": action,
            "expiry": expiry or "",
            "reason": reason or "",
        }
        self._append_csv(self.config.mitigations_path, MITIGATION_HEADERS, row)

    def close(self) -> None:
        """Release file descriptors."""

        with self._decision_lock:
            try:
                self._decision_file.close()
            except Exception:
                pass
        with self._alerts_lock:
            try:
                self._alert_file.close()
            except Exception:
                pass
        if self._pcap_writer is not None:
            try:
                self._pcap_writer.close()
            except Exception:
                pass


__all__ = ["RuntimeLogger"]
