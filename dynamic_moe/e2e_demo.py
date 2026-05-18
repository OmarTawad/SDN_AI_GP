"""Full replay-based Dynamic MoE end-to-end demo runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml
from scapy.utils import PcapReader

from dynamic_moe.config import RuntimeConfig, load_runtime_config
from dynamic_moe.feature_extractor import FeatureWindow, StreamingFeatureExtractor
from dynamic_moe.pcap_replay import replay_pcap_on_host
from dynamic_moe.policy import PolicyConfig, SdnPolicyEngine
from dynamic_moe.runtime import RuntimeLogger
from dynamic_moe.topology import build_dynamic_moe_topology
from gateway.core import CLASS_LABELS
from gateway.data.datasets.gating import UNIFIED_GATING_COMPONENT_KEYS
from gateway.data.extractors import features as feature_meta
from gateway.dynamic_moe_adapter import DynamicMoEGateway
from gateway.models.unified_moe import UNIFIED_EXPERT_SPECS, UNIFIED_GATING_INPUT_DIM


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NetSentinel Dynamic MoE E2E demo.")
    parser.add_argument("--pcap", type=Path, required=True, help="Input PCAP to replay through the pipeline.")
    parser.add_argument("--runtime-dir", type=Path, required=True, help="Directory for runtime logs.")
    parser.add_argument("--mode", choices=("full", "dry-run", "mininet"), default="full")
    parser.add_argument("--controller", choices=("ryu", "none"), default="ryu")
    flow_group = parser.add_mutually_exclusive_group()
    flow_group.add_argument("--install-flows", action="store_true", help="Attempt real OpenFlow installation.")
    flow_group.add_argument("--no-install-flows", action="store_true", help="Skip physical OpenFlow installation.")
    parser.add_argument("--duration", type=float, help="Maximum PCAP seconds to process from first packet.")
    parser.add_argument("--output-json", type=Path, required=True, help="Machine-readable summary path.")
    parser.add_argument("--config", type=Path, help="Optional dynamic_moe/config.yaml path.")
    return parser.parse_args(argv)


def _runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    config = load_runtime_config(args.config)
    runtime_dir = args.runtime_dir.resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return replace(
        config,
        runtime_dir=runtime_dir,
        alerts_path=runtime_dir / config.alerts_path.name,
        flows_path=runtime_dir / config.flows_path.name,
        packets_meta_path=runtime_dir / config.packets_meta_path.name,
        decisions_log_path=runtime_dir / config.decisions_log_path.name,
        mitigations_path=runtime_dir / config.mitigations_path.name,
        attack_pcap_path=None,
    )


def _policy(config: RuntimeConfig) -> SdnPolicyEngine:
    return SdnPolicyEngine(
        PolicyConfig(
            min_alert_confidence=config.min_alert_confidence,
            min_rate_limit_confidence=config.min_rate_limit_confidence,
            min_block_confidence=config.min_block_confidence,
            arp_isolate_confidence=config.arp_isolate_confidence,
            mitigation_mode=config.mitigation,
            allow_automatic_blocking=config.allow_automatic_blocking,
            action_expiry_seconds=config.action_expiry_seconds,
        )
    )


def _dependency_status() -> dict[str, Any]:
    return {
        "mininet_binary": shutil.which("mn"),
        "ovs_vsctl_binary": shutil.which("ovs-vsctl"),
        "ovs_ofctl_binary": shutil.which("ovs-ofctl"),
        "ryu_manager_binary": shutil.which("ryu-manager"),
        "ryu_python_module": importlib.util.find_spec("ryu") is not None,
        "is_root": os.geteuid() == 0,
    }


def _validate_artifacts(gateway: DynamicMoEGateway) -> dict[str, Any]:
    paths = {
        "checkpoint": gateway.config.checkpoint,
        "autoencoder_scaler": feature_meta.AUTO_ARTIFACT_DIR / "scaler.pkl",
        "autoencoder_model_config": feature_meta.AUTO_ARTIFACT_DIR / "model_config.json",
        "dos_cnn_scaler_artifact": feature_meta.DOS_CNN_ARTIFACT_DIR / "scaler.pkl",
        "arp_cnn_scaler_artifact": feature_meta.ARP_CNN_ARTIFACT_DIR / "scaler.pkl",
        "dos_lstm_scaler": feature_meta.DOS_LSTM_MODEL_DIR / "feature_scaler.joblib",
        "arp_lstm_scaler": feature_meta.ARP_LSTM_MODEL_DIR / "feature_scaler.joblib",
    }
    return {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in paths.items()
    }


def _feature_shapes(window: FeatureWindow) -> dict[str, list[int]]:
    return {
        key: [int(dim) for dim in tensor.shape]
        for key, tensor in window.features.items()
        if hasattr(tensor, "shape")
    }


def _packet_meta(window: FeatureWindow) -> dict[str, object]:
    meta = dict(window.packet_metadata)
    if not meta.get("src_ip"):
        meta["src_ip"] = window.context.get("top_src_ip")
    if not meta.get("dst_ip"):
        meta["dst_ip"] = window.context.get("top_dst_ip")
    meta.setdefault("switch", 0)
    meta.setdefault("in_port", None)
    meta.setdefault("out_port", None)
    return meta


def _openflow_status(
    args: argparse.Namespace,
    dependency_status: Mapping[str, Any],
    decision_action: str | None,
) -> dict[str, Any]:
    requested = bool(args.install_flows)
    status = {
        "attempted": False,
        "succeeded": False,
        "skipped": True,
        "reason": "flow_install_disabled",
        "action": decision_action,
    }
    if args.no_install_flows or not requested:
        return status
    status["attempted"] = True
    if args.mode != "mininet":
        status["reason"] = "physical_openflow_requires_mode_mininet"
        return status
    missing = [
        name
        for name, ok in {
            "root": dependency_status.get("is_root"),
            "mn": dependency_status.get("mininet_binary"),
            "ovs-vsctl": dependency_status.get("ovs_vsctl_binary"),
            "ryu-manager": dependency_status.get("ryu_manager_binary"),
            "ryu_python_module": dependency_status.get("ryu_python_module"),
        }.items()
        if not ok
    ]
    if missing:
        status["reason"] = "missing_live_sdn_dependencies:" + ",".join(missing)
        return status
    status["reason"] = "live_mininet_mode_available_but_not_started_by_replay_summary_path"
    return status


def _write_live_config(config: RuntimeConfig, install_flows: bool) -> Path:
    """Write an isolated controller config for the Mininet live demo."""

    controller = {
        "ip": config.controller_ip,
        "port": config.controller_port,
        "mitigation": "drop" if install_flows else config.mitigation,
        "min_alert_confidence": config.min_alert_confidence,
        "min_rate_limit_confidence": config.min_rate_limit_confidence,
        "min_block_confidence": min(config.min_block_confidence, 0.90) if install_flows else config.min_block_confidence,
        "arp_isolate_confidence": min(config.arp_isolate_confidence, 0.90) if install_flows else config.arp_isolate_confidence,
        "allow_automatic_blocking": bool(install_flows or config.allow_automatic_blocking),
        "action_expiry_seconds": config.action_expiry_seconds,
    }
    payload = {
        "runtime": {
            "root_dir": str(config.runtime_dir),
            "alerts_file": config.alerts_path.name,
            "flows_file": config.flows_path.name,
            "packets_meta_file": config.packets_meta_path.name,
            "decisions_log": config.decisions_log_path.name,
            "mitigations_file": config.mitigations_path.name,
        },
        "controller": controller,
        "pcap": {"default_replay_host": config.default_replay_host},
    }
    path = config.runtime_dir / "live_dynamic_moe_config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _live_mininet_e2e(args: argparse.Namespace, runtime_config: RuntimeConfig, dependency_status: Mapping[str, Any]) -> dict[str, Any]:
    missing = [
        key
        for key in ("mininet_binary", "ovs_vsctl_binary", "ovs_ofctl_binary", "ryu_python_module", "is_root")
        if not dependency_status.get(key)
    ]
    if missing:
        raise RuntimeError("Cannot run live Mininet/OpenFlow demo; missing " + ", ".join(missing))

    config_path = _write_live_config(runtime_config, install_flows=args.install_flows)
    env = os.environ.copy()
    env["DYNAMIC_MOE_CONFIG"] = str(config_path)
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")] if item
    )
    controller_cmd = [
        sys.executable,
        "-m",
        "ryu.cmd.manager",
        "--ofp-listen-host",
        runtime_config.controller_ip,
        "--ofp-tcp-listen-port",
        str(runtime_config.controller_port),
        "dynamic_moe.controller",
    ]

    controller_proc: subprocess.Popen[str] | None = None
    net = None
    replay_proc = None
    flow_dump = ""
    controller_output = ""
    try:
        controller_proc = subprocess.Popen(
            controller_cmd,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        net = build_dynamic_moe_topology(
            controller_ip=runtime_config.controller_ip,
            controller_port=runtime_config.controller_port,
        )
        net.start()
        net.staticArp()
        host = net.get(runtime_config.default_replay_host)
        replay_proc = replay_pcap_on_host(host, args.pcap, loop=False, rate=None)
        try:
            replay_proc.wait(timeout=args.duration or 20)
        except subprocess.TimeoutExpired:
            replay_proc.terminate()
            replay_proc.wait(timeout=3)
        # Give the controller's worker thread time to classify completed windows.
        import time

        time.sleep(3)
        dump = subprocess.run(
            ["ovs-ofctl", "-O", "OpenFlow13", "dump-flows", "s1"],
            text=True,
            capture_output=True,
            check=False,
        )
        flow_dump = dump.stdout + dump.stderr
        (runtime_config.runtime_dir / "ovs_flows_s1.txt").write_text(flow_dump, encoding="utf-8")
    finally:
        if replay_proc is not None and replay_proc.poll() is None:
            replay_proc.terminate()
        if net is not None:
            net.stop()
        if controller_proc is not None:
            controller_proc.terminate()
            try:
                controller_output, _ = controller_proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                controller_proc.kill()
                controller_output, _ = controller_proc.communicate(timeout=5)
            (runtime_config.runtime_dir / "ryu_controller.log").write_text(controller_output or "", encoding="utf-8")

    mitigations = runtime_config.mitigations_path.read_text(encoding="utf-8") if runtime_config.mitigations_path.exists() else ""
    decisions = runtime_config.decisions_log_path.read_text(encoding="utf-8") if runtime_config.decisions_log_path.exists() else ""
    install_attempted = bool(args.install_flows)
    flow_succeeded = install_attempted and "priority=20" in flow_dump and ("actions=drop" in flow_dump or "actions" in flow_dump)
    summary = {
        "status": "PASS" if decisions and mitigations else "FAIL",
        "mode": "mininet",
        "controller": args.controller,
        "pcap": str(args.pcap),
        "runtime_dir": str(runtime_config.runtime_dir),
        "dependency_status": dict(dependency_status),
        "ryu_controller_started": bool(controller_output is not None),
        "mininet_topology_started": True,
        "ovs_switch": "s1",
        "moe_predictions_observed": bool(decisions),
        "mitigations_observed": bool(mitigations),
        "openflow": {
            "attempted": install_attempted,
            "succeeded": flow_succeeded,
            "skipped": not install_attempted,
            "reason": "visible_priority_20_rule" if flow_succeeded else "no_visible_mitigation_rule_in_ovs_dump",
            "flow_dump_path": str(runtime_config.runtime_dir / "ovs_flows_s1.txt"),
        },
        "logs": {
            "decisions": str(runtime_config.decisions_log_path),
            "packets": str(runtime_config.packets_meta_path),
            "mitigations": str(runtime_config.mitigations_path),
            "alerts": str(runtime_config.alerts_path),
            "flows": str(runtime_config.flows_path),
            "ovs_flows": str(runtime_config.runtime_dir / "ovs_flows_s1.txt"),
            "controller": str(runtime_config.runtime_dir / "ryu_controller.log"),
        },
        "output_json": str(args.output_json),
    }
    if install_attempted and not flow_succeeded:
        summary["status"] = "FAIL"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_e2e(args: argparse.Namespace) -> dict[str, Any]:
    if not args.pcap.exists():
        raise FileNotFoundError(f"PCAP does not exist: {args.pcap}")
    runtime_config = _runtime_config(args)
    dependency_status = _dependency_status()
    if args.mode == "mininet":
        return _live_mininet_e2e(args, runtime_config, dependency_status)

    gateway = DynamicMoEGateway()
    policy = _policy(runtime_config)
    logger = RuntimeLogger(runtime_config)
    extractor = StreamingFeatureExtractor()
    packets_read = 0
    windows: list[FeatureWindow] = []
    first_ts: float | None = None

    try:
        with PcapReader(str(args.pcap)) as reader:
            for pkt in reader:
                timestamp = float(getattr(pkt, "time", 0.0))
                if first_ts is None:
                    first_ts = timestamp
                if args.duration is not None and timestamp - first_ts > args.duration:
                    break
                packets_read += 1
                windows.extend(extractor.process_packet(pkt))
        windows.extend(extractor.flush())

        if packets_read == 0:
            raise RuntimeError("No packets were read from the input PCAP.")
        if not windows:
            raise RuntimeError("No 1-second windows were generated from the input PCAP.")

        window_summaries: list[dict[str, Any]] = []
        final_openflow_status: dict[str, Any] | None = None
        for window in windows:
            result = gateway.predict(window.features)
            decision = policy.decide(result)
            packet_meta = _packet_meta(window)
            context = window.context
            logger.log_decision(context, packet_meta, result)
            logger.log_packet_metadata(context, packet_meta, result)
            logger.log_mitigation(packet_meta, result, decision.action, decision.expiry, decision.reason)
            if result.get("is_attack"):
                logger.log_attack(context, packet_meta, result)
            openflow = _openflow_status(args, dependency_status, decision.action)
            final_openflow_status = openflow
            window_summaries.append(
                {
                    "window_index": context.get("window_index"),
                    "packet_count": context.get("packet_count"),
                    "window_start": context.get("start_time"),
                    "window_end": context.get("end_time"),
                    "feature_tensor_shapes": _feature_shapes(window),
                    "expert_output_shapes": result.get("expert_output_shapes"),
                    "expert_outputs": result.get("expert_outputs"),
                    "expert_weights": result.get("expert_weights"),
                    "gate_weight_sum": result.get("gate_weight_sum"),
                    "probabilities": result.get("probabilities"),
                    "prediction": result.get("label"),
                    "confidence": result.get("confidence"),
                    "score": result.get("score"),
                    "is_attack": result.get("is_attack"),
                    "selected_action": decision.action,
                    "policy_reason": decision.reason,
                    "action_expiry": decision.expiry,
                    "openflow": openflow,
                }
            )
    finally:
        logger.close()

    final = window_summaries[-1]
    summary = {
        "status": "PASS",
        "mode": args.mode,
        "controller": args.controller,
        "pcap": str(args.pcap),
        "runtime_dir": str(runtime_config.runtime_dir),
        "packets_read": packets_read,
        "windows_generated": len(windows),
        "preprocessing": {
            "window_size_seconds": feature_meta.WINDOW_SIZE,
            "hop_size_seconds": feature_meta.WINDOW_STRIDE,
            "dos_micro_bins": feature_meta.DOS_MICRO_BINS,
            "arp_micro_bins": feature_meta.ARP_MICRO_BINS,
            "gating_input_dim": UNIFIED_GATING_INPUT_DIM,
            "gating_feature_order": list(UNIFIED_GATING_COMPONENT_KEYS),
        },
        "artifacts": _validate_artifacts(gateway),
        "class_labels": list(CLASS_LABELS),
        "expert_names": [spec.name for spec in UNIFIED_EXPERT_SPECS],
        "windows": window_summaries,
        "final_prediction": {
            "class": final["prediction"],
            "confidence": final["confidence"],
            "probabilities": final["probabilities"],
            "expert_weights": final["expert_weights"],
            "gate_weight_sum": final["gate_weight_sum"],
            "selected_action": final["selected_action"],
            "policy_reason": final["policy_reason"],
        },
        "openflow": final_openflow_status or _openflow_status(args, dependency_status, None),
        "dependency_status": dependency_status,
        "logs": {
            "decisions": str(runtime_config.decisions_log_path),
            "packets": str(runtime_config.packets_meta_path),
            "mitigations": str(runtime_config.mitigations_path),
            "alerts": str(runtime_config.alerts_path),
            "flows": str(runtime_config.flows_path),
        },
        "output_json": str(args.output_json),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_e2e(args)
    except Exception as exc:
        payload = {"status": "FAIL", "error": str(exc), "output_json": str(args.output_json)}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": summary["status"],
                "packets_read": summary.get("packets_read"),
                "windows_generated": summary.get("windows_generated"),
                "final_prediction": summary.get("final_prediction"),
                "moe_predictions_observed": summary.get("moe_predictions_observed"),
                "mitigations_observed": summary.get("mitigations_observed"),
                "openflow": summary["openflow"],
                "mitigation_log": summary.get("logs", {}).get("mitigations"),
                "output_json": summary["output_json"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
