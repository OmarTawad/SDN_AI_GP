"""Entry-point that wires together Mininet, Ryu, and the MoE adapter."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from mininet.cli import CLI

from .config import DEFAULT_CONFIG_PATH, RuntimeConfig, load_runtime_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from .pcap_replay import replay_pcap_on_host
from .topology import build_dynamic_moe_topology

LOGGER = logging.getLogger("dynamic_moe.runner")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic MoE Mininet experiment harness")
    parser.add_argument("--pcap", type=Path, help="Optional PCAP file to replay once the topology is online.")
    parser.add_argument("--loop-pcap", action="store_true", help="Replay the PCAP continuously.")
    parser.add_argument("--rate", help="Packet rate hint passed to tcpreplay (pps).")
    parser.add_argument("--replay-host", help="Mininet host that emits the replay traffic.")
    parser.add_argument("--controller-ip", default="127.0.0.1", help="IP for the remote Ryu controller.")
    parser.add_argument("--controller-port", default=6633, type=int, help="OpenFlow port for the controller.")
    parser.add_argument("--ryu-app", default="dynamic_moe.controller", help="Ryu application to spawn.")
    parser.add_argument("--no-controller", action="store_true", help="Skip launching ryu-manager (use external instance).")
    parser.add_argument("--no-cli", action="store_true", help="Do not attach the Mininet CLI; exit after replay finishes.")
    parser.add_argument("--default-threshold", type=float, help="Override the default attack probability threshold (0-1).")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to dynamic_moe/config.yaml (overrides env if provided).",
    )
    return parser.parse_args(argv)


def _launch_controller(args: argparse.Namespace, runtime_config: RuntimeConfig) -> Optional[subprocess.Popen]:
    if args.no_controller:
        LOGGER.info("Skipping controller launch; assume ryu-manager is running externally.")
        return None
    if _is_port_in_use(args.controller_ip, args.controller_port):
        LOGGER.info(
            "Controller port %s:%d already bound; skipping internal ryu-manager (use --no-controller to silence this).",
            args.controller_ip,
            args.controller_port,
        )
        return None
    cmd = [
        "ryu-manager",
        "--ofp-tcp-listen-host",
        args.controller_ip,
        "--ofp-tcp-listen-port",
        str(args.controller_port),
        args.ryu_app,
    ]
    env = os.environ.copy()
    env["DYNAMIC_MOE_CONFIG"] = str(runtime_config.config_path)
    env["PYTHONPATH"] = _build_pythonpath(env)
    if args.default_threshold is not None:
        env["DYNAMIC_MOE_DEFAULT_THRESHOLD"] = str(args.default_threshold)
    LOGGER.info("Starting controller: %s", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def _build_pythonpath(env: dict[str, str]) -> str:
    """Aggregate project and user site-package paths so sudoed processes import dependencies."""

    segments: list[str] = []
    project = str(PROJECT_ROOT)
    segments.append(project)

    sudo_user = env.get("SUDO_USER") or os.environ.get("SUDO_USER")
    if sudo_user:
        sudo_home = Path(os.path.expanduser(f"~{sudo_user}"))
    else:
        sudo_home = Path(os.path.expanduser("~"))
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    user_site = sudo_home / f".local/lib/python{version}/site-packages"
    if user_site.exists():
        segments.append(str(user_site))

    existing = env.get("PYTHONPATH")
    if existing:
        segments.append(existing)

    # preserve order while removing duplicates
    deduped: list[str] = []
    seen: set[str] = set()
    for item in segments:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return os.pathsep.join(deduped)


def _ensure_root() -> None:
    if os.geteuid() != 0:
        raise SystemExit("This script must be run as root for Mininet.")


def _is_port_in_use(host: str, port: int) -> bool:
    """Return True if a TCP port is already bound."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _configure_logging() -> None:
    """Configure project logging without altering Mininet's plain CLI output."""

    dyn_logger = logging.getLogger("dynamic_moe")
    for handler in list(dyn_logger.handlers):
        dyn_logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    handler.setLevel(logging.INFO)
    dyn_logger.setLevel(logging.INFO)
    dyn_logger.addHandler(handler)
    dyn_logger.propagate = False


def main(argv: Optional[list[str]] = None) -> None:
    _ensure_root()
    _configure_logging()
    args = _parse_args(argv)
    runtime_config = load_runtime_config(args.config)
    controller_proc = None
    net = None
    replay_proc = None
    try:
        controller_proc = _launch_controller(args, runtime_config)
        time.sleep(1.5)  # allow controller to bind
        net = build_dynamic_moe_topology(controller_ip=args.controller_ip, controller_port=args.controller_port)
        net.start()
        net.staticArp()
        LOGGER.info("Topology started with %d hosts.", len(net.hosts))
        if args.pcap:
            replay_host_name = args.replay_host or runtime_config.default_replay_host
            host = net.get(replay_host_name)
            replay_proc = replay_pcap_on_host(host, args.pcap, loop=args.loop_pcap, rate=args.rate)
        if args.no_cli:
            LOGGER.info("CLI disabled; waiting for replay (Ctrl+C to exit).")
            while replay_proc and replay_proc.poll() is None:
                time.sleep(1)
        else:
            CLI(net)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted; shutting down.")
    finally:
        if replay_proc and replay_proc.poll() is None:
            replay_proc.terminate()
            try:
                replay_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                replay_proc.kill()
        if net is not None:
            net.stop()
        if controller_proc is not None:
            controller_proc.send_signal(signal.SIGINT)
            try:
                controller_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                controller_proc.terminate()
    LOGGER.info("Dynamic MoE experiment finished.")


if __name__ == "__main__":
    main()
