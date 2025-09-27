#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dae.cli import run_infer_live
from dae.config import load_config
from dae.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live anomaly detection on a network interface")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--iface", required=True, help="Network interface to sniff")
    parser.add_argument("--duration", type=int, default=60, help="Sniff duration in seconds")
    parser.add_argument("--out", required=True, help="Output JSON report path")
    parser.add_argument("--csv", help="Optional CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(Path(args.config))
    run_infer_live(
        config,
        args.iface,
        args.duration,
        Path(args.out),
        Path(args.csv) if args.csv else None,
    )
    get_logger("live_script").info("live_inference_complete", report=str(args.out))


if __name__ == "__main__":
    main()
