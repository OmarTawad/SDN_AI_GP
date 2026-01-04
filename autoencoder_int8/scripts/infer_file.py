#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dae.cli import run_infer_file
from dae.config import load_config
from dae.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anomaly detection on a PCAP file")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--pcap", required=True, help="PCAP file to score")
    parser.add_argument("--out", required=True, help="Output JSON report path")
    parser.add_argument("--csv", help="Optional CSV to store per-window scores")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(Path(args.config))
    result = run_infer_file(
        config,
        Path(args.pcap),
        Path(args.out),
        Path(args.csv) if args.csv else None,
    )
    report = result["report"]
    get_logger("infer_script").info(
        "inference_report",
        decision=report["decision"],
        anomalous=report["anomalous_windows"],
        total=report["total_windows"],
    )


if __name__ == "__main__":
    main()
