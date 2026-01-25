#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dae.cli import run_extract
from dae.config import load_config
from dae.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract feature windows from PCAP files")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--pcaps", nargs="+", required=True, help="PCAP glob patterns")
    parser.add_argument("--out", required=True, help="Output Parquet path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(Path(args.config))
    stats = run_extract(config, args.pcaps, Path(args.out))
    get_logger("extract_script").info(
        "extraction_complete",
        packets=stats.packets,
        windows=stats.windows,
        files=stats.files,
        output=args.out,
    )


if __name__ == "__main__":
    main()
