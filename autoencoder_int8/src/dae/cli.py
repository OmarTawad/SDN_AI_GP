from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence

from scapy.utils import wrpcap
from scapy.sendrecv import sniff

from .config import Config, ensure_directories, load_config
from .extract import ExtractionStats, extract_pcaps
from .infer import infer_pcap
from .logging import configure_logging, get_logger
from .train import train_model
from .utils_io import iter_pcap_files


def run_extract(config: Config, patterns: Sequence[str], output_path: Path) -> ExtractionStats:
    ensure_directories(
        config,
        [
            ("paths", "windows_dir"),
        ],
    )
    files = list(iter_pcap_files(patterns))
    if not files:
        raise FileNotFoundError("No PCAP files matched the provided patterns")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return extract_pcaps(files, config, output_path)


def run_train(config: Config, windows_path: Path):
    ensure_directories(
        config,
        [
            ("paths", "artifacts_dir"),
        ],
    )
    return train_model(config, windows_path)


def run_infer_file(config: Config, pcap_path: Path, report_path: Path, csv_path: Path | None = None):
    ensure_directories(
        config,
        [
            ("paths", "reports_dir"),
        ],
    )
    result = infer_pcap(config, pcap_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result["report"], indent=2), encoding="utf-8")

    if csv_path is None:
        csv_path = report_path.with_suffix(".csv")
    result["details"].to_csv(csv_path, index=False)

    return result


def run_infer_live(
    config: Config,
    interface: str,
    duration: int,
    report_path: Path,
    csv_path: Path | None = None,
):
    logger = get_logger("live")
    logger.info("sniff_start", interface=interface, duration=duration)
    packets = sniff(iface=interface, timeout=duration, store=True)
    if not packets:
        raise RuntimeError("No packets captured during live sniffing")
    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
        wrpcap(tmp.name, packets)
        temp_path = Path(tmp.name)
    logger.info("sniff_complete", packets=len(packets), temp_file=str(temp_path))
    try:
        return run_infer_file(config, temp_path, report_path, csv_path)
    finally:
        temp_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dae", description="Autoencoder-based network anomaly detection CLI")
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract", help="Extract windows from PCAP files")
    extract_parser.add_argument("--config", required=True, help="Path to config.yaml")
    extract_parser.add_argument("--pcaps", nargs="+", required=True, help="PCAP file patterns")
    extract_parser.add_argument("--out", required=True, help="Output Parquet path")

    train_parser = subparsers.add_parser("train", help="Train the autoencoder model")
    train_parser.add_argument("--config", required=True, help="Path to config.yaml")
    train_parser.add_argument("--windows", required=True, help="Parquet file with training windows")

    infer_parser = subparsers.add_parser("infer-file", help="Run inference on a PCAP file")
    infer_parser.add_argument("--config", required=True, help="Path to config.yaml")
    infer_parser.add_argument("--pcap", required=True, help="PCAP file to score")
    infer_parser.add_argument("--out", required=True, help="Report JSON output path")
    infer_parser.add_argument("--csv", help="Optional CSV output path for window details")

    live_parser = subparsers.add_parser("infer-live", help="Run live inference on an interface")
    live_parser.add_argument("--config", required=True, help="Path to config.yaml")
    live_parser.add_argument("--iface", required=True, help="Network interface to sniff")
    live_parser.add_argument("--duration", type=int, default=60, help="Sniff duration in seconds")
    live_parser.add_argument("--out", required=True, help="Report JSON output path")
    live_parser.add_argument("--csv", help="Optional CSV output path for window details")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    config = load_config(Path(args.config))

    if args.command == "extract":
        stats = run_extract(config, args.pcaps, Path(args.out))
        get_logger("cli").info(
            "extract_done",
            packets=stats.packets,
            windows=stats.windows,
            files=stats.files,
            output=args.out,
        )
    elif args.command == "train":
        artifacts = run_train(config, Path(args.windows))
        get_logger("cli").info(
            "train_done",
            model=str(artifacts.model_path),
            scaler=str(artifacts.scaler_path),
        )
    elif args.command == "infer-file":
        result = run_infer_file(config, Path(args.pcap), Path(args.out), Path(args.csv) if args.csv else None)
        get_logger("cli").info(
            "infer_file_done",
            decision=result["report"]["decision"],
            anomalous=result["report"]["anomalous_windows"],
        )
    elif args.command == "infer-live":
        run_infer_live(
            config,
            args.iface,
            int(args.duration),
            Path(args.out),
            Path(args.csv) if args.csv else None,
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
