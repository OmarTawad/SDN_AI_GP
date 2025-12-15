"""Command-line configuration handling for MoE gate training.

----
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from gateway.data.datasets.cache import CACHE_ROOT


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training script."""

    parser = argparse.ArgumentParser(description="Train the MoE gate on streaming PCAP features or cached tensors.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=17, help="Random seed (default: 17)")
    default_threads = max(1, min(2, os.cpu_count() or 1))
    parser.add_argument(
        "--num-threads",
        type=int,
        default=default_threads,
        help="Torch CPU thread pool size (default: min(2, available cores)).",
    )
    parser.add_argument("--gating-hidden", type=int, default=128, help="Hidden width for the gating MLP (default: 128)")
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=80,
        help="Cap processed windows per PCAP (0 disables the cap).",
    )
    parser.add_argument(
        "--max-total-windows",
        type=int,
        default=1200,
        help="Global window cap across all files (0 disables the cap).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit on optimizer steps per epoch.",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=200,
        help="Emit a progress log every N processed windows (0 disables logging).",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=None,
        help="Skip PCAP files larger than this size in MB.",
    )
    parser.add_argument(
        "--max-packets-per-file",
        type=int,
        default=None,
        help="Stop reading a PCAP after this many packets (0 disables the cap).",
    )
    parser.add_argument(
        "--max-packets-per-window",
        type=int,
        default=None,
        help="Limit stored packets per window (default: no limit).",
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=None,
        help="Abort processing a PCAP after this many seconds (0 disables the timeout).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory containing cached window tensors (default: gateway/cache).",
    )
    parser.add_argument(
        "--use-cache",
        type=str,
        choices=("auto", "on", "off"),
        default="auto",
        help="Use cached tensors (auto tries cache, on requires cache, off disables it).",
    )
    parser.add_argument(
        "--auto-recon-weight",
        type=float,
        default=0.0,
        help="Optional weight for negative autoencoder scores as auxiliary loss.",
    )
    return parser.parse_args()


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    num_threads: int
    gating_hidden: int
    auto_recon_weight: float
    max_windows_per_file: Optional[int]
    max_total_windows: Optional[int]
    max_batches: Optional[int]
    status_interval: Optional[int]
    max_file_size_mb: Optional[float]
    max_packets_per_file: Optional[int]
    file_timeout: Optional[float]
    max_packets_per_window: Optional[int]
    cache_dir: Optional[Path]
    use_cache: str
    tasks: Tuple[str, ...] = ("dos", "arp")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        return cls(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            num_threads=max(1, args.num_threads),
            gating_hidden=max(1, args.gating_hidden),
            auto_recon_weight=max(0.0, float(args.auto_recon_weight)),
            max_windows_per_file=args.max_windows_per_file if args.max_windows_per_file > 0 else None,
            max_total_windows=args.max_total_windows if args.max_total_windows > 0 else None,
            max_batches=args.max_batches,
            status_interval=args.status_interval if args.status_interval and args.status_interval > 0 else None,
            max_file_size_mb=args.max_file_size_mb if args.max_file_size_mb and args.max_file_size_mb > 0 else None,
            max_packets_per_file=(
                args.max_packets_per_file if args.max_packets_per_file and args.max_packets_per_file > 0 else None
            ),
            file_timeout=args.file_timeout if args.file_timeout and args.file_timeout > 0 else None,
            max_packets_per_window=(
                args.max_packets_per_window if args.max_packets_per_window and args.max_packets_per_window > 0 else None
            ),
            cache_dir=Path(args.cache_dir).resolve() if args.cache_dir else None,
            use_cache=args.use_cache,
        )

    @property
    def file_size_bytes(self) -> Optional[int]:
        if self.max_file_size_mb is None:
            return None
        return int(self.max_file_size_mb * 1024 * 1024)

    @property
    def cache_base(self) -> Path:
        return self.cache_dir or CACHE_ROOT


__all__ = ["TrainingConfig", "parse_args"]

