"""Configuration and CLI parsing for unified MoE global evaluation."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from gateway.core import PATHS
from gateway.env import CACHE_ROOT


@dataclass(frozen=True)
class EvaluationConfig:
    """Runtime settings for global unified MoE evaluation."""

    checkpoint: Path
    split: str
    seed: int
    batch_size: int
    num_threads: int
    use_cache: str
    cache_dir: Optional[Path]
    output_dir: Path
    temperature: Optional[float]
    threshold: Optional[float]
    max_windows_per_file: Optional[int]
    max_total_windows: Optional[int]
    status_interval: Optional[int]
    max_file_size_mb: Optional[float]
    max_packets_per_file: Optional[int]
    max_packets_per_window: Optional[int]
    file_timeout: Optional[float]
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    tasks: Tuple[str, ...] = ("dos", "arp")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EvaluationConfig":
        return cls(
            checkpoint=args.checkpoint.resolve(),
            split=str(args.split).lower(),
            seed=int(args.seed),
            batch_size=max(1, int(args.batch_size)),
            num_threads=max(1, int(args.num_threads)),
            use_cache=args.use_cache,
            cache_dir=args.cache_dir.resolve() if args.cache_dir else None,
            output_dir=args.output_dir.resolve(),
            temperature=float(args.temperature) if args.temperature is not None else None,
            threshold=float(args.threshold) if args.threshold is not None else None,
            max_windows_per_file=args.max_windows_per_file if args.max_windows_per_file > 0 else None,
            max_total_windows=args.max_total_windows if args.max_total_windows > 0 else None,
            status_interval=args.status_interval if args.status_interval > 0 else None,
            max_file_size_mb=args.max_file_size_mb if args.max_file_size_mb > 0 else None,
            max_packets_per_file=args.max_packets_per_file if args.max_packets_per_file > 0 else None,
            max_packets_per_window=args.max_packets_per_window if args.max_packets_per_window > 0 else None,
            file_timeout=args.file_timeout if args.file_timeout > 0 else None,
        )

    @property
    def cache_base(self) -> Path:
        return self.cache_dir or CACHE_ROOT

    @property
    def file_size_bytes(self) -> Optional[int]:
        if self.max_file_size_mb is None:
            return None
        return int(self.max_file_size_mb * 1024 * 1024)


def parse_args(argv: Optional[Sequence[str]] = None) -> EvaluationConfig:
    """Parse evaluation CLI arguments into an ``EvaluationConfig``."""

    parser = argparse.ArgumentParser(description="Run global attack-vs-normal evaluation for the unified MoE model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PATHS.default_checkpoint,
        help="Path to unified MoE checkpoint (default: gateway/unified_moe.pt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Requested dataset split to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Deterministic split seed.")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size.")
    default_threads = max(1, min(2, os.cpu_count() or 1))
    parser.add_argument("--num-threads", type=int, default=default_threads, help="Torch CPU thread count.")
    parser.add_argument(
        "--use-cache",
        type=str,
        default="auto",
        choices=("auto", "on", "off"),
        help="Cache policy: auto tries cache first, on requires cache, off disables cache.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache root directory (default: gateway/cache).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.gateway_root / "eval",
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional fixed temperature. If omitted, fit on validation split.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fixed attack threshold. If omitted, fit best F1 on validation split.",
    )
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=0,
        help="Cap windows per file (0 disables the cap).",
    )
    parser.add_argument(
        "--max-total-windows",
        type=int,
        default=0,
        help="Global window cap across the selected split (0 disables the cap).",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=0,
        help="Log streaming progress every N windows (0 disables).",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=0.0,
        help="Skip files larger than this size in MB (0 disables).",
    )
    parser.add_argument(
        "--max-packets-per-file",
        type=int,
        default=0,
        help="Packet cap per file when streaming (0 disables).",
    )
    parser.add_argument(
        "--max-packets-per-window",
        type=int,
        default=0,
        help="Packet cap per window when streaming (0 disables).",
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=0.0,
        help="Streaming timeout per file in seconds (0 disables).",
    )
    args = parser.parse_args(argv)
    return EvaluationConfig.from_args(args)


__all__ = ["EvaluationConfig", "parse_args"]
