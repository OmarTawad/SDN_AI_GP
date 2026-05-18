"""Inference configuration and CLI parsing utilities.


"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from gateway.core import PATHS


@dataclass(frozen=True)
class InferenceArgs:
    """Arguments controlling the inference workflow."""

    pcap: Path
    checkpoint: Path
    batch_size: int
    max_windows: int | None
    num_threads: int
    gating_hidden: int | None
    attack_threshold: float
    attack_prob_threshold: float
    high_confidence_threshold: float
    min_high_confidence_windows: int
    min_attack_windows: int
    log_windows: bool


def parse_args(argv: Sequence[str] | None = None) -> InferenceArgs:
    """Parse CLI arguments into an :class:`InferenceArgs` dataclass.

    Args:
        argv: Optional sequence of arguments overriding ``sys.argv``.

    Returns:
        InferenceArgs: Populated inference configuration.
    """

    parser = argparse.ArgumentParser(description="Run unified MoE inference over a PCAP file.")
    parser.add_argument("pcap", type=Path, help="Path to the PCAP file to evaluate.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PATHS.default_checkpoint,
        help="Path to a trained unified MoE checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Number of windows per batch.")
    parser.add_argument("--max-windows", type=int, default=0, help="Optional cap on processed windows.")
    parser.add_argument("--num-threads", type=int, default=1, help="Torch thread pool size.")
    parser.add_argument("--gating-hidden", type=int, default=None, help="Override gating hidden size.")
    parser.add_argument(
        "--attack-threshold",
        type=float,
        default=0.25,
        help="Attack fraction threshold (higher values demand more consensus).",
    )
    parser.add_argument(
        "--attack-prob-threshold",
        type=float,
        default=0.40,
        help="Attack probability threshold.",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.85,
        help="High confidence threshold.",
    )
    parser.add_argument(
        "--min-high-confidence-windows",
        type=int,
        default=25,
        help="Minimum high-confidence windows for promotion.",
    )
    parser.add_argument(
        "--min-attack-windows",
        type=int,
        default=50,
        help="Minimum attack windows for promotion.",
    )
    parser.add_argument(
        "--log-windows",
        action="store_true",
        help="Print one attack/normal log line per processed window.",
    )
    namespace = parser.parse_args(argv)
    max_windows = namespace.max_windows if namespace.max_windows > 0 else None
    return InferenceArgs(
        pcap=namespace.pcap.resolve(),
        checkpoint=namespace.checkpoint.resolve(),
        batch_size=int(namespace.batch_size),
        max_windows=max_windows,
        num_threads=int(namespace.num_threads),
        gating_hidden=namespace.gating_hidden,
        attack_threshold=float(namespace.attack_threshold),
        attack_prob_threshold=float(namespace.attack_prob_threshold),
        high_confidence_threshold=float(namespace.high_confidence_threshold),
        min_high_confidence_windows=int(namespace.min_high_confidence_windows),
        min_attack_windows=int(namespace.min_attack_windows),
        log_windows=bool(namespace.log_windows),
    )


__all__ = ["InferenceArgs", "parse_args"]
