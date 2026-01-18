"""Core configuration objects and constants for the gateway package.


"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    """Filesystem locations required by the gateway package.

    Attributes:
        project_root: Root directory of the monorepo.
        gateway_root: Root path of the gateway package.
        cache_root: Directory that stores cached tensors.
        samples_root: Directory containing sample PCAP captures.
        default_checkpoint: Default unified MoE checkpoint path.
    """

    project_root: Path
    gateway_root: Path
    cache_root: Path
    samples_root: Path
    default_checkpoint: Path


def build_path_config() -> PathConfig:
    """Construct a :class:`PathConfig` describing important filesystem paths.

    Returns:
        PathConfig: Immutable container with resolved paths.
    """

    gateway_root = Path(__file__).resolve().parents[1]
    project_root = gateway_root.parent
    cache_root = gateway_root / "cache"
    samples_root = project_root / "samples"
    default_checkpoint = gateway_root / "unified_moe.pt"
    return PathConfig(
        project_root=project_root,
        gateway_root=gateway_root,
        cache_root=cache_root,
        samples_root=samples_root,
        default_checkpoint=default_checkpoint,
    )


PATHS = build_path_config()

CLASS_LABELS: tuple[str, ...] = ("normal", "dos", "arp")

__all__ = ["PATHS", "PathConfig", "CLASS_LABELS", "build_path_config"]
