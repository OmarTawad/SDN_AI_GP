#src/arp_detector/__init__.py
"""ARP spoofing detector package."""

import os
# Force hide GPU to stop PyTorch warnings and ensure CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("arp-detector")
except PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"

__all__ = ["__version__"]
