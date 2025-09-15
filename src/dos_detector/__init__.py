"""DoS detector package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dos-detector")
except PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"

__all__ = ["__version__"]
