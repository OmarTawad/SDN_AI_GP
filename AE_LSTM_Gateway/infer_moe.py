"""Legacy shim for unified MoE inference."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gateway.inference.configuration import parse_args
from gateway.inference.infer_runner import run


def main() -> None:
    """Invoke the inference runner using CLI arguments."""

    run(parse_args())


if __name__ == "__main__":
    main()
