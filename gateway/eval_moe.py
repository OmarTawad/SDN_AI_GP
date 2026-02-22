"""CLI entrypoint for unified MoE global evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gateway.evaluation.configuration import parse_args


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and execute evaluation."""

    config = parse_args(argv)
    from gateway.evaluation.runner import run

    run(config)


if __name__ == "__main__":
    main()
