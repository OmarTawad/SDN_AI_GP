#src/dos_detector/utils/logging.py
"""Structured logging utilities."""

from __future__ import annotations

import logging
<<<<<<< HEAD
from typing import Any, Dict
=======
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7

try:  # pragma: no cover - optional dependency
    import structlog
except ImportError:  # pragma: no cover
    structlog = None  # type: ignore


<<<<<<< HEAD
def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for console-friendly JSON logging."""

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
=======
def configure_logging(level: str = "INFO", log_file: Optional[Path | str] = None) -> None:
    """Configure structlog for console-friendly JSON logging."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_path,
                maxBytes=1_000_000,
                backupCount=3,
                encoding="utf-8",
            )
        )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7
    if structlog is None:
        logging.getLogger(__name__).warning("structlog not available; using stdlib logging")
        return
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a structured logger."""

    if structlog is None:
        return logging.getLogger(name)  # type: ignore[return-value]
    return structlog.get_logger(name)


def log_config(logger: structlog.BoundLogger, config: Dict[str, Any]) -> None:
    """Log a configuration snapshot."""

<<<<<<< HEAD
    logger.info("config", **config)
=======
    logger.info("config", **config)
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7
