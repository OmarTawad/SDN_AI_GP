#src/arp_detector/utils/logging.py
"""Structured logging utilities."""

from __future__ import annotations

import logging
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import structlog
except ImportError:  # pragma: no cover
    structlog = None  # type: ignore


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for console-friendly JSON logging."""

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
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

    logger.info("config", **config)