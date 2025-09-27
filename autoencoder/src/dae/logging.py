from __future__ import annotations

import logging
from typing import Optional

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    structlog.configure(
        processors=[
            timestamper,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=level)


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    if not structlog.is_configured():
        configure_logging()
    return structlog.get_logger(name)
