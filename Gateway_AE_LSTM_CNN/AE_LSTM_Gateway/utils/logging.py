"""Logging utilities for the gateway package.

----
"""

from __future__ import annotations

import logging
from typing import Optional

LOGGER_NAME = "gateway"


def configure_logger(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a module-level logger.

    Args:
        level: Logging level applied to the root gateway logger.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Fetch a child logger under the gateway namespace.

    Args:
        name: Optional child logger suffix.

    Returns:
        logging.Logger: Child logger bound to the gateway namespace.
    """

    base_logger = configure_logger()
    if not name:
        return base_logger
    return base_logger.getChild(name)


__all__ = ["configure_logger", "get_logger", "LOGGER_NAME"]

