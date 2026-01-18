"""Training subpackage exposing CLI entrypoints."""

from __future__ import annotations

from .configuration import TrainingConfig, parse_args
from .runner import main, run_training

__all__ = ["TrainingConfig", "main", "parse_args", "run_training"]

