"""Unified MoE global evaluation package."""

from __future__ import annotations

__all__ = ["EvaluationConfig", "parse_args", "run"]


def __getattr__(name: str):
    if name in {"EvaluationConfig", "parse_args"}:
        from .configuration import EvaluationConfig, parse_args

        return {"EvaluationConfig": EvaluationConfig, "parse_args": parse_args}[name]
    if name == "run":
        from .runner import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
