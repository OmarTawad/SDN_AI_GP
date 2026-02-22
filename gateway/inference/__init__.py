"""Inference subpackage initialisation.


"""

from __future__ import annotations

__all__ = ["InferenceArgs", "main", "parse_args", "run"]


def __getattr__(name: str):
    if name in {"InferenceArgs", "parse_args"}:
        from .configuration import InferenceArgs, parse_args

        return {"InferenceArgs": InferenceArgs, "parse_args": parse_args}[name]
    if name in {"main", "run"}:
        from .infer_runner import main, run

        return {"main": main, "run": run}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
