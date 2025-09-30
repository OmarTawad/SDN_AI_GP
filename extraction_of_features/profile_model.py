#!/usr/bin/env python3
"""Profile a PyTorch model and export human- and machine-readable reports."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a PyTorch model, profile it, and write detailed reports."
    )
    parser.add_argument("--model-path", required=True, help="Path to the Python file with the model.")
    parser.add_argument("--class-name", required=True, help="Class name of the model to instantiate.")
    parser.add_argument(
        "--input-shape",
        required=True,
        help="Comma-separated shape for a dummy input tensor, e.g. '1,3,224,224'.",
    )
    parser.add_argument(
        "--out-file",
        required=True,
        help="Destination text file for the human-readable report (JSON written alongside).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to profile on (default: cpu).",
    )
    parser.add_argument(
        "--init-args",
        default=None,
        help="Optional Python literal list/tuple with positional args for model init.",
    )
    parser.add_argument(
        "--init-kwargs",
        default=None,
        help="Optional Python literal dict with keyword args for model init.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Show the full traceback when an error occurs.",
    )
    return parser.parse_args()


def ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Model file '{path}' does not exist.")
    if not path.is_file():
        raise FileNotFoundError(f"Model path '{path}' is not a file.")


def parse_shapes(shape_str: str) -> List[Tuple[int, ...]]:
    groups = [chunk.strip() for chunk in shape_str.split(";") if chunk.strip()]
    if not groups:
        raise ValueError("--input-shape must specify at least one shape.")
    shapes: List[Tuple[int, ...]] = []
    for group in groups:
        pieces = [part.strip() for part in group.split(",") if part.strip()]
        if not pieces:
            raise ValueError("Each shape must contain at least one dimension.")
        dims: List[int] = []
        for part in pieces:
            try:
                value = int(part)
            except ValueError as exc:  # pragma: no cover - argument validation
                raise ValueError(f"Invalid dimension '{part}' in --input-shape.") from exc
            if value <= 0:
                raise ValueError("All dimensions in --input-shape must be positive integers.")
            dims.append(value)
        shapes.append(tuple(dims))
    return shapes


def parse_literal(name: str, raw: str | None, expected_types: Tuple[type, ...]) -> Any:
    if raw is None:
        return [] if list in expected_types else {}
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:  # pragma: no cover - argument validation
        raise ValueError(f"--{name} must be a valid Python literal.") from exc
    if not isinstance(value, expected_types):
        expected_names = ", ".join(t.__name__ for t in expected_types)
        raise ValueError(f"--{name} must evaluate to one of: {expected_names}.")
    return value


def import_module_from_path(path: Path) -> ModuleType:
    module_name = f"profile_target_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from '{path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_model_class(module: ModuleType, class_name: str) -> type:
    candidate = getattr(module, class_name, None)
    if candidate is None:
        available = ", ".join(attr for attr in dir(module) if not attr.startswith("_"))
        raise AttributeError(
            f"Class '{class_name}' not found in module. Available attributes: {available}"
        )
    if not isinstance(candidate, type) or not issubclass(candidate, torch.nn.Module):
        raise TypeError(f"Attribute '{class_name}' is not a torch.nn.Module subclass.")
    return candidate


def instantiate_model(model_cls: type, args: Iterable[Any], kwargs: Dict[str, Any]) -> torch.nn.Module:
    try:
        model = model_cls(*args, **kwargs)
    except TypeError as exc:
        raise TypeError(
            "Failed to instantiate model. Adjust --init-args/--init-kwargs to match the constructor."
        ) from exc
    if not isinstance(model, torch.nn.Module):  # pragma: no cover - safety
        raise TypeError("Instantiated object is not a torch.nn.Module.")
    return model


def load_dependencies():
    missing = []
    try:
        from torchinfo import summary
    except ModuleNotFoundError:
        missing.append("torchinfo")
        summary = None  # type: ignore[assignment]
    try:
        from fvcore.nn import FlopCountAnalysis
    except ModuleNotFoundError:
        missing.append("fvcore")
        FlopCountAnalysis = None  # type: ignore[assignment]
    if missing:
        raise ModuleNotFoundError(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Install them via 'pip install torchinfo fvcore'."
        )
    return summary, FlopCountAnalysis


def human_readable_count(value: float) -> str:
    suffixes = ["", "K", "M", "G", "T", "P"]
    magnitude = 0
    abs_val = float(abs(value))
    while abs_val >= 1000 and magnitude < len(suffixes) - 1:
        abs_val /= 1000.0
        magnitude += 1
    formatted = f"{abs_val:,.2f}" if magnitude else f"{int(value):,}"
    return f"{formatted} {suffixes[magnitude]}".strip()


def bytes_to_mb(value: int) -> float:
    return value / 1_000_000 if value else 0.0


def profile_with_torchinfo(
    summary_fn,
    model: torch.nn.Module,
    dummy_inputs: Iterable[Any] | Any,
) -> Any:
    # torchinfo consumes concrete input examples to capture shapes and MACs.
    if isinstance(dummy_inputs, tuple):
        input_data = dummy_inputs
    elif isinstance(dummy_inputs, list):
        input_data = tuple(dummy_inputs)
    else:
        input_data = (dummy_inputs,)
    return summary_fn(
        model,
        input_data=input_data,
        mode="eval",
        verbose=0,
        depth=10,
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds", "trainable"),
        row_settings=("var_names", "depth"),
    )


def profile_flops(
    flop_count_cls,
    model: torch.nn.Module,
    dummy_inputs: Iterable[Any] | Any,
) -> Dict[str, Any]:
    logging.getLogger("fvcore.nn.jit_handles").setLevel(logging.ERROR)
    logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)
    try:
        buffer = io.StringIO()
        with torch.no_grad(), contextlib.redirect_stdout(buffer):
            if isinstance(dummy_inputs, tuple):
                input_args = dummy_inputs
            elif isinstance(dummy_inputs, list):
                input_args = tuple(dummy_inputs)
            else:
                input_args = (dummy_inputs,)
            analyzer = flop_count_cls(model, input_args)
            total = int(analyzer.total())
            per_module = analyzer.by_module()
            per_operator = analyzer.by_operator()
    except Exception as exc:  # pragma: no cover - external dependency failures
        warnings = [line.strip() for line in buffer.getvalue().splitlines() if line.strip()] if "buffer" in locals() else []
        return {
            "total_flops": None,
            "per_module": {},
            "per_operator": {},
            "warnings": warnings,
            "error": str(exc),
        }
    warnings = [line.strip() for line in buffer.getvalue().splitlines() if line.strip()]
    return {
        "total_flops": total,
        "per_module": {name or "<model>": int(value) for name, value in per_module.items()},
        "per_operator": {name: int(value) for name, value in per_operator.items()},
        "warnings": warnings,
        "error": None,
    }


def build_layer_summaries(summary_obj: Any) -> List[Dict[str, Any]]:
    layers: List[Dict[str, Any]] = []
    for layer in summary_obj.summary_list:
        layers.append(
            {
                "name": layer.var_name or "<root>",
                "type": layer.class_name,
                "depth": layer.depth,
                "is_leaf": layer.is_leaf_layer,
                "input_shape": layer.input_size,
                "output_shape": layer.output_size,
                "kernel_size": layer.kernel_size,
                "num_params": layer.num_params,
                "trainable_params": layer.trainable_params,
                "macs": layer.macs,
                "param_bytes": layer.param_bytes,
                "output_bytes": layer.output_bytes,
            }
        )
    return layers


def build_text_report(
    summary_obj: Any,
    summary_str: str,
    flops: Dict[str, Any],
    param_stats: Dict[str, int],
    memory_stats: Dict[str, float],
) -> str:
    lines: List[str] = []
    lines.append("Model Profiling Report")
    lines.append("=" * 80)
    lines.append("")
    lines.append("TORCHINFO SUMMARY")
    lines.append("-" * 80)
    lines.append(summary_str)
    lines.append("")
    lines.append("FLOPs (fvcore)")
    lines.append("-" * 80)
    if flops.get("total_flops") is None:
        lines.append("Total FLOPs: unavailable (" + flops.get("error", "unknown error") + ")")
    else:
        total_flops = flops["total_flops"]
        lines.append(
            f"Total FLOPs: {total_flops:,} ({human_readable_count(total_flops)})"
        )
        lines.append("Per-module FLOPs:")
        for name, value in flops["per_module"].items():
            lines.append(f"  - {name or '<model>'}: {value:,}")
    if flops.get("warnings"):
        lines.append("Warnings:")
        for message in flops["warnings"]:
            lines.append(f"  * {message}")
    lines.append("")
    lines.append("Parameters")
    lines.append("-" * 80)
    lines.append(f"Total params: {param_stats['total']:,}")
    lines.append(f"Trainable params: {param_stats['trainable']:,}")
    lines.append(f"Non-trainable params: {param_stats['non_trainable']:,}")
    lines.append("")
    lines.append("Memory Footprint (approximate)")
    lines.append("-" * 80)
    lines.append(f"Input size (MB): {memory_stats['input_mb']:.2f}")
    lines.append(f"Forward/backward size (MB): {memory_stats['fwd_bwd_mb']:.2f}")
    lines.append(f"Params size (MB): {memory_stats['param_mb']:.2f}")
    lines.append(f"Estimated total size (MB): {memory_stats['total_mb']:.2f}")
    return "\n".join(lines)


def write_reports(text_path: Path, text_content: str, json_content: Dict[str, Any]) -> Path:
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text_content)
    json_path = text_path.with_suffix(".json") if text_path.suffix else text_path.with_name(text_path.name + ".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_content, indent=2))
    return json_path


def main() -> int:
    args = parse_args()
    try:
        model_path = Path(args.model_path).resolve()
        ensure_file_exists(model_path)
        input_shapes = parse_shapes(args.input_shape)
        positional_args = parse_literal("init-args", args.init_args, (list, tuple))
        keyword_args = parse_literal("init-kwargs", args.init_kwargs, (dict,))
        module = import_module_from_path(model_path)
        model_cls = resolve_model_class(module, args.class_name)
        model = instantiate_model(model_cls, positional_args, keyword_args)
        device = torch.device(args.device)
        model.to(device)
        model.eval()
        dummy_tensors = [torch.randn(*shape, device=device) for shape in input_shapes]
        if len(dummy_tensors) == 1:
            dummy_input = dummy_tensors[0]
            dummy_input_args: Tuple[Any, ...] = (dummy_tensors[0],)
            dummy_dtype: Any = str(dummy_tensors[0].dtype)
        else:
            dummy_input = tuple(dummy_tensors)
            dummy_input_args = tuple(dummy_tensors)
            dummy_dtype = [str(t.dtype) for t in dummy_tensors]
        summary_fn, flop_counter_cls = load_dependencies()
        with torch.no_grad():
            summary_obj = profile_with_torchinfo(summary_fn, model, dummy_input_args)
        summary_str = str(summary_obj)
        flops = profile_flops(flop_counter_cls, model, dummy_input_args)
        total_params = int(summary_obj.total_params)
        trainable_params = int(summary_obj.trainable_params)
        non_trainable = total_params - trainable_params
        memory_stats = {
            "input_bytes": int(summary_obj.total_input),
            "fwd_bwd_bytes": int(summary_obj.total_output_bytes),
            "param_bytes": int(summary_obj.total_param_bytes),
        }
        memory_stats["total_bytes"] = (
            memory_stats["input_bytes"]
            + memory_stats["fwd_bwd_bytes"]
            + memory_stats["param_bytes"]
        )
        memory_stats.update(
            {
                "input_mb": bytes_to_mb(memory_stats["input_bytes"]),
                "fwd_bwd_mb": bytes_to_mb(memory_stats["fwd_bwd_bytes"]),
                "param_mb": bytes_to_mb(memory_stats["param_bytes"]),
                "total_mb": bytes_to_mb(memory_stats["total_bytes"]),
            }
        )
        param_stats = {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": non_trainable,
        }
        layers = build_layer_summaries(summary_obj)
        json_report: Dict[str, Any] = {
            "model": {
                "file": str(model_path),
                "class_name": args.class_name,
                "device": str(device),
            },
            "input_shapes": [list(shape) for shape in input_shapes],
            "dummy_input_dtype": dummy_dtype,
            "parameter_counts": param_stats,
            "memory_bytes": {
                "input": memory_stats["input_bytes"],
                "forward_backward": memory_stats["fwd_bwd_bytes"],
                "parameters": memory_stats["param_bytes"],
                "total": memory_stats["total_bytes"],
            },
            "memory_mb": {
                "input": memory_stats["input_mb"],
                "forward_backward": memory_stats["fwd_bwd_mb"],
                "parameters": memory_stats["param_mb"],
                "total": memory_stats["total_mb"],
            },
            "torchinfo": {
                "total_mult_adds": int(summary_obj.total_mult_adds),
                "summary_text": summary_str,
                "layers": layers,
            },
            "fvcore": flops,
        }
        text_report = build_text_report(summary_obj, summary_str, flops, param_stats, memory_stats)
        text_path = Path(args.out_file).resolve()
        json_path = write_reports(text_path, text_report, json_report)
        print(f"Report written to {text_path} and {json_path}")
        return 0
    except Exception as exc:  # pragma: no cover - CLI error handling
        if args.traceback:
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
