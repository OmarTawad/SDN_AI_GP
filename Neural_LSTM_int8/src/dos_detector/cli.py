#src/dos_detector/cli.py
"""Typer CLI for the DoS detector."""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import glob
import json
from pathlib import Path
from typing import Optional

import typer

from .config import load_config
from .data.processor import FeaturePipeline
from .inference.pipeline import InferencePipeline
from .training.supervised_trainer import SupervisedTrainer
from .utils.io import ensure_dir

app = typer.Typer(add_completion=False)


def _format_summary(report: dict, name: str) -> str:
    """Return a compact CLI summary for a single inference run."""

    return " ".join(
        [
            f"[{name}]",
            f"decision={report.get('final_decision', 'n/a')}",
            f"max_prob={report.get('max_prob', 0.0):.6f}",
            f"num_attack_windows={report.get('num_attack_windows', 0)}",
            f"top_mac={report.get('top_mac') or 'n/a'}",
            f"top_ip={report.get('top_ip') or 'n/a'}",
        ]
    )


def _resolve_pcaps(pattern: str) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise typer.BadParameter(f"No PCAPs matched pattern: {pattern}")
    return paths


@app.command()
def extract_features(
    pcaps: str = typer.Argument(..., help="Glob pattern for PCAP files"),
    out: Optional[Path] = typer.Option(None, help="Output directory for processed features"),
    limit: int = typer.Option(0, help="Limit number of windows to process per file"),
    config_path: Path = typer.Option(Path("configs/config.yaml"), help="Path to configuration file"),
) -> None:
    """Extract features from PCAP files and persist them to Parquet."""

    config = load_config(config_path)
    pipeline = FeaturePipeline(config)
    target_dir = out or config.paths.processed_dir
    ensure_dir(target_dir)
    paths = _resolve_pcaps(pcaps)
    paths = _resolve_pcaps(pcaps)
    pipeline.process_files(paths, target_dir, limit=limit)
    typer.echo(f"Processed {len(paths)} PCAPs → {target_dir}")


@app.command("train-supervised")
def train_supervised(config_path: Path = typer.Option(Path("configs/config.yaml"), help="Configuration path")) -> None:
    """Train the supervised sequence detector."""

    config = load_config(config_path)
    trainer = SupervisedTrainer(config)
    metrics = trainer.train()
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def infer(
    pcap: Path = typer.Argument(..., help="Path to PCAP file"),
    out: Path = typer.Option(Path("reports/prediction.json"), help="Output JSON path"),
    config_path: Path = typer.Option(Path("configs/config.yaml"), help="Configuration path"),
    quantized: Optional[bool] = typer.Option(
        None,
        "--quantized/--no-quantized",
        help="Override config quantization flag for int8 inference",
    ),
    quantized_checkpoint: Optional[Path] = typer.Option(None, help="Optional int8 checkpoint override"),
) -> None:
    """Run inference on a single PCAP."""

    config = load_config(config_path)
    if quantized is not None:
        config.quantization.enabled = quantized
    if quantized_checkpoint is not None:
        config.quantization.checkpoint_path = quantized_checkpoint
    pipeline = InferencePipeline(config)
    report = pipeline.infer(pcap)
    ensure_dir(out.parent)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"Wrote inference report → {out}")
    typer.echo(_format_summary(report, pcap.name))


@app.command("batch-infer")
def batch_infer(
    pcaps: str = typer.Argument(..., help="Glob pattern for PCAP files"),
    out_dir: Path = typer.Option(Path("reports"), help="Directory for JSON reports"),
    config_path: Path = typer.Option(Path("configs/config.yaml"), help="Configuration path"),
    quantized: Optional[bool] = typer.Option(
        None,
        "--quantized/--no-quantized",
        help="Override config quantization flag for int8 inference",
    ),
    quantized_checkpoint: Optional[Path] = typer.Option(None, help="Optional int8 checkpoint override"),
) -> None:
    """Run inference on multiple PCAPs."""

    config = load_config(config_path)
    if quantized is not None:
        config.quantization.enabled = quantized
    if quantized_checkpoint is not None:
        config.quantization.checkpoint_path = quantized_checkpoint
    pipeline = InferencePipeline(config)
    ensure_dir(out_dir)
    for path in _resolve_pcaps(pcaps):
        report = pipeline.infer(path)
        target = out_dir / f"{path.stem}.json"
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"→ {target}")
        typer.echo(_format_summary(report, path.name))


if __name__ == "__main__":
    app()
