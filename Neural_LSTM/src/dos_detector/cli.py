#src/dos_detector/cli.py
"""Typer CLI for the DoS detector."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Optional

import typer
from torch.utils.data import DataLoader

from .config import load_config
from .data.dataset import SequenceDataset, collate_fn
from .data.processor import FeaturePipeline
from .fusion import FusionSample, ScoreFusion
from .inference.pipeline import InferencePipeline
from .training.autoencoder_trainer import AutoencoderTrainer
from .training.supervised_trainer import SupervisedTrainer
from .utils.io import ensure_dir, load_dataframe

app = typer.Typer(add_completion=False)


def _resolve_pcaps(pattern: str) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise typer.BadParameter(f"No PCAPs matched pattern: {pattern}")
    return paths


@app.command()
def extract_features(
    pcaps: str = typer.Argument(..., help="Glob pattern for PCAP files"),
    out: Optional[Path] = typer.Option(None, help="Output directory for processed features"),
    config_path: Path = typer.Option(Path("configs/config.yaml"), help="Path to configuration file"),
) -> None:
    """Extract features from PCAP files and persist them to Parquet."""

    config = load_config(config_path)
    pipeline = FeaturePipeline(config)
    target_dir = out or config.paths.processed_dir
    ensure_dir(target_dir)
    paths = _resolve_pcaps(pcaps)
    pipeline.process_files(paths, target_dir)
    typer.echo(f"Processed {len(paths)} PCAPs → {target_dir}")


@app.command("train-supervised")
def train_supervised(config_path: Path = typer.Option(Path("configs/config.yaml"), help="Configuration path")) -> None:
    """Train the supervised sequence detector."""

    config = load_config(config_path)
    trainer = SupervisedTrainer(config)
    metrics = trainer.train()
    typer.echo(json.dumps(metrics, indent=2))


@app.command("train-ae")
def train_autoencoder(config_path: Path = typer.Option(Path("configs/config.yaml"), help="Configuration path")) -> None:
    """Train the sequence autoencoder."""

    config = load_config(config_path)
    trainer = AutoencoderTrainer(config)
    metrics = trainer.train()
    typer.echo(json.dumps(metrics, indent=2))


@app.command("calibrate-fusion")
def calibrate_fusion(config_path: Path = typer.Option(Path("configs/config.yaml"), help="Configuration path")) -> None:
    """Calibrate the logistic fusion layer on validation data."""

    config = load_config(config_path)
    pipeline = InferencePipeline(config)
    val_files = config.data.val_files or [entry["pcap"] for entry in pipeline.manifest.get("frames", [])]
    samples: list[FusionSample] = []
    limit = config.fusion.validation_limit
    for name in val_files:
        path = config.paths.processed_dir / f"{Path(name).stem}.parquet"
        if not path.exists():
            typer.echo(f"Skipping missing processed file: {path}")
            continue
        frame = load_dataframe(path)
        features_supervised = frame.copy()
        features_autoencoder = frame.copy()
        features_supervised[pipeline.feature_columns] = pipeline.scaler.transform(frame[pipeline.feature_columns])
        features_autoencoder[pipeline.feature_columns] = pipeline.ae_scaler.transform(frame[pipeline.feature_columns])
        sup_dataset = SequenceDataset([features_supervised], pipeline.feature_columns, config.labels.family_mapping, config.windowing)
        ae_dataset = SequenceDataset([features_autoencoder], pipeline.feature_columns, config.labels.family_mapping, config.windowing)
        sup_loader = DataLoader(sup_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        ae_loader = DataLoader(ae_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        window_store = pipeline._run_supervised(frame, sup_loader)
        pipeline._run_autoencoder(ae_loader, window_store)
        results = pipeline._assemble_results(frame, window_store)
        for entry in results:
            row = frame.loc[frame["window_index"] == entry["index"]]
            label = int(row["attack"].iloc[0]) if not row.empty else 0
            normalized = max(0.0, (entry["ae_error"] - pipeline.ae_baseline.get("ae_error_mean", 0.0)) / max(pipeline.ae_baseline.get("ae_error_std", 1.0), 1e-6))
            sample_features = [entry["supervised_prob"], normalized, float(entry["features"].get("ssdp_share", 0.0))]
            samples.append(FusionSample(features=sample_features, label=label))
            if limit is not None and len(samples) >= limit:
                break
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise typer.BadParameter("No validation samples available for fusion calibration.")
    fusion = ScoreFusion(config.fusion.feature_names)
    fusion.fit(samples)
    fusion.save(config.paths.fusion_model_path)
    typer.echo(f"Saved fusion calibrator → {config.paths.fusion_model_path}")


@app.command()
def infer(
    pcap: Path = typer.Argument(..., help="Path to PCAP file"),
    out: Path = typer.Option(Path("reports/prediction.json"), help="Output JSON path"),
    config_path: Path = typer.Option(Path("configs/config.yaml"), help="Configuration path"),
) -> None:
    """Run inference on a single PCAP."""

    config = load_config(config_path)
    pipeline = InferencePipeline(config)
    report = pipeline.infer(pcap)
    ensure_dir(out.parent)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"Wrote inference report → {out}")


@app.command("batch-infer")
def batch_infer(
    pcaps: str = typer.Argument(..., help="Glob pattern for PCAP files"),
    out_dir: Path = typer.Option(Path("reports"), help="Directory for JSON reports"),
    config_path: Path = typer.Option(Path("configs/config.yaml"), help="Configuration path"),
) -> None:
    """Run inference on multiple PCAPs."""

    config = load_config(config_path)
    pipeline = InferencePipeline(config)
    ensure_dir(out_dir)
    for path in _resolve_pcaps(pcaps):
        report = pipeline.infer(path)
        target = out_dir / f"{path.stem}.json"
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"→ {target}")


if __name__ == "__main__":
    app()