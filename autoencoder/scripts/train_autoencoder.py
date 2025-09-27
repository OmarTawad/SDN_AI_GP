#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dae.cli import run_train
from dae.config import load_config
from dae.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the autoencoder on windowed features")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--windows", required=True, help="Parquet file containing training windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(Path(args.config))
    artifacts = run_train(config, Path(args.windows))
    logger = get_logger("train_script")
    logger.info("training_artifacts", model=str(artifacts.model_path), scaler=str(artifacts.scaler_path))


if __name__ == "__main__":
    main()
