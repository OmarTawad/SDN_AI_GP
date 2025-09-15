"""Configuration loader utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .types import (
    AutoencoderModelConfig,
    AutoencoderTrainingConfig,
    Config,
    DataConfig,
    ExplainabilityConfig,
    FeatureConfig,
    FusionConfig,
    LabelsConfig,
    LiveConfig,
    ModelConfig,
    PathsConfig,
    PlausibilityConfig,
    PostProcessingConfig,
    SupervisedModelConfig,
    SupervisedTrainingConfig,
    TrainingConfig,
    WindowingConfig,
    resolve_paths,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(path: Path) -> Config:
    """Load a configuration file and return a :class:`Config`."""

    raw = _load_yaml(path)
    base = path.parent

    path_payload = {key: Path(value) for key, value in raw["paths"].items()}
    paths = PathsConfig(**path_payload)
    windowing = WindowingConfig(**raw["windowing"])  # type: ignore[arg-type]
    feature = FeatureConfig(**raw["feature"])  # type: ignore[arg-type]
    label_payload = raw["labels"].copy()
    if label_payload.get("intervals_csv") is not None:
        label_payload["intervals_csv"] = Path(label_payload["intervals_csv"])
    labels = LabelsConfig(**label_payload)  # type: ignore[arg-type]
    data = DataConfig(**raw["data"])  # type: ignore[arg-type]
    training = TrainingConfig(
        supervised=SupervisedTrainingConfig(**raw["training"]["supervised"]),
        autoencoder=AutoencoderTrainingConfig(**raw["training"]["autoencoder"]),
    )
    model = ModelConfig(
        supervised=SupervisedModelConfig(**raw["model"]["supervised"]),
        autoencoder=AutoencoderModelConfig(**raw["model"]["autoencoder"]),
    )
    fusion = FusionConfig(**raw["fusion"])  # type: ignore[arg-type]
    plausibility = PlausibilityConfig(**raw["postprocessing"]["plausibility"])
    postprocessing = PostProcessingConfig(
        plausibility=plausibility,
        **{k: v for k, v in raw["postprocessing"].items() if k != "plausibility"},
    )
    explainability = ExplainabilityConfig(**raw["explainability"])  # type: ignore[arg-type]
    live = LiveConfig(**raw["live"])  # type: ignore[arg-type]

    config = Config(
        seed=raw["seed"],
        paths=paths,
        windowing=windowing,
        feature=feature,
        labels=labels,
        data=data,
        training=training,
        model=model,
        fusion=fusion,
        postprocessing=postprocessing,
        explainability=explainability,
        live=live,
    )
    return resolve_paths(config, base)


__all__ = ["load_config"]
