# src/arp_detector/config/types.py

"""Configuration dataclasses and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class PathsConfig:
    """Filesystem paths used by the pipeline."""

    raw_pcap_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    scaler_path: Path
    supervised_model_path: Path
    manifest_path: Path
    metrics_path: Path


@dataclass
class WindowingConfig:
    """Window and sequence configuration."""

    window_size: float
    hop_size: float
    sequence_length: int
    sequence_stride: int
    max_windows: Optional[int] = None


@dataclass
class FeatureConfig:
    """Feature extraction configuration for ARP spoofing."""

    # Placeholder for future tunables; the extractor currently has no configurable knobs.
    pass


@dataclass
class LabelsConfig:
    """Label configuration."""

    intervals_csv: Optional[Path]
    default_family: str
    family_mapping: Dict[str, int]
    attack_families: Sequence[str]


@dataclass
class DataConfig:
    """Dataset split configuration."""

    train_files: Sequence[str]
    val_files: Sequence[str]
    test_files: Sequence[str]
    normal_families: Sequence[str]


@dataclass
class SupervisedTrainingConfig:
    """Hyperparameters for the supervised training pipeline."""

    batch_size: int
    num_workers: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    bce_pos_weight: float
    early_stopping_patience: int
    precision_mode: str
    max_train_batches: Optional[int]
    max_val_batches: Optional[int]


@dataclass
class SupervisedModelConfig:
    """Architecture parameters for the supervised detector."""

    input_dropout: float
    rnn_type: str
    hidden_size: int
    num_layers: int
    bidirectional: bool
    dropout: float
    attention: bool
    attention_heads: int


@dataclass
class PostProcessingConfig:
    """Decision gating configuration."""

    tau_window: float
    tau_file: float
    min_attack_windows: int


@dataclass
class ExplainabilityConfig:
    """Explainability configuration."""

    shap_background_size: int
    top_features: int
    attention: bool


@dataclass
class LiveConfig:
    """Live sniffing configuration."""

    interface: str
    flush_seconds: float


@dataclass
class Config:
    """Root configuration object."""

    seed: int
    paths: PathsConfig
    windowing: WindowingConfig
    feature: FeatureConfig
    labels: LabelsConfig
    data: DataConfig
    training: "TrainingConfig"
    model: "ModelConfig"
    postprocessing: PostProcessingConfig
    explainability: ExplainabilityConfig
    live: LiveConfig


@dataclass
class TrainingConfig:
    """Grouped training configuration."""

    supervised: SupervisedTrainingConfig


@dataclass
class ModelConfig:
    """Grouped model configuration."""

    supervised: SupervisedModelConfig


def expand_path(base: Path, path: Path) -> Path:
    """Resolve a path relative to a base directory."""

    if path.is_absolute():
        return path
    return (base / path).resolve()


def resolve_paths(config: Config, root: Optional[Path] = None) -> Config:
    """Resolve all filesystem paths relative to a root directory."""

    base = root or Path.cwd()
    config.paths.raw_pcap_dir = expand_path(base, config.paths.raw_pcap_dir)
    config.paths.processed_dir = expand_path(base, config.paths.processed_dir)
    config.paths.models_dir = expand_path(base, config.paths.models_dir)
    config.paths.reports_dir = expand_path(base, config.paths.reports_dir)
    config.paths.scaler_path = expand_path(base, config.paths.scaler_path)
    config.paths.supervised_model_path = expand_path(base, config.paths.supervised_model_path)
    config.paths.manifest_path = expand_path(base, config.paths.manifest_path)
    config.paths.metrics_path = expand_path(base, config.paths.metrics_path)
    if config.labels.intervals_csv is not None:
        config.labels.intervals_csv = expand_path(base, config.labels.intervals_csv)
    return config
