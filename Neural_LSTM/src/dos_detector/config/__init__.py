from __future__ import annotations
from pathlib import Path
import yaml

from .types import (
    PathsConfig, WindowingConfig, FeatureConfig, LabelsConfig, DataConfig,
    SupervisedTrainingConfig, AutoencoderTrainingConfig, TrainingConfig,
    SupervisedModelConfig, AutoencoderModelConfig, ModelConfig,
    FusionConfig, PlausibilityConfig, PostProcessingConfig,
    ExplainabilityConfig, LiveConfig, Config, resolve_paths
)

def _to_path(p) -> Path:
    return p if isinstance(p, Path) else Path(p)

def load_config(path: Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Paths (coerce to Path objects)
    p = raw["paths"]
    paths = PathsConfig(
        raw_pcap_dir=_to_path(p["raw_pcap_dir"]),
        processed_dir=_to_path(p["processed_dir"]),
        models_dir=_to_path(p["models_dir"]),
        reports_dir=_to_path(p["reports_dir"]),
        scaler_path=_to_path(p["scaler_path"]),
        supervised_model_path=_to_path(p["supervised_model_path"]),
        ae_model_path=_to_path(p["ae_model_path"]),
        ae_scaler_path=_to_path(p["ae_scaler_path"]),
        fusion_model_path=_to_path(p["fusion_model_path"]),
        manifest_path=_to_path(p["manifest_path"]),
        metrics_path=_to_path(p["metrics_path"]),
    )

    # Windowing / Feature
    w = raw["windowing"]; fcfg = raw["feature"]
    windowing = WindowingConfig(**w)
    feature = FeatureConfig(**fcfg)

    # Labels (intervals_csv may be null)
    l = raw["labels"]
    intervals = l.get("intervals_csv")
    labels = LabelsConfig(
        intervals_csv=_to_path(intervals) if intervals is not None else None,
        default_family=l["default_family"],
        family_mapping=l["family_mapping"],
        attack_families=l["attack_families"],
    )

    # Data
    d = raw["data"]
    data = DataConfig(
        train_files=d.get("train_files", []),
        val_files=d.get("val_files", []),
        test_files=d.get("test_files", []),
        normal_families=d.get("normal_families", ["normal"]),
    )

    # Training
    t = raw["training"]
    sup = SupervisedTrainingConfig(**t["supervised"])
    ae = AutoencoderTrainingConfig(**t["autoencoder"])
    training = TrainingConfig(supervised=sup, autoencoder=ae)

    # Model
    m = raw["model"]
    sm = SupervisedModelConfig(**m["supervised"])
    am = AutoencoderModelConfig(**m["autoencoder"])
    model = ModelConfig(supervised=sm, autoencoder=am)

    # Fusion
    fusion = FusionConfig(**raw["fusion"])

    # Post-processing
    pp = raw["postprocessing"]
    pl = pp["plausibility"]
    postprocessing = PostProcessingConfig(
        tau_window=pp["tau_window"],
        tau_file=pp["tau_file"],
        consecutive_required=pp["consecutive_required"],
        min_attack_windows=pp["min_attack_windows"],
        cooldown_windows=pp["cooldown_windows"],
        smoothing=pp["smoothing"],
        plausibility=PlausibilityConfig(**pl),
    )

    # Explainability / Live
    explainability = ExplainabilityConfig(**raw["explainability"])
    live = LiveConfig(**raw["live"])

    cfg = Config(
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

    # resolve any relative paths against the config file's directory
    cfg = resolve_paths(cfg, root=path.parent)
    return cfg
