from __future__ import annotations
from pathlib import Path
import yaml

from .types import (
    PathsConfig, WindowingConfig, FeatureConfig, LabelsConfig, DataConfig,
    SupervisedTrainingConfig, TrainingConfig,
    SupervisedModelConfig, ModelConfig,
    PostProcessingConfig,
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
        manifest_path=_to_path(p["manifest_path"]),
        metrics_path=_to_path(p["metrics_path"]),
    )

    # Windowing / Feature
    w = raw["windowing"]; fcfg = raw.get("feature") or {}
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
    training = TrainingConfig(supervised=sup)

    # Model
    m = raw["model"]
    sm = SupervisedModelConfig(**m["supervised"])
    model = ModelConfig(supervised=sm)

    # Post-processing
    pp = raw["postprocessing"]
    postprocessing = PostProcessingConfig(
        tau_window=pp["tau_window"],
        tau_file=pp["tau_file"],
        min_attack_windows=pp.get("min_attack_windows", 1),
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
        postprocessing=postprocessing,
        explainability=explainability,
        live=live,
    )

    # resolve any relative paths against the config file's directory
    cfg = resolve_paths(cfg, root=path.parent)
    return cfg
