"""Feature extraction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..config import Config
from ..config.types import LabelsConfig
from ..utils.io import ensure_dir, save_dataframe, save_json
from ..utils.logging import get_logger
from .labels import AttackInterval, label_windows, load_attack_intervals
from .pcap_reader import PCAPMetadata, read_pcap, summarize_packets
from .windowing import WindowBuilder, WindowingParams
from ..features.feature_engineering import FeatureExtractor


class FeaturePipeline:
    """End-to-end feature extraction for PCAP files."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.window_builder = WindowBuilder(
            WindowingParams(
                window_size=config.windowing.window_size,
                hop_size=config.windowing.hop_size,
                max_windows=config.windowing.max_windows,
            )
        )
        self.extractor = FeatureExtractor(config.feature, config.windowing.window_size)
        self.logger = get_logger(__name__)
        self.intervals = self._load_intervals(config.labels)

    def _load_intervals(self, labels_config: LabelsConfig) -> Dict[str, List[AttackInterval]]:
        if labels_config.intervals_csv is None:
            return {}
        if not labels_config.intervals_csv.exists():
            self.logger.warning("Interval file missing: %s", labels_config.intervals_csv)
            return {}
        return load_attack_intervals(labels_config.intervals_csv, labels_config)

    def process_files(self, pcap_paths: Iterable[Path], output_dir: Path) -> List[PCAPMetadata]:
        ensure_dir(output_dir)
        metadata: List[PCAPMetadata] = []
        feature_columns: Optional[List[str]] = None
        manifest_entries: List[Dict[str, object]] = []

        for path in pcap_paths:
            frame, meta = self._process_single(path)
            metadata.append(meta)
            if frame.empty:
                continue
            feature_path = output_dir / f"{path.stem}.parquet"
            save_dataframe(feature_path, frame)
            manifest_entries.append(
                {
                    "pcap": path.name,
                    "rows": len(frame),
                    "attack_windows": int(frame["attack"].sum()),
                    "file_label": int(frame["attack"].any()),
                }
            )
            if feature_columns is None:
                feature_columns = [
                    column
                    for column in frame.columns
                    if column
                    not in {
                        "pcap",
                        "window_index",
                        "window_start",
                        "window_end",
                        "attack",
                        "family",
                        "family_index",
                    }
                ]
        manifest = {
            "feature_columns": feature_columns or [],
            "frames": manifest_entries,
            "metadata_columns": ["pcap", "window_index", "window_start", "window_end"],
            "label_columns": ["attack", "family", "family_index"],
        }
        save_json(self.config.paths.manifest_path, manifest)
        return metadata

    def _process_single(self, path: Path) -> tuple[pd.DataFrame, PCAPMetadata]:
        packets = read_pcap(path)
        meta = summarize_packets(packets, path)
        windows = self.window_builder.build(packets)
        features = self.extractor.extract(windows)
        intervals = self.intervals.get(path.name, [])
        if intervals and meta.start_time > 1e6:
            adjusted: List[AttackInterval] = []
            for interval in intervals:
                if interval.start < 1e6 and interval.end < 1e6:
                    adjusted.append(
                        AttackInterval(
                            start=interval.start + meta.start_time,
                            end=interval.end + meta.start_time,
                            family=interval.family,
                        )
                    )
                else:
                    adjusted.append(interval)
            intervals = adjusted
        window_labels = label_windows(windows, intervals, self.config.labels)
        attack = [label.attack for label in window_labels]
        families = [label.family for label in window_labels]
        family_indices = [
            self.config.labels.family_mapping.get(fam, self.config.labels.family_mapping.get("other", 0))
            for fam in families
        ]
        features["pcap"] = path.name
        features["attack"] = attack
        features["family"] = families
        features["family_index"] = family_indices
        return features, meta

    def process_single(self, path: Path) -> tuple[pd.DataFrame, PCAPMetadata]:
        """Public wrapper to process a single PCAP."""

        return self._process_single(path)


__all__ = ["FeaturePipeline"]
