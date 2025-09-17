#src/dos_detector/data/dataset.py
"""Dataset helpers for training."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config.types import WindowingConfig
from .structures import SequenceSample


def _build_sequence_samples(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    family_mapping: Dict[str, int],
    sequence_length: int,
    sequence_stride: int,
) -> List[SequenceSample]:
    samples: List[SequenceSample] = []
    features = frame[feature_columns].to_numpy(dtype=np.float32)
    binary_labels = frame["attack"].to_numpy(dtype=np.float32)
    fallback = family_mapping.get("other", 0)
    family_labels = frame["family"].map(lambda fam: family_mapping.get(fam, fallback)).to_numpy(dtype=np.int64)
    window_starts = frame["window_start"].to_list()
    for start in range(0, len(frame) - sequence_length + 1, sequence_stride):
        end = start + sequence_length
        window_slice = slice(start, end)
        sample = SequenceSample(
            features=features[window_slice].tolist(),
            binary_labels=binary_labels[window_slice].astype(int).tolist(),
            family_labels=family_labels[window_slice].tolist(),
            metadata={
                "start_index": int(frame["window_index"].iloc[start]),
                "end_index": int(frame["window_index"].iloc[end - 1]),
                "window_start": window_starts[start],
                "window_end": frame["window_end"].iloc[end - 1],
                "pcap": frame["pcap"].iloc[0],
            },
        )
        samples.append(sample)
    if samples:
        last_start = len(frame) - sequence_length
        if last_start >= 0 and (last_start % sequence_stride) != 0:
            start = last_start
            end = start + sequence_length
            window_slice = slice(start, end)
            sample = SequenceSample(
                features=features[window_slice].tolist(),
                binary_labels=binary_labels[window_slice].astype(int).tolist(),
                family_labels=family_labels[window_slice].tolist(),
                metadata={
                    "start_index": int(frame["window_index"].iloc[start]),
                    "end_index": int(frame["window_index"].iloc[end - 1]),
                    "window_start": window_starts[start],
                    "window_end": frame["window_end"].iloc[end - 1],
                    "pcap": frame["pcap"].iloc[0],
                },
            )
            if sample.metadata not in [s.metadata for s in samples]:
                samples.append(sample)
    return samples


class SequenceDataset(Dataset[Dict[str, torch.Tensor]]):
    """PyTorch dataset returning sequence tensors."""

    def __init__(
        self,
        frames: Sequence[pd.DataFrame],
        feature_columns: Sequence[str],
        family_mapping: Dict[str, int],
        windowing: WindowingConfig,
    ) -> None:
        self.feature_columns = list(feature_columns)
        samples: List[SequenceSample] = []
        for frame in frames:
            samples.extend(
                _build_sequence_samples(
                    frame=frame,
                    feature_columns=self.feature_columns,
                    family_mapping=family_mapping,
                    sequence_length=windowing.sequence_length,
                    sequence_stride=windowing.sequence_stride,
                )
            )
        self.samples = samples

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        sample = self.samples[index]
        features = torch.tensor(sample.features, dtype=torch.float32)
        binary = torch.tensor(sample.binary_labels, dtype=torch.float32)
        family = torch.tensor(sample.family_labels, dtype=torch.long)
        return {
            "features": features,
            "binary_labels": binary,
            "family_labels": family,
            "metadata": sample.metadata,
        }


def filter_normal_sequences(samples: Sequence[SequenceSample]) -> List[SequenceSample]:
    """Return sequences that contain only normal windows."""

    return [sample for sample in samples if not any(sample.binary_labels)]


def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""

    features = torch.stack([item["features"] for item in batch], dim=0)
    binary = torch.stack([item["binary_labels"] for item in batch], dim=0)
    family = torch.stack([item["family_labels"] for item in batch], dim=0)
    return {
        "features": features,
        "binary_labels": binary,
        "family_labels": family,
        "metadata": [item["metadata"] for item in batch],
    }


__all__ = ["SequenceDataset", "collate_fn", "filter_normal_sequences"]