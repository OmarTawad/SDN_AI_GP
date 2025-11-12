#src/dos_detector/data/dataset.py
"""Dataset helpers for training."""

from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ..config.types import WindowingConfig
from .structures import SequenceSample
from ..utils.io import resolve_processed_frame, stream_dataframe

DEFAULT_CHUNK_SIZE = 50_000
META_COLUMNS = ["attack", "family", "window_index", "window_start", "window_end", "pcap"]


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


class StreamingSequenceDataset(IterableDataset[Dict[str, Any]]):
    """Memory-efficient dataset that streams window sequences from disk."""

    def __init__(
        self,
        files: Sequence[str],
        processed_dir: Path,
        feature_columns: Sequence[str],
        family_mapping: Dict[str, int],
        windowing: WindowingConfig,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        scaler: Any | None = None,
        shuffle_files: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.processed_dir = Path(processed_dir)
        self.feature_columns = list(feature_columns)
        self.family_mapping = dict(family_mapping)
        self.windowing = windowing
        self.chunk_size = max(1_000, int(chunk_size))
        self.scaler = scaler
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.family_default = self.family_mapping.get("other", 0)
        self.required_columns = list(dict.fromkeys([*self.feature_columns, *META_COLUMNS]))

    def __iter__(self) -> Iterator[Dict[str, Any]]:  # type: ignore[override]
        files = self._select_files_for_worker()
        for name in files:
            yield from self._stream_file(name)

    def _select_files_for_worker(self) -> List[str]:
        if not self.files:
            return []
        worker = get_worker_info()
        files = self.files.copy()
        rng_seed = self.seed if worker is None else worker.seed
        if self.shuffle_files:
            random.Random(rng_seed).shuffle(files)
        if worker is None or worker.num_workers is None or worker.num_workers <= 1:
            return files
        return files[worker.id :: worker.num_workers]

    def _resolve_frame_path(self, name: str) -> Path:
        return resolve_processed_frame(self.processed_dir, name)

    def _stream_file(self, name: str) -> Iterator[Dict[str, Any]]:
        path = self._resolve_frame_path(name)
        buffer = pd.DataFrame(columns=self.required_columns)
        stride = max(1, self.windowing.sequence_stride)
        seq_len = self.windowing.sequence_length
        for chunk in stream_dataframe(path, columns=self.required_columns, chunk_size=self.chunk_size):
            if chunk.empty:
                continue
            if not buffer.empty:
                chunk = pd.concat([buffer, chunk], ignore_index=True)
            chunk = chunk.reset_index(drop=True)
            feature_block = chunk[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
            if self.scaler is not None:
                feature_block = self.scaler.transform(feature_block).astype(np.float32, copy=False)
            binary_block = chunk["attack"].to_numpy(dtype=np.float32, copy=True)
            family_block = (
                chunk["family"]
                .map(lambda fam: self.family_mapping.get(str(fam), self.family_default))
                .to_numpy(dtype=np.int64, copy=False)
            )
            window_index = chunk["window_index"].to_numpy(dtype=np.int64, copy=True)
            window_start = chunk["window_start"].to_numpy(copy=True)
            window_end = chunk["window_end"].to_numpy(copy=True)
            pcap_values = chunk["pcap"].astype(str).to_numpy(copy=False)
            start = 0
            total = len(chunk)
            while start + seq_len <= total:
                end = start + seq_len
                metadata = {
                    "start_index": int(window_index[start]),
                    "end_index": int(window_index[end - 1]),
                    "window_start": float(window_start[start]),
                    "window_end": float(window_end[end - 1]),
                    "pcap": str(pcap_values[start]),
                }
                yield {
                    "features": feature_block[start:end],
                    "binary_labels": binary_block[start:end],
                    "family_labels": family_block[start:end],
                    "metadata": metadata,
                }
                start += stride
            buffer = chunk.iloc[start:].copy()
            del chunk
            gc.collect()


def filter_normal_sequences(samples: Sequence[SequenceSample]) -> List[SequenceSample]:
    """Return sequences that contain only normal windows."""

    return [sample for sample in samples if not any(sample.binary_labels)]


def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""

    def _to_tensor(value, dtype):
        if torch.is_tensor(value):
            return value.to(dtype=dtype)
        return torch.as_tensor(value, dtype=dtype)

    features = torch.stack([_to_tensor(item["features"], torch.float32) for item in batch], dim=0)
    binary = torch.stack([_to_tensor(item["binary_labels"], torch.float32) for item in batch], dim=0)
    family = torch.stack([_to_tensor(item["family_labels"], torch.long) for item in batch], dim=0)
    return {
        "features": features,
        "binary_labels": binary,
        "family_labels": family,
        "metadata": [item["metadata"] for item in batch],
    }


__all__ = ["SequenceDataset", "StreamingSequenceDataset", "collate_fn", "filter_normal_sequences"]
