"""Window feature assembly helpers for streaming datasets.


"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from gateway.data.extractors.features import (
    ARP_FEATURE_EXTRACTOR,
    ARP_LSTM_SCALER,
    ARP_LSTM_SEQUENCE_LENGTH,
    ARP_MICRO_BINS,
    ARP_WINDOW_SIZE,
    DOS_FEATURE_EXTRACTOR,
    DOS_LSTM_SCALER,
    DOS_LSTM_SEQUENCE_LENGTH,
    DOS_MICRO_BINS,
    DOS_WINDOW_SIZE,
    TOP_UDP_PORTS,
    WINDOW_SIZE,
    compute_arp_sequence_features,
    compute_arp_static_features,
    compute_dos_sequence_features,
    compute_dos_static_features,
    prepare_arp_static,
    prepare_auto_tensor,
    prepare_dos_static,
)
from gateway.data.datasets.packet import build_arp_window, build_dos_window
from gateway.data.structures.windowing import SequenceState, WindowBuffer
from gateway.moe_model import (
    ARP_CNN_SEQ_IN_DIM,
    ARP_CNN_STATIC_DIM,
    ARP_LSTM_INPUT_DIM,
    DOS_CNN_SEQ_IN_DIM,
    DOS_CNN_STATIC_DIM,
    DOS_LSTM_INPUT_DIM,
)


def assemble_window_features(
    buffer: WindowBuffer,
    tasks: Sequence[str],
    dos_state: Optional[SequenceState],
    arp_state: Optional[SequenceState],
) -> Optional[Dict[str, Tensor]]:
    """Convert a completed window buffer to model-ready tensors."""

    auto_stats = buffer.auto_acc.finalize()
    if auto_stats is None:
        return None
    truncated = False
    auto_tensor = torch.from_numpy(prepare_auto_tensor(auto_stats))
    features: Dict[str, Tensor] = {"auto": auto_tensor}

    if "dos" in tasks:
        if dos_state is None:
            raise RuntimeError("DoS task selected but DOS state not initialised.")
        dos_rows = buffer.task_rows.get("dos", [])
        if dos_rows:
            try:
                dos_seq_np, dos_extras = compute_dos_sequence_features(
                    dos_rows,
                    buffer.bin_indices.get("dos", []),
                    DOS_MICRO_BINS,
                    TOP_UDP_PORTS,
                )
                dos_static_vec, _, _ = compute_dos_static_features(
                    dos_rows,
                    DOS_MICRO_BINS,
                    dos_extras["per_bin_total_pkts"],
                    TOP_UDP_PORTS,
                    WINDOW_SIZE,
                )
                dos_cnn_seq_tensor = torch.from_numpy(dos_seq_np.astype(np.float32))
                dos_cnn_static_tensor = torch.from_numpy(prepare_dos_static(dos_static_vec.astype(np.float32)))
                dos_window = build_dos_window(buffer.index, buffer.start, buffer.end, dos_rows)
                dos_features = DOS_FEATURE_EXTRACTOR._features_for_window(dos_window)
                dos_lstm_seq_tensor = dos_state.update(dos_features, DOS_LSTM_SCALER)
            except Exception:
                dos_cnn_seq_tensor = torch.zeros((DOS_MICRO_BINS, DOS_CNN_SEQ_IN_DIM), dtype=torch.float32)
                dos_cnn_static_tensor = torch.zeros(DOS_CNN_STATIC_DIM, dtype=torch.float32)
                dos_lstm_seq_tensor = torch.zeros(
                    (DOS_LSTM_SEQUENCE_LENGTH, DOS_LSTM_INPUT_DIM),
                    dtype=torch.float32,
                )
        else:
            dos_cnn_seq_tensor = torch.zeros((DOS_MICRO_BINS, DOS_CNN_SEQ_IN_DIM), dtype=torch.float32)
            dos_cnn_static_tensor = torch.zeros(DOS_CNN_STATIC_DIM, dtype=torch.float32)
            dos_lstm_seq_tensor = torch.zeros(
                (DOS_LSTM_SEQUENCE_LENGTH, DOS_LSTM_INPUT_DIM),
                dtype=torch.float32,
            )
        truncated = truncated or buffer.truncated
        features.update(
            {
                "dos_cnn_seq": dos_cnn_seq_tensor,
                "dos_cnn_static": dos_cnn_static_tensor,
                "dos_lstm_seq": dos_lstm_seq_tensor,
            }
        )

    if "arp" in tasks:
        if arp_state is None:
            raise RuntimeError("ARP task selected but ARP state not initialised.")
        arp_rows = buffer.task_rows.get("arp", [])
        if arp_rows:
            try:
                arp_seq_np, arp_extras = compute_arp_sequence_features(
                    arp_rows,
                    buffer.bin_indices.get("arp", []),
                    ARP_MICRO_BINS,
                )
                arp_static_vec, _, _ = compute_arp_static_features(
                    arp_rows,
                    ARP_MICRO_BINS,
                    arp_extras,
                    ARP_WINDOW_SIZE,
                )
                arp_cnn_seq_tensor = torch.from_numpy(arp_seq_np.astype(np.float32))
                arp_cnn_static_tensor = torch.from_numpy(prepare_arp_static(arp_static_vec.astype(np.float32)))
                arp_window = build_arp_window(buffer.index, buffer.start, buffer.end, arp_rows)
                arp_features = ARP_FEATURE_EXTRACTOR._features_for_window(arp_window)
                arp_lstm_seq_tensor = arp_state.update(arp_features, ARP_LSTM_SCALER)
            except Exception:
                arp_cnn_seq_tensor = torch.zeros((ARP_MICRO_BINS, ARP_CNN_SEQ_IN_DIM), dtype=torch.float32)
                arp_cnn_static_tensor = torch.zeros(ARP_CNN_STATIC_DIM, dtype=torch.float32)
                arp_lstm_seq_tensor = torch.zeros(
                    (ARP_LSTM_SEQUENCE_LENGTH, ARP_LSTM_INPUT_DIM),
                    dtype=torch.float32,
                )
        else:
            arp_cnn_seq_tensor = torch.zeros((ARP_MICRO_BINS, ARP_CNN_SEQ_IN_DIM), dtype=torch.float32)
            arp_cnn_static_tensor = torch.zeros(ARP_CNN_STATIC_DIM, dtype=torch.float32)
            arp_lstm_seq_tensor = torch.zeros(
                (ARP_LSTM_SEQUENCE_LENGTH, ARP_LSTM_INPUT_DIM),
                dtype=torch.float32,
            )
        truncated = truncated or buffer.truncated
        features.update(
            {
                "arp_cnn_seq": arp_cnn_seq_tensor,
                "arp_cnn_static": arp_cnn_static_tensor,
                "arp_lstm_seq": arp_lstm_seq_tensor,
            }
        )

    if not features:
        return None
    if truncated:
        features["_truncated"] = torch.tensor([1], dtype=torch.float32)
    return features


__all__ = ["assemble_window_features"]

