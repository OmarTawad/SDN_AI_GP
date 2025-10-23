"""Training loop utilities for the unified MoE gate."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import IterableDataset
from tqdm import tqdm

from gateway.core import class_id_to_name
from gateway.training.data import set_dataset_logger


def compute_batch_budget(config) -> Optional[int]:
    if config.max_total_windows is None:
        return None
    batch_cap = max(1, math.ceil(config.max_total_windows / max(1, config.batch_size)))
    print(
        f"[Config] Window budget of {config.max_total_windows} windows allows up to "
        f"{batch_cap} batches per epoch (batch_size={config.batch_size})."
    )
    return batch_cap


def resolve_effective_max_batches(config, batches_cap: Optional[int]) -> Optional[int]:
    effective = config.max_batches
    if effective is not None and batches_cap is not None and batches_cap < effective:
        print(
            f"[Config] Adjusting max_batches from {effective} to {batches_cap} to align with the window budget."
        )
        return batches_cap
    if effective is None and batches_cap is not None:
        print(
            "[Config] No explicit max_batches provided; the window budget will implicitly cap "
            f"the epoch at {batches_cap} batches."
        )
        return batches_cap
    return effective


def train_epoch(
    epoch_index: int,
    num_epochs: int,
    dataloader: Iterable,
    dataset: IterableDataset,
    model,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    effective_max_batches: Optional[int],
    auto_recon_weight: float,
) -> Tuple[float, float, int, bool]:
    epoch_loss = 0.0
    epoch_correct = 0
    total_samples = 0
    batch_count = 0
    printed_shapes = False
    stop_training = False
    wants_attention = auto_recon_weight > 0.0
    progress_total = effective_max_batches
    progress = tqdm(dataloader, desc=f"Epoch {epoch_index + 1}/{num_epochs}", unit="batch", total=progress_total)
    set_dataset_logger(dataset, progress.write)

    try:
        for batch_idx, (features, targets) in enumerate(progress, start=1):
            if not printed_shapes:
                shape_bits = []
                for key in ("gating_input", "auto", "dos_cnn_seq", "dos_lstm_seq", "arp_cnn_seq", "arp_lstm_seq"):
                    tensor = features.get(key)
                    if isinstance(tensor, Tensor):
                        shape_bits.append(f"{key}={tuple(tensor.shape)}")
                if shape_bits:
                    progress.write("[Shapes] " + " ".join(shape_bits))
                printed_shapes = True

            optimizer.zero_grad()
            if wants_attention:
                logits, attention = model(features, return_attention=True)
            else:
                logits = model(features)
                attention = None

            targets = targets.to(logits.device)
            loss = criterion(logits, targets)
            aux_value = 0.0
            if wants_attention and attention is not None:
                auto_scores = attention["expert_outputs"][:, 0]
                recon_loss = -auto_scores.mean()
                loss = loss + auto_recon_weight * recon_loss
                aux_value = float(recon_loss.item())

            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            epoch_loss += loss_value
            preds = logits.argmax(dim=1)
            batch_correct = int((preds == targets).sum().item())
            epoch_correct += batch_correct
            sample_count = targets.size(0)
            total_samples += sample_count
            batch_count += 1
            accuracy = batch_correct / max(1, sample_count)
            postfix: Dict[str, str] = {"loss": f"{loss_value:.4f}", "acc": f"{accuracy * 100:.1f}%"}
            if wants_attention:
                postfix["aux"] = f"{aux_value:.4f}"
            progress.set_postfix(**postfix)
            log_line = (
                f"Epoch {epoch_index + 1} Batch {batch_idx}: "
                f"loss={loss_value:.4f} acc={accuracy * 100:.1f}%"
            )
            if wants_attention:
                log_line += f" aux_recon={aux_value:.4f}"
            progress.write(log_line)
            if effective_max_batches is not None and batch_count >= effective_max_batches:
                progress.write(f"Reached max_batches={effective_max_batches}; stopping early.")
                stop_training = True
                break
    finally:
        set_dataset_logger(dataset, None)
        progress.close()

    avg_accuracy = epoch_correct / max(1, total_samples)
    return epoch_loss, avg_accuracy, batch_count, stop_training


__all__ = [
    "compute_batch_budget",
    "resolve_effective_max_batches",
    "train_epoch",
]

