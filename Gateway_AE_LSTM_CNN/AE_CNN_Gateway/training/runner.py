"""Training runner for the unified MoE gate."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from gateway.data.datasets.cache import discover_pcaps
from gateway.models.unified_moe import UNIFIED_GATING_INPUT_DIM, build_unified_moe
from gateway.training.configuration import TrainingConfig, parse_args
from gateway.training.data import build_dataset, set_dataset_budget, set_dataset_epoch
from gateway.training.loop import (
    compute_batch_budget,
    resolve_effective_max_batches,
    train_epoch,
)
from gateway.training.reporting import summarise


def run_training(config: TrainingConfig) -> None:
    torch.set_num_threads(max(1, config.num_threads))
    torch.manual_seed(config.seed)

    files = discover_pcaps()
    dataset, used_cache = build_dataset(config, files)
    if not files and not used_cache:
        raise RuntimeError(
            "No labelled PCAP files found in 'samples/'. Provide labels.csv or class subdirectories."
        )

    batches_cap = compute_batch_budget(config)
    effective_max_batches = resolve_effective_max_batches(config, batches_cap)

    window_budget_for_logs: Optional[int] = config.max_total_windows
    if window_budget_for_logs is None and effective_max_batches is not None:
        window_budget_for_logs = effective_max_batches * config.batch_size
    set_dataset_budget(dataset, window_budget_for_logs)

    device = torch.device("cpu")
    model = build_unified_moe(device=device, gating_hidden_dim=config.gating_hidden)
    print(
        f"[MoE] Unified gating input={UNIFIED_GATING_INPUT_DIM} hidden={config.gating_hidden} "
        f"experts={len(model.experts)}"
    )
    if config.auto_recon_weight > 0.0:
        print(f"[Config] Autoencoder auxiliary weight={config.auto_recon_weight}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=config.learning_rate)

    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    for epoch in range(config.epochs):
        set_dataset_epoch(dataset, epoch)
        epoch_loss, epoch_acc, batch_count, stop_training = train_epoch(
            epoch_index=epoch,
            num_epochs=config.epochs,
            dataloader=dataloader,
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            effective_max_batches=effective_max_batches,
            auto_recon_weight=config.auto_recon_weight,
        )
        average_loss = epoch_loss / batch_count if batch_count else float("nan")
        print(
            f"Epoch {epoch + 1} stats: loss={average_loss:.4f} "
            f"acc={epoch_acc * 100:.2f}% batches={batch_count}"
        )
        if stop_training:
            break

    model.eval()
    output_path = Path(__file__).resolve().parent.parent / "unified_moe.pt"
    torch.save({"state_dict": model.state_dict(), "gating_hidden": config.gating_hidden}, output_path)
    print(f"Saved unified MoE weights to {output_path}")

    if used_cache:
        print("[Cache] Training consumed cached tensors.")
    summarise(dataset)


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_args(args)
    run_training(config)


__all__ = ["main", "run_training"]

