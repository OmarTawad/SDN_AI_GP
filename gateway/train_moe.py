"""Train the Mixture-of-Experts gating network on streaming or cached features."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from gateway.data_pipeline import (
    CACHE_ROOT,
    CachedMoEDataset,
    MoEDataset,
    class_id_to_name,
    discover_pcaps,
    load_cache_entries,
    tasks_slug,
)
from gateway.unified_moe_model import UNIFIED_GATING_INPUT_DIM, UnifiedMoE, build_unified_moe


# ---------------------------------------------------------------------------
# Command-line handling
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MoE gate on streaming PCAP features or cached tensors.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=17, help="Random seed (default: 17)")
    default_threads = max(1, min(2, os.cpu_count() or 1))
    parser.add_argument(
        "--num-threads",
        type=int,
        default=default_threads,
        help="Torch CPU thread pool size (default: min(2, available cores)).",
    )
    parser.add_argument(
        "--gating-hidden",
        type=int,
        default=128,
        help="Hidden width for the unified gating MLP (default: 128).",
    )
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=80,
        help="Cap the number of processed windows per PCAP (0 disables the cap).",
    )
    parser.add_argument(
        "--max-total-windows",
        type=int,
        default=1200,
        help="Global window cap across all files (0 disables the cap).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit on the number of optimizer steps per epoch (default: run full epoch).",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=200,
        help="Emit a progress log every N processed windows (0 disables logging).",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=None,
        help="Skip PCAP files larger than this size (in megabytes).",
    )
    parser.add_argument(
        "--max-packets-per-file",
        type=int,
        default=None,
        help="Stop reading a PCAP after this many packets (0 disables the cap).",
    )
    parser.add_argument(
        "--max-packets-per-window",
        type=int,
        default=None,
        help="Limit stored packets per window (default: no limit).",
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=None,
        help="Abort processing a PCAP after this many seconds (0 disables the timeout).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory containing cached window tensors (default: gateway/cache).",
    )
    parser.add_argument(
        "--use-cache",
        type=str,
        choices=("auto", "on", "off"),
        default="auto",
        help="Use cached tensors (auto tries cache, on requires cache, off disables it).",
    )
    parser.add_argument(
        "--auto-recon-weight",
        type=float,
        default=0.0,
        help="Optional weight for negative autoencoder scores as auxiliary loss.",
    )
    return parser.parse_args()


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    num_threads: int
    gating_hidden: int
    auto_recon_weight: float
    max_windows_per_file: Optional[int]
    max_total_windows: Optional[int]
    max_batches: Optional[int]
    status_interval: Optional[int]
    max_file_size_mb: Optional[float]
    max_packets_per_file: Optional[int]
    file_timeout: Optional[float]
    max_packets_per_window: Optional[int]
    cache_dir: Optional[Path]
    use_cache: str
    tasks: Tuple[str, ...] = ("dos", "arp")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        return cls(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            num_threads=max(1, args.num_threads),
            gating_hidden=max(1, args.gating_hidden),
            auto_recon_weight=max(0.0, float(args.auto_recon_weight)),
            max_windows_per_file=args.max_windows_per_file if args.max_windows_per_file > 0 else None,
            max_total_windows=args.max_total_windows if args.max_total_windows > 0 else None,
            max_batches=args.max_batches,
            status_interval=args.status_interval if args.status_interval and args.status_interval > 0 else None,
            max_file_size_mb=args.max_file_size_mb if args.max_file_size_mb and args.max_file_size_mb > 0 else None,
            max_packets_per_file=args.max_packets_per_file if args.max_packets_per_file and args.max_packets_per_file > 0 else None,
            file_timeout=args.file_timeout if args.file_timeout and args.file_timeout > 0 else None,
            max_packets_per_window=args.max_packets_per_window if args.max_packets_per_window and args.max_packets_per_window > 0 else None,
            cache_dir=Path(args.cache_dir).resolve() if args.cache_dir else None,
            use_cache=args.use_cache,
            tasks=("dos", "arp"),
        )

    @property
    def file_size_bytes(self) -> Optional[int]:
        if self.max_file_size_mb is None:
            return None
        return int(self.max_file_size_mb * 1024 * 1024)

    @property
    def cache_base(self) -> Path:
        return self.cache_dir or CACHE_ROOT


# ---------------------------------------------------------------------------
# Dataset orchestration
# ---------------------------------------------------------------------------
def build_dataset(
    config: TrainingConfig,
    files: Sequence,
) -> Tuple[IterableDataset[Tuple[Dict[str, Tensor], Tensor]], bool]:
    entries, missing, mismatched = load_cache_entries(config.cache_base, files, config.tasks)
    slug = tasks_slug(config.tasks)
    cache_dir = config.cache_base / slug

    if config.use_cache != "off" and entries and not missing and not mismatched:
        print(f"[Cache] Using cached tensors from {cache_dir}")
        payloads: List[Dict[str, object]] = []
        for entry in entries:
            data = entry["data"]
            features = {
                key: tensor.detach().to(torch.float32)
                for key, tensor in data.get("features", {}).items()
            }
            payload: Dict[str, object] = {"features": features}
            targets = data.get("targets")
            if targets is not None:
                payload["targets"] = torch.as_tensor(targets).to(torch.long)
            labels = data.get("labels")
            if isinstance(labels, dict):
                payload["labels"] = {
                    key: torch.as_tensor(value).to(torch.float32)
                    for key, value in labels.items()
                }
            meta = dict(data.get("meta", {}))
            meta.setdefault("source_path", str(entry["path"]))
            meta.setdefault("cache_path", str(entry["cache_path"]))
            payload["meta"] = meta
            payloads.append(payload)
        dataset = CachedMoEDataset(
            cache_entries=payloads,
            tasks=config.tasks,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed,
            max_windows_per_file=config.max_windows_per_file,
            max_total_windows=config.max_total_windows,
        )
        return dataset, True

    if config.use_cache == "on":
        problems: List[str] = []
        if not entries:
            problems.append("no cache files were found")
        if missing:
            problems.append(
                "missing caches for: " + ", ".join(path.name for path in missing)
            )
        if mismatched:
            problems.append(
                "invalid caches: " + ", ".join(f"{path.name} ({reason})" for path, reason in mismatched)
            )
        raise RuntimeError("Cache usage forced with --use-cache on, but " + "; ".join(problems))

    if entries and (missing or mismatched):
        if missing:
            print(f"[Cache] Missing caches for {len(missing)} PCAPs; falling back to raw streaming.")
        if mismatched:
            print(f"[Cache] Ignoring {len(mismatched)} cache files due to mismatches.")

    dataset = MoEDataset(
        files=files,
        tasks=config.tasks,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        max_windows_per_file=config.max_windows_per_file,
        max_total_windows=config.max_total_windows,
        status_interval=config.status_interval,
        max_file_size=config.file_size_bytes,
        max_packets_per_file=config.max_packets_per_file,
        file_timeout=config.file_timeout,
        max_packets_per_window=config.max_packets_per_window,
    )
    return dataset, False


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def compute_batch_budget(config: TrainingConfig) -> Optional[int]:
    if config.max_total_windows is None:
        return None
    batch_cap = max(1, math.ceil(config.max_total_windows / max(1, config.batch_size)))
    print(
        f"[Config] Window budget of {config.max_total_windows} windows allows up to "
        f"{batch_cap} batches per epoch (batch_size={config.batch_size})."
    )
    return batch_cap


def resolve_effective_max_batches(
    config: TrainingConfig,
    batches_cap: Optional[int],
) -> Optional[int]:
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


def set_dataset_budget(
    dataset: IterableDataset,
    window_budget: Optional[int],
) -> None:
    if hasattr(dataset, "set_window_budget"):
        dataset.set_window_budget(window_budget)


def set_dataset_logger(dataset: IterableDataset, logger) -> None:
    if hasattr(dataset, "set_log_fn"):
        dataset.set_log_fn(logger)


def set_dataset_epoch(dataset: IterableDataset, epoch: int) -> None:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def train_epoch(
    epoch_index: int,
    num_epochs: int,
    dataloader: Iterable,
    dataset: IterableDataset,
    model: UnifiedMoE,
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
    progress = tqdm(
        dataloader,
        desc=f"Epoch {epoch_index + 1}/{num_epochs}",
        unit="batch",
        total=progress_total,
    )
    set_dataset_logger(dataset, progress.write)

    try:
        for batch_idx, (features, targets) in enumerate(progress, start=1):
            if not printed_shapes:
                shape_bits: List[str] = []
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
            postfix = {"loss": f"{loss_value:.4f}", "acc": f"{accuracy * 100:.1f}%"}
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


def summarise(dataset: IterableDataset) -> None:
    print("\nPCAP summary:")
    stats = getattr(dataset, "stats", {})
    skipped = getattr(dataset, "skipped", {})
    if not stats:
        print(" (no statistics available)")
        return

    label_totals: Dict[int, int] = {}
    for path, info in stats.items():
        skip_reason = info.get("skipped_reason") or skipped.get(path)
        if skip_reason:
            print(f" - {path}: skipped ({skip_reason})")
            continue
        label_id = int(info.get("label", 0))
        label_name = info.get("label_name", class_id_to_name(label_id))
        windows = int(info.get("windows", 0))
        batches = int(info.get("batches", 0))
        truncated = int(info.get("truncated_windows", 0))
        line = f" - {path}: label={label_name}, windows={windows}, batches={batches}"
        if truncated:
            line += f", truncated={truncated}"
        print(line)
        label_totals[label_id] = label_totals.get(label_id, 0) + windows

    print("\nWindow counts by class:")
    for label_id, count in sorted(label_totals.items()):
        print(f" * {class_id_to_name(label_id)}: {count} windows")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


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
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config.learning_rate)

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
    output_path = Path(__file__).resolve().parent / "unified_moe.pt"
    torch.save({"state_dict": model.state_dict(), "gating_hidden": config.gating_hidden}, output_path)
    print(f"Saved unified MoE weights to {output_path}")

    if used_cache:
        print("[Cache] Training consumed cached tensors.")
    summarise(dataset)


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_args(args)
    run_training(config)


if __name__ == "__main__":
    main()
