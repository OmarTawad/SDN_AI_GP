"""Train the Mixture-of-Experts gating network on streaming or cached features."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, replace
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
    discover_pcaps,
    load_cache_entries,
    tasks_slug,
)
from gateway.moe_model import DEFAULT_TASK_SPECS, build_multitask_moe


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
        "--tasks",
        type=str,
        default="dos,arp",
        help="Comma-separated list of tasks to train (choices: dos, arp).",
    )
    parser.add_argument(
        "--gating-hidden",
        type=int,
        default=None,
        help="Hidden width for each gating MLP (defaults to spec value).",
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
    return parser.parse_args()


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    num_threads: int
    tasks: List[str]
    gating_hidden: Optional[int]
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

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        tasks = [task.strip().lower() for task in args.tasks.split(",") if task.strip()]
        return cls(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            num_threads=max(1, args.num_threads),
            tasks=tasks,
            gating_hidden=args.gating_hidden,
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


def _ensure_tasks(tasks: Sequence[str]) -> List[str]:
    if not tasks:
        raise ValueError("No tasks specified. Choose from 'dos' and 'arp'.")
    valid_tasks = {"dos", "arp"}
    for task in tasks:
        if task not in valid_tasks:
            raise ValueError(f"Unsupported task '{task}'. Valid options: dos, arp.")
    return list(tasks)


def build_dataset(
    config: TrainingConfig,
    files: Sequence,
) -> Tuple[IterableDataset[Tuple[Dict[str, Tensor], Dict[str, Tensor]]], bool]:
    entries, missing, mismatched = load_cache_entries(config.cache_base, files, config.tasks)
    slug = tasks_slug(config.tasks)
    cache_dir = config.cache_base / slug

    if config.use_cache != "off" and entries and not missing and not mismatched:
        print(f"[Cache] Using cached tensors from {cache_dir}")
        payloads: List[Dict[str, object]] = []
        for entry in entries:
            data = entry["data"]
            features = {key: tensor.detach().to(torch.float32) for key, tensor in data.get("features", {}).items()}
            labels = {key: tensor.detach().to(torch.float32) for key, tensor in data.get("labels", {}).items()}
            meta = dict(data.get("meta", {}))
            meta.setdefault("source_path", str(entry["path"]))
            meta.setdefault("cache_path", str(entry["cache_path"]))
            payloads.append({"features": features, "labels": labels, "meta": meta})
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
    model,
    optimizer,
    criterion,
    effective_max_batches: Optional[int],
    selected_tasks: Sequence[str],
) -> Tuple[float, int, bool]:
    epoch_loss = 0.0
    batch_count = 0
    printed_shapes = False
    stop_training = False
    progress_total = effective_max_batches
    progress = tqdm(
        dataloader,
        desc=f"Epoch {epoch_index + 1}/{num_epochs}",
        unit="batch",
        total=progress_total,
    )
    set_dataset_logger(dataset, progress.write)

    try:
        for batch_idx, (features, labels) in enumerate(progress, start=1):
            if not printed_shapes:
                shape_bits = [f"auto={tuple(features['auto'].shape)}"]
                if "dos" in selected_tasks and "dos_gating" in features:
                    shape_bits.append(f"dos_gating={tuple(features['dos_gating'].shape)}")
                if "arp" in selected_tasks and "arp_gating" in features:
                    shape_bits.append(f"arp_gating={tuple(features['arp_gating'].shape)}")
                progress.write("[Shapes] " + " ".join(shape_bits))
                printed_shapes = True

            optimizer.zero_grad()
            outputs = model(features)
            losses = []
            for task_name, logits in outputs.items():
                target = labels.get(task_name)
                if target is None:
                    continue
                losses.append(criterion(logits, target))
            if not losses:
                continue
            loss = sum(losses) / len(losses)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            epoch_loss += loss_value
            batch_count += 1
            progress.set_postfix(loss=f"{loss_value:.4f}")
            progress.write(f"Epoch {epoch_index + 1} Batch {batch_idx}: loss={loss_value:.4f}")
            if effective_max_batches is not None and batch_count >= effective_max_batches:
                progress.write(f"Reached max_batches={effective_max_batches}; stopping early.")
                stop_training = True
                break
    finally:
        set_dataset_logger(dataset, None)
        progress.close()

    return epoch_loss, batch_count, stop_training


def summarise(dataset: IterableDataset, selected_tasks: Sequence[str]) -> None:
    print("\nPCAP summary:")
    label_totals: Dict[str, Dict[str, int]] = {task: {} for task in selected_tasks}

    stats = getattr(dataset, "stats", {})
    skipped = getattr(dataset, "skipped", {})

    for path, info in stats.items():
        skip_reason = info.get("skipped_reason") or skipped.get(path)
        if skip_reason:
            print(f" - {path}: skipped ({skip_reason})")
            continue
        labels = info.get("labels", {})
        windows = int(info.get("windows", 0))
        batches = int(info.get("batches", 0))
        label_desc = ", ".join(
            f"{task}={'attack' if labels.get(task, 0) else 'normal'}" for task in selected_tasks
        )
        print(f" - {path}: labels={label_desc}, windows={windows}, batches={batches}")
        for task in selected_tasks:
            totals = label_totals.setdefault(task, {})
            label_name = "attack" if labels.get(task, 0) else "normal"
            totals[label_name] = totals.get(label_name, 0) + windows

    print("\nWindow counts by task:")
    for task, counts in label_totals.items():
        for label_name, count in counts.items():
            print(f" * {task} {label_name}: {count} windows")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run_training(config: TrainingConfig) -> None:
    torch.set_num_threads(max(1, config.num_threads))
    selected_tasks = _ensure_tasks(config.tasks)

    task_specs_map = {spec.name: spec for spec in DEFAULT_TASK_SPECS}
    selected_specs = []
    for task in selected_tasks:
        spec = task_specs_map[task]
        if config.gating_hidden is not None:
            spec = replace(spec, gating_hidden_dim=config.gating_hidden)
        selected_specs.append(spec)

    files = discover_pcaps(selected_tasks)
    if not files:
        raise RuntimeError("No eligible PCAP files found in 'samples/'. Populate the directory with .pcap files.")

    dataset, used_cache = build_dataset(config, files)
    batches_cap = compute_batch_budget(config)
    effective_max_batches = resolve_effective_max_batches(config, batches_cap)

    window_budget_for_logs: Optional[int] = config.max_total_windows
    if window_budget_for_logs is None and effective_max_batches is not None:
        window_budget_for_logs = effective_max_batches * config.batch_size
    set_dataset_budget(dataset, window_budget_for_logs)

    model = build_multitask_moe(task_specs=selected_specs, device=torch.device("cpu"))
    criterion = nn.BCEWithLogitsLoss()
    gating_params: List[nn.Parameter] = []
    for spec in selected_specs:
        gating_params.extend(model.tasks[spec.name].gating.parameters())
    optimizer = torch.optim.Adam(gating_params, lr=config.learning_rate)

    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    for epoch in range(config.epochs):
        set_dataset_epoch(dataset, epoch)
        epoch_loss, batch_count, stop_training = train_epoch(
            epoch_index=epoch,
            num_epochs=config.epochs,
            dataloader=dataloader,
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            effective_max_batches=effective_max_batches,
            selected_tasks=selected_tasks,
        )
        average_loss = epoch_loss / batch_count if batch_count else float("nan")
        print(f"Epoch {epoch + 1} average loss: {average_loss:.4f}")
        if stop_training:
            break

    gate_state = {name: task.gating.state_dict() for name, task in model.tasks.items()}
    output_path = Path(__file__).resolve().parent / "moe_gate.pt"
    torch.save(gate_state, output_path)
    print(f"Saved gating weights to {output_path}")

    if used_cache:
        print("[Cache] Training consumed cached tensors.")
    summarise(dataset, selected_tasks)


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_args(args)
    run_training(config)


if __name__ == "__main__":
    main()
