#!/usr/bin/env python3
import argparse

from dos_detector.config import load_config
from dos_detector.training.supervised_trainer import SupervisedTrainer
from dos_detector.utils import configure_cpu_environment
from dos_detector.utils.io import load_json


def _normalize_sample_path(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    return name if name.startswith("samples/") else f"samples/{name}"


def main() -> None:
    configure_cpu_environment(threads=2, interop_threads=2)
    parser = argparse.ArgumentParser(description="Train the supervised model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the training config.")
    parser.add_argument("--val", default="mixed1.pcap,ssdp_flood6.pcap", help="Comma-separated filenames reserved for validation")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence_length")
    parser.add_argument("--seq-stride", type=int, default=None, help="Override sequence_stride")
    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seq_len is not None:
        cfg.windowing.sequence_length = int(args.seq_len)
    if args.seq_stride is not None:
        cfg.windowing.sequence_stride = int(args.seq_stride)
    if args.epochs is not None:
        cfg.training.supervised.max_epochs = int(args.epochs)

    manifest = load_json(cfg.paths.manifest_path)
    all_files = [entry["pcap"] for entry in manifest["frames"]]
    val_candidates = []
    for item in args.val.split(","):
        normalized = _normalize_sample_path(item)
        if normalized:
            val_candidates.append(normalized)
    val_set = set(val_candidates)
    cfg.data.val_files = sorted({f for f in all_files if f in val_set})
    cfg.data.train_files = sorted({f for f in all_files if f not in val_set})

    print(f"[cfg] seq_len/stride: {cfg.windowing.sequence_length}/{cfg.windowing.sequence_stride}")
    print(f"[cfg] train_files: {len(cfg.data.train_files)}")
    print(f"[cfg] val_files: {len(cfg.data.val_files)} → {cfg.data.val_files}")

    trainer = SupervisedTrainer(cfg)
    print("[SUP] training…")
    metrics = trainer.train()
    print("[SUP] done:", metrics)


if __name__ == "__main__":
    main()
