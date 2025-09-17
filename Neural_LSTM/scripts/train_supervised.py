#!/usr/bin/env python3
import argparse, torch
from dos_detector.config import load_config
from dos_detector.utils.io import load_json
from dos_detector.training.supervised_trainer import SupervisedTrainer

torch.set_num_threads(2)

def main():
    ap = argparse.ArgumentParser(description="Train the supervised model.")
    ap.add_argument("--val", default="mixed1.pcap,ssdp_flood6.pcap", help="Comma-separated filenames reserved for validation")
    ap.add_argument("--seq-len", type=int, default=None, help="Override sequence_length")
    ap.add_argument("--seq-stride", type=int, default=None, help="Override sequence_stride")
    ap.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    ap.add_argument("--num-workers", type=int, default=1, help="DataLoader workers")
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")
    if args.seq_len is not None:    cfg.windowing.sequence_length = int(args.seq_len)
    if args.seq_stride is not None: cfg.windowing.sequence_stride = int(args.seq_stride)
    if args.epochs is not None:     cfg.training.supervised.max_epochs = int(args.epochs)
    cfg.training.supervised.num_workers = int(args.num_workers)

    man = load_json(cfg.paths.manifest_path)
    all_files = [x["pcap"] for x in man["frames"]]
    val_set = set(f"samples/{v.strip()}" if not v.strip().startswith("samples/") else v.strip()
                  for v in args.val.split(",") if v.strip())
    cfg.data.val_files = sorted([f for f in all_files if f in val_set])
    cfg.data.train_files = sorted([f for f in all_files if f not in val_set])

    print("[cfg] seq_len/stride:", cfg.windowing.sequence_length, cfg.windowing.sequence_stride)
    print("[cfg] train_files:", len(cfg.data.train_files))
    print("[cfg] val_files:", len(cfg.data.val_files), cfg.data.val_files)

    print("[SUP] training…")
    hist = SupervisedTrainer(cfg).train()
    print("[SUP] done:", hist)

if __name__ == "__main__":
    main()
