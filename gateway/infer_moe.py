"""Inference utility for the unified Mixture-of-Experts model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gateway.data_pipeline import MoEDataset, PcapInfo, class_id_to_name  # noqa: E402
from gateway.unified_moe_model import (  # noqa: E402
    UNIFIED_EXPERT_SPECS,
    UNIFIED_GATING_INPUT_DIM,
    build_unified_moe,
)

NUM_CLASSES = 3
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "unified_moe.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified MoE inference over a PCAP file.")
    parser.add_argument("pcap", type=str, help="Path to the PCAP file to evaluate.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to a trained unified MoE checkpoint (default: gateway/unified_moe.pt).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Number of windows per inference batch.")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Optional cap on processed windows (0 disables the cap).",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Torch thread pool size for feature extraction (default: 1).",
    )
    parser.add_argument(
        "--gating-hidden",
        type=int,
        default=None,
        help="Override the gating hidden size when loading the checkpoint (normally auto-detected).",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, gating_hidden_override: int | None) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        gating_hidden = gating_hidden_override or int(payload.get("gating_hidden", 128))
        state_dict: Dict[str, torch.Tensor] = payload["state_dict"]
    else:
        gating_hidden = gating_hidden_override or 128
        state_dict = payload

    model = build_unified_moe(device=torch.device("cpu"), gating_hidden_dim=gating_hidden)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, args.num_threads))

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = load_model(checkpoint_path, gating_hidden_override=args.gating_hidden)

    pcap_path = Path(args.pcap).resolve()
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    info = PcapInfo(path=pcap_path, label=0)
    max_windows = args.max_windows if args.max_windows > 0 else None
    dataset = MoEDataset(
        files=[info],
        tasks=("dos", "arp"),
        batch_size=args.batch_size,
        shuffle=False,
        seed=42,
        max_windows_per_file=max_windows,
        max_total_windows=max_windows,
        status_interval=None,
        max_file_size=None,
        max_packets_per_file=None,
        file_timeout=None,
        max_packets_per_window=None,
    )
    dataset.set_window_budget(max_windows)
    dataset.set_log_fn(None)

    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    expert_names: List[str] = [spec.name for spec in UNIFIED_EXPERT_SPECS]

    print(
        f"[Info] Loaded checkpoint '{checkpoint_path.name}' | gating_input={UNIFIED_GATING_INPUT_DIM} "
        f"experts={len(expert_names)}"
    )
    print(f"[Info] Processing {pcap_path}")

    headers = ["Window", "Prediction", "P(normal)", "P(dos)", "P(arp)"]
    headers.extend(f"gate_{name}" for name in expert_names)
    print("\t".join(headers))

    window_offset = 0
    class_totals: Dict[int, int] = {i: 0 for i in range(NUM_CLASSES)}
    with torch.no_grad():
        for batch_features, _ in dataloader:
            logits, attention = model(batch_features, return_attention=True)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            weights = attention["weights"]

            for idx in range(predictions.size(0)):
                window_index = window_offset + idx + 1
                pred_label = int(predictions[idx].item())
                class_totals[pred_label] = class_totals.get(pred_label, 0) + 1
                probs = probabilities[idx]
                gates = weights[idx]
                row = [
                    f"{window_index:06d}",
                    class_id_to_name(pred_label),
                    f"{probs[0].item():.4f}",
                    f"{probs[1].item():.4f}",
                    f"{probs[2].item():.4f}",
                ]
                row.extend(f"{value.item():.4f}" for value in gates)
                print("\t".join(row))

            window_offset += predictions.size(0)

    print("\nPrediction counts:")
    for label_id, count in class_totals.items():
        label_name = class_id_to_name(label_id)
        print(f" - {label_name:<7}: {count}")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
