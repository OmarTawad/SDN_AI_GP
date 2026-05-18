"""High-level orchestrator for unified MoE inference.


"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Sequence

import torch

from gateway.core import CLASS_LABELS
from gateway.core.labels import class_id_to_name
from gateway.models.unified_moe import UNIFIED_EXPERT_SPECS, UNIFIED_GATING_INPUT_DIM
from gateway.utils import get_logger
from .aggregation import AggregationSummary, aggregate_predictions
from .configuration import InferenceArgs, parse_args
from .dataloader import build_dataloader
from .model_loader import load_model
from .reporting import (
    build_json_summary,
    format_window_log,
    format_verdict_banner,
    write_json_summary,
    write_window_csv,
)
from .suspicion import extract_suspicion

LOGGER = get_logger("inference.runner")


def _render_progress(processed: int, probabilities: Sequence[float]) -> None:
    """Render inline progress with current class probabilities."""

    if processed <= 0:
        return
    bar_length = 24
    max_prob = max(probabilities) if probabilities else 0.0
    filled = int(round(bar_length * max_prob))
    bar = "█" * filled + "-" * (bar_length - filled)
    sys.stdout.write(
        "\r"
        f"[{bar}] windows={processed} "
        f"P(normal)={probabilities[0]:.2f} "
        f"P(dos)={probabilities[1]:.2f} "
        f"P(arp)={probabilities[2]:.2f}"
    )
    sys.stdout.flush()


def _build_window_row(
    window_index: int,
    prediction_id: int,
    probability_vector: Sequence[float],
    gating_weights: Sequence[float],
) -> List[str]:
    """Create the printable row for a single window.

    Args:
        window_index: One-indexed window counter.
        prediction_id: Predicted class identifier.
        probability_vector: Probability distribution across classes.
        gating_weights: Expert gating weights from the MoE model.

    Returns:
        List[str]: Formatted values ready for CSV output.
    """

    label = class_id_to_name(prediction_id)
    row = [f"{window_index:06d}", label]
    row.extend(f"{prob:.4f}" for prob in probability_vector)
    row.extend(f"{weight:.4f}" for weight in gating_weights)
    return row


def run(args: InferenceArgs) -> None:
    """Execute unified MoE inference given parsed arguments.

    Args:
        args: Parsed inference arguments.
    """

    torch.set_num_threads(max(1, args.num_threads))
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.pcap.exists():
        raise FileNotFoundError(f"PCAP file not found: {args.pcap}")

    model = load_model(args.checkpoint, args.gating_hidden)
    dataloader = build_dataloader(args.pcap, args.batch_size, args.max_windows)
    expert_names = [spec.name for spec in UNIFIED_EXPERT_SPECS]

    LOGGER.info(
        "Loaded checkpoint %s | gating_input=%d experts=%d",
        args.checkpoint.name,
        UNIFIED_GATING_INPUT_DIM,
        len(expert_names),
    )
    LOGGER.info("Processing %s", args.pcap)

    headers = ["Window", "Prediction", "P(normal)", "P(dos)", "P(arp)"]
    headers.extend(f"gate_{name}" for name in expert_names)
    window_rows: List[List[str]] = []
    predictions: List[int] = []
    probability_vectors: List[Sequence[float]] = []

    processed_windows = 0
    stop_processing = False
    with torch.no_grad():
        for batch_features, _ in dataloader:
            logits, attention = model(batch_features, return_attention=True)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1).cpu()
            weights = attention["weights"].cpu()

            for idx in range(preds.size(0)):
                if args.max_windows is not None and processed_windows >= args.max_windows:
                    stop_processing = True
                    break
                window_index = processed_windows + 1
                probability_vector = probs[idx].tolist()
                gating_weights = weights[idx].tolist()
                row = _build_window_row(window_index, int(preds[idx].item()), probability_vector, gating_weights)
                window_rows.append(row)
                predictions.append(int(preds[idx].item()))
                probability_vectors.append(probability_vector)
                processed_windows += 1
                if args.log_windows:
                    print(
                        format_window_log(
                            pcap_name=args.pcap.name,
                            window_index=window_index,
                            prediction=class_id_to_name(int(preds[idx].item())),
                            probabilities=probability_vector,
                            total_windows=args.max_windows,
                        )
                    )
                else:
                    _render_progress(processed_windows, probability_vector)
            if stop_processing:
                break

    if processed_windows and not args.log_windows:
        sys.stdout.write("\n")

    aggregation: AggregationSummary = aggregate_predictions(
        predictions=predictions,
        probabilities=probability_vectors,
        high_conf_threshold=args.high_confidence_threshold,
        attack_threshold=args.attack_threshold,
        attack_prob_threshold=args.attack_prob_threshold,
        min_attack_windows=args.min_attack_windows,
        min_high_conf_windows=args.min_high_confidence_windows,
    )
    suspicion = extract_suspicion(
        args.pcap,
        confidences={label: aggregation.classes[label].mean_probability for label in CLASS_LABELS},
    )

    csv_path = args.pcap.with_suffix(".windows.csv")
    json_path = args.pcap.with_suffix(".summary.json")
    write_window_csv(csv_path, headers, window_rows)

    summary_payload = build_json_summary(
        pcap_name=args.pcap.name,
        aggregation=aggregation,
        suspicion=suspicion,
        attack_threshold=args.attack_threshold,
        attack_prob_threshold=args.attack_prob_threshold,
        high_conf_threshold=args.high_confidence_threshold,
        min_attack_windows=args.min_attack_windows,
        min_high_conf_windows=args.min_high_confidence_windows,
    )
    write_json_summary(json_path, summary_payload)

    for line in format_verdict_banner(args.pcap.name, aggregation):
        LOGGER.info(line)
    LOGGER.info("Top suspicious IPs: %s", ", ".join(suspicion.ip_addresses) or "None")
    LOGGER.info("Top suspicious MACs: %s", ", ".join(suspicion.mac_addresses) or "None")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint wrapper.

    Args:
        argv: Optional sequence overriding ``sys.argv``.
    """

    args = parse_args(argv)
    run(args)


__all__ = ["main", "run"]
