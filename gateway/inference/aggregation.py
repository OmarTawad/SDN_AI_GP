"""Aggregation logic for unified MoE inference results.


"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Sequence

from gateway.core import CLASS_LABELS, class_id_to_name


@dataclass(frozen=True)
class ClassAggregation:
    """Statistics summarising predictions for a single class."""

    label: str
    count: int
    fraction: float
    mean_probability: float
    high_conf_windows: int


@dataclass(frozen=True)
class AggregationSummary:
    """Container for per-class statistics and final verdict."""

    classes: Dict[str, ClassAggregation]
    total_windows: int
    verdict: str


def _compute_class_aggregation(
    predictions: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    high_conf_threshold: float,
) -> Dict[str, ClassAggregation]:
    """Generate per-class aggregates for predictions and probabilities.

    Args:
        predictions: Sequence of class identifiers.
        probabilities: Sequence of probability vectors per window.
        high_conf_threshold: Probability threshold considered high confidence.

    Returns:
        Dict[str, ClassAggregation]: Mapping of class label to aggregation statistics.
    """

    counts: Counter[str] = Counter()
    probability_sum: Counter[str] = Counter()
    high_conf_counts: Counter[str] = Counter()
    total_windows = len(predictions)
    safe_threshold = float(high_conf_threshold)

    for idx, class_id in enumerate(predictions):
        label = class_id_to_name(int(class_id))
        counts[label] += 1
        prob_vector = probabilities[idx]
        for class_index, class_name in enumerate(CLASS_LABELS):
            prob_value = float(prob_vector[class_index])
            probability_sum[class_name] += prob_value
            if prob_value >= safe_threshold:
                high_conf_counts[class_name] += 1

    classes: Dict[str, ClassAggregation] = {}
    for label in CLASS_LABELS:
        count = int(counts.get(label, 0))
        fraction = (count / total_windows) if total_windows else 0.0
        mean_probability = (
            float(probability_sum.get(label, 0.0)) / total_windows if total_windows else 0.0
        )
        high_conf = int(high_conf_counts.get(label, 0))
        classes[label] = ClassAggregation(
            label=label,
            count=count,
            fraction=fraction,
            mean_probability=mean_probability,
            high_conf_windows=high_conf,
        )
    return classes


def decide_verdict(
    class_stats: Dict[str, ClassAggregation],
    attack_threshold: float,
    attack_prob_threshold: float,
    min_attack_windows: int,
    min_high_conf_windows: int,
) -> str:
    """Determine the final verdict based on aggregate statistics.

    Args:
        class_stats: Per-class summarised statistics.
        attack_threshold: Minimum fraction of windows supporting an attack.
        attack_prob_threshold: Average probability threshold for escalation.
        min_attack_windows: Minimum attack windows required for verdict promotion.
        min_high_conf_windows: Minimum high-confidence windows required for promotion.

    Returns:
        str: Selected label.
    """

    total_windows = sum(stat.count for stat in class_stats.values())
    if total_windows == 0:
        return "normal"

    majority_label = max(class_stats.values(), key=lambda stat: stat.fraction).label
    majority_fraction = class_stats[majority_label].fraction
    if majority_label != "normal" and majority_fraction >= attack_threshold:
        return majority_label
    if majority_fraction >= attack_threshold:
        return majority_label

    probability_winner = max(class_stats.values(), key=lambda stat: stat.mean_probability)
    if probability_winner.label != "normal":
        prob_ok = probability_winner.mean_probability >= attack_prob_threshold
        support_ok = (
            probability_winner.count >= min_attack_windows
            or probability_winner.high_conf_windows >= min_high_conf_windows
        )
        if prob_ok and support_ok:
            return probability_winner.label

    best_candidate = None
    candidate_score = -1.0
    for label, stats in class_stats.items():
        if label == "normal":
            continue
        score = stats.fraction + stats.mean_probability
        if stats.high_conf_windows:
            score += stats.high_conf_windows / max(total_windows, 1)
        if score > candidate_score:
            candidate_score = score
            best_candidate = stats

    if not best_candidate:
        return "normal"

    meets_fraction = best_candidate.fraction >= attack_threshold
    meets_probability = (
        best_candidate.mean_probability >= attack_prob_threshold
        and best_candidate.count >= min_attack_windows
    )
    meets_high_conf = (
        best_candidate.mean_probability >= attack_prob_threshold
        and best_candidate.high_conf_windows >= min_high_conf_windows
    )
    if meets_fraction or meets_probability or meets_high_conf:
        return best_candidate.label
    return "normal"


def aggregate_predictions(
    predictions: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    high_conf_threshold: float,
    attack_threshold: float,
    attack_prob_threshold: float,
    min_attack_windows: int,
    min_high_conf_windows: int,
) -> AggregationSummary:
    """Aggregate per-window predictions and decide a verdict.

    Args:
        predictions: Sequence of predicted class identifiers.
        probabilities: Sequence of class probability vectors.
        high_conf_threshold: Probability threshold for a window to be considered high confidence.
        attack_threshold: Fraction of windows required to assert an attack verdict.
        attack_prob_threshold: Mean probability threshold for escalation.
        min_attack_windows: Minimum number of supporting windows to escalate.
        min_high_conf_windows: Minimum high-confidence windows for escalation.

    Returns:
        AggregationSummary: Summary containing per-class stats and final verdict.
    """

    class_stats = _compute_class_aggregation(predictions, probabilities, high_conf_threshold)
    verdict = decide_verdict(
        class_stats=class_stats,
        attack_threshold=attack_threshold,
        attack_prob_threshold=attack_prob_threshold,
        min_attack_windows=min_attack_windows,
        min_high_conf_windows=min_high_conf_windows,
    )
    total_windows = len(predictions)
    return AggregationSummary(classes=class_stats, total_windows=total_windows, verdict=verdict)


__all__ = [
    "AggregationSummary",
    "ClassAggregation",
    "aggregate_predictions",
    "decide_verdict",
]
