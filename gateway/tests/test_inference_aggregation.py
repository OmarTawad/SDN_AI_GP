"""Unit tests for the inference aggregation utilities."""

from __future__ import annotations

import math

import pytest

from gateway.inference.aggregation import aggregate_predictions


def test_aggregate_predictions_promotes_attack() -> None:
    """An attack class with strong support should be promoted."""

    predictions = [1] * 60 + [0] * 40
    probability_vectors = [[0.1, 0.8, 0.1]] * 60 + [[0.8, 0.1, 0.1]] * 40
    summary = aggregate_predictions(
        predictions=predictions,
        probabilities=probability_vectors,
        high_conf_threshold=0.75,
        attack_threshold=0.25,
        attack_prob_threshold=0.4,
        min_attack_windows=20,
        min_high_conf_windows=10,
    )
    assert summary.verdict == "dos"
    dos_stats = summary.classes["dos"]
    assert math.isclose(dos_stats.fraction, 0.6)
    assert dos_stats.high_conf_windows >= 60


def test_aggregate_predictions_validates_inputs() -> None:
    """Mismatched collection lengths should raise a descriptive error."""

    predictions = [0, 1]
    probability_vectors = [[0.9, 0.05, 0.05]]
    with pytest.raises(ValueError):
        aggregate_predictions(
            predictions=predictions,
            probabilities=probability_vectors,
            high_conf_threshold=0.5,
            attack_threshold=0.2,
            attack_prob_threshold=0.4,
            min_attack_windows=1,
            min_high_conf_windows=1,
        )
