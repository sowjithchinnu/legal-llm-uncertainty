"""Uncertainty metrics derived from pairwise combined similarities."""

from collections.abc import Sequence

import numpy as np


def _validated_similarities(
    pairwise_similarities: Sequence[float],
) -> np.ndarray:
    try:
        values = np.asarray(list(pairwise_similarities), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("pairwise similarities must be numeric") from exc

    if values.ndim != 1 or values.size == 0:
        raise ValueError("at least one pairwise similarity is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("pairwise similarities must be finite")
    return values


def mean_consistency(pairwise_similarities: Sequence[float]) -> float:
    """Return the mean of the pairwise combined similarities."""
    values = _validated_similarities(pairwise_similarities)
    return float(np.mean(values))


def uncertainty(pairwise_similarities: Sequence[float]) -> float:
    """Return uncertainty as one minus mean consistency."""
    return 1.0 - mean_consistency(pairwise_similarities)


def standard_deviation(pairwise_similarities: Sequence[float]) -> float:
    """Return the population standard deviation of pairwise similarities."""
    values = _validated_similarities(pairwise_similarities)
    return float(np.std(values))


def calculate_uncertainty(
    pairwise_similarities: Sequence[float],
) -> dict[str, float]:
    """Return mean consistency, uncertainty, and standard deviation together."""
    values = _validated_similarities(pairwise_similarities)
    mean = float(np.mean(values))
    return {
        "mean_consistency": mean,
        "uncertainty": 1.0 - mean,
        "standard_deviation": float(np.std(values)),
    }
