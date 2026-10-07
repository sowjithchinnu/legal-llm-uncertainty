"""Consistency and uncertainty scores for individual responses."""

from collections.abc import Sequence

import numpy as np

from src.similarity.combined import (
    _validate_lexical_weight,
    combined_similarity_matrix,
)


def _validate_responses(responses: Sequence[str]) -> None:
    if len(responses) < 2 or any(
        not isinstance(response, str) or not response.strip()
        for response in responses
    ):
        raise ValueError("at least two non-empty response strings are required")


def _individual_consistency(
    responses: Sequence[str], lexical_weight: float
) -> np.ndarray:
    _validate_responses(responses)
    _validate_lexical_weight(lexical_weight)
    matrix = np.asarray(
        combined_similarity_matrix(responses, lexical_weight), dtype=float
    )
    if matrix.shape != (len(responses), len(responses)):
        raise ValueError("combined similarity matrix has an invalid shape")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("combined similarities must be finite")

    non_diagonal = ~np.eye(len(responses), dtype=bool)
    scores = matrix[non_diagonal].reshape(len(responses), len(responses) - 1)
    return np.clip(np.mean(scores, axis=1), 0.0, 1.0)


def individual_response_consistency(
    responses: Sequence[str], lexical_weight: float = 0.5
) -> list[float]:
    """Return each response's mean combined similarity to the other responses."""
    return _individual_consistency(responses, lexical_weight).tolist()


def individual_response_uncertainty(
    responses: Sequence[str], lexical_weight: float = 0.5
) -> list[float]:
    """Return one minus each response's individual consistency score."""
    consistency = _individual_consistency(responses, lexical_weight)
    return (1.0 - consistency).tolist()


def individual_response_metrics(
    responses: Sequence[str], lexical_weight: float = 0.5
) -> dict[str, list[float]]:
    """Return per-response consistency and uncertainty in input order."""
    consistency = _individual_consistency(responses, lexical_weight)
    return {
        "consistency": consistency.tolist(),
        "uncertainty": (1.0 - consistency).tolist(),
    }
