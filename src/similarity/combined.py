"""Combined lexical and semantic similarity utilities."""

from collections.abc import Sequence

import numpy as np

from src.similarity.lexical import lexical_similarity_matrix
from src.similarity.semantic import semantic_similarity_matrix


def _validate_lexical_weight(lexical_weight: float) -> None:
    try:
        valid = 0 <= lexical_weight <= 1
    except TypeError as exc:
        raise ValueError("lexical_weight must be between 0 and 1") from exc

    if not valid:
        raise ValueError("lexical_weight must be between 0 and 1")


def combined_similarity_matrix(
    responses: Sequence[str], lexical_weight: float = 0.5
) -> np.ndarray:
    """Return the weighted element-wise combination of lexical and semantic scores."""
    _validate_lexical_weight(lexical_weight)
    lexical = lexical_similarity_matrix(responses)
    semantic = semantic_similarity_matrix(responses)
    combined = lexical_weight * lexical + (1 - lexical_weight) * semantic
    np.fill_diagonal(combined, 1.0)
    return combined


def unique_pairwise_combined_similarities(
    responses: Sequence[str], lexical_weight: float = 0.5
) -> list[float]:
    """Return each unique upper-triangle combined score once."""
    matrix = combined_similarity_matrix(responses, lexical_weight)
    row_indices, column_indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[row_indices, column_indices].tolist()


def mean_combined_similarity(
    responses: Sequence[str], lexical_weight: float = 0.5
) -> float:
    """Return the mean of the unique pairwise combined scores."""
    scores = unique_pairwise_combined_similarities(responses, lexical_weight)
    return float(np.mean(scores))
