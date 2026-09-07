"""Lexical similarity utilities based on TF-IDF and cosine similarity."""

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _validate_responses(responses: Sequence[str]) -> None:
    if len(responses) < 2 or any(
        not isinstance(response, str) or not response.strip()
        for response in responses
    ):
        raise ValueError("at least two non-empty response strings are required")


def lexical_similarity_matrix(responses: Sequence[str]) -> np.ndarray:
    """Return the pairwise TF-IDF cosine similarity matrix for responses."""
    _validate_responses(responses)

    try:
        tfidf = TfidfVectorizer().fit_transform(responses)
    except ValueError as exc:
        raise ValueError("responses must contain at least one word") from exc

    matrix = cosine_similarity(tfidf)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def unique_pairwise_similarities(responses: Sequence[str]) -> list[float]:
    """Return each unique pairwise similarity score once, excluding the diagonal."""
    matrix = lexical_similarity_matrix(responses)
    row_indices, column_indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[row_indices, column_indices].tolist()
