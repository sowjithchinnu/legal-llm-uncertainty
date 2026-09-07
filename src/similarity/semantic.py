"""Semantic similarity utilities based on sentence-transformer embeddings."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - exercised when the dependency is absent
    SentenceTransformer = None


_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _validate_responses(responses: Sequence[str]) -> None:
    if len(responses) < 2 or any(
        not isinstance(response, str) or not response.strip()
        for response in responses
    ):
        raise ValueError("at least two non-empty response strings are required")


def _embed_responses(responses: Sequence[str]) -> np.ndarray:
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required for semantic similarity"
        )

    model = SentenceTransformer(_MODEL_NAME)
    return np.asarray(model.encode(list(responses), convert_to_numpy=True))


def semantic_similarity_matrix(responses: Sequence[str]) -> np.ndarray:
    """Return the pairwise cosine similarity matrix for response embeddings."""
    _validate_responses(responses)
    matrix = cosine_similarity(_embed_responses(responses))
    np.fill_diagonal(matrix, 1.0)
    return matrix


def unique_pairwise_semantic_similarities(responses: Sequence[str]) -> list[float]:
    """Return each unique pairwise semantic similarity once."""
    matrix = semantic_similarity_matrix(responses)
    row_indices, column_indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[row_indices, column_indices].tolist()
