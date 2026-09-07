import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.similarity.semantic import (
    semantic_similarity_matrix,
    unique_pairwise_semantic_similarities,
)


class _FakeSentenceTransformer:
    _embeddings = {
        "The cat sat on the mat.": [1.0, 0.0, 0.0],
        "A cat was sitting on a mat.": [0.98, 0.2, 0.0],
        "The stock market closed higher today.": [0.0, 1.0, 0.0],
        "Quantum mechanics describes subatomic particles.": [0.0, 0.0, 1.0],
    }

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, responses, convert_to_numpy: bool = False):
        return np.array([self._embeddings[response] for response in responses])


class SemanticSimilarityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_patch = patch(
            "src.similarity.semantic.SentenceTransformer",
            _FakeSentenceTransformer,
        )
        self.model_patch.start()

    def tearDown(self) -> None:
        self.model_patch.stop()

    def test_identical_responses_have_similarity_one(self) -> None:
        matrix = semantic_similarity_matrix(
            ["The cat sat on the mat.", "The cat sat on the mat."]
        )

        self.assertAlmostEqual(matrix[0, 1], 1.0, places=6)

    def test_semantically_similar_responses_have_high_similarity(self) -> None:
        matrix = semantic_similarity_matrix(
            ["The cat sat on the mat.", "A cat was sitting on a mat."]
        )

        self.assertGreater(matrix[0, 1], 0.9)

    def test_unrelated_responses_have_lower_similarity(self) -> None:
        matrix = semantic_similarity_matrix(
            [
                "The cat sat on the mat.",
                "Quantum mechanics describes subatomic particles.",
            ]
        )

        self.assertLess(matrix[0, 1], 0.5)

    def test_matrix_is_symmetric(self) -> None:
        matrix = semantic_similarity_matrix(
            [
                "The cat sat on the mat.",
                "A cat was sitting on a mat.",
                "The stock market closed higher today.",
            ]
        )

        np.testing.assert_allclose(matrix, matrix.T)

    def test_diagonal_values_are_one(self) -> None:
        matrix = semantic_similarity_matrix(
            [
                "The cat sat on the mat.",
                "A cat was sitting on a mat.",
                "The stock market closed higher today.",
            ]
        )

        np.testing.assert_array_equal(np.diag(matrix), np.ones(3))

    def test_unique_scores_have_correct_count(self) -> None:
        scores = unique_pairwise_semantic_similarities(
            [
                "The cat sat on the mat.",
                "A cat was sitting on a mat.",
                "The stock market closed higher today.",
                "Quantum mechanics describes subatomic particles.",
            ]
        )

        self.assertEqual(len(scores), 6)

    def test_invalid_input_raises(self) -> None:
        for responses in ([], ["only one"], ["valid response", "   "]):
            with self.subTest(responses=responses):
                with self.assertRaises(ValueError):
                    semantic_similarity_matrix(responses)


if __name__ == "__main__":
    unittest.main()
