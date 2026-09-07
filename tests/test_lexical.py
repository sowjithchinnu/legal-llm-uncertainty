import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.similarity.lexical import (
    lexical_similarity_matrix,
    unique_pairwise_similarities,
)


class LexicalSimilarityTests(unittest.TestCase):
    def test_identical_responses_have_similarity_one(self) -> None:
        matrix = lexical_similarity_matrix(["same legal answer", "same legal answer"])

        np.testing.assert_allclose(matrix, np.ones((2, 2)))

    def test_different_responses_have_lower_similarity(self) -> None:
        matrix = lexical_similarity_matrix(
            ["contract formation requires offer and acceptance", "quantum mechanics studies particles"]
        )

        self.assertLess(matrix[0, 1], 1.0)

    def test_matrix_is_symmetric(self) -> None:
        matrix = lexical_similarity_matrix(
            ["the court affirmed the judgment", "the court reversed the judgment", "a statute governs procedure"]
        )

        np.testing.assert_allclose(matrix, matrix.T)

    def test_diagonal_values_are_one(self) -> None:
        matrix = lexical_similarity_matrix(["first response", "second response", "third response"])

        np.testing.assert_allclose(np.diag(matrix), np.ones(3))

    def test_unique_scores_exclude_duplicates_and_self_comparisons(self) -> None:
        scores = unique_pairwise_similarities(["alpha beta", "alpha gamma", "delta epsilon"])

        self.assertEqual(len(scores), 3)

    def test_invalid_input_raises(self) -> None:
        for responses in ([], ["only one"], ["valid response", "   "]):
            with self.subTest(responses=responses):
                with self.assertRaises(ValueError):
                    lexical_similarity_matrix(responses)


if __name__ == "__main__":
    unittest.main()
