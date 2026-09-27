import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.similarity.combined import (
    combined_similarity_matrix,
    mean_combined_similarity,
    unique_pairwise_combined_similarities,
)


RESPONSES = ["first response", "second response", "third response", "fourth response"]
LEXICAL = np.array(
    [
        [1.0, 0.2, 0.3, 0.4],
        [0.2, 1.0, 0.5, 0.6],
        [0.3, 0.5, 1.0, 0.7],
        [0.4, 0.6, 0.7, 1.0],
    ]
)
SEMANTIC = np.array(
    [
        [1.0, 0.8, 0.7, 0.6],
        [0.8, 1.0, 0.5, 0.4],
        [0.7, 0.5, 1.0, 0.3],
        [0.6, 0.4, 0.3, 1.0],
    ]
)


class CombinedSimilarityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.patches = (
            patch("src.similarity.combined.lexical_similarity_matrix", return_value=LEXICAL),
            patch("src.similarity.combined.semantic_similarity_matrix", return_value=SEMANTIC),
        )
        for patcher in self.patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patches):
            patcher.stop()

    def test_identical_responses_have_combined_similarity_one(self) -> None:
        with patch(
            "src.similarity.combined.lexical_similarity_matrix",
            return_value=np.ones((2, 2)),
        ), patch(
            "src.similarity.combined.semantic_similarity_matrix",
            return_value=np.ones((2, 2)),
        ):
            matrix = combined_similarity_matrix(["same response", "same response"])
        self.assertAlmostEqual(matrix[0, 1], 1.0)

    def test_matrix_is_symmetric(self) -> None:
        matrix = combined_similarity_matrix(RESPONSES)
        np.testing.assert_allclose(matrix, matrix.T)

    def test_diagonal_values_are_one(self) -> None:
        matrix = combined_similarity_matrix(RESPONSES)
        np.testing.assert_array_equal(np.diag(matrix), np.ones(len(RESPONSES)))

    def test_unique_scores_have_correct_count(self) -> None:
        scores = unique_pairwise_combined_similarities(RESPONSES)
        self.assertEqual(len(scores), 6)

    def test_default_weight_is_equal_weighting(self) -> None:
        matrix = combined_similarity_matrix(RESPONSES)
        np.testing.assert_allclose(matrix, (LEXICAL + SEMANTIC) / 2)

    def test_custom_weight_zero_uses_semantic_scores(self) -> None:
        np.testing.assert_allclose(combined_similarity_matrix(RESPONSES, 0), SEMANTIC)

    def test_custom_weight_one_uses_lexical_scores(self) -> None:
        np.testing.assert_allclose(combined_similarity_matrix(RESPONSES, 1), LEXICAL)

    def test_invalid_weights_raise(self) -> None:
        for weight in (-0.01, 1.01):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    combined_similarity_matrix(RESPONSES, weight)

    def test_mean_score_is_mean_of_unique_scores(self) -> None:
        expected = float(np.mean(((LEXICAL + SEMANTIC) / 2)[np.triu_indices(4, k=1)]))
        self.assertAlmostEqual(mean_combined_similarity(RESPONSES), expected)


if __name__ == "__main__":
    unittest.main()
