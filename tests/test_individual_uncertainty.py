import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.uncertainty.individual import (  # noqa: E402
    individual_response_consistency,
    individual_response_metrics,
    individual_response_uncertainty,
)


RESPONSES = ["first response", "second response", "third response"]
COMBINED = np.array(
    [
        [1.0, 0.8, 0.4],
        [0.8, 1.0, 0.6],
        [0.4, 0.6, 1.0],
    ]
)


class IndividualUncertaintyTests(unittest.TestCase):
    def test_identical_responses_have_zero_uncertainty(self) -> None:
        with patch(
            "src.uncertainty.individual.combined_similarity_matrix",
            return_value=np.ones((3, 3)),
        ):
            consistency = individual_response_consistency(["a", "a", "a"])
            uncertainty = individual_response_uncertainty(["a", "a", "a"])
        np.testing.assert_allclose(consistency, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(uncertainty, [0.0, 0.0, 0.0])

    def test_different_responses_produce_different_scores(self) -> None:
        with patch(
            "src.uncertainty.individual.combined_similarity_matrix",
            return_value=COMBINED,
        ):
            consistency = individual_response_consistency(RESPONSES)
            uncertainty = individual_response_uncertainty(RESPONSES)
        np.testing.assert_allclose(consistency, [0.6, 0.7, 0.5])
        np.testing.assert_allclose(uncertainty, [0.4, 0.3, 0.5])

    def test_returns_one_score_per_response_in_order(self) -> None:
        with patch(
            "src.uncertainty.individual.combined_similarity_matrix",
            return_value=COMBINED,
        ):
            metrics = individual_response_metrics(RESPONSES)
        self.assertEqual(len(metrics["consistency"]), len(RESPONSES))
        self.assertEqual(len(metrics["uncertainty"]), len(RESPONSES))
        self.assertEqual(metrics["consistency"], [0.6000000000000001, 0.7, 0.5])

    def test_excludes_diagonal(self) -> None:
        matrix = np.full((3, 3), 0.2)
        np.fill_diagonal(matrix, 1.0)
        with patch(
            "src.uncertainty.individual.combined_similarity_matrix",
            return_value=matrix,
        ):
            scores = individual_response_consistency(RESPONSES)
        np.testing.assert_allclose(scores, [0.2, 0.2, 0.2])

    def test_symmetric_matrix_has_expected_row_means(self) -> None:
        with patch(
            "src.uncertainty.individual.combined_similarity_matrix",
            return_value=COMBINED,
        ):
            scores = individual_response_consistency(RESPONSES)
        np.testing.assert_allclose(scores, [0.6, 0.7, 0.5])

    def test_custom_weights_are_forwarded(self) -> None:
        with patch(
            "src.uncertainty.individual.combined_similarity_matrix",
            return_value=COMBINED,
        ) as combined:
            individual_response_consistency(RESPONSES, lexical_weight=0)
            individual_response_consistency(RESPONSES, lexical_weight=1)
        self.assertEqual(
            [call.args[1] for call in combined.call_args_list],
            [0, 1],
        )

    def test_invalid_inputs_raise(self) -> None:
        for responses in ([], ["only one"], ["valid", "  "]):
            with self.subTest(responses=responses):
                with self.assertRaises(ValueError):
                    individual_response_consistency(responses)

    def test_invalid_weights_raise(self) -> None:
        for weight in (-0.01, 1.01):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    individual_response_consistency(RESPONSES, weight)


if __name__ == "__main__":
    unittest.main()
