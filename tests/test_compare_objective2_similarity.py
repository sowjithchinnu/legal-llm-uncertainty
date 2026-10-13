import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.compare_objective2_similarity import (  # noqa: E402
    ComparisonError,
    compare_pilot,
    individual_scores_from_matrix,
)


class Objective2SimilarityComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = np.array(
            [
                [1.0, 0.2, 0.8, 0.4, 0.6],
                [0.2, 1.0, 0.4, 0.6, 0.8],
                [0.8, 0.4, 1.0, 0.3, 0.5],
                [0.4, 0.6, 0.3, 1.0, 0.7],
                [0.6, 0.8, 0.5, 0.7, 1.0],
            ]
        )
        self.records = [
            {
                "question_id": 3,
                "question": "Question",
                "responses": [
                    {
                        "response_id": f"3-response-{index}",
                        "response": f"Response {index}",
                        "annotation": {
                            "correctness_label": "",
                            "rationale": "",
                            "reviewer_id": "",
                        },
                    }
                    for index in range(1, 6)
                ],
            }
        ]

    def test_mean_and_median_exclude_diagonal_and_preserve_order(self) -> None:
        mean, mean_uncertainty = individual_scores_from_matrix(self.matrix, "mean")
        median_scores, median_uncertainty = individual_scores_from_matrix(
            self.matrix, "median"
        )
        np.testing.assert_allclose(mean, [0.5, 0.5, 0.5, 0.5, 0.65])
        np.testing.assert_allclose(mean_uncertainty, [0.5, 0.5, 0.5, 0.5, 0.35])
        np.testing.assert_allclose(median_scores, [0.5, 0.5, 0.45, 0.5, 0.65])
        np.testing.assert_allclose(median_uncertainty, [0.5, 0.5, 0.55, 0.5, 0.35])

    def test_invalid_matrix_and_aggregation_raise(self) -> None:
        with self.assertRaises(ValueError):
            individual_scores_from_matrix(np.ones((2, 3)), "mean")
        with self.assertRaises(ValueError):
            individual_scores_from_matrix(np.ones((2, 2)), "mode")

    @patch(
        "experiments.compare_objective2_similarity._method_matrix",
        return_value=np.array(
            [
                [1.0, 0.2, 0.8, 0.4, 0.6],
                [0.2, 1.0, 0.4, 0.6, 0.8],
                [0.8, 0.4, 1.0, 0.3, 0.5],
                [0.4, 0.6, 0.3, 1.0, 0.7],
                [0.6, 0.8, 0.5, 0.7, 1.0],
            ]
        ),
    )
    def test_output_structure_and_response_ids(self, method_matrix: object) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "output.json"
            input_path.write_text(json.dumps(self.records), encoding="utf-8")
            result = compare_pilot(input_path, output_path)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved, result)
        self.assertEqual(result["question_count"], 1)
        self.assertEqual(
            [r["response_id"] for r in result["questions"][0]["responses"]],
            [
                "3-response-1",
                "3-response-2",
                "3-response-3",
                "3-response-4",
                "3-response-5",
            ],
        )
        self.assertEqual(
            set(result["summary_statistics"]),
            {
                "lexical_mean",
                "lexical_median",
                "semantic_mean",
                "semantic_median",
                "combined_mean",
                "combined_median",
            },
        )
        self.assertTrue(result["questions"][0]["descriptive_only"])

    def test_duplicate_question_ids_and_missing_response_fail(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "output.json"
            duplicate = dict(self.records[0])
            input_path.write_text(
                json.dumps(self.records + [duplicate]), encoding="utf-8"
            )
            with self.assertRaisesRegex(ComparisonError, "duplicate question_id"):
                compare_pilot(input_path, output_path)

            incomplete = dict(self.records[0])
            incomplete["responses"] = self.records[0]["responses"][:2]
            input_path.write_text(json.dumps([incomplete]), encoding="utf-8")
            with self.assertRaisesRegex(ComparisonError, "exactly five"):
                compare_pilot(input_path, output_path)


if __name__ == "__main__":
    unittest.main()
