import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.demo_objective1 import (  # noqa: E402
    DemoError,
    main,
    print_demo,
    run_demo,
)


class Objective1DemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {"question_id": 7, "question": "Selected legal question"},
            {"question_id": 8, "question": "Other legal question"},
        ]
        self.responses = [
            "response one",
            "response two",
            "response three",
            "response four",
            "response five",
        ]

    def _write_dataset(self, directory: str) -> Path:
        path = Path(directory) / "dataset.json"
        path.write_text(json.dumps(self.records), encoding="utf-8")
        return path

    @patch(
        "experiments.demo_objective1.individual_response_metrics",
        return_value={
            "consistency": [0.8, 0.7, 0.9, 0.6, 0.75],
            "uncertainty": [0.2, 0.3, 0.1, 0.4, 0.25],
        },
    )
    @patch(
        "experiments.demo_objective1.unique_pairwise_combined_similarities",
        return_value=[0.6, 0.8, 0.7],
    )
    @patch(
        "experiments.demo_objective1.unique_pairwise_semantic_similarities",
        return_value=[0.8, 0.9, 0.7],
    )
    @patch(
        "experiments.demo_objective1.unique_pairwise_similarities",
        return_value=[0.4, 0.5, 0.6],
    )
    @patch(
        "experiments.demo_objective1.generate_responses",
        return_value=[
            "response one",
            "response two",
            "response three",
            "response four",
            "response five",
        ],
    )
    @patch("experiments.demo_objective1.load_client")
    @patch("experiments.demo_objective1.random.choice")
    def test_selects_random_question_and_generates_five_responses(
        self,
        choice: object,
        load_client: object,
        generate: object,
        lexical: object,
        semantic: object,
        combined: object,
        individual: object,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_dataset(directory)
            choice.return_value = self.records[0]
            metrics = run_demo(path)

        choice.assert_called_once_with(self.records)
        load_client.assert_called_once()
        generate.assert_called_once()
        self.assertEqual(metrics["question_id"], 7)
        self.assertEqual(metrics["question"], "Selected legal question")
        self.assertEqual(metrics["responses"], self.responses)
        self.assertAlmostEqual(metrics["lexical_consistency"], 0.5)
        self.assertAlmostEqual(metrics["semantic_consistency"], 0.8)
        self.assertAlmostEqual(metrics["combined_consistency"], 0.7)
        self.assertAlmostEqual(metrics["uncertainty"], 0.3)

    @patch("experiments.demo_objective1.load_client")
    @patch("experiments.demo_objective1.random.choice")
    def test_api_error_is_reported_without_traceback(
        self, choice: object, load_client: object
    ) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_dataset(directory)
            choice.return_value = self.records[0]
            load_client.side_effect = ValueError("GROQ_API_KEY is not set")
            with patch("experiments.demo_objective1.parse_args") as parse_args:
                parse_args.return_value.dataset_path = path
                parse_args.return_value.model_name = "test-model"
                error = io.StringIO()
                with redirect_stderr(error):
                    result = main()

        self.assertEqual(result, 1)
        self.assertIsInstance(error.getvalue(), str)
        self.assertIn("Objective 1 demo failed", error.getvalue())
        self.assertIsInstance(DemoError("error"), DemoError)

    def test_print_contains_selected_question_and_completion(self) -> None:
        metrics = {
            "question_id": 7,
            "question": "Selected legal question",
            "responses": self.responses,
            "lexical_consistency": 0.5,
            "semantic_consistency": 0.6,
            "combined_consistency": 0.7,
            "individual_consistency": [0.8, 0.7, 0.9, 0.6, 0.75],
            "individual_uncertainty": [0.2, 0.3, 0.1, 0.4, 0.25],
            "uncertainty": 0.3,
        }
        output = io.StringIO()
        with redirect_stdout(output):
            print_demo(metrics)
        text = output.getvalue()
        self.assertIn("Selected question (ID 7)", text)
        self.assertIn("Highest individual uncertainty: response 4", text)
        self.assertTrue(text.rstrip().endswith("OBJECTIVE 1 DEMONSTRATION COMPLETE"))

    def test_invalid_response_count_raises(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_dataset(directory)
            with patch(
                "experiments.demo_objective1.random.choice",
                return_value=self.records[0],
            ), patch(
                "experiments.demo_objective1.generate_responses",
                return_value=["only one"],
            ):
                with self.assertRaises(ValueError):
                    run_demo(path)


if __name__ == "__main__":
    unittest.main()
