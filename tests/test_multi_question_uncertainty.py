import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.multi_question_uncertainty import (  # noqa: E402
    ExperimentError,
    run_experiment,
)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        response_number = len(self.calls)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=f"response {response_number} legal answer"
                    )
                )
            ]
        )


class FakeClient:
    def __init__(self) -> None:
        completions = FakeCompletions()
        self.completions = completions
        self.chat = SimpleNamespace(completions=completions)


class MultiQuestionExperimentTests(unittest.TestCase):
    @patch(
        "experiments.multi_question_uncertainty.unique_pairwise_semantic_similarities",
        return_value=[0.8, 0.9, 1.0],
    )
    def test_processes_requested_questions_and_stores_metrics(
        self, semantic_scores: object
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            output_path = root / "processed" / "experiment.json"
            dataset_path.write_text(
                json.dumps(
                    [
                        {"question_id": "q-1", "question": "First question"},
                        {"question_id": "q-2", "question": "Second question"},
                        {"question_id": "q-3", "question": "Third question"},
                    ]
                ),
                encoding="utf-8",
            )

            client = FakeClient()
            with patch(
                "experiments.multi_question_uncertainty.unique_pairwise_combined_similarities",
                return_value=[0.5, 0.7, 0.9],
            ), patch(
                "experiments.multi_question_uncertainty.unique_pairwise_similarities",
                return_value=[0.4, 0.6, 0.8],
            ):
                results = run_experiment(
                    dataset_path,
                    output_path,
                    num_questions=2,
                    num_responses=3,
                    client=client,
                )

            self.assertEqual([result["question_id"] for result in results], ["q-1", "q-2"])
            self.assertEqual(len(results[0]["responses"]), 3)
            self.assertAlmostEqual(results[0]["combined_consistency"], 0.7)
            self.assertAlmostEqual(results[0]["uncertainty"], 0.3)
            self.assertAlmostEqual(results[0]["pairwise_std"], 0.16329931618554522)
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved, results)
            self.assertEqual(len(client.completions.calls), 6)

    def test_api_failure_does_not_save_incomplete_results(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            output_path = root / "experiment.json"
            dataset_path.write_text(
                json.dumps([{"question": "First question"}]), encoding="utf-8"
            )
            client = FakeClient()

            with patch(
                "experiments.multi_question_uncertainty.generate_responses",
                side_effect=RuntimeError("simulated API failure"),
            ):
                with self.assertRaises(ExperimentError):
                    run_experiment(dataset_path, output_path, client=client)

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
