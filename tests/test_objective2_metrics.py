import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.objective2_metrics import (  # noqa: E402
    MetricsError,
    adaptive_ace,
    auroc,
    evaluate_metrics,
)


COMBINATIONS = ("lexical", "semantic", "combined")


def fixture_files(root: Path, labels: list[str], scores: list[float]) -> tuple[Path, Path]:
    responses = []
    annotated = []
    for index, (label, score) in enumerate(zip(labels, scores), 1):
        response_id = f"1-response-{index}"
        methods = {
            method: {
                aggregation: {"uncertainty": score}
                for aggregation in ("mean", "median")
            }
            for method in COMBINATIONS
        }
        responses.append({"response_id": response_id, "methods": methods})
        annotated.append({
            "response_id": response_id,
            "annotation": {
                "correctness_label": label,
                "rationale": "synthetic rationale" if label else "",
                "reviewer_id": "synthetic" if label else "",
            },
        })
    comparison = {"questions": [{"question_id": 1, "responses": responses}]}
    annotations = [{"question_id": 1, "responses": annotated}]
    comparison_path = root / "comparison.json"
    annotations_path = root / "annotations.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    return annotations_path, comparison_path


class Objective2MetricsTests(unittest.TestCase):
    def test_known_auroc_and_ace(self) -> None:
        self.assertEqual(auroc([0.9, 0.1], ["incorrect", "correct"]), 1.0)
        ace = adaptive_ace([0.0, 1.0], ["correct", "incorrect"], max_bins=2)
        self.assertIsNotNone(ace)
        self.assertEqual(ace["bin_count"], 2)
        self.assertAlmostEqual(ace["value"], 0.0)

    def test_adaptive_ace_handles_small_samples_and_empty_class(self) -> None:
        ace = adaptive_ace([0.2], ["correct"], max_bins=10)
        self.assertIsNone(ace)
        ace = adaptive_ace([0.2, 0.3], ["correct", "incorrect"], max_bins=10)
        self.assertEqual(ace["bin_count"], 2)
        with self.assertRaises(ValueError):
            adaptive_ace([0.2], ["partially_correct"])

    def test_label_filtering_and_insufficient_classes(self) -> None:
        with TemporaryDirectory() as directory:
            annotations, comparison = fixture_files(
                Path(directory),
                ["correct", "partially_correct", "uncertain", ""],
                [0.1, 0.4, 0.5, 0.9],
            )
            result = evaluate_metrics(annotations, comparison)
        self.assertIn("insufficient reviewed labels", result["status"])
        self.assertEqual(result["counts"]["partially_correct"], 1)
        self.assertEqual(result["counts"]["uncertain"], 1)
        self.assertEqual(result["counts"]["unreviewed"], 1)
        self.assertIsNone(result["methods"]["lexical_mean"]["auroc"])

    def test_metrics_are_computed_for_both_binary_classes(self) -> None:
        with TemporaryDirectory() as directory:
            annotations, comparison = fixture_files(
                Path(directory), ["correct", "incorrect", "correct", "incorrect"],
                [0.1, 0.9, 0.2, 0.8],
            )
            result = evaluate_metrics(annotations, comparison, max_bins=2)
        self.assertEqual(result["status"], "sufficient reviewed labels")
        self.assertEqual(result["methods"]["lexical_mean"]["auroc"], 1.0)
        self.assertEqual(result["methods"]["lexical_mean"]["ace"]["bin_count"], 2)

    def test_invalid_scores_and_id_mismatches_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            annotations, comparison = fixture_files(Path(directory), ["correct", "incorrect"], [0.1, 0.9])
            data = json.loads(comparison.read_text(encoding="utf-8"))
            data["questions"][0]["responses"][0]["methods"]["lexical"]["mean"]["uncertainty"] = 2
            comparison.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(MetricsError, "in \\[0, 1\\]"):
                evaluate_metrics(annotations, comparison)

            data["questions"][0]["responses"][0]["methods"]["lexical"]["mean"]["uncertainty"] = 0.1
            data["questions"][0]["responses"][0]["response_id"] = "wrong-id"
            comparison.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(MetricsError, "missing annotation IDs"):
                evaluate_metrics(annotations, comparison)


if __name__ == "__main__":
    unittest.main()
