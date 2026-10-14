import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.validate_objective2_annotations import (  # noqa: E402
    validate_annotations,
)


class Objective2AnnotationValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = {
            "questions": [
                {
                    "question_id": 7,
                    "responses": [
                        {"response_id": f"7-response-{index}"}
                        for index in range(1, 3)
                    ],
                }
            ]
        }
        self.evaluation = [
            {
                "question_id": 7,
                "responses": [
                    {
                        "response_id": "7-response-1",
                        "annotation": {
                            "correctness_label": "",
                            "rationale": "",
                            "reviewer_id": "",
                        },
                    },
                    {
                        "response_id": "7-response-2",
                        "annotation": {
                            "correctness_label": "correct",
                            "rationale": "The answer states the applicable rule.",
                            "reviewer_id": "reviewer-a",
                        },
                    },
                ],
            }
        ]

    def _write_files(self, directory: str, evaluation: object) -> tuple[Path, Path]:
        root = Path(directory)
        evaluation_path = root / "evaluation.json"
        expected_path = root / "expected.json"
        evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
        expected_path.write_text(json.dumps(self.expected), encoding="utf-8")
        return evaluation_path, expected_path

    def test_valid_unreviewed_and_reviewed_annotations_pass(self) -> None:
        with TemporaryDirectory() as directory:
            evaluation_path, expected_path = self._write_files(
                directory, self.evaluation
            )
            self.assertEqual(
                validate_annotations(evaluation_path, expected_path), []
            )

    def test_invalid_label_and_incomplete_review_are_reported(self) -> None:
        invalid = json.loads(json.dumps(self.evaluation))
        invalid[0]["responses"][0]["annotation"] = {
            "correctness_label": "maybe",
            "rationale": "",
            "reviewer_id": "",
        }
        invalid[0]["responses"][1]["annotation"]["reviewer_id"] = ""
        with TemporaryDirectory() as directory:
            evaluation_path, expected_path = self._write_files(directory, invalid)
            issues = validate_annotations(evaluation_path, expected_path)
        self.assertTrue(any("invalid correctness_label" in issue for issue in issues))
        self.assertTrue(any("requires both" in issue for issue in issues))

    def test_missing_and_duplicate_ids_are_reported(self) -> None:
        invalid = [
            {
                "question_id": 7,
                "responses": [
                    {
                        "response_id": "7-response-1",
                        "annotation": {
                            "correctness_label": "",
                            "rationale": "",
                            "reviewer_id": "",
                        },
                    },
                    {
                        "response_id": "7-response-1",
                        "annotation": {
                            "correctness_label": "",
                            "rationale": "",
                            "reviewer_id": "",
                        },
                    },
                ],
            }
        ]
        with TemporaryDirectory() as directory:
            evaluation_path, expected_path = self._write_files(directory, invalid)
            issues = validate_annotations(evaluation_path, expected_path)
        self.assertTrue(any("duplicate response_id" in issue for issue in issues))
        self.assertTrue(any("missing response_id" in issue for issue in issues))

    def test_partial_unreviewed_fields_are_reported(self) -> None:
        invalid = json.loads(json.dumps(self.evaluation))
        invalid[0]["responses"][0]["annotation"]["rationale"] = "Needs review"
        with TemporaryDirectory() as directory:
            evaluation_path, expected_path = self._write_files(directory, invalid)
            issues = validate_annotations(evaluation_path, expected_path)
        self.assertTrue(any("unreviewed annotation" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
