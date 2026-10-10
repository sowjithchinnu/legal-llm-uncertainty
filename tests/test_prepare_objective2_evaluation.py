import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prepare_objective2_evaluation import (  # noqa: E402
    PreparationError,
    prepare_evaluation,
)


class Objective2PreparationTests(unittest.TestCase):
    def _write_inputs(
        self,
        directory: str,
        *,
        raw_records: list[dict[str, object]] | None = None,
        pilot_records: list[dict[str, object]] | None = None,
    ) -> tuple[Path, Path, Path]:
        root = Path(directory)
        raw_path = root / "raw.json"
        pilot_path = root / "pilot.json"
        output_path = root / "prepared.json"
        raw_path.write_text(json.dumps(raw_records or self.raw_records), encoding="utf-8")
        pilot_path.write_text(
            json.dumps(pilot_records or [self.pilot_record]), encoding="utf-8"
        )
        return raw_path, pilot_path, output_path

    def setUp(self) -> None:
        self.responses = [f"response {index}" for index in range(1, 6)]
        self.raw_records = [
            {"question": "Question one", "answer": "Reference answer one"},
            {"question": "Question two", "answer": "Reference answer two"},
        ]
        self.pilot_record = {
            "question_id": 4,
            "question": "Question one",
            "model_name": "test-model",
            "generation_config": {"num_responses": 5},
            "responses": self.responses,
            "lexical_consistency": 0.8,
            "semantic_consistency": 0.9,
            "combined_consistency": 0.85,
            "uncertainty": 0.15,
            "pairwise_std": 0.05,
        }

    def test_successful_preparation_and_annotation_structure(self) -> None:
        with TemporaryDirectory() as directory:
            raw_path, pilot_path, output_path = self._write_inputs(directory)
            prepared = prepare_evaluation(raw_path, pilot_path, output_path)

            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved, prepared)
        self.assertEqual(prepared[0]["reference_answer"], "Reference answer one")
        self.assertIn("not guaranteed ground truth", prepared[0]["reference_answer_note"])
        self.assertEqual(prepared[0]["combined_consistency"], 0.85)
        self.assertEqual(len(prepared[0]["responses"]), 5)
        for index, response in enumerate(prepared[0]["responses"], 1):
            self.assertEqual(response["response_id"], f"4-response-{index}")
            self.assertEqual(
                response["annotation"],
                {"correctness_label": "", "rationale": "", "reviewer_id": ""},
            )

    def test_missing_match_fails(self) -> None:
        with TemporaryDirectory() as directory:
            raw_path, pilot_path, output_path = self._write_inputs(
                directory,
                raw_records=[{"question": "Other", "answer": "Answer"}],
            )
            with self.assertRaisesRegex(PreparationError, "no exact raw-dataset match"):
                prepare_evaluation(raw_path, pilot_path, output_path)

    def test_duplicate_raw_match_fails(self) -> None:
        with TemporaryDirectory() as directory:
            raw_path, pilot_path, output_path = self._write_inputs(
                directory,
                raw_records=self.raw_records
                + [{"question": "Question one", "answer": "Duplicate"}],
            )
            with self.assertRaisesRegex(PreparationError, "exact matches"):
                prepare_evaluation(raw_path, pilot_path, output_path)

    def test_duplicate_pilot_ids_fail(self) -> None:
        with TemporaryDirectory() as directory:
            duplicate = dict(self.pilot_record)
            raw_path, pilot_path, output_path = self._write_inputs(
                directory, pilot_records=[self.pilot_record, duplicate]
            )
            with self.assertRaisesRegex(PreparationError, "duplicate pilot"):
                prepare_evaluation(raw_path, pilot_path, output_path)

    def test_missing_responses_fail(self) -> None:
        with TemporaryDirectory() as directory:
            incomplete = dict(self.pilot_record)
            incomplete["responses"] = self.responses[:-1]
            raw_path, pilot_path, output_path = self._write_inputs(
                directory, pilot_records=[incomplete]
            )
            with self.assertRaisesRegex(PreparationError, "exactly 5 responses"):
                prepare_evaluation(raw_path, pilot_path, output_path)

    def test_existing_output_is_not_overwritten(self) -> None:
        with TemporaryDirectory() as directory:
            raw_path, pilot_path, output_path = self._write_inputs(directory)
            output_path.write_text("original", encoding="utf-8")
            with self.assertRaisesRegex(PreparationError, "overwrite"):
                prepare_evaluation(raw_path, pilot_path, output_path)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
