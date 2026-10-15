import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.annotate_objective2 import (  # noqa: E402
    annotate,
    load_or_initialize_output,
    load_prepared,
    save_annotations,
)


def synthetic_records() -> list[dict[str, object]]:
    return [
        {
            "question_id": 1,
            "question": "Who is the respondent?",
            "reference_answer": "The respondent is A.",
            "responses": [
                {"response_id": "1-response-1", "response": "A."},
                {"response_id": "1-response-2", "response": "B."},
            ],
        }
    ]


class Objective2AnnotationTests(unittest.TestCase):
    def test_initialize_and_atomic_save_keep_source_separate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_path = root / "prepared.json"
            output_path = root / "annotations.json"
            source = synthetic_records()
            prepared_path.write_text(json.dumps(source), encoding="utf-8")

            prepared = load_prepared(prepared_path)
            records = load_or_initialize_output(prepared, output_path)
            records[0]["responses"][0]["annotation"] = {
                "correctness_label": "correct",
                "rationale": "Matches the reference.",
                "reviewer_id": "reviewer-1",
            }
            save_annotations(records, output_path)

            self.assertNotIn("annotation", source[0]["responses"][0])
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8"))[0]["responses"][0][
                    "annotation"
                ]["correctness_label"],
                "correct",
            )

    def test_annotation_resumes_and_saves_each_response(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_path = root / "prepared.json"
            output_path = root / "annotations.json"
            prepared_path.write_text(json.dumps(synthetic_records()), encoding="utf-8")
            answers = iter(
                ["correct", "Matches.", "reviewer-1", "incorrect", "Wrong party.", "reviewer-1"]
            )
            records = annotate(
                prepared_path,
                output_path,
                input_fn=lambda _prompt: next(answers),
                output=lambda _message: None,
            )
            self.assertEqual(records[0]["responses"][1]["annotation"]["correctness_label"], "incorrect")

            resumed = annotate(
                prepared_path,
                output_path,
                input_fn=lambda _prompt: (_ for _ in ()).throw(
                    AssertionError("resume should not prompt")
                ),
                output=lambda _message: None,
            )
            self.assertEqual(resumed, records)

    def test_mismatched_resume_file_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_path = root / "prepared.json"
            output_path = root / "annotations.json"
            prepared_path.write_text(json.dumps(synthetic_records()), encoding="utf-8")
            output_path.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                annotate(
                    prepared_path,
                    output_path,
                    input_fn=lambda _prompt: "",
                    output=lambda _message: None,
                )


if __name__ == "__main__":
    unittest.main()
