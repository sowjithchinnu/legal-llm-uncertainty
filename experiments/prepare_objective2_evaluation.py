"""Prepare pilot responses for manual Objective 2 correctness evaluation."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.storage.response_store import load_results


DEFAULT_DATASET_PATH = Path("data/raw/IndicLegalQA Dataset_10K_Revised.json")
DEFAULT_PILOT_PATH = Path("data/processed/multi_question_uncertainty.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/objective2_manual_evaluation.json")
REQUIRED_METRICS = (
    "lexical_consistency",
    "semantic_consistency",
    "combined_consistency",
    "uncertainty",
    "pairwise_std",
)
RESPONSE_COUNT = 5


class PreparationError(ValueError):
    """Raised when pilot data cannot be prepared for manual evaluation."""


def _load_json_records(path: str | Path, description: str) -> list[dict[str, Any]]:
    try:
        records = load_results(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise PreparationError(f"could not load {description}: {path}") from exc
    if not all(isinstance(record, dict) for record in records):
        raise PreparationError(f"{description} must contain JSON objects")
    return records


def _load_raw_records(path: str | Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"could not load raw dataset: {path}") from exc
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise PreparationError("raw dataset must contain a JSON array of objects")
    return data


def _raw_record_by_question(
    raw_records: list[dict[str, Any]],
    target_questions: set[str],
) -> dict[str, dict[str, Any]]:
    matches: dict[str, list[dict[str, Any]]] = {}
    for record in raw_records:
        question = record.get("question")
        if isinstance(question, str) and question in target_questions:
            matches.setdefault(question, []).append(record)

    unique_records: dict[str, dict[str, Any]] = {}
    for question, records in matches.items():
        if len(records) != 1:
            raise PreparationError(
                f"raw dataset question has {len(records)} exact matches: {question}"
            )
        unique_records[question] = records[0]
    return unique_records


def _validate_pilot_record(
    record: dict[str, Any],
    seen_ids: set[str | int],
    index: int,
) -> tuple[str | int, str, list[str]]:
    question_id = record.get("question_id")
    if not isinstance(question_id, (str, int)) or isinstance(question_id, bool):
        raise PreparationError(f"pilot record {index} has an invalid question_id")
    if question_id in seen_ids:
        raise PreparationError(f"duplicate pilot question_id: {question_id}")
    seen_ids.add(question_id)

    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise PreparationError(f"pilot record {index} has no valid question")

    responses = record.get("responses")
    if not isinstance(responses, list) or len(responses) != RESPONSE_COUNT:
        raise PreparationError(
            f"pilot question {question_id} must contain exactly "
            f"{RESPONSE_COUNT} responses"
        )
    if any(not isinstance(response, str) or not response.strip() for response in responses):
        raise PreparationError(f"pilot question {question_id} has a missing response")
    return question_id, question, responses


def prepare_evaluation(
    raw_dataset_path: str | Path = DEFAULT_DATASET_PATH,
    pilot_path: str | Path = DEFAULT_PILOT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, Any]]:
    """Create manual annotation records from existing pilot data only."""
    raw_records = _load_raw_records(raw_dataset_path)
    pilot_records = _load_json_records(pilot_path, "pilot results")
    if not pilot_records:
        raise PreparationError("pilot results are empty")

    prepared: list[dict[str, Any]] = []
    seen_ids: set[str | int] = set()
    target_questions: set[str] = set()
    validated_pilot: list[tuple[str | int, str, list[str], dict[str, Any]]] = []
    for index, pilot in enumerate(pilot_records):
        question_id, question, responses = _validate_pilot_record(
            pilot, seen_ids, index
        )
        target_questions.add(question)
        validated_pilot.append((question_id, question, responses, pilot))

    raw_by_question = _raw_record_by_question(raw_records, target_questions)
    for question_id, question, responses, pilot in validated_pilot:
        raw_record = raw_by_question.get(question)
        if raw_record is None:
            raise PreparationError(
                f"no exact raw-dataset match for pilot question_id {question_id}"
            )
        reference_answer = raw_record.get("answer")
        if not isinstance(reference_answer, str) or not reference_answer.strip():
            raise PreparationError(
                f"raw record for question_id {question_id} has no reference answer"
            )

        response_records = [
            {
                "response_id": f"{question_id}-response-{response_index}",
                "response": response,
                "annotation": {
                    "correctness_label": "",
                    "rationale": "",
                    "reviewer_id": "",
                },
            }
            for response_index, response in enumerate(responses, 1)
        ]
        result = {
            "question_id": question_id,
            "question": question,
            "reference_answer": reference_answer,
            "reference_answer_note": (
                "Reference answer copied from IndicLegalQA; it is not guaranteed "
                "ground truth."
            ),
            "responses": response_records,
        }
        for metric in REQUIRED_METRICS:
            if metric in pilot:
                result[metric] = pilot[metric]
        if "model_name" in pilot:
            result["model_name"] = pilot["model_name"]
        if "generation_config" in pilot:
            result["generation_config"] = pilot["generation_config"]
        prepared.append(result)

    output = Path(output_path)
    if output.exists():
        raise PreparationError(f"refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(prepared, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--pilot-path", type=Path, default=DEFAULT_PILOT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = prepare_evaluation(
        raw_dataset_path=args.raw_dataset_path,
        pilot_path=args.pilot_path,
        output_path=args.output_path,
    )
    print(f"Saved {len(records)} evaluation record(s) to {args.output_path}")


if __name__ == "__main__":
    main()
