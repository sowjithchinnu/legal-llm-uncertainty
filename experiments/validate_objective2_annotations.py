"""Validate manual Objective 2 correctness annotations without modifying data."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_EVALUATION_PATH = Path("data/processed/objective2_manual_evaluation.json")
DEFAULT_EXPECTED_PATH = Path(
    "data/processed/objective2_similarity_comparison.json"
)
VALID_LABELS = frozenset(
    {"correct", "incorrect", "partially_correct", "uncertain"}
)
ANNOTATION_FIELDS = ("correctness_label", "rationale", "reviewer_id")


def _load_json(path: str | Path, description: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load {description}: {path}") from exc


def _expected_ids(path: str | Path) -> tuple[set[str | int], set[str]]:
    data = _load_json(path, "expected comparison output")
    questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(questions, list):
        raise ValueError("expected comparison output must contain a questions array")

    question_ids: set[str | int] = set()
    response_ids: set[str] = set()
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError(f"expected question {index} is not an object")
        question_id = question.get("question_id")
        if not isinstance(question_id, (str, int)) or isinstance(question_id, bool):
            raise ValueError(f"expected question {index} has an invalid question_id")
        question_ids.add(question_id)
        responses = question.get("responses")
        if not isinstance(responses, list):
            raise ValueError(f"expected question_id {question_id} has no responses")
        for response in responses:
            if isinstance(response, dict) and isinstance(
                response.get("response_id"), str
            ):
                response_ids.add(response["response_id"])
            else:
                raise ValueError(
                    f"expected question_id {question_id} has an invalid response_id"
                )
    return question_ids, response_ids


def validate_annotations(
    evaluation_path: str | Path = DEFAULT_EVALUATION_PATH,
    expected_path: str | Path | None = DEFAULT_EXPECTED_PATH,
) -> list[str]:
    """Return all validation issues; return an empty list when valid."""
    data = _load_json(evaluation_path, "manual evaluation data")
    if not isinstance(data, list):
        return ["manual evaluation data must be a JSON array"]

    issues: list[str] = []
    expected_questions: set[str | int] | None = None
    expected_responses: set[str] | None = None
    if expected_path is not None:
        try:
            expected_questions, expected_responses = _expected_ids(expected_path)
        except ValueError as exc:
            return [str(exc)]

    actual_questions: set[str | int] = set()
    actual_responses: set[str] = set()
    for record_index, record in enumerate(data):
        prefix = f"record {record_index}"
        if not isinstance(record, dict):
            issues.append(f"{prefix}: must be an object")
            continue

        question_id = record.get("question_id")
        if not isinstance(question_id, (str, int)) or isinstance(question_id, bool):
            issues.append(f"{prefix}: invalid question_id")
        elif question_id in actual_questions:
            issues.append(f"{prefix}: duplicate question_id {question_id!r}")
        else:
            actual_questions.add(question_id)

        responses = record.get("responses")
        if not isinstance(responses, list):
            issues.append(f"{prefix}: responses must be an array")
            continue

        local_response_ids: set[str] = set()
        for response_index, response_record in enumerate(responses, 1):
            response_prefix = f"{prefix} response {response_index}"
            if not isinstance(response_record, dict):
                issues.append(f"{response_prefix}: must be an object")
                continue
            response_id = response_record.get("response_id")
            if not isinstance(response_id, str) or not response_id.strip():
                issues.append(f"{response_prefix}: invalid response_id")
            else:
                if response_id in local_response_ids:
                    issues.append(f"{response_prefix}: duplicate response_id {response_id!r}")
                local_response_ids.add(response_id)
                if response_id in actual_responses:
                    issues.append(f"{response_prefix}: response_id reused {response_id!r}")
                actual_responses.add(response_id)

            annotation = response_record.get("annotation")
            if not isinstance(annotation, dict):
                issues.append(f"{response_prefix}: annotation must be an object")
                continue
            missing_fields = [
                field for field in ANNOTATION_FIELDS if field not in annotation
            ]
            if missing_fields:
                issues.append(
                    f"{response_prefix}: missing annotation field(s): "
                    + ", ".join(missing_fields)
                )
                continue

            label = annotation["correctness_label"]
            rationale = annotation["rationale"]
            reviewer_id = annotation["reviewer_id"]
            if not all(isinstance(value, str) for value in (label, rationale, reviewer_id)):
                issues.append(
                    f"{response_prefix}: annotation fields must all be strings"
                )
                continue
            if label == "":
                if rationale.strip() or reviewer_id.strip():
                    issues.append(
                        f"{response_prefix}: unreviewed annotation must leave "
                        "rationale and reviewer_id blank"
                    )
            elif label not in VALID_LABELS:
                issues.append(
                    f"{response_prefix}: invalid correctness_label {label!r}; "
                    f"expected one of {sorted(VALID_LABELS)} or blank"
                )
            elif not rationale.strip() or not reviewer_id.strip():
                issues.append(
                    f"{response_prefix}: reviewed annotation requires both "
                    "rationale and reviewer_id"
                )

    if expected_questions is not None:
        missing = expected_questions - actual_questions
        unexpected = actual_questions - expected_questions
        if missing:
            issues.append(f"missing question_id(s): {sorted(missing, key=str)}")
        if unexpected:
            issues.append(f"unexpected question_id(s): {sorted(unexpected, key=str)}")
    if expected_responses is not None:
        missing = expected_responses - actual_responses
        unexpected = actual_responses - expected_responses
        if missing:
            issues.append(f"missing response_id(s): {sorted(missing)}")
        if unexpected:
            issues.append(f"unexpected response_id(s): {sorted(unexpected)}")
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-path", type=Path, default=DEFAULT_EVALUATION_PATH)
    parser.add_argument("--expected-path", type=Path, default=DEFAULT_EXPECTED_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        issues = validate_annotations(args.evaluation_path, args.expected_path)
    except ValueError as exc:
        print(f"VALIDATION ERROR: {exc}", file=sys.stderr)
        return 1
    if issues:
        print(f"ANNOTATION VALIDATION FAILED ({len(issues)} issue(s)):")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"Annotation validation passed: {args.evaluation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
