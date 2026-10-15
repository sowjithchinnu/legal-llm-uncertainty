"""Interactively annotate prepared Objective 2 legal responses.

The prepared evaluation file is read-only. Annotations are written to a
separate JSON file after every response so an interrupted review can resume.
"""

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.validate_objective2_annotations import (  # noqa: E402
    ANNOTATION_FIELDS,
    DEFAULT_EVALUATION_PATH,
    VALID_LABELS,
)

DEFAULT_OUTPUT_PATH = Path("data/processed/objective2_annotations.json")


def load_prepared(path: str | Path) -> list[dict[str, Any]]:
    """Load and minimally validate the prepared evaluation records."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load prepared evaluation: {path}") from exc
    if not isinstance(data, list):
        raise ValueError("prepared evaluation must be a JSON array")
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"prepared record {index} must be an object")
        if not isinstance(record.get("question_id"), (str, int)) or isinstance(
            record.get("question_id"), bool
        ):
            raise ValueError(f"prepared record {index} has an invalid question_id")
        if not isinstance(record.get("question"), str):
            raise ValueError(f"prepared record {index} has no question")
        if not isinstance(record.get("reference_answer"), str):
            raise ValueError(f"prepared record {index} has no reference_answer")
        responses = record.get("responses")
        if not isinstance(responses, list) or not responses:
            raise ValueError(f"prepared record {index} must have responses")
        for response_index, response in enumerate(responses, 1):
            if not isinstance(response, dict) or not isinstance(
                response.get("response_id"), str
            ) or not isinstance(response.get("response"), str):
                raise ValueError(
                    f"prepared record {index} response {response_index} is invalid"
                )
    return data


def _record_keys(records: list[dict[str, Any]]) -> list[tuple[Any, list[str]]]:
    return [
        (
            record["question_id"],
            [response["response_id"] for response in record["responses"]],
        )
        for record in records
    ]


def load_or_initialize_output(
    prepared_records: list[dict[str, Any]], output_path: str | Path
) -> list[dict[str, Any]]:
    """Load resumable annotations, or create a separate blank copy."""
    path = Path(output_path)
    if not path.exists():
        records = copy.deepcopy(prepared_records)
        for record in records:
            for response in record["responses"]:
                response["annotation"] = dict.fromkeys(ANNOTATION_FIELDS, "")
        return records

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load annotation output: {path}") from exc
    if not isinstance(data, list) or not all(
        isinstance(record, dict) for record in data
    ):
        raise ValueError("annotation output must be a JSON array of objects")
    if _record_keys(data) != _record_keys(prepared_records):
        raise ValueError(
            "annotation output does not match the prepared evaluation IDs"
        )
    return data


def save_annotations(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Atomically save annotations, keeping a partially written file unusable."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary:
            json.dump(records, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _is_reviewed(response: dict[str, Any]) -> bool:
    annotation = response.get("annotation")
    return (
        isinstance(annotation, dict)
        and all(isinstance(annotation.get(field), str) for field in ANNOTATION_FIELDS)
        and bool(annotation["correctness_label"].strip())
    )


def _display_question(record: dict[str, Any], output: Callable[[str], None]) -> None:
    output(f"\nQuestion {record['question_id']}: {record['question']}")
    output(f"Reference answer: {record['reference_answer']}")
    for index, response in enumerate(record["responses"], 1):
        status = " [already annotated]" if _is_reviewed(response) else ""
        output(f"\nResponse {index} ({response['response_id']}){status}:")
        output(response["response"])


def annotate(
    prepared_path: str | Path = DEFAULT_EVALUATION_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Run the review loop and save immediately after each annotation."""
    prepared = load_prepared(prepared_path)
    records = load_or_initialize_output(prepared, output_path)
    save_annotations(records, output_path)

    for record in records:
        pending = [response for response in record["responses"] if not _is_reviewed(response)]
        if not pending:
            continue
        _display_question(record, output)
        for response in pending:
            response_number = response["response_id"]
            output(f"\nAnnotating {response_number}")
            while True:
                label = input_fn(
                    "Label [correct/incorrect/partially_correct/uncertain]: "
                ).strip()
                if label in VALID_LABELS:
                    break
                output("Invalid label; choose one of the four listed labels.")
            rationale = input_fn("Rationale: ").strip()
            while not rationale:
                output("Rationale is required.")
                rationale = input_fn("Rationale: ").strip()
            reviewer_id = input_fn("Reviewer ID: ").strip()
            while not reviewer_id:
                output("Reviewer ID is required.")
                reviewer_id = input_fn("Reviewer ID: ").strip()
            response["annotation"] = {
                "correctness_label": label,
                "rationale": rationale,
                "reviewer_id": reviewer_id,
            }
            save_annotations(records, output_path)
            output(f"Saved annotation for {response_number}.")
    output(f"\nAnnotation output: {output_path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-path", type=Path, default=DEFAULT_EVALUATION_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        annotate(args.prepared_path, args.output_path)
    except (OSError, ValueError) as exc:
        print(f"ANNOTATION ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
