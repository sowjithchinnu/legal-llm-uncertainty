"""Create and persist generated-response records for legal QA experiments."""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def create_result_record(
    question_id: str | int,
    question: str,
    responses: Sequence[str],
    model_name: str,
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a JSON-serializable record for one legal QA question."""
    return {
        "question_id": question_id,
        "question": question,
        "responses": list(responses),
        "model_name": model_name,
        "generation_config": dict(generation_config),
    }


def save_results(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Save response records as a formatted JSON array."""
    output_path = Path(path)
    output_path.write_text(
        json.dumps(list(records), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load response records from a JSON array."""
    input_path = Path(path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("result file must contain a JSON array")
    return data
