"""Compare descriptive similarity-based uncertainty methods on pilot responses."""

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.similarity.combined import combined_similarity_matrix  # noqa: E402
from src.similarity.lexical import lexical_similarity_matrix  # noqa: E402
from src.similarity.semantic import semantic_similarity_matrix  # noqa: E402


DEFAULT_INPUT_PATH = Path("data/processed/objective2_manual_evaluation.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/objective2_similarity_comparison.json")
METHODS = ("lexical", "semantic", "combined")
AGGREGATIONS = ("mean", "median")


class ComparisonError(ValueError):
    """Raised when comparison input or output validation fails."""


def individual_scores_from_matrix(
    matrix: np.ndarray, aggregation: str
) -> tuple[list[float], list[float]]:
    """Aggregate each row against all other responses, excluding its diagonal."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        raise ValueError("similarity matrix must be a square matrix with at least two rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("similarity matrix must contain only finite values")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}")

    other_scores = values[~np.eye(values.shape[0], dtype=bool)].reshape(
        values.shape[0], values.shape[0] - 1
    )
    if aggregation == "mean":
        consistency = np.mean(other_scores, axis=1)
    else:
        consistency = np.asarray([median(row) for row in other_scores])
    consistency = np.clip(consistency, 0.0, 1.0)
    return consistency.tolist(), (1.0 - consistency).tolist()


def pairwise_standard_deviation(matrix: np.ndarray) -> float:
    """Return population standard deviation of unique off-diagonal pairs."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 2:
        raise ValueError("similarity matrix must be a square matrix with at least two rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("similarity matrix must contain only finite values")
    upper_triangle = values[np.triu_indices(values.shape[0], k=1)]
    return float(np.std(upper_triangle))


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"could not load evaluation data: {path}") from exc
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ComparisonError("evaluation data must contain a JSON array of objects")
    return data


def _extract_responses(
    record: dict[str, Any], index: int
) -> tuple[list[str], list[str]]:
    question_id = record.get("question_id")
    response_records = record.get("responses")
    if not isinstance(response_records, list) or len(response_records) != 5:
        raise ComparisonError(
            f"question_id {question_id} must contain exactly five response records"
        )

    response_ids: list[str] = []
    responses: list[str] = []
    for response_index, item in enumerate(response_records, 1):
        if not isinstance(item, dict):
            raise ComparisonError(
                f"question_id {question_id} response {response_index} is not an object"
            )
        response_id = item.get("response_id")
        response = item.get("response")
        if not isinstance(response_id, str) or not response_id.strip():
            raise ComparisonError(
                f"question_id {question_id} response {response_index} has no response_id"
            )
        if not isinstance(response, str) or not response.strip():
            raise ComparisonError(
                f"question_id {question_id} response {response_index} is empty"
            )
        response_ids.append(response_id)
        responses.append(response)
    if len(set(response_ids)) != len(response_ids):
        raise ComparisonError(f"question_id {question_id} has duplicate response IDs")
    return response_ids, responses


def _method_matrix(method: str, responses: list[str]) -> np.ndarray:
    matrix_functions: dict[str, Callable[[list[str]], np.ndarray]] = {
        "lexical": lexical_similarity_matrix,
        "semantic": semantic_similarity_matrix,
        "combined": combined_similarity_matrix,
    }
    try:
        return matrix_functions[method](responses)
    except (ImportError, KeyError, ValueError) as exc:
        raise ComparisonError(f"could not compute {method} similarity") from exc


def _question_result(record: dict[str, Any], index: int) -> dict[str, Any]:
    question_id = record.get("question_id")
    if not isinstance(question_id, (str, int)) or isinstance(question_id, bool):
        raise ComparisonError(f"record {index} has an invalid question_id")
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ComparisonError(f"question_id {question_id} has no valid question")
    response_ids, responses = _extract_responses(record, index)

    method_results: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        matrix = _method_matrix(method, responses)
        method_results[method] = {
            "pairwise_std": pairwise_standard_deviation(matrix),
            "aggregations": {},
        }
        for aggregation in AGGREGATIONS:
            consistency, uncertainty = individual_scores_from_matrix(
                matrix, aggregation
            )
            method_results[method]["aggregations"][aggregation] = {
                "consistency": consistency,
                "uncertainty": uncertainty,
            }

    per_response = []
    for response_index, (response_id, response) in enumerate(
        zip(response_ids, responses), 1
    ):
        per_response.append(
            {
                "response_id": response_id,
                "response_index": response_index,
                "response": response,
                "methods": {
                    method: {
                        aggregation: {
                            "consistency": method_results[method]["aggregations"][aggregation][
                                "consistency"
                            ][response_index - 1],
                            "uncertainty": method_results[method]["aggregations"][aggregation][
                                "uncertainty"
                            ][response_index - 1],
                        }
                        for aggregation in AGGREGATIONS
                    }
                    for method in METHODS
                },
            }
        )

    summary: dict[str, dict[str, float]] = {}
    for method in METHODS:
        for aggregation in AGGREGATIONS:
            key = f"{method}_{aggregation}"
            uncertainties = method_results[method]["aggregations"][aggregation][
                "uncertainty"
            ]
            summary[key] = {
                "mean_uncertainty": float(np.mean(uncertainties)),
                "uncertainty_std": float(np.std(uncertainties)),
                "pairwise_std": method_results[method]["pairwise_std"],
            }

    return {
        "question_id": question_id,
        "question": question,
        "responses": per_response,
        "pairwise_std": {
            method: method_results[method]["pairwise_std"] for method in METHODS
        },
        "question_summary": summary,
        "descriptive_only": True,
        "correctness_evaluation": (
            "No correctness labels or correctness-derived metrics are included."
        ),
    }


def compare_pilot(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Compare six method/aggregation combinations without generating responses."""
    records = _load_records(input_path)
    if not records:
        raise ComparisonError("evaluation data is empty")

    question_results: list[dict[str, Any]] = []
    question_ids: set[str | int] = set()
    for index, record in enumerate(records):
        result = _question_result(record, index)
        if result["question_id"] in question_ids:
            raise ComparisonError(f"duplicate question_id: {result['question_id']}")
        question_ids.add(result["question_id"])
        question_results.append(result)

    summaries: dict[str, dict[str, float]] = {}
    for method in METHODS:
        for aggregation in AGGREGATIONS:
            key = f"{method}_{aggregation}"
            values = [
                question["question_summary"][key]["mean_uncertainty"]
                for question in question_results
            ]
            summaries[key] = {
                "mean_uncertainty_across_questions": float(np.mean(values)),
                "uncertainty_variation_across_questions": float(np.std(values)),
            }

    output = Path(output_path)
    if output.exists():
        raise ComparisonError(f"refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "description": (
            "Descriptive comparison of similarity-based uncertainty methods; "
            "not a correctness evaluation."
        ),
        "source_file": str(input_path),
        "question_count": len(question_results),
        "response_count_per_question": 5,
        "methods": list(METHODS),
        "aggregations": list(AGGREGATIONS),
        "questions": question_results,
        "summary_statistics": summaries,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_pilot(args.input_path, args.output_path)
    print(f"Saved comparison for {result['question_count']} questions to {args.output_path}")


if __name__ == "__main__":
    main()
