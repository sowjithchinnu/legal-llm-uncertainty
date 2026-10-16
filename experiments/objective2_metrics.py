"""Evaluate similarity-derived uncertainty against manual Objective 2 labels."""

import json
import math
from pathlib import Path
from typing import Any

METHODS = ("lexical", "semantic", "combined")
AGGREGATIONS = ("mean", "median")
COMBINATIONS = tuple(
    f"{method}_{aggregation}"
    for method in METHODS
    for aggregation in AGGREGATIONS
)
LABELS = frozenset({"correct", "incorrect", "partially_correct", "uncertain"})
BINARY_LABELS = frozenset({"correct", "incorrect"})

DEFAULT_ANNOTATIONS_PATH = Path("data/processed/objective2_annotations.json")
DEFAULT_COMPARISON_PATH = Path("data/processed/objective2_similarity_comparison.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/objective2_metrics.json")


class MetricsError(ValueError):
    """Raised when evaluation inputs are invalid or cannot be matched."""


def _load_json(path: str | Path, description: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricsError(f"could not load {description}: {path}") from exc


def _key(question_id: str | int, response_id: str) -> tuple[str | int, str]:
    return question_id, response_id


def _validate_id(
    question_id: Any, response_id: Any, location: str
) -> tuple[str | int, str]:
    if not isinstance(question_id, (str, int)) or isinstance(question_id, bool):
        raise MetricsError(f"{location} has an invalid question_id")
    if not isinstance(response_id, str) or not response_id.strip():
        raise MetricsError(f"{location} has an invalid response_id")
    return question_id, response_id


def _score(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricsError(f"{location} uncertainty must be a number")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise MetricsError(f"{location} uncertainty must be finite and in [0, 1]")
    return score


def _comparison_scores(path: str | Path) -> dict[tuple[str | int, str], dict[str, float]]:
    data = _load_json(path, "similarity comparison")
    questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(questions, list):
        raise MetricsError("similarity comparison must contain a questions array")

    scores: dict[tuple[str | int, str], dict[str, float]] = {}
    question_ids: set[str | int] = set()
    for question_index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise MetricsError(f"comparison question {question_index} is not an object")
        question_id = question.get("question_id")
        if not isinstance(question_id, (str, int)) or isinstance(question_id, bool):
            raise MetricsError(f"comparison question {question_index} has an invalid question_id")
        if question_id in question_ids:
            raise MetricsError(f"duplicate question_id in comparison: {question_id!r}")
        question_ids.add(question_id)
        responses = question.get("responses")
        if not isinstance(responses, list):
            raise MetricsError(f"comparison question {question_id!r} has no responses")
        for response_index, response in enumerate(responses, 1):
            if not isinstance(response, dict):
                raise MetricsError(f"comparison question {question_id!r} response {response_index} is invalid")
            _, response_id = _validate_id(
                question_id, response.get("response_id"),
                f"comparison question {question_id!r} response {response_index}",
            )
            key = _key(question_id, response_id)
            if key in scores:
                raise MetricsError(f"duplicate response ID in comparison: {key!r}")
            methods = response.get("methods")
            if not isinstance(methods, dict):
                raise MetricsError(f"comparison response {key!r} has no methods")
            scores[key] = {}
            for combination in COMBINATIONS:
                method, aggregation = combination.split("_")
                method_data = methods.get(method)
                aggregation_data = (
                    method_data.get(aggregation)
                    if isinstance(method_data, dict)
                    else None
                )
                if not isinstance(aggregation_data, dict):
                    raise MetricsError(f"comparison response {key!r} is missing {combination}")
                scores[key][combination] = _score(
                    aggregation_data.get("uncertainty"),
                    f"comparison response {key!r} {combination}",
                )
    if not scores:
        raise MetricsError("similarity comparison contains no responses")
    return scores


def _annotations(path: str | Path) -> dict[tuple[str | int, str], str]:
    data = _load_json(path, "manual annotations")
    if not isinstance(data, list):
        raise MetricsError("manual annotations must be a JSON array")
    annotations: dict[tuple[str | int, str], str] = {}
    question_ids: set[str | int] = set()
    for question_index, question in enumerate(data):
        if not isinstance(question, dict):
            raise MetricsError(f"annotation question {question_index} is not an object")
        question_id = question.get("question_id")
        if not isinstance(question_id, (str, int)) or isinstance(question_id, bool):
            raise MetricsError(f"annotation question {question_index} has an invalid question_id")
        if question_id in question_ids:
            raise MetricsError(f"duplicate question_id in annotations: {question_id!r}")
        question_ids.add(question_id)
        responses = question.get("responses")
        if not isinstance(responses, list):
            raise MetricsError(f"annotation question {question_id!r} has no responses")
        for response_index, response in enumerate(responses, 1):
            if not isinstance(response, dict):
                raise MetricsError(f"annotation question {question_id!r} response {response_index} is invalid")
            _, response_id = _validate_id(
                question_id, response.get("response_id"),
                f"annotation question {question_id!r} response {response_index}",
            )
            key = _key(question_id, response_id)
            if key in annotations:
                raise MetricsError(f"duplicate response ID in annotations: {key!r}")
            annotation = response.get("annotation")
            if not isinstance(annotation, dict):
                raise MetricsError(f"annotation response {key!r} has no annotation object")
            label = annotation.get("correctness_label")
            rationale = annotation.get("rationale")
            reviewer_id = annotation.get("reviewer_id")
            if not all(isinstance(value, str) for value in (label, rationale, reviewer_id)):
                raise MetricsError(f"annotation response {key!r} has invalid annotation fields")
            if label and label not in LABELS:
                raise MetricsError(f"annotation response {key!r} has invalid label {label!r}")
            if label and (not rationale.strip() or not reviewer_id.strip()):
                raise MetricsError(f"annotation response {key!r} is reviewed without rationale and reviewer_id")
            if not label and (rationale.strip() or reviewer_id.strip()):
                raise MetricsError(f"annotation response {key!r} has partial unreviewed fields")
            annotations[key] = label
    if not annotations:
        raise MetricsError("manual annotations contain no responses")
    return annotations


def _matched_labels(
    annotation_path: str | Path, comparison_path: str | Path
) -> tuple[dict[tuple[str | int, str], dict[str, Any]], dict[str, int]]:
    scores = _comparison_scores(comparison_path)
    labels = _annotations(annotation_path)
    missing_annotations = set(scores) - set(labels)
    unexpected_annotations = set(labels) - set(scores)
    if missing_annotations or unexpected_annotations:
        details = []
        if missing_annotations:
            details.append(f"missing annotation IDs: {sorted(missing_annotations, key=str)}")
        if unexpected_annotations:
            details.append(f"mismatched annotation IDs: {sorted(unexpected_annotations, key=str)}")
        raise MetricsError("; ".join(details))

    samples: dict[tuple[str | int, str], dict[str, Any]] = {}
    counts = {"reviewed": 0, "unreviewed": 0, "partially_correct": 0, "uncertain": 0}
    for key, label in labels.items():
        if not label:
            counts["unreviewed"] += 1
            continue
        counts["reviewed"] += 1
        if label in ("partially_correct", "uncertain"):
            counts[label] += 1
        samples[key] = {"label": label, "scores": scores[key]}
    return samples, counts


def auroc(uncertainties: list[float], labels: list[str]) -> float | None:
    """Return AUROC for incorrect-vs-correct, or None when a class is absent."""
    incorrect = [score for score, label in zip(uncertainties, labels) if label == "incorrect"]
    correct = [score for score, label in zip(uncertainties, labels) if label == "correct"]
    if not incorrect or not correct:
        return None
    wins = sum(score > other for score in incorrect for other in correct)
    ties = sum(score == other for score in incorrect for other in correct)
    return (wins + 0.5 * ties) / (len(incorrect) * len(correct))


def adaptive_ace(
    uncertainties: list[float], labels: list[str], max_bins: int = 10
) -> dict[str, Any] | None:
    """Compute ACE using up to max_bins equal-frequency, non-empty bins.

    Responses are sorted by proxy confidence (1 - uncertainty), split into
    near-equal contiguous groups, and each bin contributes equally to the
    mean absolute gap between proxy confidence and binary accuracy.
    """
    if max_bins < 1:
        raise ValueError("max_bins must be positive")
    if len(uncertainties) != len(labels):
        raise ValueError("uncertainties and labels must have equal lengths")
    if any(label not in BINARY_LABELS for label in labels):
        raise ValueError("ACE labels must be correct or incorrect")
    if not uncertainties or not {"correct", "incorrect"}.issubset(labels):
        return None

    ordered = sorted(
        ((1.0 - score, 1.0 if label == "correct" else 0.0)
         for score, label in zip(uncertainties, labels)),
        key=lambda item: item[0],
    )
    bin_count = min(max_bins, len(ordered))
    bins: list[list[tuple[float, float]]] = []
    for index in range(bin_count):
        start = index * len(ordered) // bin_count
        end = (index + 1) * len(ordered) // bin_count
        bins.append(ordered[start:end])
    details = []
    for values in bins:
        mean_confidence = sum(value[0] for value in values) / len(values)
        mean_accuracy = sum(value[1] for value in values) / len(values)
        details.append({
            "count": len(values),
            "mean_confidence": mean_confidence,
            "accuracy": mean_accuracy,
            "absolute_gap": abs(mean_confidence - mean_accuracy),
        })
    return {
        "value": sum(item["absolute_gap"] for item in details) / len(details),
        "bin_count": len(details),
        "bins": details,
        "confidence_definition": (
            "consistency-derived proxy confidence = 1 - uncertainty; "
            "not a calibrated probability"
        ),
    }


def evaluate_metrics(
    annotation_path: str | Path = DEFAULT_ANNOTATIONS_PATH,
    comparison_path: str | Path = DEFAULT_COMPARISON_PATH,
    *,
    max_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate all six combinations without changing either input file."""
    samples, counts = _matched_labels(annotation_path, comparison_path)
    binary_samples = [
        sample for sample in samples.values() if sample["label"] in BINARY_LABELS
    ]
    labels = [sample["label"] for sample in binary_samples]
    result: dict[str, Any] = {
        "description": (
            "Objective 2 metrics for similarity-derived uncertainty. "
            "AUROC and ACE use reviewed correct/incorrect labels only."
        ),
        "source_files": {
            "annotations": str(annotation_path),
            "comparison": str(comparison_path),
        },
        "counts": {
            **counts,
            "binary_reviewed": len(binary_samples),
            "correct": labels.count("correct"),
            "incorrect": labels.count("incorrect"),
        },
        "excluded_labels": {
            "partially_correct": "excluded from primary binary metrics",
            "uncertain": "excluded from primary binary metrics",
        },
        "methods": {},
    }
    sufficient = bool(labels) and {"correct", "incorrect"}.issubset(labels)
    result["status"] = (
        "sufficient reviewed labels" if sufficient
        else "insufficient reviewed labels: both correct and incorrect are required"
    )
    for combination in COMBINATIONS:
        uncertainties = [sample["scores"][combination] for sample in binary_samples]
        result["methods"][combination] = {
            "sample_count": len(uncertainties),
            "auroc": auroc(uncertainties, labels) if sufficient else None,
            "ace": adaptive_ace(uncertainties, labels, max_bins=max_bins) if sufficient else None,
        }
    return result
