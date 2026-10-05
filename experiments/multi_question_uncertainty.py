"""Run the controlled multi-question uncertainty experiment."""

import argparse
from pathlib import Path
from typing import Any

from groq import Groq

from src.generation.batch_generate import (
    DEFAULT_DATASET_PATH,
    load_questions,
)
from src.generation.response_generator import (
    DEFAULT_MODEL_NAME,
    generate_responses,
    load_client,
)
from src.similarity.combined import unique_pairwise_combined_similarities
from src.similarity.lexical import unique_pairwise_similarities
from src.similarity.semantic import unique_pairwise_semantic_similarities
from src.storage.response_store import save_results
from src.uncertainty import calculate_uncertainty, mean_consistency


DEFAULT_OUTPUT_PATH = Path("data/processed/multi_question_uncertainty.json")
DEFAULT_NUM_QUESTIONS = 20
DEFAULT_NUM_RESPONSES = 5


class ExperimentError(RuntimeError):
    """Raised when an experiment question cannot be completed."""


def _question_id(item: dict[str, Any], index: int) -> Any:
    """Return the source ID, or the source position when no ID is supplied."""
    return item.get("question_id", index)


def _build_result(
    *,
    item: dict[str, Any],
    index: int,
    responses: list[str],
    model_name: str,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"dataset record {index} has no valid question")
    if len(responses) != generation_config["num_responses"]:
        raise ValueError(
            f"generation returned {len(responses)} responses for question {index}; "
            f"expected {generation_config['num_responses']}"
        )

    lexical_scores = unique_pairwise_similarities(responses)
    semantic_scores = unique_pairwise_semantic_similarities(responses)
    combined_scores = unique_pairwise_combined_similarities(responses)
    uncertainty_metrics = calculate_uncertainty(combined_scores)

    return {
        "question_id": _question_id(item, index),
        "question": question,
        "model_name": model_name,
        "generation_config": dict(generation_config),
        "responses": responses,
        "lexical_consistency": mean_consistency(lexical_scores),
        "semantic_consistency": mean_consistency(semantic_scores),
        "combined_consistency": uncertainty_metrics["mean_consistency"],
        "uncertainty": uncertainty_metrics["uncertainty"],
        "pairwise_std": uncertainty_metrics["standard_deviation"],
    }


def run_experiment(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    num_responses: int = DEFAULT_NUM_RESPONSES,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    client: Groq | None = None,
) -> list[dict[str, Any]]:
    """Generate and measure the first ``num_questions`` source records."""
    if num_questions < 1:
        raise ValueError("num_questions must be at least 1")
    if num_responses < 2:
        raise ValueError("num_responses must be at least 2 for pairwise metrics")

    questions = load_questions(dataset_path)[:num_questions]
    if not questions:
        raise ValueError("dataset does not contain any questions")

    generation_config = {
        "num_responses": num_responses,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    generation_client = client or load_client()
    results: list[dict[str, Any]] = []

    for index, item in enumerate(questions):
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"dataset record {index} has no valid question")
        try:
            responses = generate_responses(
                question,
                generation_client,
                model_name=model_name,
                **generation_config,
            )
            results.append(
                _build_result(
                    item=item,
                    index=index,
                    responses=responses,
                    model_name=model_name,
                    generation_config=generation_config,
                )
            )
        except Exception as exc:
            raise ExperimentError(
                f"experiment failed for question {_question_id(item, index)}"
            ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS)
    parser.add_argument("--num-responses", type=int, default=DEFAULT_NUM_RESPONSES)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        num_questions=args.num_questions,
        num_responses=args.num_responses,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    print(f"Saved {len(results)} result record(s) to {args.output_path}")


if __name__ == "__main__":
    main()
