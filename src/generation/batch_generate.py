"""Generate and store responses for a limited batch of IndicLegalQA questions."""

import argparse
import json
from pathlib import Path
from typing import Any

from groq import Groq

from src.generation.response_generator import (
    DEFAULT_MODEL_NAME,
    generate_responses,
    load_client,
)
from src.storage.response_store import create_result_record, save_results


DEFAULT_DATASET_PATH = Path("data/raw/IndicLegalQA Dataset_10K_Revised.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/generated_responses.json")


class BatchGenerationError(RuntimeError):
    """Raised when response generation fails for a dataset question."""


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load IndicLegalQA records without modifying the source file."""
    dataset_path = Path(path)
    try:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load dataset: {dataset_path}") from exc

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("dataset must contain a JSON array of objects")
    return data


def generate_batch(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    num_questions: int = 1,
    num_responses: int = 5,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    client: Groq | None = None,
) -> list[dict[str, Any]]:
    """Generate responses for the first ``num_questions`` dataset records."""
    if num_questions < 1:
        raise ValueError("num_questions must be at least 1")
    if num_responses < 1:
        raise ValueError("num_responses must be at least 1")

    questions = load_questions(dataset_path)[:num_questions]
    if not questions:
        raise ValueError("dataset does not contain any questions")

    client = client or load_client()
    generation_config = {
        "num_responses": num_responses,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    results: list[dict[str, Any]] = []

    for question_id, item in enumerate(questions):
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"dataset record {question_id} has no valid question")

        try:
            responses = generate_responses(
                question,
                client,
                model_name=model_name,
                **generation_config,
            )
        except Exception as exc:
            raise BatchGenerationError(
                f"response generation failed for question {question_id}"
            ) from exc

        results.append(
            create_result_record(
                question_id=question_id,
                question=question,
                responses=responses,
                model_name=model_name,
                generation_config=generation_config,
            )
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--num-questions", type=int, default=1)
    parser.add_argument("--num-responses", type=int, default=5)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = generate_batch(
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
