"""Demonstrate the complete Objective 1 similarity-based uncertainty pipeline."""

import argparse
import random
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.generation.batch_generate import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    load_questions,
)
from src.generation.response_generator import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    generate_responses,
    load_client,
)
from src.similarity.combined import (  # noqa: E402
    unique_pairwise_combined_similarities,
)
from src.similarity.lexical import unique_pairwise_similarities  # noqa: E402
from src.similarity.semantic import unique_pairwise_semantic_similarities  # noqa: E402
from src.uncertainty import calculate_uncertainty, mean_consistency  # noqa: E402
from src.uncertainty.individual import (  # noqa: E402
    individual_response_metrics,
)


NUM_RESPONSES = 5
GENERATION_CONFIG = {
    "num_responses": NUM_RESPONSES,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
}


class DemoError(RuntimeError):
    """Raised when the Objective 1 demonstration cannot be completed."""


def _select_question(path: str | Path) -> dict[str, Any]:
    questions = load_questions(path)
    if not questions:
        raise ValueError(f"dataset does not contain any questions: {path}")
    question = random.choice(questions)
    text = question.get("question")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("randomly selected dataset record has no valid question")
    return question


def _validate_responses(responses: list[str]) -> None:
    if len(responses) != NUM_RESPONSES or any(
        not isinstance(response, str) or not response.strip()
        for response in responses
    ):
        raise ValueError(f"exactly {NUM_RESPONSES} non-empty responses are required")


def run_demo(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    client: Any = None,
) -> dict[str, Any]:
    """Select a random dataset question and compute all Objective 1 metrics."""
    record = _select_question(dataset_path)
    question = record["question"]
    try:
        responses = generate_responses(
            question,
            client or load_client(),
            model_name=model_name,
            **GENERATION_CONFIG,
        )
    except Exception as exc:
        raise DemoError(
            "response generation failed; check Groq configuration and API access"
        ) from exc

    if not isinstance(responses, list):
        raise DemoError("response generation returned an invalid response list")
    _validate_responses(responses)

    lexical_scores = unique_pairwise_similarities(responses)
    semantic_scores = unique_pairwise_semantic_similarities(responses)
    combined_scores = unique_pairwise_combined_similarities(responses)
    question_metrics = calculate_uncertainty(combined_scores)
    individual_metrics = individual_response_metrics(responses)

    return {
        "question_id": record.get("question_id"),
        "question": question,
        "responses": responses,
        "lexical_consistency": mean_consistency(lexical_scores),
        "semantic_consistency": mean_consistency(semantic_scores),
        "combined_consistency": question_metrics["mean_consistency"],
        "individual_consistency": individual_metrics["consistency"],
        "individual_uncertainty": individual_metrics["uncertainty"],
        "uncertainty": question_metrics["uncertainty"],
    }


def print_demo(metrics: dict[str, Any]) -> None:
    """Print the Objective 1 demonstration in a terminal-friendly format."""
    print("OBJECTIVE 1 — SIMILARITY-BASED UNCERTAINTY ESTIMATION")
    print()
    print(f"Selected question (ID {metrics['question_id']}): {metrics['question']}")
    print()
    print("Generated responses:")
    for index, response in enumerate(metrics["responses"], 1):
        print(f"\nResponse {index}:\n{response}")

    print("\nMean similarity metrics:")
    print(f"Mean lexical consistency: {metrics['lexical_consistency']:.6f}")
    print(f"Mean semantic consistency: {metrics['semantic_consistency']:.6f}")
    print(f"Mean combined consistency: {metrics['combined_consistency']:.6f}")

    print("\nIndividual-response metrics:")
    print("| Response index | Individual consistency | Individual uncertainty |")
    print("|---:|---:|---:|")
    for index, (consistency, uncertainty) in enumerate(
        zip(metrics["individual_consistency"], metrics["individual_uncertainty"]),
        1,
    ):
        print(f"| {index} | {consistency:.6f} | {uncertainty:.6f} |")

    highest_index = max(
        range(len(metrics["individual_uncertainty"])),
        key=metrics["individual_uncertainty"].__getitem__,
    )
    print(
        "\nOverall question-level consistency: "
        f"{metrics['combined_consistency']:.6f}"
    )
    print(f"Overall question-level uncertainty: {metrics['uncertainty']:.6f}")
    print(
        "Highest individual uncertainty: "
        f"response {highest_index + 1} "
        f"({metrics['individual_uncertainty'][highest_index]:.6f})"
    )
    print("\nOBJECTIVE 1 DEMONSTRATION COMPLETE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        print_demo(run_demo(dataset_path=args.dataset_path, model_name=args.model_name))
    except (DemoError, OSError, ValueError) as exc:
        print(f"Objective 1 demo failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
