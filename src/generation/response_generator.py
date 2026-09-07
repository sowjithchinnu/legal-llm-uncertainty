"""Generate multiple stochastic legal responses using the Groq API."""

import os

from dotenv import load_dotenv
from groq import Groq


DEFAULT_MODEL_NAME = "openai/gpt-oss-20b"

def load_client() -> Groq:
    """Create a Groq client using ``GROQ_API_KEY`` from the environment."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)


def generate_responses(
    question: str,
    client: Groq | None = None,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    num_responses: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> list[str]:
    """Generate independently sampled responses for a supplied legal question."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if num_responses < 1:
        raise ValueError("num_responses must be at least 1")

    client = client or load_client()
    responses: list[str] = []
    completion_options = {
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": max_tokens,
    }
    if model_name in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
        completion_options.update(
            reasoning_effort="low",
            include_reasoning=False,
        )

    for _ in range(num_responses):
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": question}],
            **completion_options,
        )
        responses.append(completion.choices[0].message.content or "")

    return responses
