import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.response_generator import generate_responses, load_client


question = "What are the requirements for diversity jurisdiction?"

client = load_client()

responses = generate_responses(
    question,
    client,
    num_responses=5,
    temperature=0.7,
    top_p=0.9,
)

for i, response in enumerate(responses, start=1):
    print(f"\n{'=' * 20} RESPONSE {i} {'=' * 20}")
    print(response)
