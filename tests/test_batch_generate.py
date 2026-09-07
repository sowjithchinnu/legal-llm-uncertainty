import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.batch_generate import BatchGenerationError, generate_batch
from src.storage.response_store import load_results


class FakeCompletions:
    def __init__(self, should_fail: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.should_fail = should_fail

    def create(self, **kwargs: object) -> SimpleNamespace:
        if self.should_fail:
            raise RuntimeError("simulated API failure")
        self.calls.append(kwargs)
        response_number = len(self.calls)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"response {response_number}")
                )
            ]
        )


class FakeClient:
    def __init__(self, should_fail: bool = False) -> None:
        completions = FakeCompletions(should_fail=should_fail)
        self.completions = completions
        self.chat = SimpleNamespace(completions=completions)


class BatchGenerationTests(unittest.TestCase):
    def test_generates_only_requested_questions_and_saves_results(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            output_path = root / "nested" / "results.json"
            dataset_path.write_text(
                json.dumps(
                    [
                        {"question": "First question", "answer": "First answer"},
                        {"question": "Second question", "answer": "Second answer"},
                    ]
                ),
                encoding="utf-8",
            )

            results = generate_batch(
                dataset_path,
                output_path,
                num_questions=1,
                num_responses=2,
                client=FakeClient(),
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["question"], "First question")
            self.assertEqual(len(results[0]["responses"]), 2)
            self.assertEqual(load_results(output_path), results)

    def test_api_failure_is_reported(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            dataset_path.write_text(
                json.dumps([{"question": "A legal question"}]),
                encoding="utf-8",
            )

            with self.assertRaises(BatchGenerationError):
                generate_batch(
                    dataset_path,
                    root / "results.json",
                    client=FakeClient(should_fail=True),
                )


if __name__ == "__main__":
    unittest.main()
