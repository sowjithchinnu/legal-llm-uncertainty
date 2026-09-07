import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.response_store import (
    create_result_record,
    load_results,
    save_results,
)


class ResponseStoreTests(unittest.TestCase):
    def test_save_and_load_results(self) -> None:
        records = [
            create_result_record(
                question_id=7,
                question="What is diversity jurisdiction?",
                responses=["Response one", "Response two"],
                model_name="openai/gpt-oss-20b",
                generation_config={"temperature": 0.7, "max_tokens": 256},
            )
        ]

        with TemporaryDirectory() as directory:
            path = Path(directory) / "responses.json"
            save_results(records, path)

            self.assertEqual(load_results(path), records)

    def test_load_results_requires_json_array(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "responses.json"
            path.write_text('{"question_id": 7}', encoding="utf-8")

            with self.assertRaises(ValueError):
                load_results(path)


if __name__ == "__main__":
    unittest.main()
