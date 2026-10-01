import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.uncertainty import (
    calculate_uncertainty,
    mean_consistency,
    standard_deviation,
    uncertainty,
)


class UncertaintyTests(unittest.TestCase):
    def test_mean_consistency(self) -> None:
        scores = [0.8, 0.9, 1.0]
        self.assertAlmostEqual(mean_consistency(scores), 0.9)

    def test_uncertainty_is_one_minus_mean_consistency(self) -> None:
        scores = [0.8, 0.9, 1.0]
        self.assertAlmostEqual(uncertainty(scores), 0.1)

    def test_standard_deviation(self) -> None:
        scores = [0.8, 0.9, 1.0]
        self.assertAlmostEqual(standard_deviation(scores), np.std(scores))

    def test_calculate_uncertainty_returns_all_metrics(self) -> None:
        metrics = calculate_uncertainty([0.2, 0.4, 0.6])
        self.assertAlmostEqual(metrics["mean_consistency"], 0.4)
        self.assertAlmostEqual(metrics["uncertainty"], 0.6)
        self.assertAlmostEqual(metrics["standard_deviation"], np.std([0.2, 0.4, 0.6]))

    def test_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            calculate_uncertainty([])

    def test_non_numeric_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            calculate_uncertainty([0.5, "not a score"])


if __name__ == "__main__":
    unittest.main()
