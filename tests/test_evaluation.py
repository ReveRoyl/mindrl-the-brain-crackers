"""Scoring convention and empty-category regression tests (runtime deps only)."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from agent import Agent
from evaluate_public import evaluate, official_accuracy_credit


class EvaluationTests(unittest.TestCase):
    def test_exact_ties(self):
        self.assertEqual(official_accuracy_credit({0: .5, 1: .5}, 0), .5)
        self.assertEqual(official_accuracy_credit({0: .5 + 1e-13, 1: .5 - 1e-13}, 1), 0)

    def test_first_only_and_blank_line(self):
        trajectory = {"context": {"available_actions": ["a", "b"]}, "trials": [{"action": "a", "reward": 50.0}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.jsonl"
            path.write_text(json.dumps(trajectory) + "\n\n", encoding="utf-8")
            metrics = evaluate(Agent(), path)
        self.assertEqual(metrics["trials"], 1)
        self.assertAlmostEqual(metrics["nll"], math.log(2))
        self.assertEqual(metrics["accuracy"], .5)
        self.assertIsNone(metrics["categories"]["switch"]["nll"])

    def test_empty_data_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate(Agent(), path)


if __name__ == "__main__":
    unittest.main()
