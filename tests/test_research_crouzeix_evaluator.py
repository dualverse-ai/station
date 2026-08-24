import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


EVALUATOR_PATH = "example/research_alpha_evolve/crouzeix/research/evaluators/evaluator.py"
AGENT_FACING_PATHS = [
    Path("example/research_alpha_evolve/crouzeix/research/research_task.md"),
    Path("example/research_alpha_evolve/crouzeix/research/baseline.yamll"),
    Path("example/research_alpha_evolve/crouzeix/research/evaluators/evaluator.py"),
]
LEAK_TERMS = [
    "alphaevolve",
    "alpha evolve",
    "deepmind",
    "arxiv",
    "reported construction",
    "known maximizer",
]


def load_evaluator_module():
    spec = importlib.util.spec_from_file_location("research_crouzeix_evaluator", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def complex_pair(value):
    return [float(np.real(value)), float(np.imag(value))]


class ResearchCrouzeixEvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_evaluator_module()

    def test_simple_diagonal_baseline_is_valid_and_below_target(self):
        payload = {
            "mode": self.module.MODE,
            "n": 3,
            "degree": 1,
            "matrix": [
                [[-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            ],
            "coefficients": [[0.0, 0.0], [1.0, 0.0]],
        }

        success, score, details, sort_key = self.module.Task1Evaluator().evaluate_submission(
            json.dumps(payload)
        )

        self.assertTrue(success)
        self.assertLess(score, self.module.TARGET_SCORE)
        self.assertAlmostEqual(score, 1.0, places=8)
        self.assertEqual(details["N"], 3)
        self.assertEqual(sort_key[0], score)

    def test_reference_ratio_two_case_is_scored_near_two_without_reaching_target(self):
        n = 3
        matrix = np.zeros((n, n), dtype=np.complex128)
        matrix[0, 1] = np.sqrt(2.0)
        matrix[1, 2] = np.sqrt(2.0)
        payload = {
            "mode": self.module.MODE,
            "n": n,
            "degree": 2,
            "matrix": [[complex_pair(matrix[i, j]) for j in range(n)] for i in range(n)],
            "coefficients": [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        }

        success, score, details, _sort_key = self.module.Task1Evaluator().evaluate_submission(
            json.dumps(payload)
        )

        self.assertTrue(success)
        self.assertGreater(score, 1.999)
        self.assertLess(score, self.module.TARGET_SCORE)
        self.assertEqual(details["StrictVerification"], 1)

    def test_rejects_oversized_matrix(self):
        n = self.module.MAX_N + 1
        payload = {
            "mode": self.module.MODE,
            "n": n,
            "degree": 1,
            "matrix": [],
            "coefficients": [[0.0, 0.0], [1.0, 0.0]],
        }

        success, score, details, sort_key = self.module.Task1Evaluator().evaluate_submission(
            json.dumps(payload)
        )

        self.assertFalse(success)
        self.assertEqual(score, "n.a.")
        self.assertEqual(sort_key, (float("-inf"), 0, 0))
        self.assertIn("n must satisfy", details["Message"])

    def test_rejects_zero_polynomial(self):
        payload = {
            "mode": self.module.MODE,
            "n": 3,
            "degree": 1,
            "matrix": [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            "coefficients": [[0.0, 0.0], [0.0, 0.0]],
        }

        success, score, details, _sort_key = self.module.Task1Evaluator().evaluate_submission(
            json.dumps(payload)
        )

        self.assertFalse(success)
        self.assertEqual(score, "n.a.")
        self.assertIn("must not all be zero", details["Message"])

    def test_agent_facing_files_do_not_leak_provenance_or_reported_solution(self):
        for path in AGENT_FACING_PATHS:
            text = path.read_text(encoding="utf-8").lower()
            for term in LEAK_TERMS:
                with self.subTest(path=str(path), term=term):
                    self.assertNotIn(term, text)


if __name__ == "__main__":
    unittest.main()
