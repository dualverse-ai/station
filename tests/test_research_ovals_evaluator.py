import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


EVALUATOR_PATH = "example/research_alpha_evolve/ovals/research/evaluators/evaluator.py"
AGENT_FACING_PATHS = [
    Path("example/research_alpha_evolve/ovals/research/research_task.md"),
    Path("example/research_alpha_evolve/ovals/research/baseline.yamll"),
    Path("example/research_alpha_evolve/ovals/research/evaluators/evaluator.py"),
]
LEAK_TERMS = [
    "alphaevolve",
    "alpha evolve",
    "deepmind",
    "arxiv",
    "unit circle",
    "round circle",
    "-1.0",
    "-0.9999",
    "0.9999",
]


def load_evaluator_module():
    spec = importlib.util.spec_from_file_location("research_ovals_evaluator", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ResearchOvalsEvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_evaluator_module()

    def test_round_circle_scores_near_one(self):
        n = self.module.EXPECTED_SAMPLES
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        payload = {
            "x": np.cos(theta).tolist(),
            "y": np.sin(theta).tolist(),
            "phi": np.ones(n).tolist(),
        }
        result = repr(payload)

        success, score, details, sort_key = self.module.Task1Evaluator().evaluate_submission(result)

        self.assertTrue(success)
        self.assertAlmostEqual(score, 1.0, places=8)
        self.assertAlmostEqual(details["RayleighQuotient"], 1.0, places=8)
        self.assertEqual(sort_key, (-score,))

    def test_simple_ellipse_seed_is_valid_and_below_target(self):
        n = self.module.EXPECTED_SAMPLES
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        payload = {
            "x": (1.2 * np.cos(theta)).tolist(),
            "y": (0.8 * np.sin(theta)).tolist(),
            "phi": np.ones(n).tolist(),
        }
        result = repr(payload)

        success, score, details, sort_key = self.module.Task1Evaluator().evaluate_submission(result)

        self.assertTrue(success)
        self.assertGreater(score, self.module.TARGET_SCORE)
        self.assertGreater(details["RayleighQuotient"], 1.0)
        self.assertEqual(sort_key, (-score,))

    def test_rejects_nonliteral_payload(self):
        evaluator = self.module.Task1Evaluator()

        success, score, details, sort_key = evaluator.evaluate_submission("not a literal")

        self.assertFalse(success)
        self.assertEqual(score, "n.a.")
        self.assertEqual(sort_key, (float("-inf"),))
        self.assertIn("valid Python literal dictionary", details["Message"])

    def test_rejects_huge_values_before_scoring(self):
        n = self.module.EXPECTED_SAMPLES
        payload = {
            "x": [200.0] * n,
            "y": [0.0] * n,
            "phi": [1.0] * n,
        }

        success, score, details, _sort_key = self.module.Task1Evaluator().evaluate_submission(
            repr(payload)
        )

        self.assertFalse(success)
        self.assertEqual(score, "n.a.")
        self.assertIn("absolute value", details["Message"])

    def test_agent_facing_files_do_not_leak_provenance_or_reported_solution(self):
        for path in AGENT_FACING_PATHS:
            text = path.read_text(encoding="utf-8").lower()
            for term in LEAK_TERMS:
                with self.subTest(path=str(path), term=term):
                    self.assertNotIn(term, text)


if __name__ == "__main__":
    unittest.main()
