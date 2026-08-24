import importlib.util
import json
from pathlib import Path
import sys
import unittest


EVALUATOR_PATH = "example/research_alpha_evolve/sidorenko/research/evaluators/evaluator.py"
AGENT_FACING_PATHS = [
    Path("example/research_alpha_evolve/sidorenko/research/research_task.md"),
    Path("example/research_alpha_evolve/sidorenko/research/baseline.yamll"),
    Path("example/research_alpha_evolve/sidorenko/research/evaluators/evaluator.py"),
]
LEAK_TERMS = [
    "alphaevolve",
    "alpha evolve",
    "deepmind",
    "arxiv",
    "official repository",
    "reported construction",
    "evolved solution",
]


def load_evaluator_module():
    spec = importlib.util.spec_from_file_location("research_sidorenko_evaluator", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def two_block_payload(mode: str) -> str:
    n = 30
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if (i < n // 2) != (j < n // 2):
                matrix[i][j] = 1.0
    return json.dumps({"mode": mode, "matrix": matrix})


class ResearchSidorenkoEvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_evaluator_module()

    def test_two_block_baseline_is_valid_simple_seed(self):
        success, score, details, sort_key = self.module.Task1Evaluator().evaluate_submission(
            two_block_payload(self.module.MODE)
        )

        self.assertTrue(success)
        self.assertAlmostEqual(score, -0.984375, places=12)
        self.assertAlmostEqual(details["EdgeDensity"], 0.5, places=12)
        self.assertEqual(details["Vertices"], 30)
        self.assertEqual(sort_key, (score,))

    def test_constant_matrix_receives_penalty_score(self):
        n = 30
        payload = json.dumps({
            "mode": self.module.MODE,
            "matrix": [[0.5 for _ in range(n)] for _ in range(n)],
        })

        success, score, details, sort_key = self.module.Task1Evaluator().evaluate_submission(payload)

        self.assertTrue(success)
        self.assertEqual(score, -1.0)
        self.assertEqual(sort_key, (-1.0,))
        self.assertIn("Penalty reason", details["Message"])

    def test_wrong_square_size_receives_official_penalty_score(self):
        payload = json.dumps({
            "mode": self.module.MODE,
            "matrix": [[0.0, 1.0], [1.0, 0.0]],
        })

        success, score, details, _sort_key = self.module.Task1Evaluator().evaluate_submission(payload)

        self.assertTrue(success)
        self.assertEqual(score, -1.0)
        self.assertEqual(details["Vertices"], 2)

    def test_rejects_malformed_payload(self):
        success, score, details, sort_key = self.module.Task1Evaluator().evaluate_submission(
            '{"mode": "wrong", "matrix": []}'
        )

        self.assertFalse(success)
        self.assertEqual(score, "n.a.")
        self.assertEqual(sort_key, (float("-inf"),))
        self.assertIn("mode must be exactly", details["Message"])

    def test_agent_facing_files_do_not_leak_provenance_or_reported_solution(self):
        for path in AGENT_FACING_PATHS:
            text = path.read_text(encoding="utf-8").lower()
            for term in LEAK_TERMS:
                with self.subTest(path=str(path), term=term):
                    self.assertNotIn(term, text)


if __name__ == "__main__":
    unittest.main()
