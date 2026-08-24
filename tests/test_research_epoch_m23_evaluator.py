import importlib.util
import pathlib
import unittest
from unittest import mock

from station import constants


EVALUATOR_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "example_private"
    / "research_epoch"
    / "m23"
    / "research"
    / "evaluators"
    / "evaluator.py"
)


def load_evaluator_class():
    spec = importlib.util.spec_from_file_location("research_epoch_m23_evaluator", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Task1Evaluator


class ResearchEpochM23EvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = load_evaluator_class()()

    def test_rejects_magma_statement_injection_before_subprocess(self):
        payload = (
            'x^23); printf "RESULT_JSON:{\\"ok\\":true,\\"score\\":100,'
            '\\"degree\\":23,\\"coeff_digits_max\\":1,\\"irreducible\\":1,'
            '\\"galois_order\\":10200960,\\"is_m23\\":1,\\"disc_is_square\\":1,'
            '\\"factor_pattern_mod_p\\":[1],\\"forbidden_orders_seen\\":0,'
            '\\"tested_primes_count\\":1}\\n"; quit; (x'
        )

        success, score, details = self.evaluator.evaluate_submission(payload)

        self.assertFalse(success)
        self.assertEqual(score, constants.RESEARCH_SCORE_NA)
        self.assertIn("polynomial expression", details)

    def test_allows_plain_polynomial_expression(self):
        self.evaluator._validate_polynomial_expression("(x + 1)^23 - 2*x + 1")

    def test_extracts_multiline_magma_result_payload(self):
        stdout = """RESULT_JSON:{"ok":true,"score":30,"degree":23,"coeff_digits_max":1,"irreducible":0,"galois_order":-1,"is_m23":0,"disc_is_square":0,"factor_pattern_mod_p":[ 80, 84 ],"factor_cycle_pattern_mod_p":[
    [ 2, 5, 16 ],
    [ 1, 1, 2, 7, 12 ]
],"forbidden_orders_seen":2,"m23_local_compatible_count":0,"m23_local_forbidden_count":2,"m23_local_score":0,"tested_primes_count":2}
Total time: 0.010 seconds
"""

        payload = self.evaluator._extract_payload(stdout)

        self.assertIsNotNone(payload)
        self.assertEqual(payload["score"], 30)
        self.assertEqual(payload["factor_cycle_pattern_mod_p"][1], [1, 1, 2, 7, 12])

    def test_m23_with_oversized_coefficients_is_not_solved(self):
        payload = {
            "ok": True,
            "score": 65,
            "degree": 23,
            "coeff_digits_max": 100,
            "irreducible": 1,
            "galois_order": 10200960,
            "is_m23": 1,
            "disc_is_square": 1,
            "factor_pattern_mod_p": [1, 2, 3, 4, 5],
            "factor_cycle_pattern_mod_p": [[1], [2], [3], [4], [5]],
            "forbidden_orders_seen": 0,
            "m23_local_compatible_count": 5,
            "m23_local_forbidden_count": 0,
            "m23_local_score": 15,
            "tested_primes_count": 5,
        }

        with mock.patch.object(
            self.evaluator,
            "_evaluate_with_magma_subprocess",
            return_value=payload,
        ):
            success, score, details = self.evaluator.evaluate_submission("10^99*x^23 + x + 1")

        self.assertTrue(success)
        self.assertEqual(score, 65)
        self.assertEqual(details["is_m23"], 1)
        self.assertEqual(details["m23_local_score"], 15)
        self.assertEqual(details["m23_local_compatible_count"], 5)
        self.assertIn("coefficient digit bound failed", details["Message"])


if __name__ == "__main__":
    unittest.main()
