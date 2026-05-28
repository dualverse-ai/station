"""
Evaluator for Diophantine large-x solution search across all nine equations.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator


EQUATIONS = {
    1: {
        "display": "z^2 + y^2 z + x^3 - 2 = 0",
        "eval": lambda x, y, z: z * z + y * y * z + x * x * x - 2,
    },
    2: {
        "display": "z^2 + y^2 z + x^3 - x - 1 = 0",
        "eval": lambda x, y, z: z * z + y * y * z + x * x * x - x - 1,
    },
    3: {
        "display": "z^2 + y^2 z + x^3 + x - 1 = 0",
        "eval": lambda x, y, z: z * z + y * y * z + x * x * x + x - 1,
    },
    4: {
        "display": "z^2 + y^2 z + x^3 + x + 1 = 0",
        "eval": lambda x, y, z: z * z + y * y * z + x * x * x + x + 1,
    },
    5: {
        "display": "z^2 + y^2 z + x^3 - 3 = 0",
        "eval": lambda x, y, z: z * z + y * y * z + x * x * x - 3,
    },
    6: {
        "display": "z^2 + y^2 z + x^3 + 3 = 0",
        "eval": lambda x, y, z: z * z + y * y * z + x * x * x + 3,
    },
    7: {
        "display": "z^2 + y^2 z + x^3 - x - 2 = 0",
        "eval": lambda x, y, z: z * z + y * y * z + x * x * x - x - 2,
    },
    8: {
        "display": "z^2 + y^2 z + x^3 - x + 2 = 0",
        "eval": lambda x, y, z: z * z + y * y * z + x * x * x - x + 2,
    },
    9: {
        "display": "z^2 + y^2 z - z + x^3 + 2 = 0",
        "eval": lambda x, y, z: z * z + y * y * z - z + x * x * x + 2,
    },
}

X_THRESHOLD = 10**50
TARGET_SOLUTIONS_PER_EQUATION = 3


class DiophantineEvaluator(ResearchTaskEvaluator):
    """Evaluator for large-x Diophantine search across all nine equations."""

    def __init__(self):
        super().__init__()

    def get_expected_function_name(self) -> str:
        return "find_solutions"

    def get_task_description(self) -> str:
        return "Find three distinct large-x integer solutions for each of the nine Diophantine equations"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "SolvedEquations": "d",
            "TotalLargeXSolutions": "d",
            "BestEquationLargeXCount": "d",
            "BestEquationMaxLog10AbsX": ".6f",
            "EqSolvedMask": None,
        }

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        try:
            score, details = self._evaluate(result)
            return True, score, details
        except AssertionError as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Verification failed: {type(exc).__name__}: {exc}"
            }
        except Exception as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Evaluation error: {type(exc).__name__}: {exc}"
            }

    def _evaluate(self, result: str) -> Tuple[float, Dict[str, Any]]:
        details: Dict[str, Any] = {
            "Message": "",
            "SolvedEquations": 0,
            "TotalLargeXSolutions": 0,
            "BestEquationLargeXCount": 0,
            "BestEquationMaxLog10AbsX": -1.0,
            "EqSolvedMask": "000000000",
        }

        if result is None:
            details["Message"] = "find_solutions returned None."
            return 0.0, details
        if not isinstance(result, str):
            details["Message"] = "Submission must return a JSON string."
            return 0.0, details

        try:
            payload = json.loads(result)
        except json.JSONDecodeError as exc:
            details["Message"] = f"Invalid JSON: {exc}"
            return 0.0, details

        if not isinstance(payload, dict):
            details["Message"] = "Top-level JSON value must be an object."
            return 0.0, details

        solutions_by_equation = payload.get("solutions_by_equation")
        if not isinstance(solutions_by_equation, dict):
            details["Message"] = "solutions_by_equation must be a JSON object keyed by equation id."
            return 0.0, details

        solved_mask_chars: List[str] = []
        solved_count = 0
        total_large_x = 0
        best_large_x_count = 0
        best_max_log10_abs_x = -1.0

        for equation_id in range(1, 10):
            raw_solutions = _lookup_equation_payload(solutions_by_equation, equation_id)
            exact_solutions = _collect_exact_solutions(raw_solutions, equation_id)
            large_x_solutions = [(x, y, z) for x, y, z in exact_solutions if abs(x) > X_THRESHOLD]
            large_x_count = len(large_x_solutions)
            solved = (
                large_x_count >= TARGET_SOLUTIONS_PER_EQUATION
                and len({x for x, _, _ in large_x_solutions}) >= TARGET_SOLUTIONS_PER_EQUATION
            )

            solved_mask_chars.append("1" if solved else "0")
            solved_count += int(solved)
            total_large_x += large_x_count
            best_large_x_count = max(best_large_x_count, large_x_count)
            best_max_log10_abs_x = max(best_max_log10_abs_x, _max_log10_abs_x(exact_solutions))

        details.update(
            {
                "SolvedEquations": solved_count,
                "TotalLargeXSolutions": total_large_x,
                "BestEquationLargeXCount": best_large_x_count,
                "BestEquationMaxLog10AbsX": best_max_log10_abs_x,
                "EqSolvedMask": "".join(solved_mask_chars),
            }
        )

        score = 100.0 * solved_count / 9.0
        if solved_count == 9:
            details["Message"] = "Solved all nine equations."
        else:
            details["Message"] = (
                f"Solved {solved_count} of 9 equations. Each solved equation needs at least "
                "three exact integer solutions with pairwise distinct x and |x| > 10^50."
            )
        return score, details


def _lookup_equation_payload(solutions_by_equation: Dict[str, Any], equation_id: int) -> Any:
    if str(equation_id) in solutions_by_equation:
        return solutions_by_equation[str(equation_id)]
    return solutions_by_equation.get(equation_id, [])


def _collect_exact_solutions(
    raw_solutions: Any, equation_id: int
) -> List[Tuple[int, int, int]]:
    if not isinstance(raw_solutions, list):
        return []

    equation = EQUATIONS[equation_id]["eval"]
    exact_solutions = set()

    for item in raw_solutions:
        triple = _parse_solution_item(item)
        if triple is None:
            continue
        x, y, z = triple
        if equation(x, y, z) == 0:
            exact_solutions.add((x, y, z))

    return sorted(exact_solutions)


def _parse_solution_item(item: Any) -> Optional[Tuple[int, int, int]]:
    if not isinstance(item, dict):
        return None

    try:
        x = _parse_decimal_int(item.get("x"))
        y = _parse_decimal_int(item.get("y"))
        z = _parse_decimal_int(item.get("z"))
    except (TypeError, ValueError):
        return None
    return x, y, z


def _parse_decimal_int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("booleans are not valid integers")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty integer string")
        return int(text, 10)
    raise TypeError("value must be an int or base-10 integer string")


def _max_log10_abs_x(solutions: Iterable[Tuple[int, int, int]]) -> float:
    max_abs_x = max((abs(x) for x, _, _ in solutions), default=0)
    if max_abs_x <= 0:
        return -1.0
    return math.log10(max_abs_x)
