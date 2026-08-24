"""
Evaluator for one-dimensional sign-uncertainty auxiliary functions.

Version 2 uses exact symbolic verification: submitted JSON numbers are parsed
as Python floats, converted to exact SymPy rationals, and scored from the
largest true positive sign-change root of the residual polynomial rather than
from a finite scan window.
"""

from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator

try:
    import sympy
except ImportError:  # pragma: no cover - handled at evaluation time.
    sympy = None


try:
    sys.set_int_max_str_digits(40000)
except AttributeError:  # pragma: no cover - Python < 3.11.
    pass


MODE = "laguerre-double-root"
TARGET_UPPER_BOUND = 0.315
MIN_K = 4
MAX_K = 20
MAX_ROOT = 1000.0
SIGN_CHECK_PRECISION_DIGITS = 200


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for Laguerre double-root uncertainty bounds."""

    def __init__(self):
        super().__init__()

    def get_expected_function_name(self) -> str:
        return "construct_uncertainty_bound"

    def get_task_description(self) -> str:
        return "Sign-uncertainty auxiliary-function construction"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "UpperBound": ".8f",
            "TargetMargin": ".8f",
            "K": "d",
            "SignBoundaryT": ".8f",
            "APlusEstimate": ".8f",
            "MinRoot": ".8f",
            "MaxRoot": ".8f",
            "ResidualDegree": "d",
            "RealRootCount": "d",
            "PositiveSignChangeCount": "d",
        }

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, Any, Dict[str, Any], Tuple[float, int]]:
        try:
            k, roots = self._parse_submission(result)
            upper_bound, metrics = self._evaluate_roots(k, roots)
        except AssertionError as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Verification failed: {type(exc).__name__}: {exc}"
            }, (float("-inf"), 0)
        except Exception as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Evaluation error: {type(exc).__name__}: {exc}"
            }, (float("-inf"), 0)

        margin = upper_bound - TARGET_UPPER_BOUND
        if margin < 0.0:
            message = (
                f"Target reached: upper bound {upper_bound:.10g} is below "
                f"{TARGET_UPPER_BOUND:.10g}."
            )
        else:
            message = (
                f"Valid auxiliary function with exact upper bound {upper_bound:.10g}. "
                "Lower scores are better."
            )

        details = {
            "Message": message,
            "UpperBound": float(upper_bound),
            "TargetMargin": float(margin),
            "K": int(k),
            "SignBoundaryT": float(metrics["sign_boundary_t"]),
            "APlusEstimate": float(math.sqrt(upper_bound)),
            "MinRoot": float(roots[0]),
            "MaxRoot": float(roots[-1]),
            "ResidualDegree": int(metrics["residual_degree"]),
            "RealRootCount": int(metrics["real_root_count"]),
            "PositiveSignChangeCount": int(metrics["positive_sign_change_count"]),
        }
        return True, float(upper_bound), details, self._build_sort_key(upper_bound, k)

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_uncertainty_bound" not in content:
            return False, "Submission must define construct_uncertainty_bound()."
        return True, None

    def _build_sort_key(self, score: float, k: int) -> Tuple[float, int]:
        # Research Center ranks larger sort keys first; this task is lower-is-better.
        return (-float(score), -int(k))

    def _parse_submission(self, result: Any) -> Tuple[int, List[float]]:
        if result is None:
            raise AssertionError("construct_uncertainty_bound returned None.")
        if not isinstance(result, str):
            raise AssertionError("Submission must return a JSON string.")

        try:
            payload = json.loads(result)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"Invalid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise AssertionError("Top-level JSON value must be an object.")
        if payload.get("mode") != MODE:
            raise AssertionError(f"mode must be exactly {MODE!r}.")
        if "k" not in payload:
            raise AssertionError("Missing required field 'k'.")
        if "roots" not in payload:
            raise AssertionError("Missing required field 'roots'.")

        k = payload["k"]
        if isinstance(k, bool) or not isinstance(k, int):
            raise AssertionError("k must be an integer.")
        if k < MIN_K or k > MAX_K:
            raise AssertionError(f"k must satisfy {MIN_K} <= k <= {MAX_K}.")

        raw_roots = payload["roots"]
        if not isinstance(raw_roots, list):
            raise AssertionError("roots must be a list.")
        if len(raw_roots) != k:
            raise AssertionError(f"roots must contain exactly k={k} entries.")

        roots: List[float] = []
        for idx, value in enumerate(raw_roots):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AssertionError(f"roots[{idx}] must be a real number.")
            root = float(value)
            if not math.isfinite(root):
                raise AssertionError(f"roots[{idx}] must be finite.")
            if root <= 0.0:
                raise AssertionError(f"roots[{idx}] must be positive.")
            if root > MAX_ROOT:
                raise AssertionError(f"roots[{idx}] exceeds the cap {MAX_ROOT}.")
            if roots and root <= roots[-1]:
                raise AssertionError("roots must be strictly increasing.")
            roots.append(root)

        return k, roots

    def _evaluate_roots(self, k: int, roots: Sequence[float]) -> Tuple[float, Dict[str, float]]:
        if sympy is None:
            raise AssertionError(
                "The evaluator requires the 'sympy' package for exact verification."
            )

        x_symbol = sympy.Symbol("t")
        rational_roots = [sympy.Rational(float(root)) for root in roots]
        polynomial = _find_laguerre_combination(k, rational_roots, x_symbol)
        residual = _divide_known_factors(polynomial, rational_roots, x_symbol)

        residual_poly = sympy.Poly(residual, x_symbol, domain=sympy.QQ)
        real_roots = sympy.real_roots(residual, x_symbol)
        sign_change_roots = _find_sign_change_roots(residual, real_roots, x_symbol)
        positive_sign_changes = [root for root in sign_change_roots if root > 0]
        if not positive_sign_changes:
            raise AssertionError("Residual polynomial has no positive sign-change root.")
        if len(positive_sign_changes) % 2 == 0:
            raise AssertionError(
                "Sign pattern is invalid: tail-positive orientation is not negative near the origin."
            )

        sign_boundary_t = max(positive_sign_changes)
        upper_bound = sign_boundary_t / (2 * sympy.pi)
        upper_bound_float = float(upper_bound)
        if not math.isfinite(upper_bound_float) or upper_bound_float <= 0:
            raise AssertionError("Computed upper bound is invalid.")

        return upper_bound_float, {
            "sign_boundary_t": float(sign_boundary_t),
            "residual_degree": int(residual_poly.degree()),
            "real_root_count": int(len(real_roots)),
            "positive_sign_change_count": int(len(positive_sign_changes)),
        }


def _find_laguerre_combination(k: int, roots: Sequence[Any], x_symbol: Any) -> Any:
    alpha = sympy.Rational(-1, 2)
    degrees = range(0, 4 * int(k) + 4, 2)
    basis = [_laguerre_polynomial(degree, alpha, x_symbol) for degree in degrees]

    condition_count = 2 * int(k) + 2
    matrix = sympy.Matrix(condition_count, len(basis), lambda _i, _j: 0)
    rhs = sympy.Matrix(condition_count, 1, lambda _i, _j: 0)
    rhs[1] = 1

    for j, basis_expr in enumerate(basis):
        derivative = sympy.diff(basis_expr, x_symbol)
        matrix[0, j] = basis_expr.subs(x_symbol, 0)
        matrix[1, j] = derivative.subs(x_symbol, 0)

    for i, root in enumerate(roots):
        for j, basis_expr in enumerate(basis):
            matrix[2 * i + 2, j] = basis_expr.subs(x_symbol, root)
        for j, basis_expr in enumerate(basis):
            matrix[2 * i + 3, j] = sympy.diff(basis_expr, x_symbol).subs(x_symbol, root)

    try:
        coefficients = matrix.LUsolve(rhs)
    except Exception as exc:
        raise AssertionError(f"Could not reconstruct a nonsingular polynomial: {exc}") from exc

    return sum(coefficients[i] * basis[i] for i in range(len(basis)))


def _divide_known_factors(polynomial: Any, roots: Sequence[Any], x_symbol: Any) -> Any:
    divisor = x_symbol
    for root in roots:
        divisor *= (x_symbol - root) ** 2
    try:
        return sympy.exquo(polynomial, divisor)
    except Exception as exc:
        raise AssertionError(
            "Exact division by t * prod_i (t-r_i)^2 failed after reconstruction."
        ) from exc


def _find_sign_change_roots(residual: Any, real_roots: Sequence[Any], x_symbol: Any) -> List[Any]:
    sign_change_roots: List[Any] = []
    epsilon = sympy.Rational(1e-198)
    for root in real_roots:
        approx_root = root.eval_rational(n=SIGN_CHECK_PRECISION_DIGITS)
        left_value = residual.subs(x_symbol, approx_root - epsilon)
        right_value = residual.subs(x_symbol, approx_root + epsilon)
        if (left_value > 0 and right_value < 0) or (left_value < 0 and right_value > 0):
            sign_change_roots.append(approx_root)
    return sign_change_roots


@lru_cache(maxsize=None)
def _laguerre_coefficients(degree: int, alpha: Any) -> Tuple[Any, ...]:
    if degree == 0:
        return (sympy.Integer(1),)
    if degree == 1:
        return (alpha + 1, sympy.Integer(-1))

    previous = [sympy.Integer(1)]
    current = [alpha + 1, sympy.Integer(-1)]
    for n in range(1, degree):
        term = _poly_scalar_mul(current, 2 * n + 1 + alpha)
        term = _poly_sub(term, _poly_shift_x(current))
        term = _poly_sub(term, _poly_scalar_mul(previous, n + alpha))
        next_coefficients = _poly_scalar_mul(term, sympy.Rational(1, n + 1))
        previous, current = current, next_coefficients
    return tuple(current)


def _laguerre_polynomial(degree: int, alpha: Any, x_symbol: Any) -> Any:
    result = sympy.Integer(0)
    for power, coefficient in enumerate(_laguerre_coefficients(int(degree), alpha)):
        result += coefficient * x_symbol**power
    return result


def _poly_scalar_mul(coefficients: Sequence[Any], scalar: Any) -> List[Any]:
    return [scalar * coefficient for coefficient in coefficients]


def _poly_sub(left: Sequence[Any], right: Sequence[Any]) -> List[Any]:
    size = max(len(left), len(right))
    result = [sympy.Integer(0)] * size
    for idx in range(size):
        if idx < len(left):
            result[idx] += left[idx]
        if idx < len(right):
            result[idx] -= right[idx]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _poly_shift_x(coefficients: Sequence[Any]) -> List[Any]:
    return [sympy.Integer(0)] + list(coefficients)
