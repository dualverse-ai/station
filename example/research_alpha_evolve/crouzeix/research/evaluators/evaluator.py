"""
Evaluator for Crouzeix numerical-range ratio constructions.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator


MODE = "crouzeix-ratio-v1"
TARGET_SCORE = 2.001

MIN_N = 3
MAX_N = 80
MIN_DEGREE = 1
MAX_DEGREE = 60

MAX_MATRIX_COMPONENT = 100.0
MAX_COEFFICIENT_COMPONENT = 1.0e6
MIN_COEFFICIENT_SCALE = 1.0e-14
MIN_DENOMINATOR = 1.0e-12

BASE_THETA_MIN = 4096
BASE_THETA_PER_DEGREE = 128
STRICT_SCORE_THRESHOLD = 1.8
STRICT_THETA_MIN = 16384
STRICT_THETA_PER_DEGREE = 256
BASE_GRID_OFFSETS = (0.0,)
STRICT_GRID_OFFSETS = (0.0, 0.25, 0.5, 0.75)
EDGE_FRACTIONS = (0.25, 0.5, 0.75)


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for finite Crouzeix ratio witnesses."""

    def __init__(self):
        super().__init__()

    def get_expected_function_name(self) -> str:
        return "construct_crouzeix"

    def get_task_description(self) -> str:
        return "Crouzeix numerical-range ratio construction"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "CrouzeixRatio": ".10f",
            "TargetMargin": ".10f",
            "N": "d",
            "Degree": "d",
            "NumeratorNorm": ".8e",
            "DenominatorSup": ".8e",
            "MatrixOperatorNorm": ".8f",
            "NumericalRadius": ".8f",
            "ThetaSamples": "d",
            "EdgeSamples": "d",
            "StrictVerification": "d",
            "CoefficientScale": ".8e",
        }

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, Any, Dict[str, Any], Tuple[float, int, int]]:
        try:
            matrix, coefficients, degree = self._parse_submission(result)
            score, metrics = evaluate_crouzeix_ratio(matrix, coefficients, degree)
        except AssertionError as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Verification failed: {type(exc).__name__}: {exc}"
            }, (float("-inf"), 0, 0)
        except Exception as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Evaluation error: {type(exc).__name__}: {exc}"
            }, (float("-inf"), 0, 0)

        margin = score - TARGET_SCORE
        if margin >= 0.0:
            message = (
                f"Target reached: Crouzeix ratio {score:.10g} is at least "
                f"{TARGET_SCORE:.10g}."
            )
        elif score > 2.0:
            message = (
                f"Valid Crouzeix construction with ratio {score:.10g}, above 2 "
                "but below the station target. Check numerical robustness carefully."
            )
        else:
            message = (
                f"Valid Crouzeix construction with ratio {score:.10g}. "
                "Higher scores are better."
            )

        details = {
            "Message": message,
            "CrouzeixRatio": float(score),
            "TargetMargin": float(margin),
            "N": int(matrix.shape[0]),
            "Degree": int(degree),
            "NumeratorNorm": float(metrics["numerator_norm"]),
            "DenominatorSup": float(metrics["denominator_sup"]),
            "MatrixOperatorNorm": float(metrics["matrix_operator_norm"]),
            "NumericalRadius": float(metrics["numerical_radius"]),
            "ThetaSamples": int(metrics["theta_samples"]),
            "EdgeSamples": int(metrics["edge_samples"]),
            "StrictVerification": int(metrics["strict_verification"]),
            "CoefficientScale": float(metrics["coefficient_scale"]),
        }
        return True, float(score), details, self._build_sort_key(score, matrix.shape[0], degree)

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_crouzeix" not in content:
            return False, "Submission must define construct_crouzeix()."
        return True, None

    def _build_sort_key(self, score: float, n: int, degree: int) -> Tuple[float, int, int]:
        return (float(score), int(n), int(degree))

    def _parse_submission(self, result: Any) -> Tuple[np.ndarray, np.ndarray, int]:
        if result is None:
            raise AssertionError("construct_crouzeix returned None.")
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

        n = _parse_int(payload.get("n"), "n")
        degree = _parse_int(payload.get("degree"), "degree")
        if n < MIN_N or n > MAX_N:
            raise AssertionError(f"n must satisfy {MIN_N} <= n <= {MAX_N}.")
        if degree < MIN_DEGREE or degree > MAX_DEGREE:
            raise AssertionError(
                f"degree must satisfy {MIN_DEGREE} <= degree <= {MAX_DEGREE}."
            )

        if "matrix" not in payload:
            raise AssertionError("Missing required field 'matrix'.")
        if "coefficients" not in payload:
            raise AssertionError("Missing required field 'coefficients'.")

        matrix = _parse_complex_matrix(payload["matrix"], n)
        coefficients = _parse_complex_vector(payload["coefficients"], degree + 1, "coefficients")
        return matrix, coefficients, degree


def evaluate_crouzeix_ratio(
    matrix: np.ndarray,
    coefficients: np.ndarray,
    degree: int,
) -> Tuple[float, Dict[str, float]]:
    normalized_coefficients, coefficient_scale = _normalize_coefficients(coefficients)

    matrix_operator_norm = _operator_norm(matrix)
    polynomial_matrix = _evaluate_matrix_polynomial(matrix, normalized_coefficients)
    numerator_norm = _operator_norm(polynomial_matrix)
    if not math.isfinite(numerator_norm) or numerator_norm < 0.0:
        raise AssertionError("Computed numerator norm is invalid.")

    base_theta_count = _theta_count(BASE_THETA_MIN, BASE_THETA_PER_DEGREE, degree)
    denominator, denominator_metrics = _estimate_denominator(
        matrix,
        normalized_coefficients,
        base_theta_count,
        BASE_GRID_OFFSETS,
    )
    if denominator <= MIN_DENOMINATOR:
        raise AssertionError(
            f"Denominator estimate is too small ({denominator:.12g}); "
            "the ratio is numerically unstable."
        )

    preliminary_score = numerator_norm / denominator
    strict_verification = 0
    if preliminary_score >= STRICT_SCORE_THRESHOLD:
        strict_theta_count = _theta_count(STRICT_THETA_MIN, STRICT_THETA_PER_DEGREE, degree)
        strict_denominator, strict_metrics = _estimate_denominator(
            matrix,
            normalized_coefficients,
            strict_theta_count,
            STRICT_GRID_OFFSETS,
        )
        denominator = max(denominator, strict_denominator)
        denominator_metrics = strict_metrics
        strict_verification = 1

    score = numerator_norm / denominator
    if not math.isfinite(score) or score < 0.0:
        raise AssertionError("Computed score is invalid.")

    return score, {
        "numerator_norm": numerator_norm,
        "denominator_sup": denominator,
        "matrix_operator_norm": matrix_operator_norm,
        "numerical_radius": denominator_metrics["numerical_radius"],
        "theta_samples": float(denominator_metrics["theta_samples"]),
        "edge_samples": float(denominator_metrics["edge_samples"]),
        "strict_verification": float(strict_verification),
        "coefficient_scale": coefficient_scale,
    }


def _theta_count(minimum: int, per_degree: int, degree: int) -> int:
    count = max(minimum, per_degree * max(1, degree))
    # Keep the periodic grid even, which simplifies adjacent edge sampling.
    if count % 2:
        count += 1
    return int(count)


def _estimate_denominator(
    matrix: np.ndarray,
    coefficients: np.ndarray,
    theta_count: int,
    grid_offsets: Sequence[float],
) -> Tuple[float, Dict[str, float]]:
    denominator = 0.0
    numerical_radius = 0.0
    total_theta_samples = 0
    total_edge_samples = 0

    for offset in grid_offsets:
        boundary_points = _numerical_range_boundary_points(matrix, theta_count, offset)
        numerical_radius = max(numerical_radius, float(np.max(np.abs(boundary_points))))
        total_theta_samples += len(boundary_points)

        boundary_values = np.abs(_evaluate_scalar_polynomial(coefficients, boundary_points))
        if not np.all(np.isfinite(boundary_values)):
            raise AssertionError("Boundary polynomial values are not finite.")
        denominator = max(denominator, float(np.max(boundary_values)))

        if EDGE_FRACTIONS:
            next_points = np.roll(boundary_points, -1)
            for fraction in EDGE_FRACTIONS:
                edge_points = (1.0 - fraction) * boundary_points + fraction * next_points
                edge_values = np.abs(_evaluate_scalar_polynomial(coefficients, edge_points))
                if not np.all(np.isfinite(edge_values)):
                    raise AssertionError("Edge polynomial values are not finite.")
                denominator = max(denominator, float(np.max(edge_values)))
                total_edge_samples += len(edge_points)

    if not math.isfinite(denominator) or denominator < 0.0:
        raise AssertionError("Computed denominator is invalid.")

    return denominator, {
        "numerical_radius": numerical_radius,
        "theta_samples": float(total_theta_samples),
        "edge_samples": float(total_edge_samples),
    }


def _numerical_range_boundary_points(
    matrix: np.ndarray,
    theta_count: int,
    grid_offset: float,
) -> np.ndarray:
    n = matrix.shape[0]
    matrix_adjoint = matrix.conj().T
    points = np.empty(theta_count, dtype=np.complex128)
    theta_step = 2.0 * np.pi / theta_count
    for idx in range(theta_count):
        theta = (idx + grid_offset) * theta_step
        phase = np.exp(1j * theta)
        hermitian = 0.5 * (phase * matrix + np.conjugate(phase) * matrix_adjoint)
        hermitian = 0.5 * (hermitian + hermitian.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
        if not np.all(np.isfinite(eigenvalues)):
            raise AssertionError("Numerical-range eigensolve produced nonfinite eigenvalues.")
        vector = eigenvectors[:, n - 1]
        point = np.vdot(vector, matrix @ vector)
        if not (math.isfinite(float(point.real)) and math.isfinite(float(point.imag))):
            raise AssertionError("Numerical-range boundary point is not finite.")
        points[idx] = point
    return points


def _evaluate_scalar_polynomial(coefficients: np.ndarray, values: np.ndarray) -> np.ndarray:
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = np.full(values.shape, coefficients[-1], dtype=np.complex128)
            for coefficient in coefficients[-2::-1]:
                result = result * values + coefficient
        except FloatingPointError as exc:
            raise AssertionError("Scalar polynomial evaluation overflowed.") from exc
    return result


def _evaluate_matrix_polynomial(matrix: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    identity = np.eye(n, dtype=np.complex128)
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = coefficients[-1] * identity
            for coefficient in coefficients[-2::-1]:
                result = result @ matrix + coefficient * identity
                if not _all_complex_finite(result):
                    raise AssertionError("Matrix polynomial values are not finite.")
        except FloatingPointError as exc:
            raise AssertionError("Matrix polynomial evaluation overflowed.") from exc
    return result


def _operator_norm(matrix: np.ndarray) -> float:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
        raise AssertionError("Operator norm computation failed.")
    return float(singular_values[0])


def _normalize_coefficients(coefficients: np.ndarray) -> Tuple[np.ndarray, float]:
    scale = float(np.max(np.abs(coefficients)))
    if not math.isfinite(scale) or scale < MIN_COEFFICIENT_SCALE:
        raise AssertionError("Polynomial coefficients must not all be zero.")
    return coefficients / scale, scale


def _parse_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssertionError(f"{field_name} must be an integer.")
    return int(value)


def _parse_complex_matrix(value: Any, n: int) -> np.ndarray:
    if not isinstance(value, list):
        raise AssertionError("matrix must be a list.")
    if len(value) != n:
        raise AssertionError(f"matrix must have exactly {n} rows.")

    matrix = np.empty((n, n), dtype=np.complex128)
    for row_idx, row in enumerate(value):
        if not isinstance(row, list) or len(row) != n:
            raise AssertionError(f"matrix[{row_idx}] must contain exactly {n} entries.")
        for col_idx, pair in enumerate(row):
            matrix[row_idx, col_idx] = _parse_complex_pair(
                pair,
                f"matrix[{row_idx}][{col_idx}]",
                MAX_MATRIX_COMPONENT,
            )
    if not _all_complex_finite(matrix):
        raise AssertionError("matrix contains nonfinite values.")
    return matrix


def _parse_complex_vector(value: Any, length: int, field_name: str) -> np.ndarray:
    if not isinstance(value, list):
        raise AssertionError(f"{field_name} must be a list.")
    if len(value) != length:
        raise AssertionError(f"{field_name} must contain exactly {length} entries.")

    vector = np.empty(length, dtype=np.complex128)
    for idx, pair in enumerate(value):
        vector[idx] = _parse_complex_pair(
            pair,
            f"{field_name}[{idx}]",
            MAX_COEFFICIENT_COMPONENT,
        )
    if not _all_complex_finite(vector):
        raise AssertionError(f"{field_name} contains nonfinite values.")
    return vector


def _parse_complex_pair(value: Any, field_name: str, max_component: float) -> complex:
    if not isinstance(value, list) or len(value) != 2:
        raise AssertionError(f"{field_name} must be a [real, imag] pair.")
    real = _parse_real(value[0], f"{field_name}[0]", max_component)
    imag = _parse_real(value[1], f"{field_name}[1]", max_component)
    return complex(real, imag)


def _parse_real(value: Any, field_name: str, max_abs: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"{field_name} must be a real number.")
    number = float(value)
    if not math.isfinite(number):
        raise AssertionError(f"{field_name} must be finite.")
    if abs(number) > max_abs:
        raise AssertionError(f"{field_name} exceeds the magnitude cap {max_abs}.")
    return number


def _all_complex_finite(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values.real)) and np.all(np.isfinite(values.imag)))
