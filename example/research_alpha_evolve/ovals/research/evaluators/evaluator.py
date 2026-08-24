"""
Evaluator for the Ovals eigenvalue-search task.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
import dataclasses
import enum
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
import scipy.integrate
import scipy.interpolate

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator


EXPECTED_SAMPLES = 64
SPLINE_ORDER = 4
TARGET_SCORE = 1.0
NEGATIVE_INFINITY = -1e10

T_START = 0.0
T_END = 2.0 * np.pi

# Numerical thresholds for the spline-based Ovals evaluator.
NUM_CURVE_SAMPLES = 10000
SPEED_DEGENERACY_THRESHOLD = 1e-4
CURVE_QUADRATURE_PRECISION_THRESHOLD = 1e-5
TOTAL_ARCLENGTH_DEGENERACY_THRESHOLD = 1e-3
PHI_DEGENERACY_THRESHOLD = 1e-3
PHI_QUADRATURE_PRECISION_THRESHOLD = 1e-5

# Format and stability checks around the scorer. These are deliberately loose:
# they reject numerical exploits without changing the mathematical objective.
MAX_ABS_VALUE = 100.0
MAX_ADJACENT_JUMP = 25.0
MAX_CLOSURE_JUMP = 10.0
SCORE_STABILITY_TOLERANCE = 1e-5


class ErrorMessage(enum.Enum):
    """Error messages for the curve evaluation."""

    NO_ERROR = 0
    DEGENERATE_CURVE_SPEED = 1
    DEGENERATE_PHI = 2
    LOW_QUADRATURE_PRECISION = 3
    WRONG_OUTPUT_SHAPE = 4
    FUNCTION_IS_TOO_LARGE = 5
    ZERO_FUNCTION_SCORE = 6
    STOCHASTIC_FUNCTION = 7
    POINTS_NOT_FLOATS = 8
    NEGATIVE_CURVATURE = 9
    DEGENERATE_CURVE_LENGTH = 10
    OTHER_ERROR = 11


@dataclasses.dataclass(frozen=True)
class CurveEvaluationLog:
    score: float
    error_message: ErrorMessage
    arc_length: float = float("nan")
    arc_length_error: float = float("nan")
    phi_l2: float = float("nan")
    phi_l2_error: float = float("nan")
    min_speed: float = float("nan")
    min_signed_curvature: float = float("nan")
    rayleigh_numerator: float = float("nan")
    rayleigh_error: float = float("nan")


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for spline-parametrized Ovals constructions."""

    def __init__(self):
        super().__init__()

    def get_expected_function_name(self) -> str:
        return "construct_ovals"

    def get_task_description(self) -> str:
        return "Ovals eigenvalue-search construction"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "Score": ".10f",
            "TargetMargin": ".10f",
            "RayleighQuotient": ".10f",
            "ArcLength": ".10f",
            "PhiL2": ".10f",
            "MinSpeed": ".8f",
            "MinSignedCurvature": ".8f",
            "MaxAbsValue": ".6f",
            "MaxAdjacentJump": ".6f",
            "Samples": "d",
        }

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, Any, Dict[str, Any], Tuple[float]]:
        try:
            x1, x2, phi, parse_metrics = self._parse_payload(result)
            curve_eval = evaluate_curve(x1, x2, phi)
            if curve_eval.error_message != ErrorMessage.NO_ERROR:
                raise AssertionError(f"Curve evaluation failed: {curve_eval.error_message.name}")

            score = get_score_from_log(curve_eval)
            if not np.isfinite(score):
                raise AssertionError("Computed score is not finite.")
            self._check_score_stability(x1, x2, phi, score)
        except AssertionError as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Verification failed: {type(exc).__name__}: {exc}"
            }, (float("-inf"),)
        except Exception as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Evaluation error: {type(exc).__name__}: {exc}"
            }, (float("-inf"),)

        margin = score - TARGET_SCORE
        if margin <= 0.0:
            message = (
                f"Target reached: score {score:.10g} is below "
                f"or equal to {TARGET_SCORE:.10g}."
            )
        else:
            message = (
                f"Valid Ovals construction with score {score:.10g}. "
                "Lower scores are better."
            )

        details = {
            "Message": message,
            "Score": float(score),
            "TargetMargin": float(margin),
            "RayleighQuotient": float(score),
            "ArcLength": float(curve_eval.arc_length),
            "PhiL2": float(curve_eval.phi_l2),
            "MinSpeed": float(curve_eval.min_speed),
            "MinSignedCurvature": float(curve_eval.min_signed_curvature),
            "MaxAbsValue": float(parse_metrics["max_abs_value"]),
            "MaxAdjacentJump": float(parse_metrics["max_adjacent_jump"]),
            "Samples": int(EXPECTED_SAMPLES),
        }
        return True, float(score), details, self._build_sort_key(score)

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_ovals" not in content:
            return False, "Submission must define construct_ovals()."
        return True, None

    def _build_sort_key(self, score: float) -> Tuple[float]:
        # Research Center ranks larger sort keys first; this task is lower-is-better.
        return (-float(score),)

    def _parse_payload(self, result: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        if result is None:
            raise AssertionError("construct_ovals returned None.")
        if not isinstance(result, str):
            raise AssertionError("Submission must return a string.")

        payload_text = result.strip()
        if not payload_text:
            raise AssertionError("Submission is empty.")

        try:
            payload = ast.literal_eval(payload_text)
        except (SyntaxError, ValueError) as exc:
            raise AssertionError("Payload must be a valid Python literal dictionary.") from exc

        if not isinstance(payload, dict):
            raise AssertionError("Payload must be a Python dictionary.")

        arrays = tuple(self._parse_array(payload, key) for key in ("x", "y", "phi"))
        x1, x2, phi = arrays

        stacked = np.vstack(arrays)
        max_abs_value = float(np.max(np.abs(stacked)))
        if max_abs_value > MAX_ABS_VALUE:
            raise AssertionError(
                f"All values must have absolute value <= {MAX_ABS_VALUE}; got {max_abs_value}."
            )

        max_adjacent_jump = max(_max_circular_adjacent_jump(array) for array in arrays)
        if max_adjacent_jump > MAX_ADJACENT_JUMP:
            raise AssertionError(
                f"Adjacent sample jumps must be <= {MAX_ADJACENT_JUMP}; got {max_adjacent_jump}."
            )

        max_closure_jump = max(abs(float(array[0] - array[-1])) for array in arrays)
        if max_closure_jump > MAX_CLOSURE_JUMP:
            raise AssertionError(
                f"Endpoint jumps must be <= {MAX_CLOSURE_JUMP}; got {max_closure_jump}."
            )

        return x1, x2, phi, {
            "max_abs_value": max_abs_value,
            "max_adjacent_jump": float(max_adjacent_jump),
        }

    def _parse_array(self, payload: Dict[str, Any], key: str) -> np.ndarray:
        if key not in payload:
            raise AssertionError(f"Missing required key {key!r}.")
        value = payload[key]
        if not isinstance(value, (list, tuple)):
            raise AssertionError(f"{key!r} must be a one-dimensional list.")
        try:
            array = np.asarray(value, dtype=np.float64)
        except Exception as exc:
            raise AssertionError(f"{key!r} values must be numeric.") from exc

        if array.ndim != 1:
            raise AssertionError(f"{key!r} must be one-dimensional.")
        if int(array.size) != EXPECTED_SAMPLES:
            raise AssertionError(f"{key!r} must contain exactly {EXPECTED_SAMPLES} values.")
        if not np.all(np.isfinite(array)):
            raise AssertionError(f"{key!r} must contain only finite values.")
        return array

    def _check_score_stability(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        phi: np.ndarray,
        score: float,
    ) -> None:
        shift = EXPECTED_SAMPLES // 3
        shifted_eval = evaluate_curve(
            np.roll(x1, shift),
            np.roll(x2, shift),
            np.roll(phi, shift),
        )
        if shifted_eval.error_message != ErrorMessage.NO_ERROR:
            raise AssertionError(
                f"Cyclic-shift stability check failed: {shifted_eval.error_message.name}"
            )

        shifted_score = get_score_from_log(shifted_eval)
        if abs(shifted_score - score) > SCORE_STABILITY_TOLERANCE:
            raise AssertionError(
                "Cyclic-shift stability check failed: "
                f"score changed from {score:.12g} to {shifted_score:.12g}."
            )


def _max_circular_adjacent_jump(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    circular = np.concatenate([values, values[:1]])
    return float(np.max(np.abs(np.diff(circular))))


def get_quadrature_points_and_weights(grid: np.ndarray, order: int) -> Any:
    """Generates sample points and weights for Gaussian quadrature integration."""
    diff = grid[1:] - grid[:-1]
    sample_points_pattern = (np.polynomial.legendre.leggauss(order)[0] + 1) / 2
    weights_pattern = np.polynomial.legendre.leggauss(order)[1] / 2

    sample_points = (
        np.repeat(grid[:-1][:, None], order, axis=-1)
        + np.repeat(diff[:, None], order, axis=-1) * sample_points_pattern
    )

    weights = np.repeat(diff[:, None], order, axis=-1) * weights_pattern
    return sample_points, weights


def get_cumulative_integration_fn(
    func: Callable[[np.ndarray], np.ndarray],
    order: int = 4,
    add_zero: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a cumulative integration function for a given integrand."""

    def int_func(grid: np.ndarray) -> np.ndarray:
        sample_points, weights = get_quadrature_points_and_weights(grid, order)
        func_samples = func(sample_points.flatten()).reshape(sample_points.shape)
        cumulative_sum = np.cumsum(np.sum(func_samples * weights, axis=1))

        if add_zero:
            return np.append(np.zeros(()), cumulative_sum)
        return cumulative_sum

    return int_func


def get_scipy_spline(f_vals: Sequence[float], spline_order: int):
    values = list(f_vals)
    num_points = len(values)
    values[-1] = values[0]
    xs = np.linspace(0, 2 * np.pi, num_points)
    return scipy.interpolate.make_interp_spline(
        xs,
        values,
        k=spline_order,
        bc_type="periodic",
    )


def _quad(func: Callable[[np.ndarray], np.ndarray], start: float, end: float, limit: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.integrate.IntegrationWarning)
        return scipy.integrate.quad(
            func,
            start,
            end,
            epsrel=1e-14,
            epsabs=1e-14,
            limit=limit,
        )


def evaluate_curve(
    x1: np.ndarray | Sequence[float],
    x2: np.ndarray | Sequence[float],
    phi: np.ndarray | Sequence[float],
) -> CurveEvaluationLog:
    """Returns the Ovals score log for the given curve."""
    x1s = list(np.asarray(x1, dtype=np.float64).copy())
    x2s = list(np.asarray(x2, dtype=np.float64).copy())
    phis = list(np.asarray(phi, dtype=np.float64).copy())

    x1s.append(x1s[0])
    x2s.append(x2s[0])
    phis.append(phis[0])
    x1_fn = get_scipy_spline(x1s, SPLINE_ORDER)
    x2_fn = get_scipy_spline(x2s, SPLINE_ORDER)
    phi_fn = get_scipy_spline(phis, SPLINE_ORDER)

    d1x1 = x1_fn.derivative(1)
    d1x2 = x2_fn.derivative(1)
    dx_norm_fn = lambda x: np.sqrt(d1x1(x) ** 2 + d1x2(x) ** 2)

    xs = np.linspace(T_START, T_END, NUM_CURVE_SAMPLES)
    dx_norms = dx_norm_fn(xs)
    min_speed = float(np.min(dx_norms))

    if min_speed < SPEED_DEGENERACY_THRESHOLD:
        return CurveEvaluationLog(
            score=0.0,
            error_message=ErrorMessage.DEGENERATE_CURVE_SPEED,
            min_speed=min_speed,
        )

    total_arc_length = _quad(
        dx_norm_fn,
        T_START,
        T_END,
        limit=10000,
    )
    arc_length = float(total_arc_length[0])
    arc_length_error = float(total_arc_length[1])

    if arc_length < TOTAL_ARCLENGTH_DEGENERACY_THRESHOLD:
        return CurveEvaluationLog(
            score=0.0,
            error_message=ErrorMessage.DEGENERATE_CURVE_LENGTH,
            arc_length=arc_length,
            arc_length_error=arc_length_error,
            min_speed=min_speed,
        )

    if arc_length_error > CURVE_QUADRATURE_PRECISION_THRESHOLD:
        return CurveEvaluationLog(
            score=0.0,
            error_message=ErrorMessage.LOW_QUADRATURE_PRECISION,
            arc_length=arc_length,
            arc_length_error=arc_length_error,
            min_speed=min_speed,
        )

    phi_l2_norm = _quad(
        lambda x: phi_fn(x) ** 2,
        0,
        2 * np.pi,
        limit=10000,
    )
    phi_l2 = float(phi_l2_norm[0])
    phi_l2_error = float(phi_l2_norm[1])

    if phi_l2 < PHI_DEGENERACY_THRESHOLD:
        return CurveEvaluationLog(
            score=0.0,
            error_message=ErrorMessage.DEGENERATE_PHI,
            arc_length=arc_length,
            arc_length_error=arc_length_error,
            phi_l2=phi_l2,
            phi_l2_error=phi_l2_error,
            min_speed=min_speed,
        )

    if phi_l2_error > PHI_QUADRATURE_PRECISION_THRESHOLD:
        return CurveEvaluationLog(
            score=0.0,
            error_message=ErrorMessage.LOW_QUADRATURE_PRECISION,
            arc_length=arc_length,
            arc_length_error=arc_length_error,
            phi_l2=phi_l2,
            phi_l2_error=phi_l2_error,
            min_speed=min_speed,
        )

    # Rescale to ensure total arclength is 2*pi.
    x1_fn.c = x1_fn.c * 2 * np.pi / arc_length
    x2_fn.c = x2_fn.c * 2 * np.pi / arc_length

    d1x1_fn = x1_fn.derivative(1)
    d2x1 = x1_fn.derivative(2)

    d1x2_fn = x2_fn.derivative(1)
    d2x2 = x2_fn.derivative(2)

    dx_norm_fn = lambda t: np.sqrt(d1x1_fn(t) ** 2 + d1x2_fn(t) ** 2)

    arclength_fn = get_cumulative_integration_fn(dx_norm_fn)
    t_samples = np.linspace(T_START, T_END, NUM_CURVE_SAMPLES)
    s_samples = arclength_fn(t_samples)

    t_of_s = scipy.interpolate.make_interp_spline(
        s_samples,
        t_samples[1:],
        k=SPLINE_ORDER,
    )

    d1t_of_s = t_of_s.derivative(1)
    d2t_of_s = t_of_s.derivative(2)

    s_vals = np.linspace(0, 2 * np.pi, NUM_CURVE_SAMPLES)

    d1c1_fn = lambda s: d1x1(t_of_s(s)) * d1t_of_s(s)
    d1c2_fn = lambda s: d1x2(t_of_s(s)) * d1t_of_s(s)

    d2c1_fn = lambda s: d1x1(t_of_s(s)) * d2t_of_s(s) + d2x1(
        t_of_s(s)
    ) * d1t_of_s(s)

    d2c2_fn = lambda s: d1x2(t_of_s(s)) * d2t_of_s(s) + d2x2(
        t_of_s(s)
    ) * d1t_of_s(s)

    d1phi = phi_fn.derivative(1)

    kappa = lambda s: np.sqrt(d2c1_fn(s) ** 2 + d2c2_fn(s) ** 2)
    signed_kappa = lambda s: (d1c1_fn(s) * d2c2_fn(s) - d2c1_fn(s) * d1c2_fn(s))

    kappas = kappa(s_vals)
    signed_kappas = signed_kappa(s_vals)
    min_signed_curvature = float(np.min(signed_kappas))

    if np.min(kappas) < 0.0 or min_signed_curvature < 0.0:
        return CurveEvaluationLog(
            score=0.0,
            error_message=ErrorMessage.NEGATIVE_CURVATURE,
            arc_length=arc_length,
            arc_length_error=arc_length_error,
            phi_l2=phi_l2,
            phi_l2_error=phi_l2_error,
            min_speed=min_speed,
            min_signed_curvature=min_signed_curvature,
        )

    i_x_phi = _quad(
        lambda s: (d1phi(s) ** 2 + (kappa(s) * phi_fn(s)) ** 2),
        0,
        2 * np.pi,
        limit=NUM_CURVE_SAMPLES,
    )
    rayleigh_numerator = float(i_x_phi[0])
    rayleigh_error = float(i_x_phi[1])

    if rayleigh_error > CURVE_QUADRATURE_PRECISION_THRESHOLD:
        return CurveEvaluationLog(
            score=0.0,
            error_message=ErrorMessage.LOW_QUADRATURE_PRECISION,
            arc_length=arc_length,
            arc_length_error=arc_length_error,
            phi_l2=phi_l2,
            phi_l2_error=phi_l2_error,
            min_speed=min_speed,
            min_signed_curvature=min_signed_curvature,
            rayleigh_numerator=rayleigh_numerator,
            rayleigh_error=rayleigh_error,
        )

    return CurveEvaluationLog(
        score=rayleigh_numerator / phi_l2,
        error_message=ErrorMessage.NO_ERROR,
        arc_length=arc_length,
        arc_length_error=arc_length_error,
        phi_l2=phi_l2,
        phi_l2_error=phi_l2_error,
        min_speed=min_speed,
        min_signed_curvature=min_signed_curvature,
        rayleigh_numerator=rayleigh_numerator,
        rayleigh_error=rayleigh_error,
    )


def get_score_from_log(curve_eval_log: CurveEvaluationLog) -> float:
    if curve_eval_log.error_message != ErrorMessage.NO_ERROR:
        return NEGATIVE_INFINITY
    return curve_eval_log.score


def get_score(
    x1: np.ndarray | Sequence[float],
    x2: np.ndarray | Sequence[float],
    phi: np.ndarray | Sequence[float],
) -> float:
    """Computes the Ovals Rayleigh quotient for a given configuration."""
    return get_score_from_log(evaluate_curve(x1, x2, phi))
