"""
Evaluator for nonnegative step functions with low autocorrelation peak.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, Optional, Tuple

import numpy as np

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator


MODE_HEADER = "#mode=step-v1"
MIN_STEPS = 2
MAX_STEPS = 100000
DIRECT_CONVOLUTION_MAX_STEPS = 4096


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for equal-width nonnegative step functions."""

    def __init__(self):
        super().__init__()

    def get_expected_function_name(self) -> str:
        return "construct_autocorrelation"

    def get_task_description(self) -> str:
        return "Low-peak autocorrelation step-function construction"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "UpperBound": ".8f",
            "Steps": "d",
            "Peak": ".8f",
            "ArgmaxT": ".8f",
            "NonzeroFraction": ".6f",
        }

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, Any, Dict[str, Any], Tuple[float]]:
        try:
            weights = self._parse_weights(result)
            upper_bound, metrics = self._evaluate_weights(weights)
        except AssertionError as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Verification failed: {type(exc).__name__}: {exc}"
            }, (float("-inf"),)
        except Exception as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Evaluation error: {type(exc).__name__}: {exc}"
            }, (float("-inf"),)

        details = {
            "Message": (
                f"Valid step function with upper bound {upper_bound:.10g}. "
                "Lower scores are better."
            ),
            "UpperBound": float(upper_bound),
            "Steps": int(metrics["steps"]),
            "Peak": float(metrics["peak"]),
            "ArgmaxT": float(metrics["argmax_t"]),
            "NonzeroFraction": float(metrics["nonzero_fraction"]),
        }
        return True, float(upper_bound), details, self._build_sort_key(upper_bound)

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_autocorrelation" not in content:
            return False, "Submission must define construct_autocorrelation()."
        return True, None

    def _build_sort_key(self, score: float) -> Tuple[float]:
        # Research Center ranks larger sort keys first; this task is lower-is-better.
        return (-float(score),)

    def _parse_weights(self, result: Any) -> np.ndarray:
        if result is None:
            raise AssertionError("construct_autocorrelation returned None.")
        if not isinstance(result, str):
            raise AssertionError("Submission must return a string.")

        lines = [line.strip() for line in result.splitlines() if line.strip()]
        if not lines:
            raise AssertionError("Submission is empty.")
        if lines[0] != MODE_HEADER:
            raise AssertionError(f"First non-empty line must be exactly {MODE_HEADER!r}.")
        if len(lines) < 2:
            raise AssertionError("Missing weight list after mode header.")

        payload_text = "\n".join(lines[1:])
        try:
            payload = ast.literal_eval(payload_text)
        except (SyntaxError, ValueError) as exc:
            raise AssertionError("Weight payload must be a valid Python literal list.") from exc

        if not isinstance(payload, list):
            raise AssertionError("Weight payload must be a Python list.")

        try:
            weights = np.asarray(payload, dtype=np.float64)
        except Exception as exc:
            raise AssertionError("Weights must be numeric.") from exc

        if weights.ndim != 1:
            raise AssertionError("Weights must be a one-dimensional list.")

        steps = int(weights.size)
        if steps < MIN_STEPS or steps > MAX_STEPS:
            raise AssertionError(f"Step count must satisfy {MIN_STEPS} <= n <= {MAX_STEPS}.")

        if not np.all(np.isfinite(weights)):
            raise AssertionError("All weights must be finite.")
        if np.any(weights < 0.0):
            raise AssertionError("All weights must be nonnegative.")

        max_weight = float(np.max(weights))
        if max_weight <= 0.0:
            raise AssertionError("At least one weight must be positive.")

        weights = weights / max_weight
        mass = float(np.sum(weights))
        if not np.isfinite(mass) or mass <= 0.0:
            raise AssertionError("Normalized weight mass is invalid.")
        return weights

    def _evaluate_weights(self, weights: np.ndarray) -> Tuple[float, Dict[str, float]]:
        steps = int(weights.size)
        if steps <= DIRECT_CONVOLUTION_MAX_STEPS:
            convolution = np.convolve(weights, weights)
        else:
            convolution = self._fft_convolve(weights)

        if convolution.size != 2 * steps - 1:
            raise AssertionError("Internal convolution length mismatch.")

        peak_index = int(np.argmax(convolution))
        peak_discrete = float(convolution[peak_index])
        weight_sum = float(np.sum(weights))

        if peak_discrete <= 0.0 or weight_sum <= 0.0:
            raise AssertionError("Autocorrelation peak or mass is invalid.")

        upper_bound = 2.0 * steps * peak_discrete / (weight_sum * weight_sum)
        if not np.isfinite(upper_bound):
            raise AssertionError("Computed upper bound is not finite.")

        interval_width = 1.0 / (2.0 * steps)
        return float(upper_bound), {
            "steps": float(steps),
            "peak": float(interval_width * peak_discrete),
            "argmax_t": float(-0.5 + (peak_index + 1) * interval_width),
            "nonzero_fraction": float(np.count_nonzero(weights > 0.0) / steps),
        }

    def _fft_convolve(self, weights: np.ndarray) -> np.ndarray:
        target_size = 2 * int(weights.size) - 1
        fft_size = 1 << (target_size - 1).bit_length()
        spectrum = np.fft.rfft(weights, fft_size)
        convolution = np.fft.irfft(spectrum * spectrum, fft_size)[:target_size]
        return np.maximum(convolution, 0.0)
