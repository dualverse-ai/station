"""Evaluator for the autoconvolution norm ratio of nonnegative step functions."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator, SeedBatchEvaluation


MIN_STEPS = 2
MAX_STEPS = 100000
DIRECT_CONVOLUTION_MAX_STEPS = 4096


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluate an exact norm ratio for equal-width nonnegative step functions."""

    def get_expected_function_name(self) -> str:
        return "construct_autoconvolution_ratio"

    def get_task_description(self) -> str:
        return "Indicator-like autoconvolution step-function construction"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "NormRatio": ".10f",
            "L2Squared": ".10f",
            "L1Norm": ".10f",
            "LInfinityNorm": ".10f",
            "Steps": "d",
            "NonzeroFraction": ".6f",
        }

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, Any, Dict[str, Any], Tuple[float]]:
        batch = self.evaluate_seed_batch(result, eval_id, author)
        valid_indices = [index for index, valid in enumerate(batch.valid) if bool(valid)]
        if not valid_indices:
            message = next((error for error in batch.errors if error), "All candidates were invalid.")
            return False, constants.RESEARCH_SCORE_NA, {"Message": message}, (float("-inf"),)
        winner = max(valid_indices, key=lambda index: batch.sort_keys[index])
        return True, float(batch.scores[winner]), batch.details[winner], batch.sort_keys[winner]

    def evaluate_seed_batch(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> SeedBatchEvaluation:
        candidates = self._split_candidates(result)
        seeds = []
        scores = []
        valid = []
        sort_keys = []
        details = []
        errors = []
        for candidate in candidates:
            try:
                weights = self._parse_weights(candidate)
                ratio, metrics = self._evaluate_weights(weights)
            except AssertionError as exc:
                message = f"Verification failed: {type(exc).__name__}: {exc}"
                seeds.append(None)
                scores.append(float("-inf"))
                valid.append(False)
                sort_keys.append((float("-inf"),))
                details.append({"Message": message})
                errors.append(message)
                continue
            except Exception as exc:
                message = f"Evaluation error: {type(exc).__name__}: {exc}"
                seeds.append(None)
                scores.append(float("-inf"))
                valid.append(False)
                sort_keys.append((float("-inf"),))
                details.append({"Message": message})
                errors.append(message)
                continue

            seeds.append(weights)
            scores.append(float(ratio))
            valid.append(True)
            sort_keys.append((float(ratio),))
            details.append(self._build_details(weights, ratio, metrics))
            errors.append(None)

        return SeedBatchEvaluation(
            seeds=seeds,
            scores=np.asarray(scores, dtype=np.float64),
            valid=np.asarray(valid, dtype=bool),
            sort_keys=sort_keys,
            details=details,
            errors=errors,
        )

    @staticmethod
    def _split_candidates(result: Any) -> list[Any]:
        if isinstance(result, np.ndarray):
            if result.ndim <= 1:
                return [result]
            if result.ndim == 2:
                return [result[index] for index in range(result.shape[0])]
            return [result]
        if isinstance(result, (list, tuple)):
            if not result:
                return [result]
            if all(
                not isinstance(value, bool) and isinstance(value, (int, float, np.number))
                for value in result
            ):
                return [result]
            return list(result)
        return [result]

    def _build_details(
        self,
        weights: np.ndarray,
        ratio: float,
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        return {
            "Message": (
                f"Valid step function with norm ratio {ratio:.10g}. "
                "Higher scores are better."
            ),
            "NormRatio": float(ratio),
            "L2Squared": float(metrics["l2_squared"]),
            "L1Norm": float(metrics["l1_norm"]),
            "LInfinityNorm": float(metrics["linfinity_norm"]),
            "Steps": int(weights.size),
            "NonzeroFraction": float(np.count_nonzero(weights > 0.0) / weights.size),
        }

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_autoconvolution_ratio" not in content:
            return False, "Submission must define construct_autoconvolution_ratio()."
        return True, None

    def _parse_weights(self, result: Any) -> np.ndarray:
        if result is None:
            raise AssertionError("construct_autoconvolution_ratio returned None.")
        if isinstance(result, np.ndarray):
            weights = np.asarray(result, dtype=np.float64)
        else:
            if not isinstance(result, (list, tuple)):
                raise AssertionError("Weights must be a numeric list or one-dimensional NumPy array.")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                for value in result
            ):
                raise AssertionError("Every weight must be a real numeric value.")
            weights = np.asarray(result, dtype=np.float64)

        if weights.ndim != 1:
            raise AssertionError("Weights must be one-dimensional.")
        if not MIN_STEPS <= weights.size <= MAX_STEPS:
            raise AssertionError(
                f"Step count must satisfy {MIN_STEPS} <= n <= {MAX_STEPS}."
            )
        if not np.all(np.isfinite(weights)):
            raise AssertionError("All weights must be finite.")
        if np.any(weights < 0.0):
            raise AssertionError("All weights must be nonnegative.")

        scale = float(np.max(weights))
        if scale <= 0.0:
            raise AssertionError("At least one weight must be positive.")
        weights = weights / scale
        if float(np.sum(weights)) <= 0.0:
            raise AssertionError("Normalized weight mass is invalid.")
        return weights

    def _evaluate_weights(self, weights: np.ndarray) -> Tuple[float, Dict[str, float]]:
        steps = int(weights.size)
        convolution = self._convolve(weights)
        if convolution.size != 2 * steps - 1:
            raise AssertionError("Internal convolution length mismatch.")

        interval_width = 1.0 / (2.0 * steps)
        knot_values = np.zeros(convolution.size + 2, dtype=np.float64)
        knot_values[1:-1] = interval_width * convolution

        left = knot_values[:-1]
        right = knot_values[1:]
        l2_squared = float(
            interval_width
            * np.sum(left * left + left * right + right * right)
            / 3.0
        )
        mass = interval_width * float(np.sum(weights))
        l1_norm = mass * mass
        linfinity_norm = float(np.max(knot_values))

        denominator = l1_norm * linfinity_norm
        if l2_squared <= 0.0 or denominator <= 0.0:
            raise AssertionError("Computed convolution norms are invalid.")
        ratio = l2_squared / denominator
        if not np.isfinite(ratio):
            raise AssertionError("Computed norm ratio is not finite.")

        return float(ratio), {
            "l2_squared": l2_squared,
            "l1_norm": l1_norm,
            "linfinity_norm": linfinity_norm,
        }

    def _convolve(self, weights: np.ndarray) -> np.ndarray:
        if weights.size <= DIRECT_CONVOLUTION_MAX_STEPS:
            return np.convolve(weights, weights)

        target_size = 2 * int(weights.size) - 1
        fft_size = 1 << (target_size - 1).bit_length()
        spectrum = np.fft.rfft(weights, fft_size)
        convolution = np.fft.irfft(spectrum * spectrum, fft_size)[:target_size]
        return np.maximum(convolution, 0.0)
