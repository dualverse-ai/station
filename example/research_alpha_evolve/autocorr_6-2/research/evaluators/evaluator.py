"""
Evaluator for nonnegative step functions with low autocorrelation peak.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator, SeedBatchEvaluation


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
        count = len(candidates)
        seeds: list[Any] = [None] * count
        scores = np.full(count, float("-inf"), dtype=np.float64)
        valid = np.zeros(count, dtype=bool)
        sort_keys = [(float("-inf"),)] * count
        details: list[Dict[str, Any]] = [{} for _ in range(count)]
        errors: list[Optional[str]] = [None] * count
        parsed: dict[int, np.ndarray] = {}

        for index, candidate in enumerate(candidates):
            try:
                weights = self._parse_weights(candidate)
            except AssertionError as exc:
                message = f"Verification failed: {type(exc).__name__}: {exc}"
                details[index] = {"Message": message}
                errors[index] = message
            except Exception as exc:
                message = f"Evaluation error: {type(exc).__name__}: {exc}"
                details[index] = {"Message": message}
                errors[index] = message
            else:
                parsed[index] = weights

        groups: dict[int, list[int]] = {}
        for index, weights in parsed.items():
            groups.setdefault(int(weights.size), []).append(index)

        for indices in groups.values():
            weights_group = [parsed[index] for index in indices]
            try:
                steps = int(weights_group[0].size)
                use_batched_fft = (
                    len(indices) >= 4
                    and 512 <= steps <= DIRECT_CONVOLUTION_MAX_STEPS
                )
                if use_batched_fft:
                    evaluated = self._evaluate_equal_length_batch(weights_group)
                else:
                    evaluated = [self._evaluate_weights(weights) for weights in weights_group]
            except Exception:
                evaluated = []
                for weights in weights_group:
                    try:
                        evaluated.append(self._evaluate_weights(weights))
                    except Exception as exc:
                        evaluated.append(exc)

            for index, weights, outcome in zip(indices, weights_group, evaluated):
                if isinstance(outcome, Exception):
                    message = f"Evaluation error: {type(outcome).__name__}: {outcome}"
                    details[index] = {"Message": message}
                    errors[index] = message
                    continue
                upper_bound, metrics = outcome
                seeds[index] = weights
                scores[index] = float(upper_bound)
                valid[index] = True
                sort_keys[index] = self._build_sort_key(upper_bound)
                details[index] = self._build_details(weights, upper_bound, metrics)

        for index in range(count):
            if not valid[index] and errors[index] is None:
                message = "Evaluation error: candidate was not evaluated."
                details[index] = {"Message": message}
                errors[index] = message

        return SeedBatchEvaluation(
            seeds=seeds,
            scores=scores,
            valid=valid,
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
        upper_bound: float,
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        return {
            "Message": (
                f"Valid step function with upper bound {upper_bound:.10g}. "
                "Lower scores are better."
            ),
            "UpperBound": float(upper_bound),
            "Steps": int(weights.size),
            "Peak": float(metrics["peak"]),
            "ArgmaxT": float(metrics["argmax_t"]),
            "NonzeroFraction": float(metrics["nonzero_fraction"]),
        }

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

    def _evaluate_equal_length_batch(
        self,
        weights_group: list[np.ndarray],
    ) -> list[Tuple[float, Dict[str, float]]]:
        matrix = np.stack(weights_group, axis=0)
        steps = int(matrix.shape[1])
        target_size = 2 * steps - 1
        fft_size = 1 << (target_size - 1).bit_length()
        spectrum = np.fft.rfft(matrix, fft_size, axis=1)
        convolution = np.fft.irfft(spectrum * spectrum, fft_size, axis=1)[:, :target_size]
        convolution = np.maximum(convolution, 0.0)

        peak_indices = np.argmax(convolution, axis=1)
        peak_discrete = convolution[np.arange(matrix.shape[0]), peak_indices]
        weight_sums = np.sum(matrix, axis=1)
        upper_bounds = 2.0 * steps * peak_discrete / (weight_sums * weight_sums)
        if not np.all(np.isfinite(upper_bounds)) or np.any(upper_bounds <= 0.0):
            raise AssertionError("Computed upper bound is invalid.")

        interval_width = 1.0 / (2.0 * steps)
        nonzero = np.count_nonzero(matrix > 0.0, axis=1) / float(steps)
        return [
            (
                float(upper_bounds[index]),
                {
                    "steps": float(steps),
                    "peak": float(interval_width * peak_discrete[index]),
                    "argmax_t": float(
                        -0.5 + (int(peak_indices[index]) + 1) * interval_width
                    ),
                    "nonzero_fraction": float(nonzero[index]),
                },
            )
            for index in range(matrix.shape[0])
        ]

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
