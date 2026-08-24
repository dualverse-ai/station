"""Evaluator for equal-width step functions in the minimum-overlap problem."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator, SeedBatchEvaluation


MIN_STEPS = 2
MAX_STEPS = 100000
MASS_TOLERANCE = 1e-8
DIRECT_CORRELATION_MAX_STEPS = 4096


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluate the maximum overlap of a step function and its complement."""

    def get_expected_function_name(self) -> str:
        return "construct_minimum_overlap"

    def get_task_description(self) -> str:
        return "Minimum-overlap step-function construction"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "MaximumOverlap": ".10f",
            "ArgmaxX": ".10f",
            "IntegralF": ".10f",
            "MassError": ".3e",
            "Steps": "d",
            "FMin": ".8f",
            "FMax": ".8f",
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
        parsed: dict[int, Tuple[np.ndarray, float]] = {}

        for index, candidate in enumerate(candidates):
            try:
                weights, integral_f = self._parse_weights(candidate)
            except AssertionError as exc:
                message = f"Verification failed: {type(exc).__name__}: {exc}"
                details[index] = {"Message": message}
                errors[index] = message
            except Exception as exc:
                message = f"Evaluation error: {type(exc).__name__}: {exc}"
                details[index] = {"Message": message}
                errors[index] = message
            else:
                parsed[index] = (weights, integral_f)

        groups: dict[int, list[int]] = {}
        for index, (weights, _integral_f) in parsed.items():
            groups.setdefault(int(weights.size), []).append(index)

        for indices in groups.values():
            weights_group = [parsed[index][0] for index in indices]
            try:
                steps = int(weights_group[0].size)
                use_batched_fft = (
                    len(indices) >= 4
                    and 512 <= steps <= DIRECT_CORRELATION_MAX_STEPS
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
                maximum_overlap, argmax_x = outcome
                integral_f = parsed[index][1]
                seeds[index] = weights
                scores[index] = float(maximum_overlap)
                valid[index] = True
                sort_keys[index] = (-float(maximum_overlap),)
                details[index] = self._build_details(
                    weights,
                    integral_f,
                    maximum_overlap,
                    argmax_x,
                )

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
        integral_f: float,
        maximum_overlap: float,
        argmax_x: float,
    ) -> Dict[str, Any]:
        return {
            "Message": (
                f"Valid step function with maximum overlap {maximum_overlap:.10g}. "
                "Lower scores are better."
            ),
            "MaximumOverlap": float(maximum_overlap),
            "ArgmaxX": float(argmax_x),
            "IntegralF": float(integral_f),
            "MassError": float(abs(integral_f - 1.0)),
            "Steps": int(weights.size),
            "FMin": float(np.min(weights)),
            "FMax": float(np.max(weights)),
        }

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_minimum_overlap" not in content:
            return False, "Submission must define construct_minimum_overlap()."
        return True, None

    def _parse_weights(self, result: Any) -> Tuple[np.ndarray, float]:
        if result is None:
            raise AssertionError("construct_minimum_overlap returned None.")
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
        if not MIN_STEPS <= weights.size <= MAX_STEPS:
            raise AssertionError(
                f"Step count must satisfy {MIN_STEPS} <= n <= {MAX_STEPS}."
            )
        if not np.all(np.isfinite(weights)):
            raise AssertionError("All weights must be finite.")
        if np.any(weights < 0.0) or np.any(weights > 1.0):
            raise AssertionError("All weights must lie in the interval [0, 1].")

        interval_width = 2.0 / float(weights.size)
        integral_f = interval_width * float(np.sum(weights))
        if abs(integral_f - 1.0) > MASS_TOLERANCE:
            raise AssertionError(
                f"Integral of f must equal 1 within tolerance {MASS_TOLERANCE:g}."
            )
        return weights, float(integral_f)

    def _evaluate_equal_length_batch(
        self,
        weights_group: list[np.ndarray],
    ) -> list[Tuple[float, float]]:
        matrix = np.stack(weights_group, axis=0)
        steps = int(matrix.shape[1])
        complement = 1.0 - matrix
        target_size = 2 * steps - 1
        fft_size = 1 << (target_size - 1).bit_length()
        left_spectrum = np.fft.rfft(matrix[:, ::-1], fft_size, axis=1)
        right_spectrum = np.fft.rfft(complement, fft_size, axis=1)
        correlation = np.fft.irfft(
            left_spectrum * right_spectrum,
            fft_size,
            axis=1,
        )[:, :target_size]
        correlation = np.maximum(correlation, 0.0)

        peak_indices = np.argmax(correlation, axis=1)
        peak_values = correlation[np.arange(matrix.shape[0]), peak_indices]
        interval_width = 2.0 / float(steps)
        maximum_overlaps = interval_width * peak_values
        if not np.all(np.isfinite(maximum_overlaps)) or np.any(maximum_overlaps < 0.0):
            raise AssertionError("Computed maximum overlap is invalid.")
        return [
            (
                float(maximum_overlaps[index]),
                float((int(peak_indices[index]) - (steps - 1)) * interval_width),
            )
            for index in range(matrix.shape[0])
        ]

    def _evaluate_weights(self, weights: np.ndarray) -> Tuple[float, float]:
        steps = int(weights.size)
        complement = 1.0 - weights
        correlation = self._cross_correlate(weights, complement)
        if correlation.size != 2 * steps - 1:
            raise AssertionError("Internal cross-correlation length mismatch.")

        peak_index = int(np.argmax(correlation))
        interval_width = 2.0 / float(steps)
        maximum_overlap = interval_width * float(correlation[peak_index])
        argmax_x = (peak_index - (steps - 1)) * interval_width
        if maximum_overlap < 0.0 or not np.isfinite(maximum_overlap):
            raise AssertionError("Computed maximum overlap is invalid.")
        return float(maximum_overlap), float(argmax_x)

    def _cross_correlate(self, weights: np.ndarray, complement: np.ndarray) -> np.ndarray:
        if weights.size <= DIRECT_CORRELATION_MAX_STEPS:
            return np.convolve(weights[::-1], complement)

        target_size = 2 * int(weights.size) - 1
        fft_size = 1 << (target_size - 1).bit_length()
        left_spectrum = np.fft.rfft(weights[::-1], fft_size)
        right_spectrum = np.fft.rfft(complement, fft_size)
        correlation = np.fft.irfft(
            left_spectrum * right_spectrum,
            fft_size,
        )[:target_size]
        return np.maximum(correlation, 0.0)
