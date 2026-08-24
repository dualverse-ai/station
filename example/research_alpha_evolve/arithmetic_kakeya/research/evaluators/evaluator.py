"""Evaluator for finite-grid arithmetic Kakeya entropy witnesses."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator, SeedBatchEvaluation


GRID_SIZE = 83
SLOPE_TUPLE = (1, 2)


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluate canonical finite-grid projection-entropy distributions."""

    def get_expected_function_name(self) -> str:
        return "construct_arithmetic_kakeya"

    def get_task_description(self) -> str:
        return "Finite-grid arithmetic Kakeya entropy witness"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "FullRatio": ".10f",
            "HMinus": ".10f",
            "HMaxProxy": ".10f",
            "HXPlusY": ".10f",
            "Support": "d",
            "Collapsed": "d",
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
        return (
            True,
            float(batch.scores[winner]),
            batch.details[winner],
            batch.sort_keys[winner],
        )

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
                normalized = self._parse_and_normalize(candidate)
                input_support = int(np.count_nonzero(normalized > 0.0))
                canonical = self._ensure_distinct_differences(normalized)
                proxy_ratio, metrics = self._score_canonical(canonical)
                canonical_support = int(np.count_nonzero(canonical > 0.0))
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

            seeds.append(canonical)
            scores.append(float(proxy_ratio))
            valid.append(True)
            sort_keys.append((float(proxy_ratio),))
            details.append(
                {
                    "Message": (
                        f"Valid canonical entropy witness with proxy ratio "
                        f"{proxy_ratio:.10g} and full ratio {metrics['full_ratio']:.10g}. "
                        "Higher primary scores are better."
                    ),
                    "FullRatio": float(metrics["full_ratio"]),
                    "HMinus": float(metrics["h_minus"]),
                    "HMaxProxy": float(metrics["h_max_proxy"]),
                    "HXPlusY": float(metrics["h_x_plus_y"]),
                    "Support": canonical_support,
                    "Collapsed": input_support - canonical_support,
                }
            )
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
            if result.ndim == 2:
                return [result]
            if result.ndim == 3:
                return [result[index] for index in range(result.shape[0])]
            return [result]

        if isinstance(result, (list, tuple)):
            if not result:
                return [result]

            try:
                array = np.asarray(result)
            except (TypeError, ValueError):
                array = None

            if array is not None and array.ndim == 2:
                return [result]
            if array is not None and array.ndim == 3:
                return [result[index] for index in range(len(result))]

            candidate_like = []
            for item in result:
                try:
                    candidate_like.append(np.asarray(item).ndim == 2)
                except (TypeError, ValueError):
                    candidate_like.append(False)
            if candidate_like and all(candidate_like):
                return list(result)

        return [result]

    @staticmethod
    def _parse_and_normalize(candidate: Any) -> np.ndarray:
        if candidate is None:
            raise AssertionError("construct_arithmetic_kakeya returned None.")

        try:
            raw = np.asarray(candidate)
        except (TypeError, ValueError) as exc:
            raise AssertionError("Candidate must be a rectangular numeric matrix.") from exc

        if raw.ndim != 2 or raw.shape != (GRID_SIZE, GRID_SIZE):
            raise AssertionError(
                f"Each candidate must have shape ({GRID_SIZE}, {GRID_SIZE}); got {raw.shape}."
            )

        if raw.dtype.kind not in "iuf":
            if raw.dtype.kind != "O":
                raise AssertionError("Every matrix entry must be a real numeric value.")
            for value in raw.flat:
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, float, np.number)
                ):
                    raise AssertionError("Every matrix entry must be a real numeric value.")
                if isinstance(value, (complex, np.complexfloating)):
                    raise AssertionError("Complex matrix entries are not allowed.")

        if raw.dtype.kind == "b":
            raise AssertionError("Boolean matrix entries are not allowed.")

        try:
            matrix = np.asarray(candidate, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AssertionError("Every matrix entry must be a finite real number.") from exc

        if not np.all(np.isfinite(matrix)):
            raise AssertionError("Every matrix entry must be finite.")

        matrix = np.clip(matrix, 0.0, None)
        total = float(np.sum(matrix))
        if not math.isfinite(total) or total <= 0.0:
            raise AssertionError("The clipped matrix must have positive finite total mass.")
        matrix /= total
        return matrix

    @staticmethod
    def _ensure_distinct_differences(joint_prob_xy: np.ndarray) -> np.ndarray:
        difference_groups: Dict[int, list[Tuple[Tuple[int, int], float]]] = defaultdict(list)

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                probability = float(joint_prob_xy[i, j])
                if probability > 0.0:
                    difference_groups[i - j].append(((i, j), probability))

        fixed = joint_prob_xy.copy()
        for instances in difference_groups.values():
            if len(instances) <= 1:
                continue

            ordered = sorted(instances, key=lambda item: item[1], reverse=True)
            highest_probability_indices = ordered[0][0]
            for indices, _probability in ordered[1:]:
                removed_probability = fixed[indices]
                fixed[indices] = 0.0
                fixed[highest_probability_indices] += removed_probability

            fixed /= np.sum(fixed)

        return fixed

    def _score_canonical(self, joint_prob_xy: np.ndarray) -> Tuple[float, Dict[str, float]]:
        h_joint = self._joint_entropy(joint_prob_xy)
        h_y = self._slope_entropy((0, 1), joint_prob_xy)
        h_x = self._slope_entropy((1, 0), joint_prob_xy)
        h_x_plus_2y = self._slope_entropy(SLOPE_TUPLE, joint_prob_xy)
        h_max_proxy = max(h_y, h_x, h_x_plus_2y)
        if h_max_proxy <= 0.0:
            raise AssertionError("The proxy entropy denominator must be positive.")

        proxy_ratio = h_joint / h_max_proxy
        h_minus = self._slope_entropy((1, -1), joint_prob_xy)
        h_x_plus_y = self._slope_entropy((1, 1), joint_prob_xy)
        h_max_full = max(h_x, h_y, h_x_plus_y, h_x_plus_2y)
        if h_max_full <= 0.0:
            raise AssertionError("The full entropy denominator must be positive.")
        full_ratio = h_minus / h_max_full

        values = [
            proxy_ratio,
            full_ratio,
            h_minus,
            h_max_proxy,
            h_x_plus_y,
        ]
        if not all(math.isfinite(value) for value in values):
            raise AssertionError("Computed entropy metrics must be finite.")

        return float(proxy_ratio), {
            "full_ratio": float(full_ratio),
            "h_minus": float(h_minus),
            "h_max_proxy": float(h_max_proxy),
            "h_x_plus_y": float(h_x_plus_y),
        }

    @staticmethod
    def _joint_entropy(joint_prob_xy: np.ndarray) -> float:
        entropy = 0.0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                probability = float(joint_prob_xy[i, j])
                if probability > 0.0:
                    entropy -= probability * np.log2(probability)
        return float(entropy)

    @staticmethod
    def _slope_entropy(
        slope: Tuple[int, int],
        joint_prob_xy: np.ndarray,
    ) -> float:
        projection_probabilities: Dict[int, float] = defaultdict(float)
        slope_x, slope_y = slope

        for i in range(GRID_SIZE):
            x_value = i + 1
            for j in range(GRID_SIZE):
                probability = float(joint_prob_xy[i, j])
                if probability > 0.0:
                    y_value = j + 1
                    projected_value = slope_x * x_value + slope_y * y_value
                    projection_probabilities[projected_value] += probability

        entropy = 0.0
        for probability in projection_probabilities.values():
            if probability > 0.0:
                entropy -= probability * np.log2(probability)
        return float(entropy)

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_arithmetic_kakeya" not in content:
            return False, "Submission must define construct_arithmetic_kakeya()."
        return True, None
