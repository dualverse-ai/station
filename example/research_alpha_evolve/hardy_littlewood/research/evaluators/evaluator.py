"""
Evaluator for finite interval-union constructions.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator


MODE = "interval-union-v1"
MIN_N = 2
MAX_N = 1000
MAX_ABS_VALUE = 1.0e100

# These limits are applied after total-weight normalization, so they are
# scale-free. They reject numerically degenerate constructions whose score would
# depend on double-precision endpoint roundoff rather than stable interval
# geometry.
MAX_NORMALIZED_SPAN = 1.0e9
MIN_NORMALIZED_GAP = 1.0e-12
MIN_NORMALIZED_WEIGHT = 1.0e-14
MAX_WEIGHT_DYNAMIC_RANGE = 1.0e10


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for finite interval-union witnesses."""

    def __init__(self):
        super().__init__()

    def get_expected_function_name(self) -> str:
        return "construct_interval_union"

    def get_task_description(self) -> str:
        return "Finite interval-union construction"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "LowerBound": ".8f",
            "N": "d",
            "UnionLength": ".8f",
            "TotalWeight": ".8f",
            "NonemptyIntervals": "d",
            "MaxWeight": ".8f",
            "MinGap": ".8f",
            "Span": ".8f",
        }

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, Any, Dict[str, Any], Tuple[float, int]]:
        try:
            y_values, k_values = self._parse_submission(result)
            lower_bound, metrics = self._evaluate_sequences(y_values, k_values)
        except AssertionError as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Verification failed: {type(exc).__name__}: {exc}"
            }, (float("-inf"), 0)
        except Exception as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Evaluation error: {type(exc).__name__}: {exc}"
            }, (float("-inf"), 0)

        message = (
            f"Valid interval-union construction with score {lower_bound:.10g}. "
            "Higher scores are better."
        )

        details = {
            "Message": message,
            "LowerBound": float(lower_bound),
            "N": int(len(y_values)),
            "UnionLength": float(metrics["union_length_original_scale"]),
            "TotalWeight": float(metrics["total_weight"]),
            "NonemptyIntervals": int(metrics["nonempty_intervals"]),
            "MaxWeight": float(max(k_values)),
            "MinGap": float(metrics["min_gap"]),
            "Span": float(y_values[-1] - y_values[0]),
        }
        return True, float(lower_bound), details, self._build_sort_key(lower_bound, len(y_values))

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_interval_union" not in content:
            return False, "Submission must define construct_interval_union()."
        return True, None

    def _build_sort_key(self, score: float, n: int) -> Tuple[float, int]:
        return (float(score), int(n))

    def _parse_submission(self, result: Any) -> Tuple[List[float], List[float]]:
        if result is None:
            raise AssertionError("construct_interval_union returned None.")
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
        if "y" not in payload:
            raise AssertionError("Missing required field 'y'.")
        if "k" not in payload:
            raise AssertionError("Missing required field 'k'.")

        y_values = _parse_float_list(payload["y"], "y")
        k_values = _parse_float_list(payload["k"], "k")
        n = len(y_values)
        if n < MIN_N or n > MAX_N:
            raise AssertionError(f"Sequence length must satisfy {MIN_N} <= n <= {MAX_N}.")
        if len(k_values) != n:
            raise AssertionError("y and k must have the same length.")

        for idx in range(1, n):
            if y_values[idx] <= y_values[idx - 1]:
                raise AssertionError("y must be strictly increasing.")
        for idx, value in enumerate(k_values):
            if value <= 0.0:
                raise AssertionError(f"k[{idx}] must be positive.")

        total_weight = math.fsum(k_values)
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            raise AssertionError("Total weight is invalid.")

        return y_values, k_values

    def _evaluate_sequences(
        self,
        y_values: Sequence[float],
        k_values: Sequence[float],
    ) -> Tuple[float, Dict[str, float]]:
        total_weight = math.fsum(k_values)
        y0 = y_values[0]

        # Normalize to reduce roundoff. The score is invariant under this
        # translation and common positive scaling.
        y_normalized = [(value - y0) / total_weight for value in y_values]
        k_normalized = [value / total_weight for value in k_values]
        if not all(math.isfinite(value) for value in y_normalized):
            raise AssertionError("Normalized y values are not finite.")
        if not all(math.isfinite(value) for value in k_normalized):
            raise AssertionError("Normalized k values are not finite.")

        self._check_numeric_conditioning(y_normalized, k_normalized)

        intervals: List[Tuple[float, float]] = []
        prefix = [0.0]
        running = 0.0
        for value in k_normalized:
            running += value
            prefix.append(running)

        n = len(y_normalized)
        for i in range(n):
            yi = y_normalized[i]
            for j in range(i, n):
                segment_weight = prefix[j + 1] - prefix[i]
                left = y_normalized[j] - segment_weight
                right = yi + segment_weight
                if left <= right:
                    intervals.append((left, right))

        if not intervals:
            raise AssertionError("Construction produced no nonempty intervals.")

        union_length_normalized = _merged_interval_length(intervals)
        if not math.isfinite(union_length_normalized) or union_length_normalized <= 0.0:
            raise AssertionError("Computed union length is invalid.")

        lower_bound = union_length_normalized / 2.0

        min_gap = min(y_values[idx] - y_values[idx - 1] for idx in range(1, n))
        return lower_bound, {
            "union_length_original_scale": union_length_normalized * total_weight,
            "total_weight": total_weight,
            "nonempty_intervals": float(len(intervals)),
            "min_gap": min_gap,
        }

    def _check_numeric_conditioning(
        self,
        y_normalized: Sequence[float],
        k_normalized: Sequence[float],
    ) -> None:
        span = y_normalized[-1] - y_normalized[0]
        if span > MAX_NORMALIZED_SPAN:
            raise AssertionError(
                "Degenerate construction: after normalizing total weight to 1, "
                f"the coordinate span is {span:.6g}, above the stability limit "
                f"{MAX_NORMALIZED_SPAN:.6g}. Remove far-away or near-irrelevant "
                "scale padding and keep the useful interval geometry in a stable "
                "numeric range."
            )

        min_gap = min(y_normalized[idx] - y_normalized[idx - 1] for idx in range(1, len(y_normalized)))
        if min_gap < MIN_NORMALIZED_GAP:
            raise AssertionError(
                "Degenerate construction: adjacent normalized coordinates are "
                f"separated by only {min_gap:.6g}, below the stability limit "
                f"{MIN_NORMALIZED_GAP:.6g}. The score would be sensitive to "
                "floating-point ordering and endpoint merging; use a construction "
                "whose distinct atoms remain numerically separated after normalization."
            )

        min_weight = min(k_normalized)
        max_weight = max(k_normalized)
        if min_weight < MIN_NORMALIZED_WEIGHT:
            raise AssertionError(
                "Degenerate construction: at least one normalized weight is "
                f"{min_weight:.6g}, below the stability limit "
                f"{MIN_NORMALIZED_WEIGHT:.6g}. Tiny weights can create apparent "
                "micro-interval gains that are not reliable in double precision; "
                "remove negligible atoms or rebalance the weight scale."
            )

        dynamic_range = max_weight / min_weight
        if dynamic_range > MAX_WEIGHT_DYNAMIC_RANGE:
            raise AssertionError(
                "Degenerate construction: normalized weights have dynamic range "
                f"{dynamic_range:.6g}, above the stability limit "
                f"{MAX_WEIGHT_DYNAMIC_RANGE:.6g}. Extremely unbalanced weights "
                "make the score sensitive to numerical artifacts; use a better "
                "conditioned finite construction."
            )


def _parse_float_list(value: Any, field_name: str) -> List[float]:
    if not isinstance(value, list):
        raise AssertionError(f"{field_name} must be a list.")

    parsed: List[float] = []
    for idx, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AssertionError(f"{field_name}[{idx}] must be a real number.")
        number = float(item)
        if not math.isfinite(number):
            raise AssertionError(f"{field_name}[{idx}] must be finite.")
        if abs(number) > MAX_ABS_VALUE:
            raise AssertionError(f"{field_name}[{idx}] exceeds the magnitude cap {MAX_ABS_VALUE}.")
        parsed.append(number)
    return parsed


def _merged_interval_length(intervals: Sequence[Tuple[float, float]]) -> float:
    ordered = sorted(intervals)
    total = 0.0
    current_left, current_right = ordered[0]

    for left, right in ordered[1:]:
        if left <= current_right:
            if right > current_right:
                current_right = right
        else:
            total += current_right - current_left
            current_left, current_right = left, right

    total += current_right - current_left
    return total
