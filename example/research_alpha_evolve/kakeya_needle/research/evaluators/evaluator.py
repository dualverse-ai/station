"""
Evaluator for Kakeya needle triangle-area constructions.
"""

from __future__ import annotations

import ast
import math
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator


TEST_N_VALUES = (2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129)
TARGET_RELATIVE_AREA = 0.90
BREAKPOINT_EPS = 1e-12


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for finite 2D Kakeya needle triangle constructions."""

    def __init__(self):
        super().__init__()

    def get_expected_function_name(self) -> str:
        return "construct_kakeya"

    def get_task_description(self) -> str:
        return "Kakeya needle triangle-area optimization"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "RelativeArea": ".8f",
            "TargetMargin": ".8f",
            "GeometricMeanRatio": ".8f",
            "MeanRatio": ".8f",
            "BestRatio": ".8f",
            "WorstN": "d",
            "BestN": "d",
            "ValidCount": "d",
        }

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, Any, Dict[str, Any], Tuple[float]]:
        try:
            positions_by_n = self._parse_positions(result)
            score, metrics = self._evaluate_positions(positions_by_n)
        except AssertionError as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Verification failed: {type(exc).__name__}: {exc}"
            }, (float("-inf"),)
        except Exception as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Evaluation error: {type(exc).__name__}: {exc}"
            }, (float("-inf"),)

        target_margin = score - TARGET_RELATIVE_AREA
        ratio_summary = ", ".join(
            f"n={n}: {ratio:.6g}" for n, ratio in metrics["ratios_by_n"]
        )
        if target_margin <= 0.0:
            message = (
                f"Target reached: geometric mean ratio {score:.10g} is at most "
                f"{TARGET_RELATIVE_AREA:.10g}; worst tested ratio is {metrics['worst_ratio']:.10g}. "
                f"Ratios by n: {ratio_summary}."
            )
        else:
            message = (
                f"Valid Kakeya triangle construction with geometric mean ratio {score:.10g}; "
                f"worst tested ratio is {metrics['worst_ratio']:.10g}. Lower scores are better. "
                f"Ratios by n: {ratio_summary}."
            )

        details = {
            "Message": message,
            "RelativeArea": float(score),
            "TargetMargin": float(target_margin),
            "GeometricMeanRatio": float(metrics["geometric_mean_ratio"]),
            "MeanRatio": float(metrics["mean_ratio"]),
            "BestRatio": float(metrics["best_ratio"]),
            "WorstN": int(metrics["worst_n"]),
            "BestN": int(metrics["best_n"]),
            "ValidCount": len(TEST_N_VALUES),
        }
        return True, float(score), details, self._build_sort_key(score)

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_kakeya" not in content:
            return False, "Submission must define construct_kakeya()."
        return True, None

    def _build_sort_key(self, score: float) -> Tuple[float]:
        # Research Center ranks larger sort keys first; this task is lower-is-better.
        return (-float(score),)

    def _parse_positions(self, result: Any) -> Dict[int, np.ndarray]:
        if result is None:
            raise AssertionError("construct_kakeya returned None.")
        if not isinstance(result, str):
            raise AssertionError("Submission must return a string.")

        lines = [line.strip() for line in result.splitlines() if line.strip()]
        if not lines:
            raise AssertionError("Submission is empty.")

        payload_text = "\n".join(lines)
        try:
            payload = ast.literal_eval(payload_text)
        except (SyntaxError, ValueError) as exc:
            raise AssertionError("Payload must be a valid Python literal dictionary.") from exc

        if not isinstance(payload, dict):
            raise AssertionError("Payload must be a Python dictionary.")

        positions_by_n: Dict[int, np.ndarray] = {}
        for n in TEST_N_VALUES:
            value = self._lookup_n_payload(payload, n)
            if not isinstance(value, (list, tuple)):
                raise AssertionError(f"Value for n={n} must be a list of offsets.")

            try:
                positions = np.asarray(value, dtype=np.float64)
            except Exception as exc:
                raise AssertionError(f"Offsets for n={n} must be numeric.") from exc

            if positions.ndim != 1:
                raise AssertionError(f"Offsets for n={n} must be one-dimensional.")
            if int(positions.size) != n:
                raise AssertionError(f"Offsets for n={n} must contain exactly {n} values.")
            if not np.all(np.isfinite(positions)):
                raise AssertionError(f"Offsets for n={n} must all be finite.")

            # Area is invariant under common horizontal translation. Normalizing
            # improves numerical stability for submissions with large offsets.
            positions = positions - float(np.min(positions))
            positions_by_n[n] = positions

        return positions_by_n

    def _lookup_n_payload(self, payload: Dict[Any, Any], n: int) -> Any:
        if n in payload:
            return payload[n]
        key = str(n)
        if key in payload:
            return payload[key]
        raise AssertionError(f"Missing required entry for n={n}.")

    def _evaluate_positions(self, positions_by_n: Dict[int, np.ndarray]) -> Tuple[float, Dict[str, float]]:
        ratios: List[float] = []
        ratios_by_n: List[Tuple[int, float]] = []

        for n in TEST_N_VALUES:
            positions = positions_by_n[n]
            area = triangle_union_area(positions)
            keich_area = keich_triangle_area(n)
            if area <= 0.0 or keich_area <= 0.0:
                raise AssertionError(f"Computed invalid area for n={n}.")
            ratio = area / keich_area
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise AssertionError(f"Computed invalid area ratio for n={n}.")

            ratios.append(float(ratio))
            ratios_by_n.append((n, float(ratio)))

        score = geometric_mean(ratios)
        worst_index = int(np.argmax(ratios))
        best_index = int(np.argmin(ratios))

        return score, {
            "geometric_mean_ratio": float(score),
            "mean_ratio": float(np.mean(ratios)),
            "worst_ratio": float(ratios[worst_index]),
            "best_ratio": float(ratios[best_index]),
            "worst_n": float(TEST_N_VALUES[worst_index]),
            "best_n": float(TEST_N_VALUES[best_index]),
            "ratios_by_n": ratios_by_n,
        }


def geometric_mean(values: Sequence[float]) -> float:
    return float(math.exp(sum(math.log(float(value)) for value in values) / len(values)))


@lru_cache(maxsize=None)
def keich_triangle_area(n: int) -> float:
    return triangle_union_area(np.asarray(keich_positions(n), dtype=np.float64))


def keich_positions(n: int) -> List[float]:
    k = int((n - 1).bit_length())
    if k <= 0:
        raise AssertionError("n must be at least 2.")

    positions: List[float] = []
    for i in range(n):
        fraction = i / n
        bits = []
        for _ in range(k):
            fraction *= 2.0
            bit = int(fraction)
            bits.append(bit)
            fraction -= bit
        total = 0.0
        for j, eps in enumerate(bits, start=1):
            total += (1 - j) * eps * (2.0 ** -j) / k
        positions.append(total)
    min_position = min(positions)
    return [x - min_position for x in positions]


def triangle_union_area(positions: np.ndarray) -> float:
    n = int(positions.size)
    endpoint_lines = _triangle_endpoint_lines(positions)
    breakpoints = _collect_breakpoints(endpoint_lines)

    total_area = 0.0
    for start, end in zip(breakpoints, breakpoints[1:]):
        if end - start <= BREAKPOINT_EPS:
            continue
        start_length = _union_length_at_y(positions, start)
        end_length = _union_length_at_y(positions, end)
        total_area += (end - start) * (start_length + end_length) / 2.0

    single_triangle_area = 0.5 / n
    max_possible = n * single_triangle_area
    if total_area <= 0.0 or total_area > max_possible + 1e-8:
        raise AssertionError(
            f"Internal area check failed: area={total_area}, max_possible={max_possible}."
        )
    return float(total_area)


def _triangle_endpoint_lines(positions: np.ndarray) -> List[Tuple[float, float]]:
    n = int(positions.size)
    lines: List[Tuple[float, float]] = []
    for index, x_value in enumerate(positions, start=1):
        x = float(x_value)
        lines.append((index / n, x))
        lines.append(((index - 1) / n, x + 1.0 / n))
    return lines


def _collect_breakpoints(lines: Sequence[Tuple[float, float]]) -> List[float]:
    breakpoints = [0.0, 1.0]
    for i, (slope_i, intercept_i) in enumerate(lines):
        for slope_j, intercept_j in lines[i + 1:]:
            slope_delta = slope_i - slope_j
            if abs(slope_delta) <= 1e-15:
                continue
            y_cross = (intercept_j - intercept_i) / slope_delta
            if BREAKPOINT_EPS < y_cross < 1.0 - BREAKPOINT_EPS:
                breakpoints.append(float(y_cross))

    breakpoints.sort()
    deduped = [breakpoints[0]]
    for value in breakpoints[1:]:
        if value - deduped[-1] > BREAKPOINT_EPS:
            deduped.append(value)
    if deduped[-1] != 1.0:
        deduped.append(1.0)
    return deduped


def _union_length_at_y(positions: np.ndarray, y: float) -> float:
    intervals = _triangle_intervals_at_y(positions, y)
    return _union_interval_length(intervals)


def _triangle_intervals_at_y(positions: np.ndarray, y: float) -> List[Tuple[float, float]]:
    n = int(positions.size)
    intervals: List[Tuple[float, float]] = []
    for index, x_value in enumerate(positions, start=1):
        x = float(x_value)
        left = x + (index / n) * y
        right = x + (1.0 + (index - 1) * y) / n
        if right < left:
            left, right = right, left
        intervals.append((left, right))
    return intervals


def _union_interval_length(intervals: Iterable[Tuple[float, float]]) -> float:
    sorted_intervals = sorted(intervals)
    current_left = None
    current_right = None
    total = 0.0

    for left, right in sorted_intervals:
        if current_left is None:
            current_left = left
            current_right = right
            continue
        if left <= current_right:
            current_right = max(current_right, right)
            continue
        total += current_right - current_left
        current_left = left
        current_right = right

    if current_left is not None:
        total += current_right - current_left
    return float(total)
