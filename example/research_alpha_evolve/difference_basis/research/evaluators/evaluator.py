"""
Evaluator for compact difference bases of intervals.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator


MIN_N = 1
MAX_N = 50_000_000
MIN_BASIS_SIZE = 2
MAX_BASIS_SIZE = 20_000
MAX_ABS_ENTRY = 10**12


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for finite integer difference bases."""

    def __init__(self):
        super().__init__()

    def get_expected_function_name(self) -> str:
        return "construct_difference_basis"

    def get_task_description(self) -> str:
        return "Compact difference basis construction for integer intervals"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "UpperBound": ".8f",
            "N": "d",
            "BasisSize": "d",
            "Span": "d",
            "Coverage": ".8f",
            "MissingCount": "d",
        }

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, Any, Dict[str, Any], Tuple[float, int, int]]:
        try:
            n, basis = self._parse_submission(result)
            metrics = self._verify_difference_basis(n, basis)
        except AssertionError as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Verification failed: {type(exc).__name__}: {exc}"
            }, (float("-inf"), 0, 0)
        except Exception as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Evaluation error: {type(exc).__name__}: {exc}"
            }, (float("-inf"), 0, 0)

        basis_size = int(metrics["basis_size"])
        score = float(basis_size * basis_size / n)
        message = (
            f"Valid difference basis for n={n} with |B|={basis_size} "
            f"and score {score:.10g}. Lower scores are better."
        )

        details = {
            "Message": message,
            "UpperBound": score,
            "N": int(n),
            "BasisSize": basis_size,
            "Span": int(metrics["span"]),
            "Coverage": 1.0,
            "MissingCount": 0,
        }
        return True, score, details, self._build_sort_key(score, int(n), basis_size)

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_difference_basis" not in content:
            return False, "Submission must define construct_difference_basis()."
        return True, None

    def _build_sort_key(self, score: float, n: int, basis_size: int) -> Tuple[float, int, int]:
        # Research Center ranks larger sort keys first; this task is lower-is-better.
        return (-float(score), int(n), -int(basis_size))

    def _parse_submission(self, result: Any) -> Tuple[int, np.ndarray]:
        if result is None:
            raise AssertionError("construct_difference_basis returned None.")
        if not isinstance(result, str):
            raise AssertionError("Submission must return a JSON string.")

        try:
            payload = json.loads(result)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"Invalid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise AssertionError("Top-level JSON value must be an object.")
        if "n" not in payload:
            raise AssertionError("Missing required field 'n'.")
        if "B" not in payload:
            raise AssertionError("Missing required field 'B'.")

        n = _parse_decimal_int(payload["n"], "n")
        if n < MIN_N or n > MAX_N:
            raise AssertionError(f"n must satisfy {MIN_N} <= n <= {MAX_N}.")

        raw_basis = payload["B"]
        if not isinstance(raw_basis, list):
            raise AssertionError("B must be a list of integers.")

        basis_values = [_parse_decimal_int(value, f"B[{idx}]") for idx, value in enumerate(raw_basis)]
        basis_size = len(basis_values)
        if basis_size < MIN_BASIS_SIZE or basis_size > MAX_BASIS_SIZE:
            raise AssertionError(
                f"Basis size must satisfy {MIN_BASIS_SIZE} <= |B| <= {MAX_BASIS_SIZE}."
            )
        if len(set(basis_values)) != basis_size:
            raise AssertionError("B must not contain duplicate elements.")

        min_value = min(basis_values)
        shifted = [value - min_value for value in basis_values]
        max_shifted = max(shifted)
        if max_shifted > 2 * MAX_ABS_ENTRY:
            raise AssertionError(
                f"Basis span is too large for this evaluator cap ({max_shifted} > {2 * MAX_ABS_ENTRY})."
            )

        basis = np.asarray(sorted(shifted), dtype=np.int64)
        return n, basis

    def _verify_difference_basis(self, n: int, basis: np.ndarray) -> Dict[str, int]:
        basis_size = int(basis.size)
        span = int(basis[-1] - basis[0])
        if span < n:
            raise AssertionError(f"Basis span {span} is smaller than n={n}, so coverage is impossible.")

        covered = np.zeros(n + 1, dtype=bool)
        covered_count = 0

        for upper_idx in range(1, basis_size):
            upper = int(basis[upper_idx])
            lower_start = int(np.searchsorted(basis, upper - n, side="left", sorter=None))
            if lower_start >= upper_idx:
                continue

            diffs = upper - basis[lower_start:upper_idx]
            if diffs.size == 0:
                continue

            new_mask = ~covered[diffs]
            if np.any(new_mask):
                new_diffs = diffs[new_mask]
                covered[new_diffs] = True
                covered_count += int(new_diffs.size)
                if covered_count == n:
                    return {
                        "basis_size": basis_size,
                        "span": span,
                    }

        if covered_count != n:
            missing_values = _first_missing_values(covered)
            raise AssertionError(
                f"B does not cover all values in 1..n. Covered {covered_count}/{n}; "
                f"first missing values: {missing_values}."
            )

        return {
            "basis_size": basis_size,
            "span": span,
        }


def _parse_decimal_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise AssertionError(f"{field_name} must be an integer, not a boolean.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise AssertionError(f"{field_name} must not be an empty string.")
        try:
            parsed = int(text, 10)
        except ValueError as exc:
            raise AssertionError(f"{field_name} must be a base-10 integer string.") from exc
    else:
        raise AssertionError(f"{field_name} must be an integer or base-10 integer string.")

    if abs(parsed) > MAX_ABS_ENTRY:
        raise AssertionError(f"{field_name} exceeds the absolute-value cap {MAX_ABS_ENTRY}.")
    return parsed


def _first_missing_values(covered: np.ndarray, limit: int = 10) -> List[int]:
    values: List[int] = []
    chunk_size = 1_000_000
    offset = 1
    end_limit = int(covered.size)

    while offset < end_limit and len(values) < limit:
        end = min(end_limit, offset + chunk_size)
        missing = np.flatnonzero(~covered[offset:end])
        if missing.size:
            remaining = limit - len(values)
            values.extend((missing[:remaining] + offset).astype(np.int64).tolist())
        offset = end

    return values
