"""Exact finite verifier and scorer for Nikodym sets in F_p^3."""

from __future__ import annotations

import importlib
import json
import math
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
from numba import njit


TEST_PRIMES = (5, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89)
SCORED_PRIMES = tuple(p for p in TEST_PRIMES if p > 10)
DIMENSION = 3
INVALID_INSTANCE_PENALTY = 1000.0


def _emit_eval_json(score: float, details: Dict[str, object]) -> None:
    payload = {"score": float(score), "details": details}
    print(f"EVAL_JSON: {json.dumps(payload, ensure_ascii=False)}", flush=True)


@njit
def is_valid_nikodym_numba(construction: np.ndarray, p: int, d: int) -> bool:
    """Check the punctured-line condition at every point of F_p^3."""
    if d != 3:
        return False

    dummy_tuple = (np.int64(0), np.int64(0), np.int64(0))
    construction_set = {dummy_tuple}
    construction_set.clear()
    for i in range(construction.shape[0]):
        construction_set.add(
            (construction[i, 0], construction[i, 1], construction[i, 2])
        )

    def check_punctured_line_is_contained(point_x, direction_v, const_set):
        for t in range(1, p):
            point_array = (point_x + t * direction_v) % p
            point_on_line = (point_array[0], point_array[1], point_array[2])
            if point_on_line not in const_set:
                return False
        return True

    for x1 in range(p):
        for x2 in range(p):
            for x3 in range(p):
                x = np.array([x1, x2, x3], dtype=np.int64)
                found_line_for_x = False

                for v2 in range(p):
                    if found_line_for_x:
                        break
                    for v3 in range(p):
                        v = np.array([1, v2, v3], dtype=np.int64)
                        if check_punctured_line_is_contained(x, v, construction_set):
                            found_line_for_x = True
                            break
                if found_line_for_x:
                    continue

                for v3 in range(p):
                    v = np.array([0, 1, v3], dtype=np.int64)
                    if check_punctured_line_is_contained(x, v, construction_set):
                        found_line_for_x = True
                        break
                if found_line_for_x:
                    continue

                v = np.array([0, 0, 1], dtype=np.int64)
                if check_punctured_line_is_contained(x, v, construction_set):
                    found_line_for_x = True

                if not found_line_for_x:
                    return False

    return True


def calculate_score(construction: np.ndarray, p: int, d: int) -> float:
    """Return negative unique set size, or positive infinity when invalid."""
    if (
        not isinstance(construction, np.ndarray)
        or construction.ndim != 2
        or construction.shape[1] != d
    ):
        return np.inf

    if construction.shape[0] == 0:
        return np.inf

    try:
        unique_tuples = {tuple(row) for row in construction}
        unique_construction = np.array(list(unique_tuples), dtype=np.int64)
    except Exception:
        return np.inf

    size = unique_construction.shape[0]
    if not is_valid_nikodym_numba(unique_construction, p, d):
        return np.inf

    return -float(size)


def evaluate_callable(
    construction_fn: Callable[[int, int], np.ndarray],
) -> Tuple[float, Dict[str, object]]:
    """Evaluate one submitted construction function on the fixed prime suite."""
    averaging_values: List[float] = []
    valid_ratios: List[Tuple[int, int, float]] = []
    valid_count = 0
    start_time = time.time()

    for p in TEST_PRIMES:
        try:
            construction = construction_fn(p, DIMENSION)
            raw_score = calculate_score(construction, p, DIMENSION)
        except Exception as exc:
            raw_score = np.inf
            print(
                f"p={p}, d={DIMENSION}: ERROR {type(exc).__name__}: {exc}",
                flush=True,
            )

        if raw_score != np.inf:
            valid_count += 1
            size = int(-raw_score)
            denominator = (p - 1) ** 3 + p + 1
            relative_size = size / denominator
            valid_ratios.append((p, size, float(relative_size)))
            if p > 10:
                averaging_values.append(float(relative_size))
            elapsed_minutes = (time.time() - start_time) / 60.0
            print(
                (
                    f"[{elapsed_minutes:7.2f}m] p={p}, d={DIMENSION}: VALID "
                    f"|N|={size}, relative_size={relative_size:.10g}"
                ),
                flush=True,
            )
        else:
            averaging_values.append(INVALID_INSTANCE_PENALTY)
            elapsed_minutes = (time.time() - start_time) / 60.0
            print(
                f"[{elapsed_minutes:7.2f}m] p={p}, d={DIMENSION}: INVALID",
                flush=True,
            )

    mean_value = (
        float(np.mean(np.asarray(averaging_values, dtype=np.float64)))
        if averaging_values
        else INVALID_INSTANCE_PENALTY
    )
    primary_score = -mean_value

    scored_ratios = [entry for entry in valid_ratios if entry[0] > 10]
    if valid_count == len(TEST_PRIMES) and len(scored_ratios) == len(SCORED_PRIMES):
        worst_p, _, worst_ratio = max(scored_ratios, key=lambda entry: entry[2])
        best_p, _, best_ratio = min(scored_ratios, key=lambda entry: entry[2])
        ratio_summary = ", ".join(
            f"p={p}: {size}/{(p - 1) ** 3 + p + 1}={ratio:.6g}"
            for p, size, ratio in scored_ratios
        )
        message = (
            f"Valid Nikodym constructions at all {len(TEST_PRIMES)} tested primes. "
            f"Primary score is {primary_score:.10g}; higher is better. "
            f"Scored relative sizes: {ratio_summary}."
        )
        details = {
            "Message": message,
            "MeanRelativeSize": float(mean_value),
            "WorstRelativeSize": float(worst_ratio),
            "BestRelativeSize": float(best_ratio),
            "WorstP": int(worst_p),
            "BestP": int(best_p),
            "ValidCount": int(valid_count),
            "ScoredCount": int(len(scored_ratios)),
        }
    else:
        message = (
            f"Invalid on {len(TEST_PRIMES) - valid_count} of {len(TEST_PRIMES)} "
            f"tested primes. Invalid instances contribute "
            f"{INVALID_INSTANCE_PENALTY:g} to the evaluator average."
        )
        details = {
            "Message": message,
            "MeanRelativeSize": float(mean_value),
            "WorstRelativeSize": 0.0,
            "BestRelativeSize": 0.0,
            "WorstP": 0,
            "BestP": 0,
            "ValidCount": int(valid_count),
            "ScoredCount": int(len(scored_ratios)),
        }

    if not math.isfinite(primary_score):
        raise AssertionError("Computed primary score is not finite.")
    return float(primary_score), details


def main() -> None:
    try:
        run_module = importlib.import_module("run")
        construction_fn = getattr(run_module, "search_for_best_construction", None)
        if construction_fn is None:
            raise AssertionError(
                "run.py must define search_for_best_construction(p: int, d: int)"
            )
        score, details = evaluate_callable(construction_fn)
        _emit_eval_json(score, details)
    except Exception as exc:
        _emit_eval_json(
            -INVALID_INSTANCE_PENALTY,
            {
                "Message": f"Runner failure: {type(exc).__name__}: {exc}",
                "MeanRelativeSize": INVALID_INSTANCE_PENALTY,
                "WorstRelativeSize": 0.0,
                "BestRelativeSize": 0.0,
                "WorstP": 0,
                "BestP": 0,
                "ValidCount": 0,
                "ScoredCount": 0,
            },
        )


if __name__ == "__main__":
    main()
