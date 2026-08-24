"""
Evaluator for finite prime-counting weight certificates.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator


MODE = "prime-weight-v1"
MIN_SUPPORT_SIZE = 1
MAX_SUPPORT_SIZE = 256
MAX_KEY = 1_000_000_000
WEIGHT_CLIP_MIN = -10.0
WEIGHT_CLIP_MAX = 10.0
SAMPLE_COUNT = 10_000_000
SAMPLE_RANDOM_SEED = 627
SAMPLE_RANGE_MULTIPLIER = 10
FLOOR_SUM_TOLERANCE = 1.0001
SCORE_CEILING = 1.0001
EXACT_PREFIX_LIMIT = 100_000
DEFAULT_BATCH_SIZE = 200_000


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for finite Chebyshev-type weight certificates."""

    def __init__(self):
        super().__init__()

    def get_expected_function_name(self) -> str:
        return "construct_prime_weights"

    def get_task_description(self) -> str:
        return "Prime-counting finite weight certificate"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "Score": ".10f",
            "SupportSize": "d",
            "MaxKey": "d",
            "ExactPrefixCount": "d",
            "MaxExactPrefixFloorSum": ".8f",
            "SampleCount": "d",
            "MaxSampledFloorSum": ".8f",
            "NormalizationAdjustment": ".8f",
        }

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, Any, Dict[str, Any], Tuple[float]]:
        try:
            submitted_weights = self._parse_submission(result)
            score, metrics = evaluate_weight_certificate(
                submitted_weights,
                sample_count=SAMPLE_COUNT,
            )
        except AssertionError as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Verification failed: {type(exc).__name__}: {exc}"
            }, (float("-inf"),)
        except Exception as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Evaluation error: {type(exc).__name__}: {exc}"
            }, (float("-inf"),)

        if not metrics["feasible"]:
            check_label = (
                "Exact-prefix floor-sum feasibility check"
                if metrics["failure_check"] == "exact_prefix"
                else "Sampled floor-sum feasibility check"
            )
            message = (
                f"{check_label} failed: "
                f"maximum checked floor sum was {metrics['max_floor_sum']:.10g}, "
                f"above {FLOOR_SUM_TOLERANCE:.10g}. "
                f"First violation occurred at x={metrics['first_violation_x']:.10g} "
                f"with floor sum {metrics['first_violation_value']:.10g}."
            )
            return False, constants.RESEARCH_SCORE_NA, {"Message": message}, (float("-inf"),)

        message = (
            f"Valid prime-counting weight certificate with score {score:.10g}. "
            f"Maximum sampled floor sum was {metrics['max_floor_sum']:.10g}. "
            "Higher scores are better."
        )
        details = {
            "Message": message,
            "Score": float(score),
            "SupportSize": int(metrics["support_size"]),
            "MaxKey": int(metrics["max_key"]),
            "ExactPrefixCount": int(metrics["exact_prefix_count"]),
            "MaxExactPrefixFloorSum": float(metrics["max_exact_prefix_floor_sum"]),
            "SampleCount": int(metrics["sample_count"]),
            "MaxSampledFloorSum": float(metrics["max_sampled_floor_sum"]),
            "NormalizationAdjustment": float(metrics["normalization_adjustment"]),
        }
        return True, float(score), details, (float(score),)

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_prime_weights" not in content:
            return False, "Submission must define construct_prime_weights()."
        return True, None

    def _parse_submission(self, result: Any) -> Dict[int, float]:
        if result is None:
            raise AssertionError("construct_prime_weights returned None.")
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
        if "weights" not in payload:
            raise AssertionError("Missing required field 'weights'.")
        if not isinstance(payload["weights"], dict):
            raise AssertionError("weights must be a JSON object.")

        raw_weights = payload["weights"]
        support_size = len(raw_weights)
        if support_size < MIN_SUPPORT_SIZE or support_size > MAX_SUPPORT_SIZE:
            raise AssertionError(
                f"Support size must satisfy {MIN_SUPPORT_SIZE} <= size <= {MAX_SUPPORT_SIZE}."
            )

        parsed: Dict[int, float] = {}
        for raw_key, raw_value in raw_weights.items():
            key = _parse_positive_integer_key(raw_key)
            if key > MAX_KEY:
                raise AssertionError(f"Key {key} exceeds the maximum allowed key {MAX_KEY}.")
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise AssertionError(f"Weight for key {key} must be a real number.")
            value = float(raw_value)
            if not math.isfinite(value):
                raise AssertionError(f"Weight for key {key} must be finite.")
            parsed[key] = value

        return parsed


def evaluate_weight_certificate(
    submitted_weights: Dict[int, float],
    *,
    sample_count: int = SAMPLE_COUNT,
    seed: int = SAMPLE_RANDOM_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate a finite weight certificate with fixed-seed sampled checks."""

    if sample_count <= 0:
        raise AssertionError("sample_count must be positive.")
    if batch_size <= 0:
        raise AssertionError("batch_size must be positive.")

    weights = {
        int(key): float(np.clip(value, WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX))
        for key, value in submitted_weights.items()
    }
    old_weight_at_one = float(weights.get(1, 0.0))
    reciprocal_sum = math.fsum(value / key for key, value in weights.items())
    weights[1] = old_weight_at_one - reciprocal_sum
    normalization_adjustment = float(weights[1] - old_weight_at_one)

    keys = np.asarray(list(weights.keys()), dtype=np.int64)
    values = np.asarray([weights[int(key)] for key in keys], dtype=np.float64)
    max_key = int(np.max(keys))
    upper_bound_for_sampling = float(SAMPLE_RANGE_MULTIPLIER * max_key)

    score = -math.fsum(
        float(value) * math.log(int(key)) / int(key)
        for key, value in weights.items()
    )
    if not math.isfinite(score):
        raise AssertionError("Computed score is not finite.")
    if score > SCORE_CEILING:
        raise AssertionError(
            f"Score {score:.10g} exceeds the mathematical ceiling {SCORE_CEILING:.10g}. "
            "Such a result cannot be a valid global prime-counting certificate."
        )

    exact_prefix_count = min(EXACT_PREFIX_LIMIT, int(upper_bound_for_sampling))
    exact_prefix_feasibility = _exact_integer_floor_sum_feasibility(
        keys=keys,
        values=values,
        upper_bound=exact_prefix_count,
        batch_size=batch_size,
    )

    if not exact_prefix_feasibility["feasible"]:
        feasibility = exact_prefix_feasibility
        sampled_max_floor_sum = float("nan")
    else:
        feasibility = _sample_floor_sum_feasibility(
            keys=keys,
            values=values,
            sample_count=sample_count,
            seed=seed,
            upper_bound_for_sampling=upper_bound_for_sampling,
            batch_size=batch_size,
        )
        sampled_max_floor_sum = float(feasibility["max_floor_sum"])

    metrics: Dict[str, Any] = {
        "feasible": bool(feasibility["feasible"]),
        "failure_check": feasibility.get("failure_check"),
        "support_size": int(len(weights)),
        "max_key": int(max_key),
        "exact_prefix_count": int(exact_prefix_count),
        "max_exact_prefix_floor_sum": float(exact_prefix_feasibility["max_floor_sum"]),
        "sample_count": int(sample_count),
        "max_floor_sum": float(feasibility["max_floor_sum"]),
        "max_sampled_floor_sum": sampled_max_floor_sum,
        "normalization_adjustment": float(normalization_adjustment),
        "first_violation_x": float(feasibility["first_violation_x"]),
        "first_violation_value": float(feasibility["first_violation_value"]),
    }
    return float(score), metrics


def _exact_integer_floor_sum_feasibility(
    *,
    keys: np.ndarray,
    values: np.ndarray,
    upper_bound: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Check every integer floor-sum row through the inclusive upper bound."""

    max_floor_sum = float("-inf")
    first_violation_x = float("nan")
    first_violation_value = float("nan")

    for start in range(1, int(upper_bound) + 1, int(batch_size)):
        stop = min(int(upper_bound) + 1, start + int(batch_size))
        x_values = np.arange(start, stop, dtype=np.int64)
        floor_sums = np.zeros(len(x_values), dtype=np.float64)
        for key, value in zip(keys, values):
            floor_sums += float(value) * (x_values // int(key))

        batch_max_index = int(np.argmax(floor_sums))
        batch_max = float(floor_sums[batch_max_index])
        if batch_max > max_floor_sum:
            max_floor_sum = batch_max

        violation_indices = np.flatnonzero(floor_sums > FLOOR_SUM_TOLERANCE)
        if violation_indices.size:
            first_index = int(violation_indices[0])
            first_violation_x = float(x_values[first_index])
            first_violation_value = float(floor_sums[first_index])
            return {
                "feasible": False,
                "failure_check": "exact_prefix",
                "max_floor_sum": max_floor_sum,
                "first_violation_x": first_violation_x,
                "first_violation_value": first_violation_value,
            }

    return {
        "feasible": True,
        "failure_check": None,
        "max_floor_sum": max_floor_sum,
        "first_violation_x": first_violation_x,
        "first_violation_value": first_violation_value,
    }


def _sample_floor_sum_feasibility(
    *,
    keys: np.ndarray,
    values: np.ndarray,
    sample_count: int,
    seed: int,
    upper_bound_for_sampling: float,
    batch_size: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples_remaining = int(sample_count)
    max_floor_sum = float("-inf")
    first_violation_x = float("nan")
    first_violation_value = float("nan")

    while samples_remaining > 0:
        current_batch_size = min(int(batch_size), samples_remaining)
        x_samples = rng.uniform(1.0, upper_bound_for_sampling, size=current_batch_size)
        floor_sums = np.zeros(current_batch_size, dtype=np.float64)
        for key, value in zip(keys, values):
            floor_sums += float(value) * (x_samples // int(key))

        batch_max_index = int(np.argmax(floor_sums))
        batch_max = float(floor_sums[batch_max_index])
        if batch_max > max_floor_sum:
            max_floor_sum = batch_max

        violation_indices = np.flatnonzero(floor_sums > FLOOR_SUM_TOLERANCE)
        if violation_indices.size:
            first_index = int(violation_indices[0])
            first_violation_x = float(x_samples[first_index])
            first_violation_value = float(floor_sums[first_index])
            return {
                "feasible": False,
                "failure_check": "sampled",
                "max_floor_sum": max_floor_sum,
                "first_violation_x": first_violation_x,
                "first_violation_value": first_violation_value,
            }

        samples_remaining -= current_batch_size

    return {
        "feasible": True,
        "failure_check": None,
        "max_floor_sum": max_floor_sum,
        "first_violation_x": first_violation_x,
        "first_violation_value": first_violation_value,
    }


def _parse_positive_integer_key(raw_key: Any) -> int:
    if isinstance(raw_key, bool):
        raise AssertionError("Weight keys must be positive integers written as strings.")
    if isinstance(raw_key, int):
        key = raw_key
    elif isinstance(raw_key, str):
        stripped = raw_key.strip()
        if not stripped or not stripped.isdigit():
            raise AssertionError(f"Invalid positive integer key: {raw_key!r}.")
        key = int(stripped)
    else:
        raise AssertionError(f"Invalid positive integer key: {raw_key!r}.")

    if key <= 0:
        raise AssertionError(f"Key {key} must be positive.")
    return key
