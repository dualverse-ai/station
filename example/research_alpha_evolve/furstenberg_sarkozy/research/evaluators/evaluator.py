"""Evaluator for power-difference-free subsets of square-free cyclic groups."""

from __future__ import annotations

import json
import math
from numbers import Integral
from typing import Any, Dict, Optional, Tuple

import numpy as np

from station import constants
from station.eval_research.base_evaluator import (
    ResearchTaskEvaluator,
    SeedBatchEvaluation,
)


MODE = "power-difference-v1"
ALLOWED_POWERS = frozenset({2, 3, 4, 5, 6})
MIN_MODULUS = 2
MAX_MODULUS = 1_000_000
MAX_SET_SIZE = 25_000


class Task1Evaluator(ResearchTaskEvaluator):
    """Verify cyclic witnesses and score their induced lower-bound exponent."""

    def get_expected_function_name(self) -> str:
        return "construct_power_difference_set"

    def get_task_description(self) -> str:
        return "Power-difference-free cyclic lower-bound construction"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "K": "d",
            "Modulus": "d",
            "SetSize": "d",
            "CyclicContribution": ".8f",
            "ResidueDensity": ".8f",
            "PowerResidues": "d",
        }

    def evaluate_submission(
        self,
        result: Any,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[Any, ...]:
        batch = self.evaluate_seed_batch(result, eval_id, author)
        if not bool(batch.valid[0]):
            return (
                False,
                constants.RESEARCH_SCORE_NA,
                batch.details[0],
                (float("-inf"),),
            )
        return (
            True,
            float(batch.scores[0]),
            batch.details[0],
            batch.sort_keys[0],
        )

    def evaluate_seed_batch(
        self,
        result: Any,
        eval_id: str = None,
        author: str = None,
    ) -> SeedBatchEvaluation:
        del eval_id, author
        try:
            seed, score, details, sort_key = _evaluate_candidate(result)
        except AssertionError as exc:
            message = f"Verification failed: {type(exc).__name__}: {exc}"
            return SeedBatchEvaluation(
                seeds=[None],
                scores=np.asarray([0.0], dtype=np.float64),
                valid=np.asarray([False], dtype=bool),
                sort_keys=[(float("-inf"),)],
                details=[{"Message": message}],
                errors=[message],
            )
        except Exception as exc:
            message = f"Evaluation error: {type(exc).__name__}: {exc}"
            return SeedBatchEvaluation(
                seeds=[None],
                scores=np.asarray([0.0], dtype=np.float64),
                valid=np.asarray([False], dtype=bool),
                sort_keys=[(float("-inf"),)],
                details=[{"Message": message}],
                errors=[message],
            )

        return SeedBatchEvaluation(
            seeds=[seed],
            scores=np.asarray([score], dtype=np.float64),
            valid=np.asarray([True], dtype=bool),
            sort_keys=[sort_key],
            details=[details],
            errors=[None],
        )

    def get_progress_records(
        self,
        *,
        result: Any,
        success: bool,
        score: Any,
        details: Any,
        sort_key: Any,
        eval_id: str = None,
        author: str = None,
    ) -> list[Dict[str, Any]]:
        del result, sort_key, eval_id, author
        if not success or not isinstance(details, dict):
            return []
        try:
            k = int(_metric_value(details["K"]))
            modulus = int(_metric_value(details["Modulus"]))
            set_size = int(_metric_value(details["SetSize"]))
            exponent = float(score)
        except (KeyError, TypeError, ValueError):
            return []
        if k not in ALLOWED_POWERS or not math.isfinite(exponent):
            return []
        return [
            {
                "track": f"power-difference:k={k}",
                "rank_key": [exponent],
                "value": exponent,
                "label": f"k={k} cyclic lower-bound exponent",
                "metadata": {
                    "k": k,
                    "m": modulus,
                    "set_size": set_size,
                },
            }
        ]

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        del author, agent_module
        if "def construct_power_difference_set" not in content:
            return False, "Submission must define construct_power_difference_set()."
        return True, None


def _evaluate_candidate(
    result: Any,
) -> Tuple[Dict[str, np.ndarray], float, Dict[str, Any], Tuple[float]]:
    k, modulus, residues = _parse_candidate(result)
    power_residues = _nonzero_power_residues(k, modulus)
    _verify_differences(residues, modulus, power_residues)

    log_modulus = math.log(modulus)
    cyclic_contribution = math.log(len(residues)) / (k * log_modulus)
    exponent = 1.0 - 1.0 / k + cyclic_contribution
    residue_density = len(residues) / modulus

    seed = {
        "parameters": np.asarray([k, modulus], dtype=np.int64),
        "residues": np.asarray(residues, dtype=np.int64),
    }
    details = {
        "Message": (
            f"Valid k={k} cyclic witness modulo {modulus} with "
            f"{len(residues)} residues and lower-bound exponent "
            f"{exponent:.10g}. Higher is better within the k={k} track."
        ),
        "K": k,
        "Modulus": modulus,
        "SetSize": len(residues),
        "CyclicContribution": float(cyclic_contribution),
        "ResidueDensity": float(residue_density),
        "PowerResidues": len(power_residues),
    }
    return seed, float(exponent), details, (float(cyclic_contribution),)


def _parse_candidate(result: Any) -> Tuple[int, int, list[int]]:
    if result is None:
        raise AssertionError("construct_power_difference_set returned None.")
    if not isinstance(result, str):
        raise AssertionError("Submission must return a JSON string.")

    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, ValueError) as exc:
        raise AssertionError(f"Invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise AssertionError("Top-level JSON value must be an object.")
    if payload.get("mode") != MODE:
        raise AssertionError(f"mode must be exactly {MODE!r}.")
    for field in ("k", "m", "set"):
        if field not in payload:
            raise AssertionError(f"Missing required field {field!r}.")

    k = _parse_plain_integer(payload["k"], "k")
    modulus = _parse_plain_integer(payload["m"], "m")
    if k not in ALLOWED_POWERS:
        allowed = ", ".join(str(value) for value in sorted(ALLOWED_POWERS))
        raise AssertionError(f"k must be one of {{{allowed}}}.")
    if modulus < MIN_MODULUS or modulus > MAX_MODULUS:
        raise AssertionError(
            f"m must satisfy {MIN_MODULUS} <= m <= {MAX_MODULUS}."
        )
    if not _is_square_free(modulus):
        raise AssertionError("m must be square-free.")

    raw_residues = payload["set"]
    if not isinstance(raw_residues, list):
        raise AssertionError("set must be a JSON list of integers.")
    if len(raw_residues) > MAX_SET_SIZE:
        raise AssertionError(
            f"set may contain at most {MAX_SET_SIZE} submitted integers."
        )

    normalized = {
        _parse_plain_integer(value, f"set[{index}]") % modulus
        for index, value in enumerate(raw_residues)
    }
    if not normalized:
        raise AssertionError("The normalized residue set must be nonempty.")
    residues = sorted(normalized)
    return k, modulus, residues


def _parse_plain_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise AssertionError(f"{field_name} must be an integer.")
    return int(value)


def _is_square_free(value: int) -> bool:
    divisor = 2
    while divisor * divisor <= value:
        if value % (divisor * divisor) == 0:
            return False
        divisor += 1
    return True


def _nonzero_power_residues(k: int, modulus: int) -> set[int]:
    residues = {pow(value, k, modulus) for value in range(1, modulus)}
    residues.discard(0)
    return residues


def _verify_differences(
    residues: list[int],
    modulus: int,
    power_residues: set[int],
) -> None:
    for left_index, left in enumerate(residues):
        for right_index in range(left_index + 1, len(residues)):
            right = residues[right_index]
            difference = (left - right) % modulus
            if (
                difference in power_residues
                or (-difference) % modulus in power_residues
            ):
                raise AssertionError(
                    "The normalized set contains distinct residues whose ordered "
                    f"difference is a nonzero k-th-power residue modulo m: "
                    f"{left} and {right}."
                )


def _metric_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[1]
    return value
