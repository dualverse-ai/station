"""Evaluator for the fixed-length low-peak Littlewood proxy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator, SeedBatchEvaluation


COEFFICIENT_COUNT = 89
MESH_POINTS = 1_000_000
MAX_BATCH_CANDIDATES = 64


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluate the exact finite mesh proxy documented in research_task.md."""

    def get_expected_function_name(self) -> str:
        return "construct_flat_polynomial"

    def get_task_description(self) -> str:
        return "Fixed-length low-peak Littlewood polynomial proxy"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "MeshMinimum": ".10f",
            "MeshWidth": ".10f",
            "MeritFactor": ".10f",
            "PeakIndex": "d",
            "ValleyIndex": "d",
            "SignChanges": "d",
        }

    def evaluate_submission(
        self,
        result: Any,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, Any, Dict[str, Any], Tuple[float]]:
        batch = self.evaluate_seed_batch(result, eval_id, author)
        valid_indices = [index for index, is_valid in enumerate(batch.valid) if bool(is_valid)]
        if not valid_indices:
            message = next((error for error in batch.errors if error), "All candidates were invalid.")
            return False, constants.RESEARCH_SCORE_NA, {"Message": message}, (float("-inf"),)
        winner = max(valid_indices, key=lambda index: batch.sort_keys[index])
        return (
            True,
            float(batch.scores[winner]),
            batch.details[winner],
            tuple(batch.sort_keys[winner]),
        )

    def evaluate_seed_batch(
        self,
        result: Any,
        eval_id: str = None,
        author: str = None,
    ) -> SeedBatchEvaluation:
        candidates = self._split_candidates(result)
        seeds: List[Any] = []
        scores: List[float] = []
        valid: List[bool] = []
        sort_keys: List[tuple] = []
        details: List[Dict[str, Any]] = []
        errors: List[Optional[str]] = []

        for candidate in candidates:
            try:
                coefficients = self._parse_coefficients(candidate)
                score, metrics = self._score_reference_mesh(coefficients)
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

            seeds.append(coefficients)
            scores.append(float(score))
            valid.append(True)
            sort_keys.append((-float(score),))
            details.append(self._build_details(score, metrics))
            errors.append(None)

        return SeedBatchEvaluation(
            seeds=seeds,
            scores=np.asarray(scores, dtype=np.float64),
            valid=np.asarray(valid, dtype=bool),
            sort_keys=sort_keys,
            details=details,
            errors=errors,
        )

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_flat_polynomial" not in content:
            return False, "Submission must define construct_flat_polynomial()."
        return True, None

    @staticmethod
    def _split_candidates(result: Any) -> List[Any]:
        if isinstance(result, np.ndarray):
            if result.ndim == 1:
                return [result]
            if result.ndim == 2:
                return [result[index] for index in range(result.shape[0])]
            return [result]

        if isinstance(result, (list, tuple)):
            if not result:
                return [result]
            if all(_is_real_scalar(value) for value in result):
                return [result]
            return list(result)

        return [result]

    @staticmethod
    def _parse_coefficients(candidate: Any) -> np.ndarray:
        if candidate is None:
            raise AssertionError("construct_flat_polynomial returned None as a candidate.")

        if isinstance(candidate, np.ndarray):
            if candidate.ndim != 1:
                raise AssertionError("Each candidate must be one-dimensional.")
            if np.iscomplexobj(candidate):
                raise AssertionError("Coefficients must be real.")
            raw = candidate
        elif isinstance(candidate, (list, tuple)):
            if not candidate:
                raise AssertionError("Coefficient vector is empty.")
            if not all(_is_real_scalar(value) for value in candidate):
                raise AssertionError("Every coefficient must be a real number.")
            raw = np.asarray(candidate)
        else:
            raise AssertionError("Candidate must be a numeric list or one-dimensional NumPy array.")

        if raw.size != COEFFICIENT_COUNT:
            raise AssertionError(
                f"Coefficient count must be exactly {COEFFICIENT_COUNT}; got {raw.size}."
            )
        try:
            coefficients = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise AssertionError("Coefficients must be finite real numbers.") from exc
        if not np.all(np.isfinite(coefficients)):
            raise AssertionError("All coefficients must be finite.")
        if not np.all((coefficients == -1.0) | (coefficients == 1.0)):
            raise AssertionError("Every coefficient must be exactly -1 or +1.")
        return coefficients.astype(np.int8, copy=True)

    @staticmethod
    def _score_reference_mesh(coefficients: np.ndarray) -> Tuple[float, Dict[str, float]]:
        # Keep this calculation textually aligned with the released reference
        # scorer: np.poly1d, endpoint-inclusive np.linspace, and sqrt(n + 1).
        poly_fn = np.poly1d(coefficients)
        n = len(coefficients)
        zs = np.exp(1j * np.linspace(0, 2 * np.pi, MESH_POINTS))
        modulus = np.abs(poly_fn(zs))
        score = float(np.max(modulus) / np.sqrt(n + 1))
        if not np.isfinite(score):
            raise AssertionError("Computed score is not finite.")

        autocorrelation = np.correlate(
            coefficients.astype(np.float64),
            coefficients.astype(np.float64),
            mode="full",
        )[COEFFICIENT_COUNT - 1 :]
        energy = float(2.0 * np.sum(autocorrelation[1:] ** 2))
        merit_factor = float(COEFFICIENT_COUNT**2 / energy) if energy > 0.0 else float("inf")
        if not np.isfinite(merit_factor):
            merit_factor = 0.0

        return score, {
            "mesh_minimum": float(np.min(modulus) / np.sqrt(n + 1)),
            "mesh_width": float((np.max(modulus) - np.min(modulus)) / np.sqrt(n + 1)),
            "merit_factor": merit_factor,
            "peak_index": float(np.argmax(modulus)),
            "valley_index": float(np.argmin(modulus)),
            "sign_changes": float(np.count_nonzero(coefficients[1:] != coefficients[:-1])),
        }

    @staticmethod
    def _build_details(score: float, metrics: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Message": (
                f"Valid length-{COEFFICIENT_COUNT} signed polynomial with mesh maximum "
                f"{score:.12g}. Lower scores are better; the mesh is not a continuum certificate."
            ),
            "MeshMinimum": float(metrics["mesh_minimum"]),
            "MeshWidth": float(metrics["mesh_width"]),
            "MeritFactor": float(metrics["merit_factor"]),
            "PeakIndex": int(metrics["peak_index"]),
            "ValleyIndex": int(metrics["valley_index"]),
            "SignChanges": int(metrics["sign_changes"]),
        }


def _is_real_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, np.ndarray):
        return False
    if not isinstance(value, (int, float, np.number)):
        return False
    return not np.iscomplexobj(value)
