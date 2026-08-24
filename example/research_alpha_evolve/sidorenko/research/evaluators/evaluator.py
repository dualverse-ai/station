"""
Evaluator for finite step-graphons for a fixed Sidorenko-ratio search.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator


MODE = "sidorenko-step-graphon-v1"
VERTEX_COUNT = 30
EDGE_COUNT_H = 15
VERTEX_COUNT_H = 10
MIN_MEAN_BEFORE_RESCALING = 0.01
MIN_MATRIX_RANGE = 0.25
MIN_MATRIX_STD = 0.23


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for weighted 30-step graphon candidates."""

    def __init__(self):
        super().__init__()

    def get_expected_function_name(self) -> str:
        return "construct_sidorenko"

    def get_task_description(self) -> str:
        return "Sidorenko step-graphon search"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "Score": ".10f",
            "EdgeDensity": ".8f",
            "HomDensity": ".8e",
            "MatrixStd": ".8f",
            "MatrixRange": ".8f",
            "Vertices": "d",
        }

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, Any, Dict[str, Any], Tuple[float]]:
        try:
            matrix = self._parse_submission(result)
            score, metrics = score_step_graphon(matrix)
        except AssertionError as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Verification failed: {type(exc).__name__}: {exc}"
            }, (float("-inf"),)
        except Exception as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Evaluation error: {type(exc).__name__}: {exc}"
            }, (float("-inf"),)

        reason = metrics.get("penalty_reason")
        if reason:
            message = (
                f"Parseable step-graphon received score {score:.10g}. "
                f"Penalty reason: {reason}. Higher scores are better."
            )
        else:
            message = (
                f"Valid step-graphon with score {score:.10g}. "
                f"Edge density after normalization is {metrics['edge_density']:.10g}; "
                f"weighted homomorphism density is {metrics['hom_density']:.10g}. "
                "Higher scores are better."
            )

        details = {
            "Message": message,
            "Score": float(score),
            "EdgeDensity": float(metrics["edge_density"]),
            "HomDensity": float(metrics["hom_density"]),
            "MatrixStd": float(metrics["matrix_std"]),
            "MatrixRange": float(metrics["matrix_range"]),
            "Vertices": int(metrics["vertices"]),
        }
        return True, float(score), details, (float(score),)

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_sidorenko" not in content:
            return False, "Submission must define construct_sidorenko()."
        return True, None

    def _parse_submission(self, result: Any) -> np.ndarray:
        if result is None:
            raise AssertionError("construct_sidorenko returned None.")
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
        if "matrix" not in payload:
            raise AssertionError("Missing required field 'matrix'.")

        try:
            matrix = np.asarray(payload["matrix"], dtype=np.float64)
        except Exception as exc:
            raise AssertionError("matrix must be a numeric two-dimensional list.") from exc

        if matrix.ndim != 2:
            raise AssertionError("matrix must be two-dimensional.")
        if matrix.shape[0] != matrix.shape[1]:
            raise AssertionError("matrix must be square.")
        if not np.all(np.isfinite(matrix)):
            raise AssertionError("All matrix entries must be finite.")
        return matrix


def score_step_graphon(matrix: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Score a weighted step graphon using the fixed finite ratio."""

    matrix = np.asarray(matrix, dtype=np.float64)
    vertices = int(matrix.shape[0])
    if vertices != VERTEX_COUNT:
        clipped = np.clip(matrix, 0.0, 1.0)
        return _penalty_metrics(
            clipped,
            "matrix must have exactly 30 vertices",
            vertices=vertices,
        )

    adjacency_matrix = np.clip(matrix, 0.0, 1.0)
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2.0

    mean_before_rescaling = float(np.mean(adjacency_matrix))
    if mean_before_rescaling < MIN_MEAN_BEFORE_RESCALING:
        return _penalty_metrics(
            adjacency_matrix,
            f"mean before rescaling is below {MIN_MEAN_BEFORE_RESCALING}",
            vertices=vertices,
        )

    adjacency_matrix = adjacency_matrix / (2.0 * mean_before_rescaling)
    adjacency_matrix = np.clip(adjacency_matrix, 0.0, 1.0)

    matrix_range = float(np.max(adjacency_matrix) - np.min(adjacency_matrix))
    if matrix_range < MIN_MATRIX_RANGE:
        return _penalty_metrics(
            adjacency_matrix,
            f"matrix range after normalization is below {MIN_MATRIX_RANGE}",
            vertices=vertices,
        )

    matrix_std = float(np.std(adjacency_matrix))
    if matrix_std < MIN_MATRIX_STD:
        return _penalty_metrics(
            adjacency_matrix,
            f"matrix standard deviation after normalization is below {MIN_MATRIX_STD}",
            vertices=vertices,
        )

    weighted_twice_edges = float(np.sum(adjacency_matrix))
    if weighted_twice_edges == 0.0:
        return _penalty_metrics(
            adjacency_matrix,
            "normalized matrix has no edge mass",
            vertices=vertices,
        )

    homomorphisms = count_homomorphisms_k55_minus_c10_weighted(adjacency_matrix)
    if homomorphisms == 0.0:
        return _penalty_metrics(
            adjacency_matrix,
            "weighted homomorphism count is zero",
            vertices=vertices,
        )

    numerator = (weighted_twice_edges ** EDGE_COUNT_H) - (
        vertices ** (2 * VERTEX_COUNT_H)
    ) * homomorphisms
    denominator = (vertices ** (2 * VERTEX_COUNT_H)) * homomorphisms
    score = float(np.double(numerator / denominator))
    if not math.isfinite(score):
        raise AssertionError("Computed score is not finite.")

    metrics = _matrix_metrics(adjacency_matrix, vertices=vertices)
    metrics["hom_density"] = float(homomorphisms / (vertices ** VERTEX_COUNT_H))
    return score, metrics


def count_homomorphisms_k55_minus_c10_weighted(adjacency_matrix: np.ndarray) -> float:
    """Return the weighted labelled homomorphism count for K_{5,5} - C_10."""

    matrix = np.asarray(adjacency_matrix, dtype=np.float64)
    if matrix.shape != (VERTEX_COUNT, VERTEX_COUNT):
        raise AssertionError("Internal homomorphism counter requires a 30 x 30 matrix.")

    common_neighbours = np.einsum(
        "ad,bd,cd->abc",
        matrix,
        matrix,
        matrix,
        optimize=True,
    )
    homomorphisms = np.einsum(
        "abc,bcd,cde,dea,eab->",
        common_neighbours,
        common_neighbours,
        common_neighbours,
        common_neighbours,
        common_neighbours,
        optimize=True,
    )
    return float(homomorphisms)


def _penalty_metrics(
    matrix: np.ndarray,
    reason: str,
    *,
    vertices: int,
) -> Tuple[float, Dict[str, float]]:
    metrics = _matrix_metrics(matrix, vertices=vertices)
    metrics["hom_density"] = 0.0
    metrics["penalty_reason"] = reason
    return -1.0, metrics


def _matrix_metrics(matrix: np.ndarray, *, vertices: int) -> Dict[str, float]:
    if matrix.size == 0:
        edge_density = 0.0
        matrix_std = 0.0
        matrix_range = 0.0
    else:
        edge_density = float(np.mean(matrix))
        matrix_std = float(np.std(matrix))
        matrix_range = float(np.max(matrix) - np.min(matrix))

    return {
        "edge_density": edge_density,
        "matrix_std": matrix_std,
        "matrix_range": matrix_range,
        "vertices": float(vertices),
        "hom_density": 0.0,
    }
