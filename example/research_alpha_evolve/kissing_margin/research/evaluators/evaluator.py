# station_data/rooms/research/evaluators/evaluator.py
"""
Evaluator for the fixed-count kissing-number direction-margin task.
"""

from decimal import Decimal
from fractions import Fraction
import os
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np

from station.eval_research.base_evaluator import ResearchTaskEvaluator
from station import constants
from station import file_io_utils


TARGET_DIMENSION = 11
TARGET_COUNT = 594
PAIR_BLOCK_SIZE = 256
MIN_STABLE_ROW_SCALE = float(np.sqrt(np.finfo(np.float64).tiny))
EXACT_FLOAT_GATE_TOL = 1e-8
EXACT_MAX_OVERLAP_GATE_TOL = 1e-8


class Task1Evaluator(ResearchTaskEvaluator):
    """
    Evaluator for a fixed-count kissing configuration in TARGET_DIMENSION.
    """

    def __init__(self):
        super().__init__()
        self.dim = TARGET_DIMENSION
        self.target_count = TARGET_COUNT

    def get_secondary_metrics_format(self):
        return {
            "N": "d",
            "Dimension": "d",
            "Min Pair Distance": ".12f",
            "Max Overlap": ".12f",
            "Certified": None,
        }

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None
    ) -> Tuple[bool, Any, Any, Tuple[float]]:
        """
        Score submitted direction vectors by total overlap loss after normalization.

        Lower primary score is better, so the sort key negates the loss for the
        Station's higher-is-better ranking logic.
        """
        try:
            raw_array = np.asarray(result, dtype=object)
            vectors = np.asarray(result, dtype=np.float64)
        except Exception as exc:  # pragma: no cover - best effort conversion
            return False, "n.a.", f"Could not parse submission output as numeric vectors: {exc}", (float("-inf"),)

        try:
            self._validate_vectors(vectors)
            score, metrics = self._evaluate_overlap_loss(vectors)
        except AssertionError as exc:
            return False, "n.a.", f"Verification failed: {exc}", (float("-inf"),)
        except Exception as exc:
            return False, "n.a.", f"Evaluation error: {exc}", (float("-inf"),)

        certified = "float64 numerical"
        looks_exact = self._looks_exact(raw_array)
        should_try_exact = (
            looks_exact
            and (
                score <= 0.0
                or (
                    score <= EXACT_FLOAT_GATE_TOL
                    and metrics.get("max_overlap", float("inf")) <= EXACT_MAX_OVERLAP_GATE_TOL
                )
            )
        )
        if should_try_exact:
            exact_ok, exact_message = self._verify_exact_no_overlap(raw_array)
            certified = exact_message
            if exact_ok:
                score = 0.0
                metrics["max_overlap"] = 0.0
                if metrics.get("min_pair_distance", 0.0) < 2.0:
                    metrics["min_pair_distance"] = 2.0
            elif score <= 0.0:
                storage_paths = None
                if eval_id:
                    storage_paths = self._save_scored_config(
                        vectors,
                        raw_array,
                        score,
                        metrics,
                        eval_id,
                        author,
                        certified=certified,
                        status="uncertified",
                    )
                message = self._append_storage_paths(
                    "Numerical loss was zero, but exact verification did not certify the configuration.",
                    storage_paths,
                )
                details = self._build_details(
                    metrics,
                    certified,
                    message,
                )
                return False, "n.a.", details, (float("-inf"),)

        storage_paths = None
        if eval_id:
            status = "accepted" if score <= 0.0 else "overlap"
            storage_paths = self._save_scored_config(
                vectors,
                raw_array,
                score,
                metrics,
                eval_id,
                author,
                certified=certified,
                status=status,
            )

        if score <= 0.0:
            message = (
                f"Accepted configuration: {self.target_count} direction vectors in dimension {self.dim}; "
                "the normalized centers have zero detected overlap loss."
            )
        else:
            message = (
                f"Overlap remains after direction normalization: total loss {score:.12g}; "
                f"largest pair overlap {metrics['max_overlap']:.12g}."
            )
        message = self._append_storage_paths(message, storage_paths)

        details = self._build_details(metrics, certified, message)
        return True, float(score), details, self._build_sort_key(score)

    def get_expected_function_name(self) -> str:
        return "find_kissing_configuration"

    def get_task_description(self) -> str:
        return (
            f"Fixed-count kissing direction-margin loss in dimension {self.dim} "
            f"with N={self.target_count}"
        )

    def _build_sort_key(self, score: float) -> Tuple[float]:
        # Research Center ranks larger sort keys first; the overlap loss is lower-is-better.
        return (-float(score),)

    def _validate_vectors(self, vectors: np.ndarray) -> None:
        if not isinstance(vectors, np.ndarray):
            raise AssertionError("Submission did not return a numpy-compatible array.")

        if vectors.ndim != 2:
            raise AssertionError(
                f"Expected 2D array with shape ({self.target_count}, {self.dim}); got ndim={vectors.ndim}."
            )

        num_vectors, dim = vectors.shape
        if num_vectors != self.target_count:
            raise AssertionError(
                f"Expected exactly {self.target_count} vectors, but got {num_vectors}."
            )

        if dim != self.dim:
            raise AssertionError(f"Expected dimension {self.dim}, but got {dim}.")

        if np.any(np.isnan(vectors)):
            raise AssertionError("Configuration contains NaN values.")

        if np.any(np.isinf(vectors)):
            raise AssertionError("Configuration contains infinite values.")

        row_scales = np.max(np.abs(vectors), axis=1)
        if np.any(row_scales <= 0.0):
            raise AssertionError("Every submitted vector must be non-zero.")

        smallest_scale = float(np.min(row_scales))
        if smallest_scale < MIN_STABLE_ROW_SCALE:
            raise AssertionError(
                "A submitted vector is too close to zero for stable float64 direction "
                f"normalization (minimum row scale {smallest_scale:.3e}). Rescale direction vectors before submitting."
            )

        largest_scale = float(np.max(row_scales))
        max_stable_scale = self._max_stable_row_scale()
        if largest_scale > max_stable_scale:
            raise AssertionError(
                "A submitted vector is too large for stable float64 direction "
                f"normalization (maximum row scale {largest_scale:.3e}). Rescale direction vectors before submitting."
            )

        directions = self._normalize_directions(vectors)
        direction_norms = np.linalg.norm(directions, axis=1)
        if not np.allclose(direction_norms, 1.0, rtol=1e-12, atol=1e-12):
            raise AssertionError("Direction normalization failed to produce unit vectors.")

    def _evaluate_overlap_loss(self, vectors: np.ndarray) -> Tuple[float, Dict[str, float]]:
        directions = self._normalize_directions(vectors)

        n = directions.shape[0]
        total_loss = 0.0
        min_pair_distance = float("inf")
        max_overlap = 0.0

        for start in range(0, n, PAIR_BLOCK_SIZE):
            end = min(start + PAIR_BLOCK_SIZE, n)
            dots = directions[start:end] @ directions.T
            dots = np.clip(dots, -1.0, 1.0)
            distances = np.sqrt(np.maximum(0.0, 8.0 * (1.0 - dots)))

            for local_i, global_i in enumerate(range(start, end)):
                if global_i + 1 >= n:
                    continue
                row_distances = distances[local_i, global_i + 1:]
                if row_distances.size == 0:
                    continue
                overlaps = np.maximum(0.0, 2.0 - row_distances)
                total_loss += float(np.sum(overlaps))
                row_min = float(np.min(row_distances))
                if row_min < min_pair_distance:
                    min_pair_distance = row_min
                row_max_overlap = float(np.max(overlaps))
                if row_max_overlap > max_overlap:
                    max_overlap = row_max_overlap

        return float(total_loss), {
            "n": float(n),
            "dimension": float(self.dim),
            "min_pair_distance": float(min_pair_distance),
            "max_overlap": float(max_overlap),
        }

    def _normalize_directions(self, vectors: np.ndarray) -> np.ndarray:
        row_scales = np.max(np.abs(vectors), axis=1)
        if np.any(row_scales <= 0.0):
            raise AssertionError("Every submitted vector must be non-zero.")

        scaled = vectors / row_scales[:, None]
        norms = np.linalg.norm(scaled, axis=1)
        if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
            raise AssertionError("Direction normalization produced invalid norms.")

        directions = scaled / norms[:, None]
        if not np.all(np.isfinite(directions)):
            raise AssertionError("Direction normalization produced non-finite values.")
        return directions

    def _max_stable_row_scale(self) -> float:
        return float(np.sqrt(np.finfo(np.float64).max / max(1, self.dim)))

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module
    ) -> Tuple[bool, Optional[str]]:
        if "def find_kissing_configuration" not in content:
            return False, "Submission must define find_kissing_configuration()."
        return True, None

    def _build_details(
        self,
        metrics: Dict[str, float],
        certified: str,
        message: str,
    ) -> Dict[str, Any]:
        return {
            "N": int(metrics["n"]),
            "Dimension": int(metrics["dimension"]),
            "Min Pair Distance": float(metrics["min_pair_distance"]),
            "Max Overlap": float(metrics["max_overlap"]),
            "Certified": certified,
            "Message": message,
        }

    def _looks_exact(self, raw_array: np.ndarray) -> bool:
        if raw_array.ndim != 2:
            return False

        return self._is_rational_exact(raw_array) or self._has_sympy_values(raw_array)

    def _is_rational_exact(self, raw_array: np.ndarray) -> bool:
        for value in raw_array.flat:
            if isinstance(value, (bool, np.bool_)):
                return False
            if isinstance(value, (int, np.integer, Fraction, Decimal)):
                continue
            return False
        return True

    def _has_sympy_values(self, raw_array: np.ndarray) -> bool:
        saw_sympy = False
        for value in raw_array.flat:
            if isinstance(value, (bool, np.bool_, float, np.floating, complex, np.complexfloating)):
                return False
            if isinstance(value, (int, np.integer, Fraction, Decimal)):
                continue
            if getattr(value, "is_Float", False):
                return False
            if type(value).__module__.startswith("sympy"):
                saw_sympy = True
                continue
            return False
        return saw_sympy

    def _verify_exact_no_overlap(self, raw_array: np.ndarray) -> Tuple[bool, str]:
        if self._is_rational_exact(raw_array):
            return self._verify_rational_no_overlap(raw_array)
        return self._verify_sympy_no_overlap(raw_array)

    def _verify_rational_no_overlap(self, raw_array: np.ndarray) -> Tuple[bool, str]:
        rows = []
        for row in raw_array.tolist():
            exact_row = []
            for value in row:
                if isinstance(value, (int, np.integer)):
                    exact_row.append(int(value))
                else:
                    exact_row.append(Fraction(value))
            rows.append(exact_row)

        norms = [sum(coord * coord for coord in row) for row in rows]
        for idx, norm in enumerate(norms):
            if norm <= 0:
                return False, f"exact rational verifier failed: vector {idx} is zero"

        for i in range(len(rows)):
            row_i = rows[i]
            norm_i = norms[i]
            for j in range(i + 1, len(rows)):
                dot = sum(a * b for a, b in zip(row_i, rows[j]))
                if dot <= 0:
                    continue
                if 4 * dot * dot > norm_i * norms[j]:
                    return (
                        False,
                        f"exact rational verifier failed: pair ({i}, {j}) has positive overlap",
                    )

        return True, "exact rational certified"

    def _verify_sympy_no_overlap(self, raw_array: np.ndarray) -> Tuple[bool, str]:
        try:
            import sympy as sp
        except Exception as exc:  # pragma: no cover - optional dependency
            return False, f"exact verifier unavailable: could not import sympy ({exc})"

        rows = [[sp.sympify(value) for value in row] for row in raw_array.tolist()]
        norms = [sp.simplify(sum(coord * coord for coord in row)) for row in rows]

        for idx, norm in enumerate(norms):
            positive = self._sympy_is_positive(norm, sp)
            if positive is not True:
                return False, f"symbolic verifier failed: vector {idx} is zero or not provably real-positive"

        for i in range(len(rows)):
            row_i = rows[i]
            norm_i = norms[i]
            for j in range(i + 1, len(rows)):
                dot = sp.simplify(sum(a * b for a, b in zip(row_i, rows[j])))
                nonpositive = self._sympy_is_nonpositive(dot, sp)
                if nonpositive is True:
                    continue

                margin_expr = sp.simplify(norm_i * norms[j] - 4 * dot * dot)
                nonnegative = self._sympy_is_nonnegative(margin_expr, sp)
                if nonnegative is True:
                    continue
                if nonpositive is False and nonnegative is False:
                    return False, f"symbolic verifier failed: pair ({i}, {j}) has positive overlap"
                return False, f"symbolic verifier inconclusive: pair ({i}, {j}) could not be certified"

        return True, "symbolic exact certified"

    def _sympy_is_positive(self, expr, sp) -> Optional[bool]:
        expr = sp.simplify(expr)
        if expr.is_positive is not None:
            return bool(expr.is_positive)
        answer = sp.ask(sp.Q.positive(expr))
        if answer is not None:
            return bool(answer)
        return None

    def _sympy_is_nonpositive(self, expr, sp) -> Optional[bool]:
        expr = sp.simplify(expr)
        if expr.is_nonpositive is not None:
            return bool(expr.is_nonpositive)
        answer = sp.ask(sp.Q.nonpositive(expr))
        if answer is not None:
            return bool(answer)
        return None

    def _sympy_is_nonnegative(self, expr, sp) -> Optional[bool]:
        expr = sp.simplify(expr)
        if expr.is_nonnegative is not None:
            return bool(expr.is_nonnegative)
        answer = sp.ask(sp.Q.nonnegative(expr))
        if answer is not None:
            return bool(answer)
        return None

    def _save_scored_config(
        self,
        vectors: np.ndarray,
        raw_array: Optional[np.ndarray],
        score: float,
        metrics: Dict[str, float],
        eval_id: str = None,
        author: str = None,
        certified: str = "",
        status: str = "scored",
    ) -> Optional[Dict[str, str]]:
        saved_paths: Dict[str, str] = {}
        try:
            research_room_path = os.path.join(
                constants.BASE_STATION_DATA_PATH,
                constants.ROOMS_DIR_NAME,
                constants.SHORT_ROOM_NAME_RESEARCH
            )
            configs_dir = os.path.join(
                research_room_path,
                constants.RESEARCH_INTERNAL_DIR,
                "kissing_margin_configs"
            )

            file_io_utils.ensure_dir_exists(configs_dir)

            score_token = f"{score:.12g}".replace("-", "m").replace(".", "p")
            if author and eval_id:
                author_clean = self._safe_path_token(str(author))
                safe_eval_id = self._safe_path_token(str(eval_id))
                filename = f"{author_clean}_loss_{score_token}_{safe_eval_id}.npz"
            elif eval_id:
                safe_eval_id = self._safe_path_token(str(eval_id))
                filename = f"unknown_loss_{score_token}_{safe_eval_id}.npz"
            else:
                filename = f"unknown_loss_{score_token}_legacy.npz"

            filepath = os.path.join(configs_dir, filename)
            save_data = dict(
                vectors=vectors,
                score=score,
                min_pair_distance=metrics["min_pair_distance"],
                max_overlap=metrics["max_overlap"],
                target_count=self.target_count,
                dimension=self.dim,
                eval_id=str(eval_id or ""),
                author=str(author or ""),
                certified=certified,
                status=status,
            )
            self._add_exact_payload(save_data, raw_array)
            np.savez(filepath, **save_data)
            print(f"Task1Evaluator: Saved scored configuration to {filepath}")

        except Exception as exc:  # pragma: no cover - best effort persistence
            print(f"Task1Evaluator: Failed to save configuration: {exc}")

        try:
            shared_paths = self._save_submission_artifacts(
                vectors,
                raw_array,
                score,
                metrics,
                eval_id,
                author,
                certified,
                status,
            )
            saved_paths.update(shared_paths)
        except Exception as exc:  # pragma: no cover - best effort persistence
            print(f"Task1Evaluator: Failed to save submission artifacts: {exc}")

        return saved_paths or None

    def _save_submission_artifacts(
        self,
        vectors: np.ndarray,
        raw_array: Optional[np.ndarray],
        score: float,
        metrics: Dict[str, float],
        eval_id: str,
        author: str = None,
        certified: str = "",
        status: str = "scored",
    ) -> Dict[str, str]:
        research_room_path = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            constants.SHORT_ROOM_NAME_RESEARCH
        )

        safe_eval_id = self._safe_path_token(str(eval_id))
        filename = f"eval_{safe_eval_id}.npz"
        lineage = self._extract_lineage_name(author)

        shared_relative_dir = os.path.join(
            constants.RESEARCH_STORAGE_SHARED_DIR,
            "submissions",
        )
        shared_artifact_dir = os.path.join(
            research_room_path,
            constants.RESEARCH_STORAGE_DIR,
            shared_relative_dir,
        )
        shared_display_path = (
            f"storage/{shared_relative_dir.replace(os.sep, '/')}/{filename}"
        )

        lineage_physical_prefix = constants.RESEARCH_STORAGE_LINEAGES_DIR
        lineage_display_prefix = lineage
        if lineage == constants.RESEARCH_STORAGE_SYSTEM_DIR:
            lineage_physical_prefix = constants.RESEARCH_STORAGE_SYSTEM_DIR
            lineage_display_prefix = constants.RESEARCH_STORAGE_SYSTEM_DIR

        lineage_relative_dir = os.path.join(
            lineage_physical_prefix,
            lineage,
            "submissions",
        )
        if lineage == constants.RESEARCH_STORAGE_SYSTEM_DIR:
            lineage_relative_dir = os.path.join(
                constants.RESEARCH_STORAGE_SYSTEM_DIR,
                "submissions",
            )
        lineage_artifact_dir = os.path.join(
            research_room_path,
            constants.RESEARCH_STORAGE_DIR,
            lineage_relative_dir,
        )
        lineage_display_path = (
            f"storage/{lineage_display_prefix}/submissions/{filename}"
        )

        self._write_submission_npz(
            os.path.join(shared_artifact_dir, filename),
            vectors,
            raw_array,
            score,
            metrics,
            eval_id,
            author,
            certified,
            status,
        )
        self._write_submission_npz(
            os.path.join(lineage_artifact_dir, filename),
            vectors,
            raw_array,
            score,
            metrics,
            eval_id,
            author,
            certified,
            status,
        )

        return {
            "shared": shared_display_path,
            "lineage": lineage_display_path,
        }

    def _write_submission_npz(
        self,
        path: str,
        vectors: np.ndarray,
        raw_array: Optional[np.ndarray],
        score: float,
        metrics: Dict[str, float],
        eval_id: str,
        author: str,
        certified: str,
        status: str,
    ) -> None:
        file_io_utils.ensure_dir_exists(os.path.dirname(path))
        sphere_centers = 2.0 * self._normalize_directions(vectors)
        save_data = dict(
            vectors=vectors,
            sphere_centers=sphere_centers,
            score=score,
            min_pair_distance=metrics["min_pair_distance"],
            max_overlap=metrics["max_overlap"],
            target_count=self.target_count,
            dimension=self.dim,
            eval_id=str(eval_id),
            author=str(author or ""),
            certified=str(certified or ""),
            status=str(status or "scored"),
        )
        self._add_exact_payload(save_data, raw_array)
        np.savez(path, **save_data)

    def _add_exact_payload(
        self,
        save_data: Dict[str, Any],
        raw_array: Optional[np.ndarray],
    ) -> None:
        if raw_array is None or not self._looks_exact(raw_array):
            return
        save_data["exact_vectors_repr"] = np.asarray(raw_array, dtype=str)
        save_data["exact_coordinate_format"] = "python_str"

    def _append_storage_paths(self, message: str, storage_paths: Optional[Dict[str, str]]) -> str:
        if not storage_paths:
            return message

        shared_path = storage_paths.get("shared")
        lineage_path = storage_paths.get("lineage")
        if not shared_path and not lineage_path:
            return message

        lines = [f"{message}", "", "Stored submission artifacts:"]
        if shared_path:
            lines.append(f"- Shared data: `{shared_path}`")
        if lineage_path:
            lines.append(f"- Lineage data: `{lineage_path}`")
        return "\n".join(lines)

    @staticmethod
    def _safe_path_token(value: str) -> str:
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        return token or "unknown"

    @staticmethod
    def _extract_lineage_name(author: Optional[str]) -> str:
        if not author:
            return "unknown"
        lineage = str(author).strip().split(" ")[0].lower()
        sanitized = "".join(c for c in lineage if c.isalnum() or c in {"_", "-"})
        return sanitized or "unknown"
