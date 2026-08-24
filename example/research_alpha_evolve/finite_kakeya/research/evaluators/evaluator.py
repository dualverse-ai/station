"""
Evaluator for finite-field Kakeya set constructions in F_p^d, d in {3, 4, 5}.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator


DIMENSION_PRIMES: Dict[int, Tuple[int, ...]] = {
    3: (3, 5, 7, 11, 13, 19, 23, 29, 31, 37, 41, 43, 47, 53),
    4: (3, 5, 7, 11, 13, 17, 19),
    5: (3, 5, 7, 11),
}
ALLOWED_DIMENSIONS = tuple(sorted(DIMENSION_PRIMES))

Point = Tuple[int, ...]
Direction = Tuple[int, ...]


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for exact finite-field Kakeya constructions in F_p^d."""

    def __init__(self):
        super().__init__()

    def get_expected_function_name(self) -> str:
        return "construct_kakeya"

    def get_task_description(self) -> str:
        return "Finite-field Kakeya set optimization"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "Dimension": "d",
            "CalibrationRelativeSize": ".8f",
            "MeanRatio": ".8f",
            "WorstRatio": ".8f",
            "BestRatio": ".8f",
            "WorstP": "d",
            "BestP": "d",
            "ValidCount": "d",
        }

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, Any, Dict[str, Any], Tuple[float, int]]:
        try:
            dimension, constructions = self._parse_submission(result)
            score, metrics = self._evaluate_constructions(dimension, constructions)
        except AssertionError as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Verification failed: {type(exc).__name__}: {exc}"
            }, (float("-inf"), 0)
        except Exception as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Evaluation error: {type(exc).__name__}: {exc}"
            }, (float("-inf"), 0)

        ratio_summary = ", ".join(
            f"p={p}: {size}/{reference}={ratio:.6g}"
            for p, size, reference, ratio in metrics["ratios_by_p"]
        )
        message = (
            f"Valid finite-field Kakeya construction in dimension d={dimension} "
            f"with mean relative size {score:.10g}. Lower scores are better; "
            f"worst tested ratio is {metrics['worst_ratio']:.10g}. "
            f"Per-prime values: {ratio_summary}."
        )

        details = {
            "Message": message,
            "Dimension": int(dimension),
            "CalibrationRelativeSize": float(score),
            "MeanRatio": float(metrics["mean_ratio"]),
            "WorstRatio": float(metrics["worst_ratio"]),
            "BestRatio": float(metrics["best_ratio"]),
            "WorstP": int(metrics["worst_p"]),
            "BestP": int(metrics["best_p"]),
            "ValidCount": len(DIMENSION_PRIMES[dimension]),
        }
        return True, float(score), details, self._build_sort_key(score, dimension)

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
    ) -> List[Dict[str, Any]]:
        if not success:
            return []
        try:
            dimension = int(metric_value((details or {}).get("Dimension")))
            value = float(score)
        except (TypeError, ValueError, AttributeError):
            return []
        return [
            {
                "track": f"dimension:d{dimension}",
                "rank_key": [-value],
                "value": value,
                "label": f"d={dimension} mean relative size",
                "metadata": {"dimension": dimension},
            }
        ]

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def construct_kakeya" not in content:
            return False, "Submission must define construct_kakeya()."
        return True, None

    def _build_sort_key(self, score: float, dimension: int) -> Tuple[float, int]:
        # Research Center ranks larger sort keys first; this task is lower-is-better.
        # The dimension tie-breaker makes higher-dimensional work visible when the
        # normalized score is exactly tied.
        return (-float(score), int(dimension))

    def _parse_submission(self, result: Any) -> Tuple[int, Dict[int, Tuple[Set[Point], Dict[Direction, Point]]]]:
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

        dimension = self._parse_dimension(payload)
        prime_payload = self._parse_prime_payload_container(payload, dimension)
        constructions: Dict[int, Tuple[Set[Point], Dict[Direction, Point]]] = {}
        for p in DIMENSION_PRIMES[dimension]:
            value = self._lookup_prime_payload(prime_payload, p)
            constructions[p] = self._parse_prime_construction(value, p, dimension)
        return dimension, constructions

    def _parse_dimension(self, payload: Dict[Any, Any]) -> int:
        if "dimension" not in payload and "d" not in payload:
            # Backward compatibility for the original d=3 prime-keyed payload.
            return 3
        raw_dimension = payload.get("dimension", payload.get("d"))
        if isinstance(raw_dimension, bool) or not isinstance(raw_dimension, int):
            raise AssertionError("Payload field 'dimension' must be one of 3, 4, or 5.")
        dimension = int(raw_dimension)
        if dimension not in DIMENSION_PRIMES:
            raise AssertionError("Payload field 'dimension' must be one of 3, 4, or 5.")
        return dimension

    def _parse_prime_payload_container(self, payload: Dict[Any, Any], dimension: int) -> Dict[Any, Any]:
        if "constructions" in payload:
            constructions = payload["constructions"]
            if not isinstance(constructions, dict):
                raise AssertionError("Payload field 'constructions' must be a dictionary.")
            return constructions
        if "primes" in payload:
            constructions = payload["primes"]
            if not isinstance(constructions, dict):
                raise AssertionError("Payload field 'primes' must be a dictionary.")
            return constructions
        if dimension == 3:
            return payload
        raise AssertionError("Payload must include a 'constructions' dictionary keyed by tested primes.")

    def _lookup_prime_payload(self, payload: Dict[Any, Any], p: int) -> Any:
        if p in payload:
            return payload[p]
        key = str(p)
        if key in payload:
            return payload[key]
        raise AssertionError(f"Missing required entry for p={p}.")

    def _parse_prime_construction(
        self,
        value: Any,
        p: int,
        dimension: int,
    ) -> Tuple[Set[Point], Dict[Direction, Point]]:
        if not isinstance(value, dict):
            raise AssertionError(f"Entry for p={p} must be a dictionary.")
        if "points" not in value:
            raise AssertionError(f"Entry for p={p} is missing 'points'.")
        if "witnesses" not in value:
            raise AssertionError(f"Entry for p={p} is missing 'witnesses'.")

        points = self._parse_point_set(value["points"], p, dimension)
        witnesses = self._parse_witness_map(value["witnesses"], p, dimension)
        return points, witnesses

    def _parse_point_set(self, value: Any, p: int, dimension: int) -> Set[Point]:
        if not isinstance(value, (list, tuple)):
            raise AssertionError(f"Point set for p={p} must be a list of coordinate tuples.")
        if len(value) > p ** dimension:
            raise AssertionError(f"Point list for p={p} has more entries than F_p^{dimension}.")

        points: Set[Point] = set()
        for index, raw_point in enumerate(value):
            point = self._parse_point(raw_point, p, dimension, f"points[{index}] for p={p}")
            points.add(point)
        if not points:
            raise AssertionError(f"Point set for p={p} is empty.")
        return points

    def _parse_witness_map(self, value: Any, p: int, dimension: int) -> Dict[Direction, Point]:
        if not isinstance(value, dict):
            raise AssertionError(f"Witnesses for p={p} must be a dictionary.")

        expected = set(canonical_directions(p, dimension))
        witnesses: Dict[Direction, Point] = {}
        for raw_direction, raw_point in value.items():
            direction = self._parse_direction(raw_direction, p, dimension)
            if direction not in expected:
                raise AssertionError(f"Non-canonical direction {direction} for p={p}.")
            if direction in witnesses:
                raise AssertionError(f"Duplicate witness direction {direction} for p={p}.")
            witnesses[direction] = self._parse_point(
                raw_point,
                p,
                dimension,
                f"witness for direction {direction} and p={p}",
            )

        if len(witnesses) != len(expected):
            missing = sorted(expected - set(witnesses))[:5]
            raise AssertionError(
                f"Witness map for p={p} has {len(witnesses)} entries; "
                f"expected {len(expected)}. First missing directions: {missing}."
            )

        return witnesses

    def _parse_direction(self, value: Any, p: int, dimension: int) -> Direction:
        direction = self._parse_point(value, p, dimension, f"direction key for p={p}")
        for index, coordinate in enumerate(direction):
            if coordinate == 1 and all(previous == 0 for previous in direction[:index]):
                return direction
            if coordinate != 0 and all(previous == 0 for previous in direction[:index]):
                break
        raise AssertionError(
            f"Direction {direction} for p={p} is not one of the canonical representatives."
        )

    def _parse_point(self, value: Any, p: int, dimension: int, label: str) -> Point:
        if not isinstance(value, (list, tuple)):
            raise AssertionError(f"{label} must be a length-{dimension} tuple or list.")
        if len(value) != dimension:
            raise AssertionError(f"{label} must have exactly {dimension} coordinates.")

        coordinates: List[int] = []
        for coordinate in value:
            if isinstance(coordinate, bool) or not isinstance(coordinate, int):
                raise AssertionError(f"{label} must contain integer coordinates.")
            if coordinate < 0 or coordinate >= p:
                raise AssertionError(f"{label} coordinate {coordinate} is outside 0..{p - 1}.")
            coordinates.append(int(coordinate))
        return tuple(coordinates)

    def _evaluate_constructions(
        self,
        dimension: int,
        constructions: Dict[int, Tuple[Set[Point], Dict[Direction, Point]]],
    ) -> Tuple[float, Dict[str, Any]]:
        ratios: List[float] = []
        ratios_by_p: List[Tuple[int, int, int, float]] = []

        for p in DIMENSION_PRIMES[dimension]:
            point_set, witnesses = constructions[p]
            self._verify_witness_lines(point_set, witnesses, p, dimension)

            size = len(point_set)
            reference_size = normalization_size(p, dimension)
            ratio = size / reference_size
            ratios.append(float(ratio))
            ratios_by_p.append((p, size, reference_size, float(ratio)))

        score = sum(ratios) / len(ratios)
        worst_index = max(range(len(ratios)), key=lambda index: ratios[index])
        best_index = min(range(len(ratios)), key=lambda index: ratios[index])
        primes = DIMENSION_PRIMES[dimension]

        return score, {
            "mean_ratio": float(score),
            "worst_ratio": float(ratios[worst_index]),
            "best_ratio": float(ratios[best_index]),
            "worst_p": float(primes[worst_index]),
            "best_p": float(primes[best_index]),
            "ratios_by_p": ratios_by_p,
        }

    def _verify_witness_lines(
        self,
        point_set: Set[Point],
        witnesses: Dict[Direction, Point],
        p: int,
        dimension: int,
    ) -> None:
        for direction in canonical_directions(p, dimension):
            witness = witnesses[direction]
            for point in line_points(witness, direction, p):
                if point not in point_set:
                    raise AssertionError(
                        f"Witness line for p={p}, d={dimension}, direction={direction} "
                        f"is missing point {point}."
                    )


def canonical_directions(p: int, dimension: int) -> Iterable[Direction]:
    for first_nonzero in range(dimension):
        prefix = [0] * first_nonzero + [1]
        free_count = dimension - first_nonzero - 1
        for suffix in product_range(p, free_count):
            yield tuple(prefix + list(suffix))


def product_range(p: int, length: int) -> Iterable[Tuple[int, ...]]:
    if length == 0:
        yield ()
        return
    values = [0] * length
    while True:
        yield tuple(values)
        index = length - 1
        while index >= 0:
            values[index] += 1
            if values[index] < p:
                break
            values[index] = 0
            index -= 1
        if index < 0:
            return


def line_points(witness: Point, direction: Direction, p: int) -> Iterable[Point]:
    for t in range(p):
        yield tuple((x + t * v) % p for x, v in zip(witness, direction))


def normalization_size(p: int, dimension: int) -> int:
    square_count = (p + 1) // 2
    return (p - 1) * (square_count ** (dimension - 1)) + (p ** (dimension - 1))


def metric_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        if len(value) >= 2:
            return value[1]
        return value[0]
    return value
