"""Exact evaluator for a constant-Jacobian polynomial collision."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from station.eval_research.base_evaluator import ResearchTaskEvaluator


Exponent = Tuple[int, int, int]
Polynomial = Dict[Exponent, Fraction]
Point = Tuple[Fraction, Fraction, Fraction]

ZERO_EXPONENT: Exponent = (0, 0, 0)
MAX_JSON_CHARS = 250_000
MAX_TERMS_PER_POLYNOMIAL = 128
MAX_TOTAL_TERMS = 256
MAX_TOTAL_DEGREE = 12
MAX_SCALAR_BITS = 512
MAX_SCALAR_CHARS = 512
MIN_WITNESS_POINTS = 2
MAX_WITNESS_POINTS = 8


class JacobianEvaluator(ResearchTaskEvaluator):
    """Verify a rational polynomial map and an exact noninjectivity witness."""

    def get_expected_function_name(self) -> str:
        return "construct_map"

    def get_task_description(self) -> str:
        return "Construct a constant-Jacobian polynomial map with an exact rational collision"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "JacobianConstant": "d",
            "Collision": "d",
            "JacobianTerms": "d",
            "MaxDegree": "d",
            "TotalTerms": "d",
            "WitnessPoints": "d",
        }

    def evaluate_submission(
        self,
        result: Any,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        try:
            score, details = self._evaluate(result)
        except Exception as exc:
            score = 0.0
            details = _empty_details(
                f"Invalid submission: {type(exc).__name__}: {exc}"
            )
        return True, score, details

    def _evaluate(self, result: Any) -> Tuple[float, Dict[str, Any]]:
        if result is None:
            return 0.0, _empty_details("construct_map returned None.")
        if not isinstance(result, str):
            return 0.0, _empty_details(
                "construct_map must return a JSON string."
            )
        if len(result) > MAX_JSON_CHARS:
            return 0.0, _empty_details(
                f"Returned JSON exceeds the {MAX_JSON_CHARS}-character limit."
            )

        try:
            payload = json.loads(result)
        except json.JSONDecodeError as exc:
            return 0.0, _empty_details(f"Invalid JSON: {exc}")

        if not isinstance(payload, dict):
            return 0.0, _empty_details("Top-level JSON value must be an object.")

        polynomials = _parse_polynomials(payload.get("polynomials"))
        points = _parse_points(payload.get("points"))

        max_degree = max((_polynomial_degree(poly) for poly in polynomials), default=0)
        total_terms = sum(len(poly) for poly in polynomials)
        if total_terms > MAX_TOTAL_TERMS:
            raise ValueError(
                f"Polynomials contain {total_terms} combined nonzero terms; "
                f"the limit is {MAX_TOTAL_TERMS}."
            )

        determinant = _jacobian_determinant(polynomials)
        jacobian_constant = (
            len(determinant) == 1
            and ZERO_EXPONENT in determinant
            and determinant[ZERO_EXPONENT] != 0
        )

        distinct_points = len(set(points)) == len(points)
        images = [
            tuple(_evaluate_polynomial(poly, point) for poly in polynomials)
            for point in points
        ]
        collision = distinct_points and all(image == images[0] for image in images[1:])

        solved = jacobian_constant and collision
        score = 1.0 if solved else 0.0

        if solved:
            determinant_text = _format_fraction(determinant[ZERO_EXPONENT])
            message = (
                "Verified: the Jacobian determinant is the nonzero constant "
                f"{determinant_text}, and all submitted points share one image."
            )
        else:
            failures = []
            if not jacobian_constant:
                failures.append("Jacobian determinant is not a nonzero constant")
            if not collision:
                if not distinct_points:
                    failures.append("witness points are not pairwise distinct")
                else:
                    failures.append("witness points do not share one image")
            message = "Score 0: " + "; ".join(failures) + "."

        details = {
            "Message": message,
            "JacobianConstant": int(jacobian_constant),
            "Collision": int(collision),
            "JacobianTerms": len(determinant),
            "MaxDegree": max_degree,
            "TotalTerms": total_terms,
            "WitnessPoints": len(points),
        }
        return score, details


def _empty_details(message: str) -> Dict[str, Any]:
    return {
        "Message": message,
        "JacobianConstant": 0,
        "Collision": 0,
        "JacobianTerms": 0,
        "MaxDegree": 0,
        "TotalTerms": 0,
        "WitnessPoints": 0,
    }


def _parse_polynomials(raw: Any) -> List[Polynomial]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("polynomials must be a list containing exactly P, Q, and R.")

    polynomials = []
    for polynomial_index, raw_terms in enumerate(raw):
        if not isinstance(raw_terms, list):
            raise ValueError(f"Polynomial {polynomial_index + 1} must be a term list.")
        if len(raw_terms) > MAX_TERMS_PER_POLYNOMIAL:
            raise ValueError(
                f"Polynomial {polynomial_index + 1} has more than "
                f"{MAX_TERMS_PER_POLYNOMIAL} submitted terms."
            )

        polynomial: Polynomial = {}
        for term_index, raw_term in enumerate(raw_terms):
            if not isinstance(raw_term, dict):
                raise ValueError(
                    f"Term {term_index + 1} of polynomial {polynomial_index + 1} "
                    "must be an object."
                )
            coefficient = _parse_fraction(raw_term.get("coefficient"))
            exponent = _parse_exponent(raw_term.get("powers"))
            polynomial[exponent] = polynomial.get(exponent, Fraction(0)) + coefficient
            if polynomial[exponent] == 0:
                del polynomial[exponent]
        polynomials.append(polynomial)
    return polynomials


def _parse_exponent(raw: Any) -> Exponent:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("powers must be a three-element list [x_power, y_power, z_power].")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise ValueError("powers must contain nonnegative JSON integers.")
    exponent = tuple(raw)
    if any(value < 0 for value in exponent):
        raise ValueError("powers must be nonnegative.")
    if sum(exponent) > MAX_TOTAL_DEGREE:
        raise ValueError(
            f"Term total degree {sum(exponent)} exceeds the limit {MAX_TOTAL_DEGREE}."
        )
    return exponent  # type: ignore[return-value]


def _parse_points(raw: Any) -> List[Point]:
    if not isinstance(raw, list):
        raise ValueError("points must be a list of rational coordinate triples.")
    if not MIN_WITNESS_POINTS <= len(raw) <= MAX_WITNESS_POINTS:
        raise ValueError(
            f"points must contain between {MIN_WITNESS_POINTS} and "
            f"{MAX_WITNESS_POINTS} witnesses."
        )

    points = []
    for index, raw_point in enumerate(raw):
        if not isinstance(raw_point, list) or len(raw_point) != 3:
            raise ValueError(f"Point {index + 1} must contain exactly three coordinates.")
        points.append(tuple(_parse_fraction(value) for value in raw_point))
    return points  # type: ignore[return-value]


def _parse_fraction(raw: Any) -> Fraction:
    if isinstance(raw, bool):
        raise ValueError("Boolean values are not rational scalars.")
    if isinstance(raw, int):
        value = Fraction(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("Rational scalar strings cannot be empty.")
        if len(text) > MAX_SCALAR_CHARS:
            raise ValueError("Rational scalar string is too long.")
        if text.count("/") > 1:
            raise ValueError("Rational scalars must use numerator/denominator form.")
        try:
            if "/" in text:
                numerator_text, denominator_text = text.split("/", 1)
                value = Fraction(int(numerator_text, 10), int(denominator_text, 10))
            else:
                value = Fraction(int(text, 10))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Invalid rational scalar {raw!r}.") from exc
    else:
        raise ValueError("Rational scalars must be JSON integers or base-10 strings.")

    if (
        abs(value.numerator).bit_length() > MAX_SCALAR_BITS
        or value.denominator.bit_length() > MAX_SCALAR_BITS
    ):
        raise ValueError(
            f"Rational numerator and denominator are limited to {MAX_SCALAR_BITS} bits."
        )
    return value


def _polynomial_degree(polynomial: Polynomial) -> int:
    return max((sum(exponent) for exponent in polynomial), default=0)


def _differentiate(polynomial: Polynomial, variable: int) -> Polynomial:
    derivative: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        power = exponent[variable]
        if power == 0:
            continue
        next_exponent = list(exponent)
        next_exponent[variable] -= 1
        derivative[tuple(next_exponent)] = coefficient * power  # type: ignore[index]
    return derivative


def _add_polynomials(*terms: Tuple[Fraction, Polynomial]) -> Polynomial:
    result: Polynomial = {}
    for scale, polynomial in terms:
        for exponent, coefficient in polynomial.items():
            result[exponent] = result.get(exponent, Fraction(0)) + scale * coefficient
            if result[exponent] == 0:
                del result[exponent]
    return result


def _multiply_polynomials(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            exponent = tuple(
                left_exponent[index] + right_exponent[index] for index in range(3)
            )
            result[exponent] = (
                result.get(exponent, Fraction(0))
                + left_coefficient * right_coefficient
            )
            if result[exponent] == 0:
                del result[exponent]
    return result


def _jacobian_determinant(polynomials: List[Polynomial]) -> Polynomial:
    jacobian = [
        [_differentiate(polynomial, variable) for variable in range(3)]
        for polynomial in polynomials
    ]
    a, b, c = jacobian[0]
    d, e, f = jacobian[1]
    g, h, i = jacobian[2]

    ei = _multiply_polynomials(e, i)
    fh = _multiply_polynomials(f, h)
    di = _multiply_polynomials(d, i)
    fg = _multiply_polynomials(f, g)
    dh = _multiply_polynomials(d, h)
    eg = _multiply_polynomials(e, g)

    first = _multiply_polynomials(a, _add_polynomials((Fraction(1), ei), (Fraction(-1), fh)))
    second = _multiply_polynomials(b, _add_polynomials((Fraction(1), di), (Fraction(-1), fg)))
    third = _multiply_polynomials(c, _add_polynomials((Fraction(1), dh), (Fraction(-1), eg)))
    return _add_polynomials(
        (Fraction(1), first),
        (Fraction(-1), second),
        (Fraction(1), third),
    )


def _evaluate_polynomial(polynomial: Polynomial, point: Point) -> Fraction:
    total = Fraction(0)
    for exponent, coefficient in polynomial.items():
        term = coefficient
        for coordinate, power in zip(point, exponent):
            term *= coordinate**power
        total += term
    return total


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
