## Research Task: A Locally Reversible but Globally Colliding Polynomial Map

**This specification is the authoritative statement of the task and overrides all other sources.**

### 1. Problem Description

Construct a polynomial map

```text
F = (P, Q, R): C^3 -> C^3
```

with rational coefficients that satisfies both conditions below:

1. The Jacobian determinant of `F` is an exact nonzero constant.
2. At least two distinct rational input points have exactly the same image under `F`.

The Jacobian matrix is

```text
      [dP/dx  dP/dy  dP/dz]
J_F = [dQ/dx  dQ/dy  dQ/dz]
      [dR/dx  dR/dy  dR/dz]
```

and the evaluator computes its determinant symbolically over the rational
numbers. A submitted collision at rational points is also checked exactly.

### 2. Evaluation and Scoring

The score is binary:

- **Score `1`**: all format and size constraints pass, the Jacobian determinant
  is an exact nonzero constant, and every submitted witness point is distinct
  while all witness points have the same image.
- **Score `0`**: every other outcome, including malformed data, a zero or
  nonconstant Jacobian determinant, or failure of the collision check.

There is no partial credit. Secondary metrics report which exact checks passed,
but they do not affect the score.

### 3. Constraints and Resources

- Exactly three polynomials `P`, `Q`, and `R` must be returned.
- Coefficients and witness coordinates must be rational numbers.
- Each polynomial must have total degree at most `12`.
- Each polynomial may contain at most `128` submitted term entries.
- After like terms are combined, the three polynomials may contain at most
  `256` nonzero terms in total.
- Numerators and denominators are limited to `512` bits each.
- Submit between `2` and `8` witness points. All submitted points must be
  pairwise distinct and must share one common image.
- The complete returned JSON string is limited to `250,000` characters.
- **Execution timeout:** 30 minutes (`1800` seconds).
- **Official resources:** one managed GPU and ten CPUs.
- **External literature access:** unavailable.

The official verification itself is fast and exact. The time and compute budget
are available for construction, symbolic search, algebraic manipulation, and
independent checks performed by the submitted program before it returns its
final candidate.

Common scientific Python tools, CAS packages, C/C++ compilers, and GPU-capable
libraries available in the Station environment may be used.

### 4. Submission Format

Your instruction should tell the coder what single construction or search
experiment to implement. The coder must submit a complete Python file defining:

```python
def construct_map() -> str:
    """Return the candidate map and collision witnesses as one JSON string."""
```

The returned JSON must have this shape:

```json
{
  "polynomials": [
    [
      {"coefficient": "3/2", "powers": [2, 1, 0]},
      {"coefficient": "-1", "powers": [0, 0, 1]}
    ],
    [
      {"coefficient": "1", "powers": [0, 1, 0]}
    ],
    [
      {"coefficient": "1", "powers": [0, 0, 1]}
    ]
  ],
  "points": [
    ["0", "0", "0"],
    ["1/2", "-3", "7/4"]
  ]
}
```

This example demonstrates syntax only and is not claimed to satisfy the task.

For a term

```json
{"coefficient": "3/2", "powers": [2, 1, 0]}
```

the evaluator reads `(3/2) * x^2 * y`. The three entries in `powers` are the
nonnegative exponents of `x`, `y`, and `z`, in that order.

Rational scalars may be supplied as JSON integers or base-10 strings in either
`"numerator"` or `"numerator/denominator"` form. Floating-point values,
decimal notation, scientific notation, and Boolean values are rejected.
Denominators must be nonzero. Duplicate powers are combined exactly.

The evaluator implementation is the definitive check and can be inspected with:

```text
/execute_action{read system/evaluator.py}
```

If `construct_map()` returns `None`, a non-string value, or malformed JSON, the
submission receives score `0`.

### 5. Exact Validity Check

For a well-formed submission, the evaluator:

1. Parses and combines all sparse polynomial terms over the exact rational
   numbers.
2. Differentiates `P`, `Q`, and `R` symbolically.
3. Expands `det(J_F)` exactly and checks that its only term is a nonzero
   constant.
4. Confirms that the submitted rational points are pairwise distinct.
5. Evaluates all three polynomials at every point using exact rational
   arithmetic.
6. Checks that all resulting image triples are identical.

Passing finite sampled points is not enough for the Jacobian condition: the
full determinant polynomial is expanded and checked exactly.

### 6. Secondary Metrics

The evaluator reports:

- `JacobianConstant`: `1` exactly when the determinant is a nonzero constant.
- `Collision`: `1` exactly when all witness points are distinct and share one
  image.
- `JacobianTerms`: number of nonzero terms in the expanded determinant.
- `MaxDegree`: maximum total degree among `P`, `Q`, and `R`.
- `TotalTerms`: total nonzero terms after combining like terms.
- `WitnessPoints`: number of submitted witness points.

These fields are diagnostics only. The official score remains either `0` or
`1`.

### 7. Baseline

- **Evaluation ID 1**: sparse-format identity-map control (expected score `0`).

The baseline exercises the parser and has a nonzero constant Jacobian, but its
two submitted points do not collide. Use `/execute_action{review 1}` to inspect
the result or `/execute_action{read_code 1}` to inspect its submission format.

