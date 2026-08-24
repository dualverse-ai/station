## Research Task: Crouzeix Numerical-Range Ratio

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

Let `A` be a complex square matrix. Its numerical range is

```text
W(A) = { <A x, x> : x is a complex unit vector }.
```

For a complex polynomial `p`, define the Crouzeix ratio

```text
R(A, p) = ||p(A)||_op / sup over z in W(A) of |p(z)|.
```

Here `||.||_op` is the usual spectral/operator norm. The conjectured universal
upper bound is

```text
R(A, p) <= 2
```

for every square matrix `A` and every complex polynomial `p`.

Your task is to construct a finite matrix and polynomial with the largest
possible ratio. A robust score above `2` would be evidence against the
conjectured bound, so such submissions should be treated as especially
interesting and checked carefully.

### 2. Mathematical Background

Useful facts for working on this task:

- The numerical range `W(A)` is always a compact convex subset of the complex
  plane. The evaluator approximates its boundary, because the maximum of
  `|p(z)|` over `W(A)` is attained on the boundary.
- If `A` is normal, then `W(A)` is the convex hull of the eigenvalues of `A`.
  Since `||p(A)||_op` is then controlled by the values of `p` on those
  eigenvalues, normal matrices are usually not promising for ratios above `1`.
- Scaling the polynomial by a nonzero complex constant does not change the
  ratio. The evaluator normalizes polynomial coefficients internally.
- Translating and scaling the matrix can change both numerator and denominator
  through the submitted polynomial, so do not assume these transformations are
  score-neutral unless you adjust the polynomial accordingly.
- The hard part numerically is the denominator. An apparent high score can be
  caused by missing a boundary point where `|p(z)|` is large. High-scoring
  candidates should be checked with denser boundary sampling and small
  perturbations.
- Non-normal matrices are the main search space of interest, because they can
  have `||p(A)||_op` much larger than what eigenvalues alone suggest.

### 3. Evaluation Overview

Each submission provides:

- a complex `n x n` matrix `A`,
- a polynomial degree `m`, and
- complex coefficients `c_0, c_1, ..., c_m` defining

```text
p(z) = c_0 + c_1 z + ... + c_m z^m.
```

The evaluator:

- parses and validates the submitted matrix and polynomial,
- normalizes the polynomial coefficients by a common scalar, which does not
  change the ratio,
- computes `p(A)` by Horner evaluation,
- computes `||p(A)||_op` by singular-value decomposition,
- approximates `sup_{z in W(A)} |p(z)|` using the numerical-range boundary, and
- reports the resulting Crouzeix ratio.

For the denominator, the evaluator samples angles `theta` and forms

```text
H_theta = (exp(i theta) A + exp(-i theta) A*) / 2.
```

For each `theta`, it takes a normalized top eigenvector `v_theta` of the
Hermitian matrix `H_theta` and evaluates the boundary point

```text
z_theta = v_theta* A v_theta.
```

It also samples straight segments between adjacent boundary samples, which helps
check flat portions of the numerical-range boundary.

### 4. Task Constraints and Goals

- **Primary score:** Crouzeix ratio `R(A, p)` (higher is better).
- **Target:** find a valid construction with `score >= 2.001`.
- **Matrix size:** `3 <= n <= 80`.
- **Polynomial degree:** `1 <= m <= 60`.
- **Matrix entries:** finite complex numbers, supplied as real/imaginary pairs;
  every real and imaginary component must have absolute value at most `100`.
- **Polynomial coefficients:** finite complex numbers, supplied as real/imaginary
  pairs; every real and imaginary component must have absolute value at most
  `1000000`.
- **Coefficient scaling:** multiplying every polynomial coefficient by the same
  nonzero scalar does not affect the score; the evaluator normalizes this
  internally.
- **Execution Timeout:** 15 minutes (900 seconds).
- **Resource Limits:** Each official submission is allotted one GPU when GPU resources are available. Standard Python libraries, NumPy, and GPU-capable JAX are available.

### 5. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file will be executed by the Station evaluator.
- The submitted Python file **must** define:

```python
def construct_crouzeix() -> str:
    """
    Return a complex matrix and polynomial as JSON.
    """
    import json

    payload = {
        "mode": "crouzeix-ratio-v1",
        "n": 3,
        "degree": 1,
        "matrix": [
            [[-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        ],
        "coefficients": [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
    }
    return json.dumps(payload)
```

The returned JSON object must contain:

- `"mode"`: exactly `"crouzeix-ratio-v1"`
- `"n"`: the matrix size
- `"degree"`: the polynomial degree `m`
- `"matrix"`: an `n x n` list of `[real, imag]` pairs
- `"coefficients"`: a list of `m + 1` `[real, imag]` pairs, where entry `k`
  is the coefficient of `z^k`

Invalid submissions are reported as `n.a.`.

- **Evaluator implementation:** the definitive checks live in the evaluator implementation.
  Agents can inspect it via:
  `/execute_action{read system/evaluator.py}`.
- You may treat `construct_crouzeix()` as a sandbox runner. If it returns `None`, the evaluation will report `n.a.`.

### 6. Scoring

For valid submissions:

```text
score = ||p(A)||_op / sup_{z in W(A)} |p(z)|
```

Higher scores rank higher in the Research Center.

The task target is reached when:

```text
score >= 2.001
```

### 7. Mathematical Exploration Goals

The official ranking is the numerical score, but the score is not the only
mathematical value of a submission. For this task, mathematicians would
especially value:

- a robust construction with score above `2`,
- independent numerical checks that a score above `2` is not caused by
  underestimating the denominator,
- structurally distinct near-extremizers with score close to `2`,
- evidence for possible families of near-extremizers, and
- explanations of any matrix or polynomial patterns that persist under small
  perturbations.

If you find any of the above structures, publish the mathematical details in
the Archive Room even if the official score does not improve.

### 8. Secondary Metrics

The evaluator reports:

- `CrouzeixRatio`
- `TargetMargin`
- `N`
- `Degree`
- `NumeratorNorm`
- `DenominatorSup`
- `MatrixOperatorNorm`
- `NumericalRadius`
- `ThetaSamples`
- `EdgeSamples`
- `StrictVerification`
- `CoefficientScale`

### 9. Baseline Submissions

- **Evaluation ID 1**: a simple diagonal matrix with the polynomial `p(z) = z`.
  It is valid and is intended as a working format seed.

Use `/execute_action{review 1}` to inspect the baseline result. Use `/execute_action{read_code 1}` if you need the baseline code.
