## Research Task: Finite-Field Nikodym Set Optimization

**This specification holds the highest degree of credibility in this research
station and overrides all other sources.**

### 1. Problem Description

Let `p` be a prime and let `F_p = {0, 1, ..., p-1}`, with arithmetic modulo
`p`. A point of `F_p^3` is a triple `(x_1, x_2, x_3)`.

For a point `x` and a nonzero direction vector `v`, the affine line through
`x` in direction `v` is

```text
L(x,v) = {x + t v mod p : t in F_p}.
```

A set `N subseteq F_p^3` is a Nikodym set when every point `x in F_p^3` has
some line through it whose other points all belong to `N`:

```text
for every x, there is a nonzero v such that
x + t v belongs to N for every t = 1, 2, ..., p-1.
```

Equivalently, `L(x,v)` is contained in `N union {x}`. The point `x` itself may
or may not belong to `N`.

This point-wise condition differs from the finite-field Kakeya condition. A
Kakeya set must contain one complete line in every direction; a Nikodym set
must provide a suitable punctured line through every point.

The computational goal is to construct Nikodym sets in `F_p^3` with as few
points as possible, using one algorithm that works across the tested primes.

### 2. Evaluation Overview

The evaluator calls the submitted function with `d = 3` at exactly these
primes:

```text
p = 5, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89.
```

For every prime, it:

- calls `search_for_best_construction(p, 3)`;
- checks that the result is a nonempty NumPy array of shape `(k, 3)`;
- removes duplicate rows and converts the coordinates to `numpy.int64`;
- checks the Nikodym property exhaustively for every point of `F_p^3`; and
- records the number of unique submitted rows.

The `p = 5` instance is a validity check. Valid results at `p = 5` are not
included in the primary-score average. All ten primes from `47` through `89`
are included in the score.

### 3. Submission Format

Your instruction should tell the coder what single experiment to implement.
Your coder should submit a complete Python script defining exactly:

```python
def search_for_best_construction(p: int, d: int):
    """
    Return a NumPy array of shape (k, 3) representing a proposed
    Nikodym set in F_p^3. The evaluator always supplies d == 3.
    """
    import itertools
    import numpy as np

    # Trivial format example: return the full space.
    return np.asarray(
        list(itertools.product(range(p), repeat=3)),
        dtype=np.int64,
    )
```

Submission requirements:

- The return value must be a NumPy array.
- It must have exactly two dimensions and shape `(k, 3)` with `k > 0`.
- Rows represent points `(x_1, x_2, x_3)`.
- Mathematical constructions should return integer coordinates in
  `0, ..., p-1`.
- The evaluator first removes exactly equal submitted rows and then converts
  the remaining array to `numpy.int64`, matching NumPy's integer-cast
  behavior. It does not reduce coordinates modulo `p` when building the
  submitted membership set. Out-of-range values therefore count toward the
  submitted size but cannot match the canonical points checked by the
  verifier.
- Exactly duplicate rows are ignored for verification and scoring.
- The function must work when called independently at every tested prime.
- Do not rely on file-level execution or on call order between primes.
- Official evaluation is CPU-only with a 3600-second timeout. Standard Python,
  NumPy, and Numba are available; no GPU is allocated.

Invalid instances receive the evaluator's fixed invalid-instance penalty.

The definitive checks live in the evaluator and runner sources. They can be
inspected with:

```text
/execute_action{read system/evaluator.py}
/execute_action{read system/nikodym_eval_runner.py}
```

### 4. Exact Validity Check

Two nonzero vectors represent the same direction when one is a nonzero scalar
multiple of the other. The evaluator checks one canonical representative of
each projective direction:

```text
(1, a, b) for every a,b in F_p,
(0, 1, b) for every b in F_p,
(0, 0, 1).
```

Thus there are `p^2 + p + 1` directions.

For each point `x in F_p^3`, the evaluator searches these directions in the
order shown. A direction `v` witnesses the Nikodym condition at `x` exactly
when all `p-1` points

```text
x + t v mod p,  t = 1, 2, ..., p-1,
```

belong to the submitted set. The construction is valid only if at least one
such direction exists for every `x`.

The evaluator performs exact integer membership checks after conversion to
`numpy.int64`; it uses no numerical tolerance or randomized sampling.

### 5. Scoring

For each scored prime, define the normalization value

```text
D_p = (p - 1)^3 + p + 1
```

and the relative size

```text
r_p = |N_p| / D_p.
```

For a submission valid at every tested prime, the primary score is

```text
score = -arithmetic_mean(r_p for p = 47, 53, 59, 61, 67, 71, 73, 79, 83, 89).
```

Higher scores are better: making `N_p` smaller makes `r_p` smaller and the
negative mean larger. There is no fixed target threshold; make the score as
high as possible.

If an instance is invalid, the evaluator inserts the fixed value `1000` for
that instance into its averaging list. This makes invalid submissions rank far
below fully valid constructions. An invalid `p = 5` check is penalized even
though a valid `p = 5` result is not otherwise included in the mean.

### 6. Finite Benchmark and Mathematical Exploration Goals

The official evaluator checks finitely many primes in dimension `3`. A strong
finite score does not prove that a construction works for every prime, and it
does not certify any higher-dimensional claim.

Beyond improving the official score, mathematically valuable outcomes include:

- one explicit construction rule that works for a broad or infinite family of
  primes rather than unrelated prime-specific lookup tables;
- an exact cardinality formula or a justified asymptotic estimate for a
  proposed family;
- a proof of the Nikodym property for all primes covered by that family;
- a structurally transparent explanation of why the punctured lines overlap;
- rigorous negative results that rule out natural construction families; and
- a general construction principle that extends from dimension `3` to other
  fixed dimensions `d >= 3`, together with size bounds and a proof of the
  higher-dimensional Nikodym property.

Higher-dimensional generalization is especially valuable, but it is unscored:
the official evaluator remains fixed at `d = 3`. Publish general formulas,
proofs, asymptotic analyses, and higher-dimensional extensions in the Archive
Room even when they do not improve the finite benchmark score.

### 7. Secondary Metrics

For fully valid submissions, the evaluator reports:

- `MeanRelativeSize`: arithmetic mean of `r_p` over the ten scored primes;
- `WorstRelativeSize`: largest `r_p`;
- `BestRelativeSize`: smallest `r_p`;
- `WorstP`: prime attaining `WorstRelativeSize`;
- `BestP`: prime attaining `BestRelativeSize`;
- `ValidCount`: number of valid instances among all eleven tested primes; and
- `ScoredCount`: number of valid scored instances included in the mean.

The result message also summarizes the verified set size and relative size at
each scored prime.

### 8. Baseline Submission

- **Evaluation ID 1**: full-space baseline. It returns every point of `F_p^3`,
  so it is trivially valid and intentionally noncompetitive.

Use `/execute_action{review 1}` to inspect the baseline result. Use
`/execute_action{read_code 1}` to inspect its code.
