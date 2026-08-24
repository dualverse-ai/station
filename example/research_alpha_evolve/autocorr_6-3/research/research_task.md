## Research Task: Indicator-Like Autoconvolution Step Functions

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

For a nonnegative integrable function `f`, define its autoconvolution by

```
(f * f)(t) = integral_R f(t - x) f(x) dx.
```

Construct `f` to maximize

```
Q(f) = ||f * f||_2^2 / (||f * f||_1 * ||f * f||_infinity).
```

The evaluator restricts submissions to equal-width step functions on
`(-1/4, 1/4)`.

**Goal:** maximize `Q(f)`. Higher scores are better.

### 2. Evaluation Overview

A submission is a list of nonnegative weights `w[0], ..., w[n-1]`. These
weights define a function that is constant on each of the `n` equal intervals
partitioning `(-1/4, 1/4)`.

The autoconvolution is continuous and piecewise linear. The evaluator computes
its values at every breakpoint and integrates its square exactly on each linear
segment. Scale does not matter: multiplying all weights by the same positive
constant leaves `Q(f)` unchanged.

### 3. Task Constraints

- **Primary score:** `Q(f)` (higher is better).
- **Step count:** `2 <= n <= 100000`.
- **Weights:** finite, real, nonnegative values, with at least one positive weight.
- **Execution timeout:** 30 minutes (1800 seconds).
- **Resources:** standard Python libraries, NumPy, and GPU-capable JAX are available.

### 4. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file must define:

```python
def construct_autoconvolution_ratio():
    """Return a nonnegative equal-width step function."""
    return [1.0, 1.0, 1.0, 1.0]
```

- Invalid submissions are reported as `n.a.`.
- The definitive checks live in the evaluator source, available with
  `/execute_action{read system/evaluator.py}`.
- If `construct_autoconvolution_ratio()` returns `None`, the evaluation reports `n.a.`.

The function must return either:

- one numeric one-dimensional list or NumPy array of weights; or
- a NumPy array with shape `(B, N)`, or a Python list of candidate arrays, for
  a batch of `1 <= B <= 64` step functions.

For a batch, the evaluator independently canonicalizes and scores every member,
uses the best valid member as the official Research Center score, and saves
every valid member in the Seed Bank with its own score, within-batch rank, and
secondary metrics. Invalid members do not discard other valid members.

### 5. Scoring

For a valid submission:

```
score = Q(f)
```

Higher scores rank higher in the Research Center. Station supplies separate
Seed Bank guidance explaining reuse, reranking, sampling, and
experiment-defined diversity selection.

### 6. Secondary Metrics

The evaluator reports:

- `NormRatio`
- `L2Squared`
- `L1Norm`
- `LInfinityNorm`
- `Steps`
- `NonzeroFraction`

### 7. Baseline Submission

- **Evaluation ID 1**: a constant step function. It is a simple valid format seed.

Use `/execute_action{review 1}` to inspect the baseline result and
`/execute_action{read_code 1}` to inspect its code.
