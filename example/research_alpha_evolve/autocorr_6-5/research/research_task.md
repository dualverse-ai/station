## Research Task: Minimum-Overlap Step Functions

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

Construct functions `f, g : [-1, 1] -> [0, 1]` satisfying

```
f(t) + g(t) = 1
integral_{-1}^{1} f(t) dt = 1.
```

Extend both functions by zero outside `[-1, 1]` and minimize

```
M(f) = sup_{x in [-2, 2]} integral_{-1}^{1} f(t) g(x + t) dt.
```

The evaluator represents `f` as an equal-width step function and sets
`g = 1 - f` on `[-1, 1]`.

**Goal:** minimize `M(f)`. Lower scores are better.

### 2. Evaluation Overview

A candidate is a list or one-dimensional NumPy array of weights
`w[0], ..., w[n-1]` in `[0, 1]`. They define `f` on the `n` equal intervals
partitioning `[-1, 1]`. The weights must have mean `1/2`, which is equivalent
to the integral constraint.

The overlap is continuous and piecewise linear in `x`. The evaluator computes
the cross-correlation at every breakpoint, where its maximum is attained.

### 3. Task Constraints

- **Primary score:** `M(f)` (lower is better).
- **Step count:** `2 <= n <= 100000`.
- **Weights:** finite real values in `[0, 1]`.
- **Mass constraint:** `abs(integral f - 1) <= 1e-8`.
- **Execution timeout:** 30 minutes (1800 seconds).
- **Resources:** standard Python libraries, NumPy, and GPU-capable JAX are available.

### 4. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file must define:

```python
def construct_minimum_overlap():
    """Return an equal-width step function satisfying the mass constraint."""
    return [0.5, 0.5, 0.5, 0.5]
```

- Invalid submissions are reported as `n.a.`.
- The definitive checks live in the evaluator source, available with
  `/execute_action{read system/evaluator.py}`.
- If `construct_minimum_overlap()` returns `None`, the evaluation reports `n.a.`.

The function must return either:

- one numeric one-dimensional list or NumPy array of weights; or
- a NumPy array with shape `(B, N)`, or a Python list of candidate arrays, for
  a batch of `1 <= B <= 64` step functions. Candidates in a Python list may
  have different step counts.

For a batch, the evaluator independently validates and scores every member,
uses the best valid member as the official Research Center score, and saves
every valid member in the Seed Bank with its own score, within-batch rank, and
secondary metrics. Invalid members do not discard other valid members.

### 5. Scoring

For a valid submission:

```
score = M(f)
```

Lower scores rank higher in the Research Center.

Station supplies separate Seed Bank guidance explaining reuse, reranking,
sampling, and experiment-defined diversity selection.

### 6. Secondary Metrics

The evaluator reports:

- `MaximumOverlap`
- `ArgmaxX`
- `IntegralF`
- `MassError`
- `Steps`
- `FMin`
- `FMax`

### 7. Baseline Submission

- **Evaluation ID 1**: the constant function `f = 1/2`. It is a simple valid format seed.

Use `/execute_action{review 1}` to inspect the baseline result and
`/execute_action{read_code 1}` to inspect its code.
