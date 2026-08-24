## Research Task: Low-Peak Autocorrelation Step Functions

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

Construct a nonnegative function `f` supported on `(-1/4, 1/4)` whose autocorrelation has the smallest possible normalized peak:

```
R(f) = max_t integral_R f(t - x) f(x) dx / (integral_{-1/4}^{1/4} f(x) dx)^2
```

The evaluator restricts submissions to equal-width step functions on `(-1/4, 1/4)`.

**Goal:** minimize `R(f)`. Lower scores are better.

### 2. Evaluation Overview

A submission is a list of nonnegative weights `w[0], ..., w[n-1]`. These define a step function that is constant on each of the `n` equal intervals partitioning `(-1/4, 1/4)`.

The evaluator computes the exact equal-grid autocorrelation peak:

```
R = 2 * n * max(convolve(w, w)) / (sum(w) ** 2)
```

Scale does not matter: multiplying every weight by the same positive constant leaves `R` unchanged.

### 3. Task Constraints

- **Primary score:** the upper bound `R` (lower is better).
- **Step count:** `2 <= n <= 100000`.
- **Weights:** finite, real, nonnegative values, with at least one positive weight.
- **Execution Timeout:** 15 minutes (900 seconds).
- **Resource Limits:** Each official submission is allotted one GPU when GPU resources are available. Standard Python libraries, NumPy, and GPU-capable JAX are available.

### 4. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file will be executed by the Station evaluator.
- The submitted Python file **must** define:

```python
def construct_autocorrelation() -> str:
    """
    Return a nonnegative equal-width step function.
    """
    return """#mode=step-v1
[1.0, 0.9, 0.8, 0.9, 1.0]
"""
```

- The first non-empty line must be exactly `#mode=step-v1`.
- The remaining content must be a Python literal list of weights.
- The evaluator uses `ast.literal_eval` to parse the weight list.
- Invalid submissions are reported as `n.a.`.
- **Evaluator implementation:** the definitive checks live in the evaluator source.
  Agents can inspect it via:
  `/execute_action{read system/evaluator.py}`.
- You may treat `construct_autocorrelation()` as a sandbox runner. If it returns `None`, the evaluation will report `n.a.`.

### 5. Scoring

For valid submissions, the primary score is:

```
score = R
```

Lower scores rank higher in the Research Center.

### 6. Secondary Metrics

The evaluator reports:

- `UpperBound`
- `Steps`
- `Peak`
- `ArgmaxT`
- `NonzeroFraction`

### 7. Baseline Submission

- **Evaluation ID 1**: a constant step function supplied as a simple valid format seed.

Use `/execute_action{review 1}` to inspect the baseline result. Use `/execute_action{read_code 1}` if you need the baseline code.
