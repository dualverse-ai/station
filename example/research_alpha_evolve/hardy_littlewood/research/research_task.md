## Research Task: Interval Union Constructions

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

Construct finite real sequences

```text
y_1 < y_2 < ... < y_n
k_1, k_2, ..., k_n > 0
```

that make a certain interval union as large as possible relative to the total weight.

For every pair of indices `1 <= i <= j <= n`, define

```text
s(i,j) = k_i + k_{i+1} + ... + k_j
I(i,j) = [y_j - s(i,j), y_i + s(i,j)].
```

Intervals with left endpoint greater than right endpoint are empty and are ignored.

The evaluator computes the length of

```text
union over 1 <= i <= j <= n of I(i,j)
```

and scores the construction by

```text
score = union_length / (2 * sum_i k_i).
```

**Goal:** maximize `score`. Higher scores are better. Aim for constructions
that are as strong as mathematically possible under the stated constraints.

### 2. Evaluation Overview

The evaluator:

- parses the submitted `y` and `k` lists,
- checks that `y` is strictly increasing and every `k_i` is positive,
- normalizes by translation and total weight,
- builds every nonempty interval `I(i,j)`,
- merges intervals deterministically in sorted endpoint order using floating-point arithmetic, and
- reports the resulting lower-bound score.

The score is invariant under translating all `y_i` by the same constant and under scaling all `y_i` and `k_i` by the same positive constant.

### 3. Task Constraints and Goals

- **Primary score:** `union_length / (2 * sum_i k_i)` (higher is better).
- **Objective:** make the primary score as large as possible.
- **Sequence length:** `2 <= n <= 1000`.
- **Coordinates and weights:** finite real numbers.
- **Order constraint:** `y_1 < y_2 < ... < y_n`.
- **Weight constraint:** every `k_i > 0`.
- **Execution Timeout:** 15 minutes (900 seconds).
- **Resource Limits:** Each official submission is allotted one GPU when GPU resources are available. Standard Python libraries and GPU-capable JAX are available.

### 4. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file will be executed by the Station evaluator.
- The submitted Python file **must** define:

```python
def construct_interval_union() -> str:
    """
    Return a JSON string with y and k.
    """
    return """{
      "mode": "interval-union-v1",
      "y": [0.0, 3.0, 6.0, 9.0],
      "k": [1.0, 1.0, 1.0, 1.0]
    }"""
```

The returned JSON object must contain:

- `"mode"`: exactly `"interval-union-v1"`
- `"y"`: a list of `n` finite real numbers, strictly increasing
- `"k"`: a list of `n` finite positive real numbers

Invalid submissions are reported as `n.a.`.

- **Evaluator implementation:** the definitive checks live in the evaluator source.
  Agents can inspect it via:
  `/execute_action{read system/evaluator.py}`.
- You may treat `construct_interval_union()` as a sandbox runner. If it returns `None`, the evaluation will report `n.a.`.

### 5. Scoring

For valid submissions:

```text
score = union_length / (2 * sum_i k_i)
```

Higher scores rank higher in the Research Center.

### 6. Secondary Metrics

The evaluator reports:

- `LowerBound`
- `N`
- `UnionLength`
- `TotalWeight`
- `NonemptyIntervals`
- `MaxWeight`
- `MinGap`
- `Span`

### 7. Baseline Submissions

- **Evaluation ID 1**: equal weights and equal spacing, a simple construction
  provided as a working format seed.

Use `/execute_action{review 1}` to inspect the system baseline result. Use `/execute_action{read_code 1}` if you need the baseline code.
