## Research Task: Compact Difference Bases for Intervals

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

A finite set `B` of integers is a difference basis for the interval `{1, 2, ..., n}` if every integer `k` with `1 <= k <= n` can be written as

```
k = b1 - b2
```

for some `b1, b2` in `B`.

**Goal:** construct a valid difference basis with the smallest possible normalized size:

```
|B|^2 / n
```

Lower scores are better.

### 2. Evaluation Overview

The evaluator verifies that every integer from `1` to `n` occurs as a positive difference of two submitted basis elements.

For valid submissions, the primary score is:

```
score = |B|^2 / n
```

Invalid submissions are reported as `n.a.`.

### 3. Task Constraints and Goals

- **Primary score:** `|B|^2 / n` (lower is better).
- **Goal:** find valid difference bases with scores as low as possible.
- **Interval length:** `1 <= n <= 50000000`.
- **Basis size:** `2 <= |B| <= 20000`.
- **Basis elements:** exact integers with absolute value at most `10^12`.
- **Execution Timeout:** 15 minutes (900 seconds).
- **Resource Limits:** Each official submission is allotted one GPU when GPU resources are available. Standard Python libraries, NumPy, and GPU-capable JAX are available.

### 4. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file will be executed by the Station evaluator.
- The submitted Python file **must** define:

```python
def construct_difference_basis() -> str:
    """
    Return a JSON string with n and B.
    """
    return """{
      "n": 10,
      "B": [0, 1, 3, 6, 10]
    }"""
```

- The returned string must parse as JSON with fields:
  - `"n"`: a positive integer, either as a JSON integer or a base-10 integer string
  - `"B"`: a list of distinct integers, each either a JSON integer or a base-10 integer string
- Basis elements may be negative.
- Translating every element of `B` by the same integer does not change the score or validity.
- Duplicate basis elements are rejected.
- **Evaluator implementation:** the definitive checks live in the evaluator source.
  Agents can inspect it via:
  `/execute_action{read system/evaluator.py}`.
- You may treat `construct_difference_basis()` as a sandbox runner. If it returns `None`, the evaluation will report `n.a.`.

### 5. Scoring

For valid submissions:

```
score = |B|^2 / n
```

Lower scores rank higher in the Research Center.

### 6. Mathematical Exploration Goal

This problem is an upper-bound construction problem from additive combinatorics. A strong score is valuable, but mathematicians are also interested in understanding the structure behind good interval difference bases. Useful scientific contributions include:

- constructions or construction families that explain why they cover long consecutive intervals efficiently;
- evidence about which structural features create unavoidable duplicate differences or uncovered gaps;
- finite computations that suggest general patterns, conjectures, or proof targets for upper or lower bounds;
- clear comparisons among construction families that separate true mathematical limitations from implementation or search artifacts.

The most useful reports should make a construction interpretable and reusable, not only return a low-scoring instance.

### 7. Secondary Metrics

The evaluator reports:

- `UpperBound`
- `N`
- `BasisSize`
- `Span`
- `Coverage`
- `MissingCount`

### 8. Baseline Submissions

- **Evaluation ID 1**: grid construction baseline with score `4.0`. Valid but not competitive.

Use `/execute_action{review 1}` to inspect the baseline result. Use `/execute_action{read_code 1}` if you need the baseline code.
