## Research Task: Ramsey Lower-Bound Construction for Triangular Books

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

Let `B_n` denote the triangular book graph on `n + 2` vertices, i.e. `n` triangles sharing a common edge.

Your task is to produce constructions for the tested range. For each tested `n`, the returned graph `G_n` must have exactly `4n - 2` vertices such that:

- `G_n` does **not** contain `B_{n-1}`, and
- the complement `G_n^c` does **not** contain `B_n`.

This would certify:

```text
R(B_{n-1}, B_n) > 4n - 2
```

for the tested evaluator range, matching the conjectured sharp lower bound family.

The upper bound `R(B_{n-1}, B_n) <= 4n - 1` is known. Constructions are already known for all `n <= 21` and for an infinite family of `n`.

### 2. Evaluation Overview

The evaluator uses **exact range-coverage scoring** on the range `22 <= n <= 50`.

It calls `solution_batch()` once. For each tested `n`, it:

- reads the returned graph for `n` from the batch result,
- parses the returned graph from its adjacency string,
- checks that the graph has exactly `4n - 2` vertices,
- computes the maximum number of common neighbors on every edge of `G_n`,
- computes the maximum number of common neighbors on every edge of `G_n^c`, and
- verifies the two exact book-avoidance inequalities.

For a graph `G`, a copy of `B_t` exists iff some edge of `G` lies in at least `t` triangles. Therefore:

- `G_n` avoids `B_{n-1}` iff every edge of `G_n` has at most `n - 2` common neighbors.
- `G_n^c` avoids `B_n` iff every edge of `G_n^c` has at most `n - 1` common neighbors.

**Primary score:**

- `ValidCount` is the number of tested `n` that pass all exact checks
- `score = 100 * ValidCount / 29`
- `score = 100` iff every tested `n` passes all exact checks

The per-instance validity condition remains exact. The station score tracks how much of the target family range is covered while preserving the full-range solve condition.

### 3. Task Constraints and Goals

- **Goal:** produce a valid witness family with full-range score `100`
- **Range enforced by evaluator:** all integers `n` from `22` through `50`
- **Execution expectation:** the full batch should finish within the station timeout
- **Runner policy:** the bundled runner calls `solution_batch()` once and validates all returned values in `22..50`
- **Execution Timeout:** 15 minutes (900 seconds)
- **Active experiments:** each agent can have at most 1 active experiment at a time
- **Resource Limits:** CPU-only. Standard Python libraries are available.
- **Available Packages:** `python-sat`, `z3-solver`, `ortools`, and `sage` are installed and may be used.

### 4. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file must define the following exact function:

  ```python
  def solution_batch() -> dict[int | str, str]:
      """
      Return adjacency strings for tested n values.
      """
      ...
  ```

- Keys may be integers or strings. For example, either `22` or `"22"` may be used for `n = 22`.
- Values must be adjacency strings in the format below.
- Submissions may reload saved constructions or artifacts for already solved values of `n`; the evaluator only checks the returned graphs.
- You may optionally define:

  ```python
  def test():
      """
      Optional debug-only entrypoint.
      If defined, the runner executes only test() and returns score n.a.
      """
      ...
  ```

- `test()` is useful for debugging or quick experiments without triggering official scoring.
- **Evaluator implementation:** the definitive checks live in:
  - `execute_action{read system/evaluator.py}`
  - `execute_action{read system/epoch_book_eval_runner.py}`

### 5. Adjacency String Format

The returned graph must be encoded as a binary string of length `C(4n - 2, 2)`.

Bit order is **column-major by the larger endpoint**, zero-indexed:

- `{0,1}`, `{0,2}`, `{1,2}`, `{0,3}`, `{1,3}`, `{2,3}`, ...

So for each `j = 1,2,...,N-1`, list bits for pairs `{0,j}, {1,j}, ..., {j-1,j}`.

Interpretation:

- bit `1` means the edge is present in `G`
- bit `0` means the edge is absent in `G`

Whitespace around the string is ignored. No separators are allowed inside the bitstring.

### 6. Exact Validity Check

Let `N_G(u)` denote the neighborhood of `u` in `G`.

The evaluator computes:

```text
red_max  = max_{uv in E(G)}     |N_G(u) ∩ N_G(v)|
blue_max = max_{uv in E(G^c)}   |N_{G^c}(u) ∩ N_{G^c}(v)|
```

A witness is valid for `n` exactly when:

```text
red_max  <= n - 2
blue_max <= n - 1
```

Equivalently, the evaluator checks:

- `red_excess  = max(0, red_max  - (n - 2))`
- `blue_excess = max(0, blue_max - (n - 1))`

and validity means both excesses are zero.

### 7. Secondary Metrics

The evaluator reports:

- `ValidCount`
- `WorstRedExcess`
- `WorstBlueExcess`
- `WorstN`
- `WorstViolatingPairRate`

Meanings:

- `ValidCount` is the number of tested `n` with a fully valid witness; the primary score is `100 * ValidCount / 29`
- `WorstRedExcess` is the maximum red obstruction excess over the range
- `WorstBlueExcess` is the maximum blue obstruction excess over the range
- `WorstN` is the tested `n` with the largest normalized obstruction
- `WorstViolatingPairRate` is, on `WorstN`, the larger of:
  - fraction of edges of `G_n` whose common-neighbor count exceeds `n - 2`
  - fraction of edges of `G_n^c` whose common-neighbor count exceeds `n - 1`

Apart from `ValidCount`, these are exact diagnostics, not alternate objectives.

### 8. Baseline Submissions

- **Evaluation ID 1**: Balanced complete bipartite baseline. It is evaluator-valid and produces useful diagnostics, but its score is expected to remain low because it fails many tested values of `n`.

Use `/execute_action{review 1}` to inspect the baseline result. Use `/execute_action{read_code 1}` if you need the baseline code.
