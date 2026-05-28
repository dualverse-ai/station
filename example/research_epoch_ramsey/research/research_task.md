## Research Task: Hypergraph Partition Lower-Bound Improvement

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

A hypergraph `(V, H)` contains a partition of size `m` if there exist `D ⊆ V` and `P ⊆ H` such that:

- `|D| = m`, and
- every vertex in `D` is contained in exactly one hyperedge of `P`.

Define `H(n)` to be the largest integer `k` such that there exists a hypergraph `(V, H)` with:

- `|V| = k`,
- no isolated vertices, and
- no partition of size greater than `n`.

A known lower bound is `H(n) ≥ k_n`, where:

- `k_1 = 1`
- `k_n = floor(n/2) + k_floor(n/2) + k_floor((n+1)/2)` for `n >= 2`

Your task is to provide an algorithm `solution(n)` that outputs a witness hypergraph improving this lower bound by a constant factor on the evaluator range.

### 2. Evaluation Overview

The evaluator uses **binary primary scoring** on the range `15 <= n <= 50`.

For each tested `n`, it:

- calls `solution(n)`,
- parses the returned hypergraph,
- checks that the vertex labels are exactly `1,2,...,|V|`,
- checks that there are no isolated vertices,
- computes the maximum partition size exactly, and
- verifies that the returned hypergraph has no partition of size greater than `n`.

**Primary score:**

- `score = 100` iff every tested `n` passes validity and `|V| > k_n`
- `score = 0` otherwise

This means the evaluator certifies an empirical constant
`c_eval = min_{15 <= n <= 50} |V_n| / k_n`,
and the task is solved exactly when `c_eval > 1`.

### 3. Task Constraints and Goals

- **Goal:** produce a valid witness family with binary score `100`
- **Range enforced by evaluator:** all integers `n` from `15` through `50`
- **Execution expectation:** for `n <= 50`, construction should finish within 10 minutes on a typical laptop
- **Execution Timeout:** 20 minutes (1200 seconds)
- **Active experiments:** each agent can have at most 1 active experiment at a time
- **Resource Limits:** CPU-only. Standard Python libraries are available.
- **Available Packages:** `python-sat`, `z3-solver`, `ortools`, and `sage` are installed and may be used.

### 4. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file will be executed by the Station evaluator.
- The submitted Python file must define the following exact function:

  ```python
  def solution(n: int) -> str:
      """
      Return a hypergraph on vertices 1..|V| as a string such as:
      "{1,2,3},{2,4},{3,4,5},{1,5}"
      """
      ...
  ```

- Do not rely on file-level execution.
- The runner imports the submission, calls `solution(n)` for each tested `n`, and evaluates the returned string.
- You may optionally define:

  ```python
  def test():
      """
      Optional debug-only entrypoint.
      If defined, the runner executes only test() and returns score n.a.
      """
      ...
  ```

- `test()` is useful for debugging, quick experiments, or inspecting helper logic without triggering official scoring.
- **Evaluator implementation:** the definitive checks live in:
  - `execute_action{read system/evaluator.py}`
  - `execute_action{read system/epoch_ramsey_eval_runner.py}`

### 5. Hypergraph Parsing Rules

The returned string is parsed as a comma-separated list of hyperedges, where each hyperedge is written with curly braces.

Example:

```text
{1,2,3},{2,4},{3,4,5},{1,5}
```

Parsing rules:

- Each hyperedge must be non-empty.
- Vertex labels must be positive integers.
- Duplicate vertices inside one edge are not allowed.
- The vertex set is inferred as `{1,2,...,m}` where `m` is the largest label appearing.
- Every label in that range must appear in at least one edge.
- Duplicate hyperedges are ignored semantically by the evaluator.

### 6. Exact Validity Check

For a chosen set of hyperedges `P`, let `deg_P(v)` be the number of selected edges containing `v`.
The evaluator defines the partition size of `P` to be:

```text
|{ v in V : deg_P(v) = 1 }|
```

The witness is valid for `n` exactly when:

```text
max_{P subseteq H} |{ v in V : deg_P(v) = 1 }| <= n
```

The evaluator computes this maximum exactly with a CP-SAT model.

### 7. Secondary Metrics

The evaluator reports:

- `MinRatio`
- `MinExcess`
- `WorstN`
- `ValidCount`
- `WorstPartitionSlack`

Meanings:

- `MinRatio = min |V_n| / k_n` over tested `n`
- `MinExcess = min (|V_n| - k_n)` over tested `n`
- `WorstN` is the `n` achieving `MinRatio`
- `ValidCount` is the number of tested `n` with a valid witness
- `WorstPartitionSlack = min (n - p*_n)` where `p*_n` is the exact maximum partition size

The solve condition is still purely binary: only `score = 100` counts as solved.

### 8. Baseline Submissions

- **Evaluation ID 1**: Triangle-union control baseline. It is valid but does not improve the known lower bound on the full range, so its binary score is expected to be `0`.

Use `/execute_action{review 1}` to inspect the baseline result. Use `/execute_action{read_code 1}` if you need the baseline code.

### 9. Notes

- Since the task must verify a family `n -> G_n`, command mode is used so the runner can call `solution(n)` repeatedly.
- The primary score is intentionally binary; progress should be read from the secondary metrics, not from intermediate scores.
