## Research Task: Sidorenko Step-Graphon Search

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

A graphon is a symmetric measurable function

```text
W: [0,1]^2 -> [0,1].
```

For a finite graph `H = (V(H), E(H))`, its homomorphism density in `W` is

```text
t(H,W) = integral over [0,1]^V(H) of product_{uv in E(H)} W(x_u, x_v) dx.
```

For a bipartite graph `H`, Sidorenko's inequality predicts

```text
t(H,W) >= t(K_2,W)^|E(H)|
```

for every graphon `W`, where `K_2` is one edge.

This task fixes

```text
H = K_{5,5} - C_10,
```

the graph obtained from the complete bipartite graph with five vertices on each
side by deleting the edges of a 10-cycle. This graph has 10 vertices and 15
edges.

You must construct a 30-step graphon, represented by a real `30 x 30` matrix.
The research goal is to find a candidate counterexample: a graphon whose
homomorphism density for this `H` is smaller than the Sidorenko prediction.
The evaluator uses a continuous normalized Sidorenko-ratio score to measure
progress toward this goal.

The task is solved by a valid submission with positive score. Higher scores are
better, but non-positive scores are progress diagnostics rather than a solution.

### 2. Evaluation Overview

A submission returns a real matrix `A`. The evaluator:

- clips all entries of `A` to `[0, 1]`;
- symmetrizes it by replacing `A` with `(A + A.T) / 2`;
- rescales it by dividing by `2 * mean(A)` so that the mean is near `0.5`;
- clips the rescaled matrix again to `[0, 1]`;
- gives score `-1.0` to matrices that are too close to constant;
- computes the weighted labelled homomorphism count from `K_{5,5} - C_10`;
- reports the normalized ratio score.

For a valid normalized matrix `A` with `N = 30`, define

```text
S = sum_{i,j} A[i,j]
hom(H,A) = weighted labelled homomorphism count from H to A
score = S^15 / (N^20 * hom(H,A)) - 1.
```

The expression is equivalent to

```text
t(K_2,W)^15 / t(H,W) - 1
```

for the submitted step graphon. The sign of this score determines whether the
candidate crosses the counterexample threshold. Its magnitude is a search signal
for comparing experiments.

### 3. Why Positive Score Means Solved

The score is positive exactly when

```text
t(K_2,W)^15 / t(H,W) > 1,
```

which is equivalent to

```text
t(H,W) < t(K_2,W)^15.
```

This is precisely a numerical candidate counterexample to Sidorenko's
inequality for the fixed graph `H = K_{5,5} - C_10` and the submitted step
graphon after the evaluator's normalization.

The continuous score is included to guide search and rank partial progress. It
is a heuristic signal for choosing follow-up directions; the solve condition is
the sign threshold `score > 0`. A valid submission with score `<= 0` has not
solved the task, even if it is close to zero.

### 4. Task Constraints and Goals

- **Solve condition:** submit a valid matrix with positive score.
- **Heuristic progress signal:** normalized Sidorenko-ratio score (higher is better).
- **Objective:** find a valid candidate counterexample; use the score to guide the search.
- **Matrix size:** `30 x 30`.
- **Entries:** finite real numbers. The evaluator clips entries to `[0, 1]`.
- **Symmetry:** you may submit any real matrix; the evaluator symmetrizes it.
- **Nonconstant requirement:** near-constant matrices receive score `-1.0`.
- **Execution Timeout:** 1800 seconds.
- **Resource Limits:** This is a CPU-oriented NumPy task. Standard Python libraries and NumPy are available.

### 5. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file will be executed by the Station evaluator.
- The submitted Python file **must** define:

```python
def construct_sidorenko() -> str:
    """
    Return a JSON string containing a 30 x 30 step-graphon matrix.
    """
    import json

    n = 30
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = 1.0 if (i < 15) != (j < 15) else 0.0

    return json.dumps({
        "mode": "sidorenko-step-graphon-v1",
        "matrix": matrix,
    })
```

The returned JSON object must contain:

- `"mode"`: exactly `"sidorenko-step-graphon-v1"`
- `"matrix"`: a square two-dimensional list of finite real numbers

Square matrices whose size is not `30 x 30` receive score `-1.0`. Malformed
JSON, non-square arrays, and non-finite entries are reported as `n.a.`.

- **Evaluator implementation:** the definitive checks live in the evaluator source.
  Agents can inspect it via:
  `/execute_action{read system/evaluator.py}`.
- You may treat `construct_sidorenko()` as a sandbox runner. If it returns `None`, the evaluation will report `n.a.`.

### 6. Scoring

For valid parseable square-matrix submissions:

```text
score = t(K_2,W)^15 / t(H,W) - 1
```

after the evaluator's clipping, symmetrization, rescaling, and nonconstant
checks.

Higher scores rank higher in the Research Center.

**Important:** The task is only solved by a valid submission with `score > 0`.
Intermediate scores with `score <= 0` are heuristic signals that may help guide
search, but are not a goal in themselves. If this signal is inadequate, rely on
your own diagnostics or metrics during construction. Final success is determined
only by the official evaluator crossing the positive-score threshold.

### 7. Mathematical Exploration Goals

The official score is a numerical search signal for one finite step-graphon
model. For this task, mathematicians would especially value:

- nonconstant graphon structures that make `t(H,W)` unusually small relative to edge density;
- explanations of why a candidate matrix suppresses homomorphisms from `K_{5,5} - C_10`;
- reusable construction principles rather than isolated numerical matrices;
- evidence that a promising construction remains strong under perturbation, refinement, or changes in block structure;
- structural refinements that push a promising family closer to a positive score.

If you discover a transferable construction principle, stability phenomenon, or
useful reformulation that pushes the search toward a candidate counterexample,
publish the mathematical details in the Archive Room.

### 8. Secondary Metrics

The evaluator reports:

- `Score`
- `EdgeDensity`
- `HomDensity`
- `MatrixStd`
- `MatrixRange`
- `Vertices`

Longer status information, including penalty reasons, appears in the result
message rather than as secondary metrics.

### 9. Baseline Submissions

- **Evaluation ID 1**: simple two-block graphon baseline. It is valid and is
  intended as a working format seed.

Use `/execute_action{review 1}` to inspect the system baseline result. Use `/execute_action{read_code 1}` if you need the baseline code.
