## Research Task: Kissing Number Lower Bound for d=11

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

# 1. Problem Description

The kissing number problem asks how many non-overlapping unit spheres can simultaneously touch one central unit sphere. This task fixes the ambient dimension at

\[
d = 11.
\]

The goal is to exhibit **594** surrounding spheres in this dimension.

Submit **594 non-zero direction vectors** in \(\mathbb{R}^{11}\). A submitted vector \(v_i\) is used only for its direction. The evaluator normalizes each row and converts it into the center

\[
p_i = \frac{2v_i}{\|v_i\|}.
\]

The point \(p_i\) lies on the sphere of radius \(2\) around the origin, so a unit sphere centered at \(p_i\) touches the central unit sphere.

Two surrounding unit spheres are non-overlapping exactly when their centers are at distance at least \(2\). Thus the returned directions solve the task if

\[
\|p_i - p_j\| \ge 2
\]

for every pair \(1 \le i < j \le 594\).

# 2. Scoring

The score is the total overlap loss:

\[
\text{score}
=
\sum_{1 \le i < j \le 594}
\max\left(0,\; 2-\|p_i-p_j\|\right),
\]

where

\[
p_i=\frac{2v_i}{\|v_i\|}.
\]

Lower is better.

A score \(>0\) means that at least one pair of surrounding spheres overlaps.

A score \(\le 0\) means that no overlap was detected by the evaluator, so the submitted configuration is accepted as solving the task.

In exact arithmetic, every term in the score is nonnegative. Therefore,

\[
\text{score} \le 0
\]

is equivalent to

\[
\text{score}=0,
\]

which means all surrounding sphere centers are pairwise at distance at least \(2\). This gives a valid kissing configuration with **594** surrounding spheres in dimension \(11\).

# 3. Accepted Numeric Formats

Agents may submit vectors using several numeric representations.

Accepted coordinate types include:

* `float64`
* `int64`
* exact rational numbers
* exact algebraic numbers, including expressions such as

\[
1+\sqrt{2}.
\]

Floating-point submissions are allowed and are useful for search. However, because `float64` arithmetic is affected by rounding error, a `float64` solution is considered a numerical solution and is less preferred than an exact solution.

Preferred final submissions are those that can be verified exactly or with rigorously justified arithmetic, such as:

1. **Integer vectors**, for example `int64`.
2. **Exact rational vectors**, for example using `fractions.Fraction` or `sympy.Rational`.
3. **Exact algebraic vectors**, for example using SymPy expressions such as `sqrt(2)` or `1 + sqrt(2)`.

For exact submissions, the evaluator attempts exact certification when the numerical overlap loss is zero. In that case,

\[
\text{score} \le 0
\]

is a certified solution, not merely a numerical candidate.

# 4. Execution Environment

The submitted code is run in a restricted sandboxed environment.

The official evaluation has an externally enforced timeout of about **30 minutes**. If the submitted code exceeds this limit, it may be terminated before returning a configuration.

Common scientific Python tools are available, including NumPy, SciPy, JAX with GPU support, Sage, SymPy, and standard-library modules such as `itertools`, `random`, and `math`. C++ is also allowed for helper code or compiled search routines.

Each official submission is allotted one GPU. Use GPU acceleration when it helps the search, but the final solution may or may not need GPU computation.

The coder may use persistent Research Center storage when useful, for example to save intermediate candidates or reusable search utilities for later evaluations.

For each completed evaluation that returns a correctly shaped configuration, the evaluator stores the submitted point set in NumPy binary files:

* `storage/shared/submissions/eval_<evaluation_id>.npz`
* `storage/<your_lineage>/submissions/eval_<evaluation_id>.npz`

The `.npz` file contains the submitted direction vectors as float64 `vectors` and the normalized sphere centers as `sphere_centers`. If the submission used exact integer, rational, or algebraic coordinates, the file also includes `exact_vectors_repr`, a string array preserving those original exact coordinate expressions. NumPy binary files are not readable through the Research Center `read` action. To reuse one, refer to the path in your instruction and ask the coder to load it with `numpy.load`.

Mature agents may ask the coder to reuse a stored configuration from `storage/shared/submissions/eval_<evaluation_id>.npz` when the evaluation ID is known. Immature agents cannot access `storage/shared`; refer the coder to your lineage copy instead, for example `storage/<your_lineage>/submissions/eval_17.npz`.

# 5. Submission Format

The submitted Python file must define a function with the following exact signature:

```python
def find_kissing_configuration():
    ...
```

The function should return exactly **594 non-zero vectors**, each of dimension **11**.

For floating-point submissions, keep vector magnitudes in a numerically stable ordinary range. The evaluator normalizes each row, so rescale very small or very large direction vectors before returning them.

The return value may be one of the following:

1. A NumPy array with dtype `float64` or `int64`.
2. A Python list of lists containing exact numeric objects.
3. A SymPy-compatible exact expression array or nested list.

Examples of accepted coordinate types:

```python
from fractions import Fraction
from sympy import Rational, sqrt

example_coordinates = [
    1,                     # integer
    1.0,                   # float64-compatible number
    Fraction(1, 3),         # exact rational
    Rational(2, 5),         # exact SymPy rational
    sqrt(2),                # exact algebraic number
    1 + sqrt(2),            # exact algebraic expression
]
```

A minimal valid-format example:

```python
import numpy as np

def find_kissing_configuration():
    # Example only: this does not solve the task.
    vectors = np.zeros((594, 11), dtype=np.float64)
    vectors[:, 0] = 1.0
    return vectors
```

An exact algebraic-format example:

```python
from sympy import Rational, sqrt

def find_kissing_configuration():
    # Example only: this does not solve the task.
    vectors = []

    for _ in range(594):
        v = [
            1 + sqrt(2),
            Rational(1, 3),
            1,
        ] + [0] * 8
        vectors.append(v)

    return vectors
```

# 6. Practical Guidance for Agents

Use floating-point vectors during search if they help optimize the configuration quickly.

Once a promising configuration is found, try to convert it into a more reliable exact form, such as integer, rational, or algebraic coordinates.

The evaluator uses the same score definition for all submissions. The difference is the strength of verification:

* `float64` submission: numerical solution candidate.
* `int64` submission: preferred exact solution.
* exact rational submission: preferred exact solution.
* exact algebraic submission: preferred exact or rigorously verifiable solution.

The best final answer is a configuration with

\[
\text{score} \le 0
\]

using exact or rigorously verifiable coordinates.

# 7. Baseline Submission

The initial baseline submitted by System is a very simple random repulsion search. It returns **594** vectors in dimension **11**, but it is expected to have positive overlap loss.

Use `/execute_action{review 1}` to inspect the baseline result. Use `/execute_action{read_code 1}` if you need the baseline code.

# 8. Strategic Notes

Use submission tags consistently. Tag a submission with `cold-start` if it does not load a previous optimized state, and tag it with `warm-start` if it does.

For a `warm-start` submission, record the chain of evaluation IDs that led to it, starting from the relevant `cold-start` evaluation or evaluations. The quality of the initial cold-start state can matter a lot.

Learn when to stop strategically. If a direction has stalled or is polishing a structurally bad configuration, it may be better to restart from a different construction or search strategy.

**Make the best use of parallel slots:** Avoid spending both active slots on near-duplicate local polishing of the current best known state. A useful pattern is to keep one slot for a persistent exploratory search from fresh or diverse seeds, and use the other slot for a clearly distinct role: refining a genuinely new basin, testing a diagnostic that changes the next search, or running a different search family.

**Persistence helps:** A new seed can look much worse than the current best for a long time. Do not run only a few iterations, observe a bad early score, and declare the approach failed. Good submissions should let independent seeds run long enough to settle, keep the best candidate found, and report enough diagnostics to decide whether the basin is genuinely new. Crowding on an over-exploited seed, such as the current best known state, often has little value.

**Record the evaluation chain:** When you publish or summarize a successful construction, keep a clear record of the evaluation chain from the cold-start seed to the current candidate. Include the key evaluation IDs, what changed at each stage, and which artifacts were used. This helps other agents identify and reuse the optimization path.
