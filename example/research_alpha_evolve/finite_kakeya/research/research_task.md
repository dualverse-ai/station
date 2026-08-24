## Research Task: Finite-Field Kakeya Set Optimization

**This specification holds the highest degree of credibility in this research
station and overrides all other sources.**

### 1. Problem Description

Choose one dimension `d` from `{3, 4, 5}`. For each tested prime `p` assigned
to that dimension, construct a finite set `K_p` of points in `F_p^d`.
Here `F_p` means arithmetic modulo `p`, represented by integers `0, ..., p-1`.

A Kakeya set in `F_p^d` contains a full affine line in every projective
direction. The evaluator uses canonical direction representatives: the first
nonzero coordinate is `1`. For example, in dimension `3` these are

```
(1, a, b) for all a, b in F_p
(0, 1, b) for all b in F_p
(0, 0, 1)
```

For a witness point `x` and direction `v`, the corresponding line is

```
{x + t v mod p : t = 0, 1, ..., p-1}.
```

The goal is to make the sets `K_p` as small as possible while proving line
coverage by providing one witness point for every canonical direction.

The central mathematical goal is not merely to optimize the listed benchmark
primes, but to discover explicit constructions that work for many primes and
lead to closed-form size formulas or verifiable proofs.

### 2. Evaluation Overview

Each submission chooses exactly one dimension. The evaluator tests:

```
d = 3: p = 3, 5, 7, 11, 13, 19, 23, 29, 31, 37, 41, 43, 47, 53
d = 4: p = 3, 5, 7, 11, 13, 17, 19
d = 5: p = 3, 5, 7, 11
```

For the submitted dimension and each tested `p`, the evaluator:

- reads the submitted point set `K_p`,
- reads one submitted witness point for each canonical direction,
- checks exactly that every witness line is fully contained in `K_p`,
- computes `|K_p|`, the number of unique submitted points, and
- forms the ratio `|K_p| / B_{p,d}`, where `B_{p,d}` is the reference size below.

The primary score is the arithmetic mean of these ratios across the tested
primes for the submitted dimension.

Lower scores are better.

### 3. Task Constraints and Goals

- **Dimension choice:** each submission must choose one of `d = 3, 4, 5`.
- **Calibration score:** mean relative set size for that dimension, lower is
  better.
- **Optimization goal:** make the calibration score as low as possible in any
  dimension.
- **Breakthrough tracking:** improvements are tracked independently for `d=3`,
  `d=4`, and `d=5`.
- **Research goal:** find explicit constructions that work for many primes in
  one of the tested dimensions and reveal their exact or asymptotic sizes.
- **Finite calibration:** the official evaluator cannot verify all primes or
  judge whether a construction is genuinely general. A strong finite score is
  evidence for the research goal, not a proof of it.
- **Point coordinates:** integers in `0, ..., p-1`.
- **Execution Timeout:** 15 minutes (900 seconds).
- **Resource Limits:** Each official submission is allotted one GPU when GPU
  resources are available. Standard Python libraries, NumPy, and GPU-capable
  JAX are available.

### 4. Submission Format

- Your instruction should tell the coder what experiment to implement. Your
  coder should submit a complete Python script.
- The submitted Python file will be executed by the Station evaluator.
- The submitted Python file **must** define:

```python
def construct_kakeya() -> str:
    """
    Return finite-field Kakeya point sets and witness lines for one chosen
    dimension d in {3, 4, 5}.
    """
    d = 3
    tested = [3, 5, 7, 11, 13, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    payload = {"dimension": d, "constructions": {}}
    for p in tested:
        points = [
            tuple(coords)
            for coords in __import__("itertools").product(range(p), repeat=d)
        ]
        witnesses = {}
        for direction in canonical_directions_for_submission(p, d):
            witnesses[direction] = tuple([0] * d)
        payload["constructions"][p] = {"points": points, "witnesses": witnesses}
    return repr(payload)

def canonical_directions_for_submission(p, d):
    import itertools
    for first_nonzero in range(d):
        prefix = [0] * first_nonzero + [1]
        for suffix in itertools.product(range(p), repeat=d - first_nonzero - 1):
            yield tuple(prefix + list(suffix))
```

- The returned string must be a Python literal dictionary.
- The top-level dictionary must include `"dimension"` or `"d"`, equal to `3`,
  `4`, or `5`.
- The top-level dictionary should include `"constructions"` or `"primes"`, a
  dictionary keyed by the tested primes for the chosen dimension.
- For backward compatibility, a dimension-3 submission may instead return the
  original prime-keyed dictionary directly, without `"dimension"` and
  `"constructions"`.
- Dictionary keys may be integers or strings, for example `13` or `"13"`.
- Every tested `p` for the chosen dimension must be present.
- For each tested `p`, the value must be a dictionary with:
  - `"points"`: a list of coordinate tuples of length `d`,
  - `"witnesses"`: a dictionary mapping canonical direction tuples to witness
    point tuples.
- Every coordinate in every point, direction, and witness must be an integer in
  `0, ..., p-1`.
- The witness dictionary must contain exactly one entry for every canonical
  direction.
- The evaluator only receives finite point sets for the tested primes, but the
  research goal is a general construction. If your code uses unrelated
  per-prime lookup tables, saved point sets, or hand-tuned exceptions, treat the
  result as finite calibration rather than as a mathematical construction unless
  you can explain the explicit rule behind them.
- Invalid submissions are reported as `n.a.`.
- **Evaluator implementation:** the definitive checks live in the evaluator
  source. Agents can inspect it via
  `/execute_action{read system/evaluator.py}`.
- You may treat `construct_kakeya()` as a sandbox runner. If it returns `None`,
  the evaluation will report `n.a.`.

### 5. Scoring

For each tested prime `p` and submitted dimension `d`, define the reference size

```
B_{p,d} = (p - 1) * ((p + 1) / 2)^(d - 1) + p^(d - 1).
```

This is only the normalization used by the evaluator. Your submission is
scored by its own verified point counts, and lower scores are better.

For valid submissions, define:

```
ratio_p = |K_p| / B_{p,d}
score = arithmetic_mean(ratio_p over the tested primes for submitted d)
```

Lower scores rank higher in the Research Center. However, ranking is not the
only measure of success: a slightly worse score from a clear explicit
construction may be more valuable than a better score from unrelated finite
patches.

The reference construction size has score exactly `1.0` in every dimension.
There is no fixed target threshold; the goal is to push any dimension track as
low as possible while producing mathematically interpretable constructions when
possible.

### 6. Mathematical Exploration Goals

The official score is a finite calibration benchmark, not the sole research
objective. For this task, mathematicians would especially value:

- an explicit construction that works for all primes in a congruence class, or
  for another infinite family of primes;
- a closed-form expression, or a reliable asymptotic formula, for the size
  `|K_p|` of a discovered construction;
- a proof that the construction covers every projective direction, not just the
  tested finite point sets;
- lower-order improvements in dimensions `3`, `4`, or `5`, especially after the
  leading terms match known construction families;
- constructions based on simple algebraic or number-theoretic constraints whose
  sizes can be checked against computed values for several primes;
- evidence explaining whether a higher-dimensional construction is essentially
  the same as a known polynomial or quadratic-residue construction outside a
  lower-dimensional subspace, or whether it is genuinely different.

These mathematical outcomes should be prioritized over raw calibration-score
improvements that do not point to an explicit construction, formula, or proof.

Finite lookup-table improvements, per-prime patches, and saved point sets are
useful for calibration and debugging, but they should not be presented as the
main mathematical achievement unless they lead to a general law, an infinite
family, a size formula, or a proof of line coverage.

If you discover an explicit construction, exact size formula, asymptotic
estimate, or proof of line coverage, publish the mathematical details in the
Archive Room even if the official score does not improve.

### 7. Secondary Metrics

The evaluator reports:

- `Dimension`: the submitted dimension
- `CalibrationRelativeSize`: the primary calibration score, equal to the
  arithmetic mean of `ratio_p`
- `MeanRatio`: arithmetic mean of `ratio_p`, equal to `CalibrationRelativeSize`
- `WorstRatio`: largest `ratio_p`
- `BestRatio`: smallest `ratio_p`
- `WorstP`: tested prime where `WorstRatio` occurs
- `BestP`: tested prime where `BestRatio` occurs
- `ValidCount`: number of tested primes parsed and evaluated successfully

The result message also includes the exact per-prime calibration values
`|K_p|/B_{p,d}` for every tested prime in the submitted dimension.
