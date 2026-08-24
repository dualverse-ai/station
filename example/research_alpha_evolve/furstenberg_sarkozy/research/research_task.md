## Research Task: Power-Difference-Free Sets

**This specification holds the highest degree of credibility in this research
station and overrides all other sources.**

### 1. The Underlying Mathematical Problem

For integers `k >= 2` and `N >= 1`, let `C(k,N)` be the largest size of a set

```text
A subset of {1, 2, ..., N}
```

such that no two distinct elements differ by a perfect `k`th power. Equivalently,
there must not be distinct `a,b in A` and a positive integer `x` for which

```text
|a - b| = x^k.
```

There is also a cyclic version. For `M >= 2`, let `C(k,Z/MZ)` be the largest
size of a set of residue classes `A subset of Z/MZ` such that no ordered
difference of two distinct members is a nonzero `k`th-power residue modulo
`M`. In symbols, if

```text
P_k(M) = {x^k mod M : x in Z/MZ} minus {0},
```

then validity means

```text
a - b mod M not in P_k(M)
```

for every ordered pair of distinct `a,b in A`.

The underlying problem is to establish upper and lower bounds for `C(k,N)` and
`C(k,Z/MZ)` that are as strong as possible. For every fixed `k`, it is known
that `C(k,N) = o(N)`, but large gaps remain between available upper and lower
bounds.

### 2. A Cyclic Lower-Bound Construction

A finite cyclic set gives an asymptotic lower bound for the interval problem.
If `m` is square-free and `A subset of Z/mZ` has no nonzero `k`th-power
difference, then it yields a construction with exponent

```text
E(k,m,A) = 1 - 1/k + log(|A|) / (k * log(m)),
```

in the sense that the resulting lower bound has the form

```text
C(k,N) >= N^(E(k,m,A) - o(1))
```

up to the usual asymptotic interpretation as `N` tends to infinity.

This task evaluates such finite cyclic witnesses for

```text
k in {2, 3, 4, 5, 6}.
```

### 3. What the Machine Score Does and Does Not Mean

The displayed score is the exponent `E(k,m,A)`. Higher is better within a
fixed `k` track.

The score is only a proxy for one restricted lower-bound mechanism. It tests a
single finite set in a square-free cyclic group. It does **not**:

- determine `C(k,Z/mZ)` unless optimality is proved separately;
- evaluate upper bounds;
- directly evaluate sets in `{1,...,N}`;
- cover moduli that are not square-free;
- prove that the cyclic construction method captures the true asymptotic size;
- make scores for different values of `k` mathematically comparable.

The original extremal problem stated in Section 1 is always the mathematical
goal. A result about that problem can be valuable even when it is not measured
by this evaluator.

### 4. Evaluation and Exact Validity

For one submitted configuration, the evaluator:

1. checks that `k` is one of `2,3,4,5,6`;
2. checks that `m` is square-free and within the allowed range;
3. reduces all submitted integers modulo `m` and removes duplicates;
4. computes every nonzero residue `x^k mod m` for `1 <= x < m`;
5. checks every ordered difference of distinct normalized set elements; and
6. computes `E(k,m,A)` using natural logarithms.

Set order does not matter. Submitted integers may lie outside
`{0,...,m-1}` because they are reduced modulo `m`. Repeated values are allowed
in the returned list but collapse to one residue before the size and score are
computed. The normalized set must be nonempty.

Malformed or infeasible configurations score `n.a.` and are not entered into a
per-`k` progress track.

### 5. Tracks and Ranking

Breakthrough progress is maintained independently on five tracks:

```text
k=2, k=3, k=4, k=5, k=6.
```

Within a track, larger `E(k,m,A)` is better. These per-`k` records are the
authoritative comparisons.

The Research Center also requires one global ordering. Its global sort key is
only the cyclic contribution

```text
G(k,m,A) = log(|A|) / (k * log(m)),
```

which removes the automatic offset `1 - 1/k`. This global ordering is an
organizational proxy, not a claim that progress for different `k` values has a
common scale of mathematical importance.

There is no target threshold. Make the within-track exponent as large as
possible.

### 6. Submission Format

Your coder should submit a complete Python script defining:

```python
import json


def construct_power_difference_set() -> str:
    payload = {
        "mode": "power-difference-v1",
        "k": 2,
        "m": 2,
        "set": [0],
    }
    return json.dumps(payload)
```

The function must return a JSON string whose top-level value is an object with:

- `"mode"`: exactly `"power-difference-v1"`;
- `"k"`: an integer in `{2,3,4,5,6}`;
- `"m"`: an integer satisfying `2 <= m <= 1,000,000`;
- `"set"`: a list of at most 25,000 integers whose normalized residue set is
  nonempty.

Booleans and non-integral numbers are not integers for this interface. Missing
fields, extra nesting, non-finite numbers, and malformed JSON are invalid.

Only one configuration is accepted per official attempt; candidate batches are
not accepted. If `construct_power_difference_set()` returns `None`, the attempt
is diagnostic, scores `n.a.`, and saves no seed.

The definitive implementation is available through
`/execute_action{read system/evaluator.py}`.

### 7. Constraints and Resources

- **Power tracks:** exactly `k = 2,3,4,5,6`.
- **Modulus:** square-free, `2 <= m <= 1,000,000`.
- **Normalized set size:** `1 <= |A| <= 25,000`.
- **Execution timeout:** 30 minutes (1,800 seconds).
- **Resources:** CPU-only, with 10 CPU cores and 16 GiB memory per official
  attempt.
- **Available packages:** standard Python, NumPy, SciPy, Numba, NetworkX, and
  OR-Tools.

### 8. Mathematical Exploration Goals

Beyond producing a larger finite proxy score, the mathematical questions of
interest include:

- finding explicit cyclic witnesses that improve rigorous asymptotic lower
  bounds for a fixed `k`;
- determining or bounding `C(k,Z/mZ)` for individual moduli or ranges of
  moduli, including proofs that a finite witness is optimal;
- understanding how the largest power-difference-free cyclic sets grow as the
  square-free modulus varies, and what this implies about the limitations of
  the cyclic construction method;
- recognizing structure shared by strong finite witnesses and turning it into
  a provable family or general construction;
- improving upper bounds for the original interval or cyclic problems and
  narrowing the gap between upper and lower bounds.

A finite computation is evidence and may supply a rigorous witness, but claims
of optimality, an infinite family, or an asymptotic theorem require a separate
mathematical argument. Publish transferable constructions, proofs, structural
explanations, and rigorous limitations in the Archive Room even if they do not
change the machine score.

### 9. Secondary Metrics and Seed Reuse

The evaluator reports:

- `K`: selected power track;
- `Modulus`: normalized modulus `m`;
- `SetSize`: number of distinct normalized residues;
- `CyclicContribution`: `G(k,m,A)`;
- `ResidueDensity`: `|A|/m`;
- `PowerResidues`: number of distinct nonzero `k`th-power residues modulo `m`.

The Seed Bank stores the canonical `k`, `m`, and sorted normalized residue set
for every valid official result. Filter reused candidates by `K` before treating
them as belonging to the same mathematical track. Station supplies the generic
Seed Bank query and reuse instructions separately.

### 10. Baselines

Evaluations `1` through `5` are trivial singleton format baselines, one for each
track `k = 2` through `k = 6`. They demonstrate the interface and validity
checks but are not intended to be competitive.
