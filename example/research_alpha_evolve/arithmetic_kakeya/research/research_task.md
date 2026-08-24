## Research Task: Arithmetic Kakeya Entropy Witnesses

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

For each slope `r` in `R union {infinity}`, define the projection

```text
pi_r(a,b) = a + r b       when r is finite,
pi_infinity(a,b) = b.
```

Given distinct slopes `r_1, ..., r_k, r_inf`, let
`C({r_1, ..., r_k}; r_inf)` be the smallest constant such that every pair of
finite-valued discrete random variables `X,Y`, which need not be independent,
satisfies

```text
H(pi_r_inf(X,Y))
    <= C({r_1, ..., r_k}; r_inf)
       * max_i H(pi_r_i(X,Y)).
```

Here Shannon entropy is

```text
H(Z) = -sum_z P(Z=z) log_2(P(Z=z)).
```

Changing the logarithm base does not change an entropy ratio. Rational or
integer values for `X,Y` suffice. Projective changes of slope can often be used
to normalize the distinguished slope, and one may study witnesses for which
the distinguished projection is injective on the joint support.

The arithmetic Kakeya conjecture asks whether suitable finite sets of
comparison slopes can make these constants arbitrarily close to `1`. The
mathematical goal of this task is to understand these constants, their
extremizing distributions, and their dependence on the slopes as sharply as
possible.

An explicit joint distribution gives a rigorous lower bound for a fixed
constant by taking the corresponding entropy ratio.

### 2. Fixed-Slope Scored Benchmark

The finite benchmark studies the comparison slopes

```text
0, 1, 2, infinity
```

against the distinguished slope `-1`. For a joint distribution `P`, the
original fixed-slope witness ratio is

```text
FullRatio(P) = H(X - Y)
               / max(H(X), H(Y), H(X + Y), H(X + 2Y)).
```

Make mathematically valid witness ratios as large as possible. The official
primary score is a restricted proxy described below; it is useful for ranking
finite-grid searches but does not replace the original problem.

### 3. Finite-Grid Representation

A candidate is an `83 x 83` real matrix `P`. Before evaluator preprocessing,
`P[i,j]` is the weight assigned to

```text
X = i + 1,  Y = j + 1,
```

for zero-based Python indices `i,j`.

The evaluator performs these steps in order:

1. Clip every negative entry to zero.
2. Require positive total mass and divide by the total so the matrix sums to
   one.
3. For each integer difference `d = X - Y`, inspect all positive atoms with
   that difference. Keep the atom having the largest probability, move the
   probability of every other such atom onto it, and set the other atoms to
   zero. Exact probability ties are resolved by the first atom in row-major
   order.
4. Renormalize after each difference group that required consolidation.

The resulting canonical distribution has at most one positive atom for each
value of `X-Y`. Consequently, `X-Y` is injective on its support and

```text
H(X-Y) = H(X,Y).
```

The canonical distribution, rather than the raw submitted matrix, is scored
and stored in the Seed Bank.

### 4. Primary Scoring And Proxy Limitation

For a canonical distribution, the primary score is

```text
ProxyRatio(P) = H(X,Y) / max(H(X), H(Y), H(X + 2Y)).
```

Higher scores rank higher in the Research Center. There is no numeric target;
make the score as large as possible while seeking mathematical understanding
of the original problem.

This proxy omits `H(X+Y)` from the denominator. It therefore evaluates a
restricted quantity and can be larger than `FullRatio(P)`. A high primary score
alone is not a lower bound of the same size for
`C({0,1,2,infinity}; -1)`, is not evidence for an upper bound, and does not
prove the arithmetic Kakeya conjecture. The separately reported `FullRatio`
is the valid fixed-slope lower-bound witness supplied by the canonical matrix.

### 5. Submission Format

Your instruction should tell the coder what one focused experiment to
implement. The submitted Python file must define:

```python
def construct_arithmetic_kakeya():
    """Return one 83 x 83 matrix, or a batch of such matrices."""
    return [[1.0 for _ in range(83)] for _ in range(83)]
```

The function may return:

- one numeric Python list or NumPy array with shape `(83, 83)`;
- a NumPy array with shape `(B, 83, 83)`; or
- a Python list of `B` candidate matrices;

where `1 <= B <= 64`. Every value must be a finite real number; booleans and
complex values are invalid. Negative finite values are accepted and clipped as
specified above. A candidate whose clipped entries have zero total mass is
invalid. Invalid candidates report `n.a.` and do not invalidate other members
of the same batch.

If `construct_arithmetic_kakeya()` returns `None`, the attempt is non-scorable
and reports `n.a.`.

The definitive implementation is available through
`/execute_action{read system/evaluator.py}`.

### 6. Constraints And Resources

- **Matrix shape:** exactly `(83, 83)` for each candidate.
- **Batch size:** at most 64 candidates.
- **Execution timeout:** 30 minutes (1800 seconds).
- **Resources:** each official attempt is allotted one GPU when station GPU
  resources are available and ten CPUs. Standard Python libraries, NumPy, and
  GPU-capable JAX are available.
- **Entropy arithmetic:** deterministic float64 arithmetic with base-2
  logarithms.

### 7. Mathematical Exploration Goals

The machine score is only a finite restricted proxy. Progress on the original
problem may also come from:

- coherent parameterized witness families that work across several rational
  slope sets rather than isolated lookup tables;
- exact or rationalized versions of numerical witnesses with independently
  checkable entropy calculations;
- structural explanations for the support and probability patterns of strong
  witnesses;
- conjectures and rigorous asymptotics for how the constants vary with rational
  slope parameters;
- upper-bound arguments, obstructions, or reductions bearing directly on the
  original constants, even though the machine evaluator cannot score them; and
- proofs that a proposed family has the claimed projection entropies or that a
  natural restricted family cannot improve further.

Publish transferable constructions, proofs, asymptotic laws, or negative
results in the Archive Room even when they do not improve the primary proxy
score. Always distinguish a finite numerical observation, a valid lower-bound
witness through `FullRatio`, and a theorem about the original constants.

### 8. Secondary Metrics

The evaluator reports:

- `FullRatio`: the original fixed-slope ratio including `H(X+Y)`;
- `HMinus`: `H(X-Y)` for the canonical distribution;
- `HMaxProxy`: the denominator used by the primary proxy;
- `HXPlusY`: the projection entropy omitted from the proxy denominator;
- `Support`: the number of positive atoms after canonicalization; and
- `Collapsed`: the number of positive atoms removed by consolidation.

### 9. Baseline Submission

- **Evaluation ID 1:** a uniform `83 x 83` matrix supplied only as a simple
  format and evaluator-path baseline.

Use `/execute_action{review 1}` to inspect its result and
`/execute_action{read_code 1}` to inspect its code.

