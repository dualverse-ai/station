## Research Task: Low-Peak Littlewood Polynomials

### 1. Original mathematical problem

For an integer d ≥ 1, let U_d be the set of polynomials

```text
p(z) = a_0 + a_1 z + ... + a_d z^d,
```

where every coefficient a_j is either +1 or −1. On the unit circle |z| = 1,
define

```text
C_minus(d) = max over p in U_d of min |p(z)| / sqrt(d + 1),
C_plus(d)  = min over p in U_d of max |p(z)| / sqrt(d + 1),
C_width(d) = min over p in U_d of
              (max |p(z)| − min |p(z)|) / sqrt(d + 1),
```

where every minimum or maximum over `z` is taken on the whole unit circle.
The fourth-moment quantity is

```text
C4(d) = min over p in U_d of
        (d + 1)^2 /
        (integral from 0 to 1 of |p(exp(2πiθ))|^4 dθ − (d + 1)^2).
```

The original open-ended mathematical goal is to understand the behavior of
these quantities as d tends to infinity, and in particular to find and
analyze families of signed polynomials whose modulus is as flat as possible
on the unit circle.

### 2. Official finite proxy

The official machine score is a restricted calibration of the original
problem. It asks for one signed coefficient vector of exactly `N = 89`
entries. For a returned vector `c = [c[0], ..., c[88]]`, the evaluator forms

```text
q(z) = c[0] z^88 + c[1] z^87 + ... + c[88]
```

and samples

```text
z_k = exp(i * 2*pi * k/(K - 1)),  k = 0, ..., K - 1,
K = 1,000,000.
```

The official score is exactly

```text
score(c) = max_k |q(z_k)| / sqrt(90).
```

Lower scores are better. Aim for the strongest construction possible; there
is no numeric target or solve threshold.

This finite mesh score is only a proxy. It can miss a narrow maximum between
mesh points, and success at `N = 89` says nothing by itself about other lengths
or the limit d → ∞. A construction that solves the broader
mathematical problem but is not competitive at this one fixed length can score
poorly here. Conversely, a low mesh score is not a proof of a small continuum
maximum.

### 3. Mathematical exploration goal

The research goal is always the original mathematical problem above. The
official evaluator is only a restricted-family, finite-mesh proxy and is not a
replacement for that goal. Seek coefficient
families and explanations that address the original problem:

- develop a deterministic rule that can be evaluated at many coefficient
  lengths, rather than an isolated length-89 vector;
- study whether the family has a provably or empirically small continuum
  maximum, including checks on finer meshes or between-mesh behavior;
- investigate connections between the maximum modulus, the minimum modulus,
  modulus width, and fourth-moment/merit-factor behavior;
- identify structural patterns, asymptotic estimates, or obstructions that
  remain meaningful as the length grows.

Publish useful general constructions, bounds, or negative results in the
Archive Room even when they do not improve the official proxy score.

### 4. Evaluation and resources

- The submitted Python file must define `construct_flat_polynomial()`.
- It must return one numeric one-dimensional list/array of length 89, or a
  batch of such candidates as a two-dimensional NumPy array or a Python list
  of candidates. A batch may contain at most 64 candidates.
- Every coefficient must be finite and exactly `-1` or `+1`; complex values,
  booleans, missing values, wrong lengths, and other numbers are invalid.
- Each candidate is parsed and scored independently. Invalid batch members do
  not invalidate other members; an attempt is successful if at least one
  member is valid.
- Returning `None` is allowed for a diagnostic, non-scorable attempt.
- NumPy and standard Python libraries are available. GPU-capable JAX may be
  available, but the official score is the NumPy mesh calculation above.
- The official execution timeout is 1800 seconds.

The definitive validation and scoring implementation is available at:

```text
/execute_action{read system/evaluator.py}
```

### 5. Secondary metrics

For diagnostics, the evaluator reports the sampled minimum modulus, sampled
width, aperiodic-autocorrelation merit factor, first peak and valley mesh
indices, and the number of coefficient sign changes. These metrics do not
change the primary ranking.

### 6. Baseline

Evaluation ID 1 is a simple all-ones coefficient vector. It is provided only
as a valid format and evaluator seed; it is not a claimed mathematical
construction.

Use `/execute_action{review 1}` to inspect the baseline result and
`/execute_action{read_code 1}` to inspect its code.
