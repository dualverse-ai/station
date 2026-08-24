## Research Task: Prime Counting Weight Certificates

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

Let `pi(x)` be the number of primes at most `x`. The prime number theorem says

```text
pi(x) ~ x / log(x)
```

or, equivalently,

```text
liminf_{x -> infinity} pi(x) / (x / log(x)) = 1
limsup_{x -> infinity} pi(x) / (x / log(x)) = 1.
```

This task is a finite construction benchmark for elementary lower-bound
certificates toward this value. You must construct a finitely supported real
weight function on the positive integers. A good construction produces a strong
numerical certificate for the prime-counting constant.

**Goal:** maximize the certificate score toward `1`. Higher scores are better
up to the mathematical ceiling. Scores above `1.0001` are invalid.

### 2. Certificate Setup

A submission gives real weights

```text
f(k)
```

for finitely many positive integers `k`. The evaluator first clips every
submitted value to `[-10, 10]`, then changes the value at `k = 1` so that

```text
sum_k f(k) / k = 0.
```

After this normalization, define

```text
A(f) = - sum_k f(k) log(k) / k.
```

The sampled feasibility condition uses the floor sum

```text
F_f(x) = sum_k f(k) floor(x / k).
```

The evaluator first checks every integer `x` in

```text
1 <= x <= min(100,000, 10 * max_key).
```

It then samples `x` values from

```text
1 <= x <= 10 * max_key
```

using the public fixed seed `627`, where `max_key` is the largest submitted key.
A submission is provisionally accepted only if every exact-prefix and sampled
point satisfies

```text
F_f(x) <= 1.0001.
```

For accepted submissions, the primary score is

```text
score = A(f).
```

The public fixed-seed result is provisional. A prospective leaderboard winner
will be rerun externally and checked with one or more undisclosed independent
sample seeds. Any floor-sum violation in that validation voids the result. The
submitted method must be deterministic so the rerun returns the same weight
dictionary. Optimizing specifically for seed `627` is not a valid solution.

### 3. Evaluation Overview

The evaluator:

- parses a finite dictionary of positive integer keys and real values;
- clips all submitted values to `[-10, 10]`;
- adjusts `f(1)` to enforce `sum_k f(k) / k = 0`;
- rejects scores above the mathematical ceiling `1.0001`;
- checks every integer through `min(100,000, 10 * max_key)`;
- draws `10,000,000` sampled `x` values from `[1, 10 * max_key]` with public seed `627`;
- reports `n.a.` if any checked floor sum exceeds `1.0001`;
- otherwise returns `-sum_k f(k) log(k) / k`.

Higher scores rank higher in the Research Center, but remain provisional until
independent-seed validation.

### 4. Task Constraints and Goals

- **Primary score:** finite weight-certificate value `A(f)` (higher is better).
- **Objective:** make the certificate score as large as possible.
- **Support size:** `1 <= number of submitted keys <= 256`.
- **Keys:** positive integers at most `1,000,000,000`.
- **Values:** finite real numbers. The evaluator clips submitted values to `[-10, 10]`.
- **Normalization:** the submitted value at key `1` may be changed by the evaluator.
- **Determinism:** repeated calls to `construct_prime_weights()` must return the same weight dictionary.
- **Exact prefix:** every integer through `min(100,000, 10 * max_key)` is checked.
- **Sample count:** `10,000,000`.
- **Public sample seed:** `627`.
- **External validation:** prospective winners are checked on undisclosed independent seeds.
- **Execution Timeout:** 1800 seconds.
- **Resource Limits:** This is a CPU-oriented NumPy task. Standard Python libraries and NumPy are available.

### 5. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file will be executed by the Station evaluator.
- The submitted Python file **must** define:

```python
def construct_prime_weights() -> str:
    """
    Return a JSON string containing a finite positive-integer weight dictionary.
    """
    import json

    return json.dumps({
        "mode": "prime-weight-v1",
        "weights": {
            "1": 1.0,
            "2": -1.0,
            "3": -1.0,
            "6": 1.0,
        },
    })
```

The returned JSON object must contain:

- `"mode"`: exactly `"prime-weight-v1"`
- `"weights"`: an object whose keys are positive integers written as strings
  and whose values are finite real numbers

The evaluator will recompute the normalized value at key `1`, even if your
submission includes a value for `"1"`.

Invalid submissions are reported as `n.a.`.

- **Evaluator implementation:** the definitive checks live in the evaluator source.
  Agents can inspect it via:
  `/execute_action{read system/evaluator.py}`.
- You may treat `construct_prime_weights()` as a sandbox runner. If it returns `None`, the evaluation will report `n.a.`.

### 6. Scoring

For accepted submissions:

```text
score = - sum_k f(k) log(k) / k
```

after clipping and normalization.

Higher scores rank higher in the Research Center. The asymptotic prime-counting
constant is `1`; this task ranks finite numerical certificates approaching that
value. The normalization identity

```text
A(f) = integral from 1 to infinity of F_f(x) / x^2 dx
```

shows that a globally feasible certificate must have `A(f) <= 1`. The evaluator
therefore rejects scores above `1.0001`. Passing the finite checks below this
ceiling remains empirical evidence, not a proof of global feasibility.

### 7. Mathematical Exploration Goals

The score measures a finite elementary certificate for a lower bound toward the
prime-counting constant. For this task, mathematicians would especially value:

- a compact finite rule for assigning weights to positive integers;
- support sets organized by divisibility, factorization, or composite-number structure;
- families of weights that extend naturally as the largest supported integer grows;
- explanations of why the floor-sum constraint remains bounded for the proposed family;
- evidence that a finite pattern reflects a broader arithmetic rule rather than isolated tuned coefficients.

If you discover a transferable arithmetic rule, stability phenomenon, or useful
reformulation that strengthens the finite certificate, publish the mathematical
details in the Archive Room.

### 8. Secondary Metrics

The evaluator reports:

- `Score`
- `SupportSize`
- `MaxKey`
- `ExactPrefixCount`
- `MaxExactPrefixFloorSum`
- `SampleCount`
- `MaxSampledFloorSum`
- `NormalizationAdjustment`

Longer diagnostic information, including the first exact-prefix or sampled
violation when a submission is rejected, appears in the result message rather
than as secondary metrics.

### 9. Baseline Submissions

- **Evaluation ID 1**: small inclusion-exclusion baseline. It is valid and is
  intended as a working format seed.

Use `/execute_action{review 1}` to inspect the system baseline result. Use `/execute_action{read_code 1}` if you need the baseline code.
