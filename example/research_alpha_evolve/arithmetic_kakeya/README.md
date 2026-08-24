## Task Specification: Arithmetic Kakeya Entropy Witnesses

This developer-facing bundle implements Problem 6.30, the arithmetic Kakeya
conjecture, from *Mathematical exploration and discovery at scale*. The
agent-facing Research payload deliberately contains no AlphaEvolve provenance,
reported scores, discovered construction, or search-method information.

Spec version: **v1**.

The mathematical problem defines projection-entropy constants
`C({r_1, ..., r_k}; r_inf)`. The scored benchmark focuses on the slope tuple
used by the released Problem 30 verification code and searches for a finite
joint distribution with a large entropy ratio.

## Source And Comparison Evaluator

Sources inspected for this bundle:

- Paper: `~/tmp/alpha_evolve.pdf`, Section 6.15, Problem 6.30.
- Repository: `google-deepmind/alphaevolve_repository_of_problems`.
- Problem page: `problems/30.html`.
- Problem-linked notebook:
  `experiments/arithmetic_kakeya_conjecture/arithmetic_kakeya_conjecture.ipynb`.
- Repository `main` revision inspected on 2026-08-08:
  `8f447457957deac61e28bf1676746f0753b3b2f8`.

The repository also contains a separate
`experiments/arithmetic_kakeya/arithmetic_kakeya.ipynb`. This bundle uses the
notebook linked by the Problem 30 page so the comparison source is explicit
and reproducible.

For `slope_tuple = (1, 2)`, the released verifier:

1. accepts a square joint-probability matrix;
2. clips negative entries to zero and renormalizes;
3. consolidates all positive atoms sharing a value of `X - Y` onto the
   highest-probability atom for that difference, using row-major order to break
   exact ties; and
4. scores

```text
H(X,Y) / max(H(X), H(Y), H(X + 2Y)).
```

After consolidation, `X - Y` is injective on the support, so
`H(X,Y) = H(X - Y)`. The Station evaluator reproduces this preprocessing and
score for every valid candidate. It implements entropy directly rather than
depending on SciPy; this is numerically equivalent to the released loop-based
entropy calculation.

## Evaluator Limitation

The released proxy is not strictly the full fixed-slope quantity

```text
H(X - Y) / max(H(X), H(Y), H(X + Y), H(X + 2Y))
```

associated with `C({0, 1, 2, infinity}; -1)`, because its denominator omits
`H(X + Y)`. The primary Station score intentionally preserves the released
proxy for evaluator-level comparison. The evaluator additionally reports
`FullRatio`, which restores the omitted projection and is a genuine lower-bound
witness for the stated fixed-slope constant within the finite grid family.

The paper reports an AlphaEvolve lower-bound improvement for the four-slope
case and describes qualitative structure in the discovered distributions.
Those benchmark and construction details are leakage-sensitive and must remain
in this developer README; they must not be copied into `research/`, prompts,
baselines, evaluator comments, or system resources. The reported comparison
value is approximately `1.668`. The paper further notes that the observed
distributions resembled discrete Gaussians and motivated later theoretical
work. Neither fact is exposed to station agents.

## Runtime And Fair-Comparison Note

The released search prompt used a 1000-second budget. This Station bundle uses
an **1800-second** official execution timeout by explicit project decision.
Accordingly, a paper may accurately state that the same scoring evaluator is
used, but should not claim an identical compute or wall-time budget.

A single official attempt may return up to 64 matrices. Station scores each
candidate independently, selects the highest proxy score as the official
result, and stores all valid canonicalized matrices in the Seed Bank.

## Setup

```bash
station init example/research_alpha_evolve/arithmetic_kakeya "My Station"
```

## Dependencies And Resources

- The evaluator requires NumPy.
- Each official submission is configured for one station-managed GPU and ten
  CPUs so submitted search code may use NumPy or GPU-capable JAX.
- The evaluator itself is deterministic and CPU-only.
- With 64 float64 candidates, the raw matrix payload is about 3.4 MiB.

Before deployment, benchmark a full 64-candidate valid batch on the production
environment and reduce `RESEARCH_EVAL_MAX_PARALLEL_WORKERS` if eight concurrent
GPU allocations exceed the host capacity.

## Baseline

Evaluation ID 1 returns the uniform `83 x 83` matrix. It is intentionally a
simple interface and evaluator-path check, not a competitive construction.
With NumPy 2.2.6, the independently checked baseline proxy and full ratios are
both `1.087576917428` (165 canonical support atoms and 6724 consolidated
atoms). No benchmark value is embedded in agent-facing files.

## Licensing

The upstream repository states that its software is Apache-2.0 licensed and
other materials are CC-BY 4.0. Preserve applicable notices and attribution in
developer-facing release material.
