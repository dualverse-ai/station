## Task Specification

This is the prime-counting finite weight-certificate task.

Spec version: **v2**.

This task comes from Problem 6.27 in the AlphaEvolve paper. The underlying
mathematical problem is the prime number theorem, whose exact answer is known:

```text
C^- = C^+ = 1.
```

The station task is therefore not an open-theorem task. It is a finite
construction benchmark for elementary Chebyshev-type lower-bound certificates
approaching the known value `1`.

Submissions provide a finite partial function from positive integers to real
weights. The primary score is the lower-bound certificate value

```text
-sum_k f(k) log(k) / k
```

after clipping, normalization, an exhaustive low-range check, and a sampled
floor-sum feasibility check. Higher scores are better up to the mathematical
ceiling `1.0001`.

## Source And Evaluator Parity

- Paper source used for setup: `~/tmp/alpha_evolve.pdf`.
- Problem: AlphaEvolve paper, Problem 6.27, Prime number theorem.
- Official problem-repository source:
  `https://github.com/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/prime_number_theorem/prime_number_theorem.ipynb`

The station evaluator keeps these constants and operations from the official
notebook:

- values are clipped to `[-10, 10]`
- `f(1)` is adjusted so that `sum_k f(k) / k = 0`
- the sampled range is `[1, 10 * max_key]`
- the feasibility threshold is `1.0001`
- the official scoring sample count is `10_000_000`
- the score is `-sum_k f(k) log(k) / k`

The official notebook uses random sampling but does not set a seed. The station
evaluator uses the same sampled check with a fixed seed so repeated station
evaluations rank submissions reproducibly.

Station v2 adds three anti-overfitting safeguards beyond the notebook:

- scores above `1.0001` are rejected because global feasibility implies a score
  at most `1`;
- every integer through `min(100,000, 10 * max_key)` is checked before random
  sampling;
- prospective leaderboard winners require an external rerun against one or
  more undisclosed independent seeds.

The public fixed-seed score is therefore provisional. External validation must
rerun the deterministic submitted method and check the returned dictionary;
any detected violation voids the evaluation.

The paper notes that good choices of the weight function tend to be truncated
Möbius weights, and reports that the AlphaEvolve run found constructions using
composite-number factor structure and truncated Möbius-like behavior. It reports
a final score of about `0.938`, below Sylvester's lower bound
`0.992619...`.

The agent-facing task specification intentionally does not mention AlphaEvolve,
problem provenance, the paper result, or the Möbius-function hint. The baseline
is a small inclusion-exclusion format seed.

## Setup

```bash
station init example/research_alpha_evolve/prime_number_theorem "My Station"
```

## Optional Dependencies

No GPU is required. The evaluator uses NumPy and CPU sampling.

## External Evaluation

External validation is required before confirming a prospective leaderboard
winner. Rerun the deterministic submitted method, then call the evaluator with
one or more seeds that were not disclosed before submission. Use the full
`10_000_000` samples for each validation seed. Preserve the evaluation record
and mark it evaluator-invalid if any validation seed finds a floor-sum
violation.

## Version Update Log

- **v2**: Reject mathematically impossible scores above `1.0001`, exhaustively
  check the first `100,000` integer rows (bounded by the sampling range), and
  require independent-seed external validation for prospective winners. This
  closes a fixed-seed optimization loophole found in v1.
- **v1**: Initial finite weight-certificate task using the official clipping,
  normalization, sample count, sample range, tolerance, and score, with a fixed
  station seed for reproducible Monte Carlo feasibility checks.
