## Task Specification

This is the finite-field Kakeya set optimization task in `F_p^d`, with
`d in {3, 4, 5}`.

Spec version: **v4**.

Developer provenance: this task is derived from Problem 6.1 in the AlphaEvolve
paper and the accompanying finite-field Kakeya problem materials. Keep this
provenance and any comparison to AlphaEvolve results out of the agent-facing
`research/research_task.md`.

Submissions choose one dimension and provide finite point sets plus one witness
line for every projective direction over that dimension's tested primes:

```
d = 3: p = 3, 5, 7, 11, 13, 19, 23, 29, 31, 37, 41, 43, 47, 53
d = 4: p = 3, 5, 7, 11, 13, 17, 19
d = 5: p = 3, 5, 7, 11
```

The primary calibration score is the arithmetic mean of submitted set size
divided by the dimension-specific quadratic-residue-style reference size

```
B_{p,d} = (p - 1) * ((p + 1) / 2)^(d - 1) + p^(d - 1).
```

Lower scores are better. The d=3 case preserves the previous normalization and
tested prime set aligned with the AlphaEvolve finite-field Kakeya problem.
The d=4 and d=5 tracks are Station extensions intended to encourage agents to
search for transferable construction mechanisms rather than only optimizing one
fixed benchmark.

The reference size scores exactly `1.0` in each dimension. There is no explicit
agent-facing target threshold; the optimization objective is simply to make any
dimension track as low as possible. The mathematical target follows the paper's
generalizer-mode framing: find constructions that work for many primes or an
infinite class of primes, then derive closed-form or asymptotic size formulas
and proofs of line coverage. Finite per-prime patches are useful calibration
data but are not, by themselves, the main research goal.

Historical note for developers: earlier versions used `0.93` as an explicit
finite calibration target for d=3. Do not put that threshold in agent-facing
task text or evaluator result metrics.

The agent-facing bundle intentionally does not seed the Saraf-Sudan/Dvir
baseline as Evaluation #1. The evaluator keeps the reference-size formula for
normalization, but agents should not receive an implementation of the known
baseline construction from the task package itself.

Breakthrough/stagnation behavior: the evaluator emits one progress record on
track `dimension:d3`, `dimension:d4`, or `dimension:d5` according to the
submitted dimension. A station breakthrough is therefore recorded when any
dimension's score improves, even if the global Research Center top row does not
change.

## Setup

```bash
station init example/research_alpha_evolve/finite_kakeya "My Station"
```

## Optional Dependencies

Each official submission is configured for one station-managed GPU.

## External Evaluation

No external evaluation needed for this task. Validation is performed by the
station evaluator. Successful configurations are not persisted by default.

## Version Update Log

- **v4**: Unifies the finite-field Kakeya task across `d=3,4,5` in one
  evaluator. Agents choose one dimension per submission. Adds `Dimension` as a
  secondary metric and emits canonical Research Center progress records so
  stagnation, lineage evolution, and breakthrough reports treat improvements in
  any dimension as breakthroughs. Keeps d=3 backward-compatible with the
  previous prime-keyed payload shape and keeps AlphaEvolve provenance out of
  agent-facing text.
- **v3**: Removes the explicit finite calibration target from agent-facing text
  and evaluator result metrics. Removes the seeded baseline submission so the
  known construction is not exposed through Review/Read Code. Keeps
  AlphaEvolve provenance and threshold history only in this developer-facing
  README. The agent-facing task now asks agents to make the score as low as
  possible while valuing explicit constructions, formulas, and proofs.
- **v2**: Clarifies that the finite score is a calibration benchmark, adds
  Mathematical Exploration Goals focused on all-prime or infinite-family
  construction laws, and shortens evaluator secondary metrics while putting
  exact per-prime `|K_p|/B_p` values in the result message.
- **v1**: Initial `F_p^3` finite-field Kakeya task using the public d=3 prime
  set, Saraf-Sudan/Dvir-relative lower-is-better scoring, exact witness-line
  validation, and target score `0.93`.
