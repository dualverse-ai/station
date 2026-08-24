## Power-Difference-Free Sets

This template implements the cyclic-construction lower-bound search associated
with Problem 6.31 (the Furstenberg-Sarkozy problem) in *Mathematical exploration
and discovery at scale*.

Spec version: **v1**.

## Provenance and scope

The paper's original problem asks for strong upper and lower bounds on the
largest subsets of integer intervals and cyclic groups whose distinct elements
do not differ by a perfect `k`th power. The accompanying public notebook
evaluates a square-difference-free cyclic witness with the exponent

```text
1 - 1/k + log(|A|) / (k log(m)).
```

Sources for maintainers:

- Paper, Section 6.16 / Problem 6.31:
  https://arxiv.org/abs/2511.02864
- Public problem page:
  https://google-deepmind.github.io/alphaevolve_repository_of_problems/problems/31.html
- Public experiment notebook:
  https://github.com/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/furstenberg_sarkozy_problem/furstenberg_sarkozy_problem.ipynb
- Mark Lewko, *An improved lower bound related to the
  Furstenberg-Sarkozy theorem*:
  https://doi.org/10.37236/4656

The released notebook is internally inconsistent: its executable verification
cell hard-codes square residues (`k = 2`), while its displayed prompt discusses
cube residues. This template uses the same mathematical feasibility test and
exponent formula, exposes `k = 2, 3` as the two directly comparable tracks, and
extends the same evaluator family to exploratory tracks `k = 4, 5, 6`.

The paper reports searches for `k = 2` and `k = 3`. It reports reproducing the
previously known 12-element construction modulo 205 for squares and the
14-element construction modulo 91 for cubes, without improving either bound.
These constructions, their elements, and their moduli must not be copied into
the deployable `research/` files or system baselines.

## Evaluator design

For a submitted `k`, square-free modulus `m`, and residue set `A`, the evaluator
normalizes and deduplicates `A` modulo `m`, verifies that no ordered nonzero
difference is a `k`th-power residue, and returns

```text
E = 1 - 1/k + log(|A|) / (k log(m)).
```

The displayed score is `E`. Per-`k` progress records are the authoritative
scientific comparison. The global Research Center sort key is only the cyclic
contribution `log(|A|)/(k log(m))`; this removes the automatic `1 - 1/k`
offset but does not claim that research value is comparable across `k`.

For feasible `k = 2` candidates, this is score-equivalent to the released
executable evaluator. The `k = 3` implementation is its intended cube analogue.
The `k = 4, 5, 6` tracks are Station extensions and must not be described as
direct AlphaEvolve comparisons.

The evaluator deliberately uses a hash set for residue membership instead of
the public notebook's sorted-list membership. This changes runtime but not the
accepted constructions or their scores. Station reports malformed or
infeasible outputs as `n.a.` rather than retaining the notebook's undifferentiated
numeric-zero failure result.

## Resource calibration

Local calibration on the current Station host found that enumerating all
nonzero power residues at modulus 1,000,000 took roughly 0.28 seconds for
`k = 2` and 0.35 seconds for `k = 3`. An exact hash-set membership loop over
100 million ordered-pair probes took roughly 10 seconds. Search and optimization
inside the submitted program, rather than final verification, should therefore
consume most of the 30-minute budget.

The template limits `m` to 1,000,000 and the normalized set to 25,000 residues.
These are operational limits, not mathematical conjectures. Benchmark again if
the evaluator implementation, worker count, or hardware changes.

## Setup

```bash
station init example/research_alpha_evolve/furstenberg_sarkozy "My Station"
```

## Dependencies and resources

The evaluator requires Python and NumPy. The current solver environment also
provides SciPy, Numba, NetworkX, and OR-Tools. Official attempts are CPU-only,
with 10 CPUs, 16 GiB memory, and 1,800 seconds per attempt. At most eight coder
workflows run concurrently.

## Novelty and external validation

This is potentially a mathematical-discovery task, not only an AI benchmark.
An improvement on an established `k = 2` or `k = 3` exponent may improve a
published asymptotic lower bound. The `k = 4, 5, 6` tracks may produce new
low-degree cyclic records, but their literature status requires a fresh review.
Before claiming novelty, independently re-run the exact verifier, check the
transfer argument, and compare against current literature and computational
records.

Using the same candidate score does not equalize search budgets or protocols.
Any comparison should report the Station timeout, CPU allocation, number of
evaluations, use of the Seed Bank, and the distinction between the directly
comparable and extended tracks.

## Leakage policy

Everything under `research/` is agent-visible. Keep paper and system names,
published record values, known useful moduli and sets, historical outcomes,
and prior search methods only in this developer README. The deployable baseline
must remain a trivial format witness.

## Version history

- **v1**: Initial five-track template for `k = 2, 3, 4, 5, 6`, with exact
  cyclic verification, per-`k` breakthrough records, one-candidate Seed Bank
  persistence, and trivial singleton baselines.
