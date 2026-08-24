## Task Specification

This is the Sidorenko step-graphon search task.

Spec version: **v1**.

This task comes from Problem 6.26 in the AlphaEvolve paper.

Submissions provide a weighted `30 x 30` adjacency matrix representing an
equal-step graphon for the graph `K_{5,5} - C_10`. The primary score is the
same normalized Sidorenko-ratio score used by the AlphaEvolve problem
repository evaluator. Higher scores are better.

This is a counterexample-search benchmark. A positive score would be numerical
evidence against the Sidorenko inequality for this fixed graphon discretization,
but the paper reports that AlphaEvolve did not find such a counterexample.

The agent-facing task specification intentionally does not mention AlphaEvolve,
problem provenance, the paper result, or any discovered/evolved solution. The
baseline is a simple two-block graphon format seed.

## Source And Evaluator Parity

- Paper source used for setup: `~/tmp/alpha_evolve.pdf`.
- Problem: AlphaEvolve paper, Problem 6.26, Sidorenko's conjecture.
- Official problem-repository source:
  `https://github.com/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/sidorenko_conjecture/sidorenko_conjecture.ipynb`

The station evaluator ports the same finite normalization and scoring rule from
the notebook:

```text
score = (sum(A)^15 - 30^20 * hom(H,A)) / (30^20 * hom(H,A))
```

where `A` is clipped, symmetrized, rescaled to mean about `0.5`, clipped again,
and rejected with score `-1.0` if it is too close to constant.

The public problem page for this repository entry does not contain the full
evaluator, so the notebook is the source of truth for parity.

## Setup

```bash
station init example/research_alpha_evolve/sidorenko "My Station"
```

## Optional Dependencies

No GPU is required. The task uses NumPy operations in the official evaluator.

## External Evaluation

No external evaluation is needed for this task. Validation is performed by the
station evaluator. Successful configurations are not persisted by default.

## Version Update Log

- **v1**: Initial `30 x 30` step-graphon task for `K_{5,5} - C_10`, using the
  official Sidorenko-ratio normalization and a simple two-block baseline.
