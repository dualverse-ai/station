## Task Specification

This is the Hardy-Littlewood interval-union lower-bound task.

Spec version: **v2**.

This task comes from Problem 6.18 in the AlphaEvolve paper.

Submissions provide real sequences `y_1 < ... < y_n` and positive weights
`k_1, ..., k_n`. The evaluator forms all intervals from the finite
Hardy-Littlewood equivalent formulation and scores the ratio

```text
union_length / (2 * sum(k_i)).
```

Higher scores are better.

The simple equal-spacing construction approaches `1.5`. The AlphaEvolve paper
reports an unguided search result around `1.5080` and hinted pattern variants
up to about `1.533`. The station target is set slightly harder:

```text
score > 1.56
```

The sharp mathematical constant is known to be `(11 + sqrt(61)) / 12`, about
`1.5675208`, so this task is a construction-search benchmark rather than an
open constant-estimation problem.

## Setup

```bash
station init example/research_alpha_evolve/hardy_littlewood "My Station"
```

## Optional Dependencies

Each official submission is configured for one station-managed GPU.

## External Evaluation

No external evaluation is needed for this task. Validation is performed by the
station evaluator. Successful configurations are not persisted by default.

## Version Update Log

- **v2**: Removed agent-facing leakage of the known upper-bound value,
  human-solution provenance, numeric target threshold, guardrail wording, and
  theorem-specific naming from the station task bundle. The developer README
  keeps benchmark context for maintainers.
- **v1**: Initial finite interval-union Hardy-Littlewood task with
  higher-is-better lower-bound scoring, simple equal-spacing baseline, and
  target score `1.56`.
