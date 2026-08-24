## Archived Task Specification: Autocorrelation Problem 6.2

This is the earlier unhinted layout. The active template is two directories
above this one. Its deployable files have also been scrubbed of external
benchmarks and constructions so the archived layout remains safe to reuse.

This is the low-peak autocorrelation step-function construction task.

Spec version: **v1**.

The task follows the AlphaEvolve Problem 6.2 upper-bound setup. Submissions
provide a nonnegative equal-width step function on `(-1/4, 1/4)`. The primary
score is the normalized autocorrelation upper bound `R`; lower scores are
better.

The target is `R < 1.5032`.

## Setup

This is archived historical material and is intentionally not discovered by
`station init`. Use the active `alpha_evolve/autocorr_6-2` task instead.

## Optional Dependencies

Each official submission is configured for one station-managed GPU.

## External Evaluation

No external evaluation needed for this task. Validation is performed by the
station evaluator. Successful configurations are not persisted by default.

## Version Update Log

- **v1**: Initial equal-width nonnegative step-function task with lower-is-better
  normalized autocorrelation scoring and target upper bound `1.5032`.
