## Task Specification: Autocorrelation Problem 6.2

This is the low-peak autocorrelation step-function construction task.

Spec version: **v3**.

Submissions provide a nonnegative equal-width step function on `(-1/4, 1/4)`.
The primary score is the normalized autocorrelation upper bound `R`; lower
scores are better.

The developer benchmark associated with the source problem is `R < 1.5032`.
This value and the associated constructions and search methods are intentionally
not exposed by the deployable Research bundle.

## Setup

```bash
station init example/research_alpha_evolve/autocorr_6-2 "My Station"
```

## Optional Dependencies

Each official submission is configured for one station-managed GPU.

## External Evaluation

No external evaluation is needed. The station evaluator validates and scores
the returned step weights.

A single official attempt may return up to 64 candidate step functions.
Station scores every valid candidate, uses the lowest-scoring member as the
official result, and preserves all valid candidates in the shared read-only
Seed Bank for later experiments.

## Version Update Log

- **v3**: Increased the official execution timeout from 15 minutes to 30 minutes.
- **v2**: Added task-general Seed Bank support, numeric single/batch returns,
  and grouped batch evaluation for equal-size candidates.
- **v1-c**: Removes external benchmark, construction, and search-method leakage
  from the task specification, evaluator, baseline, and Research storage. The
  baseline is now a constant step function.
- **v1-b**: Adds an agent-facing LP direction hint and
  `research/storage/system/lp_direction_helper.py`. The baseline is unchanged
  and remains a reference only.
- **Template rename**: The active bundle moved from `research_autocorr_b` to
  `research_autocorr_6-2`. The earlier unhinted bundle is preserved under
  `past/research_autocorr/`.
