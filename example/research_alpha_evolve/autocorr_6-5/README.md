## Task Specification: Autocorrelation Problem 6.5

This developer-facing bundle implements the Erdős minimum-overlap problem from
Section 6.2 of *Mathematical exploration and discovery at scale*. The paper
reports that AlphaEvolve produced a step-function upper bound of `0.380924`.

Spec version: **v3**.

The agent-facing task deliberately contains no provenance, reported
construction, or historical result. It asks agents to minimize the overlap
objective directly and begins with the constant half-density function.

## Setup

```bash
station init example/research_alpha_evolve/autocorr_6-5 "My Station"
```

## Optional Dependencies

Each official submission is configured for one station-managed GPU. The
evaluator itself uses NumPy and can run on CPU.

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
- **v1**: Initial equal-width minimum-overlap task with exact breakpoint
  cross-correlation scoring and a constant half-density baseline.
