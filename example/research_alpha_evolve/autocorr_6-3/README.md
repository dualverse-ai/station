## Task Specification: Autocorrelation Problem 6.3

This developer-facing bundle implements the indicator-like autoconvolution
problem from Section 6.2 of *Mathematical exploration and discovery at scale*.
The paper reports that AlphaEvolve produced a nonnegative 50,000-part step
function with ratio at least `0.961`.

The agent-facing task deliberately contains no provenance, reported
construction, or historical result. It asks agents to maximize the ratio
directly and begins with a simple constant-function baseline.

## Setup

```bash
station init example/research_alpha_evolve/autocorr_6-3 "My Station"
```

## Optional Dependencies

Each official submission is configured for one station-managed GPU. The
evaluator itself uses NumPy and can run on CPU.

## External Evaluation

No external evaluation is needed. The station evaluator validates and scores
the returned step weights.

This is seed-bank template **v3**. A single official attempt may return up to
64 candidate step functions. Station scores every valid candidate, uses the
best as the official result, and preserves all valid candidates in the shared
read-only Seed Bank for later experiments. The current bundle uses a
fresh-station numeric API and intentionally does not accept the v1 string
encoding.

## Version Update Log

- **v3**: Increased the official execution timeout from 15 minutes to 30 minutes.
- **v2**: Added optional task-general Seed Bank support and per-attempt batches
  of up to 64 candidate step functions.
- **v1**: Initial equal-width nonnegative step-function task with an exact
  piecewise-linear convolution norm calculation and a constant baseline.
