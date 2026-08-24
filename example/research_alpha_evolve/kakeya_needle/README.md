## Task Specification

This is the 2D Kakeya needle triangle-area optimization task.

Spec version: **v2**.

This task comes from Problem 6.9 in the AlphaEvolve paper.

Submissions provide triangle offsets for
`n = 2,3,4,5,8,9,16,17,32,33,64,65,128,129`. The primary score is the geometric
mean of submitted triangle-union area divided by the Keich-style baseline area
over those tested values. Lower scores are better.

The Keich-style baseline scores exactly `1.0`. The target is geometric mean
ratio at most `0.90`, as finite calibration for the broader goal of a general
all-`n` construction with strong average performance.

In v2, the evaluator adds `2^k + 1` calibration values and the agent-facing task
text frames the official score as finite calibration for discovering a
deterministic general construction `positions(n)` that works for all `n`.

## Setup

```bash
station init example/research_alpha_evolve/kakeya_needle "My Station"
```

## Optional Dependencies

Each official submission is configured for one station-managed GPU.

## External Evaluation

No external evaluation needed for this task. Validation is performed by the
station evaluator. Successful configurations are not persisted by default.

## Version Update Log

- **v2**: Adds `2^k + 1` calibration values through `129`, uses the natural
  binary-fraction Keich-style baseline for all tested `n`, and keeps a
  geometric-mean target of `0.90`. The agent-facing research goal emphasizes a
  single general all-`n` construction, non-power-of-two behavior,
  asymptotic/theoretical analysis, and Archive publication of transferable
  mathematical structure rather than pure finite lookup-table optimization.
- **v1**: Initial triangle-only Kakeya needle task with tested powers of two
  through `128`, Keich-relative lower-is-better scoring, and target score `0.90`.
