## Developer Notes

This is the Ovals eigenvalue-search task.

Spec version: **v2**.

This task comes from Problem 6.19 in the AlphaEvolve paper.

Agent-facing text lives in `research/research_task.md`. Keep that file focused
on the mathematical objective, accepted format, scoring rule, and baseline. Do
not include paper provenance, reported results, or maintainer notes there.

Submissions provide three equal-length real arrays representing a periodic
planar curve and test function. The primary score is the Rayleigh quotient;
lower scores are better.

The bundled baseline is a simple ellipse with a constant test function. The
target is `score <= 1.0`.

## Setup

```bash
station init example/research_alpha_evolve/ovals "My Station"
```

## Optional Dependencies

Each official submission is configured for one station-managed GPU.

## External Evaluation

No external evaluation needed for this task. Validation is performed by the
station evaluator. Successful configurations are not persisted by default.

## Version Update Log

- **v2**: Reframed the agent-facing task around calibration, rigorous evidence,
  and mathematical structure near the `1` boundary. The numerical score remains
  the Research Center benchmark, but it is not presented as mathematical
  certification.
- **v1**: Initial Ovals task with spline-based curve evaluation, natural
  lower-is-better scoring, simple-ellipse baseline, and `score <= 1.0`
  target.
