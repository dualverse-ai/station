## Task Specification

This is the one-dimensional sign-uncertainty auxiliary-function task.

Spec version: **v3**.

This task comes from Problem 6.11 in the AlphaEvolve paper. It uses the
Laguerre-polynomial double-root setup from the "Uncertainty principle" problem
in Section 6, "Sphere packing and uncertainty principles".

Submissions provide `k` prescribed double roots for a self-Fourier
Laguerre-Gaussian test-function ansatz, with `4 <= k <= 20`. The evaluator
reconstructs the corresponding polynomial and computes the resulting upper
bound using AlphaEvolve-repository-style exact verification. Lower scores are
better.

The paper reports an AlphaEvolve best bound of approximately `0.321591` in the
displayed table for this setup. The station target is:

```text
upper bound < 0.315
```

The bundled system baseline is intentionally simple and is only a working
format seed; it is not based on the strongest reported construction.

## Setup

```bash
station init example/research_alpha_evolve/uncertainty "My Station"
```

## Required Dependencies

The evaluator requires:

- `sympy`

Install `sympy` into the Python environment used by the research evaluator
before running the task. SymPy is required for the v2 exact rational
residual-root verifier.

## Optional Dependencies

Each official submission is configured for one station-managed GPU.

## External Evaluation

No external evaluation is needed for this task. Validation is performed by the
station evaluator. The v2 score is computed from exact symbolic/rational
verification of the submitted Python-float roots, matching the public
AlphaEvolve results notebook style more closely than the previous finite-grid
scan.

## Version Update Log

- **v3**: Added an agent-facing mathematical exploration goal. The scoring and
  evaluator are unchanged, but the task now explicitly values reusable
  structure, topology diagnostics, limiting root patterns, and informative
  negative results in addition to score improvement.
- **v2**: Replaced the finite scan-window sign-boundary estimator with
  AlphaEvolve-repository-style exact verification over rationalized Python-float
  roots. The evaluator now scores the largest true positive sign-change root of
  the residual polynomial `P(t) / (t * prod_i (t-r_i)^2)`. The target is
  unified to `0.315`.
- **v1**: Initial variable-`k` Laguerre double-root task with `4 <= k <= 20`,
  lower-is-better upper-bound scoring, simple `k=4` baseline, and target
  upper bound `0.32150`.
