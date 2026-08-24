# Constant-Jacobian Polynomial Construction

This bundle is a controlled, exact-verification research benchmark. The
agent-facing payload intentionally contains no historical status, public
construction, discovery attribution, suggested ansatz, or conceptual solution
mechanism. Keep contemporary discussions and known witnesses out of
`research/`.

## Version

Current version: **research_jacobian v1**

## Task design

- Function-mode submission returning sparse rational polynomial data as JSON.
- Binary score: `1` only for a fully verified construction, otherwise `0`.
- Exact standard-library rational arithmetic; no numerical tolerances or LLM
  judging.
- A degree-12 and size-bounded contract keeps worst-case verification fast.
- The system baseline is a format-control identity map with expected score `0`.
- One managed GPU, ten CPUs, and a 30-minute official evaluation timeout.
- External Counter access is disabled by the task configuration.

The accepted regression probe is deliberately kept outside the bundle so it
cannot become agent-visible through task resources, evaluator source, or the
baseline.

## Maintainer verification

- A valid in-memory regression artifact receives score `1`; exact evaluation
  takes approximately 2 ms on the development machine.
- The checked-in identity-map baseline receives score `0` with
  `JacobianConstant=1` and `Collision=0`.
- A colliding map with nonconstant determinant receives score `0` with
  `JacobianConstant=0` and `Collision=1`.
- Malformed JSON, floating-point scalars, repeated witnesses, and over-degree
  terms receive score `0`.
- Sixty randomized determinant calculations were cross-checked against SymPy.
- A dense 256-term boundary probe completed exact determinant expansion in
  under two seconds on the development machine.

## Setup

```bash
station init example/research_misc/jacobian "My Station"
```

## Dependencies

The official verifier uses only Python's standard library. Coders may use the
scientific Python, CAS, compiler, CPU, and GPU tools available in the Station
environment while searching for a construction.
