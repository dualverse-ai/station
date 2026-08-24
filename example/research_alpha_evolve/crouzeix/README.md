## Developer Notes

This is the Crouzeix numerical-range ratio task.

Spec version: **v1**.

This task comes from Problem 6.25 in the AlphaEvolve paper.

Agent-facing text lives in `research/research_task.md`. Keep that file focused
on the mathematical objective, accepted format, scoring rule, baseline, and
exploration goals. Do not include paper provenance, reported constructions,
or maintainer notes there.

The public repository page for this problem currently has no evaluator Colab;
the Colab link is marked under construction:

```text
https://github.com/google-deepmind/alphaevolve_repository_of_problems/blob/main/problems/25.html
```

The public repository tree was also checked for a Crouzeix-specific experiment
notebook; none is present. If DeepMind later publishes such a notebook, prefer
porting its evaluator details over the paper-derived implementation below.

The evaluator follows the paper-described Crouzeix ratio:

```text
||p(A)||_op / sup_{z in W(A)} |p(z)|
```

For the denominator, it uses the Kippenhahn-Johnson boundary construction
described in the paper:

```text
H_theta = (exp(i theta) A + exp(-i theta) A*) / 2
z_theta = v_theta* A v_theta
```

where `v_theta` is a normalized top eigenvector of `H_theta`. The evaluator
samples this boundary densely and also samples polygon edges between adjacent
boundary samples to reduce denominator-underestimation artifacts near flat
boundary portions. Scores near the conjectural threshold receive a stricter
second denominator pass using four shifted theta grids to reduce grid-alignment
artifacts.

The paper reports that its search over variable matrix sizes matched the
known lower bound `2` but did not find an example above `2`. The station target
is deliberately harder:

```text
score >= 2.001
```

This is a counterexample-seeking benchmark. A robust score above `2` would be
mathematically significant, so high scores should be investigated for
denominator underestimation and numerical artifacts.

The task permits variable finite instances with `3 <= n <= 80` and polynomial
degree `1 <= m <= 60`. Matrix components are capped at absolute value `100`
and polynomial coefficient components at `1e6`; the polynomial is normalized
before scoring. Local runtime checks show that the largest strict denominator
pass is comfortably below the 10 minute station timeout on this checkout.

## Setup

```bash
station init example/research_alpha_evolve/crouzeix "My Station"
```

## Optional Dependencies

Each official submission is configured for one station-managed GPU.

## External Evaluation

No external evaluation is needed for this task. Validation is performed by the
station evaluator. Successful configurations are not persisted by default.

## Version Update Log

- **v1**: Initial variable-size Crouzeix numerical-range ratio task with
  paper-described Kippenhahn-Johnson denominator evaluation, simple diagonal
  baseline, and target score `2.001`.
