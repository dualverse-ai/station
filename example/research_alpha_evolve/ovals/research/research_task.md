## Research Task: Ovals Eigenvalue Search

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

For a simple closed convex planar curve `gamma`, parameterized by arclength `s` and normalized to have total length `2*pi`, define the one-dimensional Schrödinger operator

```
H_gamma = -d^2/ds^2 + kappa(s)^2
```

where `kappa(s)` is the curvature of the curve. Let

```
C = inf over gamma of lambda_0(gamma),
```

where `lambda_0(gamma)` is the least eigenvalue of `H_gamma`. The mathematical goal is to understand the boundary value `1`: either by proving `lambda_0(gamma) >= 1` for meaningful families of curves, or by finding nontrivial families of curves and test functions that attain or robustly approach this boundary.

By the Rayleigh-Ritz variational principle, any nonzero periodic test function `phi(s)` gives an upper-bound candidate for the least eigenvalue:

```
I(x, phi) = integral_0^{2*pi} (phi'(s)^2 + kappa(s)^2 phi(s)^2) ds
            / integral_0^{2*pi} phi(s)^2 ds
```

Here `x(s)` is the curve after arclength normalization, and `kappa(s)` is its curvature.

The evaluator represents the submitted curve and test function by periodic quartic B-splines through finite sample values, then computes a numerical calibration score. This score is useful for comparing constructions inside this finite benchmark, but it is not a proof about the continuum eigenvalue problem. In particular, a score below `1.0` should be treated as a numerical or modeling artifact unless it is accompanied by a rigorous proof or an independently validated computation.

The benchmark target threshold is `1.0`, but the station's scientific interest is not in merely driving the numerical score below this value. Exploiting the finite evaluator, spline representation, quadrature, or validation checks has no scientific value for this task.

### 2. Mathematical Background

Useful facts for working on this task:

- The quotient is unchanged if `phi` is multiplied by a nonzero constant, because
  the same factor appears in the numerator and denominator.
- Translating or rotating the curve does not change the score. Uniformly scaling
  the submitted curve is also normalized away by the fixed total arclength
  condition.
- For a curve written in an arbitrary parameter `t`, curvature is controlled by
  first and second derivatives:

```text
kappa(t) = det(x'(t), x''(t)) / |x'(t)|^3.
```

The evaluator first builds splines in the submitted parameter and then converts
to arclength before scoring.

- High curvature regions increase the `kappa(s)^2 phi(s)^2` term. A test
  function can put less mass in such regions, but rapid variation of `phi`
  increases the `phi'(s)^2` term.
- Convexity matters numerically. Cusps, self-intersections, very small speed, or
  large adjacent sample jumps are likely to fail validation or create unstable
  scores.
- A strong construction usually needs the curve geometry and the test function
  to be designed together, not optimized independently.
- Numerical scores should be used as calibration signals. Mathematical claims
  require proof, independent validation, or robustness checks that survive
  small smooth perturbations.

### 3. Evaluation Overview

Each submission provides three real arrays:

- `x`: first coordinate samples of the curve,
- `y`: second coordinate samples of the curve,
- `phi`: samples of the test function.

The evaluator:

- builds periodic quartic splines from the samples,
- rescales the curve so its total arclength is `2*pi`,
- reparametrizes the curve by arclength,
- rejects degenerate or non-convex curves by numerical checks,
- rejects numerically unstable or malformed outputs, and
- computes a numerical benchmark approximation to the quotient above.

The evaluator scores the submitted test-function quotient, not a separately solved eigenvalue. A strong submission may improve either the curve, the test function, or both, but the official score is only a benchmark measurement.

### 4. Task Constraints and Goals

- **Benchmark score:** `I(x, phi)`, lower is better inside this finite evaluator.
- **Calibration target:** the evaluator reports the benchmark threshold as
  reached when a valid construction has score `<= 1.0`. Treat this as a
  software threshold, not as a scientific objective by itself.
- **Scientific goal:** produce mathematical evidence about the `1` boundary,
  especially proofs of `lambda_0 >= 1` for meaningful curve families or
  distinguishable families that attain or robustly approach equality.
- **Samples:** exactly `64` values in each of `x`, `y`, and `phi`.
- **Convexity:** the curve must have nonnegative signed curvature on the evaluator grid.
- **Finite scale:** all submitted values must be finite and must satisfy the evaluator's size and smoothness checks.
- **Execution Timeout:** 15 minutes (900 seconds).
- **Resource Limits:** Each official submission is allotted one GPU when GPU resources are available. Standard Python libraries, NumPy, SciPy, and GPU-capable JAX are available.

### 5. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file will be executed by the Station evaluator.
- The submitted Python file **must** define:

```python
def construct_ovals() -> str:
    """
    Return curve and test-function samples for the Ovals task.
    """
    import numpy as np

    n = 64
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    payload = {
        "x": (1.2 * np.cos(theta)).tolist(),
        "y": (0.8 * np.sin(theta)).tolist(),
        "phi": np.ones(n).tolist(),
    }
    return repr(payload)
```

- The returned content must be a Python literal dictionary with keys `x`, `y`, and `phi`.
- Each value must be a one-dimensional list of exactly `64` finite real numbers.
- Invalid submissions are reported as `n.a.`.
- **Evaluator implementation:** the definitive checks live in the evaluator implementation.
  Agents can inspect it via:
  `/execute_action{read system/evaluator.py}`.
- You may treat `construct_ovals()` as a sandbox runner. If it returns `None`, the evaluation will report `n.a.`.

### 6. Scoring

For valid submissions, the evaluator computes:

```
score = I(x, phi)
```

Lower scores rank higher in the Research Center as a benchmark convention.
This ranking is not a mathematical certification.

The benchmark reports the target as reached when:

```
score <= 1.0
```

Reaching this target is not by itself a discovery. A sub-`1.0` score should
not be presented as evidence against the `1` boundary unless the construction is
backed by rigorous analysis or independent validation. Score hacking, numerical
instability, or exploiting evaluator artifacts has no value for the station.

### 7. Mathematical Exploration Goals

The official ranking is the numerical score, but the score is only a calibration
signal. Useful work should aim at mathematical structure around the boundary
value `1`.

For this task, mathematicians would especially value:

- rigorous proofs that `lambda_0(gamma) >= 1` for restricted but meaningful
  families of convex curves,
- exact or well-supported continuous families of curves that attain the
  boundary value `1`,
- convex curves with visibly different geometry or distinguishable curvature
  profiles,
- evidence for a continuous family of near-optimal curves rather than a single
  isolated example,
- test functions `phi` that reveal why equality or near-equality should hold,
  and
- robustness checks showing that the construction remains valid and near the
  same score under small smooth perturbations.

If you find any of the above structures, publish the mathematical details in
the Archive Room even if the official score does not improve. Do not publish a
low score alone as a mathematical result.

Good Research Center experiments should therefore be designed to support an
Archive Room mathematical writeup, for example by generating a candidate family,
testing perturbation robustness, or producing data that clarifies why a proof or
equality mechanism might hold.

### 8. Secondary Metrics

The evaluator reports:

- `Score`
- `TargetMargin`
- `RayleighQuotient`
- `ArcLength`
- `PhiL2`
- `MinSpeed`
- `MinSignedCurvature`
- `MaxAbsValue`
- `MaxAdjacentJump`
- `Samples`

### 9. Baseline Submissions

- **Evaluation ID 1**: simple ellipse with constant test function. It is valid and is intended as a working format seed.

Use `/execute_action{review 1}` to inspect the baseline result. Use `/execute_action{read_code 1}` if you need the baseline code.
