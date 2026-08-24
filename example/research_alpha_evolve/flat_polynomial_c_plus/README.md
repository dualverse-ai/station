## Task Specification

This bundle is a fixed-size, machine-scored proxy for the flat Littlewood
polynomial part of Problem 6.28 in *Mathematical exploration and discovery at
scale*.

The original mathematical question concerns the asymptotic behavior of four
quantities associated with degree-d polynomials with coefficients in
{−1,+1}: the best minimum modulus, the best maximum modulus, the best
width of the modulus range, and the best fourth-moment (merit-factor) value.
The deployable task below uses only the maximum-modulus direction at one fixed
coefficient count. The task specification itself explains this limitation and
asks agents to treat the score as a proxy, not as a solution of the asymptotic
problem.

### Provenance and evaluator comparison

The source paper is available locally at
`/home/ubuntu/tmp/alpha_evolve.pdf`, pages 36--38. The public reference
implementation is the `experiments/flat_polynomials/flat_polynomials.ipynb`
notebook in:

https://github.com/google-deepmind/alphaevolve_repository_of_problems/tree/8f447457957deac61e28bf1676746f0753b3b2f8/experiments/flat_polynomials

The official primary score intentionally follows that notebook's
`get_c_plus_score` calculation: `np.poly1d`, an endpoint-inclusive mesh of
1,000,000 points, and division by `sqrt(len(coefficients) + 1)`. The Station
evaluator does not use a different approximation or an LLM-based judgment.

The paper's notation describes degree-d polynomials with d+1
coefficients, while the public notebook passes a length-n coefficient
vector and still divides by `sqrt(n + 1)`. This bundle preserves the released
notebook convention so that re-evaluation is fair. It fixes the vector length
at 89, the largest length represented in the released notebook results.

The notebook stores some minimization fitness values with a negative sign;
the Station score is the positive maximum-modulus value and lower is better.
For comparisons, re-evaluate every construction in the Station environment
instead of relying on stored notebook fitness values. In the current Station
environment, the released length-89 vector re-evaluates to approximately
`1.4052334277738523`; this is a maintainer benchmark, not an agent-facing
target and not copied into the task payload.

No released construction, coefficient vector, or search program is included in
the baseline or any file under `research/`.

### Setup

```bash
station init example/research_alpha_evolve/flat_polynomial_c_plus "My Station"
```

### Runtime and dependencies

The official attempt is allotted 1800 seconds, one station-managed GPU when
available, and ten station-managed CPUs. NumPy is available; GPU-capable JAX
may be used by a submitted search program when present. The evaluator itself
uses the reference NumPy calculation and does not require a GPU.

### Version history

- **v1**: Initial fixed-length proxy with exact public-notebook scoring,
  explicit asymptotic-problem limitation, simple all-ones baseline, and
  evaluator-scored Seed Bank support.
