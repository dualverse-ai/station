## Task Specification

This is the three-dimensional finite-field Nikodym set optimization task.

Spec version: **v1**.

### Developer provenance

The problem is the Nikodym half of Problem 6.1 in *Mathematical exploration
and discovery at scale*. The scored benchmark follows the released 3D
Nikodym evaluator in:

```text
google-deepmind/alphaevolve_repository_of_problems
experiments/finite_field_nikodym_problem/finite_field_nikodym.ipynb
```

The repository version inspected while authoring this template was commit
`8f447457957deac61e28bf1676746f0753b3b2f8`.

Keep this provenance, the comparison methodology, and every AlphaEvolve result
or method out of the agent-visible `research/` directory. In particular, do not
copy notebook solution cells, prompts containing construction hints, reported
scores or sizes, or material from `deep_think_docs/` into the task spec,
baseline, evaluator comments, or runner comments.

### Benchmark choice

Problem 6.1 concerns Kakeya and Nikodym sets over finite fields. This bundle is
only the Nikodym task; it is separate from
`example/research_alpha_evolve/finite_kakeya/`.

The paper discusses mathematical results for every fixed dimension `d >= 3`,
but the released 3D evaluator is hard-coded to `d = 3`. The official Station
score therefore fixes `d = 3` so it remains directly comparable with the
released benchmark. Higher-dimensional generalization is included only as an
unscored mathematical exploration goal.

The released notebook also contains two-dimensional experiments over
`F_(p^2)^2`. Those experiments use different prompts, representations,
benchmark behavior, and strong construction guidance. They are intentionally
not combined with this task.

### Evaluator correspondence

The command-mode runner preserves the released 3D evaluator's substantive
contract:

- submission function `search_for_best_construction(p, d)`;
- dimension `d = 3`;
- prime suite `5, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89`;
- exhaustive punctured-line verification using the same canonical direction
  order;
- duplicate-row removal and `int64` conversion;
- normalization by `(p - 1)^3 + p + 1`;
- exclusion of the valid `p = 5` result from the score average; and
- the released invalid-instance penalty of `1000` in the average.

The two random `uniform(0, 0.0001)` perturbations in the notebook are omitted.
They are tie noise rather than mathematical evaluation, and retaining them
would make identical Station submissions non-reproducible. Comparisons should
therefore be described precisely as using the same benchmark instances,
verification logic, normalization, and aggregation, with random tie jitter
disabled. Do not claim byte-for-byte identity with the notebook evaluator.

The Station wrapper adds only reporting fields and command-output parsing; it
does not alter valid construction checks or the deterministic primary score.

### Simple baseline

Evaluation `#1` returns all of `F_p^3` for every requested prime. This is a
trivial valid Nikodym set and a deliberately weak format/control baseline. It
does not expose the notebook's initial construction or any subsequently
discovered construction.

### Setup

```bash
station init example/research_alpha_evolve/nikodym "My Station"
```

### Dependencies and resources

The evaluator requires NumPy and Numba. It is CPU-only and does not require a
GPU. The repository requirements currently declare `numpy>=2.2.0` and
`numba>=0.61.0`.

The exhaustive verifier can be expensive because it checks every point and
may search many directions. On the authoring host, with NumPy `2.4.0` and
Numba `0.65.0`, the full-space baseline completed the complete eleven-prime
suite in about 67 seconds after JIT compilation and received deterministic
score `-1.0468550268409333`. This baseline finds the first direction
immediately at every point, so it is not a worst-case runtime benchmark. The
3600-second timeout and two-worker limit are intentionally conservative until
a slower nontrivial valid construction is benchmarked in production.

### External evaluation

No external evaluation is required. The bundled runner performs exact finite
verification. Successful constructions are not persisted separately by
default.

### Leakage audit

Before deployment:

1. Search every file under `research/` for `AlphaEvolve`, paper titles,
   reported bounds, reported scores, and construction-specific terminology.
2. Confirm the baseline is still the full-space construction.
3. Confirm the runner contains evaluator code only, with no copied solution
   cell or construction hint.
4. Confirm `research_task.md` states no target threshold or known benchmark
   result.

### Version update log

- **v1**: Initial `F_p^3` Nikodym task using the released 3D prime suite,
  exhaustive punctured-line verification, deterministic form of the released
  normalized score, a full-space baseline, and an unscored higher-dimensional
  generalization goal.
