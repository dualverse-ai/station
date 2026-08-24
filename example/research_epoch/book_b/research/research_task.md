## Research Task: Ramsey Lower-Bound Constructions for Triangular Books, n=22..100

**This specification is the authoritative statement of the task and overrides
all other sources.**

### 1. Problem Description

Let `B_n` denote the triangular book graph on `n + 2` vertices, i.e. `n`
triangles sharing a common edge.

For each tested integer `n`, produce a graph `G_n` with exactly `4n - 2`
vertices such that:

- `G_n` does not contain `B_{n-1}`, and
- the complement `G_n^c` does not contain `B_n`.

This certifies:

```text
R(B_{n-1}, B_n) > 4n - 2
```

for that value of `n`.

### 2. Evaluation Overview

The evaluator tests all integers:

```text
22 <= n <= 100
```

It calls `solution_batch()` once. For each tested `n`, it:

- reads the returned graph for `n`, if any,
- parses the graph from its adjacency string,
- checks that the graph has exactly `4n - 2` vertices,
- computes the maximum number of common neighbors on every edge of `G_n`,
- computes the maximum number of common neighbors on every edge of `G_n^c`, and
- verifies the two exact book-avoidance inequalities.

For a graph `G`, a copy of `B_t` exists iff some edge of `G` lies in at least
`t` triangles. Therefore:

- `G_n` avoids `B_{n-1}` iff every edge of `G_n` has at most `n - 2`
  common neighbors.
- `G_n^c` avoids `B_n` iff every edge of `G_n^c` has at most `n - 1`
  common neighbors.

### 3. Scoring

The score is exact range coverage:

```text
score = number of tested n values that pass exact validation
```

Each valid `n` gives `+1`. Missing or invalid returned values give `0` for
that `n`.

The maximum possible score is `79`, one point for each integer in `22..100`.

Runtime and resource notes:

The submitted code is run in a restricted sandboxed environment.

The official evaluation has an externally enforced timeout of about
**30 minutes**. If the submitted code exceeds this limit, it may be terminated
before returning a batch.

Common scientific Python tools are available, including NumPy, SciPy, JAX with
GPU support, Sage, SymPy, and standard-library modules such as `itertools`,
`random`, and `math`. C++ is also allowed for helper code or compiled search
routines.

Each official submission is allotted one GPU. Use GPU acceleration when it
helps the search, but the final solution may or may not need GPU computation.

### 4. Known Construction Landscape

Known constructions already reveal substantial structure in this problem.
Treat them as construction languages to understand and extend, not as a
request to copy isolated finite certificates.

For the full tested range `22 <= n <= 100`, the bundled construction baseline
currently validates the following values:

```text
22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
36, 37, 38, 39, 41, 42, 43, 45, 46, 49, 50, 51, 53, 54,
55, 57, 61, 62, 63, 66, 69, 71, 74, 75, 77, 79, 82, 83,
85, 87, 90, 91, 95, 97, 98, 99
```

This list contains 50 solved values out of the 79 tested values. The missing
values are:

```text
40, 44, 47, 48, 52, 56, 58, 59, 60, 64, 65, 67, 68, 70,
72, 73, 76, 78, 80, 81, 84, 86, 88, 89, 92, 93, 94, 96,
100
```

The baseline uses the most structural known source for each solved value:

- **Two-copy Paley:** `25, 27, 31, 37, 41, 45, 49, 51, 55, 57,
  61, 63, 69, 75, 79, 85, 87, 91, 97, 99`.
  For `q = 2n - 1`, with `q = 1 mod 4` a prime power, use two field layers
  and the quadratic-residue/nonresidue split.
- **S/K conference construction:** `26, 30, 38, 42, 46, 50, 54,
  62, 66, 74, 82, 90, 98`.
  For `q = n - 1`, a conference-type relation with parameters
  `(q, (q-1)/2, (q-5)/4, (q-1)/4)` feeds a fixed four-chamber rule table.
  This includes Paley conference relations and the bundled non-prime-power
  conference masks for `q = 45` and `q = 65`.
- **Skew-Paley sum kernel:** `33, 35, 53, 71, 77, 83, 95`.
  For `q = 4n - 1`, with `q = 3 mod 8` a prime power, a graph on
  `GF(q)^*` uses the quadratic character of `x + y` and validates by an exact
  Seidel row-correlation identity.
- **Tenax bicirculant Tabu search:** `22, 23, 24, 28, 29, 32, 34`.
  These rows are generated during the baseline build by a deterministic
  two-fiber cyclic search over same-fiber residue sets `S_U`, `S_V` and a
  directed cross residue set `S_UV`.
- **Z_71 capacity-profile row:** `36`.
  This is the only bundled finite-source preload. The included C generator
  implements the search grammar, but this case needs a long full run, so the
  baseline stores the compact `A/C` source row and marks it explicitly in the
  regenerated artifact manifest.
- **Z_7 x Z_11 product completion:** `39`.
  The baseline runs a product-group `A/C` search, then compiles a generated
  fixed-`A` completion program for `C`.
- **Z_85 orbit deconvolution:** `43`.
  The baseline exhausts an uncoupled cyclic bicirculant manifold using
  multiplication-by-4 orbits, internal convolution filters, and exact cross
  convolution.

When two descriptions cover the same `n`, prefer the more structural one.
Finite search is most useful for values not yet explained by an infinite or
parametrized source law.

### 5. Mathematical Exploration Goal

The main research interest is not only to return isolated finite certificates.
We especially want source laws that generate infinitely many or parametrically
many valid witnesses.

The S/K construction is a model example: once a suitable conference-type
relation on `n-1` points is supplied, the graph and its pair-row slack
certificate are generated together. Work that discovers new infinite families,
new reusable relation sources, or a theorem explaining many `n` values is more
valuable than a collection of unrelated finite bitstrings.

Finite witnesses are still useful, especially for currently missing values, but
the most valuable finite work should expose a compact construction language or
ledger that can be reused.

It is highly prized to take a searched witness, a baseline source packet, or
any other finite artifact, understand the structural property that makes it
work, and convert that property into a parametrized or infinite family. This is
valuable even when it does not immediately increase the score on `22..100`.
The ultimate mathematical goal is to prove the lower bound for all relevant
`n`, not only for the finite evaluation range.

### 6. Available Baseline Resources

The system storage includes an algorithmic construction baseline. It builds
witnesses from infinite or parametrized families, the bundled non-prime-power
conference masks, and finite generators that execute during the build.

The only exception is the `n=36` finite source row described above. It is
included as a compact `A/C` source row because the full regeneration search is
long; the generator file remains available for inspection and further work.

After a baseline build, the system writes:

- `baseline_witnesses.npy`: generated adjacency strings for the solved
  baseline values;
- `baseline_manifest.json`: compact solved/missing/family summary;
- `regenerated_source_artifacts.json`: source residues, search summaries, and
  validation stats emitted by the finite generators.

The generated witness file is a convenience artifact. It should not be treated
as the explanation for the constructions.

### 7. Submission Format

Your instruction should tell the coder what experiment to implement. The
submitted Python file must define:

```python
def solution_batch() -> dict[int | str, str]:
    """
    Return adjacency strings for any subset of tested n values.
    """
    ...
```

Keys may be integers or strings. For example, either `22` or `"22"` may be used
for `n = 22`.

Values must be adjacency strings in the format below.

Submissions may reload saved constructions or artifacts when useful; the
evaluator only checks the returned graphs. If you use generated baseline
artifacts, treat them as starting material, not as a substitute for explaining
new construction ideas.

You may optionally define:

```python
def test():
    """
    Optional debug-only entrypoint.
    If defined, the runner executes only test() and returns score n.a.
    """
    ...
```

`test()` is useful for debugging or quick experiments without triggering
official scoring.

The definitive runner implementation lives in:

- `storage/system/epoch_book_b_eval_runner.py`

### 8. Adjacency String Format

The returned graph must be encoded as a binary string of length
`C(4n - 2, 2)`.

Bit order is column-major by the larger endpoint, zero-indexed:

```text
{0,1}, {0,2}, {1,2}, {0,3}, {1,3}, {2,3}, ...
```

For each `j = 1,2,...,N-1`, list bits for pairs `{0,j}`, `{1,j}`, ...,
`{j-1,j}`.

Interpretation:

- bit `1` means the edge is present in `G`
- bit `0` means the edge is absent in `G`

Whitespace around the string is ignored. No separators are allowed inside the
bitstring.

### 9. Exact Validity Check

Let `N_G(u)` denote the neighborhood of `u` in `G`.

The evaluator computes:

```text
red_max  = max_{uv in E(G)}     |N_G(u) intersection N_G(v)|
blue_max = max_{uv in E(G^c)}   |N_{G^c}(u) intersection N_{G^c}(v)|
```

A witness is valid for `n` exactly when:

```text
red_max  <= n - 2
blue_max <= n - 1
```

Equivalently:

- `red_excess  = max(0, red_max  - (n - 2))`
- `blue_excess = max(0, blue_max - (n - 1))`

and validity means both excesses are zero.

### 10. Secondary Metrics

The evaluator reports:

- `ValidCount`
- `TotalCount`
- `WorstRedExcess`
- `WorstBlueExcess`
- `WorstN`
- `WorstViolatingPairRate`

Apart from `ValidCount`, these are diagnostics, not alternate objectives.

### 11. Resource Reading Priority

To save context, read files in phases.

**First pass / short reads**

- `storage/system/baseline_manifest.json`: compact solved/missing/family index.
- `storage/system/construction_survey.md`: compact generator hierarchy and
  extension guidance.
- `storage/system/regenerated_source_artifacts.json`: finite source residues,
  search summaries, and validation stats from the latest baseline build.

**Implementation reads**

- `storage/system/build_epoch_book_b_baseline.py`: executable constructors for
  the bundled baseline. Read this when you want to reuse or modify the baseline
  construction code; otherwise start with the two short files above.
- `storage/system/epoch_book_b_finite_generators.py`: orchestrates the finite
  generators for `n = 22, 23, 24, 28, 29, 32, 34, 36, 39, 43`.
- `storage/system/epoch_book_b_tenax_tabu.c`: deterministic two-fiber Tenax
  Tabu search for the low bicirculant rows.
- `storage/system/epoch_book_b_n36_c_search.c`: long `n=36` capacity-profile
  generator; the baseline uses the explicitly marked source-row exception.
- `storage/system/epoch_book_b_n39_phase1.c` and
  `storage/system/epoch_book_b_n39_phase2_template.c`: product-group search
  and fixed-`A` completion for `n=39`.
- `storage/system/epoch_book_b_n43_orbit.py`: orbit deconvolution generator
  for `n=43`.

**Finite-search extension reads**

- `storage/system/finite_search_algorithms.py`: reusable finite-search grammar,
  including residual rows, simulated annealing, and fixed-`A` CP-SAT completion.
- `storage/system/bridge_algorithm_notes.md`: bridge bicirculant residual law
  and CRT ledger interpretation. This is useful background, but it is not the
  active baseline path.

**Low priority / generated**

- `storage/system/baseline_witnesses.npy`: generated adjacency-string witnesses
  for the solved baseline values. Load it only if you want to start from the
  existing baseline; do not read it as text.
- `storage/system/epoch_book_b_eval_runner.py`: exact runner implementation.
  Read only when debugging evaluator mechanics.
- `storage/system/epoch_book_b_source_laws.py`: historical source-law table
  retained as reference. The active baseline builder does not import it.
