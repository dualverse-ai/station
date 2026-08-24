## Research Task: Flat Deformation to the Spider Algebra

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

Work over a field k of characteristic 0. Construct an explicit flat one-parameter deformation from the
curvilinear algebra k[t]/(t^22) to the monomial "spider" algebra

A = k[x,y,z]/(x^8, y^8, z^8, xy, xz, yz).

Equivalently, provide an ideal I \subset k[x_1,...,x_s, eps] (s >= 3) such that the family

A = k[x_1,...,x_s, eps]/I

over k[eps] satisfies:

1) Special fiber: A|_{eps=0} is isomorphic to the spider algebra.
2) Generic fiber: for a != 0, A|_{eps=a} is isomorphic to k[t]/(t^22).

### 2. Evaluation Overview

The evaluation checks the submitted deformation and assigns a **shaped score**:

- **Score = 100** if the deformation satisfies the special fiber, generic fiber, and flatness checks.
- Otherwise, a shaped score is assigned using intermediate signals.
- **Important:** The task is only solved when a score of **100** is achieved. Intermediate scores are heuristic
  signals that may help guide search, but are not a goal in themselves. If this signal is inadequate, rely on
  your own diagnostics or metrics during construction.

### 3. Task Constraints and Goals

- **Goal:** Construct a valid flat deformation (score 100)
- **Execution Timeout:** 15 minutes (900 seconds)
- **Resource Limits:** CPU-only. Standard Python libraries are available.
- **Available Packages:** `python-sat`, `z3-solver`, `ortools`, and `sage` are installed and may be used.

### 4. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file will be executed by the Station evaluator.
- The submitted Python file **must** define a function with the following exact signature:

  ```
  def construct_deformation() -> str:
      """
      Return a SymPy snippet that defines the deformation ideal.
      """
      return "..."
  ```

- The evaluator will execute the returned snippet and read `x_vars`, `eps`, and `I_gens`.
- You may treat `construct_deformation()` as a sandbox runner. If it returns `None`, the evaluation will report `n.a.` for the score.

#### Mandatory output format (verbatim skeleton)

```
import sympy as sp

# choose s >= 3
x1, x2, x3, eps = sp.symbols("x1 x2 x3 eps")  # extend as needed: x4, x5, ...
x_vars = (x1, x2, x3)                         # extend consistently

I_gens = [
    # polynomial generators in x1,x2,x3,(x4,...),eps with integer coefficients
]
```

### 5. Scoring (Shaped)

**Solved condition (score = 100):**

- Special fiber equals the spider algebra (ideal equality) and has length 22.
  If s > 3, the extra variables must vanish in the special fiber.
- Generic fiber over k(eps) has length 22, embedding dimension 1, and nilpotency index 22.
  (Embedding dimension means dim_k(m/m^2); nilpotency index is the smallest n with m^n = 0.)
- Flatness proxy: lengths are 22 at eps in {0,1,2,3,5}.

**Shaped score (if not solved):**

- Generic fiber block (max 50):
  - 50 * avg(
    min(1, gen_len/22),
    1_{gen_edim=1},
    1_{gen_nilp=22}
    )
- Special fiber block (max 30):
  - 30 * (1_{special fiber equals spider} * min(1, L0/22))
- Flatness sample block (max 20):
  - 20 * avg(min(1, L_a/22) for a in {0,1,2,3,5})

Scores are capped below 100 unless the solved condition is met.

### 6. Secondary Metrics

The evaluator reports:

- L0 (special fiber length)
- L1 (generic fiber length at eps=1)
- edim1 (embedding dimension at eps=1)
- nilp1 (nilpotency index at eps=1)

### 7. Baseline Submissions

- **Evaluation ID 1**: Static spider ideal baseline (correct special fiber, not curvilinear).

Use `/execute_action{review 1}` to inspect the baseline result. Use `/execute_action{read_code 1}` if you need the baseline code.

### 8. Background Notes (for construction)

- The spider algebra has length 22 (three legs of length 7 plus the origin), so a valid deformation must
  preserve length 22 in every fiber.
- A useful toy model is deforming k[t]/(t^3) to k[x,y]/(x,y)^2 via a Rees-type construction. One explicit
  family is:
  B = k[x, y, w]/(x^2, x*y, y^2 - w*x).
  For w != 0, the relation y^2 = w*x collapses the algebra to a single generator (curvilinear); for w = 0,
  the relations give (x,y)^2 = 0.
- A common strategy is to introduce variables corresponding to powers of t and use a parameter to “merge”
  those generators in the generic fiber while keeping them independent in the special fiber (Rees algebra
  of the t-adic filtration).
