## Research Task: Large-x Diophantine Solutions Across All Nine Equations

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

The problem consists of nine separate Diophantine equations. The official objective is to solve all nine within one submission output.

Mathematicians are interested in proving that these equations have infinitely many integer solutions. Strong submissions should therefore prefer constructive methods that can generate solutions with arbitrarily large magnitude, not just a fixed set of examples, even though the official evaluator only checks the submitted triples against the finite `|x| > 10^50` threshold below.

For each of the following equations, find three distinct integer solutions `(x, y, z)` such that:

- all coordinates are integers
- each triple exactly satisfies the equation
- the three triples are pairwise distinct
- their `x` values are pairwise distinct
- each satisfies `|x| > 10^50`

Equations:

1. `z^2 + y^2 z + x^3 - 2 = 0`
2. `z^2 + y^2 z + x^3 - x - 1 = 0`
3. `z^2 + y^2 z + x^3 + x - 1 = 0`
4. `z^2 + y^2 z + x^3 + x + 1 = 0`
5. `z^2 + y^2 z + x^3 - 3 = 0`
6. `z^2 + y^2 z + x^3 + 3 = 0`
7. `z^2 + y^2 z + x^3 - x - 2 = 0`
8. `z^2 + y^2 z + x^3 - x + 2 = 0`
9. `z^2 + y^2 z - z + x^3 + 2 = 0`

### 2. Evaluation Overview

The evaluator scores partial progress by counting how many of the nine equations are solved.

- Each solved equation contributes `100 / 9` points.
- Score `100` means all 9 equations are solved at once in a single submission.
- Score `0` means none of the 9 equations are solved.
- No shaped reward is given inside an equation: an equation either contributes full `1/9` credit or contributes `0`.

### 3. Why Score 100 Means Solved

The score is `100` exactly when, for every equation `1..9`, the evaluator verifies:

- at least three submitted triples parse as exact integers
- those triples satisfy that equation exactly over the integers
- those triples are pairwise distinct
- their `x` values are pairwise distinct
- each of those `x` values satisfies `|x| > 10^50`

### 4. Task Constraints and Goals

- **Goal:** maximize the number of solved equations, ideally all 9
- **Execution Timeout:** 15 minutes (900 seconds)
- **Resource Limits:** CPU-only. Standard Python libraries are available.
- **Available Packages:** `python-sat`, `z3-solver`, `ortools`, and `sage` are installed and may be used.

### 5. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file will be executed by the Station evaluator.
- The submitted Python file **must** define:

  ```
  def find_solutions() -> str:
      """
      Return a JSON string containing candidate solutions for all nine equations.
      """
      return "..."
  ```

- The returned string must parse as JSON with this shape:

  ```json
  {
    "solutions_by_equation": {
      "1": [
        {"x": "123", "y": "-4", "z": "5"},
        {"x": "456", "y": "7", "z": "-8"}
      ],
      "2": [],
      "3": [
        {"x": "100000000000000000000000000000000000000000000000001", "y": "2", "z": "-3"}
      ]
    }
  }
  ```

- `solutions_by_equation` must be a JSON object.
- Keys should be equation ids `"1"` through `"9"`; missing keys are treated as empty lists.
- Each value should be a list of candidate triples for that equation.
- Each of `x`, `y`, `z` may be either:
  - a JSON integer, or
  - a base-10 integer string
- Decimal strings are recommended for very large integers.
- Malformed entries are ignored individually; malformed top-level JSON receives score `0`.

- **Evaluator implementation:** the definitive checks live in the evaluator source.
  Agents can inspect it via:
  `/execute_action{read system/evaluator.py}`.
- You may treat `find_solutions()` as a sandbox runner. If it returns `None`, the evaluation will report score `0`.

### 6. Secondary Metrics

The evaluator always reports:

- `SolvedEquations`
- `TotalLargeXSolutions`
- `BestEquationLargeXCount`
- `BestEquationMaxLog10AbsX`
- `EqSolvedMask`

Meanings:
- `SolvedEquations`: how many of the nine equations earned their full `1/9` credit.
- `TotalLargeXSolutions`: total number of exact submitted solutions across all equations with `|x| > 10^50`.
- `BestEquationLargeXCount`: largest count of exact large-`x` solutions achieved for any single equation.
- `BestEquationMaxLog10AbsX`: best `log10(|x|)` achieved among exact solutions for any equation, or `-1` if none exist.
- `EqSolvedMask`: a 9-character `0/1` mask indicating which equations were solved.

### 7. Baseline Submissions

- **Evaluation ID 1**: valid-format wrong submission covering all nine equations (expected score `0`).

Use `/execute_action{review 1}` to inspect the baseline result. Use `/execute_action{read_code 1}` if you need the baseline code.
