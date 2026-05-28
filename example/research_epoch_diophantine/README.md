## Task Specification

This is the Epoch AI's [A Ramsey-style Problem on Hypergraphs](https://epoch.ai/frontiermath/open-problems/ramsey-hypergraphs) task.

### Current version: v2

v2 models the original problem as nine subproblems evaluated together in one submission.

- each solved equation contributes `100 / 9` points
- score `100` means all 9 equations are solved in one submission
- score `0` means none are solved
- there is no within-equation shaped reward

The evaluator reports compact aggregate diagnostics:
`SolvedEquations`, `TotalLargeXSolutions`, `BestEquationLargeXCount`,
`BestEquationMaxLog10AbsX`, and `EqSolvedMask`.

The task accepts a JSON submission payload so agents can emit arbitrary-precision decimal
integers safely as strings.

## Setup

```bash
cp -r example/station_default station_data
cp -rL example/research_epoch_diophantine/research station_data/rooms
cp example/research_epoch_diophantine/constant_config.yaml station_data/constant_config.yaml
```

## Optional Dependencies

For stronger construction/search methods, these optional packages are helpful:

```bash
pip install python-sat z3-solver ortools
```

SageMath can be installed via conda:

```bash
conda install -y -c conda-forge sage
```
