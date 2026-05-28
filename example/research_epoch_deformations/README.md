## Task Specification

This is the Epoch AI's [Explicit Deformations of Algebras](https://epoch.ai/frontiermath/open-problems/explicit-deformations) task.

## Setup

```bash
cp -r example/station_default station_data
cp -rL example/research_epoch_deformations/research station_data/rooms
cp example/research_epoch_deformations/constant_config.yaml station_data/constant_config.yaml
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
