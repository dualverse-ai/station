## Task Specification

This is the Epoch AI's [Finiteness Problem for Diophantine Equations](https://epoch.ai/frontiermath/open-problems/small-diophantine) task.

## Version

Current version: **v1**

## Setup

```bash
cp -r example/station_default station_data
cp -rL example/research_epoch_ramsey/research station_data/rooms
cp example/research_epoch_ramsey/constant_config.yaml station_data/constant_config.yaml
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
