## Task Specification

This is the Epoch AI's [Ramsey Numbers for Book Graphs](https://epoch.ai/frontiermath/open-problems/ramsey-book-graphs) task.

## Version

Current version: **v2**

### v2 Changes

1. Changed the bundled evaluator to test the full range `22 <= n <= 50` without stopping on the first failing instance.
4. Changed primary scoring from binary pass/fail to exact range coverage:
   - `ValidCount = #{n in [22, 50] : witness passes exact checks}`
   - `score = 100 * ValidCount / 29`
   - `score = 100` iff all tested values pass
5. Changed the required submission interface to `solution_batch() -> dict[int | str, str]`, so submissions can allocate computation across the full tested range directly.
6. Set `RESEARCH_EVAL_TIMEOUT` to `900` seconds.

## Setup

```bash
cp -r example/station_default station_data
cp -rL example/research_epoch_book/research station_data/rooms
cp example/research_epoch_book/constant_config.yaml station_data/constant_config.yaml
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
