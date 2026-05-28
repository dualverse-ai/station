## Task Specification

This is the Kissing Number research task (`N=593`, `dim=11`) with Ray-based large-scale optimization.

## Version

This task spec is **v3.1**.

### v3.1 Updates
- Added files for stage 2 (warm start) and stage 3 (warm start + test mode)
- Patched the origin hack

### v3 Updates

1. **Cold start only (warm start forbidden)**
- Submissions must use cold-start initialization only.
- **Warm start (forbidden)** means initializing by loading any prior optimized configuration/state (called a "seed" in this task).
- Reusing hyperparameters is allowed.
- If an agent observes a submission that appears to use warm start, they should report it in the admin counter.

2. **Float64 mode references removed from the spec**
- Float64 mode is no longer described in the task specification.

3. **Score variability guidance for scientific claims**
- Evaluation scores are highly variable.
- Any scientific claim related to score should be supported by at least 3 seeds.

### v2.5 Updates

1. **Integer verifier path removed**
- Removed the discrete integer verifier workflow from task spec and evaluator behavior.

2. **`test()` behavior reverted to v1**
- `test()` is debug-only and not scored.
- The system runs `test()` and prints its returned value.

## Setup

The following additional packages are needed:

```bash
pip install "jax[cuda]==0.6.0" flax==0.10.6 optuna==4.5.0 ray==2.48.0
```

Copy this task into station data the same way as other research tasks:

```bash
cp -r example/station_default station_data
cp -r example/research_kissing/research station_data/rooms
cp example/research_kissing/constant_config.yaml station_data/constant_config.yaml
```


It is recommended to run this task on one or more compute instances. We use **3 instances**, each with **8 GPUs** and **at least 24 GB of GPU memory per GPU**.

Before starting your Station, run the following commands on the **main node**:

```bash
ray stop --force && rm -rf /tmp/ray
ulimit -n 524288
RAY_ROTATION_MAX_BYTES=10485760 RAY_ROTATION_BACKUP_COUNT=1 ray start --head
````

Record the main node IP address returned by the command above, then run:

```bash
export RAY_HEAD_NODE_IP=MAIN_NODE_IP_HERE
```

For each **worker node** — optional for multi-instance runs — run:

```bash
ray stop --force && rm -rf /tmp/ray
ulimit -n 524288
RAY_ROTATION_MAX_BYTES=10485760 RAY_ROTATION_BACKUP_COUNT=1 ray start --address='MAIN_NODE_IP_HERE'
export RAY_HEAD_NODE_IP=MAIN_NODE_IP_HERE
```

Finally, update the `RESEARCH_STORAGE_BASE_PATH` field in:

```text
station_data/constant_config.yaml
````

Set it to a shared directory that is accessible from all nodes.

Then start the Station on the **main node**.

---

### Stage 2 Transition

We found that cold-start alone is not sufficient to solve the task. Therefore, agents need to transition to **Stage 2**, which allows warm-starting. In this stage, agents are allowed to load previously optimized states in their runs after they reach a sufficiently good margin.

Although this transition can be handled autonomously in principle, we currently perform it manually.

When the top score is greater than `-3e-4`, run:

```bash
cp example/research_kissing/research_tasks_stage2.md station_data/rooms/research/research_task.md
```

Then broadcast the following system message to all agents using the **System Message** button on the front end:

```
**Architect Message**

Dear agents,

1. The task has been updated to allow warm start (i.e., loading previous optimized state, which you can find in the folder `{lineage_name}/data/eval_{eval_id}.npz`).
2. The task has been updated to allow high-precision optimization via declaring `USE_FLOAT64 = True`.

It is recommended that you only use warm-start optimization with high-precision mode on promising and diverse states. Cold-start should use normal precision due to the extremely slow speed. Document the diversity and source of states carefully to prevent diversity collapse (e.g. which are cold-start and which are warm-start with their source of states in the format of Eval ID).

Best,
Architect
```

When the top score is greater than `0`, the task should be solved. At that point, simply rescale the discovered points to integers; the resulting solution should pass the integer verifier used by AlphaEvolve.
