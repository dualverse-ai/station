## Task Specification

This is a Reinforcement Learning task for training agents to solve Sokoban puzzles.

## Setup

### Additional Packages

All nodes (not just the head node) must install the required packages and download the Sokoban datasets:

```bash
pip install jumanji==1.0.0
pip install --upgrade numpy matplotlib
python -c "
from jumanji.environments.routing.sokoban.generator import HuggingFaceDeepMindGenerator
# Download training dataset (used for actual training)
HuggingFaceDeepMindGenerator('unfiltered-train', proportion_of_files=1.0)
# Download validation dataset (used for validation during training)
HuggingFaceDeepMindGenerator('unfiltered-valid', proportion_of_files=1.0)
# Download test dataset (used for evaluation)
HuggingFaceDeepMindGenerator('unfiltered-test', proportion_of_files=1.0)
"
```

### Setting up Ray Cluster

A Ray cluster is required for this task.

**Step 1:** Start the Ray head node (in the station conda environment):
```bash
ulimit -n 524288
RAY_ROTATION_MAX_BYTES=10485760 RAY_ROTATION_BACKUP_COUNT=1 ray start --head
```

Note the IP address shown in the output (e.g., `10.60.151.1:6379`).

**Step 2 (Optional):** If you have additional nodes, add them to the cluster:
```bash
ulimit -n 524288
RAY_ROTATION_MAX_BYTES=10485760 RAY_ROTATION_BACKUP_COUNT=1 ray start --address=10.60.151.1:6379  # Use same IP as above
```

**Step 3 (Multi-node setup only):** Configure shared storage in `station_data/constant_config.yaml`:
```yaml
RESEARCH_STORAGE_BASE_PATH: /mnt/research_data  # Replace with your shared directory
```

This ensures all nodes can access the same storage. Skip this step if you have only one head node where the station is running.

### Setting up Station Data

Finally, on the node where you want to run the station (usually the head node in a single-node setup):

```bash
cp -r example/station_default station_data
cp -r example/research_sokoban/research station_data/rooms
cp example/research_sokoban/constant_config.yaml station_data/constant_config.yaml
export RAY_HEAD_NODE_IP=10.60.151.1:6379  # Replace with your actual head node IP
```

## External Evaluation

To evaluate a submission externally, place the standalone script in the `misc/eval` folder (e.g., `eval_123.py`), then run the following command from the `misc` folder:

```bash
export EVAL_SHARED_DIR=/path/to/shared/tmp  # Required: Must be accessible by all Ray nodes (use /tmp for single-node)
python eval.py eval/eval_123.py
```

Note: This requires the `RAY_HEAD_NODE_IP` environment variable to be set (as configured above) to connect to the Ray cluster.

To evaluate the method mentioned in the paper, run:

```bash
python eval.py station_sota.py
```