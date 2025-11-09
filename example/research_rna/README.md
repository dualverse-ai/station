## Task Specification

This is the RNA Modeling task using the BEACON dataset.

## Setup

### Dataset

Visit the official [BEACON repository](https://github.com/terry-r123/RNABenchmark) to download the following 7 datasets from the [Google Drive folder](https://drive.google.com/drive/folders/1nBytCBey8CRYnAagwvwjDU1yFrYkBRo2):

```
Isoform
CRISPROffTarget
Modification
CRISPROnTarget
ProgrammableRNASwitches
MeanRibosomeLoading
NoncodingRNAFamily
```

Store them in a local directory (e.g., `/mnt/data/`).

**Note:** For multi-node Ray clusters, the datasets must be stored on a shared drive accessible by all Ray nodes. For single-node setups, any local directory is fine.

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
cp -r example/research_rna/research station_data/rooms
cp example/research_rna/constant_config.yaml station_data/constant_config.yaml
export RAY_HEAD_NODE_IP=10.60.151.1:6379  # Replace with your actual head node IP

# Create symbolic links to the datasets
DATA_PATH=/mnt/data  # Replace with your actual dataset directory
mkdir -p station_data/rooms/research/storage/system/data && \
for x in Isoform CRISPROffTarget Modification CRISPROnTarget ProgrammableRNASwitches MeanRibosomeLoading NoncodingRNAFamily; do
  cp -rs "$DATA_PATH/$x" "station_data/rooms/research/storage/system/data/$x"
done
```

## External Evaluation

To evaluate a submission externally, place the standalone script in the `misc/eval` folder (e.g., `eval_123.py`), then run the following command from the `misc` folder:

```bash
export EVAL_SHARED_DIR=/path/to/shared/tmp  # Required: Must be accessible by all Ray nodes (use /tmp for single-node)
export RNA_DATA_PATH=/path/to/rna/data  # Optional: Path to RNA datasets (default: /mnt/stephen/data/rna/data)
python eval.py eval/eval_123.py
```

By default, this runs a single trial (using the default seed) on each of the 7 datasets. You can run trials with different seeds by adding the `--seed {seed}` flag.

Note: This requires the `RAY_HEAD_NODE_IP` environment variable to be set (as configured above) to connect to the Ray cluster.

To evaluate the method mentioned in the paper, run:

```bash
python eval.py station_sota.py
```