## Task Specification

This research task evaluates batch integration methods on the **Human Heart** dataset (~20k cells, 19 cell types, 4 batches).

## Setup

### Environment

This task requires a custom conda environment. Run the following in the `misc` folder to set it up:

```bash
cd example/research_batch_integration/misc
bash setup_batch_integration_env.sh
conda activate batch_integration
```

### Dataset

To download and process the training and 6 testing datasets:

```bash
cd example/research_batch_integration/misc
pip install awscli
python prepare_data.py --dir /mnt/data  # Replace with your dataset directory
```

If successful, your dataset directory should contain:

```
├── cellxgene_364bd0c7.h5ad          # Raw human dataset
├── human_train.h5ad                  # Training HVG
├── human_train_solution.h5ad         # Training full
├── human_val.h5ad                    # Validation HVG
├── human_val_solution.h5ad           # Validation full
├── test_data_raw/                    # Raw test datasets
│   ├── dkd.h5ad
│   ├── gtex_v9.h5ad
│   ├── hypomap.h5ad
│   ├── immune_cell_atlas.h5ad
│   ├── mouse_pancreas_atlas.h5ad
│   └── tabula_sapiens.h5ad
└── test_data/                        # Processed test datasets
    ├── dkd_processed.h5ad
    ├── dkd_processed_solution.h5ad
    └── ... (12 files total)
```

### Setting up Station Data

```bash
cp -r example/station_default station_data
cp -r example/research_batch_integration/research station_data/rooms
cp example/research_batch_integration/constant_config.yaml station_data/constant_config.yaml

# Copy training datasets to station storage
DATA_PATH=/mnt/data  # Replace with your actual dataset directory
cp "$DATA_PATH/human_train.h5ad" station_data/rooms/research/storage/system/
cp "$DATA_PATH/human_train_solution.h5ad" station_data/rooms/research/storage/system/

# Set conda environment and enable auto-start
export CONDA_ENV_NAME=batch_integration
echo "AUTO_START: true" >> station_data/constant_config.yaml
```

## External Evaluation

There are two external evaluation options available. The first is an approximation (some metrics, such as `kbet`, may not be accurate) but requires only around 128GB RAM. The setup for the full evaluation according to OpenProblems v2.0 is more complicated and requires 500GB RAM, but this should be the only metric reported in papers.

### Approximated Version

To evaluate a submission externally, place the standalone script in the `misc/eval` folder (e.g., `eval_123.py`), then run the following command from the `misc` folder:

```bash
bash eval.sh eval/eval_123.py /mnt/data/test_data
```

Where:
- `eval/eval_123.py` is the path to your submission script
- `/mnt/data/test_data` is the path to your test datasets directory

The script will evaluate your submission on all 6 test datasets and display the final scores.

### Official Version

#### Key Steps for Evaluating on OpenProblems v2.0

#### 1. Download the OpenProblems v2.0 Repository

First, download the repo:

```bash
# Clone repo
git clone https://github.com/openproblems-bio/task_batch_integration
```

---

#### 2. Install Dependencies: viash, Nextflow, and Docker (if needed)

```bash
sudo apt-get update
sudo apt-get install openjdk-17-jdk unzip curl
wget -qO- dl.viash.io | bash
sudo mv viash /usr/local/bin
sudo chmod +x /usr/local/bin/viash
viash --help

curl -s https://get.nextflow.io | bash
chmod +x nextflow
sudo mv nextflow /usr/local/bin
nextflow help

sudo mkdir -p /etc/apt/keyrings /etc/apt/sources.list.d
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

---

#### 3. Add Your Method

Add your method under `src/methods`, ensuring your script includes the necessary **metadata in `uns`**.
If missing, a baseline embedding will be auto-generated, which will result in a **very low score**.
See `misc/station_sota` as an example, which should be placed under `src/methods` (i.e., `src/methods/station_sota`)

---

#### 4. Update Workflow Files

Modify both files below to include your method:

* `src/workflows/run_benchmark/main.nf` (e.g., after ` - name: methods/uce` add ` - name: methods/station_sota`)
* `src/workflows/run_benchmark/config.vsh.yaml` (e.g., after ` - name: methods/uce` add ` - name: methods/station_sota`)

---

#### 5. Build Your Method

```bash
viash ns build --parallel --setup cachedbuild --query '^(?!methods/)|methods/(METHOD)'
```

Replace `METHOD` with your method's folder name inside `src/methods`, e.g., `viash ns build --parallel --setup cachedbuild --query '^(?!methods/)|methods/station_sota'`

---

#### 6. Setup

```bash
rm -rf target
viash ns build --parallel
```

---

#### 7. Run Experiments

Run for the first five datasets:

```bash
DATA_PATH=/mnt/data  # Must be absolute path to your test_data directory
METHOD=station_sota  # Replace with your method name

for dataset in dkd immune_cell_atlas mouse_pancreas_atlas gtex_v9 hypomap; do
echo "=========================================="
echo "Processing dataset: ${dataset}"
echo "=========================================="

cat > tmp.yaml << EOF
param_list:
- id: ${dataset}
  input_dataset: ${DATA_PATH}/test_data/${dataset}_processed.h5ad
  input_solution: ${DATA_PATH}/test_data/${dataset}_processed_solution.h5ad
  methods_include: ["${METHOD}"]

output_state: "state.yaml"
publish_dir: "results/${METHOD}_${dataset}"
EOF

nextflow run target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker \
  -resume \
  -params-file tmp.yaml
done
```

**Note:** `DATA_PATH` must be an absolute path to match the dataset location from the setup section.

> **Note:**
> The `hypomap` dataset often encounters out-of-memory (OOM) issues for **KBET**, but this metric is not required for normalized scoring since all control methods score zero. You may need to manually extract the score log to read the other 12 metric scores.

For the `tabula_sapiens` dataset (requires up to **500 GB RAM**, others <128 GB):

```bash
DATA_PATH=/mnt/data  # Must be absolute path to your test_data directory
METHOD=station_sota  # Replace with your method name

cat > nextflow.config << 'EOF'
process {
    withLabel: lowmem { memory = 200.GB }
    withLabel: midmem { memory = 200.GB }
    withLabel: highmem { memory = 400.GB }
    withLabel: veryhighmem { memory = 600.GB }
}
EOF

cat > tmp.yaml << EOF
param_list:
- id: tabula_sapiens
  input_dataset: ${DATA_PATH}/test_data/tabula_sapiens_processed.h5ad
  input_solution: ${DATA_PATH}/test_data/tabula_sapiens_processed_solution.h5ad
  methods_include: ["${METHOD}"]

output_state: "state.yaml"
publish_dir: "results/${METHOD}_tabula_sapiens"
EOF

nextflow run target/nextflow/workflows/run_benchmark/main.nf \
    -profile docker \
    -c nextflow.config \
    -resume \
    -params-file tmp.yaml
```

---

#### 8. Normalize Scores

Download raw performance metrics from:

```
s3://openproblems-data/resources/task_batch_integration/results/run_2025-01-23_18-03-16/score_uns.yaml
```

Normalization formula:

```
(x - x_min) / (x_max - x_min)
```

Compute `x_min` and `x_max` **per (METRIC, DATASET)**, using only **control methods**:

* `embed_cell_types`
* `embed_cell_types_jittered`
* `no_integration`
* `no_integration_batch`
* `shuffle_integration`
* `shuffle_integration_by_batch`
* `shuffle_integration_by_cell_type`

If a control method lacks a score for a (METRIC, DATASET) pair, **ignore it** (do not treat as zero).
If **all control methods** lack scores (e.g., *hypomap KBET*), normalize all your scores to **0**.

We have pre-computed the normalized range in `misc/test_norm.csv` for convenience.