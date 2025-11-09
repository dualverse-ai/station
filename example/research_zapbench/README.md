## Task Specification

This is the Batch Integration of Single Cell RNA Sequencing Data dataset.

## Setup

### Environment

We need a custom conda environment, `batch_integration` for this task; please run this in `misc` folder to setup the conda env:

```bash
bash setup_batch_integration_env.sh
```

after which, do `conda activate batch_integration`

### Dataset

To download and process the training and the 6 testing dataset:

```bash
cd example/research_zapbench/misc
pip install awscli
python prepare_data.py --dir /mnt/data  # Replace with your dataset directory
```

If successful, your dataset direcotry should have:

Final Directory Structure:

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

### Setting up Station Data

Finally, do:

```bash
conda activate batch_integration
export CONDA_ENV_NAME=batch_integration
cp {DATA_PATH}/human_train.h5ad station_data/rooms/research/storage/system
cp {DATA_PATH}/human_train_solution.h5ad station_data/rooms/research/storage/system
echo "AUTO_START: true" >> station_data/constant_config.yaml
```

## External Evaluation

To evaluate a submission externally, place the standalone script in the `misc/eval` folder (e.g., `eval_123.py`), then run the following command from the `misc` folder:

```bash
export EVAL_SHARED_DIR=/path/to/shared/tmp  # Required: Must be accessible by all Ray nodes (use /tmp for single-node)
export ZAPBENCH_DATA_PATH=/path/to/zapbench/data  # Optional: Path to ZAPBench data (default: /mnt/stephen/data/zapbench/data)
bash eval.sh eval_123.py test_data_path
```

where test_data_path is the folder you download the test_data that contains the h5ad like `dkd_processed.h5ad`, e.g. `/mnt/data/test_data`