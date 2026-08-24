#!/usr/bin/env python3
"""
Test set evaluation script for Multi-Dataset RNA Sequence Modeling.

Usage:
    python eval.py eval/eval_1234.py

This script:
1. Trains on 3 datasets in parallel using Ray (same as main submission)
2. Selects the best model based on validation performance for each dataset
3. Evaluates on test set for all 3 datasets
4. Reports individual dataset scores and primary score (average)
"""

import os
import sys
import time
import uuid
import numpy as np
import argparse
import importlib.util
import contextlib
import shutil
from pathlib import Path

# Add parent research directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research'))

# Set Ray environment variables FIRST before any imports
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '1'
os.environ['TUNE_DISABLE_AUTO_RESULT_CALLBACKS'] = '1'
os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
os.environ['RAY_AIR_NEW_OUTPUT'] = '0'
os.environ['RAY_DATA_DISABLE_PROGRESS_BARS'] = '1'

# Set environment variables before importing JAX/TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_LOG_COMPILES'] = '0'

# Suppress all warnings before any imports
import warnings
warnings.filterwarnings('ignore')

import jax
import jax.numpy as jnp
from jax import random
import optax

# Import Ray
import ray
from ray import tune

# Dataset configurations (expanded from main.py)
DATASET_CONFIGS = {
    "APA": {
        "d_output": 1,
        "task_type": "regression",
        "metric": "r2",
        "data_dir": "Isoform",
        "seq_col": "seq",
        "label_col": "proximal_isoform_proportion",
        "level": "sequence",
        "max_seq_len": 186
    },
    "CRI-Off": {
        "d_output": 1,
        "task_type": "regression",
        "metric": "spearman",
        "data_dir": "CRISPROffTarget",
        "seq_col": "concat",  # Special: sgrna + target
        "label_col": "label",
        "level": "sequence",
        "max_seq_len": 46  # 23 + 23 (sgrna + target, both 23bp)
    },
    "Modif": {
        "d_output": 12,
        "task_type": "multilabel_classification",
        "metric": "auc_roc",
        "data_dir": "Modification",
        "seq_col": "sequence",
        "label_col": "label",
        "level": "sequence",
        "max_seq_len": 101
    },
    "CRI-On": {
        "d_output": 1,
        "task_type": "regression",
        "metric": "spearman",
        "data_dir": "CRISPROnTarget",
        "seq_col": "seq",
        "label_col": "label",
        "level": "sequence",
        "max_seq_len": 23
    },
    "PRS": {
        "d_output": 3,
        "task_type": "multilabel_regression",
        "metric": "r2",
        "data_dir": "ProgrammableRNASwitches",
        "seq_col": "sequence",
        "label_col": "multi",  # Special: ON, OFF, ON_OFF
        "level": "sequence",
        "max_seq_len": 148
    },
    "MRL": {
        "d_output": 1,
        "task_type": "regression",
        "metric": "r2",
        "data_dir": "MeanRibosomeLoading",
        "seq_col": "seq",
        "label_col": "label",
        "level": "sequence",
        "max_seq_len": 100
    },
    "ncRNA": {
        "d_output": 13,  # Number of ncRNA families
        "task_type": "multiclass_classification",
        "metric": "accuracy",
        "data_dir": "NoncodingRNAFamily",
        "seq_col": "sequence",
        "label_col": "label",
        "level": "sequence",
        "max_seq_len": 1182
    }
}

# Import from storage/system
from storage.system.main import load_all_functions, SilentReporter
from storage.system.train_single import (
    run_single_dataset_training, load_dataset, compute_metric,
    create_batches, MAX_EPOCHS
)

# Configuration (expanded for 7 datasets)
NUM_SAMPLES = 1  # Single seed per dataset
TRIAL_TIMEOUT = 300 * 60  # 300 min per trial
TOTAL_TIMEOUT = 300 * 60 * 7  # 300 min per trial * 7 datasets (35 hours total)
MAX_CONCURRENT = 7  # 7 datasets in parallel
SAFETY_BUFFER = 10

# Global variables for override parameters (set via command line)
OVERRIDE_BATCH_SIZE = None
OVERRIDE_MAX_EPOCHS = None
OVERRIDE_PATIENCE = None
OVERRIDE_SEED = None

from storage.system.defaults import BASE_SEED


@contextlib.contextmanager
def capture_stdout(output_buffer):
    """Pass through stdout directly (no buffering for debugging)."""
    yield  # Just pass through, don't capture anything


def load_submission_module(submission_path):
    """Load submission file as a Python module."""
    submission_path = Path(submission_path).resolve()
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    # Load as module
    spec = importlib.util.spec_from_file_location("submission", submission_path)
    submission = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = submission
    spec.loader.exec_module(submission)

    return submission


def rna_eval_trainable(config):
    """Custom trainable that trains on one dataset and evaluates on both val and test."""
    # Set JAX configuration for Ray worker process
    import os
    import sys
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'

    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore')

    # Suppress logging
    import logging
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("ray.train").setLevel(logging.ERROR)
    logging.getLogger("ray.air").setLevel(logging.ERROR)
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.worker").setLevel(logging.ERROR)

    # Get shared directory from config
    shared_eval_dir = config.get('_shared_eval_dir')
    shared_research_dir = os.path.join(shared_eval_dir, 'research')

    # Add shared directory to path to import submission
    sys.path.insert(0, shared_eval_dir)

    # Change to shared research directory and load functions
    os.chdir(shared_research_dir)
    from storage.system.main import load_all_functions
    funcs = load_all_functions(shared_research_dir)

    # Apply override parameters if specified
    import storage.system.train_single as train_single_module
    if config.get('_override_max_epochs') is not None:
        train_single_module.MAX_EPOCHS = config['_override_max_epochs']
    if config.get('_override_patience') is not None:
        train_single_module.EARLY_STOP_PATIENCE = config['_override_patience']

    # Extract configuration
    trial_base_seed = config.get('base_seed', funcs['base_seed'])
    dataset_name = config.get('dataset')
    seed_offset = config.get('seed_offset', 0)

    # Override batch_size if specified
    if config.get('_override_batch_size') is not None:
        funcs['batch_size'] = config['_override_batch_size']

    # Dataset configurations (need to define in worker scope)
    DATASET_CONFIGS_WORKER = {
        "APA": {
            "d_output": 1,
            "task_type": "regression",
            "metric": "r2",
            "data_dir": "Isoform",
            "level": "sequence",
            "max_seq_len": 186
        },
        "CRI-Off": {
            "d_output": 1,
            "task_type": "regression",
            "metric": "spearman",
            "data_dir": "CRISPROffTarget",
            "level": "sequence",
            "max_seq_len": 46
        },
        "Modif": {
            "d_output": 12,
            "task_type": "multilabel_classification",
            "metric": "auc_roc",
            "data_dir": "Modification",
            "level": "sequence",
            "max_seq_len": 101
        },
        "CRI-On": {
            "d_output": 1,
            "task_type": "regression",
            "metric": "spearman",
            "data_dir": "CRISPROnTarget",
            "level": "sequence",
            "max_seq_len": 23
        },
        "PRS": {
            "d_output": 3,
            "task_type": "multilabel_regression",
            "metric": "r2",
            "data_dir": "ProgrammableRNASwitches",
            "level": "sequence",
            "max_seq_len": 148
        },
        "MRL": {
            "d_output": 1,
            "task_type": "regression",
            "metric": "r2",
            "data_dir": "MeanRibosomeLoading",
            "level": "sequence",
            "max_seq_len": 100
        },
        "ncRNA": {
            "d_output": 13,
            "task_type": "multiclass_classification",
            "metric": "accuracy",
            "data_dir": "NoncodingRNAFamily",
            "level": "sequence",
            "max_seq_len": 1182
        }
    }

    # Create hparams dict from config (excluding our constants)
    constants = {'base_seed', '_shared_storage_path', '_submission_uuid', '_shared_eval_dir', 'dataset', 'seed_offset',
                 '_override_batch_size', '_override_max_epochs', '_override_patience'}
    hparams = {k: v for k, v in config.items() if k not in constants}

    # Add dataset-specific information to hyperparameters
    dataset_config = DATASET_CONFIGS_WORKER[dataset_name]
    hparams.update({
        'dataset': dataset_name,
        'd_output': dataset_config['d_output'],
        'task_type': dataset_config['task_type'],
        'metric': dataset_config['metric'],
        'level': dataset_config['level'],
        'max_seq_len': dataset_config['max_seq_len']
    })

    # Buffer all output to print at the end
    output_buffer = []

    with capture_stdout(output_buffer):
        print(f"\nStarting training - Dataset {dataset_name}, Seed offset {seed_offset}")

        # Set JAX to use GPU
        jax.config.update('jax_platform_name', 'gpu')

        import time
        trial_start = time.time()

        # Import the single-dataset training function
        from storage.system.train_single import run_single_dataset_training

        # Run the neural network training with the specific seed
        seed = trial_base_seed + seed_offset
        val_metric, final_params, final_opt_state = run_single_dataset_training(
            funcs=funcs,
            hparams=hparams,
            seed=seed,
            dataset_name=dataset_name,
            trial_timeout=TRIAL_TIMEOUT
        )

        # Now evaluate on test set
        print(f"\nEvaluating {dataset_name} on test set...")

        # Load test data
        from storage.system.train_single import load_dataset
        import pandas as pd

        data_dir = f"storage/system/data/{dataset_config['data_dir']}"
        test_df = pd.read_csv(f"{data_dir}/test.csv")

        # Handle dataset-specific sequence preprocessing
        if dataset_name == "CRI-Off":
            test_sequences = (test_df['sgrna'] + test_df['target']).values
        elif dataset_name == "APA":
            test_sequences = test_df['seq'].values
        elif dataset_name == "Modif":
            test_sequences = test_df['sequence'].values
        elif dataset_name == "CRI-On":
            test_sequences = test_df['seq'].values
        elif dataset_name == "PRS":
            test_sequences = test_df['sequence'].values
        elif dataset_name == "MRL":
            test_sequences = test_df['seq'].values
        elif dataset_name == "ncRNA":
            test_sequences = test_df['sequence'].values
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # One-hot encode sequences
        from storage.system.train_single import one_hot_encode_rna_sequences
        test_X = one_hot_encode_rna_sequences(test_sequences)

        # Process labels based on task type
        if dataset_config['task_type'] == 'regression':
            if dataset_name == "APA":
                test_y = test_df['proximal_isoform_proportion'].values.astype(np.float32)
            elif dataset_name in ["CRI-Off", "CRI-On", "MRL"]:
                test_y = test_df['label'].values.astype(np.float32)
        elif dataset_config['task_type'] == 'multilabel_classification':
            from storage.system.train_single import process_multilabel_labels
            test_y = process_multilabel_labels(test_df['label'].values)
        elif dataset_config['task_type'] == 'multilabel_regression':
            # PRS dataset: Multi-label regression
            test_y = test_df[['ON', 'OFF', 'ON_OFF']].values.astype(np.float32)
        elif dataset_config['task_type'] == 'multiclass_classification':
            # ncRNA dataset: Multi-class classification
            test_y = test_df['label'].values.astype(np.int64)

        test_X = jnp.array(test_X)
        test_y = jnp.array(test_y)

        # Evaluate on test set
        network = funcs['create_network'](hparams)
        batch_size = funcs.get('batch_size', 64)

        batched_test_X, batched_test_y = create_batches(test_X, test_y, batch_size)

        all_predictions = []
        all_targets = []

        for batch_X, batch_y in zip(batched_test_X, batched_test_y):
            predictions = network.apply(final_params, batch_X, deterministic=True)
            all_predictions.append(predictions)
            all_targets.append(batch_y)

        # Concatenate and trim to original test size
        all_predictions = jnp.concatenate(all_predictions)[:len(test_y)]
        all_targets = jnp.concatenate(all_targets)[:len(test_y)]

        # Compute test metric
        from storage.system.train_single import compute_metric
        test_metric = compute_metric(
            dataset_config['task_type'],
            dataset_config['metric'],
            all_predictions,
            all_targets
        )

        # Save trial data
        submission_uuid = config.get('_submission_uuid')
        trial_file_path = f"storage/shared/tmp/{submission_uuid}/trial_{dataset_name}_{seed_offset}.npz"

        trial_data = {
            'dataset': dataset_name,
            'seed': seed,
            'seed_offset': seed_offset,
            'val_metric': val_metric,
            'test_metric': test_metric,
            'task_type': dataset_config['task_type'],
            'metric': dataset_config['metric'],
            'hyperparameters': hparams
        }

        np.savez(trial_file_path, **trial_data)

        # Call complete function
        funcs['complete'](final_params, final_opt_state, trial_data)

        total_time = time.time() - trial_start
        print(f"\n{dataset_name} completed:")
        print(f"Validation {dataset_config['metric']}: {val_metric:.6f}")
        print(f"Test {dataset_config['metric']}: {test_metric:.6f}")
        print(f"Training time: {total_time:.0f}s")

    # Print all captured output at once
    for line in output_buffer:
        print(line)

    # Simple Ray report
    tune.report(metrics={"val_metric": val_metric, "test_metric": test_metric, "completed": True})

    return {"val_metric": val_metric, "test_metric": test_metric}


def run_ray_training(funcs, submission_uuid, shared_eval_dir):
    """Run Ray Tune optimization with 3 datasets in parallel (single seed per dataset).

    Args:
        funcs: Functions loaded from submission
        submission_uuid: Unique ID for this submission
        shared_eval_dir: Shared directory containing submission.py
    """
    print("=== RAY PARALLEL TRAINING ===")
    start_time = time.time()

    # Initialize Ray cluster
    ray_address = os.environ.get('RAY_HEAD_NODE_IP')
    if ray_address:
        print(f"Connecting to Ray cluster at {ray_address}")
        ray.init(address=ray_address, logging_config=ray.LoggingConfig(encoding="TEXT"))
    else:
        print("RAY_HEAD_NODE_IP not set, initializing Ray locally")
        ray.init(logging_config=ray.LoggingConfig(encoding="TEXT"))

    print("Ray cluster initialized successfully")
    resources = ray.cluster_resources()
    num_cpus = resources.get('CPU', 0)
    num_gpus = resources.get('GPU', 0)
    print(f"Available resources: {num_cpus:.0f} CPUs, {num_gpus:.0f} GPUs")

    # Get search space from agent
    search_space = funcs['define_hyperparameters']()

    # Convert plain values to Ray Tune search spaces if necessary
    for key, value in search_space.items():
        if not hasattr(value, 'sample'):
            search_space[key] = tune.choice([value])

    # Create submission-unique temp directory
    submission_temp_dir = f'storage/shared/tmp/{submission_uuid}'
    os.makedirs(submission_temp_dir, exist_ok=True)

    # Add constants to search space - grid search over datasets only (single seed)
    search_space.update({
        'dataset': tune.grid_search(["APA", "CRI-Off", "Modif", "CRI-On", "PRS", "MRL", "ncRNA"]),  # 7 datasets
        'seed_offset': 0,  # Single seed
        'base_seed': OVERRIDE_SEED if OVERRIDE_SEED is not None else funcs['base_seed'],
        '_submission_uuid': submission_uuid,
        '_shared_eval_dir': shared_eval_dir,
        '_override_batch_size': OVERRIDE_BATCH_SIZE,
        '_override_max_epochs': OVERRIDE_MAX_EPOCHS,
        '_override_patience': OVERRIDE_PATIENCE
    })

    print("="*60)
    print(f"Starting Ray Tune RNA Training")
    print(f"Configuration: 7 datasets × 1 seed = 7 trials")
    print(f"Parallel trials: {MAX_CONCURRENT}")
    print(f"Timeout: {TRIAL_TIMEOUT/60:.0f} min per trial")
    print("="*60)

    # Configure Ray Tune run
    tuner = tune.Tuner(
        tune.with_resources(
            rna_eval_trainable,
            resources={"cpu": 1, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="val_metric",
            mode="max",
            num_samples=1,  # Grid search will create 9 trials automatically
            time_budget_s=TOTAL_TIMEOUT - SAFETY_BUFFER,
            max_concurrent_trials=MAX_CONCURRENT
        ),
        param_space=search_space,
        run_config=tune.RunConfig(
            name=f"rna_eval_{submission_uuid[:8]}",
            storage_path=os.path.abspath('.'),
            stop={"completed": True},
            checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=False),
            failure_config=tune.FailureConfig(max_failures=2),
            verbose=0,
            progress_reporter=SilentReporter()
        )
    )

    # Run the optimization
    print("\nStarting Ray Tune optimization...")
    results = tuner.fit()

    print(f"Training completed in {time.time() - start_time:.0f}s")

    # Collect trial results
    trial_results = []
    if os.path.exists(submission_temp_dir):
        trial_files = os.listdir(submission_temp_dir)
        trial_files = [f for f in trial_files if f.startswith('trial_') and f.endswith('.npz')]
        trial_files.sort()

        print(f"Found {len(trial_files)} trial result files in {submission_temp_dir}")

        for trial_file in trial_files:
            trial_path = os.path.join(submission_temp_dir, trial_file)
            print(f"Loading trial results from {trial_path}")
            data = np.load(trial_path, allow_pickle=True)
            trial_results.append({
                'dataset': str(data['dataset']),
                'seed': int(data['seed']),
                'seed_offset': int(data['seed_offset']),
                'val_metric': float(data['val_metric']),
                'test_metric': float(data['test_metric']),
                'task_type': str(data['task_type']),
                'metric': str(data['metric']),
                'hyperparameters': data['hyperparameters'].item() if data['hyperparameters'].ndim == 0 else data['hyperparameters'],
                'file_path': trial_path
            })
    else:
        print(f"Trial results directory not found: {submission_temp_dir}")

    ray.shutdown()
    return trial_results, submission_temp_dir


def simple_cpu_validation(funcs):
    """Simple CPU-only validation with basic function loading, no Ray."""
    print("=== SIMPLE CPU VALIDATION ===")
    print("Validating function loading and basic compatibility")

    # Force JAX to use CPU for validation
    jax.config.update('jax_platform_name', 'cpu')

    # Sample hyperparameters from agent's search space
    search_space = funcs['define_hyperparameters']()
    hparams = {}
    for key, space_obj in search_space.items():
        if hasattr(space_obj, 'sample'):
            hparams[key] = space_obj.sample()
        else:
            hparams[key] = space_obj

    # Test each dataset configuration
    for dataset_name, dataset_config in DATASET_CONFIGS.items():
        print(f"\nTesting {dataset_name} configuration...")

        # Add dataset-specific info to hyperparameters
        test_hparams = hparams.copy()
        test_hparams.update({
            'dataset': dataset_name,
            'd_output': dataset_config['d_output'],
            'task_type': dataset_config['task_type'],
            'metric': dataset_config['metric'],
            'level': dataset_config['level'],
            'max_seq_len': dataset_config['max_seq_len']
        })

        # Test network creation
        network = funcs['create_network'](test_hparams)
        print(f"✓ {dataset_name} network creation works")

        # Test basic forward pass
        batch_size = 4
        seq_len = min(dataset_config['max_seq_len'], 100)  # Use actual max_seq_len but cap at 100 for efficiency

        dummy_input = jnp.zeros((batch_size, seq_len, 4), dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        params = network.init(key, dummy_input)
        output = network.apply(params, dummy_input)

        expected_shape = (batch_size,) if dataset_config['task_type'] == 'regression' else (batch_size, dataset_config['d_output'])
        if output.shape != expected_shape:
            raise ValueError(f"{dataset_name} output shape mismatch: got {output.shape}, expected {expected_shape}")

        print(f"✓ {dataset_name} network forward pass works - output shape: {output.shape}")

    print("✓ All functions validated successfully!")
    return True


def main(submission_path):
    """Main evaluation pipeline."""
    print("="*80)
    print("Multi-Dataset RNA Sequence Modeling - Test Set Evaluation")
    print(f"Submission: {submission_path}")
    print("="*80)

    # Create a unique shared directory for this evaluation
    eval_uuid = str(uuid.uuid4())
    # Use EVAL_SHARED_DIR environment variable, fallback to /tmp if not set
    shared_eval_base = os.environ.get('EVAL_SHARED_DIR', '/tmp')
    if not os.path.exists(shared_eval_base):
        raise FileNotFoundError(f"EVAL_SHARED_DIR '{shared_eval_base}' does not exist. Please set EVAL_SHARED_DIR to a valid shared directory accessible by all Ray nodes.")
    shared_eval_dir = f"{shared_eval_base}/rna_eval_{eval_uuid}"
    os.makedirs(shared_eval_dir, exist_ok=True)
    print(f"\n1. Created shared evaluation directory: {shared_eval_dir}")

    # Copy submission to shared directory
    submission_dest = os.path.join(shared_eval_dir, 'submission.py')
    shutil.copy(submission_path, submission_dest)
    print(f"  Copied submission to shared directory for Ray workers")

    # Copy necessary directories and files to shared directory
    research_dir = os.path.join(os.path.dirname(__file__), '..', 'research')
    shared_research_dir = os.path.join(shared_eval_dir, 'research')

    # Copy storage/system directory structure
    storage_src = os.path.join(research_dir, 'storage', 'system')
    storage_dest = os.path.join(shared_research_dir, 'storage', 'system')
    shutil.copytree(storage_src, storage_dest)

    # Replace train_single.py with eval_train_single.py (supports 7 datasets)
    eval_train_single_src = os.path.join(os.path.dirname(__file__), 'eval_train_single.py')
    train_single_dest = os.path.join(storage_dest, 'train_single.py')
    shutil.copy(eval_train_single_src, train_single_dest)

    print(f"  Replaced train_single.py with eval_train_single.py (7 datasets)")

    # Create symlink to data directory
    # Use RNA_DATA_PATH environment variable, fallback to /mnt/stephen/data/rna/data if not set
    rna_data_path = os.environ.get('RNA_DATA_PATH', '/mnt/stephen/data/rna/data')
    if not os.path.exists(rna_data_path):
        raise FileNotFoundError(f"RNA data not found at {rna_data_path}. Please set RNA_DATA_PATH environment variable to the correct data directory.")
    data_dest = os.path.join(storage_dest, 'data')
    os.symlink(rna_data_path, data_dest)
    print(f"  Created symlink to RNA data at {rna_data_path}")

    # Create storage/shared/tmp directory for results
    shared_tmp = os.path.join(shared_research_dir, 'storage', 'shared', 'tmp')
    os.makedirs(shared_tmp, exist_ok=True)

    print(f"  Copied research infrastructure to shared directory")

    # Load submission
    print("\n2. Loading submission module...")
    submission = load_submission_module(submission_path)

    # Change to shared research directory for proper imports
    original_dir = os.getcwd()
    os.chdir(shared_research_dir)

    try:
        # Load all functions with submission
        print("\n3. Loading functions from submission...")
        funcs = load_all_functions(verbose=True)

        # Run CPU validation first
        print("\n4. Running CPU validation...")
        simple_cpu_validation(funcs)

        # Create unique submission ID
        submission_uuid = str(uuid.uuid4())

        # Run Ray training with 7 datasets
        print("\n5. Training 7 datasets in parallel with Ray...")
        trial_results, temp_dir = run_ray_training(funcs, submission_uuid, shared_eval_dir)

        if not trial_results:
            print("ERROR: No trial results found!")
            return

        # Organize results by dataset
        datasets = ["APA", "CRI-Off", "Modif", "CRI-On", "PRS", "MRL", "ncRNA"]
        results_by_dataset = {ds: [] for ds in datasets}

        for trial in trial_results:
            results_by_dataset[trial['dataset']].append(trial)

        print("\n" + "="*80)
        print("TEST SET EVALUATION RESULTS")
        print("="*80)

        # Individual dataset results (single seed per dataset)
        print(f"{'Dataset':<15} {'Metric':<12} {'Seed':<10} {'Val Score':<15} {'Test Score':<15}")
        print("-" * 80)

        test_scores = []
        for dataset_name in datasets:
            dataset_trials = results_by_dataset[dataset_name]

            if not dataset_trials:
                print(f"\nWARNING: No results found for dataset {dataset_name}")
                continue

            trial = dataset_trials[0]  # Single seed
            metric_name = trial['metric'].upper()

            print(f"{dataset_name:<15} {metric_name:<12} {trial['seed']:<10} {trial['val_metric']:<15.6f} {trial['test_metric']:<15.6f}")
            test_scores.append(trial['test_metric'])

        # Overall primary score (average across datasets)
        if test_scores:
            primary_score = np.mean(test_scores)
            print("\n" + "="*80)
            print(f"PRIMARY SCORE (average across {len(test_scores)} datasets): {primary_score:.6f}")
            print("="*80)

            # Print scores on a single line separated by commas
            print("\nScores (APA, CRI-Off, Modif, CRI-On, PRS, MRL, ncRNA):")
            scores_str = ", ".join([f"{score:.5f}" for score in test_scores])
            print(scores_str)

        # Clean up temp directory
        print(f"\nCleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

        print("\n" + "="*80)
        print("Evaluation completed successfully!")
        print("="*80)

    finally:
        # Clean up shared evaluation directory
        try:
            shutil.rmtree(shared_eval_dir)
            print(f"Cleaned up shared evaluation directory: {shared_eval_dir}")
        except:
            pass
        os.chdir(original_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RNA submission on test set')
    parser.add_argument('submission', type=str, help='Path to submission file (e.g., eval/eval_1234.py)')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size (if not set, uses value from submission)')
    parser.add_argument('--max_epochs', type=int, default=None, help='Override max epochs (if not set, uses default from train_single)')
    parser.add_argument('--patience', type=int, default=None, help='Override early stopping patience (if not set, uses default from train_single)')
    parser.add_argument('--seed', type=int, default=42, help=f'Base seed for training (default: {BASE_SEED})')

    args = parser.parse_args()

    # Set global override variables (using globals() to modify module-level variables)
    import sys
    current_module = sys.modules[__name__]
    current_module.OVERRIDE_BATCH_SIZE = args.batch_size
    current_module.OVERRIDE_MAX_EPOCHS = args.max_epochs
    current_module.OVERRIDE_PATIENCE = args.patience
    current_module.OVERRIDE_SEED = args.seed

    # Print override information
    if args.batch_size is not None:
        print(f"Override: batch_size = {args.batch_size}")
    if args.max_epochs is not None:
        print(f"Override: max_epochs = {args.max_epochs}")
    if args.patience is not None:
        print(f"Override: patience = {args.patience}")
    if args.seed is not None:
        print(f"Override: seed = {args.seed}")

    main(args.submission)
