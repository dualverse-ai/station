#!/usr/bin/env python3
"""
Test set evaluation script for ZAPBench neural activity forecasting.

Usage:
    python eval.py eval/eval_1234.py

This script:
1. Trains 3 seeds in parallel using Ray (same as main submission)
2. Selects the best model based on validation MAE
3. Evaluates on test set for multiple prediction horizons (1, 4, 8, 16, 32)
4. Reports individual seed statistics and mean/std
"""

import os
import sys
import time
import uuid
import numpy as np
import argparse
import importlib.util
import contextlib
import gc
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

# Import from storage/system
from storage.system.main import load_all_functions, SilentReporter
from storage.system.train_single import (
    INPUT_HORIZON, OUTPUT_HORIZON, compute_vanilla_mae
)

# Add output capture from main.py
@contextlib.contextmanager
def capture_stdout(output_buffer):
    """Pass through stdout directly (no buffering for debugging)."""
    yield  # Just pass through, don't capture anything
from storage.system.defaults import BASE_SEED

# Configuration (same as main.py)
NUM_SAMPLES = 3
TRIAL_TIMEOUT = 300 * 60  # 300 min per trial
TOTAL_TIMEOUT = 1000 * 60  # 1000 min total
MAX_CONCURRENT = 3
WARMUP_TRIALS = 1
SAFETY_BUFFER = 10

# Test evaluation horizons
TEST_HORIZONS = [1, 4, 8, 16, 32]


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


def eval_zapbench_trainable(config):
    """Custom trainable that loads submission from shared directory."""
    # Set JAX configuration for Ray worker process
    import os
    import sys
    os.environ['JAX_PLATFORMS'] = 'cpu'
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

    # Extract BASE_SEED and trial ID from config
    trial_base_seed = config.get('base_seed', funcs['base_seed'])
    trial_number = config.get('_trial_id', 0)  # Explicitly passed trial ID

    # Create hparams dict from config (excluding our constants)
    constants = {'base_seed', '_trial_id', '_shared_storage_path', '_submission_uuid', '_shared_eval_dir', '_max_epoch', '_patience'}
    hparams = {k: v for k, v in config.items() if k not in constants}

    # Buffer all output to print at the end
    output_buffer = []

    with capture_stdout(output_buffer):
        # Get max_epoch and patience from config
        max_epoch = config.get('_max_epoch', 120)
        patience = config.get('_patience', 20)

        print(f"\n{'='*60}")
        print(f"Ray Actor - Starting Trial {trial_number}")
        print(f"Max Epochs: {max_epoch}")
        print(f"Patience: {patience}")
        print(f"{'='*60}")

        # Set JAX to use GPU
        jax.config.update('jax_platform_name', 'gpu')

        import time
        trial_start = time.time()

        # Import the single-seed training function (eval version with flexible epochs)
        from storage.system.eval_train_single import run_single_seed_training

        # Run the neural network training with the specific seed
        seed = trial_base_seed + trial_number
        val_mae, final_params, final_opt_state, step_maes = run_single_seed_training(
            funcs=funcs,
            hparams=hparams,
            seed=seed,
            trial_timeout=300*60,  # 300 min timeout
            patience=patience,
            max_epochs=max_epoch
        )

        # Save trial data
        import numpy as np
        submission_uuid = config.get('_submission_uuid')
        trial_file_path = f"storage/shared/tmp/{submission_uuid}/trial_{trial_number}.npz"
        trial_counter = trial_number
        while os.path.exists(trial_file_path):
            trial_counter += 1
            trial_file_path = f"storage/shared/tmp/{submission_uuid}/trial_{trial_counter}.npz"

        # Clear GPU memory before test evaluation to prevent OOM
        # Move final_params to CPU and clear GPU memory
        final_params_cpu = jax.tree_util.tree_map(lambda x: np.array(x), final_params)
        final_opt_state_cpu = jax.tree_util.tree_map(lambda x: np.array(x), final_opt_state)
        del final_params
        del final_opt_state
        gc.collect()
        jax.clear_caches()  # Clear JAX compilation caches

        # Evaluate on test set after clearing memory
        # Load test data and condition labels with forced copy to avoid network I/O
        # Use ZAPBENCH_DATA_PATH environment variable, fallback to /mnt/stephen/data/zapbench/data if not set
        zapbench_data_path = os.environ.get('ZAPBENCH_DATA_PATH', '/mnt/stephen/data/zapbench/data')
        test_data_path = os.path.join(zapbench_data_path, 'test_data.npy')
        test_labels_path = os.path.join(zapbench_data_path, 'test_condition_labels.npy')

        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data not found at {test_data_path}. Please set ZAPBENCH_DATA_PATH environment variable to the correct data directory.")

        test_data_np = np.load(test_data_path)
        test_data_np = np.array(test_data_np, copy=True)  # Force copy into RAM
        test_data = jnp.array(test_data_np)
        test_condition_labels = np.load(test_labels_path)
        test_condition_labels = np.array(test_condition_labels, copy=True)  # Force copy into RAM

        # Create network for evaluation and move params back to GPU
        network = funcs['create_network'](hparams)
        final_params = jax.tree_util.tree_map(lambda x: jnp.array(x), final_params_cpu)

        # Evaluate on test set for multiple horizons - do ONE forward pass for all horizons
        horizons = [1, 4, 8, 16, 32]
        batch_size = funcs['batch_size']  # Use same batch size as training

        # Compute both standard MAE and per-condition MAE
        test_results = {}
        test_results_per_condition = {}

        # 1. Standard MAE (corrected method - MAE at specific timesteps)
        horizon_maes = {h: 0.0 for h in horizons}
        all_timesteps_mae_sum = 0.0  # NEW: For same-weight average across all 32 timesteps
        num_batches = 0

        from storage.system.train_single import compute_vanilla_mae

        # Custom create_batches that respects condition boundaries
        def create_batches_condition_aware(data, condition_labels, input_horizon, output_horizon, batch_size):
            """Create batches that don't cross condition boundaries."""
            if condition_labels is None:
                raise ValueError("condition_labels is REQUIRED to prevent cross-condition contamination")

            all_inputs = []
            all_outputs = []

            # Process each condition separately
            unique_conditions = np.unique(condition_labels)
            for condition in unique_conditions:
                # Get indices for this condition
                condition_mask = condition_labels == condition
                condition_indices = np.where(condition_mask)[0]

                if len(condition_indices) < input_horizon + output_horizon:
                    continue

                # Find contiguous segments for this condition
                # (condition labels are already contiguous by construction)
                start_idx = condition_indices[0]
                end_idx = condition_indices[-1] + 1

                # Create sequences within this condition
                for i in range(start_idx, end_idx - input_horizon - output_horizon + 1):
                    # Verify this sequence doesn't cross boundaries
                    seq_labels = condition_labels[i:i+input_horizon+output_horizon]
                    if np.all(seq_labels == condition):
                        all_inputs.append(data[i:i+input_horizon])
                        all_outputs.append(data[i+input_horizon:i+input_horizon+output_horizon])

            # Keep as numpy arrays in CPU memory - convert to JAX only per batch
            all_inputs = np.stack(all_inputs)
            all_outputs = np.stack(all_outputs)

            # Create batches and convert to JAX arrays only when yielding
            num_samples = len(all_inputs)
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                # Convert only this batch to JAX arrays on GPU
                yield jnp.array(all_inputs[i:batch_end]), jnp.array(all_outputs[i:batch_end])
        for x_batch, y_batch in create_batches_condition_aware(test_data, test_condition_labels, 4, 32, batch_size):
            # Single forward pass for 32 steps
            predictions = network.apply(final_params, x_batch)

            # Compute MAE at each specific horizon timestep (not averaged)
            for horizon in horizons:
                # MAE at step h (0-indexed: horizon-1)
                pred_step_h = predictions[:, horizon-1:horizon, :]  # Only step h
                target_step_h = y_batch[:, horizon-1:horizon, :]    # Only step h
                mae = compute_vanilla_mae(pred_step_h, target_step_h)
                # Move to CPU immediately to free GPU memory
                horizon_maes[horizon] += float(mae)

            # NEW: Compute MAE across all 32 timesteps with equal weight
            all_timesteps_mae = compute_vanilla_mae(predictions, y_batch)
            all_timesteps_mae_sum += float(all_timesteps_mae)

            num_batches += 1

        # Average across batches for standard MAE
        for horizon in horizons:
            test_results[horizon] = float(horizon_maes[horizon] / num_batches)

        # NEW: Store same-weight average MAE (averaged across all 32 timesteps)
        test_results['all_timesteps'] = float(all_timesteps_mae_sum / num_batches)

        # 2. Per-condition MAE (paper's method)
        unique_conditions = np.unique(test_condition_labels)
        condition_maes = {cond: {h: 0.0 for h in horizons} for cond in unique_conditions}

        for condition in unique_conditions:
            # Get data for this condition only
            condition_mask = test_condition_labels == condition
            condition_data = test_data[condition_mask]

            if len(condition_data) == 0:
                continue

            # Evaluate on this condition's data
            condition_horizon_maes = {h: 0.0 for h in horizons}
            condition_num_batches = 0

            # Create condition-specific batches (already filtered, so no cross-contamination)
            # But we still need to ensure sequences don't go beyond this condition's data
            condition_inputs = []
            condition_outputs = []

            for i in range(len(condition_data) - 4 - 32 + 1):
                condition_inputs.append(condition_data[i:i+4])
                condition_outputs.append(condition_data[i+4:i+4+32])

            if len(condition_inputs) == 0:
                condition_num_batches = 0
                continue

            # Keep as numpy arrays in CPU memory
            condition_inputs = np.stack(condition_inputs)
            condition_outputs = np.stack(condition_outputs)

            # Create batches and convert to JAX only when needed
            for i in range(0, len(condition_inputs), batch_size):
                batch_end = min(i + batch_size, len(condition_inputs))
                # Convert only this batch to JAX arrays on GPU
                x_batch = jnp.array(condition_inputs[i:batch_end])
                y_batch = jnp.array(condition_outputs[i:batch_end])

                # Single forward pass for 32 steps
                predictions = network.apply(final_params, x_batch)

                # Compute MAE at each specific horizon timestep (not averaged)
                for horizon in horizons:
                    # MAE at step h (0-indexed: horizon-1)
                    pred_step_h = predictions[:, horizon-1:horizon, :]  # Only step h
                    target_step_h = y_batch[:, horizon-1:horizon, :]    # Only step h
                    mae = compute_vanilla_mae(pred_step_h, target_step_h)
                    # Move to CPU immediately to free GPU memory
                    condition_horizon_maes[horizon] += float(mae)

                condition_num_batches += 1

            # Average across batches for this condition
            for horizon in horizons:
                if condition_num_batches > 0:
                    condition_maes[condition][horizon] = float(condition_horizon_maes[horizon] / condition_num_batches)

        # Average across conditions (grand average)
        for horizon in horizons:
            condition_values = [condition_maes[cond][horizon] for cond in unique_conditions if condition_maes[cond][horizon] > 0]
            test_results_per_condition[horizon] = float(np.mean(condition_values)) if condition_values else 0.0

        # Calculate total time before creating trial data
        total_time = time.time() - trial_start

        # Create trial data with both test results
        trial_data = {
            'val_mae': val_mae,
            'hyperparameters': hparams,
            'seed': seed,
            'trial_number': trial_counter,
            'step_maes': step_maes,  # Validation step MAEs from training
            'test_results': test_results,
            'test_results_per_condition': test_results_per_condition,
            'condition_maes': condition_maes,  # Detailed per-condition breakdown
            'training_time': total_time  # Add training time to trial data
        }

        # Save trial data with test results
        np.savez(trial_file_path, **trial_data)

        # Call complete function (using CPU versions that we saved)
        funcs['complete'](final_params_cpu, final_opt_state_cpu, trial_data)

        print(f"\nTrial {trial_number} completed:")
        print(f"Validation MAE: {val_mae:.6f}")
        print(f"Training time: {total_time:.0f}s")
        print(f"Test MAE by horizon: {test_results}")

    # Print all captured output at once (outside capture context)
    for line in output_buffer:
        print(line)

    # Simple Ray report
    tune.report(metrics={"val_mae": val_mae, "completed": True})

    return {"val_mae": val_mae}


def run_ray_training(funcs, submission_uuid, shared_eval_dir, max_epoch=120, patience=20):
    """Run Ray Tune optimization with 3 seeds in parallel.

    Args:
        funcs: Functions loaded from submission
        submission_uuid: Unique ID for this submission
        shared_eval_dir: Shared directory containing submission.py
        max_epoch: Maximum number of training epochs
        patience: Early stopping patience
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

    # Use grid search with explicit trial IDs to ensure exactly 3 different seeds
    # This prevents seed collisions from Ray's trial naming
    search_space.update({
        '_trial_id': tune.grid_search([0, 1, 2]),  # Explicit trial IDs
        'base_seed': funcs['base_seed'],
        '_submission_uuid': submission_uuid,
        '_shared_eval_dir': shared_eval_dir,
        '_max_epoch': max_epoch,
        '_patience': patience
    })

    print("="*60)
    print(f"Starting Ray Tune Training")
    print(f"Configuration: 3 trials with explicit seed assignment")
    print(f"Seeds: {funcs['base_seed']+0}, {funcs['base_seed']+1}, {funcs['base_seed']+2}")
    print(f"Parallel trials: {MAX_CONCURRENT}")
    print(f"Timeout: {TRIAL_TIMEOUT/60:.0f} min per trial")
    print("="*60)

    # Configure Ray Tune run with grid search
    tuner = tune.Tuner(
        tune.with_resources(
            eval_zapbench_trainable,
            resources={"cpu": 1, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="val_mae",
            mode="min",
            time_budget_s=TOTAL_TIMEOUT - SAFETY_BUFFER,
            max_concurrent_trials=MAX_CONCURRENT
        ),
        param_space=search_space,
        run_config=tune.RunConfig(
            name=f"zapbench_eval_{submission_uuid[:8]}",
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
                'val_mae': float(data['val_mae']),
                'hyperparameters': data['hyperparameters'].item() if data['hyperparameters'].ndim == 0 else data['hyperparameters'],
                'seed': int(data['seed']),
                'trial_number': int(data['trial_number']),
                'test_results': data['test_results'].item(),  # Standard MAE
                'test_results_per_condition': data['test_results_per_condition'].item(),  # Per-condition MAE
                'condition_maes': data['condition_maes'].item(),  # Detailed breakdown
                'training_time': float(data['training_time']),  # Training time in seconds
                'file_path': trial_path
            })
    else:
        print(f"Trial results directory not found: {submission_temp_dir}")

    ray.shutdown()
    return trial_results, submission_temp_dir


def count_parameters(params):
    """Count total number of trainable parameters in a JAX parameter pytree."""
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    return param_count


def simple_cpu_validation(funcs):
    """Simple CPU-only validation with basic function loading, no Ray/Optuna."""
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
            # Plain value, not a distribution - use directly
            hparams[key] = space_obj

    # Test network creation
    network = funcs['create_network'](hparams)
    print("✓ Network creation works")

    # Test basic forward pass
    num_neurons = 71721  # ZAPBench dataset size
    input_horizon = 4
    output_horizon = 32
    batch_size = 4

    dummy_input = jnp.zeros((batch_size, input_horizon, num_neurons), dtype=jnp.float32)
    key = jax.random.PRNGKey(42)
    params = network.init(key, dummy_input)
    output = network.apply(params, dummy_input)
    print(f"✓ Network forward pass works - output shape: {output.shape}")

    # Count and display parameter count
    param_count = count_parameters(params)
    print(f"✓ Network parameter count: {param_count:,} parameters ({param_count/1e6:.2f}M)")

    # Test optimizer creation
    learning_rate = hparams.get('learning_rate', 0.001)
    optimizer = funcs['create_optimizer'](learning_rate)
    print("✓ Optimizer creation works")

    # Test compute_loss function
    print("Testing compute_loss...")
    dummy_target = jnp.zeros((batch_size, output_horizon, num_neurons), dtype=jnp.float32)
    # First get predictions from network
    predictions = network.apply(params, dummy_input)
    # Then compute loss with new signature
    loss = funcs['compute_loss'](predictions, dummy_target, params, dummy_input)
    print(f"✓ Compute loss works - loss value: {float(loss):.6f}")

    print("✓ All functions validated successfully!")
    return True


class TeeOutput:
    """Redirect stdout to both console and file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def main(submission_path, batch_size_override=None, max_epoch=120, patience=20):
    """Main evaluation pipeline."""
    # Setup log file
    submission_path_obj = Path(submission_path)
    log_file = submission_path_obj.parent / (submission_path_obj.stem + '.log')

    # Redirect stdout to both console and log file
    tee = TeeOutput(log_file)
    original_stdout = sys.stdout
    sys.stdout = tee

    print("="*80)
    print("ZAPBench Test Set Evaluation")
    print(f"Submission: {submission_path}")
    print(f"Log file: {log_file}")
    if batch_size_override is not None:
        print(f"Batch Size Override: {batch_size_override}")
    print(f"Max Epochs: {max_epoch}")
    print(f"Patience: {patience}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    try:
        # Create a unique shared directory for this evaluation
        import shutil
        eval_uuid = str(uuid.uuid4())
        # Use EVAL_SHARED_DIR environment variable, fallback to /tmp if not set
        shared_eval_base = os.environ.get('EVAL_SHARED_DIR', '/tmp')
        if not os.path.exists(shared_eval_base):
            print(f"Warning: EVAL_SHARED_DIR '{shared_eval_base}' does not exist, using /tmp")
            shared_eval_base = '/tmp'
        shared_eval_dir = f"{shared_eval_base}/zapbench_eval_{eval_uuid}"
        os.makedirs(shared_eval_dir, exist_ok=True)
        print(f"\n1. Created shared evaluation directory: {shared_eval_dir}")

        # Copy submission to shared directory where all Ray workers can access it
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

        # Copy eval_train_single.py from misc to shared storage/system
        eval_train_single_src = os.path.join(os.path.dirname(__file__), 'eval_train_single.py')
        eval_train_single_dest = os.path.join(storage_dest, 'eval_train_single.py')
        shutil.copy(eval_train_single_src, eval_train_single_dest)
        print(f"  Copied eval_train_single.py to shared directory for flexible epoch control")

        # Copy training data to shared directory
        data_dest_dir = os.path.join(storage_dest, 'data')
        os.makedirs(data_dest_dir, exist_ok=True)

        # Copy symlinks to actual data files
        # Use ZAPBENCH_DATA_PATH environment variable, fallback to /mnt/stephen/data/zapbench/data if not set
        zapbench_data_path = os.environ.get('ZAPBENCH_DATA_PATH', '/mnt/stephen/data/zapbench/data')

        train_data_src = os.path.join(zapbench_data_path, 'train_data.npy')
        val_data_src = os.path.join(zapbench_data_path, 'val_data.npy')
        train_labels_src = os.path.join(zapbench_data_path, 'train_condition_labels.npy')
        val_labels_src = os.path.join(zapbench_data_path, 'val_condition_labels.npy')

        # Check if paths exist
        if not os.path.exists(train_data_src):
            raise FileNotFoundError(f"Training data not found at {zapbench_data_path}. Please set ZAPBENCH_DATA_PATH environment variable to the correct data directory.")

        train_data_dest = os.path.join(data_dest_dir, 'train_data.npy')
        val_data_dest = os.path.join(data_dest_dir, 'val_data.npy')
        train_labels_dest = os.path.join(data_dest_dir, 'train_condition_labels.npy')
        val_labels_dest = os.path.join(data_dest_dir, 'val_condition_labels.npy')

        os.symlink(train_data_src, train_data_dest)
        os.symlink(val_data_src, val_data_dest)
        os.symlink(train_labels_src, train_labels_dest)
        os.symlink(val_labels_src, val_labels_dest)

        # Create storage/shared/tmp directory for results
        shared_tmp = os.path.join(shared_research_dir, 'storage', 'shared', 'tmp')
        os.makedirs(shared_tmp, exist_ok=True)

        print(f"  Copied research infrastructure and data links to shared directory")

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

            # Override batch_size if specified
            if batch_size_override is not None:
                print(f"\n   Overriding batch_size: {funcs.get('batch_size', 'N/A')} -> {batch_size_override}")
                funcs['batch_size'] = batch_size_override

            # Run CPU validation first
            print("\n4. Running CPU validation...")
            simple_cpu_validation(funcs)

            # Create unique submission ID
            submission_uuid = str(uuid.uuid4())

            # Run Ray training with 3 seeds
            print("\n5. Training 3 seeds in parallel with Ray...")
            trial_results, temp_dir = run_ray_training(funcs, submission_uuid, shared_eval_dir, max_epoch, patience)

            if not trial_results:
                print("ERROR: No trial results found!")
                return

            # Find best trial based on validation MAE
            best_trial = min(trial_results, key=lambda x: x['val_mae'])

            print("\n" + "="*60)
            print("Training Phase Results:")
            print("="*60)
            for i, trial in enumerate(sorted(trial_results, key=lambda x: x['seed'])):
                is_best = trial == best_trial
                marker = " [BEST]" if is_best else ""
                print(f"  Seed {trial['seed']}: Val MAE = {trial['val_mae']:.6f}{marker}")

            # Test results were already computed in each trial
            print("\n6. Collecting test set results from trials...")

            all_seed_results = []
            for trial in sorted(trial_results, key=lambda x: x['seed']):
                all_seed_results.append({
                    'seed': trial['seed'],
                    'val_mae': trial['val_mae'],
                    'test_results': trial['test_results'],
                    'test_results_per_condition': trial['test_results_per_condition'],
                    'condition_maes': trial['condition_maes'],
                    'training_time': trial['training_time']
                })
                print(f"  Seed {trial['seed']}: Test results collected")

            # Compute statistics across seeds
            print("\n" + "="*80)
            print("TEST SET EVALUATION RESULTS")
            print("="*80)

            # Individual seed results - simplified to just Val MAE and training time
            print("\nIndividual Seed Results:")
            print("-" * 80)
            val_maes = [sr['val_mae'] for sr in all_seed_results]
            training_times = [sr['training_time'] for sr in all_seed_results]

            for seed_result in all_seed_results:
                print(f"Seed {seed_result['seed']}: Val MAE = {seed_result['val_mae']:.6f}, Training Time = {seed_result['training_time']:.0f}s")

            mean_val_mae = np.mean(val_maes)
            mean_training_time = np.mean(training_times)
            print(f"\nAverage: Val MAE = {mean_val_mae:.6f}, Training Time = {mean_training_time:.0f}s")

            # Aggregate statistics for both methods
            print("\n" + "="*80)
            print("STANDARD MAE - Aggregate Statistics (mean ± std across seeds):")
            print("="*80)
            print(f"{'Horizon':<10} {'MAE (mean ± std)':<25} {'Individual Values':<50}")
            print("-" * 80)

            for horizon in TEST_HORIZONS:
                maes = [sr['test_results'][horizon] for sr in all_seed_results]
                mean_mae = np.mean(maes)
                std_mae = np.std(maes)
                individual_str = ", ".join([f"{mae:.6f}" for mae in maes])
                print(f"{horizon:<10} {mean_mae:.6f} ± {std_mae:.6f}      [{individual_str}]")

            print("\n" + "="*80)
            print("PER-CONDITION MAE - Aggregate Statistics (mean ± std across seeds):")
            print("="*80)
            print(f"{'Horizon':<10} {'MAE (mean ± std)':<25} {'Individual Values':<50}")
            print("-" * 80)

            # Collect mean values for averaging across horizons
            horizon_means = []
            for horizon in TEST_HORIZONS:
                maes = [sr['test_results_per_condition'][horizon] for sr in all_seed_results]
                mean_mae = np.mean(maes)
                std_mae = np.std(maes)
                individual_str = ", ".join([f"{mae:.6f}" for mae in maes])
                print(f"{horizon:<10} {mean_mae:.6f} ± {std_mae:.6f}      [{individual_str}]")
                horizon_means.append(mean_mae)

            # Print mean across all horizons
            overall_mean = np.mean(horizon_means)
            print(f"\nMean across horizons: {overall_mean:.6f}")

            # NEW: Print same-weight average MAE (averaged across all 32 timesteps)
            all_timesteps_maes = [sr['test_results']['all_timesteps'] for sr in all_seed_results]
            mean_all_timesteps = np.mean(all_timesteps_maes)
            std_all_timesteps = np.std(all_timesteps_maes)
            individual_all_timesteps_str = ", ".join([f"{mae:.6f}" for mae in all_timesteps_maes])
            print(f"Same-weight average MAE (all 32 timesteps): {mean_all_timesteps:.6f} ± {std_all_timesteps:.6f}      [{individual_all_timesteps_str}]")

            # Print copy-paste friendly format
            print("\nCopy-paste format:")
            print(",".join([str(h) for h in TEST_HORIZONS]))
            print(",".join([f"{m:.6f}" for m in horizon_means]))

            # Overall summary
            print("\n" + "="*60)
            print("Summary:")
            print("="*60)
            print(f"Best validation model: Seed {best_trial['seed']} (Val MAE: {best_trial['val_mae']:.6f})")

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
            # Inner finally - restore working directory
            os.chdir(original_dir)

    finally:
        # Clean up shared evaluation directory
        try:
            shutil.rmtree(shared_eval_dir)
            print(f"Cleaned up shared evaluation directory: {shared_eval_dir}")
        except:
            pass

        # Restore stdout and close log file
        sys.stdout = original_stdout
        tee.close()
        print(f"Results saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ZAPBench submission on test set')
    parser.add_argument('submission', type=str, help='Path to submission file (e.g., eval/eval_1234.py)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (optional, overrides submission batch_size)')
    parser.add_argument('--max_epoch', type=int, default=120,
                        help='Maximum number of training epochs (default: 120)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')

    args = parser.parse_args()
    main(args.submission, batch_size_override=args.batch_size, max_epoch=args.max_epoch, patience=args.patience)
