#!/usr/bin/env python3
"""
Ray Tune distributed hyperparameter optimization for Multi-Dataset RNA Sequence Modeling.

This script uses Ray for distributed training across nodes, with each trial
training on a different RNA dataset (APA, Modif, PRS). 3 trials run
to evaluate performance across all datasets.

Usage: python main.py
Environment: Expects RAY_HEAD_NODE_IP environment variable for cluster connection
"""

import os
import sys
import time
import uuid
import tempfile
import numpy as np
from io import StringIO
import contextlib

# Set Ray environment variables FIRST before any imports
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
os.environ['RAY_DEDUP_LOGS'] = '0'  # Don't mask repeated logs
os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '1'
os.environ['TUNE_DISABLE_AUTO_RESULT_CALLBACKS'] = '1'
os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
os.environ['RAY_AIR_NEW_OUTPUT'] = '0'  # Use legacy output engine (no unicode box table)
os.environ['RAY_DATA_DISABLE_PROGRESS_BARS'] = '1'  # Disable progress bars

# Set environment variables before importing JAX/TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs (ERROR level only)
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'

# Force offline mode for HuggingFace datasets (use cached data only)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# Main node always gets 0 GPUs - force CPU mode to prevent CUDA initialization errors
# Ray workers will override this to use GPU when they have GPU access
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_LOG_COMPILES'] = '0'  # Suppress JAX compilation logs

# Suppress all warnings before any imports
import warnings
warnings.filterwarnings('ignore')  # Suppress ALL warnings

import jax
import jax.numpy as jnp
import optax
from functools import partial

# ============================================================================
# CONFIGURATION - CHANGE THESE AS NEEDED
# ============================================================================
NUM_SAMPLES = 7          # Number of parallel trials (one per dataset)
TRIAL_TIMEOUT = 30 * 60  # Timeout per trial in seconds (30 minutes)
TOTAL_TIMEOUT = 240 * 60  # Total timeout for all trials (4 hours for 7 datasets)
MAX_CONCURRENT = 7       # Max concurrent trials (all 7 datasets in parallel)
WARMUP_TRIALS = 0        # No warmup for grid search
SAFETY_BUFFER = 10       # Safety margin in seconds
# ============================================================================

# Dataset configurations
DATASET_CONFIGS = {
    "APA": {
        "d_output": 1,
        "task_type": "regression",
        "metric": "r2",
        "level": "sequence",
        "max_seq_len": 186
    },
    "CRI-Off": {
        "d_output": 1,
        "task_type": "regression",
        "metric": "spearman",
        "level": "sequence",
        "max_seq_len": 43
    },
    "Modif": {
        "d_output": 12,
        "task_type": "multilabel_classification",
        "metric": "auc_roc",
        "level": "sequence",
        "max_seq_len": 101
    },
    "CRI-On": {
        "d_output": 1,
        "task_type": "regression",
        "metric": "spearman",
        "level": "sequence",
        "max_seq_len": 23
    },
    "PRS": {
        "d_output": 3,
        "task_type": "multilabel_regression",
        "metric": "r2",
        "level": "sequence",
        "max_seq_len": 148
    },
    "MRL": {
        "d_output": 1,
        "task_type": "regression",
        "metric": "r2",
        "level": "sequence",
        "max_seq_len": 59
    },
    "ncRNA": {
        "d_output": 13,
        "task_type": "multiclass_classification",
        "metric": "accuracy",
        "level": "sequence",
        "max_seq_len": 100
    }
}

# Ray imports - defer until needed
try:
    import ray
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


if RAY_AVAILABLE:
    class SilentReporter(tune.ProgressReporter):
        """Custom reporter that suppresses all Ray Tune output."""
        def should_report(self, trials, done=False):
            """Never report anything."""
            return False

        def report(self, trials, done, *sys_info):
            """Do nothing when reporting."""
            pass

# Ray logging configuration (already set above)
import logging
# Changed from CRITICAL to ERROR to show Ray Tune trial failures
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("ray.worker").setLevel(logging.ERROR)
logging.getLogger("ray.serve").setLevel(logging.ERROR)
logging.getLogger("ray.train").setLevel(logging.ERROR)  # Show train module errors
logging.getLogger("ray.air").setLevel(logging.ERROR)  # Show AIR storage errors

# Suppress additional noisy loggers
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


def load_all_functions(working_dir=None, verbose=False):
    """Load all functions (defaults and submission) from current context.

    Args:
        working_dir: Optional directory to change to before importing
        verbose: If True, print which functions are from submission vs defaults

    Returns:
        dict: Dictionary containing all functions needed
    """
    if working_dir:
        original_cwd = os.getcwd()
        os.chdir(working_dir)

    # Import defaults from storage/system (relative to current working directory)
    sys.path.append('storage/system')
    from defaults import (
        default_define_hyperparameters,
        default_create_network,
        hardcoded_compute_loss,  # Use hardcoded loss
        default_create_optimizer,
        default_complete,
        BASE_SEED
    )

    # Import submission
    import submission

    # Load functions with fallback to defaults
    define_hyperparameters = getattr(submission, '_define_hyperparameters', default_define_hyperparameters)
    create_network = getattr(submission, 'create_network', default_create_network)
    # IMPORTANT: Loss is now hardcoded, agents cannot override
    compute_loss = hardcoded_compute_loss
    create_optimizer = getattr(submission, 'create_optimizer', default_create_optimizer)
    complete = getattr(submission, 'complete', default_complete)
    base_seed = getattr(submission, 'BASE_SEED', BASE_SEED)
    batch_size = getattr(submission, 'BATCH_SIZE', 64)

    # Print function sources if verbose
    if verbose:
        print("\nFunction sources:")
        print(f"  _define_hyperparameters: {'from submission' if define_hyperparameters != default_define_hyperparameters else 'using default'}")
        print(f"  create_network: {'from submission' if create_network != default_create_network else 'using default'}")
        print(f"  compute_loss: HARDCODED (agents cannot override)")
        print(f"  create_optimizer: {'from submission' if create_optimizer != default_create_optimizer else 'using default'}")
        print(f"  complete: {'from submission' if complete != default_complete else 'using default'}")
        print(f"  BASE_SEED: {base_seed} ({'from submission' if hasattr(submission, 'BASE_SEED') else 'using time-based default'}")
        print(f"  BATCH_SIZE: {batch_size} ({'from submission' if hasattr(submission, 'BATCH_SIZE') else 'using default'})")
        print("="*60)

    functions = {
        'define_hyperparameters': define_hyperparameters,
        'create_network': create_network,
        'compute_loss': compute_loss,
        'create_optimizer': create_optimizer,
        'complete': complete,
        'base_seed': base_seed,
        'batch_size': batch_size,
        # Store defaults for comparison
        '_default_define_hyperparameters': default_define_hyperparameters,
        '_default_create_network': default_create_network,
        '_default_create_optimizer': default_create_optimizer,
        '_default_complete': default_complete
    }

    if working_dir:
        os.chdir(original_cwd)

    return functions


@contextlib.contextmanager
def capture_stdout(output_buffer):
    """Context manager to capture stdout and add to output buffer."""
    old_stdout = sys.stdout
    stdout_buffer = StringIO()
    sys.stdout = stdout_buffer
    try:
        yield
    finally:
        sys.stdout = old_stdout
        captured = stdout_buffer.getvalue()
        if captured.strip():  # Only add if there's actual content
            for line in captured.rstrip().split('\n'):
                output_buffer.append(line)


def rna_trainable(config):
    """Ray Tune trainable function for RNA sequence modeling."""
    # Set JAX configuration for Ray worker process
    import os
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'

    # Force offline mode in worker
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'

    # Suppress all warnings in worker
    import warnings
    warnings.filterwarnings('ignore')  # Suppress ALL warnings

    # Suppress all logging in worker (changed to ERROR to show failures)
    import logging
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("ray.train").setLevel(logging.ERROR)
    logging.getLogger("ray.air").setLevel(logging.ERROR)
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.worker").setLevel(logging.ERROR)

    # Buffer all output to print at the end
    output_buffer = []

    with capture_stdout(output_buffer):
        # Load all functions from shared storage directory first
        shared_storage_path = config.get('_shared_storage_path')
        funcs = load_all_functions(shared_storage_path)

        # Stay in shared storage directory for the rest of execution
        os.chdir(shared_storage_path)

        # Get dataset assignment from config
        dataset_name = config.get('dataset')
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Valid datasets: {list(DATASET_CONFIGS.keys())}")

        dataset_config = DATASET_CONFIGS[dataset_name]

        # Extract base seed from config
        trial_base_seed = config.get('base_seed', funcs['base_seed'])

        # Get trial number from Ray Tune session for reproducible seeding
        tune_context = tune.get_context()
        trial_dir_name = tune_context.get_storage().trial_dir_name
        # Use dataset name hash for consistent seeding instead of trial number
        dataset_seed_offset = hash(dataset_name) % 1000
        seed = trial_base_seed + dataset_seed_offset

        # Create hparams dict from config, excluding our constants and adding dataset info
        constants = {'base_seed', '_shared_storage_path', '_submission_uuid', 'dataset'}
        hparams = {k: v for k, v in config.items() if k not in constants}

        # Add dataset-specific information to hyperparameters
        hparams.update({
            'dataset': dataset_name,
            'd_output': dataset_config['d_output'],
            'task_type': dataset_config['task_type'],
            'metric': dataset_config['metric']
        })

        print(f"\nStarting RNA training on {dataset_name} dataset")
        print(f"Task type: {dataset_config['task_type']}, Output dim: {dataset_config['d_output']}")
        print(f"Metric: {dataset_config['metric']}, Timeout: {TRIAL_TIMEOUT/60:.1f}min")

        # Set JAX to use GPU
        jax.config.update('jax_platform_name', 'gpu')

        trial_start = time.time()

        # Import the single-dataset training function
        from train_single import run_single_dataset_training

        # Run the neural network training with the specific dataset
        val_metric, final_params, final_opt_state, diagnostics = run_single_dataset_training(
            funcs=funcs,
            hparams=hparams,
            seed=seed,
            dataset_name=dataset_name,
            trial_timeout=TRIAL_TIMEOUT
        )

        # Save trial data with conflict resolution
        submission_uuid = config.get('_submission_uuid')
        trial_file_path = f"storage/shared/tmp/{submission_uuid}/trial_{dataset_name}.npz"

        # Get function sources from config (passed from main thread)
        function_sources = config.get('_function_sources', {})

        # Prepare trial data dictionary
        trial_data = {
            'val_metric': val_metric,
            'dataset': dataset_name,
            'task_type': dataset_config['task_type'],
            'metric': dataset_config['metric'],
            'hyperparameters': hparams,
            'seed': seed,
            'dataset_name': dataset_name,
            'diagnostics': diagnostics,
            'function_sources': function_sources
        }

        # Save trial results
        np.savez(trial_file_path, **trial_data)

        # Call the complete function with final params and trial data
        funcs['complete'](final_params, final_opt_state, trial_data)

        total_time = time.time() - trial_start
        print(f"\n{dataset_name} training completed:")
        print(f"Validation {dataset_config['metric']}: {val_metric:.6f}")
        print(f"Training time: {total_time:.0f}s")

    # Print all captured output directly in the worker
    for line in output_buffer:
        print(line)

    # Report results to main process
    tune.report(metrics={"val_metric": val_metric, "dataset": dataset_name, "completed": True})

    return {"val_metric": val_metric, "dataset": dataset_name}


def simple_cpu_validation():
    """Simple CPU-only validation with basic function loading, no Ray."""
    print("=== SIMPLE CPU VALIDATION ===")
    print("Validating function loading and basic compatibility")

    # Force JAX to use CPU for validation
    jax.config.update('jax_platform_name', 'cpu')

    # Load all functions
    funcs = load_all_functions()

    # Sample hyperparameters from agent's search space
    search_space = funcs['define_hyperparameters']()
    hparams = {}
    for key, space_obj in search_space.items():
        if hasattr(space_obj, 'sample'):
            hparams[key] = space_obj.sample()
        else:
            # Plain value, not a distribution - use directly
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
        seq_len = dataset_config['max_seq_len']  # Use dataset-specific sequence length
        d_input = 4  # All datasets use 4-channel one-hot encoding

        dummy_input = jnp.zeros((batch_size, seq_len, d_input), dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        params = network.init(key, dummy_input)
        output = network.apply(params, dummy_input)

        # Determine expected shape based on task_type (all are sequence-level)
        if dataset_config['task_type'] == 'regression':
            expected_shape = (batch_size,)
        elif dataset_config['task_type'] == 'multiclass_classification':
            expected_shape = (batch_size, dataset_config['d_output'])
        else:  # multilabel_classification or multilabel_regression
            expected_shape = (batch_size, dataset_config['d_output'])

        if output.shape != expected_shape:
            raise ValueError(f"{dataset_name} output shape mismatch: got {output.shape}, expected {expected_shape}")

        print(f"✓ {dataset_name} network forward pass works - output shape: {output.shape}")

        # Test optimizer creation
        learning_rate = test_hparams.get('learning_rate', 0.001)
        optimizer = funcs['create_optimizer'](learning_rate)
        print(f"✓ {dataset_name} optimizer creation works")

        # Note: Loss function is hardcoded, no need to test here

    print("Validation successful - all functions work correctly for all datasets!\n")
    return True


def run_optimization():
    """Run Ray Tune optimization."""
    if not RAY_AVAILABLE:
        raise ImportError("Ray is not installed. Please install with: pip install ray[tune]")

    print("=== FULL RAY TRAINING MODE ===")
    start_time = time.time()

    # Initialize Ray cluster (suppress verbose output)
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

    # Load all functions with verbose logging
    funcs = load_all_functions(verbose=True)

    # Get function sources (do this once in main thread)
    function_sources = {
        '_define_hyperparameters': 'submission' if funcs['define_hyperparameters'] != funcs['_default_define_hyperparameters'] else 'default',
        'create_network': 'submission' if funcs['create_network'] != funcs['_default_create_network'] else 'default',
        'compute_loss': 'hardcoded',
        'create_optimizer': 'submission' if funcs['create_optimizer'] != funcs['_default_create_optimizer'] else 'default',
        'complete': 'submission' if funcs['complete'] != funcs['_default_complete'] else 'default'
    }

    # Get search space from agent
    search_space = funcs['define_hyperparameters']()

    # Convert plain values to Ray Tune search spaces if necessary
    for key, value in search_space.items():
        if not hasattr(value, 'sample'):  # Plain value, not a tune distribution
            search_space[key] = tune.choice([value])  # Wrap in tune.choice with single option

    # Create submission-unique temp directory
    submission_uuid = str(uuid.uuid4())
    submission_temp_dir = f'storage/shared/tmp/{submission_uuid}'
    os.makedirs(submission_temp_dir, exist_ok=True)

    # Add constants to search space - CRUCIAL: Use grid_search for guaranteed dataset assignment
    search_space.update({
        'dataset': tune.grid_search(["APA", "CRI-Off", "Modif", "CRI-On", "PRS", "MRL", "ncRNA"]),  # Guarantees 7 trials, one per dataset
        'base_seed': funcs['base_seed'],
        '_shared_storage_path': os.path.abspath('.'),  # Pass current directory to workers
        '_submission_uuid': submission_uuid,  # Pass unique submission ID to workers
        '_function_sources': function_sources  # Pass function sources to workers
    })

    mode_str = "MULTI-DATASET RNA TRAINING"
    print("="*60)
    print(f"Starting Ray Tune RNA Sequence Modeling ({mode_str})")
    print(f"Configuration: 7 datasets (APA, CRI-Off, Modif, CRI-On, PRS, MRL, ncRNA) for multi-task evaluation")
    print(f"Parallel trials: {MAX_CONCURRENT}, Total trials: 7")
    print(f"Timeout: {TRIAL_TIMEOUT/60:.1f} min per trial, {TOTAL_TIMEOUT/60:.1f} min total")
    print("="*60)

    # Ensure Ray Tune storage directory exists and is accessible to all nodes
    ray_storage_path = os.path.abspath('.')  # Current shared directory
    ray_results_dir = os.path.join(ray_storage_path, "ray_results")
    os.makedirs(ray_results_dir, exist_ok=True)

    # Configure Ray Tune run - Use BasicVariantGenerator for grid search (no search algorithm needed)
    tuner = tune.Tuner(
        tune.with_resources(
            rna_trainable,
            resources={"cpu": 1, "gpu": 1}  # Each trial requests 1 GPU
        ),
        tune_config=tune.TuneConfig(
            metric="val_metric",
            mode="max",  # Maximize validation metrics (R2, Spearman, AUC-ROC are all "higher is better")
            num_samples=1,  # Since we use grid_search, this means 1 sample per grid combination
            time_budget_s=TOTAL_TIMEOUT - SAFETY_BUFFER,
            max_concurrent_trials=MAX_CONCURRENT
        ),
        param_space=search_space,
        run_config=tune.RunConfig(
            name=f"rna_multi_dataset_{submission_uuid[:8]}",
            storage_path=ray_storage_path,  # Use shared storage accessible to all nodes
            checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=False),
            failure_config=tune.FailureConfig(max_failures=5),  # Allow 5 retries per dataset
            verbose=0,  # Suppress Ray Tune's verbose output
            progress_reporter=SilentReporter()  # Use custom silent reporter
        )
    )

    # Run the optimization
    print("\nStarting Ray Tune multi-dataset training...")
    results = tuner.fit()


    # Simple completion reporting - all evaluation logic moved to auto evaluator
    print(f"SUBMISSION_UUID: {submission_uuid}")  # Key line for evaluator to parse
    print(f"Total time: {time.time() - start_time:.0f}s")
    print("="*60)

    ray.shutdown()
    return results


def main():
    """Main function - runs simple validation first, then full Ray training."""
    # First run simple CPU validation
    validation_success = simple_cpu_validation()
    if not validation_success:
        print("Validation failed, stopping.")
        return None

    # Then run full Ray training
    print("Starting full Ray multi-dataset training...")
    result = run_optimization()
    print("Full training completed!")
    return result


if __name__ == "__main__":
    # CRITICAL: Prevent recursive Ray initialization in workers
    # Ray workers may re-execute main.py when trying to deserialize functions defined here
    # This happens when agent submissions modify sys.path at import time (e.g., Principia's bag_of_kmers.py)
    if os.environ.get('RAY_WORKER_MODE'):
        # We're inside a Ray worker that's trying to re-import main.py
        # Exit immediately to prevent infinite recursion
        sys.exit(0)

    # Check if we're already inside a Ray runtime (alternative detection)
    if 'RAY_ADDRESS' in os.environ and os.environ.get('RAY_RUNTIME_STARTED'):
        # Another sign we're in a worker context
        sys.exit(0)

    # Set flag to prevent recursion in child processes
    os.environ['RAY_WORKER_MODE'] = '1'

    # Check for test function first (same as original main.py)
    import submission
    if hasattr(submission, 'test'):
        print("=== Test Mode Detected ===")
        print("Running test() function from submission...")
        test_result = submission.test()
        print(f"Test completed. Result: {test_result}")
        print("=== Test Mode Complete ===")
        sys.exit(0)

    # Normal Ray optimization mode
    result = main()