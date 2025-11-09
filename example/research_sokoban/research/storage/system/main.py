#!/usr/bin/env python3
"""
Ray Tune distributed hyperparameter optimization for Sokoban RL task.

This script uses Ray for distributed training across nodes, with each trial 
requesting 1 GPU and training a single seed of the RL agent. 4 trials run
in parallel to get statistical robustness from multiple seeds.

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

# Ray imports
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter

# RL-specific imports 
from functools import partial


class SilentReporter(tune.ProgressReporter):
    """Custom reporter that suppresses all Ray Tune output."""
    def should_report(self, trials, done=False):
        """Never report anything."""
        return False
    
    def report(self, trials, done, *sys_info):
        """Do nothing when reporting."""
        pass


# Suppress Optuna logging except for warnings/errors
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# Configuration
SAFETY_BUFFER = 10       # 10 seconds safety margin
TOTAL_TIMEOUT = 180 * 60  # 2 hour total for full run
TRIAL_TIMEOUT = 45 * 60   # 30 minutes per trial

# Fixed parameters for Sokoban RL
NUM_SAMPLES = 4  # 4 seeds for statistical robustness
MAX_CONCURRENT = 4  # All 4 run in parallel
WARMUP_TRIALS = 4  # No hyperparameter tuning, just 4 seeds

# RL Training parameters
CONFIG = {
    'num_envs': 64,
    'unroll_length': 20,
    'total_timesteps': 50_000_000,
    'time_limit': 120,
    'seed': 42,
    'max_training_minutes': 30,  # 30 minutes per seed
    'timeout_check_interval': 100,
}


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
        default_training_step,
        default_create_optimizer,
        default_complete,
        BASE_SEED
    )
    
    # Import submission
    import submission
    
    # Load functions with fallback to defaults
    define_hyperparameters = getattr(submission, '_define_hyperparameters', default_define_hyperparameters)
    create_network = getattr(submission, 'create_network', default_create_network)
    training_step = getattr(submission, 'training_step', default_training_step)
    create_optimizer = getattr(submission, 'create_optimizer', default_create_optimizer)
    complete = getattr(submission, 'complete', default_complete)
    base_seed = getattr(submission, 'BASE_SEED', BASE_SEED)
    
    # Print function sources if verbose
    if verbose:
        print("\nFunction sources:")
        print(f"  _define_hyperparameters: {'from submission' if define_hyperparameters != default_define_hyperparameters else 'using default'}")
        print(f"  create_network: {'from submission' if create_network != default_create_network else 'using default'}")
        print(f"  training_step: {'from submission' if training_step != default_training_step else 'using default'}")
        print(f"  create_optimizer: {'from submission' if create_optimizer != default_create_optimizer else 'using default'}")
        print(f"  complete: {'from submission' if complete != default_complete else 'using default'}")
        print(f"  BASE_SEED: {base_seed} ({'from submission' if hasattr(submission, 'BASE_SEED') else 'using time-based default'})")
        print("="*60)
    
    functions = {
        'define_hyperparameters': define_hyperparameters,
        'create_network': create_network,
        'training_step': training_step,
        'create_optimizer': create_optimizer,
        'complete': complete,
        'base_seed': base_seed
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


def sokoban_trainable(config):
    """Ray Tune trainable function for Sokoban RL training."""
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
        
        # Extract BASE_SEED from config (passed by our space function)
        trial_base_seed = config.get('base_seed', funcs['base_seed'])
        
        # Get trial number from Ray Tune session for reproducible seeding
        tune_context = tune.get_context()
        trial_dir_name = tune_context.get_storage().trial_dir_name
        # Extract sequential trial number from dir name like "sokoban_trainable_abc123_2_param1=0.5_date"
        # The trial number is after the trial_id and before the parameters
        parts = trial_dir_name.split('_')
        trial_number = 0  # fallback
        for i, part in enumerate(parts):
            if part.isdigit():
                trial_number = int(part)
                break
        
        # Create hparams dict from config (excluding our constants)
        constants = {'base_seed', '_shared_storage_path', '_submission_uuid'}
        hparams = {k: v for k, v in config.items() if k not in constants}
        
        print(f"\nStarting Sokoban RL training - Seed {trial_number} with timeout {TRIAL_TIMEOUT/60:.1f}min")
        
        # Set JAX to use GPU
        jax.config.update('jax_platform_name', 'gpu')
        
        trial_start = time.time()
        
        # Import the single-seed training function
        from train_single import run_single_seed_training
        
        # Run the RL training with the specific seed
        seed = trial_base_seed + trial_number
        solve_rate, final_params, final_opt_state = run_single_seed_training(
            funcs=funcs,
            hparams=hparams,
            seed=seed,
            config=CONFIG,
            trial_timeout=TRIAL_TIMEOUT
        )

        # Save trial data with conflict resolution
        submission_uuid = config.get('_submission_uuid')
        trial_file_path = f"storage/shared/tmp/{submission_uuid}/trial_{trial_number}.npz"
        trial_counter = trial_number
        while os.path.exists(trial_file_path):
            trial_counter += 1
            trial_file_path = f"storage/shared/tmp/{submission_uuid}/trial_{trial_counter}.npz"

        # Prepare trial data dictionary
        trial_data = {
            'solve_rate': solve_rate,
            'hyperparameters': hparams,
            'seed': seed,
            'trial_number': trial_counter
        }

        # Save trial results
        np.savez(trial_file_path, **trial_data)

        # Call the complete function with final params and trial data
        funcs['complete'](final_params, final_opt_state, trial_data)
        
        total_time = time.time() - trial_start
        print(f"\nSeed {trial_number} completed:")
        print(f"Solve rate: {solve_rate:.3f}")
        print(f"Training time: {total_time:.0f}s")
    
    # Print all captured output at once (outside capture context)
    for line in output_buffer:
        print(line)
    
    # Simple Ray report without checkpoint
    tune.report(metrics={"solve_rate": solve_rate, "completed": True})
    
    return {"solve_rate": solve_rate}


def simple_cpu_validation():
    """Simple CPU-only validation with basic function loading, no Ray/Optuna."""
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
            print(f"Warning: {key} doesn't have sample() method")
    
    # Test network creation
    network = funcs['create_network'](hparams)
    print("✓ Network creation works")
    
    # Test basic forward pass
    dummy_obs = jnp.zeros((4, 8, 8, 8), dtype=jnp.float32)
    dummy_done = jnp.zeros((4,), dtype=bool)
    key = jax.random.PRNGKey(42)
    params = network.init(key, dummy_obs, dummy_done)
    outputs = network.apply(params, dummy_obs, dummy_done)
    print(f"✓ Network forward pass works - {len(outputs)} outputs")
    
    # Test optimizer creation
    optimizer = funcs['create_optimizer']()
    print("✓ Optimizer creation works")

    # Test training_step function
    print("Testing training_step...")
    # Create a dummy batch for testing
    batch_size = 4
    sequence_length = 20
    dummy_batch = {
        'observations': jnp.zeros((batch_size, sequence_length, 8, 8, 8), dtype=jnp.float32),
        'actions': jnp.zeros((batch_size, sequence_length), dtype=jnp.int32),
        'rewards': jnp.zeros((batch_size, sequence_length), dtype=jnp.float32),
        'dones': jnp.zeros((batch_size, sequence_length), dtype=bool),
        'values': jnp.zeros((batch_size, sequence_length), dtype=jnp.float32),
        'final_values': jnp.zeros(batch_size, dtype=jnp.float32),
        'initial_rnn_states': None  # Optional RNN states
    }

    # Initialize optimizer state
    opt_state = optimizer.init(params)

    # Test training_step - let exceptions pass through for debugger
    new_params, new_opt_state = funcs['training_step'](
        network, optimizer, params, opt_state, dummy_batch, hparams
    )

    # Verify outputs are valid
    if new_params is None:
        raise ValueError("training_step returned None for params")
    if new_opt_state is None:
        raise ValueError("training_step returned None for opt_state")

    print("✓ Training step works")

    # Test complete function
    print("Testing complete function...")
    # Create dummy trial data
    dummy_trial_data = {
        'solve_rate': 0.5,
        'hyperparameters': hparams,
        'seed': 42,
        'trial_number': 0
    }

    # Call complete function - let exceptions pass through for debugger
    funcs['complete'](params, opt_state, dummy_trial_data)
    print("✓ Complete function works")

    print("Validation successful - all functions work correctly!\n")

    return True


def run_optimization():
    """Run Ray Tune optimization."""
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
    
    # Get search space from agent
    search_space = funcs['define_hyperparameters']()
    
    # Create submission-unique temp directory
    submission_uuid = str(uuid.uuid4())
    submission_temp_dir = f'storage/shared/tmp/{submission_uuid}'
    os.makedirs(submission_temp_dir, exist_ok=True)
    
    # Add constants to search space
    # Use RESEARCH_STORAGE_HOST_PATH if set (for Docker mode with external Ray workers)
    # Otherwise use the absolute path of current directory (for sandbox/direct mode)
    storage_path = os.environ.get('RESEARCH_STORAGE_HOST_PATH', os.path.abspath('.'))

    search_space.update({
        'base_seed': funcs['base_seed'],
        '_shared_storage_path': storage_path,  # Pass correct path to Ray workers
        '_submission_uuid': submission_uuid  # Pass unique submission ID to workers
    })

    if os.environ.get('RESEARCH_STORAGE_HOST_PATH'):
        print(f"Using host storage path for Ray workers: {storage_path}")
    
    # Create Optuna search algorithm
    sampler = optuna.samplers.TPESampler(n_startup_trials=WARMUP_TRIALS, seed=funcs['base_seed'])
    optuna_search = OptunaSearch(
        metric="solve_rate",
        mode="max",  # Maximize solve rate
        sampler=sampler
    )
    
    # Limit concurrent trials
    search_alg = ConcurrencyLimiter(optuna_search, max_concurrent=MAX_CONCURRENT)
    
    mode_str = "FULL RAY TRAINING"
    print("="*60)
    print(f"Starting Ray Tune Sokoban RL Optimization ({mode_str})")
    print(f"Configuration: 4 seeds, {CONFIG['total_timesteps']:,} steps per seed")
    print(f"Parallel trials: {MAX_CONCURRENT}, Total trials: {NUM_SAMPLES}")
    print(f"Timeout: {TRIAL_TIMEOUT/60:.1f} min per trial, {TOTAL_TIMEOUT/60:.1f} min total")
    print("="*60)
    
    # Function source logging is handled by load_all_functions(verbose=True)
    
    # Ensure Ray Tune storage directory exists and is accessible to all nodes
    # Use the same storage path logic as above for consistency
    ray_storage_path = os.environ.get('RESEARCH_STORAGE_HOST_PATH', os.path.abspath('.'))
    ray_results_dir = os.path.join(ray_storage_path, "ray_results")
    os.makedirs(ray_results_dir, exist_ok=True)
    
    # Configure Ray Tune run
    tuner = tune.Tuner(
        tune.with_resources(
            sokoban_trainable,
            resources={"cpu": 1, "gpu": 1}  # Each trial requests 1 GPU
        ),
        tune_config=tune.TuneConfig(
            metric="solve_rate",
            mode="max",
            search_alg=search_alg,
            num_samples=NUM_SAMPLES,
            time_budget_s=TOTAL_TIMEOUT - SAFETY_BUFFER,
            max_concurrent_trials=MAX_CONCURRENT
        ),
        param_space=search_space,
        run_config=tune.RunConfig(
            name=f"sokoban_ray_optimization_{submission_uuid[:8]}",
            storage_path=ray_storage_path,  # Use shared storage accessible to all nodes
            stop={"solve_rate": 1.0, "completed": True},  # Stop if perfect solve rate achieved
            checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=False),
            failure_config=tune.FailureConfig(max_failures=2),  # Auto-retry failed trials on different nodes
            verbose=0,  # Suppress Ray Tune's verbose output
            progress_reporter=SilentReporter()  # Use custom silent reporter
        )
    )
    
    # Run the optimization
    print("\nStarting Ray Tune optimization...")
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
    print("Starting full Ray training run...")
    result = run_optimization()
    print("Full training completed!")
    return result


if __name__ == "__main__":
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