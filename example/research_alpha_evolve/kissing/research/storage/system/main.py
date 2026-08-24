#!/usr/bin/env python3
"""
Ray Tune distributed hyperparameter optimization for Kissing Number task.

This script uses Ray for distributed training across nodes, with each trial 
requesting 1 GPU and using Optuna backend for hyperparameter search.

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

# Main node always gets 0 GPUs - force CPU mode to prevent CUDA initialization errors
# Ray workers will override this to use GPU when they have GPU access
os.environ['JAX_PLATFORMS'] = 'cpu'

os.environ['JAX_LOG_COMPILES'] = '0'  # Suppress JAX compilation logs

# Suppress fork() warning for Ray + JAX
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*os.fork.*')

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
import optuna
from functools import partial

# Ray imports
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter

class SilentReporter(tune.ProgressReporter):
    """Custom reporter that suppresses all Ray Tune output."""
    def should_report(self, trials, done=False):
        """Never report anything."""
        return False

    def report(self, trials, done, *sys_info):
        """Do nothing when reporting."""
        pass

# Suppress Optuna logging except for warnings/errors
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ray logging configuration (already set above)
import logging
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("ray.worker").setLevel(logging.ERROR)
logging.getLogger("ray.serve").setLevel(logging.ERROR)
logging.getLogger("ray.train").setLevel(logging.ERROR)  # Suppress train module logs
logging.getLogger("ray.air").setLevel(logging.ERROR)  # Suppress AIR storage logs

# Suppress additional noisy loggers
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)  
logging.getLogger("absl").setLevel(logging.ERROR)

# Configuration
SAFETY_BUFFER = 10       # 10 seconds safety margin
TOTAL_TIMEOUT = 180 * 60  # 3 hours total for full run
TRIAL_TIMEOUT = 12 * 60   # 12 minutes per trial

# Fixed parameters
DIM = 11
N = 593  # Fixed number of spheres
BATCH_SIZE = 128
MINI_BATCH_SIZE = 32  # Process in mini-batches for better performance
NUM_MINI_BATCHES = BATCH_SIZE // MINI_BATCH_SIZE  # 4 mini-batches
MAX_STEPS = 1000000
NUM_SAMPLES = 32
MAX_CONCURRENT = 8
WARMUP_TRIALS = 8

def _run_test_mode(submission_module):
    print("=== Test Mode Detected ===")
    print("Running test() function from submission...")
    test_result = submission_module.test()
    print(f"Test completed. Result: {test_result}")
    print("=== Test Mode Complete ===")


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
        default_get_optimizer,
        default_get_init_fn,
        default_update_func,
        default_post_hook,
        BASE_SEED,
        N as DEFAULT_N,
        compute_margin_batch
    )
    
    # Import submission
    import submission
    
    # Load functions with fallback to defaults
    define_hyperparameters = getattr(submission, '_define_hyperparameters', default_define_hyperparameters)
    get_optimizer = getattr(submission, '_get_optimizer', default_get_optimizer)
    get_init_fn = getattr(submission, '_get_init_fn', default_get_init_fn)
    update_func = getattr(submission, '_update_func', default_update_func)
    post_hook = getattr(submission, '_post_hook', default_post_hook)
    base_seed = getattr(submission, 'BASE_SEED', BASE_SEED)
    n = getattr(submission, 'N', DEFAULT_N)  # Allow submission to override N
    use_float64 = bool(getattr(submission, 'USE_FLOAT64', False))
    
    # Print function sources if verbose
    if verbose:
        print("\nFunction sources:")
        print(f"  _define_hyperparameters: {'from submission' if define_hyperparameters != default_define_hyperparameters else 'using default'}")
        print(f"  _get_optimizer: {'from submission' if get_optimizer != default_get_optimizer else 'using default'}")
        print(f"  _get_init_fn: {'from submission' if get_init_fn != default_get_init_fn else 'using default'}")
        print(f"  _update_func: {'from submission' if update_func != default_update_func else 'using default'}")
        print(f"  _post_hook: {'from submission' if post_hook != default_post_hook else 'using default'}")
        print(f"  BASE_SEED: {base_seed} ({'from submission' if hasattr(submission, 'BASE_SEED') else 'using time-based default'})")
        print(f"  N: {n} ({'from submission' if hasattr(submission, 'N') else 'using default 593'})")
        print(f"  USE_FLOAT64: {use_float64}")
        print("="*60)
    
    functions = {
        'define_hyperparameters': define_hyperparameters,
        'get_optimizer': get_optimizer,
        'get_init_fn': get_init_fn,
        'update_func': update_func,
        'post_hook': post_hook,
        'base_seed': base_seed,
        'n': n,
        'use_float64': use_float64,
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






def kissing_trainable(config):
    """Ray Tune trainable function that calls train_single for the actual work."""
    # Set JAX configuration for Ray worker process
    import os
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'

    # Suppress all logging in worker
    import logging
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("ray.train").setLevel(logging.ERROR)
    logging.getLogger("ray.air").setLevel(logging.ERROR)
    logging.getLogger("ray").setLevel(logging.ERROR)

    # Buffer all output to print at the end
    output_buffer = []

    with capture_stdout(output_buffer):
        # Load all functions from shared storage directory first
        shared_storage_path = config.get('_shared_storage_path')
        funcs = load_all_functions(shared_storage_path)

        # Stay in shared storage directory for the rest of execution
        os.chdir(shared_storage_path)

        # Extract BASE_SEED and N from config
        trial_base_seed = config.get('base_seed', funcs['base_seed'])
        trial_n = config.get('n', funcs['n'])
        trial_use_float64 = bool(config.get('use_float64', funcs.get('use_float64', False)))
        if trial_use_float64:
            os.environ['JAX_ENABLE_X64'] = '1'
            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_default_matmul_precision", "highest")

        # Get trial number from Ray Tune session
        tune_context = tune.get_context()
        trial_dir_name = tune_context.get_storage().trial_dir_name
        parts = trial_dir_name.split('_')
        trial_number = 0  # fallback
        for i, part in enumerate(parts):
            if part.isdigit():
                trial_number = int(part)
                break

        # Create hparams dict from config (excluding our constants)
        constants = {'base_seed', '_shared_storage_path', '_submission_uuid', 'n', 'use_float64'}
        hparams = {k: v for k, v in config.items() if k not in constants}

        # Import the single trial training function
        from train_single import run_single_trial_optimization

        # Run the optimization for this trial
        submission_uuid = config.get('_submission_uuid')
        final_margin, best_configuration, density = run_single_trial_optimization(
            funcs=funcs,
            hparams=hparams,
            trial_number=trial_number,
            trial_n=trial_n,
            submission_uuid=submission_uuid,
            trial_timeout=TRIAL_TIMEOUT,
            verbose=True
        )

    # Print all captured output at once (outside capture context)
    for line in output_buffer:
        print(line)

    # Simple Ray report without checkpoint
    tune.report(metrics={"margin": final_margin, "completed": True})

    return {"margin": final_margin}


def simple_cpu_validation():
    """Simple CPU-only validation with single training step, no Ray/Optuna."""
    print("=== SIMPLE CPU VALIDATION ===")
    print("Running single training step with batch_size=8, no Ray/Optuna")
    
    # Force JAX to use CPU for validation
    jax.config.update('jax_platform_name', 'cpu')
    
    # Load all functions
    funcs = load_all_functions()
    if funcs.get('use_float64', False):
        os.environ['JAX_ENABLE_X64'] = '1'
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
    
    # Sample hyperparameters from agent's search space
    search_space = funcs['define_hyperparameters']()
    hparams = {}
    for key, space_obj in search_space.items():
        if hasattr(space_obj, 'sample'):
            hparams[key] = space_obj.sample()
        else:
            print(f"Warning: {key} doesn't have sample() method")
    
    # Create optimizer and initializer
    optimizer = funcs['get_optimizer'](hparams)
    init_fn = funcs['get_init_fn'](hparams)
    dtype = jnp.float64 if funcs.get('use_float64', False) else jnp.float32
    
    # Initialize parameters with small batch
    key = jax.random.PRNGKey(42)
    
    key, subkey = jax.random.split(key)
    n = funcs['n']  # Get N from functions
    sphere_centers = init_fn(subkey, (8, n, DIM), dtype=dtype)  # batch_size=8
    
    opt_state = optimizer.init(sphere_centers)
    
    # Single training step - handle both 3-tuple and 4-tuple returns
    key, subkey = jax.random.split(key)
    update_result = funcs['update_func'](
        sphere_centers=sphere_centers,
        opt_state=opt_state,
        global_step=0,
        rng=subkey,
        optimizer=optimizer,
        constraints_data=None,
        hparams=hparams
    )
    
    if len(update_result) == 4:
        sphere_centers, opt_state, loss, margins = update_result
        print(f"Agent returned margins - shape: {margins.shape}")
    else:
        sphere_centers, opt_state, loss = update_result
        print("Agent returned standard 3-tuple format")
    
    print("Validation successful - agent functions work correctly!\n")
    
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
    search_space.update({
        'base_seed': funcs['base_seed'],
        'n': funcs['n'],  # Pass N to workers
        'use_float64': funcs['use_float64'],
        '_shared_storage_path': os.path.abspath('.'),  # Pass current directory to workers
        '_submission_uuid': submission_uuid  # Pass unique submission ID to workers
    })
    
    # Create Optuna search algorithm
    sampler = optuna.samplers.TPESampler(n_startup_trials=WARMUP_TRIALS, seed=funcs['base_seed'])
    optuna_search = OptunaSearch(
        metric="margin",
        mode="max",  # We want to maximize margin
        sampler=sampler
    )
    
    # Limit concurrent trials
    search_alg = ConcurrencyLimiter(optuna_search, max_concurrent=MAX_CONCURRENT)
    
    mode_str = "FULL RAY TRAINING"
    print("="*60)
    print(f"Starting Ray Tune Kissing Number Optimization ({mode_str})")
    print(f"Configuration: batch_size={BATCH_SIZE} (mini-batch={MINI_BATCH_SIZE}x{NUM_MINI_BATCHES}), N={funcs['n']}, dim={DIM}, max_steps={MAX_STEPS}")
    print(f"Parallel trials: {MAX_CONCURRENT}, Total trials: {NUM_SAMPLES}")
    print(f"Timeout: {TRIAL_TIMEOUT/60:.1f} min per trial, {TOTAL_TIMEOUT/60:.1f} min total")
    print("="*60)
    
    # Function source logging is handled by load_all_functions(verbose=True)
    
    # Ensure Ray Tune storage directory exists and is accessible to all nodes
    ray_storage_path = os.path.abspath('.')  # Current shared directory
    ray_results_dir = os.path.join(ray_storage_path, "ray_results")
    os.makedirs(ray_results_dir, exist_ok=True)
    
    # Configure Ray Tune run
    tuner = tune.Tuner(
        tune.with_resources(
            kissing_trainable,
            resources={"cpu": 1, "gpu": 1}  # Each trial requests 1 GPU
        ),
        tune_config=tune.TuneConfig(
            metric="margin",
            mode="max",  # We want to maximize margin
            search_alg=search_alg,
            num_samples=NUM_SAMPLES,
            time_budget_s=TOTAL_TIMEOUT - SAFETY_BUFFER,
            max_concurrent_trials=MAX_CONCURRENT
        ),
        param_space=search_space,
        run_config=tune.RunConfig(
            name=f"kissing_ray_optimization_{submission_uuid[:8]}",
            storage_path=ray_storage_path,  # Use shared storage accessible to all nodes
            stop={"completed": True},  # Stop when trials complete naturally
            checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=False),
            failure_config=tune.FailureConfig(max_failures=3),  # Auto-retry failed trials on different nodes
            verbose=0,  # Suppress Ray Tune's verbose output
            progress_reporter=SilentReporter()  # Use custom silent reporter
        )
    )
    
    # Run the optimization
    print("\nStarting Ray Tune optimization...")
    results = tuner.fit()
    
    # Simple completion reporting - all evaluation logic moved to auto evaluator
    print(f"SUBMISSION_UUID: {submission_uuid}")  # Key line for evaluator to parse
    print(f"NUM_TRIALS: {NUM_SAMPLES}")  # Key line for evaluator to know configured trial count
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
        _run_test_mode(submission)
        sys.exit(0)
    
    # Normal Ray optimization mode
    result = main()
