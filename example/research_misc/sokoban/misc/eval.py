#!/usr/bin/env python3
"""
Standalone evaluation script for testing submissions on the test set.
Uses identical setup as research counter evaluation but evaluates on test set instead of validation set.

Usage:
    cd misc
    python eval.py submission_path.py [--num-seeds N] [--timeout SECONDS]

Example:
    python eval.py eval/eval_drc.py --num-seeds 4 --timeout 1800
"""

import os
import sys
import argparse
import time
import tempfile
import shutil
import glob
import re
import numpy as np
import uuid
from pathlib import Path

# Add research directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()  # misc/
RESEARCH_DIR = SCRIPT_DIR.parent / "research"  # research/
STORAGE_SYSTEM_DIR = RESEARCH_DIR / "storage" / "system"

sys.path.insert(0, str(STORAGE_SYSTEM_DIR))
sys.path.insert(0, str(RESEARCH_DIR))

# Suppress warnings before any major imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['JAX_LOG_COMPILES'] = '0'
os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '1'

import warnings
warnings.filterwarnings('ignore')

# Import JAX but force CPU-only mode to prevent GPU initialization in head process
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp


def create_test_env(reward_adjustment: float = 0.09, time_limit: int = 120, num_levels: int = 1000):
    """Create test environment wrapper using unfiltered-test dataset.

    Uses a fixed random subset of 1000 levels from the test set.
    The subset is deterministic (seed 42) for reproducible evaluation.
    """
    # Import here to avoid GPU initialization in head process
    from jumanji.environments.routing.sokoban.generator import HuggingFaceDeepMindGenerator
    from env import SokobanEnvWrapper

    # Load full test dataset
    generator = HuggingFaceDeepMindGenerator('unfiltered-test', proportion_of_files=1.0)

    # Create fixed subset using independent random state
    test_rng = np.random.RandomState(42)  # Fixed seed for reproducible subset
    total_levels = len(generator._fixed_grids)

    # If test set has fewer levels than requested, use all of them
    actual_num_levels = min(num_levels, total_levels)
    indices = test_rng.choice(total_levels, size=actual_num_levels, replace=False)
    indices = np.sort(indices)

    # Create wrapper with test dataset
    wrapper = SokobanEnvWrapper(
        dataset_name="unfiltered-test",
        reward_adjustment=reward_adjustment,
        time_limit=time_limit
    )

    # Override with subset
    wrapper.fixed_grids = generator._fixed_grids[indices]
    wrapper.variable_grids = generator._variable_grids[indices]
    wrapper.num_levels = actual_num_levels

    return wrapper


def sokoban_eval_trainable(config):
    """Custom trainable that loads submission from shared directory and uses TEST dataset."""
    # Set environment variables for Ray worker process
    import os
    import sys
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['JAX_LOG_COMPILES'] = '0'

    # Suppress all warnings in worker
    import warnings
    warnings.filterwarnings('ignore')

    # Suppress all logging in worker
    import logging
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("ray.train").setLevel(logging.ERROR)
    logging.getLogger("ray.air").setLevel(logging.ERROR)
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.worker").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    # No buffering - print directly for real-time progress
    output_buffer = []

    if True:  # Keep indentation structure
        # Get shared directory from config
        shared_eval_dir = config.get('_shared_eval_dir')
        shared_research_dir = os.path.join(shared_eval_dir, 'research')

        # Add shared directory to path to import submission
        sys.path.insert(0, shared_eval_dir)

        # Change to shared research directory and load functions
        os.chdir(shared_research_dir)

        # Add storage/system to path for imports
        sys.path.append('storage/system')

        # Import defaults
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

        funcs = {
            'define_hyperparameters': define_hyperparameters,
            'create_network': create_network,
            'training_step': training_step,
            'create_optimizer': create_optimizer,
            'complete': complete,
            'base_seed': base_seed
        }

        # Import the evaluation training function (uses test dataset directly)
        from eval_train_single import run_single_seed_training

        # Extract BASE_SEED from config
        trial_base_seed = config.get('base_seed', funcs['base_seed'])

        # Get trial number from Ray Tune session for reproducible seeding
        from ray import tune
        tune_context = tune.get_context()
        trial_dir_name = tune_context.get_storage().trial_dir_name
        parts = trial_dir_name.split('_')
        trial_number = 0
        for i, part in enumerate(parts):
            if part.isdigit():
                trial_number = int(part)
                break

        # Create hparams dict from config (excluding our constants)
        constants = {'base_seed', '_shared_eval_dir', '_submission_uuid', '_num_levels', '_trial_timeout', '_batch_size'}
        hparams = {k: v for k, v in config.items() if k not in constants}

        # Get timeout and batch_size from config
        trial_timeout = config.get('_trial_timeout', 30*60)
        batch_size = config.get('_batch_size', 64)

        print(f"\nStarting Sokoban RL training (TEST SET) - Seed {trial_number}, batch_size {batch_size}, timeout {trial_timeout/60:.1f}min")

        # Set JAX to use GPU
        import jax
        jax.config.update('jax_platform_name', 'gpu')

        import time
        trial_start = time.time()

        # RL Training parameters (matching paper defaults)
        CONFIG = {
            'num_envs': batch_size,  # Number of parallel environments (default 64, paper uses 32)
            'unroll_length': 20,  # BPTT unroll length (paper: 20)
            'total_timesteps': 50_000_000,
            'time_limit': 120,
            'seed': 42,
            'max_training_minutes': trial_timeout / 60,
            'timeout_check_interval': 100,
        }

        # Run the RL training with the specific seed
        seed = trial_base_seed + trial_number
        solve_rate_sampling, solve_rate_greedy, final_params, final_opt_state = run_single_seed_training(
            funcs=funcs,
            hparams=hparams,
            seed=seed,
            config=CONFIG,
            trial_timeout=trial_timeout
        )

        # Save trial data with conflict resolution
        submission_uuid = config.get('_submission_uuid')
        trial_file_path = f"storage/shared/tmp/{submission_uuid}/trial_{trial_number}.npz"
        trial_counter = trial_number
        while os.path.exists(trial_file_path):
            trial_counter += 1
            trial_file_path = f"storage/shared/tmp/{submission_uuid}/trial_{trial_counter}.npz"

        # Prepare trial data dictionary
        import numpy as np
        trial_data = {
            'solve_rate': solve_rate_sampling,
            'solve_rate_greedy': solve_rate_greedy,
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
        print(f"Solve rate (sampling): {solve_rate_sampling:.3f}")
        print(f"Solve rate (greedy): {solve_rate_greedy:.3f}")
        print(f"Training time: {total_time:.0f}s")

    # Simple Ray report without checkpoint (use sampling as primary metric)
    from ray import tune
    tune.report(metrics={"solve_rate": solve_rate_sampling, "completed": True})


def evaluate_submission(submission_path: str, num_seeds: int = 4, timeout: int = 36000,
                       num_levels: int = 1000, batch_size: int = 64, verbose: bool = True):
    """Evaluate a submission file on the test set.

    Args:
        submission_path: Path to the submission.py file
        num_seeds: Number of seeds to run (default: 4)
        timeout: Timeout per seed in seconds (default: 36000 = 10 hours)
        num_levels: Number of test levels to evaluate on (default: 1000)
        batch_size: Number of parallel environments (default: 64, paper uses 32)
        verbose: Print detailed progress

    Returns:
        dict: Evaluation results with scores and statistics
    """
    submission_path = Path(submission_path).absolute()

    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    if verbose:
        print("="*60)
        print(f"Evaluating submission: {submission_path.name}")
        print(f"Test set: {num_levels} levels from unfiltered-test")
        print(f"Seeds: {num_seeds}, Timeout: {timeout}s per seed")
        print("="*60)

    # Create a unique shared directory for this evaluation (must be on shared filesystem for Ray cluster)
    eval_uuid = str(uuid.uuid4())
    # Use EVAL_SHARED_DIR environment variable, fallback to /tmp if not set
    shared_eval_base = os.environ.get('EVAL_SHARED_DIR', '/tmp')
    if not os.path.exists(shared_eval_base):
        raise FileNotFoundError(f"EVAL_SHARED_DIR '{shared_eval_base}' does not exist. Please set EVAL_SHARED_DIR to a valid shared directory accessible by all Ray nodes.")
    shared_eval_dir = f"{shared_eval_base}/sokoban_eval_{eval_uuid}"
    os.makedirs(shared_eval_dir, exist_ok=True)

    if verbose:
        print(f"\n1. Created shared evaluation directory: {shared_eval_dir}")

    # Copy submission to shared directory where all Ray workers can access it
    submission_dest = os.path.join(shared_eval_dir, 'submission.py')
    shutil.copy(submission_path, submission_dest)

    # Copy eval_train_single.py to shared directory for Ray workers
    eval_train_single_src = SCRIPT_DIR / 'eval_train_single.py'
    eval_train_single_dest = os.path.join(shared_eval_dir, 'eval_train_single.py')
    shutil.copy(eval_train_single_src, eval_train_single_dest)

    if verbose:
        print(f"   Copied submission and eval_train_single to shared directory for Ray workers")

    # Copy necessary directories and files to shared directory
    research_dir = RESEARCH_DIR
    shared_research_dir = os.path.join(shared_eval_dir, 'research')

    # Copy storage/system directory structure
    storage_src = os.path.join(research_dir, 'storage', 'system')
    storage_dest = os.path.join(shared_research_dir, 'storage', 'system')
    shutil.copytree(storage_src, storage_dest)

    # Create storage/shared/tmp directory for results
    shared_tmp_dir = os.path.join(shared_research_dir, 'storage', 'shared', 'tmp')
    os.makedirs(shared_tmp_dir, exist_ok=True)

    if verbose:
        print(f"   Copied research infrastructure to shared directory")

    # Change to shared research directory
    original_cwd = os.getcwd()
    os.chdir(shared_research_dir)

    try:
        # Add shared eval dir to path to import submission
        sys.path.insert(0, shared_eval_dir)
        import submission

        # Add storage/system to path for imports
        sys.path.insert(0, storage_dest)

        # Get define_hyperparameters and other functions
        from defaults import (
            default_define_hyperparameters,
            default_create_network,
            default_training_step,
            default_create_optimizer,
            default_complete,
            BASE_SEED
        )

        define_hyperparameters = getattr(submission, '_define_hyperparameters', default_define_hyperparameters)
        create_network = getattr(submission, 'create_network', default_create_network)
        training_step = getattr(submission, 'training_step', default_training_step)
        create_optimizer = getattr(submission, 'create_optimizer', default_create_optimizer)
        complete = getattr(submission, 'complete', default_complete)
        base_seed = getattr(submission, 'BASE_SEED', BASE_SEED)

        funcs = {
            'define_hyperparameters': define_hyperparameters,
            'create_network': create_network,
            'training_step': training_step,
            'create_optimizer': create_optimizer,
            'complete': complete,
            'base_seed': base_seed
        }

        if verbose:
            print(f"\n2. Loaded submission module")
            print(f"   BASE_SEED: {base_seed}")

        # Run CPU validation first (matching research counter behavior)
        if verbose:
            print("\n" + "="*60)
            print("SIMPLE CPU VALIDATION")
            print("="*60)
            print("Validating function loading and basic compatibility")

        # Force JAX to use CPU for validation
        jax.config.update('jax_platform_name', 'cpu')

        # Sample hyperparameters from agent's search space
        search_space = funcs['define_hyperparameters']()
        hparams = {}
        for key, space_obj in search_space.items():
            if hasattr(space_obj, 'sample'):
                hparams[key] = space_obj.sample()

        # Test network creation
        network = funcs['create_network'](hparams)
        if verbose:
            print("✓ Network creation works")

        # Count total parameters in the network
        def count_params(params):
            """Count total number of parameters in a pytree."""
            return sum(x.size for x in jax.tree_util.tree_leaves(params))

        # Initialize params to count them
        dummy_obs_count = jnp.zeros((4, 8, 8, 8), dtype=jnp.float32)
        dummy_done_count = jnp.zeros((4,), dtype=bool)
        key_count = jax.random.PRNGKey(42)
        params_count = network.init(key_count, dummy_obs_count, dummy_done_count)
        total_params = count_params(params_count)

        if verbose:
            print(f"✓ Total network parameters: {total_params:,}")

        # Test basic forward pass
        dummy_obs = jnp.zeros((4, 8, 8, 8), dtype=jnp.float32)
        dummy_done = jnp.zeros((4,), dtype=bool)
        key = jax.random.PRNGKey(42)
        params = network.init(key, dummy_obs, dummy_done)
        outputs = network.apply(params, dummy_obs, dummy_done)
        if verbose:
            print(f"✓ Network forward pass works - {len(outputs)} outputs")

        # Test optimizer creation
        import optax
        optimizer = funcs['create_optimizer']()
        if verbose:
            print("✓ Optimizer creation works")

        # Test training_step function
        if verbose:
            print("Testing training_step...")
        validation_batch_size = 4
        sequence_length = 20
        dummy_batch = {
            'observations': jnp.zeros((validation_batch_size, sequence_length, 8, 8, 8), dtype=jnp.float32),
            'actions': jnp.zeros((validation_batch_size, sequence_length), dtype=jnp.int32),
            'rewards': jnp.zeros((validation_batch_size, sequence_length), dtype=jnp.float32),
            'dones': jnp.zeros((validation_batch_size, sequence_length), dtype=bool),
            'values': jnp.zeros((validation_batch_size, sequence_length), dtype=jnp.float32),
            'final_values': jnp.zeros(validation_batch_size, dtype=jnp.float32),
            'initial_rnn_states': None
        }

        opt_state = optimizer.init(params)
        new_params, new_opt_state = funcs['training_step'](
            network, optimizer, params, opt_state, dummy_batch, hparams
        )

        if new_params is None:
            raise ValueError("training_step returned None for params")
        if new_opt_state is None:
            raise ValueError("training_step returned None for opt_state")

        if verbose:
            print("✓ Training step works")

        # Test complete function
        if verbose:
            print("Testing complete function...")
        dummy_trial_data = {
            'solve_rate': 0.5,
            'hyperparameters': hparams,
            'seed': 42,
            'trial_number': 0
        }
        funcs['complete'](params, opt_state, dummy_trial_data)
        if verbose:
            print("✓ Complete function works")
            print("Validation successful - all functions work correctly!\n")

        # Now proceed with Ray Tune evaluation
        import ray
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch
        from ray.tune.search import ConcurrencyLimiter
        import optuna

        # Import SilentReporter from main
        from main import SilentReporter

        # Fixed parameters
        NUM_SAMPLES = num_seeds
        MAX_CONCURRENT = 4
        WARMUP_TRIALS = num_seeds
        SAFETY_BUFFER = 10
        TOTAL_TIMEOUT = timeout * num_seeds
        TRIAL_TIMEOUT = timeout

        # Initialize Ray cluster
        ray_address = os.environ.get('RAY_HEAD_NODE_IP')
        if ray_address:
            if verbose:
                print(f"\n3. Connecting to Ray cluster at {ray_address}")
            ray.init(address=ray_address, logging_config=ray.LoggingConfig(encoding="TEXT"))
        else:
            if verbose:
                print(f"\n3. RAY_HEAD_NODE_IP not set, initializing Ray locally")
            ray.init(logging_config=ray.LoggingConfig(encoding="TEXT"))

        if verbose:
            print("   Ray cluster initialized successfully")
            resources = ray.cluster_resources()
            num_cpus = resources.get('CPU', 0)
            num_gpus = resources.get('GPU', 0)
            print(f"   Available resources: {num_cpus:.0f} CPUs, {num_gpus:.0f} GPUs")

        # Get search space
        search_space = define_hyperparameters()

        # Create submission-unique temp directory
        submission_uuid = str(uuid.uuid4())
        submission_temp_dir = os.path.join(shared_tmp_dir, submission_uuid)
        os.makedirs(submission_temp_dir, exist_ok=True)

        # Add constants to search space
        search_space.update({
            'base_seed': base_seed,
            '_shared_eval_dir': shared_eval_dir,
            '_submission_uuid': submission_uuid,
            '_num_levels': num_levels,
            '_trial_timeout': TRIAL_TIMEOUT,
            '_batch_size': batch_size
        })

        # Create Optuna search algorithm
        sampler = optuna.samplers.TPESampler(n_startup_trials=WARMUP_TRIALS, seed=base_seed)
        optuna_search = OptunaSearch(
            metric="solve_rate",
            mode="max",
            sampler=sampler
        )

        # Limit concurrent trials
        search_alg = ConcurrencyLimiter(optuna_search, max_concurrent=MAX_CONCURRENT)

        if verbose:
            print("\n" + "="*60)
            print(f"Starting Ray Tune Sokoban RL Evaluation (TEST SET)")
            print(f"Configuration: {num_seeds} seeds, 50,000,000 steps per seed")
            print(f"Parallel trials: {MAX_CONCURRENT}, Total trials: {NUM_SAMPLES}")
            print(f"Timeout: {TRIAL_TIMEOUT/60:.1f} min per trial, {TOTAL_TIMEOUT/60:.1f} min total")
            print("="*60)

        # Configure Ray Tune run
        tuner = tune.Tuner(
            tune.with_resources(
                sokoban_eval_trainable,
                resources={"cpu": 1, "gpu": 1}
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
                name=f"sokoban_test_eval_{submission_uuid[:8]}",
                storage_path=os.path.abspath('.'),
                stop={"solve_rate": 1.0, "completed": True},
                checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=False),
                failure_config=tune.FailureConfig(max_failures=2),
                verbose=0,
                progress_reporter=SilentReporter()
            )
        )

        # Run the optimization
        if verbose:
            print("\n4. Starting Ray Tune optimization...")
        start_time = time.time()
        results = tuner.fit()

        # Parse results from submission temp directory
        trial_files = glob.glob(os.path.join(submission_temp_dir, "trial_*.npz"))
        solve_rates_sampling = []
        solve_rates_greedy = []
        for trial_file in sorted(trial_files):
            data = np.load(trial_file, allow_pickle=True)
            solve_rates_sampling.append(float(data['solve_rate']))
            solve_rates_greedy.append(float(data['solve_rate_greedy']))

        if verbose:
            print(f"\n5. Evaluation completed")
            print(f"   SUBMISSION_UUID: {submission_uuid}")
            print(f"   Total time: {time.time() - start_time:.0f}s")

        # Cleanup
        ray.shutdown()

        # Calculate statistics for both modes
        mean_solve_rate_sampling = float(np.mean(solve_rates_sampling))
        std_solve_rate_sampling = float(np.std(solve_rates_sampling))
        best_solve_rate_sampling = float(np.max(solve_rates_sampling))

        mean_solve_rate_greedy = float(np.mean(solve_rates_greedy))
        std_solve_rate_greedy = float(np.std(solve_rates_greedy))
        best_solve_rate_greedy = float(np.max(solve_rates_greedy))

        results_dict = {
            'individual_scores_sampling': solve_rates_sampling,
            'individual_scores_greedy': solve_rates_greedy,
            'mean_solve_rate_sampling': mean_solve_rate_sampling,
            'std_solve_rate_sampling': std_solve_rate_sampling,
            'best_solve_rate_sampling': best_solve_rate_sampling,
            'mean_solve_rate_greedy': mean_solve_rate_greedy,
            'std_solve_rate_greedy': std_solve_rate_greedy,
            'best_solve_rate_greedy': best_solve_rate_greedy,
            'num_seeds': num_seeds,
            'num_test_levels': num_levels,
        }

        if verbose:
            print("\n" + "="*60)
            print("FINAL RESULTS")
            print("="*60)
            print("\nSampling mode:")
            individual_scores_str = ", ".join([f"{score:.3f}" for score in solve_rates_sampling])
            print(f"  Individual seed scores: [{individual_scores_str}]")
            print(f"  Mean solve rate: {mean_solve_rate_sampling:.3f}")
            print(f"  Std solve rate: {std_solve_rate_sampling:.3f}")
            print(f"  Best solve rate: {best_solve_rate_sampling:.3f}")
            print(f"  Final score (percentage): {mean_solve_rate_sampling * 100:.1f}%")
            print("\nGreedy mode:")
            individual_scores_str = ", ".join([f"{score:.3f}" for score in solve_rates_greedy])
            print(f"  Individual seed scores: [{individual_scores_str}]")
            print(f"  Mean solve rate: {mean_solve_rate_greedy:.3f}")
            print(f"  Std solve rate: {std_solve_rate_greedy:.3f}")
            print(f"  Best solve rate: {best_solve_rate_greedy:.3f}")
            print(f"  Final score (percentage): {mean_solve_rate_greedy * 100:.1f}%")
            print("="*60)

        return results_dict

    finally:
        # Restore working directory
        os.chdir(original_cwd)

        # Cleanup shared directory
        if verbose:
            print(f"\n6. Cleaning up shared directory: {shared_eval_dir}")
        shutil.rmtree(shared_eval_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Sokoban RL submissions on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py eval_drc.py
  python eval.py eval_drc.py --num-seeds 2 --timeout 900
  python eval.py eval_drc.py --num-levels 500
        """
    )

    parser.add_argument(
        'submission',
        type=str,
        help='Path to submission file (e.g., eval_drc.py)'
    )

    parser.add_argument(
        '--num-seeds',
        type=int,
        default=10,
        help='Number of seeds to run (default: 10)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=36000,
        help='Timeout per seed in seconds (default: 36000 = 10 hours)'
    )

    parser.add_argument(
        '--num-levels',
        type=int,
        default=1000,
        help='Number of test levels to evaluate on (default: 1000)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Number of parallel environments (default: 32)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed progress output'
    )

    args = parser.parse_args()

    try:
        results = evaluate_submission(
            submission_path=args.submission,
            num_seeds=args.num_seeds,
            timeout=args.timeout,
            num_levels=args.num_levels,
            batch_size=args.batch_size,
            verbose=not args.quiet
        )

        # Exit with success
        sys.exit(0)

    except Exception as e:
        print(f"\nError during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
