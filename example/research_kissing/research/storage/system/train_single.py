#!/usr/bin/env python3
"""
Single trial optimization for Kissing Number Margin Optimization.

This file contains the core optimization logic for a single trial, without
Ray cluster management complexity. Agents should read this file to understand
how the optimization works.

The main entry point is run_single_trial_optimization() which executes one
complete optimization run with a specific set of hyperparameters.
"""

import os
import time
import inspect
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import optax
from functools import partial

# Fixed parameters for the task
DIM = 11
BATCH_SIZE = 128
MINI_BATCH_SIZE = 32  # Process in mini-batches for better performance
NUM_MINI_BATCHES = BATCH_SIZE // MINI_BATCH_SIZE  # 4 mini-batches
MAX_STEPS = 1000000

# Import defaults to access compute_margin_batch
import sys
sys.path.append('storage/system')
from defaults import compute_margin_batch


@jit
def compute_margins_and_best(sphere_centers):
    """Compute margins for batch and return best margin with corresponding configuration."""
    margins = compute_margin_batch(sphere_centers)

    # Find best in batch
    best_idx = jnp.argmax(margins)
    current_margin = margins[best_idx]
    current_best = sphere_centers[best_idx]

    return current_margin, current_best


def create_batched_update(update_func, optimizer, constraints_data, hparams, num_steps, n, param_dtype):
    """
    Create a JIT-compiled function that runs num_steps updates entirely on GPU.

    This function batches multiple gradient updates together for efficiency,
    tracking the best solution found across all steps.

    Args:
        update_func: The agent's update function (from submission or defaults)
        optimizer: Optax optimizer instance
        constraints_data: Constraint-related data (not used)
        hparams: Hyperparameters dictionary
        num_steps: Number of update steps to batch together
        n: Number of spheres to optimize

    Returns:
        A JIT-compiled function that performs batched updates
    """

    # Test update function signature with dummy data to determine return format
    import jax
    key = jax.random.PRNGKey(42)
    dummy_params = jax.random.normal(key, (4, n, 11), dtype=param_dtype)  # Mini batch size
    dummy_opt_state = optimizer.init(dummy_params)

    test_result = update_func(
        sphere_centers=dummy_params,
        opt_state=dummy_opt_state,
        global_step=0,
        rng=key,
        optimizer=optimizer,
        constraints_data=constraints_data,
        hparams=hparams
    )

    returns_margins = len(test_result) == 4

    if returns_margins:
        @jit
        def batched_update(params, opt_state, start_step, key, best_margin_so_far, best_config_so_far):
            def scan_fn(carry, step_offset):
                params, opt_state, key, best_margin, best_config = carry
                key, subkey = jax.random.split(key)
                global_step = start_step + step_offset

                # Store pre-update params for margin tracking
                pre_update_params = params

                # Agent returns margins (computed on pre-update params)
                params, opt_state, loss, margins = update_func(
                    sphere_centers=params,
                    opt_state=opt_state,
                    global_step=global_step,
                    rng=subkey,
                    optimizer=optimizer,
                    constraints_data=constraints_data,
                    hparams=hparams
                )

                # Find best margin and corresponding configuration from PRE-UPDATE params
                best_idx = jnp.argmax(margins)
                current_margin = margins[best_idx]
                current_best = pre_update_params[best_idx]  # Use pre-update config that margins correspond to

                # Update best if current is better (higher margin)
                is_better = current_margin > best_margin
                new_best_margin = jnp.where(is_better, current_margin, best_margin)
                new_best_config = jnp.where(is_better, current_best, best_config)

                return (params, opt_state, key, new_best_margin, new_best_config), loss

            # Initialize with current solution
            initial_margin, initial_best = compute_margins_and_best(params)

            is_initial_better = initial_margin > best_margin_so_far
            init_best_margin = jnp.where(is_initial_better, initial_margin, best_margin_so_far)
            init_best_config = jnp.where(is_initial_better, initial_best, best_config_so_far)

            (params, opt_state, key, final_best_margin, final_best_config), losses = jax.lax.scan(
                scan_fn,
                (params, opt_state, key, init_best_margin, init_best_config),
                jnp.arange(num_steps)
            )
            return params, opt_state, key, losses[-1], final_best_margin, final_best_config

    else:
        @jit
        def batched_update(params, opt_state, start_step, key, best_margin_so_far, best_config_so_far):
            def scan_fn(carry, step_offset):
                params, opt_state, key, best_margin, best_config = carry
                key, subkey = jax.random.split(key)
                global_step = start_step + step_offset

                # Agent returns standard format
                params, opt_state, loss = update_func(
                    sphere_centers=params,
                    opt_state=opt_state,
                    global_step=global_step,
                    rng=subkey,
                    optimizer=optimizer,
                    constraints_data=constraints_data,
                    hparams=hparams
                )

                # Compute margins ourselves
                current_margin, current_best = compute_margins_and_best(params)

                # Update best if current is better (higher margin)
                is_better = current_margin > best_margin
                new_best_margin = jnp.where(is_better, current_margin, best_margin)
                new_best_config = jnp.where(is_better, current_best, best_config)

                return (params, opt_state, key, new_best_margin, new_best_config), loss

            # Initialize with current solution
            initial_margin, initial_best = compute_margins_and_best(params)

            is_initial_better = initial_margin > best_margin_so_far
            init_best_margin = jnp.where(is_initial_better, initial_margin, best_margin_so_far)
            init_best_config = jnp.where(is_initial_better, initial_best, best_config_so_far)

            (params, opt_state, key, final_best_margin, final_best_config), losses = jax.lax.scan(
                scan_fn,
                (params, opt_state, key, init_best_margin, init_best_config),
                jnp.arange(num_steps)
            )
            return params, opt_state, key, losses[-1], final_best_margin, final_best_config

    return batched_update


def run_single_trial_optimization(funcs, hparams, trial_number, trial_n, submission_uuid, trial_timeout, verbose=False):
    """
    Run a single trial optimization for the kissing number problem.

    Args:
        funcs: Dictionary containing all optimization functions
        hparams: Hyperparameters for this trial
        trial_number: Trial number for reproducible seeding
        trial_n: Number of spheres to optimize
        submission_uuid: Unique submission identifier
        trial_timeout: Maximum time allowed for this trial (seconds)
        verbose: Whether to print progress information

    Returns:
        tuple: (final_margin, best_configuration, density) where density is 1.0 for kissing number
    """
    if verbose:
        print(f"\nStarting trial {trial_number} with N={trial_n}, timeout {trial_timeout:.1f}s")

    # Initialize compact logging table
    step_markers = []
    loss_values = []
    margin_values = []

    # Set JAX to use GPU
    jax.config.update('jax_platform_name', 'gpu')
    if funcs.get('use_float64', False):
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
    param_dtype = jnp.float64 if funcs.get('use_float64', False) else jnp.float32
    np_dtype = np.float64 if funcs.get('use_float64', False) else np.float32

    trial_start = time.time()

    # Create optimizer and initializer using hyperparameters
    optimizer = funcs['get_optimizer'](hparams)
    init_fn = funcs['get_init_fn'](hparams)

    # Initialize parameters with trial-specific seed for reproducibility
    key = jax.random.PRNGKey(funcs['base_seed'] + trial_number)

    # Initialize parameters - create mini-batches
    sphere_centers_list = []
    opt_state_list = []

    for i in range(NUM_MINI_BATCHES):
        key, subkey = jax.random.split(key)
        # Use slightly different seed for each mini-batch
        mini_key = jax.random.PRNGKey(funcs['base_seed'] + trial_number + i)
        mini_centers = init_fn(mini_key, (MINI_BATCH_SIZE, trial_n, DIM), dtype=param_dtype)
        mini_opt_state = optimizer.init(mini_centers)

        sphere_centers_list.append(mini_centers)
        opt_state_list.append(mini_opt_state)

    best_margin = float('-inf')
    best_configuration = None
    best_mini_batch_idx = None

    post_hook = funcs['post_hook']
    post_hook_accepts_info = False
    post_hook_accepts_kwargs = False
    try:
        post_hook_sig = inspect.signature(post_hook)
        post_hook_params = post_hook_sig.parameters
        post_hook_accepts_info = 'info' in post_hook_params
        post_hook_accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in post_hook_params.values()
        )
    except (TypeError, ValueError):
        pass

    # Define checkpoints for margin evaluation: 10k increments only (more efficient)
    checkpoints = []
    # Add 10k increments from 10k up to MAX_STEPS
    for step in range(10000, MAX_STEPS + 1, 10000):
        checkpoints.append(step)

    # Pre-compile batched update functions for each checkpoint interval
    batched_updates = {}
    prev_checkpoint = 0

    for checkpoint in checkpoints:
        steps_in_batch = checkpoint - prev_checkpoint
        if steps_in_batch not in batched_updates:
            batch_fn = create_batched_update(
                funcs['update_func'],
                optimizer,
                None,
                hparams,
                steps_in_batch,
                trial_n,
                param_dtype,
            )
            # Force compilation with dummy call using first mini-batch
            _, _, _, loss_dummy, _, _ = batch_fn(
                sphere_centers_list[0],
                opt_state_list[0],
                0,
                key,
                best_margin,
                jnp.zeros((trial_n, DIM), dtype=param_dtype),
            )
            loss_dummy.block_until_ready()  # Ensure compilation completes
            batched_updates[steps_in_batch] = batch_fn
        prev_checkpoint = checkpoint

    # Batched training loop - process intervals between checkpoints
    current_step = 0
    for checkpoint_idx, checkpoint in enumerate(checkpoints):
        # Check timeout before starting batch
        elapsed = time.time() - trial_start
        if elapsed > trial_timeout:
            if verbose:
                print(f"Trial {trial_number} timeout before step {current_step}")
            break

        steps_in_batch = checkpoint - current_step
        batch_fn = batched_updates[steps_in_batch]

        # Process 4 mini-batches sequentially, reusing the same compiled graph
        final_loss = None

        for mini_batch_idx in range(NUM_MINI_BATCHES):
            # Use slightly different key for each mini-batch (vary by mini_batch_idx)
            key, mini_key = jax.random.split(key)

            # Initialize best_config for this mini-batch if needed
            if best_configuration is None:
                best_config_for_batch = jnp.zeros((trial_n, DIM), dtype=param_dtype)
            else:
                best_config_for_batch = jnp.array(best_configuration, dtype=param_dtype)

            # Run batched update for this mini-batch
            sphere_centers_list[mini_batch_idx], opt_state_list[mini_batch_idx], mini_key, loss, new_best_margin, new_best_config = batch_fn(
                sphere_centers_list[mini_batch_idx],
                opt_state_list[mini_batch_idx],
                current_step,
                mini_key,
                best_margin,
                best_config_for_batch
            )
            loss.block_until_ready()
            final_loss = loss  # Keep last loss for logging

            # Update global best from JAX results
            new_best_margin_float = float(new_best_margin)
            if new_best_margin_float > best_margin:
                best_margin = new_best_margin_float
                best_configuration = np.array(new_best_config, dtype=np_dtype)
                best_mini_batch_idx = mini_batch_idx

            # Post-hook only at checkpoints (not JIT-compiled) - call per mini-batch
            if post_hook_accepts_info or post_hook_accepts_kwargs:
                info = {
                    "best_margin": best_margin,
                    "best_configuration": best_configuration,
                    "best_mini_batch_index": best_mini_batch_idx,
                    "mini_batch_index": mini_batch_idx
                }
                post_hook(
                    sphere_centers_list[mini_batch_idx],
                    opt_state_list[mini_batch_idx],
                    current_step,
                    mini_key,
                    optimizer,
                    None,
                    hparams,
                    loss,
                    trial_number,
                    info=info
                )
            else:
                post_hook(
                    sphere_centers_list[mini_batch_idx],
                    opt_state_list[mini_batch_idx],
                    current_step,
                    mini_key,
                    optimizer,
                    None,
                    hparams,
                    loss,
                    trial_number
                )

        current_step = checkpoint

        elapsed = time.time() - trial_start
        # Collect data for compact logging every 100k steps
        if current_step % 100000 == 0 or current_step == MAX_STEPS:
            step_markers.append(f"{current_step/1000000:.1f}m")
            loss_values.append(f"{float(final_loss):.3f}")
            margin_values.append(f"{best_margin:.3f}")

        # Early stopping for excellent solutions (margin >= 0 means constraint satisfied)
        if best_margin >= 0:
            score = best_margin
            if verbose:
                print(f"  Early stopping: Margin: {score:.6f}")
            break

    # Final checkpoint check if we completed all steps without hitting any checkpoints
    if best_configuration is None:
        # Compute margins for all mini-batches and combine them
        all_margins = []
        for mini_batch_idx in range(NUM_MINI_BATCHES):
            mini_margins = compute_margin_batch(sphere_centers_list[mini_batch_idx])
            all_margins.append(mini_margins)

        # Combine all mini-batch margins
        combined_margins = jnp.concatenate(all_margins, axis=0)  # Combine 4 mini-batches of 32 = 128 total

        margins_np = np.array(combined_margins)
        best_idx = np.argmax(margins_np)
        best_margin = margins_np[best_idx]
        # Determine which mini-batch and index within that mini-batch
        mini_batch_idx = best_idx // MINI_BATCH_SIZE
        idx_in_mini_batch = best_idx % MINI_BATCH_SIZE
        best_configuration = np.array(sphere_centers_list[mini_batch_idx][idx_in_mini_batch], dtype=np_dtype)

    # Final report
    final_margin = best_margin if best_margin > float('-inf') else -1000.0

    # Save trial data with conflict resolution
    trial_file_path = f"storage/shared/tmp/{submission_uuid}/trial_{trial_number}.npz"
    trial_counter = trial_number
    while os.path.exists(trial_file_path):
        trial_counter += 1
        trial_file_path = f"storage/shared/tmp/{submission_uuid}/trial_{trial_counter}.npz"

    if best_configuration is not None:
        # WARNING: This file is temporary and will be aggregated by the evaluator
        # Final results across all trials are saved to storage/{lineage_name}/data/eval_{eval_id}.npz
        np.savez(trial_file_path,
                 sphere_centers=best_configuration,
                 margin=final_margin,
                 steps_run=int(current_step),
                 hyperparameters=hparams,
                 trial_number=trial_counter)

    # Print compact logging table
    if verbose and step_markers:
        print("\nStep | " + " | ".join(step_markers))
        print("--- | " + " | ".join(["---"] * len(step_markers)))
        print("Loss | " + " | ".join(loss_values))
        print("Margin | " + " | ".join(margin_values))

    # Print final results
    if verbose:
        score = final_margin
        final_step = current_step / 1000000
        total_time = time.time() - trial_start
        print(f"\nBest score: {score:.6f}")
        print(f"Margin: {final_margin:.6f}")
        print(f"Total time: {total_time:.0f}s")
        print(f"Completed at: {final_step:.1f}m step")

    # Return final results (density is always 1.0 for kissing number - all coordinates are used)
    return final_margin, best_configuration, 1.0
