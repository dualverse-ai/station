#!/usr/bin/env python3
"""
Single-seed training script for ZAPBench neural activity forecasting.
This is adapted to work with Ray workers.
"""

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
import optax

# Set environment variables before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
os.environ['JAX_LOG_COMPILES'] = '0'

# Suppress all warnings
import warnings
warnings.filterwarnings('ignore')

# Training configuration
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10
INPUT_HORIZON = 4
OUTPUT_HORIZON = 32


def make_train_step(network, optimizer, compute_loss_fn):
    """Create a training step function with the network, optimizer, and loss.

    Args:
        network: Network object with apply method
        optimizer: Optimizer for parameter updates
        compute_loss_fn: Loss function (predictions, targets, params, x) -> scalar

    Returns:
        JIT-compiled training step function

    Notes:
        Networks can specify mutable collections by setting network.mutable attribute
        to a list of collection names (e.g., ['batch_stats']). If not specified,
        the system tries to infer from params structure.
    """
    # Check if network explicitly declares mutable collections
    mutable_collections = getattr(network, 'mutable', None)

    # Check if network needs RNG for stochastic layers (e.g., Dropout)
    needs_rng = getattr(network, 'needs_rng', False)

    @jit
    def train_step(params, opt_state, x, y, rng=None):
        """Single training step."""
        def loss_with_params(params):
            # Determine which collections to make mutable during training
            if mutable_collections is not None:
                # Agent explicitly specified mutable collections
                mutable = mutable_collections
            elif isinstance(params, dict) and 'params' in params:
                # Flax-style params with potential mutable state
                # Common mutable collections: batch_stats, intermediates
                mutable = [k for k in params.keys() if k in ['batch_stats', 'intermediates']]
            else:
                # Simple params dict - no mutable state
                mutable = []

            # Call network with appropriate mutable setting
            if mutable:
                # Prepare rngs dict if network needs RNG
                if needs_rng and rng is not None:
                    predictions, updates = network.apply(
                        params, x, training=True, mutable=mutable, rngs={'dropout': rng}
                    )
                else:
                    predictions, updates = network.apply(
                        params, x, training=True, mutable=mutable
                    )
                return compute_loss_fn(predictions, y, params, x), updates
            else:
                # No mutable state or simple model
                if needs_rng and rng is not None:
                    predictions = network.apply(params, x, training=True, rngs={'dropout': rng})
                else:
                    predictions = network.apply(params, x, training=True)
                return compute_loss_fn(predictions, y, params, x), None

        (loss, updates), grads = value_and_grad(loss_with_params, has_aux=True)(params)

        # Update mutable collections if present (e.g., batch_stats)
        if updates is not None and isinstance(updates, dict):
            for key, value in updates.items():
                if isinstance(params, dict):
                    params[key] = value

        # Handle parameter updates based on structure
        if isinstance(params, dict) and 'params' in params:
            # Flax-style: params = {'params': {...}, 'batch_stats': {...}}
            # Gradients are only for 'params' part
            param_grads = grads['params']
            trainable_params = params['params']

            # Update trainable parameters
            optimizer_updates, opt_state = optimizer.update(param_grads, opt_state, trainable_params)
            params['params'] = optax.apply_updates(trainable_params, optimizer_updates)
        else:
            # Simple params dict - apply updates directly
            optimizer_updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, optimizer_updates)

        return params, opt_state, loss

    return train_step


def load_data():
    """Load pre-processed ZAPBench data from storage/system/data."""
    # Data is in storage/system/data relative to shared storage root
    print("Loading ZAPBench data with condition labels...")
    # Keep as numpy arrays in CPU RAM - JAX will convert batches on-demand
    train_data = np.load('storage/system/data/train_data.npy')
    val_data = np.load('storage/system/data/val_data.npy')

    # Load condition labels
    train_labels = np.load('storage/system/data/train_condition_labels.npy')
    val_labels = np.load('storage/system/data/val_condition_labels.npy')

    # Force copy into RAM to avoid memory-mapped files hitting network
    train_data = np.array(train_data, copy=True)
    val_data = np.array(val_data, copy=True)
    train_labels = np.array(train_labels, copy=True)
    val_labels = np.array(val_labels, copy=True)

    print(f"Train: {train_data.shape}, Val: {val_data.shape} [numpy in CPU RAM]")
    print(f"Train labels: {train_labels.shape}, Val labels: {val_labels.shape}")
    return train_data, val_data, train_labels, val_labels


def create_batches(data, input_horizon, output_horizon, batch_size, condition_labels, shuffle_key=None):
    """Create batches for training, respecting condition boundaries.

    Args:
        data: Time series data
        input_horizon: Input sequence length
        output_horizon: Output sequence length
        batch_size: Batch size
        condition_labels: REQUIRED condition labels to prevent cross-condition sequences
        shuffle_key: Random key for shuffling
    """
    if condition_labels is None:
        raise ValueError("condition_labels is REQUIRED to prevent cross-condition contamination")

    timesteps, features = data.shape
    total_horizon = input_horizon + output_horizon

    # Create sequences that don't cross condition boundaries
    valid_indices = []
    for i in range(timesteps - total_horizon + 1):
        # Check if this sequence stays within one condition
        seq_labels = condition_labels[i:i+total_horizon]
        if np.all(seq_labels == seq_labels[0]):
            valid_indices.append(i)

    indices = np.array(valid_indices)
    num_samples = len(indices)

    if num_samples == 0:
        raise ValueError(f"No valid sequences found that don't cross condition boundaries")

    # Shuffle if key provided
    if shuffle_key is not None:
        indices = random.permutation(shuffle_key, indices)

    # Create batches
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]

        # Collect input and output windows
        input_batch = []
        output_batch = []

        for idx in batch_indices:
            idx = int(idx)
            input_batch.append(data[idx:idx + input_horizon])
            output_batch.append(data[idx + input_horizon:idx + input_horizon + output_horizon])

        yield jnp.array(input_batch), jnp.array(output_batch)


def compute_vanilla_mae(predictions, targets):
    """Standard MAE for fair evaluation - ALWAYS used for validation metric."""
    return jnp.mean(jnp.abs(predictions - targets))


def evaluate_with_step_maes(network, params, data, batch_size, condition_labels):
    """Evaluate model on dataset computing both overall and step-wise MAEs.

    Args:
        network: Network object with apply method
        params: Network parameters
        data: Dataset to evaluate on
        batch_size: Batch size for evaluation
        condition_labels: REQUIRED condition labels to prevent cross-condition sequences

    Returns:
        Tuple of (overall_mae, step_maes_dict) where step_maes_dict has keys [1,4,8,16,32]
    """
    if condition_labels is None:
        raise ValueError("condition_labels is REQUIRED for evaluation to prevent cross-condition contamination")

    total_abs_error = 0
    total_samples = 0

    # For step-wise MAEs
    step_horizons = [1, 4, 8, 16, 32]
    step_abs_errors = {h: 0.0 for h in step_horizons}
    step_samples = {h: 0 for h in step_horizons}

    for x_batch, y_batch in create_batches(data, INPUT_HORIZON, OUTPUT_HORIZON, batch_size,
                                          condition_labels, shuffle_key=None):
        # CRITICAL: Use training=False for evaluation (BatchNorm uses accumulated stats, Dropout disabled)
        predictions = network.apply(params, x_batch, training=False)

        # Compute total absolute error for this batch (not mean)
        batch_abs_error = jnp.sum(jnp.abs(predictions - y_batch))
        # Count total elements in this batch
        batch_samples = predictions.size
        # CRITICAL: Move to CPU immediately to free GPU memory
        total_abs_error += float(batch_abs_error)
        total_samples += batch_samples

        # Compute step-wise MAEs (always, minimal overhead)
        batch_size_actual = predictions.shape[0]
        num_neurons = predictions.shape[2]
        for horizon in step_horizons:
            # Extract only the h-th timestep (0-indexed: horizon-1)
            pred_step = predictions[:, horizon-1, :]
            target_step = y_batch[:, horizon-1, :]
            step_error = jnp.sum(jnp.abs(pred_step - target_step))
            step_samples[horizon] += batch_size_actual * num_neurons
            # Move to CPU immediately
            step_abs_errors[horizon] += float(step_error)

    overall_mae = total_abs_error / total_samples if total_samples > 0 else 0

    # Always compute step MAEs
    step_maes = {}
    for horizon in step_horizons:
        step_maes[horizon] = step_abs_errors[horizon] / step_samples[horizon] if step_samples[horizon] > 0 else 0

    return overall_mae, step_maes


def run_single_seed_training(funcs, hparams, seed, trial_timeout=300, patience=None, max_epochs=None):
    """
    Run training for a single seed with given hyperparameters.

    Args:
        funcs: Dictionary of functions from load_all_functions
        hparams: Hyperparameter dictionary
        seed: Random seed for this trial
        trial_timeout: Maximum time in seconds for training
        patience: Early stopping patience (default: uses EARLY_STOPPING_PATIENCE constant)
        max_epochs: Maximum number of epochs (default: uses MAX_EPOCHS constant)

    Returns:
        tuple: (best_val_mae, final_params, final_opt_state)
    """
    print(f"Starting ZAPBench training with seed {seed}")
    print(f"Hyperparameters: {hparams}")
    print(f"Batch size: {funcs['batch_size']}")
    print(f"Timeout: {trial_timeout/60:.1f} minutes")

    # Use provided patience or default to constant
    early_stopping_patience = patience if patience is not None else EARLY_STOPPING_PATIENCE
    # Use provided max_epochs or default to constant
    max_epochs_to_use = max_epochs if max_epochs is not None else MAX_EPOCHS
    print(f"Max epochs: {max_epochs_to_use}, Patience: {early_stopping_patience}")

    # Load data
    train_data, val_data, train_labels, val_labels = load_data()

    # Initialize random key
    rng = random.PRNGKey(seed)
    rng, init_rng = random.split(rng)
    num_neurons = train_data.shape[1]

    # Create network and parameters
    network = funcs['create_network'](hparams)
    dummy_input = jnp.zeros((1, INPUT_HORIZON, num_neurons))
    params = network.init(init_rng, dummy_input)

    # Create optimizer and initialize with trainable params only
    learning_rate = hparams.get('learning_rate', 0.001)
    optimizer = funcs['create_optimizer'](learning_rate)

    # Extract trainable parameters (excluding mutable state like batch_stats)
    if isinstance(params, dict) and 'params' in params:
        trainable_params = params['params']
    else:
        trainable_params = params

    opt_state = optimizer.init(trainable_params)

    # Get compute_loss function and batch size
    compute_loss_fn = funcs['compute_loss']
    batch_size = funcs['batch_size']

    # Create training step function with network and loss
    train_step = make_train_step(network, optimizer, compute_loss_fn)

    # Check if network needs RNG (for Dropout, etc.)
    needs_rng = getattr(network, 'needs_rng', False)

    # Training loop
    best_val_mae = float('inf')
    best_params = params
    best_step_maes = None  # Will store step MAEs for best model
    patience_counter = 0
    start_time = time.time()

    # Print header for training table
    print("\n" + "="*40)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val MAE':^12}")
    print("="*40)

    for epoch in range(max_epochs_to_use):
        epoch_start = time.time()
        rng, shuffle_rng = random.split(rng)

        # Training
        train_losses = []

        for x_batch, y_batch in create_batches(train_data, INPUT_HORIZON, OUTPUT_HORIZON,
                                                batch_size, train_labels, shuffle_rng):
            # Split RNG for dropout if network needs it
            if needs_rng:
                rng, dropout_rng = random.split(rng)
                params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch, dropout_rng)
            else:
                params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            # CRITICAL: Move loss to CPU immediately to free GPU memory
            train_losses.append(float(loss))

        # Compute epoch metrics with step MAEs
        train_loss = np.mean(train_losses)  # Average loss from training batches
        val_mae, step_maes = evaluate_with_step_maes(network, params, val_data, batch_size, val_labels)

        # Early stopping check
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = params
            best_step_maes = step_maes  # Save step MAEs for best model
            patience_counter = 0
        else:
            patience_counter += 1

        # Check timeout
        elapsed_time = time.time() - start_time

        # Print epoch row
        print(f"{epoch+1:^7} | {train_loss:^12.6f} | {val_mae:^12.6f}")

        if elapsed_time > trial_timeout:
            print("="*40)
            print(f"Timeout after {elapsed_time:.1f}s at epoch {epoch+1}")
            break

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print("="*40)
            print(f"Early stopping at epoch {epoch+1} (patience {early_stopping_patience})")
            break

    # Final results
    print("="*40)
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Best validation MAE: {best_val_mae:.6f}")

    return best_val_mae, best_params, opt_state, best_step_maes