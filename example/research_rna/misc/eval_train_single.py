#!/usr/bin/env python3
"""
Single-dataset training script for Multi-Dataset RNA Sequence Modeling.
This is adapted to work with Ray workers and different RNA datasets.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
import optax
from scipy.stats import spearmanr

# Set environment variables before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
os.environ['JAX_LOG_COMPILES'] = '0'

# Suppress all warnings
import warnings
warnings.filterwarnings('ignore')

# Training configuration
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10  # Stop if no improvement for 10 epochs

# Dataset configurations (expanded for evaluation with 7 datasets)
DATASET_CONFIGS = {
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


def one_hot_encode_rna_sequences(sequences):
    """One-hot encode RNA sequences with complete IUPAC nucleotide support."""
    mapping = np.array([
        [1, 0, 0, 0],          # A
        [0, 1, 0, 0],          # T/U
        [0, 0, 1, 0],          # C
        [0, 0, 0, 1],          # G
        [0.0, 0.0, 0.0, 0.0],  # N (any)
        [0.0, 0.0, 0.0, 0.0],  # X (unknown)
        # IUPAC ambiguous nucleotides (probabilistic encoding)
        [0.0, 0.5, 0.0, 0.5],  # K = G or T
        [0.5, 0.0, 0.5, 0.0],  # M = A or C
        [0.5, 0.0, 0.0, 0.5],  # R = A or G
        [0.0, 0.0, 0.5, 0.5],  # S = C or G
        [0.5, 0.5, 0.0, 0.0],  # W = A or T
        [0.0, 0.5, 0.5, 0.0],  # Y = C or T
        [0.0, 0.33, 0.33, 0.33],  # B = C or G or T (not A)
        [0.33, 0.33, 0.33, 0.0],  # D = A or G or T (not C)
        [0.33, 0.0, 0.33, 0.33],  # H = A or C or T (not G)
        [0.33, 0.33, 0.0, 0.33]   # V = A or C or G (not T)
    ])
    char_to_int = {'A': 0, 'T': 1, 'U': 1, 'C': 2, 'G': 3, 'N': 4, 'X': 5,
                   'K': 6, 'M': 7, 'R': 8, 'S': 9, 'W': 10, 'Y': 11,
                   'B': 12, 'D': 13, 'H': 14, 'V': 15}

    # Find max length and pad sequences
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = [seq.upper().ljust(max_len, 'N') for seq in sequences]

    # Convert to indices - FAIL on unknown characters
    seq_indices = []
    for seq in padded_seqs:
        seq_idx = []
        for char in seq:
            if char not in char_to_int:
                raise ValueError(f"Unknown nucleotide character: '{char}' in sequence. Valid: {list(char_to_int.keys())}")
            seq_idx.append(char_to_int[char])
        seq_indices.append(seq_idx)

    # One-hot encode
    encoded = np.array([mapping[seq] for seq in seq_indices], dtype=np.float32)
    return encoded


def process_multilabel_labels(label_strings):
    """Convert multi-label strings to binary arrays."""
    processed_labels = []

    for i, label_str in enumerate(label_strings):
        # Validate label string
        if not isinstance(label_str, str):
            raise ValueError(f"Label at index {i} is not a string: {type(label_str)}")

        # Split space-separated values and convert to integers
        try:
            label_array = [int(x) for x in label_str.strip().split()]
        except ValueError as e:
            raise ValueError(f"Invalid multi-label format at index {i}: '{label_str}'") from e

        # Validate binary values (0 or 1)
        invalid_labels = [x for x in label_array if x not in [0, 1]]
        if invalid_labels:
            raise ValueError(f"Invalid binary labels at index {i}: {set(invalid_labels)}. Valid: 0,1")

        processed_labels.append(label_array)

    return np.array(processed_labels, dtype=np.float32)  # Float for BCE loss


def load_dataset(dataset_name: str):
    """Load and preprocess a specific RNA dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Valid datasets: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]
    data_dir = f"storage/system/data/{config['data_dir']}"

    # Load train and validation sets
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/val.csv")

    # Handle dataset-specific sequence column names and preprocessing
    if dataset_name == "CRI-Off":
        # Create concatenated sequences: sgRNA + target
        train_sequences = (train_df['sgrna'] + train_df['target']).values
        val_sequences = (val_df['sgrna'] + val_df['target']).values
    elif dataset_name == "APA":
        train_sequences = train_df['seq'].values
        val_sequences = val_df['seq'].values
    elif dataset_name == "Modif":
        train_sequences = train_df['sequence'].values
        val_sequences = val_df['sequence'].values
    elif dataset_name == "CRI-On":
        train_sequences = train_df['seq'].values
        val_sequences = val_df['seq'].values
    elif dataset_name == "PRS":
        train_sequences = train_df['sequence'].values
        val_sequences = val_df['sequence'].values
    elif dataset_name == "MRL":
        train_sequences = train_df['seq'].values
        val_sequences = val_df['seq'].values
    elif dataset_name == "ncRNA":
        train_sequences = train_df['sequence'].values
        val_sequences = val_df['sequence'].values
    else:
        raise ValueError(f"Sequence column not configured for dataset: {dataset_name}")

    # One-hot encode sequences
    train_X = one_hot_encode_rna_sequences(train_sequences)
    val_X = one_hot_encode_rna_sequences(val_sequences)

    # Process labels based on task type
    if config['task_type'] == 'regression':
        if dataset_name == "APA":
            train_y = train_df['proximal_isoform_proportion'].values.astype(np.float32)
            val_y = val_df['proximal_isoform_proportion'].values.astype(np.float32)
        elif dataset_name in ["CRI-Off", "CRI-On", "MRL"]:
            train_y = train_df['label'].values.astype(np.float32)
            val_y = val_df['label'].values.astype(np.float32)
    elif config['task_type'] == 'multilabel_classification':
        # Modif dataset: space-separated binary labels
        train_y = process_multilabel_labels(train_df['label'].values)
        val_y = process_multilabel_labels(val_df['label'].values)
    elif config['task_type'] == 'multilabel_regression':
        # PRS dataset: Multi-label regression (ON, OFF, ON_OFF)
        train_y = train_df[['ON', 'OFF', 'ON_OFF']].values.astype(np.float32)
        val_y = val_df[['ON', 'OFF', 'ON_OFF']].values.astype(np.float32)
    elif config['task_type'] == 'multiclass_classification':
        # ncRNA dataset: Multi-class classification
        train_y = train_df['label'].values.astype(np.int64)
        val_y = val_df['label'].values.astype(np.int64)

    return (jnp.array(train_X), jnp.array(train_y)), (jnp.array(val_X), jnp.array(val_y))


def make_train_step(network, optimizer, compute_loss_fn, hparams):
    """Create a training step function with the network, optimizer, and loss.

    Args:
        network: Network object with apply method
        optimizer: Optimizer for parameter updates
        compute_loss_fn: Loss function (predictions, targets, params, x, hparams) -> scalar
        hparams: Hyperparameters dict

    Returns:
        JIT-compiled training step function
    """
    @jit
    def train_step(params, opt_state, x, y, rng_key):
        """Single training step."""
        def loss_with_params(params):
            predictions = network.apply(params, x, deterministic=False, rng_key=rng_key)
            return compute_loss_fn(predictions, y, params, x, hparams)

        loss, grads = value_and_grad(loss_with_params)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step


def make_eval_step(network):
    """Create an evaluation step function."""
    @jit
    def eval_step(params, x, y):
        """Single evaluation step."""
        predictions = network.apply(params, x, deterministic=True)
        return predictions

    return eval_step


def create_batches(sequences, labels, batch_size, shuffle_key=None):
    """Create batched data."""
    n_samples = sequences.shape[0]

    if shuffle_key is not None:
        indices = jax.random.permutation(shuffle_key, n_samples)
        sequences = sequences[indices]
        labels = labels[indices]

    # Pad to full batches
    n_batches = (n_samples + batch_size - 1) // batch_size
    padded_size = n_batches * batch_size

    if padded_size > n_samples:
        pad_size = padded_size - n_samples
        sequences = jnp.concatenate([
            sequences,
            jnp.zeros((pad_size, *sequences.shape[1:]), dtype=sequences.dtype)
        ])

        # Handle both 1D and 2D label shapes
        if labels.ndim == 1:
            # Sequence-level labels (APA, CRI-Off)
            label_pad = jnp.zeros(pad_size, dtype=labels.dtype)
        else:
            # Multi-label (Modif)
            label_pad = jnp.zeros((pad_size, *labels.shape[1:]), dtype=labels.dtype)

        labels = jnp.concatenate([labels, label_pad])

    # Reshape sequences and labels
    batched_sequences = sequences.reshape(n_batches, batch_size, *sequences.shape[1:])

    if labels.ndim == 1:
        # 1D labels: (n_samples,) → (n_batches, batch_size)
        batched_labels = labels.reshape(n_batches, batch_size)
    else:
        # 2D labels: (n_samples, num_labels) → (n_batches, batch_size, num_labels)
        batched_labels = labels.reshape(n_batches, batch_size, *labels.shape[1:])

    return batched_sequences, batched_labels


def compute_r2_score(y_true, y_pred):
    """Compute R² score without sklearn."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0 if ss_res != 0 else 1.0

    return 1 - (ss_res / ss_tot)


def compute_auc_roc_macro(y_true, y_score):
    """Compute macro-averaged AUC-ROC without sklearn."""
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        # Get scores and labels for this class
        scores = y_score[:, i]
        labels = y_true[:, i]

        # Skip if only one class present
        if len(np.unique(labels)) < 2:
            continue

        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = labels[sorted_indices]

        # Compute TPR and FPR
        n_pos = np.sum(labels)
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            continue

        tpr = np.zeros(len(labels) + 1)
        fpr = np.zeros(len(labels) + 1)

        tp = 0
        fp = 0

        for j, label in enumerate(sorted_labels):
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr[j + 1] = tp / n_pos
            fpr[j + 1] = fp / n_neg

        # Compute AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)
        auc_scores.append(auc)

    if not auc_scores:
        return 0.0

    return np.mean(auc_scores)


def compute_metric(task_type: str, metric: str, predictions, targets):
    """Compute task-specific metric."""
    predictions = np.array(predictions)
    targets = np.array(targets)

    if task_type == "regression":
        if metric == "r2":
            return compute_r2_score(targets, predictions)
        elif metric == "spearman":
            corr, _ = spearmanr(targets, predictions)
            if np.isnan(corr):
                print(f"Warning: Spearman correlation is NaN")
                return 0.0
            return corr
    elif task_type == "multilabel_classification":
        if metric == "auc_roc":
            # predictions should be probabilities (sigmoid applied)
            if predictions.ndim == 1:
                raise ValueError("AUC-ROC requires probability scores, not predictions")

            # Apply sigmoid to convert logits to probabilities
            probabilities = 1.0 / (1.0 + np.exp(-predictions))  # sigmoid

            try:
                return compute_auc_roc_macro(targets, probabilities)
            except Exception as e:
                print(f"Warning: AUC-ROC calculation failed: {e}")
                return 0.0
    elif task_type == "multilabel_regression":
        # Multi-label regression: compute average R2 across all labels
        if metric == "r2":
            r2_scores = []
            for i in range(targets.shape[1]):
                r2 = compute_r2_score(targets[:, i], predictions[:, i])
                r2_scores.append(r2)
            return np.mean(r2_scores)
    elif task_type == "multiclass_classification":
        # Multi-class classification: compute accuracy
        if metric == "accuracy":
            # predictions are logits, get argmax for predicted class
            if predictions.ndim > 1:
                pred_classes = np.argmax(predictions, axis=1)
            else:
                pred_classes = predictions
            accuracy = np.mean(pred_classes == targets)
            return accuracy

    raise ValueError(f"Unknown metric: {metric} for task type: {task_type}")


def run_single_dataset_training(funcs, hparams, seed, dataset_name, trial_timeout):
    """Run training on a single RNA dataset.

    Args:
        funcs: Dictionary of functions (network creation, loss, optimizer, etc.)
        hparams: Hyperparameters including dataset-specific info
        seed: Random seed for reproducibility
        dataset_name: Name of dataset to train on
        trial_timeout: Timeout in seconds

    Returns:
        Tuple of (best_val_metric, final_params, final_opt_state)
    """

    # Load dataset
    (train_X, train_y), (val_X, val_y) = load_dataset(dataset_name)

    config = DATASET_CONFIGS[dataset_name]
    task_type = config['task_type']
    metric = config['metric']

    # Initialize network
    network = funcs['create_network'](hparams)

    # Initialize parameters
    key = random.PRNGKey(seed)
    init_key, train_key = random.split(key)

    dummy_input = jnp.ones((1, *train_X.shape[1:]))
    params = network.init(init_key, dummy_input)

    # Create optimizer
    learning_rate = hparams.get('learning_rate', 0.001)
    optimizer = funcs['create_optimizer'](learning_rate)
    opt_state = optimizer.init(params)

    # Force use system default loss for multiclass classification and multilabel regression
    if task_type == 'multiclass_classification':
        def multiclass_cross_entropy_loss(predictions, targets, params, x, hparams):
            """Cross-entropy loss for multiclass classification."""
            # predictions: (batch_size, num_classes) logits
            # targets: (batch_size,) class indices
            log_probs = jax.nn.log_softmax(predictions, axis=-1)
            one_hot_targets = jax.nn.one_hot(targets, num_classes=predictions.shape[-1])
            return -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))

        compute_loss_fn = multiclass_cross_entropy_loss
    elif task_type == 'multilabel_regression':
        def multilabel_mse_loss(predictions, targets, params, x, hparams):
            """MSE loss for multilabel regression."""
            # predictions: (batch_size, num_labels)
            # targets: (batch_size, num_labels)
            return jnp.mean((predictions - targets) ** 2)

        compute_loss_fn = multilabel_mse_loss
    else:
        compute_loss_fn = funcs['compute_loss']

    # Create training and evaluation functions
    train_step = make_train_step(network, optimizer, compute_loss_fn, hparams)
    eval_step = make_eval_step(network)

    # Training configuration
    batch_size = funcs.get('batch_size', 64)

    print(f"\n{dataset_name} Training ({task_type}, {metric})")
    print(f"Data: {train_X.shape[0]} train, {val_X.shape[0]} val, seq_len={train_X.shape[1]}")

    # Training loop
    best_val_metric = -float('inf')
    best_params = params
    epochs_without_improvement = 0
    start_time = time.time()

    # Print header for training table
    print("="*45)
    print(f"{'Epoch':^5}|{'Loss':^8}|{'Val ' + metric.upper():^10}|{'Best':^8}")
    print("="*45)

    for epoch in range(MAX_EPOCHS):
        # Check timeout
        if time.time() - start_time > trial_timeout:
            print("="*45)
            print(f"Timeout epoch {epoch+1}")
            break

        # Shuffle training data
        train_key, shuffle_key = random.split(train_key)
        batched_train_X, batched_train_y = create_batches(
            train_X, train_y, batch_size, shuffle_key
        )

        # Training epoch
        epoch_loss = 0.0
        num_batches = len(batched_train_X)

        for batch_X, batch_y in zip(batched_train_X, batched_train_y):
            # Generate new RNG key for dropout in this batch
            train_key, dropout_key = random.split(train_key)
            params, opt_state, loss = train_step(params, opt_state, batch_X, batch_y, dropout_key)
            epoch_loss += loss

        avg_epoch_loss = epoch_loss / num_batches

        # Validation
        batched_val_X, batched_val_y = create_batches(val_X, val_y, batch_size)

        all_predictions = []
        all_targets = []

        for batch_X, batch_y in zip(batched_val_X, batched_val_y):
            predictions = eval_step(params, batch_X, batch_y)
            all_predictions.append(predictions)
            all_targets.append(batch_y)

        # Concatenate and trim to original validation size
        all_predictions = jnp.concatenate(all_predictions)[:len(val_y)]
        all_targets = jnp.concatenate(all_targets)[:len(val_y)]

        # Compute validation metric
        val_metric = compute_metric(task_type, metric, all_predictions, all_targets)

        # Track best model and early stopping
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_params = params
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"{epoch+1:^5}|{avg_epoch_loss:^8.4f}|{val_metric:^10.4f}|{best_val_metric:^8.4f}")

        # Early stopping check
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print("="*45)
            print(f"Early stop at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # Print final summary
    print("="*45)
    total_time = time.time() - start_time
    print(f"Best:{best_val_metric:.4f} Time:{total_time:.0f}s")

    return best_val_metric, best_params, opt_state