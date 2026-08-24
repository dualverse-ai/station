"""
Default implementations for Multi-Dataset RNA Sequence Modeling task.
"""

import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
import flax.linen as nn
import optax
import time
from typing import Tuple, Dict, Any, Optional

# Default base seed for reproducibility
BASE_SEED = int(time.time()) % 1000000


def default_define_hyperparameters():
    """Define default hyperparameter values."""
    return {
        'learning_rate': 0.001,
        'hidden_dim': 512,
        'dropout_rate': 0.1
    }


class VanillaMLP(nn.Module):
    """Adaptive RNA network for sequence-level predictions."""
    d_input: int = 4  # One-hot RNA encoding
    d_output: int = 1
    task_type: str = "regression"
    hidden_dim: int = 512
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic=True):
        """
        Forward pass for RNA sequence prediction.

        Args:
            x: Input sequences (batch_size, seq_len, d_input)
            deterministic: Whether to use deterministic mode (no dropout)

        Returns:
            - Sequence-level regression: (batch,)
            - Sequence-level multilabel: (batch, d_output)
        """
        # x shape: (batch, seq_len, d_input)
        x = nn.Dense(self.hidden_dim)(x)  # (batch, seq_len, hidden_dim)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # Sequence-level: project then pool
        x = nn.Dense(self.d_output)(x)  # (batch, seq_len, d_output)
        x = jnp.mean(x, axis=1)  # (batch, d_output)

        if self.task_type == "regression":
            return x.squeeze(-1)  # (batch,)
        else:
            return x  # (batch, d_output) for multilabel classification or regression


class DefaultRNANetwork:
    """Default network wrapper that implements the required interface."""

    def __init__(self, d_input=4, d_output=1, task_type="regression", hidden_dim=512, dropout_rate=0.1):
        self.network = VanillaMLP(
            d_input=d_input,
            d_output=d_output,
            task_type=task_type,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

    def init(self, rng_key, dummy_input):
        """Initialize network parameters."""
        params_key, dropout_key = random.split(rng_key)
        variables = self.network.init({'params': params_key, 'dropout': dropout_key},
                                     dummy_input, deterministic=True)
        return variables['params']

    def apply(self, params, x, deterministic=True, rng_key=None):
        """Apply network forward pass."""
        if deterministic:
            # No dropout needed for deterministic mode
            return self.network.apply({'params': params}, x, deterministic=True)
        else:
            # Need dropout PRNG for training
            if rng_key is None:
                # Generate a random key if not provided (for backward compatibility)
                rng_key = random.PRNGKey(0)
            return self.network.apply({'params': params}, x, deterministic=False,
                                    rngs={'dropout': rng_key})


def default_create_network(hparams: Dict[str, Any]):
    """Create and return the default RNA network."""
    d_output = hparams.get("d_output", 1)
    task_type = hparams.get("task_type", "regression")
    hidden_dim = hparams.get("hidden_dim", 512)
    dropout_rate = hparams.get("dropout_rate", 0.1)

    # All datasets use 4-channel input (one-hot RNA)
    d_input = 4

    return DefaultRNANetwork(
        d_input=d_input,
        d_output=d_output,
        task_type=task_type,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    )


def hardcoded_compute_loss(predictions, targets, params, x, hparams):
    """
    Hardcoded loss function - agents CANNOT override this.

    Args:
        predictions: Network output
            - Sequence regression: (batch,)
            - Sequence multilabel classification: (batch, d_output)
            - Sequence multilabel regression: (batch, d_output)
        targets: Ground truth (same shape as predictions)
        params: Network parameters (unused)
        x: Input data (unused)
        hparams: Hyperparameters dict with 'task_type'

    Returns:
        Scalar loss value
    """
    task_type = hparams.get('task_type', 'regression')

    if task_type == 'regression':
        # Standard MSE for APA (single-output regression)
        return jnp.mean((predictions - targets) ** 2)

    elif task_type == 'multilabel_regression':
        # MSE for PRS (multi-output regression)
        return jnp.mean((predictions - targets) ** 2)

    elif task_type == 'multilabel_classification':
        # Binary cross-entropy for Modif
        return optax.sigmoid_binary_cross_entropy(predictions, targets).mean()

    else:
        raise ValueError(f"Unknown task_type: {task_type}")


# Keep old name for backward compatibility, but this should not be used
def default_compute_loss(predictions, targets, params, x):
    """DEPRECATED: Use hardcoded_compute_loss instead. This is kept for compatibility only."""
    # This should never be called in the new system
    raise NotImplementedError("Loss function is now hardcoded. Use hardcoded_compute_loss instead.")


def default_create_optimizer(learning_rate: float = 0.001) -> optax.GradientTransformation:
    """Create and return the default optimizer."""
    return optax.adamw(learning_rate=learning_rate, weight_decay=0.01)


def default_complete(params: Any, opt_state: Any, trial_data: Dict[str, Any]) -> None:
    """Default completion function called when training finishes."""
    pass