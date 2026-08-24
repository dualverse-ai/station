"""
Default implementations for ZAPBench neural activity forecasting task.
"""

import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
import optax
import flax.linen as nn
import time
from typing import Tuple, Dict, Any, Optional

# Default base seed for reproducibility
BASE_SEED = int(time.time()) % 1000000


def default_define_hyperparameters():
    """Define default hyperparameter values."""
    return {'learning_rate': 0.001}


class SharedNeuronMLP(nn.Module):
    """
    Shared-neuron MLP baseline with BatchNorm and Dropout.
    Processes the last 4 activation values through a shared MLP for all neurons.
    This means zero independence between neurons - all neurons share the same weights.
    """
    hidden_size: int = 64
    output_horizon: int = 32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        Forward pass through shared MLP.

        Args:
            x: Input shape (batch_size, 4, num_neurons)
            training: Whether in training mode (for BatchNorm and Dropout)

        Returns:
            Output shape (batch_size, 32, num_neurons)
        """
        batch_size, input_horizon, num_neurons = x.shape

        # Reshape to process all neurons together
        # From (batch_size, 4, num_neurons) to (batch_size * num_neurons, 4)
        x = jnp.transpose(x, (0, 2, 1))  # (batch_size, num_neurons, 4)
        x = jnp.reshape(x, (batch_size * num_neurons, input_horizon))

        # Shared MLP: input_dim (4) -> hidden (64) -> output (32)
        # Layer 1 with BatchNorm and Dropout
        x = nn.Dense(self.hidden_size)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Output layer
        x = nn.Dense(self.output_horizon)(x)

        # Reshape back to (batch_size, output_horizon, num_neurons)
        x = jnp.reshape(x, (batch_size, num_neurons, self.output_horizon))
        x = jnp.transpose(x, (0, 2, 1))  # (batch_size, 32, num_neurons)

        return x


class SharedMLPWrapper:
    """
    Wrapper to make Flax model compatible with the training system.
    Handles BatchNorm state and Dropout RNG requirements.
    """

    def __init__(self):
        self.model = SharedNeuronMLP()
        # Declare mutable collections (BatchNorm stats)
        self.mutable = ['batch_stats']
        # Declare RNG requirement (for Dropout)
        self.needs_rng = True

    def init(self, rng_key, dummy_input):
        """Initialize parameters with BatchNorm stats."""
        # Split keys for params and dropout
        rng_params, rng_dropout = random.split(rng_key)
        variables = self.model.init(
            {'params': rng_params, 'dropout': rng_dropout},
            dummy_input,
            training=True
        )
        return variables

    def apply(self, params, x, training=False, mutable=None, rngs=None):
        """Apply network with proper handling of BatchNorm and Dropout.

        Args:
            params: Network parameters (including batch_stats)
            x: Input data
            training: Whether in training mode
            mutable: Mutable collections to update (e.g., ['batch_stats'])
            rngs: Random number generators (e.g., {'dropout': key})
        """
        if mutable is not None:
            # Training mode with mutable batch_stats
            output, updates = self.model.apply(
                params,
                x,
                training=training,
                mutable=mutable,
                rngs=rngs if rngs is not None else {}
            )
            return output, updates
        else:
            # Inference mode
            return self.model.apply(
                params,
                x,
                training=training,
                rngs=rngs if rngs is not None else {}
            )


def default_create_network(hparams: Dict[str, Any]):
    """Create and return the default baseline network (shared-neuron MLP)."""
    return SharedMLPWrapper()


def default_compute_loss(predictions, targets, params, x):
    """Default loss function: Mean Absolute Error.

    Args:
        predictions: Network output [batch, timesteps, features]
        targets: Ground truth [batch, timesteps, features]
        params: Network parameters (unused in default, available for regularization)
        x: Input data (unused in default, available for input-dependent losses)

    Returns:
        Scalar loss value (MAE)
    """
    return jnp.mean(jnp.abs(predictions - targets))


def default_create_optimizer(learning_rate: float = 0.001) -> optax.GradientTransformation:
    """Create and return the default optimizer for training."""
    return optax.adam(learning_rate=learning_rate, eps=1e-8)


def default_complete(params: Any, opt_state: Any, trial_data: Dict[str, Any]) -> None:
    """Default completion function called when training finishes."""