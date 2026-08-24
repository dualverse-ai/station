"""
CNN baseline submission for distributed RL on Sokoban.
Uses system defaults with basic hyperparameter search space.
"""
import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
import flax.linen as nn
import optax
from ray import tune
import time
from typing import Tuple, Dict, Any, Optional


def default_create_optimizer(learning_rate: float = 4e-4) -> optax.GradientTransformation:
    """Create and return the default optimizer for training."""
    return optax.adam(learning_rate=learning_rate, eps=1e-5)

# Uses system defaults for training_step and optimizer
