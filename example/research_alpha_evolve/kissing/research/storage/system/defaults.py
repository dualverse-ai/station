"""
Default implementations for kissing number optimization functions.
These are used when agent doesn't provide their own implementations.
"""

import jax
import jax.numpy as jnp
import optax
import time

# Default base seed for reproducibility (agents can override in their submission.py)
BASE_SEED = int(time.time()) % 1000000

# Fixed parameters
DIM = 11
N = 593  # Fixed number of spheres


def default_define_hyperparameters():
    """
    Define Ray Tune search space for hyperparameters.
    Returns a dictionary with Ray Tune search space definitions.
    """
    from ray import tune
    search_space = {
        'learning_rate': tune.loguniform(0.001, 0.1),
        'init_scale': tune.loguniform(0.01, 1.0)
    }
    return search_space


def default_get_optimizer(hparams):
    """
    Create optimizer based on hyperparameters.
    Default uses Adam optimizer.
    
    Args:
        hparams: Dictionary of hyperparameters from define_hyperparameters
    """
    return optax.adam(hparams['learning_rate'])


def default_get_init_fn(hparams):
    """
    Create initializer based on hyperparameters.
    Default uses normal initialization.
    
    Args:
        hparams: Dictionary of hyperparameters from define_hyperparameters
    """
    init_scale = hparams['init_scale']
    
    def init_fn(key, shape, dtype=jnp.float32):
        # Initialize with normal distribution
        return jax.random.normal(key, shape, dtype=dtype) * init_scale
    
    return init_fn


@jax.jit
def compute_margin_batch(sphere_centers):
    """Compute margin for a batch of sphere center configurations (EFFICIENT VERSION)."""
    batch_size = sphere_centers.shape[0]
    n = sphere_centers.shape[1]  # Get N from shape

    # Compute norms for each configuration
    norms = jnp.linalg.norm(sphere_centers, axis=2)  # (batch_size, N)
    max_norms = jnp.max(norms, axis=1)  # (batch_size,)

    # Normalize each configuration by its max norm (avoid division by zero)
    max_norms = jnp.maximum(max_norms, 1e-10)
    normalized_centers = sphere_centers / max_norms[:, None, None]  # (batch_size, N, DIM)

    # Efficient pairwise distance computation using ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
    dots = jnp.einsum('bid,bjd->bij', normalized_centers, normalized_centers)  # (batch_size, N, N)
    norms_sq = jnp.sum(normalized_centers**2, axis=2, keepdims=True)  # (batch_size, N, 1)
    squared_distances = norms_sq + norms_sq.transpose(0, 2, 1) - 2 * dots  # (batch_size, N, N)

    # Mask diagonal (self-distances are 0)
    mask = ~jnp.eye(n, dtype=bool)

    # Get minimum pairwise distance for each batch
    # Use a large value for diagonal entries to exclude them from min
    masked_distances = jnp.where(mask[None, :, :], squared_distances, jnp.inf)
    min_squared_distances = jnp.min(masked_distances, axis=(1, 2))  # (batch_size,)
    min_distances = jnp.sqrt(jnp.maximum(min_squared_distances, 1e-20))  # Avoid sqrt(0)

    # Margin = min_distance - max_norm (max_norm is 1 after normalization)
    margins = min_distances - 1.0

    return margins


def default_update_func(sphere_centers, opt_state, global_step, rng, optimizer, constraints_data, hparams):
    """
    Default update function with simple gradient descent on margin loss.
    
    Args:
        sphere_centers: Current parameters, shape (batch_size, N, DIM)
        opt_state: Optimizer state
        global_step: Current training step
        rng: Random key (not used in default implementation)
        optimizer: Optimizer object
        constraints_data: Constraint-related data (not used in default)
        hparams: Dictionary of hyperparameters from define_hyperparameters
    
    Returns:
        Can return either:
        - (sphere_centers, opt_state, loss) - system will compute margins
        - (sphere_centers, opt_state, loss, margins) - margins shape (batch_size,)
    """
    # Compute margins and loss, reusing the computation
    def loss_and_margins_fn(params):
        margins = compute_margin_batch(params)
        loss = jnp.mean(-margins)  # Negative margin as loss
        return loss, margins
    
    (loss, margins), grads = jax.value_and_grad(loss_and_margins_fn, has_aux=True)(sphere_centers)
    
    # Apply optimizer updates
    updates, opt_state = optimizer.update(grads, opt_state, sphere_centers)
    sphere_centers = optax.apply_updates(sphere_centers, updates)
    
    # Return pre-update margins (for best tracking of previous step configuration)
    return sphere_centers, opt_state, loss, margins


def default_post_hook(sphere_centers, opt_state, global_step, rng, optimizer, constraints_data, hparams, loss_value, trial_number):
    """
    Default post-hook function (does nothing).

    Args:
        sphere_centers: Current parameters, shape (batch_size, N, DIM)
        opt_state: Optimizer state
        global_step: Current training step
        rng: Random key
        optimizer: Optimizer object
        constraints_data: Constraint-related data (not used)
        hparams: Dictionary of hyperparameters
        loss_value: Loss value from the update step
        trial_number: Trial number for this experiment
    """
    pass
