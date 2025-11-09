"""
Default implementations for Sokoban RL optimization functions with Ray Tune.
These are used when agent doesn't provide their own implementations.
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
import flax.linen as nn
import optax
from ray import tune
import time
from typing import Tuple, Dict, Any, Optional

# Default base seed for reproducibility (agents can override in their submission.py)
BASE_SEED = int(time.time()) % 1000000


def default_define_hyperparameters():
    """
    Define default hyperparameter search space.
    Returns a dictionary with Ray Tune search space definitions.
    """
    # Fixed values (agents can define ranges in their own _define_hyperparameters)
    search_space = {
        'learning_rate': tune.choice([4e-4]),
        'entropy_coef': tune.choice([0.01]),
        'value_loss_coef': tune.choice([0.5]),
        'cnn_features_1': tune.choice([32]),
        'cnn_features_2': tune.choice([64]),
        'lstm_features': tune.choice([256])
    }
    
    return search_space


class DefaultResidualCNN(nn.Module):
    """Default CNN with residual connections for Sokoban."""
    cnn_features_1: int
    cnn_features_2: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, done: jnp.ndarray, rnn_state: Optional[Dict] = None) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Forward pass through the network."""
        batch_size = x.shape[0]
        
        # First convolutional block
        x = nn.Conv(features=self.cnn_features_1, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        
        # Second block with residual connection
        conv2_in = x
        x = nn.Conv(features=self.cnn_features_1, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.cnn_features_1, kernel_size=(3, 3), padding='SAME')(x)
        x = x + conv2_in  # Residual connection
        x = nn.relu(x)
        
        # Third block - increase channels
        x = nn.Conv(features=self.cnn_features_2, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        
        # Fourth block with residual connection
        conv4_in = x
        x = nn.Conv(features=self.cnn_features_2, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.cnn_features_2, kernel_size=(3, 3), padding='SAME')(x)
        x = x + conv4_in  # Residual connection
        x = nn.relu(x)
        
        # Flatten spatial dimensions
        x = x.reshape((batch_size, -1))
        
        # Shared MLP backbone
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        
        # Separate heads for policy and value
        policy_hidden = nn.Dense(features=64)(x)
        policy_hidden = nn.relu(policy_hidden)
        policy_logits = nn.Dense(features=4)(policy_hidden)
        
        value_hidden = nn.Dense(features=64)(x)
        value_hidden = nn.relu(value_hidden)
        value = nn.Dense(features=1)(value_hidden).squeeze(-1)
        
        return policy_logits, value, None


def default_create_network(hparams: Dict[str, Any]):
    """Create and return the default agent's neural network architecture."""
    return DefaultResidualCNN(
        cnn_features_1=hparams['cnn_features_1'],
        cnn_features_2=hparams['cnn_features_2']
    )


def default_create_optimizer(learning_rate: float = 4e-4) -> optax.GradientTransformation:
    """Create and return the default optimizer for training."""
    return optax.adam(learning_rate=learning_rate, eps=1e-5)


@jit
def default_calculate_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.97,
    gae_lambda: float = 0.97,
    next_values: jnp.ndarray = None,
) -> jnp.ndarray:
    """Calculate Generalized Advantage Estimation (GAE)."""
    assert next_values is not None, "next_values is required for bootstrapping"
    
    values_t_plus_1 = jnp.concatenate([values[:, 1:], next_values[:, None]], axis=1)
    not_done_mask = (1.0 - dones.astype(jnp.float32))
    deltas = rewards + gamma * values_t_plus_1 * not_done_mask - values
    
    def scan_fn(advantage_t_plus_1, t):
        delta_t = deltas[:, t]
        not_done_t = not_done_mask[:, t]
        advantage_t = delta_t + gamma * gae_lambda * not_done_t * advantage_t_plus_1
        return advantage_t, advantage_t
    
    _, advantages = lax.scan(
        scan_fn,
        jnp.zeros(values.shape[0]),
        jnp.arange(values.shape[1] - 1, -1, -1),
    )
    
    advantages = advantages[::-1].T
    return advantages


def default_training_step(
    network: Any,
    optimizer: Any,
    params: Any,
    opt_state: Any,
    batch: Dict[str, jnp.ndarray],
    hparams: Dict[str, Any],
) -> Tuple[Any, Any]:
    """Default training step for policy gradient with hyperparameters"""
    
    # Extract batch data
    observations = batch['observations']
    actions = batch['actions']
    rewards = batch['rewards']
    dones = batch['dones']
    old_values = batch['values']
    final_values = batch['final_values']
    initial_rnn_states = batch.get('initial_rnn_states', None)
    
    # Calculate advantages using GAE
    advantages = default_calculate_gae(rewards, old_values, dones, next_values=final_values)
    returns = advantages + old_values
    
    batch_size, trajectory_length = observations.shape[:2]
    
    def loss_fn(params):
        # Process trajectories sequentially with RNN if RNN states are provided
        if initial_rnn_states is not None:
            # Process each environment's trajectory with RNN
            def process_env_trajectory(env_idx):
                # Get initial RNN state for this environment with stop_gradient
                env_rnn_state = jax.tree.map(lambda x: jax.lax.stop_gradient(x[env_idx]), initial_rnn_states)
                
                # Process trajectory sequentially
                def step_fn(carry, t):
                    rnn_state = carry
                    obs = observations[env_idx, t]
                    prev_done = jnp.where(t > 0, dones[env_idx, t-1], False)
                    
                    # Add batch dimension for network call
                    obs_batched = obs[None, ...]
                    prev_done_batched = prev_done[None, ...]
                    rnn_state_batched = jax.tree.map(lambda x: x[None, ...], rnn_state) if rnn_state is not None else None
                    
                    # Get outputs from network (with batch dimension)
                    logits_batched, value_batched, new_rnn_state_batched = network.apply(
                        params, obs_batched, prev_done_batched, rnn_state_batched
                    )
                    
                    # Remove batch dimension
                    logits = logits_batched[0]
                    value = value_batched[0]
                    new_rnn_state = jax.tree.map(lambda x: x[0], new_rnn_state_batched) if new_rnn_state_batched is not None else None
                    
                    return new_rnn_state, (logits, value)
                
                _, (env_logits, env_values) = lax.scan(
                    step_fn, env_rnn_state, jnp.arange(trajectory_length)
                )
                return env_logits, env_values
            
            # Process all environments
            all_logits, all_values = vmap(process_env_trajectory)(jnp.arange(batch_size))
        else:
            # Non-RNN network: process all timesteps at once
            obs_reshaped = observations.reshape(batch_size * trajectory_length, *observations.shape[2:])
            dones_reshaped = dones.reshape(batch_size * trajectory_length)
            
            # Call network with batched observations
            logits_flat, values_flat, _ = network.apply(params, obs_reshaped, dones_reshaped)
            
            # Reshape back to (batch_size, trajectory_length, ...)
            all_logits = logits_flat.reshape(batch_size, trajectory_length, -1)
            all_values = values_flat.reshape(batch_size, trajectory_length)
        
        # Calculate log probabilities
        log_probs_all = jax.nn.log_softmax(all_logits, axis=-1)
        action_indices = jnp.arange(batch_size)[:, None], jnp.arange(trajectory_length), actions
        log_probs = log_probs_all[action_indices]
        
        # Policy gradient loss
        advantages_detached = jax.lax.stop_gradient(advantages)
        policy_loss = -jnp.mean(advantages_detached * log_probs)
        
        # Value loss
        returns_detached = jax.lax.stop_gradient(returns)
        value_loss = 0.5 * jnp.mean((all_values - returns_detached) ** 2)
        
        # Entropy bonus
        probs = jax.nn.softmax(all_logits, axis=-1)
        entropy = -jnp.mean(jnp.sum(probs * log_probs_all, axis=-1))
        
        # Total loss with hyperparameters
        entropy_coef = hparams['entropy_coef']
        value_loss_coef = hparams['value_loss_coef']
        total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
        return total_loss
    
    # Calculate gradients and update
    loss_and_grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(params)
    
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state


def default_complete(params: Any, opt_state: Any, trial_data: Dict[str, Any]) -> None:
    """
    Default completion function called when a trial training finishes.
    This is a no-op by default, but agents can override to perform custom post-training actions.

    Args:
        params: Final trained network parameters
        opt_state: Final optimizer state
        trial_data: Dictionary containing:
            - solve_rate: Final solve rate achieved
            - hyperparameters: Hyperparameters used for training
            - seed: Random seed used
            - trial_number: Trial number (0-3)

    Returns:
        None
    """
    # Default implementation: no-op
    pass