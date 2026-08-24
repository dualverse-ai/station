"""
DRC(3,3) architecture following "An Investigation of Model-Free Planning" (Guez et al., ICML 2019).

Architecture: 3 stacked ConvLSTM layers (D=3) with 3 internal ticks per timestep (N=3).
Features:
- Top-down skip connections: h3 from previous tick feeds back to layer 1
- Encoded observation skip connections: input fed to all layers
- Pool-and-inject: spatial pooling (max+mean) aggregated and injected to next tick
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
import flax.linen as nn
import optax
from typing import Tuple, Dict, Any, Optional


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell."""
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    
    @nn.compact
    def __call__(self, carry, inputs):
        h, c = carry
        # Concatenate hidden state and input
        combined = jnp.concatenate([inputs, h], axis=-1)
        
        # Gates
        gates = nn.Conv(
            features=4 * self.features,
            kernel_size=self.kernel_size,
            padding='SAME'
        )(combined)
        
        # Split into input, forget, cell, and output gates
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        
        # Apply activations
        i = nn.sigmoid(i)
        f = nn.sigmoid(f)
        g = nn.tanh(g)
        o = nn.sigmoid(o)
        
        # Update cell state
        c = f * c + i * g
        # Update hidden state
        h = o * nn.tanh(c)
        
        return (h, c), h


class SokobanConvLSTM(nn.Module):
    """DRC(3,3): 3 stacked ConvLSTM layers with 3 internal ticks per timestep.

    Implements the architecture from Guez et al. ICML 2019 with pool-and-inject.
    """
    internal_ticks: int = 3  # N=3 from DRC(3,3) paper
    
    def setup(self):
        # CNN layers - all 32 channels (stride=1 keeps 8x8 spatial)
        self.conv = nn.Conv(features=32, kernel_size=(3, 3), strides=1, padding='SAME')

        # Three ConvLSTM cells - 32 channels each (D=3)
        self.convlstm1 = ConvLSTMCell(features=32)
        self.convlstm2 = ConvLSTMCell(features=32)
        self.convlstm3 = ConvLSTMCell(features=32)

        # Pool-and-inject projection layers (one per ConvLSTM layer)
        self.pool_proj1 = nn.Dense(features=32)
        self.pool_proj2 = nn.Dense(features=32)
        self.pool_proj3 = nn.Dense(features=32)

        # MLP layer
        self.fc = nn.Dense(features=256)

        # Policy and value heads
        self.policy_head = nn.Dense(features=4)
        self.value_head = nn.Dense(features=1)
    
    def initialize_state(self, batch_size: int):
        """Initialize ConvLSTM hidden states (all 8x8x32)."""
        return {
            'h1': jnp.zeros((batch_size, 8, 8, 32)),
            'c1': jnp.zeros((batch_size, 8, 8, 32)),
            'h2': jnp.zeros((batch_size, 8, 8, 32)),
            'c2': jnp.zeros((batch_size, 8, 8, 32)),
            'h3': jnp.zeros((batch_size, 8, 8, 32)),
            'c3': jnp.zeros((batch_size, 8, 8, 32)),
        }

    def pool_and_inject(self, h: jnp.ndarray, pool_proj) -> jnp.ndarray:
        """Pool-and-inject mechanism from the paper.

        Applies max and mean pooling spatially, projects through linear layer,
        then tiles back to 8x8 spatial dimensions.

        Args:
            h: Hidden state (batch, 8, 8, 32)
            pool_proj: Dense layer for projection

        Returns:
            Tiled pooled tensor (batch, 8, 8, 32)
        """
        # Pool spatially (both max and mean as in paper)
        max_pooled = jnp.max(h, axis=(1, 2))  # (batch, 32)
        mean_pooled = jnp.mean(h, axis=(1, 2))  # (batch, 32)
        pooled = jnp.concatenate([max_pooled, mean_pooled], axis=-1)  # (batch, 64)

        # Project through linear layer
        projected = pool_proj(pooled)  # (batch, 32)

        # Tile back to 8x8 spatial dimensions
        tiled = jnp.tile(projected[:, None, None, :], (1, 8, 8, 1))  # (batch, 8, 8, 32)

        return tiled
    
    def __call__(self, x: jnp.ndarray, done: jnp.ndarray, rnn_state: Optional[Dict] = None) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        # Input shape: (batch, 8, 8, 8) - ALWAYS batched
        batch_size = x.shape[0]
        
        # Initialize or reset RNN states
        if rnn_state is None:
            rnn_state = self.initialize_state(batch_size)
        
        # Reset states where episodes are done
        if done is not None:
            done_mask = done.reshape(batch_size, 1, 1, 1)
            for key in rnn_state:
                rnn_state[key] = rnn_state[key] * (1 - done_mask)
        
        # CNN layers
        x = self.conv(x)
        
        # Store CNN encoded observation for ConvLSTM stack
        convlstm_input = x
        
        # Run N=3 internal ticks with the same encoded observation
        current_rnn_state = rnn_state

        for tick in range(self.internal_ticks):
            # Pool-and-inject from previous tick's hidden states
            # At tick 0: use current_rnn_state (from previous timestep)
            # At tick>0: use outputs from previous tick
            if tick == 0:
                # Use hidden states from previous timestep
                pooled1 = self.pool_and_inject(current_rnn_state['h1'], self.pool_proj1)
                pooled2 = self.pool_and_inject(current_rnn_state['h2'], self.pool_proj2)
                pooled3 = self.pool_and_inject(current_rnn_state['h3'], self.pool_proj3)
            else:
                # Use outputs from previous tick
                pooled1 = self.pool_and_inject(prev_h1, self.pool_proj1)
                pooled2 = self.pool_and_inject(prev_h2, self.pool_proj2)
                pooled3 = self.pool_and_inject(prev_h3, self.pool_proj3)

            # Layer 1: encoded obs + top-down skip (h3 from prev tick) + pooled
            layer1_input = jnp.concatenate([convlstm_input, current_rnn_state['h3'], pooled1], axis=-1)
            (current_rnn_state['h1'], current_rnn_state['c1']), h1_output = self.convlstm1(
                (current_rnn_state['h1'], current_rnn_state['c1']), layer1_input
            )

            # Layer 2: encoded obs + h1 from current tick + pooled
            layer2_input = jnp.concatenate([convlstm_input, h1_output, pooled2], axis=-1)
            (current_rnn_state['h2'], current_rnn_state['c2']), h2_output = self.convlstm2(
                (current_rnn_state['h2'], current_rnn_state['c2']), layer2_input
            )

            # Layer 3: encoded obs + h2 from current tick + pooled
            layer3_input = jnp.concatenate([convlstm_input, h2_output, pooled3], axis=-1)
            (current_rnn_state['h3'], current_rnn_state['c3']), h3_output = self.convlstm3(
                (current_rnn_state['h3'], current_rnn_state['c3']), layer3_input
            )

            # Save outputs for next tick's pool-and-inject
            prev_h1, prev_h2, prev_h3 = h1_output, h2_output, h3_output

        # Update the rnn_state with final tick's state
        rnn_state = current_rnn_state
        
        # Use the final tick's output
        x = h3_output

        # Flatten all spatial dimensions
        x = x.reshape((batch_size, -1))
        
        # MLP
        x = self.fc(x)
        x = nn.relu(x)
        
        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)  # Remove last dimension to get (batch_size,)
        
        return policy_logits, value, rnn_state


def create_network(hparams: Dict[str, Any]):
    """Create and return the agent's neural network architecture.

    Args:
        hparams: Hyperparameters dictionary (not used in this architecture)
    """
    return SokobanConvLSTM()


def _define_hyperparameters():
    """Define hyperparameters for DRC(3,3) following Guez et al. ICML 2019.

    Paper hyperparameters (Appendix training_setup):
    - Learning rate: 4e-4 (annealed to 0 over 1.5e9 steps, but we use fixed for simplicity)
    - Adam: β1=0.9, β2=0.999, ε=1e-4
    - Entropy coef: 0.01
    - Baseline (value) loss weight: 0.5
    - L2 norm on logits: 1e-3
    - L2 regularization on final layers (policy/value heads): 1e-5
    - λ-returns: λ=0.97
    - Discount: γ=0.97
    - BPTT unroll: 20
    - Batch size: 32
    """
    from ray import tune

    # Fixed hyperparameters from the paper
    return {
        'learning_rate': tune.choice([4e-4]),              # Paper: 4e-4
        'entropy_coef': tune.choice([0.01]),               # Paper: 0.01
        'value_loss_coef': tune.choice([0.5]),             # Paper: 0.5 (baseline loss weight)
        'l2_logits_coef': tune.choice([1e-3]),             # Paper: 1e-3 (L2 norm on logits)
        'l2_final_layers_coef': tune.choice([1e-5]),       # Paper: 1e-5 (L2 on final layers)
    }


def create_optimizer(learning_rate: float = 4e-4):
    """Create optimizer with linear LR decay to 0.

    Paper: Adam with β1=0.9, β2=0.999, ε=1e-4, lr=4e-4 (annealed to 0)

    Args:
        learning_rate: Initial learning rate (default 4e-4 from paper)
    """
    # Hardcoded training parameters for LR schedule
    total_steps = 50_000_000
    batch_size = 32  # Paper uses 32
    unroll_length = 20

    # Calculate total number of optimizer updates
    num_updates = total_steps // (batch_size * unroll_length)

    # Linear decay from learning_rate to 0
    lr_schedule = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0.0,
        transition_steps=num_updates
    )

    return optax.adam(
        learning_rate=lr_schedule,
        b1=0.9,
        b2=0.999,
        eps=1e-4
    )


@jit
def calculate_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.97,
    gae_lambda: float = 0.97,
    next_values: jnp.ndarray = None,
) -> jnp.ndarray:
    """Calculate Generalized Advantage Estimation (GAE).

    Copied from defaults to avoid import dependency.
    """
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


def training_step(
    network: Any,
    optimizer: Any,
    params: Any,
    opt_state: Any,
    batch: Dict[str, jnp.ndarray],
    hparams: Dict[str, Any],
) -> Tuple[Any, Any]:
    """Custom training step with L2 regularization matching paper.

    Paper regularization:
    - L2 norm on logits: weight 1e-3
    - L2 regularization on final layers (policy/value heads): weight 1e-5
    """
    # Extract batch data
    observations = batch['observations']
    actions = batch['actions']
    rewards = batch['rewards']
    dones = batch['dones']
    old_values = batch['values']
    final_values = batch['final_values']
    initial_rnn_states = batch.get('initial_rnn_states', None)

    # Calculate advantages using GAE
    advantages = calculate_gae(rewards, old_values, dones, next_values=final_values)
    returns = advantages + old_values

    batch_size, trajectory_length = observations.shape[:2]

    # Get regularization coefficients
    l2_logits_coef = hparams.get('l2_logits_coef', 1e-3)
    l2_final_layers_coef = hparams.get('l2_final_layers_coef', 1e-5)

    def loss_fn(params):
        # Process trajectories with RNN
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

        # Value loss (MSE without the 0.5 factor - will be scaled by value_loss_coef)
        returns_detached = jax.lax.stop_gradient(returns)
        value_loss = jnp.mean((all_values - returns_detached) ** 2)

        # Entropy bonus
        probs = jax.nn.softmax(all_logits, axis=-1)
        entropy = -jnp.mean(jnp.sum(probs * log_probs_all, axis=-1))

        # Base loss with hyperparameters
        entropy_coef = hparams['entropy_coef']
        value_loss_coef = hparams['value_loss_coef']  # Paper: 0.5
        base_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

        # L2 regularization on logits (paper: weight 1e-3)
        # Use mean to normalize across batch and trajectory
        logits_l2 = jnp.mean(all_logits ** 2)

        # L2 regularization on final layers (policy_head and value_head weights)
        # Paper: weight 1e-5 on linear layers that compute baseline value and logits
        final_layers_l2 = 0.0

        # Regularize policy head weights (computes logits)
        if 'params' in params and 'policy_head' in params['params']:
            policy_kernel = params['params']['policy_head']['kernel']
            final_layers_l2 += jnp.mean(policy_kernel ** 2)

        # Regularize value head weights (computes baseline value)
        if 'params' in params and 'value_head' in params['params']:
            value_kernel = params['params']['value_head']['kernel']
            final_layers_l2 += jnp.mean(value_kernel ** 2)

        # Total loss with regularization
        total_loss = base_loss + l2_logits_coef * logits_l2 + l2_final_layers_coef * final_layers_l2

        return total_loss

    # Calculate gradients and update
    loss_and_grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(params)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state

