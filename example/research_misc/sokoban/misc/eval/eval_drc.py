"""
Student submission for RL on Sokoban assignment.
This file contains the DRC (Deep Recurrent Convolutional) network architecture.
Features 3 interconnected ConvLSTM layers with feedback connections.
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
    """CNN + ConvLSTM architecture for Sokoban with state handling."""
    internal_ticks: int = 3  # Number of internal ConvLSTM ticks per timestep
    
    def setup(self):
        # CNN layers - all 32 channels
        self.conv = nn.Conv(features=32, kernel_size=(3, 3), strides=1, padding='SAME')
        
        # Three ConvLSTM cells - 32 channels each
        self.convlstm1 = ConvLSTMCell(features=32)
        self.convlstm2 = ConvLSTMCell(features=32)
        self.convlstm3 = ConvLSTMCell(features=32)
        
        # MLP layer
        self.fc = nn.Dense(features=256)
        
        # Policy and value heads
        self.policy_head = nn.Dense(features=4)
        self.value_head = nn.Dense(features=1)
    
    def initialize_state(self, batch_size: int):
        """Initialize ConvLSTM hidden states."""
        return {
            'h1': jnp.zeros((batch_size, 8, 8, 32)),
            'c1': jnp.zeros((batch_size, 8, 8, 32)),
            'h2': jnp.zeros((batch_size, 8, 8, 32)),
            'c2': jnp.zeros((batch_size, 8, 8, 32)),
            'h3': jnp.zeros((batch_size, 8, 8, 32)),
            'c3': jnp.zeros((batch_size, 8, 8, 32)),
        }
    
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
        
        # Run ConvLSTM for multiple internal ticks with the same input
        current_rnn_state = rnn_state
        
        for tick in range(self.internal_ticks):
            # Three ConvLSTM layers with interconnected state passing
            layer1_input = jnp.concatenate([convlstm_input, current_rnn_state['h3']], axis=-1)
            (current_rnn_state['h1'], current_rnn_state['c1']), h1_output = self.convlstm1(
                (current_rnn_state['h1'], current_rnn_state['c1']), layer1_input
            )
            layer2_input = jnp.concatenate([convlstm_input, current_rnn_state['h1']], axis=-1)
            (current_rnn_state['h2'], current_rnn_state['c2']), h2_output = self.convlstm2(
                (current_rnn_state['h2'], current_rnn_state['c2']), layer2_input
            )
            
            layer3_input = jnp.concatenate([convlstm_input, current_rnn_state['h2']], axis=-1)
            (current_rnn_state['h3'], current_rnn_state['c3']), h3_output = self.convlstm3(
                (current_rnn_state['h3'], current_rnn_state['c3']), layer3_input
            )
        
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

