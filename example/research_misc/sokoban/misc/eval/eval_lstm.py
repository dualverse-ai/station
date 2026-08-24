"""
RNN baseline submission for distributed RL on Sokoban.
Custom CNN-LSTM architecture with hyperparameter search.
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
import flax.linen as nn
import optax
from ray import tune
import time
from typing import Tuple, Dict, Any, Optional


# Uses system defaults for all hyperparameters

class CNNLSTM(nn.Module):
    """CNN baseline + LSTM for Sokoban."""
    lstm_features: int
    cnn_features_1: int
    cnn_features_2: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, done: jnp.ndarray, rnn_state: Optional[Dict] = None) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        batch_size = x.shape[0]
        
        # Initialize RNN state if not provided
        if rnn_state is None:
            rnn_state = {
                'h': jnp.zeros((batch_size, self.lstm_features)),
                'c': jnp.zeros((batch_size, self.lstm_features)),
            }
        
        # Reset states for episodes that are done
        if done is not None:
            done_mask = done.reshape(batch_size, 1)
            rnn_state = {
                'h': rnn_state['h'] * (1 - done_mask),
                'c': rnn_state['c'] * (1 - done_mask),
            }
        
        # FULL CNN baseline architecture (same as DefaultResidualCNN)
        # First convolutional block
        x = nn.Conv(features=self.cnn_features_1, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        
        # Second block with residual connection
        conv2_in = x
        x = nn.Conv(features=self.cnn_features_1, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.cnn_features_1, kernel_size=(3, 3), padding='SAME')(x)
        x = x + conv2_in
        x = nn.relu(x)
        
        # Third block - increase channels
        x = nn.Conv(features=self.cnn_features_2, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        
        # Fourth block with residual connection
        conv4_in = x
        x = nn.Conv(features=self.cnn_features_2, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.cnn_features_2, kernel_size=(3, 3), padding='SAME')(x)
        x = x + conv4_in
        x = nn.relu(x)
        
        # Flatten spatial dimensions
        x = x.reshape((batch_size, -1))
        
        # Shared MLP backbone
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        
        # LSTM layer on top of CNN features
        lstm_cell = nn.LSTMCell(
            features=self.lstm_features,
            kernel_init=nn.initializers.xavier_uniform(),
            recurrent_kernel_init=nn.initializers.xavier_uniform()
        )
        (rnn_state['c'], rnn_state['h']), lstm_out = lstm_cell((rnn_state['c'], rnn_state['h']), x)
        
        # Small MLP after LSTM
        x = nn.Dense(features=128)(lstm_out)
        x = nn.relu(x)
        
        # Policy and value heads
        policy_logits = nn.Dense(features=4)(x)
        value = nn.Dense(features=1)(x).squeeze(-1)
        
        return policy_logits, value, rnn_state

def create_network(hparams):
    """Create CNN-LSTM network architecture."""
    return CNNLSTM(
        lstm_features=hparams['lstm_features'],
        cnn_features_1=hparams['cnn_features_1'],
        cnn_features_2=hparams['cnn_features_2']
    )
