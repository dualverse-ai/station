"""
Default baseline submission for ZAPBench neural activity forecasting.
Uses all system defaults - no custom implementations.
"""

def _define_hyperparameters():
    """Use exact same hyperparameters as default."""
    return {'learning_rate': 0.001}

# Uses system defaults for:
# - create_network: Per-neuron MLP (4->128->32)
# - compute_loss: MAE loss
# - create_optimizer: Adam optimizer
# - complete: No-op