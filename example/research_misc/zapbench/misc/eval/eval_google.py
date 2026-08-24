#!/usr/bin/env python3
"""
JAX/Flax implementation functionally equivalent to the TensorFlow/Keras submission.
This recreates the exact same architecture and training procedure.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random
import optax
import numpy as np


class FeatureExtractor(nn.Module):
    """
    JAX/Flax version of the TensorFlow FeatureExtractor.
    Functionally equivalent feature engineering for ZapBench model.
    """
    num_timesteps_context: int
    num_neurons: int

    def setup(self):
        # Constants matching TensorFlow version exactly
        self.GLOBAL_STATE_DIM = 96  # Increased from 64 (matching raw_google.py)
        self.EMBEDDING_DIM = 96  # Increased from 64 (matching raw_google.py)

        # Index for the most recent past activity value (t-0)
        self.last_value_index = self.num_timesteps_context - 1
        # Index for the first value in context
        self.first_value_index = 0

        # Learned Global Brain State Encoder (matching TF TimeDistributed + Dense)
        self.global_state_encoder = nn.Dense(
            features=self.GLOBAL_STATE_DIM,
            name='global_state_dense'
        )

        # Neuron ID Embedding Layer (matching TF Embedding)
        self.neuron_embedding_layer = nn.Embed(
            num_embeddings=self.num_neurons,
            features=self.EMBEDDING_DIM,
            name='neuron_embedding'
        )

        # 1D CNN layer (matching TF Conv1D exactly)
        # TF Conv1D with kernel_size=4 on input length 4 produces output length 1
        self.conv1d_context_processor = nn.Conv(
            features=64,  # 64 filters
            kernel_size=(self.num_timesteps_context,),  # kernel_size = context length
            padding='VALID',  # No padding to ensure kernel_size=input_size gives output_size=1
            name='conv1d_context_processor'
        )

    def __call__(self, series_input):
        """
        Forward pass matching TensorFlow version exactly.
        Args:
          series_input: Shape (batch_size, C, NUM_NEURONS)
        Returns:
          Combined features tensor: (B*N, total_feature_dim)
        """
        batch_size = series_input.shape[0]
        num_neurons = series_input.shape[2]

        # Transpose to (batch_size, NUM_NEURONS, C) for per-neuron processing
        raw_neuron_data = jnp.transpose(series_input, (0, 2, 1))  # (B, N, C)

        # Reshape for Conv1D: (batch_size * num_neurons, timesteps_context, 1)
        raw_neuron_context_for_temporal_layer = jnp.expand_dims(
            raw_neuron_data.reshape(-1, self.num_timesteps_context), axis=-1
        )  # (B*N, C, 1)

        # Process with 1D CNN (matching TF version)
        temporal_conv_features = self.conv1d_context_processor(raw_neuron_context_for_temporal_layer)
        temporal_conv_features = nn.relu(temporal_conv_features)  # Apply relu activation

        # The Conv output should be (B*N, 1, 64) with kernel_size=4 and input length=4
        # Squeeze the time dimension (axis=1) to get (B*N, 64)
        processed_temporal_features = jnp.squeeze(temporal_conv_features, axis=1)  # (B*N, 64)

        # Calculate statistical features (matching TF version exactly)
        neuron_means = jnp.mean(raw_neuron_data, axis=2, keepdims=True)  # (B, N, 1)
        neuron_stds = jnp.std(raw_neuron_data, axis=2, keepdims=True)   # (B, N, 1)
        neuron_mins = jnp.min(raw_neuron_data, axis=2, keepdims=True)   # (B, N, 1)
        neuron_maxs = jnp.max(raw_neuron_data, axis=2, keepdims=True)   # (B, N, 1)
        neuron_last_value = raw_neuron_data[:, :, self.last_value_index:self.last_value_index+1]  # (B, N, 1)
        neuron_first_value = raw_neuron_data[:, :, self.first_value_index:self.first_value_index+1]  # (B, N, 1)

        # Median calculation (matching TF version for C=4)
        sorted_neuron_data = jnp.sort(raw_neuron_data, axis=2)
        neuron_medians = (sorted_neuron_data[:, :, self.num_timesteps_context // 2 - 1] +
                         sorted_neuron_data[:, :, self.num_timesteps_context // 2]) / 2.0
        neuron_medians = jnp.expand_dims(neuron_medians, axis=-1)  # (B, N, 1)

        neuron_ranges = neuron_maxs - neuron_mins  # (B, N, 1)

        # Delta features (matching TF version)
        neuron_deltas = raw_neuron_data[:, :, 1:] - raw_neuron_data[:, :, :-1]  # (B, N, C-1)

        # Flatten statistical features to (B*N, 1) format
        neuron_means_flat = neuron_means.reshape(-1, 1)
        neuron_stds_flat = neuron_stds.reshape(-1, 1)
        neuron_mins_flat = neuron_mins.reshape(-1, 1)
        neuron_maxs_flat = neuron_maxs.reshape(-1, 1)
        neuron_last_value_flat = neuron_last_value.reshape(-1, 1)
        neuron_first_value_flat = neuron_first_value.reshape(-1, 1)
        neuron_medians_flat = neuron_medians.reshape(-1, 1)
        neuron_ranges_flat = neuron_ranges.reshape(-1, 1)
        neuron_deltas_flat = neuron_deltas.reshape(-1, self.num_timesteps_context - 1)

        # Global Time-Dependent Aggregates (matching TF TimeDistributed + GlobalAveragePooling1D)
        # Apply dense layer to each timestep
        global_brain_state_embeddings = jnp.stack([
            nn.relu(self.global_state_encoder(series_input[:, t, :]))  # Apply to each timestep
            for t in range(self.num_timesteps_context)
        ], axis=1)  # (B, C, GLOBAL_STATE_DIM)

        # Global average pooling across timesteps (matching TF GlobalAveragePooling1D)
        global_context_aggregated = jnp.mean(global_brain_state_embeddings, axis=1)  # (B, GLOBAL_STATE_DIM)

        # Tile for each neuron (matching TF version)
        global_context_final = jnp.tile(
            global_context_aggregated[:, None, :], (1, num_neurons, 1)
        ).reshape(-1, self.GLOBAL_STATE_DIM)  # (B*N, GLOBAL_STATE_DIM)

        # Generate neuron IDs and embed (matching TF version)
        neuron_ids_base = jnp.arange(self.num_neurons, dtype=jnp.int32)  # (N,)
        neuron_ids_final = jnp.tile(neuron_ids_base, batch_size)  # (B*N,)
        neuron_embedding = self.neuron_embedding_layer(neuron_ids_final)  # (B*N, EMBEDDING_DIM)

        # Combine all features (matching exact order from TF version)
        combined_features_for_mlp = jnp.concatenate([
            processed_temporal_features,  # (B*N, 64) - 1D CNN temporal features
            neuron_means_flat,           # (B*N, 1)
            neuron_stds_flat,            # (B*N, 1)
            neuron_mins_flat,            # (B*N, 1)
            neuron_maxs_flat,            # (B*N, 1)
            neuron_last_value_flat,      # (B*N, 1)
            neuron_first_value_flat,     # (B*N, 1)
            neuron_medians_flat,         # (B*N, 1)
            neuron_ranges_flat,          # (B*N, 1)
            neuron_deltas_flat,          # (B*N, C-1)
            global_context_final,        # (B*N, GLOBAL_STATE_DIM)
            neuron_embedding             # (B*N, EMBEDDING_DIM)
        ], axis=1)

        return combined_features_for_mlp


class ResidualBlock(nn.Module):
    """
    JAX/Flax version of TensorFlow ResidualBlock.
    Dense + BatchNormalization + Dropout + skip connection.
    """
    units: int
    dropout_rate: float

    def setup(self):
        self.dense = nn.Dense(features=self.units, name='dense')
        # CRITICAL: Use epsilon=1e-3 to match Keras default (Flax default is 1e-5)
        self.bn = nn.BatchNorm(epsilon=1e-3, name='batch_norm')
        self.dropout = nn.Dropout(rate=self.dropout_rate, name='dropout')

    def __call__(self, inputs, training=False):
        """Forward pass matching TF version exactly."""
        x = self.dense(inputs)
        x = self.bn(x, use_running_average=not training)
        x = self.dropout(x, deterministic=not training)

        # Add residual connection + activation (matching TF version)
        output = nn.relu(x + inputs)
        return output


class GoogleEquivalentModel(nn.Module):
    """
    JAX/Flax version of the TensorFlow Model class.
    Functionally equivalent enhanced MLP for multivariate time series forecasting.
    """
    num_timesteps_context: int
    num_neurons: int
    prediction_window_length: int

    def setup(self):
        # Store constants (matching TF version exactly)
        self.RESIDUAL_BLOCK_UNITS = 256

        # Feature extractor (matching TF version)
        self.feature_extractor = FeatureExtractor(
            num_timesteps_context=self.num_timesteps_context,
            num_neurons=self.num_neurons,
            name='feature_extractor_layer'
        )

        # Initial batch normalization (matching TF version)
        # CRITICAL: Use epsilon=1e-3 to match Keras default (Flax default is 1e-5)
        self.initial_batchnorm = nn.BatchNorm(epsilon=1e-3, name='initial_bn')

        # Initial projection layer (matching TF version)
        self.initial_projection_dense = nn.Dense(
            features=self.RESIDUAL_BLOCK_UNITS,
            name='initial_projection_dense'
        )

        # Residual Blocks (matching TF version exactly)
        self.res_block1 = ResidualBlock(
            units=self.RESIDUAL_BLOCK_UNITS,
            dropout_rate=0.3,
            name='res_block1'
        )
        self.res_block2 = ResidualBlock(
            units=self.RESIDUAL_BLOCK_UNITS,
            dropout_rate=0.3,
            name='res_block2'
        )
        self.res_block3 = ResidualBlock(
            units=self.RESIDUAL_BLOCK_UNITS,
            dropout_rate=0.3,
            name='res_block3'
        )
        self.res_block4 = ResidualBlock(
            units=self.RESIDUAL_BLOCK_UNITS,
            dropout_rate=0.3,
            name='res_block4'
        )

        # Final output layer (matching TF softplus activation)
        self.final_output_dense = nn.Dense(
            features=self.prediction_window_length,
            name='final_output_dense'
        )

    def __call__(self, series_input, training=False):
        """
        Forward pass matching TF version exactly.
        Args:
          series_input: Shape (batch_size, C, NUM_NEURONS)
          training: Boolean for BatchNorm and Dropout
        Returns:
          Predictions: Shape (batch_size * NUM_NEURONS, H)
        """
        # Use feature extractor to generate combined features
        combined_features_for_mlp = self.feature_extractor(series_input)

        # Apply initial batch normalization (matching TF version)
        x = self.initial_batchnorm(combined_features_for_mlp, use_running_average=not training)

        # Project features and apply relu (matching TF version)
        x = nn.relu(self.initial_projection_dense(x))

        # Pass through residual blocks (matching TF version)
        x = self.res_block1(x, training=training)
        x = self.res_block2(x, training=training)
        x = self.res_block3(x, training=training)
        x = self.res_block4(x, training=training)

        # Final prediction layer with softplus (matching TF version)
        output = self.final_output_dense(x)
        output = nn.softplus(output)  # TF version uses softplus activation

        return output  # Shape: (B*N, H)


# ModelInterface for station compatibility
class ModelInterface:
    def __init__(self, model_class, hparams):
        self.model = model_class(
            num_timesteps_context=4,  # INPUT_HORIZON
            num_neurons=hparams['num_neurons'],
            prediction_window_length=32  # OUTPUT_HORIZON
        )
        # Explicitly declare mutable collections and RNG requirements for train_single.py
        self.mutable = ['batch_stats']
        self.needs_rng = True  # Dropout enabled

    def init(self, rng_key, dummy_input):
        rng_params, rng_dropout = jax.random.split(rng_key)
        return self.model.init({'params': rng_params, 'dropout': rng_dropout}, dummy_input, training=True)

    def apply(self, params, x, training=False, mutable=None, rngs=None):
        # Get model output: (B*N, H)
        # CRITICAL: Pass training flag for BatchNorm and Dropout
        if mutable is not None:
            # Training mode with mutable batch_stats and optional RNG
            if rngs is not None:
                output, updates = self.model.apply(
                    params, x, training=training, mutable=mutable, rngs=rngs
                )
            else:
                output, updates = self.model.apply(params, x, training=training, mutable=mutable)
        else:
            # Evaluation mode
            if rngs is not None:
                output = self.model.apply(params, x, training=training, rngs=rngs)
            else:
                output = self.model.apply(params, x, training=training)
            updates = None

        # Reshape to match station's expected format: (B, H, N)
        batch_size = x.shape[0]
        num_neurons = x.shape[2]
        prediction_length = output.shape[1]

        # Reshape from (B*N, H) to (B, N, H) then transpose to (B, H, N)
        output = output.reshape(batch_size, num_neurons, prediction_length)
        output = jnp.transpose(output, (0, 2, 1))  # (B, H, N)

        if mutable is not None:
            return output, updates
        else:
            return output


def cosine_annealing_warm_restarts_schedule(initial_lr, first_decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.0, steps_per_epoch=1):
    """
    JAX/Optax version of TensorFlow's CosineAnnealingWarmRestarts.
    Properly implements warm restarts with increasing cycle lengths.
    Uses JAX-compatible operations for JIT compilation.
    """
    def schedule(step):
        # Convert step to epoch (use integer division)
        epoch = step // steps_per_epoch

        # Handle epoch 0 case
        def epoch_zero_lr():
            return initial_lr

        def compute_lr():
            # Manually unroll cycles for epochs 0-120 (covers typical training)
            # Cycle 1: epochs 0-9 (10 epochs)
            # Cycle 2: epochs 10-29 (20 epochs)
            # Cycle 3: epochs 30-69 (40 epochs)
            # Cycle 4: epochs 70-149 (80 epochs)

            # Determine which cycle and progress within it
            in_cycle_1 = epoch < 10
            in_cycle_2 = (epoch >= 10) & (epoch < 30)
            in_cycle_3 = (epoch >= 30) & (epoch < 70)
            in_cycle_4 = (epoch >= 70) & (epoch < 150)

            # Cycle 1: epochs 0-9
            progress_c1 = epoch / first_decay_steps
            lr_c1 = initial_lr

            # Cycle 2: epochs 10-29
            progress_c2 = (epoch - 10) / (first_decay_steps * t_mul)
            lr_c2 = initial_lr * m_mul

            # Cycle 3: epochs 30-69
            progress_c3 = (epoch - 30) / (first_decay_steps * t_mul * t_mul)
            lr_c3 = initial_lr * m_mul * m_mul

            # Cycle 4: epochs 70-149
            progress_c4 = (epoch - 70) / (first_decay_steps * t_mul * t_mul * t_mul)
            lr_c4 = initial_lr * m_mul * m_mul * m_mul

            # Beyond cycle 4, use cycle 4 parameters
            progress_c5 = (epoch - 70) / (first_decay_steps * t_mul * t_mul * t_mul)
            lr_c5 = initial_lr * m_mul * m_mul * m_mul

            # Select the appropriate cycle
            progress = jnp.where(in_cycle_1, progress_c1,
                       jnp.where(in_cycle_2, progress_c2,
                       jnp.where(in_cycle_3, progress_c3,
                       jnp.where(in_cycle_4, progress_c4, progress_c5))))

            cycle_initial_lr = jnp.where(in_cycle_1, lr_c1,
                               jnp.where(in_cycle_2, lr_c2,
                               jnp.where(in_cycle_3, lr_c3,
                               jnp.where(in_cycle_4, lr_c4, lr_c5))))

            # Cosine annealing formula
            cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
            current_lr = (cycle_initial_lr - cycle_initial_lr * alpha) * cosine_decay + cycle_initial_lr * alpha

            return current_lr

        return jnp.where(epoch == 0, epoch_zero_lr(), compute_lr())

    return schedule


def _define_hyperparameters():
    """Hyperparameters matching TensorFlow version exactly."""
    return {
        'learning_rate': 1e-3,  # initial_lr_for_schedule from TF version
    }


def create_network(hparams):
    """Create network matching TF Model exactly."""
    hp = hparams.copy()
    hp['num_neurons'] = 71721  # ZAPBench dataset size
    return ModelInterface(GoogleEquivalentModel, hparams=hp)


def compute_loss(predictions, targets, params, x):
    """MAE loss matching TF version exactly."""
    # TF version compiles with loss='mae' which is mean absolute error
    return jnp.mean(jnp.abs(predictions - targets))


def create_optimizer(learning_rate=1e-3):
    """
    Create optimizer matching TF version with cosine annealing warm restarts.

    Uses the same schedule as raw_google.py:
    - First cycle: 10 epochs
    - Cycle length multiplier: 2.0 (doubles each restart)
    - Max LR multiplier: 1.0 (keeps max LR same)
    """
    # Calculate steps per epoch based on approximate data size
    # ZAPBench has ~5000 training samples
    steps_per_epoch = 5052 // BATCH_SIZE  # ~158 steps per epoch with batch_size=32

    # Create cosine annealing schedule matching raw_google.py
    schedule = cosine_annealing_warm_restarts_schedule(
        initial_lr=learning_rate,
        first_decay_steps=10,  # 10 epochs for first cycle
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0,
        steps_per_epoch=steps_per_epoch
    )

    # Adam optimizer with scheduled learning rate
    # CRITICAL: Use eps=1e-7 to match Keras default (optax default is 1e-8)
    return optax.adam(learning_rate=schedule, eps=1e-7)


# Batch size for training (B*N expansion in reshape causes high memory usage)
BATCH_SIZE = 8  # Standard batch size for better BatchNorm statistics
