import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm
import pandas as pd
from zapbench import constants
from zapbench.ts_forecasting import data_source
import grain.python as grain
# Custom Learning Rate Scheduler for Cosine Annealing with Warm Restarts
class CosineAnnealingWarmRestarts(keras.callbacks.Callback):
    def __init__(self, initial_lr, first_decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.0):
        super().__init__()
        self.initial_lr = initial_lr
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        self.current_cycle_steps = 0
        self.cycle_length = float(first_decay_steps)
        self.lrs = [] # To store learning rates per epoch
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
            raise ValueError('Model optimizer is not set. Ensure model.compile() is called before fit().')
        
        # Access the learning_rate attribute, which is a tf.Variable for Adam
        # This is more robust than 'lr' which might be an alias or not always present.
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        # Calculate the learning rate for the current epoch
        if epoch == 0:
            current_lr = self.initial_lr
        else:
            # Check if we finished a cycle and need to restart
            if (epoch - self.current_cycle_steps) >= self.cycle_length:
                self.current_cycle_steps += self.cycle_length
                self.cycle_length *= self.t_mul
                self.initial_lr *= self.m_mul # Update initial_lr for the new cycle
            
            # Calculate progress within the current cycle
            progress_in_cycle = (epoch - self.current_cycle_steps) / self.cycle_length
            
            # Cosine annealing formula
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress_in_cycle))
            current_lr = (self.initial_lr - self.initial_lr * self.alpha) * cosine_decay + self.initial_lr * self.alpha
        
        # Set the optimizer's learning rate using assign for tf.Variable
        self.model.optimizer.learning_rate.assign(current_lr)
        self.lrs.append(current_lr)
        # print(f"Epoch {epoch+1}: Learning rate set to {current_lr:.6f}") # For debugging
# Custom Residual Block Layer
class ResidualBlock(keras.layers.Layer):
    """
    A simple residual block for MLP.
    Consists of Dense, BatchNormalization, Dropout, and a skip connection.
    The output dimension of the Dense layer must match the input dimension for the skip connection.
    """
    def __init__(self, units, dropout_rate, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(units, activation=None) # Activation applied after addition
        self.bn = keras.layers.BatchNormalization()
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.activation_fn = keras.activations.get(activation)
    def call(self, inputs, training=False):
        x = self.dense(inputs)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        # Add residual connection: original input + transformed input
        output = self.activation_fn(x + inputs) # Assuming inputs and x have same shape (units)
        return output
class FeatureExtractor(keras.layers.Layer):
    """
    Custom Keras Layer to encapsulate all feature engineering for the ZapBench model.
    Takes raw neural activity and generates rich features for a shared-weight MLP.
    Improvements for C=4 context:
    - Added more rolling window statistics: median, range, first value.
    - Consolidates temporal feature extraction using a 1D CNN for the short context.
    - Added explicit delta (rate of change) features.
    - Relies on 1D CNN temporal features, enhanced rolling window statistics, a learned global brain state,
      and neuron ID embeddings to form comprehensive features.
    """
    def __init__(self, num_timesteps_context: int, num_neurons: int, **kwargs):
        super().__init__(**kwargs)
        self.NUM_NEURONS = num_neurons
        self.NUM_TIMESTEPS_CONTEXT = num_timesteps_context
        # Index for the most recent past activity value (t-0)
        self.last_value_index = num_timesteps_context - 1
        # Index for the first value in context
        self.first_value_index = 0
        # Learned Global Brain State Encoder (Improved Aggregation)
        # Increased dimension for potentially richer global context
 
        self.GLOBAL_STATE_DIM = 96 # Increased from 64
        self.global_state_encoder = keras.layers.TimeDistributed(
            keras.layers.Dense(self.GLOBAL_STATE_DIM, activation='relu', name='global_state_dense'),
            name='global_state_encoder_in_extractor'
        )
        self.global_state_aggregator_pooling = keras.layers.GlobalAveragePooling1D(
            name='global_state_aggregator_pooling_in_extractor'
        )
        # Neuron ID Embedding Layer
 
        self.EMBEDDING_DIM = 96 # Increased from 64
        self.neuron_embedding_layer = keras.layers.Embedding(
            input_dim=self.NUM_NEURONS,
            output_dim=self.EMBEDDING_DIM,
            name='neuron_embedding_in_extractor'
        )
        self.flatten_embedding = keras.layers.Flatten()
        # 1D CNN layer to process the raw temporal context for each neuron
        self.conv1d_context_processor = keras.layers.Conv1D(
            filters=64, # Increased output filters for 1D convolution
            kernel_size=self.NUM_TIMESTEPS_CONTEXT, # Use kernel size equal to context length to capture full window
            activation='relu',
            name='conv1d_context_processor_in_extractor',
        )
    def call(self, series_input: tf.Tensor) -> tf.Tensor:
        """
        Performs the forward pass of the feature extractor.
        Args:
          series_input: Raw neural activity data, shape (batch_size, C, NUM_NEURONS).
        Returns:
          A single tensor combining all extracted features, shaped (B*N, total_feature_dim).
        """
        batch_size = tf.shape(series_input)[0]
        num_neurons = tf.shape(series_input)[2] # Use dynamic num_neurons from input tensor
        # Transpose to (batch_size, NUM_NEURONS, C) for easier per-neuron context processing.
        raw_neuron_data = tf.transpose(series_input, perm=[0, 2, 1]) # Shape: (B, N, C)
        # Reshape for Conv1D input: (batch_size * num_neurons, timesteps_context, 1)
        raw_neuron_context_for_temporal_layer = tf.expand_dims(
            tf.reshape(raw_neuron_data, [-1, self.NUM_TIMESTEPS_CONTEXT]), axis=-1
        ) # Shape: (B*N, C, 1)
        # Process raw context with 1D CNN
        # input shape for conv1d_context_processor will be (B*N, C, 1)
        temporal_conv_features = self.conv1d_context_processor(raw_neuron_context_for_temporal_layer) # Shape: (B*N, 1, filters)
        # Squeeze the redundant dimension, as kernel_size=C effectively performs pooling
        processed_temporal_features = tf.squeeze(temporal_conv_features, axis=1) # Shape: (B*N, filters)
        # Calculate mean, std, min, max, last_value, first_value, median, range for each neuron's C context window.
        neuron_means = tf.reduce_mean(raw_neuron_data, axis=2, keepdims=True) # Shape: (B, N, 1)
        neuron_stds = tf.math.reduce_std(raw_neuron_data, axis=2, keepdims=True) # Shape: (B, N, 1)
        neuron_mins = tf.reduce_min(raw_neuron_data, axis=2, keepdims=True) # Shape (B, N, 1)
        neuron_maxs = tf.reduce_max(raw_neuron_data, axis=2, keepdims=True) # Shape (B, N, 1)
        neuron_last_value = raw_neuron_data[:, :, self.last_value_index:self.last_value_index+1] # Shape (B, N, 1)
        neuron_first_value = raw_neuron_data[:, :, self.first_value_index:self.first_value_index+1] # Shape (B, N, 1)
        
        # For median on C=4, sort and take average of middle two
        sorted_neuron_data = tf.sort(raw_neuron_data, axis=2) # Sort along the context dimension
        # Note: For C=4, indices are 0, 1, 2, 3. Middle two are at index 1 and 2.
        # This is (C/2 - 1) and (C/2).
        neuron_medians = (sorted_neuron_data[:, :, self.NUM_TIMESTEPS_CONTEXT // 2 -1 ] +
                          sorted_neuron_data[:, :, self.NUM_TIMESTEPS_CONTEXT // 2 ]) / 2.0 # Shape (B, N)
        neuron_medians = tf.expand_dims(neuron_medians, axis=-1) # Shape (B, N, 1)
        neuron_ranges = neuron_maxs - neuron_mins # Shape (B, N, 1)
        # Calculate delta features (differences between consecutive timesteps)
        # For C=4, we'll have (t-0 - t-1), (t-1 - t-2), (t-2 - t-3)
        # raw_neuron_data shape: (B, N, C)
        # Deltas along the time dimension: (B, N, C-1)
        neuron_deltas = raw_neuron_data[:, :, 1:] - raw_neuron_data[:, :, :-1] # Shape (B, N, C-1)
        # Flatten these to match the (B*N) batching
        neuron_means_flat = tf.reshape(neuron_means, [-1, 1]) # Shape: (B*N, 1)
        neuron_stds_flat = tf.reshape(neuron_stds, [-1, 1]) # Shape: (B*N, 1)
        neuron_mins_flat = tf.reshape(neuron_mins, [-1, 1]) # Shape (B*N, 1)
        neuron_maxs_flat = tf.reshape(neuron_maxs, [-1, 1]) # Shape (B*N, 1)
        neuron_last_value_flat = tf.reshape(neuron_last_value, [-1, 1]) # Shape (B*N, 1)
        neuron_first_value_flat = tf.reshape(neuron_first_value, [-1, 1]) # Shape (B*N, 1)
        neuron_medians_flat = tf.reshape(neuron_medians, [-1, 1]) # Shape (B*N, 1)
        neuron_ranges_flat = tf.reshape(neuron_ranges, [-1, 1]) # Shape (B*N, 1)
        neuron_deltas_flat = tf.reshape(neuron_deltas, [-1, self.NUM_TIMESTEPS_CONTEXT - 1]) # Shape (B*N, C-1)
        # --- 2. Global Time-Dependent Aggregates (Learned) ---
        # Input series_input (B, C, N) -> TimeDistributed Dense -> (B, C, GLOBAL_STATE_DIM)
        global_brain_state_embeddings = self.global_state_encoder(series_input) # Shape: (B, C, GLOBAL_STATE_DIM)
        # Aggregate the learned global states across the context timesteps using GlobalAveragePooling1D
        # (B, C, GLOBAL_STATE_DIM) -> GlobalAveragePooling1D -> (B, GLOBAL_STATE_DIM)
        global_context_aggregated = self.global_state_aggregator_pooling(global_brain_state_embeddings) # Shape: (B, GLOBAL_STATE_DIM)
        # Tile this aggregated global context to match (B*N) for concatenation with other features
        global_context_final = tf.reshape(
            tf.tile(global_context_aggregated[:, tf.newaxis, :], [1, num_neurons, 1]),
            [-1, self.GLOBAL_STATE_DIM]
        ) # Shape: (B*N, GLOBAL_STATE_DIM)
        # --- 3. Generate Neuron IDs dynamically for the current batch and embed ---
        neuron_ids_base = tf.range(self.NUM_NEURONS, dtype=tf.int32) # Shape: (N,)
        # Repeat neuron IDs for each sample in the batch to match the (B*N) flattened structure
        X_neuron_ids_final = tf.tile(neuron_ids_base, [batch_size]) # Shape: (B*N,)
        X_neuron_ids_final = tf.reshape(X_neuron_ids_final, [-1, 1]) # Shape: (B*N, 1)
        neuron_embedding = self.neuron_embedding_layer(X_neuron_ids_final) # Shape: (B*N, 1, EMBEDDING_DIM)
        neuron_embedding = self.flatten_embedding(neuron_embedding) # Shape: (B*N, EMBEDDING_DIM)
        # Combine all features (1D CNN temporal features, statistics, global context, neuron embedding, DELTAS)
        combined_features_for_mlp = tf.concat([
            processed_temporal_features, # (B*N, 64) - NEW: Temporal features from 1D CNN
            neuron_means_flat,       # (B*N, 1)
            neuron_stds_flat,        # (B*N, 1)
            neuron_mins_flat,        # (B*N, 1)
            neuron_maxs_flat,        # (B*N, 1)
            neuron_last_value_flat,  # (B*N, 1)
            neuron_first_value_flat, # (B*N, 1)
            neuron_medians_flat,     # (B*N, 1)
            neuron_ranges_flat,      # (B*N, 1)
            neuron_deltas_flat,      # (B*N, C-1) - NEW: Delta features
            global_context_final,    # (B*N, GLOBAL_STATE_DIM)
            neuron_embedding         # (B*N, EMBEDDING_DIM)
        ], axis=1) # Shape: (B*N, 64 + 8 + (C-1) + GLOBAL_STATE_DIM + EMBEDDING_DIM)
        return combined_features_for_mlp
class Model(keras.Model):
  """
  An enhanced Multi-Layer Perceptron (MLP) model for multivariate time series forecasting.
  This model now uses a refined FeatureExtractor that consolidates 1D CNN temporal processing
  alongside rich statistical, delta, and global brain state features, and neuron ID embeddings.
  The extracted features are processed by a deep residual network.
  Training is enhanced with a Cosine Annealing Warm Restarts learning rate schedule.
  """
  def __init__(self, num_timesteps_context: int, num_neurons: int, prediction_window_length: int):
    super().__init__() # Initialize the base Keras Model class
    # Store constants passed during instantiation.
    self.NUM_NEURONS = num_neurons
    self.PREDICTION_WINDOW_LENGTH = prediction_window_length
    self.NUM_TIMESTEPS_CONTEXT = num_timesteps_context
    # Encapsulate all feature engineering into a dedicated layer
    self.feature_extractor = FeatureExtractor(
        num_timesteps_context=num_timesteps_context,
        num_neurons=num_neurons,
        name='feature_extractor_layer'
    )
    self.RESIDUAL_BLOCK_UNITS = 256 # Consistent hidden layer size for residual blocks
    # Initial batch normalization and projection layer
    self.initial_batchnorm = keras.layers.BatchNormalization(name='initial_bn')
    # Initial projection layer to match the dimension of subsequent residual blocks
    # This layer will now take the concatenated features directly from the FeatureExtractor.
    self.initial_projection_dense = keras.layers.Dense(
        self.RESIDUAL_BLOCK_UNITS, activation='relu', name='initial_projection_dense'
    )
    # Residual Blocks - Adjusted dropout rate
    self.res_block1 = ResidualBlock(self.RESIDUAL_BLOCK_UNITS, dropout_rate=0.3, name='res_block1')
    self.res_block2 = ResidualBlock(self.RESIDUAL_BLOCK_UNITS, dropout_rate=0.3, name='res_block2')
    self.res_block3 = ResidualBlock(self.RESIDUAL_BLOCK_UNITS, dropout_rate=0.3, name='res_block3')
    self.res_block4 = ResidualBlock(self.RESIDUAL_BLOCK_UNITS, dropout_rate=0.3, name='res_block4')
    # Final output layer
    self.final_output_dense = keras.layers.Dense(
        self.PREDICTION_WINDOW_LENGTH,
        activation='softplus', # Retained softplus activation
        name='final_output_dense'
    )
  def call(self, series_input: tf.Tensor, training: bool = False) -> tf.Tensor:
    """
    Performs the forward pass of the model, now using the refined FeatureExtractor
    to provide all combined features.
    Args:
      series_input: Raw neural activity data, shape (batch_size, C, NUM_NEURONS).
      training: Boolean indicating whether the model is in training mode. Used by BatchNormalization and Dropout.
    Returns:
      Predicted neural activity, shape (batch_size * NUM_NEURONS, H).
    """
    # Use the encapsulated feature extractor layer to generate all combined features
    combined_features_for_mlp = self.feature_extractor(series_input)
    # Apply initial batch normalization
    x = self.initial_batchnorm(combined_features_for_mlp, training=training)
    # Project features to the dimension of residual blocks
    x = self.initial_projection_dense(x)
    # Pass through residual blocks
    x = self.res_block1(x, training=training)
    x = self.res_block2(x, training=training)
    x = self.res_block3(x, training=training)
    x = self.res_block4(x, training=training)
    # Final prediction layer
    output = self.final_output_dense(x) # Shape: (B*N, H)
    return output
  def _raw_data_generator(self, data_loader_instance):
    """
    A Python generator to yield raw (series_input, series_output) pairs
    from a grain.DataLoader. These are NumPy arrays as provided by grain.
    """
    for element in data_loader_instance:
      yield element['series_input'], element['series_output']
  @tf.function # Decorate for graph mode execution
  def _prepare_targets_for_training(self, series_input_batch: tf.Tensor, series_output_batch: tf.Tensor):
    """
    Prepares the target tensor (y_true) for training by reshaping it
    to match the model's output shape (batch_size * NUM_NEURONS, H).
    The series_input_batch is returned as is, for the model's `call` method.
    """
    # Reshape the output batch (targets) from (B, H, N) to (B, N, H)
    y_target = tf.transpose(series_output_batch, perm=[0, 2, 1]) # Shape: (B, N, H)
    y_target = tf.reshape(y_target, [-1, self.PREDICTION_WINDOW_LENGTH]) # Shape: (B*N, H)
    return series_input_batch, y_target
  def fit(self, train_data_loader, val_data_loader=None, epochs=10):
    """
    Trains the shared-weight MLP model using the provided data loaders via Keras's Model.fit().
    Args:
      train_data_loader: A grain.DataLoader instance for training data.
      val_data_loader: An optional grain.DataLoader instance for validation data.
      epochs: Number of epochs to train for.
    """
    print(f"\nStarting training for {epochs} epochs using Keras model.fit()...")
    # Create tf.data.Dataset from generators for efficient data pipeline.
    # Set num_epochs=1 for grain.IndexSampler, and tf.data.Dataset.from_generator handles
    # restarting the data source for each epoch provided to model.fit.
    train_dataset = tf.data.Dataset.from_generator(
        lambda: self._raw_data_generator(train_data_loader),
        output_signature=(
            tf.TensorSpec(shape=(None, self.NUM_TIMESTEPS_CONTEXT, self.NUM_NEURONS), dtype=tf.float32),
            tf.TensorSpec(shape=(None, self.PREDICTION_WINDOW_LENGTH, self.NUM_NEURONS), dtype=tf.float32)
        )
    ).map(self._prepare_targets_for_training, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    validation_dataset = None
    callbacks = []
    if val_data_loader:
        validation_dataset = tf.data.Dataset.from_generator(
            lambda: self._raw_data_generator(val_data_loader),
            output_signature=(
                tf.TensorSpec(shape=(None, self.NUM_TIMESTEPS_CONTEXT, self.NUM_NEURONS), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.PREDICTION_WINDOW_LENGTH, self.NUM_NEURONS), dtype=tf.float32)
            )
        )
        validation_dataset = validation_dataset.map(self._prepare_targets_for_training, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        # EarlyStopping remains
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1))
    # Add Cosine Annealing Warm Restarts Learning Rate Callback
    initial_lr_for_schedule = 1e-3 # Starting LR for the first cycle
    first_decay_epochs = 10 # Length of the first LR cycle in epochs
    callbacks.append(CosineAnnealingWarmRestarts(
        initial_lr=initial_lr_for_schedule, 
        first_decay_steps=first_decay_epochs, 
        t_mul=2.0, # Double cycle length each restart
        m_mul=1.0 # Keep max LR constant each restart
    ))
    # Compile the model with Adam optimizer and MAE loss.
    # The learning rate is now managed by the CosineAnnealingWarmRestarts callback.
    self.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_lr_for_schedule), loss='mae')
    # Use the parent's fit method
    super().fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=1 # Show Keras training progress bar
    )
    print(f"Training completed after {epochs} epochs (or early stopping).")
  def predict_non_batched(self, series_input: np.ndarray) -> np.ndarray:
    """Predict the series_output for a single timestep given the series_input.
    Args:
      series_input: np.ndarray of shape (context, neuron)
    Returns:
      np.ndarray of shape (steps_ahead, neuron)
    """
    # Convert input NumPy array to TensorFlow tensor and add a batch dimension.
    series_input_tensor = tf.expand_dims(tf.convert_to_tensor(series_input, dtype=tf.float32), axis=0) # Shape: (1, C, N)
    # Make predictions using the model's `call` method.
    # The `call` method performs all feature engineering internally.
    # The output will be (1 * NUM_NEURONS, PREDICTION_WINDOW_LENGTH).
    predictions_flat = self(series_input_tensor, training=False) # Use self() to invoke the call method
    # Reshape predictions back to (NUM_NEURONS, H) then transpose to (H, NUM_NEURONS).
    predictions_per_neuron = tf.reshape(predictions_flat, [self.NUM_NEURONS, self.PREDICTION_WINDOW_LENGTH]) # Shape: (N, H)
    prediction = tf.transpose(predictions_per_neuron, perm=[1, 0]) # Shape: (H, N)
    return prediction.numpy()
# Dynamically infer the necessary constants from the data source for model instantiation.
NUM_NEURONS_INFERRED = source[0]['series_input'].shape[1]
PREDICTION_WINDOW_LENGTH_INFERRED = constants.PREDICTION_WINDOW_LENGTH
NUM_TIMESTEPS_CONTEXT_INFERRED = num_timesteps_context
# Define the number of epochs to train for model.fit
epochs_to_train_for_model_fit = 120
# Setup train data loader: ensure num_epochs is 1 for consistent tf.data.Dataset generation per epoch
index_sampler = grain.IndexSampler(
    num_records=len(source),
    num_epochs=num_epochs, # This should be 1 as defined globally initially
    shard_options=grain.ShardOptions(
        shard_index=0, shard_count=1, drop_remainder=True),
    shuffle=shuffle,
    seed=101
)
data_loader = grain.DataLoader(
    data_source=source,
    sampler=index_sampler,
    operations=[
        grain.Batch(
            batch_size=batch_size, drop_remainder=True)
    ],
    worker_count=0
)
# Setup validation data loader: ensure num_epochs is 1
val_sources = []
for condition_id in constants.CONDITIONS_TRAIN: # Use training conditions for validation split
  config_val = data_source.TensorStoreTimeSeriesConfig(
      input_spec=data_utils.adjust_spec_for_condition_and_split(
          condition=condition_id,
          split='val', # Crucially, use 'val' split here
          spec=data_utils.get_spec('240930_traces'),
          num_timesteps_context=num_timesteps_context),
      timesteps_input=num_timesteps_context,
      timesteps_output=constants.PREDICTION_WINDOW_LENGTH,
  )
  val_sources.append(data_source.TensorStoreTimeSeries(config_val, prefetch=True))
val_source = data_source.ConcatenatedTensorStoreTimeSeries(*val_sources)
val_batch_size = batch_size # Use same batch size for validation
val_index_sampler = grain.IndexSampler(
    num_records=len(val_source),
    num_epochs=num_epochs, # This should be 1
    shard_options=grain.ShardOptions(
        shard_index=0, shard_count=1, drop_remainder=True),
    shuffle=False, # No need to shuffle validation data
    seed=101 # Consistent seed
)
val_data_loader = grain.DataLoader(
    data_source=val_source,
    sampler=val_index_sampler,
    operations=[
        grain.Batch(
            batch_size=val_batch_size, drop_remainder=True)
    ],
    worker_count=0
)
print(f'\nValidation data loader created with {len(val_source)} records and batch size {val_batch_size}.')
# Instantiate the model by passing the required constants.
model = Model(
    num_timesteps_context=NUM_TIMESTEPS_CONTEXT_INFERRED,
    num_neurons=NUM_NEURONS_INFERRED,
    prediction_window_length=PREDICTION_WINDOW_LENGTH_INFERRED
)
# Perform a dummy forward pass using the actual batch_size to build the model
# and ensure all layer shapes are determined, including those within FeatureExtractor.
dummy_input_for_build = tf.zeros((batch_size, NUM_TIMESTEPS_CONTEXT_INFERRED, NUM_NEURONS_INFERRED), dtype=tf.float32)
_ = model(dummy_input_for_build) # Call the model once to build its layers
print("\nShared-Weight MLP Model Summary (with encapsulated and consolidated FeatureExtractor and Residual Blocks):")
model.summary()
total_params = model.count_params()
# The input dimension to the MLP is now determined by the FeatureExtractor's output
# It's (conv1d_filters + 8 stats + (C-1) deltas + GLOBAL_STATE_DIM + EMBEDDING_DIM)
total_mlp_input_dim = (model.feature_extractor.conv1d_context_processor.filters +
                       8 + # number of static features (mean, std, min, max, last, first, median, range)
                       (model.NUM_TIMESTEPS_CONTEXT - 1) + # number of delta features
                       model.feature_extractor.GLOBAL_STATE_DIM +
                       model.feature_extractor.EMBEDDING_DIM)
print(f"\nTotal combined model parameters: {total_params:,}.")
print(f"This compact model shares its weights across {model.NUM_NEURONS} neurons to predict their future activity.")
print(f"This design is critical for managing the vast number of outputs ({model.NUM_NEURONS} neurons * {model.PREDICTION_WINDOW_LENGTH} steps), providing both local and global context, neuron identity, and now rate of change, with a total input dimension to the MLP of {total_mlp_input_dim}.")
# Now using the encapsulated 'fit' method with validation data.
model.fit(train_data_loader=data_loader, val_data_loader=val_data_loader, epochs=epochs_to_train_for_model_fit)
