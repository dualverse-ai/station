"""
Single-seed training script for Sokoban RL using Policy Gradient.
This is adapted from the original train.py to work with Ray workers.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'

# Force offline mode for HuggingFace datasets (use cached data only)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# Suppress all warnings before any imports
import warnings
warnings.filterwarnings('ignore')  # Suppress ALL warnings

import sys
import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
import optax
import numpy as np
from typing import Dict, Tuple, Any, Optional
import time
from functools import partial

from env import create_train_env, create_valid_env

# Global constants for JIT compatibility
NUM_ENVS = 64
UNROLL_LENGTH = 20


def validate_network_outputs(network: Any, params: Any, config: Dict):
    """
    Performs a non-JIT-compiled check on the network's output shapes.
    """
    print("Performing initial network output shape validation...")
    try:
        batch_size = config.get('num_envs', 64)
        dummy_obs = jnp.zeros((batch_size, 8, 8, 8), dtype=jnp.float32)
        dummy_done = jnp.zeros((batch_size,), dtype=bool)
        
        outputs = network.apply(params, dummy_obs, dummy_done, None)
        
        if len(outputs) == 3:
            logits, value, _ = outputs
        elif len(outputs) == 2:
            logits, value = outputs
        else:
            raise ValueError(f"Expected 2 or 3 outputs, got {len(outputs)}")
        
        expected_logits_shape = (batch_size, 4)
        expected_value_shape = (batch_size,)
        
        if logits.shape != expected_logits_shape:
            raise ValueError(f"Logits shape mismatch: expected {expected_logits_shape}, got {logits.shape}")
        
        if value.shape != expected_value_shape:
            if value.shape == (batch_size, 1):
                print("Warning: value output has shape (B, 1), recommended (B,)")
            else:
                raise ValueError(f"Value shape mismatch: expected {expected_value_shape}, got {value.shape}")
        
        print("Network output shapes are correct.")
    except Exception as e:
        print(f"Network validation error: {e}")
        raise e


@partial(jit, static_argnums=(0,))
def select_action(
    network: Any, 
    params: Any, 
    observation: jnp.ndarray, 
    key: jax.random.PRNGKey, 
    rnn_state: Any = None, 
    done: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
    """Select action using the policy network."""
    # Add batch dimension for single observation
    observation_batched = observation[None, ...]
    done_batched = done[None, ...] if done is not None else None
    
    # Add batch dimension to RNN state if needed
    if rnn_state is not None:
        rnn_state_batched = jax.tree.map(lambda x: x[None, ...], rnn_state)
    else:
        rnn_state_batched = None
    
    # Get network outputs (always batched)
    outputs = network.apply(params, observation_batched, done_batched, rnn_state_batched)
    if len(outputs) == 3:
        logits, value, new_rnn_state = outputs
    else:
        logits, value = outputs
        new_rnn_state = rnn_state
    
    # Remove batch dimension
    logits = logits[0]
    value = value.reshape(())  # Convert to scalar
    
    # Remove batch dimension from RNN state if present
    if new_rnn_state is not None:
        new_rnn_state = jax.tree.map(lambda x: x[0], new_rnn_state)
    
    # Sample action
    action = jax.random.categorical(key, logits)
    
    # Calculate log probability
    log_probs = jax.nn.log_softmax(logits)
    log_prob = log_probs[action]
    
    return action, log_prob, value, new_rnn_state


def collect_trajectory(
    env: Any,
    network: Any,
    params: Any,
    env_states: Any,
    observations: jnp.ndarray,
    episode_returns: jnp.ndarray,
    episode_lengths: jnp.ndarray,
    key: jax.random.PRNGKey,
    rnn_states: Any = None,
) -> Tuple[Dict[str, jnp.ndarray], Any, Any, jnp.ndarray, jnp.ndarray, Tuple, Any]:
    """Collect a trajectory of experience from parallel environments."""
    
    def env_step(carry, unused):
        env_states, key, observations, episode_returns, episode_lengths, rnn_states, prev_dones = carry
        key, *action_keys = jax.random.split(key, NUM_ENVS + 1)
        
        # Select actions
        actions, log_probs, values, new_rnn_states = vmap(
            select_action, in_axes=(None, None, 0, 0, 0, 0)
        )(network, params, observations, jnp.array(action_keys), rnn_states, prev_dones)
        
        # Step environments
        step_results = vmap(env.step)(env_states, actions)
        next_observations, rewards, dones, truncated_dones, infos = step_results
        
        # Extract new states and track solved episodes
        env_states = vmap(lambda info: info['state'])(infos)
        solved = vmap(lambda info: info['solved'])(infos)
        
        # Update episode tracking
        episode_returns = episode_returns + rewards
        episode_lengths = episode_lengths + 1
        
        # Track completed episodes
        completed_mask = dones.astype(jnp.float32)
        step_completed_returns = episode_returns * completed_mask
        step_completed_lengths = episode_lengths * completed_mask
        step_episodes_solved = solved.astype(jnp.float32) * completed_mask
        
        # Create transition
        transition = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'log_probs': log_probs,
            'completed_returns': step_completed_returns,
            'completed_lengths': step_completed_lengths,
            'episodes_solved': step_episodes_solved,
            'episodes_completed': completed_mask,
            'rnn_states': rnn_states,
        }
        
        # Reset done environments
        key, *reset_keys = jax.random.split(key, NUM_ENVS + 1)
        
        def maybe_reset(state, obs, done, key):
            def reset_fn(_):
                new_obs, new_info = env.reset(key)
                return new_obs, new_info
            def keep_fn(_):
                return obs, {'state': state, 'level_id': jnp.array(-1)}
            return lax.cond(done, reset_fn, keep_fn, None)
        
        reset_results = vmap(maybe_reset)(env_states, next_observations, dones, jnp.array(reset_keys))
        next_observations, reset_infos = reset_results
        env_states = vmap(lambda info: info['state'])(reset_infos)
        
        # Reset episode stats
        episode_returns = jnp.where(dones, 0.0, episode_returns)
        episode_lengths = jnp.where(dones, 0, episode_lengths)
        
        return (env_states, key, next_observations, episode_returns, episode_lengths, new_rnn_states, dones), transition
    
    # Collect trajectory
    initial_dones = jnp.zeros(NUM_ENVS, dtype=bool)
    final_carry, trajectory = lax.scan(
        env_step, 
        (env_states, key, observations, episode_returns, episode_lengths, rnn_states, initial_dones), 
        None, 
        length=UNROLL_LENGTH
    )
    
    # Unpack final carry
    final_env_states, final_key, final_observations, final_episode_returns, final_episode_lengths, final_rnn_states, _ = final_carry
    
    # Stack trajectory: (T, B, ...) -> (B, T, ...)
    trajectory = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), trajectory)
    
    # Get final values for GAE bootstrapping
    final_dones = jnp.zeros(NUM_ENVS, dtype=bool)
    final_values_output = network.apply(params, final_observations, final_dones, final_rnn_states)
    if len(final_values_output) == 3:
        _, final_values, _ = final_values_output
    else:
        _, final_values = final_values_output
    
    final_values = final_values.reshape(-1)
    
    trajectory['final_values'] = final_values
    trajectory['initial_rnn_states'] = rnn_states
    
    # Calculate episode statistics
    total_episodes = jnp.sum(trajectory['episodes_completed'])
    total_returns = jnp.sum(trajectory['completed_returns'])
    total_lengths = jnp.sum(trajectory['completed_lengths'])
    total_solved = jnp.sum(trajectory['episodes_solved'])
    
    mean_return = jnp.where(total_episodes > 0, total_returns / total_episodes, 0.0)
    mean_length = jnp.where(total_episodes > 0, total_lengths / total_episodes, 0.0)
    solve_rate = jnp.where(total_episodes > 0, total_solved / total_episodes, 0.0)
    
    stats = (mean_return, mean_length, solve_rate, total_episodes)
    
    return trajectory, final_env_states, final_observations, final_episode_returns, final_episode_lengths, stats, final_rnn_states


def evaluate_agent(
    env: Any,
    network: Any,
    params: Any,
    key: jax.random.PRNGKey,
    time_limit: int = 120,
) -> Dict[str, float]:
    """Evaluate agent on all test levels."""
    
    @jit
    def run_episode(key, level_id):
        """Run a single episode and return whether it was solved."""
        key, reset_key = jax.random.split(key)
        initial_obs, info = env.reset(reset_key, level_id)
        state = info['state']
        
        # Initialize RNN state if needed
        try:
            dummy_done = jnp.array([False])
            outputs = network.apply(params, initial_obs[None, ...], dummy_done, None)
            if len(outputs) == 3 and outputs[2] is not None:
                rnn_state = jax.tree.map(lambda x: x[0], outputs[2])
            else:
                rnn_state = None
        except:
            rnn_state = None
        
        def step_fn(carry, unused):
            state, key, done, rnn_state, prev_done, current_obs = carry
            key, action_key = jax.random.split(key)
            
            # Select action
            action, _, _, new_rnn_state = select_action(
                network, params, current_obs, action_key, rnn_state, prev_done
            )
            
            # Step environment
            next_obs, reward, step_done, truncated, info = env.step(state, action)
            next_state = info['state']
            solved = info['solved']
            
            done = done | step_done
            
            return (next_state, key, done, new_rnn_state, done, next_obs), solved
        
        _, episode_solved = lax.scan(
            step_fn, 
            (state, key, False, rnn_state, False, initial_obs), 
            None, 
            length=time_limit
        )
        
        return jnp.any(episode_solved)
    
    # Run evaluation on all levels
    episodes_to_run = env.num_levels
    key, shuffle_key = jax.random.split(key)
    level_ids = jax.random.permutation(shuffle_key, env.num_levels)
    
    keys = jax.random.split(key, episodes_to_run)
    solved = vmap(run_episode)(keys, level_ids)
    solve_rate = jnp.mean(solved.astype(jnp.float32))
    
    return {
        'solve_rate': float(solve_rate),
        'num_episodes': episodes_to_run,
        'solved_count': int(jnp.sum(solved)),
    }


def run_single_seed_training(funcs, hparams, seed, config, trial_timeout):
    """Run single-seed Sokoban RL training and return final solve rate."""

    # Use the submission's training_step function directly
    training_step = funcs['training_step']

    NUM_UPDATES = config['total_timesteps'] // (config['num_envs'] * config['unroll_length'])
    
    # Create environments
    train_env = create_train_env(time_limit=config['time_limit'])
    # Use validation dataset for evaluation (fixed 1000 subset)
    test_env = create_valid_env(time_limit=config['time_limit'])
    
    # Initialize network
    network = funcs['create_network'](hparams)
    
    # Create dummy observation for initialization
    dummy_obs = jnp.zeros((1, 8, 8, 8), dtype=jnp.float32)
    dummy_done = jnp.zeros((1,), dtype=bool)
    
    # Initialize parameters and optimizer
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = network.init(init_key, dummy_obs, dummy_done)
    
    # Validate network output shapes
    validate_network_outputs(network, params, config)
    
    # Create optimizer with hyperparameters
    learning_rate = hparams['learning_rate']
    optimizer = funcs['create_optimizer'](learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    
    # Initialize environments
    key, *env_keys = jax.random.split(key, config['num_envs'] + 1)
    reset_results = vmap(train_env.reset)(jnp.array(env_keys), None)
    initial_observations, initial_infos = reset_results
    env_states = vmap(lambda info: info['state'])(initial_infos)
    
    # Initialize RNN states if needed
    try:
        dummy_dones = jnp.zeros(config['num_envs'], dtype=bool)
        outputs = network.apply(params, initial_observations, dummy_dones, None)
        if len(outputs) == 3 and outputs[2] is not None:
            _, _, rnn_states = outputs
        else:
            rnn_states = None
    except:
        rnn_states = None
    
    # Create JIT-compiled training step with hparams in closure (like matrix task)
    def create_training_step_jit():
        @jit
        def training_step_compiled(params, opt_state, trajectory):
            # hparams captured from outer scope, training_step called with it
            return training_step(network, optimizer, params, opt_state, trajectory, hparams)
        return training_step_compiled
    
    # JIT compile functions exactly like matrix task
    print("Compiling JAX functions...")
    collect_trajectory_jit = jit(
        partial(collect_trajectory, train_env, network)
    )
    training_step_jit = create_training_step_jit()
    
    # Training state
    episode_returns = jnp.zeros(config['num_envs'])
    episode_lengths = jnp.zeros(config['num_envs'], dtype=jnp.int32)
    current_observations = initial_observations
    
    # Compact logging like matrix task
    step_markers = []
    return_values = []
    solve_values = []
    
    # Training loop
    start_time = time.time()
    timeout_reached = False
    final_update = 0
    
    for update in range(NUM_UPDATES):
        # Collect trajectory
        key, collect_key = jax.random.split(key)
        trajectory, env_states, current_observations, episode_returns, episode_lengths, stats, rnn_states = collect_trajectory_jit(
            params, env_states, current_observations, episode_returns, episode_lengths, collect_key, rnn_states
        )
        
        # Update parameters 
        params, opt_state = training_step_jit(params, opt_state, trajectory)
        
        # Update RNN states for next rollout (CRITICAL for RNN networks!)
        # This carries the final RNN states from this rollout to the next
        
        # Check timeout
        if update % config['timeout_check_interval'] == 0:
            elapsed_time = time.time() - start_time
            if elapsed_time > trial_timeout:
                timeout_reached = True
                final_update = update + 1
                break
        
        # Collect data for compact logging every 5M steps (3906 updates)
        if update % 3906 == 0 or update == NUM_UPDATES - 1:
            mean_return, mean_length, solve_rate, num_episodes = stats
            timesteps = (update + 1) * config['num_envs'] * config['unroll_length']
            step_markers.append(f"{timesteps/1000000:.1f}M")
            return_values.append(f"{float(mean_return):.2f}")
            solve_values.append(f"{float(solve_rate):.3f}")
    
    # Final evaluation
    key, eval_key = jax.random.split(key)
    results = evaluate_agent(test_env, network, params, eval_key, config['time_limit'])
    final_solve_rate = results['solve_rate']
    
    # Print compact table at end (like matrix task)
    if step_markers:
        print("\nStep | " + " | ".join(step_markers))
        print("--- | " + " | ".join(["---"] * len(step_markers)))
        print("Return | " + " | ".join(return_values))
        print("Solve | " + " | ".join(solve_values))
    
    # Final results
    total_time = time.time() - start_time
    if timeout_reached:
        print(f"\nTimeout after {final_update} updates ({total_time:.0f}s)")
    else:
        print(f"\nCompleted {NUM_UPDATES} updates ({total_time:.0f}s)")
    
    print(f"Final test solve rate: {final_solve_rate:.3f}")

    return final_solve_rate, params, opt_state