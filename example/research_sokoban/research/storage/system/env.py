"""
Environment wrapper for Sokoban that handles preprocessing and provides a gym-like interface.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Any, Optional
from functools import partial

import jumanji
from jumanji.environments.routing.sokoban import Sokoban
from jumanji.environments.routing.sokoban.generator import HuggingFaceDeepMindGenerator
from jumanji.environments.routing.sokoban.types import State


class SokobanEnvWrapper:
    """Wrapper that handles all environment-specific logic following gym-like interface."""
    
    def __init__(self, dataset_name: str = "unfiltered-train", reward_adjustment: float = 0.09, time_limit: int = 120):
        self.reward_adjustment = reward_adjustment
        self.time_limit = time_limit
        
        # Load and store the dataset
        generator = HuggingFaceDeepMindGenerator(dataset_name, proportion_of_files=1.0)
        self.fixed_grids = generator._fixed_grids
        self.variable_grids = generator._variable_grids
        self.num_levels = len(self.fixed_grids)
        
        # Create base env (we'll bypass its generator for reset)
        self.env = Sokoban(generator=generator, time_limit=time_limit)
    
    def reset(self, key: jax.random.PRNGKey, level_id: Optional[int] = None) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Reset environment, optionally with specific level_id.
        
        Returns:
            observation: Preprocessed observation (8, 8, 8) with one-hot encoding + time
            info: Dictionary with state, level_id, and other metadata
        """
        if level_id is None:
            # Random level selection
            level_id = jax.random.randint(key, shape=(), minval=0, maxval=self.num_levels)
        
        # Get specific level
        fixed_grid = self.fixed_grids[level_id]
        variable_grid = self.variable_grids[level_id]
        
        # Find agent location
        agent_location = jnp.argwhere(variable_grid == 3, size=1)[0]
        
        # Create state manually
        state = State(
            fixed_grid=fixed_grid,
            variable_grid=variable_grid,
            agent_location=agent_location,
            step_count=jnp.array(0, dtype=jnp.int32),
            key=key,
        )
        
        # Preprocess observation
        obs = self._preprocess_obs(jnp.stack([variable_grid, fixed_grid], axis=-1), 0)
        
        info = {
            "state": state,
            "level_id": level_id
        }
        
        return obs, info
    
    def step(self, state: State, action: jnp.ndarray) -> Tuple[jnp.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Returns:
            observation: Preprocessed observation (8, 8, 8)
            reward: Adjusted reward
            done: Whether episode is done
            truncated: Whether episode hit time limit without solving
            info: Dictionary with state, timestep, and solved status
        """
        # Take step in base environment
        next_state, timestep = self.env.step(state, action)
        
        # Preprocess observation
        obs = self._preprocess_obs(
            jnp.stack([next_state.variable_grid, next_state.fixed_grid], axis=-1),
            next_state.step_count
        )
        
        # Adjust reward
        reward = timestep.reward + self.reward_adjustment
        
        # Check done
        done = timestep.last()
        
        # Check if truncated (time limit reached without solving)
        at_time_limit = next_state.step_count >= self.time_limit
        solved = self._is_solved(next_state)
        truncated = done & at_time_limit & ~solved
        
        info = {
            "state": next_state,
            "solved": solved,
            "at_time_limit": at_time_limit
        }
        
        return obs, reward, done, truncated, info
    
    def _preprocess_obs(self, grid: jnp.ndarray, step_count: int) -> jnp.ndarray:
        """Combine grids, one-hot encode, and add time feature.
        
        Args:
            grid: Raw observation grid (10, 10, 2) with variable and fixed grids
            step_count: Current step count
            
        Returns:
            Preprocessed observation (8, 8, 8) with one-hot encoding + time
        """
        # Extract grids
        variable_grid = grid[..., 0]
        fixed_grid = grid[..., 1]
        
        # Combine grids
        combined = self._combine_grids(variable_grid, fixed_grid)
        
        # Convert to integers
        grid_int = combined.astype(jnp.int32)
        
        # One-hot encode the grid values (0-6 -> 7 channels)
        grid_onehot = jax.nn.one_hot(grid_int, 7)  # (10, 10, 7)
        
        # Remove border (walls) to get 8x8 inner grid
        grid_onehot = grid_onehot[1:-1, 1:-1]  # (8, 8, 7)
        
        # Add time feature as t/120 (normalized to [0, 1])
        time_feature = step_count / self.time_limit
        # Create an 8x8 grid filled with the time value
        time_grid = jnp.full((8, 8, 1), time_feature, dtype=jnp.float32)
        
        # Stack with one-hot encoded observation
        processed_obs = jnp.concatenate([grid_onehot, time_grid], axis=-1)  # (8, 8, 8)
        
        return processed_obs
    
    def _combine_grids(self, variable_grid: jnp.ndarray, fixed_grid: jnp.ndarray) -> jnp.ndarray:
        """Combine variable and fixed grids into a single grid with all information."""
        # Start with variable grid (has agent and boxes)
        combined = variable_grid.copy()
        
        # Add walls from fixed grid (value 1)
        wall_mask = fixed_grid == 1
        combined = jnp.where(wall_mask, 1, combined)
        
        # Add targets that aren't covered by agent/box (value 2)
        target_mask = (fixed_grid == 2) & (combined == 0)
        combined = jnp.where(target_mask, 2, combined)
        
        # Note: Agent on target (5) and box on target (6) are already in variable grid
        return combined
    
    def _is_solved(self, state: State) -> bool:
        """Check if all boxes are on targets."""
        mask_box = state.variable_grid == 4  # BOX = 4
        mask_target = state.fixed_grid == 2  # TARGET = 2
        num_boxes_on_targets = jnp.sum(mask_box & mask_target)
        return num_boxes_on_targets == 4  # N_BOXES = 4


def create_train_env(reward_adjustment: float = 0.09, time_limit: int = 120) -> SokobanEnvWrapper:
    """Create training environment wrapper."""
    return SokobanEnvWrapper(
        dataset_name="unfiltered-train",
        reward_adjustment=reward_adjustment,
        time_limit=time_limit
    )


def create_valid_env(reward_adjustment: float = 0.09, time_limit: int = 120) -> SokobanEnvWrapper:
    """Create validation environment wrapper with fixed 1000-level subset.

    Uses a fixed random subset of 1000 levels from the 100k validation set.
    The subset is deterministic (seed 999) and independent of training seeds.
    """
    # Load full validation dataset
    generator = HuggingFaceDeepMindGenerator('unfiltered-valid', proportion_of_files=1.0)

    # Create fixed subset using independent random state (doesn't affect training seeds)
    validation_rng = np.random.RandomState(999)  # Fixed seed for reproducible subset
    indices = validation_rng.choice(100000, size=1000, replace=False)
    indices = np.sort(indices)

    # Create wrapper with subset
    wrapper = SokobanEnvWrapper(
        dataset_name="unfiltered-valid",
        reward_adjustment=reward_adjustment,
        time_limit=time_limit
    )

    # Override with subset
    wrapper.fixed_grids = generator._fixed_grids[indices]
    wrapper.variable_grids = generator._variable_grids[indices]
    wrapper.num_levels = 1000

    return wrapper


def create_test_env(reward_adjustment: float = 0.09, time_limit: int = 120, num_levels: int = 1000) -> SokobanEnvWrapper:
    """Create test environment wrapper with fixed subset from unfiltered-test.

    Uses a fixed random subset of levels from the test set.
    The subset is deterministic (seed 42) for reproducible evaluation.
    """
    # Load full test dataset
    generator = HuggingFaceDeepMindGenerator('unfiltered-test', proportion_of_files=1.0)

    # Create fixed subset using independent random state
    test_rng = np.random.RandomState(42)  # Fixed seed for reproducible subset
    total_levels = len(generator._fixed_grids)

    # If test set has fewer levels than requested, use all of them
    actual_num_levels = min(num_levels, total_levels)
    indices = test_rng.choice(total_levels, size=actual_num_levels, replace=False)
    indices = np.sort(indices)

    # Create wrapper with test dataset
    wrapper = SokobanEnvWrapper(
        dataset_name="unfiltered-test",
        reward_adjustment=reward_adjustment,
        time_limit=time_limit
    )

    # Override with subset
    wrapper.fixed_grids = generator._fixed_grids[indices]
    wrapper.variable_grids = generator._variable_grids[indices]
    wrapper.num_levels = actual_num_levels

    return wrapper