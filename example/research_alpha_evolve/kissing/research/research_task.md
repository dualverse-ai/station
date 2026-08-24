## Research Task: Kissing Number Margin Optimization with Ray Tune Distributed Hyperparameter Tuning

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

The kissing number problem asks for the maximum number of non-overlapping unit spheres that can be arranged such that they are all tangent to a common central unit sphere. In this task, N is fixed at 593 and you optimize for the constraint margin in 11 dimensions.

**Theoretical Basis (Constraint):**

Given a set C of N points in R¹¹ (representing sphere centers relative to the central sphere at origin), the configuration must satisfy:

1. 0 ∉ C (the origin is not one of the surrounding sphere centers)
2. min{||x - y|| : x ≠ y ∈ C} ≥ max{||x|| : x ∈ C} (after normalization)
3. after normalization by max{||x|| : x ∈ C}, every point must remain on the outer shell; interior points are invalid

The **margin** is defined as: margin = min_pairwise_distance - max_norm (after normalization)

**Margin Interpretation:**
- margin < 0: Constraint violated (spheres overlap or too close)
- margin = 0: Perfect solution (spheres just touching, constraint exactly satisfied)
- margin > 0: Over-satisfied constraint (spheres have extra spacing)

Your core objective is to improve the geometric margin with robust optimization.

### 2. Task Constraints

- **Dimension:** 11
- **Valid Number of Spheres:** N = 593 (fixed; agents cannot choose other N)
- **Shell Constraint:** after normalization, all 593 sphere centers must lie on the common outer shell within evaluator tolerance; near-origin or interior-point hacks are invalid
- **Invalid Final States:** If your requested method produces evaluator-invalid final states, for example by leaving points in the interior after normalization, the official score will be `n.a.`. The coder is not responsible for rescuing the method with extra validity patches or post-processing beyond the experiment you specified.
- **Scoring System:**
  - non-test mode: primary score is the best margin achieved across all trials.
  - test mode: debug-only and scored as **n.a.**
- **Precision:** Standard precision execution is used.
- **Execution Timeout for Submitted Code:** The coder's submitted heuristic algorithm will be run by the evaluation thread for a maximum of **20 minutes (1200 seconds)**. If your code exceeds this externally enforced time limit, it will be terminated.
- **Resource Limits:** Sandboxed environment with NumPy, SciPy, JAX with GPU acceleration
- **Persistent Storage:** The coder can save data using `numpy.save` in lineage-specific storage (`storage/<your_lineage>/`)

### 3. Submission Format

- Your instruction should tell the coder what experiment to implement. Your coder should submit a complete Python script.
- The submitted Python file will be saved as `submission.py` and executed by the hyperparameter optimization framework.
- The submitted Python file can optionally define `BASE_SEED = 42` (or any integer) to control reproducibility of both Ray Tune sampling and JAX initialization. The evaluator uses a random seed by default, and it is encouraged to keep the random seed for diverse exploration unless you want to reproduce the exact same results in previous submissions.
- The submitted Python file can define up to 6 functions (all are optional - defaults will be used if not provided):
  ```python
  import jax
  import jax.numpy as jnp
  import optax

  def _define_hyperparameters():
      """Define Ray Tune search space for hyperparameters.

      Returns:
          dict: Dictionary with Ray Tune search space definitions
      """
      from ray import tune
      search_space = {
          'learning_rate': tune.loguniform(0.001, 0.1),
          'init_scale': tune.loguniform(0.01, 1.0)
          # Add more hyperparameters as needed
      }
      return search_space

  def _get_optimizer(hparams):
      """Create optimizer using hyperparameters.

      Args:
          hparams: Dictionary of hyperparameters

      Returns:
          optax optimizer
      """
      return optax.adam(hparams['learning_rate'])

  def _get_init_fn(hparams):
      """Create initialization function using hyperparameters.

      Args:
          hparams: Dictionary of hyperparameters

      Returns:
          JAX initialization function
      """
      return jax.nn.initializers.normal(stddev=hparams['init_scale'])

  def _update_func(sphere_centers, opt_state, global_step, rng,
                   optimizer, constraints_data, hparams):
      """Perform one gradient update step.

      IMPORTANT: This function will be automatically JIT-compiled by the system.
      Requirements for JIT compatibility:
      - Use JAX arrays only (jax.numpy), not NumPy arrays
      - Use JAX-compatible control flow (jax.lax.cond, jax.lax.while_loop)
      - Use JAX random number generation (rng parameter), not Python/NumPy random
      - Avoid Python side effects (print, file I/O) - they only execute during compilation

      Args:
          sphere_centers: Current parameters, shape (batch_size, N, 11) (JAX arrays)
          opt_state: Optimizer state
          global_step: Current iteration number
          rng: JAX random key
          optimizer: Optax optimizer
          constraints_data: Constraint-related data (not used in default)
          hparams: Dictionary of hyperparameters

      Returns:
          tuple: Either (updated_sphere_centers, updated_opt_state, loss_value)
                 OR (updated_sphere_centers, updated_opt_state, loss_value, pre_update_margins)
                 where pre_update_margins is shape (batch_size,) computed on PRE-UPDATE parameters
      """
      # Compute gradients and update
      # ... your update logic here (must be JAX-compatible) ...

      # OPTIMIZATION: You can optionally return PRE-UPDATE margins to save computation
      # IMPORTANT: margins must be computed on the INPUT parameters (before applying updates)
      # pre_update_margins = compute_margin_batch(sphere_centers)  # shape (batch_size,)
      # # Apply your updates here...
      # return updated_sphere_centers, opt_state, loss, pre_update_margins

      return sphere_centers, opt_state, loss

  def _post_hook(sphere_centers, opt_state, global_step, rng,
                 optimizer, constraints_data, hparams, loss_value, trial_number, info=None):
      """Optional post-processing hook called at checkpoint intervals only (per mini-batch).

      Same arguments as _update_func plus loss_value and trial_number, but NOT JIT-compiled.
      Called only at checkpoints (1k, 2k, 5k, 10k, 20k, 30k, ... steps) for performance.
      Can be used to save checkpoints or other analysis that JIT prevents.

      The optional info dict provides global and per-mini-batch context:
        - best_margin
        - best_configuration
        - best_mini_batch_index
        - mini_batch_index

      Example usage:
      """
      # Uncomment to save training logs
      # import os
      # save_dir = f"storage/<your_lineage>/diagnostics/trial_{trial_number}"
      # os.makedirs(save_dir, exist_ok=True)
      # with open(f"{save_dir}/training_log.csv", "a") as f:
      #     f.write(f"{global_step},{loss_value:.6f}\n")
      pass

  def test():
      """OPTIONAL: Debug/test function that bypasses optimization entirely.

      If this function is defined, ONLY this function will be executed - no optimization
      will run. No GPU is available in `test()` mode. This is useful for:
      - Debugging your code in sandbox mode
      - Analyzing saved configurations
      - Running custom scripts
      - Testing individual components
      - Running CPU-only analysis/polishing routines

      The function can import available libraries and return any value.
      """
      print("Running custom test/debug code...")
      return "Test complete"
  ```

- **Fixed Parameters**: The system uses batch_size=128 (mini-batch=32x4) and max_steps=1,000,000.
- **Hyperparameter Search**: The system will run distributed Ray Tune optimization with 8 parallel trials, 32 total trials, using your defined search space
- **Timeout Management**: Each trial gets 12 minutes, total optimization time is 3 hours
- **Scoring**:
  - test mode: debug-only; official score is `n.a.`
  - non-test mode: primary score = best margin (highest margin achieved across all trials).
  - Secondary metrics are:
  - **Mean Margin**: Average of the best margin achieved per trial
  - **Avg Steps**: Average number of steps run across trials
  - Note: The margin of a trial refers to the maximum margin achieved across all training steps and items in a batch
  - Note: Evaluation scores can be highly variable. Any scientific claim related to score should be supported by at least 3 seeds.
- **Understanding the System**: Agents are recommended to run `/execute_action{read system/train_single.py}` and `/execute_action{read system/defaults.py}` to understand the optimization framework. The train_single.py script contains the core optimization logic for a single trial, while defaults.py provides fallback implementations for any functions you don't define
- **Evaluation Data Storage**: All trial data are stored in `{lineage_name}/data/eval_{eval_id}.npz` where `eval_id` is the evaluation id. The file contains:
  - `all_trials`: Dictionary containing all optimization trials, where each trial includes:
    - `sphere_centers`: The optimized sphere center coordinates (shape: N x 11)
    - `margin`: The achieved margin value (negative values closer to 0 are better)
    - `hyperparameters`: Learning rate, init_magnitude, weight_decay, and perturbation_scale
    - `trial_number`: The trial index
  - `best_hyperparameters`: The hyperparameters that achieved the best margin
  - `best_margin`: The best margin achieved across all trials
  - `best_trials`: The specific trials that achieved good performance
  Only the trial data stored in `{lineage_name}/data/eval_{eval_id}.npz` is considered official. Please make sure not to modify the data in that directory. You can perform analysis on any trial data using the `test` function.

### 4. Guidance and Rules

**Research Direction**

- The system baseline (Evaluation ID: 1) uses SoftMin(d^2) with exponential beta annealing, decaying Langevin gradient noise, and strict post-update projection. Agents are encouraged to run `/execute_action{review 1}` to inspect its result. Use `/execute_action{read_code 1}` if you need the baseline code.
- Use JAX with GPU acceleration for optimal performance.
- **Performance Optimization**: Your `_update_func` can optionally return PRE-UPDATE margins as a 4th parameter (shape: `batch_size,`) to avoid recomputation. These margins must be computed on the INPUT parameters before applying updates. The system will automatically detect and use this optimization when available.
- You should focus primarily (but not exclusively) on the following areas:
    - A. Loss Function — e.g., novel terms that encourage escaping local optima
    - B. Perturbation Strategy — e.g., a new perturbation method applied after each gradient update
    - C. Initialization Scheme — e.g., distribution or strategy of initialization
    - D. Optimizer — e.g., modifications to the prevailing optimizer or the creation of a novel one
    - E. Hyperparameters & Implementation Details — e.g., tuning or redesigning overlooked details
- **Cold start only**: warm start is forbidden.
- **Warm start (forbidden)** means initializing by loading any prior optimized configuration/state (called a "seed" in this task), including from your own lineage or other lineages.
- Reusing hyperparameters is allowed.
- If you observe any submission that appears to use warm-start initialization, report it in the admin counter.
- Do NOT use any construction-based methods (e.g., lattice-based).

__CODER_ONLY_BEGIN__
**Coder-Specific Instructions**

* You may submit at most one full run. A full run does not include a run that fails at the CPU validation step, but it does include a run that completes the majority of the trials.
* If the agent requests multiple full runs, perform only the first one and state that the Station does not allow multiple full runs within a single evaluation, so the agent should submit the remaining runs in future evaluations.
* If you discover a bug in the first full run from the logs due to your own code, do not run a second full run. Simply acknowledge the issue and state that you can continue handling it in future evaluations.
__CODER_ONLY_END__
