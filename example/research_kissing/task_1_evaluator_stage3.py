# station_data/rooms/research/evaluators/task_1_evaluator.py
"""
Evaluator for Research Task 1: Kissing Number Hyperparameter Optimization
"""

import os
import sys
import re
import glob
import shutil
import itertools
import numpy as np
from typing import Tuple, Optional

# Add station package to path for imports
station_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if station_path not in sys.path:
    sys.path.insert(0, station_path)

from station.eval_research.base_evaluator import ResearchTaskEvaluator
from station import constants
from station import file_io_utils


class Task1Evaluator(ResearchTaskEvaluator):
    """
    Evaluator for Research Task 1: Kissing Number Hyperparameter Optimization
    """
    
    def __init__(self):
        super().__init__()
        # Fixed parameters for this task
        self.dim = 11
        self.default_n = 593  # Fixed number of spheres
    
    def get_execution_mode(self) -> str:
        """Use command mode for this task."""
        return "command"
    
    def get_execution_command(self) -> str:
        """
        Return the command to execute for this task.
        The submission code will be saved as submission.py in the working directory.
        """
        # Use storage-relative path that works in both Docker and Python sandbox modes
        return "python storage/system/main.py"
    
    def get_secondary_metrics_format(self):
        """Define secondary metrics with appropriate formatting."""
        return {
            "N": "d",  # Integer format for number of spheres
            "Mean Margin": ".6f",  # 6 decimal places
            "Avg Steps": ".1f"  # 1 decimal place
        }
    
    def evaluate_submission(self, result: any = None, eval_id: str = None, author: str = None) -> Tuple[bool, any, str, Tuple]:
        """
        This method is called after code execution completes.
        For this task, we parse the UUID from output, load the result, and calculate margin score.
        Returns: (success, score, details, sort_key)
        """
        if result is None:
            return False, 'n.a.', "No output received from optimization script", (float('-inf'),)
        
        try:
            # Parse the output
            output_str = str(result)
            
            # Check if this is test mode and run discrete end-game verification.
            if "=== Test Mode Detected ===" in output_str:
                uuid_match = re.search(r'SUBMISSION_UUID:\s*([a-f0-9-]+)', output_str)
                if not uuid_match:
                    return False, 'n.a.', "Test mode - missing SUBMISSION_UUID", (float('-inf'),)
                submission_uuid = uuid_match.group(1)

                research_room_path = os.path.join(
                    constants.BASE_STATION_DATA_PATH,
                    constants.ROOMS_DIR_NAME,
                    constants.SHORT_ROOM_NAME_RESEARCH
                )
                state_file = os.path.join(
                    research_room_path,
                    constants.RESEARCH_STORAGE_DIR,
                    constants.RESEARCH_STORAGE_SHARED_DIR,
                    'data',
                    f"test_state_{submission_uuid}.npz",
                )
                if not os.path.exists(state_file):
                    return False, 'n.a.', "Test mode - no discrete state saved", (float('-inf'),)

                data = np.load(state_file, allow_pickle=True)
                if 'sphere_centers' not in data.files:
                    return False, 'n.a.', "Test mode - test_state.npz missing sphere_centers", (float('-inf'),)

                state = np.asarray(data['sphere_centers'])
                if state.ndim != 2:
                    return False, 'n.a.', f"Test mode - invalid state rank: {state.shape}", (float('-inf'),)
                if state.shape == (self.dim, self.default_n):
                    state = state.T
                if state.shape != (self.default_n, self.dim):
                    return False, 'n.a.', f"Test mode - invalid state shape: {state.shape}", (float('-inf'),)

                official_test_file = os.path.join(
                    research_room_path,
                    constants.RESEARCH_STORAGE_DIR,
                    constants.RESEARCH_STORAGE_SHARED_DIR,
                    'data',
                    f"eval_{eval_id}_test.npz",
                )
                np.savez(official_test_file, sphere_centers=np.asarray(state))

                passed, min_d2, max_n2, ratio = self._verify_integer_state(state)
                margin = self._evaluate_margin(np.asarray(state, dtype=np.float64))
                message = (
                    ("Discrete integer verifier passed in test mode.\n" if passed else "Discrete integer verifier failed in test mode.\n")
                    + f"min_squared_distance: {min_d2}\n"
                    + f"max_squared_norm: {max_n2}\n"
                    + f"ratio: {ratio:.12f}\n"
                    + f"normalized_margin: {margin:.12f}\n"
                    + f"official_test_file: {official_test_file}"
                )
                details = {
                    "N": self.default_n,
                    "Mean Margin": margin,
                    "Avg Steps": 0.0,
                    "Message": message
                }

                if passed:
                    return True, 100.0, details, (100.0,)
                return False, 'n.a.', details, (float('-inf'),)
            
            # Look for the SUBMISSION_UUID line
            uuid_match = re.search(r'SUBMISSION_UUID:\s*([a-f0-9-]+)', output_str)
            if uuid_match:
                submission_uuid = uuid_match.group(1)

                # Parse NUM_TRIALS from output (default to 32 if not found)
                num_trials_match = re.search(r'NUM_TRIALS:\s*(\d+)', output_str)
                num_trials = int(num_trials_match.group(1)) if num_trials_match else 32
                
                # Set up paths
                research_room_path = os.path.join(
                    constants.BASE_STATION_DATA_PATH,
                    constants.ROOMS_DIR_NAME,
                    constants.SHORT_ROOM_NAME_RESEARCH
                )
                tmp_dir = os.path.join(
                    research_room_path,
                    constants.RESEARCH_STORAGE_DIR,
                    constants.RESEARCH_STORAGE_SHARED_DIR,
                    'tmp'
                )
                submission_dir = os.path.join(tmp_dir, submission_uuid)
                
                if not os.path.exists(submission_dir):
                    return False, 'n.a.', f"Submission directory not found: {submission_dir}", (float('-inf'),)
                
                # Load all trial files
                trial_files = glob.glob(os.path.join(submission_dir, "trial_*.npz"))
                if not trial_files:
                    return False, 'n.a.', f"No trial files found in {submission_dir}", (float('-inf'),)
                
                trial_data = {}
                steps_runs = []
                best_margin = float('-inf')
                best_configuration = None
                submission_n = None  # Track the N used in submission

                for trial_file in trial_files:
                    trial_num = os.path.basename(trial_file).replace('trial_', '').replace('.npz', '')
                    data = np.load(trial_file, allow_pickle=True)
                    sphere_centers = data['sphere_centers']
                    margin = float(data['margin'])
                    steps_run = int(data['steps_run']) if 'steps_run' in data.files else 0
                    trial_n = sphere_centers.shape[0]  # Get N from sphere_centers shape
                    if submission_n is None:
                        submission_n = trial_n
                    
                    # Verify the configuration using our evaluation function
                    try:
                        verified_margin = self._evaluate_margin(sphere_centers)
                        # Use verified margin if it's close to the reported one
                        if abs(verified_margin - margin) < 1e-3:
                            final_margin = verified_margin
                        else:
                            print(f"Warning: Margin mismatch for trial {trial_num}. Reported: {margin:.6f}, Verified: {verified_margin:.6f}")
                            final_margin = verified_margin
                    except Exception as e:
                        return False, 'n.a.', f"Verification failed for trial {trial_num}: {str(e)}", (float('-inf'),)
                    
                    trial_data[trial_num] = {
                        'sphere_centers': sphere_centers,
                        'margin': final_margin,
                        'n': trial_n,
                        'steps_run': steps_run,
                        'hyperparameters': data['hyperparameters'].item() if data['hyperparameters'].ndim == 0 else data['hyperparameters'],
                        'trial_number': int(data['trial_number'])
                    }
                    steps_runs.append(steps_run)
                    
                    # Track best
                    if final_margin > best_margin:
                        best_margin = final_margin
                        best_configuration = sphere_centers
                
                # Get best trial's hyperparameters
                best_trial = max(trial_data.values(), key=lambda x: x['margin'])
                best_hparams = best_trial['hyperparameters']

                # Calculate secondary metrics
                # Best Margin: highest margin achieved across all trials
                best_margin_secondary = max(trial['margin'] for trial in trial_data.values())
                # Mean Margin: average of best margin per trial (in this case, all trials are individual)
                mean_margin = sum(trial['margin'] for trial in trial_data.values()) / len(trial_data)
                # Avg Steps: average number of steps run across trials
                avg_steps = sum(steps_runs) / len(steps_runs) if steps_runs else 0.0

                # Check if N is in valid set
                valid_n_values = {593}
                if submission_n not in valid_n_values:
                    # Invalid N: primary score is disqualified
                    final_score = float('-inf')
                    sort_key = (float('-inf'),)
                else:
                    # Valid N: primary score is best margin across all trials
                    final_score = float(best_margin_secondary)
                    sort_key = (final_score,)
                
                eval_data = {
                    'all_trials': trial_data,
                    'final_score': final_score,
                    'best_margin': best_margin_secondary,
                    'mean_margin': mean_margin,
                    'avg_steps': avg_steps,
                    'n': submission_n,
                    'num_trials': len(trial_data),
                    'submission_uuid': submission_uuid,
                    'best_hyperparameters': best_hparams
                }
                persist_eval_artifact = (author or "").strip().lower() != "system"
                if persist_eval_artifact:
                    # Save organized evaluation data to author's lineage storage.
                    author_lineage = self._extract_lineage_name(author)
                    eval_data_dir = os.path.join(
                        research_room_path,
                        "storage",
                        "lineages",
                        author_lineage,
                        "data",
                    )
                    file_io_utils.ensure_dir_exists(eval_data_dir)
                    eval_file = os.path.join(eval_data_dir, f"eval_{eval_id}.npz")
                    np.savez(eval_file, **{k: v for k, v in eval_data.items()})
                
                # Save to internal storage with proper naming
                save_result = self._save_successful_config(
                    best_configuration, best_margin_secondary, eval_id, author,
                    {'best_hyperparameters': best_hparams, 'num_trials': len(trial_data), 'n': submission_n}
                )
                
                # Clean up submission temp directory
                try:
                    shutil.rmtree(submission_dir)
                except:
                    pass  # Ignore cleanup errors
                
                # Build details with secondary metrics
                message_str = f"Total trials: {len(trial_data)}\nBest hyperparameters:"
                for hparam_key, hparam_value in best_hparams.items():
                    if hparam_key not in {'base_seed', '_shared_storage_path', '_submission_uuid'}:  # Skip constants
                        if isinstance(hparam_value, (int, float)):
                            message_str += f"\n  {hparam_key}: {hparam_value:.6f}"
                        else:
                            message_str += f"\n  {hparam_key}: {hparam_value}"
                if not persist_eval_artifact:
                    message_str += "\nNote: System baseline artifacts are not persisted to storage."

                details = {
                    "N": submission_n,
                    "Mean Margin": mean_margin,
                    "Avg Steps": avg_steps,
                    "Message": message_str
                }

                return True, final_score, details, sort_key
            else:
                # Try to find error messages
                if "timeout" in output_str.lower():
                    return False, 'n.a.', "Optimization timed out before finding solution", (float('-inf'),)
                elif "OPTIMIZATION FAILED" in output_str:
                    return False, 'n.a.', "Optimization failed - no valid solution found", (float('-inf'),)
                elif "error" in output_str.lower():
                    error_lines = [line for line in output_str.split('\n') if 'error' in line.lower()]
                    error_msg = error_lines[0] if error_lines else "Unknown error during optimization"
                    return False, 'n.a.', f"Optimization failed: {error_msg}", (float('-inf'),)
                else:
                    return False, 'n.a.', "Could not parse submission UUID from output", (float('-inf'),)
            
        except Exception as e:
            return False, 'n.a.', f"Error evaluating results: {str(e)}", (float('-inf'),)
    
    def get_expected_function_name(self) -> str:
        """Not used for command mode tasks."""
        return "dummy_function"  # Required by base class but not used in command mode
    
    def get_task_description(self) -> str:
        return "Kissing Number Hyperparameter Optimization"
    
    def _evaluate_margin(self, sphere_centers: np.ndarray) -> float:
        """Calculate normalized margin and reject interior-point shell hacks."""

        n = sphere_centers.shape[0]  # Get N from shape

        if sphere_centers.shape[1] != self.dim:
            raise AssertionError(f"Expected dimension {self.dim}, got {sphere_centers.shape[1]}")
        
        # Check for inf or nan values
        if np.any(np.isnan(sphere_centers)) or np.any(np.isinf(sphere_centers)):
            raise AssertionError("Configuration contains NaN or infinite values")

        norms = np.linalg.norm(sphere_centers, axis=1)
        max_norm = np.max(norms)

        if max_norm <= 1e-10:
            raise AssertionError("All points too close to origin")

        shell_tol = 1e-3
        normalized_norms = norms / max_norm
        min_normalized_norm = np.min(normalized_norms)
        if min_normalized_norm < 1.0 - shell_tol:
            raise AssertionError(
                f"Configuration contains interior points: min normalized norm {min_normalized_norm:.6f}"
            )

        normalized_centers = sphere_centers / max_norm

        # Calculate pairwise distances
        diff = normalized_centers[:, None, :] - normalized_centers[None, :, :]
        squared_distances = np.sum(diff**2, axis=2)
        
        # Get minimum pairwise distance (excluding diagonal)
        mask = ~np.eye(squared_distances.shape[0], dtype=bool)
        min_squared_distance = np.min(squared_distances[mask])
        min_distance = np.sqrt(min_squared_distance)
        
        # Max norm is 1 after normalization
        margin = min_distance - 1.0
        
        return float(margin)  # Convert to Python float for JSON serialization

    def _verify_integer_state(self, sphere_centers: np.ndarray):
        """Run the exact integer verifier for end-game test mode."""
        centers_i64 = np.around(sphere_centers).astype(np.int64)

        def sq(v):
            return sum(pow(int(x), 2) for x in v)

        squared_norms = [sq(list(center)) for center in centers_i64]
        min_squared_norm = min(squared_norms)
        max_squared_norm = max(squared_norms)
        min_squared_distance = min(
            sq(list(a - b)) for a, b in itertools.combinations(centers_i64, 2)
        )
        ratio = float(min_squared_distance) / float(max_squared_norm) if max_squared_norm > 0 else -1.0
        passed = (min_squared_norm > 1e-6) and (min_squared_distance >= max_squared_norm)
        return passed, int(min_squared_distance), int(max_squared_norm), ratio

    def _save_successful_config(self, sphere_centers: np.ndarray, margin: float, eval_id: str = None, author: str = None, data: dict = None) -> str:
        """Save successful configuration to internal/configurations directory."""
        try:
            # Create internal/configurations directory path
            research_room_path = os.path.join(
                constants.BASE_STATION_DATA_PATH,
                constants.ROOMS_DIR_NAME,
                constants.SHORT_ROOM_NAME_RESEARCH
            )
            configs_dir = os.path.join(
                research_room_path,
                constants.RESEARCH_INTERNAL_DIR,
                "configurations"
            )
            
            # Ensure directory exists
            file_io_utils.ensure_dir_exists(configs_dir)
            
            # Create filename with format: {author}_{margin}_{eval_id}.npz
            if author and eval_id:
                author_clean = author.replace(" ", "_")
                filename = f"{author_clean}_{margin:.6f}_{eval_id}.npz"
            elif eval_id:
                filename = f"unknown_{margin:.6f}_{eval_id}.npz"
            else:
                filename = f"unknown_{margin:.6f}_legacy.npz"
            
            filepath = os.path.join(configs_dir, filename)
            
            # Save the configuration and hyperparameters
            save_data = {
                'sphere_centers': sphere_centers,
                'margin': margin,
                'n': sphere_centers.shape[0],  # Save actual N from sphere_centers shape
                'dim': self.dim
            }
            
            # Add hyperparameters if available
            if data is not None and 'best_hyperparameters' in data:
                # best_hyperparameters is saved as an object array containing a dict
                hparams = data['best_hyperparameters']
                save_data['best_hyperparameters'] = hparams
                if 'num_trials' in data:
                    save_data['num_trials'] = int(data['num_trials'])
                if 'n' in data:
                    save_data['n'] = int(data['n'])
            
            np.savez(filepath, **save_data)
            
            # Successfully saved
            return "saved"
            
        except Exception as e:
            # Failed to save
            print(f"Failed to save configuration: {str(e)}")
            return "error"

    @staticmethod
    def _extract_lineage_name(author: Optional[str]) -> str:
        """Extract and sanitize lineage name from author string."""
        if not author:
            return "unknown"
        lineage = str(author).strip().split(" ")[0].lower()
        sanitized = "".join(c for c in lineage if c.isalnum() or c in {"_", "-"})
        return sanitized or "unknown"
