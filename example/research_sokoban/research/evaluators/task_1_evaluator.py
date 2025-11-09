# station_data/rooms/research/evaluators/task_1_evaluator.py
"""
Evaluator for Research Task 1: Reinforcement Learning on Sokoban with Ray Distributed Training
"""

import os
import sys
import re
import glob
import shutil
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
    Evaluator for Research Task 1: RL on Sokoban with Ray Distributed Training
    """
    
    def __init__(self):
        super().__init__("1")
    
    def get_execution_mode(self) -> str:
        """Use command mode for this task."""
        return "command"
    
    def get_execution_command(self) -> str:
        """
        Return the command to execute for this task.
        The submission code will be saved as submission.py in the working directory.
        """
        # Use storage-relative path that works in both Docker and Python sandbox modes
        # Let JAX automatically choose available backend (CPU fallback when no GPUs visible)
        return "CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS='' python storage/system/main.py"
    
    def evaluate_submission(self, result: any = None, eval_id: str = None, author: str = None) -> Tuple[bool, any, str]:
        """
        This method is called after code execution completes.
        For this task, we parse the UUID from output, load the results, and aggregate 4 seed scores.
        Returns: (success, score, details)
        """
        if result is None:
            return False, 'n.a.', "No output received from Ray optimization script"
        
        try:
            # Parse the output
            output_str = str(result)
            
            # Check if this is test mode - if so, don't attempt UUID loading
            if "=== Test Mode Detected ===" in output_str:
                return False, 'n.a.', "Test mode - no scoring"
            
            # Look for the SUBMISSION_UUID line
            uuid_match = re.search(r'SUBMISSION_UUID:\s*([a-f0-9-]+)', output_str)
            if uuid_match:
                submission_uuid = uuid_match.group(1)
                
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
                    return False, 'n.a.', f"Submission directory not found: {submission_dir}"
                
                # Load all trial files (expect 4 seeds)
                trial_files = glob.glob(os.path.join(submission_dir, "trial_*.npz"))
                if not trial_files:
                    return False, 'n.a.', f"No trial files found in {submission_dir}"
                
                trial_data = {}
                for trial_file in trial_files:
                    trial_num = os.path.basename(trial_file).replace('trial_', '').replace('.npz', '')
                    data = np.load(trial_file, allow_pickle=True)
                    trial_data[trial_num] = {
                        'solve_rate': float(data['solve_rate']),
                        'hyperparameters': data['hyperparameters'].item() if data['hyperparameters'].ndim == 0 else data['hyperparameters'],
                        'seed': int(data['seed']),
                        'trial_number': int(data['trial_number'])
                    }
                
                if len(trial_data) == 0:
                    return False, 'n.a.', f"No valid trial data found in {submission_dir}"
                
                # Calculate statistics
                solve_rates = [trial['solve_rate'] for trial in trial_data.values()]
                mean_solve_rate = float(np.mean(solve_rates))
                std_solve_rate = float(np.std(solve_rates))
                
                # Find best trial (highest solve rate)
                best_trial_id = max(trial_data.keys(), key=lambda k: trial_data[k]['solve_rate'])
                best_trial = trial_data[best_trial_id]
                best_solve_rate = best_trial['solve_rate']
                best_hparams = best_trial['hyperparameters']
                
                # Save organized evaluation data to shared storage
                eval_data_dir = os.path.join(research_room_path, "storage", "shared", "data")
                file_io_utils.ensure_dir_exists(eval_data_dir)
                eval_file = os.path.join(eval_data_dir, f"eval_{eval_id}.npz")
                
                eval_data = {
                    'all_trials': trial_data,
                    'mean_solve_rate': mean_solve_rate,
                    'best_solve_rate': best_solve_rate,
                    'best_trial_hyperparameters': best_hparams,
                    'individual_scores': solve_rates,
                    'num_trials': len(trial_data),
                    'submission_uuid': submission_uuid
                }
                np.savez(eval_file, **{k: v for k, v in eval_data.items()})
                
                # Clean up submission temp directory
                try:
                    shutil.rmtree(submission_dir)
                except:
                    pass  # Ignore cleanup errors
                
                # Build details message showing individual scores and best hyperparameters
                individual_scores_str = ", ".join([f"{score:.3f}" for score in solve_rates])
                
                hparam_details = []
                for hparam_key, hparam_value in best_hparams.items():
                    if hparam_key not in {'base_seed', '_shared_storage_path', '_submission_uuid'}:  # Skip constants
                        if isinstance(hparam_value, (int, float)) or hasattr(hparam_value, 'dtype'):  # Include numpy scalars
                            hparam_details.append(f"  {hparam_key}: {float(hparam_value):.6f}")
                        else:
                            hparam_details.append(f"  {hparam_key}: {hparam_value}")
                
                details_message = (
                    f"Individual seed scores: [{individual_scores_str}]\n"
                    f"Mean: {mean_solve_rate:.3f}, Std: {std_solve_rate:.3f}\n"
                    f"Best trial: {best_trial_id} ({best_solve_rate:.3f})\n"
                    f"Best trial hyperparameters:\n" + "\n".join(hparam_details)
                )
                
                # Convert to percentage for final score display
                final_score = float(round(mean_solve_rate * 100, 1))
                
                return True, final_score, details_message
            else:
                # Try to find error messages
                if "timeout" in output_str.lower():
                    return False, 'n.a.', "Ray optimization timed out before finding solution"
                elif "OPTIMIZATION FAILED" in output_str:
                    return False, 'n.a.', "Ray optimization failed - no valid solution found"
                elif "error" in output_str.lower():
                    error_lines = [line for line in output_str.split('\n') if 'error' in line.lower()]
                    error_msg = error_lines[0] if error_lines else "Unknown error during Ray optimization"
                    return False, 'n.a.', f"Ray optimization failed: {error_msg}"
                else:
                    return False, 'n.a.', "Could not parse submission UUID from output"
            
        except Exception as e:
            return False, 'n.a.', f"Error evaluating results: {str(e)}"
    
    def get_expected_function_name(self) -> str:
        """Not used for command mode tasks."""
        return "dummy_function"  # Required by base class but not used in command mode
    
    def get_task_description(self) -> str:
        return "Reinforcement Learning on Sokoban with Ray Distributed Training"