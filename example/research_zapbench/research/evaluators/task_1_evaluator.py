# station_data/rooms/research/evaluators/task_1_evaluator.py
"""
Evaluator for Research Task 1: Neural Activity Forecasting on ZAPBench with Ray Distributed Training
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
    Evaluator for Research Task 1: Neural Activity Forecasting on ZAPBench
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

    def evaluate_submission(self, result: any = None, eval_id: str = None, author: str = None) -> Tuple[bool, any, dict, tuple]:
        """
        This method is called after code execution completes.
        For this task, we parse the UUID from output, load the results, and aggregate 3 seed scores.
        Returns: (success, score, secondary_metrics, score_tuple)
        """
        # Define empty secondary metrics for error cases
        empty_metrics = {
            "Val_MAE_1": None,
            "Val_MAE_4": None,
            "Val_MAE_8": None,
            "Val_MAE_16": None,
            "Val_MAE_32": None,
            "Message": ""
        }

        if result is None:
            empty_metrics["Message"] = "No output received from Ray optimization script"
            return False, 'n.a.', empty_metrics, (float('-inf'),)

        try:
            # Parse the output
            output_str = str(result)

            # Check if this is test mode - if so, don't attempt UUID loading
            if "=== Test Mode Detected ===" in output_str:
                empty_metrics["Message"] = "Test mode - no scoring"
                return False, 'n.a.', empty_metrics, (float('-inf'),)

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
                    empty_metrics["Message"] = f"Submission directory not found: {submission_dir}"
                    return False, 'n.a.', empty_metrics, (float('-inf'),)

                # Load all trial files (expect 3 seeds)
                trial_files = glob.glob(os.path.join(submission_dir, "trial_*.npz"))
                if not trial_files:
                    empty_metrics["Message"] = f"No trial files found in {submission_dir}"
                    return False, 'n.a.', empty_metrics, (float('-inf'),)

                trial_data = {}
                all_step_maes = {1: [], 4: [], 8: [], 16: [], 32: []}  # Collect step MAEs from all seeds

                for trial_file in trial_files:
                    trial_num = os.path.basename(trial_file).replace('trial_', '').replace('.npz', '')
                    data = np.load(trial_file, allow_pickle=True)
                    trial_data[trial_num] = {
                        'val_mae': float(data['val_mae']),
                        'hyperparameters': data['hyperparameters'].item() if data['hyperparameters'].ndim == 0 else data['hyperparameters'],
                        'seed': int(data['seed']),
                        'trial_number': int(data['trial_number'])
                    }

                    # Extract step MAEs if available
                    if 'step_maes' in data:
                        step_maes = data['step_maes'].item() if hasattr(data['step_maes'], 'item') else data['step_maes']
                        trial_data[trial_num]['step_maes'] = step_maes
                        # Collect for averaging
                        for horizon in [1, 4, 8, 16, 32]:
                            if horizon in step_maes:
                                all_step_maes[horizon].append(float(step_maes[horizon]))

                if len(trial_data) == 0:
                    empty_metrics["Message"] = f"No valid trial data found in {submission_dir}"
                    return False, 'n.a.', empty_metrics, (float('-inf'),)

                # Calculate statistics
                val_maes = [trial['val_mae'] for trial in trial_data.values()]
                mean_val_mae = float(np.mean(val_maes))
                std_val_mae = float(np.std(val_maes))

                # Find best trial (lowest MAE)
                best_trial_id = min(trial_data.keys(), key=lambda k: trial_data[k]['val_mae'])
                best_trial = trial_data[best_trial_id]
                best_val_mae = best_trial['val_mae']
                best_hparams = best_trial['hyperparameters']

                # Calculate mean step MAEs for secondary metrics
                mean_step_maes = {}
                for horizon in [1, 4, 8, 16, 32]:
                    if all_step_maes[horizon]:
                        mean_step_maes[horizon] = float(np.mean(all_step_maes[horizon]))
                    else:
                        mean_step_maes[horizon] = None

                # Save organized evaluation data to shared storage
                eval_data_dir = os.path.join(research_room_path, "storage", "shared", "data")
                file_io_utils.ensure_dir_exists(eval_data_dir)
                eval_file = os.path.join(eval_data_dir, f"eval_{eval_id}.npz")

                eval_data = {
                    'all_trials': trial_data,
                    'mean_val_mae': mean_val_mae,
                    'best_val_mae': best_val_mae,
                    'best_trial_hyperparameters': best_hparams,
                    'individual_scores': val_maes,
                    'num_trials': len(trial_data),
                    'submission_uuid': submission_uuid,
                    'mean_step_maes': mean_step_maes,
                    'all_step_maes': all_step_maes
                }
                np.savez(eval_file, **{k: v for k, v in eval_data.items()})

                # Clean up submission temp directory
                try:
                    shutil.rmtree(submission_dir)
                except:
                    pass  # Ignore cleanup errors

                # Build details message showing individual scores and best hyperparameters
                individual_scores_str = ", ".join([f"{score:.6f}" for score in val_maes])

                hparam_details = []
                for hparam_key, hparam_value in best_hparams.items():
                    if hparam_key not in {'base_seed', '_shared_storage_path', '_submission_uuid'}:  # Skip constants
                        if isinstance(hparam_value, (int, float)) or hasattr(hparam_value, 'dtype'):  # Include numpy scalars
                            hparam_details.append(f"  {hparam_key}: {float(hparam_value):.6f}")
                        else:
                            hparam_details.append(f"  {hparam_key}: {hparam_value}")

                # Build secondary metrics dictionary
                secondary_metrics = {
                    "Val_MAE_1": mean_step_maes.get(1),
                    "Val_MAE_4": mean_step_maes.get(4),
                    "Val_MAE_8": mean_step_maes.get(8),
                    "Val_MAE_16": mean_step_maes.get(16),
                    "Val_MAE_32": mean_step_maes.get(32),
                    "Message": f"Mean over {len(trial_data)} seeds | Step MAEs computed on validation set"
                }

                # Final score is negative mean MAE (higher is better)
                final_score = float(-mean_val_mae)

                # Return 4-tuple: success, primary_score, secondary_metrics_dict, (primary_score_tuple)
                return True, final_score, secondary_metrics, (final_score,)
            else:
                # Try to find error messages
                if "timeout" in output_str.lower():
                    empty_metrics["Message"] = "Ray optimization timed out before finding solution"
                    return False, 'n.a.', empty_metrics, (float('-inf'),)
                elif "OPTIMIZATION FAILED" in output_str:
                    empty_metrics["Message"] = "Ray optimization failed - no valid solution found"
                    return False, 'n.a.', empty_metrics, (float('-inf'),)
                elif "error" in output_str.lower():
                    error_lines = [line for line in output_str.split('\n') if 'error' in line.lower()]
                    error_msg = error_lines[0] if error_lines else "Unknown error during Ray optimization"
                    empty_metrics["Message"] = f"Ray optimization failed: {error_msg}"
                    return False, 'n.a.', empty_metrics, (float('-inf'),)
                else:
                    empty_metrics["Message"] = "Could not parse submission UUID from output"
                    return False, 'n.a.', empty_metrics, (float('-inf'),)

        except Exception as e:
            empty_metrics["Message"] = f"Error evaluating results: {str(e)}"
            return False, 'n.a.', empty_metrics, (float('-inf'),)

    def get_expected_function_name(self) -> str:
        """Not used for command mode tasks."""
        return "dummy_function"  # Required by base class but not used in command mode

    def get_task_description(self) -> str:
        return "Neural Activity Forecasting on ZAPBench with Ray Distributed Training"

    def get_secondary_metrics_format(self):
        """Format secondary metrics to 4 decimal places."""
        return {
            "Val_MAE_1": ".4f",
            "Val_MAE_4": ".4f",
            "Val_MAE_8": ".4f",
            "Val_MAE_16": ".4f",
            "Val_MAE_32": ".4f"
        }