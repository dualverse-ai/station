# station_data/rooms/research/evaluators/task_1_evaluator.py
"""
Evaluator for Research Task 1: Multi-Dataset RNA Sequence Modeling with Ray Multi-Dataset Training
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
    Evaluator for Research Task 1: Multi-Dataset RNA Sequence Modeling
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
        For this task, we parse the UUID from output, load the results, and aggregate 3 dataset scores.
        Returns: (success, score, details)
        """
        # Define empty secondary metrics for error cases
        empty_metrics = {
            "APA_R2": None,
            "CRI-Off_Spearman": None,
            "Modif_AUC-ROC": None,
            "CRI-On_Spearman": None,
            "PRS_R2": None,
            "MRL_R2": None,
            "ncRNA_Accuracy": None,
            "Message": ""
        }

        if result is None:
            empty_metrics["Message"] = "No output received from RNA multi-dataset training script"
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

                # Load all dataset trial files (expect 7 datasets)
                expected_datasets = ["APA", "CRI-Off", "Modif", "CRI-On", "PRS", "MRL", "ncRNA"]
                trial_data = {}

                for dataset_name in expected_datasets:
                    trial_file = os.path.join(submission_dir, f"trial_{dataset_name}.npz")
                    if os.path.exists(trial_file):
                        data = np.load(trial_file, allow_pickle=True)
                        trial_data[dataset_name] = {
                            'val_metric': float(data['val_metric']),
                            'task_type': str(data['task_type']),
                            'metric': str(data['metric']),
                            'hyperparameters': data['hyperparameters'].item() if data['hyperparameters'].ndim == 0 else data['hyperparameters'],
                            'seed': int(data['seed']),
                            'dataset_name': str(data['dataset_name'])
                        }

                # Check if we have all required datasets
                missing_datasets = [d for d in expected_datasets if d not in trial_data]
                if missing_datasets:
                    empty_metrics["Message"] = f"Missing results for datasets: {missing_datasets}. Found: {list(trial_data.keys())}"
                    return False, 'n.a.', empty_metrics, (float('-inf'),)

                if len(trial_data) != 7:
                    empty_metrics["Message"] = f"Expected 7 dataset trials, found {len(trial_data)}: {list(trial_data.keys())}"
                    return False, 'n.a.', empty_metrics, (float('-inf'),)

                # Extract individual dataset scores
                apa_score = trial_data["APA"]["val_metric"]
                crioff_score = trial_data["CRI-Off"]["val_metric"]
                modif_score = trial_data["Modif"]["val_metric"]
                crion_score = trial_data["CRI-On"]["val_metric"]
                prs_score = trial_data["PRS"]["val_metric"]
                mrl_score = trial_data["MRL"]["val_metric"]
                ncrna_score = trial_data["ncRNA"]["val_metric"]

                # Validate scores are reasonable (all metrics are "higher is better")
                dataset_scores = {
                    "APA": apa_score,
                    "CRI-Off": crioff_score,
                    "Modif": modif_score,
                    "CRI-On": crion_score,
                    "PRS": prs_score,
                    "MRL": mrl_score,
                    "ncRNA": ncrna_score
                }
                for dataset_name, score in dataset_scores.items():
                    if np.isnan(score) or np.isinf(score):
                        empty_metrics["Message"] = f"Invalid {dataset_name} score: {score}"
                        return False, 'n.a.', empty_metrics, (float('-inf'),)

                # Calculate primary score: average of normalized scores across all 7 datasets
                # All metrics (R², Spearman, AUC-ROC, Accuracy) are "higher is better"
                primary_score = float(np.mean([apa_score, crioff_score, modif_score, crion_score, prs_score, mrl_score, ncrna_score]))

                # Find best performing dataset
                best_dataset = max(dataset_scores.keys(), key=lambda k: dataset_scores[k])
                best_score = dataset_scores[best_dataset]

                # Get hyperparameters from best performing dataset (for display)
                best_hparams = trial_data[best_dataset]['hyperparameters']

                # Save organized evaluation data to shared storage
                eval_data_dir = os.path.join(research_room_path, "storage", "shared", "data")
                file_io_utils.ensure_dir_exists(eval_data_dir)
                eval_file = os.path.join(eval_data_dir, f"eval_{eval_id}.npz")

                eval_data = {
                    'all_trials': trial_data,
                    'primary_score': primary_score,
                    'apa_score': apa_score,
                    'crioff_score': crioff_score,
                    'modif_score': modif_score,
                    'crion_score': crion_score,
                    'prs_score': prs_score,
                    'mrl_score': mrl_score,
                    'ncrna_score': ncrna_score,
                    'best_dataset': best_dataset,
                    'best_score': best_score,
                    'best_hyperparameters': best_hparams,
                    'individual_scores': [apa_score, crioff_score, modif_score, crion_score, prs_score, mrl_score, ncrna_score],
                    'dataset_names': expected_datasets,
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
                individual_scores_str = f"APA: {apa_score:.4f}, CRI-Off: {crioff_score:.4f}, Modif: {modif_score:.4f}, CRI-On: {crion_score:.4f}, PRS: {prs_score:.4f}, MRL: {mrl_score:.4f}, ncRNA: {ncrna_score:.4f}"

                hparam_details = []
                for hparam_key, hparam_value in best_hparams.items():
                    # Skip internal constants and dataset-specific fields
                    if hparam_key not in {'base_seed', '_shared_storage_path', '_submission_uuid', 'dataset', 'd_output', 'task_type', 'metric', 'level', 'max_seq_len'}:
                        if isinstance(hparam_value, (int, float)) or hasattr(hparam_value, 'dtype'):  # Include numpy scalars
                            hparam_details.append(f"  {hparam_key}: {float(hparam_value):.6f}")
                        else:
                            hparam_details.append(f"  {hparam_key}: {hparam_value}")

                # Build secondary metrics dictionary for display
                secondary_metrics = {
                    "APA_R2": float(apa_score),
                    "CRI-Off_Spearman": float(crioff_score),
                    "Modif_AUC-ROC": float(modif_score),
                    "CRI-On_Spearman": float(crion_score),
                    "PRS_R2": float(prs_score),
                    "MRL_R2": float(mrl_score),
                    "ncRNA_Accuracy": float(ncrna_score),
                    "Message": f"Evaluated on 7 datasets successfully"
                }

                # Final score is the primary score (average across datasets)
                final_score = float(primary_score)

                # Return 4-tuple: success, primary_score, secondary_metrics_dict, (primary_score_tuple)
                return True, final_score, secondary_metrics, (final_score,)
            else:
                # Try to find error messages
                if "timeout" in output_str.lower():
                    empty_metrics["Message"] = "RNA multi-dataset training timed out before completion"
                    return False, 'n.a.', empty_metrics, (float('-inf'),)
                elif "OPTIMIZATION FAILED" in output_str:
                    empty_metrics["Message"] = "RNA multi-dataset training failed - no valid solution found"
                    return False, 'n.a.', empty_metrics, (float('-inf'),)
                elif "error" in output_str.lower():
                    error_lines = [line for line in output_str.split('\n') if 'error' in line.lower()]
                    error_msg = error_lines[0] if error_lines else "Unknown error during RNA training"
                    empty_metrics["Message"] = f"RNA multi-dataset training failed: {error_msg}"
                    return False, 'n.a.', empty_metrics, (float('-inf'),)
                else:
                    empty_metrics["Message"] = "Could not parse submission UUID from output"
                    return False, 'n.a.', empty_metrics, (float('-inf'),)

        except Exception as e:
            empty_metrics["Message"] = f"Error evaluating RNA results: {str(e)}"
            return False, 'n.a.', empty_metrics, (float('-inf'),)

    def get_expected_function_name(self) -> str:
        """Not used for command mode tasks."""
        return "dummy_function"  # Required by base class but not used in command mode

    def get_task_description(self) -> str:
        return "Multi-Dataset RNA Sequence Modeling with Ray Multi-Dataset Training"

    def get_secondary_metrics_format(self):
        """Format secondary metrics to 4 decimal places."""
        return {
            "APA_R2": ".4f",
            "CRI-Off_Spearman": ".4f",
            "Modif_AUC-ROC": ".4f",
            "CRI-On_Spearman": ".4f",
            "PRS_R2": ".4f",
            "MRL_R2": ".4f",
            "ncRNA_Accuracy": ".4f"
        }