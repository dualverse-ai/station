# Copyright 2025 DualverseAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# station/eval_research/evaluation_helpers.py
"""
Helper methods for research evaluation operations.
"""

import os
import re
import time
import yaml
import fcntl
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from station import constants
from station import file_io_utils


def truncate_stderr(stderr_output: str) -> str:
    """Truncate stderr to maximum allowed characters."""
    if not stderr_output:
        return stderr_output
    
    max_chars = constants.RESEARCH_EVAL_STDERR_MAX_CHARS
    if len(stderr_output) > max_chars:
        return stderr_output[:max_chars] + f"\n\n[... stderr truncated after {max_chars:,} characters]"
    return stderr_output


def _load_pending_evaluations(self) -> List[Dict[str, Any]]:
    """Load pending research evaluations from YAMLL file"""
    if not file_io_utils.file_exists(self.pending_evaluations_file):
        return []
        
    try:
        pending_evals = file_io_utils.load_yaml_lines(self.pending_evaluations_file)
        
        # Check for evaluations that have exceeded their timeout
        current_time = time.time()
        for eval_data in pending_evals:
            if eval_data.get(constants.EVALUATION_STATUS_KEY) == constants.EVALUATION_STATUS_RUNNING:
                start_timestamp = eval_data.get(constants.EVALUATION_START_TIMESTAMP_KEY, current_time)
                elapsed_time = current_time - start_timestamp
                
                if elapsed_time > self.timeout:
                    # This evaluation has timed out - mark it
                    eval_id = eval_data.get(constants.EVALUATION_ID_KEY)
                    print(f"AutoResearchEvaluator: Evaluation {eval_id} has exceeded timeout ({elapsed_time:.0f}s > {self.timeout}s)")
                    # The evaluation thread will handle the actual termination
        
        # Filter for pending evaluations only (not running ones)
        # Pick up any evaluation that is NOT explicitly marked as running, completed, or failed
        return [eval_data for eval_data in pending_evals 
                if eval_data.get(constants.EVALUATION_STATUS_KEY) not in [constants.EVALUATION_STATUS_RUNNING, "completed", "failed"]]
    except Exception as e:
        self._push_log_event("auto_research_error", {
            "error": f"Failed to load pending research evaluations: {str(e)}",
            "file": self.pending_evaluations_file
        })
        return []


def _should_retry_evaluation(self, eval_entry: Dict[str, Any]) -> bool:
    """Check if an evaluation should be retried based on failure count"""
    retry_count = eval_entry.get("auto_eval_retry_count", 0)
    return retry_count < self.max_retry_attempts


def _is_parallel_evaluation_enabled(self, task_id: str) -> bool:
    """Check if parallel evaluation is enabled for a specific research task"""
    try:
        tasks_path = os.path.join(self.research_room_path, constants.RESEARCH_TASKS_FILENAME)
        if file_io_utils.file_exists(tasks_path):
            tasks_data = file_io_utils.load_yaml(tasks_path)
            if isinstance(tasks_data, list):
                for task in tasks_data:
                    if isinstance(task, dict) and str(task.get(constants.RESEARCH_TASK_ID_KEY)) == task_id:
                        return task.get(constants.RESEARCH_TASK_PARALLEL_EVAL_KEY, False)
        return False
    except Exception as e:
        print(f"AutoResearchEvaluator: Error checking parallel evaluation setting for task {task_id}: {e}")
        return False


def _filter_cuda_banner(self, stdout_text: str) -> str:
    """Filter out CUDA banner from Docker stdout output."""
    if not stdout_text:
        return stdout_text
    
    # Pattern to match the CUDA banner from start to end
    cuda_banner_pattern = r'=+\s*\n\s*==\s*CUDA\s*==\s*\n\s*=+.*?convenience\.\s*\n'
    
    # Remove the CUDA banner with DOTALL flag to match across newlines
    filtered_text = re.sub(cuda_banner_pattern, '', stdout_text, flags=re.DOTALL)
    
    # Clean up any extra newlines left behind
    filtered_text = re.sub(r'\n{3,}', '\n\n', filtered_text)
    
    return filtered_text


def _check_completed_futures(self):
    """Check for completed futures and clean them up"""
    if not self.active_futures:
        return
        
    completed_ids = []
    for eval_id, future in self.active_futures.items():
        if future.done():
            completed_ids.append(eval_id)
            try:
                # Get the result to ensure any exceptions are raised
                future.result()
                print(f"AutoResearchEvaluator: Evaluation {eval_id} thread completed")
            except Exception as e:
                print(f"AutoResearchEvaluator: Evaluation {eval_id} completed with error: {e}")
    
    # Remove completed futures and deallocate their GPUs
    for eval_id in completed_ids:
        del self.active_futures[eval_id]
        # Note: GPU deallocation is handled in the finally block of _evaluate_single_submission
        
    if completed_ids:
        remaining = len(self.active_futures)
        print(f"AutoResearchEvaluator: Cleaned up {len(completed_ids)} completed evaluations, {remaining} still active")


def _save_evaluation_result(self, eval_entry: Dict[str, Any], success: bool, score: float, details: str, logs: str):
    """Save evaluation result to completed evaluations"""
    try:
        file_io_utils.ensure_dir_exists(self.evaluations_dir)
        
        timestamp = int(time.time())
        eval_id = eval_entry.get(constants.EVALUATION_ID_KEY)
        filename = f"evaluation_{eval_id}.yaml"
        filepath = os.path.join(self.evaluations_dir, filename)
        
        result_data = eval_entry.copy()
        # Always use 'n.a.' for failed evaluations
        final_score = score if success else constants.RESEARCH_SCORE_NA
        result_data[constants.EVALUATION_SCORE_KEY] = final_score
        result_data[constants.EVALUATION_LOGS_KEY] = logs
        result_data["evaluation_timestamp"] = timestamp
        
        # Debug: Check score type before saving
        print(f"AutoResearchEvaluator: Saving evaluation {eval_id} - Success: {success}, Score: {final_score} (type: {type(final_score)})")
        result_data["evaluation_datetime"] = datetime.fromtimestamp(timestamp).isoformat()
        result_data["evaluation_details"] = details
        result_data["evaluation_success"] = success
        
        file_io_utils.save_yaml(result_data, filepath)
        print(f"AutoResearchEvaluator: Saved evaluation result to {filepath}")
        
    except Exception as e:
        print(f"AutoResearchEvaluator: Failed to save evaluation result: {e}")


def _save_failed_evaluation(self, eval_entry: Dict[str, Any], error: str, logs: str):
    """Save failed evaluation result"""
    # For import errors, try to extract more specific information from logs
    if "Import failed:" in error and "IMPORT_ERROR:" in logs:
        # Already have detailed import error
        details = error
    elif "Failed to import" in error and "IMPORT_ERROR:" in logs:
        # Extract the actual import error from logs
        for line in logs.split('\n'):
            if line.strip().startswith("IMPORT_ERROR:"):
                # Extract the specific error message
                import_msg = line.strip()[13:].strip()  # Remove "IMPORT_ERROR:" prefix
                details = f"Import failed: {import_msg}"
                break
        else:
            details = f"Execution failed: {error}"
    else:
        details = f"Execution failed: {error}"
    
    self._save_evaluation_result(eval_entry, False, constants.RESEARCH_SCORE_NA, details, logs)


def _remove_from_pending_evaluations(self, completed_eval_entry: Dict[str, Any]) -> bool:
    """Remove completed evaluation from pending evaluations file. Returns True if removed, False if not found or failed."""
    max_retries = 5
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            if not file_io_utils.file_exists(self.pending_evaluations_file):
                return False  # File doesn't exist, item not found
            
            # Use file locking to prevent race conditions
            with open(self.pending_evaluations_file, 'r+', encoding='utf-8') as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    # Load all evaluations
                    f.seek(0)
                    content = f.read()
                    if not content.strip():
                        return False  # Empty file, item not found
                    
                    all_evaluations = file_io_utils.load_yaml_lines(self.pending_evaluations_file)
                    
                    # Remove the completed evaluation entirely
                    remaining_evaluations = []
                    completed_id = completed_eval_entry.get(constants.EVALUATION_ID_KEY)
                    found = False
                    
                    for eval_data in all_evaluations:
                        eval_id = eval_data.get(constants.EVALUATION_ID_KEY)
                        if eval_id != completed_id:
                            # Keep evaluations that are NOT the completed one
                            remaining_evaluations.append(eval_data)
                        else:
                            found = True
                    
                    if not found:
                        return False  # Item not found in file
                    
                    # Write back to file
                    f.seek(0)
                    f.truncate()
                    for eval_data in remaining_evaluations:
                        f.write("---\n")
                        yaml.dump(eval_data, f, default_flow_style=False, allow_unicode=True)
                    
                    print(f"AutoResearchEvaluator: Removed completed evaluation {completed_id} from pending file")
                    return True  # Successfully removed
                    
                except (IOError, OSError) as lock_error:
                    # File is locked, wait and retry
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        raise lock_error
                        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"AutoResearchEvaluator: Retry {attempt + 1}/{max_retries} failed to remove evaluation: {e}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                print(f"AutoResearchEvaluator: Failed to remove completed evaluation after {max_retries} attempts: {e}")
                return False  # Failed after all retries


def _update_evaluation_status_to_running(self, eval_entry: Dict[str, Any], start_timestamp: float, start_tick: int) -> None:
    """Update evaluation status to running with multi-tick tracking information"""
    try:
        # Load all pending evaluations
        pending_evals = []
        if file_io_utils.file_exists(self.pending_evaluations_file):
            pending_evals = file_io_utils.load_yaml_lines(self.pending_evaluations_file)
        
        # Find and update the specific evaluation
        updated = False
        for i, eval_data in enumerate(pending_evals):
            if eval_data.get(constants.EVALUATION_ID_KEY) == eval_entry.get(constants.EVALUATION_ID_KEY):
                # Update with running status and tick tracking
                eval_data[constants.EVALUATION_STATUS_KEY] = constants.EVALUATION_STATUS_RUNNING
                eval_data[constants.EVALUATION_START_TIMESTAMP_KEY] = start_timestamp
                eval_data[constants.EVALUATION_START_TICK_KEY] = start_tick
                eval_data[constants.EVALUATION_MAX_ALLOWED_TICKS_KEY] = constants.RESEARCH_EVAL_MAX_TICK
                pending_evals[i] = eval_data
                updated = True
                break
        
        if updated:
            # Write back the updated evaluations by rewriting the entire file
            # First, clear the file
            with open(self.pending_evaluations_file, 'w') as f:
                pass  # Create empty file
            
            # Then append each evaluation
            for eval_data in pending_evals:
                file_io_utils.append_yaml_line(eval_data, self.pending_evaluations_file)
            
            print(f"AutoResearchEvaluator: Updated evaluation {eval_entry.get(constants.EVALUATION_ID_KEY)} to running status (submitted at tick {start_tick})")
        
    except Exception as e:
        print(f"AutoResearchEvaluator: Failed to update evaluation status to running: {e}")


def _update_evaluation_failure_count(self, eval_entry: Dict[str, Any]):
    """Update failure count for an evaluation and save back to pending file"""
    try:
        eval_id = eval_entry.get(constants.EVALUATION_ID_KEY)
        
        # Load all pending evaluations
        all_evaluations = file_io_utils.load_yaml_lines(self.pending_evaluations_file)
        
        # Find and update the specific evaluation entry
        for i, pending_eval in enumerate(all_evaluations):
            if pending_eval.get(constants.EVALUATION_ID_KEY) == eval_id:
                retry_count = pending_eval.get("auto_eval_retry_count", 0) + 1
                all_evaluations[i]["auto_eval_retry_count"] = retry_count
                all_evaluations[i]["last_auto_eval_failure"] = time.time()
                
                if retry_count >= self.max_retry_attempts:
                    self._push_log_event("auto_research_error", {
                        "eval_id": eval_id,
                        "status": "max_retries_exceeded",
                        "retry_count": retry_count,
                        "message": f"Research evaluation {eval_id} exceeded {self.max_retry_attempts} retry attempts"
                    })
                    print(f"AutoResearchEvaluator: Evaluation {eval_id} exceeded max retries ({self.max_retry_attempts})")
                    
                    # Mark as failed
                    all_evaluations[i][constants.EVALUATION_SCORE_KEY] = constants.RESEARCH_SCORE_NA
                
                break
        
        # Save updated evaluations back to file
        with open(self.pending_evaluations_file, 'w', encoding='utf-8') as f:
            for eval_data in all_evaluations:
                f.write("---\n")
                yaml.dump(eval_data, f, default_flow_style=False, allow_unicode=True)
                
    except Exception as e:
        print(f"AutoResearchEvaluator: Failed to update evaluation failure count: {e}")


def find_conda_python(conda_env_name: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Find the Python executable for a conda environment dynamically.
    
    Args:
        conda_env_name: Name of the conda environment
        env: Optional environment variables dict (defaults to os.environ.copy())
        
    Returns:
        Path to Python executable, or None if not found
    """
    import subprocess
    
    if env is None:
        env = os.environ.copy()
    
    python_path = None
    
    # First, try to get the conda executable path from the environment variable set by deploy.sh
    conda_executable = env.get('CONDA_BIN_PATH')
    if conda_executable and os.path.exists(conda_executable) and os.access(conda_executable, os.X_OK):
        print(f"Using CONDA_BIN_PATH from environment: {conda_executable}")
    else:
        # Fallback to trying to find it in PATH or common locations if not set or invalid
        print("CONDA_BIN_PATH not found or invalid, attempting to locate conda executable...")
        conda_exec_candidates = [
            os.path.join(os.path.expanduser('~root'), 'miniconda3', 'bin', 'conda'),
            os.path.join(os.path.expanduser('~'), 'miniconda3', 'bin', 'conda'),
            os.path.join(os.path.expanduser('~'), 'miniforge', 'bin', 'conda'),
            '/opt/conda/bin/conda',
            '/usr/local/bin/conda',
            '/usr/bin/conda',
        ]
        for candidate in conda_exec_candidates:
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                conda_executable = candidate
                print(f"Found conda executable at: {conda_executable}")
                break
        
        if not conda_executable:
            # Final fallback: rely on PATH (less reliable in sudo contexts)
            try:
                result = subprocess.run(["which", "conda"], capture_output=True, text=True, env=env, check=False)
                if result.returncode == 0 and result.stdout.strip():
                    conda_executable = result.stdout.strip()
                    print(f"Found conda executable using 'which': {conda_executable}")
            except FileNotFoundError:
                pass # 'which' command not found

    if not conda_executable:
        print(f"ERROR: 'conda' executable not found. Please ensure it's installed and accessible, or set CONDA_BIN_PATH.")
        return None

    # Now use the found conda executable to get the python path for the specific environment
    command = [conda_executable, "run", "-n", conda_env_name, "which", "python"]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=env, check=True)
        python_path = result.stdout.strip()
        if python_path and os.path.exists(python_path) and os.access(python_path, os.X_OK):
            print(f"Found Python for conda env '{conda_env_name}' at {python_path}")
            return python_path
        else:
            print(f"ERROR: Python executable not found or not executable in conda env '{conda_env_name}': {python_path}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to get Python path for conda env '{conda_env_name}'. Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while finding conda python: {e}")
        return None


def setup_conda_env(conda_env_name: str, env: Dict[str, str]) -> bool:
    """
    Configure environment variables for a conda environment.
    
    Args:
        conda_env_name: Name of the conda environment
        env: Environment variables dict to update
        
    Returns:
        True if environment was configured successfully
    """
    import subprocess
    
    try:
        # Find conda executable path
        conda_executable = env.get('CONDA_BIN_PATH')
        if not conda_executable or not os.path.exists(conda_executable):
            try:
                result = subprocess.run(["which", "conda"], capture_output=True, text=True, check=True)
                conda_executable = result.stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"Error: 'conda' executable not found and CONDA_BIN_PATH is not set or invalid.")
                return False

        # Use conda to find the environment prefix
        result = subprocess.run(
            [conda_executable, 'run', '-n', conda_env_name, 'python', '-c', 'import sys; print(sys.prefix)'],
            capture_output=True,
            text=True,
            env=env
        )
        if result.returncode == 0:
            conda_prefix = result.stdout.strip()
            conda_bin_path = os.path.join(conda_prefix, 'bin')
            
            # Update PATH to prioritize the conda environment
            current_path = env.get('PATH', '')
            env['PATH'] = f'{conda_bin_path}:{current_path}'
            
            # Set CONDA environment variables
            env['CONDA_DEFAULT_ENV'] = conda_env_name
            env['CONDA_PREFIX'] = conda_prefix
            
            print(f"Configured conda environment '{conda_env_name}' at {conda_prefix}")
            return True
        else:
            print(f"Could not find conda environment '{conda_env_name}': {result.stderr}")
            return False
    except Exception as e:
        print(f"Error configuring conda environment: {e}")
        return False