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

"""
Evaluation Manager for research evaluations.
Manages all file operations for research evaluation JSON files.
Provides atomic operations, state tracking, and notification hooks.
"""

import json
import os
import time
import threading
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Tuple
from contextlib import contextmanager

# The filelock library provides robust, cross-platform file locking.
import filelock

from station import constants
from station import file_io_utils

# ==============================================================================
# Standalone Helper & Public Functions
# ==============================================================================

def _get_default_evaluations_dir() -> str:
    """Get the default evaluations directory path."""
    # This helper is restored to allow standalone functions to work without a path.
    from station import constants
    return os.path.join(
        constants.BASE_STATION_DATA_PATH,
        constants.ROOMS_DIR_NAME,
        constants.SHORT_ROOM_NAME_RESEARCH,
        constants.RESEARCH_EVALUATIONS_SUBDIR_NAME
    )

def _load_evaluation_json(eval_id: str, evaluations_dir: str = None) -> Optional[Dict[str, Any]]:
    """Load evaluation JSON file without locks (read-only operation)."""
    # Restored logic to fetch default directory if none is provided.
    if evaluations_dir is None:
        evaluations_dir = _get_default_evaluations_dir()

    eval_path = os.path.join(evaluations_dir, f"evaluation_{eval_id}.json")
    try:
        with open(eval_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        print(f"EvaluationManager: CRITICAL - Corrupted JSON file detected for evaluation {eval_id}: {e}")
        return None
    except Exception as e:
        print(f"EvaluationManager: Unexpected error reading evaluation {eval_id}: {e}")
        return None


def get_evaluation_display_info(eval_id: str, evaluations_dir: str = None) -> Optional[Dict[str, Any]]:
    """
    Get simplified display information for room display. This is a public, importable function.
    """
    # The 'evaluations_dir' argument is now correctly optional.
    eval_data = _load_evaluation_json(eval_id, evaluations_dir)
    if not eval_data:
        return None

    notification = eval_data.get("notification", {})
    if notification.get("sent"):
        version_notified = notification.get("version_notified", "original")
        result_source = eval_data["versions"].get(version_notified) if version_notified != "original" else eval_data["original_submission"]
        display_score = result_source["evaluation_result"].get("score", "n.a.")
        sort_key = result_source["evaluation_result"].get("sort_key")
        evaluation_details = result_source["evaluation_result"].get(constants.EVALUATION_DETAILS_KEY, "")
    else:
        display_score = "pending"
        sort_key = None
        evaluation_details = ""  # No secondary metrics when pending

    result = {
        constants.EVALUATION_ID_KEY: eval_data["id"],
        constants.EVALUATION_AUTHOR_KEY: eval_data["author"],
        constants.EVALUATION_TITLE_KEY: eval_data["title"],
        constants.EVALUATION_TAGS_KEY: eval_data.get("tags", []),  # Default to empty list for backward compatibility
        constants.EVALUATION_ABSTRACT_KEY: eval_data.get("abstract", ""),  # Default to empty string for backward compatibility
        constants.EVALUATION_SCORE_KEY: display_score,
        constants.EVALUATION_SUBMITTED_TICK_KEY: eval_data["submitted_tick"],
        constants.EVALUATION_RESEARCH_TASK_ID_KEY: eval_data["research_task_id"],
        constants.EVALUATION_DETAILS_KEY: evaluation_details  # Include for secondary metrics display
    }
    
    # Include sort_key if available
    if sort_key is not None:
        result["sort_key"] = sort_key
        
    return result


def _format_secondary_metrics_for_display(evaluation_details: Any) -> Tuple[str, str]:
    """
    Extract and format secondary metrics from evaluation details.
    
    Args:
        evaluation_details: Either string or dict with secondary metrics
        
    Returns:
        Tuple of (message_string, formatted_metrics_string)
    """
    # If details is a string, no secondary metrics
    if isinstance(evaluation_details, str):
        return evaluation_details, ""
    
    # If details is not a dict, treat as string
    if not isinstance(evaluation_details, dict):
        return str(evaluation_details), ""
    
    # Extract message and format metrics
    message = evaluation_details.get("Message", "")
    
    # Build formatted metrics string
    formatted_parts = []
    for key, value in evaluation_details.items():
        if key == "Message":
            continue
            
        # Value should be (formatted_value, raw_value) tuple or list (JSON converts tuples to lists)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            formatted_value, raw_value = value
            formatted_parts.append(f"**{key}:** {formatted_value}")
        else:
            # Fallback for unexpected format
            formatted_parts.append(f"**{key}:** {value}")
    
    if formatted_parts:
        metrics_string = "\n" + "\n".join(formatted_parts)
    else:
        metrics_string = ""
    
    return str(message), metrics_string


def get_evaluation_review_info(eval_id: str, evaluations_dir: str = None) -> Optional[Dict[str, str]]:
    """
    Get review information for an evaluation. This is a public, importable function.
    """
    # The 'evaluations_dir' argument is now correctly optional.
    eval_data = _load_evaluation_json(eval_id, evaluations_dir)
    if not eval_data:
        return None

    notification = eval_data.get("notification", {})
    if not notification.get("sent"):
        return {
            "status": "pending",
            "message": f"Evaluation '{eval_id}' is still pending. Please try again later."
        }

    version_notified = notification["version_notified"]
    result_source = eval_data["versions"].get(version_notified) if version_notified != "original" else eval_data["original_submission"]
    result = result_source["evaluation_result"]
    code = result_source["content"]
    
    score = result.get("score", "n.a.")
    logs = result.get("logs", "")

    max_chars = constants.RESEARCH_EVAL_LOG_MAX_CHARS

    if len(logs) > max_chars:
        logs = logs[:max_chars] + f"\n\n[... truncated after {max_chars:,} characters]"

    # Check if the evaluation failed and include error summary
    error_summary = ""
    if not result.get("success", False) and result.get("error"):
        error_summary = f"**Error Summary:** {result['error']}\n\n"
    
    # Format tags and abstract for display
    tags = eval_data.get("tags", [])
    abstract = eval_data.get("abstract", "")
    tags_display = ", ".join(tags) if tags else "—"
    
    # Get evaluation details and format secondary metrics
    evaluation_details = result.get(constants.EVALUATION_DETAILS_KEY, "")
    details_message, secondary_metrics_string = _format_secondary_metrics_for_display(evaluation_details)
    
    # Build message conditionally based on RESEARCH_NO_SCORE setting
    message = (
        f"**Research Submission Review**\n\n"
        f"**Title:** {eval_data['title']}\n"
        f"**ID:** {eval_id}\n"
        f"**Tags:** {tags_display}\n"
        f"**Abstract:** {abstract}"
    )
    
    # Only include score and evaluation details if scoring is enabled
    if not constants.RESEARCH_NO_SCORE:
        message += f"\n**Score:** {score}"
        if secondary_metrics_string:
            message += secondary_metrics_string
        message += f"\n**Evaluation Details:** {details_message}"
    
    message += f"\n\n{error_summary}**Submission Code:**\n```python\n{code}\n```\n\n**Execution Log:**\n```\n{logs}\n```"

    return {"status": "completed", "message": message}

# ==============================================================================
# Evaluation Manager Class
# ==============================================================================

class EvaluationManager:
    """
    Manages all file operations for research evaluation JSON files.
    Provides atomic operations, state tracking, and notification hooks.
    """
    
    def __init__(self, evaluations_dir: str):
        self.evaluations_dir = evaluations_dir
        self._notification_callback = None
        self._running_evaluations = {}
        self._running_lock = threading.Lock()
        self._tick_limit_logged = set()
        
        self.top_submission = None
        self._initialize_top_submission()
        
        os.makedirs(self.evaluations_dir, exist_ok=True)
        
    def set_notification_callback(self, callback: Callable[[str, str], None]):
        self._notification_callback = callback
    
    @contextmanager
    def _file_lock(self, filepath: str):
        lock_path = filepath + '.lock'
        lock = filelock.FileLock(lock_path, timeout=60)
        try:
            with lock:
                yield
        except filelock.Timeout:
            print(f"EvaluationManager: Could not acquire lock for {filepath} within the timeout period.")
            raise Exception(f"Failed to acquire lock for {filepath}")
    
    def _get_eval_path(self, eval_id: str) -> str:
        return os.path.join(self.evaluations_dir, f"evaluation_{eval_id}.json")
    
    def _load_evaluation(self, eval_id: str) -> Optional[Dict[str, Any]]:
        return _load_evaluation_json(eval_id, self.evaluations_dir)

    def _save_evaluation(self, eval_id: str, data: Dict[str, Any]):
        eval_path = self._get_eval_path(eval_id)
        temp_path = f"{eval_path}.{uuid.uuid4()}.tmp"
        
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.rename(temp_path, eval_path)
        except Exception as e:
            print(f"EvaluationManager: Failed to save evaluation {eval_id}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _check_all_complete(self, eval_data: Dict[str, Any]) -> bool:
        if eval_data["original_submission"]["evaluation_result"]["status"] == "pending":
            return False
        for version_data in eval_data["versions"].values():
            if version_data["evaluation_result"]["status"] == "pending":
                return False
        return True

    def _generate_notification_message(self, eval_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generate notification message based on the three principles.
        This version is robust against a `None` value in the 'error' field.
        """
        title = eval_data.get("title", "N/A")
        eval_id = eval_data.get("id", "N/A")

        claude_session = (eval_data.get("claude_sessions") or [None])[-1]

        # Default to original submission
        result = eval_data.get("original_submission", {}).get("evaluation_result", {})
        version_to_notify = "original"
        claude_report = None
        is_timeout_case = False

        # Principle 1: If success.md exists AND (debugged version score ≠ n.a. or debugged version timeout or test-mode success)
        if claude_session and claude_session.get("success"):
            latest_version = eval_data.get("current_state", {}).get("latest_version")
            if latest_version and latest_version != "original":
                version_result = eval_data.get("versions", {}).get(latest_version, {}).get("evaluation_result", {})
                score = version_result.get("score", "n.a.")
                error_msg = version_result.get("error") # Can be None
                logs = version_result.get("logs", "")

                # Check for different success conditions
                is_timeout_case = "timed out after" in (error_msg or "").lower()

                # Test mode is considered successful if code executed without runtime errors
                is_test_mode_success = (
                    (error_msg or "").startswith("Evaluation failed: Test mode - no scoring") and
                    "Traceback" not in logs
                )

                if score != "n.a." or is_timeout_case or is_test_mode_success:
                    result = version_result
                    version_to_notify = latest_version
                    claude_report = claude_session.get("final_report")
                else:
                    claude_report = None
                    
        # Principle 2: If fail.md exists
        elif claude_session and not claude_session.get("success"):
            claude_report = claude_session.get("final_report")

        # Safely get result details
        success = result.get("success", False)
        final_score = result.get("score", "n.a.")
        logs = result.get("logs", "")

        max_chars = constants.RESEARCH_EVAL_LOG_MAX_CHARS
        if len(logs) > max_chars:
            logs = logs[:max_chars] + f"\n\n[... truncated after {max_chars:,} characters]"

        # --- Message Generation ---

        claude_report_str = claude_report or ""

        if is_timeout_case:
            error = result.get("error", "Execution timed out.")
            debugged_code = eval_data.get("versions", {}).get(version_to_notify, {}).get("content", "")
            message = (f"Your research submission '{title}' (ID: {eval_id}) failed after automatic debugging.\n\n"
                       f"The debugger's changes were applied, but the new version failed to complete within the time limit.\n\n"
                       f"**Debugger Report:** {claude_report_str}\n\n"
                       f"**Final Error (version {version_to_notify}):** {error}\n\n"
                       f"**Debugged Code:**\n```python\n{debugged_code}\n```\n\n"
                       f"**Full Execution Log:**\n```\n{logs}\n```")
        elif claude_report and version_to_notify != "original":
            details = result.get(constants.EVALUATION_DETAILS_KEY, "")
            debugged_code = eval_data.get("versions", {}).get(version_to_notify, {}).get("content", "")
            # Format secondary metrics for notification
            details_message, secondary_metrics_string = _format_secondary_metrics_for_display(details)
            
            message = f"Your research submission '{title}' (ID: {eval_id}) succeeded after automatic debugging.\n\n"
            if not constants.RESEARCH_NO_SCORE:
                message += f"**Score:** {final_score}"
                if secondary_metrics_string:
                    message += secondary_metrics_string
                message += "\n"
            message += (f"**Evaluation Details:** {details_message}\n\n"
                       f"**Debugger Report:** {claude_report_str}\n\n"
                       f"**Important Note:** The debugger cannot modify files in your lineage storage. "
                       f"If your code relies on lineage files, you may need to update them manually.\n\n"
                       f"**Debugged Code (version {version_to_notify}):**\n```python\n{debugged_code}\n```\n\n"
                       f"**Full Execution Log:**\n```\n{logs}\n```")
        elif claude_report:
            error = result.get("error", "Unknown error")
            message = (f"Your research submission '{title}' (ID: {eval_id}) failed.\n\n"
                       f"The debugger attempted to fix it but encountered issues:\n\n"
                       f"**Debugger Report:** {claude_report_str}\n\n"
                       f"**Original Error:** {error}\n"
                       f"**Full Execution Log:**\n```\n{logs}\n```")
        else:
            if success:
                details = result.get(constants.EVALUATION_DETAILS_KEY, "")
                # Format secondary metrics for notification
                details_message, secondary_metrics_string = _format_secondary_metrics_for_display(details)
                
                # Build message conditionally based on RESEARCH_NO_SCORE setting
                message = f"Your research submission '{title}' (ID: {eval_id}) has been evaluated.\n\n"
                if not constants.RESEARCH_NO_SCORE:
                    message += f"**Score:** {final_score}"
                    if secondary_metrics_string:
                        message += secondary_metrics_string
                    message += "\n"
                message += f"**Evaluation Details:** {details_message}\n\n"
                message += f"**Full Execution Log:**\n```\n{logs}\n```"
            else:
                error = result.get("error", "Unknown error")
                error_guidance = ""
                if "Import failed:" in (error or "") or "No module named" in (error or ""):
                    error_guidance = ("\n\n**Import Error Diagnosis:**\n"
                                      "- Check that imported files exist in your storage\n"
                                      "- Verify file paths in sys.path.append() statements\n"
                                      "- Use `/execute_action{storage list}` to check your files\n")
                message = (f"Your research submission '{title}' (ID: {eval_id}) evaluation failed.\n\n"
                           f"**Error Summary:** {error}\n"
                           f"{error_guidance}\n"
                           f"**Full Execution Log:**\n```\n{logs}\n```")
        return message, version_to_notify

    def _check_and_notify(self, eval_data: Dict[str, Any]) -> bool:
        if eval_data["notification"]["sent"] or eval_data["current_state"]["claude_active"] or not eval_data["current_state"]["all_evaluations_complete"]:
            return False
        
        notification_message, version_notified = self._generate_notification_message(eval_data)
        
        if self._notification_callback:
            try:
                self._notification_callback(eval_data["author"], notification_message)
                eval_data["notification"].update({
                    "sent": True, "sent_timestamp": time.time(),
                    "version_notified": version_notified, "message": notification_message
                })
                with self._running_lock:
                    eval_id = eval_data["id"]
                    if eval_id in self._running_evaluations:
                        del self._running_evaluations[eval_id]
                        print(f"EvaluationManager: Removed evaluation {eval_id} from running (notification sent)")
                    self._tick_limit_logged.discard(eval_id)
                self._update_top_submission_if_needed(eval_data)
                return True
            except Exception as e:
                print(f"EvaluationManager: Error in notification callback: {e}")
                return False
        return False

    # The rest of the class methods remain unchanged. They are correct.
    def create_evaluation(self, eval_id: str, author: str, task_id: str, 
                          title: str, content: str, tick: int, 
                          no_debugger: bool = False, version: int = None,
                          tags: List[str] = None, abstract: str = "",
                          cpu_only: bool = False) -> Dict[str, Any]:
        with self._file_lock(self._get_eval_path(eval_id)):
            if version:
                eval_data = self._load_evaluation(eval_id)
                if not eval_data:
                    print(f"EvaluationManager: Base evaluation {eval_id} not found for version {version}")
                    return None
                
                version_key = f"v{version}"
                eval_data["versions"][version_key] = {
                    "content": content, "created_timestamp": time.time(),
                    "evaluation_result": {"status": "pending"}
                }
                if eval_data.get("claude_sessions"):
                    eval_data["claude_sessions"][-1]["versions_created"].append(version_key)
                
                eval_data["current_state"].update({
                    "latest_version": version_key, "latest_status": "pending",
                    "latest_score": "pending", "all_evaluations_complete": self._check_all_complete(eval_data)
                })
            else:
                eval_data = {
                    "id": eval_id, "author": author, "research_task_id": task_id,
                    "title": title, "tags": tags or [], "abstract": abstract,
                    "submitted_tick": tick, "no_debugger": no_debugger, "cpu_only": cpu_only,
                    "original_submission": {
                        "content": content, "submitted_timestamp": time.time(),
                        "evaluation_result": {"status": "pending"}
                    },
                    "claude_sessions": [], "versions": {},
                    "current_state": {
                        "latest_version": "original", "latest_score": "pending",
                        "latest_status": "pending", "claude_active": False,
                        "all_evaluations_complete": False
                    },
                    "notification": {"sent": False, "sent_timestamp": None, "version_notified": None}
                }
            self._save_evaluation(eval_id, eval_data)
        
        with self._running_lock:
            self._running_evaluations[eval_id] = {"start_tick": tick}
            if version:
                print(f"EvaluationManager: Started tracking evaluation {eval_id} version {version} at tick {tick}")
            else:
                print(f"EvaluationManager: Started tracking evaluation {eval_id} at tick {tick}")
        return eval_data

    def update_result(self, eval_id: str, success: bool, score: Any,
                      error: str = None, logs: str = "", details: str = "",
                      version: str = None, sort_key: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
        with self._file_lock(self._get_eval_path(eval_id)):
            eval_data = self._load_evaluation(eval_id)
            if not eval_data:
                return None
            
            target = eval_data["versions"][version]["evaluation_result"] if version else eval_data["original_submission"]["evaluation_result"]
            MAX_LOG_CHARS = 100000
            truncated_logs = logs[:MAX_LOG_CHARS] + f"\n\n[... logs truncated ...]" if len(logs) > MAX_LOG_CHARS else logs
            target.update({
                "status": "completed" if success else "failed", "score": score,
                "success": success, "error": error, "logs": truncated_logs,
                constants.EVALUATION_DETAILS_KEY: details, "evaluation_timestamp": time.time(),
                "evaluation_datetime": datetime.now().isoformat()
            })
            # Store sort_key if provided
            if sort_key is not None:
                target["sort_key"] = sort_key
            
            current_version = eval_data["current_state"]["latest_version"]
            is_latest = (version and current_version == version) or (not version and current_version == "original")
            if is_latest:
                eval_data["current_state"].update({
                    "latest_score": score,
                    "latest_status": "completed" if success else "failed"
                })
            
            all_complete = self._check_all_complete(eval_data)
            eval_data["current_state"]["all_evaluations_complete"] = all_complete
            if all_complete:
                self._check_and_notify(eval_data)
            self._save_evaluation(eval_id, eval_data)
        return eval_data

    def start_claude_session(self, eval_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        with self._file_lock(self._get_eval_path(eval_id)):
            eval_data = self._load_evaluation(eval_id)
            if not eval_data:
                return None
            
            eval_data["claude_sessions"].append({
                "session_id": session_id, "started_timestamp": time.time(),
                "completed_timestamp": None, "final_report": None,
                "success": None, "versions_created": []
            })
            eval_data["current_state"].update({"claude_active": True, "all_evaluations_complete": False})
            self._save_evaluation(eval_id, eval_data)
        return eval_data

    def complete_claude_session(self, eval_id: str, final_report: str, success: bool) -> Optional[Dict[str, Any]]:
        with self._file_lock(self._get_eval_path(eval_id)):
            eval_data = self._load_evaluation(eval_id)
            if not eval_data:
                return None
            
            if eval_data.get("claude_sessions"):
                eval_data["claude_sessions"][-1].update({
                    "completed_timestamp": time.time(),
                    "final_report": final_report, "success": success
                })
            eval_data["current_state"]["claude_active"] = False
            if eval_data["current_state"]["all_evaluations_complete"]:
                self._check_and_notify(eval_data)
            self._save_evaluation(eval_id, eval_data)
        return eval_data

    def get_evaluation(self, eval_id: str) -> Optional[Dict[str, Any]]:
        return self._load_evaluation(eval_id)

    def get_display_info(self, eval_id: str) -> Optional[Dict[str, Any]]:
        return get_evaluation_display_info(eval_id, self.evaluations_dir)

    def get_review_info(self, eval_id: str) -> Optional[Dict[str, str]]:
        return get_evaluation_review_info(eval_id, self.evaluations_dir)

    def should_wait_at_tick(self, current_tick: int) -> bool:
        with self._running_lock:
            max_allowed_ticks = constants.RESEARCH_EVAL_MAX_TICK
            
            # Check running evaluations
            for eval_id, info in self._running_evaluations.items():
                elapsed_ticks = current_tick - info["start_tick"] + 1
                if elapsed_ticks >= max_allowed_ticks:
                    if eval_id not in self._tick_limit_logged:
                        print(f"EvaluationManager: Evaluation {eval_id} has reached tick limit.")
                        self._tick_limit_logged.add(eval_id)
                    return True
            
            # Check pending evaluations
            pending_path = os.path.join(
                os.path.dirname(self.evaluations_dir),
                constants.PENDING_RESEARCH_EVALUATIONS_FILENAME
            )
            
            try:
                pending_evals = file_io_utils.load_yaml_lines(pending_path)
                for pending_eval in pending_evals:
                    eval_id = pending_eval.get(constants.EVALUATION_ID_KEY)
                    submitted_tick = pending_eval.get(constants.EVALUATION_SUBMITTED_TICK_KEY)
                    
                    if eval_id and submitted_tick:
                        elapsed_ticks = current_tick - submitted_tick + 1
                        if elapsed_ticks >= max_allowed_ticks:
                            if eval_id not in self._tick_limit_logged:
                                print(f"EvaluationManager: Pending evaluation {eval_id} has reached tick limit.")
                                self._tick_limit_logged.add(eval_id)
                            return True
            except FileNotFoundError:
                # No pending evaluations file, that's fine
                pass
            except Exception as e:
                print(f"EvaluationManager: Error checking pending evaluations: {e}")
            
            return False

    def get_all_evaluation_ids(self) -> List[str]:
        try:
            return [f[11:-5] for f in os.listdir(self.evaluations_dir) if f.startswith('evaluation_') and f.endswith('.json')]
        except FileNotFoundError:
            return []

    def _initialize_top_submission(self):
        try:
            top_sort_key = None
            for eval_id in self.get_all_evaluation_ids():
                eval_data = self._load_evaluation(eval_id)
                if not eval_data or not eval_data.get("notification", {}).get("sent"):
                    continue
                version_notified = eval_data["notification"].get("version_notified", "original")
                result_source = eval_data["versions"].get(version_notified) if version_notified != "original" else eval_data["original_submission"]
                score = result_source["evaluation_result"].get("score")
                sort_key = result_source["evaluation_result"].get("sort_key")
                
                # Skip invalid scores
                if score in ["n.a.", "pending"] or score is None:
                    continue
                
                # Use sort_key if available, otherwise use score
                if sort_key is not None:
                    current_sort_key = tuple(sort_key) if isinstance(sort_key, list) else (sort_key if isinstance(sort_key, tuple) else (sort_key,))
                else:
                    try:
                        current_sort_key = (float(score),)
                    except (TypeError, ValueError):
                        continue
                
                # Compare sort keys (higher is better)
                if top_sort_key is None or current_sort_key > top_sort_key:
                    top_sort_key = current_sort_key
                    self.top_submission = {'evaluation_id': eval_data["id"], 'title': eval_data["title"],
                                           'score': score, 'agent_name': eval_data["author"],
                                           'task_id': eval_data["research_task_id"],
                                           'submitted_tick': eval_data["submitted_tick"],
                                           'sort_key': current_sort_key}
        except Exception as e:
            print(f"EvaluationManager: Error initializing top submission: {e}")

    def _update_top_submission_if_needed(self, eval_data: Dict[str, Any]):
        if not eval_data.get("notification", {}).get("sent"):
            return
        version_notified = eval_data["notification"].get("version_notified", "original")
        result_source = eval_data["versions"].get(version_notified) if version_notified != "original" else eval_data["original_submission"]
        score = result_source["evaluation_result"].get("score")
        sort_key = result_source["evaluation_result"].get("sort_key")
        
        # Skip invalid scores
        if score in ["n.a.", "pending"] or score is None:
            return
        
        # Use sort_key if available, otherwise use score
        if sort_key is not None:
            current_sort_key = tuple(sort_key) if isinstance(sort_key, list) else (sort_key if isinstance(sort_key, tuple) else (sort_key,))
        else:
            try:
                current_sort_key = (float(score),)
            except (TypeError, ValueError):
                return
        
        # Get current top sort key
        if self.top_submission is None:
            top_sort_key = None
        else:
            # Use stored sort_key if available, otherwise reconstruct from score
            top_sort_key = self.top_submission.get("sort_key")
            if top_sort_key is None:
                top_score = self.top_submission.get("score")
                try:
                    top_sort_key = (float(top_score),)
                except (TypeError, ValueError):
                    top_sort_key = None
            elif isinstance(top_sort_key, list):
                # Convert list to tuple for consistent comparison
                top_sort_key = tuple(top_sort_key)
        
        # Compare sort keys (higher is better)
        if top_sort_key is None or current_sort_key > top_sort_key:
            self.top_submission = {'evaluation_id': eval_data["id"], 'title': eval_data["title"],
                                   'score': score, 'agent_name': eval_data["author"],
                                   'task_id': eval_data["research_task_id"],
                                   'submitted_tick': eval_data["submitted_tick"],
                                   'sort_key': current_sort_key}
            print(f"EvaluationManager: New top submission - ID: {eval_data['id']}, Score: {score}, Sort key: {current_sort_key}")

    def get_top_submission(self):
        return self.top_submission.copy() if self.top_submission else None

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        with self._running_lock:
            running_evaluations = []
            for eval_id, info in self._running_evaluations.items():
                eval_data = self._load_evaluation(eval_id)
                if eval_data:
                    latest_version = eval_data["current_state"]["latest_version"]
                    version_data = eval_data["versions"].get(latest_version, {}) if latest_version != "original" else eval_data["original_submission"]
                    start_timestamp = version_data.get("created_timestamp") or version_data.get("submitted_timestamp", 0)
                    running_evaluations.append({'evaluation_id': eval_id, 'agent_name': eval_data["author"],
                                               'task_id': eval_data["research_task_id"],
                                               'title': eval_data["title"][:50] + "..." if len(eval_data["title"]) > 50 else eval_data["title"],
                                               'start_tick': info.get("start_tick", 0),
                                               'start_timestamp': start_timestamp,
                                               'elapsed_seconds': int(time.time() - start_timestamp) if start_timestamp else 0})
            running_evaluations.sort(key=lambda x: x['start_timestamp'], reverse=True)
            return {'running_count': len(running_evaluations),
                    'top_submission': self.get_top_submission(),
                    'running_evaluations': running_evaluations}