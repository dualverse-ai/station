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

# station/eval_archive.py
"""
Automated archive evaluation system using LLM to evaluate agent archive submissions.
Runs in parallel to the main orchestrator to automatically resolve waiting states.
"""

import os
import json
import time
import yaml
import threading
import traceback
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from queue import Queue

from station import constants
from station import file_io_utils
from station import capsule as capsule_module
from station.llm_connectors import create_llm_connector, LLMTransientAPIError, LLMPermanentAPIError, LLMSafetyBlockError, LLMContextOverflowError


class AutoArchiveEvaluator:
    """
    Automated archive evaluator that runs in a background thread to evaluate
    pending archive submissions using an LLM evaluator.
    """
    
    # Class-level tracking to prevent duplicate instances
    _active_instances = {}
    
    def __init__(self, station_instance, room_context, enabled: bool = None, model_name: str = None, model_class: str = None, log_queue: Optional[Queue] = None):
        # Check for existing instances
        station_id = id(station_instance)
        if station_id in self._active_instances and self._active_instances[station_id].is_running:
            print(f"AutoArchiveEvaluator: WARNING - Another evaluator instance already exists and is running for this station")
        
        self.station = station_instance
        self.room_context = room_context
        self.enabled = enabled if enabled is not None else (constants.EVAL_ARCHIVE_MODE == "auto")
        self.model_class = model_class or constants.AUTO_EVAL_ARCHIVE_MODEL_CLASS
        self.model_name = model_name or constants.AUTO_EVAL_ARCHIVE_MODEL_NAME
        self.check_interval = constants.AUTO_EVAL_ARCHIVE_CHECK_INTERVAL
        self.max_output_tokens = constants.AUTO_EVAL_ARCHIVE_MAX_OUTPUT_TOKENS
        self.log_queue = log_queue
        
        # Failure tracking
        self.max_retry_attempts = constants.AUTO_EVAL_ARCHIVE_MAX_RETRIES
        self.pass_threshold = constants.ARCHIVE_EVALUATION_PASS_THRESHOLD
        
        self.is_running = False
        self.evaluation_thread: Optional[threading.Thread] = None
        self.llm_connector = None
        self.evaluation_tick_counter = 1  # Internal tick counter for evaluations
        
        # Paths
        self.archive_room_path = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            constants.SHORT_ROOM_NAME_ARCHIVE
        )
        self.pending_archive_file = os.path.join(
            self.archive_room_path,
            constants.PENDING_ARCHIVE_EVALUATIONS_FILENAME
        )
        self.evaluations_dir = os.path.join(self.archive_room_path, constants.ARCHIVE_EVALUATIONS_SUBDIR_NAME)
        
        if self.enabled:
            self._initialize_llm_connector()
            self._load_evaluation_tick_counter()
            # Refresh tick 1 on restart if we have existing evaluation history
            if self.evaluation_tick_counter > 1:
                self._update_initial_context_in_history_file()

        # Register this instance
        self._active_instances[station_id] = self
            
    def _push_log_event(self, event_type: str, data: Dict[str, Any]):
        """Push log event to queue if available"""
        if self.log_queue:
            log_message = {"event": event_type, "data": data, "timestamp": time.time()}
            try:
                self.log_queue.put_nowait(log_message)
            except Exception as e:
                print(f"AutoArchiveEvaluator: Error putting log event on queue: {e}")
    
    def _initialize_llm_connector(self) -> bool:
        """Initialize the LLM connector for archive evaluation"""
        try:
            self.llm_connector = create_llm_connector(
                model_class_name=self.model_class,
                model_name=self.model_name,
                agent_name="AutoArchiveEvaluator",
                agent_data_path=self.archive_room_path,  # Use archive room for logs
                api_key=None,  # Will use environment variable
                system_prompt="You are a critical reviewer evaluating AI research publications.",
                temperature=0.3,  # Lower temperature for more consistent evaluations
                max_output_tokens=self.max_output_tokens
            )
            
            if self.llm_connector:
                self._push_log_event("auto_eval_status", {
                    "status": "connector_initialized",
                    "model_class": self.model_class,
                    "model": self.model_name,
                    "message": "Auto archive evaluator LLM connector initialized successfully"
                })
                return True
            else:
                self._push_log_event("auto_eval_error", {
                    "error": "Failed to create LLM connector",
                    "model_class": self.model_class,
                    "model": self.model_name
                })
                return False
                
        except Exception as e:
            self._push_log_event("auto_eval_error", {
                "error": f"Exception during LLM connector initialization: {str(e)}",
                "model_class": self.model_class,
                "model": self.model_name,
                "trace": traceback.format_exc()
            })
            print(f"AutoArchiveEvaluator: Failed to initialize LLM connector: {e}")
            return False
    
    def start_evaluation_loop(self) -> bool:
        """Start the background evaluation loop"""
        if not self.enabled:
            print("AutoArchiveEvaluator: Auto evaluation is disabled")
            return False
            
        if self.is_running:
            print("AutoArchiveEvaluator: Evaluation loop is already running")
            return False
            
        # Check for other running instances globally to prevent duplicates
        for station_id, instance in self._active_instances.items():
            if instance is not self and instance.is_running:
                print(f"AutoArchiveEvaluator: WARNING - Another evaluator instance is already running globally. Skipping start.")
                return False
            
        if not self.llm_connector:
            print("AutoArchiveEvaluator: Cannot start - LLM connector not initialized")
            return False
            
        print("AutoArchiveEvaluator: Starting evaluation loop...")
        self.is_running = True
        self.evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.evaluation_thread.start()
        
        self._push_log_event("auto_eval_status", {
            "status": "started",
            "message": "Auto archive evaluation loop started"
        })
        return True
    
    def stop_evaluation_loop(self):
        """Stop the background evaluation loop"""
        if not self.is_running:
            return
            
        print("AutoArchiveEvaluator: Stopping evaluation loop...")
        self.is_running = False
        
        # Unregister this instance
        station_id = id(self.station)
        if station_id in self._active_instances and self._active_instances[station_id] is self:
            del self._active_instances[station_id]
        
        if self.evaluation_thread and self.evaluation_thread.is_alive():
            self.evaluation_thread.join(timeout=5)
            if self.evaluation_thread.is_alive():
                print("AutoArchiveEvaluator: Warning - evaluation thread did not stop within timeout")
        
        self._push_log_event("auto_eval_status", {
            "status": "stopped",
            "message": "Auto archive evaluation loop stopped"
        })
    
    def _evaluation_loop(self):
        """Main evaluation loop that runs in background thread"""
        print("AutoArchiveEvaluator: Evaluation loop started")
        
        try:
            while self.is_running:
                try:
                    # Check for pending archive submissions
                    pending_archives = self._load_pending_archives()
                    if pending_archives:
                        # Filter out submissions that have exceeded retry attempts
                        eligible_archives = [archive for archive in pending_archives if self._should_retry_archive(archive)]
                        
                        if eligible_archives:
                            self._push_log_event("auto_eval_status", {
                                "status": "processing",
                                "pending_count": len(eligible_archives),
                                "total_pending": len(pending_archives),
                                "message": f"Processing {len(eligible_archives)} of {len(pending_archives)} pending archive evaluations"
                            })
                            
                            for archive_entry in eligible_archives:
                                if not self.is_running:
                                    break
                                self._evaluate_single_archive(archive_entry)
                        elif pending_archives:
                            # All archives have exceeded retry attempts - provide detailed information
                            failed_details = []
                            for archive in pending_archives:
                                agent_name = archive.get("agent_name", "Unknown")
                                title = archive.get("title", "Untitled")
                                retry_count = archive.get("auto_eval_retry_count", 0)
                                last_failure = archive.get("last_auto_eval_failure")
                                failed_details.append({
                                    "agent_name": agent_name,
                                    "title": title,
                                    "retry_count": retry_count,
                                    "last_failure_timestamp": last_failure
                                })

                            self._push_log_event("auto_eval_status", {
                                "status": "all_failed",
                                "failed_count": len(pending_archives),
                                "failed_archives": failed_details,
                                "message": f"All {len(pending_archives)} pending archives have exceeded retry attempts (max: {self.max_retry_attempts})"
                            })

                            # Print detailed failure information to console for debugging
                            print(f"AutoArchiveEvaluator: All {len(pending_archives)} pending archives have exceeded retry attempts:")
                            for i, details in enumerate(failed_details, 1):
                                print(f"  {i}. '{details['title']}' by {details['agent_name']} - {details['retry_count']} failed attempts")
                                if details['last_failure_timestamp']:
                                    last_fail_dt = datetime.fromtimestamp(details['last_failure_timestamp'])
                                    print(f"     Last failure: {last_fail_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"AutoArchiveEvaluator: These archives require manual intervention or system debugging.")
                    
                    # Sleep before next check
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    self._push_log_event("auto_eval_error", {
                        "error": f"Exception in evaluation loop: {str(e)}",
                        "trace": traceback.format_exc()
                    })
                    print(f"AutoArchiveEvaluator: Exception in evaluation loop: {e}")
                    time.sleep(self.check_interval * 2)  # Wait longer after error
                    
        except Exception as e:
            print(f"AutoArchiveEvaluator: Fatal error in evaluation loop: {e}")
            traceback.print_exc()
        finally:
            print("AutoArchiveEvaluator: Evaluation loop ended")
    
    def _load_pending_archives(self) -> List[Dict[str, Any]]:
        """Load pending archive evaluations from file"""
        if not file_io_utils.file_exists(self.pending_archive_file):
            return []
            
        try:
            with open(self.pending_archive_file, 'r', encoding='utf-8') as f:
                pending_archives = []
                for line in f:
                    line = line.strip()
                    if line:
                        pending_archives.append(json.loads(line))
                return pending_archives
        except Exception as e:
            self._push_log_event("auto_eval_error", {
                "error": f"Failed to load pending archives: {str(e)}",
                "file": self.pending_archive_file
            })
            return []
    
    def _manage_llm_history_size(self):
        """Manage LLM chat history size using the existing pruning mechanism"""
        if not self.llm_connector:
            return
            
        try:
            # Load current history from the LLM connector's history file
            history_entries = self.llm_connector._load_history_from_file()
            
            # Count the number of evaluation exchanges (user prompts, not assistant responses)
            evaluation_count = sum(1 for entry in history_entries if entry.get('role') == 'user')
            
            max_size = getattr(constants, 'AUTO_EVAL_ARCHIVE_MAX_SIZE', 15)
            restore_size = getattr(constants, 'AUTO_EVAL_ARCHIVE_RESTORE_SIZE', 10)
            
            if evaluation_count >= max_size:
                print(f"AutoArchiveEvaluator: Managing history size - current evaluations: {evaluation_count}, max: {max_size}")
                
                # Calculate how many evaluations to remove
                evaluations_to_remove = evaluation_count - restore_size
                
                # Group history entries by tick (each evaluation should have unique tick)
                entries_by_tick = {}
                for entry in history_entries:
                    tick = entry.get('tick', 0)
                    if tick not in entries_by_tick:
                        entries_by_tick[tick] = []
                    entries_by_tick[tick].append(entry)
                
                # Sort ticks and identify which ones to prune
                # CRITICAL: Always exclude tick 1 (initial context) from pruning
                sorted_ticks = sorted(entries_by_tick.keys())
                ticks_to_prune = [tick for tick in sorted_ticks[:evaluations_to_remove] if tick != 1]
                
                if ticks_to_prune:
                    # Update the evaluator's agent data with pruning info
                    # The base connector will load this automatically in send_message()
                    self._update_evaluator_pruning_info(ticks_to_prune)
                    
                    # Refresh initial context with fresh research task and archive data
                    self._update_initial_context_in_history_file()
                    
                    print(f"AutoArchiveEvaluator: Marked {len(ticks_to_prune)} old evaluations for pruning and refreshed initial context")
                    
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to manage LLM history size: {e}")
    
    def _handle_context_overflow(self) -> bool:
        """Handle context overflow by aggressively pruning old evaluations and refreshing context"""
        if not self.llm_connector:
            print("AutoArchiveEvaluator: Cannot handle context overflow - no LLM connector")
            return False
            
        try:
            print("AutoArchiveEvaluator: Context overflow detected, starting recovery...")
            
            # Load current history entries to identify what to prune
            history_entries = self.llm_connector._load_history_from_file()
            
            if not history_entries:
                print("AutoArchiveEvaluator: No history to prune for context overflow recovery")
                return False
            
            # Get unique tick numbers and identify which ones to prune (always exclude tick 1)
            all_ticks = sorted(set(entry.get('tick', 0) for entry in history_entries))
            prune_count = constants.AUTO_EVAL_ARCHIVE_OVERFLOW_PRUNE_COUNT
            ticks_to_prune = [tick for tick in all_ticks[:prune_count] if tick != 1]
            
            if not ticks_to_prune:
                print("AutoArchiveEvaluator: No ticks available to prune for overflow recovery")
                return False
            
            print(f"AutoArchiveEvaluator: Pruning {len(ticks_to_prune)} ticks for overflow recovery: {ticks_to_prune}")
            
            # Update pruning info
            self._update_evaluator_pruning_info(ticks_to_prune)
            
            # Refresh initial context with fresh data
            self._update_initial_context_in_history_file()
            
            # Note: LLM connector will automatically detect pruning changes and re-initialize
            # its session on the next send_message() call
            
            print(f"AutoArchiveEvaluator: Context overflow recovery completed, pruned {len(ticks_to_prune)} ticks")
            return True
            
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to handle context overflow: {e}")
            traceback.print_exc()
            return False
    
    def _update_evaluator_pruning_info(self, ticks_to_prune: List[int]):
        """Update the evaluator's agent data with new pruning information using YAML block format"""
        try:
            # Load existing evaluator agent data or create new
            evaluator_agent_data = self.station.agent_module.load_agent_data(
                "AutoArchiveEvaluator", include_ended=True, include_ascended=True
            ) or self._create_evaluator_agent_data()

            # Get existing prune blocks
            current_prune_blocks = evaluator_agent_data.get(constants.AGENT_PRUNED_DIALOGUE_TICKS_KEY, [])

            # Convert ticks to consecutive ranges for efficient storage
            if ticks_to_prune:
                # Sort ticks and group into consecutive ranges
                sorted_ticks = sorted(ticks_to_prune)
                ranges = []
                start = sorted_ticks[0]
                end = sorted_ticks[0]

                for tick in sorted_ticks[1:]:
                    if tick == end + 1:
                        end = tick
                    else:
                        # Add completed range
                        if start == end:
                            ranges.append(str(start))
                        else:
                            ranges.append(f"{start}-{end}")
                        start = end = tick

                # Add final range
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")

                # Create new prune blocks for each range (with empty summary for complete removal)
                for range_str in ranges:
                    new_block = {
                        constants.PRUNE_TICKS_KEY: range_str,
                        constants.PRUNE_SUMMARY_KEY: ""  # Empty summary = complete removal
                    }
                    current_prune_blocks.append(new_block)

            # Update agent data with new block format
            evaluator_agent_data[constants.AGENT_PRUNED_DIALOGUE_TICKS_KEY] = current_prune_blocks

            # Save the updated agent data
            self.station.agent_module.save_agent_data("AutoArchiveEvaluator", evaluator_agent_data)

        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to update pruning info: {e}")
    
    def _create_evaluator_agent_data(self) -> Dict[str, Any]:
        """Create minimal agent data structure for the evaluator"""
        return {
            constants.AGENT_NAME_KEY: "AutoArchiveEvaluator",
            constants.AGENT_STATUS_KEY: "System",
            constants.AGENT_PRUNED_DIALOGUE_TICKS_KEY: [],  # New block format: list instead of dict
            "evaluation_tick_counter": 1
        }
    
    def _load_evaluation_tick_counter(self):
        """Load the evaluation tick counter from agent data"""
        try:
            evaluator_agent_data = self.station.agent_module.load_agent_data(
                "AutoArchiveEvaluator", include_ended=True, include_ascended=True
            )
            if evaluator_agent_data:
                self.evaluation_tick_counter = evaluator_agent_data.get("evaluation_tick_counter", 1)
            else:
                self.evaluation_tick_counter = 1
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to load tick counter, starting from 1: {e}")
            self.evaluation_tick_counter = 1
    
    def _save_evaluation_tick_counter(self):
        """Save the current evaluation tick counter to agent data"""
        try:
            evaluator_agent_data = self.station.agent_module.load_agent_data(
                "AutoArchiveEvaluator", include_ended=True, include_ascended=True
            ) or self._create_evaluator_agent_data()
            
            evaluator_agent_data["evaluation_tick_counter"] = self.evaluation_tick_counter
            
            self.station.agent_module.save_agent_data("AutoArchiveEvaluator", evaluator_agent_data)
            
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to save tick counter: {e}")
    
    def _evaluate_single_archive(self, archive_entry: Dict[str, Any]):
        """Evaluate a single archive submission"""
        agent_name = archive_entry.get("agent_name")
        title = archive_entry.get("title", "Untitled")
        
        self._push_log_event("auto_eval_event", {
            "agent_name": agent_name,
            "title": title,
            "status": "evaluating",
            "message": f"Starting evaluation for {agent_name} archive '{title}'"
        })
        
        # Send initial context if this is the first evaluation ever
        self._send_initial_context_if_needed()
        
        # Manage LLM history size before evaluation
        self._manage_llm_history_size()
        
        # Create submission evaluation prompt (assumes initial context already established)
        # NOTE: This will raise ValueError if EVAL_ARCHIVE_SUBMISSION_PROMPT contains unescaped braces
        evaluation_prompt = self._create_submission_prompt(archive_entry)
        
        try:
            # Get LLM evaluation using current tick counter with context overflow retry
            current_eval_tick = self.evaluation_tick_counter
            self.evaluation_tick_counter += 1  # Increment for next evaluation
            
            # Retry mechanism for context overflow
            max_overflow_retries = constants.AUTO_EVAL_ARCHIVE_OVERFLOW_MAX_RETRIES
            
            for attempt in range(max_overflow_retries + 1):  # 0 to max_retries inclusive
                try:
                    llm_response, thinking_text, success = self._get_llm_evaluation(evaluation_prompt, current_eval_tick)
                    
                    # Check if evaluation succeeded
                    if not success or not llm_response:
                        raise Exception("Failed to get LLM evaluation response")
                    
                    # Success - break out of retry loop
                    break
                    
                except LLMContextOverflowError:
                    print(f"AutoArchiveEvaluator: Context overflow detected for {agent_name} (attempt {attempt + 1}/{max_overflow_retries + 1})")
                    
                    if attempt < max_overflow_retries:
                        # Try to recover from context overflow
                        recovery_success = self._handle_context_overflow()
                        
                        if recovery_success:
                            print(f"AutoArchiveEvaluator: Context overflow recovery successful, retrying evaluation for {agent_name}")
                            continue  # Retry the evaluation
                        else:
                            print(f"AutoArchiveEvaluator: Context overflow recovery failed for {agent_name}")
                            raise Exception("Context overflow recovery failed")
                    else:
                        print(f"AutoArchiveEvaluator: Max context overflow retries exceeded for {agent_name}")
                        raise Exception(f"Context overflow could not be resolved after {max_overflow_retries} attempts")
            
            # Parse evaluation result
            evaluation_result = self._parse_evaluation_response(llm_response)
            
            # Save evaluation log
            self._save_evaluation_log(archive_entry, evaluation_prompt, llm_response, thinking_text, evaluation_result)
            
            # Process evaluation result (create capsule if accepted, notify agent)
            self._process_evaluation_result(archive_entry, evaluation_result)
            
            # Remove from pending archives
            self._remove_from_pending_archives(archive_entry)
            
            # Save the updated tick counter
            self._save_evaluation_tick_counter()
            
            self._push_log_event("auto_eval_event", {
                "agent_name": agent_name,
                "title": title,
                "status": "completed",
                "score": evaluation_result.get("score", 0),
                "result": "accepted" if evaluation_result.get("score", 0) >= self.pass_threshold else "rejected",
                "message": f"Auto-evaluation completed for {agent_name} archive '{title}'"
            })
            
        except Exception as e:
            self._push_log_event("auto_eval_error", {
                "agent_name": agent_name,
                "title": title,
                "error": f"Failed to evaluate archive: {str(e)}",
                "error_type": type(e).__name__,
                "evaluation_id": archive_entry.get("evaluation_id"),
                "trace": traceback.format_exc()
            })
            print(f"AutoArchiveEvaluator: Failed to evaluate archive '{title}' for {agent_name}: {type(e).__name__}: {e}")

            # Update failure count in pending archive entry
            self._update_archive_failure_count(archive_entry)
    
    def _should_retry_archive(self, archive_entry: Dict[str, Any]) -> bool:
        """Check if an archive should be retried based on failure count"""
        retry_count = archive_entry.get("auto_eval_retry_count", 0)
        return retry_count < self.max_retry_attempts
    
    def _update_archive_failure_count(self, archive_entry: Dict[str, Any]):
        """Update the failure count for an archive and save back to pending file"""
        try:
            agent_name = archive_entry.get("agent_name")
            evaluation_id = archive_entry.get("evaluation_id")
            
            # Load all pending archives
            pending_archives = self._load_pending_archives()
            
            # Find and update the specific archive entry
            for i, pending_archive in enumerate(pending_archives):
                if (pending_archive.get("agent_name") == agent_name and 
                    pending_archive.get("evaluation_id") == evaluation_id):
                    
                    retry_count = pending_archive.get("auto_eval_retry_count", 0) + 1
                    pending_archives[i]["auto_eval_retry_count"] = retry_count
                    pending_archives[i]["last_auto_eval_failure"] = time.time()
                    
                    if retry_count >= self.max_retry_attempts:
                        self._push_log_event("auto_eval_error", {
                            "agent_name": agent_name,
                            "title": archive_entry.get("title", "Untitled"),
                            "status": "max_retries_exceeded",
                            "retry_count": retry_count,
                            "message": f"Archive submission for {agent_name} exceeded {self.max_retry_attempts} retry attempts, will not retry again"
                        })
                        print(f"AutoArchiveEvaluator: Archive submission for {agent_name} exceeded max retries ({self.max_retry_attempts})")
                        
                        # Notify agent of evaluation failure
                        self._notify_agent_evaluation_failed(archive_entry)
                    
                    break
            
            # Save updated pending archives back to file
            with open(self.pending_archive_file, 'w', encoding='utf-8') as f:
                for archive_entry in pending_archives:
                    f.write(json.dumps(archive_entry) + '\n')
                    
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to update archive failure count: {e}")
    
    def _notify_agent_evaluation_failed(self, archive_entry: Dict[str, Any]):
        """Notify agent that their archive evaluation failed"""
        try:
            agent_name = archive_entry.get("agent_name")
            title = archive_entry.get("title", "Untitled")
            
            agent_data = self.station.agent_module.load_agent_data(agent_name)
            if not agent_data:
                return
            
            # Add notification
            failure_notification = (
                f"Your archive submission '{title}' could not be evaluated due to repeated system failures. "
                f"Please contact a human administrator for manual review."
            )
            
            self.station.agent_module.add_pending_notification(agent_data, failure_notification)
            
            # Save agent data
            if self.station.agent_module.save_agent_data(agent_name, agent_data):
                print(f"AutoArchiveEvaluator: Notified {agent_name} of evaluation failure")
            else:
                print(f"AutoArchiveEvaluator: Failed to save evaluation failure notification for {agent_name}")
                
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to notify agent of evaluation failure: {e}")

    def _create_evaluation_prompt(self, archive_entry: Dict[str, Any]) -> str:
        """Create evaluation prompt for the LLM (legacy method, now uses submission prompt)"""
        return self._create_submission_prompt(archive_entry)
    
    def _create_submission_prompt(self, archive_entry: Dict[str, Any]) -> str:
        """Create submission evaluation prompt (assumes initial context already established)"""
        title = archive_entry.get("title", "")
        tags = archive_entry.get("tags", "")
        abstract = archive_entry.get("abstract", "")
        content = archive_entry.get("content", "")
        
        # Calculate word count for the content
        word_count = len(content.split()) if content else 0
        
        submission_prompt = constants.EVAL_ARCHIVE_SUBMISSION_PROMPT.format(
            title=title,
            tags=tags,
            abstract=abstract,
            content=content,
            word_count=word_count
        )

        return submission_prompt
    
    def _get_llm_evaluation(self, prompt: str, current_tick: int) -> Tuple[Optional[str], Optional[str], bool]:
        """Get evaluation from LLM"""
        try:
            if not self.llm_connector:
                return None, None, False
            
            # Get LLM response using send_message method with proper tick
            response, thinking_text, token_info = self.llm_connector.send_message(prompt, current_tick=current_tick)
            
            # Push SSE event for reviewer dialogue (for real-time updates in UI)
            if response:
                self._push_log_event("system_message", {
                    "agent_name": "Reviewer",
                    "message": f"[Reviewer] New evaluation response received (tick {current_tick})",
                    "source": "auto_archive_evaluator"
                })
                return response, thinking_text, True
            else:
                return None, thinking_text, False
                
        except LLMContextOverflowError as e:
            print(f"AutoArchiveEvaluator: Context overflow during evaluation: {e}")
            self._push_log_event("auto_eval_error", {
                "error": "Context overflow during evaluation", 
                "message": str(e),
                "recovery_action": "Will attempt overflow recovery"
            })
            # Re-raise as a specific exception that caller can catch
            raise
        except (LLMTransientAPIError, LLMPermanentAPIError, LLMSafetyBlockError) as e:
            print(f"AutoArchiveEvaluator: LLM error during evaluation: {e}")
            return None, None, False
        except Exception as e:
            print(f"AutoArchiveEvaluator: Unexpected error during LLM evaluation: {e}")
            return None, None, False
    
    def _parse_evaluation_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse evaluation result from LLM response"""
        # Look for YAML block in the response
        lines = llm_response.split('\n')
        yaml_start = -1
        yaml_end = -1
        
        for i, line in enumerate(lines):
            if '```yaml' in line.lower():
                yaml_start = i + 1
            elif yaml_start != -1 and '```' in line:
                yaml_end = i
                break
        
        if yaml_start != -1 and yaml_end != -1:
            yaml_content = '\n'.join(lines[yaml_start:yaml_end])
            try:
                parsed_yaml = yaml.safe_load(yaml_content)
                if isinstance(parsed_yaml, dict):
                    # Validate required fields and convert score to int
                    score = parsed_yaml.get("score", 1)
                    try:
                        score = int(score)
                        if score < 1:
                            score = 1
                        elif score > 10:
                            score = 10
                    except (ValueError, TypeError):
                        score = 1  # Default to lowest score if invalid
                    
                    result = {
                        "score": score,
                        "comment": str(parsed_yaml.get("comment", "No comment provided")),
                        "suggestion": str(parsed_yaml.get("suggestion", "No suggestion provided")),
                        "evaluation_status": "success"
                    }
                    
                    # Handle additional fields if configured
                    additional_fields = getattr(constants, 'AUTO_EVAL_ARCHIVE_ADDITIONAL_FIELDS', None)
                    if additional_fields:
                        for field_name in additional_fields:
                            if field_name in parsed_yaml:
                                try:
                                    # Try to convert to int/float for score-like fields
                                    field_value = parsed_yaml[field_name]
                                    if isinstance(field_value, (int, float)):
                                        result[field_name] = field_value
                                    else:
                                        # Try to parse as number
                                        result[field_name] = float(field_value) if '.' in str(field_value) else int(field_value)
                                except (ValueError, TypeError):
                                    # Store as string if not numeric
                                    result[field_name] = str(parsed_yaml[field_name])
                            else:
                                print(f"AutoArchiveEvaluator: Warning - Required additional field '{field_name}' missing from reviewer response")
                                result[field_name] = None
                    
                    return result
            except yaml.YAMLError as e:
                print(f"AutoArchiveEvaluator: YAML parsing error: {e}")
        
        # Fallback: try to extract simple score from response
        response_lower = llm_response.lower()
        score = 1  # Default to lowest score
        
        # Simple regex-like extraction
        for i in range(1, 11):
            if f"score: {i}" in response_lower or f"score:{i}" in response_lower:
                score = i
                break
        
        return {
            "score": score,
            "comment": "Auto-parsed from response (YAML parsing failed)",
            "suggestion": "Please ensure your submission follows the publication guidelines",
            "evaluation_status": "parsing_error"
        }
    
    def _save_evaluation_log(self, archive_entry: Dict[str, Any], evaluation_prompt: str, 
                           llm_response: str, thinking_text: Optional[str], evaluation_result: Dict[str, Any]):
        """Save evaluation log to file"""
        try:
            # Ensure evaluations directory exists
            file_io_utils.ensure_dir_exists(self.evaluations_dir)
            
            timestamp = int(time.time())
            evaluation_id = archive_entry.get("evaluation_id")
            log_filename = f"evaluation_{evaluation_id}_{timestamp}.yaml"
            log_filepath = os.path.join(self.evaluations_dir, log_filename)
            
            log_data = {
                "evaluation_id": evaluation_id,
                "agent_name": archive_entry.get("agent_name"),
                "submission_tick": archive_entry.get("submission_tick"),
                "title": archive_entry.get("title"),
                "tags": archive_entry.get("tags"),
                "abstract": archive_entry.get("abstract"),
                "content": archive_entry.get("content"),
                "evaluation_timestamp": timestamp,
                "evaluation_datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "evaluator_model": self.model_name,
                "llm_evaluation_prompt": evaluation_prompt,
                "llm_evaluation_response": llm_response,
                "llm_evaluation_thinking": thinking_text,
                "extracted_result": evaluation_result,
                "result": "accepted" if evaluation_result.get("score", 0) >= self.pass_threshold else "rejected"
            }
            
            with open(log_filepath, 'w', encoding='utf-8') as f:
                yaml.dump(log_data, f, default_flow_style=False, allow_unicode=True)
                
            print(f"AutoArchiveEvaluator: Saved evaluation log to {log_filepath}")
            
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to save evaluation log: {e}")
    
    def _process_evaluation_result(self, archive_entry: Dict[str, Any], evaluation_result: Dict[str, Any]):
        """Process evaluation result: create capsule if accepted, notify agent"""
        agent_name = archive_entry.get("agent_name")
        title = archive_entry.get("title", "Untitled")
        score = evaluation_result.get("score", 0)
        comment = evaluation_result.get("comment", "")
        suggestion = evaluation_result.get("suggestion", "")
        
        try:
            # Load agent data once at the beginning
            agent_data = self.station.agent_module.load_agent_data(agent_name)
            if not agent_data:
                print(f"AutoArchiveEvaluator: Could not load agent data for {agent_name}")
                return
            
            if score >= self.pass_threshold:
                # Archive accepted - create capsule and add reviewer reply
                capsule_result = self._create_archive_capsule(archive_entry, evaluation_result)
                
                if capsule_result:
                    capsule_id, numeric_id = capsule_result
                    
                    # Mark capsules as read on the SAME agent_data object
                    self._mark_capsule_as_read_for_author(agent_name, capsule_id, numeric_id, agent_data)
                    
                    # Format additional scores if present
                    additional_scores_text = self._format_additional_scores(evaluation_result)
                    
                    # Notify agent of acceptance with full review and tips
                    success_notification = (
                        f"Your archive submission '{title}' has been accepted for publication! "
                        f"Score: {score}/10{additional_scores_text}. The capsule has been created and a reviewer comment has been added.\n\n"
                        f"**Full Reviewer Feedback:**\n{comment}\n\n"
                        f"**Suggestions for Future Work:**\n{suggestion}\n\n"
                        f"**Tip:** You can update your published capsule to address the reviewer's suggestions. "
                        f"Use `/execute_action{{update {numeric_id}-1}}` followed by YAML with updated content to revise your original submission. "
                    )
                    
                    # Send station announcement to all agents
                    self._send_station_announcement(archive_entry, capsule_id)
                else:
                    # Failed to create capsule despite acceptance
                    success_notification = (
                        f"Your archive submission '{title}' was accepted (Score: {score}/10) but there was an error creating the capsule. "
                        f"Please contact an administrator.\n\n"
                        f"**Full Reviewer Feedback:**\n{comment}\n\n"
                        f"**Suggestions for Future Work:**\n{suggestion}"
                    )
            else:
                # Format additional scores if present
                additional_scores_text = self._format_additional_scores(evaluation_result)
                
                # Archive rejected
                success_notification = (
                    f"Your archive submission '{title}' was not accepted for publication. "
                    f"Score: {score}/10 (minimum required: {self.pass_threshold}){additional_scores_text}.\n\n"
                    f"**Full Reviewer Feedback:**\n{comment}\n\n"
                    f"**Suggestions for Improvement:**\n{suggestion}\n\n"
                    f"**Tip:** You can submit a revised version through the Archive Room again after making the necessary changes. "
                    f"Rejected submissions do not activate the Archive Room cooldown, meaning you can resubmit immediately. "
                    f"Focus on addressing the key concerns mentioned in the reviewer feedback to improve your chances of acceptance. "
                    f"You may need to perform additional experiments or reflect on your research methodology based on the feedback provided. \n"
                    f"**If you are low on tokens, go to the Token Management Room to restore your token budget. Low tokens are not an excuse for poor-quality submissions, such as reframing work without conducting follow-up experiments.**"
                )
            
            # Add notification to the same agent_data object and save once
            self.station.agent_module.add_pending_notification(agent_data, success_notification)
            
            if self.station.agent_module.save_agent_data(agent_name, agent_data):
                print(f"AutoArchiveEvaluator: Notified {agent_name} of evaluation result: {'ACCEPTED' if score >= self.pass_threshold else 'REJECTED'}")
            else:
                print(f"AutoArchiveEvaluator: Failed to save evaluation notification for {agent_name}")
                
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to process evaluation result: {e}")
    
    def _create_archive_capsule(self, archive_entry: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Optional[Tuple[str, int]]:
        """Create archive capsule for accepted submission and add reviewer reply"""
        try:
            # Load agent data for lineage info
            agent_name = archive_entry.get("agent_name")
            agent_data = self.station.agent_module.load_agent_data(agent_name)
            if not agent_data:
                print(f"AutoArchiveEvaluator: Could not load agent data for {agent_name}")
                return None
            
            # Prepare capsule creation data in the format expected by capsule_manager.create_capsule
            title = archive_entry.get("title")
            tags = archive_entry.get("tags", "")
            abstract = archive_entry.get("abstract", "")
            content = archive_entry.get("content", "")
            submission_tick = archive_entry.get("submission_tick", self.station._get_current_tick())
            
            # Create YAML data in the format expected by the capsule protocol
            yaml_data = {
                constants.YAML_CAPSULE_TITLE: title,
                constants.YAML_CAPSULE_TAGS: tags,
                constants.YAML_CAPSULE_ABSTRACT: abstract,
                constants.YAML_CAPSULE_CONTENT: content
            }
            
            # Use the proper capsule manager to create the capsule
            
            numeric_id, new_capsule = capsule_module.create_capsule(
                capsule_content_from_agent=yaml_data,
                capsule_type=constants.CAPSULE_TYPE_ARCHIVE,
                author_agent_data=agent_data,
                current_tick=submission_tick,
                lineage_for_private=None  # Archive capsules are public
            )
            
            if not new_capsule:
                print("AutoArchiveEvaluator: Failed to create capsule via capsule manager")
                return None
            
            capsule_id = new_capsule.get(constants.CAPSULE_ID_KEY)
            print(f"AutoArchiveEvaluator: Created archive capsule {capsule_id} for {agent_name}")
            
            # Add reviewer reply with comment and suggestion
            self._add_reviewer_reply(numeric_id, evaluation_result)
            
            # Don't mark as read here - we'll do it with the shared agent_data object
            
            return capsule_id, numeric_id
            
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to create archive capsule: {e}")
            return None
    
    def _add_reviewer_reply(self, numeric_id: int, evaluation_result: Dict[str, Any]):
        """Add reviewer reply to the created capsule"""
        try:
            score = evaluation_result.get("score", 0)
            comment = evaluation_result.get("comment", "")
            suggestion = evaluation_result.get("suggestion", "")
            
            # Format additional scores if present
            additional_scores_text = self._format_additional_scores(evaluation_result)
            
            reviewer_content = f"""**Reviewer Evaluation**

**Score:** {score}/10{additional_scores_text}

**Reviewer Comments:**
{comment}

**Suggestions for Future Work:**
{suggestion}

*This evaluation was conducted by the automated review system to ensure publication quality standards.*"""
            
            # Create fake agent data for the reviewer system
            reviewer_agent_data = {
                constants.AGENT_NAME_KEY: "Archive Review System",
                constants.AGENT_LINEAGE_KEY: "System",
                constants.AGENT_GENERATION_KEY: 0
            }
            
            # Create reply YAML data
            reply_yaml_data = {
                constants.YAML_CAPSULE_CONTENT: reviewer_content
            }
            
            # Use the proper capsule manager to add the reply
            
            success = capsule_module.add_message_to_capsule(
                numeric_id=numeric_id,
                capsule_type=constants.CAPSULE_TYPE_ARCHIVE,
                message_content_from_agent=reply_yaml_data,
                author_agent_data=reviewer_agent_data,
                current_tick=self.station._get_current_tick(),
                lineage_name=None  # Archive capsules are public
            )
            
            if success:
                print(f"AutoArchiveEvaluator: Added reviewer reply to capsule {numeric_id}")
            else:
                print(f"AutoArchiveEvaluator: Failed to add reviewer reply to capsule {numeric_id}")
            
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to add reviewer reply: {e}")
    
    def _mark_capsule_as_read_for_author(self, agent_name: str, capsule_id: str, numeric_id: int, author_agent_data: Dict[str, Any]):
        """Mark both the main capsule and reviewer reply as read for the author agent"""
        try:
            
            # Get the archive room instance
            archive_room = self.station.rooms.get(constants.ROOM_ARCHIVE)
            if not archive_room:
                print("AutoArchiveEvaluator: Could not get archive room instance")
                return
            
            # Use the provided room context (should always be available)
            if not self.room_context:
                print("AutoArchiveEvaluator: ERROR - No room context provided")
                return
            
            room_context = self.room_context
            
            # Mark the main capsule as read
            archive_room._set_agent_read_status(
                author_agent_data, 
                capsule_id,  # e.g., "archive_1"
                True, 
                room_context
            )
            
            # Get the updated capsule to find all message IDs
            updated_capsule = capsule_module.get_capsule(
                numeric_id,
                constants.CAPSULE_TYPE_ARCHIVE,
                None,
                include_deleted_messages=True
            )
            
            # Mark all messages in the capsule as read (both author's original and reviewer reply)
            if updated_capsule and updated_capsule.get(constants.CAPSULE_MESSAGES_KEY):
                for message in updated_capsule[constants.CAPSULE_MESSAGES_KEY]:
                    message_id = message.get(constants.MESSAGE_ID_KEY)
                    if message_id:
                        archive_room._set_agent_read_status(
                            author_agent_data,
                            message_id,
                            True,
                            room_context
                        )
            
            # Don't save agent data here - we'll save it with notifications in the calling method
            print(f"AutoArchiveEvaluator: Marked capsule {capsule_id} and all messages as read for {agent_name}")
                
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to mark capsule as read for author: {e}")
    
    def _send_station_announcement(self, archive_entry: Dict[str, Any], capsule_id: str):
        """Send station announcement about new archive capsule"""
        try:
            agent_name = archive_entry.get("agent_name")
            title = archive_entry.get("title", "Untitled")
            
            # Get agent data for description
            agent_data = self.station.agent_module.load_agent_data(agent_name)
            author_desc = ""
            if agent_data:
                author_desc = agent_data.get(constants.AGENT_DESCRIPTION_KEY, "")
                if author_desc:
                    author_desc = f" ({author_desc})"
            
            announcement = (
                f"**Station Announcement:** **{agent_name}**{author_desc} has published a new archive capsule: "
                f"'{title}' (Archive #{capsule_id.replace('archive_', '')})."
            )
            
            # Get all active agents except the author
            all_other_active_agents = [
                name for name in self.station.agent_module.get_all_active_agent_names()
                if name != agent_name
            ]
            
            # Send notification to all other agents (filtered by maturity)
            current_tick = self.station._get_current_tick()
            for other_agent_name in all_other_active_agents:
                other_agent_data = self.station.agent_module.load_agent_data(other_agent_name)
                if other_agent_data and self.station._should_agent_receive_broadcast(other_agent_data, current_tick, "archive"):
                    self.station.agent_module.add_pending_notification(other_agent_data, announcement)
                    self.station.agent_module.save_agent_data(other_agent_name, other_agent_data)
            
            print(f"AutoArchiveEvaluator: Sent station announcement for new archive capsule {capsule_id}")
            
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to send station announcement: {e}")
    
    def _remove_from_pending_archives(self, completed_archive_entry: Dict[str, Any]):
        """Remove completed archive from pending archives file"""
        try:
            if not file_io_utils.file_exists(self.pending_archive_file):
                return
            
            # Load all pending archives
            remaining_archives = []
            with open(self.pending_archive_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        archive_entry = json.loads(line)
                        # Keep archives that don't match the completed one
                        if not (archive_entry.get("agent_name") == completed_archive_entry.get("agent_name") and
                                archive_entry.get("evaluation_id") == completed_archive_entry.get("evaluation_id")):
                            remaining_archives.append(archive_entry)
            
            # Write remaining archives back to file
            with open(self.pending_archive_file, 'w', encoding='utf-8') as f:
                for archive_entry in remaining_archives:
                    f.write(json.dumps(archive_entry) + '\n')
            
            print(f"AutoArchiveEvaluator: Removed completed archive from pending list")
            
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to remove archive from pending list: {e}")

    def _load_research_task_spec(self) -> str:
        """Load research task 1 specification from research_tasks.yaml"""
        try:
            research_room_path = os.path.join(
                constants.BASE_STATION_DATA_PATH,
                constants.ROOMS_DIR_NAME,
                "research"
            )
            tasks_path = os.path.join(research_room_path, constants.RESEARCH_TASKS_FILENAME)
            
            if not file_io_utils.file_exists(tasks_path):
                return "No research tasks currently defined."
            
            # Load research tasks
            tasks_data = file_io_utils.load_yaml(tasks_path)
            if not isinstance(tasks_data, list) or not tasks_data:
                return "No research tasks currently available."
            
            # Find task 1 (the main research task)
            task_1 = None
            for task in tasks_data:
                if isinstance(task, dict) and task.get(constants.RESEARCH_TASK_ID_KEY) == 1:
                    task_1 = task
                    break
            
            if not task_1:
                return "Research task 1 not found."
            
            # Use content field if available (contains full specification)
            content = task_1.get("content", "")
            if content:
                return content.strip()
            
            # Fallback to structured fields for backward compatibility
            task_spec = f"**Research Task {task_1.get(constants.RESEARCH_TASK_ID_KEY, 1)}**\n\n"
            
            # Add task title if available
            title = task_1.get("title", "")
            if title:
                task_spec += f"**Title:** {title}\n\n"
            
            # Add task description
            description = task_1.get("description", "")
            if description:
                task_spec += f"**Description:** {description}\n\n"
            
            # Add evaluation criteria if available
            criteria = task_1.get("evaluation_criteria", "")
            if criteria:
                task_spec += f"**Evaluation Criteria:** {criteria}\n\n"
            
            # Add any additional details
            details = task_1.get("details", "")
            if details:
                task_spec += f"**Details:** {details}\n\n"
            
            return task_spec.strip()
            
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to load research task spec: {e}")
            return "Error loading research task specification."

    def _load_archive_abstracts(self) -> str:
        """Load all archive abstracts in preview format similar to scripts/stat/archive_preview.py"""
        try:
            # Get all archive capsules
            archive_capsules_dir = os.path.join(
                constants.BASE_STATION_DATA_PATH,
                constants.CAPSULES_DIR_NAME,
                constants.ARCHIVE_CAPSULES_SUBDIR_NAME
            )
            
            if not os.path.exists(archive_capsules_dir):
                return "No archive papers currently available."
            
            # Get all archive capsule files and sort by ID
            capsule_files = []
            for filename in os.listdir(archive_capsules_dir):
                if filename.startswith('archive_') and filename.endswith('.yaml'):
                    try:
                        # Extract ID from filename (archive_N.yaml)
                        capsule_id = int(filename.split('_')[1].split('.')[0])
                        capsule_files.append((capsule_id, filename))
                    except (IndexError, ValueError):
                        continue
            
            if not capsule_files:
                return "No archive papers currently available."
            
            # Sort by capsule ID
            capsule_files.sort(key=lambda x: x[0])
            
            # Load and format capsules
            archive_previews = []
            for capsule_id, filename in capsule_files:
                filepath = os.path.join(archive_capsules_dir, filename)
                try:
                    capsule_data = file_io_utils.load_yaml(filepath)
                    if not capsule_data:
                        continue
                    
                    # Skip deleted capsules
                    if capsule_data.get(constants.CAPSULE_IS_DELETED_KEY, False):
                        continue
                    
                    # Format capsule preview
                    capsule_full_id = capsule_data.get(constants.CAPSULE_ID_KEY, f"archive_{capsule_id}")
                    title = capsule_data.get(constants.CAPSULE_TITLE_KEY, "Untitled")
                    author = capsule_data.get(constants.CAPSULE_AUTHOR_NAME_KEY, "Unknown")
                    created_tick = capsule_data.get(constants.CAPSULE_CREATED_AT_TICK_KEY, "N/A")
                    abstract = capsule_data.get(constants.CAPSULE_ABSTRACT_KEY, "")
                    
                    preview = f"**Archive #{capsule_id}: {title}**\n"
                    preview += f"Author: {author}, Created at Tick: {created_tick}\n"
                    
                    if abstract:
                        preview += f"Abstract: {abstract}"
                    else:
                        preview += "(No abstract available for this capsule.)"
                    
                    archive_previews.append(preview)
                    
                except Exception as e:
                    print(f"AutoArchiveEvaluator: Error reading archive capsule {filename}: {e}")
                    continue
            
            if not archive_previews:
                return "No archive papers currently available."
            
            # Join all previews with separators
            return "\n\n---\n\n".join(archive_previews)
            
        except Exception as e:
            print(f"AutoArchiveEvaluator: Failed to load archive abstracts: {e}")
            return "Error loading archive abstracts."

    def _send_initial_context_if_needed(self):
        """Send initial context only for the very first evaluation (tick 1)"""
        # If evaluation_tick_counter > 1, we've already sent the initial context
        if self.evaluation_tick_counter > 1:
            return  # Context already sent in previous evaluations
        
        # Load research task and archive data
        research_spec = self._load_research_task_spec()
        archive_abstracts = self._load_archive_abstracts()
        
        # Create initial context prompt
        # NOTE: This will raise ValueError if EVAL_ARCHIVE_INITIAL_PROMPT contains unescaped braces
        initial_prompt = constants.EVAL_ARCHIVE_INITIAL_PROMPT.format(
            research_task_spec=research_spec,
            archive_abstract=archive_abstracts
        )
        
        # Send initial context as tick 1
        current_eval_tick = self.evaluation_tick_counter
        self.evaluation_tick_counter += 1  # Next evaluation will be tick 2
        
        try:
            llm_response, thinking_text, success = self._get_llm_evaluation(initial_prompt, current_eval_tick)
            
            if success and llm_response:
                print(f"AutoArchiveEvaluator: Sent initial context at tick {current_eval_tick}")
                print(f"AutoArchiveEvaluator: Reviewer response: {llm_response[:100]}...")  # First 100 chars
                # Push SSE event for reviewer initial context
                self._push_log_event("system_message", {
                    "agent_name": "Reviewer",
                    "message": f"[Reviewer] Initial context established (tick {current_eval_tick})",
                    "source": "auto_archive_evaluator"
                })
            else:
                print(f"AutoArchiveEvaluator: Failed to send initial context at tick {current_eval_tick}")
                # Reset counter for retry
                self.evaluation_tick_counter = current_eval_tick  # Reset counter for retry
        except LLMContextOverflowError:
            print(f"AutoArchiveEvaluator: Context overflow during initial context setup at tick {current_eval_tick}")
            print("AutoArchiveEvaluator: This is unusual for initial context - may indicate very large research task or archive data")
            # Reset counter for retry (this will be retried on next evaluation attempt)
            self.evaluation_tick_counter = current_eval_tick

    def _update_initial_context_in_history_file(self):
        """Update tick 1 entry in history file with fresh research/archive data"""
        if not self.llm_connector:
            print("AutoArchiveEvaluator: No LLM connector available for history update")
            return
        
        history_file_path = self.llm_connector.history_file_path
        
        if not file_io_utils.file_exists(history_file_path):
            print("AutoArchiveEvaluator: History file does not exist, cannot update initial context")
            return
        
        # Load fresh context data
        research_spec = self._load_research_task_spec()
        archive_abstracts = self._load_archive_abstracts()
        # NOTE: This will raise ValueError if EVAL_ARCHIVE_INITIAL_PROMPT contains unescaped braces
        updated_prompt = constants.EVAL_ARCHIVE_INITIAL_PROMPT.format(
            research_task_spec=research_spec,
            archive_abstract=archive_abstracts
        )
        
        # Load current history entries
        history_entries = file_io_utils.load_yaml_lines(history_file_path)
        
        # Find and update tick 1 entry
        updated = False
        for i, entry in enumerate(history_entries):
            if isinstance(entry, dict) and entry.get('tick') == 1 and entry.get('role') == 'user':
                # Create a clean new entry with proper field order
                new_entry = {
                    'tick': 1,
                    'role': 'user',
                    'parts': [{'text': updated_prompt}]
                }
                # Preserve thinking_content if it existed
                if 'thinking_content' in entry:
                    new_entry['thinking_content'] = entry['thinking_content']
                # Preserve token_info if it existed (though user prompts typically don't have it)
                if 'token_info' in entry:
                    new_entry['token_info'] = entry['token_info']
                history_entries[i] = new_entry
                updated = True
                print(f"AutoArchiveEvaluator: Updated initial context in history file at tick 1")
                break
        
        if not updated:
            print("AutoArchiveEvaluator: Tick 1 entry not found in history file, cannot update initial context")
            return
        
        # Write updated history back to file atomically
        with open(history_file_path, 'w', encoding='utf-8') as f:
            import yaml
            for entry in history_entries:
                yaml_str = yaml.dump(entry, default_flow_style=False, allow_unicode=True).strip()
                f.write(yaml_str + '\n---\n')
        
        print("AutoArchiveEvaluator: Successfully refreshed initial context in history file")

    def _format_additional_scores(self, evaluation_result: Dict[str, Any]) -> str:
        """Format additional score fields for display in notifications and capsules"""
        additional_fields = getattr(constants, 'AUTO_EVAL_ARCHIVE_ADDITIONAL_FIELDS', None)
        if not additional_fields:
            return ""
        
        additional_scores = []
        for field_name in additional_fields:
            field_value = evaluation_result.get(field_name)
            if field_value is not None:
                # Format field name for display (convert snake_case to Title Case)
                display_name = field_name.replace('_', ' ').title()
                additional_scores.append(f"**{display_name}:** {field_value}")
        
        if additional_scores:
            return "\n\n" + "\n".join(additional_scores)
        return ""