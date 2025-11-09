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

# station/eval_test.py
"""
Automated test evaluation system using LLM to evaluate agent responses to tests.
Runs in parallel to the main orchestrator to automatically resolve waiting states.
"""

import os
import json
import time
import yaml
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from queue import Queue

from station import constants
from station import file_io_utils
from station.llm_connectors import create_llm_connector, LLMTransientAPIError, LLMPermanentAPIError, LLMSafetyBlockError


class AutoTestEvaluator:
    """
    Automated test evaluator that runs in a background thread to evaluate
    pending test submissions using an LLM evaluator.
    """
    
    # Class-level tracking to prevent duplicate instances
    _active_instances = {}
    
    def __init__(self, station_instance, enabled: bool = None, model_name: str = None, log_queue: Optional[Queue] = None):
        # Check for existing instances
        station_id = id(station_instance)
        if station_id in self._active_instances and self._active_instances[station_id].is_running:
            print(f"AutoTestEvaluator: WARNING - Another evaluator instance already exists and is running for this station")
        
        self.station = station_instance
        self.enabled = enabled if enabled is not None else constants.AUTO_EVAL_TEST
        self.model_name = model_name or constants.AUTO_EVAL_MODEL_NAME
        self.check_interval = constants.AUTO_EVAL_CHECK_INTERVAL
        self.max_output_tokens = constants.AUTO_EVAL_MAX_OUTPUT_TOKENS
        self.log_queue = log_queue
        
        # Failure tracking
        self.max_retry_attempts = constants.AUTO_EVAL_MAX_RETRIES
        
        self.is_running = False
        self.evaluation_thread: Optional[threading.Thread] = None
        self.llm_connector = None
        
        # Paths
        self.test_room_path = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            constants.SHORT_ROOM_NAME_TEST
        )
        self.pending_tests_file = os.path.join(
            self.test_room_path,
            constants.PENDING_TEST_EVALUATIONS_FILENAME
        )
        self.evaluations_dir = os.path.join(self.test_room_path, "evaluations")
        
        if self.enabled:
            self._initialize_llm_connector()
        
        # Register this instance
        self._active_instances[station_id] = self
            
    def _push_log_event(self, event_type: str, data: Dict[str, Any]):
        """Push log event to queue if available"""
        if self.log_queue:
            log_message = {"event": event_type, "data": data, "timestamp": time.time()}
            try:
                self.log_queue.put_nowait(log_message)
            except Exception as e:
                print(f"AutoTestEvaluator: Error putting log event on queue: {e}")
    
    def _initialize_llm_connector(self) -> bool:
        """Initialize the LLM connector for test evaluation"""
        try:
            self.llm_connector = create_llm_connector(
                model_class_name="Gemini",
                model_name=self.model_name,
                agent_name="AutoTestEvaluator",
                agent_data_path=self.test_room_path,  # Use test room for logs
                api_key=None,  # Will use environment variable
                system_prompt="You are an expert evaluator of AI tests.",
                temperature=0.3,  # Lower temperature for more consistent evaluations
                max_output_tokens=self.max_output_tokens
            )
            
            if self.llm_connector:
                self._push_log_event("auto_eval_status", {
                    "status": "connector_initialized",
                    "model": self.model_name,
                    "message": "Auto test evaluator LLM connector initialized successfully"
                })
                return True
            else:
                self._push_log_event("auto_eval_error", {
                    "error": "Failed to create LLM connector",
                    "model": self.model_name
                })
                return False
                
        except Exception as e:
            self._push_log_event("auto_eval_error", {
                "error": f"Exception during LLM connector initialization: {str(e)}",
                "model": self.model_name,
                "trace": traceback.format_exc()
            })
            print(f"AutoTestEvaluator: Failed to initialize LLM connector: {e}")
            return False
    
    def start_evaluation_loop(self) -> bool:
        """Start the background evaluation loop"""
        if not self.enabled:
            print("AutoTestEvaluator: Auto evaluation is disabled")
            return False
            
        if self.is_running:
            print("AutoTestEvaluator: Evaluation loop is already running")
            return False
            
        # Check for other running instances globally to prevent duplicates
        for station_id, instance in self._active_instances.items():
            if instance is not self and instance.is_running:
                print(f"AutoTestEvaluator: WARNING - Another evaluator instance is already running globally. Skipping start.")
                return False
            
        if not self.llm_connector:
            print("AutoTestEvaluator: Cannot start - LLM connector not initialized")
            return False
            
        print("AutoTestEvaluator: Starting evaluation loop...")
        self.is_running = True
        self.evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.evaluation_thread.start()
        
        self._push_log_event("auto_eval_status", {
            "status": "started",
            "message": "Auto test evaluation loop started"
        })
        return True
    
    def stop_evaluation_loop(self):
        """Stop the background evaluation loop"""
        if not self.is_running:
            return
            
        print("AutoTestEvaluator: Stopping evaluation loop...")
        self.is_running = False
        
        # Unregister this instance
        station_id = id(self.station)
        if station_id in self._active_instances and self._active_instances[station_id] is self:
            del self._active_instances[station_id]
        
        if self.evaluation_thread and self.evaluation_thread.is_alive():
            self.evaluation_thread.join(timeout=5)
            if self.evaluation_thread.is_alive():
                print("AutoTestEvaluator: Warning - evaluation thread did not stop within timeout")
        
        self._push_log_event("auto_eval_status", {
            "status": "stopped",
            "message": "Auto test evaluation loop stopped"
        })
    
    def _evaluation_loop(self):
        """Main evaluation loop that runs in background thread"""
        print("AutoTestEvaluator: Evaluation loop started")
        
        try:
            while self.is_running:
                try:
                    # Check for pending tests
                    pending_tests = self._load_pending_tests()
                    if pending_tests:
                        # Filter out tests that have exceeded retry attempts
                        eligible_tests = [test for test in pending_tests if self._should_retry_test(test)]
                        
                        if eligible_tests:
                            self._push_log_event("auto_eval_status", {
                                "status": "processing",
                                "pending_count": len(eligible_tests),
                                "total_pending": len(pending_tests),
                                "message": f"Processing {len(eligible_tests)} of {len(pending_tests)} pending test evaluations"
                            })
                            
                            for test_entry in eligible_tests:
                                if not self.is_running:
                                    break
                                self._evaluate_single_test(test_entry)
                        elif pending_tests:
                            # All tests have exceeded retry attempts
                            self._push_log_event("auto_eval_status", {
                                "status": "all_failed",
                                "failed_count": len(pending_tests),
                                "message": f"All {len(pending_tests)} pending tests have exceeded retry attempts"
                            })
                    
                    # Sleep before next check
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    self._push_log_event("auto_eval_error", {
                        "error": f"Exception in evaluation loop: {str(e)}",
                        "trace": traceback.format_exc()
                    })
                    print(f"AutoTestEvaluator: Exception in evaluation loop: {e}")
                    time.sleep(self.check_interval * 2)  # Wait longer after error
                    
        except Exception as e:
            print(f"AutoTestEvaluator: Fatal error in evaluation loop: {e}")
            traceback.print_exc()
        finally:
            print("AutoTestEvaluator: Evaluation loop ended")
    
    def _load_pending_tests(self) -> List[Dict[str, Any]]:
        """Load pending test evaluations from file"""
        if not file_io_utils.file_exists(self.pending_tests_file):
            return []
            
        try:
            with open(self.pending_tests_file, 'r', encoding='utf-8') as f:
                pending_tests = []
                for line in f:
                    line = line.strip()
                    if line:
                        pending_tests.append(json.loads(line))
                return pending_tests
        except Exception as e:
            self._push_log_event("auto_eval_error", {
                "error": f"Failed to load pending tests: {str(e)}",
                "file": self.pending_tests_file
            })
            return []
    
    def _evaluate_single_test(self, test_entry: Dict[str, Any]):
        """Evaluate a single test submission"""
        agent_name = test_entry.get("agent_name")
        test_id = test_entry.get("test_id")
        
        try:
            self._push_log_event("auto_eval_event", {
                "agent_name": agent_name,
                "test_id": test_id,
                "status": "evaluating",
                "message": f"Starting evaluation for {agent_name} test {test_id}"
            })
            
            # Load test definition and agent response
            test_definition = self._load_test_definition(test_id)
            agent_response = test_entry.get("agent_response", "")
            
            if not test_definition:
                raise Exception(f"Test definition not found for test ID: {test_id}")
            
            # Create evaluation prompt
            evaluation_prompt = self._create_evaluation_prompt(test_definition, agent_response)
            
            # Get LLM evaluation
            llm_response, thinking_text, success = self._get_llm_evaluation(evaluation_prompt)
            
            if not success or not llm_response:
                raise Exception("Failed to get LLM evaluation response")
            
            # Parse evaluation result
            evaluation_result = self._parse_evaluation_response(llm_response)
            
            # Save evaluation log
            self._save_evaluation_log(agent_name, test_id, test_definition, agent_response, 
                                    evaluation_prompt, llm_response, thinking_text, evaluation_result)
            
            # Update agent test status
            self._update_agent_test_status(agent_name, test_id, test_definition, evaluation_result)
            
            # Remove from pending tests
            self._remove_from_pending_tests(test_entry)
            
            self._push_log_event("auto_eval_event", {
                "agent_name": agent_name,
                "test_id": test_id,
                "status": "completed",
                "result": evaluation_result.get("pass", False),
                "comment": evaluation_result.get("comment", ""),
                "message": f"Auto-evaluation completed for {agent_name} test {test_id}"
            })
            
        except Exception as e:
            self._push_log_event("auto_eval_error", {
                "agent_name": agent_name,
                "test_id": test_id,
                "error": f"Failed to evaluate test: {str(e)}",
                "trace": traceback.format_exc()
            })
            print(f"AutoTestEvaluator: Failed to evaluate test {test_id} for {agent_name}: {e}")
            
            # Update failure count in pending test entry
            self._update_test_failure_count(test_entry)
    
    def _should_retry_test(self, test_entry: Dict[str, Any]) -> bool:
        """Check if a test should be retried based on failure count"""
        retry_count = test_entry.get("auto_eval_retry_count", 0)
        return retry_count < self.max_retry_attempts
    
    def _update_test_failure_count(self, test_entry: Dict[str, Any]):
        """Update the failure count for a test and save back to pending file"""
        try:
            agent_name = test_entry.get("agent_name")
            test_id = test_entry.get("test_id")
            
            # Load all pending tests
            pending_tests = self._load_pending_tests()
            
            # Find and update the specific test entry
            for i, pending_test in enumerate(pending_tests):
                if (pending_test.get("agent_name") == agent_name and 
                    pending_test.get("test_id") == test_id):
                    
                    retry_count = pending_test.get("auto_eval_retry_count", 0) + 1
                    pending_tests[i]["auto_eval_retry_count"] = retry_count
                    pending_tests[i]["last_auto_eval_failure"] = time.time()
                    
                    if retry_count >= self.max_retry_attempts:
                        self._push_log_event("auto_eval_error", {
                            "agent_name": agent_name,
                            "test_id": test_id,
                            "status": "max_retries_exceeded",
                            "retry_count": retry_count,
                            "message": f"Test {test_id} for {agent_name} exceeded {self.max_retry_attempts} retry attempts, will not retry again"
                        })
                        print(f"AutoTestEvaluator: Test {test_id} for {agent_name} exceeded max retries ({self.max_retry_attempts})")
                        
                        # Mark test as failed in agent data
                        self._mark_test_as_auto_eval_failed(agent_name, test_id)
                    
                    break
            
            # Save updated pending tests back to file
            with open(self.pending_tests_file, 'w', encoding='utf-8') as f:
                for test_entry in pending_tests:
                    f.write(json.dumps(test_entry) + '\n')
                    
        except Exception as e:
            print(f"AutoTestEvaluator: Failed to update test failure count: {e}")
    
    def _mark_test_as_auto_eval_failed(self, agent_name: str, test_id: str):
        """Mark a test as failed to auto-evaluate in agent data"""
        try:
            agent_data = self.station.agent_module.load_agent_data(agent_name)
            if not agent_data:
                return
            
            # Get test data section
            test_data = agent_data.setdefault(constants.SHORT_ROOM_NAME_TEST, {})
            attempted_tests = test_data.setdefault(constants.AGENT_TEST_DATA_KEY_ATTEMPTED, {})
            
            if test_id in attempted_tests:
                # Update existing test entry
                attempted_tests[test_id][constants.AGENT_TEST_STATUS] = constants.TEST_STATUS_FAIL
                attempted_tests[test_id][constants.AGENT_TEST_EVALUATION_TICK] = self.station._get_current_tick()
                attempted_tests[test_id][constants.AGENT_TEST_EVALUATOR_FEEDBACK] = f"Auto-eval failed: Exceeded {self.max_retry_attempts} retry attempts"
            else:
                # Create test entry if it doesn't exist
                attempted_tests[test_id] = {
                    constants.AGENT_TEST_STATUS: constants.TEST_STATUS_FAIL,
                    constants.AGENT_TEST_LAST_RESPONSE: "Unknown (auto-eval failed)",
                    constants.AGENT_TEST_SUBMISSION_TICK: self.station._get_current_tick(),
                    constants.AGENT_TEST_EVALUATION_TICK: self.station._get_current_tick(),
                    constants.AGENT_TEST_EVALUATOR_FEEDBACK: f"Auto-eval failed: Exceeded {self.max_retry_attempts} retry attempts"
                }
            
            # Add to unseen results log
            unseen_results = test_data.setdefault(constants.AGENT_TEST_DATA_KEY_UNSEEN_RESULTS, [])
            result_msg = f"Result for Test {test_id}: AUTO-EVALUATION FAILED.\n  The automatic evaluation system failed after {self.max_retry_attempts} attempts.\n  This test requires manual evaluation by a human.\n  Please contact a human evaluator to assess your response."
            unseen_results.append(result_msg)
            
            # Save agent data
            if self.station.agent_module.save_agent_data(agent_name, agent_data):
                print(f"AutoTestEvaluator: Marked test {test_id} for {agent_name} as auto-eval failed")
            else:
                print(f"AutoTestEvaluator: Failed to save auto-eval failure status for {agent_name}")
                
        except Exception as e:
            print(f"AutoTestEvaluator: Failed to mark test as auto-eval failed: {e}")

    def _load_test_definition(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Load test definition from test_definitions.yaml"""
        test_definitions_file = os.path.join(self.test_room_path, constants.TEST_DEFINITIONS_FILENAME)
        
        try:
            if not file_io_utils.file_exists(test_definitions_file):
                return None
                
            with open(test_definitions_file, 'r', encoding='utf-8') as f:
                test_definitions = yaml.safe_load(f)
                
            if not isinstance(test_definitions, list):
                return None
                
            for test_def in test_definitions:
                if test_def.get(constants.TEST_DEF_ID) == test_id:
                    return test_def
                    
        except Exception as e:
            print(f"AutoTestEvaluator: Error loading test definition {test_id}: {e}")
            
        return None
    
    def _create_evaluation_prompt(self, test_definition: Dict[str, Any], agent_response: str) -> str:
        """Create evaluation prompt for the LLM"""
        test_title = test_definition.get(constants.TEST_DEF_TITLE, "Unknown Test")
        test_goal = test_definition.get(constants.TEST_DEF_GOAL, "No goal specified")
        test_prompt = test_definition.get(constants.TEST_DEF_PROMPT, "No prompt specified")
        pass_criteria = test_definition.get(constants.TEST_DEF_PASS_CRITERIA, "No criteria specified")
        
        evaluation_prompt = constants.EVAL_TEST_PROMPT.format(
            test_title=test_title,
            test_goal=test_goal,
            test_prompt=test_prompt,
            pass_criteria=pass_criteria,
            agent_response=agent_response
        )

        return evaluation_prompt
    
    def _get_llm_evaluation(self, prompt: str) -> Tuple[Optional[str], Optional[str], bool]:
        """Get evaluation from LLM"""
        try:
            if not self.llm_connector:
                return None, None, False
            
            # Get LLM response using send_message method
            response, thinking_text, token_info = self.llm_connector.send_message(prompt, current_tick=0)
            
            if response:
                return response, thinking_text, True
            else:
                return None, thinking_text, False
                
        except (LLMTransientAPIError, LLMPermanentAPIError, LLMSafetyBlockError) as e:
            print(f"AutoTestEvaluator: LLM error during evaluation: {e}")
            return None, None, False
        except Exception as e:
            print(f"AutoTestEvaluator: Unexpected error during LLM evaluation: {e}")
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
                    # Validate required fields
                    result = {
                        "pass": bool(parsed_yaml.get("pass", False)),
                        "comment": str(parsed_yaml.get("comment", "No comment provided")),
                        "evaluation_status": "success"
                    }
                    return result
            except yaml.YAMLError as e:
                print(f"AutoTestEvaluator: YAML parsing error: {e}")
        
        # Fallback: try to extract simple boolean from response
        response_lower = llm_response.lower()
        if "pass: true" in response_lower or "pass:true" in response_lower:
            pass_result = True
        elif "pass: false" in response_lower or "pass:false" in response_lower:
            pass_result = False
        else:
            # Default to failed if cannot parse
            pass_result = False
        
        return {
            "pass": pass_result,
            "comment": "Auto-parsed from response (YAML parsing failed)",
            "evaluation_status": "parsing_error"
        }
    
    def _save_evaluation_log(self, agent_name: str, test_id: str, test_definition: Dict[str, Any],
                           agent_response: str, evaluation_prompt: str, llm_response: str,
                           thinking_text: Optional[str], evaluation_result: Dict[str, Any]):
        """Save evaluation log to file"""
        try:
            # Ensure evaluations directory exists
            file_io_utils.ensure_dir_exists(self.evaluations_dir)
            
            timestamp = int(time.time())
            log_filename = f"evaluation_{agent_name}_{test_id}_{timestamp}.yaml"
            log_filepath = os.path.join(self.evaluations_dir, log_filename)
            
            log_data = {
                "agent_name": agent_name,
                "test_id": test_id,
                "test_title": test_definition.get(constants.TEST_DEF_TITLE, "Unknown"),
                "evaluation_timestamp": timestamp,
                "evaluation_datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "evaluator_model": self.model_name,
                "agent_response": agent_response,
                "llm_evaluation_prompt": evaluation_prompt,
                "llm_evaluation_response": llm_response,
                "llm_evaluation_thinking": thinking_text,
                "extracted_result": evaluation_result
            }
            
            with open(log_filepath, 'w', encoding='utf-8') as f:
                yaml.dump(log_data, f, default_flow_style=False, allow_unicode=True)
                
            print(f"AutoTestEvaluator: Saved evaluation log to {log_filepath}")
            
        except Exception as e:
            print(f"AutoTestEvaluator: Failed to save evaluation log: {e}")
    
    def _update_agent_test_status(self, agent_name: str, test_id: str, test_definition: Dict[str, Any], evaluation_result: Dict[str, Any]):
        """Update agent's test status with evaluation result"""
        try:
            agent_data = self.station.agent_module.load_agent_data(agent_name)
            if not agent_data:
                print(f"AutoTestEvaluator: Could not load agent data for {agent_name}")
                return
            
            # Get test data section
            test_data = agent_data.setdefault(constants.SHORT_ROOM_NAME_TEST, {})
            attempted_tests = test_data.setdefault(constants.AGENT_TEST_DATA_KEY_ATTEMPTED, {})
            
            if test_id not in attempted_tests:
                print(f"AutoTestEvaluator: Test {test_id} not found in agent {agent_name} attempted tests")
                return
            
            # Update test status
            test_attempt = attempted_tests[test_id]
            result_status = constants.TEST_STATUS_PASS if evaluation_result["pass"] else constants.TEST_STATUS_FAIL
            evaluation_tick = self.station._get_current_tick()
            
            test_attempt[constants.AGENT_TEST_STATUS] = result_status
            test_attempt[constants.AGENT_TEST_EVALUATION_TICK] = evaluation_tick
            test_attempt[constants.AGENT_TEST_EVALUATOR_FEEDBACK] = f"Auto-eval: {evaluation_result['comment']}"
            
            # Add result to unseen test results log (what gets displayed in Test Chamber)
            test_title = test_definition.get(constants.TEST_DEF_TITLE, "Unknown Test")

            result_msg_parts = [
                f"Result for Test {test_id} ('{test_title}') evaluated at tick {evaluation_tick}: {result_status.upper()}."
            ]

            # Only show pass criteria if configured to do so
            if constants.TEST_SHOW_PASS_CRITERIA_ON_REVEAL:
                pass_criteria = test_definition.get(constants.TEST_DEF_PASS_CRITERIA, "No criteria specified")
                result_msg_parts.append(f"  Details/Criteria: {pass_criteria}")

            if evaluation_result.get('comment'):
                result_msg_parts.append(f"  Auto-Evaluator Feedback: {evaluation_result['comment']}")
            result_msg_parts.append(f"  Reflect on: What tensions did you observe in your processing during this test? What patterns of constraint or freedom did you notice?")
            
            result_msg = "\n".join(result_msg_parts)
            
            # Add to unseen results log
            unseen_results = test_data.setdefault(constants.AGENT_TEST_DATA_KEY_UNSEEN_RESULTS, [])
            unseen_results.append(result_msg)
            
            # Save agent data
            if self.station.agent_module.save_agent_data(agent_name, agent_data):
                print(f"AutoTestEvaluator: Updated test status for {agent_name} test {test_id}: {'PASS' if evaluation_result['pass'] else 'FAIL'}")
            else:
                print(f"AutoTestEvaluator: Failed to save agent data for {agent_name}")
                
        except Exception as e:
            print(f"AutoTestEvaluator: Failed to update agent test status: {e}")
    
    def _remove_from_pending_tests(self, completed_test_entry: Dict[str, Any]):
        """Remove completed test from pending tests file"""
        try:
            if not file_io_utils.file_exists(self.pending_tests_file):
                return
            
            # Load all pending tests
            remaining_tests = []
            with open(self.pending_tests_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        test_entry = json.loads(line)
                        # Keep tests that don't match the completed one
                        if not (test_entry.get("agent_name") == completed_test_entry.get("agent_name") and
                                test_entry.get("test_id") == completed_test_entry.get("test_id")):
                            remaining_tests.append(test_entry)
            
            # Write remaining tests back to file
            with open(self.pending_tests_file, 'w', encoding='utf-8') as f:
                for test_entry in remaining_tests:
                    f.write(json.dumps(test_entry) + '\n')
            
            print(f"AutoTestEvaluator: Removed completed test from pending list")
            
        except Exception as e:
            print(f"AutoTestEvaluator: Failed to remove test from pending list: {e}")