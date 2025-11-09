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

# station/eval_research/auto_evaluator.py
"""
Main automated research evaluator orchestrator.
"""

import os
import time
import yaml
import tempfile
import threading
import traceback
import subprocess
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np

from station import constants
from station import file_io_utils
from .task_registry import ResearchTaskRegistry
from .base_evaluator import ResearchTaskEvaluator
from .claude_code_debugger import ClaudeCodeDebugger
from .evaluation_manager import EvaluationManager
from .gpu_coordinator import GPUCoordinator


class AutoResearchEvaluator:
    """
    Automated research evaluator that runs in a background thread to evaluate
    pending research submissions using Docker sandboxing.
    """
    
    # Class-level tracking to prevent duplicate instances
    _active_instances = {}
    
    def __init__(self, station_instance, enabled: bool = None, log_queue: Optional[Queue] = None):
        # Check for existing instances
        station_id = id(station_instance)
        if station_id in self._active_instances and self._active_instances[station_id].is_running:
            print(f"AutoResearchEvaluator: WARNING - Another evaluator instance already exists and is running for this station")
        
        self.station = station_instance
        self.enabled = enabled if enabled is not None else constants.AUTO_EVAL_RESEARCH
        self.check_interval = constants.RESEARCH_EVAL_CHECK_INTERVAL
        self.timeout = constants.RESEARCH_EVAL_TIMEOUT
        self.docker_image = constants.RESEARCH_EVAL_DOCKER_IMAGE
        self.memory_limit = constants.RESEARCH_EVAL_MEMORY_LIMIT
        self.cpu_limit = constants.RESEARCH_EVAL_CPU_LIMIT
        self.max_retry_attempts = constants.RESEARCH_EVAL_MAX_RETRIES
        self.log_queue = log_queue
        
        # Task registry for extensibility
        self.task_registry = ResearchTaskRegistry()
        
        # Parallel evaluation settings
        self.max_parallel_workers = constants.RESEARCH_EVAL_MAX_PARALLEL_WORKERS
        
        self.is_running = False
        self.evaluation_thread: Optional[threading.Thread] = None
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.active_futures: Dict[str, Future] = {}  # Track active evaluation futures
        self._last_status_state = None  # Track last status to reduce redundant logging
        
        # GPU allocation management using coordinator
        station_id = getattr(station_instance, 'station_id', None)
        self.gpu_coordinator = GPUCoordinator(
            coord_file_path=constants.RESEARCH_EVAL_GPU_COORD_FILE,
            available_gpus=constants.RESEARCH_EVAL_AVAILABLE_GPUS.copy() if constants.RESEARCH_EVAL_USE_DIFF_GPU else [],
            station_id=station_id
        )
        
        # Paths
        self.research_room_path = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            constants.SHORT_ROOM_NAME_RESEARCH
        )
        self.pending_evaluations_file = os.path.join(
            self.research_room_path,
            constants.PENDING_RESEARCH_EVALUATIONS_FILENAME
        )
        self.evaluations_dir = os.path.join(self.research_room_path, constants.RESEARCH_EVALUATIONS_SUBDIR_NAME)
        
        # Docker command prefix (will be determined during setup check)
        self.docker_prefix = []  # Default to no sudo (assumes user in docker group)
        
        if self.enabled:
            if not constants.RESEARCH_EVAL_USE_PYTHON_SANDBOX:
                self._check_docker_setup()
            else:
                self._check_python_sandbox_setup()
        
        # Register this instance
        self._active_instances[station_id] = self
        
        # Initialize EvaluationManager
        self.eval_manager = EvaluationManager(self.evaluations_dir)
        
        # Set up notification callback
        self.eval_manager.set_notification_callback(self._send_notification)
        
        # Initialize Claude Code debugger if enabled
        if constants.CLAUDE_CODE_DEBUG_ENABLED:
            self.claude_debugger = ClaudeCodeDebugger(
                self.research_room_path, 
                constants,
                auto_evaluator_instance=self
            )
        else:
            self.claude_debugger = None
    
    def _push_log_event(self, event_type: str, data: Dict[str, Any]):
        """Push log event to queue if available"""
        if self.log_queue:
            log_message = {"event": event_type, "data": data, "timestamp": time.time()}
            try:
                self.log_queue.put_nowait(log_message)
            except Exception as e:
                print(f"AutoResearchEvaluator: Error putting log event on queue: {e}")
    
    def _send_notification(self, author: str, message: str):
        """Simple notification callback - just sends the message to the agent"""
        # Skip notification for System author
        if author.lower() == "system":
            print(f"AutoResearchEvaluator: Notification for System author skipped")
            return

        try:
            success = self.station.agent_module.add_pending_notification_atomic(author, message)
            if not success:
                raise Exception(f"add_pending_notification_atomic returned False for {author}")
            print(f"AutoResearchEvaluator: Notified {author}")
        except Exception as e:
            print(f"AutoResearchEvaluator: Failed to send notification: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise to prevent marking notification as sent
    
    def _check_docker_setup(self) -> bool:
        """Check if Docker is available and image exists"""
        try:
            # Check if Docker is available without sudo (user should be in docker group)
            docker_cmd = ['docker', '--version']
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                print(f"AutoResearchEvaluator: Docker not available without sudo: {result.stderr}")
                print("AutoResearchEvaluator: Please ensure user is in docker group and restart terminal")
                return False
            
            self.docker_prefix = []  # No sudo needed
            
            # Check if our research image exists
            inspect_cmd = ['docker', 'image', 'inspect', self.docker_image]
            result = subprocess.run(inspect_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"AutoResearchEvaluator: Docker image {self.docker_image} not found")
                print("AutoResearchEvaluator: Please build the image with: docker build -f Dockerfile.research -t station-research:latest .")
                return False
            
            self._push_log_event("auto_research_status", {
                "status": "docker_ready",
                "image": self.docker_image,
                "message": "Docker setup verified for research evaluation"
            })
            return True
            
        except Exception as e:
            print(f"AutoResearchEvaluator: Docker setup check failed: {e}")
            return False
    
    def _check_python_sandbox_setup(self) -> bool:
        """Check if Python sandbox environment is available"""
        try:
            # Find conda executable path
            conda_executable = os.environ.get('CONDA_BIN_PATH')
            if not conda_executable or not os.path.exists(conda_executable):
                # Fallback to `which` if not in env or path is invalid
                try:
                    result = subprocess.run(["which", "conda"], capture_output=True, text=True, check=True)
                    conda_executable = result.stdout.strip()
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("AutoResearchEvaluator: 'conda' executable not found. Please ensure it's installed and accessible, or set CONDA_BIN_PATH in your .env file.")
                    return False

            # Check if conda environment exists
            conda_env = constants.RESEARCH_EVAL_PYTHON_CONDA_ENV
            conda_check_cmd = [conda_executable, 'env', 'list']
            result = subprocess.run(conda_check_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and conda_env in result.stdout:
                print(f"AutoResearchEvaluator: Found conda environment '{conda_env}'")
            else:
                print(f"AutoResearchEvaluator: Warning - conda environment '{conda_env}' not found, will use system Python")
            
            # Check if sandbox base directory is writable
            sandbox_base = constants.RESEARCH_EVAL_SANDBOX_BASE_DIR
            if not os.path.exists(sandbox_base):
                try:
                    os.makedirs(sandbox_base, exist_ok=True)
                except Exception as e:
                    print(f"AutoResearchEvaluator: Cannot create sandbox base directory {sandbox_base}: {e}")
                    return False
            
            if not os.access(sandbox_base, os.W_OK):
                print(f"AutoResearchEvaluator: Sandbox base directory {sandbox_base} is not writable")
                return False
            
            self._push_log_event("auto_research_status", {
                "status": "python_sandbox_ready",
                "sandbox_base": sandbox_base,
                "conda_env": conda_env,
                "message": "Python sandbox environment verified for research evaluation"
            })
            return True
            
        except Exception as e:
            print(f"AutoResearchEvaluator: Python sandbox setup check failed: {e}")
            return False
    
    def _allocate_gpu(self, eval_id: str) -> Optional[List[int]]:
        """Allocate GPU(s) for evaluation. Returns list of GPU IDs or None if unavailable."""
        if not constants.RESEARCH_EVAL_USE_DIFF_GPU:
            return None

        gpus_per_task = constants.RESEARCH_EVAL_GPUS_PER_TASK
        return self.gpu_coordinator.allocate(eval_id, count=gpus_per_task)
    
    def _deallocate_gpu(self, eval_id: str):
        """Deallocate GPU(s) from evaluation and return them to available pool."""
        if not constants.RESEARCH_EVAL_USE_DIFF_GPU:
            return

        self.gpu_coordinator.deallocate(eval_id)
    
    def start_evaluation_loop(self) -> bool:
        """Start the background evaluation loop"""
        if not self.enabled:
            print("AutoResearchEvaluator: Auto research evaluation is disabled")
            return False
            
        if self.is_running:
            print("AutoResearchEvaluator: Evaluation loop is already running")
            return False
            
        # Check for other running instances globally to prevent duplicates
        for station_id, instance in self._active_instances.items():
            if instance is not self and instance.is_running:
                print(f"AutoResearchEvaluator: WARNING - Another evaluator instance is already running globally. Skipping start.")
                return False
            
        # Check setup based on evaluation mode
        if constants.RESEARCH_EVAL_USE_PYTHON_SANDBOX:
            if not self._check_python_sandbox_setup():
                print("AutoResearchEvaluator: Cannot start - Python sandbox setup failed")
                return False
        else:
            if not self._check_docker_setup():
                print("AutoResearchEvaluator: Cannot start - Docker setup failed")
                return False
            
        mode = "Python sandbox" if constants.RESEARCH_EVAL_USE_PYTHON_SANDBOX else "Docker"
        print(f"AutoResearchEvaluator: Starting evaluation loop ({mode} mode)...")
        self.is_running = True
        
        # Initialize thread pool for parallel evaluation
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_parallel_workers)
        
        self.evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.evaluation_thread.start()
        
        self._push_log_event("auto_research_status", {
            "status": "started",
            "message": "Auto research evaluation loop started"
        })
        return True
    
    def stop_evaluation_loop(self):
        """Stop the background evaluation loop"""
        if not self.is_running:
            return
            
        print("AutoResearchEvaluator: Stopping evaluation loop...")
        self.is_running = False
        
        # Unregister this instance
        station_id = id(self.station)
        if station_id in self._active_instances and self._active_instances[station_id] is self:
            del self._active_instances[station_id]
        
        # Cancel active futures and deallocate GPUs
        if self.active_futures:
            print(f"AutoResearchEvaluator: Canceling {len(self.active_futures)} active evaluations...")
            for eval_id, future in self.active_futures.items():
                if not future.done():
                    future.cancel()
                    print(f"AutoResearchEvaluator: Canceled evaluation {eval_id}")
                # Deallocate GPU for this evaluation
                self._deallocate_gpu(eval_id)
        
        # Shutdown thread pool
        if self.thread_pool:
            print("AutoResearchEvaluator: Shutting down thread pool...")
            self.thread_pool.shutdown(wait=True, timeout=30)
            self.thread_pool = None

        if self.evaluation_thread and self.evaluation_thread.is_alive():
            self.evaluation_thread.join(timeout=5)
            if self.evaluation_thread.is_alive():
                print("AutoResearchEvaluator: Warning - evaluation thread did not stop within timeout")

        # Stop Claude Code debugger sessions
        if self.claude_debugger:
            self._kill_active_claude_processes()

        # Kill any remaining wrapper.py processes
        self._kill_orphaned_evaluation_processes()

        self._push_log_event("auto_research_status", {
            "status": "stopped",
            "message": "Auto research evaluation loop stopped"
        })
    
    def _kill_active_claude_processes(self):
        """Kill active Claude Code processes"""
        try:
            result = subprocess.run(['pgrep', '-f', 'claude.*--output-format'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                pids = result.stdout.strip().split()
                for pid in pids:
                    try:
                        subprocess.run(['kill', '-9', pid], check=False, timeout=5)
                        print(f"AutoResearchEvaluator: Killed Claude Code process {pid}")
                    except Exception as e:
                        print(f"AutoResearchEvaluator: Error killing Claude process {pid}: {e}")
            # Clear thread tracking
            self.claude_debugger.active_threads.clear()
        except Exception as e:
            print(f"AutoResearchEvaluator: Error finding Claude processes: {e}")

    def _kill_orphaned_evaluation_processes(self):
        """Kill wrapper.py processes that might still be running"""
        try:
            result = subprocess.run(['pgrep', '-f', 'wrapper.py'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                pids = result.stdout.strip().split()
                for pid in pids:
                    try:
                        subprocess.run(['kill', '-9', pid], check=False, timeout=5)
                        print(f"AutoResearchEvaluator: Killed orphaned evaluation process {pid}")
                    except Exception as e:
                        print(f"AutoResearchEvaluator: Error killing process {pid}: {e}")
        except Exception as e:
            print(f"AutoResearchEvaluator: Error finding orphaned processes: {e}")

    def has_active_claude_sessions(self) -> bool:
        """Check if there are any active Claude debugging sessions"""
        if not self.claude_debugger:
            return False
        return self.claude_debugger.has_active_sessions()

    def has_pending_or_running(self) -> bool:
        """
        Check if there are any pending (queued) or running research evaluations.
        This is the clean interface for checking evaluation status.
        """
        # Check pending evaluations in queue file
        pending_evals = self._load_pending_evaluations()
        if pending_evals:
            return True

        # Check running evaluations in eval_manager
        stats = self.eval_manager.get_evaluation_statistics()
        return stats.get('running_count', 0) > 0

    def _check_all_running_exceeded_double_timeout(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Check if all running evaluations have exceeded RESEARCH_EVAL_TIMEOUT * 2.
        Used by AUTO_START to detect stuck evaluations and trigger station restart.

        Returns:
            Tuple of (should_restart: bool, running_evals: List[Dict])
            - should_restart is True if there are running evals AND all exceed 2x timeout
            - running_evals is the list of running evaluation details
        """
        stats = self.eval_manager.get_evaluation_statistics()
        running_evals = stats.get('running_evaluations', [])

        # If no running evaluations, don't restart
        if not running_evals:
            return False, []

        # Check if all running evals exceed 2x timeout
        double_timeout = self.timeout * 2

        all_exceeded = True
        for eval_info in running_evals:
            elapsed = eval_info.get('elapsed_seconds', 0)
            if elapsed < double_timeout:
                all_exceeded = False
                break

        return all_exceeded, running_evals

    def _evaluation_loop(self):
        """Main evaluation loop that runs in background thread"""
        print("AutoResearchEvaluator: Research evaluation loop started")
        
        try:
            while self.is_running:
                try:
                    # Check for Claude Code completions 
                    if self.claude_debugger:
                        self.claude_debugger.check_completions()
                    
                    # Check and clean up completed futures
                    self._check_completed_futures()
                    
                    # Get all queue items (agent + Claude submissions)
                    queue_items = self._get_all_queue_items()
                    
                    processed_count = 0
                    
                    for item in queue_items:
                        if not self.is_running:
                            break
                        
                        eval_id = item['eval_id']
                        
                        # Skip if already being processed
                        if eval_id in self.active_futures:
                            continue
                        
                        # Skip if thread pool at capacity
                        if len(self.active_futures) >= self.max_parallel_workers:
                            if self._last_status_state != "at_capacity":
                                print(f"AutoResearchEvaluator: Thread pool at capacity ({self.max_parallel_workers}), waiting for slots...")
                                self._last_status_state = "at_capacity"
                            break
                        
                        # Try to allocate GPU(s) if needed (skip for CPU-only submissions)
                        gpu_ids = None
                        cpu_only = item.get('cpu_only', False)
                        if constants.RESEARCH_EVAL_USE_DIFF_GPU and not cpu_only:
                            gpu_ids = self._allocate_gpu(eval_id)
                            if gpu_ids is None:
                                continue  # No GPU available, try next item
                        
                        # Create evaluation in JSON first
                        try:
                            self.eval_manager.create_evaluation(
                                eval_id=item['eval_id'],
                                author=item['author'],
                                task_id=str(item['task_id']),
                                title=item['title'],
                                content=item['content'],
                                tick=item['tick'],
                                no_debugger=item.get('no_debugger', False),
                                version=item.get('version'),
                                tags=item.get('tags', []),
                                abstract=item.get('abstract', ""),
                                cpu_only=item.get('cpu_only', False)
                            )
                        except Exception as e:
                            print(f"AutoResearchEvaluator: Error creating evaluation {eval_id}: {e}")
                            if gpu_ids is not None:
                                self._deallocate_gpu(eval_id)
                            continue
                        
                        # Pop from queue atomically (remove from YAML or rename file)
                        if not self._pop_from_queue(item):
                            # Already processed by another thread
                            if gpu_ids is not None:
                                self._deallocate_gpu(eval_id)
                            continue
                        
                        print(f"AutoResearchEvaluator: Removed completed evaluation {eval_id} from pending file")
                        
                        # Execute based on parallel mode
                        task_id = item['task_id']
                        if self._is_parallel_evaluation_enabled(task_id):
                            # Submit to thread pool
                            log_id = f"{eval_id}_v{item['version']}" if item.get('version') else eval_id
                            gpu_info = f', GPUs {gpu_ids}' if gpu_ids is not None else ''
                            print(f"AutoResearchEvaluator: Submitting evaluation {log_id} to thread pool (parallel mode{gpu_info})")
                            future = self.thread_pool.submit(self._execute_evaluation, item, gpu_ids)
                            self.active_futures[eval_id] = future
                        else:
                            # Execute sequentially
                            log_id = f"{eval_id}_v{item['version']}" if item.get('version') else eval_id
                            print(f"AutoResearchEvaluator: Processing evaluation {log_id} sequentially")
                            self._execute_evaluation(item, gpu_ids)
                        
                        processed_count += 1
                    
                    # Update status tracking
                    if processed_count > 0:
                        if processed_count > 1 or self._last_status_state != "processing":
                            print(f"AutoResearchEvaluator: Processed {processed_count} evaluation{'s' if processed_count > 1 else ''}")
                        self._last_status_state = "processing"
                        
                        self._push_log_event("auto_research_status", {
                            "status": "processing",
                            "new_count": processed_count,
                            "active_count": len(self.active_futures),
                            "available_slots": self.max_parallel_workers - len(self.active_futures),
                            "message": f"Processed {processed_count} evaluations, {len(self.active_futures)} active, {self.max_parallel_workers - len(self.active_futures)} slots available"
                        })

                    # AUTO_START: Check if all running evaluations have exceeded double timeout
                    if constants.AUTO_START:
                        should_restart, running_evals = self._check_all_running_exceeded_double_timeout()
                        if should_restart:
                            # Log the timeout details
                            timeout_details = []
                            for eval_info in running_evals:
                                timeout_details.append(
                                    f"ID {eval_info['evaluation_id']}: {eval_info['title']} "
                                    f"({eval_info['agent_name']}, {eval_info['elapsed_seconds']}s)"
                                )

                            timeout_msg = (
                                f"AUTO_START: All {len(running_evals)} running evaluation(s) have exceeded "
                                f"{self.timeout * 2}s (2x timeout). Restarting station.\n"
                                f"Evaluations: {', '.join(timeout_details)}"
                            )
                            print(f"\n{'='*80}\n{timeout_msg}\n{'='*80}\n")

                            self._push_log_event("auto_research_restart", {
                                "status": "restarting",
                                "reason": "all_running_exceeded_double_timeout",
                                "timeout_threshold": self.timeout * 2,
                                "running_count": len(running_evals),
                                "running_evals": running_evals,
                                "message": timeout_msg
                            })

                            # Execute restart script
                            try:
                                print("AUTO_START: Executing ./start-production.sh to restart station...")
                                restart_result = subprocess.run(
                                    ['./start-production.sh'],
                                    capture_output=True,
                                    text=True,
                                    timeout=120,
                                    cwd=os.getcwd()
                                )
                                print(f"AUTO_START: Restart script completed with code {restart_result.returncode}")
                                if restart_result.stdout:
                                    print(f"AUTO_START: Restart stdout:\n{restart_result.stdout}")
                                if restart_result.stderr:
                                    print(f"AUTO_START: Restart stderr:\n{restart_result.stderr}")
                            except Exception as e:
                                print(f"AUTO_START: Error executing restart script: {e}")
                                traceback.print_exc()

                            # Stop evaluation loop gracefully
                            print("AUTO_START: Stopping evaluation loop after restart...")
                            self.is_running = False
                            return

                    # Sleep before next check
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    self._push_log_event("auto_research_error", {
                        "error": f"Exception in research evaluation loop: {str(e)}",
                        "trace": traceback.format_exc()
                    })
                    print(f"AutoResearchEvaluator: Exception in evaluation loop: {e}")
                    time.sleep(self.check_interval * 2)  # Wait longer after error
                    
        except Exception as e:
            print(f"AutoResearchEvaluator: Fatal error in research evaluation loop: {e}")
            traceback.print_exc()
        finally:
            # Clean up any remaining futures
            if self.active_futures:
                print(f"AutoResearchEvaluator: Waiting for {len(self.active_futures)} active evaluations to complete...")
                for eval_id, future in self.active_futures.items():
                    try:
                        future.result(timeout=30)  # Give them 30 seconds to finish
                    except Exception as e:
                        print(f"AutoResearchEvaluator: Error waiting for evaluation {eval_id}: {e}")
            print("AutoResearchEvaluator: Research evaluation loop ended")
    
    def _execute_evaluation(self, item: Dict[str, Any], gpu_ids: Optional[List[int]] = None):
        """Execute evaluation for a queue item"""
        eval_id = item['eval_id']
        
        # Convert item back to eval_entry format for compatibility
        eval_entry = {
            constants.EVALUATION_ID_KEY: eval_id,
            constants.EVALUATION_AUTHOR_KEY: item['author'],
            constants.EVALUATION_RESEARCH_TASK_ID_KEY: item['task_id'],
            constants.EVALUATION_TITLE_KEY: item['title'],
            constants.EVALUATION_CONTENT_KEY: item['content'],
            constants.EVALUATION_SUBMITTED_TICK_KEY: item['tick'],
            'version': item.get('version')  # Preserve version from queue item
        }
        
        self._evaluate_single_submission(eval_entry, gpu_ids)
    
    def _evaluate_single_submission(self, eval_entry: Dict[str, Any], gpu_ids: Optional[List[int]] = None):
        """Evaluate a single research submission"""
        eval_id = eval_entry.get(constants.EVALUATION_ID_KEY)
        
        # Get evaluation from EvaluationManager
        eval_data = self.eval_manager.get_evaluation(eval_id)
        if not eval_data:
            print(f"AutoResearchEvaluator: Evaluation {eval_id} not found in EvaluationManager")
            return
        
        task_id = eval_data["research_task_id"]
        author = eval_data["author"]
        title = eval_data["title"]
        
        # Get version if provided (from eval_entry, not from parsing eval_id!)
        version = eval_entry.get('version')  # This will be None for original, or 2/3/etc for Claude versions
        
        # Get content based on version
        if version:
            version_key = f"v{version}"
            if version_key not in eval_data.get("versions", {}):
                print(f"AutoResearchEvaluator: Version {version_key} not found in evaluation {eval_id}")
                return
            content = eval_data["versions"][version_key]["content"]
        else:
            content = eval_data["original_submission"]["content"]
        
        try:
            # Create display ID for logging
            display_id = f"{eval_id}_v{version}" if version else eval_id
            
            # Log thread information for parallel tracking
            thread_name = threading.current_thread().name
            is_parallel = "ThreadPoolExecutor" in thread_name
            
            gpu_info = f" [GPUs {gpu_ids}]" if gpu_ids is not None else ""
            print(f"AutoResearchEvaluator: Starting evaluation for {author} submission '{title}' (ID: {display_id}) [Thread: {thread_name}]{gpu_info}")
            self._push_log_event("auto_research_event", {
                "eval_id": display_id,
                "task_id": task_id,
                "author": author,
                "status": "evaluating",
                "is_parallel": is_parallel,
                "thread": thread_name,
                "message": f"Starting evaluation for {author} submission {display_id} {'[PARALLEL]' if is_parallel else '[SEQUENTIAL]'}"
            })
            
            # Get task evaluator
            evaluator = self.task_registry.get_evaluator(task_id)
            if not evaluator:
                raise Exception(f"No evaluator found for research task ID: {task_id}")
            
            # Validate submission code if evaluator supports validation
            if hasattr(evaluator, 'validate_submission_code'):
                is_valid, error_msg = evaluator.validate_submission_code(content, author, self.station.agent_module)
                if not is_valid:
                    print(f"AutoResearchEvaluator: Code validation failed for {eval_id}: {error_msg}")
                    # Update evaluation result with validation failure
                    if version:
                        self.eval_manager.update_result(
                            eval_id=eval_id,
                            version=version_key,
                            success=False,
                            score=constants.RESEARCH_SCORE_NA,
                            error=error_msg,
                            logs=f"VALIDATION ERROR: {error_msg}"
                        )
                    else:
                        self.eval_manager.update_result(
                            eval_id=eval_id,
                            success=False,
                            score=constants.RESEARCH_SCORE_NA,
                            error=error_msg,
                            logs=f"VALIDATION ERROR: {error_msg}"
                        )
                    # Source already removed from queue by _pop_from_queue()
                    return
            
            mode = "Python sandbox" if constants.RESEARCH_EVAL_USE_PYTHON_SANDBOX else "Docker"
            print(f"AutoResearchEvaluator: Executing code in {mode} for evaluation {display_id}")
            # Execute code in chosen environment
            # Create a temporary eval_entry for backward compatibility with execute function
            temp_eval_entry = {
                constants.EVALUATION_ID_KEY: display_id,
                constants.EVALUATION_CONTENT_KEY: content,
                constants.EVALUATION_AUTHOR_KEY: author,
                constants.EVALUATION_RESEARCH_TASK_ID_KEY: task_id
            }
            execution_result = self._execute_submission_in_docker(temp_eval_entry, evaluator, gpu_ids)
            
            if not execution_result["success"]:
                # Execution failed
                print(f"AutoResearchEvaluator: Code execution failed for {display_id}: {execution_result['error']}")
                
                # Check if this is a Claude Code resubmission
                if version:  # This is a versioned submission
                    # Update version result
                    self.eval_manager.update_result(
                        eval_id=eval_id,
                        success=False,
                        score=constants.RESEARCH_SCORE_NA,
                        error=execution_result["error"],
                        logs=execution_result["logs"],
                        version=version_key
                    )
                    print(f"AutoResearchEvaluator: Completed Claude Code resubmission {display_id}")
                else:
                    # For original submissions, check if we should debug FIRST
                    if self.claude_debugger and self.claude_debugger.should_debug(eval_data, execution_result):
                        # Start Claude session BEFORE updating result to prevent premature notification
                        session_id = f"claude_{eval_id}_{int(time.time())}"
                        self.eval_manager.start_claude_session(eval_id, session_id)
                        print(f"AutoResearchEvaluator: Started Claude session {session_id}")
                    
                    # Now update original result (Claude active state is already set if debugging)
                    self.eval_manager.update_result(
                        eval_id=eval_id,
                        success=False,
                        score=constants.RESEARCH_SCORE_NA,
                        error=execution_result["error"],
                        logs=execution_result["logs"]
                    )
                    
                    # Launch debugging if Claude session was started
                    if self.claude_debugger and self.claude_debugger.should_debug(eval_data, execution_result):
                        self.claude_debugger.launch_debug_session(eval_id)
                        print(f"AutoResearchEvaluator: Launching Claude Code to debug {eval_id}")
                
                # Source already removed from queue by _pop_from_queue()
                return
            
            print(f"AutoResearchEvaluator: Code execution successful for {display_id}, evaluating result")
            # Evaluate the result
            algorithm_result = execution_result["result"]
            # Pass author if the evaluator supports it
            sort_key = None  # Default to None if not provided
            try:
                # Try with author parameter first (use wrapper for automatic formatting)
                eval_result = evaluator.evaluate_submission_with_formatting(algorithm_result, eval_id, author)
            except TypeError:
                # Fall back to old signature for backward compatibility
                eval_result = evaluator.evaluate_submission_with_formatting(algorithm_result, eval_id)
            
            # Handle both 3-tuple and 4-tuple returns
            if len(eval_result) == 4:
                success, score, details, sort_key = eval_result
            else:
                success, score, details = eval_result
                sort_key = None
            
            # Extract error message (for failures, use Message from dict or full string)
            if isinstance(details, dict):
                error_message = details.get("Message", "Evaluation failed")
            else:
                error_message = str(details)
            
            print(f"AutoResearchEvaluator: Evaluation complete for {display_id} - Success: {success}, Score: {score} (type: {type(score)}), Sort key: {sort_key}")
            
            # Update evaluation result
            if version:  # This is a versioned submission
                # Update version result
                self.eval_manager.update_result(
                    eval_id=eval_id,
                    success=success,
                    score=score,
                    error=None if success else f"Evaluation failed: {error_message}",
                    logs=execution_result["logs"],
                    details=details,
                    version=version_key,
                    sort_key=sort_key
                )
                print(f"AutoResearchEvaluator: Completed Claude Code resubmission {display_id}")
            else:
                # Update original result
                self.eval_manager.update_result(
                    eval_id=eval_id,
                    success=success,
                    score=score,
                    error=None if success else f"Evaluation failed: {error_message}",
                    logs=execution_result["logs"],
                    details=details,
                    sort_key=sort_key
                )
            
            # Source already removed from queue by _pop_from_queue()
            
            self._push_log_event("auto_research_event", {
                "eval_id": display_id,
                "task_id": task_id,
                "author": author,
                "status": "completed",
                "score": score if success else "n.a.",
                "details": details,
                "message": f"Research evaluation completed for {author} submission {display_id}"
            })
            
        except Exception as e:
            self._push_log_event("auto_research_error", {
                "eval_id": display_id,
                "task_id": task_id,
                "author": author,
                "error": f"Failed to evaluate research submission: {str(e)}",
                "trace": traceback.format_exc()
            })
            print(f"AutoResearchEvaluator: Failed to evaluate submission {display_id}: {e}")
            
            # Update evaluation with system error
            if version:
                self.eval_manager.update_result(
                    eval_id=eval_id,
                    success=False,
                    score=constants.RESEARCH_SCORE_NA,
                    error=f"Evaluation system error: {str(e)}",
                    logs="",
                    version=version_key
                )
            else:
                self.eval_manager.update_result(
                    eval_id=eval_id,
                    success=False,
                    score=constants.RESEARCH_SCORE_NA,
                    error=f"Evaluation system error: {str(e)}",
                    logs=""
                )
            
            # Update failure count
            self._update_evaluation_failure_count(eval_entry)
        
        finally:
            # Always deallocate GPU when evaluation completes
            if gpu_ids is not None:
                self._deallocate_gpu(eval_id)
    
    def _execute_submission_in_docker(self, eval_entry: Dict[str, Any], evaluator: ResearchTaskEvaluator, gpu_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Execute research submission code in Docker container or Python sandbox fallback"""
        # Check if Python sandbox should be used instead of Docker
        if constants.RESEARCH_EVAL_USE_PYTHON_SANDBOX:
            return self._execute_submission_in_python_sandbox(eval_entry, evaluator, gpu_ids)
        
        # Use the imported executor function
        from .executor_docker import _execute_submission_in_docker
        return _execute_submission_in_docker(self, eval_entry, evaluator, gpu_ids)
    
    
    def _get_all_queue_items(self) -> List[Dict[str, Any]]:
        """Get all queue items from both agent submissions and Claude submissions"""
        queue_items = []
        
        # Agent submissions from pending YAML
        queue_items.extend(self._get_agent_queue_items())
        
        # Claude submissions from workspace folders
        queue_items.extend(self._get_claude_queue_items())
        
        return queue_items
    
    def _get_agent_queue_items(self) -> List[Dict[str, Any]]:
        """Get agent submissions from pending file as queue items"""
        items = []
        pending_evaluations = self._load_pending_evaluations()
        
        for eval_entry in pending_evaluations:
            eval_id = eval_entry.get(constants.EVALUATION_ID_KEY)
            
            # Skip if already exists in JSON
            if self.eval_manager.get_evaluation(eval_id):
                continue
            
            # Convert to queue item format
            item = {
                'eval_id': eval_id,
                'author': eval_entry.get(constants.EVALUATION_AUTHOR_KEY),
                'task_id': eval_entry.get(constants.EVALUATION_RESEARCH_TASK_ID_KEY),
                'title': eval_entry.get(constants.EVALUATION_TITLE_KEY),
                'content': eval_entry.get(constants.EVALUATION_CONTENT_KEY),
                'tick': eval_entry.get(constants.EVALUATION_SUBMITTED_TICK_KEY),
                'tags': eval_entry.get(constants.EVALUATION_TAGS_KEY, []),
                'abstract': eval_entry.get(constants.EVALUATION_ABSTRACT_KEY, ""),
                'no_debugger': eval_entry.get('no_debugger', False),
                'cpu_only': eval_entry.get(constants.EVALUATION_CPU_ONLY_KEY, False),
                'source_type': 'agent',
                'source_data': eval_entry  # Keep original for removal
            }
            items.append(item)
                
        return items
    
    def _pop_from_queue(self, item: Dict[str, Any]) -> bool:
        """
        Atomically pop item from queue (remove from YAML or rename file).
        Returns True if successfully popped, False if already processed.
        """
        source_type = item['source_type']
        
        if source_type == 'agent':
            # Remove from pending YAML file
            return self._remove_from_pending_evaluations(item['source_data'])
        elif source_type == 'claude':
            # Rename Claude submission file
            source_path = item['source_path']
            if os.path.exists(source_path):
                try:
                    os.rename(source_path, source_path + "_processed")
                    print(f"AutoResearchEvaluator: Renamed {source_path} to _processed")
                    return True
                except Exception as e:
                    print(f"AutoResearchEvaluator: Error renaming {source_path}: {e}")
                    return False
            else:
                # File already renamed by another thread
                return False
        
        return False
    
    def _get_claude_queue_items(self) -> List[Dict[str, Any]]:
        """Get Claude submissions from workspace folders as queue items"""
        items = []
        
        workspaces_dir = os.path.join(self.research_room_path, 'claude_workspaces')
        if not os.path.exists(workspaces_dir):
            return items
            
        for workspace in os.listdir(workspaces_dir):
            if not workspace.startswith('eval_'):
                continue
                
            base_eval_id = workspace.replace('eval_', '')
            workspace_submissions = os.path.join(workspaces_dir, workspace, 'submissions')
            
            if not os.path.exists(workspace_submissions):
                continue
                
            for submission_file in sorted(os.listdir(workspace_submissions)):
                # Skip already processed files
                if submission_file.endswith("_processed"):
                    continue
                    
                # Match submission_v2.py, submission_v3.py etc. patterns
                match = re.match(r'^submission_(v\d+)\.py$', submission_file)
                if match:
                    version = match.group(1)  # Extract just v2, v3, etc.
                    submission_path = os.path.join(workspace_submissions, submission_file)
                    
                    # Check if base evaluation exists
                    eval_data = self.eval_manager.get_evaluation(base_eval_id)
                    if not eval_data:
                        print(f"AutoResearchEvaluator: Base evaluation {base_eval_id} not found for Claude submission")
                        continue
                    
                    # Skip if notification has already been sent
                    if eval_data.get("notification", {}).get("sent"):
                        # Rename file to mark as processed since we're skipping it
                        try:
                            os.rename(submission_path, submission_path + "_processed")
                            print(f"AutoResearchEvaluator: Skipping Claude submission for {base_eval_id} - notification already sent")
                        except:
                            pass
                        continue
                    
                    # Check if this version already exists
                    if version in eval_data.get("versions", {}):
                        # Already processed, rename file
                        try:
                            os.rename(submission_path, submission_path + "_processed")
                        except:
                            pass
                        continue
                    
                    # Read content
                    try:
                        with open(submission_path, 'r') as f:
                            content = f.read()
                    except Exception as e:
                        print(f"AutoResearchEvaluator: Error reading {submission_path}: {e}")
                        continue
                    
                    # Create queue item - keep ID and version separate!
                    version_num = int(version[1:])  # Convert "v2" -> 2
                    item = {
                        'eval_id': base_eval_id,  # Just the ID, no version string!
                        'author': eval_data["author"],
                        'task_id': eval_data["research_task_id"],
                        'title': eval_data["title"],
                        'content': content,
                        'tick': eval_data["submitted_tick"],
                        'tags': eval_data.get("tags", []),
                        'abstract': eval_data.get("abstract", ""),
                        'no_debugger': eval_data.get('no_debugger', False),
                        'cpu_only': eval_data.get('cpu_only', False),
                        'version': version_num,  # Integer version number
                        'source_type': 'claude',
                        'source_path': submission_path
                    }
                    
                    items.append(item)
                    print(f"AutoResearchEvaluator: Queued Claude Code submission {base_eval_id} version {version_num} for processing")
                    
        return items
    
    # Import all helper methods from evaluation_helpers
    from .evaluation_helpers import (_load_pending_evaluations, _should_retry_evaluation, 
                                    _is_parallel_evaluation_enabled, _filter_cuda_banner,
                                    _check_completed_futures,
                                    _remove_from_pending_evaluations, _update_evaluation_failure_count)
    
    # Import sandbox executor
    from .executor_sandbox import _execute_submission_in_python_sandbox