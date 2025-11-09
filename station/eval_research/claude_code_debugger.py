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

# station/eval_research/claude_code_debugger.py
"""
Claude Code debugger for automatically fixing failed research submissions.
"""

import os
import time
import threading
import subprocess
import shutil
import re
import glob
from typing import Dict, Any, Optional, Tuple

from station import file_io_utils
from station import constants


class ClaudeCodeDebugger:
    """Manages Claude Code debugging sessions for failed evaluations"""
    
    def __init__(self, research_room_path: str, constants_module=None, auto_evaluator_instance=None):
        """
        Initialize Claude Code debugger.
        
        Args:
            research_room_path: Path to research room data directory
            constants_module: Constants module (defaults to station.constants)
            auto_evaluator_instance: Reference to auto evaluator for notifications
        """
        self.research_room_path = research_room_path
        self.constants = constants_module or constants
        self.auto_evaluator = auto_evaluator_instance
        self.claude_workspaces_path = os.path.join(research_room_path, 'claude_workspaces')
        self.active_threads = []  # Track active debugging threads
        self.rate_limiter = threading.Semaphore(self.constants.CLAUDE_CODE_DEBUG_MAX_CONCURRENT)
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs(self.claude_workspaces_path, exist_ok=True)
        os.makedirs(os.path.join(self.claude_workspaces_path, 'history'), exist_ok=True)
        
    def should_debug(self, eval_data: Dict[str, Any], execution_result: Dict[str, Any]) -> bool:
        """
        Determine if this failure should be debugged.
        
        Args:
            eval_data: Evaluation data from EvaluationManager
            execution_result: Result from execution with 'success', 'error', 'logs' keys
            
        Returns:
            True if Claude Code should debug this failure
        """
        if not self.constants.CLAUDE_CODE_DEBUG_ENABLED:
            return False
        
        # Check if agent opted out of debugging
        if eval_data.get('no_debugger', False):
            return False
        
        # Check if execution timed out
        error_msg = execution_result.get('error', '')
        if error_msg.startswith('Execution timed out after'):
            return False  # Don't debug timeouts - code might be correct but just slow
            
        # Debug all other errors (syntax errors, import errors, runtime errors, etc.)
        return True
        
    def launch_debug_session(self, eval_id: str):
        """
        Launch Claude Code subprocess for debugging.
        
        Args:
            eval_id: Evaluation ID to debug
        """
        # Launch in separate thread to avoid blocking
        thread = threading.Thread(
            target=self._run_debug_session,
            args=(eval_id,),
            daemon=True
        )
        thread.start()
        # Track active thread
        self.active_threads.append(thread)
        # Clean up completed threads
        self.active_threads = [t for t in self.active_threads if t.is_alive()]
        # Return the thread so caller can wait if needed
        return thread
        
    def _get_lineage_name(self, agent_name: str) -> str:
        """
        Extract lineage name from agent name.
        
        Args:
            agent_name: Full agent name (e.g., "Alice_R3", "Praxis I")
            
        Returns:
            Lineage name (e.g., "Alice", "praxis")
        """
        # Agent names are like "Alice_R3" or "Bob_Guest" 
        # But could also be "Praxis I" or similar
        # Lineage is the base name before underscore
        if '_' in agent_name:
            lineage = agent_name.split('_')[0]
        else:
            # For names like "Praxis I", take the first word
            lineage = agent_name.split()[0] if ' ' in agent_name else agent_name
        return lineage
        
    def _setup_workspace(self, eval_id: str, eval_data: Dict[str, Any]) -> str:
        """
        Set up isolated workspace for Claude Code.
        
        Args:
            eval_id: Evaluation ID
            eval_data: Evaluation data from EvaluationManager
            
        Returns:
            Path to workspace directory
            
        Raises:
            Exception: If workspace setup fails
        """
        workspace_dir = os.path.join(self.claude_workspaces_path, f"eval_{eval_id}")
        
        try:
            # Create workspace structure
            os.makedirs(os.path.join(workspace_dir, "submissions"), exist_ok=True)
            os.makedirs(os.path.join(workspace_dir, "tmp"), exist_ok=True)
            os.makedirs(os.path.join(workspace_dir, "storage"), exist_ok=True)
            os.makedirs(os.path.join(workspace_dir, "scripts"), exist_ok=True)
            
            # Create evaluation YAML in old format for backward compatibility with Claude prompt
            eval_yaml_data = {
                self.constants.EVALUATION_ID_KEY: eval_id,
                self.constants.EVALUATION_AUTHOR_KEY: eval_data["author"],
                self.constants.EVALUATION_TITLE_KEY: eval_data["title"],
                self.constants.EVALUATION_RESEARCH_TASK_ID_KEY: eval_data["research_task_id"],
                self.constants.EVALUATION_SUBMITTED_TICK_KEY: eval_data["submitted_tick"],
                self.constants.EVALUATION_CONTENT_KEY: eval_data["original_submission"]["content"],
                self.constants.EVALUATION_LOGS_KEY: eval_data["original_submission"]["evaluation_result"].get("logs", ""),
                "error": eval_data["original_submission"]["evaluation_result"].get("error", "")
            }
            eval_path = os.path.join(workspace_dir, "evaluation.yaml")
            file_io_utils.save_yaml(eval_yaml_data, eval_path)
            
            # Get lineage name for monitor script
            author_name = eval_data["author"]
            lineage_name = self._get_lineage_name(author_name)
            
            # Create monitor script in workspace root
            monitor_script = os.path.join(workspace_dir, "monitor_evaluation.py")
            self._create_monitor_script(monitor_script, eval_id, lineage_name)
            
            # Set up storage symlinks
            
            # Read-only symlinks for shared and system
            shared_target = os.path.abspath(os.path.join(
                self.research_room_path, 
                self.constants.RESEARCH_STORAGE_DIR, 
                self.constants.RESEARCH_STORAGE_SHARED_DIR
            ))
            system_target = os.path.abspath(os.path.join(
                self.research_room_path, 
                self.constants.RESEARCH_STORAGE_DIR, 
                self.constants.RESEARCH_STORAGE_SYSTEM_DIR
            ))
            
            if os.path.exists(shared_target):
                os.symlink(shared_target, os.path.join(workspace_dir, "storage/shared"), target_is_directory=True)
            if os.path.exists(system_target):
                # Create symlink but ensure the source directory has proper read-only permissions
                # First, verify and fix permissions on the actual system directory
                if os.stat(system_target).st_mode & 0o200:  # Check if writable
                    print(f"ClaudeCodeDebugger: Warning - system directory has write permissions, fixing...")
                    # Fix permissions on the real system directory
                    for root, dirs, files in os.walk(system_target):
                        for d in dirs:
                            os.chmod(os.path.join(root, d), 0o555)  # r-xr-xr-x for dirs
                        for f in files:
                            os.chmod(os.path.join(root, f), 0o444)  # r--r--r-- for files
                    os.chmod(system_target, 0o555)
                
                # Now create the symlink
                os.symlink(system_target, os.path.join(workspace_dir, "storage/system"), target_is_directory=True)
            
            # Read-write symlink for author's lineage (normalize to lowercase)
            lineage_name_normalized = lineage_name.lower()
            lineage_path = os.path.join(
                self.research_room_path, 
                self.constants.RESEARCH_STORAGE_DIR,
                self.constants.RESEARCH_STORAGE_LINEAGES_DIR,
                lineage_name_normalized
            )
            if os.path.exists(lineage_path):
                # Create symlink as storage/praxis instead of storage/lineage
                symlink_target = os.path.join(workspace_dir, "storage", lineage_name_normalized)
                os.symlink(
                    os.path.abspath(lineage_path),
                    symlink_target,
                    target_is_directory=True
                )
            
            return workspace_dir
        except Exception as e:
            # Clean up partially created workspace
            if os.path.exists(workspace_dir):
                # Skip cleanup for now
                # self._cleanup_workspace(workspace_dir)
                pass
            raise
    
    def _create_monitor_script(self, script_path: str, eval_id: str, lineage_name: str):
        """
        Create the monitor script for Claude Code to use.
        
        Args:
            script_path: Path where to create the script
            eval_id: Evaluation ID to monitor
            lineage_name: Name of the agent's lineage
        """
        # Read template
        template_path = os.path.join(os.path.dirname(__file__), 'monitor_evaluation_template.py')
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Replace placeholders
        script_content = template_content.replace('{EVAL_ID}', str(eval_id))
        script_content = script_content.replace('{TIMEOUT}', str(self.constants.CLAUDE_CODE_MONITOR_TIMEOUT))
        script_content = script_content.replace('{LINEAGE_NAME}', lineage_name.lower())
        
        # Write the customized script
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        
    def _run_debug_session(self, eval_id: str):
        """
        Run debugging session in thread.
        
        Args:
            eval_id: Evaluation ID to debug
        """
        with self.rate_limiter:  # Rate limiting
            try:
                # Get evaluation from EvaluationManager
                eval_data = self.auto_evaluator.eval_manager.get_evaluation(eval_id)
                if not eval_data:
                    print(f"ClaudeCodeDebugger: Could not load evaluation {eval_id} from EvaluationManager")
                    return
                
                # Get lineage name
                agent_name = eval_data["author"]
                lineage_name = self._get_lineage_name(agent_name).lower()
                
                # Set up workspace
                workspace_dir = self._setup_workspace(eval_id, eval_data)
                
                session_id = f'claude_{eval_id}_{int(time.time())}'
                
                # Build prompt with specific eval_id and lineage name
                prompt = self._build_prompt(eval_id, lineage_name)
                
                # Prepare environment - optionally remove API key to use standard auth
                env = os.environ.copy()
                if self.constants.CLAUDE_CODE_USE_STANDARD_AUTH:
                    # Remove API key to force standard Claude.ai authentication (often cheaper)
                    env.pop('ANTHROPIC_API_KEY', None)
                
                # Ensure proxy settings from constants are applied
                if self.constants.LLM_HTTP_PROXY:
                    env['http_proxy'] = self.constants.LLM_HTTP_PROXY
                    env['HTTP_PROXY'] = self.constants.LLM_HTTP_PROXY  # Some tools check uppercase
                if self.constants.LLM_HTTPS_PROXY:
                    env['https_proxy'] = self.constants.LLM_HTTPS_PROXY
                    env['HTTPS_PROXY'] = self.constants.LLM_HTTPS_PROXY  # Some tools check uppercase
                
                # Configure conda environment for Python execution
                # This ensures Claude Code uses the same conda environment as the Python sandbox
                from station.eval_research.evaluation_helpers import setup_conda_env
                conda_env = self.constants.RESEARCH_EVAL_PYTHON_CONDA_ENV
                
                # Set up conda environment in the env dict
                if not setup_conda_env(conda_env, env):
                    print(f"ClaudeCodeDebugger: Could not configure conda environment '{conda_env}', using system Python")
                
                # --- Find Claude Executable ---
                claude_executable = os.environ.get('CLAUDE_BIN_PATH')
                if not claude_executable or not os.path.exists(claude_executable) or not os.access(claude_executable, os.X_OK):
                    print(f"ClaudeCodeDebugger: CLAUDE_BIN_PATH not set or invalid: {claude_executable}. Attempting to find 'claude' in PATH...")
                    try:
                        import shutil
                        claude_executable = shutil.which('claude')
                    except Exception as e:
                        print(f"ClaudeCodeDebugger: Error using shutil.which: {e}")
                
                if not claude_executable:
                    raise FileNotFoundError("Claude executable not found. Please ensure 'claude' is installed and in your PATH, or set the CLAUDE_BIN_PATH environment variable.")

                # Add the directory of the claude executable to the PATH
                claude_dir = os.path.dirname(claude_executable)
                if claude_dir:
                    env['PATH'] = f"{claude_dir}:{env.get('PATH', '')}"

                # Create history directory for this evaluation
                history_dir = os.path.join(self.claude_workspaces_path, 'history', f'eval_{eval_id}_{session_id}')
                os.makedirs(history_dir, exist_ok=True)
                
                # Create output files
                stdout_file = os.path.join(history_dir, 'claude_output.json')  # JSON output
                stderr_file = os.path.join(history_dir, 'claude_stderr.txt')
                metadata_file = os.path.join(history_dir, 'session_metadata.yaml')
                
                # Save metadata
                metadata = {
                    'eval_id': eval_id,
                    'session_id': session_id,
                    'started_at': time.time(),
                    'workspace_dir': workspace_dir,
                    'author': eval_data["author"],
                    'title': eval_data["title"],
                    'error_type': eval_data["original_submission"]["evaluation_result"].get("error", "unknown")
                }
                file_io_utils.save_yaml(metadata, metadata_file)
                
                # Files for output already defined above
                
                print(f"ClaudeCodeDebugger: Launching Claude session {session_id}")
                print(f"ClaudeCodeDebugger: Workspace: {workspace_dir}")
                print(f"ClaudeCodeDebugger: History dir: {history_dir}")
                print(f"ClaudeCodeDebugger: About to run subprocess.run()...")
                
                # Launch subprocess with retry logic
                max_retries = self.constants.CLAUDE_CODE_LAUNCH_MAX_RETRIES
                retry_delay = self.constants.CLAUDE_CODE_LAUNCH_RETRY_DELAY
                result = None
                
                for attempt in range(max_retries):
                    try:
                        # Use the configured timeout for Claude to complete
                        timeout_seconds = self.constants.CLAUDE_CODE_DEBUG_TIMEOUT
                        
                        result = subprocess.run([
                        claude_executable, '-p', prompt,
                        '--verbose',  # Add verbose flag for full output
                        '--output-format', 'json',  # Get JSON output for complete history
                        '--allowedTools', 'Read,Write,Bash,Python',
                        '--disallowedTools', 'Write(storage/shared/**),Edit(storage/shared/**),Write(storage/system/**),Edit(storage/system/**),MultiEdit(storage/shared/**),MultiEdit(storage/system/**),Write(storage/{lineage_name}/**),Edit(storage/{lineage_name}/**),MultiEdit(storage/{lineage_name}/**)'.replace('{lineage_name}', lineage_name),
                        '--max-turns', '200'
                        ], env=env, capture_output=True, text=True, cwd=workspace_dir, 
                        timeout=timeout_seconds)  # Add timeout to prevent hanging
                        
                        print(f"ClaudeCodeDebugger: subprocess.run() returned")
                        break  # Success, exit retry loop
                        
                    except FileNotFoundError as e:
                        if 'claude' in str(e) and attempt < max_retries - 1:
                            print(f"ClaudeCodeDebugger: Claude command not found (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            raise  # Re-raise on last attempt or non-claude errors
                    except subprocess.TimeoutExpired as e:
                        print(f"ClaudeCodeDebugger: Claude timed out after {timeout_seconds}s")
                        # Save partial output if available
                        result = e
                        if hasattr(e, 'stdout') and e.stdout:
                            result.stdout = e.stdout
                        else:
                            result.stdout = ""
                        if hasattr(e, 'stderr') and e.stderr:
                            result.stderr = e.stderr
                        else:
                            result.stderr = ""
                        result.returncode = -1
                        break  # Exit retry loop on timeout
                    except Exception as e:
                        print(f"ClaudeCodeDebugger: Exception in subprocess.run(): {e}")
                        raise
                
                # Save output with explicit file sync
                try:
                    # Write stdout
                    with open(stdout_file, 'w') as f:
                        f.write(result.stdout if result.stdout else "")
                        f.flush()
                        os.fsync(f.fileno())  # Force write to disk
                    print(f"ClaudeCodeDebugger: Saved stdout to {stdout_file}")
                    
                    # Write stderr
                    with open(stderr_file, 'w') as f:
                        f.write(result.stderr if result.stderr else "")
                        f.flush()
                        os.fsync(f.fileno())  # Force write to disk
                    print(f"ClaudeCodeDebugger: Saved stderr to {stderr_file}")
                    
                    # Verify files were written
                    stdout_size = os.path.getsize(stdout_file) if os.path.exists(stdout_file) else 0
                    stderr_size = os.path.getsize(stderr_file) if os.path.exists(stderr_file) else 0
                    print(f"ClaudeCodeDebugger: Written files - stdout: {stdout_size} bytes, stderr: {stderr_size} bytes")
                    
                except Exception as e:
                    print(f"ClaudeCodeDebugger: Error saving output files: {e}")
                    import traceback
                    traceback.print_exc()
                
                print(f"ClaudeCodeDebugger: Claude completed with return code: {result.returncode}")
                print(f"ClaudeCodeDebugger: Stdout length: {len(result.stdout) if result.stdout else 0} chars")
                print(f"ClaudeCodeDebugger: Stderr length: {len(result.stderr) if result.stderr else 0} chars")
                
                # Try to extract summary from JSON output
                if result.stdout:
                    try:
                        import json
                        # Parse JSON array output
                        json_output = json.loads(result.stdout)
                        # Validate JSON structure
                        if not isinstance(json_output, list):
                            raise ValueError("Expected JSON array output")
                        
                        # Look for the final assistant message
                        final_message = None
                        for item in reversed(json_output):
                            if not isinstance(item, dict):
                                continue
                            if item.get('type') == 'assistant':
                                message = item.get('message', {})
                                if isinstance(message, dict):
                                    content = message.get('content', [])
                                    if isinstance(content, list):
                                        for c in content:
                                            if isinstance(c, dict) and c.get('type') == 'text':
                                                final_message = c.get('text', '')
                                                break
                                if final_message:
                                    break
                        
                        if final_message:
                            print(f"ClaudeCodeDebugger: Claude final message:\n{final_message[:500]}")
                        else:
                            print(f"ClaudeCodeDebugger: Claude output (first 500 chars):\n{result.stdout[:500]}")
                    except Exception as e:
                        print(f"ClaudeCodeDebugger: Could not parse JSON output: {e}")
                        print(f"ClaudeCodeDebugger: Raw output (first 500 chars):\n{result.stdout[:500]}")
                
                # Handle completion immediately after Claude finishes
                # This is the main completion handling - we don't need check_completions for this
                self._handle_completion(eval_id, workspace_dir)
                
            except Exception as e:
                print(f"ClaudeCodeDebugger: Error launching session for {eval_id}: {e}")
                import traceback
                traceback.print_exc()
                
                # Mark debugging as failed but complete the evaluation
                try:
                    print(f"ClaudeCodeDebugger: Claude debugging failed after all attempts, completing evaluation without debugging")
                    
                    # Get evaluation data
                    eval_data = self.auto_evaluator.eval_manager.get_evaluation(eval_id)
                    if eval_data:
                        # Mark Claude session as failed with no report
                        # This will ensure the agent gets notified with the original error
                        self.auto_evaluator.eval_manager.complete_claude_session(
                            eval_id, 
                            final_report=None,  # No report since Claude didn't run
                            success=False
                        )
                        print(f"ClaudeCodeDebugger: Completed failed session for {eval_id}, agent will see original error")
                    else:
                        print(f"ClaudeCodeDebugger: Could not find evaluation {eval_id} to mark as complete")
                        
                except Exception as complete_err:
                    print(f"ClaudeCodeDebugger: Error completing failed session: {complete_err}")
                    import traceback
                    traceback.print_exc()
            
    def _build_prompt(self, eval_id: str, lineage_name: str) -> str:
        """
        Build Claude Code prompt with specific eval_id and lineage name.
        
        Args:
            eval_id: Evaluation ID
            lineage_name: Name of the agent's lineage (e.g., 'praxis', 'alice')
            
        Returns:
            Prompt string for Claude Code
        """
        return f'''You are debugging failed research submission {eval_id} in an isolated workspace.

IMPORTANT: How the system works:
- When you write submissions/submission_v2.py, it is AUTOMATICALLY fetched and executed
- You do NOT need to run it yourself
- The system will create a new evaluation file when execution completes
- Your goal: Fix errors so the code runs WITHOUT CRASHING (not necessarily to completion)

CRITICAL: About submission files:
- Each submission file (v2, v3, v4, etc.) must be a COMPLETE, WORKING implementation
- Do NOT create test files or partial implementations to "see if it works"
- The system will use ANY successful submission as the final output
- If you need to test something, use the tmp/ directory, not submissions/
- Only write to submissions/ when you have a complete fix ready

WORKSPACE STRUCTURE:
- evaluation.yaml - The failed evaluation with error logs
- submissions/ - Write fixed code here (submission_v2.py, submission_v3.py, etc.)
- tmp/ - Your scratch space for testing snippets
- monitor_evaluation.py - Use this to check if your fix worked
- storage/shared/ - READ-ONLY shared data
- storage/system/ - READ-ONLY system files (train.py, env.py) - NEVER modify or create files here!
- storage/{lineage_name}/ - READ-ONLY author's lineage data (e.g., storage/{lineage_name}/)

CRITICAL RULES:
- NEVER write, create, or modify ANY files in storage/system/ directory
- Always write fixed submissions to the submissions/ directory (e.g., submissions/submission_v2.py)
- Do NOT save submissions to storage/system/submission.py - this is incorrect!

PYTHON ENVIRONMENT:
- Python tools use the conda environment: {self.constants.RESEARCH_EVAL_PYTHON_CONDA_ENV}
- This matches the evaluation system's Python environment
- All required packages (JAX, Flax, etc.) are pre-installed

IMPORTANT NOTE ABOUT LINEAGE FILES:
- The lineage directory storage/{lineage_name}/ is READ-ONLY - you cannot modify these files
- ONLY copy functions that have bugs. Keep imports for functions that work correctly!
- If the agent's code imports functions from their lineage directory that have bugs, you MUST:
  1. Copy ONLY the problematic function(s) from storage/{lineage_name}/my_helper.py into submission_v2.py
  2. Fix the bugs in the copied function
  3. Remove ONLY the import for the buggy function, keep other working imports
- Example: If submission has "from storage.{lineage_name}.my_helper import func1, func2, func3" and only func2 has bugs:
  1. Keep the import for func1 and func3 (they work fine)
  2. Read storage/{lineage_name}/my_helper.py to get func2's code
  3. Copy func2 into submission_v2.py and fix the bugs there
  4. Change import to: "from storage.{lineage_name}.my_helper import func1, func3"
- If ALL imported functions work correctly, keep all imports unchanged

PROCESS:
1. Read evaluation.yaml to understand the error (check the 'logs' field for error details)
2. Analyze if the error is:
   a) Simple fix in main code (syntax, imports, shape mismatches) → Proceed to fix
   b) Bug in imported lineage function → Copy function to submission and fix there
   c) Fundamental logic error requiring rewrite → Skip to step 6
3. Apply the fix:
   - If error is in main submission code: Write fixed code to submissions/submission_v2.py
   - If error is in imported lineage function: 
     * Read the function from storage/{lineage_name}/file.py
     * Copy ONLY the buggy function into submissions/submission_v2.py (not the entire file)
     * Fix the bug in the copied function
     * Update imports: remove only the buggy function, keep working functions
   - MUST use v2, v3, v4 naming for submissions in submissions/ directory
4. CRITICAL VERIFICATION STEP - Use Bash tool to execute:
   python monitor_evaluation.py 2
   
   This script will check if your fix worked by monitoring for the new evaluation.
   DO NOT proceed without running this command and checking the exit code!
   
5. Based on monitor exit code:
   - Exit 0: Success! Score achieved. Write report_success.md
   - Exit 1: Success! Code is running without crashing. Write report_success.md
   - Exit 2: Failed with new errors. Fix and try v3
6. Continue until success or {self.constants.CLAUDE_CODE_DEBUG_MAX_ATTEMPTS} attempts

WHEN TO GIVE UP:
- The algorithm/approach is fundamentally wrong
- Fixing would require completely different logic
- After {self.constants.CLAUDE_CODE_DEBUG_MAX_ATTEMPTS} attempts with different errors each time
- The task requirements were misunderstood

IMPORTANT: Report naming based on outcome:
- If you successfully fixed the code: Write report_success.md
- If you gave up after failed attempts: Write report_failed.md

FINAL REPORT (report_success.md OR report_failed.md) MUST INCLUDE:
# Debug Report for Evaluation {eval_id}

## Summary
[Success/Failed - brief explanation]

## Root Cause
[What was wrong with the original code]

## Fix Applied (if any)
[What you changed and why]

## Recommendation (if gave up)
[Why the code needs fundamental rework]'''
        
    def check_completions(self):
        """Check for completed sessions via report files"""
        # Note: Since we use subprocess.run() which blocks until completion,
        # all completion handling is done directly in the worker thread.
        # This method is kept for API compatibility.
        # Clean up completed threads
        self.active_threads = [t for t in self.active_threads if t.is_alive()]
        
    def has_active_sessions(self) -> bool:
        """
        Check if any debugging sessions are active.
        
        Returns:
            True if there are active sessions
        """
        # Clean up completed threads
        self.active_threads = [t for t in self.active_threads if t.is_alive()]
        return len(self.active_threads) > 0
        
    def _handle_completion(self, eval_id: str, workspace_dir: str):
        """
        Handle completion - copy appropriate report to main directory and update EvaluationManager.
        
        Args:
            eval_id: Evaluation ID
            workspace_dir: Path to workspace directory
        """
        try:
            # Get evaluation data
            eval_data = self.auto_evaluator.eval_manager.get_evaluation(eval_id)
            if not eval_data:
                print(f"ClaudeCodeDebugger: Could not find evaluation {eval_id} in EvaluationManager")
                return
            
            # Scan for any file ending with report_success.md or report_failed.md
            success_report_workspace = None
            failed_report_workspace = None
            
            if os.path.exists(workspace_dir):
                for filename in os.listdir(workspace_dir):
                    if filename.endswith('report_success.md'):
                        success_report_workspace = os.path.join(workspace_dir, filename)
                    elif filename.endswith('report_failed.md'):
                        failed_report_workspace = os.path.join(workspace_dir, filename)
            
            evaluations_dir = os.path.join(self.research_room_path, self.constants.RESEARCH_EVALUATIONS_SUBDIR_NAME)
            
            completion_status = None
            final_report = ""
            success = False
            
            if success_report_workspace and os.path.exists(success_report_workspace):
                # Copy success report
                main_report = os.path.join(evaluations_dir, f'report_{eval_id}_success.md')
                shutil.copy2(success_report_workspace, main_report)
                with open(success_report_workspace, 'r') as f:
                    final_report = f.read()
                completion_status = 'success'
                success = True
            elif failed_report_workspace and os.path.exists(failed_report_workspace):
                # Copy failed report
                main_report = os.path.join(evaluations_dir, f'report_{eval_id}_failed.md')
                shutil.copy2(failed_report_workspace, main_report)
                with open(failed_report_workspace, 'r') as f:
                    final_report = f.read()
                completion_status = 'failed'
                success = False
            else:
                # No report found (timeout/crash) - follow principle 3
                print(f"ClaudeCodeDebugger: No report files found for {eval_id} - completing session without report")
                completion_status = 'no_report'
                final_report = None
                success = False
            
            # Complete Claude session in EvaluationManager
            self.auto_evaluator.eval_manager.complete_claude_session(
                eval_id, final_report, success
            )
            
            print(f"ClaudeCodeDebugger: Completed Claude session for {eval_id} with status: {completion_status}")
            
            # Save workspace snapshot to history
            self._save_workspace_snapshot(eval_id, workspace_dir, completion_status)
            
            # Skip cleanup for now
            self._cleanup_workspace(workspace_dir)
            
        except Exception as e:
            print(f"ClaudeCodeDebugger: Error handling completion for {eval_id}: {e}")
            import traceback
            traceback.print_exc()
        
    def _save_workspace_snapshot(self, eval_id: str, workspace_dir: str, completion_status: Optional[str] = None):
        """
        Save a snapshot of the workspace to history for debugging.
        
        Args:
            eval_id: Evaluation ID
            workspace_dir: Path to workspace directory
            completion_status: 'success', 'failed', or None
        """
        try:
            # Find the latest history directory for this eval
            history_base = os.path.join(self.claude_workspaces_path, 'history')
            if not os.path.exists(history_base):
                os.makedirs(history_base, exist_ok=True)
            
            # Find the most recent session directory
            session_dirs = [d for d in os.listdir(history_base) if d.startswith(f'eval_{eval_id}_')]
            if not session_dirs:
                # Create one if none exists (for manual debugging)
                session_id = f'manual_{int(time.time())}'
                history_dir = os.path.join(history_base, f'eval_{eval_id}_{session_id}')
                os.makedirs(history_dir, exist_ok=True)
            else:
                latest_session = sorted(session_dirs)[-1]
                history_dir = os.path.join(history_base, latest_session)
            
            # Save workspace structure
            workspace_structure_file = os.path.join(history_dir, 'workspace_final_structure.txt')
            with open(workspace_structure_file, 'w') as f:
                f.write(f"Final workspace structure for eval {eval_id}:\n")
                f.write(f"Completion status: {completion_status or 'unknown'}\n\n")
                
                for root, dirs, files in os.walk(workspace_dir):
                    level = root.replace(workspace_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    f.write(f"{indent}{os.path.basename(root)}/\n")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        f.write(f"{subindent}{file}\n")
            
            # Copy important files to history
            files_to_copy = [
                ('report_success.md', 'final_report_success.md'),
                ('report_failed.md', 'final_report_failed.md'),
                ('submissions/submission_v2.py', 'submission_v2.py'),
                ('submissions/submission_v3.py', 'submission_v3.py'),
                ('submissions/submission_v4.py', 'submission_v4.py'),
                ('submissions/submission_v5.py', 'submission_v5.py'),
                ('evaluation.yaml', 'original_evaluation.yaml'),
                ('monitor_calls.log', 'monitor_calls.log'),  # Add monitor call log
            ]
            
            for src_path, dst_name in files_to_copy:
                src_full = os.path.join(workspace_dir, src_path)
                if os.path.exists(src_full):
                    dst_full = os.path.join(history_dir, dst_name)
                    shutil.copy2(src_full, dst_full)
            
            # Update metadata with completion info
            metadata_file = os.path.join(history_dir, 'session_metadata.yaml')
            if os.path.exists(metadata_file):
                metadata = file_io_utils.load_yaml(metadata_file)
                metadata['completed_at'] = time.time()
                metadata['completion_status'] = completion_status
                metadata['duration_seconds'] = metadata.get('completed_at', 0) - metadata.get('started_at', 0)
                file_io_utils.save_yaml(metadata, metadata_file)
            
            print(f"ClaudeCodeDebugger: Saved workspace snapshot to {history_dir}")
            
        except Exception as e:
            print(f"ClaudeCodeDebugger: Error saving workspace snapshot: {e}")
    
    def _cleanup_workspace(self, workspace_dir: str):
        """
        Clean up workspace after completion.
        
        Args:
            workspace_dir: Path to workspace directory
        """
        # Validate path to prevent directory traversal
        workspace_dir = os.path.abspath(workspace_dir)
        expected_parent = os.path.abspath(self.claude_workspaces_path)
        
        if not workspace_dir.startswith(expected_parent):
            print(f"ClaudeCodeDebugger: Refusing to clean up directory outside workspace: {workspace_dir}")
            return
            
        try:
            # Remove symlinks first (important on some systems)
            for storage_type in ['shared', 'system', 'lineage']:
                symlink_path = os.path.join(workspace_dir, f"storage/{storage_type}")
                if os.path.islink(symlink_path):
                    os.unlink(symlink_path)
            
            # Then remove the entire workspace
            shutil.rmtree(workspace_dir)
        except Exception as e:
            print(f"ClaudeCodeDebugger: Error cleaning workspace {workspace_dir}: {e}")
        
    def get_debug_report(self, eval_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Read and parse debug report if exists.
        
        Args:
            eval_id: Evaluation ID
            
        Returns:
            Tuple of (status, report_content) where status is 'success', 'failed', or None
        """
        evaluations_dir = os.path.join(self.research_room_path, self.constants.RESEARCH_EVALUATIONS_SUBDIR_NAME)
        success_report_path = os.path.join(evaluations_dir, f'report_{eval_id}_success.md')
        failed_report_path = os.path.join(evaluations_dir, f'report_{eval_id}_failed.md')
        
        if os.path.exists(success_report_path):
            with open(success_report_path, 'r') as f:
                return ('success', f.read())
        elif os.path.exists(failed_report_path):
            with open(failed_report_path, 'r') as f:
                return ('failed', f.read())
        return (None, None)
        
        
    
            
