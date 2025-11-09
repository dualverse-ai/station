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

# station/eval_research/executor_sandbox.py
"""
Python sandbox execution methods for research evaluation.
"""

import os

# Force direct subprocess import to avoid asyncio integration issues
import subprocess as _original_subprocess

# Disable asyncio subprocess integration to prevent "child watchers are only available on the default loop" error
# This forces subprocess.Popen to use traditional synchronous behavior in worker threads
os.environ['PYTHONASYNCIO'] = '0'

# Ensure we use the original subprocess.Popen without any asyncio wrapping
subprocess = _original_subprocess
GEVENT_AVAILABLE = False

# Override gevent subprocess to prevent asyncio conflicts in worker threads
try:
    import gevent
    # Don't use gevent.subprocess as it may still trigger asyncio issues
    print("AutoResearchEvaluator: Using standard subprocess to avoid asyncio conflicts.")
except ImportError:
    pass

import tempfile
import uuid
import numpy as np
import pickle
import signal
import shutil
from typing import Dict, Any, Optional

from station import constants
from station import file_io_utils
from station.eval_research.evaluation_helpers import truncate_stderr


def _execute_submission_in_python_sandbox(self, eval_entry: Dict[str, Any], evaluator, gpu_ids: Optional[list] = None) -> Dict[str, Any]:
    """Execute research submission code in Python sandbox using conda environment"""
    content = eval_entry.get(constants.EVALUATION_CONTENT_KEY, "")
    author = eval_entry.get(constants.EVALUATION_AUTHOR_KEY, "Unknown")
    
    timeout = self.timeout
    
    # Determine temp directory location based on shared storage config
    research_room_abs_path = os.path.abspath(self.research_room_path)
    storage_base_path = os.path.join(research_room_abs_path, constants.RESEARCH_STORAGE_DIR)
    
    # Check if storage is a symlink (indicates shared storage is being used)
    if os.path.islink(storage_base_path) and constants.RESEARCH_STORAGE_BASE_PATH:
        # Resolve the symlink to get the actual shared storage path
        real_storage_path = os.path.realpath(storage_base_path)
        tmp_base = os.path.join(real_storage_path, "tmp")
        
        # Ensure tmp directory exists
        os.makedirs(tmp_base, exist_ok=True)
        
        # Create a unique subdirectory in the shared tmp folder
        temp_dir_name = str(uuid.uuid4())
        temp_dir = os.path.join(tmp_base, temp_dir_name)
        os.makedirs(temp_dir)
        
        # Use context manager to ensure cleanup
        import contextlib
        @contextlib.contextmanager
        def cleanup_temp_dir():
            try:
                yield temp_dir
            finally:
                try:
                    # Handle read-only files/directories during cleanup
                    def handle_remove_readonly(func, path, exc):
                        os.chmod(path, 0o755)
                        func(path)
                    shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
                except Exception as e:
                    print(f"Warning: Could not clean up temp directory {temp_dir}: {e}")
        
        temp_context = cleanup_temp_dir()
    else:
        # Use system temp directory (original behavior)
        temp_context = tempfile.TemporaryDirectory()
    
    with temp_context as temp_dir:
        # Set up storage symlinks
        storage_dir = os.path.join(temp_dir, "storage")
        os.makedirs(storage_dir, exist_ok=True)
        
        # Resolve storage base path in case it's a symlink to shared storage
        storage_base = os.path.join(research_room_abs_path, constants.RESEARCH_STORAGE_DIR)
        if os.path.islink(storage_base):
            # Use the real path if storage is symlinked to shared location
            storage_base = os.path.realpath(storage_base)
        
        shared_storage_path = os.path.join(storage_base, constants.RESEARCH_STORAGE_SHARED_DIR)
        system_storage_path = os.path.join(storage_base, constants.RESEARCH_STORAGE_SYSTEM_DIR)
        architect_storage_path = os.path.join(storage_base, "architect")
        try:
            author_data = self.station.agent_module.load_agent_data(author)
            author_lineage = author_data.get(constants.AGENT_LINEAGE_KEY, "unknown").lower() if author_data else "unknown"
        except Exception:
            author_lineage = "unknown"
        author_lineage_storage_path = os.path.join(storage_base, constants.RESEARCH_STORAGE_LINEAGES_DIR, author_lineage)
        lineages_base_path = os.path.join(storage_base, constants.RESEARCH_STORAGE_LINEAGES_DIR)
        file_io_utils.ensure_dir_exists(shared_storage_path)
        file_io_utils.ensure_dir_exists(system_storage_path)
        file_io_utils.ensure_dir_exists(author_lineage_storage_path)
        file_io_utils.ensure_dir_exists(architect_storage_path)
        if os.path.exists(shared_storage_path): os.symlink(shared_storage_path, os.path.join(storage_dir, "shared"))
        if os.path.exists(system_storage_path):
            # Use symlink for system storage since it's already read-only
            os.symlink(system_storage_path, os.path.join(storage_dir, "system"))
        if os.path.exists(architect_storage_path): os.symlink(architect_storage_path, os.path.join(storage_dir, "architect"))
        if os.path.exists(author_lineage_storage_path):
            os.symlink(author_lineage_storage_path, os.path.join(storage_dir, "lineage"))
            os.symlink(author_lineage_storage_path, os.path.join(storage_dir, author_lineage))
        if constants.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS and os.path.exists(lineages_base_path):
            try:
                for lineage_name in os.listdir(lineages_base_path):
                    lineage_path = os.path.join(lineages_base_path, lineage_name)
                    if os.path.isdir(lineage_path):
                        if lineage_name != author_lineage:
                            if not os.path.exists(os.path.join(storage_dir, lineage_name)):
                                os.symlink(lineage_path, os.path.join(storage_dir, lineage_name))
                            capitalized_name = lineage_name.capitalize()
                            if capitalized_name != lineage_name and not os.path.exists(os.path.join(storage_dir, capitalized_name)):
                                os.symlink(lineage_path, os.path.join(storage_dir, capitalized_name))
            except OSError: pass
        if os.path.exists(author_lineage_storage_path):
            author_capitalized = author_lineage.capitalize()
            if author_capitalized != author_lineage and not os.path.exists(os.path.join(storage_dir, author_capitalized)):
                os.symlink(author_lineage_storage_path, os.path.join(storage_dir, author_capitalized))
        
        execution_mode = evaluator.get_execution_mode() if hasattr(evaluator, 'get_execution_mode') else "function"
        
        if execution_mode == "command":
            submission_filename = evaluator.get_submission_filename()
            submission_path = os.path.join(temp_dir, submission_filename)
            with open(submission_path, 'w', encoding='utf-8') as f:
                f.write(content)
            execution_command = evaluator.get_execution_command()
        else:
            run_py_path = os.path.join(temp_dir, "run.py")
            with open(run_py_path, 'w', encoding='utf-8') as f:
                f.write(content)

        if execution_mode == "command":
            # ** FIXED: This wrapper now streams output instead of blocking **
            wrapper_content = f"""
import sys, subprocess, os
try:
    cmd = {execution_command!r}
    print(f"Executing command: {{cmd}}")
    sys.stdout.flush() 

    env = os.environ.copy()
    env['PYTHONPATH'] = '.' + os.pathsep + env.get('PYTHONPATH', '')

    # Use subprocess.run without capturing output to let it stream.
    # The parent process will capture this wrapper's output, which now
    # includes the output of the command in real-time.
    result = subprocess.run(
        cmd,
        shell=True,
        text=True,
        env=env
    )
    sys.exit(result.returncode)
except Exception as e:
    print(f"EXECUTION_ERROR: Failed to launch command: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        else:
            # Function mode wrapper is unchanged and correct.
            wrapper_content = f"""
import sys, traceback, numpy as np, pickle
try:
    from run import {evaluator.get_expected_function_name()}
except ImportError as e:
    print(f"IMPORT_ERROR: Cannot import {evaluator.get_expected_function_name()}: {{e}}", file=sys.stderr)
    print("IMPORT_DETAILS: Full import error traceback:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print("IMPORT_HINT: Check that all imported modules exist and have correct names.", file=sys.stderr)
    print("IMPORT_HINT: Common issues: typos in module names, missing .py files, wrong file paths.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"IMPORT_ERROR: Error importing {evaluator.get_expected_function_name()}: {{e}}", file=sys.stderr)
    print("IMPORT_DETAILS: Full error traceback:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print("IMPORT_HINT: This may be a syntax error or other issue in your code.", file=sys.stderr)
    sys.exit(1)
try:
    result = {evaluator.get_expected_function_name()}()
    if hasattr(result, 'shape') and hasattr(result, 'dtype'):
        np.save('result.npy', result)
        print(f"EXECUTION_SUCCESS: Function returned result with shape {{result.shape}} and dtype {{result.dtype}}")
    else:
        # Save non-array results (like tuples) using pickle
        with open('result.pkl', 'wb') as f:
            pickle.dump(result, f)
        print(f"EXECUTION_SUCCESS: Function returned result (non-array): {{type(result)}}")
except Exception as e:
    print(f"EXECUTION_ERROR: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
        # Use a unique name to avoid collisions with user code that might import 'wrapper'
        wrapper_path = os.path.join(temp_dir, "__sandbox_execution_wrapper__.py")
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_content)
        
        try:
            env = os.environ.copy()
            if gpu_ids is not None: env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
            if 'HF_HOME' not in env and 'HOME' in env: env['HF_HOME'] = os.path.join(env['HOME'], '.cache', 'huggingface')
            if 'XDG_CACHE_HOME' not in env and 'HOME' in env: env['XDG_CACHE_HOME'] = os.path.join(env['HOME'], '.cache')
            for proxy in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']: env.pop(proxy, None)
            
            # Force disable asyncio subprocess integration in the child process environment
            env['PYTHONASYNCIO'] = '0'
            
            from station.eval_research.evaluation_helpers import find_conda_python
            conda_env_name = constants.RESEARCH_EVAL_PYTHON_CONDA_ENV
            python_path = find_conda_python(conda_env_name, env)
            if not python_path: raise Exception(f"No suitable Python executable found for conda environment '{conda_env_name}'")

            sandbox_cmd = [python_path, '-u', wrapper_path]
            print(f"AutoResearchEvaluator: Executing in Python sandbox with command: {' '.join(sandbox_cmd)}")

            # Define memory limit function if configured
            def set_resource_limits():
                """Set memory limits for the subprocess"""
                if constants.RESEARCH_EVAL_MEMORY_LIMIT:
                    import resource
                    # Parse memory limit (e.g., "64g" -> 64 * 1024^3 bytes)
                    memory_str = str(constants.RESEARCH_EVAL_MEMORY_LIMIT).lower()
                    if memory_str.endswith('g'):
                        memory_gb = float(memory_str[:-1])
                        memory_bytes = int(memory_gb * 1024 * 1024 * 1024)
                    elif memory_str.endswith('m'):
                        memory_mb = float(memory_str[:-1])
                        memory_bytes = int(memory_mb * 1024 * 1024)
                    else:
                        # Assume bytes if no suffix
                        memory_bytes = int(memory_str)

                    # Set virtual memory limit
                    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
                    print(f"AutoResearchEvaluator: Set memory limit to {memory_str}")

            # Use preexec_fn if memory limit is set, otherwise use start_new_session for compatibility
            if constants.RESEARCH_EVAL_MEMORY_LIMIT:
                # When memory limit is set, use preexec_fn to set limits
                process = subprocess.Popen(
                    sandbox_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    cwd=temp_dir,
                    preexec_fn=set_resource_limits,
                    start_new_session=True
                )
            else:
                # Original behavior when no memory limit
                process = subprocess.Popen(
                    sandbox_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    cwd=temp_dir,
                    start_new_session=True
                )
            
            stdout_output, stderr_output = process.communicate(timeout=timeout)
            returncode = process.returncode
        
        except subprocess.TimeoutExpired:
            print(f"AutoResearchEvaluator: Process timed out after {timeout} seconds. Terminating process group.")
            try:
                # Use os.killpg with the process's session ID (which is its PID due to start_new_session=True)
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass 
            # FIX: Add a short timeout to the final communicate to prevent hangs if killpg fails
            try:
                stdout_output, stderr_output = process.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                stdout_output = ""
                stderr_output = "Process failed to terminate even after SIGKILL and timed out again."
            
            truncated_stderr = truncate_stderr(stderr_output)
            execution_logs = f"PYTHON SANDBOX TIMEOUT:\nSTDOUT:\n{stdout_output}\n\nSTDERR:\n{truncated_stderr}"
            error_msg = f"Execution timed out after {self.timeout} seconds\n\n**Optimization Tips:**\n1. Use GPU-accelerated code when possible (JAX, CuPy, etc.)\n2. Vectorize operations instead of loops\n3. Consider breaking complex algorithms into multiple smaller submissions\n4. Profile your code to identify bottlenecks\n5. Use JIT compilation (@jit decorators) for computational kernels"
            
            return {"success": False, "error": error_msg, "logs": execution_logs}
        
        except Exception as e:
            return {"success": False, "error": f"Python sandbox execution error: {str(e)}", "logs": str(e)}

        truncated_stderr = truncate_stderr(stderr_output)
        execution_logs = f"PYTHON SANDBOX EXECUTION:\nSTDOUT:\n{stdout_output}\n\nSTDERR:\n{truncated_stderr}"

        success = (returncode == 0 and "EXECUTION_SUCCESS" in stdout_output) if execution_mode == "function" else (returncode == 0)

        if success:
            if execution_mode == "command":
                return {"success": True, "result": stdout_output, "logs": execution_logs}
            else:
                result_npy_path = os.path.join(temp_dir, "result.npy")
                result_pkl_path = os.path.join(temp_dir, "result.pkl")
                
                if os.path.exists(result_npy_path):
                    loaded_result = np.load(result_npy_path)
                    return {"success": True, "result": loaded_result, "logs": execution_logs}
                elif os.path.exists(result_pkl_path):
                    with open(result_pkl_path, 'rb') as f:
                        loaded_result = pickle.load(f)
                    return {"success": True, "result": loaded_result, "logs": execution_logs}
                else:
                    result_txt_path = os.path.join(temp_dir, "result.txt")
                    result_text = ""
                    if os.path.exists(result_txt_path):
                        with open(result_txt_path, 'r') as f: result_text = f.read()
                    return {"success": False, "error": f"Function returned non-array result: {result_text}", "logs": execution_logs}

        if "IMPORT_ERROR" in stderr_output:
            main_error_line = next((line for line in stderr_output.split('\n') if "IMPORT_ERROR" in line), "")
            main_error = main_error_line.replace("IMPORT_ERROR: ", "").strip()
            error_msg = f"Import failed: {main_error}" if main_error else "Import failed. Check logs for details."
        elif "EXECUTION_ERROR" in stderr_output:
            main_error_line = next((line for line in stderr_output.split('\n') if "EXECUTION_ERROR" in line), "")
            main_error = main_error_line.replace("EXECUTION_ERROR: ", "").strip()
            error_msg = f"Runtime error during execution: {main_error}" if main_error else "Runtime error. Check logs for details."
        else:
            error_msg = f"Execution failed with exit code {returncode}. Check logs for details."
        
        return {"success": False, "error": error_msg, "logs": execution_logs}