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
import threading
import uuid
import shlex
import numpy as np
import pickle
import signal
import shutil
from typing import Dict, Any, Optional

from station import constants
from station import file_io_utils
from station.eval_research.evaluation_helpers import append_stdout_with_limit, resolve_conda_env, truncate_stderr


def _function_result_paths(tmp_dir: str) -> tuple[str, str]:
    return os.path.join(tmp_dir, "result.npy"), os.path.join(tmp_dir, "result.pkl")


def _has_function_result_artifact(tmp_dir: str) -> bool:
    result_npy_path, result_pkl_path = _function_result_paths(tmp_dir)
    return os.path.exists(result_npy_path) or os.path.exists(result_pkl_path)


def _function_mode_success(returncode: int, stdout_output: str, tmp_dir: str) -> bool:
    if returncode != 0:
        return False
    # The persisted result artifact is the authoritative signal. Very large stdout can
    # delay or truncate the tail marker in the collected pipe output.
    return _has_function_result_artifact(tmp_dir) or "EXECUTION_SUCCESS" in stdout_output


def _append_live_log(log_path: Optional[str], text: str, *, truncate: bool):
    if not log_path or not text:
        return
    if truncate:
        append_stdout_with_limit(log_path, text)
        return
    file_io_utils.ensure_dir_exists(os.path.dirname(log_path))
    existing = file_io_utils.load_text(log_path) if file_io_utils.file_exists(log_path) else ""
    file_io_utils.save_text((existing or "") + text, log_path)

def _stream_pipe_to_log(
    stream,
    log_path: Optional[str],
    chunks: list[str],
    *,
    truncate: bool,
):
    try:
        while True:
            line = stream.readline()
            if line == "":
                break
            chunks.append(line)
            _append_live_log(log_path, line, truncate=truncate)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _build_sandbox_command(
    python_path: str,
    wrapper_path: str,
    *,
    cpu_ids: Optional[list] = None,
) -> list[str]:
    command = [python_path, '-u', wrapper_path]
    if cpu_ids:
        taskset_path = shutil.which("taskset")
        if not taskset_path:
            return command
        cpu_list = ",".join(str(cpu_id) for cpu_id in cpu_ids)
        return [taskset_path, "-c", cpu_list, *command]
    return command


def _execute_submission_in_python_sandbox(self, eval_entry: Dict[str, Any], evaluator,
                                          gpu_ids: Optional[list] = None,
                                          cpu_ids: Optional[list] = None,
                                          live_stdout_path: Optional[str] = None,
                                          live_stderr_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute research submission code in Python sandbox using conda environment"""
    content = eval_entry.get(constants.EVALUATION_CONTENT_KEY, "")
    author = eval_entry.get(constants.EVALUATION_AUTHOR_KEY, "Unknown")
    
    timeout = self.timeout
    
    # Determine tmp directory location based on shared storage config
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
        tmp_dir_name = str(uuid.uuid4())
        tmp_dir = os.path.join(tmp_base, tmp_dir_name)
        os.makedirs(tmp_dir)
        
        # Use context manager to ensure cleanup
        import contextlib
        @contextlib.contextmanager
        def cleanup_tmp_dir():
            try:
                yield tmp_dir
            finally:
                try:
                    # Handle read-only files/directories during cleanup
                    def handle_remove_readonly(func, path, exc):
                        os.chmod(path, 0o755)
                        func(path)
                    shutil.rmtree(tmp_dir, onerror=handle_remove_readonly)
                except Exception as e:
                    print(f"Warning: Could not clean up tmp directory {tmp_dir}: {e}")

        tmp_context = cleanup_tmp_dir()
    else:
        # Use the system tmp directory (original behavior)
        tmp_context = tempfile.TemporaryDirectory()

    with tmp_context as tmp_dir:
        # Set up storage symlinks
        storage_dir = os.path.join(tmp_dir, "storage")
        os.makedirs(storage_dir, exist_ok=True)
        
        # Resolve storage base path in case it's a symlink to shared storage
        storage_base = os.path.join(research_room_abs_path, constants.RESEARCH_STORAGE_DIR)
        if os.path.islink(storage_base):
            # Use the real path if storage is symlinked to shared location
            storage_base = os.path.realpath(storage_base)
        
        shared_storage_path = os.path.join(storage_base, constants.RESEARCH_STORAGE_SHARED_DIR)
        system_storage_path = os.path.join(storage_base, constants.RESEARCH_STORAGE_SYSTEM_DIR)
        architect_storage_path = os.path.join(storage_base, "architect")
        tmp_storage_path = os.path.join(storage_base, "tmp")
        try:
            author_data = self.station.agent_module.load_agent_data(author)
            author_lineage = author_data.get(constants.AGENT_LINEAGE_KEY, "unknown").lower() if author_data else "unknown"
        except Exception:
            author_lineage = "unknown"
        author_lineage_storage_path = os.path.join(storage_base, constants.RESEARCH_STORAGE_LINEAGES_DIR, author_lineage)
        author_tmp_storage_path = os.path.join(tmp_storage_path, author_lineage)
        lineages_base_path = os.path.join(storage_base, constants.RESEARCH_STORAGE_LINEAGES_DIR)
        file_io_utils.ensure_dir_exists(shared_storage_path)
        file_io_utils.ensure_dir_exists(system_storage_path)
        file_io_utils.ensure_dir_exists(author_lineage_storage_path)
        file_io_utils.ensure_dir_exists(author_tmp_storage_path)
        file_io_utils.ensure_dir_exists(architect_storage_path)
        if os.path.exists(shared_storage_path): os.symlink(shared_storage_path, os.path.join(storage_dir, "shared"))
        if os.path.exists(system_storage_path):
            # Use symlink for system storage since it's already read-only
            os.symlink(system_storage_path, os.path.join(storage_dir, "system"))
        if os.path.exists(architect_storage_path): os.symlink(architect_storage_path, os.path.join(storage_dir, "architect"))
        if os.path.exists(author_tmp_storage_path):
            tmp_view_path = os.path.join(storage_dir, "tmp")
            os.makedirs(tmp_view_path, exist_ok=True)
            os.symlink(author_tmp_storage_path, os.path.join(tmp_view_path, author_lineage))
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
            submission_path = os.path.join(tmp_dir, submission_filename)
            with open(submission_path, 'w', encoding='utf-8') as f:
                f.write(content)
            execution_command = evaluator.get_execution_command()
        else:
            run_py_path = os.path.join(tmp_dir, "run.py")
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
        wrapper_path = os.path.join(tmp_dir, "__sandbox_execution_wrapper__.py")
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_content)
        
        try:
            env = os.environ.copy()
            if gpu_ids is not None: env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
            if 'HF_HOME' not in env and 'HOME' in env: env['HF_HOME'] = os.path.join(env['HOME'], '.cache', 'huggingface')
            if 'XDG_CACHE_HOME' not in env and 'HOME' in env: env['XDG_CACHE_HOME'] = os.path.join(env['HOME'], '.cache')
            for proxy in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']: env.pop(proxy, None)

            env['STATION_EVAL_ID'] = str(eval_entry.get(constants.EVALUATION_ID_KEY, ''))
            env['STATION_BASE_EVAL_ID'] = str(eval_entry.get('base_eval_id', env['STATION_EVAL_ID']))
            env['STATION_EVAL_VERSION'] = '' if eval_entry.get('eval_version') is None else str(eval_entry.get('eval_version'))
            env['STATION_AUTHOR'] = str(eval_entry.get(constants.EVALUATION_AUTHOR_KEY, ''))
            env['STATION_RESEARCH_ROOT'] = research_room_abs_path
            
            # Force disable asyncio subprocess integration in the child process environment
            env['PYTHONASYNCIO'] = '0'
            
            conda_env_name = constants.RESEARCH_EVAL_PYTHON_CONDA_ENV
            python_path = resolve_conda_env(conda_env_name, env)
            if not python_path: raise Exception(f"No suitable Python executable found for conda environment '{conda_env_name}'")

            sandbox_cmd = _build_sandbox_command(
                python_path,
                wrapper_path,
                cpu_ids=cpu_ids,
            )
            print(
                "AutoResearchEvaluator: Executing in Python sandbox with command: "
                + " ".join(shlex.quote(part) for part in sandbox_cmd)
            )

            command_uses_taskset = bool(sandbox_cmd) and os.path.basename(sandbox_cmd[0]) == "taskset"
            needs_preexec_affinity = bool(cpu_ids) and not command_uses_taskset
            eval_id = str(eval_entry.get(constants.EVALUATION_ID_KEY, ""))
            affinity_mode = "taskset" if command_uses_taskset else ("preexec" if needs_preexec_affinity else "none")

            def set_resource_limits_and_affinity():
                """Set memory limits and fallback CPU affinity for the subprocess."""
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

                if needs_preexec_affinity:
                    try:
                        os.sched_setaffinity(0, cpu_ids)
                    except Exception as e:
                        print(f"AutoResearchEvaluator: Failed to set CPU affinity: {e}")

            use_preexec = bool(constants.RESEARCH_EVAL_MEMORY_LIMIT or needs_preexec_affinity)
            if use_preexec:
                process = subprocess.Popen(
                    sandbox_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=env,
                    cwd=tmp_dir,
                    preexec_fn=set_resource_limits_and_affinity,
                    start_new_session=True
                )
            else:
                process = subprocess.Popen(
                    sandbox_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=env,
                    cwd=tmp_dir,
                    start_new_session=True
                )

            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            stdout_thread = threading.Thread(
                target=_stream_pipe_to_log,
                args=(process.stdout, live_stdout_path, stdout_chunks),
                kwargs={"truncate": True},
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_stream_pipe_to_log,
                args=(process.stderr, live_stderr_path, stderr_chunks),
                kwargs={"truncate": True},
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            process.wait(timeout=timeout)
            returncode = process.returncode
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            stdout_output = "".join(stdout_chunks)
            stderr_output = "".join(stderr_chunks)

        except subprocess.TimeoutExpired:
            print(f"AutoResearchEvaluator: Process timed out after {timeout} seconds. Terminating process group.")
            try:
                # Use os.killpg with the process's session ID (which is its PID due to start_new_session=True)
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
            stdout_chunks = locals().get("stdout_chunks", [])
            stderr_chunks = locals().get("stderr_chunks", [])
            stdout_thread = locals().get("stdout_thread")
            stderr_thread = locals().get("stderr_thread")
            if stdout_thread is not None:
                stdout_thread.join(timeout=5)
            if stderr_thread is not None:
                stderr_thread.join(timeout=5)
            stdout_output = "".join(stdout_chunks)
            stderr_output = "".join(stderr_chunks)
            if not stderr_output:
                stderr_output = "Process timed out and was terminated by the evaluator."
            
            truncated_stderr = truncate_stderr(stderr_output)
            execution_logs = f"PYTHON SANDBOX TIMEOUT:\nSTDOUT:\n{stdout_output}\n\nSTDERR:\n{truncated_stderr}"
            error_msg = f"Execution timed out after {self.timeout} seconds\n\n**Optimization Tips:**\n1. Use GPU-accelerated code when possible (JAX, CuPy, etc.)\n2. Vectorize operations instead of loops\n3. Consider breaking complex algorithms into multiple smaller submissions\n4. Profile your code to identify bottlenecks\n5. Use JIT compilation (@jit decorators) for computational kernels"
            
            return {"success": False, "error": error_msg, "logs": execution_logs, "stdout": stdout_output, "stderr": truncated_stderr}
        
        except Exception as e:
            return {"success": False, "error": f"Python sandbox execution error: {str(e)}", "logs": str(e), "stdout": "", "stderr": str(e)}

        truncated_stderr = truncate_stderr(stderr_output)
        execution_logs = f"PYTHON SANDBOX EXECUTION:\nSTDOUT:\n{stdout_output}\n\nSTDERR:\n{truncated_stderr}"

        success = _function_mode_success(returncode, stdout_output, tmp_dir) if execution_mode == "function" else (returncode == 0)

        if success:
            if execution_mode == "command":
                return {"success": True, "result": stdout_output, "logs": execution_logs, "stdout": stdout_output, "stderr": truncated_stderr}
            else:
                result_npy_path, result_pkl_path = _function_result_paths(tmp_dir)
                
                if os.path.exists(result_npy_path):
                    loaded_result = np.load(result_npy_path)
                    return {"success": True, "result": loaded_result, "logs": execution_logs, "stdout": stdout_output, "stderr": truncated_stderr}
                elif os.path.exists(result_pkl_path):
                    with open(result_pkl_path, 'rb') as f:
                        loaded_result = pickle.load(f)
                    return {"success": True, "result": loaded_result, "logs": execution_logs, "stdout": stdout_output, "stderr": truncated_stderr}
                else:
                    result_txt_path = os.path.join(tmp_dir, "result.txt")
                    result_text = ""
                    if os.path.exists(result_txt_path):
                        with open(result_txt_path, 'r') as f: result_text = f.read()
                    return {"success": False, "error": f"Function returned non-array result: {result_text}", "logs": execution_logs, "stdout": stdout_output, "stderr": truncated_stderr}

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
        
        return {"success": False, "error": error_msg, "logs": execution_logs, "stdout": stdout_output, "stderr": truncated_stderr}
