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

# station/eval_research/executor_docker.py
"""
Docker execution methods for research evaluation.
"""

import os
import subprocess
import tempfile
import time
import pickle
import numpy as np
from typing import Dict, Any, Optional

from station import constants
from station import file_io_utils
from station.eval_research.evaluation_helpers import truncate_stderr


def _execute_submission_in_docker(self, eval_entry: Dict[str, Any], evaluator, gpu_ids: Optional[list] = None) -> Dict[str, Any]:
    """Execute research submission code in Docker container"""
    content = eval_entry.get(constants.EVALUATION_CONTENT_KEY, "")
    author = eval_entry.get(constants.EVALUATION_AUTHOR_KEY, "Unknown")
    
    # Use the configured timeout directly
    remaining_timeout = self.timeout
    
    # Prepare storage paths
    shared_storage_path = os.path.join(
        self.research_room_path, 
        constants.RESEARCH_STORAGE_DIR, 
        constants.RESEARCH_STORAGE_SHARED_DIR
    )
    
    system_storage_path = os.path.join(
        self.research_room_path,
        constants.RESEARCH_STORAGE_DIR,
        constants.RESEARCH_STORAGE_SYSTEM_DIR
    )
    
    # Architect hint storage path
    architect_storage_path = os.path.join(
        self.research_room_path,
        constants.RESEARCH_STORAGE_DIR,
        "architect"
    )
    
    # Get author's lineage for lineage-specific storage
    try:
        author_data = self.station.agent_module.load_agent_data(author)
        if author_data:
            author_lineage_original = author_data.get(constants.AGENT_LINEAGE_KEY, "unknown")
            author_lineage = author_lineage_original.lower()  # normalized for filesystem
        else:
            author_lineage_original = "unknown"
            author_lineage = "unknown"
    except Exception:
        author_lineage_original = "unknown"
        author_lineage = "unknown"
    
    author_lineage_storage_path = os.path.join(
        self.research_room_path,
        constants.RESEARCH_STORAGE_DIR,
        constants.RESEARCH_STORAGE_LINEAGES_DIR,
        author_lineage
    )
    
    # Get all lineage storage directories for read-only mounting
    lineages_base_path = os.path.join(
        self.research_room_path,
        constants.RESEARCH_STORAGE_DIR,
        constants.RESEARCH_STORAGE_LINEAGES_DIR
    )
    
    # Ensure storage directories exist
    file_io_utils.ensure_dir_exists(shared_storage_path)
    file_io_utils.ensure_dir_exists(system_storage_path)
    file_io_utils.ensure_dir_exists(author_lineage_storage_path)
    file_io_utils.ensure_dir_exists(architect_storage_path)
    
    # Create temporary directory for execution
    with tempfile.TemporaryDirectory() as temp_dir:
        # Check execution mode
        execution_mode = evaluator.get_execution_mode() if hasattr(evaluator, 'get_execution_mode') else "function"
        
        if execution_mode == "command":
            # Command mode: Save submission with specified filename and run command
            submission_filename = evaluator.get_submission_filename()
            submission_path = os.path.join(temp_dir, submission_filename)
            with open(submission_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Get the command to execute
            execution_command = evaluator.get_execution_command()
            
            if execution_command.startswith("python ") and " -u " not in execution_command:
                # Replace the first instance of "python " with "python -u "
                execution_command = execution_command.replace("python ", "python -u ", 1)

        else:
            # Function mode (default): Write submission code to run.py
            run_py_path = os.path.join(temp_dir, "run.py")
            with open(run_py_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Create execution wrapper based on mode
        if execution_mode == "command":
            # Command mode wrapper
            wrapper_content = f"""
import sys
import subprocess
import os

# Set up matplotlib config directory to prevent warnings
os.environ['MPLCONFIGDIR'] = '/tmp/.matplotlib'
os.makedirs('/tmp/.matplotlib', exist_ok=True)

# Run the execution command
try:
    cmd = {execution_command!r}
    print(f"Executing command: {{cmd}}")
    # Set PYTHONPATH to include /app directory so train.py can import submission
    env = os.environ.copy()
    env['PYTHONPATH'] = '/app' + os.pathsep + env.get('PYTHONPATH', '')
    env['MPLCONFIGDIR'] = '/tmp/.matplotlib'  # Ensure it's passed to subprocess
    env['PYTHONWARNINGS'] = 'ignore::UserWarning'  # Suppress UserWarnings in subprocess
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    
    # Output stdout and stderr
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    # Exit with same return code as the command
    sys.exit(result.returncode)
except Exception as e:
    print(f"EXECUTION_ERROR: Failed to run command: {{e}}")
    sys.exit(1)
"""
        else:
            # Function mode wrapper (existing behavior)
            wrapper_content = f"""
import sys
import traceback
import pickle
import numpy as np
import warnings
import os

# Suppress warnings from matplotlib and jumanji
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['MPLCONFIGDIR'] = '/tmp/.matplotlib'

# Create matplotlib directory to prevent warnings
os.makedirs('/tmp/.matplotlib', exist_ok=True)

# Import the submitted code
try:
    from run import {evaluator.get_expected_function_name()}
except ImportError as e:
    print(f"IMPORT_ERROR: Cannot import {evaluator.get_expected_function_name()}: {{e}}")
    print("IMPORT_DETAILS: Full import error traceback:")
    traceback.print_exc()
    print("IMPORT_HINT: Check that all imported modules exist and have correct names.")
    print("IMPORT_HINT: Common issues: typos in module names, missing .py files, wrong file paths.")
    sys.exit(1)
except Exception as e:
    print(f"IMPORT_ERROR: Error importing {evaluator.get_expected_function_name()}: {{e}}")
    print("IMPORT_DETAILS: Full error traceback:")
    traceback.print_exc()
    print("IMPORT_HINT: This may be a syntax error or other issue in your code.")
    sys.exit(1)

try:
    # Execute the function
    result = {evaluator.get_expected_function_name()}()
    
    # Serialize result using numpy
    if hasattr(result, 'shape') and hasattr(result, 'dtype'):
        # Save as numpy array
        np.save('/app/result.npy', result)
        print(f"EXECUTION_SUCCESS: Function returned result with shape {{result.shape}} and dtype {{result.dtype}}")
    else:
        # Save non-array results (like tuples) using pickle
        with open('/app/result.pkl', 'wb') as f:
            pickle.dump(result, f)
        print(f"EXECUTION_SUCCESS: Function returned result (non-array): {{type(result)}}")
    
except Exception as e:
    print(f"EXECUTION_ERROR: {{e}}")
    traceback.print_exc()
    sys.exit(1)
"""
        
        wrapper_path = os.path.join(temp_dir, "wrapper.py")
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_content)
        
        # Build Docker command with storage mounts
        docker_cmd = self.docker_prefix + [
            'docker', 'run', '--rm',
            '-v', f'{temp_dir}:/app',
            '-v', f'{shared_storage_path}:/storage/shared:rw',
            '-v', f'{system_storage_path}:/storage/system:ro',
            '-v', f'{architect_storage_path}:/storage/architect:ro',
        ]
        
        # Mount author's own lineage storage as read-write
        docker_cmd.extend(['-v', f'{author_lineage_storage_path}:/storage/{author_lineage}:rw'])
        
        # Mount all lineage directories with both original case and lowercase access
        # Only if cross-lineage storage access is allowed
        if constants.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS and os.path.exists(lineages_base_path):
            try:
                for lineage_name in os.listdir(lineages_base_path):
                    lineage_path = os.path.join(lineages_base_path, lineage_name)
                    if os.path.isdir(lineage_path):
                        if lineage_name != author_lineage:
                            # Mount other lineages as read-only with original case
                            docker_cmd.extend(['-v', f'{lineage_path}:/storage/{lineage_name}:ro'])
                            
                            # Also mount with lowercase name for convenience (if different)
                            if lineage_name.lower() != lineage_name:
                                docker_cmd.extend(['-v', f'{lineage_path}:/storage/{lineage_name.lower()}:ro'])
            except OSError:
                pass  # Ignore errors listing lineage directories
        
        # If author's original lineage case differs from normalized case (e.g., "Cognita" vs "cognita"),
        # mount the same storage to both the original case path for backwards compatibility
        if author_lineage_original != author_lineage:
            docker_cmd.extend(['-v', f'{author_lineage_storage_path}:/storage/{author_lineage_original}:rw'])
        
        # Check if Ray cluster connectivity is needed
        ray_address = os.environ.get('RAY_HEAD_NODE_IP')

        # Add remaining Docker options
        docker_options = [
            '--gpus', 'all',  # Enable GPU access
            '--cpus', self.cpu_limit,
            '--user', 'researcher'
        ]

        # SECURITY CONSIDERATIONS for conditional network access:
        # 1. Network is ONLY enabled when RAY_HEAD_NODE_IP environment variable is set
        # 2. This allows Docker containers to connect to external Ray clusters for distributed training
        # 3. Risk: Containers with network access could potentially reach other network services
        # 4. Mitigation: Only enable for trusted research tasks that genuinely need Ray cluster access
        # 5. Alternative: Consider using custom Docker network with firewall rules to restrict to Ray port only
        # 6. Using 'host' network mode for full bidirectional Ray communication:
        #    - Ray workers need to receive connections from the head node for data transfer
        #    - Bridge network's NAT would block these inbound connections
        #    - Host network gives container same network stack as host machine
        # 7. For production use, consider network policies or firewall rules to limit access

        # Conditional network access: enable only for Ray cluster connectivity
        if ray_address:
            # Use host network for full bidirectional Ray communication
            # Ray requires both outbound (worker->head) and inbound (head->worker) connectivity
            docker_options.extend(['--network', 'host'])
            self.log(f"Network enabled (host mode) for Ray cluster connectivity at {ray_address}", "warning")
        else:
            # Maintain isolation for non-Ray tasks (default secure mode)
            docker_options.extend(['--network', 'none'])

        # Add memory limit only if configured
        if self.memory_limit:
            docker_options.extend(['--memory', self.memory_limit])

        docker_cmd.extend(docker_options)

        # Pass environment variables
        # Set CUDA_VISIBLE_DEVICES if specific GPUs are allocated
        if gpu_ids is not None:
            gpu_list = ','.join(map(str, gpu_ids))
            docker_cmd.extend(['-e', f'CUDA_VISIBLE_DEVICES={gpu_list}'])

        # Pass Ray cluster address if defined
        if ray_address:
            docker_cmd.extend(['-e', f'RAY_HEAD_NODE_IP={ray_address}'])
            self.log(f"Passing RAY_HEAD_NODE_IP={ray_address} to container", "info")

        # Pass storage path mapping for Ray workers (Option B)
        if constants.RESEARCH_STORAGE_BASE_PATH and ray_address:
            # Get the real storage path on host for Ray workers
            real_storage = os.path.realpath(shared_storage_path)
            docker_cmd.extend(['-e', f'RESEARCH_STORAGE_HOST_PATH={real_storage}'])
            self.log(f"Passing RESEARCH_STORAGE_HOST_PATH={real_storage} for Ray workers", "info")
        
        docker_cmd.extend([
            self.docker_image,
            'bash', '-c', f'cd / && timeout {remaining_timeout} python -u /app/wrapper.py'
        ])
        
        try:
            # Use Popen for better partial output capture
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout_output, stderr_output = process.communicate(timeout=remaining_timeout + 300)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout_output, stderr_output = process.communicate()
                returncode = 124
            
            # Filter out CUDA banner from stdout
            stdout_output = self._filter_cuda_banner(stdout_output)
            
            # Truncate stderr to prevent log flooding
            truncated_stderr = truncate_stderr(stderr_output)
            execution_logs = f"STDOUT:\n{stdout_output}\n\nSTDERR:\n{truncated_stderr}"
            
            # Check if execution was successful based on mode
            if execution_mode == "command":
                # Command mode: return full stdout as result if command succeeded
                if returncode == 0:
                    return {
                        "success": True,
                        "result": stdout_output,  # Return FULL stdout for evaluator to parse
                        "logs": execution_logs
                    }
            else:
                # Function mode: check for EXECUTION_SUCCESS and load numpy result
                if returncode == 0 and "EXECUTION_SUCCESS" in stdout_output:
                    # Load the numpy result
                    result_npy_path = os.path.join(temp_dir, "result.npy")
                    result_txt_path = os.path.join(temp_dir, "result.txt")
                    
                    result_pkl_path = os.path.join(temp_dir, "result.pkl")
                    
                    if os.path.exists(result_npy_path):
                        algorithm_result = np.load(result_npy_path)
                        return {
                            "success": True,
                            "result": algorithm_result,
                            "logs": execution_logs
                        }
                    elif os.path.exists(result_pkl_path):
                        with open(result_pkl_path, 'rb') as f:
                            algorithm_result = pickle.load(f)
                        return {
                            "success": True,
                            "result": algorithm_result,
                            "logs": execution_logs
                        }
                    elif os.path.exists(result_txt_path):
                        with open(result_txt_path, 'r') as f:
                            result_text = f.read()
                        
                        return {
                            "success": False,
                            "error": f"Function returned non-array result: {result_text}",
                            "logs": execution_logs
                        }
                    else:
                        return {
                            "success": False,
                            "error": "Function executed but no result file found",
                            "logs": execution_logs
                        }
            
            # If we get here, execution failed in either mode
            # Determine error type
            if returncode == 124:  # timeout exit code
                error_msg = f"Execution timed out after {self.timeout} seconds\n\n**Optimization Tips:**\n1. Use GPU-accelerated code when possible (JAX, CuPy, etc.)\n2. Vectorize operations instead of loops\n3. Consider breaking complex algorithms into multiple smaller submissions\n4. Profile your code to identify bottlenecks\n5. Use JIT compilation (@jit decorators) for computational kernels"
            elif "IMPORT_ERROR" in stdout_output:
                # Extract the specific import error from stdout
                import_error_lines = []
                for line in stdout_output.split('\n'):
                    if line.strip() and ('IMPORT_ERROR' in line or 'IMPORT_DETAILS' in line or 'IMPORT_HINT' in line):
                        import_error_lines.append(line.strip())
                
                if import_error_lines:
                    # Get the main error message
                    main_error = import_error_lines[0] if import_error_lines else "Import error"
                    # Remove the IMPORT_ERROR: prefix for cleaner message
                    if main_error.startswith("IMPORT_ERROR: "):
                        main_error = main_error[14:]
                    error_msg = f"Import failed: {main_error}"
                else:
                    error_msg = "Failed to import required function from submitted code"
            elif "EXECUTION_ERROR" in stdout_output:
                error_msg = "Runtime error during function execution"
            else:
                error_msg = f"Docker execution failed with exit code {returncode}"
            
            return {
                "success": False,
                "error": error_msg,
                "logs": execution_logs
            }
                
        except subprocess.TimeoutExpired as e:
            # Try to get partial output from the exception
            stdout_output = getattr(e, 'stdout', '') or ""
            stderr_output = getattr(e, 'stderr', '') or ""
            truncated_stderr = truncate_stderr(stderr_output)
            execution_logs = f"STDOUT:\n{stdout_output}\n\nSTDERR:\n{truncated_stderr}"
            return {
                "success": False,
                "error": f"Docker execution timed out after {self.timeout + 30} seconds",
                "logs": execution_logs
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Docker execution error: {str(e)}",
                "logs": str(e)
            }