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
Small shared helpers for Research Center runtime configuration.
"""

from __future__ import annotations

import os
import subprocess
from typing import Dict, Optional

from station import constants
from station import file_io_utils


_CONDA_ENV_CACHE: dict[tuple[str, str, str], Optional[str]] = {}
PERSISTED_STDOUT_TRUNCATION_MARKER = "[truncated due to exceeds char limit]"


def truncate_stderr(stderr_output: str) -> str:
    """Truncate stderr to the configured maximum length."""
    if not stderr_output:
        return stderr_output

    max_chars = constants.RESEARCH_EVAL_STDERR_MAX_CHARS
    if len(stderr_output) > max_chars:
        return stderr_output[:max_chars] + f"\n\n[... stderr truncated after {max_chars:,} characters]"
    return stderr_output


def truncate_persisted_stdout(stdout_output: str, *, marker: str = PERSISTED_STDOUT_TRUNCATION_MARKER) -> str:
    if not stdout_output:
        return stdout_output

    max_chars = getattr(constants, "RESEARCH_EVAL_PERSISTED_STDOUT_MAX_CHARS", 1000000)
    if len(stdout_output) <= max_chars:
        return stdout_output

    if max_chars <= len(marker):
        return marker[:max_chars]

    preserved = stdout_output[: max_chars - len(marker) - 1].rstrip()
    if preserved:
        return preserved + "\n" + marker
    return marker


def save_stdout_with_limit(path: str, text: str):
    file_io_utils.save_text(truncate_persisted_stdout(text or ""), path)


def append_stdout_with_limit(path: Optional[str], text: str):
    if not path or not text:
        return

    file_io_utils.ensure_dir_exists(os.path.dirname(path))
    existing = file_io_utils.load_text(path) if file_io_utils.file_exists(path) else ""
    if existing and existing.rstrip().endswith(PERSISTED_STDOUT_TRUNCATION_MARKER):
        return
    file_io_utils.save_text(truncate_persisted_stdout((existing or "") + text), path)


def _find_conda_executable(env: Dict[str, str]) -> Optional[str]:
    conda_executable = env.get("CONDA_BIN_PATH")
    if conda_executable and os.path.exists(conda_executable) and os.access(conda_executable, os.X_OK):
        return conda_executable

    conda_exec_candidates = [
        os.path.join(os.path.expanduser("~root"), "miniconda3", "bin", "conda"),
        os.path.join(os.path.expanduser("~"), "miniconda3", "bin", "conda"),
        os.path.join(os.path.expanduser("~"), "miniforge", "bin", "conda"),
        "/opt/conda/bin/conda",
        "/usr/local/bin/conda",
        "/usr/bin/conda",
    ]
    for candidate in conda_exec_candidates:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate

    try:
        result = subprocess.run(["which", "conda"], capture_output=True, text=True, env=env, check=False)
    except FileNotFoundError:
        return None

    discovered = result.stdout.strip()
    if result.returncode == 0 and discovered:
        return discovered
    return None


def _is_executable_file(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(path) and os.access(path, os.X_OK)


def _candidate_python_from_prefix(prefix: Optional[str]) -> Optional[str]:
    if not prefix:
        return None
    candidate = os.path.join(prefix, "bin", "python")
    return candidate if _is_executable_file(candidate) else None


def _prefix_from_python(python_path: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.realpath(python_path)))


def _normalize_conda_env_vars(conda_env_name: str, env: Dict[str, str], python_path: str, conda_executable: Optional[str]) -> None:
    conda_prefix = _prefix_from_python(python_path)
    conda_bin_path = os.path.join(conda_prefix, "bin")
    path_parts = [part for part in str(env.get("PATH", "")).split(os.pathsep) if part]
    if conda_bin_path in path_parts:
        path_parts.remove(conda_bin_path)
    path_parts.insert(0, conda_bin_path)
    env["PATH"] = os.pathsep.join(path_parts)
    env["CONDA_DEFAULT_ENV"] = conda_env_name
    env["CONDA_PREFIX"] = conda_prefix
    if conda_executable:
        env["CONDA_BIN_PATH"] = conda_executable


def _candidate_conda_prefixes(conda_env_name: str, env: Dict[str, str], conda_executable: Optional[str]) -> list[str]:
    prefixes: list[str] = []
    conda_prefix = env.get("CONDA_PREFIX")
    if env.get("CONDA_DEFAULT_ENV") == conda_env_name and conda_prefix:
        prefixes.append(conda_prefix)

    if conda_executable:
        conda_root = os.path.dirname(os.path.dirname(conda_executable))
        prefixes.append(os.path.join(conda_root, "envs", conda_env_name))
        prefixes.append(os.path.join(conda_root, conda_env_name))

    home = os.path.expanduser("~")
    prefixes.append(os.path.join(home, "miniconda3", "envs", conda_env_name))
    prefixes.append(os.path.join(home, "miniforge", "envs", conda_env_name))
    prefixes.append(os.path.join("/opt/conda", "envs", conda_env_name))
    prefixes.append(os.path.join("/usr/local/miniconda3", "envs", conda_env_name))
    return prefixes


def _resolve_python_from_prefixes(prefixes: list[str]) -> Optional[str]:
    seen: set[str] = set()
    for prefix in prefixes:
        normalized = os.path.realpath(prefix) if prefix else prefix
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        python_path = _candidate_python_from_prefix(normalized)
        if python_path:
            return python_path
    return None


def _resolve_python_with_conda_run(conda_env_name: str, env: Dict[str, str], conda_executable: Optional[str]) -> Optional[str]:
    if not conda_executable:
        return None
    command = [conda_executable, "run", "-n", conda_env_name, "which", "python"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=env, check=True)
    except (subprocess.CalledProcessError, OSError):
        return None
    resolved = result.stdout.strip()
    return resolved if _is_executable_file(resolved) else None


def _resolve_conda_python_path(conda_env_name: str, env: Dict[str, str], conda_executable: Optional[str]) -> Optional[str]:
    explicit_python = env.get("STATION_PYTHON_EXECUTABLE")
    if _is_executable_file(explicit_python):
        return explicit_python
    return (
        _resolve_python_from_prefixes(_candidate_conda_prefixes(conda_env_name, env, conda_executable))
        or _resolve_python_with_conda_run(conda_env_name, env, conda_executable)
    )


def resolve_conda_env(conda_env_name: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Resolve a conda environment, normalize PATH/CONDA vars, and return Python.
    """
    runtime_env = env if env is not None else os.environ.copy()

    explicit_python = runtime_env.get("STATION_PYTHON_EXECUTABLE")
    if _is_executable_file(explicit_python):
        _normalize_conda_env_vars(conda_env_name, runtime_env, explicit_python, None)
        return explicit_python

    if runtime_env.get("CONDA_DEFAULT_ENV") == conda_env_name:
        active_python = _candidate_python_from_prefix(runtime_env.get("CONDA_PREFIX"))
        if active_python:
            _normalize_conda_env_vars(conda_env_name, runtime_env, active_python, None)
            return active_python

    conda_executable = _find_conda_executable(runtime_env)
    cache_key = (
        conda_env_name,
        runtime_env.get("STATION_PYTHON_EXECUTABLE", ""),
        runtime_env.get("CONDA_PREFIX", ""),
        conda_executable or "",
    )
    if cache_key in _CONDA_ENV_CACHE:
        python_path = _CONDA_ENV_CACHE[cache_key]
    else:
        python_path = _resolve_conda_python_path(conda_env_name, runtime_env, conda_executable)
        _CONDA_ENV_CACHE[cache_key] = python_path

    if not python_path:
        return None

    _normalize_conda_env_vars(conda_env_name, runtime_env, python_path, conda_executable)
    return python_path


def find_conda_python(conda_env_name: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Resolve the Python executable for a conda environment.

    Also normalizes PATH/CONDA variables in ``env`` when a mutable environment
    is provided, so external tools resolve from the same env as Python.
    """
    return resolve_conda_env(conda_env_name, env)


def setup_conda_env(conda_env_name: str, env: Dict[str, str]) -> bool:
    """
    Update PATH/CONDA variables for the requested conda environment.
    """
    return bool(resolve_conda_env(conda_env_name, env))
