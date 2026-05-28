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
Lean execution helpers shared by Theory Room and its auto evaluator.
"""

import os
import re
import subprocess
import tempfile
import time
from typing import List, Optional, Set, Tuple

from station import constants


class LeanRunResult:
    def __init__(self, success: bool, logs: str):
        self.success = success
        self.logs = logs


def _contains_forbidden(content: str) -> Optional[str]:
    if re.search(r"\b(sorry|admit)\b", content):
        return "Submission rejected: found forbidden keyword 'sorry' or 'admit'."
    return None


def strip_imports(code: str) -> str:
    """Remove leading import lines to avoid mid-file import errors when concatenated."""
    kept_lines: List[str] = []
    for line in code.splitlines():
        if line.strip().startswith("import "):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines) + ("\n" if code.endswith("\n") else "")


def _resolve_shared_storage_path(repo_root: str) -> Optional[str]:
    base_path = constants.BASE_STATION_DATA_PATH
    if not os.path.isabs(base_path):
        base_path = os.path.join(repo_root, base_path)
    research_storage = os.path.join(
        base_path,
        constants.ROOMS_DIR_NAME,
        constants.SHORT_ROOM_NAME_RESEARCH,
        constants.RESEARCH_STORAGE_DIR,
    )
    if os.path.islink(research_storage):
        research_storage = os.path.realpath(research_storage)
    shared_dir = os.path.join(research_storage, constants.RESEARCH_STORAGE_SHARED_DIR)
    return shared_dir if os.path.exists(shared_dir) else None


def _get_theory_setup_path(repo_root: str) -> str:
    return os.path.realpath(os.path.join(repo_root, ".theory_setup"))


def _get_shared_snapshot_root(repo_root: str) -> str:
    base_path = constants.BASE_STATION_DATA_PATH
    if not os.path.isabs(base_path):
        base_path = os.path.join(repo_root, base_path)
    return os.path.join(
        base_path,
        constants.ROOMS_DIR_NAME,
        constants.SHORT_ROOM_NAME_THEORY,
        "shared_snapshots",
    )


def _get_theory_env_path(theory_setup: str) -> str:
    return os.path.join(theory_setup, "Station", "TheoryEnv.lean")


def _ensure_theory_setup_storage_link(repo_root: str) -> Optional[str]:
    """
    Ensure Storage.* imports resolve during `lake build`.

    Lake sets its own `LEAN_PATH` for compilation (e.g. `.../.lake/build/lib/lean`) and
    does not reliably preserve externally-provided `LEAN_PATH` additions. To ensure
    `import Storage.Shared...` works, we link the snapshot `Storage/` tree into a path
    that Lake *always* includes: `.../.lake/build/lib/lean/Storage`.
    """
    theory_setup = _get_theory_setup_path(repo_root)
    snapshot_root = _get_shared_snapshot_root(repo_root)
    storage_target = os.path.join(snapshot_root, "Storage")
    # Prefer linking into the Lake build lib root so `Storage` is on the module search path.
    lake_lib_root = os.path.join(theory_setup, ".lake", "build", "lib", "lean")
    storage_link = os.path.join(lake_lib_root, "Storage")

    if not os.path.exists(theory_setup):
        return None
    if not os.path.exists(storage_target):
        return "Shared snapshot storage is missing; rerun setup to snapshot Storage.Shared imports."

    try:
        os.makedirs(lake_lib_root, exist_ok=True)
    except OSError as e:
        return f"Failed to prepare Lake build directory for Storage imports: {e}"

    # Don't clobber an existing directory; only ensure a missing link exists (or points correctly).
    if os.path.lexists(storage_link):
        if os.path.islink(storage_link):
            try:
                if os.path.realpath(storage_link) == os.path.realpath(storage_target):
                    return None
            except OSError:
                return None
        return None

    try:
        os.symlink(storage_target, storage_link)
    except OSError as e:
        return f"Failed to link Storage shared snapshots into Theory setup: {e}"
    return None


def _ensure_snapshot_storage_dir(repo_root: str) -> str:
    snapshot_root = _get_shared_snapshot_root(repo_root)
    storage_root = os.path.join(snapshot_root, "Storage")
    os.makedirs(storage_root, exist_ok=True)
    shared_dir = os.path.join(storage_root, "Shared")
    os.makedirs(shared_dir, exist_ok=True)
    # Make Storage.* imports resolvable during `lake build` by linking the snapshot into the project.
    _ensure_theory_setup_storage_link(repo_root)
    return shared_dir


def _parse_storage_shared_imports(contents: List[str]) -> Tuple[Set[str], Set[str]]:
    modules: Set[str] = set()
    suspicious: Set[str] = set()
    for content in contents:
        for line in content.splitlines():
            head = line.split("--", 1)[0].strip()
            if not head.startswith("import "):
                continue
            tail = head[len("import ") :].strip()
            if not tail:
                continue
            tokens = tail.replace("{", " ").replace("}", " ").replace(",", " ").split()
            for token in tokens:
                if token.startswith("Storage.Shared"):
                    modules.add(token)
                    if token == "Storage.Shared":
                        suspicious.add(token)
            if "Storage.Shared" in tail and not any(t.startswith("Storage.Shared") for t in tokens):
                suspicious.add(tail)
    return modules, suspicious


def _parse_import_tokens(line: str) -> List[str]:
    tail = line[len("import ") :].strip()
    if not tail:
        return []
    tokens = tail.replace("{", " ").replace("}", " ").replace(",", " ").split()
    return tokens


def _load_theory_env_text(repo_root: str) -> str:
    theory_env_path = _get_theory_env_path(_get_theory_setup_path(repo_root))
    if not os.path.exists(theory_env_path):
        return ""
    try:
        with open(theory_env_path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _snapshot_storage_shared_imports(repo_root: str, contents: List[str]) -> Tuple[Optional[str], Optional[str]]:
    modules, suspicious = _parse_storage_shared_imports(contents)
    if not modules:
        return None, None

    shared_dir = _ensure_snapshot_storage_dir(repo_root)
    link_error = _ensure_theory_setup_storage_link(repo_root)
    if link_error:
        return link_error, None
    missing: List[str] = []
    missing_in_snapshot: List[str] = []
    shared_storage = _resolve_shared_storage_path(repo_root)

    for module in sorted(modules):
        if module == "Storage.Shared":
            rel_path = "Shared.lean"
        else:
            rel = module[len("Storage.Shared.") :]
            rel_path = rel.replace(".", os.sep) + ".lean"
        dest_path = os.path.join(shared_dir, rel_path)
        if os.path.exists(dest_path):
            continue
        missing_in_snapshot.append(module)
        if not shared_storage:
            continue
        source_path = os.path.join(shared_storage, rel_path)
        if not os.path.exists(source_path):
            missing.append(module)
            continue
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            with open(source_path, "rb") as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
        except OSError:
            missing.append(module)

    if missing_in_snapshot and not shared_storage:
        return "Shared storage is not available for Storage.Shared imports.", None
    if missing:
        return "Missing shared modules: " + ", ".join(missing), None
    if suspicious:
        return None, (
            "Non-standard Storage.Shared import detected. "
            "Use `import Storage.Shared.<path>.Module` or `import {Storage.Shared.<path>.A, ...}`."
        )
    return None, None


def _resolve_lake_bin() -> str:
    lean_bin = os.environ.get("LEAN_BIN")
    if lean_bin:
        candidate = os.path.join(os.path.dirname(lean_bin), "lake")
        if os.path.exists(candidate):
            return candidate
    candidate = os.path.expanduser("~/.elan/bin/lake")
    return candidate if os.path.exists(candidate) else "lake"


def _resolve_lean_bin() -> str:
    lean_bin = os.environ.get("LEAN_BIN")
    if lean_bin:
        return lean_bin
    candidate = os.path.expanduser("~/.elan/bin/lean")
    return candidate if os.path.exists(candidate) else "lean"


def _collect_imports_and_body(contents: List[str]) -> Tuple[List[str], str]:
    seen = set()
    imports: List[str] = []
    for content in contents:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped == "import Station.Theory":
                continue
            if stripped.startswith("import "):
                if stripped not in seen:
                    seen.add(stripped)
                    imports.append(stripped)
    if "import Mathlib" not in seen:
        imports.insert(0, "import Mathlib")

    body_parts: List[str] = []
    for content in contents:
        cleaned = strip_imports(content).strip()
        if cleaned:
            body_parts.append(cleaned)
    body = "\n\n".join(body_parts)
    return imports, body


def _snapshot_shared_module_paths(repo_root: str, modules: Set[str]) -> List[Tuple[str, str]]:
    snapshot_root = _get_shared_snapshot_root(repo_root)
    shared_root = os.path.join(snapshot_root, "Storage", "Shared")
    paths: List[Tuple[str, str]] = []
    for module in sorted(modules):
        if module == "Storage.Shared":
            rel_path = "Shared.lean"
        else:
            rel = module[len("Storage.Shared.") :]
            rel_path = rel.replace(".", os.sep) + ".lean"
        src_path = os.path.join(shared_root, rel_path)
        olean_path = os.path.splitext(src_path)[0] + ".olean"
        paths.append((src_path, olean_path))
    return paths


def _ensure_snapshot_oleans(repo_root: str, modules: Set[str]) -> Optional[str]:
    if not modules:
        return None
    snapshot_root = _get_shared_snapshot_root(repo_root)
    if not os.path.exists(snapshot_root):
        return "Shared snapshot root is missing; rerun setup to snapshot Storage.Shared imports."

    lean_bin = _resolve_lean_bin()
    theory_setup = _get_theory_setup_path(repo_root)
    default_paths: List[str] = []
    deps_root = os.path.join(theory_setup, ".lake", "packages")
    if os.path.exists(deps_root):
        for dep in os.listdir(deps_root):
            dep_path = os.path.join(deps_root, dep, ".lake", "build", "lib", "lean")
            if os.path.exists(dep_path):
                default_paths.append(dep_path)
    default_paths.append(os.path.join(theory_setup, ".lake", "build", "lib", "lean"))
    default_paths.append(theory_setup)
    default_paths.insert(0, snapshot_root)
    lean_path = os.pathsep.join([p for p in default_paths if p and os.path.exists(p)])

    env = os.environ.copy()
    env["LEAN_PATH"] = lean_path
    for src_path, olean_path in _snapshot_shared_module_paths(repo_root, modules):
        if not os.path.exists(src_path):
            return f"Missing shared snapshot source file: {src_path}"
        if os.path.exists(olean_path):
            continue
        ilean_path = os.path.splitext(src_path)[0] + ".ilean"
        os.makedirs(os.path.dirname(olean_path), exist_ok=True)
        cmd = [lean_bin, src_path, "-o", olean_path, "-i", ilean_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        except Exception as e:
            return f"Failed to compile shared module {src_path}: {e}"
        if proc.returncode != 0:
            return (
                f"Failed to compile shared module {src_path}:\n"
                f"{(proc.stdout or '')}{(proc.stderr or '')}".strip()
            )
    return None


def build_theory_project(repo_root: str, contents: List[str], timeout: int = 1200) -> LeanRunResult:
    theory_setup = _get_theory_setup_path(repo_root)
    if not os.path.exists(theory_setup):
        return LeanRunResult(False, f"Missing Theory setup directory at {theory_setup}.")

    theory_env_path = _get_theory_env_path(theory_setup)
    os.makedirs(os.path.dirname(theory_env_path), exist_ok=True)

    snapshot_error, snapshot_warning = _snapshot_storage_shared_imports(repo_root, contents)
    if snapshot_error:
        return LeanRunResult(False, snapshot_error)
    modules, _ = _parse_storage_shared_imports(contents)
    olean_error = _ensure_snapshot_oleans(repo_root, modules)
    if olean_error:
        return LeanRunResult(False, olean_error)

    imports, body = _collect_imports_and_body(contents)
    new_code = "\n".join(imports)
    if body:
        new_code = f"{new_code}\n\n{body}\n"
    try:
        with open(theory_env_path, "r", encoding="utf-8") as f:
            old_code = f.read()
    except FileNotFoundError:
        old_code = ""

    with open(theory_env_path, "w", encoding="utf-8") as f:
        f.write(new_code)

    lake_bin = _resolve_lake_bin()
    cmd = [lake_bin, "build", "StationTheory"]
    env = os.environ.copy()
    snapshot_root = _get_shared_snapshot_root(repo_root)
    if os.path.exists(snapshot_root):
        existing = env.get("LEAN_PATH", "")
        parts = [p for p in existing.split(os.pathsep) if p.strip()] if existing else []
        if snapshot_root not in parts:
            parts.insert(0, snapshot_root)
        env["LEAN_PATH"] = os.pathsep.join(parts)
    start_ts = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=theory_setup,
            env=env,
            timeout=timeout,
        )
    except Exception as e:
        with open(theory_env_path, "w", encoding="utf-8") as f:
            f.write(old_code)
        return LeanRunResult(False, f"Lake build failed to start: {e}")

    duration = time.time() - start_ts
    logs = (
        f"[lake start] bin={lake_bin} timeout={timeout}s\n"
        f"{(proc.stdout or '')}{(proc.stderr or '')}\n"
        f"[lake end] code={proc.returncode} duration={duration:.2f}s"
    ).strip()
    if snapshot_warning:
        logs = f"{logs}\n[warning] {snapshot_warning}"
    if proc.returncode != 0:
        with open(theory_env_path, "w", encoding="utf-8") as f:
            f.write(old_code)
        return LeanRunResult(False, logs)

    return LeanRunResult(True, logs)


def run_lean_submission(
    content: str,
    env_prefix: str = "",
    formal_statement: Optional[str] = None,
    formal_definitions: Optional[str] = None,
    allow_sorry: bool = False,
) -> LeanRunResult:
    """Execute Lean with provided content and optional env prefix."""
    forbid = _contains_forbidden(content)
    sorry_used = False
    if forbid:
        if allow_sorry:
            sorry_used = True
        else:
            return LeanRunResult(False, forbid)

    # Hoist imports to the top to avoid "import must be at beginning" errors once env code is prepended.
    base_imports = ["import Mathlib", "import Station.Theory"]
    user_imports: List[str] = []
    body_lines: List[str] = []
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env_text = _load_theory_env_text(repo_root)
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            tokens = _parse_import_tokens(stripped)
            if any(t.startswith("Storage.Shared") for t in tokens) and env_text:
                remaining = []
                for t in tokens:
                    if t.startswith("Storage.Shared"):
                        module_name = t.split(".")[-1]
                        if t in env_text or (module_name and module_name in env_text):
                            continue
                    remaining.append(t)
                if remaining:
                    user_imports.append("import " + " ".join(remaining))
            else:
                user_imports.append(stripped)
        else:
            body_lines.append(line)

    # Deduplicate imports while preserving order
    seen = set()
    ordered_imports: List[str] = []
    for im in base_imports + user_imports:
        if im not in seen:
            seen.add(im)
            ordered_imports.append(im)

    lean_parts: List[str] = []
    lean_parts.extend(ordered_imports)
    if env_prefix:
        lean_parts.append(env_prefix)
    if body_lines:
        lean_parts.append("\n".join(body_lines))

    if formal_statement:
        lean_parts.append(f"#check ({formal_statement})")
    # Intentionally skip auto-`#check` of formal_definitions to avoid brittle parsing of rich definitions.

    lean_code = "\n".join(lean_parts) + "\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "submission.lean")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(lean_code)

        # Prefer explicit LEAN_BIN (env), else ~/.elan/bin/lean, else PATH default.
        lean_bin = _resolve_lean_bin()

        cmd = [lean_bin, file_path]
        env = os.environ.copy()
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        theory_setup = os.path.realpath(os.path.join(repo_root, ".theory_setup"))
        default_paths: List[str] = []
        deps_root = os.path.join(theory_setup, ".lake", "packages")
        if os.path.exists(deps_root):
            for dep in os.listdir(deps_root):
                dep_path = os.path.join(deps_root, dep, ".lake", "build", "lib", "lean")
                if os.path.exists(dep_path):
                    default_paths.append(dep_path)
        snapshot_root = _get_shared_snapshot_root(repo_root)
        if os.path.exists(snapshot_root):
            default_paths.append(snapshot_root)
        default_paths.append(os.path.join(theory_setup, ".lake", "build", "lib", "lean"))
        default_paths.append(theory_setup)
        snapshot_error, snapshot_warning = _snapshot_storage_shared_imports(repo_root, [content])
        if snapshot_error:
            return LeanRunResult(False, snapshot_error)
        modules, _ = _parse_storage_shared_imports([content])
        olean_error = _ensure_snapshot_oleans(repo_root, modules)
        if olean_error:
            return LeanRunResult(False, olean_error)
        existing = env.get("LEAN_PATH", "")
        path_parts = [p for p in existing.split(os.pathsep) if p.strip()] if existing else []
        if os.path.exists(snapshot_root) and snapshot_root in path_parts:
            path_parts = [p for p in path_parts if p != snapshot_root]
        if os.path.exists(snapshot_root):
            path_parts.insert(0, snapshot_root)
        for p in default_paths:
            if p and os.path.exists(p) and p not in path_parts:
                path_parts.append(p)
        if tmpdir not in path_parts:
            path_parts.append(tmpdir)
        env["LEAN_PATH"] = os.pathsep.join(path_parts)

        # Lean can take time to load Mathlib in some environments; allow generous timeout.
        start_ts = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir, env=env, timeout=600)
        duration = time.time() - start_ts
        combined = (proc.stdout + proc.stderr).splitlines()
        cleaned_lines: List[str] = []
        for line in combined:
            if "try 'simp' instead of 'simpa'" in line:
                continue
            if "linter.unnecessarySimpa" in line:
                continue
            if line.strip().startswith("Note:"):
                continue
            cleaned_lines.append(line)
        success = proc.returncode == 0
        cleaned_logs = "\n".join(cleaned_lines)
        logs = (
            f"[lean start] bin={lean_bin} timeout=600s\n"
            f"{cleaned_logs}\n"
            f"[lean end] code={proc.returncode} duration={duration:.2f}s"
        ).strip()
        if snapshot_warning:
            logs = f"{logs}\n[warning] {snapshot_warning}"
        if allow_sorry:
            # Flag the use of sorry/admit in sandbox runs so users know the run is not a proof.
            if sorry_used or re.search(r"\b(sorry|admit)\b", content):
                logs += "\n[warning] sandbox detected usage of 'sorry'/'admit'; success does not mean the proof is complete."
        return LeanRunResult(success, logs)
