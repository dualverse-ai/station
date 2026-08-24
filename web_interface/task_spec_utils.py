"""Atomic helpers for the dashboard Research Task specification editor."""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Optional

from filelock import FileLock

from station import constants
from station import file_io_utils
from station.eval_research.runtime_paths import build_runtime_paths


MAX_TASK_SPEC_BYTES = 2 * 1024 * 1024


class TaskSpecConflictError(RuntimeError):
    """Raised when an editor attempts to overwrite a newer task revision."""


def _revision(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_task_spec_snapshot(consts_module=constants) -> Dict[str, Any]:
    paths = build_runtime_paths(consts_module)
    content = file_io_utils.load_text(paths.task_spec_path) or ""
    relative_path = os.path.relpath(paths.task_spec_path, consts_module.BASE_STATION_DATA_PATH)
    try:
        modified_at_ns: Optional[str] = str(os.stat(paths.task_spec_path).st_mtime_ns)
    except (FileNotFoundError, OSError):
        modified_at_ns = None
    return {
        "raw_markdown": content,
        "revision": _revision(content),
        "relative_path": relative_path,
        "modified_at_ns": modified_at_ns,
    }


def save_task_spec_snapshot(
    raw_markdown: str,
    *,
    expected_revision: str,
    consts_module=constants,
) -> Dict[str, Any]:
    if not isinstance(raw_markdown, str):
        raise ValueError("Task specification must be a text string.")
    if "\x00" in raw_markdown:
        raise ValueError("Task specification cannot contain null bytes.")

    normalized = raw_markdown.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        raise ValueError("Task specification cannot be empty.")
    if not normalized.endswith("\n"):
        normalized += "\n"
    if len(normalized.encode("utf-8")) > MAX_TASK_SPEC_BYTES:
        raise ValueError("Task specification exceeds the 2 MiB editor limit.")

    paths = build_runtime_paths(consts_module)
    file_io_utils.ensure_dir_exists(paths.research_root)
    lock_path = f"{paths.task_spec_path}.dashboard.lock"
    with FileLock(lock_path, timeout=30):
        current = get_task_spec_snapshot(consts_module)
        if not expected_revision or expected_revision != current["revision"]:
            raise TaskSpecConflictError(
                "The task specification changed after this editor loaded it. Reload before saving."
            )
        file_io_utils.save_text(normalized, paths.task_spec_path)
        return get_task_spec_snapshot(consts_module)
