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

import fcntl
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List

import yaml

from station import file_io_utils


_PROCESS_LOCK = threading.RLock()


@contextmanager
def _queue_lock(queue_path: str, timeout_seconds: float = 30.0):
    lock_path = f"{queue_path}.lock"
    file_io_utils.ensure_dir_exists(os.path.dirname(lock_path))
    deadline = time.monotonic() + timeout_seconds
    with _PROCESS_LOCK, open(lock_path, "a+", encoding="utf-8") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out locking external report queue: {queue_path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load(queue_path: str) -> List[Dict[str, Any]]:
    if not file_io_utils.file_exists(queue_path):
        return []
    return [entry for entry in file_io_utils.load_yaml_lines(queue_path) if isinstance(entry, dict)]


def _save_unlocked(queue_path: str, entries: Iterable[Dict[str, Any]]) -> None:
    documents = list(entries)
    content = yaml.safe_dump_all(
        documents,
        explicit_start=True,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    file_io_utils.save_text(content, queue_path)


def append(queue_path: str, entry: Dict[str, Any], id_key: str) -> bool:
    """Append one entry atomically, returning False when its ID is already queued."""
    with _queue_lock(queue_path):
        entries = load(queue_path)
        entry_id = str(entry.get(id_key, ""))
        if entry_id and any(str(existing.get(id_key, "")) == entry_id for existing in entries):
            return False
        entries.append(entry)
        _save_unlocked(queue_path, entries)
        return True


def remove(queue_path: str, report_id: str, id_key: str) -> bool:
    """Remove all queue entries for a report ID atomically."""
    with _queue_lock(queue_path):
        entries = load(queue_path)
        remaining = [entry for entry in entries if str(entry.get(id_key, "")) != str(report_id)]
        if len(remaining) == len(entries):
            return False
        _save_unlocked(queue_path, remaining)
        return True
