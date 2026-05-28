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
Thread-safe helpers for Theory Room storage and queue management.
"""

import json
import os
import threading
from typing import Any, Dict, List, Optional

from station import constants
from station import file_io_utils
from .lean_runner import strip_imports


LEMMA_FILE = "lemmas.yamll"  # legacy flat file (kept for back-compat if present)
THEORY_FILE = "theories.yamll"  # legacy flat file (kept for back-compat if present)
LEMMA_DIR = "lemmas"
THEORY_DIR = "theories"
INDEX_FILE = "index.json"
ENV_FILE = "env.lean"


class TheoryStorageManager:
    """Manages Theory Room storage with simple locking for cross-thread access."""

    def __init__(self, base_path: str):
        self.base_path = base_path
        file_io_utils.ensure_dir_exists(self.base_path)
        self._index_cache: Dict[str, Dict[str, Any]] = {"lemma": {}, "theory": {}}
        self._env_cache: List[str] = []
        self._index_lock = threading.Lock()
        self._env_lock = threading.Lock()
        self._pending_lock = threading.RLock()
        self._ensure_files()

    # ---------- initialization ----------
    def _ensure_files(self) -> None:
        # Legacy yamll files (if present) are left untouched; new items go to per-item files.
        for fname in [LEMMA_FILE, THEORY_FILE]:
            fpath = os.path.join(self.base_path, fname)
            if not os.path.exists(fpath):
                file_io_utils.ensure_dir_exists(os.path.dirname(fpath))
                with open(fpath, "w", encoding="utf-8"):
                    pass

        # Ensure per-item directories and index
        file_io_utils.ensure_dir_exists(os.path.join(self.base_path, LEMMA_DIR))
        file_io_utils.ensure_dir_exists(os.path.join(self.base_path, THEORY_DIR))
        index_path = os.path.join(self.base_path, INDEX_FILE)
        if not os.path.exists(index_path):
            self._save_index({"lemma": {}, "theory": {}})
        else:
            self._load_index()

    # ---------- index helpers ----------
    def _load_index(self) -> None:
        index_path = os.path.join(self.base_path, INDEX_FILE)
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._index_cache = {"lemma": data.get("lemma", {}), "theory": data.get("theory", {})}
        except Exception:
            self._index_cache = {"lemma": {}, "theory": {}}

    def _save_index(self, data: Dict[str, Dict[str, Any]]) -> None:
        index_path = os.path.join(self.base_path, INDEX_FILE)
        file_io_utils.ensure_dir_exists(os.path.dirname(index_path))
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_items(self, kind: str) -> List[Dict[str, Any]]:
        # Prefer index cache
        with self._index_lock:
            entries = self._index_cache.get(kind, {})
            if entries:
                return [v for _, v in sorted(entries.items(), key=lambda kv: int(kv[0]))]

        # Fallback: load from legacy yamll if index empty
        items: List[Dict[str, Any]] = []
        fname = LEMMA_FILE if kind == "lemma" else THEORY_FILE
        fpath = os.path.join(self.base_path, fname)
        if os.path.exists(fpath):
            try:
                items = file_io_utils.load_yaml_lines(fpath) or []
            except Exception:
                items = []
            # Populate index from legacy for future runs
            with self._index_lock:
                for it in items:
                    if "id" in it:
                        self._index_cache.setdefault(kind, {})[str(it["id"])] = it
                self._save_index(self._index_cache)
        return items

    def next_id(self, kind: str) -> int:
        with self._index_lock:
            return self._compute_next_id_locked(kind)

    def _compute_next_id_locked(self, kind: str) -> int:
        entries = self._index_cache.get(kind, {})
        ids = [int(k) for k in entries.keys()] if entries else []
        next_id = (max(ids + [0]) + 1)
        return next_id

    def append_item(self, kind: str, item: Dict[str, Any]) -> None:
        # Persist per-item file
        dir_name = LEMMA_DIR if kind == "lemma" else THEORY_DIR
        file_io_utils.ensure_dir_exists(os.path.join(self.base_path, dir_name))
        file_path = os.path.join(self.base_path, dir_name, f"{kind}_{item['id']}.yaml")
        file_io_utils.save_yaml(item, file_path)

        # Update index cache and save
        with self._index_lock:
            self._index_cache.setdefault(kind, {})[str(item["id"])] = item
            self._save_index(self._index_cache)

    def add_verified_item(self, kind: str, item_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Atomically allocate an ID, persist the item, and update the index."""
        with self._index_lock:
            new_id = self._compute_next_id_locked(kind)
            item = {"id": new_id, **item_fields}
            dir_name = LEMMA_DIR if kind == "lemma" else THEORY_DIR
            file_io_utils.ensure_dir_exists(os.path.join(self.base_path, dir_name))
            file_path = os.path.join(self.base_path, dir_name, f"{kind}_{item['id']}.yaml")
            file_io_utils.save_yaml(item, file_path)
            self._index_cache.setdefault(kind, {})[str(item["id"])] = item
            self._save_index(self._index_cache)
            return item

    # ---------- env helpers ----------
    def get_env_code(self) -> str:
        with self._env_lock:
            if self._env_cache:
                return "".join(self._env_cache)

        fpath = os.path.join(self.base_path, ENV_FILE)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    raw = f.read()
                data = strip_imports(raw)
                with self._env_lock:
                    self._env_cache = [data]
                return data
            except Exception:
                return ""
        return ""

    def append_env_code(self, code: str) -> None:
        fpath = os.path.join(self.base_path, ENV_FILE)
        file_io_utils.ensure_dir_exists(os.path.dirname(fpath))
        cleaned = strip_imports(code)
        with self._env_lock:
            with open(fpath, "a", encoding="utf-8") as f:
                f.write(cleaned)
            self._env_cache.append(cleaned)

    # ---------- pending queue helpers ----------
    @property
    def pending_path(self) -> str:
        return os.path.join(self.base_path, constants.PENDING_THEORY_EVALUATIONS_FILENAME)

    def load_pending(self) -> List[Dict[str, Any]]:
        with self._pending_lock:
            return self._load_pending_unlocked()

    def _load_pending_unlocked(self) -> List[Dict[str, Any]]:
        return file_io_utils.load_yaml_lines(self.pending_path) or []

    def append_pending(self, entry: Dict[str, Any]) -> None:
        with self._pending_lock:
            file_io_utils.append_yaml_line(entry, self.pending_path)

    def remove_pending(self, queue_id: str) -> None:
        with self._pending_lock:
            entries = self._load_pending_unlocked()
            remaining = [e for e in entries if str(e.get("queue_id")) != str(queue_id)]
            self._rewrite_pending(remaining)

    def rewrite_pending(self, entries: List[Dict[str, Any]]) -> None:
        with self._pending_lock:
            self._rewrite_pending(entries)

    def _rewrite_pending(self, entries: List[Dict[str, Any]]) -> None:
        file_io_utils.ensure_dir_exists(os.path.dirname(self.pending_path))
        # Truncate then append entries to maintain yamll structure
        with open(self.pending_path, "w", encoding="utf-8"):
            pass
        for entry in entries:
            file_io_utils.append_yaml_line(entry, self.pending_path)
