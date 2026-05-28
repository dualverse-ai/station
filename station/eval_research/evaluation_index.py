"""SQLite read model for Research Center evaluation metadata.

Evaluation YAML files remain authoritative. This module stores only the compact
fields needed for list, dashboard statistics, queue scheduling, and top-score
lookups.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from station import constants
from station import file_io_utils
from station import index_paths


SCHEMA_VERSION = "3"
_INDEX_LOCK = threading.RLock()
_PROCESS_REBUILD_REQUESTED_SCOPES: set[str] = set()

TERMINAL_EVALUATION_STATUSES = {"completed", "success", "failed", "blocked", "partial"}
ACTIVE_EVALUATION_STATUSES = {"queued", "running"}
QUEUED_EVALUATION_STATUSES = {"queued"}
RUNNING_EVALUATION_STATUSES = {"running"}
ARTIFACT_MIGRATION_BLOB_FIELDS = ("submission_snapshot", "stdout", "stdout_visible", "stderr", "coder_report")
ARTIFACT_MIGRATION_KEYS = ("submission", "stdout", "stderr", "report")


def should_rebuild_from_process_args() -> bool:
    import sys

    if os.environ.get("STATION_REBUILD_DB", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return any(arg in {"--rebuild-db", "--rebuild_db"} for arg in sys.argv[1:])


def get_database_path(evaluations_dir: Optional[str] = None) -> str:
    base_path = _base_path_from_evaluations_dir(evaluations_dir)
    return index_paths.get_station_index_database_path(base_path)


def ensure_research_evaluation_index(
    evaluations_dir: str,
    *,
    rebuild: bool = False,
    log_status: bool = False,
) -> None:
    scope = _scope_key(evaluations_dir)
    with _INDEX_LOCK:
        db_path = get_database_path(evaluations_dir)
        if rebuild:
            if scope not in _PROCESS_REBUILD_REQUESTED_SCOPES:
                print(f"ResearchIndex: rebuild requested path={db_path!r} evaluations_dir={scope!r}")
                rebuild_research_evaluation_index(evaluations_dir)
                _PROCESS_REBUILD_REQUESTED_SCOPES.add(scope)
            elif log_status:
                print(f"ResearchIndex: ready path={db_path!r} evaluations_dir={scope!r}")
        elif _needs_rebuild(evaluations_dir):
            rebuild_research_evaluation_index(evaluations_dir)
        elif log_status:
            print(f"ResearchIndex: ready path={db_path!r} evaluations_dir={scope!r}")


def rebuild_research_evaluation_index(evaluations_dir: str) -> None:
    scope = _scope_key(evaluations_dir)
    with _INDEX_LOCK:
        db_path = get_database_path(evaluations_dir)
        file_io_utils.ensure_dir_exists(os.path.dirname(db_path))
        print(f"ResearchIndex: rebuilding path={db_path!r} evaluations_dir={scope!r}")
        conn = _connect(evaluations_dir)
        indexed_count = 0
        try:
            _configure_database(conn, setup_wal=True)
            conn.execute("BEGIN IMMEDIATE")
            _create_schema(conn)
            _clear_scope_unlocked(conn, scope)
            for eval_id, path in _iter_evaluation_files(evaluations_dir):
                data = file_io_utils.load_yaml(path)
                if not isinstance(data, dict):
                    continue
                _upsert_evaluation_unlocked(conn, scope, data, path)
                indexed_count += 1
            _recompute_top_submission_unlocked(conn, scope)
            _set_scope_schema_unlocked(conn, scope)
            conn.commit()
            print(
                "ResearchIndex: rebuild complete "
                f"path={db_path!r} evaluations_dir={scope!r} evaluations={indexed_count}"
            )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def upsert_evaluation(eval_data: Dict[str, Any], evaluations_dir: str) -> None:
    if not isinstance(eval_data, dict):
        return
    scope = _scope_key(evaluations_dir)
    eval_id = str(eval_data.get("id", "")).strip()
    if not eval_id:
        return
    file_path = _get_yaml_eval_path(eval_id, evaluations_dir)
    with _INDEX_LOCK:
        ensure_research_evaluation_index(evaluations_dir)
        conn = _connect(evaluations_dir)
        try:
            _configure_database(conn)
            with conn:
                _upsert_evaluation_unlocked(conn, scope, eval_data, file_path)
                _recompute_top_submission_unlocked(conn, scope)
                _set_scope_schema_unlocked(conn, scope)
        finally:
            conn.close()


def delete_evaluation(eval_id: str, evaluations_dir: str) -> None:
    scope = _scope_key(evaluations_dir)
    eval_id = str(eval_id)
    with _INDEX_LOCK:
        ensure_research_evaluation_index(evaluations_dir)
        conn = _connect(evaluations_dir)
        try:
            _configure_database(conn)
            with conn:
                conn.execute(
                    "DELETE FROM research_evaluation_tags WHERE evaluations_dir = ? AND eval_id = ?",
                    (scope, eval_id),
                )
                conn.execute(
                    "DELETE FROM research_evaluations WHERE evaluations_dir = ? AND eval_id = ?",
                    (scope, eval_id),
                )
                _recompute_top_submission_unlocked(conn, scope)
                _set_scope_schema_unlocked(conn, scope)
        finally:
            conn.close()


def get_display_info(eval_id: str, evaluations_dir: str) -> Optional[Dict[str, Any]]:
    ensure_research_evaluation_index(evaluations_dir)
    scope = _scope_key(evaluations_dir)
    conn = _connect(evaluations_dir)
    try:
        _configure_database(conn)
        row = conn.execute(
            "SELECT * FROM research_evaluations WHERE evaluations_dir = ? AND eval_id = ?",
            (scope, str(eval_id)),
        ).fetchone()
        return _row_to_display_info(row) if row is not None else None
    finally:
        conn.close()


def list_display_infos(evaluations_dir: str) -> List[Dict[str, Any]]:
    ensure_research_evaluation_index(evaluations_dir)
    scope = _scope_key(evaluations_dir)
    conn = _connect(evaluations_dir)
    try:
        _configure_database(conn)
        rows = conn.execute(
            """
            SELECT *
            FROM research_evaluations
            WHERE evaluations_dir = ?
            ORDER BY eval_id_num ASC, eval_id ASC
            """,
            (scope,),
        ).fetchall()
        return [_row_to_display_info(row) for row in rows]
    finally:
        conn.close()


def get_all_evaluation_ids(evaluations_dir: str) -> List[str]:
    ensure_research_evaluation_index(evaluations_dir)
    scope = _scope_key(evaluations_dir)
    conn = _connect(evaluations_dir)
    try:
        _configure_database(conn)
        rows = conn.execute(
            """
            SELECT eval_id
            FROM research_evaluations
            WHERE evaluations_dir = ?
            ORDER BY eval_id_num ASC, eval_id ASC
            """,
            (scope,),
        ).fetchall()
        return [str(row["eval_id"]) for row in rows]
    finally:
        conn.close()


def needs_artifact_migration(evaluations_dir: str) -> bool:
    ensure_research_evaluation_index(evaluations_dir)
    scope = _scope_key(evaluations_dir)
    conn = _connect(evaluations_dir)
    try:
        _configure_database(conn)
        value = conn.execute(
            """
            SELECT 1
            FROM research_evaluations
            WHERE evaluations_dir = ? AND needs_artifact_migration = 1
            LIMIT 1
            """,
            (scope,),
        ).fetchone()
        return value is not None
    finally:
        conn.close()


def get_artifact_migration_eval_ids(evaluations_dir: str) -> List[str]:
    return _list_ids_by_flag(evaluations_dir, "needs_artifact_migration = 1")


def search_abstracts(evaluations_dir: str, pattern: str, limit: int = 50) -> Tuple[int, List[Dict[str, Any]]]:
    ensure_research_evaluation_index(evaluations_dir)
    regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    scope = _scope_key(evaluations_dir)
    conn = _connect(evaluations_dir)
    try:
        _configure_database(conn)
        rows = conn.execute(
            """
            SELECT eval_id, title, abstract
            FROM research_evaluations
            WHERE evaluations_dir = ?
            ORDER BY eval_id_num DESC, eval_id DESC
            """,
            (scope,),
        ).fetchall()
        matches = [
            {
                constants.EVALUATION_ID_KEY: str(row["eval_id"]),
                constants.EVALUATION_TITLE_KEY: row["title"] or "(untitled)",
                constants.EVALUATION_ABSTRACT_KEY: row["abstract"] or "",
            }
            for row in rows
            if regex.search(row["abstract"] or "")
        ]
        shown_limit = max(0, int(limit))
        return len(matches), matches[:shown_limit]
    finally:
        conn.close()


def get_next_evaluation_id(evaluations_dir: str) -> str:
    ensure_research_evaluation_index(evaluations_dir)
    scope = _scope_key(evaluations_dir)
    conn = _connect(evaluations_dir)
    try:
        _configure_database(conn)
        value = conn.execute(
            "SELECT MAX(eval_id_num) FROM research_evaluations WHERE evaluations_dir = ?",
            (scope,),
        ).fetchone()[0]
        return str(int(value or 0) + 1)
    finally:
        conn.close()


def get_top_submission(evaluations_dir: str) -> Optional[Dict[str, Any]]:
    ensure_research_evaluation_index(evaluations_dir)
    scope = _scope_key(evaluations_dir)
    conn = _connect(evaluations_dir)
    try:
        _configure_database(conn)
        row = conn.execute(
            "SELECT top_submission_json FROM research_evaluation_scopes WHERE evaluations_dir = ?",
            (scope,),
        ).fetchone()
        if row is None:
            return None
        return _json_load_any(row["top_submission_json"])
    finally:
        conn.close()


def get_active_evaluations(evaluations_dir: str) -> List[Dict[str, Any]]:
    ensure_research_evaluation_index(evaluations_dir)
    scope = _scope_key(evaluations_dir)
    conn = _connect(evaluations_dir)
    try:
        _configure_database(conn)
        rows = conn.execute(
            """
            SELECT *
            FROM research_evaluations
            WHERE evaluations_dir = ? AND is_active = 1
            ORDER BY start_timestamp DESC, eval_id_num DESC, eval_id DESC
            """,
            (scope,),
        ).fetchall()
        return _prepare_active_evaluation_summaries([_row_to_active_summary(row) for row in rows])
    finally:
        conn.close()


def get_evaluation_statistics(evaluations_dir: str) -> Dict[str, Any]:
    active_evaluations = get_active_evaluations(evaluations_dir)
    queued_evaluations = [
        item for item in active_evaluations
        if str(item.get("top_level_status", "")).strip().lower() in QUEUED_EVALUATION_STATUSES
    ]

    def _is_running_job(item: Dict[str, Any]) -> bool:
        if str(item.get("top_level_status", "")).strip().lower() not in RUNNING_EVALUATION_STATUSES:
            return False
        if bool(item.get("coder_active")) or item.get("execution_source") == "direct":
            return True
        status = str(item.get("status", "")).strip().lower()
        latest_attempt_status = str(item.get("latest_attempt_status", "")).strip().lower()
        return status in {"attempt_queued", "attempt_running"} and latest_attempt_status in {"queued", "running"}

    running_evaluations = [
        item
        for item in active_evaluations
        if _is_running_job(item)
    ]
    return {
        "running_count": len(running_evaluations),
        "queued_count": len(queued_evaluations),
        "top_submission": get_top_submission(evaluations_dir),
        "running_evaluations": running_evaluations,
        "queued_evaluations": queued_evaluations,
    }


def get_active_eval_ids_for_author(evaluations_dir: str, author: str) -> List[str]:
    return _list_ids_by_flag(
        evaluations_dir,
        "is_active = 1 AND author_lower = ?",
        [str(author or "").lower()],
    )


def get_queued_instruction_eval_ids(evaluations_dir: str) -> List[str]:
    return _list_ids_by_flag(evaluations_dir, "is_queued_instruction = 1")


def get_running_instruction_eval_ids(evaluations_dir: str) -> List[str]:
    return _list_ids_by_flag(evaluations_dir, "is_running_instruction = 1")


def get_resuming_instruction_eval_ids(evaluations_dir: str) -> List[str]:
    return _list_ids_by_flag(evaluations_dir, "is_resuming_instruction = 1")


def get_unfinished_instruction_eval_ids(evaluations_dir: str) -> List[str]:
    return _list_ids_by_flag(evaluations_dir, "is_unfinished_instruction = 1")


def get_retryable_blocked_instruction_eval_ids(evaluations_dir: str) -> List[str]:
    return _list_ids_by_flag(evaluations_dir, "is_retryable_blocked = 1")


def get_pending_notification_eval_ids(evaluations_dir: str) -> List[str]:
    return _list_ids_by_flag(evaluations_dir, "has_pending_notification = 1")


def get_active_coder_count(evaluations_dir: str) -> int:
    ensure_research_evaluation_index(evaluations_dir)
    scope = _scope_key(evaluations_dir)
    conn = _connect(evaluations_dir)
    try:
        _configure_database(conn)
        return int(conn.execute(
            """
            SELECT COUNT(*)
            FROM research_evaluations
            WHERE evaluations_dir = ? AND active_coder = 1
            """,
            (scope,),
        ).fetchone()[0])
    finally:
        conn.close()


def get_recent_attempt_summaries(
    evaluations_dir: str,
    *,
    author: Optional[str] = None,
    lineage: Optional[str] = None,
    limit: int = 5,
    exclude_eval_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ensure_research_evaluation_index(evaluations_dir)
    scope = _scope_key(evaluations_dir)
    where = ["evaluations_dir = ?", "final_exists = 1"]
    params: List[Any] = [scope]
    if author is not None:
        where.append("author_lower = ?")
        params.append(str(author or "").lower())
    if lineage is not None:
        where.append("lineage_lower = ?")
        params.append(str(lineage or "").lower())
    if exclude_eval_id is not None:
        where.append("eval_id != ?")
        params.append(str(exclude_eval_id))
    conn = _connect(evaluations_dir)
    try:
        _configure_database(conn)
        rows = conn.execute(
            f"""
            SELECT eval_id, abstract, score_json, submitted_tick
            FROM research_evaluations
            WHERE {' AND '.join(where)}
            ORDER BY submitted_tick DESC, eval_id_num DESC, eval_id DESC
            LIMIT ?
            """,
            [*params, max(0, int(limit))],
        ).fetchall()
        return [
            {
                "id": str(row["eval_id"]),
                "abstract": row["abstract"] or "",
                "score": _json_load_any(row["score_json"], constants.RESEARCH_SCORE_NA),
                "submitted_tick": row["submitted_tick"] or 0,
            }
            for row in rows
        ]
    finally:
        conn.close()


def should_wait_at_tick(evaluations_dir: str, current_tick: int, max_allowed_ticks: int) -> bool:
    ensure_research_evaluation_index(evaluations_dir)
    scope = _scope_key(evaluations_dir)
    conn = _connect(evaluations_dir)
    try:
        _configure_database(conn)
        rows = conn.execute(
            """
            SELECT submitted_tick
            FROM research_evaluations
            WHERE evaluations_dir = ?
              AND is_active = 1
              AND top_level_status IN ('queued', 'running')
              AND submitted_tick IS NOT NULL
            """,
            (scope,),
        ).fetchall()
        for row in rows:
            elapsed_ticks = int(current_tick) - int(row["submitted_tick"]) + 1
            if elapsed_ticks >= int(max_allowed_ticks):
                return True
        return False
    finally:
        conn.close()


def _list_ids_by_flag(evaluations_dir: str, condition: str, params: Optional[Sequence[Any]] = None) -> List[str]:
    ensure_research_evaluation_index(evaluations_dir)
    scope = _scope_key(evaluations_dir)
    conn = _connect(evaluations_dir)
    try:
        _configure_database(conn)
        rows = conn.execute(
            f"""
            SELECT eval_id
            FROM research_evaluations
            WHERE evaluations_dir = ? AND {condition}
            ORDER BY eval_id_num ASC, eval_id ASC
            """,
            [scope, *(params or [])],
        ).fetchall()
        return [str(row["eval_id"]) for row in rows]
    finally:
        conn.close()


def _connect(evaluations_dir: Optional[str] = None) -> sqlite3.Connection:
    db_path = get_database_path(evaluations_dir)
    file_io_utils.ensure_dir_exists(os.path.dirname(db_path))
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _configure_database(conn: sqlite3.Connection, *, setup_wal: bool = False) -> None:
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    if setup_wal:
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.DatabaseError:
            pass


def _needs_rebuild(evaluations_dir: str) -> bool:
    db_path = get_database_path(evaluations_dir)
    if not os.path.exists(db_path):
        return True
    scope = _scope_key(evaluations_dir)
    conn = _connect(evaluations_dir)
    try:
        _configure_database(conn)
        row = conn.execute(
            "SELECT schema_version FROM research_evaluation_scopes WHERE evaluations_dir = ?",
            (scope,),
        ).fetchone()
        return row is None or str(row["schema_version"]) != SCHEMA_VERSION
    except sqlite3.DatabaseError as exc:
        if "no such table" in str(exc).lower():
            return True
        raise RuntimeError(f"ResearchIndex: database unavailable path={db_path!r}") from exc
    finally:
        conn.close()


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS research_evaluation_scopes (
            evaluations_dir TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            top_submission_json TEXT,
            rebuilt_at REAL
        );

        CREATE TABLE IF NOT EXISTS research_evaluations (
            evaluations_dir TEXT NOT NULL,
            eval_id TEXT NOT NULL,
            eval_id_num INTEGER,
            file_path TEXT NOT NULL,
            file_mtime_ns INTEGER NOT NULL,
            author TEXT,
            author_lower TEXT,
            lineage TEXT,
            lineage_lower TEXT,
            title TEXT,
            abstract TEXT,
            tags_json TEXT NOT NULL DEFAULT '[]',
            submitted_tick INTEGER,
            submitted_timestamp REAL,
            status TEXT,
            display_status TEXT,
            top_level_status TEXT,
            latest_attempt_status TEXT,
            score_json TEXT,
            score_numeric REAL,
            details_json TEXT,
            sort_key_json TEXT,
            final_exists INTEGER NOT NULL DEFAULT 0,
            success_score INTEGER NOT NULL DEFAULT 0,
            is_instruction INTEGER NOT NULL DEFAULT 0,
            is_coder_managed INTEGER NOT NULL DEFAULT 0,
            is_active INTEGER NOT NULL DEFAULT 0,
            is_queued_instruction INTEGER NOT NULL DEFAULT 0,
            is_running_instruction INTEGER NOT NULL DEFAULT 0,
            active_coder INTEGER NOT NULL DEFAULT 0,
            is_resuming_instruction INTEGER NOT NULL DEFAULT 0,
            is_unfinished_instruction INTEGER NOT NULL DEFAULT 0,
            is_retryable_blocked INTEGER NOT NULL DEFAULT 0,
            has_pending_notification INTEGER NOT NULL DEFAULT 0,
            has_required_artifacts INTEGER NOT NULL DEFAULT 0,
            has_inline_blobs INTEGER NOT NULL DEFAULT 0,
            needs_artifact_migration INTEGER NOT NULL DEFAULT 0,
            current_attempt INTEGER,
            start_timestamp REAL,
            coder_status TEXT,
            coder_active INTEGER NOT NULL DEFAULT 0,
            execution_source TEXT,
            system_baseline INTEGER NOT NULL DEFAULT 0,
            updated_at REAL,
            PRIMARY KEY (evaluations_dir, eval_id)
        );

        CREATE INDEX IF NOT EXISTS idx_research_eval_scope_order
            ON research_evaluations(evaluations_dir, eval_id_num DESC, eval_id DESC);
        CREATE INDEX IF NOT EXISTS idx_research_eval_active
            ON research_evaluations(evaluations_dir, is_active, top_level_status, start_timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_research_eval_author_active
            ON research_evaluations(evaluations_dir, author_lower, is_active);
        CREATE INDEX IF NOT EXISTS idx_research_eval_lineage_recent
            ON research_evaluations(evaluations_dir, lineage_lower, final_exists, submitted_tick DESC);
        CREATE INDEX IF NOT EXISTS idx_research_eval_pending_notification
            ON research_evaluations(evaluations_dir, has_pending_notification);
        CREATE INDEX IF NOT EXISTS idx_research_eval_artifact_migration
            ON research_evaluations(evaluations_dir, needs_artifact_migration, eval_id_num ASC, eval_id ASC);

        CREATE TABLE IF NOT EXISTS research_evaluation_tags (
            evaluations_dir TEXT NOT NULL,
            eval_id TEXT NOT NULL,
            tag_lower TEXT NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (evaluations_dir, eval_id, tag_lower)
        );
        CREATE INDEX IF NOT EXISTS idx_research_eval_tags_lookup
            ON research_evaluation_tags(evaluations_dir, tag_lower);
        """
    )
    _ensure_research_evaluation_columns(conn)


def _ensure_research_evaluation_columns(conn: sqlite3.Connection) -> None:
    existing = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(research_evaluations)").fetchall()
    }
    missing_columns = {
        "latest_attempt_status": "TEXT",
        "has_required_artifacts": "INTEGER NOT NULL DEFAULT 0",
        "has_inline_blobs": "INTEGER NOT NULL DEFAULT 0",
        "needs_artifact_migration": "INTEGER NOT NULL DEFAULT 0",
    }
    for column, definition in missing_columns.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE research_evaluations ADD COLUMN {column} {definition}")


def _clear_scope_unlocked(conn: sqlite3.Connection, scope: str) -> None:
    conn.execute("DELETE FROM research_evaluation_tags WHERE evaluations_dir = ?", (scope,))
    conn.execute("DELETE FROM research_evaluations WHERE evaluations_dir = ?", (scope,))
    conn.execute("DELETE FROM research_evaluation_scopes WHERE evaluations_dir = ?", (scope,))


def _set_scope_schema_unlocked(conn: sqlite3.Connection, scope: str) -> None:
    conn.execute(
        """
        INSERT INTO research_evaluation_scopes(evaluations_dir, schema_version, top_submission_json, rebuilt_at)
        VALUES(?, ?, COALESCE((SELECT top_submission_json FROM research_evaluation_scopes WHERE evaluations_dir = ?), NULL), ?)
        ON CONFLICT(evaluations_dir) DO UPDATE SET
            schema_version = excluded.schema_version,
            rebuilt_at = excluded.rebuilt_at
        """,
        (scope, SCHEMA_VERSION, scope, time.time()),
    )


def _upsert_evaluation_unlocked(
    conn: sqlite3.Connection,
    scope: str,
    eval_data: Dict[str, Any],
    file_path: str,
) -> None:
    eval_id = str(eval_data.get("id", "")).strip()
    if not eval_id:
        return

    display_info = _build_display_info(eval_data)
    active_summary = _build_active_evaluation_summary(eval_data)
    final = eval_data.get("final") or {}
    status = _normalize_evaluation_status(eval_data.get("status")) or "queued"
    score = display_info.get(constants.EVALUATION_SCORE_KEY, constants.RESEARCH_SCORE_NA)
    details = display_info.get(constants.EVALUATION_DETAILS_KEY, "")
    sort_key = display_info.get("sort_key")
    tags = _clean_string_list(display_info.get(constants.EVALUATION_TAGS_KEY))
    notification = eval_data.get("notification") or {}
    coder = eval_data.get("coder", {}) or {}
    coder_substate = str(coder.get("status") or "").strip().lower()
    is_instruction_eval = "instruction" in eval_data
    is_coder_managed = (
        is_instruction_eval
        and not final
        and not eval_data.get("submission_mode") == "direct"
        and not eval_data.get("system_baseline")
    )
    score_success = (
        bool(final)
        and status in TERMINAL_EVALUATION_STATUSES
        and score not in (constants.RESEARCH_SCORE_PENDING, constants.RESEARCH_SCORE_NA, None)
    )
    has_required_artifacts = _has_required_artifacts(eval_data)
    has_inline_blobs = _has_inline_blobs(eval_data)
    needs_artifact_migration = has_inline_blobs or not has_required_artifacts

    if active_summary:
        top_level_status = active_summary.get("top_level_status")
        display_status = active_summary.get("status")
        latest_attempt_status = active_summary.get("latest_attempt_status")
        start_timestamp = active_summary.get("start_timestamp")
        execution_source = active_summary.get("execution_source")
    else:
        top_level_status = status
        display_status = status
        latest_attempt_status = (_normalize_evaluation_status((_get_final_attempt(eval_data) or {}).get("status")) or "")
        start_timestamp = eval_data.get("submitted_timestamp") or 0
        execution_source = "direct" if bool(eval_data.get("submission_mode") == "direct" or eval_data.get("system_baseline")) else "coder"

    conn.execute(
        """
        INSERT INTO research_evaluations(
            evaluations_dir, eval_id, eval_id_num, file_path, file_mtime_ns,
            author, author_lower, lineage, lineage_lower, title, abstract, tags_json,
            submitted_tick, submitted_timestamp, status, display_status, top_level_status,
            latest_attempt_status, score_json, score_numeric, details_json, sort_key_json, final_exists, success_score,
            is_instruction, is_coder_managed, is_active, is_queued_instruction, is_running_instruction,
            active_coder, is_resuming_instruction, is_unfinished_instruction, is_retryable_blocked,
            has_pending_notification, has_required_artifacts, has_inline_blobs, needs_artifact_migration,
            current_attempt, start_timestamp, coder_status, coder_active, execution_source,
            system_baseline, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(evaluations_dir, eval_id) DO UPDATE SET
            eval_id_num = excluded.eval_id_num,
            file_path = excluded.file_path,
            file_mtime_ns = excluded.file_mtime_ns,
            author = excluded.author,
            author_lower = excluded.author_lower,
            lineage = excluded.lineage,
            lineage_lower = excluded.lineage_lower,
            title = excluded.title,
            abstract = excluded.abstract,
            tags_json = excluded.tags_json,
            submitted_tick = excluded.submitted_tick,
            submitted_timestamp = excluded.submitted_timestamp,
            status = excluded.status,
            display_status = excluded.display_status,
            top_level_status = excluded.top_level_status,
            latest_attempt_status = excluded.latest_attempt_status,
            score_json = excluded.score_json,
            score_numeric = excluded.score_numeric,
            details_json = excluded.details_json,
            sort_key_json = excluded.sort_key_json,
            final_exists = excluded.final_exists,
            success_score = excluded.success_score,
            is_instruction = excluded.is_instruction,
            is_coder_managed = excluded.is_coder_managed,
            is_active = excluded.is_active,
            is_queued_instruction = excluded.is_queued_instruction,
            is_running_instruction = excluded.is_running_instruction,
            active_coder = excluded.active_coder,
            is_resuming_instruction = excluded.is_resuming_instruction,
            is_unfinished_instruction = excluded.is_unfinished_instruction,
            is_retryable_blocked = excluded.is_retryable_blocked,
            has_pending_notification = excluded.has_pending_notification,
            has_required_artifacts = excluded.has_required_artifacts,
            has_inline_blobs = excluded.has_inline_blobs,
            needs_artifact_migration = excluded.needs_artifact_migration,
            current_attempt = excluded.current_attempt,
            start_timestamp = excluded.start_timestamp,
            coder_status = excluded.coder_status,
            coder_active = excluded.coder_active,
            execution_source = excluded.execution_source,
            system_baseline = excluded.system_baseline,
            updated_at = excluded.updated_at
        """,
        (
            scope,
            eval_id,
            _eval_id_num(eval_id),
            os.path.abspath(file_path),
            _file_mtime_ns(file_path),
            _as_optional_str(display_info.get(constants.EVALUATION_AUTHOR_KEY)),
            str(display_info.get(constants.EVALUATION_AUTHOR_KEY, "") or "").lower(),
            _as_optional_str(eval_data.get("lineage")),
            str(eval_data.get("lineage", "") or "").lower(),
            _as_optional_str(display_info.get(constants.EVALUATION_TITLE_KEY)),
            _as_optional_str(display_info.get(constants.EVALUATION_ABSTRACT_KEY)),
            _json_dumps(tags),
            _as_optional_int(display_info.get(constants.EVALUATION_SUBMITTED_TICK_KEY)),
            _as_optional_float(eval_data.get("submitted_timestamp")),
            status,
            str(display_status or ""),
            str(top_level_status or status),
            str(latest_attempt_status or ""),
            _json_dumps(score),
            _as_optional_float(score),
            _json_dumps(details),
            _json_dumps(sort_key) if sort_key is not None else None,
            1 if bool(final) else 0,
            1 if score_success else 0,
            1 if is_instruction_eval else 0,
            1 if is_coder_managed else 0,
            1 if active_summary is not None else 0,
            1 if status == "queued" and is_coder_managed else 0,
            1 if status == "running" and is_coder_managed else 0,
            1 if status == "running" and is_coder_managed and bool(coder.get("active")) else 0,
            1 if status == "running" and coder_substate in {"pending_resume", "resuming"} and is_coder_managed else 0,
            1 if is_coder_managed and status in {"queued", "running", "blocked"} else 0,
            1 if status == "blocked" and is_coder_managed and not bool(coder.get("active")) else 0,
            1 if bool(final) and status in TERMINAL_EVALUATION_STATUSES and not notification.get("sent") else 0,
            1 if has_required_artifacts else 0,
            1 if has_inline_blobs else 0,
            1 if needs_artifact_migration else 0,
            _as_optional_int(eval_data.get("current_attempt")),
            _as_optional_float(start_timestamp),
            coder_substate,
            1 if bool(coder.get("active")) else 0,
            str(execution_source or "coder"),
            1 if bool(eval_data.get("system_baseline")) else 0,
            time.time(),
        ),
    )

    conn.execute(
        "DELETE FROM research_evaluation_tags WHERE evaluations_dir = ? AND eval_id = ?",
        (scope, eval_id),
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO research_evaluation_tags(evaluations_dir, eval_id, tag_lower, tag)
        VALUES(?, ?, ?, ?)
        """,
        [(scope, eval_id, tag.lower(), tag) for tag in tags],
    )


def _recompute_top_submission_unlocked(conn: sqlite3.Connection, scope: str) -> None:
    rows = conn.execute(
        """
        SELECT eval_id, title, author, submitted_tick, tags_json, abstract, score_json, sort_key_json
        FROM research_evaluations
        WHERE evaluations_dir = ? AND success_score = 1
        """,
        (scope,),
    ).fetchall()
    top_submission = None
    for row in rows:
        candidate = _candidate_from_row(row)
        if candidate and _should_replace_top_submission(candidate, top_submission):
            top_submission = candidate
    conn.execute(
        """
        INSERT INTO research_evaluation_scopes(evaluations_dir, schema_version, top_submission_json, rebuilt_at)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(evaluations_dir) DO UPDATE SET
            schema_version = excluded.schema_version,
            top_submission_json = excluded.top_submission_json
        """,
        (scope, SCHEMA_VERSION, _json_dumps(top_submission) if top_submission else None, time.time()),
    )


def _candidate_from_row(row: sqlite3.Row) -> Optional[Dict[str, Any]]:
    score = _json_load_any(row["score_json"])
    sort_key = _json_load_any(row["sort_key_json"]) if row["sort_key_json"] else None
    normalized_sort_key = _normalize_sort_key(sort_key, score)
    if normalized_sort_key is None:
        return None
    return {
        "evaluation_id": str(row["eval_id"]),
        "title": row["title"],
        "score": score,
        "agent_name": row["author"],
        "submitted_tick": row["submitted_tick"],
        "tags": _json_load_list(row["tags_json"]),
        "abstract": row["abstract"] or "",
        "sort_key": list(normalized_sort_key),
    }


def _row_to_display_info(row: sqlite3.Row) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        constants.EVALUATION_ID_KEY: str(row["eval_id"]),
        constants.EVALUATION_AUTHOR_KEY: row["author"] or "Unknown",
        constants.EVALUATION_TITLE_KEY: row["title"] or "Untitled",
        constants.EVALUATION_TAGS_KEY: _json_load_list(row["tags_json"]),
        constants.EVALUATION_ABSTRACT_KEY: row["abstract"] or "",
        constants.EVALUATION_SCORE_KEY: _json_load_any(row["score_json"], constants.RESEARCH_SCORE_NA),
        constants.EVALUATION_SUBMITTED_TICK_KEY: row["submitted_tick"] or 0,
        constants.EVALUATION_DETAILS_KEY: _json_load_any(row["details_json"], ""),
        constants.EVALUATION_STATUS_KEY: row["status"] or "queued",
    }
    if row["sort_key_json"]:
        result["sort_key"] = _json_load_any(row["sort_key_json"])
    return result


def _row_to_active_summary(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "evaluation_id": str(row["eval_id"]),
        "agent_name": row["author"],
        "title": row["title"] or "",
        "start_tick": row["submitted_tick"] or 0,
        "start_timestamp": row["start_timestamp"] or 0,
        "status": row["display_status"] or row["status"] or "",
        "top_level_status": row["top_level_status"] or row["status"] or "",
        "latest_attempt_status": row["latest_attempt_status"] or "",
        "coder_active": bool(row["coder_active"]),
        "execution_source": row["execution_source"] or "coder",
        "system_baseline": bool(row["system_baseline"]),
        "submitted_tick": row["submitted_tick"] or 0,
        "author_lower": row["author_lower"] or "",
        "lineage_lower": row["lineage_lower"] or "",
    }


def _prepare_active_evaluation_summaries(indexed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now = time.time()
    running: List[Dict[str, Any]] = []
    for item in indexed:
        start_timestamp = item.get("start_timestamp") or 0
        item["elapsed_seconds"] = int(now - start_timestamp) if start_timestamp else 0
        item.pop("author_lower", None)
        item.pop("lineage_lower", None)
        running.append(item)
    running.sort(key=lambda item: item.get("start_timestamp", 0), reverse=True)
    return running


def _build_display_info(eval_data: Dict[str, Any]) -> Dict[str, Any]:
    final = eval_data.get("final") or {}
    status = _normalize_evaluation_status(eval_data.get("status")) or "queued"

    if final and status in TERMINAL_EVALUATION_STATUSES:
        score = final.get("primary_score", constants.RESEARCH_SCORE_NA)
        details = final.get(constants.EVALUATION_DETAILS_KEY, "")
        sort_key = final.get("sort_key")
    else:
        score = constants.RESEARCH_SCORE_PENDING
        details = ""
        sort_key = None

    result = {
        constants.EVALUATION_ID_KEY: str(eval_data.get("id")),
        constants.EVALUATION_AUTHOR_KEY: eval_data.get("author", "Unknown"),
        constants.EVALUATION_TITLE_KEY: eval_data.get("title", "Untitled"),
        constants.EVALUATION_TAGS_KEY: eval_data.get("tags", []),
        constants.EVALUATION_ABSTRACT_KEY: eval_data.get("abstract", ""),
        constants.EVALUATION_SCORE_KEY: score,
        constants.EVALUATION_SUBMITTED_TICK_KEY: eval_data.get("submitted_tick", 0),
        constants.EVALUATION_DETAILS_KEY: details,
        constants.EVALUATION_STATUS_KEY: status,
    }
    if sort_key is not None:
        result["sort_key"] = sort_key
    return result


def _build_active_evaluation_summary(eval_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    top_level_status = _normalize_evaluation_status(eval_data.get("status")) or "queued"
    if top_level_status not in ACTIVE_EVALUATION_STATUSES:
        return None

    latest_attempt = _get_final_attempt(eval_data) or {}
    coder = eval_data.get("coder", {}) or {}
    coder_started = coder.get("started_timestamp")
    latest_attempt_started = latest_attempt.get("started_timestamp")
    start_timestamp = coder_started or latest_attempt_started or eval_data.get("submitted_timestamp") or 0
    substate = str(coder.get("status") or "").strip().lower()
    latest_attempt_status = _normalize_evaluation_status(latest_attempt.get("status")) or ""
    is_direct_evaluation = bool(eval_data.get("submission_mode") == "direct" or eval_data.get("system_baseline"))
    display_status = top_level_status
    if top_level_status == "running" and substate == "attempt_running":
        if latest_attempt_status == "queued":
            display_status = "attempt_queued"
        elif latest_attempt_status == "running":
            display_status = "attempt_running"
        else:
            display_status = substate
    elif top_level_status == "running" and bool(coder.get("active")) and substate:
        display_status = substate
    elif top_level_status == "running" and latest_attempt_status:
        if latest_attempt_status == "queued":
            display_status = "attempt_queued"
        elif latest_attempt_status == "running":
            display_status = "attempt_running"
        else:
            display_status = latest_attempt_status
    elif top_level_status == "running" and substate and substate != "queued":
        display_status = substate

    return {
        "evaluation_id": str(eval_data.get("id")),
        "agent_name": eval_data.get("author"),
        "title": eval_data.get("title", ""),
        "start_tick": eval_data.get("submitted_tick", 0),
        "start_timestamp": start_timestamp,
        "status": display_status,
        "top_level_status": top_level_status,
        "latest_attempt_status": latest_attempt_status,
        "coder_active": bool(coder.get("active")),
        "execution_source": "direct" if is_direct_evaluation else "coder",
        "system_baseline": bool(eval_data.get("system_baseline")),
        "submitted_tick": eval_data.get("submitted_tick", 0),
        "author_lower": str(eval_data.get("author", "")).lower(),
        "lineage_lower": str(eval_data.get("lineage", "")).lower(),
    }


def _has_required_artifacts(eval_data: Dict[str, Any]) -> bool:
    artifacts = eval_data.get("artifacts")
    return isinstance(artifacts, dict) and all(
        isinstance(artifacts.get(key), str) and artifacts.get(key).strip()
        for key in ARTIFACT_MIGRATION_KEYS
    )


def _has_inline_blobs(eval_data: Dict[str, Any]) -> bool:
    attempts = eval_data.get("attempts") or []
    if isinstance(attempts, list):
        for attempt in attempts:
            if isinstance(attempt, dict) and any(key in attempt for key in ARTIFACT_MIGRATION_BLOB_FIELDS):
                return True

    final = eval_data.get("final")
    return isinstance(final, dict) and any(key in final for key in ARTIFACT_MIGRATION_BLOB_FIELDS)


def _normalize_evaluation_status(status: Optional[str]) -> Optional[str]:
    if status is None:
        return None
    normalized = str(status).strip().lower()
    if not normalized:
        return None
    if normalized == "success":
        return "completed"
    if normalized in {"running", "coder_running", "attempt_running", "waiting_for_attempt", "waiting_for_report"}:
        return "running"
    if normalized == "queued":
        return "queued"
    if normalized in {"completed", "failed", "blocked", "partial"}:
        return normalized
    return normalized


def _get_final_attempt(eval_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    attempts = eval_data.get("attempts") or []
    if not attempts:
        return None
    return attempts[-1]


def _normalize_sort_key(sort_key: Any, score: Any) -> Optional[Tuple]:
    if sort_key is not None:
        raw_items = tuple(sort_key) if isinstance(sort_key, (list, tuple)) else (sort_key,)
        normalized_items = []
        for item in raw_items:
            normalized_item = _normalize_sort_key_component(item)
            if normalized_item is None:
                normalized_items = []
                break
            normalized_items.append(normalized_item)
        if normalized_items:
            return tuple(normalized_items)
    try:
        return (float(score),)
    except (TypeError, ValueError):
        return None


def _normalize_sort_key_component(value: Any) -> Optional[Any]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.startswith("*"):
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        pass
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _is_numeric_singleton_tuple(key: Any) -> bool:
    return isinstance(key, tuple) and len(key) == 1 and isinstance(key[0], (int, float))


def _coerce_tick(tick: Any) -> Optional[int]:
    try:
        return int(tick)
    except (TypeError, ValueError):
        return None


def _coerce_eval_id(eval_id: Any) -> Tuple[int, Any]:
    try:
        return (0, int(str(eval_id)))
    except (TypeError, ValueError):
        return (1, str(eval_id))


def _should_replace_top_submission(candidate: Dict[str, Any], current_top: Optional[Dict[str, Any]]) -> bool:
    if current_top is None:
        return True
    candidate_sort_key = _normalize_sort_key(candidate.get("sort_key"), candidate.get("score"))
    if candidate_sort_key is None:
        return False
    top_sort_key = _normalize_sort_key(current_top.get("sort_key"), current_top.get("score"))
    if top_sort_key is None:
        return True
    eps = getattr(constants, "BREAKTHROUGH_EPS", 1e-8)
    if _is_numeric_singleton_tuple(candidate_sort_key) and _is_numeric_singleton_tuple(top_sort_key):
        if candidate_sort_key[0] > top_sort_key[0] + eps:
            return True
        if candidate_sort_key[0] + eps < top_sort_key[0]:
            return False
    else:
        if candidate_sort_key > top_sort_key:
            return True
        if candidate_sort_key < top_sort_key:
            return False
    candidate_tick = _coerce_tick(candidate.get("submitted_tick"))
    top_tick = _coerce_tick(current_top.get("submitted_tick"))
    if candidate_tick is not None and top_tick is not None and candidate_tick != top_tick:
        return candidate_tick < top_tick
    return _coerce_eval_id(candidate.get("evaluation_id")) < _coerce_eval_id(current_top.get("evaluation_id"))


def _iter_evaluation_files(evaluations_dir: str) -> Iterable[Tuple[str, str]]:
    if not os.path.isdir(evaluations_dir):
        return
    pattern = re.compile(rf"^(.+){re.escape(constants.RESEARCH_EVALUATION_FILE_EXTENSION)}$")
    for filename in file_io_utils.list_files(evaluations_dir, constants.RESEARCH_EVALUATION_FILE_EXTENSION):
        if filename.startswith("."):
            continue
        match = pattern.match(filename)
        if not match:
            continue
        yield match.group(1), os.path.join(evaluations_dir, filename)


def _get_yaml_eval_path(eval_id: str, evaluations_dir: str) -> str:
    return os.path.join(evaluations_dir, f"{eval_id}{constants.RESEARCH_EVALUATION_FILE_EXTENSION}")


def _base_path_from_evaluations_dir(evaluations_dir: Optional[str]) -> str:
    if not evaluations_dir:
        return constants.BASE_STATION_DATA_PATH
    path = Path(evaluations_dir).resolve()
    if (
        path.name == constants.RESEARCH_EVALUATIONS_SUBDIR_NAME
        and path.parent.name == constants.RESEARCH_CENTER_SUBDIR_NAME
        and path.parent.parent.name == constants.ROOMS_DIR_NAME
    ):
        return str(path.parent.parent.parent)
    return str(path.parent)


def _scope_key(evaluations_dir: str) -> str:
    return os.path.abspath(evaluations_dir)


def _eval_id_num(eval_id: str) -> Optional[int]:
    try:
        return int(str(eval_id))
    except (TypeError, ValueError):
        return None


def _file_mtime_ns(path: str) -> int:
    try:
        return os.stat(path).st_mtime_ns
    except (FileNotFoundError, OSError):
        return 0


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _as_optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_string_list(raw_value: Any) -> List[str]:
    if isinstance(raw_value, str):
        values = [item.strip() for item in raw_value.split(",")]
    elif isinstance(raw_value, list):
        values = [str(item).strip() for item in raw_value]
    else:
        values = []
    seen = set()
    result: List[str] = []
    for value in values:
        if not value or value.lower() in seen:
            continue
        seen.add(value.lower())
        result.append(value)
    return result


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_load_any(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _json_load_list(value: Any) -> List[str]:
    parsed = _json_load_any(value, [])
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item)]
