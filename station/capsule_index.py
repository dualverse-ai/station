"""SQLite read model for capsule metadata.

YAML capsule files remain authoritative. This module maintains a rebuildable
SQLite index for list, search, and room-render paths that only need capsule
metadata and active message IDs.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from station import constants
from station import file_io_utils
from station import index_paths


SCHEMA_VERSION = "2"
_INDEX_LOCK = threading.RLock()
_SORT_INDEX_READY_PATHS: set[str] = set()

_CAPSULE_SORT_EXPRESSIONS = {
    "numeric_id": "m.numeric_id",
    "title": "m.title COLLATE NOCASE",
    "author": "m.author_name COLLATE NOCASE",
    "created_at_tick": "m.created_at_tick",
    "last_updated_at_tick": "m.last_updated_at_tick",
    "word_count": "m.word_count_total",
    "message_count": "m.total_message_count",
    "question_status": "m.question_status COLLATE NOCASE",
    "question_net_upvote": "m.question_net_upvote",
}


def get_database_path() -> str:
    return index_paths.get_station_index_database_path(constants.BASE_STATION_DATA_PATH)


def should_rebuild_from_process_args() -> bool:
    import sys

    if os.environ.get("STATION_REBUILD_DB", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return any(arg in {"--rebuild-db", "--rebuild_db"} for arg in sys.argv[1:])


def ensure_capsule_index(*, rebuild: bool = False, log_status: bool = False) -> None:
    with _INDEX_LOCK:
        db_path = get_database_path()
        if rebuild:
            print(f"CapsuleIndex: rebuild requested path={db_path!r}")
            rebuild_capsule_index()
        elif _needs_rebuild():
            rebuild_capsule_index()
        elif log_status:
            print(f"CapsuleIndex: ready path={db_path!r}")
        if db_path not in _SORT_INDEX_READY_PATHS:
            with index_paths.get_station_index_write_lock():
                _ensure_sort_indexes_unlocked(db_path)
            _SORT_INDEX_READY_PATHS.add(db_path)


def rebuild_capsule_index() -> None:
    with _INDEX_LOCK:
        with index_paths.get_station_index_write_lock():
            db_path = get_database_path()
            file_io_utils.ensure_dir_exists(os.path.dirname(db_path))
            print(f"CapsuleIndex: rebuilding path={db_path!r}")
            conn = _connect()
            indexed_count = 0
            try:
                _configure_database(conn, setup_wal=True)
                conn.execute("BEGIN IMMEDIATE")
                _drop_schema(conn)
                _create_schema(conn)
                for capsule_type, lineage_name, numeric_id, path in _iter_all_capsule_files():
                    _upsert_file_unlocked(conn, capsule_type, lineage_name, numeric_id, path)
                    indexed_count += 1
                _set_schema_version(conn)
                conn.commit()
                print(f"CapsuleIndex: rebuild complete path={db_path!r} capsules={indexed_count}")
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()


def upsert_capsule(capsule_data: Dict[str, Any], file_path: str, lineage_name: Optional[str] = None) -> None:
    if not isinstance(capsule_data, dict):
        return
    ensure_capsule_index()
    with index_paths.get_station_index_write_lock():
        conn = _connect()
        try:
            _configure_database(conn)
            with conn:
                _upsert_capsule_unlocked(conn, capsule_data, file_path, lineage_name=lineage_name)
        finally:
            conn.close()


def list_capsules(
    capsule_type: str,
    lineage_name: Optional[str] = None,
    *,
    agent_read_status: Optional[Dict[str, bool]] = None,
    tag_filter: Optional[str] = None,
    visible_agent_name: Optional[str] = None,
    exclude_capsule_ids: Optional[Sequence[str]] = None,
    include_deleted: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
    sort_by: str = "numeric_id",
    sort_direction: str = "desc",
) -> Tuple[List[Dict[str, Any]], int]:
    ensure_capsule_index()
    lineage_key = _lineage_key(capsule_type, lineage_name)
    where, params = _build_where_clause(
        capsule_type,
        lineage_key,
        tag_filter=tag_filter,
        visible_agent_name=visible_agent_name,
        exclude_capsule_ids=exclude_capsule_ids,
        include_deleted=include_deleted,
    )
    conn = _connect()
    try:
        _configure_database(conn)
        total = int(conn.execute(f"SELECT COUNT(*) FROM capsule_metadata m WHERE {where}", params).fetchone()[0])
        sort_expression = _CAPSULE_SORT_EXPRESSIONS.get(str(sort_by), _CAPSULE_SORT_EXPRESSIONS["numeric_id"])
        direction = "ASC" if str(sort_direction).lower() == "asc" else "DESC"
        query = (
            f"SELECT m.* FROM capsule_metadata m WHERE {where} "
            f"ORDER BY {sort_expression} {direction}, m.numeric_id DESC"
        )
        query_params = list(params)
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            query_params.extend([max(0, int(limit)), max(0, int(offset))])
        rows = conn.execute(query, query_params).fetchall()
        return [_row_to_metadata(row, agent_read_status) for row in rows], total
    finally:
        conn.close()


def get_capsule_metadata(
    numeric_id: int,
    capsule_type: str,
    lineage_name: Optional[str] = None,
    *,
    agent_read_status: Optional[Dict[str, bool]] = None,
    include_deleted: bool = False,
) -> Optional[Dict[str, Any]]:
    ensure_capsule_index()
    lineage_key = _lineage_key(capsule_type, lineage_name)
    conn = _connect()
    try:
        _configure_database(conn)
        row = conn.execute(
            """
            SELECT *
            FROM capsule_metadata
            WHERE capsule_type = ? AND lineage_key = ? AND numeric_id = ?
            """,
            (capsule_type, lineage_key, int(numeric_id)),
        ).fetchone()
        if row is None:
            return None
        if not include_deleted and bool(row["is_deleted"]):
            return None
        return _row_to_metadata(row, agent_read_status)
    finally:
        conn.close()


def get_capsules_by_full_ids(
    capsule_type: str,
    lineage_name: Optional[str],
    capsule_ids: Sequence[str],
    *,
    agent_read_status: Optional[Dict[str, bool]] = None,
    visible_agent_name: Optional[str] = None,
    include_deleted: bool = False,
) -> List[Dict[str, Any]]:
    requested_ids = [str(value) for value in capsule_ids if value]
    if not requested_ids:
        return []

    ensure_capsule_index()
    lineage_key = _lineage_key(capsule_type, lineage_name)
    placeholders = ",".join("?" for _ in requested_ids)
    where = [
        "m.capsule_type = ?",
        "m.lineage_key = ?",
        f"m.capsule_id IN ({placeholders})",
    ]
    params: List[Any] = [capsule_type, lineage_key, *requested_ids]
    if not include_deleted:
        where.append("m.is_deleted = 0")
    if visible_agent_name:
        where.append(
            """
            (
                m.author_name = ?
                OR EXISTS (
                    SELECT 1
                    FROM capsule_recipients r
                    WHERE r.capsule_type = m.capsule_type
                      AND r.lineage_key = m.lineage_key
                      AND r.numeric_id = m.numeric_id
                      AND r.recipient_lower = ?
                )
            )
            """
        )
        params.extend([visible_agent_name, visible_agent_name.lower()])

    conn = _connect()
    try:
        _configure_database(conn)
        rows = conn.execute(
            f"SELECT m.* FROM capsule_metadata m WHERE {' AND '.join(where)}",
            params,
        ).fetchall()
        return [_row_to_metadata(row, agent_read_status) for row in rows]
    finally:
        conn.close()


def _connect() -> sqlite3.Connection:
    db_path = get_database_path()
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


def _ensure_sort_indexes_unlocked(db_path: str) -> None:
    """Add dashboard sort indexes without rebuilding the YAML-backed read model."""
    conn = _connect()
    try:
        _configure_database(conn)
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_capsule_scope_created
                ON capsule_metadata(capsule_type, lineage_key, is_deleted, created_at_tick DESC, numeric_id DESC);
            CREATE INDEX IF NOT EXISTS idx_capsule_scope_updated
                ON capsule_metadata(capsule_type, lineage_key, is_deleted, last_updated_at_tick DESC, numeric_id DESC);
            CREATE INDEX IF NOT EXISTS idx_capsule_scope_message_count
                ON capsule_metadata(capsule_type, lineage_key, is_deleted, total_message_count DESC, numeric_id DESC);
            CREATE INDEX IF NOT EXISTS idx_capsule_scope_title
                ON capsule_metadata(capsule_type, lineage_key, is_deleted, title COLLATE NOCASE, numeric_id DESC);
            CREATE INDEX IF NOT EXISTS idx_capsule_scope_author_sort
                ON capsule_metadata(capsule_type, lineage_key, is_deleted, author_name COLLATE NOCASE, numeric_id DESC);
            CREATE INDEX IF NOT EXISTS idx_capsule_question_status_sort
                ON capsule_metadata(capsule_type, lineage_key, is_deleted, question_status COLLATE NOCASE, numeric_id DESC);
            CREATE INDEX IF NOT EXISTS idx_capsule_question_vote_sort
                ON capsule_metadata(capsule_type, lineage_key, is_deleted, question_net_upvote DESC, numeric_id DESC);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _needs_rebuild() -> bool:
    db_path = get_database_path()
    if not os.path.exists(db_path):
        return True
    conn = _connect()
    try:
        _configure_database(conn)
        row = conn.execute(
            "SELECT value FROM index_metadata WHERE key = 'capsule_schema_version'"
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT value FROM index_metadata WHERE key = 'schema_version'"
            ).fetchone()
        return row is None or str(row["value"]) != SCHEMA_VERSION
    except sqlite3.DatabaseError as exc:
        if "no such table" in str(exc).lower():
            return True
        raise RuntimeError(f"CapsuleIndex: database unavailable path={db_path!r}") from exc
    finally:
        conn.close()


def _drop_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS capsule_tags;
        DROP TABLE IF EXISTS capsule_recipients;
        DROP TABLE IF EXISTS capsule_metadata;
        """
    )


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS capsule_metadata (
            capsule_type TEXT NOT NULL,
            lineage_key TEXT NOT NULL,
            numeric_id INTEGER NOT NULL,
            capsule_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_mtime_ns INTEGER NOT NULL,
            author_name TEXT,
            author_lineage TEXT,
            author_generation INTEGER,
            created_at_tick INTEGER,
            last_updated_at_tick INTEGER,
            title TEXT,
            abstract TEXT,
            word_count_total INTEGER NOT NULL DEFAULT 0,
            total_message_count INTEGER NOT NULL DEFAULT 0,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            reviewer_score REAL,
            question_status TEXT,
            question_net_upvote INTEGER,
            question_solved_by_message_id TEXT,
            tags_json TEXT NOT NULL DEFAULT '[]',
            recipients_json TEXT NOT NULL DEFAULT '[]',
            active_message_ids_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (capsule_type, lineage_key, numeric_id)
        );

        CREATE INDEX IF NOT EXISTS idx_capsule_scope_order
            ON capsule_metadata(capsule_type, lineage_key, is_deleted, numeric_id DESC);
        CREATE INDEX IF NOT EXISTS idx_capsule_scope_created
            ON capsule_metadata(capsule_type, lineage_key, is_deleted, created_at_tick DESC, numeric_id DESC);
        CREATE INDEX IF NOT EXISTS idx_capsule_scope_updated
            ON capsule_metadata(capsule_type, lineage_key, is_deleted, last_updated_at_tick DESC, numeric_id DESC);
        CREATE INDEX IF NOT EXISTS idx_capsule_scope_message_count
            ON capsule_metadata(capsule_type, lineage_key, is_deleted, total_message_count DESC, numeric_id DESC);
        CREATE INDEX IF NOT EXISTS idx_capsule_scope_title
            ON capsule_metadata(capsule_type, lineage_key, is_deleted, title COLLATE NOCASE, numeric_id DESC);
        CREATE INDEX IF NOT EXISTS idx_capsule_scope_author_sort
            ON capsule_metadata(capsule_type, lineage_key, is_deleted, author_name COLLATE NOCASE, numeric_id DESC);
        CREATE INDEX IF NOT EXISTS idx_capsule_question_status_sort
            ON capsule_metadata(capsule_type, lineage_key, is_deleted, question_status COLLATE NOCASE, numeric_id DESC);
        CREATE INDEX IF NOT EXISTS idx_capsule_question_vote_sort
            ON capsule_metadata(capsule_type, lineage_key, is_deleted, question_net_upvote DESC, numeric_id DESC);
        CREATE INDEX IF NOT EXISTS idx_capsule_author
            ON capsule_metadata(capsule_type, lineage_key, author_name, created_at_tick);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_capsule_path
            ON capsule_metadata(file_path);

        CREATE TABLE IF NOT EXISTS capsule_tags (
            capsule_type TEXT NOT NULL,
            lineage_key TEXT NOT NULL,
            numeric_id INTEGER NOT NULL,
            tag_lower TEXT NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (capsule_type, lineage_key, numeric_id, tag_lower)
        );
        CREATE INDEX IF NOT EXISTS idx_capsule_tags_lookup
            ON capsule_tags(capsule_type, lineage_key, tag_lower);

        CREATE TABLE IF NOT EXISTS capsule_recipients (
            capsule_type TEXT NOT NULL,
            lineage_key TEXT NOT NULL,
            numeric_id INTEGER NOT NULL,
            recipient_lower TEXT NOT NULL,
            recipient TEXT NOT NULL,
            PRIMARY KEY (capsule_type, lineage_key, numeric_id, recipient_lower)
        );
        CREATE INDEX IF NOT EXISTS idx_capsule_recipients_lookup
            ON capsule_recipients(capsule_type, lineage_key, recipient_lower);
        """
    )


def _set_schema_version(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO index_metadata(key, value)
        VALUES('capsule_schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (SCHEMA_VERSION,),
    )


def _build_where_clause(
    capsule_type: str,
    lineage_key: str,
    *,
    tag_filter: Optional[str],
    visible_agent_name: Optional[str],
    exclude_capsule_ids: Optional[Sequence[str]],
    include_deleted: bool,
) -> Tuple[str, List[Any]]:
    where = ["m.capsule_type = ?", "m.lineage_key = ?"]
    params: List[Any] = [capsule_type, lineage_key]
    if not include_deleted:
        where.append("m.is_deleted = 0")
    if tag_filter:
        where.append(
            """
            EXISTS (
                SELECT 1
                FROM capsule_tags t
                WHERE t.capsule_type = m.capsule_type
                  AND t.lineage_key = m.lineage_key
                  AND t.numeric_id = m.numeric_id
                  AND t.tag_lower = ?
            )
            """
        )
        params.append(str(tag_filter).strip().lower())
    if visible_agent_name:
        where.append(
            """
            (
                m.author_name = ?
                OR EXISTS (
                    SELECT 1
                    FROM capsule_recipients r
                    WHERE r.capsule_type = m.capsule_type
                      AND r.lineage_key = m.lineage_key
                      AND r.numeric_id = m.numeric_id
                      AND r.recipient_lower = ?
                )
            )
            """
        )
        params.extend([visible_agent_name, visible_agent_name.lower()])
    excluded = [str(value) for value in (exclude_capsule_ids or []) if value]
    if excluded:
        where.append("m.capsule_id NOT IN (" + ",".join("?" for _ in excluded) + ")")
        params.extend(excluded)
    return " AND ".join(where), params


def _upsert_file_unlocked(
    conn: sqlite3.Connection,
    capsule_type: str,
    lineage_name: Optional[str],
    numeric_id: int,
    path: str,
) -> None:
    data = file_io_utils.load_yaml(path)
    if not isinstance(data, dict):
        return
    _upsert_capsule_unlocked(conn, data, path, lineage_name=lineage_name, numeric_id=numeric_id, capsule_type=capsule_type)


def _upsert_capsule_unlocked(
    conn: sqlite3.Connection,
    capsule_data: Dict[str, Any],
    file_path: str,
    *,
    lineage_name: Optional[str] = None,
    numeric_id: Optional[int] = None,
    capsule_type: Optional[str] = None,
) -> None:
    capsule_id = str(capsule_data.get(constants.CAPSULE_ID_KEY) or "")
    capsule_type = str(capsule_data.get(constants.CAPSULE_TYPE_KEY) or capsule_type or "")
    if not capsule_id or not capsule_type:
        return

    numeric_id = _extract_numeric_id(capsule_id) if numeric_id is None else int(numeric_id)
    if numeric_id is None:
        return

    lineage_value = capsule_data.get(constants.CAPSULE_LINEAGE_ASSOCIATION_KEY) or lineage_name
    lineage_key = _lineage_key(capsule_type, lineage_value)
    tags = _clean_string_list(capsule_data.get(constants.CAPSULE_TAGS_KEY))
    recipients = _clean_string_list(capsule_data.get(constants.CAPSULE_RECIPIENTS_KEY))
    active_message_ids = _active_message_ids(capsule_data)
    reviewer_score = _extract_reviewer_score(capsule_data) if capsule_type == constants.CAPSULE_TYPE_ARCHIVE else None
    question_status = _question_status(capsule_data) if capsule_type == constants.CAPSULE_TYPE_QUESTION else None
    question_net_upvote = (
        _as_optional_int(capsule_data.get(constants.QUESTION_NET_UPVOTE_KEY))
        if capsule_type == constants.CAPSULE_TYPE_QUESTION else None
    )
    question_solved_by_message_id = (
        _as_optional_str(capsule_data.get(constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY))
        if capsule_type == constants.CAPSULE_TYPE_QUESTION else None
    )
    file_mtime_ns = _file_mtime_ns(file_path)

    conn.execute(
        """
        INSERT INTO capsule_metadata(
            capsule_type, lineage_key, numeric_id, capsule_id, file_path, file_mtime_ns,
            author_name, author_lineage, author_generation, created_at_tick, last_updated_at_tick,
            title, abstract, word_count_total, total_message_count, is_deleted, reviewer_score,
            question_status, question_net_upvote, question_solved_by_message_id,
            tags_json, recipients_json, active_message_ids_json
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(capsule_type, lineage_key, numeric_id) DO UPDATE SET
            capsule_id = excluded.capsule_id,
            file_path = excluded.file_path,
            file_mtime_ns = excluded.file_mtime_ns,
            author_name = excluded.author_name,
            author_lineage = excluded.author_lineage,
            author_generation = excluded.author_generation,
            created_at_tick = excluded.created_at_tick,
            last_updated_at_tick = excluded.last_updated_at_tick,
            title = excluded.title,
            abstract = excluded.abstract,
            word_count_total = excluded.word_count_total,
            total_message_count = excluded.total_message_count,
            is_deleted = excluded.is_deleted,
            reviewer_score = excluded.reviewer_score,
            question_status = excluded.question_status,
            question_net_upvote = excluded.question_net_upvote,
            question_solved_by_message_id = excluded.question_solved_by_message_id,
            tags_json = excluded.tags_json,
            recipients_json = excluded.recipients_json,
            active_message_ids_json = excluded.active_message_ids_json
        """,
        (
            capsule_type,
            lineage_key,
            int(numeric_id),
            capsule_id,
            os.path.abspath(file_path),
            int(file_mtime_ns or 0),
            _as_optional_str(capsule_data.get(constants.CAPSULE_AUTHOR_NAME_KEY)),
            _as_optional_str(capsule_data.get(constants.CAPSULE_AUTHOR_LINEAGE_KEY)),
            _as_optional_int(capsule_data.get(constants.CAPSULE_AUTHOR_GENERATION_KEY)),
            _as_optional_int(capsule_data.get(constants.CAPSULE_CREATED_AT_TICK_KEY)),
            _as_optional_int(capsule_data.get(constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY)),
            _as_optional_str(capsule_data.get(constants.CAPSULE_TITLE_KEY)),
            _as_optional_str(capsule_data.get(constants.CAPSULE_ABSTRACT_KEY)),
            _as_optional_int(capsule_data.get(constants.CAPSULE_WORD_COUNT_TOTAL_KEY)) or 0,
            len(active_message_ids),
            1 if bool(capsule_data.get(constants.CAPSULE_IS_DELETED_KEY, False)) else 0,
            reviewer_score,
            question_status,
            question_net_upvote,
            question_solved_by_message_id,
            _json_dumps(tags),
            _json_dumps(recipients),
            _json_dumps(active_message_ids),
        ),
    )

    conn.execute(
        "DELETE FROM capsule_tags WHERE capsule_type = ? AND lineage_key = ? AND numeric_id = ?",
        (capsule_type, lineage_key, int(numeric_id)),
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO capsule_tags(capsule_type, lineage_key, numeric_id, tag_lower, tag)
        VALUES(?, ?, ?, ?, ?)
        """,
        [(capsule_type, lineage_key, int(numeric_id), tag.lower(), tag) for tag in tags],
    )

    conn.execute(
        "DELETE FROM capsule_recipients WHERE capsule_type = ? AND lineage_key = ? AND numeric_id = ?",
        (capsule_type, lineage_key, int(numeric_id)),
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO capsule_recipients(capsule_type, lineage_key, numeric_id, recipient_lower, recipient)
        VALUES(?, ?, ?, ?, ?)
        """,
        [(capsule_type, lineage_key, int(numeric_id), recipient.lower(), recipient) for recipient in recipients],
    )


def _delete_key_unlocked(conn: sqlite3.Connection, capsule_type: str, lineage_key: str, numeric_id: int) -> None:
    key = (capsule_type, lineage_key, int(numeric_id))
    conn.execute(
        "DELETE FROM capsule_tags WHERE capsule_type = ? AND lineage_key = ? AND numeric_id = ?",
        key,
    )
    conn.execute(
        "DELETE FROM capsule_recipients WHERE capsule_type = ? AND lineage_key = ? AND numeric_id = ?",
        key,
    )
    conn.execute(
        "DELETE FROM capsule_metadata WHERE capsule_type = ? AND lineage_key = ? AND numeric_id = ?",
        key,
    )


def _row_to_metadata(row: sqlite3.Row, agent_read_status: Optional[Dict[str, bool]]) -> Dict[str, Any]:
    active_message_ids = _json_loads(row["active_message_ids_json"])
    metadata: Dict[str, Any] = {
        constants.CAPSULE_ID_KEY: row["capsule_id"],
        constants.CAPSULE_TYPE_KEY: row["capsule_type"],
        constants.CAPSULE_AUTHOR_NAME_KEY: row["author_name"],
        constants.CAPSULE_AUTHOR_LINEAGE_KEY: row["author_lineage"],
        constants.CAPSULE_AUTHOR_GENERATION_KEY: row["author_generation"],
        constants.CAPSULE_CREATED_AT_TICK_KEY: row["created_at_tick"],
        constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY: row["last_updated_at_tick"],
        constants.CAPSULE_TITLE_KEY: row["title"],
        constants.CAPSULE_TAGS_KEY: _json_loads(row["tags_json"]),
        constants.CAPSULE_ABSTRACT_KEY: row["abstract"],
        constants.CAPSULE_WORD_COUNT_TOTAL_KEY: row["word_count_total"] or 0,
        constants.CAPSULE_IS_DELETED_KEY: bool(row["is_deleted"]),
        "total_message_count": row["total_message_count"] or 0,
    }
    if row["lineage_key"]:
        metadata[constants.CAPSULE_LINEAGE_ASSOCIATION_KEY] = row["lineage_key"]
    recipients = _json_loads(row["recipients_json"])
    if recipients:
        metadata[constants.CAPSULE_RECIPIENTS_KEY] = recipients
    if row["reviewer_score"] is not None:
        metadata["reviewer_score"] = row["reviewer_score"]
    if row["capsule_type"] == constants.CAPSULE_TYPE_QUESTION:
        metadata[constants.QUESTION_STATUS_KEY] = row["question_status"] or constants.QUESTION_STATUS_PENDING
        metadata[constants.QUESTION_NET_UPVOTE_KEY] = row["question_net_upvote"] or 0
        metadata[constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY] = row["question_solved_by_message_id"]

    unread_count = 0
    if agent_read_status is not None:
        unread_count = sum(1 for msg_id in active_message_ids if not agent_read_status.get(msg_id, False))
    metadata[constants.CAPSULE_UNREAD_MESSAGE_COUNT_KEY] = unread_count
    return metadata


def _iter_all_capsule_files() -> Iterable[Tuple[str, Optional[str], int, str]]:
    for capsule_type in (
        constants.CAPSULE_TYPE_PUBLIC,
        constants.CAPSULE_TYPE_MAIL,
        constants.CAPSULE_TYPE_ARCHIVE,
        constants.CAPSULE_TYPE_QUESTION,
    ):
        yield from _iter_scope_capsule_files(capsule_type, None)

    private_root = os.path.join(
        constants.BASE_STATION_DATA_PATH,
        constants.CAPSULES_DIR_NAME,
        constants.PRIVATE_CAPSULES_SUBDIR_NAME,
    )
    if not os.path.isdir(private_root):
        return
    for dirname in sorted(os.listdir(private_root)):
        dir_path = os.path.join(private_root, dirname)
        if not os.path.isdir(dir_path) or not dirname.startswith("lineage_"):
            continue
        lineage_name = dirname[len("lineage_") :]
        yield from _iter_scope_capsule_files(constants.CAPSULE_TYPE_PRIVATE, lineage_name)


def _iter_scope_capsule_files(
    capsule_type: str,
    lineage_name: Optional[str] = None,
) -> Iterable[Tuple[str, Optional[str], int, str]]:
    dir_path, prefix = _scope_dir_and_prefix(capsule_type, lineage_name)
    if not os.path.isdir(dir_path):
        return
    pattern = re.compile(f"^{re.escape(prefix)}(\\d+){re.escape(constants.YAML_EXTENSION)}$")
    for filename in file_io_utils.list_files(dir_path, constants.YAML_EXTENSION):
        match = pattern.match(filename)
        if not match:
            continue
        try:
            numeric_id = int(match.group(1))
        except ValueError:
            continue
        yield capsule_type, lineage_name, numeric_id, os.path.join(dir_path, filename)


def _scope_dir_and_prefix(capsule_type: str, lineage_name: Optional[str]) -> Tuple[str, str]:
    base_capsules_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.CAPSULES_DIR_NAME)
    if capsule_type == constants.CAPSULE_TYPE_PUBLIC:
        return os.path.join(base_capsules_path, constants.PUBLIC_CAPSULES_SUBDIR_NAME), "public_"
    if capsule_type == constants.CAPSULE_TYPE_MAIL:
        return os.path.join(base_capsules_path, constants.MAIL_CAPSULES_SUBDIR_NAME), "mail_"
    if capsule_type == constants.CAPSULE_TYPE_ARCHIVE:
        return os.path.join(base_capsules_path, constants.ARCHIVE_CAPSULES_SUBDIR_NAME), "archive_"
    if capsule_type == constants.CAPSULE_TYPE_QUESTION:
        return os.path.join(base_capsules_path, constants.QUESTION_CAPSULES_SUBDIR_NAME), "question_"
    if capsule_type == constants.CAPSULE_TYPE_PRIVATE:
        if not lineage_name:
            raise ValueError("Lineage name required for private capsules.")
        safe_lineage_name = _safe_lineage_name(lineage_name)
        return (
            os.path.join(base_capsules_path, constants.PRIVATE_CAPSULES_SUBDIR_NAME, f"lineage_{safe_lineage_name}"),
            f"{safe_lineage_name}_private_",
        )
    raise ValueError(f"Unknown capsule type: {capsule_type}")


def _lineage_key(capsule_type: str, lineage_name: Optional[str]) -> str:
    if capsule_type != constants.CAPSULE_TYPE_PRIVATE:
        return ""
    return _safe_lineage_name(lineage_name or "")


def _safe_lineage_name(lineage_name: str) -> str:
    return "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in str(lineage_name))


def _extract_numeric_id(capsule_id: str) -> Optional[int]:
    match = re.search(r"(\d+)$", str(capsule_id))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _active_message_ids(capsule_data: Dict[str, Any]) -> List[str]:
    messages = capsule_data.get(constants.CAPSULE_MESSAGES_KEY, [])
    if not isinstance(messages, list):
        return []
    return [
        str(message.get(constants.MESSAGE_ID_KEY))
        for message in messages
        if isinstance(message, dict)
        and message.get(constants.MESSAGE_ID_KEY)
        and not message.get(constants.MESSAGE_IS_DELETED_KEY, False)
    ]


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


def _extract_reviewer_score(capsule_data: Dict[str, Any]) -> Optional[float]:
    messages = capsule_data.get(constants.CAPSULE_MESSAGES_KEY, [])
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        content = str(message.get(constants.MESSAGE_CONTENT_KEY, "") or "")
        if "Reviewer Evaluation" not in content:
            continue
        match = re.search(r"\*\*Score:\*\*\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10", content)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _question_status(capsule_data: Dict[str, Any]) -> str:
    status = str(capsule_data.get(constants.QUESTION_STATUS_KEY) or "").strip().lower()
    allowed = {
        constants.QUESTION_STATUS_PENDING,
        constants.QUESTION_STATUS_OPEN,
        constants.QUESTION_STATUS_REDACTED,
        constants.QUESTION_STATUS_SOLVED,
        constants.QUESTION_STATUS_RETIRED,
    }
    return status if status in allowed else constants.QUESTION_STATUS_PENDING


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


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: Any) -> List[str]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item)]
