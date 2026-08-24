"""SQLite read model for archive reviewer evaluation logs.

Archive evaluation YAML logs remain authoritative. This module stores only the
compact fields needed for lineage-evolution aggregate counts.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Any, Dict

from station import constants
from station import file_io_utils
from station import index_paths


SCHEMA_VERSION = "1"
_INDEX_LOCK = threading.RLock()


def get_archive_evaluations_dir() -> str:
    return os.path.join(
        constants.BASE_STATION_DATA_PATH,
        constants.ROOMS_DIR_NAME,
        constants.SHORT_ROOM_NAME_ARCHIVE,
        constants.ARCHIVE_EVALUATIONS_SUBDIR_NAME,
    )


def get_database_path() -> str:
    return index_paths.get_station_index_database_path(constants.BASE_STATION_DATA_PATH)


def should_rebuild_from_process_args() -> bool:
    import sys

    if os.environ.get("STATION_REBUILD_DB", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return any(arg in {"--rebuild-db", "--rebuild_db"} for arg in sys.argv[1:])


def ensure_archive_evaluation_index(*, rebuild: bool = False, log_status: bool = False) -> None:
    with _INDEX_LOCK:
        db_path = get_database_path()
        if rebuild:
            print(f"ArchiveEvalIndex: rebuild requested path={db_path!r}")
            rebuild_archive_evaluation_index()
        elif _needs_rebuild():
            rebuild_archive_evaluation_index()
        elif log_status:
            print(f"ArchiveEvalIndex: ready path={db_path!r}")


def rebuild_archive_evaluation_index() -> None:
    with _INDEX_LOCK:
        with index_paths.get_station_index_write_lock():
            db_path = get_database_path()
            file_io_utils.ensure_dir_exists(os.path.dirname(db_path))
            print(f"ArchiveEvalIndex: rebuilding path={db_path!r}")
            conn = _connect()
            indexed_count = 0
            try:
                _configure_database(conn, setup_wal=True)
                conn.execute("BEGIN IMMEDIATE")
                _create_schema(conn)
                conn.execute("DELETE FROM archive_evaluation_metadata")
                for path in _iter_archive_evaluation_files():
                    try:
                        data = file_io_utils.load_yaml(path)
                        if not isinstance(data, dict):
                            continue
                        _upsert_archive_evaluation_unlocked(conn, data, path)
                        indexed_count += 1
                    except Exception:
                        continue
                _set_schema_version(conn)
                conn.commit()
                print(f"ArchiveEvalIndex: rebuild complete path={db_path!r} evaluations={indexed_count}")
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()


def upsert_archive_evaluation(log_data: Dict[str, Any], file_path: str) -> None:
    if not isinstance(log_data, dict):
        return
    ensure_archive_evaluation_index()
    with index_paths.get_station_index_write_lock():
        conn = _connect()
        try:
            _configure_database(conn)
            with conn:
                _upsert_archive_evaluation_unlocked(conn, log_data, file_path)
                _set_schema_version(conn)
        finally:
            conn.close()


def count_high_quality_papers_by_lineage(score_threshold: float = 8.0) -> Dict[str, int]:
    ensure_archive_evaluation_index()
    conn = _connect()
    try:
        _configure_database(conn)
        rows = conn.execute(
            """
            SELECT agent_lineage, COUNT(*) AS paper_count
            FROM archive_evaluation_metadata
            WHERE result = 'accepted'
              AND score IS NOT NULL
              AND score >= ?
              AND agent_lineage IS NOT NULL
              AND agent_lineage != ''
            GROUP BY agent_lineage
            """,
            (float(score_threshold),),
        ).fetchall()
        return {str(row["agent_lineage"]): int(row["paper_count"]) for row in rows}
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
    if setup_wal:
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.DatabaseError:
            pass


def _needs_rebuild() -> bool:
    db_path = get_database_path()
    if not os.path.exists(db_path):
        return True
    conn = _connect()
    try:
        _configure_database(conn)
        row = conn.execute(
            "SELECT value FROM index_metadata WHERE key = 'archive_evaluation_schema_version'"
        ).fetchone()
        return row is None or str(row["value"]) != SCHEMA_VERSION
    except sqlite3.DatabaseError as exc:
        if "no such table" in str(exc).lower():
            return True
        raise RuntimeError(f"ArchiveEvalIndex: database unavailable path={db_path!r}") from exc
    finally:
        conn.close()


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS archive_evaluation_metadata (
            file_path TEXT PRIMARY KEY,
            file_mtime_ns INTEGER NOT NULL,
            evaluation_id TEXT,
            agent_name TEXT,
            agent_lineage TEXT,
            result TEXT,
            score REAL
        );

        CREATE INDEX IF NOT EXISTS idx_archive_eval_quality
            ON archive_evaluation_metadata(result, score, agent_lineage);
        """
    )


def _set_schema_version(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO index_metadata(key, value)
        VALUES('archive_evaluation_schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (SCHEMA_VERSION,),
    )


def _upsert_archive_evaluation_unlocked(
    conn: sqlite3.Connection,
    log_data: Dict[str, Any],
    file_path: str,
) -> None:
    extracted_result = log_data.get("extracted_result", {})
    score = None
    if isinstance(extracted_result, dict):
        score = _as_optional_float(extracted_result.get("score"))
    agent_name = str(log_data.get("agent_name") or "")
    conn.execute(
        """
        INSERT INTO archive_evaluation_metadata(
            file_path, file_mtime_ns, evaluation_id, agent_name, agent_lineage, result, score
        )
        VALUES(?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            file_mtime_ns = excluded.file_mtime_ns,
            evaluation_id = excluded.evaluation_id,
            agent_name = excluded.agent_name,
            agent_lineage = excluded.agent_lineage,
            result = excluded.result,
            score = excluded.score
        """,
        (
            os.path.abspath(file_path),
            _file_mtime_ns(file_path),
            _as_optional_str(log_data.get("evaluation_id")),
            agent_name,
            _extract_lineage_from_agent_name(agent_name),
            str(log_data.get("result") or ""),
            score,
        ),
    )


def _iter_archive_evaluation_files():
    evaluations_dir = get_archive_evaluations_dir()
    if not os.path.isdir(evaluations_dir):
        return
    for filename in file_io_utils.list_files(evaluations_dir, constants.YAML_EXTENSION):
        yield os.path.join(evaluations_dir, filename)


def _extract_lineage_from_agent_name(agent_name: str) -> str:
    if not agent_name or agent_name.startswith("Guest_") or agent_name.lower() == "system":
        return ""
    parts = agent_name.split()
    return parts[0] if parts else ""


def _file_mtime_ns(path: str) -> int:
    try:
        return os.stat(path).st_mtime_ns
    except (FileNotFoundError, OSError):
        return 0


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
