#!/usr/bin/env python3
"""Check whether an existing SQLite index has an older schema version.

This is intentionally read-only. Missing indexes return "no" because normal
station startup already rebuilds derived indexes when they are absent.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _connect_readonly(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        is not None
    )


def _capsule_index_version_changed(capsule_index) -> bool:
    db_path = capsule_index.get_database_path()
    if not os.path.exists(db_path):
        return False

    conn = _connect_readonly(db_path)
    try:
        if not _table_exists(conn, "index_metadata"):
            return False
        row = conn.execute(
            "SELECT value FROM index_metadata WHERE key = 'capsule_schema_version'"
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT value FROM index_metadata WHERE key = 'schema_version'"
            ).fetchone()
        return row is not None and str(row[0]) != capsule_index.SCHEMA_VERSION
    finally:
        conn.close()


def _research_index_version_changed(constants, evaluation_index, build_runtime_paths) -> bool:
    paths = build_runtime_paths(constants)
    if not os.path.isdir(paths.evaluations_dir):
        return False

    db_path = evaluation_index.get_database_path(paths.evaluations_dir)
    if not os.path.exists(db_path):
        return False

    conn = _connect_readonly(db_path)
    try:
        if not _table_exists(conn, "research_evaluation_scopes"):
            return False
        row = conn.execute(
            "SELECT schema_version FROM research_evaluation_scopes WHERE evaluations_dir = ?",
            (os.path.abspath(paths.evaluations_dir),),
        ).fetchone()
        return row is not None and str(row[0]) != evaluation_index.SCHEMA_VERSION
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="Station data root to inspect.")
    args = parser.parse_args()

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        from station import capsule_index
        from station import constants
        from station.eval_research import evaluation_index
        from station.eval_research.runtime_paths import build_runtime_paths

    if args.data_root:
        constants.BASE_STATION_DATA_PATH = args.data_root

    needs_migration = _capsule_index_version_changed(capsule_index) or _research_index_version_changed(
        constants,
        evaluation_index,
        build_runtime_paths,
    )
    print("yes" if needs_migration else "no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
