"""Shared path helpers for the station SQLite index database."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Optional

from station import constants
from station import file_io_utils


INDEX_DB_PATH_ENV = "STATION_INDEX_DB_PATH"
INDEX_DB_DIR_ENV = "STATION_INDEX_DB_DIR"
_STATION_INDEX_WRITE_LOCK = threading.RLock()


def get_station_index_write_lock() -> threading.RLock:
    """Return the process-wide writer lock for the shared station index DB."""
    return _STATION_INDEX_WRITE_LOCK


def get_station_index_database_path(base_path: Optional[str] = None) -> str:
    """Return the SQLite index path for a station data root.

    Explicit tmp-file overrides are made station-specific so multiple stations
    can run on one machine without contending for the same SQLite file.
    """
    resolved_base = os.path.abspath(base_path or constants.BASE_STATION_DATA_PATH)

    override_path = os.environ.get(INDEX_DB_PATH_ENV)
    if override_path:
        return _resolve_override_path(override_path, resolved_base)

    override_dir = os.environ.get(INDEX_DB_DIR_ENV)
    if override_dir:
        return os.path.join(
            os.path.abspath(os.path.expanduser(override_dir)),
            _station_index_filename(resolved_base, constants.STATION_INDEX_DB_FILENAME),
        )

    return os.path.join(
        resolved_base,
        constants.STATION_INDEX_DIR_NAME,
        constants.STATION_INDEX_DB_FILENAME,
    )


def get_station_index_key(base_path: Optional[str] = None) -> str:
    resolved_base = os.path.abspath(base_path or constants.BASE_STATION_DATA_PATH)
    station_id = _read_station_id(resolved_base)
    if station_id:
        return _sanitize_component(station_id)
    return "path_" + hashlib.sha1(resolved_base.encode("utf-8")).hexdigest()[:12]


def _resolve_override_path(raw_path: str, base_path: str) -> str:
    expanded = os.path.abspath(os.path.expanduser(raw_path))
    if raw_path.endswith(os.sep) or os.path.isdir(expanded):
        return os.path.join(expanded, _station_index_filename(base_path, constants.STATION_INDEX_DB_FILENAME))

    if _is_under_tmp(expanded):
        filename = os.path.basename(expanded)
        station_key = get_station_index_key(base_path)
        if station_key not in filename:
            filename = _station_index_filename(base_path, filename)
        return os.path.join(os.path.dirname(expanded), filename)

    return expanded


def _station_index_filename(base_path: str, filename: str) -> str:
    return f"{get_station_index_key(base_path)}_{filename}"


def _read_station_id(base_path: str) -> str:
    config_path = os.path.join(base_path, constants.STATION_CONFIG_FILENAME)
    config = file_io_utils.load_yaml(config_path)
    if not isinstance(config, dict):
        return ""
    station_id = config.get(constants.STATION_ID_KEY)
    return str(station_id).strip() if station_id is not None else ""


def _sanitize_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if sanitized:
        return sanitized[:96]
    return "station"


def _is_under_tmp(path: str) -> bool:
    try:
        return Path(path).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
    except OSError:
        return os.path.abspath(path).startswith(os.path.abspath(tempfile.gettempdir()) + os.sep)
