from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from station import file_io_utils
from station.multistart import paths


STATE_FILENAME = "state.yaml"
JOB_LOG_FILENAME = "job.log"
ORIGIN_DIR_NAME = "origin_station_data"
ADMIN_DIR_NAME = "admin"
CONTROL_KEY = "control"
CONTROL_PAUSED = "paused"
CONTROL_RUNNING = "running"
SHUTDOWN_REQUESTED_KEY = "shutdown_requested"
SHUTDOWN_REQUESTED_AT_KEY = "shutdown_requested_at"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = file_io_utils.load_yaml(str(path))
    return data if isinstance(data, dict) else {}


def save_yaml_mapping(path: Path, data: dict[str, Any]) -> None:
    file_io_utils.save_yaml(data, str(path))


def load_current_job(repo: Path | None = None) -> dict[str, Any]:
    return load_yaml_mapping(paths.current_job_path(repo))


def save_current_job(repo: Path, data: dict[str, Any]) -> None:
    save_yaml_mapping(paths.current_job_path(repo), data)


def clear_current_job(repo: Path) -> None:
    path = paths.current_job_path(repo)
    if path.exists():
        path.unlink()


def job_dir(repo: Path, branch_tick: int, job_id: str) -> Path:
    return paths.multistart_root(repo) / f"{int(branch_tick)}_{job_id}"


def state_path(job_path: Path) -> Path:
    return job_path / STATE_FILENAME


def load_job_state(job_path: Path) -> dict[str, Any]:
    return load_yaml_mapping(state_path(job_path))


def save_job_state(job_path: Path, data: dict[str, Any]) -> None:
    data["updated_at"] = utc_now()
    save_yaml_mapping(state_path(job_path), data)


def job_control(data: dict[str, Any]) -> str:
    raw = str(data.get(CONTROL_KEY) or CONTROL_RUNNING).strip().lower()
    return raw if raw else CONTROL_RUNNING


def job_paused(data: dict[str, Any]) -> bool:
    return job_control(data) == CONTROL_PAUSED


def set_job_control(job_path: Path, control: str) -> dict[str, Any]:
    payload = load_job_state(job_path)
    payload[CONTROL_KEY] = control
    save_job_state(job_path, payload)
    return payload


def append_job_log(job_path: Path, message: str) -> None:
    file_io_utils.ensure_dir_exists(str(job_path))
    line = f"{utc_now()} {message.rstrip()}\n"
    with (job_path / JOB_LOG_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_station_config(data_root: Path) -> dict[str, Any]:
    return load_yaml_mapping(data_root / "station_config.yaml")


def branch_dir(job_path: Path, seed: int) -> Path:
    return job_path / f"station_data_s{int(seed)}"


def origin_dir(job_path: Path) -> Path:
    return job_path / ORIGIN_DIR_NAME


def admin_dir(job_path: Path) -> Path:
    return job_path / ADMIN_DIR_NAME


def append_yamll(path: Path, record: dict[str, Any]) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("---\n")
        yaml.safe_dump(record, handle, sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000)
        handle.flush()
        os.fsync(handle.fileno())
