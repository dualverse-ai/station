from __future__ import annotations

import os
from pathlib import Path


ROOT_NAME = "station_multistart"
CURRENT_JOB_FILENAME = "current_job.yaml"
CONTROLLER_PID_FILENAME = "controller.pid"
CONTROLLER_SOCK_FILENAME = "controller.sock"
CONTROLLER_LOG_FILENAME = "controller.log"
PENDING_INIT_FILENAME = "pending_init.yaml"
PENDING_STAGNATION_FILENAME = "pending_stagnation.yaml"


def repo_root() -> Path:
    return Path(os.environ.get("STATION_REPO_ROOT") or os.getcwd()).resolve()


def multistart_root(repo: Path | None = None) -> Path:
    return (repo or repo_root()) / ROOT_NAME


def current_job_path(repo: Path | None = None) -> Path:
    return multistart_root(repo) / CURRENT_JOB_FILENAME


def controller_pid_path(repo: Path | None = None) -> Path:
    return multistart_root(repo) / CONTROLLER_PID_FILENAME


def controller_sock_path(repo: Path | None = None) -> Path:
    return multistart_root(repo) / CONTROLLER_SOCK_FILENAME


def controller_log_path(repo: Path | None = None) -> Path:
    return multistart_root(repo) / CONTROLLER_LOG_FILENAME


def pending_init_path(repo: Path | None = None) -> Path:
    return multistart_root(repo) / PENDING_INIT_FILENAME


def pending_stagnation_path(repo: Path | None = None) -> Path:
    return multistart_root(repo) / PENDING_STAGNATION_FILENAME


def live_station_data_path(repo: Path | None = None) -> Path:
    return (repo or repo_root()) / "station_data"


def draft_station_data_path(repo: Path | None = None) -> Path:
    return (repo or repo_root()) / "draft_station_data"
