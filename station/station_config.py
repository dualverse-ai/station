from __future__ import annotations

import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from station import __version__, constants


def current_git_commit(repo: str | Path | None = None) -> str:
    cwd = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def ensure_top_submission_config_defaults(config_data: dict[str, Any]) -> None:
    config_data.setdefault(constants.STATION_CONFIG_TOP_EVALUATION_ID, None)
    config_data.setdefault(constants.STATION_CONFIG_TOP_TITLE, None)
    config_data.setdefault(constants.STATION_CONFIG_TOP_SCORE, None)
    config_data.setdefault(constants.STATION_CONFIG_TOP_SORT_KEY, None)
    config_data.setdefault(constants.STATION_CONFIG_TOP_TICK, None)
    config_data.setdefault(constants.STATION_CONFIG_TOP_AGENT_NAME, None)
    config_data.setdefault(constants.STATION_CONFIG_TOP_TAGS, [])
    config_data.setdefault(constants.STATION_CONFIG_TOP_ABSTRACT, "")


def top_submission_from_config(config_data: dict[str, Any]) -> dict[str, Any] | None:
    """Return the compact top-submission cache persisted in station config."""
    top_id = config_data.get(constants.STATION_CONFIG_TOP_EVALUATION_ID)
    if top_id is None:
        return None
    return {
        "evaluation_id": str(top_id),
        "title": config_data.get(constants.STATION_CONFIG_TOP_TITLE),
        "score": config_data.get(constants.STATION_CONFIG_TOP_SCORE),
        "sort_key": config_data.get(constants.STATION_CONFIG_TOP_SORT_KEY),
        "submitted_tick": config_data.get(constants.STATION_CONFIG_TOP_TICK),
        "agent_name": config_data.get(constants.STATION_CONFIG_TOP_AGENT_NAME),
        "tags": list(config_data.get(constants.STATION_CONFIG_TOP_TAGS) or []),
        "abstract": config_data.get(constants.STATION_CONFIG_TOP_ABSTRACT) or "",
    }


def build_default_station_config(
    *,
    station_name: str = "",
    station_id: str | None = None,
    git_commit: str | None = None,
    spawn_date: str | None = None,
) -> dict[str, Any]:
    resolved_spawn_date = spawn_date or datetime.now().isoformat()
    config = {
        constants.STATION_CONFIG_CURRENT_TICK: 0,
        constants.STATION_CONFIG_AGENT_TURN_ORDER: [],
        constants.STATION_CONFIG_STATION_STATUS: "Healthy",
        constants.STATION_CONFIG_SOFTWARE_VERSION: __version__,
        constants.STATION_CONFIG_NEXT_AGENT_INDEX: 0,
        constants.STATION_CONFIG_NAME: station_name,
        constants.STATION_CONFIG_DESCRIPTION: "",
        constants.STATION_CONFIG_LAST_SUPERVISOR_DEPARTURE_TICK: 0,
        constants.STATION_CONFIG_STAGNATION_COUNTER: 0,
        constants.STATION_ID_KEY: station_id or str(uuid.uuid4()),
        "version": __version__,
        "git_commit": git_commit if git_commit is not None else current_git_commit(),
        "spawn_date": resolved_spawn_date,
        "status_history": [{"status": "Healthy", "start_tick": 0}],
    }
    ensure_top_submission_config_defaults(config)
    return config


def apply_station_config_defaults(
    config_data: dict[str, Any] | None,
    *,
    station_name: str | None = None,
    git_commit: str | None = None,
) -> tuple[dict[str, Any], bool]:
    config = dict(config_data or {})
    defaults = build_default_station_config(
        station_name=station_name or "",
        station_id=str(config.get(constants.STATION_ID_KEY) or "") or None,
        git_commit=git_commit if git_commit is not None else config.get("git_commit"),
        spawn_date=config.get("spawn_date"),
    )
    changed = False
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            changed = True

    if station_name is not None and config.get(constants.STATION_CONFIG_NAME) != station_name:
        config[constants.STATION_CONFIG_NAME] = station_name
        changed = True

    if not config.get(constants.STATION_ID_KEY):
        config[constants.STATION_ID_KEY] = str(uuid.uuid4())
        changed = True

    if not config.get("status_history"):
        current_tick = config.get(constants.STATION_CONFIG_CURRENT_TICK, 0)
        current_status = config.get(constants.STATION_CONFIG_STATION_STATUS, "Healthy")
        config["status_history"] = [{"status": current_status, "start_tick": current_tick}]
        changed = True

    return config, changed
