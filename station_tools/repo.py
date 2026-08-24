"""Helpers for station checkout metadata and local process state."""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StationMetadata:
    path: Path
    station_id: str
    station_name: str
    current_tick: int | None = None
    top_score: Any = None


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def station_config_path(repo: Path) -> Path:
    return repo / "station_data" / "station_config.yaml"


def read_station_metadata(repo: Path) -> StationMetadata:
    data = load_yaml_mapping(station_config_path(repo))
    return StationMetadata(
        path=repo,
        station_id=str(data.get("station_id") or "").strip(),
        station_name=str(data.get("station_name") or repo.name).strip(),
        current_tick=_optional_int(data.get("current_tick")),
        top_score=data.get("top_score"),
    )


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def parse_env_file(repo: Path) -> dict[str, str]:
    env_path = repo / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        value = strip_inline_comment(raw_value).strip()
        try:
            parts = shlex.split(value, comments=False, posix=True)
            value = parts[0] if parts else ""
        except ValueError:
            value = value.strip("'\"")
        values[key] = value
    return values


def read_station_https_port(repo: Path) -> int | None:
    nginx_config = repo / "deployment" / "nginx.conf"
    if not nginx_config.exists():
        return None
    text = nginx_config.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"\blisten\s+(?:\[[^\]]+\]:|[^:;\s]+:)?(?P<port>\d+)\b[^;]*\bssl\b")
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group("port"))
    return None


def has_live_process_from_pid_file(pid_file: Path) -> bool:
    if not pid_file.is_file():
        return False
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    result = subprocess.run(["ps", "-p", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0


def station_service_running(repo: Path) -> bool:
    return has_live_process_from_pid_file(repo / "deployment" / "gunicorn.pid") or has_live_process_from_pid_file(
        repo / "deployment" / "nginx.pid"
    )
