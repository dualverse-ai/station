"""Configuration loading for the multi-station CLI."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


DEFAULT_STATION_PATTERNS = ("~/station", "~/station_*")
CONFIG_FILENAME = "station_tools.toml"
CONFIG_DIRNAME = "station-tools"


@dataclass(frozen=True)
class ToolsConfig:
    path: Path | None = None
    station_patterns: tuple[str, ...] = DEFAULT_STATION_PATTERNS
    env: Mapping[str, str] = field(default_factory=dict)
    hooks: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


def default_config_path() -> Path:
    return Path.home() / ".config" / CONFIG_DIRNAME / CONFIG_FILENAME


def load_config(path: Path | None = None) -> ToolsConfig:
    config_path = path or default_config_path()
    if not config_path.exists():
        return ToolsConfig(path=None)

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    raw_patterns = data.get("station_patterns", DEFAULT_STATION_PATTERNS)
    if isinstance(raw_patterns, str):
        patterns = (raw_patterns,)
    else:
        patterns = tuple(str(item) for item in raw_patterns if str(item).strip())
    if not patterns:
        patterns = DEFAULT_STATION_PATTERNS

    raw_env = data.get("env", {})
    env = {str(key): str(value) for key, value in raw_env.items()} if isinstance(raw_env, dict) else {}

    hooks: dict[str, dict[str, str]] = {}
    raw_hooks = data.get("hooks", {})
    if isinstance(raw_hooks, dict):
        for scope, values in raw_hooks.items():
            if isinstance(values, dict):
                hooks[str(scope)] = {str(key): str(value) for key, value in values.items()}

    return ToolsConfig(
        path=config_path,
        station_patterns=patterns,
        env=env,
        hooks=hooks,
    )


def build_hook_env(config: ToolsConfig) -> dict[str, str]:
    env = os.environ.copy()
    for key, raw_value in config.env.items():
        value = raw_value.replace("${PATH}", env.get("PATH", ""))
        value = value.replace("$PATH", env.get("PATH", ""))
        env[key] = value
    return env
