"""Station repository discovery and target parsing."""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Selection:
    repos: tuple[Path, ...]
    skipped: tuple[str, ...] = ()


def display_path(path: Path) -> str:
    expanded = path.expanduser()
    try:
        return "~/" + str(expanded.resolve().relative_to(Path.home()))
    except (OSError, ValueError):
        return str(expanded)


def suffix_for_repo(path: Path) -> str:
    name = path.name
    if name == "station":
        return "1"
    if name.startswith("station_"):
        return name[len("station_") :]
    return name


def token_to_path(token: str) -> Path:
    raw = token.strip()
    if raw == "1":
        return Path.home() / "station"
    if raw.isdigit():
        return Path.home() / f"station_{raw}"
    if raw.startswith("~/"):
        return Path(raw).expanduser()
    if raw.startswith("/") or raw.startswith("."):
        return Path(raw).expanduser()
    if raw.startswith("station"):
        return Path.home() / raw
    return Path.home() / f"station_{raw}"


def split_target_tokens(args: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for arg in args:
        for comma_part in str(arg).split(","):
            for token in comma_part.split():
                cleaned = token.strip()
                if cleaned:
                    tokens.append(cleaned)
    return tokens


def is_station_repo(path: Path, require_git: bool = False, require_start: bool = False) -> bool:
    if not path.is_dir():
        return False
    if require_git and not (path / ".git").is_dir():
        return False
    if require_start and not (path / "start.sh").is_file():
        return False
    return (
        (path / "station_data" / "station_config.yaml").is_file()
        or (path / "station_multistart" / "current_job.yaml").is_file()
        or (path / "station_multistart" / "current_job").is_file()
        or (path / "start.sh").is_file()
    )


def discover_repos(patterns: Iterable[str], require_git: bool = False, require_start: bool = False) -> tuple[Path, ...]:
    repos: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser()) if pattern.startswith("~") else pattern
        for raw in glob.glob(expanded):
            path = Path(raw).expanduser()
            key = path.resolve() if path.exists() else path
            if key in seen:
                continue
            if is_station_repo(path, require_git=require_git, require_start=require_start):
                seen.add(key)
                repos.append(path)
    return tuple(sorted(repos, key=lambda item: item.name))


def resolve_token(token: str, discovered: Sequence[Path]) -> Path:
    raw = token.strip()
    explicit = raw.startswith(("~/", "/", ".")) or raw.startswith("station") or raw.isdigit()
    if explicit:
        return token_to_path(raw)

    matches = [repo for repo in discovered if suffix_for_repo(repo) == raw]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = ", ".join(display_path(path) for path in matches)
        raise ValueError(f"ambiguous station suffix '{raw}': {choices}")
    return token_to_path(raw)


def select_repos(
    target_args: Sequence[str],
    patterns: Iterable[str],
    require_git: bool = False,
    require_start: bool = False,
) -> Selection:
    discovered = discover_repos(patterns, require_git=require_git, require_start=require_start)
    raw_repos = list(discovered)
    tokens = split_target_tokens(target_args)
    if tokens:
        raw_repos = [resolve_token(token, discovered) for token in tokens]

    repos: list[Path] = []
    skipped: list[str] = []
    seen: set[Path] = set()
    for repo in raw_repos:
        key = repo.resolve() if repo.exists() else repo
        if key in seen:
            continue
        seen.add(key)
        if is_station_repo(repo, require_git=require_git, require_start=require_start):
            repos.append(repo)
        else:
            skipped.append(str(repo))
    return Selection(repos=tuple(repos), skipped=tuple(skipped))


def targets_or_current(target_args: Sequence[str]) -> Sequence[str]:
    """Default an omitted checkout selector to the current Station root.

    Commands invoked outside a Station checkout retain their existing behavior:
    an empty target list lets ``select_repos`` use configured discovery.
    """
    if split_target_tokens(target_args):
        return target_args
    current = Path.cwd()
    if is_station_repo(current):
        return (str(current),)
    return target_args
