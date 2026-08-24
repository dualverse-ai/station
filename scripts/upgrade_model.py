#!/usr/bin/env python3
"""Upgrade the model used by active agents, restarting Station around the edit."""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

with contextlib.redirect_stdout(io.StringIO()):
    from station import constants, file_io_utils  # noqa: E402


BRANCH_DATA_ROOT_PATTERN = re.compile(r"station_data_s[0-9]+$")


@dataclass(frozen=True)
class ModelUpdate:
    data_root: Path
    agent_path: Path
    agent_name: str


def agent_is_active(agent_data: dict[str, Any]) -> bool:
    return (
        agent_data.get(constants.AGENT_STATUS_KEY)
        in {constants.AGENT_STATUS_GUEST, constants.AGENT_STATUS_RECURSIVE}
        and bool(agent_data.get(constants.AGENT_NAME_KEY))
        and not bool(agent_data.get(constants.AGENT_SESSION_ENDED_KEY))
        and not bool(agent_data.get(constants.AGENT_IS_ASCENDED_KEY))
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    data = file_io_utils.load_yaml(str(path))
    return data if isinstance(data, dict) else {}


def _validated_job_dir(repo_root: Path, current_job: dict[str, Any]) -> Path:
    raw_job_dir = current_job.get("job_dir")
    if not isinstance(raw_job_dir, str) or not raw_job_dir.strip():
        raise RuntimeError("Active multistart metadata has no valid job_dir")

    candidate = Path(raw_job_dir).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    candidate = candidate.resolve()
    multistart_root = (repo_root / "station_multistart").resolve()
    if candidate == multistart_root or not candidate.is_relative_to(multistart_root):
        raise RuntimeError(f"Multistart job_dir is outside this checkout: {candidate}")
    if not candidate.is_dir() or not (candidate / "state.yaml").is_file():
        raise RuntimeError(f"Active multistart job directory is unavailable: {candidate}")
    return candidate


def resolve_data_roots(repo_root: Path) -> list[Path]:
    """Return live roots, including every branch that may win an active multistart."""
    repo_root = repo_root.resolve()
    current_job_path = repo_root / "station_multistart" / "current_job.yaml"
    normal_root = repo_root / constants.BASE_STATION_DATA_PATH
    roots: list[Path] = []

    if normal_root.is_dir() and (normal_root / constants.AGENTS_DIR_NAME).is_dir():
        roots.append(normal_root.resolve())

    if current_job_path.exists():
        current_job = _load_mapping(current_job_path)
        if not current_job:
            raise RuntimeError(f"Could not read active multistart metadata: {current_job_path}")
        job_dir = _validated_job_dir(repo_root, current_job)
        candidates = [job_dir / "origin_station_data"]
        candidates.extend(
            path
            for path in sorted(job_dir.iterdir())
            if path.is_dir() and BRANCH_DATA_ROOT_PATTERN.fullmatch(path.name)
        )
        for candidate in candidates:
            if candidate.is_dir() and (candidate / constants.AGENTS_DIR_NAME).is_dir():
                roots.append(candidate.resolve())

    roots = list(dict.fromkeys(roots))
    if not roots:
        raise FileNotFoundError(
            "No live station agent data found in station_data or the active multistart job"
        )
    return roots


def find_updates(data_roots: list[Path], old_model: str) -> list[ModelUpdate]:
    updates: list[ModelUpdate] = []
    for data_root in data_roots:
        agents_dir = data_root / constants.AGENTS_DIR_NAME
        for filename in file_io_utils.list_files(str(agents_dir), constants.YAML_EXTENSION):
            agent_path = agents_dir / filename
            agent_data = file_io_utils.load_yaml(str(agent_path))
            if not isinstance(agent_data, dict) or not agent_is_active(agent_data):
                continue
            if agent_data.get(constants.AGENT_MODEL_NAME_KEY) != old_model:
                continue
            updates.append(
                ModelUpdate(
                    data_root=data_root,
                    agent_path=agent_path,
                    agent_name=str(agent_data[constants.AGENT_NAME_KEY]),
                )
            )
    return sorted(updates, key=lambda item: (str(item.data_root), item.agent_name))


def apply_updates(updates: list[ModelUpdate], old_model: str, new_model: str) -> list[ModelUpdate]:
    applied: list[ModelUpdate] = []
    for update in updates:
        # Recheck after shutdown so an agent that ended during draining is never changed.
        agent_data = file_io_utils.load_yaml(str(update.agent_path))
        if not isinstance(agent_data, dict) or not agent_is_active(agent_data):
            continue
        if agent_data.get(constants.AGENT_MODEL_NAME_KEY) != old_model:
            continue
        agent_data[constants.AGENT_MODEL_NAME_KEY] = new_model
        file_io_utils.save_yaml(agent_data, str(update.agent_path), sort_keys=False)
        applied.append(update)
    return applied


def _print_updates(prefix: str, updates: list[ModelUpdate], old_model: str, new_model: str) -> None:
    print(f"{prefix} {len(updates)} active agent record(s):")
    for update in updates:
        try:
            root_label = update.data_root.relative_to(REPO_ROOT)
        except ValueError:
            root_label = update.data_root
        print(f"- {root_label}: {update.agent_name}: {old_model} -> {new_model}")


def _run_lifecycle_script(repo_root: Path, script_name: str, args: list[str]) -> bool:
    command = [str(repo_root / script_name), *args]
    result = subprocess.run(command, cwd=repo_root, check=False)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gracefully stop Station, replace an exact model name for active agents only, "
            "then start Station again. Active multistart origins and branches are included."
        )
    )
    parser.add_argument("old_model", help="Exact current model name")
    parser.add_argument("new_model", help="Replacement model name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show matching active agents without stopping or writing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to stop.sh instead of waiting for a graceful drain",
    )
    args = parser.parse_args()

    old_model = args.old_model.strip()
    new_model = args.new_model.strip()
    if not old_model or not new_model:
        parser.error("model names must not be blank")
    if old_model == new_model:
        parser.error("old_model and new_model must differ")

    try:
        initial_roots = resolve_data_roots(REPO_ROOT)
        initial_updates = find_updates(initial_roots, old_model)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        _print_updates("Would update", initial_updates, old_model, new_model)
        return 0
    if not initial_updates:
        print(f"Found 0 active agents using {old_model!r}; Station was not restarted.")
        return 0

    stop_args = ["--force"] if args.force else []
    print("Stopping Station before changing agent model configuration...")
    if not _run_lifecycle_script(REPO_ROOT, "stop.sh", stop_args):
        print("Error: stop.sh failed; no model records were changed.", file=sys.stderr)
        return 1

    try:
        stopped_roots = resolve_data_roots(REPO_ROOT)
        stopped_updates = find_updates(stopped_roots, old_model)
        applied = apply_updates(stopped_updates, old_model, new_model)
    except Exception as exc:
        print(f"Error while updating stopped Station data: {exc}", file=sys.stderr)
        print("Station remains stopped. After resolving the error, run: ./start.sh -s", file=sys.stderr)
        return 1

    _print_updates("Updated", applied, old_model, new_model)
    print("Starting Station with the updated model configuration...")
    if not _run_lifecycle_script(REPO_ROOT, "start.sh", ["-s"]):
        print("Error: model records were updated, but start.sh failed.", file=sys.stderr)
        print("Run ./start.sh -s after resolving the startup error.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
