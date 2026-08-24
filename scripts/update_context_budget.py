#!/usr/bin/env python3
"""Update context compaction budgets for active agent YAML files."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

with contextlib.redirect_stdout(io.StringIO()):
    from station import constants, file_io_utils  # noqa: E402


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("context budget must be positive")
    return parsed


def _default_station_data_path() -> Path:
    return (REPO_ROOT / constants.BASE_STATION_DATA_PATH).resolve()


def _agent_is_active(agent_data: Dict[str, Any]) -> bool:
    if agent_data.get(constants.AGENT_STATUS_KEY) not in {
        constants.AGENT_STATUS_GUEST,
        constants.AGENT_STATUS_RECURSIVE,
    }:
        return False
    if not agent_data.get(constants.AGENT_NAME_KEY):
        return False
    return not bool(agent_data.get(constants.AGENT_SESSION_ENDED_KEY)) and not bool(
        agent_data.get(constants.AGENT_IS_ASCENDED_KEY)
    )


def update_active_agent_context_budgets(
    station_data_path: Path,
    context_budget: int,
    dry_run: bool = False,
) -> List[Tuple[str, Any, int]]:
    agents_dir = station_data_path / constants.AGENTS_DIR_NAME
    if not file_io_utils.dir_exists(str(agents_dir)):
        raise FileNotFoundError(f"Agents directory not found: {agents_dir}")

    updates: List[Tuple[str, Any, int]] = []
    for filename in file_io_utils.list_files(str(agents_dir), constants.YAML_EXTENSION):
        agent_path = agents_dir / filename
        agent_data = file_io_utils.load_yaml(str(agent_path))
        if not isinstance(agent_data, dict) or not _agent_is_active(agent_data):
            continue

        old_budget = agent_data.get(constants.AGENT_TOKEN_BUDGET_MAX_KEY)
        if old_budget == context_budget:
            continue

        agent_data[constants.AGENT_TOKEN_BUDGET_MAX_KEY] = context_budget
        updates.append((filename[: -len(constants.YAML_EXTENSION)], old_budget, context_budget))
        if not dry_run:
            file_io_utils.save_yaml(agent_data, str(agent_path), sort_keys=False)

    updates.sort(key=lambda item: item[0])
    return updates


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Update the context compaction budget for all active agent YAML files. "
            "The value is stored in the existing agent context-budget field and "
            "takes effect after restart."
        )
    )
    parser.add_argument("context_budget", type=_parse_positive_int, help="New context budget, e.g. 300000")
    parser.add_argument(
        "--station-data",
        default=str(_default_station_data_path()),
        help="Path to station_data (default: repo-local station_data)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print affected agents without writing YAML")
    args = parser.parse_args()

    station_data_path = Path(args.station_data).expanduser().resolve()
    try:
        updates = update_active_agent_context_budgets(
            station_data_path,
            args.context_budget,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    action = "Would update" if args.dry_run else "Updated"
    if not updates:
        print(f"{action} 0 active agents.")
        return 0

    print(f"{action} {len(updates)} active agents in {station_data_path}:")
    for agent_name, old_budget, new_budget in updates:
        print(f"- {agent_name}: {old_budget} -> {new_budget}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
