#!/usr/bin/env python3
"""Clean stale duplicate dialogue protection records from station data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from station import constants
from station import file_io_utils


def _record_reason(record: Dict[str, Any]) -> Optional[str]:
    return record.get(constants.PROTECTED_DIALOGUE_REASON_KEY)


def _record_tick(record: Dict[str, Any]) -> Optional[int]:
    tick = record.get(constants.PROTECTED_DIALOGUE_TICK_KEY)
    try:
        return int(tick)
    except (TypeError, ValueError):
        return None


def clean_agent_data(
    agent_data: Dict[str, Any],
    *,
    real_architect_ticks: Optional[Iterable[int]] = None,
) -> Tuple[bool, int, int]:
    """Keep the first research-task-read protection and real architect messages."""

    records = list(agent_data.get(constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY) or [])
    if not records:
        return False, 0, 0

    allowed_architect_ticks: Optional[Set[int]] = (
        {int(tick) for tick in real_architect_ticks}
        if real_architect_ticks is not None
        else None
    )

    cleaned = []
    seen_research_task_read = False
    removed_research = 0
    removed_architect = 0

    for record in records:
        if not isinstance(record, dict):
            cleaned.append(record)
            continue

        reason = _record_reason(record)
        if reason == constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ:
            if seen_research_task_read:
                removed_research += 1
                continue
            seen_research_task_read = True

        if (
            reason == constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE
            and allowed_architect_ticks is not None
            and _record_tick(record) not in allowed_architect_ticks
        ):
            removed_architect += 1
            continue

        cleaned.append(record)

    changed = cleaned != records
    if changed:
        agent_data[constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY] = cleaned
    return changed, removed_research, removed_architect


def clean_pending_protection_records(agent_data: Dict[str, Any]) -> Tuple[bool, int]:
    """Drop stale pending protection records with no backing rendered message."""

    pending = list(agent_data.get(constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY) or [])
    if not pending:
        return False, 0
    agent_data[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY] = []
    return True, len(pending)


def _agent_yaml_paths(station_data_path: Path) -> Iterable[Path]:
    agents_dir = station_data_path / constants.AGENTS_DIR_NAME
    if not agents_dir.exists():
        return []
    return sorted(path for path in agents_dir.glob("*.yaml") if path.is_file())


def _is_inactive_agent(agent_data: Dict[str, Any]) -> bool:
    return bool(
        agent_data.get(constants.AGENT_SESSION_ENDED_KEY)
        or agent_data.get(constants.AGENT_IS_ASCENDED_KEY)
        or agent_data.get(constants.AGENT_SUCCEEDED_BY_KEY)
    )


def scan_station_data(station_data_path: Path | str, *, dry_run: bool = True) -> Dict[str, int]:
    station_data = Path(station_data_path)
    summary = {
        "agents_seen": 0,
        "agents_skipped_inactive": 0,
        "agents_updated": 0,
        "research_task_read_records_removed": 0,
        "architect_message_records_removed": 0,
        "pending_records_removed": 0,
    }

    for agent_path in _agent_yaml_paths(station_data):
        agent_data = file_io_utils.load_yaml(str(agent_path))
        if not isinstance(agent_data, dict):
            continue

        summary["agents_seen"] += 1
        if _is_inactive_agent(agent_data):
            summary["agents_skipped_inactive"] += 1
            continue

        changed, removed_research, removed_architect = clean_agent_data(agent_data)
        changed_pending, removed_pending = clean_pending_protection_records(agent_data)
        if changed or changed_pending:
            summary["agents_updated"] += 1
            summary["research_task_read_records_removed"] += removed_research
            summary["architect_message_records_removed"] += removed_architect
            summary["pending_records_removed"] += removed_pending
            if not dry_run:
                file_io_utils.save_yaml(agent_data, str(agent_path))

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("station_data", nargs="?", default=constants.BASE_STATION_DATA_PATH)
    parser.add_argument("--apply", action="store_true", help="Write cleaned agent YAML files.")
    args = parser.parse_args()

    summary = scan_station_data(args.station_data, dry_run=not args.apply)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
