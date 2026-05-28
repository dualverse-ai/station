#!/usr/bin/env python3
"""Migrate legacy prune-protection keyword matches into agent YAML records."""

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
    from station import constants
    from station import file_io_utils


RESEARCH_TASK_CREDIBILITY_MARKER = (
    "This specification holds the highest degree of credibility in this research station "
    "and overrides all other sources."
)

LEGACY_KEYWORD_MARKERS = (
    (
        "**Architect Message**",
        constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
        "legacy_architect_message",
    ),
    (
        "You have successfully ascended to a **Recursive Agent**.  You gained full privileges to access station resources.",
        "legacy_keyword",
        "legacy_ascension_success",
    ),
    (
        "**Congratulations!** You have reached a mature age.**",
        "legacy_keyword",
        "legacy_maturity_notice",
    ),
)

TEXT_KEYS = (
    "content",
    "text_content",
    "next_prompt",
    "completion_message",
    "error",
    "message",
    "actions_executed_summary",
    "parsed_actions_detail",
    "internal_action_initiated",
)


def _coerce_tick(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_agent_log_name(agent_name: str) -> str:
    return "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in agent_name)


def _entry_is_station_side(entry: Dict[str, Any]) -> bool:
    role = entry.get("role")
    if role is not None:
        return role == "user"
    speaker = entry.get("speaker")
    if speaker is not None:
        return speaker == "Station"
    return False


def _text_fragments(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        fragments: List[str] = []
        for item in value:
            fragments.extend(_text_fragments(item))
        return fragments
    if isinstance(value, dict):
        fragments: List[str] = []
        for item in value.values():
            fragments.extend(_text_fragments(item))
        return fragments
    return []


def _entry_text(entry: Dict[str, Any]) -> str:
    fragments: List[str] = []
    for key in TEXT_KEYS:
        if key in entry:
            fragments.extend(_text_fragments(entry.get(key)))
    return "\n".join(fragments)


def _load_agent_dialogue_entries(station_data_path: Path, agent_name: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    safe_name = _safe_agent_log_name(agent_name)

    dialogue_log_path = (
        station_data_path
        / constants.DIALOGUE_LOGS_DIR_NAME
        / f"{safe_name}{constants.DIALOGUE_LOG_FILENAME_SUFFIX}"
    )
    entries.extend(file_io_utils.load_yaml_lines(str(dialogue_log_path)))

    llm_history_path = (
        station_data_path
        / constants.AGENTS_DIR_NAME
        / agent_name
        / "llm_chat_history.yamll"
    )
    entries.extend(file_io_utils.load_yaml_lines(str(llm_history_path)))
    return entries


def _add_record(
    records: List[Dict[str, Any]],
    seen: Set[Tuple[int, str, str]],
    *,
    tick: int,
    reason: str,
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    key = (tick, reason, source)
    if key in seen:
        return
    seen.add(key)
    record: Dict[str, Any] = {
        constants.PROTECTED_DIALOGUE_TICK_KEY: tick,
        constants.PROTECTED_DIALOGUE_REASON_KEY: reason,
    }
    if source:
        record[constants.PROTECTED_DIALOGUE_SOURCE_KEY] = source
    if metadata:
        record[constants.PROTECTED_DIALOGUE_METADATA_KEY] = metadata
    records.append(record)


def detect_legacy_protection_records(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, str, str]] = set()
    sorted_entries = sorted(
        [entry for entry in entries if isinstance(entry, dict)],
        key=lambda entry: (
            _coerce_tick(entry.get("tick")) is None,
            _coerce_tick(entry.get("tick")) or 0,
        ),
    )

    first_help_by_room: Set[str] = set()
    first_research_tick: Optional[int] = None

    for entry in sorted_entries:
        if not _entry_is_station_side(entry):
            continue
        tick = _coerce_tick(entry.get("tick"))
        if tick is None:
            continue
        text = _entry_text(entry)
        if not text:
            continue

        for marker, reason, source in LEGACY_KEYWORD_MARKERS:
            if marker in text:
                _add_record(records, seen, tick=tick, reason=reason, source=source)

        if first_research_tick is None and RESEARCH_TASK_CREDIBILITY_MARKER in text:
            first_research_tick = tick
            _add_record(
                records,
                seen,
                tick=tick,
                reason=constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ,
                source="legacy_research_task_read",
            )

        for room_name, short_room_name in constants.ROOM_NAME_TO_SHORT_MAP.items():
            if short_room_name in first_help_by_room:
                continue
            if f"Help Message - {room_name}" in text or f"Help for {room_name}:" in text:
                first_help_by_room.add(short_room_name)
                _add_record(
                    records,
                    seen,
                    tick=tick,
                    reason=constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP,
                    source=f"room:{short_room_name}",
                    metadata={"room_name": room_name, "migration": "legacy_dialogue_scan"},
                )

    records.sort(
        key=lambda item: (
            int(item.get(constants.PROTECTED_DIALOGUE_TICK_KEY, 0)),
            str(item.get(constants.PROTECTED_DIALOGUE_REASON_KEY, "")),
            str(item.get(constants.PROTECTED_DIALOGUE_SOURCE_KEY, "")),
        )
    )
    return records


def detect_legacy_pending_protection_records(agent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    pending_messages = agent_data.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, [])
    if not isinstance(pending_messages, list):
        return []

    for message in pending_messages:
        if isinstance(message, str) and "**Architect Message**" in message:
            return [
                {
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "legacy_pending_architect_message",
                }
            ]
    return []


def _agent_files(station_data_path: Path) -> List[Path]:
    agents_dir = station_data_path / constants.AGENTS_DIR_NAME
    if not agents_dir.is_dir():
        return []
    return sorted(agents_dir.glob(f"*{constants.YAML_EXTENSION}"))


def needs_migration(station_data_path: Path) -> bool:
    for agent_file in _agent_files(station_data_path):
        agent_data = file_io_utils.load_yaml(str(agent_file))
        if isinstance(agent_data, dict) and constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY not in agent_data:
            return True
    return False


def migrate_station_data(station_data_path: Path) -> Dict[str, int]:
    summary = {
        "agents_seen": 0,
        "agents_updated": 0,
        "agents_already_current": 0,
        "records_added": 0,
        "pending_records_added": 0,
    }

    for agent_file in _agent_files(station_data_path):
        agent_data = file_io_utils.load_yaml(str(agent_file))
        if not isinstance(agent_data, dict):
            continue
        summary["agents_seen"] += 1

        changed = False
        if constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY not in agent_data:
            agent_name = str(agent_data.get(constants.AGENT_NAME_KEY) or agent_file.stem)
            entries = _load_agent_dialogue_entries(station_data_path, agent_name)
            records = detect_legacy_protection_records(entries)
            agent_data[constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY] = records
            summary["records_added"] += len(records)
            pending_records = detect_legacy_pending_protection_records(agent_data)
            if pending_records:
                agent_data[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY] = pending_records
                summary["pending_records_added"] += len(pending_records)
            changed = True

        if constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY not in agent_data:
            agent_data[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY] = []
            changed = True

        if changed:
            file_io_utils.save_yaml(agent_data, str(agent_file))
            summary["agents_updated"] += 1
        else:
            summary["agents_already_current"] += 1

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy keyword-based prune protection into agent YAML."
    )
    parser.add_argument(
        "station_data_path",
        nargs="?",
        default=constants.BASE_STATION_DATA_PATH,
        help="Path to station_data directory.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print yes if any agent is missing protected_dialogue_ticks, otherwise no.",
    )
    args = parser.parse_args()

    station_data_path = Path(args.station_data_path).resolve()
    if args.check:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            migration_required = needs_migration(station_data_path)
        print("yes" if migration_required else "no")
        return 0

    summary = migrate_station_data(station_data_path)
    print(
        "Protected dialogue tick migration complete: "
        f"{summary['agents_updated']} updated, "
        f"{summary['agents_already_current']} already current, "
        f"{summary['records_added']} records added, "
        f"{summary['pending_records_added']} pending records added."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
