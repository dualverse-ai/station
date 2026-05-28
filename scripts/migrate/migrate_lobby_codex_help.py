#!/usr/bin/env python3
"""Inject the Station Codex into legacy first-turn Lobby help for active agents."""

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
    from station import constants
    from station import file_io_utils


CODEX_MARKER = "This is the Codex that underly the Research Station's philosophy:"
LOBBY_HELP_MARKERS = ("Help Message - Lobby", "Help for Lobby:")
FIRST_MISSION_MARKER = "### Your First Mission"
LEGACY_LEARN_RULES = (
    "1. **Learn the Rules:** Read your system prompt carefully. "
    "It includes the Station Codex and core operating principles."
)
UPDATED_LEARN_RULES = (
    "1. **Learn the Rules:** Read this Lobby help message and your system prompt carefully. "
    "The Station Codex is shown above, and your system prompt defines your operating role."
)


def _safe_agent_log_name(agent_name: str) -> str:
    return "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in agent_name)


def _coerce_tick(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_active_agent(agent_data: Any) -> bool:
    if not isinstance(agent_data, dict):
        return False
    if agent_data.get(constants.AGENT_SESSION_ENDED_KEY):
        return False
    if agent_data.get(constants.AGENT_IS_ASCENDED_KEY):
        return False
    return bool(agent_data.get(constants.AGENT_NAME_KEY))


def _entry_is_station_side(entry: Dict[str, Any]) -> bool:
    role = entry.get("role")
    if role is not None:
        return role == "user"
    speaker = entry.get("speaker")
    if speaker is not None:
        return speaker == "Station"
    return False


def _entry_tick(entry: Dict[str, Any]) -> Optional[int]:
    return _coerce_tick(entry.get("tick"))


def _first_tick(entries: List[Dict[str, Any]]) -> Optional[int]:
    ticks = [
        tick
        for tick in (_entry_tick(entry) for entry in entries if isinstance(entry, dict))
        if tick is not None
    ]
    return min(ticks) if ticks else None


def _load_codex_text(station_data_path: Path) -> str:
    codex_text = file_io_utils.load_text(str(station_data_path / constants.CODEX_FILENAME))
    if not isinstance(codex_text, str):
        return ""
    return codex_text.strip()


def _codex_block(codex_text: str) -> str:
    return f"{CODEX_MARKER}\n\n{codex_text}".strip()


def _looks_like_legacy_lobby_help(text: str) -> bool:
    if CODEX_MARKER in text:
        return False
    if FIRST_MISSION_MARKER not in text:
        return False
    return any(marker in text for marker in LOBBY_HELP_MARKERS)


def _inject_codex_into_text(text: str, codex_text: str) -> Tuple[str, bool]:
    if not isinstance(text, str) or not _looks_like_legacy_lobby_help(text):
        return text, False

    updated_text = text.replace(LEGACY_LEARN_RULES, UPDATED_LEARN_RULES)
    codex_block = _codex_block(codex_text)
    mission_index = updated_text.find(FIRST_MISSION_MARKER)
    separator_index = updated_text.rfind("------", 0, mission_index)

    if separator_index >= 0:
        before_separator = updated_text[:separator_index].rstrip()
        after_separator = updated_text[separator_index:].lstrip()
        updated_text = f"{before_separator}\n\n{codex_block}\n\n{after_separator}"
    else:
        before_mission = updated_text[:mission_index].rstrip()
        after_mission = updated_text[mission_index:].lstrip()
        updated_text = f"{before_mission}\n\n{codex_block}\n\n{after_mission}"

    return updated_text, updated_text != text


def _update_nested_text(value: Any, codex_text: str) -> Tuple[Any, bool]:
    if isinstance(value, str):
        return _inject_codex_into_text(value, codex_text)
    if isinstance(value, list):
        changed = False
        updated_items: List[Any] = []
        for item in value:
            updated_item, item_changed = _update_nested_text(item, codex_text)
            updated_items.append(updated_item)
            changed = changed or item_changed
        return updated_items, changed
    if isinstance(value, dict):
        changed = False
        updated_dict: Dict[Any, Any] = {}
        for key, item in value.items():
            updated_item, item_changed = _update_nested_text(item, codex_text)
            updated_dict[key] = updated_item
            changed = changed or item_changed
        return updated_dict, changed
    return value, False


def _update_entries(entries: List[Dict[str, Any]], codex_text: str) -> Tuple[List[Dict[str, Any]], int]:
    first_tick = _first_tick(entries)
    if first_tick is None:
        return entries, 0

    updated_entries: List[Dict[str, Any]] = []
    changed_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            updated_entries.append(entry)
            continue
        if _entry_tick(entry) != first_tick or not _entry_is_station_side(entry):
            updated_entries.append(entry)
            continue

        updated_entry, changed = _update_nested_text(entry, codex_text)
        updated_entries.append(updated_entry)
        if changed:
            changed_count += 1

    return updated_entries, changed_count


def _dump_yaml_documents(entries: List[Dict[str, Any]]) -> str:
    docs = [
        yaml.safe_dump(
            entry,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=1000,
        ).rstrip()
        for entry in entries
    ]
    if not docs:
        return ""
    return "\n---\n".join(docs) + "\n"


def _rewrite_yaml_lines(path: Path, entries: List[Dict[str, Any]]) -> None:
    file_io_utils.save_text(_dump_yaml_documents(entries), str(path))


def _surface_paths(station_data_path: Path, agent_name: str) -> List[Path]:
    safe_name = _safe_agent_log_name(agent_name)
    return [
        station_data_path
        / constants.DIALOGUE_LOGS_DIR_NAME
        / f"{safe_name}{constants.DIALOGUE_LOG_FILENAME_SUFFIX}",
        station_data_path
        / constants.AGENTS_DIR_NAME
        / agent_name
        / "llm_chat_history.yamll",
    ]


def _active_agent_names(station_data_path: Path) -> List[str]:
    agents_dir = station_data_path / constants.AGENTS_DIR_NAME
    if not agents_dir.is_dir():
        return []

    names: List[str] = []
    for agent_file in sorted(agents_dir.glob(f"*{constants.YAML_EXTENSION}")):
        agent_data = file_io_utils.load_yaml(str(agent_file))
        if not _is_active_agent(agent_data):
            continue
        names.append(str(agent_data[constants.AGENT_NAME_KEY]))
    return names


def migrate_station_data(station_data_path: Path, *, apply: bool) -> Dict[str, Any]:
    codex_text = _load_codex_text(station_data_path)
    summary: Dict[str, Any] = {
        "active_agents": 0,
        "candidate_files": 0,
        "changed_files": 0,
        "changed_entries": 0,
        "codex_missing": not bool(codex_text),
    }
    if not codex_text:
        return summary

    for agent_name in _active_agent_names(station_data_path):
        summary["active_agents"] += 1
        for path in _surface_paths(station_data_path, agent_name):
            if not path.exists():
                continue
            entries = file_io_utils.load_yaml_lines(str(path))
            if not entries:
                continue
            updated_entries, changed_entries = _update_entries(entries, codex_text)
            if changed_entries <= 0:
                continue
            summary["candidate_files"] += 1
            summary["changed_entries"] += changed_entries
            if apply:
                _rewrite_yaml_lines(path, updated_entries)
                summary["changed_files"] += 1
    return summary


def needs_migration(station_data_path: Path) -> bool:
    summary = migrate_station_data(station_data_path, apply=False)
    return int(summary.get("changed_entries", 0)) > 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inject Codex text into active agents' legacy first-turn Lobby help/history."
    )
    parser.add_argument("--data-root", default=constants.BASE_STATION_DATA_PATH)
    parser.add_argument("--check", action="store_true", help="Print yes if migration is needed.")
    args = parser.parse_args()

    station_data_path = Path(args.data_root)
    if args.check:
        print("yes" if needs_migration(station_data_path) else "no")
        return 0

    summary = migrate_station_data(station_data_path, apply=True)
    if summary.get("codex_missing"):
        print(f"Lobby Codex migration skipped: {station_data_path / constants.CODEX_FILENAME} is missing or empty.")
    else:
        print(
            "Lobby Codex migration complete: "
            f"active_agents={summary['active_agents']} "
            f"changed_files={summary['changed_files']} "
            f"changed_entries={summary['changed_entries']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
