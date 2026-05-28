"""Fast dashboard summaries for agent YAML records.

The agent files remain the source of truth.  This module deliberately reads
only top-level scalar fields needed by the dashboard agent list so `/api/agents`
does not deserialize long histories and notification bodies for every agent.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set

from station import constants, file_io_utils


_AGENT_SUMMARY_KEYS: Set[str] = {
    constants.AGENT_STATUS_KEY,
    constants.AGENT_IS_ASCENDED_KEY,
    constants.AGENT_ASCENDED_TO_NAME_KEY,
    constants.AGENT_SESSION_ENDED_KEY,
    constants.AGENT_MODEL_NAME_KEY,
    constants.AGENT_MODEL_PROVIDER_CLASS_KEY,
    constants.AGENT_SESSION_END_REQUESTED_KEY,
    constants.AGENT_ROLE_KEY,
}

_TEMPORAL_CHAT_KEYS: Set[str] = {"base_tick", "updated_at"}
_HUMAN_INTERVENTION_KEYS: Set[str] = {
    constants.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG,
    constants.AGENT_HUMAN_INTERACTION_ID_KEY,
    constants.AGENT_HUMAN_INTERACTION_IDS_KEY,
}
_HUMAN_INTERVENTION_LIST_KEYS: Set[str] = {
    constants.AGENT_HUMAN_INTERACTION_IDS_KEY,
}


def get_all_agents_summary(*, base_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return dashboard-ready summaries without full YAML deserialization."""
    root = base_path or constants.BASE_STATION_DATA_PATH
    agents_dir = os.path.join(root, constants.AGENTS_DIR_NAME)
    if not file_io_utils.dir_exists(agents_dir):
        file_io_utils.ensure_dir_exists(agents_dir)
        return []

    agents_info: List[Dict[str, Any]] = []
    agent_files = file_io_utils.list_files(agents_dir, constants.YAML_EXTENSION)
    for agent_file_name in agent_files:
        agent_name = agent_file_name.removesuffix(constants.YAML_EXTENSION)
        if agent_name == "AutoArchiveEvaluator":
            continue

        agent_path = os.path.join(agents_dir, agent_file_name)
        agent_data = read_top_level_scalars(agent_path, _AGENT_SUMMARY_KEYS)
        status = _display_status(agent_data)

        agent_info: Dict[str, Any] = {
            "name": agent_name,
            "status": status,
            "model_name": agent_data.get(constants.AGENT_MODEL_NAME_KEY, ""),
            "model_provider_class": agent_data.get(constants.AGENT_MODEL_PROVIDER_CLASS_KEY, ""),
            "session_end_requested": bool(agent_data.get(constants.AGENT_SESSION_END_REQUESTED_KEY, False)),
            "is_supervisor": agent_data.get(constants.AGENT_ROLE_KEY) == constants.ROLE_SUPERVISOR,
        }

        temporal_chat_path = os.path.join(
            root,
            constants.TEMPORAL_CHAT_DIR_NAME,
            f"{agent_name}{constants.YAML_EXTENSION}",
        )
        temporal_chat_record = read_top_level_scalars(temporal_chat_path, _TEMPORAL_CHAT_KEYS)
        if temporal_chat_record:
            agent_info["temporal_chat_exists"] = True
            agent_info["temporal_chat_base_tick"] = temporal_chat_record.get("base_tick")
            agent_info["temporal_chat_updated_at"] = temporal_chat_record.get("updated_at")
        else:
            agent_info["temporal_chat_exists"] = False

        agents_info.append(agent_info)
    return agents_info


def read_top_level_scalars(path: str, keys: Set[str]) -> Dict[str, Any]:
    """Read selected top-level scalar YAML fields from `path`.

    This is intentionally narrow: nested values, block scalars, and sequences are
    ignored because the dashboard list does not need them.
    """
    return read_top_level_fields(path, keys, list_keys=set())


def get_agent_human_intervention_fields(agent_name: str, *, base_path: Optional[str] = None) -> Dict[str, Any]:
    """Return only human-intervention fields from one agent YAML file."""
    root = base_path or constants.BASE_STATION_DATA_PATH
    agent_path = os.path.join(
        root,
        constants.AGENTS_DIR_NAME,
        f"{agent_name}{constants.YAML_EXTENSION}",
    )
    return read_top_level_fields(
        agent_path,
        _HUMAN_INTERVENTION_KEYS,
        list_keys=_HUMAN_INTERVENTION_LIST_KEYS,
    )


def agent_has_human_intervention_request(fields: Dict[str, Any]) -> bool:
    """Return True when lightweight agent fields indicate a pending request."""
    if fields.get(constants.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG, False):
        return True
    request_ids = fields.get(constants.AGENT_HUMAN_INTERACTION_IDS_KEY)
    if isinstance(request_ids, list) and request_ids:
        return True
    return bool(fields.get(constants.AGENT_HUMAN_INTERACTION_ID_KEY))


def read_top_level_fields(path: str, keys: Set[str], *, list_keys: Set[str]) -> Dict[str, Any]:
    """Read selected top-level YAML scalar fields and simple block lists."""
    if not os.path.exists(path):
        return {}

    values: Dict[str, Any] = {}
    active_list_key: Optional[str] = None
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped_line = raw_line.lstrip()
            if active_list_key and stripped_line.startswith("- "):
                values.setdefault(active_list_key, []).append(_parse_scalar(stripped_line[2:].strip()))
                continue
            if active_list_key:
                active_list_key = None
                if len(values) == len(keys):
                    break

            if not raw_line or raw_line[0].isspace() or ":" not in raw_line:
                if raw_line and raw_line.strip():
                    active_list_key = None
                continue
            key, raw_value = raw_line.split(":", 1)
            key = key.strip()
            if key not in keys:
                continue
            stripped_value = raw_value.strip()
            if key in list_keys and stripped_value == "":
                values[key] = []
                active_list_key = key
            else:
                values[key] = _parse_scalar(stripped_value)
            if len(values) == len(keys) and active_list_key is None:
                break
    return values


def _display_status(agent_data: Dict[str, Any]) -> str:
    status = agent_data.get(constants.AGENT_STATUS_KEY, "Unknown")
    if agent_data.get(constants.AGENT_IS_ASCENDED_KEY, False):
        ascended_to = agent_data.get(constants.AGENT_ASCENDED_TO_NAME_KEY) or "N/A"
        return f"Ascended (to {ascended_to})"
    if agent_data.get(constants.AGENT_SESSION_ENDED_KEY, False):
        return "Session Ended"
    if status is None:
        return "Unknown"
    return str(status)


def _parse_scalar(value: str) -> Any:
    if value == "[]":
        return []
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
