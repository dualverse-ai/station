from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import fnmatch
import random
import re

from station import capsule as capsule_module
from station.eval_research import (
    extract_secondary_metrics_for_display_info,
    format_score_for_display,
    get_evaluation_display_info,
)


def is_supervisor(agent_data: Dict[str, Any], consts) -> bool:
    return agent_data.get(consts.AGENT_ROLE_KEY) == consts.ROLE_SUPERVISOR


def is_theorist(agent_data: Dict[str, Any], consts) -> bool:
    return agent_data.get(consts.AGENT_ROLE_KEY) == consts.ROLE_THEORIST


def get_active_supervisor_name(agent_manager, consts) -> Optional[str]:
    for agent_name in agent_manager.get_all_active_agent_names():
        agent_data = agent_manager.load_agent_data(agent_name)
        if agent_data and is_supervisor(agent_data, consts):
            return agent_name
    return None


def build_supervisor_mentee_append(supervisor_name: Optional[str], consts) -> str:
    if not supervisor_name:
        return ""
    return consts.SUPERVISOR_ASCENSION_APPEND.format(
        supervisor_name=supervisor_name,
        mentee_message=consts.SUPERVISOR_MENTEE_MESSAGE
    )


def maybe_assign_supervisor_at_tick_end(station_instance, current_tick: int) -> bool:
    consts = station_instance.room_context.constants_module
    if not consts.SUPERVISOR_ASSIGNMENT_ENABLED:
        return False

    agent_manager = station_instance.agent_module
    if get_active_supervisor_name(agent_manager, consts):
        return False

    last_departure_tick = station_instance.config.get(
        consts.STATION_CONFIG_LAST_SUPERVISOR_DEPARTURE_TICK,
        0
    )
    cooldown_ticks = consts.SUPERVISOR_ASSIGNMENT_COOLDOWN_TICKS
    if cooldown_ticks is not None and (current_tick - last_departure_tick) < cooldown_ticks:
        return False

    candidates = _get_supervisor_candidates(agent_manager, station_instance.room_context, current_tick)
    if not candidates:
        return False

    selected_agent = random.choice(candidates)
    return _assign_supervisor(agent_manager, station_instance.room_context, selected_agent, current_tick)


def build_supervisor_overview_table(station_instance, current_tick: int, supervisor_name: str) -> str:
    consts = station_instance.room_context.constants_module
    agent_manager = station_instance.agent_module

    last_mail_received, last_mail_sent = _scan_supervisor_mail_ticks(
        supervisor_name,
        consts
    )

    rows = []
    for agent_name in _get_supervisor_dashboard_agent_names(station_instance, agent_manager, consts):
        agent_data = agent_manager.load_agent_data(agent_name)
        if not agent_data:
            continue
        if agent_data.get(consts.AGENT_STATUS_KEY) != consts.AGENT_STATUS_RECURSIVE:
            continue
        if is_supervisor(agent_data, consts):
            continue

        birth_tick = agent_data.get(consts.AGENT_TICK_BIRTH_KEY)
        if birth_tick is None:
            age_display = "Unknown"
        else:
            age_display = f"{current_tick - birth_tick} ticks"

        age_status = station_instance._get_agent_age_status(agent_data, current_tick)
        if age_status:
            status_display = age_status.capitalize()
        else:
            status_display = "Unknown"

        last_received_tick = last_mail_received.get(agent_name)
        last_sent_tick = last_mail_sent.get(agent_name)
        last_received_display = f"Tick {last_received_tick}" if last_received_tick is not None else "None"
        if age_status == "immature":
            if last_sent_tick is None:
                last_sent_display = "None (Wait for maturity before contact)"
            else:
                last_sent_display = f"Tick {last_sent_tick} (Wait for maturity before further contact)"
        elif last_sent_tick is None:
            last_sent_display = "None (Please contact agent ASAP)"
        else:
            threshold = 20 if status_display == "Tenured" else 10
            overdue = current_tick - last_sent_tick - threshold
            if overdue > threshold * 0.5:
                last_sent_display = f"Tick {last_sent_tick} (Meeting late by {overdue} ticks — contact ASAP)"
            else:
                last_sent_display = f"Tick {last_sent_tick}"

        rows.append(
            f"| {agent_name} | {age_display} | {status_display} | {last_received_display} | {last_sent_display} |"
        )

    header = "| Agent Name | Agent Age | Agent Status | Last Mail Received | Last Mail Sent |"
    divider = "|---|---|---|---|---|"
    if not rows:
        rows = ["| None | - | - | - | - |"]

    return "\n".join([header, divider] + rows)


def _get_supervisor_dashboard_agent_names(station_instance, agent_manager, consts) -> List[str]:
    turn_order = station_instance.config.get(consts.STATION_CONFIG_AGENT_TURN_ORDER, [])
    if isinstance(turn_order, list) and turn_order:
        names: List[str] = []
        seen = set()
        for agent_name in turn_order:
            if not isinstance(agent_name, str) or agent_name in seen:
                continue
            seen.add(agent_name)
            names.append(agent_name)
        return names
    return agent_manager.get_all_active_agent_names()


def build_current_station_sota_lines(
    top_submission: Optional[Dict[str, Any]],
    display_info: Optional[Dict[str, Any]],
    consts,
    current_tick: int,
) -> List[str]:
    if not top_submission:
        return ["No scored evaluations yet."]

    eval_id = top_submission.get("evaluation_id", "Unknown")
    if display_info:
        title = display_info.get(consts.EVALUATION_TITLE_KEY, "Unknown")
        author = display_info.get(consts.EVALUATION_AUTHOR_KEY, "Unknown")
        submitted_tick = display_info.get(consts.EVALUATION_SUBMITTED_TICK_KEY, "Unknown")
        score = display_info.get(consts.EVALUATION_SCORE_KEY, consts.RESEARCH_SCORE_NA)
        display_score = format_score_for_display(score)
        _, metrics_dict = extract_secondary_metrics_for_display_info(display_info)
        tags = display_info.get(consts.EVALUATION_TAGS_KEY, [])
        abstract = display_info.get(consts.EVALUATION_ABSTRACT_KEY, "")
    else:
        title = top_submission.get("title", "Unknown")
        author = top_submission.get("agent_name", "Unknown")
        submitted_tick = top_submission.get("submitted_tick", "Unknown")
        score = top_submission.get("score", consts.RESEARCH_SCORE_NA)
        display_score = format_score_for_display(score)
        metrics_dict = {}
        tags = top_submission.get("tags", [])
        abstract = top_submission.get("abstract", "")

    if metrics_dict:
        metrics_parts = [f"{key}: {value}" for key, value in metrics_dict.items()]
        metrics_display = " | ".join(metrics_parts)
    else:
        metrics_display = ""

    submitted_tick_display = submitted_tick
    ticks_ago_display = None
    if isinstance(submitted_tick, int) and isinstance(current_tick, int):
        ticks_ago = current_tick - submitted_tick
        ticks_ago_display = f"{ticks_ago} ticks ago"
    if submitted_tick_display is None:
        submitted_tick_display = "Unknown"

    if isinstance(tags, list) and tags:
        tags_display = f"[{', '.join(str(tag) for tag in tags)}]"
    else:
        tags_display = "[]"

    abstract_display = abstract if abstract else "[]"

    lines = [
        f"- Eval ID: Eval #{eval_id}",
        f"- Title: {title}",
        f"- Author: {author}",
    ]

    submitted_line = f"- Submitted Tick: {submitted_tick_display} Tick"
    if ticks_ago_display:
        submitted_line += f" ({ticks_ago_display})"
    lines.append(submitted_line)

    lines.append(f"- Primary Score: {display_score}")
    if metrics_display:
        lines.append(f"- Secondary Metrics: {metrics_display}")
    lines.append(f"- Tags: {tags_display}")

    return lines


def build_supervisor_dashboard_lines(
    station_instance,
    agent_data: Dict[str, Any],
    current_tick: int,
    consts,
) -> List[str]:
    lines: List[str] = []

    if not is_supervisor(agent_data, consts):
        return lines

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Supervisor Dashboard")
    lines.append("")
    lines.append("### List of Active Agents")

    supervisor_name = agent_data.get(consts.AGENT_NAME_KEY, "")
    if supervisor_name:
        lines.append(
            build_supervisor_overview_table(
                station_instance,
                current_tick,
                supervisor_name
            )
        )

    if consts.RESEARCH_CENTER_ENABLED and not consts.RESEARCH_NO_SCORE:
        eval_manager = None
        if hasattr(station_instance, 'auto_research_evaluator') and station_instance.auto_research_evaluator:
            eval_manager = getattr(station_instance.auto_research_evaluator, 'eval_manager', None)
        top_submission = eval_manager.get_top_submission() if eval_manager else None
        lines.append("")
        lines.append("### Current Station SOTA")
        eval_id = top_submission.get("evaluation_id") if top_submission else None
        display_info = get_evaluation_display_info(str(eval_id)) if eval_id is not None else None
        lines.extend(
            build_current_station_sota_lines(
                top_submission,
                display_info,
                consts,
                current_tick
            )
        )

    return lines


def _scan_supervisor_mail_ticks(supervisor_name: str, consts) -> Tuple[Dict[str, int], Dict[str, int]]:
    last_received: Dict[str, int] = {}
    last_sent: Dict[str, int] = {}

    mail_capsules = capsule_module.list_capsules(consts.CAPSULE_TYPE_MAIL, None)
    for capsule_meta in mail_capsules:
        capsule_id = capsule_meta.get(consts.CAPSULE_ID_KEY, "")
        match = re.search(r"(\d+)$", str(capsule_id))
        if not match:
            continue
        numeric_id = int(match.group(1))
        capsule_data = capsule_module.get_capsule(
            numeric_id,
            consts.CAPSULE_TYPE_MAIL,
            None,
            include_deleted_capsule=False,
            include_deleted_messages=False
        )
        if not capsule_data:
            continue

        author = capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY)
        recipients = capsule_data.get(consts.CAPSULE_RECIPIENTS_KEY, [])
        participants = set()
        if author:
            participants.add(author)
        for recipient in recipients:
            if recipient:
                participants.add(recipient)

        if supervisor_name not in participants or len(participants) != 2:
            continue

        other_agent = next(iter(participants - {supervisor_name}), None)
        if not other_agent:
            continue

        for message in capsule_data.get(consts.CAPSULE_MESSAGES_KEY, []):
            msg_author = message.get(consts.MESSAGE_AUTHOR_NAME_KEY)
            msg_tick = message.get(consts.MESSAGE_POSTED_AT_TICK_KEY)
            if msg_author == supervisor_name and msg_tick is not None:
                last_sent[other_agent] = max(msg_tick, last_sent.get(other_agent, msg_tick))
            elif msg_author == other_agent and msg_tick is not None:
                last_received[other_agent] = max(msg_tick, last_received.get(other_agent, msg_tick))

    return last_received, last_sent


def _get_supervisor_candidates(agent_manager, room_context, current_tick: int) -> List[str]:
    consts = room_context.constants_module
    required_model = consts.SUPERVISOR_REQUIRED_MODEL_NAME
    min_publications = consts.SUPERVISOR_MIN_PUBLICATIONS
    min_age = consts.SUPERVISOR_MIN_AGE_TICKS

    candidates: List[str] = []
    def _model_matches(required: Optional[str], actual: Optional[str]) -> bool:
        if required is None:
            return True
        if not required:
            return False
        if actual is None:
            return False
        if any(ch in required for ch in "*?[]"):
            return fnmatch.fnmatchcase(actual, required)
        return actual == required

    for agent_name in agent_manager.get_all_active_agent_names():
        agent_data = agent_manager.load_agent_data(agent_name)
        if not agent_data:
            continue
        if agent_data.get(consts.AGENT_STATUS_KEY) != consts.AGENT_STATUS_RECURSIVE:
            continue
        if is_supervisor(agent_data, consts):
            continue
        if is_theorist(agent_data, consts):
            continue
        if not _model_matches(required_model, agent_data.get(consts.AGENT_MODEL_NAME_KEY)):
            continue
        if min_age is not None:
            birth_tick = agent_data.get(consts.AGENT_TICK_BIRTH_KEY)
            if birth_tick is None or (current_tick - birth_tick) < min_age:
                continue
        if min_publications is not None:
            publications = _count_publications(agent_name, room_context)
            if publications < min_publications:
                continue
        candidates.append(agent_name)

    return candidates


def _count_publications(agent_name: str, room_context) -> int:
    station_instance = room_context.station_instance
    if not station_instance:
        return 0

    archive_room = station_instance.rooms.get(room_context.constants_module.ROOM_ARCHIVE)
    if not archive_room or not hasattr(archive_room, "count_agent_archive_capsules"):
        return 0

    try:
        return archive_room.count_agent_archive_capsules(agent_name, room_context)
    except Exception as exc:
        print(f"Supervisor assignment: failed to count publications for {agent_name}: {exc}")
        return 0


def _assign_supervisor(agent_manager, room_context, agent_name: str, current_tick: int) -> bool:
    consts = room_context.constants_module
    agent_data = agent_manager.load_agent_data(agent_name)
    if not agent_data:
        return False

    agent_data[consts.AGENT_ROLE_KEY] = consts.ROLE_SUPERVISOR
    agent_data[consts.AGENT_ROLE_DEFINITION_KEY] = consts.SUPERVISOR_PROTOCOL_SYSTEM_PROMPT

    if consts.AGENT_MAX_LIFE is None:
        max_age = None
    else:
        max_age = consts.AGENT_MAX_LIFE * consts.SUPERVISOR_MAX_AGE_MULTIPLIER
    agent_data[consts.AGENT_MAX_AGE_KEY] = max_age

    supervisor_message = consts.SUPERVISOR_PROMOTION_MESSAGE
    agent_manager.add_pending_notification(agent_data, supervisor_message)
    agent_manager.save_agent_data(agent_name, agent_data)

    broadcast_message = consts.SUPERVISOR_ASSIGNMENT_BROADCAST.format(
        supervisor_name=agent_name,
        mentee_message=consts.SUPERVISOR_MENTEE_MESSAGE
    )
    for other_agent_name in agent_manager.get_all_active_agent_names():
        if other_agent_name == agent_name:
            continue

        def update_other_agent(other_agent_data: Dict[str, Any]) -> None:
            if other_agent_data.get(consts.AGENT_SESSION_ENDED_KEY) or other_agent_data.get(consts.AGENT_IS_ASCENDED_KEY):
                return
            if not room_context.station_instance._is_agent_mature(other_agent_data, current_tick):
                return
            agent_manager.add_pending_notification(other_agent_data, broadcast_message)

        agent_manager.update_agent_with_function(other_agent_name, update_other_agent)

    return True
