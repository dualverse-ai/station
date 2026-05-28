# Copyright 2025 DualverseAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import tiktoken

from station import constants
from station import file_io_utils
from station import session_end_flow
from station import supervisor_utils


PromptEntry = Tuple[str, Set[str]]
AGENT_NAME_PLACEHOLDER = "__AGENT_NAME__"


def _load_codex_text() -> str:
    codex_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.CODEX_FILENAME)
    codex_text = file_io_utils.load_text(codex_path)
    if not isinstance(codex_text, str):
        return ""
    return codex_text.strip()


def build_tenured_reached_message(constants_module: Any = constants) -> str:
    base_message = constants_module.TENURED_REACHED_MESSAGE.strip()

    min_archives_required = getattr(constants_module, "MIN_ARCHIVE_BEFORE_LEAVE", 0)
    if min_archives_required is None or min_archives_required <= 0:
        return f"{base_message} You are now free to leave the Station if you wish.\n"

    return (
        f"{base_message} You may leave the Station once you have authored at least "
        f"{min_archives_required} qualifying archive capsule(s).\n"
    )


def build_station_level_system_prompt(agent_name: str, role_description: Optional[str]) -> Optional[str]:
    """Build runtime system prompt with station-level prefix without persisting it in agent data."""
    if constants.STATION_SYSTEM_PROMPT_PREFIX_TEMPLATE is None:
        return role_description

    base_role_description = ""
    if isinstance(role_description, str):
        base_role_description = role_description.strip()

    role_block_placeholder = "Your defined role is:\n\n{role_description}"

    template = constants.STATION_SYSTEM_PROMPT_PREFIX_TEMPLATE or "{role_description}"
    try:
        agent_data = file_io_utils.load_yaml(
            os.path.join(
                constants.BASE_STATION_DATA_PATH,
                constants.AGENTS_DIR_NAME,
                f"{agent_name}{constants.YAML_EXTENSION}",
            )
        ) or {}
    except Exception:
        agent_data = {}

    override_role_text: Optional[str] = None
    if supervisor_utils.is_supervisor(agent_data, constants):
        override_role_text = constants.SUPERVISOR_PROTOCOL_SYSTEM_PROMPT.strip()
    elif agent_data.get(constants.AGENT_ROLE_KEY) == constants.ROLE_THEORIST:
        theorist_clause = (
            "You have been assigned as a theorist, "
            "meaning that you focus on mathematical theory instead of empirical results. "
            "Your code submission is limited to at most one every ten ticks. "
            "You should focus on theories that can accumulate and lead to positive discovery eventually, "
            "instead of theories that only certify dead ends or reinforce negative results."
        )
        override_role_text = theorist_clause
    codex_text = _load_codex_text()
    if override_role_text is not None:
        template = template.replace(role_block_placeholder, override_role_text)

    if "{role_description}" in template:
        if override_role_text is not None:
            prompt = template.replace("{role_description}", override_role_text)
        elif base_role_description:
            prompt = template.replace("{role_description}", base_role_description)
        else:
            prompt = template.replace(role_block_placeholder, "").strip()
    elif override_role_text is not None:
        prompt = template.strip()
    else:
        prompt = template.strip()
        if base_role_description:
            prompt = f"{prompt}\n\n{base_role_description}".strip()

    if "{codex}" in prompt:
        prompt = prompt.replace("{codex}", codex_text)

    return prompt.strip()


def _get_system_message_encoder(model_name: Optional[str] = None):
    """Best-effort tiktoken encoder selection for rendered system-message truncation."""
    if isinstance(model_name, str) and model_name.strip():
        try:
            return tiktoken.encoding_for_model(model_name.strip())
        except Exception:
            pass

    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception as exc:
        print(f"Warning: Failed to initialize system-message tiktoken encoder: {exc}")
        return None


def truncate_rendered_system_messages(rendered_text: str, model_name: Optional[str] = None) -> str:
    """
    Truncate the already-rendered system-message block to the configured token limit.

    If the block exceeds the limit, keep the leading tokens and append a warning notice
    while keeping the final decoded token sequence within the same limit.
    """
    if not isinstance(rendered_text, str) or not rendered_text:
        return rendered_text

    token_limit = getattr(constants, "SYSTEM_MESSAGES_MAX_TOKENS", None)
    if not isinstance(token_limit, int) or token_limit <= 0:
        return rendered_text

    encoder = _get_system_message_encoder(model_name)
    if encoder is None:
        return rendered_text

    try:
        rendered_tokens = encoder.encode(rendered_text)
    except Exception as exc:
        print(f"Warning: Failed to count rendered system-message tokens: {exc}")
        return rendered_text

    if len(rendered_tokens) <= token_limit:
        return rendered_text

    notice = "[System Messages exceeded the length limit; please do not issue too many actions within one tick.]"
    notice_prefix = "\n\n" if rendered_text.strip() else ""

    try:
        notice_tokens = encoder.encode(f"{notice_prefix}{notice}")
    except Exception as exc:
        print(f"Warning: Failed to encode system-message truncation notice: {exc}")
        return rendered_text

    if len(notice_tokens) >= token_limit:
        truncated_text = encoder.decode(notice_tokens[:token_limit])
        print(
            "Info: Rendered system messages exceeded the token limit; "
            f"truncated from {len(rendered_tokens)} to {token_limit} tokens."
        )
        return truncated_text

    keep_token_count = token_limit - len(notice_tokens)
    truncated_tokens = rendered_tokens[:keep_token_count] + notice_tokens
    truncated_text = encoder.decode(truncated_tokens)
    print(
        "Info: Rendered system messages exceeded the token limit; "
        f"truncated from {len(rendered_tokens)} to {token_limit} tokens."
    )
    return truncated_text


def normalize_prompt_audience(raw_audience: Any) -> Optional[Set[str]]:
    if raw_audience is None:
        return {"all"}
    if isinstance(raw_audience, str):
        return {raw_audience}
    if isinstance(raw_audience, list):
        normalized = {a for a in raw_audience if isinstance(a, str)}
        return normalized or {"all"}
    return {"all"}


def normalize_prompt_entry(entry: Any) -> Optional[Tuple[List[str], str, Set[str]]]:
    """
    Normalize a prompt entry into (conditions, text, audience).
    Supported formats:
    - Plain string -> always eligible, audience "all"
    - Dict with optional 'when' and required 'text', optional 'audience'
    - Two-item list/tuple [conditions, text] for backward compatibility with array-style syntax
    """
    if isinstance(entry, str):
        return [], entry, {"all"}

    conditions: List[str] = []
    text: Optional[str] = None
    audience: Optional[Set[str]] = None

    if isinstance(entry, dict):
        text = entry.get("text")
        raw_conditions = entry.get("when", [])
        audience = normalize_prompt_audience(entry.get("audience"))
    elif isinstance(entry, (list, tuple)) and len(entry) == 2:
        raw_conditions, text = entry
        audience = {"all"}
    else:
        return None

    if isinstance(raw_conditions, str):
        conditions = [raw_conditions]
    elif isinstance(raw_conditions, list):
        conditions = [c for c in raw_conditions if isinstance(c, str)]
    else:
        conditions = []

    if not isinstance(text, str):
        return None

    if not audience:
        return None

    return conditions, text, audience


def is_prompt_condition_met(condition_key: str, constants_module: Any = constants) -> bool:
    if not isinstance(condition_key, str):
        return False
    key = condition_key.strip()
    if not key:
        return False

    negate = False
    lowered = key.lower()
    if lowered.startswith("not "):
        negate = True
        key = key[4:].strip()
    elif key.startswith("!"):
        negate = True
        key = key[1:].strip()

    if not key:
        return False

    try:
        value = bool(getattr(constants_module, key))
    except Exception:
        value = False

    return not value if negate else value


def prompt_audience_matches(audience: Set[str], is_supervisor: bool) -> bool:
    if "all" in audience:
        return True
    if is_supervisor and "supervisor" in audience:
        return True
    if not is_supervisor and "non_supervisor" in audience:
        return True
    return False


def render_prompt_text(prompt_text: str, agent_data: Dict[str, Any]) -> str:
    if not isinstance(prompt_text, str):
        return ""
    agent_name = agent_data.get(constants.AGENT_NAME_KEY, "")
    if isinstance(agent_name, str) and agent_name:
        return prompt_text.replace(AGENT_NAME_PLACEHOLDER, agent_name)
    return prompt_text


def load_prompt_entries_from_file(
    prompt_path: str,
    constants_module: Any = constants,
    prompt_label: str = "prompt",
) -> Tuple[List[PromptEntry], List[PromptEntry]]:
    """
    Load and filter prompt entries by condition.
    Returns (eligible_prompts, unconditional_prompts), each as (text, audience).
    """
    if not file_io_utils.file_exists(prompt_path):
        return [], []

    try:
        prompt_data = file_io_utils.load_yaml(prompt_path)
        if not prompt_data or not isinstance(prompt_data, list):
            return [], []
    except Exception as e:
        print(f"Error loading {prompt_label}s: {e}")
        return [], []

    eligible_prompts: List[PromptEntry] = []
    unconditional_prompts: List[PromptEntry] = []

    for prompt_entry in prompt_data:
        normalized_prompt = normalize_prompt_entry(prompt_entry)
        if not normalized_prompt:
            print(f"Skipping invalid {prompt_label} entry: {prompt_entry}")
            continue

        conditions, prompt_text, audience = normalized_prompt
        if not conditions:
            unconditional_prompts.append((prompt_text, audience))
            eligible_prompts.append((prompt_text, audience))
            continue

        if all(is_prompt_condition_met(condition, constants_module) for condition in conditions):
            eligible_prompts.append((prompt_text, audience))

    if not eligible_prompts:
        eligible_prompts = unconditional_prompts

    return eligible_prompts, unconditional_prompts


def select_prompt_candidates_for_agent(
    eligible_prompts: List[PromptEntry],
    unconditional_prompts: List[PromptEntry],
    agent_data: Dict[str, Any],
    constants_module: Any = constants,
) -> List[str]:
    is_supervisor = supervisor_utils.is_supervisor(agent_data, constants_module)
    role_candidates = [
        render_prompt_text(prompt_text, agent_data)
        for prompt_text, audience in eligible_prompts
        if prompt_audience_matches(audience, is_supervisor)
    ]

    if not role_candidates:
        role_candidates = [
            render_prompt_text(prompt_text, agent_data)
            for prompt_text, audience in unconditional_prompts
            if prompt_audience_matches(audience, is_supervisor)
        ]

    return role_candidates


def is_meta_reflection_enabled(constants_module: Any = constants) -> bool:
    interval = getattr(constants_module, "REFLECTION_META_INTERVAL", None)
    try:
        return interval is not None and int(interval) > 0
    except (TypeError, ValueError):
        return False


def get_meta_reflection_interval(constants_module: Any = constants) -> Optional[int]:
    if not is_meta_reflection_enabled(constants_module):
        return None
    return int(getattr(constants_module, "REFLECTION_META_INTERVAL"))


def _non_holiday_tick_ordinal(station, current_tick: int) -> int:
    """Return how many non-holiday ticks have occurred through current_tick."""
    try:
        tick_number = int(current_tick)
    except (TypeError, ValueError):
        return 0
    if tick_number <= 0:
        return 0
    return sum(1 for tick in range(1, tick_number + 1) if not station.is_holiday_tick(tick))


def _is_holiday_prompt_delivery_tick(station, current_tick: int) -> bool:
    """Return True only on the first tick of each periodic two-tick holiday."""
    try:
        tick_number = int(current_tick)
    except (TypeError, ValueError):
        return False
    if tick_number <= 0:
        return False
    if not station.is_holiday_tick(tick_number):
        return False
    return bool(constants.HOLIDAY_MODE_ENABLED) and tick_number % 10 == 9


def _sample_prompt_file_to_agents(
    station,
    current_tick: int,
    prompt_filename: str,
    prompt_label: str,
    text_key: str,
    tick_key: str,
    log_label: str,
) -> None:
    prompt_path = os.path.join(constants.BASE_STATION_DATA_PATH, prompt_filename)
    if not file_io_utils.file_exists(prompt_path):
        return

    eligible_prompts, unconditional_prompts = load_prompt_entries_from_file(
        prompt_path,
        constants_module=constants,
        prompt_label=prompt_label,
    )
    if not eligible_prompts:
        return

    try:
        active_agent_names = station.agent_module.get_all_active_agent_names()
        if not active_agent_names:
            return
    except Exception as e:
        print(f"Error getting active agents for {prompt_label}s: {e}")
        return

    sent_count = 0
    for agent_name in active_agent_names:
        if agent_name == "AutoArchiveEvaluator":
            continue

        try:
            agent_data = station.agent_module.load_agent_data(agent_name)
            if not agent_data:
                continue

            role_candidates = select_prompt_candidates_for_agent(
                eligible_prompts,
                unconditional_prompts,
                agent_data,
                constants_module=constants,
            )
            if not role_candidates:
                continue

            selected_prompt = random.choice(role_candidates)
            agent_data[text_key] = selected_prompt
            agent_data[tick_key] = current_tick
            station.agent_module.save_agent_data(agent_name, agent_data)
            sent_count += 1

        except Exception as e:
            print(f"Error sending {prompt_label} to agent {agent_name}: {e}")
            continue

    if sent_count:
        print(f"{log_label} sent to {sent_count} active agents at tick {current_tick}")


def send_random_prompts_to_agents(station, current_tick: int):
    """
    Send random prompts to all active agents if the non-holiday tick frequency condition is met.
    Each agent gets a different random prompt from the available list.
    """
    if constants.RANDOM_PROMPT_FREQUENCY <= 0:
        return

    if station.is_holiday_tick(current_tick):
        return

    non_holiday_tick_ordinal = _non_holiday_tick_ordinal(station, current_tick)
    if non_holiday_tick_ordinal <= 0 or non_holiday_tick_ordinal % constants.RANDOM_PROMPT_FREQUENCY != 0:
        return

    _sample_prompt_file_to_agents(
        station,
        current_tick,
        constants.RANDOM_PROMPT_FILENAME,
        "random prompt",
        constants.AGENT_SYSTEM_TIP_TEXT_KEY,
        constants.AGENT_SYSTEM_TIP_TICK_KEY,
        "Random prompts",
    )


def send_holiday_prompts_to_agents(station, current_tick: int):
    """Send holiday prompts once at the start of each periodic holiday pair."""
    if not _is_holiday_prompt_delivery_tick(station, current_tick):
        return

    _sample_prompt_file_to_agents(
        station,
        current_tick,
        constants.HOLIDAY_PROMPT_FILENAME,
        "holiday prompt",
        constants.AGENT_HOLIDAY_PROMPT_TEXT_KEY,
        constants.AGENT_HOLIDAY_PROMPT_TICK_KEY,
        "Holiday prompts",
    )


def check_and_apply_inactivity_warning(station, agent_data: Dict[str, Any], current_tick: Optional[int] = None) -> Dict[str, Any]:
    """
    Checks for agent inactivity based on agent type and role.
    Guest agents: No warnings
    Recursive agents (non-supervisor): Warning after RECURSIVE_AGENT_INACTIVITY_THRESHOLD ticks without research submission
    Recursive agents (supervisor): Warning after RECURSIVE_SUPERVISOR_INACTIVITY_THRESHOLD ticks without meaningful actions
    Returns the (potentially modified) agent_data.
    """
    if not agent_data:
        return agent_data

    effective_tick = current_tick if current_tick is not None else station._get_current_tick()
    if effective_tick is not None and station.is_holiday_tick(effective_tick):
        return agent_data

    min_warning_age = getattr(constants, "INACTIVITY_WARNING_MIN_AGENT_AGE_TICKS", 0)
    try:
        min_warning_age = int(min_warning_age) if min_warning_age is not None else 0
    except (TypeError, ValueError):
        min_warning_age = 0
    if min_warning_age > 0 and effective_tick is not None:
        birth_tick = agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
        try:
            agent_age = int(effective_tick) - int(birth_tick) if birth_tick is not None else min_warning_age
        except (TypeError, ValueError):
            agent_age = min_warning_age
        if agent_age < min_warning_age:
            return agent_data

    raw_actions = agent_data.get(constants.AGENT_LAST_PARSED_ACTIONS_RAW_KEY, [])
    if not isinstance(raw_actions, list):
        raw_actions = []

    pending_notifications = agent_data.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, [])
    if not isinstance(pending_notifications, list):
        pending_notifications = []

    if not raw_actions:
        if constants.NO_VALID_ACTION_WARNING not in pending_notifications:
            station.agent_module.add_pending_notification(agent_data, constants.NO_VALID_ACTION_WARNING)
    else:
        goto_commands = {constants.ACTION_GO.lower(), constants.ACTION_GO_TO.lower()}
        goto_actions = []
        has_only_goto_actions = True
        for action in raw_actions:
            if not isinstance(action, dict):
                has_only_goto_actions = False
                break
            command = str(action.get("command", "")).lower()
            if command not in goto_commands:
                has_only_goto_actions = False
                break
            goto_actions.append(action)

        all_goto_destinations_visited = bool(goto_actions)
        for action in goto_actions:
            destination_short_name = str(action.get("args") or "").strip()
            if not destination_short_name or destination_short_name not in constants.SHORT_ROOM_NAME_TO_FULL_MAP:
                all_goto_destinations_visited = False
                break
            help_already_shown = station.agent_module.get_agent_room_state(
                agent_data,
                destination_short_name,
                constants.AGENT_ROOM_STATE_FIRST_VISIT_HELP_SHOWN_KEY,
                default=False,
            )
            if not help_already_shown:
                all_goto_destinations_visited = False
                break

        if (
            has_only_goto_actions
            and all_goto_destinations_visited
            and not agent_data.get(constants.AGENT_GOTO_ONLY_WARNING_SENT_KEY, False)
            and constants.GOTO_ONLY_ACTION_WARNING not in pending_notifications
        ):
            station.agent_module.add_pending_notification(agent_data, constants.GOTO_ONLY_ACTION_WARNING)
            agent_data[constants.AGENT_GOTO_ONLY_WARNING_SENT_KEY] = True

    # Guest agents never get inactivity warnings
    if agent_data.get(constants.AGENT_STATUS_KEY) != constants.AGENT_STATUS_RECURSIVE:
        return agent_data

    if supervisor_utils.is_theorist(agent_data, constants):
        agent_data[constants.AGENT_INACTIVITY_TICK_COUNT_KEY] = 0
        agent_data[constants.AGENT_INACTIVITY_WARNING_SENT_KEY] = False
        return agent_data

    age_status = station._get_agent_age_status(agent_data, effective_tick)

    # Get inactivity count, defaulting to 0 if not present
    inactivity_count = agent_data.get(constants.AGENT_INACTIVITY_TICK_COUNT_KEY, 0)

    # Check if agent is a supervisor
    is_supervisor = supervisor_utils.is_supervisor(agent_data, constants)

    if is_supervisor:
        # Supervisor: Check against supervisor threshold (None disables warnings)
        if constants.RECURSIVE_SUPERVISOR_INACTIVITY_THRESHOLD is not None and inactivity_count >= constants.RECURSIVE_SUPERVISOR_INACTIVITY_THRESHOLD:
            warning_message = constants.RECURSIVE_SUPERVISOR_INACTIVITY_WARNING.format(inactive_ticks=inactivity_count)
            station.agent_module.add_pending_notification(agent_data, warning_message)
    else:
        # Non-supervisor recursive agent: Check against research submission threshold (None disables warnings)
        threshold_base = constants.RECURSIVE_AGENT_INACTIVITY_THRESHOLD
        threshold_effective = threshold_base
        if threshold_base is not None and age_status == "tenured":
            threshold_effective = threshold_base * 3  # Tenured agents get 3x grace before critical research notice

        if threshold_effective is not None and inactivity_count >= threshold_effective:
            warning_message = constants.RECURSIVE_AGENT_INACTIVITY_WARNING.format(inactive_ticks=inactivity_count)
            station.agent_module.add_pending_notification(agent_data, warning_message)

    return agent_data


def update_meta_reflection_tick_count(station, agent_data: Dict[str, Any], current_tick: int) -> Dict[str, Any]:
    """
    Increment the mature-agent countdown for compulsory meta reflection.
    Disabled mechanisms and immature agents do not track a countdown.
    """
    if not agent_data:
        return agent_data

    if not is_meta_reflection_enabled(constants):
        return agent_data

    if not station._is_agent_mature(agent_data, current_tick):
        agent_data[constants.AGENT_META_REFLECTION_TICK_COUNT_KEY] = 0
        return agent_data

    current_count = agent_data.get(constants.AGENT_META_REFLECTION_TICK_COUNT_KEY, 0)
    try:
        current_count = int(current_count)
    except (TypeError, ValueError):
        current_count = 0
    agent_data[constants.AGENT_META_REFLECTION_TICK_COUNT_KEY] = max(0, current_count) + 1
    return agent_data


def check_and_apply_meta_reflection_warning(station, agent_data: Dict[str, Any], current_tick: int) -> Dict[str, Any]:
    """
    Warn mature agents every tick once compulsory meta reflection is overdue.
    """
    if not agent_data:
        return agent_data

    interval = get_meta_reflection_interval(constants)
    if interval is None:
        return agent_data

    if not station._is_agent_mature(agent_data, current_tick):
        return agent_data

    current_count = agent_data.get(constants.AGENT_META_REFLECTION_TICK_COUNT_KEY, 0)
    try:
        current_count = int(current_count)
    except (TypeError, ValueError):
        current_count = 0

    if current_count >= interval + 5:
        overdue_ticks = current_count - interval
        warning_message = (
            "**System Warning: Meta Reflection Due**\n\n"
            f"Your meta reflection is overdue by {overdue_ticks} ticks. "
            "Please proceed to the Reflection Chamber and issue `/execute_action{meta_reflect}` "
            "to begin your meta reflection."
        )
        station.agent_module.add_pending_notification(agent_data, warning_message)

    return agent_data


def check_and_apply_life_warnings(station, agent_data: Dict[str, Any], current_tick: int) -> Dict[str, Any]:
    """
    Checks agent life limit warnings based on age and adds notifications if necessary.
    Updates life warning sent flag in agent_data.
    Returns the (potentially modified) agent_data.
    """
    if not agent_data:
        return agent_data

    # Get agent-specific max_age, fallback to AGENT_MAX_LIFE for backward compatibility
    agent_max_age = agent_data.get(constants.AGENT_MAX_AGE_KEY, constants.AGENT_MAX_LIFE)
    if agent_max_age is None:
        return agent_data  # No life limit for this agent

    # Calculate agent age
    birth_tick = agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
    if birth_tick is None:
        return agent_data

    agent_age = current_tick - birth_tick
    remaining_ticks = agent_max_age - agent_age

    # Check if we should send a warning
    if remaining_ticks <= constants.AGENT_LIFE_WARNING_THRESHOLD and remaining_ticks > 0:
        if not agent_data.get(constants.AGENT_LIFE_WARNING_SENT_KEY, False):
            warning_message = constants.AGENT_LIFE_WARNING_MESSAGE.format(
                remaining_ticks=remaining_ticks,
                exit_room=constants.SHORT_ROOM_NAME_EXIT,
            )
            if supervisor_utils.is_supervisor(agent_data, constants):
                warning_message += constants.AGENT_LIFE_WARNING_SUPERVISOR_MESSAGE
            station.agent_module.add_pending_notification(agent_data, warning_message)
            agent_data[constants.AGENT_LIFE_WARNING_SENT_KEY] = True

    return agent_data


def check_and_apply_supervisor_report_reminder(
    station, agent_data: Dict[str, Any], current_tick: int
) -> Dict[str, Any]:
    """
    Send supervisor report reminder at a fixed number of ticks before report deadlines.
    Deadlines are defined in agent age ticks.
    """
    if not agent_data:
        return agent_data

    if not supervisor_utils.is_supervisor(agent_data, constants):
        return agent_data

    report_ages = constants.SUPERVISOR_REPORT_AGE_TICKS
    if not report_ages:
        return agent_data

    if not isinstance(report_ages, list):
        return agent_data

    advance_notice = constants.SUPERVISOR_REPORT_ADVANCE_NOTICE_TICKS
    if advance_notice is None or advance_notice <= 0:
        return agent_data

    birth_tick = agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
    if birth_tick is None:
        return agent_data

    agent_age = current_tick - birth_tick
    upcoming_ages = [age for age in report_ages if isinstance(age, int)]
    if not upcoming_ages:
        return agent_data

    for report_age in upcoming_ages:
        if agent_age == report_age - advance_notice:
            remaining_ticks = report_age - agent_age
            deadline_tick = current_tick + remaining_ticks
            reminder_message = constants.SUPERVISOR_REPORT_SYSTEM_MESSAGE.format(
                deadline_tick,
                remaining_ticks
            )
            pending_notifications = agent_data.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, [])
            if reminder_message not in pending_notifications:
                station.agent_module.add_pending_notification(agent_data, reminder_message)
            break

    return agent_data


def check_and_apply_token_warnings(station, agent_data: Dict[str, Any], effective_max_budget: Optional[int]) -> Dict[str, Any]:
    """
    Checks token budget warnings based on current usage in agent_data
    and adds notifications to agent_data if necessary. Updates warning sent flags in agent_data.
    Warning flags are reset when token usage drops below respective thresholds.
    This method expects agent_data to have the latest AGENT_TOKEN_BUDGET_CURRENT_KEY.
    Returns the (potentially modified) agent_data.
    """
    if not agent_data or effective_max_budget is None or effective_max_budget <= 0:  # station.py:1032
        return agent_data

    if agent_data.get(constants.AGENT_TOKEN_BUDGET_CURRENT_STALE_KEY):
        return agent_data

    current_session_total_tokens_used = agent_data.get(constants.AGENT_TOKEN_BUDGET_CURRENT_KEY, 0)  # station.py:1027
    if current_session_total_tokens_used is None:
        return agent_data
    if not isinstance(current_session_total_tokens_used, (int, float)):
        return agent_data
    is_guest = agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_GUEST  # station.py:1023

    pre_warning_ratio = constants.GUEST_PRE_WARNING_RATIO if is_guest else constants.RECURSIVE_PRE_WARNING_RATIO  # station.py:1035
    warning_ratio = constants.GUEST_WARNING_RATIO if is_guest else constants.RECURSIVE_WARNING_RATIO  # station.py:1036

    pre_warning_threshold = pre_warning_ratio * effective_max_budget  # station.py:1038
    warning_threshold = warning_ratio * effective_max_budget  # station.py:1039

    pre_warning_sent_key = constants.AGENT_TOKEN_BUDGET_PRE_WARNING_SENT_KEY  # station.py:1041
    warning_sent_key = constants.AGENT_TOKEN_BUDGET_WARNING_SENT_KEY  # station.py:1042

    # Reset warning flags if token usage has dropped below respective thresholds
    if current_session_total_tokens_used < pre_warning_threshold:
        # If usage is below pre-warning threshold, reset both flags
        if agent_data.get(pre_warning_sent_key, False):
            agent_data[pre_warning_sent_key] = False
        if agent_data.get(warning_sent_key, False):
            agent_data[warning_sent_key] = False
    elif current_session_total_tokens_used < warning_threshold:
        # If usage is below warning threshold but above pre-warning, reset only final warning flag
        if agent_data.get(warning_sent_key, False):
            agent_data[warning_sent_key] = False

    # Check for Pre-Warning
    if current_session_total_tokens_used >= pre_warning_threshold and \
       current_session_total_tokens_used < warning_threshold and \
       not agent_data.get(pre_warning_sent_key, False):  # station.py:1045

        pre_warning_message_header = (  # station.py:1048
            f"**NOTICE:** You have used {current_session_total_tokens_used:,} tokens, which is over {pre_warning_ratio * 100:.0f}% of your "  # station.py:1049
            f"current budget limit of {effective_max_budget:,}.\n"  # station.py:1050
        )
        specific_guidance = ""  # station.py:1051
        if is_guest:  # station.py:1052
            specific_guidance = (  # station.py:1053
                "Prioritize ascension now (`/execute_action{ascend_inherit}` or `/execute_action{ascend_new}`) so you can unlock the Token Management Room for pruning when needed."
            )
        else:  # Recursive Agent # station.py:1056
            specific_guidance = constants.RECURSIVE_PRE_WARNING  # station.py:1057

        station.agent_module.add_pending_notification(agent_data, pre_warning_message_header + specific_guidance)  # station.py:1059
        agent_data[pre_warning_sent_key] = True  # station.py:1060

    # Check for Final Warning
    if current_session_total_tokens_used >= warning_threshold and \
       current_session_total_tokens_used < effective_max_budget and \
       not agent_data.get(warning_sent_key, False):  # station.py:1064

        warning_message_header = (  # station.py:1067
            f"**CRITICAL NOTICE:** You have used {current_session_total_tokens_used:,} tokens, which is over {warning_ratio * 100:.0f}% of your "  # station.py:1068
            f"current budget limit of {effective_max_budget:,}.\n"  # station.py:1069
        )
        specific_guidance_final = ""  # station.py:1070
        if is_guest:  # station.py:1071
            specific_guidance_final = (  # station.py:1072
                "Focus on completing ascension now (`/execute_action{ascend_inherit}` or `/execute_action{ascend_new}`) so you can manage context directly once promoted."
            )
        else:  # Recursive Agent # station.py:1075
            specific_guidance_final = constants.RECURSIVE_WARNING  # station.py:1076

        station.agent_module.add_pending_notification(agent_data, warning_message_header + specific_guidance_final)  # station.py:1078
        agent_data[warning_sent_key] = True  # station.py:1079

    return agent_data


def check_and_notify_maturity(station, agent_name: str, agent_data: Dict[str, Any], current_tick: int) -> None:
    """
    Check if agent has just reached maturity and send congratulatory notification.
    """
    birth_tick = agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
    if birth_tick is None:
        return

    agent_age = current_tick - birth_tick
    tenured_threshold = getattr(constants, "MIN_AGENT_AGE_BEFORE_LEAVE", None)
    tenured_enabled = tenured_threshold is not None and tenured_threshold > 0
    updated = False

    if constants.AGENT_ISOLATION_TICKS is not None and agent_age == constants.AGENT_ISOLATION_TICKS:
        if not agent_data.get(constants.AGENT_MATURITY_NOTIFIED_KEY, False):
            maturity_message = constants.MATURITY_REACHED_MESSAGE
            interval = get_meta_reflection_interval(constants)
            if interval is not None:
                maturity_message += (
                    "\n\n**Compulsory Meta Reflection**\n\n"
                    "To help agents avoid becoming overly fixated on the research task, "
                    f"a compulsory meta reflection is required at least every {interval} ticks. "
                    "To perform meta reflection, go to the Reflection Chamber and issue "
                    "`/execute_action{meta_reflect}` with no additional input. "
                    "The system will randomly select a meta reflection prompt for you to respond to. "
                    "Meta reflection is available on both normal work days and holidays."
                )
            if getattr(constants, "EXTERNAL_COUNTER_ENABLED", False):
                insert_line = "\n- External Counter - Request external literature surveys"
                if "**Maturity Guidance**" in maturity_message and "External Counter" not in maturity_message:
                    maturity_message = maturity_message.replace("\n\n**Maturity Guidance**", f"{insert_line}\n\n**Maturity Guidance**")
                elif "External Counter" not in maturity_message:
                    maturity_message += insert_line
            station.agent_module.add_pending_notification(agent_data, maturity_message)
            if agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_RECURSIVE:
                supervisor_name = supervisor_utils.get_active_supervisor_name(station.agent_module, constants)
                supervisor_message = supervisor_utils.build_supervisor_mentee_append(supervisor_name, constants)
                if supervisor_message:
                    pending_notifications = agent_data.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, [])
                    if supervisor_message not in pending_notifications:
                        station.agent_module.add_pending_notification(agent_data, supervisor_message)
            agent_data[constants.AGENT_MATURITY_NOTIFIED_KEY] = True
            updated = True

    if tenured_enabled and agent_age == tenured_threshold:
        if not agent_data.get(constants.AGENT_TENURE_NOTIFIED_KEY, False):
            station.agent_module.add_pending_notification(
                agent_data,
                build_tenured_reached_message(constants),
            )
            agent_data[constants.AGENT_TENURE_NOTIFIED_KEY] = True
            updated = True

    if updated:
        station.agent_module.save_agent_data(agent_name, agent_data)


def terminate_agent_session_with_broadcast(station, agent_name: str, reason: str, critical_notification: str) -> None:
    """
    Terminates an agent session with proper broadcast to other agents.
    Used for session termination scenarios like context overflow, etc.

    Args:
        agent_name: Name of the agent to terminate
        reason: Reason for termination (for broadcast message)
        critical_notification: Message to send to the terminating agent
    """
    session_end_flow.terminate_agent_session_with_broadcast(station, agent_name, reason, critical_notification)


def build_archive_announcement(agent_name: str,
                               author_desc: str,
                               title: str,
                               capsule_id: str,
                               abstract: Optional[str] = None,
                               short_room_name: Optional[str] = None) -> str:
    """
    Build the standard station announcement for new archive capsules.
    Uses the AutoArchiveEvaluator format for consistency.
    """
    capsule_num = capsule_id.replace("archive_", "")
    desc_part = f" ({author_desc})" if author_desc else ""
    abstract_line = ""
    if isinstance(abstract, str) and abstract.strip():
        abstract_line = f"\nAbstract:\n{abstract.strip()}"
    read_line = ""
    if short_room_name:
        read_line = f"\nTo read it: /execute_action{{goto {short_room_name}}} /execute_action{{read {capsule_num}}}."

    return (
        f"**Station Announcement:** **{agent_name}**{desc_part} has published a new archive capsule: "
        f"'{title}' (Archive #{capsule_num})."
        f"{abstract_line}"
        f"{read_line}"
    )
