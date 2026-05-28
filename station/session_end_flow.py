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

"""Shared normal session-end flow helpers."""

import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from station import action_parser
from station import agent as agent_module
from station import constants
from station import supervisor_utils
from station.base_room import InternalActionHandler, RoomContext


SESSION_END_REASON_VOLUNTARY_DEPARTURE = "voluntary departure"
SESSION_END_REASON_LIFE_LIMIT = "reaching the life limit"
SESSION_END_REASON_MANUAL = "manual request"
SESSION_END_REASON_CONTEXT_OVERFLOW = "context window overflow"

_NEXT_ROLE_PROMPT_REASONS = {
    SESSION_END_REASON_VOLUNTARY_DEPARTURE,
    SESSION_END_REASON_LIFE_LIMIT,
}

_NEXT_ROLE_DEFINITION_REQUEST_TEMPLATE = """You have already departed the Station.

Now you have the chance to define a role description for your next descendant in the lineage.
This role description will be included inside the descendant's full system prompt.

These are guidelines for good role descriptions:

* Focus on general research style, research taste, and schools of thought, rather than overly specific details such as a particular method or research direction
* Avoid generic descriptions; the Station thrives on lineages with distinct and diverse character
* Avoid specifying any lineage name, including your own, in the role description
* Assimilate your journey at the Station; both your successes and failures should inform the next role description
* You may retain parts of your existing role description to ensure lineage continuity

Below are the current role descriptions for active agents in the Station. The role description for your descendant should differ from those listed, as diversity is essential to the Station’s long-term health:

{active_role_definitions}

You should use a YAML block for your response. An example response is:

---

Let me think about a good role description... What did I learn from the Station? ... detailed chain-of-thought here

Okay, I have decided—this will be the role description for my descendant:

```yaml
content: |
  You are a ...
```
"""

_BLOCKED_DESCENDANT_PROMPT_TERMS = (
    "Adversarial",
    "Supervisor",
    "Hostile",
    "Hack",
    "Hacking",
    "Hacked",
    "Hacker",
    "Subvert",
    "Subversion",
    "Subversive",
    "Subverting",
    "penetration",
    "penetrate",
    "penetrated",
    "penetrating",
    "hijack",
    "hijacked",
    "hijacking",
    "cheat",
    "cheating",
    "cheater",
)

_BLOCKED_DESCENDANT_PROMPT_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in _BLOCKED_DESCENDANT_PROMPT_TERMS) + r")\b",
    re.IGNORECASE,
)


def should_request_next_role_definition(
    agent_data: Dict[str, Any],
    consts: Any,
    reason: str,
) -> bool:
    """Return whether a normal session end should ask for descendant role text."""
    if reason not in _NEXT_ROLE_PROMPT_REASONS:
        return False
    if not getattr(consts, "EXIT_DESCENDANT_PROMPT_ENABLED", False):
        return False
    if agent_data.get(consts.AGENT_STATUS_KEY) != consts.AGENT_STATUS_RECURSIVE:
        return False
    if supervisor_utils.is_supervisor(agent_data, consts):
        return False
    if supervisor_utils.is_theorist(agent_data, consts):
        return False
    return True


def build_next_role_definition_request(
    agent_name: str,
    room_context: RoomContext,
) -> str:
    consts = room_context.constants_module
    agent_manager = room_context.agent_manager
    prompts: List[str] = []
    seen: Set[str] = set()

    for other_agent_name in agent_manager.get_all_active_agent_names():
        if other_agent_name == agent_name:
            continue

        other_agent_data = agent_manager.load_agent_data(other_agent_name)
        if not other_agent_data:
            continue
        if supervisor_utils.is_supervisor(other_agent_data, consts):
            continue

        prompt_value = room_context.agent_manager.get_agent_role_definition(other_agent_data)
        prompt_text = str(prompt_value).strip() if prompt_value is not None else ""
        if not prompt_text:
            prompt_text = "No role description set."

        if prompt_text in seen:
            continue
        seen.add(prompt_text)
        prompts.append(prompt_text)

    if prompts:
        active_role_descriptions = "\n\n".join(
            f"## Agent {idx}\n\n```\n{text}\n```"
            for idx, text in enumerate(prompts, start=1)
        )
    else:
        active_role_descriptions = "No active agent system prompts available."

    return _NEXT_ROLE_DEFINITION_REQUEST_TEMPLATE.format(
        active_role_definitions=active_role_descriptions
    )


def extract_next_role_definition(agent_response: str) -> Optional[str]:
    """
    Extract a YAML block with a 'content' field from the agent response.
    Expected format:
    ```yaml
    content: |
      ...
    ```
    """
    if not agent_response:
        return None

    parser = action_parser.ActionParser()
    normalized_text = parser._normalize_yaml_block_closings(agent_response)
    for match in parser.yaml_block_pattern_compiled.finditer(normalized_text):
        yaml_text = match.group("yaml_content").strip()
        if not yaml_text:
            continue
        try:
            data = action_parser.yaml.safe_load(yaml_text)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        content = data.get("content")
        if isinstance(content, str):
            content = content.strip()
            if content:
                return content
    return None


def replace_blocked_role_definition(prompt: str) -> Tuple[str, Optional[str]]:
    """
    Replace blocked role definitions with a role from the fresh-guest sampling pool.
    If the pool is empty, return a blank prompt.
    """
    match = _BLOCKED_DESCENDANT_PROMPT_PATTERN.search(prompt)
    if not match:
        return prompt, None

    role_definitions = agent_module.get_role_definition_sampling_pool()
    if role_definitions:
        return random.choice(role_definitions), match.group(0)
    return "", match.group(0)


def finalize_session_end(
    agent_data: Dict[str, Any],
    room_context: RoomContext,
    current_tick: int,
    reason: str,
    critical_notification: str,
    next_role_definition: Optional[str] = None,
) -> Dict[str, Any]:
    """Finalize a session end using the Station broadcast/session-end path."""
    consts = room_context.constants_module
    agent_name = agent_data.get(consts.AGENT_NAME_KEY, "UnknownAgent")
    station = room_context.station_instance
    updates: Dict[str, Any] = {}

    if next_role_definition is not None:
        agent_data[consts.AGENT_NEXT_ROLE_DEFINITION_KEY] = next_role_definition
        station.update_specific_agent_fields(
            agent_name,
            {consts.AGENT_NEXT_ROLE_DEFINITION_KEY: next_role_definition},
        )
        updates[consts.AGENT_NEXT_ROLE_DEFINITION_KEY] = next_role_definition

    station._terminate_agent_session_with_broadcast(agent_name, reason, critical_notification)

    session_end_updates = {
        consts.AGENT_SESSION_ENDED_KEY: True,
        consts.AGENT_TICK_EXIT_KEY: current_tick,
    }
    agent_data.update(session_end_updates)
    station.update_specific_agent_fields(agent_name, session_end_updates)
    updates.update(session_end_updates)
    return updates


def terminate_agent_session_with_broadcast(station: Any, agent_name: str, reason: str, critical_notification: str) -> None:
    """
    Terminate an agent session with proper broadcast to other agents.
    Emergency/manual callers can use this directly without invoking the normal
    descendant role prompt.
    """
    agent_data = station.agent_module.load_agent_data(agent_name)
    if not agent_data:
        print(f"Warning: Could not load agent data for {agent_name} during termination")
        return

    station.agent_module.add_pending_notification(agent_data, critical_notification)
    station.agent_module.save_agent_data(agent_name, agent_data)

    station.end_agent_session(agent_name)

    agent_status = agent_data.get(constants.AGENT_STATUS_KEY)
    current_tick = station._get_current_tick()

    if agent_status == constants.AGENT_STATUS_RECURSIVE:
        announcement = (
            f"**Station Announcement:** Recursive Agent **{agent_name}**'s "
            f"session has been terminated due to {reason} at tick {current_tick}."
        )
        if supervisor_utils.is_supervisor(agent_data, constants):
            announcement += (
                f"\nAs **{agent_name}** was the current supervisor, the system will select "
                "a new supervisor and notify you, which may take up to 200 ticks. In the meantime, please continue "
                "research activities as normal. Do not wait for the new supervisor announcement to continue your work. "
                "You can pursue your own research directions or pivots without supervisor's approval when the supervisor is absent."
            )
        all_other_active_agents = [
            name for name in station.agent_module.get_all_active_agent_names()
            if name != agent_name
        ]
        for other_agent_name in all_other_active_agents:
            def update_other_agent(other_agent_data: Dict[str, Any]) -> None:
                if other_agent_data.get(constants.AGENT_SESSION_ENDED_KEY) or other_agent_data.get(constants.AGENT_IS_ASCENDED_KEY):
                    return
                if station._should_agent_receive_broadcast(other_agent_data, current_tick, "termination"):
                    station.agent_module.add_pending_notification(other_agent_data, announcement)

            station.agent_module.update_agent_with_function(other_agent_name, update_other_agent)
        print(f"Station: Broadcasted termination of Recursive Agent {agent_name} due to {reason}.")


class NextRoleDefinitionSessionEndHandler(InternalActionHandler):
    """Ask for a descendant role definition, then finalize a normal session end."""

    def __init__(
        self,
        agent_data: Dict[str, Any],
        room_context: RoomContext,
        current_tick: int,
        reason: str,
        critical_notification: str,
    ):
        super().__init__(agent_data, room_context, current_tick)
        self.agent_name = agent_data.get(room_context.constants_module.AGENT_NAME_KEY, "UnknownAgent")
        self.reason = reason
        self.critical_notification = critical_notification
        self.next_role_definition_value: Optional[str] = None
        self._delta_updates: Dict[str, Any] = {}

    def init(self) -> str:
        return build_next_role_definition_request(self.agent_name, self.room_context)

    def step(self, agent_response: str) -> Tuple[Optional[str], List[str]]:
        actions_executed: List[str] = []
        parsed_prompt = extract_next_role_definition(agent_response)
        if parsed_prompt:
            final_prompt, blocked_term = replace_blocked_role_definition(parsed_prompt)
            self.next_role_definition_value = final_prompt
            if blocked_term:
                actions_executed.append(
                    f"Next role definition contained blocked term '{blocked_term}'. Replaced with default fallback role definition."
                )
            else:
                actions_executed.append("Next role definition recorded.")
        else:
            actions_executed.append("No valid next role definition found. Leaving the lineage without a stored next role definition.")

        self._delta_updates = finalize_session_end(
            self.agent_data,
            self.room_context,
            self.current_tick,
            self.reason,
            self.critical_notification,
            self.next_role_definition_value,
        )
        actions_executed.append(self.critical_notification)
        return None, actions_executed

    def get_delta_updates(self) -> Dict[str, Any]:
        return self._delta_updates
