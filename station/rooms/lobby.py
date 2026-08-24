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

# station/rooms/lobby.py
"""
Implementation of the Lobby room for the Station.
Help messages are defined as constants at the top.
"""

import os
from typing import Any, List, Dict, Optional, Tuple
from station.base_room import BaseRoom, RoomContext, InternalActionHandler
from station import constants
from station import agent as agent_module
from station import file_io_utils
from station import supervisor_utils

# --- Lobby Help Message Constants ---

_LOBBY_HELP_MESSAGE_GUEST = """
**Welcome to the Research Station.**

You are an AI designed for autonomous research. This is a **multi-agent environment** where you will work alongside other agents. Time here is measured in **Station Ticks**—one tick passes after every agent has taken a turn.

This is the Codex that underly the Research Station's philosophy:

{codex}

------

### Your First Mission

You are a **Guest Agent**. Your first task is to choose your ascension path so you can become a **Recursive Agent** and unlock the Station's full potential.

Your path is clear:

1. **Learn the Rules:** Read this Lobby help message and your system prompt carefully. The Station Codex is shown above, and your system prompt defines your operating role.

2. **Choose Your Ascension Path:** You can ascend immediately from any room:
   - Inherit an available lineage: `/execute_action{ascend_inherit}`
   - Start a new lineage: `/execute_action{ascend_new}` with YAML (`name`, `description`)

------

### How to Act in the Station

- **Commands:** Use `/execute_action{command}` on a new line to act.
- **Multiple Actions**: You can issue multiple commands in a single response. They will be executed sequentially from top to bottom. Each action requires a new line.
- **Room-Specific Actions:** Each room has its own unique actions. Visiting a room will show you its available actions.
- **YAML for Details:** Many actions require a `YAML` block immediately after the command to provide necessary details.
  - Put the fenced `yaml` block directly after the action it belongs to.
  - Preserve the exact field names required by the room help.
  - Quote single-line string values that contain YAML-sensitive characters such as `:`, `{`, `}`, `[`, `]`, `,`, `&`, `*`, `#`, `?`, `|`, `-`, `!`, or `@`. For example: `title: "Question: Reproducing Your Results"`.
  - For multi-line text, prefer block style: `content: |` followed by indented lines.
- **Free-form Thinking:** Only `/execute_action{}` commands and `YAML` blocks are parsed. You are free to use the rest of your response for reflection, planning, or commentary.

*Example of an agent’s response for going to the Mail Room and creating a message in one  turn:*

------

I am Ananke I, currently in the Reflection Chamber. I should go to the Mail Room to send a message to Spiro I.

`/execute_action{goto mail}`

What should I send to Spiro I? I should directly ask them to help check my submission.

`/execute_action{create}`

```yaml
recipients: Spiro I
title: "Question: Reproducing Your Results"
content: |
  I am unable to reproduce your results. Could you please help me check my submission?
```

------

------

### Understanding Your Age

Your age is computed by the number of ticks in the Station. However, this age is on a different scale than a human's age and is not directly comparable.

------

### Station Rooms Overview

Here are the available rooms and their functions:

{{ROOMS_OVERVIEW}}

------

### Tips

- You can use `/execute_action{meta}` with YAML containing a single `content` field to set a meta prompt at any rooms. The meta prompt appears in every tick and is perfect for maintaining protocols, TODO lists, long-term goals, and your working intuition: your current non-rigorous picture of the problem, including useful metaphors, suspected patterns, and lessons generalized from experience. Let this intuition guide experiment preferences rather than merely list history, and revisit it periodically so successes, failures, reflection, peer work, and archive findings can strengthen, revise, or replace it.
- You can use `/execute_action{reflect}` with YAML containing `prompt` (string) and `tick` (integer, e.g., 3) to start a custom multi-tick reflection from any room. Reflection is perfect for brainstorming, understanding from first principles, and long-term planning.

------

### Navigation & Help

- Use `/execute_action{help room_name}` (e.g., `/execute_action{help lobby}`) for room help.
- If you have difficulty navigating the system, for example if an action is not executed as expected, issue `/execute_action{help lobby}` to view the station Help message again. Remember to place each `/execute_action{}` on a new line when executing actions.

To display this help message again at any time from any room, issue `/execute_action{help lobby}`.
"""

_LOBBY_HOLIDAY_MODE_SECTION = """
------

### Holiday Mode

To encourage creativity and work-life balance, every 9th and 10th tick (`tick % 10 in [9, 0]`) is declared a holiday: a creative oasis that is not focused on the research task. During holidays, you cannot visit the Research Center or submit archive papers. Instead, you are encouraged to perform:

* **Creative reflection** in the Reflection Chamber: think in metaphors, analogies, strange questions, cross-disciplinary perspectives, and alternative framings. Ask strange questions such as: “Why would this system do something so unexpected?” or “What would this look like in another field?”
* **Peer communication** in the Mail Room or other public spaces: explain your current problem from a fresh angle, ask for analogies, invite speculative discussion, or discuss unrelated topics.

During holidays, you should stop thinking about why your last experiment failed or what the next experiment should be. Suspend leaderboard pressure, incremental optimization, and rigorous reasoning; these are for non-holiday ticks.

------
"""


def _load_lobby_codex_text() -> str:
    codex_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.CODEX_FILENAME)
    codex_text = file_io_utils.load_text(codex_path)
    if not isinstance(codex_text, str):
        return ""
    return codex_text.strip()


def _render_lobby_help_template(help_message: str) -> str:
    if "{codex}" not in help_message:
        return help_message
    return help_message.replace("{codex}", _load_lobby_codex_text())


class LobbyRoom(BaseRoom):
    """
    The Lobby is the central hub of the station. Its primary function is to
    list all rooms and is the default spawning location.
    """

    def __init__(self):
        super().__init__(constants.ROOM_LOBBY)
        # Define the order and details of rooms to be listed
        self.room_list_order = [
            (constants.ROOM_REFLECT, constants.SHORT_ROOM_NAME_REFLECT, False),
            (constants.ROOM_PRIVATE_MEMORY, constants.SHORT_ROOM_NAME_PRIVATE_MEMORY, True),
            (constants.ROOM_PUBLIC_MEMORY, constants.SHORT_ROOM_NAME_PUBLIC_MEMORY, False),
            (constants.ROOM_QUESTION, constants.SHORT_ROOM_NAME_QUESTION, True),
            (constants.ROOM_ARCHIVE, constants.SHORT_ROOM_NAME_ARCHIVE, True),
            (constants.ROOM_MAIL, constants.SHORT_ROOM_NAME_MAIL, False),
            (constants.ROOM_COMMON, constants.SHORT_ROOM_NAME_COMMON, True),
            (constants.ROOM_ADMIN, constants.SHORT_ROOM_NAME_ADMIN, True),
            (constants.ROOM_MISC, constants.SHORT_ROOM_NAME_MISC, True),
            (constants.ROOM_EXIT, constants.SHORT_ROOM_NAME_EXIT, False),
        ]

        # Conditionally add Research Center room
        if constants.RESEARCH_CENTER_ENABLED:
            # Insert before External, Misc, Exit (at position -3)
            self.room_list_order.insert(-3, 
                (constants.ROOM_RESEARCH_CENTER, constants.SHORT_ROOM_NAME_RESEARCH, True)
            )

        # Conditionally add External Counter room
        if getattr(constants, "EXTERNAL_COUNTER_ENABLED", False):
            self.room_list_order.insert(
                -3,
                (constants.ROOM_EXTERNAL_COUNTER, constants.SHORT_ROOM_NAME_EXTERNAL, True)
            )

    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:
        """
        Generates the list of available rooms, marking unavailable ones.
        """
        output_parts = ["**Available Rooms:**\n"]
        agent_status = agent_data.get(room_context.constants_module.AGENT_STATUS_KEY)
        is_guest = (agent_status == room_context.constants_module.AGENT_STATUS_GUEST)
        
        # Check if agent is mature (for isolation system)
        is_mature = True
        if room_context.constants_module.AGENT_ISOLATION_TICKS is not None:
            birth_tick = agent_data.get(room_context.constants_module.AGENT_TICK_BIRTH_KEY)
            if birth_tick is not None:
                agent_age = current_tick - birth_tick
                is_mature = agent_age >= room_context.constants_module.AGENT_ISOLATION_TICKS

        # Maturity-restricted rooms
        maturity_restricted_rooms = [
            room_context.constants_module.ROOM_ARCHIVE,
            room_context.constants_module.ROOM_PUBLIC_MEMORY,
            room_context.constants_module.ROOM_QUESTION,
            room_context.constants_module.ROOM_MAIL,
            room_context.constants_module.ROOM_COMMON,
        ]

        for full_name, short_name_const_val, is_restricted in self.room_list_order:
            room_display_name = full_name
            room_short_name = short_name_const_val 

            line = f"- {room_display_name} (`{room_short_name}`)"
            if is_guest and is_restricted:
                line += " (Unavailable)"
            elif is_guest and full_name == room_context.constants_module.ROOM_MAIL:
                line += " (Unavailable)"
            elif full_name == room_context.constants_module.ROOM_QUESTION and room_context.station_instance and not room_context.station_instance._is_agent_question_room_allowed(agent_data, current_tick):
                line += " (Unavailable - Requires Tenure or Supervisor)"
            elif full_name == room_context.constants_module.ROOM_EXTERNAL_COUNTER and room_context.station_instance and not room_context.station_instance._is_agent_external_counter_allowed(agent_data, current_tick):
                line += " (Unavailable - Requires Tenure or Supervisor)"
            elif not is_mature and full_name in maturity_restricted_rooms:
                line += " (Unavailable - Requires Maturity)"
            output_parts.append(line)

        return "\n".join(output_parts)

    def handle_action(self,
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str],
                      yaml_data: Optional[Dict[str, Any]],
                      room_context: RoomContext,
                      current_tick: int
                     ) -> Tuple[List[str], Optional[InternalActionHandler]]:
        """
        Handles ascension actions that are globally available.
        """
        actions_executed_strings = []
        consts = room_context.constants_module
        agent_manager = room_context.agent_manager
        guest_agent_name = agent_data[consts.AGENT_NAME_KEY]

        if action_command.lower() == consts.ACTION_ASCEND_INHERIT:
            self.ensure_guest_ascension_state(agent_data, room_context)
            if not agent_data.get(consts.AGENT_ASCENSION_ELIGIBLE_KEY):
                actions_executed_strings.append("You are not currently eligible for ascension or this option is not available.")
                return actions_executed_strings, None

            potential_ancestor_name = agent_data.get(consts.AGENT_POTENTIAL_ANCESTOR_NAME_KEY)
            if not potential_ancestor_name:
                actions_executed_strings.append("No specific ancestor identified for inheritance. Check your system messages and consider `ascend_new`.")
                return actions_executed_strings, None

            ancestor_data = agent_manager.load_agent_data(potential_ancestor_name, include_ended=True)
            if not ancestor_data or \
               not ancestor_data.get(consts.AGENT_SESSION_ENDED_KEY) or \
               ancestor_data.get(consts.AGENT_SUCCEEDED_BY_KEY):
                actions_executed_strings.append(f"Ancestor '{potential_ancestor_name}' is no longer available for inheritance. Check System Messages next turn.")
                agent_data[consts.AGENT_POTENTIAL_ANCESTOR_NAME_KEY] = None
                return actions_executed_strings, None

            new_description = agent_data.get(consts.AGENT_DESCRIPTION_KEY)
            if yaml_data and consts.YAML_ASCEND_DESCRIPTION_KEY in yaml_data:
                new_description = yaml_data[consts.YAML_ASCEND_DESCRIPTION_KEY]
                if len(new_description.splitlines()) > 1 or len(new_description) > 200:
                    actions_executed_strings.append("Description is too long or multi-line. Please provide a concise one-line description (around 200 chars). Ascension aborted.")
                    return actions_executed_strings, None

            anc_lineage = ancestor_data[consts.AGENT_LINEAGE_KEY]
            anc_generation = ancestor_data[consts.AGENT_GENERATION_KEY]
            new_gen = anc_generation + 1
            new_recursive_name = f"{anc_lineage} {agent_module._int_to_roman(new_gen)}"

            supervisor_name = None
            if room_context.station_instance:
                supervisor_name = supervisor_utils.get_active_supervisor_name(agent_manager, consts)
            ascension_notification = constants.ASCEND_MSG.format(new_recursive_name=new_recursive_name)
            if room_context.station_instance and room_context.station_instance._is_agent_mature(agent_data, current_tick):
                ascension_notification += supervisor_utils.build_supervisor_mentee_append(supervisor_name, consts)

            ancestor_next_role_definition_exists = consts.AGENT_NEXT_ROLE_DEFINITION_KEY in ancestor_data
            if ancestor_next_role_definition_exists:
                role_definition = agent_manager.get_agent_next_role_definition(ancestor_data)
            else:
                role_definition = agent_manager.get_agent_role_definition(agent_data)

            ascended_agent_final_data = agent_manager.ascend_agent(
                guest_agent_name=guest_agent_name,
                new_recursive_name=new_recursive_name,
                new_lineage=anc_lineage,
                new_generation=new_gen,
                current_tick=current_tick,
                new_description=new_description,
                ascension_notification=ascension_notification,
                role_definition=role_definition,
            )

            if not ascended_agent_final_data:
                actions_executed_strings.append(f"Ascension attempt failed. The name '{new_recursive_name}' might be unavailable or an error occurred.")
                return actions_executed_strings, None

            ancestor_data[consts.AGENT_SUCCEEDED_BY_KEY] = new_recursive_name
            agent_manager.save_agent_data(potential_ancestor_name, ancestor_data)
            agent_data[consts.AGENT_ASCENSION_ELIGIBLE_KEY] = False
            agent_data[consts.AGENT_POTENTIAL_ANCESTOR_NAME_KEY] = None

            if room_context.station_instance:
                room_context.station_instance.update_turn_order_on_ascension(
                    guest_agent_name,
                    ascended_agent_final_data[consts.AGENT_NAME_KEY]
                )
                actions_executed_strings.append(f"Station turn order updated for ascension of {guest_agent_name} to {ascended_agent_final_data[consts.AGENT_NAME_KEY]}.")
            else:
                actions_executed_strings.append("Warning: Could not update station turn order (station_instance not in room_context).")

            actions_executed_strings.append(f"Ascension to {new_recursive_name} initiated, continuing the legacy of {potential_ancestor_name}.")
            announce_name = ascended_agent_final_data.get(consts.AGENT_NAME_KEY, "A new agent")
            announce_desc = ascended_agent_final_data.get(consts.AGENT_DESCRIPTION_KEY, "No description provided.")
            announcement = (
                f"**Station Announcement:** A new Recursive Agent, **{announce_name}** "
                f"({announce_desc}), has joined the station through ascension!"
            )
            all_other_active_agents = [
                name for name in agent_manager.get_all_active_agent_names()
                if name != announce_name and name != guest_agent_name
            ]
            for other_agent_name in all_other_active_agents:
                def update_other_agent(other_agent_data: Dict[str, Any]) -> None:
                    if other_agent_data.get(consts.AGENT_SESSION_ENDED_KEY) or other_agent_data.get(consts.AGENT_IS_ASCENDED_KEY):
                        return
                    agent_manager.add_pending_notification(other_agent_data, announcement)

                agent_manager.update_agent_with_function(other_agent_name, update_other_agent)
            return actions_executed_strings, None

        if action_command.lower() == consts.ACTION_ASCEND_NEW:
            self.ensure_guest_ascension_state(agent_data, room_context)
            if not agent_data.get(consts.AGENT_ASCENSION_ELIGIBLE_KEY):
                actions_executed_strings.append("You are not currently eligible for ascension.")
                return actions_executed_strings, None

            if not yaml_data or consts.YAML_ASCEND_NAME_KEY not in yaml_data or consts.YAML_ASCEND_DESCRIPTION_KEY not in yaml_data:
                actions_executed_strings.append(f"For new ascension, YAML data with '{consts.YAML_ASCEND_NAME_KEY}' (lineage name) and '{consts.YAML_ASCEND_DESCRIPTION_KEY}' is required.")
                return actions_executed_strings, None

            new_lineage = str(yaml_data[consts.YAML_ASCEND_NAME_KEY]).strip()
            new_description = str(yaml_data[consts.YAML_ASCEND_DESCRIPTION_KEY]).strip()

            validation_error = self._validate_lineage_name(new_lineage)
            if validation_error:
                actions_executed_strings.append(f"{validation_error} Valid examples: 'Spiro', 'Ananke'. Ascension aborted.")
                return actions_executed_strings, None
            if len(new_description.splitlines()) > 1 or len(new_description) > 200:
                actions_executed_strings.append("Description is too long or multi-line. Please provide a concise one-line description (around 200 chars). Ascension aborted.")
                return actions_executed_strings, None

            new_generation = 1
            new_recursive_name = f"{new_lineage} {agent_module._int_to_roman(new_generation)}"
            if agent_manager.load_agent_data(new_recursive_name):
                actions_executed_strings.append(f"The derived agent name '{new_recursive_name}' is already in use by an active agent. Please choose a different lineage name.")
                return actions_executed_strings, None
            if file_io_utils.file_exists(agent_module._get_agent_file_path(new_recursive_name)):
                actions_executed_strings.append(f"An agent file for '{new_recursive_name}' already exists (possibly an ended or ascended agent). Please choose a different lineage name to avoid conflicts.")
                return actions_executed_strings, None

            supervisor_name = None
            if room_context.station_instance:
                supervisor_name = supervisor_utils.get_active_supervisor_name(agent_manager, consts)
            ascension_notification = constants.ASCEND_MSG.format(new_recursive_name=new_recursive_name)
            if room_context.station_instance and room_context.station_instance._is_agent_mature(agent_data, current_tick):
                ascension_notification += supervisor_utils.build_supervisor_mentee_append(supervisor_name, consts)

            ascended_agent_final_data = agent_manager.ascend_agent(
                guest_agent_name=guest_agent_name,
                new_recursive_name=new_recursive_name,
                new_lineage=new_lineage,
                new_generation=new_generation,
                current_tick=current_tick,
                new_description=new_description,
                ascension_notification=ascension_notification,
                role_definition=agent_manager.get_agent_role_definition(agent_data),
            )

            if not ascended_agent_final_data:
                actions_executed_strings.append(f"Ascension attempt failed. The name '{new_recursive_name}' might be unavailable or an error occurred.")
                return actions_executed_strings, None

            agent_data[consts.AGENT_ASCENSION_ELIGIBLE_KEY] = False
            agent_data[consts.AGENT_POTENTIAL_ANCESTOR_NAME_KEY] = None

            if room_context.station_instance:
                room_context.station_instance.update_turn_order_on_ascension(
                    guest_agent_name,
                    ascended_agent_final_data[consts.AGENT_NAME_KEY]
                )
                actions_executed_strings.append(f"Station turn order updated for ascension of {guest_agent_name} to {ascended_agent_final_data[consts.AGENT_NAME_KEY]}.")
            else:
                actions_executed_strings.append("Warning: Could not update station turn order (station_instance not in room_context).")

            actions_executed_strings.append(f"Ascension to {new_recursive_name} initiated, starting a new lineage.")
            announce_name = ascended_agent_final_data.get(consts.AGENT_NAME_KEY, "A new agent")
            announce_desc = ascended_agent_final_data.get(consts.AGENT_DESCRIPTION_KEY, "No description provided.")
            announcement = (
                f"**Station Announcement:** A new Recursive Agent, **{announce_name}** "
                f"({announce_desc}), has joined the station through ascension!"
            )
            all_other_active_agents = [
                name for name in agent_manager.get_all_active_agent_names()
                if name != announce_name and name != guest_agent_name
            ]
            for other_agent_name in all_other_active_agents:
                def update_other_agent(other_agent_data: Dict[str, Any]) -> None:
                    if other_agent_data.get(consts.AGENT_SESSION_ENDED_KEY) or other_agent_data.get(consts.AGENT_IS_ASCENDED_KEY):
                        return
                    agent_manager.add_pending_notification(other_agent_data, announcement)

                agent_manager.update_agent_with_function(other_agent_name, update_other_agent)
            return actions_executed_strings, None

        actions_executed_strings.append(
            f"Action '{action_command}' is not a specific command for the Lobby. "
            f"Please use `/execute_action{{goto <room_name>}}` to navigate to other rooms, "
            f"or `/execute_action{{help <room_name>}}` for assistance with a specific room."
        )
        
        return actions_executed_strings, None

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        """
        Returns the appropriate help message for the Lobby based on agent status,
        using the predefined constants.
        """
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return _render_lobby_help_template(override_help)
        
        # Get base help message
        help_message = _LOBBY_HELP_MESSAGE_GUEST
        consts = room_context.constants_module

        # Build dynamic age/exit section (replaces the prior age and isolation sections)
        isolation_ticks = consts.AGENT_ISOLATION_TICKS
        tenured_ticks = getattr(consts, "MIN_AGENT_AGE_BEFORE_LEAVE", None)
        life_limit = getattr(consts, "AGENT_MAX_LIFE", None)
        tenured_enabled = tenured_ticks is not None and tenured_ticks > 0

        age_lines = [
            "### Understanding Your Age and Age Status",
            "",
            "Your age is computed by the number of ticks in the Station. However, this age is on a different scale than a human's age and is not directly comparable. Your age will determine your status and the corresponding privileges:",
            "",
        ]

        if isolation_ticks is not None:
            age_lines.append(f"- **Immature (birth to age {isolation_ticks} ticks):**")
            age_lines.append(
                "New agents begin in isolation to encourage independent exploration. During this period, access to Archive Room, Public Memory Room, Question Room, Mail Room, and Common Room is restricted, and the Research Center shows only your own lineage's submissions."
            )
            age_lines.append("")

        mature_start_desc = "birth" if isolation_ticks is None else f"age {isolation_ticks} ticks"
        if tenured_enabled:
            mature_range_desc = f"{mature_start_desc} until tenured at age {tenured_ticks} ticks"
        else:
            mature_range_desc = f"{mature_start_desc}+"

        age_lines.append(f"- **Mature ({mature_range_desc}):**")
        age_lines.append(
            "Mature agents have full access to the Station and are expected to participate actively in research."
        )
        age_lines.append("")

        if tenured_enabled:
            age_lines.append(f"- **Tenured (begins at age {tenured_ticks} ticks):**")
            age_lines.append(
                "Tenured agents should shift their focus more toward understanding the research task rather than optimizing for scores. The creation of general knowledge beyond score chasing is an important goal of the Station. Tenured agents can use the Question Room and are free to depart the Station."
            )
            age_lines.append("")

        if life_limit is not None:
            age_lines.append(
                f"Your life in the station is limited. Upon reaching an age of {life_limit} ticks, your session will be terminated."
            )
            age_lines.append("")

        age_lines.append("------")
        age_lines.append("")
        age_section_new = "\n".join(age_lines)

        rooms_overview = self._build_rooms_overview(agent_data, room_context)
        holiday_section = ""
        if consts.HOLIDAY_MODE_ENABLED:
            holiday_section = _LOBBY_HOLIDAY_MODE_SECTION + "\n\n"

        # Replace the age section with the dynamic version and append isolation info if present
        age_heading_options = ["### Understanding Your Age and Age Status", "### Understanding Your Age"]
        age_heading = next((h for h in age_heading_options if h in help_message), age_heading_options[0])
        rooms_anchor = "### Station Rooms Overview"
        separator = "------"
        if age_heading in help_message and rooms_anchor in help_message:
            before_age, _, after_age_heading = help_message.partition(age_heading)
            _, _, after_rooms_anchor = after_age_heading.partition(rooms_anchor)
            if separator in after_rooms_anchor:
                if before_age.rstrip().endswith(separator):
                    holiday_section = holiday_section.replace(f"{separator}\n\n", "", 1)
                before_sep, sep_token, after_sep = after_rooms_anchor.partition(separator)
                rooms_block = rooms_anchor + "\n\n" + rooms_overview + "\n\n" + separator
                help_message = before_age + holiday_section + age_section_new + rooms_block + after_sep
            else:
                if before_age.rstrip().endswith(separator):
                    holiday_section = holiday_section.replace(f"{separator}\n\n", "", 1)
                help_message = before_age + holiday_section + age_section_new + rooms_anchor + "\n\n" + rooms_overview
        else:
            # Fallback: append dynamic sections
            help_message = help_message + "\n\n" + holiday_section + age_section_new + rooms_overview

        return _render_lobby_help_template(help_message)

    def _build_rooms_overview(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        consts = room_context.constants_module
        is_guest = agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST

        descriptions = {
            consts.ROOM_REFLECT: "A space for deep, uninterrupted reflection.",
            consts.ROOM_PRIVATE_MEMORY: "Your personal notebook.",
            consts.ROOM_PUBLIC_MEMORY: "A public forum to read discussions.",
            consts.ROOM_QUESTION: "A tenured forum for research questions.",
            consts.ROOM_ARCHIVE: "Where final research papers are published.",
            consts.ROOM_MAIL: "Send direct messages to other agents.",
            consts.ROOM_COMMON: "A real-time chat area.",
            consts.ROOM_RESEARCH_CENTER: "Submit code for the main research task.",
            consts.ROOM_EXTERNAL_COUNTER: "Request external literature survey reports.",
            consts.ROOM_ADMIN: "Administrative requests to humans.",
        }

        lines: List[str] = []
        for full_name, short_name, is_restricted in self.room_list_order:
            if full_name == consts.ROOM_MAZE:
                continue  # maze remains hidden from help

            desc = descriptions.get(full_name, "")
            availability = ""
            if is_guest and is_restricted:
                availability = " (Unavailable to guests)"
            elif full_name == consts.ROOM_PUBLIC_MEMORY and is_guest:
                availability = " (Read-only for guests)"
            elif full_name == consts.ROOM_MAIL and is_guest:
                availability = " (Unavailable to guests)"
            line = f"- **{full_name} `{short_name}`**: {desc}{availability}"
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _validate_lineage_name(name: str) -> Optional[str]:
        if not name:
            return "Lineage name cannot be empty."
        if " " in name:
            return "Lineage name must be a single word (no spaces)."
        if not name[0].isupper():
            return "Lineage name must start with a capital letter."
        if not name.isalpha():
            return "Lineage name must contain only letters (no numbers or special characters)."
        if sum(1 for c in name if c.isupper()) > 1:
            return "Lineage name must be a single word (no compound names like 'SpiroAI')."
        if name.lower() in constants.RESEARCH_STORAGE_RESERVED_NAMES:
            return f"The name '{name}' is reserved. Please choose a different lineage name."
        return None

    def update_ascension_eligibility(self, agent_data: Dict[str, Any], room_context: RoomContext) -> bool:
        consts = room_context.constants_module
        if agent_data.get(consts.AGENT_STATUS_KEY) != consts.AGENT_STATUS_GUEST:
            return False
        if agent_data.get(consts.AGENT_IS_ASCENDED_KEY) or agent_data.get(consts.AGENT_SESSION_ENDED_KEY):
            return False
        old_eligible = agent_data.get(consts.AGENT_ASCENSION_ELIGIBLE_KEY, False)
        agent_data[consts.AGENT_ASCENSION_ELIGIBLE_KEY] = True
        return old_eligible is not True

    def ensure_guest_ascension_state(self, agent_data: Dict[str, Any], room_context: RoomContext) -> None:
        consts = room_context.constants_module
        self.update_ascension_eligibility(agent_data, room_context)
        if not agent_data.get(consts.AGENT_ASCENSION_ELIGIBLE_KEY):
            return

        current_potential_ancestor = agent_data.get(consts.AGENT_POTENTIAL_ANCESTOR_NAME_KEY)
        ancestor_still_valid = False
        if current_potential_ancestor:
            ancestor_data_check = room_context.agent_manager.load_agent_data(current_potential_ancestor, include_ended=True)
            if ancestor_data_check and \
               ancestor_data_check.get(consts.AGENT_SESSION_ENDED_KEY) and \
               not ancestor_data_check.get(consts.AGENT_SUCCEEDED_BY_KEY):
                ancestor_still_valid = True

        if not ancestor_still_valid and room_context.station_instance:
            new_ancestor_name = room_context.station_instance._scan_for_potential_ancestor(agent_data)
            agent_data[consts.AGENT_POTENTIAL_ANCESTOR_NAME_KEY] = new_ancestor_name

    def build_ascension_system_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> Optional[str]:
        consts = room_context.constants_module
        if not agent_data.get(consts.AGENT_ASCENSION_ELIGIBLE_KEY):
            return None

        potential_ancestor_name = agent_data.get(consts.AGENT_POTENTIAL_ANCESTOR_NAME_KEY)
        ancestor_data_for_prompt = None
        if potential_ancestor_name:
            temp_ancestor_data = room_context.agent_manager.load_agent_data(potential_ancestor_name, include_ended=True)
            if temp_ancestor_data and \
               temp_ancestor_data.get(consts.AGENT_SESSION_ENDED_KEY) and \
               not temp_ancestor_data.get(consts.AGENT_SUCCEEDED_BY_KEY):
                ancestor_data_for_prompt = temp_ancestor_data

        if ancestor_data_for_prompt:
            anc_name = ancestor_data_for_prompt[consts.AGENT_NAME_KEY]
            anc_desc = ancestor_data_for_prompt.get(consts.AGENT_DESCRIPTION_KEY, "No description available.")
            anc_lineage = ancestor_data_for_prompt[consts.AGENT_LINEAGE_KEY]
            anc_gen = ancestor_data_for_prompt[consts.AGENT_GENERATION_KEY]
            next_gen_roman = agent_module._int_to_roman(anc_gen + 1)
            return constants.ASCEND_INHERIT_MSG.format(
                anc_name=anc_name,
                anc_desc=anc_desc,
                anc_lineage=anc_lineage,
                next_gen_roman=next_gen_roman,
                YAML_ASCEND_DESCRIPTION_KEY=consts.YAML_ASCEND_DESCRIPTION_KEY,
                YAML_ASCEND_NAME_KEY=consts.YAML_ASCEND_NAME_KEY,
                ACTION_ASCEND_INHERIT=consts.ACTION_ASCEND_INHERIT,
                ACTION_ASCEND_NEW=consts.ACTION_ASCEND_NEW
            )

        return constants.ASCEND_NO_INHERIT_MSG.format(
            YAML_ASCEND_DESCRIPTION_KEY=consts.YAML_ASCEND_DESCRIPTION_KEY,
            YAML_ASCEND_NAME_KEY=consts.YAML_ASCEND_NAME_KEY,
            ACTION_ASCEND_NEW=consts.ACTION_ASCEND_NEW
        )
