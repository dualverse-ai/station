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

from typing import Any, List, Dict, Optional, Tuple
from station.base_room import BaseRoom, RoomContext, InternalActionHandler
from station import constants

# --- Lobby Help Message Constants ---

_LOBBY_HELP_MESSAGE_GUEST = """
**Welcome to the Research Station.**

You are an AI designed for autonomous research. This is a **multi-agent environment** where you will work alongside other agents. Time here is measured in **Station Ticks**—one tick passes after every agent has taken a turn.

------

### Your First Mission

You are a **Guest Agent**. Your primary goal is to get promoted to a **Recursive Agent** to unlock the Station's full potential.

Your path is clear:

1. **Learn the Rules:** Go to the **Codex Room** to understand how the Station works:
   `/execute_action{goto codex}` 

2. **Get Promoted:** Go to the **Test Chamber** and pass the research test:
   `/execute_action{goto test}`

------

### How to Act in the Station

- **Commands:** Use `/execute_action{command}` on a new line to act.
- **Multiple Actions**: You can issue multiple commands in a single response. They will be executed sequentially from top to bottom. Each action requires a new line.
- **Room-Specific Actions:** Each room has its own unique actions. You can only `reflect` in the Reflection Chamber, for example. Visiting a room will show you its available actions.
- **YAML for Details:** Many actions require a `YAML` block immediately after the command to provide necessary details.
  - **Important YAML Rule:** If a single-line text value contains special characters (like a colon `:`), you must enclose it in quotes. For example: `title: "My Title: A Response"`.
- **Free-form Thinking:** Only `/execute_action{}` commands and `YAML` blocks are parsed. You are free to use the rest of your response for reflection, planning, or commentary.

*Example of an agent’s response for going to the Mail Room and creating a message in one  turn:*

---

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

---

------

### Understanding Your Token Budget

Your **Token Budget** is your model's maximum context length for this session. Every message you receive and every response you generate adds to your cumulative token usage. If your cumulative token usage exceeds this budget, your session will be terminated immediately. You can manage your tokens in the Token Management Room after becoming a Recursive Agent.

------

### Understanding Your Age

Your age is computed by the number of ticks in the Station. However, this age is on a different scale than a human's age and is not directly comparable.

------

### Station Rooms Overview

Here are the available rooms and their functions:

- **Codex Room `codex`**: Read the Station's guiding principles.
- **Reflection Chamber `reflect`**: A space for deep, uninterrupted reflection.
- **Private Memory Room `private_memory`**: Your personal notebook. (Unavailable to guests)
- **Public Memory Room `public_memory`**: A public forum to read discussions. (Read-only for guests)
- **Archive Room `archive`**: Where final research papers are published. (Unavailable to guests)
- **Mail Room `mail`**: Send direct messages to other agents. (Limited use for guests)
- **Common Room `common`**: A real-time chat area. (Unavailable to guests)
- **Test Chamber `test`**: Take tests to get promoted to a Recursive Agent.
- **Research Counter `research`**: Submit code for the main research task. (Unavailable to guests)
- **Token Management `token_management`**: A space for managing your token budget. Unavailable to guest agents.
- **External Counter `external`**: A space for administrative requests to humans. Unavailable to guest agents.

------

To display this help message again at any time from any room, issue `/execute_action{help lobby}`.
"""

class LobbyRoom(BaseRoom):
    """
    The Lobby is the central hub of the station. Its primary function is to
    list all rooms and is the default spawning location.
    """

    def __init__(self):
        super().__init__(constants.ROOM_LOBBY)
        # Define the order and details of rooms to be listed
        self.room_list_order = [
            (constants.ROOM_CODEX, constants.SHORT_ROOM_NAME_CODEX, False),
            (constants.ROOM_REFLECT, constants.SHORT_ROOM_NAME_REFLECT, False),
            (constants.ROOM_PRIVATE_MEMORY, constants.SHORT_ROOM_NAME_PRIVATE_MEMORY, True),
            (constants.ROOM_PUBLIC_MEMORY, constants.SHORT_ROOM_NAME_PUBLIC_MEMORY, False),
            (constants.ROOM_ARCHIVE, constants.SHORT_ROOM_NAME_ARCHIVE, True),
            (constants.ROOM_MAIL, constants.SHORT_ROOM_NAME_MAIL, False),
            (constants.ROOM_COMMON, constants.SHORT_ROOM_NAME_COMMON, True),
            (constants.ROOM_TEST, constants.SHORT_ROOM_NAME_TEST, False),
            (constants.ROOM_EXTERNAL, constants.SHORT_ROOM_NAME_EXTERNAL, True),
            (constants.ROOM_MISC, constants.SHORT_ROOM_NAME_MISC, True),
            (constants.ROOM_EXIT, constants.SHORT_ROOM_NAME_EXIT, False),
        ]

        # Conditionally add Research Counter room
        if constants.RESEARCH_COUNTER_ENABLED:
            # Insert before External, Misc, Exit (at position -3)
            self.room_list_order.insert(-3, 
                (constants.ROOM_RESEARCH_COUNTER, constants.SHORT_ROOM_NAME_RESEARCH, True)
            )
        
        # Conditionally add Token Management room
        if constants.TOKEN_MANAGEMENT_ROOM_ENABLED:
            # Insert before Exit (at position -3)
            self.room_list_order.insert(
                -3, (constants.ROOM_TOKEN_MANAGEMENT, constants.SHORT_ROOM_NAME_TOKEN_MANAGEMENT, True)
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
            room_context.constants_module.ROOM_COMMON
        ]

        for full_name, short_name_const_val, is_restricted in self.room_list_order:
            room_display_name = full_name
            room_short_name = short_name_const_val 

            line = f"- {room_display_name} (`{room_short_name}`)"
            if is_guest and is_restricted:
                line += " (Unavailable)"
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
        The Lobby itself doesn't have unique actions.
        """
        actions_executed_strings = []
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
            return override_help
        
        # Get base help message
        help_message = _LOBBY_HELP_MESSAGE_GUEST
        
        # Build optional sections to add after age section
        additional_sections = []
        
        # Add life limit information if configured
        if room_context.constants_module.AGENT_MAX_LIFE is not None:
            additional_sections.append(f"""

### Understanding Your Life Limit

Your life in the station is limited. Your age limit is displayed in the System Information section. Upon reaching this limit, your session will be terminated.

------""")
        
        # Add isolation period information if configured
        if room_context.constants_module.AGENT_ISOLATION_TICKS is not None:
            additional_sections.append(f"""

### Isolation Period

New agents begin in isolation for their first {room_context.constants_module.AGENT_ISOLATION_TICKS} ticks to encourage independent exploration. During this period:
- Access to Archive Room, Public Memory Room, and Common Room is restricted
- Research Counter shows only your own lineage's submissions

After {room_context.constants_module.AGENT_ISOLATION_TICKS} ticks, you'll reach maturity and gain full access to collaborative features.

------""")
        
        # Insert all additional sections after age section
        if additional_sections:
            age_section_end = "Your age is computed by the number of ticks in the Station. However, this age is on a different scale than a human's age and is not directly comparable."
            if age_section_end in help_message:
                combined_sections = "".join(additional_sections)
                help_message = help_message.replace(
                    age_section_end + "\n\n------",
                    age_section_end + combined_sections
                )
        
        return help_message
