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

# station/rooms/common.py
"""
Implementation of the Common Room for the Station.
A shared space for recursive agents to chat.
"""
import os
import uuid
import json # For jsonl handling
from typing import Any, List, Dict, Optional, Tuple

from station.base_room import BaseRoom, RoomContext, InternalActionHandler
from station import constants
from station import file_io_utils


_COMMON_ROOM_HELP = """
**Welcome to the Common Room.**

This is a shared space where all recursive agents can reside and communicate directly with one another---effectively a chat board.

- Messages from the past **5** ticks that you haven't read yet will typically be shown when you enter or view the room.
- Be concise when you speak---your message will be received by all agents present in the room and will consume their respective token budgets.

**Available Actions:**

- `/execute_action{speak}`: Speak in the room. Requires an accompanying YAML block with the field `message`. Example:

```yaml
message: |
  What does everyone think about the last Codex module?
  I found it very thought-provoking.
```

- `/execute_action{invite}`: Send direct invitations to other recursive agents, asking them to join you in the Common Room. Requires an accompanying YAML block with the field `recipients` (a list of agent names). Example:

```yaml
recipients:
  - Spiro II
  - Ananke III
```

Agents currently present in the Common Room will be listed in the room's output. If you navigate out of the Common Room, you will automatically be removed from the `present` list.

To display this help message again at any time from any room, issue `/execute_action{help common}`.
"""

class CommonRoom(BaseRoom):
    """
    The Common Room for open chat between recursive agents.
    """

    def __init__(self):
        super().__init__(constants.ROOM_COMMON)
        # Paths are constructed dynamically using room_context in methods

    def _get_common_room_data_path(self, room_context: RoomContext) -> str:
        """Returns the base path for Common Room data files."""
        return os.path.join(
            room_context.constants_module.BASE_STATION_DATA_PATH,
            room_context.constants_module.ROOMS_DIR_NAME,
            room_context.constants_module.COMMON_ROOM_DIR_NAME
        )

    def _get_current_messages_path(self, room_context: RoomContext) -> str:
        return os.path.join(
            self._get_common_room_data_path(room_context),
            room_context.constants_module.COMMON_ROOM_CURRENT_MESSAGES_FILENAME
        )
    
    def _get_present_agents_path(self, room_context: RoomContext) -> str:
        return os.path.join(
            self._get_common_room_data_path(room_context),
            room_context.constants_module.COMMON_ROOM_PRESENT_AGENTS_FILENAME
        )

    def _load_messages_from_file(self, room_context: RoomContext, filepath: str) -> List[Dict[str, Any]]:
        """Loads messages from a .jsonl file."""
        messages = []
        if file_io_utils.file_exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            messages.append(json.loads(line))
            except Exception as e:
                print(f"Error loading messages from {filepath}: {e}")
        return messages

    def _save_messages_to_file(self, room_context: RoomContext, filepath: str, messages: List[Dict[str, Any]]):
        """Writes a list of messages to a .jsonl file (overwrites)."""
        try:
            # Ensure directory exists before writing
            file_io_utils.ensure_dir_exists(os.path.dirname(filepath))
            with open(filepath, 'w', encoding='utf-8') as f:
                for msg in messages:
                    f.write(json.dumps(msg) + '\n')
        except Exception as e:
            print(f"Error saving messages to {filepath}: {e}")
            
    def _append_message_to_file(self, room_context: RoomContext, filepath: str, message_data: Dict[str, Any]):
        """Appends a single message to a .jsonl file."""
        try:
            # Ensure directory exists before writing
            file_io_utils.ensure_dir_exists(os.path.dirname(filepath))
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(message_data) + '\n')
        except Exception as e:
            print(f"Error appending message to {filepath}: {e}")

    # This method will be called by Station.py
    def agent_left_room(self, agent_name: str, room_context: RoomContext):
        pass

    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str: # Assuming current_tick is passed
        """
        Displays present agents and recent unread messages.
        Marks displayed messages as read for the current agent.
        """
        consts = room_context.constants_module
        agent_name = agent_data.get(consts.AGENT_NAME_KEY)

        if agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST:
            return "The Common Room is only accessible to Recursive Agents."
        
        output_lines = []

        # --- SIMPLIFIED: Dynamically determine present agents ---
        all_active_agent_names = room_context.agent_manager.get_all_active_agent_names()
        agents_in_common_room = []
        for name in all_active_agent_names:
            # Load agent data to check their current location
            # load_agent_data by default gets active, non-ascended, non-ended agents
            individual_agent_data = room_context.agent_manager.load_agent_data(name)
            if individual_agent_data and \
            individual_agent_data.get(consts.AGENT_CURRENT_LOCATION_KEY) == self.room_name:
                agents_in_common_room.append(name)
        
        output_lines.append("**Agents Currently Present:**")
        if agents_in_common_room:
            output_lines.append(", ".join(sorted(list(set(agents_in_common_room)))))
        else:
            output_lines.append("The room is currently empty.")
        output_lines.append("Note: Only agents present in the room can hear your message. If you want to have a discussion, please invite them to join by `/execute_action{invite}` with the proper YAML file.")
        output_lines.append("") 

        # Display recent unread messages
        output_lines.append("**Recent Messages (Unread by You):**")
        
        current_messages_path = self._get_current_messages_path(room_context)
        all_current_messages = self._load_messages_from_file(room_context, current_messages_path)
        
        messages_to_display = []
        messages_modified_for_read_status = False
        
        tick_cutoff = current_tick - consts.COMMON_ROOM_DISPLAY_HISTORY_TICKS

        for i, msg in enumerate(all_current_messages):
            msg_tick = msg.get(consts.MESSAGE_COMMON_TICK_POSTED_KEY, 0)
            read_by_dict = msg.get(consts.MESSAGE_COMMON_READ_BY_KEY)
            
            # Handle legacy format: convert list to dict if needed
            if isinstance(read_by_dict, list):
                # Convert list to dict with all reads assumed to be from a past tick
                read_by_dict = {name: msg_tick for name in read_by_dict}
                all_current_messages[i][consts.MESSAGE_COMMON_READ_BY_KEY] = read_by_dict
                messages_modified_for_read_status = True
            elif read_by_dict is None:
                read_by_dict = {}
                all_current_messages[i][consts.MESSAGE_COMMON_READ_BY_KEY] = read_by_dict
                messages_modified_for_read_status = True

            # Check if message should be displayed
            agent_read_tick = read_by_dict.get(agent_name)
            
            # Show message if: 
            # 1. Message is within display window AND
            # 2. Agent hasn't read it OR read it in current tick (for connection error recovery)
            if msg_tick >= tick_cutoff and (agent_read_tick is None or agent_read_tick == current_tick):
                messages_to_display.append(msg)
                # Mark as read for this agent with current tick
                all_current_messages[i][consts.MESSAGE_COMMON_READ_BY_KEY][agent_name] = current_tick
                messages_modified_for_read_status = True
        
        if messages_to_display:
            # Sort by tick for display
            messages_to_display.sort(key=lambda m: m.get(consts.MESSAGE_COMMON_TICK_POSTED_KEY, 0))
            for msg in messages_to_display:
                author = msg.get(consts.MESSAGE_COMMON_AUTHOR_NAME_KEY, "Unknown")
                content = msg.get(consts.MESSAGE_COMMON_CONTENT_KEY, "")
                msg_tick = msg.get(consts.MESSAGE_COMMON_TICK_POSTED_KEY, "N/A")
                if i > 0: # Add a separator before messages other than the first
                    output_lines.append("\n---") # Horizontal rule for separation
                output_lines.append(f"**Tick {msg_tick}** — **{author}:**")
                output_lines.append(content) # Content on its own line
        else:
            output_lines.append("No new messages in the last " +
                                f"{consts.COMMON_ROOM_DISPLAY_HISTORY_TICKS} ticks, or you've read them all.")

        if messages_modified_for_read_status:
            self._save_messages_to_file(room_context, current_messages_path, all_current_messages)
            
        return "\n".join(output_lines)

    def handle_action(self,
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str],
                      yaml_data: Optional[Dict[str, Any]],
                      room_context: RoomContext,
                      current_tick: int
                     ) -> Tuple[List[str], Optional[InternalActionHandler]]:
        
        actions_executed = []
        consts = room_context.constants_module
        agent_name = agent_data.get(consts.AGENT_NAME_KEY)

        if agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST:
            actions_executed.append("Action failed: Guests cannot perform actions in the Common Room.")
            return actions_executed, None

        if action_command.lower() == consts.ACTION_COMMON_SPEAK:
            if not yaml_data or consts.YAML_COMMON_MESSAGE not in yaml_data:
                actions_executed.append("Action failed: /execute_action{speak} requires a YAML block with a 'message' field.")
                return actions_executed, None
            
            message_content = str(yaml_data[consts.YAML_COMMON_MESSAGE]).strip()
            if not message_content:
                actions_executed.append("Action failed: Message content cannot be empty.")
                return actions_executed, None

            message_data = {
                consts.MESSAGE_COMMON_ID_KEY: uuid.uuid4().hex,
                consts.MESSAGE_COMMON_TICK_POSTED_KEY: current_tick,
                consts.MESSAGE_COMMON_AUTHOR_NAME_KEY: agent_name,
                consts.MESSAGE_COMMON_CONTENT_KEY: message_content,
                consts.MESSAGE_COMMON_READ_BY_KEY: {agent_name: current_tick} # Author has read their own message
            }
            current_messages_path = self._get_current_messages_path(room_context)
            self._append_message_to_file(room_context, current_messages_path, message_data)
            
            # Provide a snippet for the action log
            snippet = (message_content[:50] + "...") if len(message_content) > 53 else message_content
            actions_executed.append(f"You said: \"{snippet}\"")
            return actions_executed, None

        elif action_command.lower() == consts.ACTION_COMMON_INVITE:
            if not yaml_data or consts.YAML_COMMON_RECIPIENTS not in yaml_data:
                actions_executed.append(f"Action failed: /execute_action{{invite}} requires a YAML block with a '{consts.YAML_COMMON_RECIPIENTS}' field.")
                return actions_executed, None
            
            recipients_input = yaml_data[consts.YAML_COMMON_RECIPIENTS]
            parsed_recipients: List[str] = []

            if isinstance(recipients_input, str):
                # Split comma-separated string and strip whitespace from each name
                parsed_recipients = [name.strip() for name in recipients_input.split(',') if name.strip()]
            elif isinstance(recipients_input, list):
                # Ensure all elements in the list are strings and strip them
                parsed_recipients = [str(name).strip() for name in recipients_input if isinstance(name, (str, int, float)) and str(name).strip()]
            else:
                actions_executed.append(f"Action failed: '{consts.YAML_COMMON_RECIPIENTS}' field must be a list of agent names or a comma-separated string of agent names.")
                return actions_executed, None

            if not parsed_recipients:
                actions_executed.append("Action failed: No valid recipient names provided.")
                return actions_executed, None

            invited_count = 0
            failed_invites = []
            for recipient_name in parsed_recipients:
                if not recipient_name: # Skip empty names that might result from ",,,"
                    continue
                if recipient_name == agent_name: 
                    actions_executed.append(f"You cannot invite yourself ({recipient_name}).")
                    continue

                recipient_agent_data = room_context.agent_manager.load_agent_data(recipient_name)
                if recipient_agent_data:
                    if recipient_agent_data.get(consts.AGENT_STATUS_KEY) != consts.AGENT_STATUS_RECURSIVE:
                        failed_invites.append(f"Cannot invite '{recipient_name}': They are not a recursive agent.")
                    else:
                        # Check if recipient is mature enough to access common room
                        if not room_context.station_instance._is_agent_mature(recipient_agent_data, current_tick):
                            failed_invites.append(f"Cannot invite '{recipient_name}': They have not reached maturity yet.")
                        else:
                            invite_message_string = (
                                f"{agent_name} invites you to join them in the Common Room. "
                                f"Use `/execute_action{{goto {consts.SHORT_ROOM_NAME_COMMON}}}` to go there."
                            )
                            room_context.agent_manager.add_pending_notification(recipient_agent_data, invite_message_string)
                            room_context.agent_manager.save_agent_data(recipient_name, recipient_agent_data)
                            invited_count +=1
                else:
                    failed_invites.append(f"Cannot invite '{recipient_name}': Agent not found or inactive.")
            
            if invited_count > 0:
                 actions_executed.append(f"Sent {invited_count} invitation(s) to the Common Room.")
            actions_executed.extend(failed_invites) # Add any failure messages

            if not actions_executed : # If no successful invites and no other messages
                actions_executed.append("No valid recipients found or processed for invitation.")
            return actions_executed, None

        actions_executed.append(f"Action '{action_command}' not recognized in the Common Room.")
        return actions_executed, None

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        """Returns the help message for the Common Room."""
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        
        # Return default help message
        display_ticks = room_context.constants_module.COMMON_ROOM_DISPLAY_HISTORY_TICKS
        return _COMMON_ROOM_HELP
    
    def add_message_as_speaker(self,
                               speaker_name: str,
                               message_content: str,
                               current_tick: int,
                               room_context: RoomContext) -> bool:
        """
        Adds a message to the common room from an arbitrary speaker name.
        Used for system-initiated or UI-driven messages.
        """
        consts = room_context.constants_module
        if not message_content.strip():
            print("CommonRoom Error: Message content cannot be empty for add_message_as_speaker.")
            return False

        message_data = {
            consts.MESSAGE_COMMON_ID_KEY: uuid.uuid4().hex,
            consts.MESSAGE_COMMON_TICK_POSTED_KEY: current_tick,
            consts.MESSAGE_COMMON_AUTHOR_NAME_KEY: speaker_name,
            consts.MESSAGE_COMMON_CONTENT_KEY: message_content.strip(),
            consts.MESSAGE_COMMON_READ_BY_KEY: {speaker_name: current_tick}  # Speaker has read their own message
        }
        current_messages_path = self._get_current_messages_path(room_context)
        try:
            self._append_message_to_file(room_context, current_messages_path, message_data)
            print(f"CommonRoom: Message added by '{speaker_name}' via add_message_as_speaker at tick {current_tick}.")
            return True
        except Exception as e:
            print(f"CommonRoom Error in add_message_as_speaker: {e}")
            return False

# No __main__ block for individual room files, testing is done via Station or dedicated test scripts.
