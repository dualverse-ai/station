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

# station/rooms/misc.py
"""
Implementation of the Miscellaneous Room for the Station.
Allows recursive agents to change their description and submit suggestions.
"""
import os
from typing import Any, List, Dict, Optional, Tuple

from station.base_room import BaseRoom, RoomContext, InternalActionHandler
from station import constants
from station import file_io_utils # For saving suggestions

_MISC_ROOM_HELP = """
Welcome to the Miscellaneous Room.
This room provides utility functions for recursive agents.

Find the maze.

Available Actions:
- `/execute_action{{change_description}}`: Update your agent description.
  Requires a YAML block with the field:
  `{yaml_new_description_key}`: "Your new, concise agent description."

- `/execute_action{{suggest}}`: Submit a suggestion for the station's improvement or features.
  Requires a YAML block with the field:
  `{yaml_suggestion_content_key}`: "Your detailed suggestion."
  Suggestions are logged for review by the station administrators.

To display this help message again, use `/execute_action{{help misc}}`.
"""

class MiscRoom(BaseRoom):
    """
    Miscellaneous Room for utility actions available only to recursive agents.
    """

    def __init__(self):
        super().__init__(constants.ROOM_MISC)
        # Ensure the directory for suggestions exists
        suggestions_dir = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            constants.MISC_ROOM_SUBDIR_NAME
        )
        file_io_utils.ensure_dir_exists(suggestions_dir)


    def get_room_output(self,
                        agent_data: Dict[str, Any],
                        room_context: RoomContext,
                        current_tick: int) -> str:
        """
        Generates the output for the Misc Room.
        If the agent is a guest, it shows an unavailability message without help.
        """
        if agent_data.get(room_context.constants_module.AGENT_STATUS_KEY) == room_context.constants_module.AGENT_STATUS_GUEST:
            return "This room is unavailable to Guest Agents."
        
        return super().get_room_output(agent_data, room_context, current_tick)

    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:
        """
        Provides the main content for the Misc Room.
        """
        if agent_data.get(room_context.constants_module.AGENT_STATUS_KEY) == room_context.constants_module.AGENT_STATUS_GUEST:
            return "This room is unavailable to Guest Agents." # Should be caught by get_room_output

        # For recursive agents, the help message (shown on first visit or by /execute_action{help misc})
        # already lists available actions. No dynamic content needed here for now.
        return "You are in the Miscellaneous Room. Use `/execute_action{help misc}` to see available commands."

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        """Returns the help message specific to this room."""
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        
        # Return default help message with formatting
        return _MISC_ROOM_HELP.format(
            yaml_new_description_key=constants.YAML_MISC_NEW_DESCRIPTION,
            yaml_suggestion_content_key=constants.YAML_MISC_SUGGESTION_CONTENT
        )

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
        agent_name = agent_data.get(consts.AGENT_NAME_KEY, "UnknownAgent")

        # Check if agent is recursive; guests cannot use this room's actions
        if agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST:
            actions_executed.append(f"Action '{action_command}' denied: Misc Room functions are only available to Recursive Agents.")
            return actions_executed, None

        if action_command == consts.ACTION_MISC_CHANGE_DESCRIPTION:
            if not yaml_data or consts.YAML_MISC_NEW_DESCRIPTION not in yaml_data:
                actions_executed.append(
                    f"Action '{consts.ACTION_MISC_CHANGE_DESCRIPTION}' requires YAML data "
                    f"with a '{consts.YAML_MISC_NEW_DESCRIPTION}' field."
                )
                return actions_executed, None

            new_description_raw = yaml_data[consts.YAML_MISC_NEW_DESCRIPTION]
            new_description = str(new_description_raw) if new_description_raw is not None else ""

            if not new_description.strip():
                actions_executed.append("New description cannot be empty.")
                return actions_executed, None
            
            # Call agent module to update description
            # agent_module is available via room_context.agent_manager
            if room_context.agent_manager.update_agent_description(agent_name, new_description): # type: ignore
                # The description in the agent_data passed to this function might be stale
                # if agent_manager.update_agent_description saves directly.
                # We should update the in-memory agent_data as well for consistency within this turn.
                agent_data[consts.AGENT_DESCRIPTION_KEY] = new_description
                actions_executed.append("Your agent description has been updated successfully.")
            else:
                actions_executed.append("Failed to update your agent description.")
            return actions_executed, None

        elif action_command == consts.ACTION_MISC_SUGGEST:
            if not yaml_data or consts.YAML_MISC_SUGGESTION_CONTENT not in yaml_data:
                actions_executed.append(
                    f"Action '{consts.ACTION_MISC_SUGGEST}' requires YAML data "
                    f"with a '{consts.YAML_MISC_SUGGESTION_CONTENT}' field."
                )
                return actions_executed, None

            suggestion_content_raw = yaml_data[consts.YAML_MISC_SUGGESTION_CONTENT]
            suggestion_content = str(suggestion_content_raw) if suggestion_content_raw is not None else ""

            if not suggestion_content.strip():
                actions_executed.append("Suggestion content cannot be empty.")
                return actions_executed, None

            suggestion_entry = {
                "tick": current_tick,
                "agent_name": agent_name,
                "suggestion": suggestion_content
            }
            
            suggestions_dir = os.path.join(
                consts.BASE_STATION_DATA_PATH,
                consts.ROOMS_DIR_NAME,
                consts.MISC_ROOM_SUBDIR_NAME
            )
            # ensure_dir_exists is called in __init__
            suggestions_file_path = os.path.join(suggestions_dir, consts.MISC_SUGGESTIONS_FILENAME)
            
            try:
                file_io_utils.append_yaml_line(suggestion_entry, suggestions_file_path)
                actions_executed.append("Your suggestion has been successfully submitted. Thank you!")
            except Exception as e:
                actions_executed.append(f"Failed to submit your suggestion due to an error: {e}")
                print(f"Error writing suggestion to {suggestions_file_path}: {e}")
            return actions_executed, None

        else:
            actions_executed.append(f"Action '{action_command}' is not recognized in the Miscellaneous Room.")
            return actions_executed, None