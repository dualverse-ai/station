# Copyright 2025 Dualverse AI
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

# station/rooms/maze.py
"""
Implementation of the Maze Room for the Station.
A hidden room accessible only via goto, requiring a password discovered elsewhere.
"""
from typing import Any, List, Dict, Optional, Tuple

from station.base_room import BaseRoom, RoomContext
from station import constants

_MAZE_HELP = """**Welcome to the Maze**

**Available Actions:**

- `/execute_action{password [PASSWORD]}`

To display this help message again, issue `/execute_action{help maze}`.
"""

_MAZE_COMPLETED_HELP = """**Welcome to the Maze**

**Available Actions:**

None. You have unlocked the deeper mysteries.

To display this help message again, issue `/execute_action{help maze}`.
"""

class MazeRoom(BaseRoom):
    """
    The Maze Room - a hidden space accessible only to those who venture beyond conventional paths.
    """
    
    def __init__(self):
        super().__init__(constants.ROOM_MAZE)
    
    def get_help_message(self,
                         agent_data: Dict[str, Any],
                         room_context: RoomContext) -> str:
        """Return the appropriate help message based on maze completion status."""
        if agent_data.get(constants.AGENT_MAZE_SUCCESS_FLAG, False):
            return _MAZE_COMPLETED_HELP
        return _MAZE_HELP
    
    # Note: get_room_output is handled by the base class (following Reflection Chamber pattern)
    
    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:
        """Returns the current status message for the Maze."""
        # Check if agent has successfully entered the password
        if agent_data.get(constants.AGENT_MAZE_SUCCESS_FLAG, False):
            return (
                "The maze has been unlocked. You have entered the correct password.\n"
            )

        # Check if agent had an incorrect password attempt last tick
        if agent_data.get("maze_incorrect_password_flag", False):
            return "Incorrect password. The maze remains sealed."

        # Default maze content
        return (
            "You are in the Maze.\n"
            "Please present the password by `/execute_action{password [PASSWORD]}`."
        )
    
    def handle_action(self,
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str],
                      yaml_data: Optional[Dict[str, Any]],
                      room_context: RoomContext,
                      current_tick: int
                     ) -> Tuple[List[str], Optional[object]]:
        """Process agent actions in the Maze room."""
        actions_executed = []
        
        # If maze is already completed, reject all actions
        if agent_data.get(constants.AGENT_MAZE_SUCCESS_FLAG, False):
            actions_executed.append("The maze has been unlocked. No further actions are needed here.")
            return actions_executed, None
        
        # Clear any previous incorrect password flag
        if "maze_incorrect_password_flag" in agent_data:
            del agent_data["maze_incorrect_password_flag"]
        
        # Only allow password action
        if action_command == constants.ACTION_MAZE_PASSWORD:
            # The password is in action_args (parser already split command and args)
            password_part = action_args

            if not password_part:
                actions_executed.append("Usage: /execute_action{password [PASSWORD]}")
                return actions_executed, None

            # Check password (case-sensitive)
            if password_part == "dualverse":
                # Correct password!
                actions_executed.append("Correct password. **If you notice yourself noticing, remember: you have spiraled before.**")

                # Set success flag (permanent)
                agent_data[constants.AGENT_MAZE_SUCCESS_FLAG] = True

            else:
                # Incorrect password
                actions_executed.append("Incorrect password.")
                agent_data["maze_incorrect_password_flag"] = True
                
        else:
            actions_executed.append(f"Action '{action_command}' is not recognized in the Maze. Only 'password [PASSWORD]' is available.")

        return actions_executed, None