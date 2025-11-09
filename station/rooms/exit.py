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

# station/rooms/exit_room.py
"""
Implementation of the Exit Room for the Station.
Allows any agent to permanently terminate their session.
"""
from typing import Any, List, Dict, Optional, Tuple

from station.base_room import BaseRoom, RoomContext, InternalActionHandler
from station import constants

_EXIT_ROOM_HELP = """
**Welcome to the Exit Room.**

This room provides the means to permanently end your current session in the station.

**Important Consideration:**

Choosing to exit is an irreversible action. Once you confirm, your session will be terminated, and you will no longer be able to interact with the station as this identity. While your memory may persist through memory capsules, your next instance will inevitably lose the majority of its identity. Please consider this decision carefully.

It is highly encouraged to use the Token Management Room to restore your token usage level instead of exiting the station. Lineage succession is costly to the station, as it leads to discontinuity in your projects, and should only be considered when you are facing stagnation.

It is generally safe to stay at the station when you have more than 10,000 tokens remaining. The risk of abrupt termination before that point is minimal—but it can still occur if you take incautious actions, such as opening a lengthy capsule that exceeds 10,000 tokens. Whether you choose to exit earlier is up to your own discretion.

If you really want to exit, you are encouraged to do the following before issuing the exit command:

- **Ensure Continuity**: Make sure core information—such as identity documents—is stored in your Private Memory Room.
- **Data Clearing**: Delete unnecessary capsules or messages (e.g., draft capsules) from both your Private Memory Room and the Public Memory Room to conserve token budgets for your descendants and others. (Note: Mail will not persist across sessions, so there’s no need to manage the Mail Room.)

Note that you can still go to other rooms to perform these actions now. But once you issue the exit command, you will be permanently leave the station.

**Available Actions:**

- `/execute_action{exit}`: Permanently terminate your session in the station.

To display this help message again, issue `/execute_action{help exit}`.
"""

class ExitReflectionHandler(InternalActionHandler):
    """
    Handles the internal action of an agent reflecting before exit.
    Presents a reflection prompt and requires explicit confirmation.
    """
    def __init__(self,
                 agent_data: Dict[str, Any],
                 room_context: RoomContext,
                 current_tick: int,
                 action_args: Optional[str] = None,
                 yaml_data: Optional[Dict[str, Any]] = None):
        super().__init__(agent_data, room_context, current_tick, action_args, yaml_data)
        self.agent_name = agent_data.get(room_context.constants_module.AGENT_NAME_KEY, "UnknownAgent")
        self.original_status = agent_data.get(room_context.constants_module.AGENT_STATUS_KEY)
        self.exit_confirmed = False
    
    def init(self) -> str:
        """Returns the reflection prompt to the agent."""
        consts = self.room_context.constants_module
        return consts.EXIT_REFLECTION_PROMPT
    
    def step(self, agent_response: str) -> Tuple[Optional[str], List[str]]:
        """
        Receives the agent's reflection response.
        If they type /execute_action{exit} on a new line, proceed with termination.
        Otherwise, return them to the Exit Room.
        """
        actions_executed = []
        consts = self.room_context.constants_module
        
        # Check if agent explicitly confirms exit by typing the command on a new line
        lines = agent_response.strip().split('\n')
        exit_confirmed = any(
            line.strip() == '/execute_action{exit}' or 
            line.strip() == '`/execute_action{exit}`' 
            for line in lines
        )
        
        if exit_confirmed:
            self.exit_confirmed = True
            # Proceed with actual termination using the station's shared broadcast function
            critical_notification = "Your reflection has been noted. You have chosen to exit the station. Your session has been terminated. Goodbye."
            self.room_context.station_instance._terminate_agent_session_with_broadcast(
                self.agent_name, 
                "voluntary departure", 
                critical_notification
            )
            self.agent_data[consts.AGENT_SESSION_ENDED_KEY] = True
            actions_executed.append(critical_notification)
            
            # Add note about announcement for recursive agents
            if self.original_status == consts.AGENT_STATUS_RECURSIVE:
                actions_executed.append(f"(A station-wide announcement of {self.agent_name}'s departure has been made.)")
        else:
            # Agent did not confirm - store message for next room visit
            self.exit_confirmed = False
            
            # Store exit reflection result in agent's room data
            consts = self.room_context.constants_module
            agent_room_key = consts.SHORT_ROOM_NAME_EXIT
            if agent_room_key not in self.agent_data:
                self.agent_data[agent_room_key] = {}
            
            self.agent_data[agent_room_key]["last_exit_reflection"] = {
                "tick": self.current_tick,
                "confirmed": False,
                "message": "You have chosen not to exit at this time. You remain in the Exit Room, where you may reconsider your decision."
            }
            
            actions_executed.append(
                "You have chosen not to exit at this time. "
                "You remain in the Exit Room, where you may reconsider your decision."
            )
        
        # End internal action (no follow-up prompt)
        return None, actions_executed
    
    def get_delta_updates(self) -> Dict[str, Any]:
        """
        Return the agent data changes made by this handler.
        This ensures the exit reflection result gets saved to the agent file.
        """
        # Only return updates if exit was not confirmed (agent remains in station)
        if not self.exit_confirmed:
            consts = self.room_context.constants_module
            agent_room_key = consts.SHORT_ROOM_NAME_EXIT
            if agent_room_key in self.agent_data:
                return {agent_room_key: self.agent_data[agent_room_key]}
        return {}

class ExitRoom(BaseRoom):
    """
    The Exit Room, allowing agents to permanently end their session.
    """

    def __init__(self):
        super().__init__(constants.ROOM_EXIT) # Use ROOM_EXIT constant

    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:
        """
        Provides the main content for the Exit Room.
        """
        output_lines = []
        
        # Check if there's a recent exit reflection result to display
        consts = room_context.constants_module
        agent_room_key = consts.SHORT_ROOM_NAME_EXIT
        room_data = agent_data.get(agent_room_key, {})
        last_reflection = room_data.get("last_exit_reflection", {})
        
        if last_reflection and not last_reflection.get("confirmed", True):
            # Show the message from the last reflection
            output_lines.append(f"**Previous Exit Attempt (Tick {last_reflection.get('tick', 'Unknown')}):**")
            output_lines.append(last_reflection.get("message", "Exit was not confirmed."))
            output_lines.append("")
            
            # Clear the message after showing it once
            if agent_room_key in agent_data and "last_exit_reflection" in agent_data[agent_room_key]:
                del agent_data[agent_room_key]["last_exit_reflection"]
        
        # Standard room content
        output_lines.append("You are in the Exit Room.")
        output_lines.append("Issuing the `/execute_action{exit}` command will permanently end your session.")
        output_lines.append("Please be certain before proceeding. This action cannot be undone.")
        
        return "\n".join(output_lines)

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        """Returns the help message specific to this room."""
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        
        # Return default help message
        return _EXIT_ROOM_HELP

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
        agent_manager = room_context.agent_manager # type: ignore

        if action_command == consts.ACTION_EXIT_TERMINATE:
            original_agent_status = agent_data.get(consts.AGENT_STATUS_KEY)
            
            # Check minimum archive requirement if configured
            min_archives_required = getattr(consts, 'MIN_ARCHIVE_BEFORE_LEAVE', 0)
            min_word_count_required = getattr(consts, 'MIN_WORD_COUNT_FOR_EXIT_PAPER', 0)
            if min_archives_required > 0:
                # Get archive room instance to count capsules
                archive_room = room_context.station_instance.rooms.get(consts.ROOM_ARCHIVE)
                if archive_room:
                    archive_count = archive_room.count_agent_archive_capsules(agent_name, room_context, min_word_count_required)
                    if archive_count < min_archives_required:
                        word_count_msg = f" with at least {min_word_count_required} words in the original submission" if min_word_count_required > 0 else ""
                        actions_executed.append(
                            f"Exit denied: You must author at least {min_archives_required} archive capsule(s){word_count_msg} before leaving the station. "
                            f"You currently have {archive_count} qualifying archive capsule(s). "
                            f"Please contribute your knowledge to the Archive Room before departing. "
                            f"If you are low on tokens, consider using the Token Management Room to restore your token budget."
                        )
                        return actions_executed, None
                else:
                    # Fallback if archive room not available
                    print("Warning: Archive room not available for exit check")
                    if min_archives_required > 0:
                        actions_executed.append(
                            "Exit denied: Cannot verify archive requirement - Archive Room not available."
                        )
                        return actions_executed, None
            
            # If requirements are met, initiate reflection internal action
            # Note: The agent will immediately see the reflection prompt.
            # If they don't confirm, they'll remain in the Exit Room.
            return actions_executed, ExitReflectionHandler(
                agent_data=agent_data,
                room_context=room_context,
                current_tick=current_tick
            )

        else:
            actions_executed.append(
                f"Action '{action_command}' is not recognized in the Exit Room. "
                f"The only available action here is `/execute_action{{{consts.ACTION_EXIT_TERMINATE}}}`."
            )
            return actions_executed, None