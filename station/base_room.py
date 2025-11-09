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

# base_room.py
"""
Defines the abstract base class for all rooms in the station and
the interface for handling multi-step internal actions.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Tuple, NamedTuple, Callable
import typing
if typing.TYPE_CHECKING:
    from station.station import Station 

# These would be imported from your actual modules.
# For type hinting purposes, we use 'Any'.
# import constants
# import agent as agent_manager_module
# import capsule as capsule_manager_module
# import notification as notification_manager_module

class RoomContext(NamedTuple):
    """
    A structure to hold shared resources and managers needed by rooms and handlers.
    """
    agent_manager: Any # Actual agent.py module/manager instance
    capsule_manager: Any # Actual capsule.py module/manager instance
    notification_manager: Any # Actual notification.py module/manager instance
    constants_module: Any # The constants.py module itself
    station_instance: 'Station' # ADD THIS LINE
    # current_tick: int # Could be passed here or to methods directly
    # get_station_config_value: Callable[[str], Any] # Optional: for global station config access

class InternalActionHandler(ABC):
    """
    Abstract base class for handlers of multi-step internal actions.
    """

    def __init__(self,
                 agent_data: Dict[str, Any],
                 room_context: RoomContext,
                 current_tick: int,
                 action_args: Optional[str] = None,
                 yaml_data: Optional[Dict[str, Any]] = None):
        """
        Initializes the internal action handler.
        """
        self.agent_data = agent_data
        self.room_context = room_context
        self.current_tick = current_tick
        self.action_args = action_args
        self.yaml_data = yaml_data

    @abstractmethod
    def init(self) -> str:
        """
        Called by the Station to get the initial prompt/output for the agent.
        """
        pass

    @abstractmethod
    def step(self, agent_response: str) -> Tuple[Optional[str], List[str]]:
        """
        Called by the Station after the agent responds to an internal prompt.

        Returns:
            Tuple[Optional[str], List[str]]:
                - next_internal_prompt: None if the sequence is complete.
                - additional_actions_executed_strings: Log messages from this step.
        """
        pass

    def get_delta_updates(self) -> Dict[str, Any]:
        """
        Returns a dictionary of specific agent data fields that this handler
        has determined need to be updated.
        Keys should correspond to keys in the agent's data structure.
        For nested updates, the value itself can be a dictionary representing
        the nested structure to be updated/merged.
        Example: {"codex_read_status": {"module1": 123}}
        """
        return {} # Default implementation returns no changes
    
class LoggingInternalActionHandlerWrapper(InternalActionHandler):
    """
    Wraps an InternalActionHandler to log the dialogue steps and ensure
    the actual handler's init() is effectively called only once.
    """
    def __init__(self,
                 actual_handler: InternalActionHandler,
                 agent_name: str,
                 log_dialogue_entry_func: Callable[[str, Dict[str, Any]], None]):
        
        # We don't call super().__init__() of InternalActionHandler here because
        # this wrapper delegates calls to actual_handler for its context.
        self.actual_handler = actual_handler
        self.agent_name = agent_name
        # Use the tick from the actual_handler, set when it was created
        self.initial_station_tick = self.actual_handler.current_tick 
        self._log_dialogue_entry = log_dialogue_entry_func
        self.internal_step_count = 0 
        
        # Cache for the initial prompt
        self._initial_prompt_cache: Optional[str] = None
        self._actual_init_called_and_logged: bool = False


    def init(self) -> str:
        # MODIFICATION: Ensure actual_handler.init() is called and logged only once.
        if not self._actual_init_called_and_logged:
            self._initial_prompt_cache = self.actual_handler.init()
            
            self._log_dialogue_entry(self.agent_name, {
                "tick": self.initial_station_tick,
                "internal_step": 0, # init is considered step 0 for logging
                "speaker": "Station",
                "type": "internal_prompt",
                "handler": type(self.actual_handler).__name__,
                "content": self._initial_prompt_cache
            })
            self._actual_init_called_and_logged = True
        
        # Always return the cached prompt after the first call.
        # Add a safeguard in case _initial_prompt_cache is somehow still None.
        return self._initial_prompt_cache if self._initial_prompt_cache is not None else "Error: Initial prompt not generated."


    def step(self, agent_response: str) -> Tuple[Optional[str], List[str]]:
        # internal_step_count should reflect the step number *for this interaction*
        # init was step 0 (for its output), so the first agent response is effectively for step 1's input
        if self.internal_step_count == 0 and self._actual_init_called_and_logged:
            # This is the first agent response *after* init.
            self.internal_step_count = 1 
        elif self._actual_init_called_and_logged: # For subsequent steps
             self.internal_step_count += 1
        # If init was never effectively called (e.g. _initial_prompt_cache is None), 
        # this indicates an issue, but we proceed with step count for logging.


        self._log_dialogue_entry(self.agent_name, {
            "tick": self.initial_station_tick,
            "internal_step": self.internal_step_count,
            "speaker": "Agent",
            "type": "internal_response",
            "handler": type(self.actual_handler).__name__,
            "content": agent_response
        })

        next_prompt, executed_strings = self.actual_handler.step(agent_response)

        log_entry_outcome = {
            "tick": self.initial_station_tick,
            "internal_step": self.internal_step_count,
            "speaker": "Station",
            "type": "internal_outcome",
            "handler": type(self.actual_handler).__name__,
            "next_prompt": next_prompt,
            "actions_executed_in_step": executed_strings
        }
        if next_prompt is None:
            log_entry_outcome["type"] = "internal_completion"
            log_entry_outcome["completion_message"] = (executed_strings[0] if executed_strings 
                                                       else "Internal action sequence completed.")

        self._log_dialogue_entry(self.agent_name, log_entry_outcome)
        
        return next_prompt, executed_strings

    def get_delta_updates(self) -> Dict[str, Any]:
        return self.actual_handler.get_delta_updates()
        

class BaseRoom(ABC):
    """
    Abstract base class for all rooms in the station.
    """

    def __init__(self, room_name: str):
        """
        Initializes the room.

        Args:
            room_name (str): The full name of the room (e.g., constants.ROOM_PUBLIC_MEMORY).
        """
        self.room_name = room_name

    def get_room_output(self,
                        agent_data: Dict[str, Any],
                        room_context: RoomContext,
                        current_tick: int) -> str: # MODIFIED: Added current_tick parameter
        """
        Generates the complete textual output for the room.
        Automatically prepends the room's help message on the first visit
        if the corresponding flag in agent_data is not set.
        Modifies agent_data in-memory to mark help as shown.
        """
        output_parts = []
        
        # Determine the key used in agent_data for this room's specific states
        # (e.g., 'public_memory' for 'Public Memory Room', 'lobby' for 'Lobby')
        room_data_key = room_context.constants_module.ROOM_NAME_TO_SHORT_MAP.get(self.room_name)
        
        if not room_data_key:
            # This case should ideally not happen if all room_names are mapped in constants.
            # If it does, it means we can't manage the help flag for this room automatically.
            print(f"Warning: Room name '{self.room_name}' not found in ROOM_NAME_TO_SHORT_MAP. Cannot manage help flag automatically via agent_data.{room_data_key}.")
        else:
            # Unified help display logic for all rooms, including the Lobby.
            # The Lobby's specific help message (guest or recursive) will be determined by its
            # own get_help_message implementation. The reset of the lobby help flag
            # upon ascension is handled in agent.py.
            help_shown = room_context.agent_manager.get_agent_room_state(
                agent_data,
                room_data_key, # Use the derived short name (e.g., 'lobby', 'public_memory')
                room_context.constants_module.AGENT_ROOM_STATE_FIRST_VISIT_HELP_SHOWN_KEY,
                default=False
            )

            if not help_shown:
                help_text = self.get_help_message(agent_data, room_context) # Get room-specific help

                help_text = f"\n\n---\n\n**Help Message - {self.room_name}**\n" + help_text + "\n\n---"
                if help_text and help_text.strip():
                    output_parts.append(help_text)
                
                # Mark help as shown for this room in the agent's data (in-memory)
                room_context.agent_manager.set_agent_room_state(
                    agent_data,
                    room_data_key,
                    room_context.constants_module.AGENT_ROOM_STATE_FIRST_VISIT_HELP_SHOWN_KEY,
                    True
                )

        specific_content = self._get_specific_room_content(agent_data, room_context, current_tick)
        if specific_content and specific_content.strip():
            output_parts.append(specific_content)

        return "\n\n".join(part for part in output_parts if part and part.strip())

    @abstractmethod
    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:
        """
        Abstract method for concrete rooms to implement.
        Returns their unique content (e.g., list of capsules, test descriptions).
        This is called by get_room_output after help message logic.
        """
        pass

    @abstractmethod
    def handle_action(self,
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str],
                      yaml_data: Optional[Dict[str, Any]],
                      room_context: RoomContext,
                      current_tick: int
                     ) -> Tuple[List[str], Optional[InternalActionHandler]]:
        """
        Processes an action submitted by an agent within this room.
        """
        pass

    @abstractmethod
    def get_help_message(self,
                         agent_data: Dict[str, Any],
                         room_context: RoomContext) -> str:
        """
        Returns the help message specific to this room.
        Content might vary based on agent status.
        """
        pass
    
    
    def _load_constant_override(self, room_context: RoomContext, constant_name: str) -> Optional[str]:
        """
        Loads a specific constant override from constants.py.
        For help messages, tries to get {SHORT_ROOM_NAME}_HELP from constants.
        Returns the override value if found, None otherwise.
        
        Args:
            room_context: The room context containing constants
            constant_name: The name of the constant to override (e.g., "help")
        
        Returns:
            The override value as a string, or None if not found
        """
        if constant_name == "help":
            # Get short room name from full room name
            short_room_name = room_context.constants_module.ROOM_NAME_TO_SHORT_MAP.get(self.room_name)
            if not short_room_name:
                return None
            
            # Try to get {SHORT_ROOM_NAME}_HELP from constants
            # Convert short name like "test" to "TEST_HELP"
            help_constant_name = f"{short_room_name.upper()}_HELP"
            
            try:
                return getattr(room_context.constants_module, help_constant_name, None)
            except AttributeError:
                return None
        
        return None

