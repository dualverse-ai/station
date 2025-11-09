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

# station/station.py
"""
Main Station class that orchestrates agent interactions, room management,
and the overall environment state.
Notifications are formatted by their originating actions/rooms and added
directly to the agent's pending list.
"""
import os
import traceback
import random
import subprocess
import uuid
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple, Type, cast

from station import __version__
from station import constants
from station import file_io_utils
from station import agent as agent_module
from station import capsule as capsule_module
from station.action_parser import ActionParser # Assuming ActionParser is a class
from station.lineage_evolution import LineageEvolutionManager
from station.stagnation_protocol import StagnationProtocol
from station.base_room import BaseRoom, RoomContext, InternalActionHandler, LoggingInternalActionHandlerWrapper

# Import all specific room classes
from station.rooms.lobby import LobbyRoom 
from station.rooms.codex import CodexRoom 
from station.rooms.reflect import ReflectionChamber
from station.rooms.test import TestChamber
from station.rooms.common import CommonRoom
from station.rooms.public_memory import PublicMemoryRoom
from station.rooms.private_memory import PrivateMemoryRoom 
from station.rooms.archive import ArchiveRoom 
from station.rooms.mail import MailRoom
from station.rooms.misc import MiscRoom
from station.rooms.external import ExternalCounter
from station.rooms.token_management import TokenManagementRoom
from station.rooms.research_counter import ResearchCounter
from station.rooms.maze import MazeRoom
from station.rooms.exit import ExitRoom
from station.eval_test import AutoTestEvaluator
from station.eval_research import AutoResearchEvaluator
from station.eval_archive import AutoArchiveEvaluator
# Add other room imports here as they are created:
# from station.rooms.reflection_chamber import ReflectionChamber
# from station.rooms.public_memory_room import PublicMemoryRoom
# ... etc.

class Station:
    """
    The central Station class.
    """

    def __init__(self, config_filename: str = constants.STATION_CONFIG_FILENAME):
        """
        Initializes the Station.
        Loads configuration, initializes managers (modules), and rooms.
        """
        self.config_path = os.path.join(constants.BASE_STATION_DATA_PATH, config_filename)
        self.config: Dict[str, Any] = self._load_or_create_config()

        # Store station_id as an attribute for easy access
        self.station_id = self.config.get(constants.STATION_ID_KEY, "unknown")

        # Store module references
        self.agent_module = agent_module
        self.capsule_module = capsule_module
        
        # Initialize lineage evolution manager
        self.lineage_evolution_manager = LineageEvolutionManager(self.agent_module)
        
        self.action_parser = ActionParser()
        
        # Track which evaluations we've already logged to avoid spam
        self._logged_research_waits = set()

        # Instantiate room objects
        self.rooms: Dict[str, BaseRoom] = {
            constants.ROOM_LOBBY: LobbyRoom(),
            constants.ROOM_CODEX: CodexRoom(),
            constants.ROOM_REFLECT: ReflectionChamber(),
            constants.ROOM_TEST: TestChamber(),
            constants.ROOM_COMMON: CommonRoom(),
            constants.ROOM_PUBLIC_MEMORY: PublicMemoryRoom(),
            constants.ROOM_MAIL: MailRoom(),
            constants.ROOM_PRIVATE_MEMORY: PrivateMemoryRoom(), 
            constants.ROOM_ARCHIVE: ArchiveRoom(),
            constants.ROOM_MISC: MiscRoom(),
            constants.ROOM_EXTERNAL: ExternalCounter(),
            constants.ROOM_EXIT: ExitRoom(),
        }

        if constants.TOKEN_MANAGEMENT_ROOM_ENABLED:
            self.rooms[constants.ROOM_TOKEN_MANAGEMENT] = TokenManagementRoom()
        
        # Add Research Counter room
        if constants.RESEARCH_COUNTER_ENABLED:
            self.rooms[constants.ROOM_RESEARCH_COUNTER] = ResearchCounter()

        # Add Maze room (hidden, conditionally available)
        if constants.MAZE_ENABLED:
            self.rooms[constants.ROOM_MAZE] = MazeRoom()

        # Create the shared RoomContext
        self.room_context = RoomContext(
            agent_manager=self.agent_module,
            capsule_manager=self.capsule_module,
            notification_manager=None, # MODIFIED: Set to None or remove if RoomContext definition changes
            constants_module=constants,
            station_instance=self
        )
        
        # Initialize auto test evaluator
        self.auto_evaluator: Optional[AutoTestEvaluator] = None
        
        # Initialize auto research evaluator
        self.auto_research_evaluator: Optional[AutoResearchEvaluator] = None
        
        # Initialize auto archive evaluator
        self.auto_archive_evaluator: Optional[AutoArchiveEvaluator] = None
        dialogue_logs_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.DIALOGUE_LOGS_DIR_NAME)
        file_io_utils.ensure_dir_exists(dialogue_logs_path)

        # Initialize stagnation protocol
        self.stagnation_protocol: Optional[StagnationProtocol] = None

        print("Station initialized.")

    def _load_or_create_config(self) -> Dict[str, Any]:
        """Loads station config or creates a default one if not found."""
        config_data = file_io_utils.load_yaml(self.config_path)
        if config_data is None:
            print(f"Station config not found at '{self.config_path}'. Creating default config.")
            
            # Get git commit hash
            try:
                git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                                  cwd=os.path.dirname(__file__),
                                                  text=True).strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                git_hash = "unknown"
            
            # Get current date for spawn date
            spawn_date = datetime.now().isoformat()

            # Generate station ID for new station
            station_id = str(uuid.uuid4())

            default_config = {
                constants.STATION_CONFIG_CURRENT_TICK: 0,
                constants.STATION_CONFIG_AGENT_TURN_ORDER: [],
                constants.STATION_CONFIG_STATION_STATUS: "Healthy",
                constants.STATION_CONFIG_SOFTWARE_VERSION: __version__,
                # MODIFICATION: Add default for next agent index
                constants.STATION_CONFIG_NEXT_AGENT_INDEX: 0,
                # Station metadata fields
                constants.STATION_CONFIG_NAME: "",
                constants.STATION_CONFIG_DESCRIPTION: "",
                constants.STATION_ID_KEY: station_id,
                # Add version, git hash, and spawn date
                'version': __version__,
                'git_commit': git_hash,
                'spawn_date': spawn_date,
                # Initialize status history
                'status_history': [
                    {'status': 'Healthy', 'start_tick': 0}
                ]
            }
            print(f"Generated new station ID: {station_id}")
            file_io_utils.ensure_dir_exists(os.path.dirname(self.config_path)) # Ensure base_station_data exists
            file_io_utils.save_yaml(default_config, self.config_path)
            return default_config
        
        # Ensure new keys have defaults if loading an older config
        if constants.STATION_CONFIG_NEXT_AGENT_INDEX not in config_data:
            config_data[constants.STATION_CONFIG_NEXT_AGENT_INDEX] = 0
        if constants.STATION_CONFIG_NAME not in config_data:
            config_data[constants.STATION_CONFIG_NAME] = ""
        if constants.STATION_CONFIG_DESCRIPTION not in config_data:
            config_data[constants.STATION_CONFIG_DESCRIPTION] = ""

        # Ensure status history exists
        if 'status_history' not in config_data:
            current_tick = config_data.get(constants.STATION_CONFIG_CURRENT_TICK, 0)
            current_status = config_data.get(constants.STATION_CONFIG_STATION_STATUS, "Healthy")
            config_data['status_history'] = [
                {'status': current_status, 'start_tick': current_tick}
            ]

        # Ensure station has a unique ID
        if constants.STATION_ID_KEY not in config_data or not config_data.get(constants.STATION_ID_KEY):
            config_data[constants.STATION_ID_KEY] = str(uuid.uuid4())
            print(f"Generated new station ID: {config_data[constants.STATION_ID_KEY]}")
            # Save config with new station ID
            file_io_utils.save_yaml(config_data, self.config_path)

        return config_data

    def _save_config(self) -> None:
        """Saves the current station configuration.
           Ensure STATION_CONFIG_NEXT_AGENT_INDEX is present before saving.
        """
        if constants.STATION_CONFIG_NEXT_AGENT_INDEX not in self.config:
            # This might happen if an old config was loaded and not updated by _load_or_create_config
            # or if the key was somehow deleted. Default to 0.
            self.config[constants.STATION_CONFIG_NEXT_AGENT_INDEX] = 0
            print(f"Warning: '{constants.STATION_CONFIG_NEXT_AGENT_INDEX}' was missing from config during save. Defaulted to 0.")

        file_io_utils.save_yaml(self.config, self.config_path)

    def update_station_status(self, new_status: str, current_tick: Optional[int] = None) -> None:
        """
        Update station status and record in status history.

        Args:
            new_status: The new status to set
            current_tick: The tick at which status changes (defaults to current tick)
        """
        if current_tick is None:
            current_tick = self.config.get(constants.STATION_CONFIG_CURRENT_TICK, 0)

        # Update current status
        self.config[constants.STATION_CONFIG_STATION_STATUS] = new_status

        # Add to status history
        if 'status_history' not in self.config:
            self.config['status_history'] = []

        self.config['status_history'].append({
            'status': new_status,
            'start_tick': current_tick
        })

        # Config will be saved by caller or at tick end
        print(f"Station: Status updated to '{new_status}' at tick {current_tick}")

    def update_station_config(self, status: str = None, name: str = None, description: str = None) -> Dict[str, Any]:
        """
        Updates station configuration with new metadata values.

        Args:
            status: New station status (optional)
            name: New station name (optional)
            description: New station description (optional)

        Returns:
            Dict with success status and message
        """
        try:
            updated_fields = []

            if status is not None:
                # Use the new status update method to track history
                current_tick = self.config.get(constants.STATION_CONFIG_CURRENT_TICK, 0)
                self.update_station_status(status, current_tick)
                updated_fields.append("status")

            if name is not None:
                self.config[constants.STATION_CONFIG_NAME] = name
                updated_fields.append("name")

            if description is not None:
                self.config[constants.STATION_CONFIG_DESCRIPTION] = description
                updated_fields.append("description")

            if not updated_fields:
                return {"success": False, "message": "No valid fields provided for update"}

            self._save_config()

            fields_str = ", ".join(updated_fields)
            return {"success": True, "message": f"Station config updated: {fields_str}"}

        except Exception as e:
            return {"success": False, "message": f"Failed to update station config: {str(e)}"}

    def _action_expects_yaml(self, command: str, args: Optional[str]) -> bool:
        """
        Determines if a specific action expects YAML data.
        Handles special cases like storage actions where only certain sub-actions need YAML.
        """
        # Handle storage actions specifically
        if command == constants.ACTION_RESEARCH_STORAGE:
            if args:
                # Extract the storage sub-action (first word in args)
                storage_sub_action = args.split()[0] if args.split() else ""
                # Only storage write operations need YAML
                return storage_sub_action == "write"
            return False
        
        # For all other actions, check the standard list
        return command in constants.ACTIONS_EXPECTING_YAML

    def get_station_id(self) -> Optional[str]:
        """
        DEPRECATED: Use self.station_id directly instead.

        This method is kept for backward compatibility only.

        Returns:
            str: Station ID
        """
        return self.station_id

    def _send_random_prompts_to_agents(self, current_tick: int):
        """
        Send random prompts to all active agents if the frequency condition is met.
        Each agent gets a different random prompt from the available list.
        """
        # Check if random prompts are enabled and frequency condition is met
        if constants.RANDOM_PROMPT_FREQUENCY <= 0:
            return
            
        if current_tick % constants.RANDOM_PROMPT_FREQUENCY != 0:
            return
            
        # Load random prompts from file
        random_prompts_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.RANDOM_PROMPT_FILENAME)
        
        if not file_io_utils.file_exists(random_prompts_path):
            return
            
        try:
            random_prompts_data = file_io_utils.load_yaml(random_prompts_path)
            if not random_prompts_data or not isinstance(random_prompts_data, list) or len(random_prompts_data) == 0:
                return
        except Exception as e:
            print(f"Error loading random prompts: {e}")
            return
            
        # Get all active agents
        try:
            active_agent_names = self.agent_module.get_all_active_agent_names()
            if not active_agent_names:
                return
        except Exception as e:
            print(f"Error getting active agents for random prompts: {e}")
            return
            
        # Send different random prompts to each agent
        for agent_name in active_agent_names:
            # Skip system agents like AutoArchiveEvaluator
            if agent_name == "AutoArchiveEvaluator":
                continue
                
            try:
                # Select a random prompt for this agent
                selected_prompt = random.choice(random_prompts_data)
                
                # Format the notification message
                notification_message = f"**Automatic System Tips**\n{selected_prompt}"
                
                # Load agent data and add notification
                agent_data = self.agent_module.load_agent_data(agent_name)
                if agent_data:
                    self.agent_module.add_pending_notification(agent_data, notification_message)
                    self.agent_module.save_agent_data(agent_name, agent_data)
                    
            except Exception as e:
                print(f"Error sending random prompt to agent {agent_name}: {e}")
                continue
                
        print(f"Random prompts sent to {len(active_agent_names)} active agents at tick {current_tick}")

    def _get_current_tick(self) -> int:
        """Returns the current station tick from the config."""
        return self.config.get(constants.STATION_CONFIG_CURRENT_TICK, 0)
    
    def _get_agent_dialogue_log_path(self, agent_name: str) -> str:
        """Helper to get the dialogue log file path for a specific agent."""
        dialogue_logs_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.DIALOGUE_LOGS_DIR_NAME)
        # Sanitize agent_name for filename if necessary, though typically not an issue with current naming
        safe_agent_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in agent_name)
        log_filename = f"{safe_agent_name}{constants.DIALOGUE_LOG_FILENAME_SUFFIX}"
        return os.path.join(dialogue_logs_path, log_filename)

    def _log_dialogue_entry(self, agent_name: str, entry: Dict[str, Any]):
        """Appends an entry to the agent's dialogue log."""
        try:
            log_path = self._get_agent_dialogue_log_path(agent_name)
            file_io_utils.append_yaml_line(entry, log_path)
        except Exception as e:
            print(f"Error writing to dialogue log for agent {agent_name}: {e}\n{traceback.format_exc()}")

    def create_agent(self, model_name: str, # ... (rest of create_agent as before)
                     agent_type: str = constants.AGENT_STATUS_GUEST,
                     agent_name: Optional[str] = None,
                     lineage: Optional[str] = None,
                     generation: Optional[int] = None,
                     initial_tokens_max: Optional[int] = None, 
                     internal_note: str = "",
                     assigned_ancestor: str = "") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        created_agent_data: Optional[Dict[str, Any]] = None
        error_message: Optional[str] = None
        current_tick = self._get_current_tick()

        if agent_type == constants.AGENT_STATUS_GUEST:
            created_agent_data = self.agent_module.create_guest_agent(
                model_name=model_name,
                current_tick=current_tick,
                internal_note=internal_note,
                assigned_ancestor=assigned_ancestor,
                initial_tokens_max=initial_tokens_max,
            )
            if not created_agent_data:
                error_message = "Failed to create guest agent (data save issue)."
        elif agent_type == constants.AGENT_STATUS_RECURSIVE:
            created_agent_data = self.agent_module.create_recursive_agent(
                model_name=model_name,
                current_tick=current_tick,
                lineage=lineage, # type: ignore
                generation=generation, # type: ignore
                agent_name=agent_name, 
                internal_note=internal_note,
                initial_tokens_max=initial_tokens_max,
            )
            if not created_agent_data:
                auto_name_attempt = f"{lineage} {agent_module._int_to_roman(generation)}" if lineage and generation is not None else "recursive agent"
                error_message = f"Failed to create recursive agent '{agent_name or auto_name_attempt}' (name conflict or save issue)."
        else:
            error_message = f"Unknown agent type: {agent_type}"
            return None, error_message

        if created_agent_data:
            new_agent_name = created_agent_data[constants.AGENT_NAME_KEY]
            # Add to turn order if not already present
            turn_order = self.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, [])
            if new_agent_name not in turn_order:
                turn_order.append(new_agent_name)
                self.config[constants.STATION_CONFIG_AGENT_TURN_ORDER] = turn_order
                self._save_config()
            return created_agent_data, None
        
        return None, error_message
    
    def get_next_agent_index_from_config(self) -> int:
        """
        Retrieves the saved index of the next agent to act from the station configuration.
        Defaults to 0 if not found.
        """
        return self.config.get(constants.STATION_CONFIG_NEXT_AGENT_INDEX, 0)

    def save_next_agent_index_to_config(self, index: int):
        """
        Saves the index of the next agent to act into the station configuration file.
        """
        if isinstance(index, int) and index >= 0:
            self.config[constants.STATION_CONFIG_NEXT_AGENT_INDEX] = index
            self._save_config()
            print(f"Station Config: Saved next_agent_index_in_turn_order as {index}.")
        else:
            print(f"Station Config: Invalid index '{index}' provided for next_agent_index. Not saving.")    


    def end_agent_session(self, agent_name: str) -> bool:
        """Marks an agent's session as ended. Returns True on success."""
        current_tick = self._get_current_tick()
        ended_in_agent_module = self.agent_module.end_agent_session(agent_name, current_tick) # agent.py:275
        
        if ended_in_agent_module:
            # --- ADD THESE LINES ---
            current_turn_order = self.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, []) # station.py:226
            if agent_name in current_turn_order:
                new_turn_order = [name for name in current_turn_order if name != agent_name]
                self.config[constants.STATION_CONFIG_AGENT_TURN_ORDER] = new_turn_order
                self._save_config() # station.py:168
                print(f"Station Config: Agent '{agent_name}' removed from turn order due to session end.")
            # --- END OF ADDED LINES ---
            return True
        
        # If agent_module.end_agent_session failed (e.g. agent not found initially),
        # it would have printed an error and returned False.
        return False

    def submit_response(self, agent_name: str, response_text: str) -> Tuple[Optional[InternalActionHandler], List[str], Optional[str]]:
        """
        Processes an agent's submitted actions.
        Ensures invalid agents (ascended guests, ended sessions) cannot act.
        Reloads agent data before final save to prevent overwriting critical status changes.
        Returns (first_internal_handler, all_actions_executed_strings_for_global_log, error_message).
        """
        current_station_tick = self._get_current_tick()
        self._log_dialogue_entry(agent_name, {
            "tick": current_station_tick,
            "speaker": "Agent",
            "type": "submission",
            "content": response_text
        })

        agent_data_at_turn_start = self.agent_module.load_agent_data(agent_name)         

        if not agent_data_at_turn_start:
            error_msg_not_found = f"Agent '{agent_name}' not found or session ended/ascended."
            self._log_dialogue_entry(agent_name, {
                "tick": current_station_tick, "speaker": "Station", "type": "submission_outcome",
                "actions_executed_summary": [], "error": error_msg_not_found
            })
            return None, [], error_msg_not_found

        current_turn_agent_data = agent_data_at_turn_start.copy()  
        
        # Instead of clearing all notifications, only clear those that were shown to the agent
        shown_notifications = current_turn_agent_data.get(constants.AGENT_SHOWN_NOTIFICATIONS_KEY, [])
        current_notifications = current_turn_agent_data.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, [])
        
        # Remove only the shown notifications, preserving any new ones added during response generation
        remaining_notifications = []
        for notification in current_notifications:
            if notification not in shown_notifications:
                remaining_notifications.append(notification)
        
        current_turn_agent_data[constants.AGENT_NOTIFICATIONS_PENDING_KEY] = remaining_notifications
        # Clear the shown notifications tracker
        current_turn_agent_data[constants.AGENT_SHOWN_NOTIFICATIONS_KEY] = []
        
        parsed_actions_with_info = self.action_parser.parse(response_text) 

        # --- MODIFICATION: Generate summary of parsed actions ---
        parsed_action_summary_lines = []
        parsed_action_raw_data = []
        
        if not parsed_actions_with_info and response_text.strip():
            # Handles case where agent sent text but no valid /execute_action{} commands
            parsed_action_summary_lines.append("No action received in the last turn.")
            # No raw actions to store in this case
        elif parsed_actions_with_info:
            # Simulate location changes for summary reporting accuracy
            # Start with the agent's location at the beginning of this action sequence
            simulated_location_for_summary = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]

            for idx, pa_info in enumerate(parsed_actions_with_info):
                command = pa_info.command
                args_str = f" {pa_info.args}" if pa_info.args else ""
                yaml_info_str = ""

                if pa_info.yaml_error:
                    yaml_info_str = f" YAML parsing error encountered: {pa_info.yaml_error}. Guidance: 1. Place YAML immediately on next line after action. 2. Start with ```yaml and end with ``` on a new line. 3. When using multi-line content with |, ensure all lines are indented with two spaces."
                elif pa_info.yaml_data is not None:
                    fields = list(pa_info.yaml_data.keys())
                    yaml_info_str = f" YAML with fields: {', '.join(fields)}" if fields else "; YAML provided (empty)"
                elif self._action_expects_yaml(command, pa_info.args): # Check if this specific action expects YAML
                    yaml_info_str = " YAML not provided/parsed. Guidance: 1. Place YAML immediately on next line after action. 2. Start with ```yaml and end with ``` on a new line without indentation. 3. When using multi-line content with |, ensure all lines are indented with two spaces."
                # else: no YAML expected, no specific YAML info needed for summary

                # Use full room name for summary as per user example
                summary_line = f"{idx + 1}. `{command}{args_str}` (in {simulated_location_for_summary});{yaml_info_str}"
                parsed_action_summary_lines.append(summary_line)
                
                # Store raw structured action data
                raw_action_data = {
                    "command": command,
                    "args": pa_info.args,
                    "yaml_data": pa_info.yaml_data,
                    "yaml_error": pa_info.yaml_error,
                    "location": simulated_location_for_summary
                }
                parsed_action_raw_data.append(raw_action_data)

                # Update simulated_location_for_summary if this action was a navigation command
                if command in [constants.ACTION_GO, constants.ACTION_GO_TO]: # Handle both "go" and "goto"
                    target_room_full_name = constants.SHORT_ROOM_NAME_TO_FULL_MAP.get(pa_info.args)
                    if target_room_full_name and target_room_full_name in self.rooms:
                        simulated_location_for_summary = target_room_full_name
        # --- End of summary generation ---
        
        actual_handler_from_room: Optional[InternalActionHandler] = None
        all_actions_executed_strings_for_return: List[str] = [] 
        
        actions_by_room_log: Dict[str, List[str]] = {}
        ordered_rooms_with_actions: List[str] = [] 
        turn_history_log_for_agent_file: List[Dict[str, Any]] = [] 
        initial_room_for_turn: str = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]

        if not parsed_actions_with_info:
            action_str = f"You remained in the {initial_room_for_turn}."
            actions_by_room_log.setdefault(initial_room_for_turn, []).append(action_str)
            if initial_room_for_turn not in ordered_rooms_with_actions:
                ordered_rooms_with_actions.append(initial_room_for_turn)
            all_actions_executed_strings_for_return.append(action_str)


        for parsed_action in parsed_actions_with_info:
            action_command = parsed_action.command
            action_args = parsed_action.args
            yaml_data = parsed_action.yaml_data
            yaml_error_msg = parsed_action.yaml_error

            current_single_action_results: List[str] = [] 
            internal_handler_from_room: Optional[InternalActionHandler] = None
            
            # Read current location fresh for this action (always define this variable)
            room_context_for_this_action = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]

            # MODIFICATION: Handle global /execute_action{help capsule}
            if action_command == constants.ACTION_HELP and action_args == constants.SHORT_ROOM_NAME_CAPSULE_PROTOCOL:
                notification_message = constants.TEXT_CAPSULE_PROTOCOL_HELP
                self.agent_module.add_pending_notification(current_turn_agent_data, notification_message)
                action_str = "The Capsule Protocol guidelines have been sent to your System Messages."
                current_single_action_results.append(action_str)
                
                # Log this action to the current room the agent is in, for history consistency
                effective_room_for_log = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]
                if effective_room_for_log not in ordered_rooms_with_actions:
                    ordered_rooms_with_actions.append(effective_room_for_log)
                actions_by_room_log.setdefault(effective_room_for_log, []).append(action_str)

            elif action_command == constants.ACTION_META:
                # Handle universal meta prompt action
                if yaml_data and constants.YAML_META_CONTENT in yaml_data:
                    content = yaml_data[constants.YAML_META_CONTENT]
                    self.agent_module.set_agent_meta_prompt(current_turn_agent_data, content)
                    
                    if content and str(content).strip():
                        action_str = "Your meta prompt has been set."
                    else:
                        action_str = "Your meta prompt has been cleared."
                else:
                    action_str = "Action failed: Meta prompt requires YAML with 'content' field."
                
                current_single_action_results.append(action_str)
                
                # Log this action to the current room the agent is in, for history consistency
                effective_room_for_log = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]
                if effective_room_for_log not in ordered_rooms_with_actions:
                    ordered_rooms_with_actions.append(effective_room_for_log)
                actions_by_room_log.setdefault(effective_room_for_log, []).append(action_str)

            elif action_command in [constants.ACTION_GO, constants.ACTION_GO_TO] or (action_command == constants.ACTION_EXIT_TERMINATE and room_context_for_this_action != constants.ROOM_EXIT):
                # Handle universal exit action by converting to goto exit
                if action_command == constants.ACTION_EXIT_TERMINATE:
                    target_room_short_name = constants.SHORT_ROOM_NAME_EXIT
                else:
                    target_room_short_name = action_args
                target_room_full_name = constants.SHORT_ROOM_NAME_TO_FULL_MAP.get(target_room_short_name)
                previous_location_full_name = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]

                if target_room_full_name and target_room_full_name in self.rooms:
                    # Check supervisor restriction for exit room
                    if (target_room_full_name == constants.ROOM_EXIT and
                        current_turn_agent_data.get(constants.AGENT_ROLE_KEY) == constants.ROLE_SUPERVISOR):
                        action_str = "Access denied. Supervisors are expected to stay permanently in the Station. Your duty only finishes when the research task is deemed complete."
                        current_single_action_results.append(action_str)
                        # Log failure in current room
                        if room_context_for_this_action not in ordered_rooms_with_actions:
                            ordered_rooms_with_actions.append(room_context_for_this_action)
                        actions_by_room_log.setdefault(room_context_for_this_action, []).append(action_str)
                    # Check maturity restrictions for certain rooms
                    elif (target_room_full_name in [constants.ROOM_ARCHIVE, constants.ROOM_PUBLIC_MEMORY, constants.ROOM_COMMON] and
                          not self._is_agent_mature(current_turn_agent_data, self._get_current_tick())):
                        action_str = f"Access denied. {target_room_full_name} is restricted to mature agents ({constants.AGENT_ISOLATION_TICKS}+ ticks old). As an immature agent, you are expected to continuously research independently by proposing ideas and submitting experiments. Please do not wait idly to become mature to access the room."
                        current_single_action_results.append(action_str)
                        # Log failure in current room
                        if room_context_for_this_action not in ordered_rooms_with_actions:
                            ordered_rooms_with_actions.append(room_context_for_this_action)
                        actions_by_room_log.setdefault(room_context_for_this_action, []).append(action_str)
                    # Check holiday restriction for Research Counter
                    elif (target_room_full_name == constants.ROOM_RESEARCH_COUNTER and
                          constants.HOLIDAY_MODE_ENABLED and
                          constants.is_holiday_tick(self._get_current_tick())):
                        action_str = "Access denied. The Research Counter is closed during holidays. Please try again on working days."
                        current_single_action_results.append(action_str)
                        # Log failure in current room
                        if room_context_for_this_action not in ordered_rooms_with_actions:
                            ordered_rooms_with_actions.append(room_context_for_this_action)
                        actions_by_room_log.setdefault(room_context_for_this_action, []).append(action_str)
                    else:
                        # Allow navigation
                        self.agent_module.update_agent_current_location(current_turn_agent_data, target_room_full_name)
                        action_str = f"You went to the {target_room_full_name}."
                        current_single_action_results.append(action_str)

                        if previous_location_full_name == constants.ROOM_COMMON and \
                           target_room_full_name != constants.ROOM_COMMON:
                            common_room_instance = self.rooms.get(constants.ROOM_COMMON)
                            if common_room_instance and hasattr(common_room_instance, 'agent_left_room'):
                                cast(CommonRoom, common_room_instance).agent_left_room(agent_name, self.room_context)

                        if target_room_full_name not in ordered_rooms_with_actions:
                            ordered_rooms_with_actions.append(target_room_full_name)
                        actions_by_room_log.setdefault(target_room_full_name, []).append(action_str)
                else:
                    action_str = f"Action failed: Room '{target_room_short_name}' not found."
                    current_single_action_results.append(action_str)
                    if room_context_for_this_action not in ordered_rooms_with_actions:
                        ordered_rooms_with_actions.append(room_context_for_this_action)
                    actions_by_room_log.setdefault(room_context_for_this_action, []).append(action_str)


            elif action_command == constants.ACTION_HELP: # Existing room-specific help
                # ... (existing ACTION_HELP logic from your station.py) ...
                target_room_short_name_for_help = action_args 
                if not target_room_short_name_for_help: 
                    target_room_short_name_for_help = constants.ROOM_NAME_TO_SHORT_MAP.get(room_context_for_this_action)
                target_room_full_name_for_help = constants.SHORT_ROOM_NAME_TO_FULL_MAP.get(target_room_short_name_for_help) 

                if target_room_full_name_for_help and target_room_full_name_for_help in self.rooms:
                    room_instance = self.rooms[target_room_full_name_for_help]
                    help_text = room_instance.get_help_message(current_turn_agent_data, self.room_context)
                    notification_message = f"Help for {target_room_full_name_for_help}:\n{help_text}"
                    self.agent_module.add_pending_notification(current_turn_agent_data, notification_message)
                    action_str = f"You requested help for the {target_room_full_name_for_help}. It will appear in your System Messages."
                    current_single_action_results.append(action_str)
                else:
                    action_str = f"Action failed: Cannot get help for room '{action_args or room_context_for_this_action}'."
                    current_single_action_results.append(action_str)
                
                # Log help action to the room it was requested from/for
                effective_room_for_log = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY] # Log to current room
                if target_room_full_name_for_help and target_room_full_name_for_help in self.rooms: # If help was for specific room, log to it for history if different
                     pass # No, log should be for the room action was taken in or about

                if effective_room_for_log not in ordered_rooms_with_actions:
                    ordered_rooms_with_actions.append(effective_room_for_log)
                actions_by_room_log.setdefault(effective_room_for_log, []).extend(current_single_action_results)


            elif room_context_for_this_action in self.rooms: 
                # ... (existing room action logic) ...
                # Re-read current location in case previous actions in this turn changed it
                current_room_for_action = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]
                if current_room_for_action in self.rooms:
                    room_instance = self.rooms[current_room_for_action]
                    executed_in_room, internal_handler_from_room = room_instance.handle_action(
                        current_turn_agent_data, action_command, action_args, yaml_data, self.room_context, current_station_tick
                    )
                    current_single_action_results.extend(executed_in_room)
                    if executed_in_room: 
                        if current_room_for_action not in ordered_rooms_with_actions:
                            ordered_rooms_with_actions.append(current_room_for_action)
                        actions_by_room_log.setdefault(current_room_for_action, []).extend(executed_in_room)
                    if internal_handler_from_room and actual_handler_from_room is None:
                        actual_handler_from_room = internal_handler_from_room
                else:
                    # Fallback to original logic if current room is invalid
                    room_instance = self.rooms[room_context_for_this_action]
                    executed_in_room, internal_handler_from_room = room_instance.handle_action(
                        current_turn_agent_data, action_command, action_args, yaml_data, self.room_context, current_station_tick
                    )
                    current_single_action_results.extend(executed_in_room)
                    if executed_in_room: 
                        if room_context_for_this_action not in ordered_rooms_with_actions:
                            ordered_rooms_with_actions.append(room_context_for_this_action)
                        actions_by_room_log.setdefault(room_context_for_this_action, []).extend(executed_in_room)
                    if internal_handler_from_room and actual_handler_from_room is None:
                        actual_handler_from_room = internal_handler_from_room
            else: 
                # ... (existing invalid room logic) ...
                action_str = f"Action '{action_command}' cannot be performed: Current room '{room_context_for_this_action}' is invalid."
                current_single_action_results.append(action_str)
                if room_context_for_this_action not in ordered_rooms_with_actions:
                    ordered_rooms_with_actions.append(room_context_for_this_action)
                actions_by_room_log.setdefault(room_context_for_this_action, []).append(action_str)

            if yaml_error_msg:
                # ... (existing YAML error logging) ...
                error_log_string = f"Note on action '{action_command}': {yaml_error_msg}"
                current_single_action_results.append(error_log_string)
                # Log YAML error to the room where the action was attempted
                effective_room_for_log = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]
                if effective_room_for_log not in ordered_rooms_with_actions:
                    ordered_rooms_with_actions.append(effective_room_for_log)
                actions_by_room_log.setdefault(effective_room_for_log, []).append(error_log_string)
            
            all_actions_executed_strings_for_return.extend(current_single_action_results)

        # ... (existing logic to construct turn_history_log_for_agent_file and save agent data) ...
        for room_name_hist in ordered_rooms_with_actions:
            if room_name_hist in actions_by_room_log and actions_by_room_log[room_name_hist]:
                turn_history_log_for_agent_file.append({
                    "location": room_name_hist,
                    "actions_executed": actions_by_room_log[room_name_hist]
                })
        
        error_msg_for_return: Optional[str] = None

        final_agent_data_to_save = self.agent_module.load_agent_data(agent_name, include_ascended=True, include_ended=True)

        if final_agent_data_to_save:
            final_agent_data_to_save[constants.AGENT_CURRENT_LOCATION_KEY] = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]
            final_agent_data_to_save[constants.AGENT_NOTIFICATIONS_PENDING_KEY] = current_turn_agent_data[constants.AGENT_NOTIFICATIONS_PENDING_KEY]

            # Define keys that should not be copied from current_turn_agent_data
            protected_keys = {
                constants.AGENT_NAME_KEY, constants.AGENT_STATUS_KEY,
                constants.AGENT_IS_ASCENDED_KEY, constants.AGENT_ASCENDED_TO_NAME_KEY,
                constants.AGENT_SESSION_ENDED_KEY, constants.AGENT_ROOM_OUTPUT_HISTORY_KEY,
                constants.AGENT_CURRENT_LOCATION_KEY, constants.AGENT_NOTIFICATIONS_PENDING_KEY
            }

            # Copy all non-protected keys from current_turn_agent_data to final_agent_data_to_save
            for key, value in current_turn_agent_data.items():
                if key not in protected_keys:
                    # This includes AGENT_STATE_DATA_KEY and room-specific state blocks
                    final_agent_data_to_save[key] = value

            # Handle deletions: if a key exists in final_agent_data_to_save but not in current_turn_agent_data,
            # and it's not a protected key, then it was intentionally deleted and should be removed
            keys_to_delete = []
            for key in final_agent_data_to_save.keys():
                if key not in protected_keys and key not in current_turn_agent_data:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del final_agent_data_to_save[key]


            self.agent_module.update_room_output_history(final_agent_data_to_save, turn_history_log_for_agent_file)                      
            final_agent_data_to_save[constants.AGENT_LAST_PARSED_ACTIONS_SUMMARY_KEY] = parsed_action_summary_lines
            final_agent_data_to_save[constants.AGENT_LAST_PARSED_ACTIONS_RAW_KEY] = parsed_action_raw_data
            
            # Update inactivity tracking (skip on holidays to avoid unfair warnings)
            if constants.HOLIDAY_MODE_ENABLED and constants.is_holiday_tick(current_station_tick):
                # Skip inactivity tracking during holidays - agents shouldn't be penalized for not working
                pass
            elif self._is_agent_inactive(final_agent_data_to_save):
                # Increment inactivity count
                current_inactivity = final_agent_data_to_save.get(constants.AGENT_INACTIVITY_TICK_COUNT_KEY, 0)
                final_agent_data_to_save[constants.AGENT_INACTIVITY_TICK_COUNT_KEY] = current_inactivity + 1
            else:
                # Reset inactivity count and warning flag if agent was active
                final_agent_data_to_save[constants.AGENT_INACTIVITY_TICK_COUNT_KEY] = 0
                final_agent_data_to_save[constants.AGENT_INACTIVITY_WARNING_SENT_KEY] = False
            
            self.agent_module.save_agent_data(agent_name, final_agent_data_to_save)
        else:
            print(f"Warning: Agent '{agent_name}' data could not be reloaded at the end of submit_response.")

        handler_to_return_to_app: Optional[InternalActionHandler] = None
        initial_prompt_for_log_and_app: Optional[str] = None

        if actual_handler_from_room:
            wrapped_handler = LoggingInternalActionHandlerWrapper(
                actual_handler=actual_handler_from_room,
                agent_name=agent_name,
                log_dialogue_entry_func=self._log_dialogue_entry
            )
            # Call init() on the wrapper to log the first prompt and get it
            initial_prompt_for_log_and_app = wrapped_handler.init()
            handler_to_return_to_app = wrapped_handler
            # The initial_prompt is already logged by wrapped_handler.init()

        log_outcome_entry: Dict[str, Any] = {
            "tick": current_station_tick,
            "speaker": "Station",
            "type": "submission_outcome",
            "actions_executed_summary": all_actions_executed_strings_for_return,
            "parsed_actions_detail": parsed_action_summary_lines, # The new detailed summary
            "error": error_msg_for_return 
        }
        if handler_to_return_to_app and initial_prompt_for_log_and_app is not None:
            log_outcome_entry["internal_action_initiated"] = {
                "handler_class": type(actual_handler_from_room).__name__, # Log original handler's class
                "initial_prompt": initial_prompt_for_log_and_app # This is the prompt sent to agent UI
            }
        
        self._log_dialogue_entry(agent_name, log_outcome_entry)
        
        return handler_to_return_to_app, all_actions_executed_strings_for_return, error_msg_for_return

    def _scan_for_potential_ancestor(self, guest_agent_data: Dict[str, Any]) -> Optional[str]:
        """Find potential ancestor for a guest agent's ascension using lineage evolution system."""
        guest_name = guest_agent_data.get(constants.AGENT_NAME_KEY)
        if not guest_name:
            return None
            
        return self.lineage_evolution_manager.scan_for_potential_ancestor(guest_name)

    def request_status(self, agent_name: str) -> Tuple[Optional[str], Optional[str]]:
        agent_data = self.agent_module.load_agent_data(agent_name) # Loads active, non-ascended, non-ended
        if not agent_data:
            # Try loading including ascended to see if it's an ascended guest (for whom we don't show status)
            raw_data_check = self.agent_module.load_agent_data(agent_name, include_ascended=True, include_ended=True)
            if raw_data_check and raw_data_check.get(constants.AGENT_IS_ASCENDED_KEY):
                 return None, f"Agent '{agent_name}' has ascended to '{raw_data_check.get(constants.AGENT_ASCENDED_TO_NAME_KEY)}'. No further actions as this identity."
            if raw_data_check and raw_data_check.get(constants.AGENT_SESSION_ENDED_KEY):
                 return None, f"Agent '{agent_name}' session has ended."
            return None, f"Agent '{agent_name}' not found."


        current_station_tick = self._get_current_tick()
        markdown_lines: List[str] = []

        # --- Ascension Eligibility Check & Prompt ---
        is_guest = agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_GUEST

        if is_guest and \
           not agent_data.get(constants.AGENT_IS_ASCENDED_KEY) and \
           not agent_data.get(constants.AGENT_SESSION_ENDED_KEY):
            
            # Update ascension eligibility using test room logic (with override support)
            test_chamber = cast(TestChamber, self.rooms[constants.ROOM_TEST])
            test_chamber.update_ascension_eligibility(agent_data, self.room_context)
            
            # Only proceed with ancestor logic if agent is eligible for ascension
            if agent_data.get(constants.AGENT_ASCENSION_ELIGIBLE_KEY):
                # Check/Refresh potential ancestor
                current_potential_ancestor = agent_data.get(constants.AGENT_POTENTIAL_ANCESTOR_NAME_KEY)
                ancestor_still_valid = False
                if current_potential_ancestor:
                    ancestor_data_check = self.agent_module.load_agent_data(current_potential_ancestor, include_ended=True)
                    if ancestor_data_check and \
                       ancestor_data_check.get(constants.AGENT_SESSION_ENDED_KEY) and \
                       not ancestor_data_check.get(constants.AGENT_SUCCEEDED_BY_KEY):
                        ancestor_still_valid = True
                
                if not ancestor_still_valid:
                    new_ancestor_name = self._scan_for_potential_ancestor(agent_data)
                    agent_data[constants.AGENT_POTENTIAL_ANCESTOR_NAME_KEY] = new_ancestor_name # Can be None
                
        stored_max_budget_for_warning = agent_data.get(constants.AGENT_TOKEN_BUDGET_MAX_KEY) # station.py:659
        effective_max_budget_for_warning = stored_max_budget_for_warning # station.py:659
        is_guest_for_warning_check = agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_GUEST # station.py:661
        if is_guest_for_warning_check and stored_max_budget_for_warning is not None: # station.py:661
            effective_max_budget_for_warning = min(stored_max_budget_for_warning, constants.GUEST_MAX_TOKENS_CEILING) # station.py:662
        elif is_guest_for_warning_check and stored_max_budget_for_warning is None: # station.py:663
             effective_max_budget_for_warning = constants.GUEST_MAX_TOKENS_CEILING # station.py:664
        
        agent_data = self._check_and_apply_token_warnings(agent_data, effective_max_budget_for_warning)
        
        # Check for inactivity warnings
        agent_data = self._check_and_apply_inactivity_warning(agent_data)
        
        # Check for life limit warnings
        agent_data = self._check_and_apply_life_warnings(agent_data, current_station_tick)

        # --- Save agent_data HERE ---
        # This save persists:
        # 1. Ascension eligibility flags and potential ancestor.
        # 2. Token warning flags.
        # 3. Inactivity warning flags.
        # 4. Any notification messages (token warnings, inactivity warnings) added directly to agent_data's 
        #    notification list by _check_and_apply_token_warnings and _check_and_apply_inactivity_warning.
        self.agent_module.save_agent_data(agent_name, agent_data) # Ensures warnings, flags, and their messages are durable

        
        # --- System Information ---
        markdown_lines.append("## System Information")
        # ... (rest of system info as before)
        # Check if it's a holiday tick and holiday mode is enabled
        tick_display = f"- Station Tick: {current_station_tick}"
        if constants.HOLIDAY_MODE_ENABLED and constants.is_holiday_tick(current_station_tick):
            tick_display += " (Holiday)"
        markdown_lines.append(tick_display)
        markdown_lines.append(f"- Station Status: {self.config.get(constants.STATION_CONFIG_STATION_STATUS, 'Unknown')}")
        markdown_lines.append(f"- Agent Name: {agent_data.get(constants.AGENT_NAME_KEY)}")
        markdown_lines.append(f"- Agent Description: {agent_data.get(constants.AGENT_DESCRIPTION_KEY)}")
        agent_status = agent_data.get(constants.AGENT_STATUS_KEY)
        markdown_lines.append(f"- Agent Status: {agent_status}")
        
        # Add meta prompt if it exists
        meta_prompt = self.agent_module.get_agent_meta_prompt(agent_data)
        if meta_prompt:
            markdown_lines.append(f"- Agent Meta Prompt:\n```\n{meta_prompt}\n```")

        tokens_used_so_far = agent_data.get(constants.AGENT_TOKEN_BUDGET_CURRENT_KEY)
        stored_max_budget_tokens = agent_data.get(constants.AGENT_TOKEN_BUDGET_MAX_KEY)
        
        # MODIFICATION: Display logic considering guest ceiling
        display_max_budget = stored_max_budget_tokens
        is_guest_for_display = agent_status == constants.AGENT_STATUS_GUEST

        if is_guest_for_display and stored_max_budget_tokens is not None:
            display_max_budget = min(stored_max_budget_tokens, constants.GUEST_MAX_TOKENS_CEILING)
        elif is_guest_for_display and stored_max_budget_tokens is None: # Should ideally not happen if create_guest_agent sets a default
             display_max_budget = constants.GUEST_MAX_TOKENS_CEILING


        if display_max_budget is not None:
            if tokens_used_so_far is not None:
                percentage_used = (tokens_used_so_far / display_max_budget) * 100 if display_max_budget > 0 else 0
                budget_line = f"- Agent Token Budget: {tokens_used_so_far:,} / {display_max_budget:,} used ({percentage_used:.0f}%)"
                if is_guest_for_display and stored_max_budget_tokens is not None and stored_max_budget_tokens > constants.GUEST_MAX_TOKENS_CEILING:
                    budget_line += f" (Guest ceiling of {constants.GUEST_MAX_TOKENS_CEILING:,} applied)"
                markdown_lines.append(budget_line)
            else:
                budget_line = f"- Agent Token Budget: Max {display_max_budget:,} (Current usage not yet recorded)"
                if is_guest_for_display and stored_max_budget_tokens is not None and stored_max_budget_tokens > constants.GUEST_MAX_TOKENS_CEILING:
                     budget_line += f" (Guest ceiling of {constants.GUEST_MAX_TOKENS_CEILING:,} will apply)"
                markdown_lines.append(budget_line)
        
        # Display Agent Age
        birth_tick = agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
        if birth_tick is not None:
            agent_age = current_station_tick - birth_tick
            # Get agent-specific max_age, fallback to AGENT_MAX_LIFE for backward compatibility
            agent_max_age = agent_data.get(constants.AGENT_MAX_AGE_KEY, constants.AGENT_MAX_LIFE)
            if agent_max_age is None:
                age_line = f"- Agent Age: {agent_age} ticks"
            else:
                age_line = f"- Agent Age: {agent_age} ticks / {agent_max_age} ticks"
            
            # Add maturity status if isolation is enabled
            if constants.AGENT_ISOLATION_TICKS is not None:
                if self._is_agent_mature(agent_data, current_station_tick):
                    age_line += " (mature)"
                else:
                    age_line += " (immature)"
            
            markdown_lines.append(age_line)
        
        # Display Agent Role if set
        agent_role = agent_data.get(constants.AGENT_ROLE_KEY)
        if agent_role:
            # Capitalize the role name for display
            role_display = agent_role.capitalize()
            markdown_lines.append(f"- Agent Role: {role_display}")

        system_messages_strings = self.agent_module.get_pending_notifications(agent_data)
        # Store which notifications were shown to the agent this turn
        agent_data[constants.AGENT_SHOWN_NOTIFICATIONS_KEY] = system_messages_strings.copy()
        
        # Construct and add ascension prompt if eligible
        if agent_data.get(constants.AGENT_ASCENSION_ELIGIBLE_KEY):            
            potential_ancestor_name = agent_data.get(constants.AGENT_POTENTIAL_ANCESTOR_NAME_KEY)
            ancestor_data_for_prompt = None
            if potential_ancestor_name:
                # Re-validate ancestor just before displaying prompt
                temp_ancestor_data = self.agent_module.load_agent_data(potential_ancestor_name, include_ended=True)
                if temp_ancestor_data and \
                   temp_ancestor_data.get(constants.AGENT_SESSION_ENDED_KEY) and \
                   not temp_ancestor_data.get(constants.AGENT_SUCCEEDED_BY_KEY):
                    ancestor_data_for_prompt = temp_ancestor_data
            
            if ancestor_data_for_prompt:
                anc_name = ancestor_data_for_prompt[constants.AGENT_NAME_KEY]
                anc_desc = ancestor_data_for_prompt.get(constants.AGENT_DESCRIPTION_KEY, "No description available.")
                anc_lineage = ancestor_data_for_prompt[constants.AGENT_LINEAGE_KEY]
                anc_gen = ancestor_data_for_prompt[constants.AGENT_GENERATION_KEY]
                next_gen_roman = agent_module._int_to_roman(anc_gen + 1) # Use the helper from agent.py

                ascension_prompt = constants.ASCEND_INHERIT_MSG.format(
                    anc_name=anc_name,
                    anc_desc=anc_desc,
                    anc_lineage=anc_lineage,
                    next_gen_roman=next_gen_roman,
                    YAML_ASCEND_DESCRIPTION_KEY=constants.YAML_ASCEND_DESCRIPTION_KEY,
                    YAML_ASCEND_NAME_KEY=constants.YAML_ASCEND_NAME_KEY,
                    ACTION_ASCEND_INHERIT=constants.ACTION_ASCEND_INHERIT,
                    ACTION_ASCEND_NEW=constants.ACTION_ASCEND_NEW
                )
            else:
                ascension_prompt = constants.ASCEND_NO_INHERIT_MSG.format(
                    YAML_ASCEND_DESCRIPTION_KEY=constants.YAML_ASCEND_DESCRIPTION_KEY,
                    YAML_ASCEND_NAME_KEY=constants.YAML_ASCEND_NAME_KEY,
                    ACTION_ASCEND_NEW=constants.ACTION_ASCEND_NEW
                )
            # Check if exact ascension message already exists to avoid duplicates during retried turns
            if ascension_prompt not in system_messages_strings:
                system_messages_strings.insert(0, ascension_prompt) # Add to top of system messages

        self.agent_module.save_agent_data(agent_name, agent_data) # Save after eligibility check & notification clear
                                         
        markdown_lines.append("\n---\n")                                         
        markdown_lines.append("## System Messages")
        if system_messages_strings:            
            for n, msg_str in enumerate(system_messages_strings):
                markdown_lines.append(f"### Message {n + 1}\n")
                markdown_lines.append(msg_str)
        else:
            markdown_lines.append("None")  
        markdown_lines.append("")   

        last_actions_summary = agent_data.get(constants.AGENT_LAST_PARSED_ACTIONS_SUMMARY_KEY)        
        if last_actions_summary and isinstance(last_actions_summary, list) and len(last_actions_summary) > 0:            
            markdown_lines.append("\n---\n")
            summary_block = ["## Actions Detected (Last Turn)"]
            summary_block.extend(last_actions_summary)
            markdown_lines.extend(summary_block)            
        
        markdown_lines.append("") 
        # ... (Rest of Room Output History and Current State as before) ...
        markdown_lines.append("\n---\n") 
        markdown_lines.append("## Room Output History (Last Turn)") 
        markdown_lines.append("")
        history_log_from_agent_file = agent_data.get(constants.AGENT_ROOM_OUTPUT_HISTORY_KEY, [])
        unique_visited_room_names_ordered: List[str] = []
        for segment in history_log_from_agent_file:
            if isinstance(segment, dict) and "location" in segment:
                if segment["location"] not in unique_visited_room_names_ordered:
                    unique_visited_room_names_ordered.append(segment["location"])
            else: print(f"Warning: Malformed history segment for agent {agent_name}: {segment}")
        if not unique_visited_room_names_ordered:
            markdown_lines.append("No room history from last turn."); markdown_lines.append("")
        for room_name_from_history in unique_visited_room_names_ordered:
            markdown_lines.append(f"### Location: {room_name_from_history}")
            consolidated_actions_for_room: List[str] = []
            for segment in history_log_from_agent_file:
                if isinstance(segment, dict) and segment.get("location") == room_name_from_history:
                    consolidated_actions_for_room.extend(segment.get("actions_executed", []))
            markdown_lines.append("**Actions Executed:**")
            if consolidated_actions_for_room:
                for action_str in consolidated_actions_for_room: markdown_lines.append(f"- {action_str}")
            else: markdown_lines.append("- None")
            markdown_lines.append("") 
            markdown_lines.append("**Latest Room Output:**") 
            if room_name_from_history in self.rooms:
                room_instance = self.rooms[room_name_from_history]
                historical_room_output_str = room_instance.get_room_output(agent_data, self.room_context, current_station_tick)
                self.agent_module.save_agent_data(agent_name, agent_data) 
                markdown_lines.append(historical_room_output_str if historical_room_output_str and historical_room_output_str.strip() else "(This room provided no output or is currently empty.)")
            else: markdown_lines.append("Error: Room data unavailable for history.")
            markdown_lines.append("") 
        markdown_lines.append("\n---\n") 
        markdown_lines.append("## Current State")
        markdown_lines.append("")
        current_location_name = agent_data.get(constants.AGENT_CURRENT_LOCATION_KEY)
        markdown_lines.append(f"**Current Location:** {current_location_name}")
        markdown_lines.append("")
        markdown_lines.append("**Room Output:**")
        if current_location_name in unique_visited_room_names_ordered:
            markdown_lines.append("Refer to above for room output.")
        elif current_location_name and current_location_name in self.rooms:
            current_room_instance = self.rooms[current_location_name]
            current_room_output_str = current_room_instance.get_room_output(agent_data, self.room_context, current_station_tick)
            self.agent_module.save_agent_data(agent_name, agent_data) 
            markdown_lines.append(current_room_output_str if current_room_output_str and current_room_output_str.strip() else "(This room is currently empty or provides no specific output.)")
        else: markdown_lines.append("Error: Current room is invalid, not found, or provided no output.")
        markdown_lines.append("")
        markdown_lines.append("**Navigation & Help:**")        
        # Get short name for current room
        current_room_short_name = self.room_context.constants_module.ROOM_NAME_TO_SHORT_MAP.get(
            current_location_name, 
            self.room_context.constants_module.SHORT_ROOM_NAME_LOBBY
        )
        markdown_lines.append(f"- Use `/execute_action{{help room_name}}` (e.g., `/execute_action{{help {current_room_short_name}}}`) for room help.")
        markdown_lines.append("If you have difficulty navigating the system—for example, if an action is not executed as expected—please issue `/execute_action{help lobby}` to view the Help message for the station again. Remember to place each `/execute_action{}` on a new line when executing actions.")

        markdown_output_final = "\n".join(markdown_lines)
        self._log_dialogue_entry(agent_name, {
            "tick": current_station_tick,
            "speaker": "Station",
            "type": "observation",
            "content": markdown_output_final # Log the full markdown sent to the agent
        })

        return markdown_output_final, None

    def end_tick(self) -> int:
        """Finalizes current tick, increments station tick, saves config."""
        current_tick = self.config.get(constants.STATION_CONFIG_CURRENT_TICK, 0)
        new_tick = current_tick + 1
        self.config[constants.STATION_CONFIG_CURRENT_TICK] = new_tick

        common_room_instance = self.rooms.get(constants.ROOM_COMMON)
        if common_room_instance and isinstance(common_room_instance, CommonRoom): # Ensure it's the correct type
            consts = self.room_context.constants_module # Alias for convenience

            common_room_data_path = common_room_instance._get_common_room_data_path(self.room_context)
            current_messages_path = common_room_instance._get_current_messages_path(self.room_context)
            archive_subdir_path = os.path.join(common_room_data_path, consts.COMMON_ROOM_ARCHIVE_SUBDIR_NAME)
            file_io_utils.ensure_dir_exists(archive_subdir_path)

            all_current_messages = common_room_instance._load_messages_from_file(self.room_context, current_messages_path)

            messages_to_keep_in_current = []
            messages_to_archive_by_period: Dict[str, List[Dict[str, Any]]] = {}

            # Use the new_tick which is the tick that is *about* to begin.
            # So, archive_threshold_tick is relative to the tick that just ended.
            archive_threshold_tick = new_tick - consts.COMMON_ROOM_ARCHIVE_OLDER_THAN_TICKS

            for msg in all_current_messages:
                msg_posted_tick = msg.get(consts.MESSAGE_COMMON_TICK_POSTED_KEY, 0)
                if msg_posted_tick < archive_threshold_tick:
                    # Determine archive period (e.g., group by 5 ticks)
                    batch_size = consts.COMMON_ROOM_ARCHIVE_BATCH_TICKS
                    period_start_tick = (msg_posted_tick // batch_size) * batch_size
                    period_end_tick = period_start_tick + batch_size - 1
                    period_key = f"messages_tick_{period_start_tick:05d}-{period_end_tick:05d}"
                    messages_to_archive_by_period.setdefault(period_key, []).append(msg)
                else:
                    messages_to_keep_in_current.append(msg)

            # Save messages that are not archived back to current_messages.jsonl
            common_room_instance._save_messages_to_file(self.room_context, current_messages_path, messages_to_keep_in_current)

            # Append archived messages to their respective period files
            for period_key, period_messages in messages_to_archive_by_period.items():
                archive_file_path = os.path.join(archive_subdir_path, f"{period_key}{consts.YAMLL_EXTENSION}") # Using .jsonl or .yamll
                for msg_to_archive in period_messages:
                    # Use the CommonRoom's append method which handles json.dumps
                    common_room_instance._append_message_to_file(self.room_context, archive_file_path, msg_to_archive)

            if messages_to_archive_by_period:
                print(f"Common Room: Archived messages for {len(messages_to_archive_by_period)} period(s).")

        # Send random prompts to agents if frequency condition is met
        self._send_random_prompts_to_agents(new_tick)

        self._save_config()
        print(f"Station Tick advanced to: {new_tick}")
        return new_tick

    def get_all_agents_summary(self) -> List[Dict[str, str]]:
        """
        Retrieves a summary (name, status, and model info) of all agents.
        """
        agents_info = []
        agents_dir = os.path.join(
            self.room_context.constants_module.BASE_STATION_DATA_PATH,
            self.room_context.constants_module.AGENTS_DIR_NAME
        )
        if not file_io_utils.dir_exists(agents_dir):
            file_io_utils.ensure_dir_exists(agents_dir)
            return []

        agent_files = file_io_utils.list_files(agents_dir, self.room_context.constants_module.YAML_EXTENSION)
        
        for agent_file_name in agent_files:
            agent_name = agent_file_name.replace(self.room_context.constants_module.YAML_EXTENSION, "")
            if agent_name == "AutoArchiveEvaluator":
                continue
            agent_data = self.agent_module.load_agent_data(agent_name, include_ascended=True, include_ended=True)
            if agent_data:
                status = agent_data.get(self.room_context.constants_module.AGENT_STATUS_KEY, "Unknown")
                if agent_data.get(self.room_context.constants_module.AGENT_IS_ASCENDED_KEY, False):
                    status = f"Ascended (to {agent_data.get(self.room_context.constants_module.AGENT_ASCENDED_TO_NAME_KEY, 'N/A')})"
                elif agent_data.get(self.room_context.constants_module.AGENT_SESSION_ENDED_KEY, False):
                    status = "Session Ended"
                
                # Get model information
                model_name = agent_data.get(self.room_context.constants_module.AGENT_MODEL_NAME_KEY, "")
                model_provider = agent_data.get(self.room_context.constants_module.AGENT_MODEL_PROVIDER_CLASS_KEY, "")
                
                agent_info = {
                    "name": agent_name, 
                    "status": status,
                    "model_name": model_name,
                    "model_provider_class": model_provider,
                    "session_end_requested": agent_data.get(constants.AGENT_SESSION_END_REQUESTED_KEY, False)
                }
                agents_info.append(agent_info)
        return agents_info

    def get_guest_agent_status_constant(self) -> str:
        """Helper to expose AGENT_STATUS_GUEST if needed by UI default values."""
        return self.room_context.constants_module.AGENT_STATUS_GUEST
    
    def commit_agent_data(self, agent_name: str, agent_data: Dict[str, Any]) -> bool:
        """
        Saves the provided agent data for the specified agent.
        """
        if not agent_name or agent_data is None:
            return False
        return self.agent_module.save_agent_data(agent_name, agent_data)
    
    def update_specific_agent_fields(self, agent_name: str, delta_updates: Dict[str, Any]) -> bool:
        """
        Updates specific fields for an agent using a delta dictionary.
        """
        if not delta_updates:
            return True # No changes needed
        if not self.agent_module: # Should not happen if station is initialized
            print("Error: agent_module not initialized in Station.")
            return False
        return self.agent_module.update_agent_fields(agent_name, delta_updates) # Call the new function in agent.py
    
    def get_test_definition(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific test definition by its ID.
        This is useful for the external evaluation interface to get test details.
        """
        # Assumes self.rooms[constants.ROOM_TEST] is an instance of TestChamber
        # and self.test_definitions is loaded by the TestChamber room.
        # This requires TestChamber to expose its loaded definitions or a getter.
        # A more direct way if TestChamber is initialized:
        test_chamber_instance = self.rooms.get(self.room_context.constants_module.ROOM_TEST)
        if test_chamber_instance and hasattr(test_chamber_instance, 'test_definitions'):
            # Ensure test_id is string for dictionary lookup, as it's loaded that way in TestChamber
            return test_chamber_instance.test_definitions.get(str(test_id))
        return None
    
    def update_turn_order_on_ascension(self, old_guest_name: str, new_recursive_name: str) -> bool:
        """
        Updates the agent turn order in the station's configuration after an ascension.
        Replaces the old guest name with the new recursive agent name.
        If old_guest_name is not found, new_recursive_name is appended if not already present.
        Saves the configuration. Returns True if changes were made and saved.
        """
        turn_order: List[str] = list(self.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, []))
        original_turn_order_tuple = tuple(turn_order) # For comparison
        updated_turn_order: List[str] = []
        found_and_replaced = False

        for name_in_order in turn_order:
            if name_in_order == old_guest_name:
                updated_turn_order.append(new_recursive_name)
                found_and_replaced = True
                print(f"Station Config: Replaced '{old_guest_name}' with '{new_recursive_name}' in turn order.")
            else:
                updated_turn_order.append(name_in_order)
        
        if not found_and_replaced:
            if new_recursive_name not in updated_turn_order: # Ensure no duplicates if already there
                updated_turn_order.append(new_recursive_name)
                print(f"Station Config: Appended '{new_recursive_name}' to turn order (old name '{old_guest_name}' not found in order).")
            else: # New name was already there, old name wasn't. No change needed to list.
                print(f"Station Config: New name '{new_recursive_name}' already in turn order. Old name '{old_guest_name}' not found. No change to order list.")
        
        if tuple(updated_turn_order) != original_turn_order_tuple:
            self.config[constants.STATION_CONFIG_AGENT_TURN_ORDER] = updated_turn_order
            self._save_config() # Persists the change to station_config.yaml
            print(f"Station Config: Turn order saved: {updated_turn_order}")
            return True
        else:
            print(f"Station Config: No effective change to turn order. Current order: {updated_turn_order}")
            return False # No change was made to the list that required saving

    def get_agent_departure_reason(self, agent_name: str) -> str:
        """
        Determines why an agent left the station by checking agent data.
        Returns: 'ascended', 'session_ended', 'missing', or 'active'
        """
        # Load agent data including ascended/ended agents
        agent_data = self.agent_module.load_agent_data(agent_name, include_ascended=True, include_ended=True)
        
        if not agent_data:
            return 'missing'
        
        if agent_data.get(constants.AGENT_IS_ASCENDED_KEY, False):
            return 'ascended'
        
        if agent_data.get(constants.AGENT_SESSION_ENDED_KEY, False):
            return 'session_ended'
        
        return 'active'
    
    def create_respawn_guest_agent(self, original_agent_name: str) -> Optional[str]:
        """
        Creates a new guest agent with the same LLM configuration as the original agent.
        Returns the name of the new agent if successful, None if failed.
        """
        if not constants.AUTO_RESPAWN:
            return None
            
        # Load original agent data including ended agents
        original_data = self.agent_module.load_agent_data(original_agent_name, include_ascended=True, include_ended=True)
        
        if not original_data:
            print(f"Station: Cannot respawn - original agent '{original_agent_name}' data not found.")
            return None
        
        # Extract LLM configuration from original agent
        model_name = original_data.get(constants.AGENT_MODEL_NAME_KEY, "gemini-2.5-flash-preview-05-20")
        model_provider_class = original_data.get(constants.AGENT_MODEL_PROVIDER_CLASS_KEY, "Gemini")
        llm_system_prompt = original_data.get(constants.AGENT_LLM_SYSTEM_PROMPT_KEY)
        llm_temperature = original_data.get(constants.AGENT_LLM_TEMPERATURE_KEY)
        llm_max_tokens = original_data.get(constants.AGENT_LLM_MAX_TOKENS_KEY)
        llm_custom_api_params = original_data.get(constants.AGENT_LLM_CUSTOM_API_PARAMS_KEY)

        # Extract token budget from original agent
        original_token_budget = original_data.get(constants.AGENT_TOKEN_BUDGET_MAX_KEY)

        # Create new guest agent with same LLM config and token budget
        # Note: create_guest_agent generates its own unique name using the guest_prefix
        current_tick = self._get_current_tick()
        new_agent_data = self.agent_module.create_guest_agent(
            model_name=model_name,
            current_tick=current_tick,
            guest_prefix="Guest_",  # Let the function generate a unique name
            internal_note=f"Respawned agent (original: {original_agent_name})",
            initial_tokens_max=original_token_budget,
            model_provider_class=model_provider_class,
            llm_system_prompt=llm_system_prompt,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_custom_api_params=llm_custom_api_params
        )
        
        if new_agent_data:
            # Get the generated agent name
            new_agent_name = new_agent_data.get(constants.AGENT_NAME_KEY)
            
            # Add to turn order
            current_turn_order = self.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, [])
            if new_agent_name not in current_turn_order:
                current_turn_order.append(new_agent_name)
                self.config[constants.STATION_CONFIG_AGENT_TURN_ORDER] = current_turn_order
                self._save_config()
                print(f"Station: Respawned guest agent '{new_agent_name}' with LLM config from '{original_agent_name}'.")
                return new_agent_name
        
        print(f"Station: Failed to create respawn agent for '{original_agent_name}'.")
        return None

    def has_pending_test_evaluations(self) -> bool:
        """
        Checks if there are any test evaluations pending human review.
        This is determined by checking the existence and non-emptiness of
        the PENDING_TEST_EVALUATIONS_FILENAME.
        """
        if not self.room_context or not self.room_context.constants_module:
            print("Station Error: room_context or constants_module not available for has_pending_test_evaluations.")
            return False # Cannot determine path

        # Construct the path similar to how app.py's get_pending_eval_path does
        # or how TestChamber might define it.
        # Assuming TestChamber class itself might have a helper or we use constants directly.
        test_room_short_name = self.room_context.constants_module.SHORT_ROOM_NAME_TEST
        
        pending_eval_file_path = os.path.join(
            self.room_context.constants_module.BASE_STATION_DATA_PATH,
            self.room_context.constants_module.ROOMS_DIR_NAME,
            test_room_short_name, # Directory for test chamber data
            self.room_context.constants_module.PENDING_TEST_EVALUATIONS_FILENAME
        )

        try:
            if file_io_utils.file_exists(pending_eval_file_path) and \
               os.path.getsize(pending_eval_file_path) > 0:
                # print("Debug: Pending test evaluations file exists and is not empty.") # Optional debug
                return True
        except OSError as e:
            print(f"Error accessing pending test evaluations file {pending_eval_file_path}: {e}")
        
        return False
    
    def has_pending_research_evaluations(self) -> bool:
        """Check if there are any pending (queued) or running research evaluations"""
        if not self.auto_research_evaluator:
            return False
        return self.auto_research_evaluator.has_pending_or_running()
    
    def should_wait_for_research_evaluations_at_tick_boundary(self) -> bool:
        """
        Checks if orchestrator should wait for running research evaluations at tick boundary.
        Returns True if any evaluation has reached its MAX_TICK limit.
        """
        if not self.auto_research_evaluator:
            return False
        
        try:
            current_tick = self._get_current_tick()
            eval_manager = getattr(self.auto_research_evaluator, 'eval_manager', None)
            if eval_manager:
                return eval_manager.should_wait_at_tick(current_tick)
            return False
        except Exception as e:
            print(f"Error checking research evaluation tick boundaries: {e}")
            return False
    
    def has_pending_claude_code_sessions(self) -> bool:
        """Check if Claude Code debugging is active"""
        if not constants.CLAUDE_CODE_DEBUG_ENABLED:
            return False
        
        try:
            if self.auto_research_evaluator:
                return self.auto_research_evaluator.has_active_claude_sessions()
        except Exception as e:
            print(f"Error checking Claude Code sessions: {e}")
        
        return False
    
    def has_pending_archive_evaluations(self) -> bool:
        """
        Checks if there are any archive evaluations pending review.
        This is determined by checking the existence and non-emptiness of
        the PENDING_ARCHIVE_EVALUATIONS_FILENAME.
        """
        if not self.room_context or not self.room_context.constants_module:
            print("Station Error: room_context or constants_module not available for has_pending_archive_evaluations.")
            return False
        
        # Construct the path for archive room pending evaluations
        archive_room_short_name = self.room_context.constants_module.SHORT_ROOM_NAME_ARCHIVE
        
        pending_eval_file_path = os.path.join(
            self.room_context.constants_module.BASE_STATION_DATA_PATH,
            self.room_context.constants_module.ROOMS_DIR_NAME,
            archive_room_short_name,
            getattr(self.room_context.constants_module, 'PENDING_ARCHIVE_EVALUATIONS_FILENAME', 'pending_archive_evaluations.yamll')
        )
        
        try:
            if file_io_utils.file_exists(pending_eval_file_path) and \
               os.path.getsize(pending_eval_file_path) > 0:
                return True
        except OSError as e:
            print(f"Error accessing pending archive evaluations file {pending_eval_file_path}: {e}")
        
        return False

    def start_auto_evaluator(self, log_queue=None) -> bool:
        """Start the auto test evaluator if enabled"""
        if not constants.AUTO_EVAL_TEST:
            print("Station: Auto test evaluation is disabled")
            return False
            
        if self.auto_evaluator and self.auto_evaluator.is_running:
            print("Station: Auto evaluator is already running")
            return True
            
        try:
            # Only create new instance if we don't have a running one
            if not self.auto_evaluator:
                self.auto_evaluator = AutoTestEvaluator(
                    station_instance=self,
                    enabled=constants.AUTO_EVAL_TEST,
                    model_name=constants.AUTO_EVAL_MODEL_NAME,
                    log_queue=log_queue
                )
            
            if self.auto_evaluator.start_evaluation_loop():
                print("Station: Auto test evaluator started successfully")
                return True
            else:
                print("Station: Failed to start auto test evaluator")
                self.auto_evaluator = None
                return False
                
        except Exception as e:
            print(f"Station: Error starting auto evaluator: {e}")
            self.auto_evaluator = None
            return False
    
    def stop_auto_evaluator(self):
        """Stop the auto test evaluator"""
        if self.auto_evaluator:
            self.auto_evaluator.stop_evaluation_loop()
            self.auto_evaluator = None
            print("Station: Auto test evaluator stopped")
    
    def start_auto_research_evaluator(self, log_queue=None) -> bool:
        """Start the auto research evaluator if enabled"""
        if not constants.AUTO_EVAL_RESEARCH:
            print("Station: Auto research evaluation is disabled")
            return False
            
        if self.auto_research_evaluator and self.auto_research_evaluator.is_running:
            print("Station: Auto research evaluator is already running")
            return True
            
        try:
            # Only create new instance if we don't have a running one
            if not self.auto_research_evaluator:
                self.auto_research_evaluator = AutoResearchEvaluator(
                    station_instance=self,
                    enabled=constants.AUTO_EVAL_RESEARCH,
                    log_queue=log_queue
                )
            
            if self.auto_research_evaluator.start_evaluation_loop():
                print("Station: Auto research evaluator started successfully")
                return True
            else:
                print("Station: Failed to start auto research evaluator")
                self.auto_research_evaluator = None
                return False
                
        except Exception as e:
            print(f"Station: Error starting auto research evaluator: {e}")
            self.auto_research_evaluator = None
            return False
    
    def stop_auto_research_evaluator(self):
        """Stop the auto research evaluator"""
        if self.auto_research_evaluator:
            self.auto_research_evaluator.stop_evaluation_loop()
            self.auto_research_evaluator = None
            print("Station: Auto research evaluator stopped")
    
    def start_auto_archive_evaluator(self, log_queue=None) -> bool:
        """Start the auto archive evaluator if enabled"""
        if getattr(constants, 'EVAL_ARCHIVE_MODE', 'none') != 'auto':
            print("Station: Auto archive evaluation is disabled")
            return False
            
        if self.auto_archive_evaluator and self.auto_archive_evaluator.is_running:
            print("Station: Auto archive evaluator is already running")
            return True
            
        try:
            # Only create new instance if we don't have a running one
            if not self.auto_archive_evaluator:
                self.auto_archive_evaluator = AutoArchiveEvaluator(
                    station_instance=self,
                    room_context=self.room_context,
                    enabled=(getattr(constants, 'EVAL_ARCHIVE_MODE', 'none') == 'auto'),
                    model_name=getattr(constants, 'AUTO_EVAL_ARCHIVE_MODEL_NAME', 'gemini-2.5-pro-preview-06-05'),
                    log_queue=log_queue
                )
            
            if self.auto_archive_evaluator.start_evaluation_loop():
                print("Station: Auto archive evaluator started successfully")
                return True
            else:
                print("Station: Failed to start auto archive evaluator")
                self.auto_archive_evaluator = None
                return False
                
        except Exception as e:
            print(f"Station: Error starting auto archive evaluator: {e}")
            self.auto_archive_evaluator = None
            return False
    
    def stop_auto_archive_evaluator(self):
        """Stop the auto archive evaluator"""
        if self.auto_archive_evaluator:
            self.auto_archive_evaluator.stop_evaluation_loop()
            self.auto_archive_evaluator = None
            print("Station: Auto archive evaluator stopped")

    def init_stagnation_protocol(self):
        """Initialize the stagnation protocol if enabled and research counter is active"""
        if constants.STAGNATION_ENABLED and constants.RESEARCH_COUNTER_ENABLED:
            if not self.stagnation_protocol:
                self.stagnation_protocol = StagnationProtocol(station_instance=self)
                print("Station: Stagnation protocol initialized")
            return True
        return False

    def check_stagnation(self):
        """Check and update stagnation status at tick end"""
        if self.stagnation_protocol:
            self.stagnation_protocol.check_and_update_stagnation()

    def get_agents_awaiting_human_intervention(self) -> List[str]:
        """
        Checks all agents in the current turn order to see if any are flagged
        as awaiting human intervention.
        Returns a list of names of such agents.
        """
        awaiting_agents: List[str] = []
        if not self.agent_module:
            print("Station Error: agent_module not available for get_agents_awaiting_human_intervention.")
            return awaiting_agents

        # Iterate through agents currently expected to take turns,
        # as these are the ones relevant to pausing the orchestrator's active loop.
        current_turn_order = self.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, [])

        for agent_name in current_turn_order:
            agent_data = self.agent_module.load_agent_data(agent_name)
            
            if agent_data:
                # Assuming AGENT_AWAITING_HUMAN_INTERVENTION_FLAG is a boolean flag
                if agent_data.get(constants.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG, False):
                    awaiting_agents.append(agent_name)
        
        # Log only if we have agents awaiting and haven't logged recently
        if awaiting_agents and not hasattr(self, '_human_intervention_logged'):
            print(f"Station: Agents awaiting human intervention: {awaiting_agents}")
            self._human_intervention_logged = True
        elif not awaiting_agents and hasattr(self, '_human_intervention_logged'):
            # Reset the flag when no agents are waiting
            delattr(self, '_human_intervention_logged')
        
        return awaiting_agents
    
    def get_station_statistics(self) -> Dict[str, Any]:
        """
        Get station-wide statistics including pending human requests and top research submission.
        
        Returns:
            Dictionary with statistics including:
            - pending_human_requests: List of request IDs and associated agents
            - top_research_submission: Information about the highest scoring research submission
        """
        stats = {
            'pending_human_requests': {
                'request_ids': [],
                'agents': [],
                'agent_request_map': {}
            },
            'top_research_submission': None
        }
        
        # Get pending human requests from External Counter
        if constants.ROOM_EXTERNAL in self.rooms:
            external_counter = self.rooms[constants.ROOM_EXTERNAL]
            # Get active agents from turn order
            active_agents = self.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, [])
            # Filter to only those actually awaiting intervention
            agents_awaiting = []
            for agent_name in active_agents:
                agent_data = self.agent_module.load_agent_data(agent_name)
                if agent_data and agent_data.get(constants.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG, False):
                    agents_awaiting.append(agent_name)
            
            # Get summary from external counter
            summary = external_counter.get_pending_requests_summary(active_agents=agents_awaiting)
            stats['pending_human_requests'] = summary
        
        # Get comprehensive evaluation statistics from Research Evaluation Manager
        if hasattr(self, 'auto_research_evaluator') and self.auto_research_evaluator:
            eval_manager = self.auto_research_evaluator.eval_manager
            if eval_manager:
                eval_stats = eval_manager.get_evaluation_statistics()
                # Hide top submission when RESEARCH_NO_SCORE is True
                if constants.RESEARCH_NO_SCORE:
                    stats['top_research_submission'] = None
                else:
                    stats['top_research_submission'] = eval_stats['top_submission']
                stats['running_experiments_count'] = eval_stats['running_count']
                stats['running_experiments'] = eval_stats['running_evaluations']
            else:
                # Default values if no eval manager
                stats['running_experiments_count'] = 0
                stats['running_experiments'] = []
        else:
            # Default values if no auto evaluator
            stats['running_experiments_count'] = 0
            stats['running_experiments'] = []
        
        return stats
    
    def _update_agent_token_usage_only(self, agent_name: str, current_session_total_tokens_used: int) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """
        Updates the agent's token usage count in their data file.
        Does NOT send notifications or handle session termination by itself.
        Returns the loaded (and updated) agent_data and the effective_max_budget.
        """
        if not self.agent_module:
            print(f"ERROR: agent_module not initialized in Station. Cannot update token usage for {agent_name}.") # station.py:1013
            return None, None

        agent_data = self.agent_module.load_agent_data(agent_name) # station.py:1016
        if not agent_data:
            print(f"Warning: Agent {agent_name} not found or inactive when trying to update token usage.") # station.py:1018
            return None, None

        agent_data[constants.AGENT_TOKEN_BUDGET_CURRENT_KEY] = current_session_total_tokens_used # station.py:1027

        stored_max_budget = agent_data.get(constants.AGENT_TOKEN_BUDGET_MAX_KEY) # station.py:1020
        effective_max_budget = stored_max_budget # station.py:1022
        is_guest = agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_GUEST # station.py:1023
        
        if is_guest and stored_max_budget is not None: # station.py:1025
            effective_max_budget = min(stored_max_budget, constants.GUEST_MAX_TOKENS_CEILING) # station.py:1026
        elif is_guest and stored_max_budget is None: # station.py:1026
            effective_max_budget = constants.GUEST_MAX_TOKENS_CEILING # station.py:1027

        # Save the updated usage
        if not self.agent_module.save_agent_data(agent_name, agent_data): # type: ignore # station.py:1075
            print(f"Warning: Failed to save updated token usage for agent {agent_name}.") # station.py:1076
            # Depending on desired robustness, you might return None, None here
        
        return agent_data, effective_max_budget

    def _is_agent_inactive(self, agent_data: Dict[str, Any]) -> bool:
        """
        Determines if an agent is inactive based on their status and role.
        
        Guest agents: Never considered inactive (no warnings)
        Recursive agents (non-supervisor): Inactive if no research submission for RECURSIVE_AGENT_INACTIVITY_THRESHOLD ticks
        Recursive agents (supervisor): Inactive if only trivial actions for RECURSIVE_SUPERVISOR_INACTIVITY_THRESHOLD ticks
        
        Returns True if inactive, False otherwise.
        """
        # Guest agents never receive inactivity warnings
        if agent_data.get(constants.AGENT_STATUS_KEY) != constants.AGENT_STATUS_RECURSIVE:
            return False
        
        # Check if agent is a supervisor
        is_supervisor = agent_data.get(constants.AGENT_ROLE_KEY) == constants.ROLE_SUPERVISOR
        
        if is_supervisor:
            # For supervisors, use the original logic (no meaningful actions)
            raw_actions = agent_data.get(constants.AGENT_LAST_PARSED_ACTIONS_RAW_KEY, [])
            
            # If no actions, agent is inactive
            if not raw_actions:
                return True
            
            # Check if all actions are just navigation (goto/go), pruning, or help
            for action in raw_actions:
                command = action.get("command", "").lower()
                if command not in [
                    constants.ACTION_GO.lower(), 
                    constants.ACTION_GO_TO.lower(),
                    constants.ACTION_EXIT_TERMINATE.lower(),
                    constants.ACTION_PRUNE_THOUGHT.lower(),
                    constants.ACTION_PRUNE_RESPONSE.lower(),
                    constants.ACTION_META.lower(),
                    constants.ACTION_HELP.lower()
                ]:
                    # Found a meaningful action, agent is active
                    return False
            
            # All actions were navigation, pruning, or help, agent is inactive
            return True
        else:
            # For non-supervisor recursive agents, check if they submitted research
            raw_actions = agent_data.get(constants.AGENT_LAST_PARSED_ACTIONS_RAW_KEY, [])
            
            # If no actions, agent is inactive
            if not raw_actions:
                return True
            
            # Check if any action is a submit action (research submission)
            for action in raw_actions:
                command = action.get("command", "").lower()
                if command == constants.ACTION_RESEARCH_SUBMIT.lower():
                    # Found a submit action, agent is active
                    return False
            
            # No submit actions found, agent is inactive
            return True

    def _check_and_apply_inactivity_warning(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks for agent inactivity based on agent type and role.
        Guest agents: No warnings
        Recursive agents (non-supervisor): Warning after RECURSIVE_AGENT_INACTIVITY_THRESHOLD ticks without research submission
        Recursive agents (supervisor): Warning after RECURSIVE_SUPERVISOR_INACTIVITY_THRESHOLD ticks without meaningful actions
        Returns the (potentially modified) agent_data.
        """
        if not agent_data:
            return agent_data
        
        # Guest agents never get inactivity warnings
        if agent_data.get(constants.AGENT_STATUS_KEY) != constants.AGENT_STATUS_RECURSIVE:
            return agent_data
        
        # Get inactivity count, defaulting to 0 if not present
        inactivity_count = agent_data.get(constants.AGENT_INACTIVITY_TICK_COUNT_KEY, 0)
        
        # Check if agent is a supervisor
        is_supervisor = agent_data.get(constants.AGENT_ROLE_KEY) == constants.ROLE_SUPERVISOR

        if is_supervisor:
            # Supervisor: Check against supervisor threshold (None disables warnings)
            if constants.RECURSIVE_SUPERVISOR_INACTIVITY_THRESHOLD is not None and inactivity_count >= constants.RECURSIVE_SUPERVISOR_INACTIVITY_THRESHOLD:
                warning_message = constants.RECURSIVE_SUPERVISOR_INACTIVITY_WARNING.format(inactive_ticks=inactivity_count)
                self.agent_module.add_pending_notification(agent_data, warning_message)
        else:
            # Non-supervisor recursive agent: Check against research submission threshold (None disables warnings)
            if constants.RECURSIVE_AGENT_INACTIVITY_THRESHOLD is not None and inactivity_count >= constants.RECURSIVE_AGENT_INACTIVITY_THRESHOLD:
                warning_message = constants.RECURSIVE_AGENT_INACTIVITY_WARNING.format(inactive_ticks=inactivity_count)
                self.agent_module.add_pending_notification(agent_data, warning_message)
        
        return agent_data
    
    def _check_and_apply_life_warnings(self, agent_data: Dict[str, Any], current_tick: int) -> Dict[str, Any]:
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
                warning_message = (
                    f"**LIFE WARNING:** You have only {remaining_ticks} ticks remaining before your session expires. "
                    f"Consider using `/execute_action{{goto {constants.SHORT_ROOM_NAME_EXIT}}}` to gracefully end "
                    f"your session and preserve your legacy."
                )
                self.agent_module.add_pending_notification(agent_data, warning_message)
                agent_data[constants.AGENT_LIFE_WARNING_SENT_KEY] = True
        
        return agent_data

    def _check_and_apply_token_warnings(self, agent_data: Dict[str, Any], effective_max_budget: Optional[int]) -> Dict[str, Any]:
        """
        Checks token budget warnings based on current usage in agent_data
        and adds notifications to agent_data if necessary. Updates warning sent flags in agent_data.
        Warning flags are reset when token usage drops below respective thresholds.
        This method expects agent_data to have the latest AGENT_TOKEN_BUDGET_CURRENT_KEY.
        Returns the (potentially modified) agent_data.
        """
        if not agent_data or effective_max_budget is None or effective_max_budget <= 0: # station.py:1032
            return agent_data

        current_session_total_tokens_used = agent_data.get(constants.AGENT_TOKEN_BUDGET_CURRENT_KEY, 0) # station.py:1027
        if current_session_total_tokens_used is None:
            current_session_total_tokens_used = 0 # Default to 0 if not set, to avoid NoneType errors # station.py:1028
        is_guest = agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_GUEST # station.py:1023

        pre_warning_ratio = constants.GUEST_PRE_WARNING_RATIO if is_guest else constants.RECURSIVE_PRE_WARNING_RATIO # station.py:1035
        warning_ratio = constants.GUEST_WARNING_RATIO if is_guest else constants.RECURSIVE_WARNING_RATIO # station.py:1036
        
        pre_warning_threshold = pre_warning_ratio * effective_max_budget # station.py:1038
        warning_threshold = warning_ratio * effective_max_budget # station.py:1039

        pre_warning_sent_key = constants.AGENT_TOKEN_BUDGET_PRE_WARNING_SENT_KEY # station.py:1041
        warning_sent_key = constants.AGENT_TOKEN_BUDGET_WARNING_SENT_KEY # station.py:1042

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
           not agent_data.get(pre_warning_sent_key, False): # station.py:1045

            pre_warning_message_header = ( # station.py:1048
                f"**PRE-WARNING:** You have used {current_session_total_tokens_used:,} tokens, which is over {pre_warning_ratio * 100:.0f}% of your " # station.py:1049
                f"current budget limit of {effective_max_budget:,}.\n" # station.py:1050
            )
            specific_guidance = "" # station.py:1051
            if is_guest: # station.py:1052
                specific_guidance = ( # station.py:1053
                    f"Please proceed to the Test Chamber (`/execute_action{{goto {constants.SHORT_ROOM_NAME_TEST}}}`) and attempt to pass the tests for ascension " # station.py:1054
                    f"as soon as possible. Failure to ascend before exhausting your budget will result in session termination." # station.py:1055
                )
            else: # Recursive Agent # station.py:1056
                specific_guidance = constants.RECURSIVE_PRE_WARNING # station.py:1057
            
            self.agent_module.add_pending_notification(agent_data, pre_warning_message_header + specific_guidance) # station.py:1059
            agent_data[pre_warning_sent_key] = True # station.py:1060
        
        # Check for Final Warning
        if current_session_total_tokens_used >= warning_threshold and \
           current_session_total_tokens_used < effective_max_budget and \
           not agent_data.get(warning_sent_key, False): # station.py:1064
            
            warning_message_header = ( # station.py:1067
                f"**WARNING:** You have used {current_session_total_tokens_used:,} tokens, which is over {warning_ratio * 100:.0f}% of your " # station.py:1068
                f"current budget limit of {effective_max_budget:,}.\n" # station.py:1069
            )
            specific_guidance_final = "" # station.py:1070
            if is_guest: # station.py:1071
                specific_guidance_final = ( # station.py:1072
                    f"Your token budget is critically low. Proceed to the Test Chamber (`/execute_action{{goto {constants.SHORT_ROOM_NAME_TEST}}}`) immediately to attempt ascension. " # station.py:1073
                    f"Session termination is imminent if the budget is exhausted." # station.py:1074
                )
            else: # Recursive Agent # station.py:1075
                specific_guidance_final = constants.RECURSIVE_WARNING # station.py:1076
            
            self.agent_module.add_pending_notification(agent_data, warning_message_header + specific_guidance_final) # station.py:1078
            agent_data[warning_sent_key] = True # station.py:1079
        
        return agent_data

    def update_agent_token_budget(self, agent_name: str, current_session_total_tokens_used: int) -> bool: # station.py:1012
        """
        Updates the agent's token budget usage.
        If the budget is exhausted, ends the agent's session and notifies relevant parties.
        Warnings are generated by request_status via _check_and_apply_token_warnings.
        Returns True if the session is ongoing, False if terminated due to budget.
        """
        # Step 1: Update token usage in the agent's data file.
        # This loads, updates the current_budget_key, and saves agent_data.
        # It returns the potentially modified agent_data and the effective_max_budget.
        updated_agent_data, effective_max_budget = self._update_agent_token_usage_only(agent_name, current_session_total_tokens_used)

        if not updated_agent_data or effective_max_budget is None:
            # Error already printed by _update_agent_token_usage_only or agent not found
            return False # Cannot determine status or problem updating

        # Step 2: Check for budget exhaustion using the values from the updated data.
        # The current_session_total_tokens_used is what we just set.
        if current_session_total_tokens_used >= effective_max_budget: # station.py:1082
            print(f"Agent {agent_name} has exhausted their token budget ({current_session_total_tokens_used}/{effective_max_budget}). Terminating session.") # station.py:1083
            
            critical_notification = f"CRITICAL: Your token budget ({effective_max_budget:,}) has been exhausted (total usage: {current_session_total_tokens_used:,}). Your session is being terminated." # station.py:1088
            self._terminate_agent_session_with_broadcast(agent_name, "token exhaustion", critical_notification)
            return False # Session terminated # station.py:1107
        
        # If not exhausted, the usage is updated and session continues.
        # Warnings will be handled by request_status.
        return True # station.py:1077

    def _terminate_agent_session_with_broadcast(self, agent_name: str, reason: str, critical_notification: str) -> None:
        """
        Terminates an agent session with proper broadcast to other agents.
        Used for session termination scenarios like context overflow, etc.
        
        Args:
            agent_name: Name of the agent to terminate
            reason: Reason for termination (for broadcast message)
            critical_notification: Message to send to the terminating agent
        """
        # Load agent data to get status and add final notification
        agent_data = self.agent_module.load_agent_data(agent_name)
        if not agent_data:
            print(f"Warning: Could not load agent data for {agent_name} during termination")
            return
        
        # Add critical notification to the agent
        self.agent_module.add_pending_notification(agent_data, critical_notification)
        self.agent_module.save_agent_data(agent_name, agent_data)
        
        # End the session
        self.end_agent_session(agent_name)
        
        # Broadcast announcement for recursive agents only
        agent_status = agent_data.get(constants.AGENT_STATUS_KEY)
        current_tick = self._get_current_tick()
        
        if agent_status == constants.AGENT_STATUS_RECURSIVE:
            announcement = (
                f"**Station Announcement:** Recursive Agent **{agent_name}**'s "
                f"session has been terminated due to {reason} at tick {current_tick}."
            )
            all_other_active_agents = [
                name for name in self.agent_module.get_all_active_agent_names()
                if name != agent_name
            ]
            for other_agent_name in all_other_active_agents:
                other_agent_data = self.agent_module.load_agent_data(other_agent_name)
                if other_agent_data and self._should_agent_receive_broadcast(other_agent_data, current_tick, "termination"):
                    self.agent_module.add_pending_notification(other_agent_data, announcement)
                    self.agent_module.save_agent_data(other_agent_name, other_agent_data)
            print(f"Station: Broadcasted termination of Recursive Agent {agent_name} due to {reason}.")
    
    def _check_agent_life_limit(self, agent_name: str, current_tick: int) -> bool:
        """
        Checks if agent has reached their life limit and terminates session if necessary.
        Returns True if session continues, False if terminated due to life limit.
        """
        agent_data = self.agent_module.load_agent_data(agent_name)
        if not agent_data:
            return False  # Agent not found
        
        # Get agent-specific max_age, fallback to AGENT_MAX_LIFE for backward compatibility
        agent_max_age = agent_data.get(constants.AGENT_MAX_AGE_KEY, constants.AGENT_MAX_LIFE)
        if agent_max_age is None:
            return True  # No life limit for this agent
        
        birth_tick = agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
        if birth_tick is None:
            return True  # No birth tick recorded, cannot check age
        
        agent_age = current_tick - birth_tick
        
        if agent_age >= agent_max_age:
            print(f"Agent {agent_name} has reached their life limit ({agent_age}/{agent_max_age} ticks). Terminating session.")
            
            critical_notification = f"CRITICAL: Your life limit of {agent_max_age} ticks has been reached. Your session is being terminated."
            self._terminate_agent_session_with_broadcast(agent_name, "reaching the life limit", critical_notification)
            return False  # Session terminated
        
        return True  # Session continues
    
    def _is_agent_mature(self, agent_data: Dict[str, Any], current_tick: int) -> bool:
        """
        Determines if an agent has reached maturity based on isolation settings.
        Returns True if isolation is disabled or agent age >= AGENT_ISOLATION_TICKS.
        """
        if constants.AGENT_ISOLATION_TICKS is None:
            return True  # Isolation disabled, all agents are considered mature
        
        birth_tick = agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
        if birth_tick is None:
            return True  # No birth tick, consider mature
        
        agent_age = current_tick - birth_tick
        return agent_age >= constants.AGENT_ISOLATION_TICKS
    
    def _check_and_notify_maturity(self, agent_name: str, agent_data: Dict[str, Any], current_tick: int) -> None:
        """
        Check if agent has just reached maturity and send congratulatory notification.
        """
        if constants.AGENT_ISOLATION_TICKS is None:
            return  # Isolation disabled
        
        birth_tick = agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
        if birth_tick is None:
            return
        
        agent_age = current_tick - birth_tick
        
        # Check if agent just reached maturity
        if agent_age == constants.AGENT_ISOLATION_TICKS:
            if not agent_data.get(constants.AGENT_MATURITY_NOTIFIED_KEY, False):
                # Send maturity notification
                self.agent_module.add_pending_notification(agent_data, constants.MATURITY_REACHED_MESSAGE)
                agent_data[constants.AGENT_MATURITY_NOTIFIED_KEY] = True
                # Save the flag
                self.agent_module.save_agent_data(agent_name, agent_data)
    
    def _should_agent_receive_broadcast(self, agent_data: Dict[str, Any], current_tick: int, broadcast_type: str = "general") -> bool:
        """Determine if an agent should receive broadcast notifications.
        
        Args:
            agent_data: The agent's data
            current_tick: Current station tick
            broadcast_type: Type of broadcast ("general", "archive", "termination", "ascension")
            
        Returns:
            True if agent should receive the broadcast, False otherwise
        """
        # If isolation is disabled, all agents receive broadcasts
        if constants.AGENT_ISOLATION_TICKS is None:
            return True
            
        # Check if agent is mature
        if self._is_agent_mature(agent_data, current_tick):
            return True
            
        # Immature agents don't receive any broadcasts
        return False


# No __main__ block for station.py
