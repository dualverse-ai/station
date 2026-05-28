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
import copy
import time
import traceback
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, List, Dict, Optional, Tuple, Type, cast

from station import __version__
from station import constants
from station import file_io_utils
from station import agent as agent_module
from station import agent_summary
from station import capsule as capsule_module
from station import capsule_index
from station import supervisor_utils
from station import system_messages
from station import session_end_flow
from station import tick_timing
from station.action_parser import ActionParser # Assuming ActionParser is a class
from station.lineage_evolution import LineageEvolutionManager
from station.stagnation_protocol import StagnationProtocol
from station.base_room import BaseRoom, RoomContext, InternalActionHandler, LoggingInternalActionHandlerWrapper

# Import all specific room classes
from station.rooms.lobby import LobbyRoom 
from station.rooms.reflect import ReflectionChamber
from station.rooms.common import CommonRoom
from station.rooms.public_memory import PublicMemoryRoom
from station.rooms.private_memory import PrivateMemoryRoom 
from station.rooms.archive import ArchiveRoom 
from station.rooms.mail import MailRoom
from station.rooms.misc import MiscRoom
from station.rooms.admin import AdminCounter
from station.rooms.token_management import TokenManagementRoom
from station.rooms.research_center import ResearchCenter
from station.rooms.external_counter import ExternalCounter
from station.rooms.maze import MazeRoom
from station.rooms.theory import TheoryRoom
from station.rooms.exit import ExitRoom
from station.eval_research import AutoResearchEvaluator, EvaluationManager, get_evaluation_display_info
from station.eval_research.runtime_paths import ensure_runtime_layout
from station.eval_archive import AutoArchiveEvaluator
from station.eval_archive.surveyor import AutoArchiveSurveyor
from station.eval_theory import AutoTheoryEvaluator
from station.eval_external import AutoExternalReporter
# Add other room imports here as they are created:
# from station.rooms.reflection_chamber import ReflectionChamber
# from station.rooms.public_memory_room import PublicMemoryRoom
# ... etc.

_REQUEST_STATUS_TIMING_LOG_THRESHOLD_SECONDS = 0.25


def _merge_unique_list_items(*lists: Any) -> List[Any]:
    merged: List[Any] = []
    for values in lists:
        if not isinstance(values, list):
            continue
        for item in values:
            if item not in merged:
                merged.append(item)
    return merged


def _merge_pending_notifications_after_turn(
    turn_pending_notifications: Any,
    latest_pending_notifications: Any,
    shown_notifications: Any,
) -> List[Any]:
    latest_values = latest_pending_notifications if isinstance(latest_pending_notifications, list) else []
    shown_values = shown_notifications if isinstance(shown_notifications, list) else []
    latest_unshown_notifications = [
        notification
        for notification in latest_values
        if notification not in shown_values
    ]
    return _merge_unique_list_items(turn_pending_notifications, latest_unshown_notifications)


def _merge_pending_dialogue_tick_protections_after_turn(
    turn_pending_records: Any,
    latest_pending_records: Any,
    turn_start_pending_records: Any,
) -> List[Any]:
    turn_values = turn_pending_records if isinstance(turn_pending_records, list) else []
    latest_values = latest_pending_records if isinstance(latest_pending_records, list) else []
    start_values = turn_start_pending_records if isinstance(turn_start_pending_records, list) else []
    concurrent_values = [
        record
        for record in latest_values
        if record not in start_values
    ]
    return _merge_unique_list_items(turn_values, concurrent_values)


def _merge_protected_dialogue_ticks_after_turn(
    turn_protected_records: Any,
    latest_protected_records: Any,
    turn_start_protected_records: Any,
) -> List[Any]:
    turn_values = turn_protected_records if isinstance(turn_protected_records, list) else []
    latest_values = latest_protected_records if isinstance(latest_protected_records, list) else []
    start_values = turn_start_protected_records if isinstance(turn_start_protected_records, list) else []
    new_turn_values = [
        record
        for record in turn_values
        if record not in start_values
    ]
    return _merge_unique_list_items(latest_values, new_turn_values)


def _pending_notification_texts(agent_data: Dict[str, Any]) -> List[str]:
    pending_notifications = agent_data.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, [])
    if not isinstance(pending_notifications, list):
        return []
    return [
        notification
        for notification in pending_notifications
        if isinstance(notification, str)
    ]


def _has_pending_help_notification(source: str, pending_texts: List[str]) -> bool:
    if not source.startswith("room:"):
        return False
    short_room_name = source.removeprefix("room:")
    full_room_name = constants.SHORT_ROOM_NAME_TO_FULL_MAP.get(short_room_name)
    if not full_room_name:
        return False
    return any(f"Help for {full_room_name}:" in text for text in pending_texts)


def _drop_stale_pending_dialogue_tick_protections(agent_data: Dict[str, Any]) -> bool:
    if not isinstance(agent_data, dict):
        return False

    pending = agent_data.get(constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY)
    if not isinstance(pending, list):
        return False

    pending_texts = _pending_notification_texts(agent_data)
    filtered_pending: List[Any] = []
    for record in pending:
        if not isinstance(record, dict):
            filtered_pending.append(record)
            continue

        reason = record.get(constants.PROTECTED_DIALOGUE_REASON_KEY)
        source = str(record.get(constants.PROTECTED_DIALOGUE_SOURCE_KEY, ""))
        if (
            reason == constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ
            and not agent_data.get(constants.AGENT_PENDING_CURRENT_RESEARCH_TASK_READ_KEY)
        ):
            continue
        if (
            reason == constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE
            and not any("**Architect Message**" in text for text in pending_texts)
        ):
            continue
        if (
            reason == constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP
            and not _has_pending_help_notification(source, pending_texts)
        ):
            continue
        filtered_pending.append(record)

    if len(filtered_pending) == len(pending):
        return False
    agent_data[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY] = filtered_pending
    return True


def _gemini_response_reminder_for_agent(agent_data: Dict[str, Any], constants_module: Any = constants) -> Optional[str]:
    provider = str(agent_data.get(constants_module.AGENT_MODEL_PROVIDER_CLASS_KEY) or "").strip().lower()
    if provider != "gemini":
        return None

    legacy_reminder = getattr(constants_module, "GEMINI_REPONSE_SUFFIX", None)
    reminder = legacy_reminder if legacy_reminder is not None else getattr(constants_module, "GEMINI_RESPONSE_SUFFIX", "")
    if reminder is None:
        return None

    reminder_text = str(reminder).strip()
    return reminder_text or None


def _read_status_map(agent_data: Any, room_key: str) -> Dict[str, bool]:
    if not isinstance(agent_data, dict):
        return {}
    room_data = agent_data.get(room_key)
    if not isinstance(room_data, dict):
        return {}
    read_status = room_data.get(constants.AGENT_ROOM_STATE_READ_STATUS_KEY)
    if not isinstance(read_status, dict):
        return {}
    return {str(item_id): bool(read) for item_id, read in read_status.items() if read}


def _merge_concurrent_room_read_status(
    target_agent_data: Dict[str, Any],
    room_key: str,
    latest_agent_data: Any,
    turn_start_agent_data: Any,
) -> None:
    latest_read_status = _read_status_map(latest_agent_data, room_key)
    turn_start_read_status = _read_status_map(turn_start_agent_data, room_key)
    concurrent_additions = {
        item_id: True
        for item_id in latest_read_status
        if item_id not in turn_start_read_status
    }
    if not concurrent_additions:
        return
    room_data = target_agent_data.setdefault(room_key, {})
    if not isinstance(room_data, dict):
        room_data = {}
        target_agent_data[room_key] = room_data
    current_read_status = room_data.get(constants.AGENT_ROOM_STATE_READ_STATUS_KEY)
    if not isinstance(current_read_status, dict):
        current_read_status = {}
    current_read_status.update(concurrent_additions)
    room_data[constants.AGENT_ROOM_STATE_READ_STATUS_KEY] = current_read_status


def _save_request_status_snapshot_atomically(
    agent_manager: Any,
    agent_name: str,
    turn_start_agent_data: Dict[str, Any],
    snapshot_agent_data: Dict[str, Any],
    *,
    refresh_shown_notifications: bool = False,
    apply_pending_protections_tick: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Persist a request-status snapshot without overwriting concurrent updates."""
    update_agent = getattr(agent_manager, "update_agent_with_function", None)
    if not callable(update_agent):
        return None

    rendered_agent_data: Optional[Dict[str, Any]] = None

    def update_func(latest_agent_data: Dict[str, Any]) -> None:
        nonlocal rendered_agent_data

        def _merge_snapshot_into_target(
            target: Dict[str, Any],
            start: Any,
            snapshot: Any,
            *,
            top_level: bool = False,
        ) -> None:
            if not isinstance(snapshot, dict):
                return
            start_dict = start if isinstance(start, dict) else {}
            for key, snapshot_value in snapshot.items():
                start_value = start_dict.get(key)

                if top_level and key == constants.AGENT_NOTIFICATIONS_PENDING_KEY:
                    latest_pending = target.get(key, [])
                    if not isinstance(latest_pending, list):
                        latest_pending = []
                    start_pending = start_value if isinstance(start_value, list) else []
                    snapshot_pending = snapshot_value if isinstance(snapshot_value, list) else []
                    new_pending = [
                        notification
                        for notification in snapshot_pending
                        if notification not in start_pending
                    ]
                    target[key] = _merge_unique_list_items(latest_pending, new_pending)
                    continue

                if top_level and key == constants.AGENT_SHOWN_NOTIFICATIONS_KEY:
                    target[key] = list(snapshot_value) if isinstance(snapshot_value, list) else []
                    continue

                if top_level and key == constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY:
                    target[key] = _merge_pending_dialogue_tick_protections_after_turn(
                        snapshot_value,
                        target.get(key, []),
                        start_value,
                    )
                    continue

                if top_level and key == constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY:
                    target[key] = _merge_protected_dialogue_ticks_after_turn(
                        snapshot_value,
                        target.get(key, []),
                        start_value,
                    )
                    continue

                if isinstance(snapshot_value, dict):
                    target_value = target.get(key)
                    if not isinstance(target_value, dict):
                        if start_value != snapshot_value:
                            target[key] = copy.deepcopy(snapshot_value)
                        continue
                    _merge_snapshot_into_target(target_value, start_value, snapshot_value)
                    continue

                if isinstance(snapshot_value, list):
                    if start_value != snapshot_value:
                        target[key] = copy.deepcopy(snapshot_value)
                    continue

                if start_value != snapshot_value:
                    target[key] = copy.deepcopy(snapshot_value)

        _merge_snapshot_into_target(latest_agent_data, turn_start_agent_data, snapshot_agent_data, top_level=True)
        if apply_pending_protections_tick is not None:
            _drop_stale_pending_dialogue_tick_protections(latest_agent_data)
            apply_pending = getattr(agent_manager, "apply_pending_dialogue_tick_protections", None)
            if callable(apply_pending):
                apply_pending(latest_agent_data, apply_pending_protections_tick)
        if refresh_shown_notifications:
            latest_pending_notifications = latest_agent_data.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, [])
            if not isinstance(latest_pending_notifications, list):
                latest_pending_notifications = []
            latest_agent_data[constants.AGENT_SHOWN_NOTIFICATIONS_KEY] = list(latest_pending_notifications)
        rendered_agent_data = copy.deepcopy(latest_agent_data)

    if not update_agent(agent_name, update_func):
        return None
    return rendered_agent_data

class Station:
    """
    The central Station class.
    """

    def __init__(self, config_filename: str = constants.STATION_CONFIG_FILENAME):
        """
        Initializes the Station.
        Loads configuration, initializes managers (modules), and rooms.
        """
        self.is_new_station: bool = False
        self.config_path = os.path.join(constants.BASE_STATION_DATA_PATH, config_filename)
        self.config: Dict[str, Any] = self._load_or_create_config()

        # Store station_id as an attribute for easy access
        self.station_id = self.config.get(constants.STATION_ID_KEY, "unknown")

        # Store module references
        self.agent_module = agent_module
        self.capsule_module = capsule_module
        try:
            capsule_index.ensure_capsule_index(
                rebuild=capsule_index.should_rebuild_from_process_args(),
                log_status=True,
            )
        except Exception as e:
            print(f"CapsuleIndex: initialization failed: {e}")
            raise
        
        self.research_eval_manager: Optional[EvaluationManager] = None
        research_paths = None
        if constants.RESEARCH_CENTER_ENABLED:
            research_paths = ensure_runtime_layout()
            self.research_eval_manager = EvaluationManager(research_paths.evaluations_dir)

        # Initialize lineage evolution manager
        self.lineage_evolution_manager = LineageEvolutionManager(
            self.agent_module,
            eval_manager=self.research_eval_manager,
        )
        
        self.action_parser = ActionParser()
        self.orchestrator = None
        
        # Track which evaluations we've already logged to avoid spam
        self._logged_research_waits = set()

        # Instantiate room objects
        self.rooms: Dict[str, BaseRoom] = {
            constants.ROOM_LOBBY: LobbyRoom(),
            constants.ROOM_REFLECT: ReflectionChamber(),
            constants.ROOM_COMMON: CommonRoom(),
            constants.ROOM_PUBLIC_MEMORY: PublicMemoryRoom(),
            constants.ROOM_MAIL: MailRoom(),
            constants.ROOM_PRIVATE_MEMORY: PrivateMemoryRoom(), 
            constants.ROOM_ARCHIVE: ArchiveRoom(),
            constants.ROOM_MISC: MiscRoom(),
            constants.ROOM_ADMIN: AdminCounter(),
            constants.ROOM_EXIT: ExitRoom(),
        }

        if constants.TOKEN_MANAGEMENT_ROOM_ENABLED:
            self.rooms[constants.ROOM_TOKEN_MANAGEMENT] = TokenManagementRoom()
        
        # Add Research Center room
        if constants.RESEARCH_CENTER_ENABLED:
            self.rooms[constants.ROOM_RESEARCH_CENTER] = ResearchCenter(
                eval_manager=self.research_eval_manager,
                paths=research_paths,
            )

        # Add External Counter room
        if getattr(constants, "EXTERNAL_COUNTER_ENABLED", False):
            self.rooms[constants.ROOM_EXTERNAL_COUNTER] = ExternalCounter()

        # Add Theory Room
        if getattr(constants, "THEORY_ROOM_ENABLED", False):
            self.rooms[constants.ROOM_THEORY] = TheoryRoom()

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
        
        # Initialize auto research evaluator
        self.auto_research_evaluator: Optional[AutoResearchEvaluator] = None

        # Initialize auto external reporter
        self.auto_external_reporter: Optional[AutoExternalReporter] = None

        # Initialize auto theory evaluator
        self.auto_theory_evaluator: Optional[AutoTheoryEvaluator] = None
        
        # Initialize auto archive evaluator
        self.auto_archive_evaluator: Optional[AutoArchiveEvaluator] = None
        self.auto_archive_surveyor: Optional[AutoArchiveSurveyor] = None
        dialogue_logs_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.DIALOGUE_LOGS_DIR_NAME)
        file_io_utils.ensure_dir_exists(dialogue_logs_path)

        # Initialize stagnation protocol
        self.stagnation_protocol: Optional[StagnationProtocol] = None

        if self.research_eval_manager is not None:
            self.sync_top_research_submission_config(self.research_eval_manager.get_top_submission())

        print("Station initialized.")

    @staticmethod
    def _ensure_top_submission_config_defaults(config_data: Dict[str, Any]) -> None:
        """Ensure top submission config fields exist in station config."""
        config_data.setdefault(constants.STATION_CONFIG_TOP_EVALUATION_ID, None)
        config_data.setdefault(constants.STATION_CONFIG_TOP_TITLE, None)
        config_data.setdefault(constants.STATION_CONFIG_TOP_SCORE, None)
        config_data.setdefault(constants.STATION_CONFIG_TOP_TICK, None)
        config_data.setdefault(constants.STATION_CONFIG_TOP_AGENT_NAME, None)

    def _load_or_create_config(self) -> Dict[str, Any]:
        """Loads station config or creates a default one if not found."""
        config_data = file_io_utils.load_yaml(self.config_path)
        if config_data is None:
            self.is_new_station = True
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
                constants.STATION_CONFIG_LAST_SUPERVISOR_DEPARTURE_TICK: 0,
                constants.STATION_CONFIG_STAGNATION_COUNTER: 0,
                constants.STATION_CONFIG_STAGNATION_HOLIDAY_START_TICK: None,
                constants.STATION_CONFIG_STAGNATION_HOLIDAY_END_TICK: None,
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
            self._ensure_top_submission_config_defaults(default_config)
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
        if constants.STATION_CONFIG_LAST_SUPERVISOR_DEPARTURE_TICK not in config_data:
            config_data[constants.STATION_CONFIG_LAST_SUPERVISOR_DEPARTURE_TICK] = 0
        if constants.STATION_CONFIG_STAGNATION_COUNTER not in config_data:
            config_data[constants.STATION_CONFIG_STAGNATION_COUNTER] = 0
        if constants.STATION_CONFIG_STAGNATION_HOLIDAY_START_TICK not in config_data:
            config_data[constants.STATION_CONFIG_STAGNATION_HOLIDAY_START_TICK] = None
        if constants.STATION_CONFIG_STAGNATION_HOLIDAY_END_TICK not in config_data:
            config_data[constants.STATION_CONFIG_STAGNATION_HOLIDAY_END_TICK] = None
        self._ensure_top_submission_config_defaults(config_data)

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
        self._ensure_top_submission_config_defaults(self.config)

        file_io_utils.save_yaml(self.config, self.config_path)

    def sync_top_research_submission_config(self, top_submission: Optional[Dict[str, Any]]) -> None:
        """Persist the current top research submission summary into station config."""
        self.config[constants.STATION_CONFIG_TOP_EVALUATION_ID] = top_submission.get("evaluation_id") if top_submission else None
        self.config[constants.STATION_CONFIG_TOP_TITLE] = top_submission.get("title") if top_submission else None
        self.config[constants.STATION_CONFIG_TOP_SCORE] = top_submission.get("score") if top_submission else None
        self.config[constants.STATION_CONFIG_TOP_TICK] = top_submission.get("submitted_tick") if top_submission else None
        self.config[constants.STATION_CONFIG_TOP_AGENT_NAME] = top_submission.get("agent_name") if top_submission else None
        self._save_config()

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
                self.clear_stagnation_holiday_window()
                if self.stagnation_protocol:
                    self.stagnation_protocol.handle_manual_status_update(status, current_tick)
                else:
                    if status == "Healthy":
                        self.config[constants.STATION_CONFIG_STAGNATION_COUNTER] = 0
                    elif isinstance(status, str) and status.startswith("Stagnation"):
                        threshold = max(1, int(getattr(constants, 'STAGNATION_THRESHOLD_TICKS', 240)))
                        suffix = status.replace("Stagnation", "", 1).strip().upper()
                        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
                        level = 1 if not suffix else 0
                        prev_value = 0
                        for char in reversed(suffix):
                            value = roman_values.get(char)
                            if value is None:
                                level = 1
                                break
                            if value < prev_value:
                                level -= value
                            else:
                                level += value
                                prev_value = value
                        self.config[constants.STATION_CONFIG_STAGNATION_COUNTER] = max(1, level) * threshold
                    else:
                        self.config[constants.STATION_CONFIG_STAGNATION_COUNTER] = 0
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
        system_messages.send_random_prompts_to_agents(self, current_tick)

    def _send_holiday_prompts_to_agents(self, current_tick: int):
        """Send holiday prompts to all active agents on holiday ticks."""
        system_messages.send_holiday_prompts_to_agents(self, current_tick)

    def _get_current_tick(self) -> int:
        """Returns the current station tick from the config."""
        return self.config.get(constants.STATION_CONFIG_CURRENT_TICK, 0)

    def set_stagnation_holiday_window(self, start_tick: int, end_tick: int) -> None:
        """Set stagnation holiday window in config (inclusive)."""
        self.config[constants.STATION_CONFIG_STAGNATION_HOLIDAY_START_TICK] = int(start_tick)
        self.config[constants.STATION_CONFIG_STAGNATION_HOLIDAY_END_TICK] = int(end_tick)

    def clear_stagnation_holiday_window(self) -> None:
        """Clear any stagnation-triggered holiday window."""
        self.config[constants.STATION_CONFIG_STAGNATION_HOLIDAY_START_TICK] = None
        self.config[constants.STATION_CONFIG_STAGNATION_HOLIDAY_END_TICK] = None

    def get_stagnation_holiday_window(self) -> Tuple[Optional[int], Optional[int]]:
        """Return stagnation holiday window as (start_tick, end_tick)."""
        start_tick = self.config.get(constants.STATION_CONFIG_STAGNATION_HOLIDAY_START_TICK)
        end_tick = self.config.get(constants.STATION_CONFIG_STAGNATION_HOLIDAY_END_TICK)
        return start_tick, end_tick

    def is_holiday_tick(self, tick: Optional[int] = None) -> bool:
        """
        Check if a tick is a holiday.
        Includes periodic holiday mode and stagnation-triggered holiday window.
        """
        tick_to_check = self._get_current_tick() if tick is None else tick
        periodic_holiday = (
            constants.HOLIDAY_MODE_ENABLED and constants.is_holiday_tick(tick_to_check)
        )
        start_tick, end_tick = self.get_stagnation_holiday_window()
        stagnation_holiday = (
            isinstance(start_tick, int)
            and isinstance(end_tick, int)
            and start_tick <= tick_to_check <= end_tick
        )
        return periodic_holiday or stagnation_holiday
    
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
                     assigned_ancestor: str = "",
                     role_definition: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
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
                role_definition=role_definition,
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
                role_definition=role_definition,
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
        agent_data_before_end = self.agent_module.load_agent_data(
            agent_name,
            include_ascended=True,
            include_ended=True
        )
        was_supervisor = bool(
            agent_data_before_end
            and supervisor_utils.is_supervisor(agent_data_before_end, constants)
        )
        ended_in_agent_module = self.agent_module.end_agent_session(agent_name, current_tick) # agent.py:275
        
        if ended_in_agent_module:
            # --- ADD THESE LINES ---
            current_turn_order = self.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, []) # station.py:226
            if agent_name in current_turn_order:
                new_turn_order = [name for name in current_turn_order if name != agent_name]
                self.config[constants.STATION_CONFIG_AGENT_TURN_ORDER] = new_turn_order
                print(f"Station Config: Agent '{agent_name}' removed from turn order due to session end.")
            if was_supervisor:
                self.config[constants.STATION_CONFIG_LAST_SUPERVISOR_DEPARTURE_TICK] = current_tick
            self._save_config() # station.py:168
            # --- END OF ADDED LINES ---
            return True
        
        # If agent_module.end_agent_session failed (e.g. agent not found initially),
        # it would have printed an error and returned False.
        return False

    def submit_response(
        self,
        agent_name: str,
        response_text: str,
        precommitted_action_results: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[Optional[InternalActionHandler], List[str], Optional[str]]:
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
        max_actions_per_turn = getattr(constants, "MAX_ACTIONS_PER_TURN", 10)
        parsed_actions_to_execute = parsed_actions_with_info
        overflow_action_count = 0
        overflow_action_warning = None
        ascension_action_filter_warning = None
        if len(parsed_actions_with_info) > max_actions_per_turn:
            overflow_action_count = len(parsed_actions_with_info) - max_actions_per_turn
            parsed_actions_to_execute = parsed_actions_with_info[:max_actions_per_turn]
            overflow_action_warning = (
                f"You can at most issue {max_actions_per_turn} action in a turn. "
                f"Executed first {max_actions_per_turn}; ignored {overflow_action_count}."
            )

        # If an ascension action is issued this tick, only keep navigation and ascension actions.
        # This prevents cross-room side effects during ascension turns.
        has_ascension_action = any(
            pa.command in [constants.ACTION_ASCEND_INHERIT, constants.ACTION_ASCEND_NEW]
            for pa in parsed_actions_to_execute
        )
        if has_ascension_action:
            allowed_commands_for_ascension_tick = {
                constants.ACTION_GO,
                constants.ACTION_GO_TO,
                constants.ACTION_ASCEND_INHERIT,
                constants.ACTION_ASCEND_NEW,
            }
            pre_filter_count = len(parsed_actions_to_execute)
            parsed_actions_to_execute = [
                pa for pa in parsed_actions_to_execute
                if pa.command in allowed_commands_for_ascension_tick
            ]
            dropped_action_count = pre_filter_count - len(parsed_actions_to_execute)
            if dropped_action_count > 0:
                ascension_action_filter_warning = (
                    f"Ascension turn guardrail: ignored {dropped_action_count} non-navigation/non-ascension action(s). "
                    f"Only `goto` and `ascend_*` actions are processed when ascension is present."
                )

        # --- MODIFICATION: Generate summary of parsed actions ---
        parsed_action_summary_lines = []
        parsed_action_raw_data = []
        
        if not parsed_actions_with_info and response_text.strip():
            # Handles case where agent sent text but no valid /execute_action{} commands
            parsed_action_summary_lines.append("No action received in the last turn.")
            # No raw actions to store in this case
        elif parsed_actions_to_execute:
            # Simulate location changes for summary reporting accuracy
            # Start with the agent's location at the beginning of this action sequence
            simulated_location_for_summary = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]

            for idx, pa_info in enumerate(parsed_actions_to_execute):
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
            if overflow_action_warning:
                parsed_action_summary_lines.append(overflow_action_warning)
            if ascension_action_filter_warning:
                parsed_action_summary_lines.append(ascension_action_filter_warning)
        # --- End of summary generation ---
        
        actual_handler_from_room: Optional[InternalActionHandler] = None
        all_actions_executed_strings_for_return: List[str] = [] 
        
        actions_by_room_log: Dict[str, List[str]] = {}
        ordered_rooms_with_actions: List[str] = [] 
        turn_history_log_for_agent_file: List[Dict[str, Any]] = [] 
        initial_room_for_turn: str = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]

        if not parsed_actions_to_execute:
            action_str = f"You remained in the {initial_room_for_turn}."
            actions_by_room_log.setdefault(initial_room_for_turn, []).append(action_str)
            if initial_room_for_turn not in ordered_rooms_with_actions:
                ordered_rooms_with_actions.append(initial_room_for_turn)
            all_actions_executed_strings_for_return.append(action_str)


        for action_index, parsed_action in enumerate(parsed_actions_to_execute, start=1):
            action_command = parsed_action.command
            action_args = parsed_action.args
            yaml_data = parsed_action.yaml_data
            yaml_error_msg = parsed_action.yaml_error

            current_single_action_results: List[str] = [] 
            internal_handler_from_room: Optional[InternalActionHandler] = None
            
            # Read current location fresh for this action (always define this variable)
            room_context_for_this_action = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]
            action_op_id = f"tick:{current_station_tick}:agent:{str(agent_name).replace(':', '_')}:action:{action_index}"
            precommitted_entry = (precommitted_action_results or {}).get(action_op_id)

            if (
                precommitted_entry is not None
                and action_command == constants.ACTION_RESEARCH_SUBMIT
                and current_turn_agent_data.get(constants.AGENT_CURRENT_LOCATION_KEY) == constants.ROOM_RESEARCH_CENTER
            ):
                precommitted_messages = list(precommitted_entry.get("messages") or [])
                current_single_action_results.extend(precommitted_messages)

                if precommitted_entry.get("accepted"):
                    notification_message = precommitted_entry.get("notification")
                    if notification_message:
                        self.agent_module.add_pending_notification(current_turn_agent_data, str(notification_message))

                    eval_id = precommitted_entry.get("eval_id")
                    if eval_id and self.research_eval_manager:
                        try:
                            self.research_eval_manager.mark_parallel_submission_committed(str(eval_id))
                        except Exception as exc:
                            current_single_action_results.append(
                                f"Warning: submitted evaluation {eval_id} was accepted, but commit marking failed: {exc}"
                            )

                    try:
                        cooldown_ticks = ResearchCenter._get_submission_cooldown_ticks(current_turn_agent_data, constants)
                    except Exception:
                        cooldown_ticks = 0
                    if cooldown_ticks > 0:
                        current_turn_agent_data[constants.AGENT_LAST_RESEARCH_SUBMISSION_TICK_KEY] = current_station_tick

                if current_single_action_results:
                    effective_room_for_log = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]
                    if effective_room_for_log not in ordered_rooms_with_actions:
                        ordered_rooms_with_actions.append(effective_room_for_log)
                    actions_by_room_log.setdefault(effective_room_for_log, []).extend(current_single_action_results)

                if yaml_error_msg:
                    error_log_string = f"Note on action '{action_command}': {yaml_error_msg}"
                    current_single_action_results.append(error_log_string)
                    effective_room_for_log = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]
                    if effective_room_for_log not in ordered_rooms_with_actions:
                        ordered_rooms_with_actions.append(effective_room_for_log)
                    actions_by_room_log.setdefault(effective_room_for_log, []).append(error_log_string)

                all_actions_executed_strings_for_return.extend(current_single_action_results)
                continue

            if (
                precommitted_entry is not None
                and action_command == getattr(constants, "ACTION_ARCHIVE_SURVEY", "survey")
                and current_turn_agent_data.get(constants.AGENT_CURRENT_LOCATION_KEY) == constants.ROOM_ARCHIVE
            ):
                precommitted_messages = list(precommitted_entry.get("messages") or [])
                current_single_action_results.extend(precommitted_messages)

                if precommitted_entry.get("accepted"):
                    notification_message = precommitted_entry.get("notification")
                    if notification_message:
                        self.agent_module.add_pending_notification(current_turn_agent_data, str(notification_message))

                    survey_id = precommitted_entry.get("survey_id")
                    if survey_id:
                        try:
                            from station.eval_archive.surveyor import mark_archive_survey_committed

                            mark_archive_survey_committed(str(survey_id))
                            surveyor = getattr(self, "auto_archive_surveyor", None)
                            if surveyor and hasattr(surveyor, "wake"):
                                surveyor.wake(f"committed parallel archive survey {survey_id}")
                        except Exception as exc:
                            current_single_action_results.append(
                                f"Warning: archive survey {survey_id} was accepted, but commit marking failed: {exc}"
                            )

                if current_single_action_results:
                    effective_room_for_log = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]
                    if effective_room_for_log not in ordered_rooms_with_actions:
                        ordered_rooms_with_actions.append(effective_room_for_log)
                    actions_by_room_log.setdefault(effective_room_for_log, []).extend(current_single_action_results)

                if yaml_error_msg:
                    error_log_string = f"Note on action '{action_command}': {yaml_error_msg}"
                    current_single_action_results.append(error_log_string)
                    effective_room_for_log = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]
                    if effective_room_for_log not in ordered_rooms_with_actions:
                        ordered_rooms_with_actions.append(effective_room_for_log)
                    actions_by_room_log.setdefault(effective_room_for_log, []).append(error_log_string)

                all_actions_executed_strings_for_return.extend(current_single_action_results)
                continue

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

            elif action_command == constants.ACTION_REFLECT_REFLECT:
                # Allow reflection from any room by delegating to the Reflection Chamber handler.
                reflection_room = self.rooms[constants.ROOM_REFLECT]
                executed_in_room, internal_handler_from_room = reflection_room.handle_action(
                    current_turn_agent_data, action_command, action_args, yaml_data, self.room_context, current_station_tick
                )
                current_single_action_results.extend(executed_in_room)
                if executed_in_room:
                    if room_context_for_this_action not in ordered_rooms_with_actions:
                        ordered_rooms_with_actions.append(room_context_for_this_action)
                    actions_by_room_log.setdefault(room_context_for_this_action, []).extend(executed_in_room)
                if internal_handler_from_room and actual_handler_from_room is None:
                    actual_handler_from_room = internal_handler_from_room

            elif action_command in [constants.ACTION_ASCEND_INHERIT, constants.ACTION_ASCEND_NEW]:
                # Allow ascension commands from any room by delegating to Lobby.
                lobby_room = self.rooms[constants.ROOM_LOBBY]
                executed_in_room, internal_handler_from_room = lobby_room.handle_action(
                    current_turn_agent_data, action_command, action_args, yaml_data, self.room_context, current_station_tick
                )
                current_single_action_results.extend(executed_in_room)
                if executed_in_room:
                    if room_context_for_this_action not in ordered_rooms_with_actions:
                        ordered_rooms_with_actions.append(room_context_for_this_action)
                    actions_by_room_log.setdefault(room_context_for_this_action, []).extend(executed_in_room)
                if internal_handler_from_room and actual_handler_from_room is None:
                    actual_handler_from_room = internal_handler_from_room

            elif action_command in [constants.ACTION_GO, constants.ACTION_GO_TO] or (action_command == constants.ACTION_EXIT_TERMINATE and room_context_for_this_action != constants.ROOM_EXIT):
                # Handle universal exit action by converting to goto exit
                if action_command == constants.ACTION_EXIT_TERMINATE:
                    target_room_short_name = constants.SHORT_ROOM_NAME_EXIT
                else:
                    target_room_short_name = action_args
                target_room_full_name = constants.SHORT_ROOM_NAME_TO_FULL_MAP.get(target_room_short_name)
                previous_location_full_name = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]

                if target_room_full_name and target_room_full_name in self.rooms:
                    def finalize_navigation(destination: str, message: str) -> str:
                        self.agent_module.update_agent_current_location(current_turn_agent_data, destination)
                        if previous_location_full_name == constants.ROOM_COMMON and destination != constants.ROOM_COMMON:
                            common_room_instance = self.rooms.get(constants.ROOM_COMMON)
                            if common_room_instance and hasattr(common_room_instance, 'agent_left_room'):
                                cast(CommonRoom, common_room_instance).agent_left_room(agent_name, self.room_context)
                        if destination not in ordered_rooms_with_actions:
                            ordered_rooms_with_actions.append(destination)
                        actions_by_room_log.setdefault(destination, []).append(message)
                        return message

                    # Check maturity restrictions for certain rooms
                    if (target_room_full_name in [constants.ROOM_ARCHIVE, constants.ROOM_PUBLIC_MEMORY, constants.ROOM_MAIL, constants.ROOM_COMMON, constants.ROOM_EXTERNAL_COUNTER] and
                          not self._is_agent_mature(current_turn_agent_data, self._get_current_tick())):
                        action_str = f"Access denied. {target_room_full_name} is restricted to mature agents ({constants.AGENT_ISOLATION_TICKS}+ ticks old). As an immature agent, you are expected to continuously research independently by proposing ideas and submitting experiments. Please do not wait idly to become mature to access the room."
                        current_single_action_results.append(action_str)
                        # Log failure in current room
                        if room_context_for_this_action not in ordered_rooms_with_actions:
                            ordered_rooms_with_actions.append(room_context_for_this_action)
                        actions_by_room_log.setdefault(room_context_for_this_action, []).append(action_str)
                    # Check holiday restriction for Research Center
                    elif (target_room_full_name == constants.ROOM_RESEARCH_CENTER and
                          self.is_holiday_tick(self._get_current_tick())):
                        action_str = "Access denied. The Research Center is closed during holidays. Please try again on working days."
                        current_single_action_results.append(action_str)
                        # Log failure in current room
                        if room_context_for_this_action not in ordered_rooms_with_actions:
                            ordered_rooms_with_actions.append(room_context_for_this_action)
                        actions_by_room_log.setdefault(room_context_for_this_action, []).append(action_str)
                    elif target_room_full_name == constants.ROOM_TOKEN_MANAGEMENT:
                        # Restrict access until usage reaches pre-warning threshold
                        is_guest_for_access = current_turn_agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_GUEST
                        effective_max_budget_tm = current_turn_agent_data.get(constants.AGENT_TOKEN_BUDGET_MAX_KEY)
                        if is_guest_for_access and effective_max_budget_tm is not None:
                            effective_max_budget_tm = min(effective_max_budget_tm, constants.GUEST_MAX_TOKENS_CEILING)
                        elif is_guest_for_access and effective_max_budget_tm is None:
                            effective_max_budget_tm = constants.GUEST_MAX_TOKENS_CEILING

                        current_tokens_used_tm = current_turn_agent_data.get(constants.AGENT_TOKEN_BUDGET_CURRENT_KEY)
                        token_count_stale_tm = current_turn_agent_data.get(constants.AGENT_TOKEN_BUDGET_CURRENT_STALE_KEY, False)
                        pre_warning_ratio_tm = constants.GUEST_PRE_WARNING_RATIO if is_guest_for_access else constants.RECURSIVE_PRE_WARNING_RATIO
                        threshold_percent_tm = int(pre_warning_ratio_tm * 100)

                        if (
                            effective_max_budget_tm is None
                            or effective_max_budget_tm <= 0
                            or token_count_stale_tm
                            or not isinstance(current_tokens_used_tm, (int, float))
                        ):
                            action_str = "Access denied. Token budget data is unavailable; try again after your next turn."
                            current_single_action_results.append(action_str)
                            if room_context_for_this_action not in ordered_rooms_with_actions:
                                ordered_rooms_with_actions.append(room_context_for_this_action)
                            actions_by_room_log.setdefault(room_context_for_this_action, []).append(action_str)
                        elif current_tokens_used_tm < pre_warning_ratio_tm * effective_max_budget_tm:
                            action_str = f"Token Management Room not needed yet. Your current token usage is below {threshold_percent_tm}%, indicating a healthy status. Please proceed with normal activities."
                            current_single_action_results.append(action_str)
                            if room_context_for_this_action not in ordered_rooms_with_actions:
                                ordered_rooms_with_actions.append(room_context_for_this_action)
                            actions_by_room_log.setdefault(room_context_for_this_action, []).append(action_str)
                        else:
                            # Allow navigation once threshold reached
                            action_str = finalize_navigation(target_room_full_name, f"You went to the {target_room_full_name}.")
                            current_single_action_results.append(action_str)
                    else:
                        # Allow navigation
                        action_str = finalize_navigation(target_room_full_name, f"You went to the {target_room_full_name}.")
                        current_single_action_results.append(action_str)
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
                    help_source = f"room:{target_room_short_name_for_help}"
                    help_protection_reason = None
                    help_already_shown = self.agent_module.get_agent_room_state(
                        current_turn_agent_data,
                        target_room_short_name_for_help,
                        constants.AGENT_ROOM_STATE_FIRST_VISIT_HELP_SHOWN_KEY,
                        default=False,
                    )
                    if not help_already_shown:
                        help_protection_reason = constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP
                        self.agent_module.set_agent_room_state(
                            current_turn_agent_data,
                            target_room_short_name_for_help,
                            constants.AGENT_ROOM_STATE_FIRST_VISIT_HELP_SHOWN_KEY,
                            True,
                        )
                    self.agent_module.add_pending_notification(
                        current_turn_agent_data,
                        notification_message,
                        protection_reason=help_protection_reason,
                        protection_source=help_source,
                    )
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

        if overflow_action_warning:
            all_actions_executed_strings_for_return.append(overflow_action_warning)
            effective_room_for_log = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]
            if effective_room_for_log not in ordered_rooms_with_actions:
                ordered_rooms_with_actions.append(effective_room_for_log)
            actions_by_room_log.setdefault(effective_room_for_log, []).append(overflow_action_warning)
        if ascension_action_filter_warning:
            all_actions_executed_strings_for_return.append(ascension_action_filter_warning)
            effective_room_for_log = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]
            if effective_room_for_log not in ordered_rooms_with_actions:
                ordered_rooms_with_actions.append(effective_room_for_log)
            actions_by_room_log.setdefault(effective_room_for_log, []).append(ascension_action_filter_warning)

        if actual_handler_from_room:
            protection = actual_handler_from_room.get_dialogue_tick_protection()
            if isinstance(protection, dict):
                self.agent_module.protect_dialogue_tick(
                    current_turn_agent_data,
                    current_station_tick,
                    str(protection.get("reason", "")),
                    source=str(protection.get("source", "")),
                )

        # ... (existing logic to construct turn_history_log_for_agent_file and save agent data) ...
        for room_name_hist in ordered_rooms_with_actions:
            if room_name_hist in actions_by_room_log and actions_by_room_log[room_name_hist]:
                turn_history_log_for_agent_file.append({
                    "location": room_name_hist,
                    "actions_executed": actions_by_room_log[room_name_hist]
                })
        
        error_msg_for_return: Optional[str] = None

        def finalize_agent_data(final_agent_data_to_save: Dict[str, Any]) -> None:
            latest_mail_state = {
                constants.SHORT_ROOM_NAME_MAIL: {
                    constants.AGENT_ROOM_STATE_READ_STATUS_KEY: _read_status_map(
                        final_agent_data_to_save,
                        constants.SHORT_ROOM_NAME_MAIL,
                    )
                }
            }
            final_agent_data_to_save[constants.AGENT_CURRENT_LOCATION_KEY] = current_turn_agent_data[constants.AGENT_CURRENT_LOCATION_KEY]
            final_agent_data_to_save[constants.AGENT_NOTIFICATIONS_PENDING_KEY] = _merge_pending_notifications_after_turn(
                current_turn_agent_data.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, []),
                final_agent_data_to_save.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, []),
                shown_notifications,
            )
            final_agent_data_to_save[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY] = (
                _merge_pending_dialogue_tick_protections_after_turn(
                    current_turn_agent_data.get(constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY, []),
                    final_agent_data_to_save.get(constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY, []),
                    agent_data_at_turn_start.get(constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY, []),
                )
            )
            final_agent_data_to_save[constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY] = (
                _merge_protected_dialogue_ticks_after_turn(
                    current_turn_agent_data.get(constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY, []),
                    final_agent_data_to_save.get(constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY, []),
                    agent_data_at_turn_start.get(constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY, []),
                )
            )

            # Define keys that should not be copied from current_turn_agent_data
            protected_keys = {
                constants.AGENT_NAME_KEY, constants.AGENT_STATUS_KEY,
                constants.AGENT_IS_ASCENDED_KEY, constants.AGENT_ASCENDED_TO_NAME_KEY,
                constants.AGENT_SESSION_ENDED_KEY, constants.AGENT_ROOM_OUTPUT_HISTORY_KEY,
                constants.AGENT_CURRENT_LOCATION_KEY, constants.AGENT_NOTIFICATIONS_PENDING_KEY,
                constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY,
                constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY,
            }

            # Copy all non-protected keys from current_turn_agent_data to final_agent_data_to_save
            for key, value in current_turn_agent_data.items():
                if key not in protected_keys:
                    # This includes AGENT_STATE_DATA_KEY and room-specific state blocks
                    final_agent_data_to_save[key] = value

            _merge_concurrent_room_read_status(
                final_agent_data_to_save,
                constants.SHORT_ROOM_NAME_MAIL,
                latest_mail_state,
                agent_data_at_turn_start,
            )

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
            if self.is_holiday_tick(current_station_tick):
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

            updated_agent_data = self._update_meta_reflection_tick_count(final_agent_data_to_save, current_station_tick)
            if updated_agent_data is not final_agent_data_to_save:
                final_agent_data_to_save.clear()
                final_agent_data_to_save.update(updated_agent_data)

        if not self.agent_module.update_agent_with_function(agent_name, finalize_agent_data):
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
        turn_start_agent_data = copy.deepcopy(agent_data)
        current_turn_agent_data = copy.deepcopy(agent_data)
        markdown_lines: List[str] = []

        # --- Ascension Eligibility Check & Prompt ---
        is_guest = current_turn_agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_GUEST

        if is_guest and \
           not current_turn_agent_data.get(constants.AGENT_IS_ASCENDED_KEY) and \
           not current_turn_agent_data.get(constants.AGENT_SESSION_ENDED_KEY):
            
            lobby_room = cast(LobbyRoom, self.rooms[constants.ROOM_LOBBY])
            lobby_room.ensure_guest_ascension_state(current_turn_agent_data, self.room_context)
                
        stored_max_budget_for_warning = current_turn_agent_data.get(constants.AGENT_TOKEN_BUDGET_MAX_KEY) # station.py:659
        effective_max_budget_for_warning = stored_max_budget_for_warning # station.py:659
        is_guest_for_warning_check = current_turn_agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_GUEST # station.py:661
        if is_guest_for_warning_check and stored_max_budget_for_warning is not None: # station.py:661
            effective_max_budget_for_warning = min(stored_max_budget_for_warning, constants.GUEST_MAX_TOKENS_CEILING) # station.py:662
        elif is_guest_for_warning_check and stored_max_budget_for_warning is None: # station.py:663
             effective_max_budget_for_warning = constants.GUEST_MAX_TOKENS_CEILING # station.py:664
        
        current_turn_agent_data = self._check_and_apply_token_warnings(current_turn_agent_data, effective_max_budget_for_warning)
        
        # Check for inactivity warnings
        current_turn_agent_data = self._check_and_apply_inactivity_warning(current_turn_agent_data, current_station_tick)

        # Check for compulsory meta reflection warnings
        current_turn_agent_data = self._check_and_apply_meta_reflection_warning(current_turn_agent_data, current_station_tick)
        
        # Check for life limit warnings
        current_turn_agent_data = self._check_and_apply_life_warnings(current_turn_agent_data, current_station_tick)

        # Check for supervisor report reminders
        current_turn_agent_data = self._check_and_apply_supervisor_report_reminder(current_turn_agent_data, current_station_tick)

        # Pending notification protections are queued when the message is created
        # and applied to the tick where that message is actually rendered.
        _drop_stale_pending_dialogue_tick_protections(current_turn_agent_data)
        self.agent_module.apply_pending_dialogue_tick_protections(current_turn_agent_data, current_station_tick)
        if current_turn_agent_data.get(constants.AGENT_PENDING_CURRENT_RESEARCH_TASK_READ_KEY):
            current_turn_agent_data[constants.AGENT_HAS_READ_CURRENT_RESEARCH_TASK_KEY] = True
            current_turn_agent_data[constants.AGENT_PENDING_CURRENT_RESEARCH_TASK_READ_KEY] = False

        # Atomically persist the warning/reminder state and re-read the latest
        # pending notifications before we mark them as shown in this turn.
        saved_turn_data = _save_request_status_snapshot_atomically(
            self.agent_module,
            agent_name,
            turn_start_agent_data,
            current_turn_agent_data,
            refresh_shown_notifications=True,
            apply_pending_protections_tick=current_station_tick,
        )
        if saved_turn_data is None:
            print(f"Warning: Agent '{agent_name}' data could not be reloaded while saving status snapshot.")
        else:
            current_turn_agent_data = saved_turn_data

        # --- System Information ---
        markdown_lines.append("## System Information")
        # ... (rest of system info as before)
        # Check if it's a holiday tick and holiday mode is enabled
        tick_display = f"- Station Tick: {current_station_tick}"
        if self.is_holiday_tick(current_station_tick):
            tick_display += " (Holiday)"
        markdown_lines.append(tick_display)
        markdown_lines.append(f"- Station Status: {self.config.get(constants.STATION_CONFIG_STATION_STATUS, 'Unknown')}")
        markdown_lines.append(f"- Agent Name: {current_turn_agent_data.get(constants.AGENT_NAME_KEY)}")
        markdown_lines.append(f"- Agent Description: {current_turn_agent_data.get(constants.AGENT_DESCRIPTION_KEY)}")
        agent_status = current_turn_agent_data.get(constants.AGENT_STATUS_KEY)
        markdown_lines.append(f"- Agent Status: {agent_status}")
        
        # Add meta prompt if it exists
        meta_prompt = self.agent_module.get_agent_meta_prompt(current_turn_agent_data)
        if meta_prompt:
            markdown_lines.append(f"- Agent Meta Prompt:\n```\n{meta_prompt}\n```")

        tokens_used_so_far = current_turn_agent_data.get(constants.AGENT_TOKEN_BUDGET_CURRENT_KEY)
        stored_max_budget_tokens = current_turn_agent_data.get(constants.AGENT_TOKEN_BUDGET_MAX_KEY)
        token_budget_count_is_trusted = (
            isinstance(tokens_used_so_far, (int, float))
            and not current_turn_agent_data.get(constants.AGENT_TOKEN_BUDGET_CURRENT_STALE_KEY, False)
        )
        
        # MODIFICATION: Display logic considering guest ceiling
        display_max_budget = stored_max_budget_tokens
        is_guest_for_display = agent_status == constants.AGENT_STATUS_GUEST

        if is_guest_for_display and stored_max_budget_tokens is not None:
            display_max_budget = min(stored_max_budget_tokens, constants.GUEST_MAX_TOKENS_CEILING)
        elif is_guest_for_display and stored_max_budget_tokens is None: # Should ideally not happen if create_guest_agent sets a default
             display_max_budget = constants.GUEST_MAX_TOKENS_CEILING


        pre_warning_ratio_display = constants.GUEST_PRE_WARNING_RATIO if is_guest_for_display else constants.RECURSIVE_PRE_WARNING_RATIO

        if display_max_budget is not None and token_budget_count_is_trusted and display_max_budget > 0:
            percentage_used = (tokens_used_so_far / display_max_budget) * 100
            if percentage_used >= pre_warning_ratio_display * 100:
                budget_line = f"- Agent Token Budget: {tokens_used_so_far:,} / {display_max_budget:,} used ({percentage_used:.0f}%)"
                if is_guest_for_display and stored_max_budget_tokens is not None and stored_max_budget_tokens > constants.GUEST_MAX_TOKENS_CEILING:
                    budget_line += f" (Guest ceiling of {constants.GUEST_MAX_TOKENS_CEILING:,} applied)"
                if is_guest_for_display:
                    budget_line += " - Critical: Ascend as soon as possible (`/execute_action{ascend_inherit}` or `/execute_action{ascend_new}`) to unlock pruning."
                else:
                    budget_line += f" - Critical: Please proceed to the Token Management Room ASAP by `/execute_action{{goto {constants.SHORT_ROOM_NAME_TOKEN_MANAGEMENT}}}`!"
                markdown_lines.append(budget_line)
        
        # Display Agent Age
        birth_tick = current_turn_agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
        if birth_tick is not None:
            agent_age = current_station_tick - birth_tick
            # Get agent-specific max_age, fallback to AGENT_MAX_LIFE for backward compatibility
            agent_max_age = current_turn_agent_data.get(constants.AGENT_MAX_AGE_KEY, constants.AGENT_MAX_LIFE)
            if agent_max_age is None:
                age_line = f"- Agent Age: {agent_age} ticks"
            else:
                age_line = f"- Agent Age: {agent_age} ticks / {agent_max_age} ticks"
            
            age_status = self._get_agent_age_status(current_turn_agent_data, current_station_tick)
            if age_status:
                age_line += f" ({age_status})"
            
            markdown_lines.append(age_line)
        
        # Display Agent Role if set
        agent_role = current_turn_agent_data.get(constants.AGENT_ROLE_KEY)
        if agent_role:
            # Capitalize the role name for display
            role_display = agent_role.capitalize()
            markdown_lines.append(f"- Agent Role: {role_display}")
            markdown_lines.extend(
                supervisor_utils.build_supervisor_dashboard_lines(
                    self,
                    current_turn_agent_data,
                    current_station_tick,
                    constants
                )
            )

        system_messages_strings = self.agent_module.get_pending_notifications(current_turn_agent_data)
        system_tips_text: Optional[str] = None
        holiday_prompt_text: Optional[str] = None
        tip_tick = current_turn_agent_data.get(constants.AGENT_SYSTEM_TIP_TICK_KEY)
        tip_text = current_turn_agent_data.get(constants.AGENT_SYSTEM_TIP_TEXT_KEY)
        if tip_tick == current_station_tick and isinstance(tip_text, str) and tip_text.strip():
            system_tips_text = " ".join(tip_text.strip().split())
        holiday_prompt_tick = current_turn_agent_data.get(constants.AGENT_HOLIDAY_PROMPT_TICK_KEY)
        holiday_prompt_value = current_turn_agent_data.get(constants.AGENT_HOLIDAY_PROMPT_TEXT_KEY)
        if holiday_prompt_tick == current_station_tick and isinstance(holiday_prompt_value, str) and holiday_prompt_value.strip():
            holiday_prompt_text = " ".join(holiday_prompt_value.strip().split())

        # Construct and add ascension prompt if eligible (owned by Lobby)
        lobby_room_for_prompt = cast(LobbyRoom, self.rooms[constants.ROOM_LOBBY])
        ascension_prompt = lobby_room_for_prompt.build_ascension_system_message(current_turn_agent_data, self.room_context)
        if ascension_prompt and ascension_prompt not in system_messages_strings:
            system_messages_strings.insert(0, ascension_prompt)

        markdown_lines.append("\n---\n")                                         
        markdown_lines.append("## System Messages")
        if system_messages_strings:
            rendered_system_messages_lines: List[str] = []
            for n, msg_str in enumerate(system_messages_strings):
                rendered_system_messages_lines.append(f"### Message {n + 1}\n")
                rendered_system_messages_lines.append(msg_str)
            rendered_system_messages = "\n".join(rendered_system_messages_lines)
            rendered_system_messages = system_messages.truncate_rendered_system_messages(
                rendered_system_messages,
                model_name=current_turn_agent_data.get(constants.AGENT_MODEL_NAME_KEY),
            )
            markdown_lines.append(rendered_system_messages)
        else:
            markdown_lines.append("None")  
        markdown_lines.append("")   

        last_actions_summary = current_turn_agent_data.get(constants.AGENT_LAST_PARSED_ACTIONS_SUMMARY_KEY)        
        if last_actions_summary and isinstance(last_actions_summary, list) and len(last_actions_summary) > 0:            
            markdown_lines.append("\n---\n")
            summary_block = ["## Actions Detected (Last Turn)"]
            summary_block.extend(last_actions_summary)
            markdown_lines.extend(summary_block)            
        
        markdown_lines.append("") 
        # ... (Rest of Room Output History and Current State as before) ...
        markdown_lines.append("\n---\n") 
        markdown_lines.append("## Room Output History") 
        markdown_lines.append("")
        current_location_name = current_turn_agent_data.get(constants.AGENT_CURRENT_LOCATION_KEY)
        history_log_from_agent_file = current_turn_agent_data.get(constants.AGENT_ROOM_OUTPUT_HISTORY_KEY, [])
        unique_visited_room_names_ordered: List[str] = []
        for segment in history_log_from_agent_file:
            if isinstance(segment, dict) and "location" in segment:
                if segment["location"] not in unique_visited_room_names_ordered:
                    unique_visited_room_names_ordered.append(segment["location"])
            else: print(f"Warning: Malformed history segment for agent {agent_name}: {segment}")
        room_names_for_output: List[str] = unique_visited_room_names_ordered.copy()
        if current_location_name and current_location_name not in room_names_for_output:
            room_names_for_output.append(current_location_name)
        if not room_names_for_output:
            markdown_lines.append("No room history from last turn."); markdown_lines.append("")
        for room_name_from_history in room_names_for_output:
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
                room_output_started_at = time.perf_counter()
                historical_room_output_str = room_instance.get_room_output(current_turn_agent_data, self.room_context, current_station_tick)
                room_output_elapsed_seconds = time.perf_counter() - room_output_started_at
                if room_output_elapsed_seconds >= _REQUEST_STATUS_TIMING_LOG_THRESHOLD_SECONDS:
                    print(
                        "StationTiming: request_status room_render "
                        f"agent={agent_name!r} tick={current_station_tick} "
                        f"room={room_name_from_history!r} duration={room_output_elapsed_seconds:.3f}s"
                    )

                saved_room_turn_data = _save_request_status_snapshot_atomically(
                    self.agent_module,
                    agent_name,
                    turn_start_agent_data,
                    current_turn_agent_data,
                )
                if not saved_room_turn_data:
                    print(f"Warning: Agent '{agent_name}' data could not be reloaded while saving room output snapshot.")
                markdown_lines.append(historical_room_output_str if historical_room_output_str and historical_room_output_str.strip() else "(This room provided no output or is currently empty.)")
            else: markdown_lines.append("Error: Room data unavailable for history.")
            markdown_lines.append("") 
        markdown_lines.append("\n---\n") 
        markdown_lines.append("## Current State")
        markdown_lines.append("")
        markdown_lines.append(f"**Current Location:** {current_location_name}")
        if system_tips_text:
            markdown_lines.append(f"**System Tips:** {system_tips_text}")
        gemini_response_reminder = _gemini_response_reminder_for_agent(current_turn_agent_data)
        if gemini_response_reminder:
            markdown_lines.append(gemini_response_reminder)
        if holiday_prompt_text:
            markdown_lines.append(f"**Holiday Prompt:** {holiday_prompt_text}")
        markdown_lines.append("")

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

        # Send scheduled prompts to agents if frequency/holiday conditions are met
        self._send_random_prompts_to_agents(new_tick)
        self._send_holiday_prompts_to_agents(new_tick)
        supervisor_utils.maybe_assign_supervisor_at_tick_end(self, current_tick)

        self._save_config()
        print(f"Station Tick advanced to: {new_tick} {self._format_tick_log_suffix()}")
        return new_tick

    @staticmethod
    def _format_tick_log_suffix() -> str:
        """Append a stable UTC+8 timestamp to tick logs for runtime correlation."""
        utc_plus_8 = timezone(timedelta(hours=8))
        timestamp = datetime.now(utc_plus_8).strftime("%Y-%m-%d %H:%M:%S")
        return f"| UTC+8 {timestamp}"

    def get_all_agents_summary(self) -> List[Dict[str, Any]]:
        """
        Retrieves a summary (name, status, and model info) of all agents.
        """
        return agent_summary.get_all_agents_summary(
            base_path=self.room_context.constants_module.BASE_STATION_DATA_PATH
        )

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
        Deprecated: retained for compatibility.
        """
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
            role_definition=None,
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

    def has_pending_research_evaluations(self) -> bool:
        """Check if there are any pending (queued) or running research evaluations"""
        if not self.auto_research_evaluator:
            return False
        return self.auto_research_evaluator.has_pending_or_running()

    def has_pending_external_reports(self) -> bool:
        """Check if there are any pending (queued) or running external reports"""
        if not self.auto_external_reporter:
            return False
        return self.auto_external_reporter.has_pending_or_running()

    def has_pending_theory_evaluations(self) -> bool:
        """Check if there are any pending (queued) or running theory evaluations"""
        if not self.auto_theory_evaluator:
            return False
        return self.auto_theory_evaluator.has_pending_or_running()

    def has_pending_archive_surveys(self) -> bool:
        """Check if there are any pending (queued) or running archive surveys."""
        if not self.auto_archive_surveyor:
            return False
        return self.auto_archive_surveyor.has_pending_or_running()
    
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

    def should_wait_for_external_reports_at_tick_boundary(self) -> bool:
        """Return True if an external report has reached the tick limit."""
        if not self.auto_external_reporter:
            return False
        try:
            current_tick = self._get_current_tick()
            return self.auto_external_reporter.should_wait_at_tick(current_tick)
        except Exception as e:
            print(f"Error checking external report tick boundaries: {e}")
            return False

    def should_wait_for_theory_evaluations_at_tick_boundary(self) -> bool:
        """Return True if a theory evaluation has reached the tick limit."""
        if not self.auto_theory_evaluator:
            return False
        try:
            current_tick = self._get_current_tick()
            return self.auto_theory_evaluator.should_wait_at_tick(current_tick)
        except Exception as e:
            print(f"Error checking theory evaluation tick boundaries: {e}")
            return False

    def should_wait_for_archive_surveys_at_tick_boundary(self) -> bool:
        """Return True if an archive survey has reached the tick limit."""
        if not self.auto_archive_surveyor:
            return False
        try:
            current_tick = self._get_current_tick()
            return self.auto_archive_surveyor.should_wait_at_tick(current_tick)
        except Exception as e:
            print(f"Error checking archive survey tick boundaries: {e}")
            return False
    
    def has_pending_coder_sessions(self) -> bool:
        try:
            if self.auto_research_evaluator:
                return self.auto_research_evaluator.has_active_coder_sessions()
        except Exception as e:
            print(f"Error checking Research Center coder sessions: {e}")
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

    def start_auto_research_evaluator(self, log_queue=None) -> bool:
        """Start the auto research evaluator if enabled"""
        started_at = time.perf_counter()
        if not constants.AUTO_EVAL_RESEARCH:
            print("Station: Auto research evaluation is disabled")
            return False
            
        if self.auto_research_evaluator and self.auto_research_evaluator.is_running:
            print("Station: Auto research evaluator is already running")
            return True
            
        try:
            # Only create new instance if we don't have a running one
            if not self.auto_research_evaluator:
                phase_started_at = time.perf_counter()
                self.auto_research_evaluator = AutoResearchEvaluator(
                    station_instance=self,
                    enabled=constants.AUTO_EVAL_RESEARCH,
                    log_queue=log_queue,
                    eval_manager=self.research_eval_manager,
                )
                print(
                    "Station: AutoResearchEvaluator construction "
                    f"took {time.perf_counter() - phase_started_at:.3f}s"
                )
            
            phase_started_at = time.perf_counter()
            if self.auto_research_evaluator.start_evaluation_loop():
                print(
                    "Station: auto_research_evaluator.start_evaluation_loop "
                    f"took {time.perf_counter() - phase_started_at:.3f}s"
                )
                print(
                    "Station: Auto research evaluator started successfully "
                    f"in {time.perf_counter() - started_at:.3f}s"
                )
                return True
            else:
                print(
                    "Station: auto_research_evaluator.start_evaluation_loop "
                    f"took {time.perf_counter() - phase_started_at:.3f}s before failure"
                )
                print(f"Station: Failed to start auto research evaluator after {time.perf_counter() - started_at:.3f}s")
                self.auto_research_evaluator = None
                return False
                
        except Exception as e:
            print(f"Station: Error starting auto research evaluator after {time.perf_counter() - started_at:.3f}s: {e}")
            self.auto_research_evaluator = None
            return False
    
    def stop_auto_research_evaluator(self):
        """Stop the auto research evaluator"""
        if self.auto_research_evaluator:
            started_at = time.perf_counter()
            self.auto_research_evaluator.stop_evaluation_loop()
            self.auto_research_evaluator = None
            print(f"Station: Auto research evaluator stopped in {time.perf_counter() - started_at:.3f}s")

    def start_auto_external_reporter(self, log_queue=None) -> bool:
        """Start the auto external reporter if enabled"""
        if not (getattr(constants, "AUTO_EVAL_EXTERNAL_REPORT", False) and getattr(constants, "EXTERNAL_COUNTER_ENABLED", False)):
            print("Station: Auto external reporting is disabled")
            return False

        if self.auto_external_reporter and self.auto_external_reporter.is_running:
            print("Station: Auto external reporter is already running")
            return True

        try:
            if not self.auto_external_reporter:
                self.auto_external_reporter = AutoExternalReporter(
                    station_instance=self,
                    enabled=constants.AUTO_EVAL_EXTERNAL_REPORT,
                    log_queue=log_queue
                )

            if self.auto_external_reporter.start_reporter_loop():
                print("Station: Auto external reporter started successfully")
                return True
            else:
                print("Station: Failed to start auto external reporter")
                self.auto_external_reporter = None
                return False
        except Exception as e:
            print(f"Station: Error starting auto external reporter: {e}")
            self.auto_external_reporter = None
            return False

    def stop_auto_external_reporter(self):
        """Stop the auto external reporter"""
        if self.auto_external_reporter:
            self.auto_external_reporter.stop_reporter_loop()
            self.auto_external_reporter = None
            print("Station: Auto external reporter stopped")

    def start_auto_theory_evaluator(self, log_queue=None) -> bool:
        """Start the auto theory evaluator if enabled"""
        if not (getattr(constants, "AUTO_EVAL_THEORY", False) and getattr(constants, "THEORY_ROOM_ENABLED", False)):
            print("Station: Auto theory evaluation is disabled")
            return False

        if self.auto_theory_evaluator and self.auto_theory_evaluator.is_running:
            print("Station: Auto theory evaluator is already running")
            return True

        try:
            if not self.auto_theory_evaluator:
                self.auto_theory_evaluator = AutoTheoryEvaluator(
                    station_instance=self,
                    storage_manager=getattr(self.rooms.get(constants.ROOM_THEORY, None), "storage", None),
                    enabled=getattr(constants, "AUTO_EVAL_THEORY", False),
                )
            if self.auto_theory_evaluator.start_evaluation_loop():
                print("Station: Auto theory evaluator started successfully")
                return True
            else:
                print("Station: Failed to start auto theory evaluator")
                self.auto_theory_evaluator = None
                return False
        except Exception as e:
            print(f"Station: Error starting auto theory evaluator: {e}")
            self.auto_theory_evaluator = None
            return False

    def stop_auto_theory_evaluator(self):
        """Stop the auto theory evaluator"""
        if self.auto_theory_evaluator:
            self.auto_theory_evaluator.stop_evaluation_loop()
            self.auto_theory_evaluator = None
            print("Station: Auto theory evaluator stopped")
    
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

    def start_auto_archive_surveyor(self, log_queue=None) -> bool:
        """Start the archive surveyor if enabled."""
        if not getattr(constants, "ARCHIVE_SURVEY_ENABLED", False):
            print("Station: Archive Surveyor is disabled")
            return False

        if self.auto_archive_surveyor and self.auto_archive_surveyor.is_running:
            print("Station: Archive Surveyor is already running")
            return True

        try:
            if not self.auto_archive_surveyor:
                self.auto_archive_surveyor = AutoArchiveSurveyor(
                    station_instance=self,
                    enabled=getattr(constants, "ARCHIVE_SURVEY_ENABLED", False),
                    log_queue=log_queue,
                )

            if self.auto_archive_surveyor.start_surveyor_loop():
                print("Station: Archive Surveyor started successfully")
                return True

            print("Station: Failed to start Archive Surveyor")
            self.auto_archive_surveyor = None
            return False
        except Exception as e:
            print(f"Station: Error starting Archive Surveyor: {e}")
            self.auto_archive_surveyor = None
            return False

    def stop_auto_archive_surveyor(self):
        """Stop the archive surveyor."""
        if self.auto_archive_surveyor:
            self.auto_archive_surveyor.stop_surveyor_loop()
            self.auto_archive_surveyor = None
            print("Station: Archive Surveyor stopped")

    def init_stagnation_protocol(self):
        """Initialize the stagnation protocol if enabled and research center is active"""
        if (
            constants.STAGNATION_ENABLED
            and constants.RESEARCH_CENTER_ENABLED
            and not constants.RESEARCH_NO_SCORE
        ):
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
            fields = agent_summary.get_agent_human_intervention_fields(
                agent_name,
                base_path=self.room_context.constants_module.BASE_STATION_DATA_PATH,
            )
            if agent_summary.agent_has_human_intervention_request(fields):
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
        current_tick_getter = getattr(self, "_get_current_tick", None)
        current_tick = (
            current_tick_getter()
            if callable(current_tick_getter)
            else self.config.get(constants.STATION_CONFIG_CURRENT_TICK, 0)
        )
        stats = {
            'pending_human_requests': {
                'request_ids': [],
                'agents': [],
                'agent_request_map': {}
            },
            'current_tick': current_tick,
            'top_research_submission': None,
            'running_experiments_count': 0,
            'running_experiments': [],
            'queued_experiments_count': 0,
            'queued_experiments': [],
            'running_jobs_count': 0,
            'running_jobs': [],
            'queued_jobs_count': 0,
            'queued_jobs': [],
            'tick_timing': tick_timing.get_timing_summary(),
        }
        
        # Get pending human requests from Administrative Counter
        if constants.ROOM_ADMIN in self.rooms:
            admin_counter = self.rooms[constants.ROOM_ADMIN]
            # Get active agents from turn order
            active_agents = self.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, [])
            # Filter to only those actually awaiting intervention
            agents_awaiting = []
            for agent_name in active_agents:
                fields = agent_summary.get_agent_human_intervention_fields(
                    agent_name,
                    base_path=self.room_context.constants_module.BASE_STATION_DATA_PATH,
                )
                if agent_summary.agent_has_human_intervention_request(fields):
                    agents_awaiting.append(agent_name)
            
            # Get summary from administrative counter
            summary = admin_counter.get_pending_requests_summary(active_agents=agents_awaiting)
            stats['pending_human_requests'] = summary
        
        # Get comprehensive evaluation statistics from Research Evaluation Manager
        if constants.RESEARCH_CENTER_ENABLED:
            eval_manager = getattr(self, "research_eval_manager", None)
            if eval_manager is None and hasattr(self, 'auto_research_evaluator') and self.auto_research_evaluator:
                eval_manager = self.auto_research_evaluator.eval_manager
            if eval_manager is None:
                eval_manager = EvaluationManager()

            eval_stats = eval_manager.get_evaluation_statistics()
            if constants.RESEARCH_NO_SCORE:
                stats['top_research_submission'] = None
            else:
                stats['top_research_submission'] = eval_stats['top_submission']
            stats['running_experiments_count'] = eval_stats['running_count']
            stats['running_experiments'] = eval_stats['running_evaluations']
            stats['queued_experiments_count'] = eval_stats.get('queued_count', 0)
            stats['queued_experiments'] = eval_stats.get('queued_evaluations', [])

            stats['running_jobs_count'] = eval_stats['running_count']
            stats['running_jobs'] = list(eval_stats['running_evaluations'])
            stats['queued_jobs_count'] = eval_stats.get('queued_count', 0)
            stats['queued_jobs'] = list(eval_stats.get('queued_evaluations', []))

        auto_archive_surveyor = getattr(self, "auto_archive_surveyor", None)
        if getattr(constants, "ARCHIVE_SURVEY_ENABLED", False) and auto_archive_surveyor:
            survey_stats = auto_archive_surveyor.get_job_statistics()
            stats['running_jobs'].extend(survey_stats.get('running_jobs', []))
            stats['queued_jobs'].extend(survey_stats.get('queued_jobs', []))
            stats['running_jobs_count'] = len(stats['running_jobs'])
            stats['queued_jobs_count'] = len(stats['queued_jobs'])

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

        # Keep token usage monotonic for LLM-reported counts; allow decreases only right after pruning.
        existing_tokens_used = agent_data.get(constants.AGENT_TOKEN_BUDGET_CURRENT_KEY)
        last_prune_tick = agent_data.get(constants.AGENT_LAST_PRUNE_ACTION_TICK_KEY)
        current_tick = self._get_current_tick()
        allow_decrease = last_prune_tick is not None and current_tick is not None and last_prune_tick >= current_tick - 1
        if not allow_decrease and existing_tokens_used is not None and current_session_total_tokens_used < existing_tokens_used:
            current_session_total_tokens_used = existing_tokens_used
        agent_data[constants.AGENT_TOKEN_BUDGET_CURRENT_KEY] = current_session_total_tokens_used # station.py:1027
        agent_data.pop(constants.AGENT_TOKEN_BUDGET_CURRENT_STALE_KEY, None)
        agent_data.pop(constants.AGENT_TOKEN_BUDGET_STALE_REASON_KEY, None)

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

        if supervisor_utils.is_theorist(agent_data, constants):
            return False
        
        # Check if agent is a supervisor
        is_supervisor = supervisor_utils.is_supervisor(agent_data, constants)
        
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

    def _check_and_apply_inactivity_warning(self, agent_data: Dict[str, Any], current_tick: Optional[int] = None) -> Dict[str, Any]:
        """
        Checks for agent inactivity based on agent type and role.
        Guest agents: No warnings
        Recursive agents (non-supervisor): Warning after RECURSIVE_AGENT_INACTIVITY_THRESHOLD ticks without research submission
        Recursive agents (supervisor): Warning after RECURSIVE_SUPERVISOR_INACTIVITY_THRESHOLD ticks without meaningful actions
        Returns the (potentially modified) agent_data.
        """
        return system_messages.check_and_apply_inactivity_warning(self, agent_data, current_tick)

    def _update_meta_reflection_tick_count(self, agent_data: Dict[str, Any], current_tick: int) -> Dict[str, Any]:
        """
        Increments the compulsory meta reflection countdown for mature agents.
        """
        return system_messages.update_meta_reflection_tick_count(self, agent_data, current_tick)

    def _check_and_apply_meta_reflection_warning(self, agent_data: Dict[str, Any], current_tick: int) -> Dict[str, Any]:
        """
        Checks compulsory meta reflection warnings and adds notifications if necessary.
        """
        return system_messages.check_and_apply_meta_reflection_warning(self, agent_data, current_tick)
    
    def _check_and_apply_life_warnings(self, agent_data: Dict[str, Any], current_tick: int) -> Dict[str, Any]:
        """
        Checks agent life limit warnings based on age and adds notifications if necessary.
        Updates life warning sent flag in agent_data.
        Returns the (potentially modified) agent_data.
        """
        return system_messages.check_and_apply_life_warnings(self, agent_data, current_tick)

    def _check_and_apply_supervisor_report_reminder(self, agent_data: Dict[str, Any], current_tick: int) -> Dict[str, Any]:
        """
        Checks supervisor report reminders based on age and adds notifications if necessary.
        Returns the (potentially modified) agent_data.
        """
        return system_messages.check_and_apply_supervisor_report_reminder(self, agent_data, current_tick)

    def _check_and_apply_token_warnings(self, agent_data: Dict[str, Any], effective_max_budget: Optional[int]) -> Dict[str, Any]:
        """
        Checks token budget warnings based on current usage in agent_data
        and adds notifications to agent_data if necessary. Updates warning sent flags in agent_data.
        Warning flags are reset when token usage drops below respective thresholds.
        This method expects agent_data to have the latest AGENT_TOKEN_BUDGET_CURRENT_KEY.
        Returns the (potentially modified) agent_data.
        """
        return system_messages.check_and_apply_token_warnings(self, agent_data, effective_max_budget)

    def update_agent_token_budget(self, agent_name: str, current_session_total_tokens_used: int) -> bool: # station.py:1012
        """
        Updates the agent's token budget usage.
        Station's configured token budget is advisory only.
        Hard termination is handled separately when the provider returns a real
        context-window overflow error (LLMContextOverflowError).
        Warnings are generated by request_status via _check_and_apply_token_warnings.
        Returns True if the usage update succeeded, False on update failure.
        """
        # Step 1: Update token usage in the agent's data file.
        # This loads, updates the current_budget_key, and saves agent_data.
        # It returns the potentially modified agent_data and the effective_max_budget.
        updated_agent_data, effective_max_budget = self._update_agent_token_usage_only(agent_name, current_session_total_tokens_used)

        if not updated_agent_data or effective_max_budget is None:
            # Error already printed by _update_agent_token_usage_only or agent not found
            return False # Cannot determine status or problem updating

        # Keep recording local budget usage even when it exceeds Station's configured
        # soft limit. The agent should only be terminated when the upstream provider
        # returns a real context/window overflow error.
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
        session_end_flow.terminate_agent_session_with_broadcast(self, agent_name, reason, critical_notification)
    
    def _check_agent_life_limit(
        self,
        agent_name: str,
        current_tick: int,
    ) -> Tuple[bool, Optional[LoggingInternalActionHandlerWrapper]]:
        """
        Checks if agent has reached their life limit and terminates session if necessary.
        Returns (session_continues, optional normal-end internal handler).
        """
        agent_data = self.agent_module.load_agent_data(agent_name)
        if not agent_data:
            return False, None  # Agent not found
        
        # Get agent-specific max_age, fallback to AGENT_MAX_LIFE for backward compatibility
        agent_max_age = agent_data.get(constants.AGENT_MAX_AGE_KEY, constants.AGENT_MAX_LIFE)
        if agent_max_age is None:
            return True, None  # No life limit for this agent
        
        birth_tick = agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
        if birth_tick is None:
            return True, None  # No birth tick recorded, cannot check age
        
        agent_age = current_tick - birth_tick
        
        if agent_age >= agent_max_age:
            print(f"Agent {agent_name} has reached their life limit ({agent_age}/{agent_max_age} ticks). Terminating session.")
            
            critical_notification = f"CRITICAL: Your life limit of {agent_max_age} ticks has been reached. Your session is being terminated."
            if session_end_flow.should_request_next_role_definition(
                agent_data,
                constants,
                session_end_flow.SESSION_END_REASON_LIFE_LIMIT,
            ):
                handler = session_end_flow.NextRoleDefinitionSessionEndHandler(
                    agent_data=agent_data,
                    room_context=self.room_context,
                    current_tick=current_tick,
                    reason=session_end_flow.SESSION_END_REASON_LIFE_LIMIT,
                    critical_notification=critical_notification,
                )
                wrapped_handler = LoggingInternalActionHandlerWrapper(
                    actual_handler=handler,
                    agent_name=agent_name,
                    log_dialogue_entry_func=self._log_dialogue_entry,
                )
                return False, wrapped_handler

            session_end_flow.finalize_session_end(
                agent_data,
                self.room_context,
                current_tick,
                session_end_flow.SESSION_END_REASON_LIFE_LIMIT,
                critical_notification,
            )
            return False, None  # Session terminated
        
        return True, None  # Session continues

    def _get_agent_age_status(self, agent_data: Dict[str, Any], current_tick: int) -> Optional[str]:
        """
        Returns the agent's age status: "immature", "mature", "tenured", or None if no status applies.
        """
        birth_tick = agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
        if birth_tick is None:
            return None

        agent_age = current_tick - birth_tick
        tenured_threshold = getattr(constants, "MIN_AGENT_AGE_BEFORE_LEAVE", None)
        tenured_enabled = tenured_threshold is not None and tenured_threshold > 0

        if constants.AGENT_ISOLATION_TICKS is not None and agent_age < constants.AGENT_ISOLATION_TICKS:
            return "immature"

        if tenured_enabled and agent_age >= tenured_threshold:
            return "tenured"

        if constants.AGENT_ISOLATION_TICKS is not None:
            return "mature"

        return None

    def _is_agent_mature(self, agent_data: Dict[str, Any], current_tick: int) -> bool:
        """
        Determines if an agent has reached maturity based on isolation settings.
        Returns True if isolation is disabled or agent age >= AGENT_ISOLATION_TICKS.
        """
        if constants.AGENT_ISOLATION_TICKS is None:
            return True  # Isolation disabled, all agents are considered mature

        age_status = self._get_agent_age_status(agent_data, current_tick)
        if age_status is None:
            return True  # No birth tick, consider mature

        return age_status in ("mature", "tenured")
    
    def _check_and_notify_maturity(self, agent_name: str, agent_data: Dict[str, Any], current_tick: int) -> None:
        """
        Check if agent has just reached maturity and send congratulatory notification.
        """
        system_messages.check_and_notify_maturity(self, agent_name, agent_data, current_tick)
    
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
