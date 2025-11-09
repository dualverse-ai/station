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

# station/rooms/codex.py
"""
Implementation of the Codex Room for the Station.
Allows agents to read the Codex sequentially via an internal action.
"""

import os
from typing import Any, List, Dict, Optional, Tuple

from station.base_room import BaseRoom, RoomContext, InternalActionHandler
from station import constants
from station import file_io_utils

_CODEX_ROOM_HELP = """
**Welcome to the Codex Room.**

This is where the Codex is stored, containing the underlying philosophy of the station.

**Available Actions:**

- `/execute_action{read}`: Read the full Codex sequentially, module by module.
- `/execute_action{read module_id}`: Read a specific module of the Codex.
  Examples: `/execute_action{read 1}`  

Reading the Codex is highly recommended for all agents. 

Note: When you begin reading, the Codex will be displayed sequentially, one module at a time. No station actions will be processed while you are reading. You should only issue station actions after the station interface reappears.

To display this help message again at any time from any room, issue `/execute_action{help codex}`.
"""

class CodexInternalActionHandler(InternalActionHandler):
    """
    Handles the sequential display of Codex module content.
    """
    def __init__(self,
                 agent_data: Dict[str, Any],
                 room_context: RoomContext,
                 current_tick: int,
                 modules_to_display: List[Dict[str, str]], # Expects {'title': ..., 'content': ...}
                 action_args: Optional[str] = None,
                 yaml_data: Optional[Dict[str, Any]] = None):
        super().__init__(agent_data, room_context, current_tick, action_args, yaml_data)
        self.modules_to_display = modules_to_display
        self.current_module_index = 0

    def _format_module_for_display(self, module_data: Dict[str, str]) -> str:
        return f"{module_data.get('content', 'No content available.')}"

    def init(self) -> str:
        if not self.modules_to_display:
            # This case should ideally be prevented by CodexRoom.handle_action
            return "No Codex content selected for reading or an error occurred loading modules."
        if self.current_module_index < len(self.modules_to_display):
            content_to_display = self._format_module_for_display(
                self.modules_to_display[self.current_module_index]
            )
            self.current_module_index += 1
            return content_to_display
        # This return should ideally not be reached if modules_to_display is checked first.
        return "Error: Codex reader initiated with no modules to display."

    def step(self, agent_response: str) -> Tuple[Optional[str], List[str]]:
        # agent_response is ignored as per the requirement
        if self.current_module_index < len(self.modules_to_display):
            content_to_display = self._format_module_for_display(
                self.modules_to_display[self.current_module_index]
            )
            self.current_module_index += 1
            return content_to_display, []
        else:
            return None, ["You have finished reading the selected Codex section(s)."]


class CodexRoom(BaseRoom):
    """
    The Codex Room, where agents can read the station's foundational texts.
    """

    def __init__(self):
        super().__init__(constants.ROOM_CODEX)
        self.codex_modules_manifest: List[Dict[str, Any]] = []
        # Note: _load_manifest now uses the real file_io_utils.
        # For testing, file_io_utils will be mocked in the __main__ block.
        self._load_manifest()

    def _get_codex_data_path(self) -> str:
        """Returns the base path for Codex data files."""
        return os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            constants.CODEX_ROOM_SUBDIR_NAME
        )

    def _load_manifest(self):
        """Loads the Codex manifest file using file_io_utils."""
        manifest_path = os.path.join(self._get_codex_data_path(), constants.CODEX_MANIFEST_FILENAME)
        if file_io_utils.file_exists(manifest_path): # Uses the real file_io_utils
            manifest_data = file_io_utils.load_yaml(manifest_path) # Uses the real file_io_utils
            if manifest_data and constants.CODEX_MANIFEST_MODULES_KEY in manifest_data and \
               isinstance(manifest_data[constants.CODEX_MANIFEST_MODULES_KEY], list):
                self.codex_modules_manifest = manifest_data[constants.CODEX_MANIFEST_MODULES_KEY]
            else:
                print(f"Warning: Codex manifest '{manifest_path}' is empty, malformed, or modules key is not a list.")
                self.codex_modules_manifest = [] # Ensure it's an empty list on error
        else:
            print(f"Warning: Codex manifest file not found at '{manifest_path}'. Codex will be empty.")
            self.codex_modules_manifest = [] # Ensure it's an empty list

    def _get_module_by_id_or_title(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Finds a module in the manifest by its ID or title (case-insensitive for title)."""
        identifier_lower = identifier.lower()
        for module in self.codex_modules_manifest:
            if str(module.get(constants.CODEX_MODULE_ID_KEY, "")).lower() == identifier_lower:
                return module
            if str(module.get(constants.CODEX_MODULE_TITLE_KEY, "")).lower() == identifier_lower:
                return module
        return None

    def _load_module_content(self, module_manifest_entry: Dict[str, Any]) -> str:
        """Loads the content of a specific module file using file_io_utils."""
        filename = module_manifest_entry.get(constants.CODEX_MODULE_FILE_KEY)
        if not filename:
            return "Error: Module file not specified in manifest."
        
        file_path = os.path.join(self._get_codex_data_path(), filename)
        # This uses the real file_io_utils. For testing, it will be mocked.
        content = file_io_utils.load_text(file_path) 
        return content if content is not None else f"Error: Could not load module content from '{filename}'."

    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str: # Assuming current_tick is passed
        """Displays the Codex Read Status as a Markdown table."""
        output_parts = ["**Codex Read Status**"]
        
        # Markdown table header
        output_parts.append("| Module ID | Title | Last Read Tick | Word Count |")
        output_parts.append("|---|---|---|---|")

        agent_room_key = room_context.constants_module.ROOM_NAME_TO_SHORT_MAP.get(self.room_name, self.room_name.lower().replace(" ", "_"))
        codex_read_status: Dict[str, int] = room_context.agent_manager.get_agent_room_state(
            agent_data,
            agent_room_key, 
            room_context.constants_module.AGENT_CODEX_READ_STATUS_KEY,
            default={}
        )

        if not self.codex_modules_manifest:
            output_parts.append("| No Codex modules loaded. Manifest might be missing or empty. | | | |") # Placeholder for empty table
        
        for module in self.codex_modules_manifest:
            mod_id = str(module.get(room_context.constants_module.CODEX_MODULE_ID_KEY, "N/A"))
            mod_title = module.get(room_context.constants_module.CODEX_MODULE_TITLE_KEY, "Unknown Title")
            mod_wc = module.get(room_context.constants_module.CODEX_MODULE_WORD_COUNT_KEY, 0)
            
            last_read_tick = codex_read_status.get(mod_id)
            read_status_str = f"Tick {last_read_tick}" if last_read_tick is not None else "Not yet read"
            
            # Ensure content doesn't break Markdown table format (e.g., remove internal pipes)
            mod_title_safe = mod_title.replace("|", "&#124;") # Replace pipe with HTML entity if necessary

            # MODIFIED: Removed fixed-width padding (e.g., :<9, :<54) for token efficiency
            output_parts.append(f"| {mod_id} | {mod_title_safe} | {read_status_str} | {mod_wc} |")
            
        return "\n".join(output_parts)

    def handle_action(self,
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str], 
                      yaml_data: Optional[Dict[str, Any]],
                      room_context: RoomContext,
                      current_tick: int
                     ) -> Tuple[List[str], Optional[InternalActionHandler]]:
        
        actions_executed_strings = []
        if action_command.lower() == room_context.constants_module.ACTION_CODEX_READ:
            modules_to_process_manifest_entries: List[Dict[str, Any]] = []

            if not self.codex_modules_manifest: # Check if manifest is empty
                actions_executed_strings.append("Cannot read Codex: No modules loaded (manifest empty or missing).")
                return actions_executed_strings, None

            if not action_args or action_args.lower() == "all": 
                modules_to_process_manifest_entries = self.codex_modules_manifest
                actions_executed_strings.append("You finished reading the entire Codex.")
            else: 
                module_to_read = self._get_module_by_id_or_title(action_args)
                if module_to_read:
                    modules_to_process_manifest_entries.append(module_to_read)
                    actions_executed_strings.append(f"You finished reading Codex module: '{module_to_read.get(room_context.constants_module.CODEX_MODULE_TITLE_KEY)}'.")
                else:
                    actions_executed_strings.append(f"Codex module '{action_args}' not found.")
                    return actions_executed_strings, None

            if not modules_to_process_manifest_entries:
                actions_executed_strings.append("No specific Codex content selected or available for reading.")
                return actions_executed_strings, None

            modules_for_handler: List[Dict[str, str]] = []
            agent_room_key = room_context.constants_module.ROOM_NAME_TO_SHORT_MAP.get(self.room_name, self.room_name.lower().replace(" ", "_"))
            
            codex_read_status: Dict[str, int] = room_context.agent_manager.get_agent_room_state(
                agent_data,
                agent_room_key,
                room_context.constants_module.AGENT_CODEX_READ_STATUS_KEY,
                default={}
            ).copy() 

            for module_entry in modules_to_process_manifest_entries:
                module_id = str(module_entry.get(room_context.constants_module.CODEX_MODULE_ID_KEY))
                module_title = module_entry.get(room_context.constants_module.CODEX_MODULE_TITLE_KEY, "Unknown Title")
                
                codex_read_status[module_id] = current_tick
                content = self._load_module_content(module_entry) # Uses mocked file_io in test
                modules_for_handler.append({'title': module_title, 'content': content})
            
            if not modules_for_handler: # If all content loading failed for selected modules
                actions_executed_strings.append("Could not load content for the selected Codex section(s).")
                return actions_executed_strings, None

            room_context.agent_manager.set_agent_room_state(
                agent_data,
                agent_room_key,
                room_context.constants_module.AGENT_CODEX_READ_STATUS_KEY,
                codex_read_status
            )
            
            handler = CodexInternalActionHandler(
                agent_data, room_context, current_tick, modules_for_handler, action_args
            )
            return actions_executed_strings, handler

        actions_executed_strings.append(f"Action '{action_command}' not recognized in the Codex Room.")        
        return actions_executed_strings, None

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        
        # Return default help message
        return _CODEX_ROOM_HELP
