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

# station/rooms/capsule_room_base.py
"""
Abstract base class for rooms that handle capsules using a shared protocol
(e.g., Private Memory, Public Memory, Archive, Mail Room).
Includes unread message count display and @mention notification processing.
"""
import re
from abc import abstractmethod
from typing import Any, List, Dict, Optional, Tuple

from station.base_room import BaseRoom, RoomContext, InternalActionHandler
from station import constants
from station import capsule as capsule_manager 
from station import agent as agent_manager 


class CapsuleHandlerBaseRoom(BaseRoom):
    """
    Base class for rooms implementing the Capsule Protocol.
    Handles common actions like create, reply, read, delete, pin, etc.
    """

    @abstractmethod
    def _get_capsule_type(self) -> str:
        """Concrete rooms must return their capsule type (e.g., constants.CAPSULE_TYPE_PUBLIC)."""
        pass

    @abstractmethod
    def _get_lineage_for_capsule_operations(self, agent_data: Dict[str, Any], room_context: RoomContext) -> Optional[str]:
        """For Private Memory, returns the agent's lineage. For other types, returns None."""
        pass

    @abstractmethod
    def _check_action_permission(self,
                                 action_command: str,
                                 agent_data: Dict[str, Any],
                                 room_context: RoomContext,
                                 capsule_data: Optional[Dict[str, Any]] = None,
                                 target_id_str: Optional[str] = None, 
                                 target_numeric_id: Optional[int] = None) -> bool:
        """Concrete rooms implement permission checks for actions."""
        pass

    def _get_additional_yaml_fields_for_create(self) -> List[str]:
        """Concrete rooms list additional *required* YAML fields for /execute_action{create}."""
        return []

    def _get_room_specific_header_elements(self,
                                           agent_data: Dict[str, Any],
                                           room_context: RoomContext,
                                           current_tick: int) -> List[str]:
        """Concrete rooms can override to add specific info before capsule listings."""
        return []
    
    @abstractmethod 
    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        """Concrete rooms must provide their help message."""
        pass

    @abstractmethod
    def _after_capsule_created(self, 
                               new_capsule_data: Dict[str, Any], 
                               creator_agent_data: Dict[str, Any], 
                               room_context: RoomContext, 
                               current_tick: int):
        """Hook for notifications after capsule creation. Implemented by concrete rooms."""
        pass

    @abstractmethod
    def _after_reply_added(self, 
                           target_capsule_data: Dict[str, Any], 
                           new_message_data: Dict[str, Any], 
                           replier_agent_data: Dict[str, Any], 
                           room_context: RoomContext, 
                           current_tick: int):
        """Hook for notifications after a reply. Implemented by concrete rooms."""
        pass


    def _get_agent_room_data_key(self, room_context: RoomContext) -> str:
        """Gets the key used in agent_data for this room's specific state (e.g., 'public_memory')."""
        # Ensure constants_module is accessed correctly from room_context
        short_name = room_context.constants_module.ROOM_NAME_TO_SHORT_MAP.get(self.room_name)
        if not short_name:
            # Fallback or error if room_name isn't in the map
            print(f"Warning: Short name for room '{self.room_name}' not found in ROOM_NAME_TO_SHORT_MAP. Using a derived key.")
            return self.room_name.lower().replace(" ", "_").replace("_room", "")
        return short_name


    def _get_agent_read_status(self, agent_data: Dict[str, Any], room_context: RoomContext) -> Dict[str, bool]:
        room_key = self._get_agent_room_data_key(room_context)
        return room_context.agent_manager.get_agent_room_state(
            agent_data, room_key, room_context.constants_module.AGENT_ROOM_STATE_READ_STATUS_KEY, default={}
        )

    def _set_agent_read_status(self, agent_data: Dict[str, Any], item_id_str: str, read: bool, room_context: RoomContext):
        room_key = self._get_agent_room_data_key(room_context)
        current_read_statuses = self._get_agent_read_status(agent_data, room_context).copy()
        if read:
            current_read_statuses[item_id_str] = True
        else: 
            current_read_statuses.pop(item_id_str, None)
        room_context.agent_manager.set_agent_room_state(
            agent_data, room_key, room_context.constants_module.AGENT_ROOM_STATE_READ_STATUS_KEY, current_read_statuses
        )

    def _get_pinned_capsules_ids(self, agent_data: Dict[str, Any], room_context: RoomContext) -> List[str]:
        room_key = self._get_agent_room_data_key(room_context)
        return room_context.agent_manager.get_agent_room_state(
            agent_data, room_key, room_context.constants_module.AGENT_ROOM_STATE_PINNED_CAPSULES_KEY, default=[]
        )
    
    def _set_pinned_capsules_ids(self, agent_data: Dict[str, Any], pinned_ids: List[str], room_context: RoomContext):
        room_key = self._get_agent_room_data_key(room_context)
        room_context.agent_manager.set_agent_room_state(
            agent_data, room_key, room_context.constants_module.AGENT_ROOM_STATE_PINNED_CAPSULES_KEY, pinned_ids
        )

    def _get_current_page(self, agent_data: Dict[str, Any], room_context: RoomContext) -> int:
        room_key = self._get_agent_room_data_key(room_context)
        page = room_context.agent_manager.get_agent_room_state(
            agent_data, room_key, room_context.constants_module.AGENT_ROOM_STATE_CURRENT_PAGE_KEY, default=1
        )
        return page if isinstance(page, int) and page > 0 else 1


    def _set_current_page(self, agent_data: Dict[str, Any], page_num: int, room_context: RoomContext):
        room_key = self._get_agent_room_data_key(room_context)
        room_context.agent_manager.set_agent_room_state(
            agent_data, room_key, room_context.constants_module.AGENT_ROOM_STATE_CURRENT_PAGE_KEY, page_num
        )

    def _format_capsule_for_list_display(self, capsule_metadata: Dict[str, Any], agent_read_status: Dict[str, bool], room_context: RoomContext) -> str:
        """Formats a single capsule's metadata for display in a Markdown table row."""
        consts = room_context.constants_module
        capsule_id_str = capsule_metadata.get(consts.CAPSULE_ID_KEY, "N/A")
        numeric_id_part = capsule_id_str 
        try:
            match = re.search(r'(\d+)$', capsule_id_str)
            if match: numeric_id_part = match.group(1)
        except Exception: pass

        title = capsule_metadata.get(consts.CAPSULE_TITLE_KEY, "No Title")
        author = capsule_metadata.get(consts.CAPSULE_AUTHOR_NAME_KEY, "Unknown")
        date_tick = capsule_metadata.get(consts.CAPSULE_CREATED_AT_TICK_KEY, "N/A")
        word_count = capsule_metadata.get(consts.CAPSULE_WORD_COUNT_TOTAL_KEY, 0)
        total_messages = capsule_metadata.get('total_message_count', 0) 
        unread_count = capsule_metadata.get(consts.CAPSULE_UNREAD_MESSAGE_COUNT_KEY, 0)
        
        # MODIFICATION: Change status string for fully read capsules
        status_str = ""
        if total_messages == 0:
            status_str = "(No Msgs)" # Or keep empty: ""
        elif unread_count > 0:
            status_str = f"({unread_count} unread)"
        else: # unread_count is 0 and total_messages > 0
            status_str = "(All Read)"
        
        title_safe = title.replace("|", "&#124;")
        author_safe = author.replace("|", "&#124;")

        base_row_parts = [
            f" {numeric_id_part} ",
            f" {title_safe} ",
            f" {author_safe} "
        ]
        
        mail_specific_cols = []
        if self._get_capsule_type() == consts.CAPSULE_TYPE_MAIL:
            recipients_list = capsule_metadata.get(consts.CAPSULE_RECIPIENTS_KEY, [])
            recipients_display = ", ".join(recipients_list) if recipients_list else "N/A"
            recipients_display_safe = recipients_display.replace("|", "&#124;")
            mail_specific_cols.append(f" {recipients_display_safe} ")

        end_row_parts = [
            f" Tick {date_tick} ",
            f" {word_count} ",
            f" {total_messages} ",
            f" {status_str} " # Use the updated status_str
        ]
        
        full_row_content = "|".join(base_row_parts + mail_specific_cols + end_row_parts)
        return f"|{full_row_content}|"

    def get_room_output(self,
                        agent_data: Dict[str, Any],
                        room_context: RoomContext,
                        current_tick: int) -> str:
        """
        Generates the complete textual output for the room, including room-specific help
        and potentially the global Capsule Protocol help.
        Modifies agent_data in-memory to mark help as shown.
        """
        # Determine if room-specific help *will be* shown by the super() call.
        # This check needs to happen BEFORE super().get_room_output() modifies the flag.
        room_data_key_for_specific_help = self._get_agent_room_data_key(room_context)
        will_show_room_specific_help = not room_context.agent_manager.get_agent_room_state(
            agent_data,
            room_data_key_for_specific_help,
            room_context.constants_module.AGENT_ROOM_STATE_FIRST_VISIT_HELP_SHOWN_KEY,
            default=False
        )

        # Call super to get base output, which includes room-specific help if it's a first visit to *this* room.
        # This call will also update the AGENT_ROOM_STATE_FIRST_VISIT_HELP_SHOWN_KEY for this specific room.
        base_output_str = super().get_room_output(agent_data, room_context, current_tick)

        # Now, manage and append Capsule Protocol help based on global flag and context.
        additional_help_parts = []
        capsule_protocol_help_globally_shown = room_context.agent_manager.get_agent_room_state(
            agent_data,
            room_context.constants_module.AGENT_STATE_DATA_KEY, # Use the global state key
            room_context.constants_module.AGENT_STATE_CAPSULE_PROTOCOL_HELP_SHOWN_KEY,
            default=False
        )

        if not capsule_protocol_help_globally_shown:
            additional_help_parts.append("\n\n---\n\n" + room_context.constants_module.TEXT_CAPSULE_PROTOCOL_HELP + "\n\n---")
            room_context.agent_manager.set_agent_room_state(
                agent_data,
                room_context.constants_module.AGENT_STATE_DATA_KEY, # Use the global state key
                room_context.constants_module.AGENT_STATE_CAPSULE_PROTOCOL_HELP_SHOWN_KEY,
                True
            )
        # If global help was already shown, but this specific room's help was just displayed, add a reminder.
        elif will_show_room_specific_help:
            additional_help_parts.append(
                f"\n\n---\n\nNote: For detailed capsule commands, use `/execute_action{{help {room_context.constants_module.SHORT_ROOM_NAME_CAPSULE_PROTOCOL}}}`.\n\n---"
            )
        
        if additional_help_parts:
            return base_output_str + "".join(additional_help_parts)
        
        return base_output_str

    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:
        output_lines = []
        consts = room_context.constants_module
        capsule_type = self._get_capsule_type()
        # Use room_context when calling _get_lineage_for_capsule_operations
        lineage = self._get_lineage_for_capsule_operations(agent_data, room_context) 
        agent_read_status = self._get_agent_read_status(agent_data, room_context)
        agent_name = agent_data.get(consts.AGENT_NAME_KEY) # Get current agent's name

        header_elements = self._get_room_specific_header_elements(agent_data, room_context, current_tick)
        if header_elements:
            output_lines.extend(header_elements); output_lines.append("")

        table_header_parts = [" ID ", " Title ", " Author "]
        table_separator_parts = [":----", ":-------------------------------", ":----------------"]
        if capsule_type == consts.CAPSULE_TYPE_MAIL:
            table_header_parts.append(" Recipients ")
            table_separator_parts.append(":-----------------")
        table_header_parts.extend([" Date ", " Words ", " Msgs ", " Status "])
        table_separator_parts.extend([":----------", ":------:", ":-----:", ":---------"])
        
        dynamic_table_header = "|" + "|".join(table_header_parts) + "|"
        dynamic_table_separator = "|" + "|".join(table_separator_parts) + "|"

        # Fetch all capsule metadata initially
        all_capsule_metadata_unfiltered = capsule_manager.list_capsules(capsule_type, lineage, agent_read_status)

        # --- Filter mail capsules for visibility ---
        if capsule_type == consts.CAPSULE_TYPE_MAIL and agent_name:
            authorized_capsules_metadata = []
            for meta in all_capsule_metadata_unfiltered:
                is_author = agent_name == meta.get(consts.CAPSULE_AUTHOR_NAME_KEY)
                is_recipient = agent_name in meta.get(consts.CAPSULE_RECIPIENTS_KEY, [])
                if is_author or is_recipient:
                    authorized_capsules_metadata.append(meta)
            all_capsule_metadata = authorized_capsules_metadata 
        else:
            all_capsule_metadata = all_capsule_metadata_unfiltered
        # --- End of mail filtering ---

        pinned_ids_full = self._get_pinned_capsules_ids(agent_data, room_context)
        if pinned_ids_full:
            output_lines.append("**Pinned Capsules**")
            # --- MODIFICATION: Filter pinned items based on the (already filtered) all_capsule_metadata ---
            valid_pinned_capsules_metadata = []
            # Create a set of valid capsule IDs from the filtered list for quick lookup
            valid_capsule_ids_for_display = {cap_meta.get(consts.CAPSULE_ID_KEY) for cap_meta in all_capsule_metadata}

            for full_capsule_id_str_pinned in pinned_ids_full:
                if full_capsule_id_str_pinned in valid_capsule_ids_for_display:
                    # Find the metadata from the already filtered all_capsule_metadata
                    capsule_meta = next((meta for meta in all_capsule_metadata if meta.get(consts.CAPSULE_ID_KEY) == full_capsule_id_str_pinned), None)
                    if capsule_meta: # Should always be found if ID is in valid_capsule_ids_for_display
                        valid_pinned_capsules_metadata.append(capsule_meta)
            # --- END MODIFICATION for pinned items ---
            
            if valid_pinned_capsules_metadata: # Display only if there are valid pinned items
                output_lines.append(dynamic_table_header)
                output_lines.append(dynamic_table_separator)
                # Sort the valid pinned capsules for display
                valid_pinned_capsules_metadata.sort(key=lambda x: x.get(consts.CAPSULE_CREATED_AT_TICK_KEY, 0), reverse=True)
                for capsule_meta in valid_pinned_capsules_metadata:
                    output_lines.append(self._format_capsule_for_list_display(capsule_meta, agent_read_status, room_context))
                output_lines.append("")
            # else: No valid pinned items to display, or list was empty. No explicit message needed here.


        current_page = self._get_current_page(agent_data, room_context)
        page_size = consts.DEFAULT_PAGE_SIZE_CAPSULES
        
        # --- MODIFICATION: REMOVE this redundant call to list_capsules ---
        # all_capsule_metadata = capsule_manager.list_capsules(capsule_type, lineage, agent_read_status) # <<< REMOVE THIS LINE
        # --- END MODIFICATION ---
        
        # Now, non_pinned_capsules should be derived from the (potentially filtered) all_capsule_metadata
        non_pinned_capsules = [cap for cap in all_capsule_metadata if cap.get(consts.CAPSULE_ID_KEY) not in pinned_ids_full]

        total_items = len(non_pinned_capsules)
        total_pages = (total_items + page_size - 1) // page_size if page_size > 0 else 0
        if total_items == 0: total_pages = 1 # If no items, still show Page 1/1

        current_page = max(1, min(current_page, total_pages if total_items > 0 else 1)) 
        self._set_current_page(agent_data, current_page, room_context) 

        start_index = (current_page - 1) * page_size
        end_index = start_index + page_size
        paginated_capsules = non_pinned_capsules[start_index:end_index]

        output_lines.append(f"**List of Capsules (Page {current_page} / {total_pages})**")
        if paginated_capsules:
            output_lines.append(dynamic_table_header)
            output_lines.append(dynamic_table_separator)
            for capsule_meta in paginated_capsules:
                output_lines.append(self._format_capsule_for_list_display(capsule_meta, agent_read_status, room_context))
        else:
            # Use the specific message for mail room if it's mail type
            if capsule_type == consts.CAPSULE_TYPE_MAIL:
                output_lines.append("No mail available that you are authorized to view on this page." if total_items > 0 else "No mail available that you are authorized to view.")
            else:
                output_lines.append("No capsules to display on this page." if total_items > 0 else "No capsules available in this room.")
        
        if total_pages > 1 :
            output_lines.append("")
            output_lines.append(f"(Use `/execute_action{{page N}}` to navigate between pages 1-{total_pages}.)")
        
        return "\n".join(output_lines)


    def handle_action(self,
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str],
                      yaml_data: Optional[Dict[str, Any]],
                      room_context: RoomContext,
                      current_tick: int
                     ) -> Tuple[List[str], Optional[Any]]: # InternalActionHandler type hint
        actions_executed = []
        consts = room_context.constants_module
        agent_name = agent_data.get(consts.AGENT_NAME_KEY, "UnknownAgent")
        capsule_type = self._get_capsule_type()
        lineage = self._get_lineage_for_capsule_operations(agent_data, room_context)

        # Helper to parse "X" or "X-Y" into (cap_id, msg_idx_str, full_msg_id_str)
        def parse_single_target_id_arg(target_arg_str: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
            if not target_arg_str: return None, None, None
            parts = target_arg_str.split('-', 1)
            try:
                capsule_num_id = int(parts[0])
                msg_idx_str = parts[1] if len(parts) > 1 else None
                # Use the class's lineage and capsule_type context for generating full ID
                _, cap_full_id_prefix = capsule_manager._get_capsule_file_prefix_and_full_id(capsule_type, capsule_num_id, lineage) # type: ignore
                full_msg_id = f"{cap_full_id_prefix}-{msg_idx_str}" if msg_idx_str else None
                return capsule_num_id, msg_idx_str, full_msg_id
            except ValueError: 
                return None, None, None

        # Helper to parse ranges like "1:5" or "1-2:1-6" and expand them
        def parse_range_arg(range_arg_str: str) -> Tuple[List[str], Optional[str]]:
            """
            Parses range arguments and returns expanded list of IDs.
            Returns: (list_of_ids, error_message)
            
            Examples:
            - "1:5" -> ["1", "2", "3", "4", "5"]
            - "1-2:1-6" -> ["1-2", "1-3", "1-4", "1-5", "1-6"]
            - "1-4:4-2" -> [], "Range should be cross message or cross capsule, not mixture of both"
            """
            if ':' not in range_arg_str:
                return [range_arg_str], None
            
            try:
                start_part, end_part = range_arg_str.split(':', 1)
                start_part = start_part.strip()
                end_part = end_part.strip()
                
                # Early validation for empty parts
                if not start_part or not end_part:
                    raise ValueError("Empty range part")
                
                # Parse start and end parts
                start_has_msg = '-' in start_part
                end_has_msg = '-' in end_part
                
                # Validate message ID format if hyphens are present
                if start_has_msg:
                    start_cap_parts = start_part.split('-', 1)
                    if len(start_cap_parts) != 2 or not start_cap_parts[0] or not start_cap_parts[1]:
                        raise ValueError("Invalid message ID format")
                
                if end_has_msg:
                    end_cap_parts = end_part.split('-', 1)
                    if len(end_cap_parts) != 2 or not end_cap_parts[0] or not end_cap_parts[1]:
                        raise ValueError("Invalid message ID format")
                
                if start_has_msg and end_has_msg:
                    # Both are message IDs (e.g., "1-2:1-6")
                    start_cap_parts = start_part.split('-', 1)
                    end_cap_parts = end_part.split('-', 1)
                    
                    start_cap = int(start_cap_parts[0])
                    start_msg = int(start_cap_parts[1])
                    end_cap = int(end_cap_parts[0])
                    end_msg = int(end_cap_parts[1])
                    
                    # Check for mixed range types
                    if start_cap != end_cap:
                        return [], "Range should be cross message or cross capsule, not mixture of both"
                    
                    # Same capsule, range across messages
                    if start_msg > end_msg:
                        return [], f"Invalid range: {start_msg} > {end_msg} (start > end)"
                    
                    result = []
                    for msg_num in range(start_msg, end_msg + 1):
                        result.append(f"{start_cap}-{msg_num}")
                    return result, None
                    
                elif not start_has_msg and not end_has_msg:
                    # Both are capsule IDs (e.g., "1:5")
                    start_cap = int(start_part)
                    end_cap = int(end_part)
                    
                    if start_cap > end_cap:
                        return [], f"Invalid range: {start_cap} > {end_cap} (start > end)"
                    
                    result = []
                    for cap_num in range(start_cap, end_cap + 1):
                        result.append(str(cap_num))
                    return result, None
                    
                else:
                    # Mixed types: one has message, one doesn't
                    return [], "Range should be cross message or cross capsule, not mixture of both"
                    
            except ValueError:
                return [], f"Invalid range format: '{range_arg_str}'"

        # --- CREATE ---
        if action_command == consts.ACTION_CAPSULE_CREATE:
            # ... (logic for create remains as previously defined)
            if not self._check_action_permission(action_command, agent_data, room_context):
                actions_executed.append(f"Permission denied to {action_command} in this room."); return actions_executed, None
            if not yaml_data: actions_executed.append(f"Action '{action_command}' requires YAML data."); return actions_executed, None
            required_fields = [consts.YAML_CAPSULE_TITLE, consts.YAML_CAPSULE_CONTENT]; required_fields.extend(self._get_additional_yaml_fields_for_create())
            missing_fields = [field for field in required_fields if field not in yaml_data or not yaml_data[field]]
            if missing_fields: actions_executed.append(f"Missing YAML fields for create: {', '.join(missing_fields)}"); return actions_executed, None
            
            numeric_id, new_capsule = capsule_manager.create_capsule(yaml_data, capsule_type, agent_data, current_tick, lineage) # type: ignore
            if new_capsule:
                new_capsule_full_id = new_capsule[consts.CAPSULE_ID_KEY]
                new_capsule_title = new_capsule.get(consts.CAPSULE_TITLE_KEY, "Untitled")
                actions_executed.append(f"Successfully created capsule #{numeric_id} ('{new_capsule_title}'). Full ID: {new_capsule_full_id}")
                self._set_agent_read_status(agent_data, new_capsule_full_id, True, room_context)
                if new_capsule[consts.CAPSULE_MESSAGES_KEY]:
                    first_msg_data = new_capsule[consts.CAPSULE_MESSAGES_KEY][0]
                    first_msg_id = first_msg_data[consts.MESSAGE_ID_KEY]
                    self._set_agent_read_status(agent_data, first_msg_id, True, room_context)
                    # Mention processing is handled by room-specific _after_capsule_created hooks
                self._after_capsule_created(new_capsule, agent_data, room_context, current_tick) 
            else: actions_executed.append("Failed to create capsule.")
            return actions_executed, None

        # --- REPLY ---
        elif action_command == consts.ACTION_CAPSULE_REPLY:
            # ... (logic for reply remains as previously defined) ...
            if not action_args : actions_executed.append(f"Usage: /execute_action{{{consts.ACTION_CAPSULE_REPLY} [capsule_numeric_id]}} with YAML."); return actions_executed, None
            # Reply targets a single capsule ID, not a message ID directly in args
            try:
                target_numeric_id_for_reply = int(action_args.split('-')[0]) # Take only capsule part if "X-Y" is given by mistake
            except ValueError:
                actions_executed.append(f"Invalid capsule ID for reply: {action_args}"); return actions_executed, None

            if not yaml_data or consts.YAML_CAPSULE_CONTENT not in yaml_data: actions_executed.append("Reply requires YAML data with 'content'."); return actions_executed, None

            target_capsule_for_perm_check = capsule_manager.get_capsule(target_numeric_id_for_reply, capsule_type, lineage, include_deleted_capsule=True) # type: ignore
            if target_capsule_for_perm_check is None: # Target capsule must exist
                print(f"DEBUG: get_capsule params: id={target_numeric_id_for_reply}, type='{capsule_type}', lineage='{lineage}'")
                actions_executed.append(f"Unable to find capsule #{target_numeric_id_for_reply}."); return actions_executed, None
            if not self._check_action_permission(action_command, agent_data, room_context, target_capsule_for_perm_check, target_numeric_id=target_numeric_id_for_reply):
                actions_executed.append(f"Permission denied or capsule #{target_numeric_id_for_reply} not found/suitable for reply."); return actions_executed, None
            if target_capsule_for_perm_check.get(consts.CAPSULE_IS_DELETED_KEY): actions_executed.append(f"Cannot reply to deleted capsule #{target_numeric_id_for_reply}."); return actions_executed, None

            success = capsule_manager.add_message_to_capsule(target_numeric_id_for_reply, capsule_type, yaml_data, agent_data, current_tick, lineage) # type: ignore
            if success:
                updated_target_capsule = capsule_manager.get_capsule(target_numeric_id_for_reply, capsule_type, lineage, include_deleted_messages=True) # type: ignore
                if updated_target_capsule and updated_target_capsule[consts.CAPSULE_MESSAGES_KEY]:
                    newly_added_msg_data = updated_target_capsule[consts.CAPSULE_MESSAGES_KEY][-1]
                    msg_id_str = newly_added_msg_data[consts.MESSAGE_ID_KEY]
                    actions_executed.append(f"Replied to capsule #{target_numeric_id_for_reply}. Your message ID is {msg_id_str}.")
                    self._set_agent_read_status(agent_data, msg_id_str, True, room_context)
                    # Mention processing is handled by room-specific _after_reply_added hooks
                    self._after_reply_added(target_capsule_for_perm_check, newly_added_msg_data, agent_data, room_context, current_tick) 
                else: actions_executed.append(f"Replied to capsule #{target_numeric_id_for_reply}, but could not confirm message details.")
            else: actions_executed.append(f"Failed to reply to capsule #{target_numeric_id_for_reply}.")
            return actions_executed, None

        # --- PREVIEW ---
        elif action_command == consts.ACTION_CAPSULE_PREVIEW:
            if not action_args:
                actions_executed.append(f"Usage: /execute_action{{{consts.ACTION_CAPSULE_PREVIEW} <id1,id2,... or all>}}")
                return actions_executed, None

            # Handle "all" case
            if action_args.strip().lower() == "all":
                # Get all capsules
                current_agent_read_status = self._get_agent_read_status(agent_data, room_context)
                all_capsules = capsule_manager.list_capsules(
                    capsule_type, 
                    lineage, 
                    agent_read_status=current_agent_read_status
                )
                
                if not all_capsules:
                    actions_executed.append("No capsules found to preview.")
                    return actions_executed, None
                
                # Sort capsules in ascending order (oldest first) for preview all
                all_capsules.sort(key=lambda x: x.get(consts.CAPSULE_CREATED_AT_TICK_KEY, 0))
                
                all_preview_parts = []
                any_successful_preview = False
                for capsule_metadata in all_capsules:
                    capsule_id_str = capsule_metadata.get(consts.CAPSULE_ID_KEY)
                    if not capsule_id_str:
                        continue
                    
                    # Extract numeric ID from full capsule ID (e.g., "archive_1" -> "1")
                    numeric_id = None
                    try:
                        match = re.search(r'(\d+)$', capsule_id_str)
                        if match:
                            numeric_id = int(match.group(1))
                    except Exception:
                        pass
                    
                    if numeric_id and self._check_action_permission(action_command, agent_data, room_context, capsule_metadata, target_numeric_id=numeric_id):
                        title = capsule_metadata.get(consts.CAPSULE_TITLE_KEY, "Untitled")
                        author = capsule_metadata.get(consts.CAPSULE_AUTHOR_NAME_KEY, "Unknown")
                        created_tick = capsule_metadata.get(consts.CAPSULE_CREATED_AT_TICK_KEY, "N/A")
                        abstract = capsule_metadata.get(consts.CAPSULE_ABSTRACT_KEY)
                        
                        preview_str = f"**Preview for Capsule #{numeric_id}: {title}**\n" \
                                      f"Author: {author}, Created at Tick: {created_tick}\n"
                        if abstract:
                            preview_str += f"Abstract: {abstract}"
                        else:
                            preview_str += "(No abstract available for this capsule.)"
                        all_preview_parts.append(preview_str)
                        any_successful_preview = True
                
                if all_preview_parts:
                    room_context.agent_manager.add_pending_notification(agent_data, "\n\n---\n\n".join(all_preview_parts)) # type: ignore
                actions_executed.append(f"Preview command processed for all capsules." + 
                                       (" Preview(s) sent to System Messages." if any_successful_preview else " No accessible capsules found."))
                return actions_executed, None

            # Parse individual IDs and ranges
            target_capsule_id_strs = [s.strip() for s in action_args.split(',') if s.strip()]
            if not target_capsule_id_strs:
                actions_executed.append("No valid capsule IDs provided for preview.")
                return actions_executed, None

            # Expand ranges into individual IDs
            expanded_capsule_ids = []
            for id_or_range in target_capsule_id_strs:
                expanded_ids, error_msg = parse_range_arg(id_or_range)
                if error_msg:
                    actions_executed.append(f"Preview error: {error_msg}")
                    return actions_executed, None
                
                # For preview, validate that expanded IDs are capsule-only (no message parts)
                for expanded_id in expanded_ids:
                    if '-' in expanded_id:
                        actions_executed.append(f"Preview error: Message-specific IDs not supported for preview: '{expanded_id}'. Use capsule IDs only.")
                        return actions_executed, None
                    expanded_capsule_ids.append(expanded_id)

            all_preview_parts = []
            processed_ids_for_log = []
            any_successful_preview = False

            for cap_id_str in expanded_capsule_ids:
                try:
                    target_cap_num_id = int(cap_id_str)
                except ValueError:
                    all_preview_parts.append(f"Skipping invalid capsule ID for preview: '{cap_id_str}'.")
                    processed_ids_for_log.append(f"{cap_id_str} (invalid)")
                    continue
                
                # For preview, we fetch metadata which includes the abstract
                # Permission check is implicitly for read-like access
                capsule_metadata = capsule_manager.get_capsule_metadata(target_cap_num_id, capsule_type, lineage, agent_read_status=None) # type: ignore

                if not self._check_action_permission(action_command, agent_data, room_context, capsule_metadata, target_numeric_id=target_cap_num_id): # type: ignore
                    all_preview_parts.append(f"Preview failed: Permission denied or capsule #{target_cap_num_id} not found.")
                    processed_ids_for_log.append(f"{cap_id_str} (no access/found)")
                    continue

                if capsule_metadata:
                    title = capsule_metadata.get(consts.CAPSULE_TITLE_KEY, "Untitled")
                    author = capsule_metadata.get(consts.CAPSULE_AUTHOR_NAME_KEY, "Unknown")
                    created_tick = capsule_metadata.get(consts.CAPSULE_CREATED_AT_TICK_KEY, "N/A")
                    abstract = capsule_metadata.get(consts.CAPSULE_ABSTRACT_KEY)

                    preview_str = f"**Preview for Capsule #{target_cap_num_id}: {title}**\n" \
                                  f"Author: {author}, Created at Tick: {created_tick}\n"
                    if abstract:
                        preview_str += f"Abstract: {abstract}"
                    else:
                        preview_str += "(No abstract available for this capsule.)"
                    all_preview_parts.append(preview_str)
                    any_successful_preview = True
                else: # Should be caught by permission check if not found
                    all_preview_parts.append(f"Could not retrieve metadata for capsule #{target_cap_num_id}.")
                processed_ids_for_log.append(cap_id_str)
            
            if all_preview_parts:
                room_context.agent_manager.add_pending_notification(agent_data, "\n\n---\n\n".join(all_preview_parts)) # type: ignore
            
            if processed_ids_for_log:
                actions_executed.append(f"Preview command processed for: {', '.join(processed_ids_for_log)}." +
                                        (" Preview(s) sent to System Messages." if any_successful_preview else " No previews generated."))
            elif not actions_executed:
                actions_executed.append("Preview command could not be processed for the given IDs.")
            return actions_executed, None

        # --- READ ---
        elif action_command == consts.ACTION_CAPSULE_READ:
            if not action_args:
                actions_executed.append(f"Usage: /execute_action{{{consts.ACTION_CAPSULE_READ} <id1,id2-msg,...>}}")
                return actions_executed, None

            target_id_arg_strings = [s.strip() for s in action_args.split(',') if s.strip()]
            if not target_id_arg_strings:
                actions_executed.append("No valid IDs provided for read.")
                return actions_executed, None

            # Expand ranges into individual IDs
            expanded_target_ids = []
            for id_or_range in target_id_arg_strings:
                expanded_ids, error_msg = parse_range_arg(id_or_range)
                if error_msg:
                    actions_executed.append(f"Read error: {error_msg}")
                    return actions_executed, None
                expanded_target_ids.extend(expanded_ids)

            all_content_parts = []
            all_items_marked_read_this_turn = set() 
            processed_ids_for_log = []
            any_content_generated_or_status_shown = False

            for single_target_arg_str in expanded_target_ids:
                current_item_content_parts = []
                target_cap_num_id, target_msg_idx_str, target_full_msg_id_str_opt = parse_single_target_id_arg(single_target_arg_str)

                if target_cap_num_id is None:
                    current_item_content_parts.append(f"Invalid ID format: '{single_target_arg_str}'.")
                    all_content_parts.extend(current_item_content_parts)
                    processed_ids_for_log.append(f"{single_target_arg_str} (invalid)")
                    any_content_generated_or_status_shown = True
                    continue

                agent_read_statuses = self._get_agent_read_status(agent_data, room_context)
                is_reading_specific_message = bool(target_full_msg_id_str_opt)
                _, cap_full_id_str_for_capsule_itself = capsule_manager._get_capsule_file_prefix_and_full_id(capsule_type, target_cap_num_id, lineage) # type: ignore
                
                # MODIFICATION Point for Read Logic
                if is_reading_specific_message:
                    # --- Reading a specific message ---
                    if target_full_msg_id_str_opt and agent_read_statuses.get(target_full_msg_id_str_opt, False):
                        already_read_message = (
                            f"Message '{single_target_arg_str}' has already been read. "
                            f"Use `/execute_action{{{consts.ACTION_CAPSULE_UNREAD} {single_target_arg_str}}}` to mark it as unread if you wish to view it again."
                        )
                        current_item_content_parts.append(already_read_message)
                        any_content_generated_or_status_shown = True
                    else:
                        # Fetch capsule to get the specific message
                        capsule_to_read = capsule_manager.get_capsule(target_cap_num_id, capsule_type, lineage, include_deleted_messages=False) # type: ignore
                        if not self._check_action_permission(action_command, agent_data, room_context, capsule_to_read, target_numeric_id=target_cap_num_id):
                            current_item_content_parts.append(f"Read failed: Permission denied or item '{single_target_arg_str}' not found.")
                            any_content_generated_or_status_shown = True
                        elif capsule_to_read and target_full_msg_id_str_opt:
                            found_msg = next((m for m in capsule_to_read.get(consts.CAPSULE_MESSAGES_KEY,[]) if m[consts.MESSAGE_ID_KEY] == target_full_msg_id_str_opt), None)
                            if found_msg:
                                current_item_content_parts.append(f"**Message {target_full_msg_id_str_opt} from Capsule '{capsule_to_read[consts.CAPSULE_TITLE_KEY]}':**")
                                current_item_content_parts.append(f"Author: {found_msg[consts.MESSAGE_AUTHOR_NAME_KEY]} (Tick {found_msg[consts.MESSAGE_POSTED_AT_TICK_KEY]})")
                                if found_msg.get(consts.MESSAGE_TITLE_KEY): current_item_content_parts.append(f"Title: {found_msg[consts.MESSAGE_TITLE_KEY]}")
                                current_item_content_parts.append(f"Content:\n{found_msg[consts.MESSAGE_CONTENT_KEY]}")
                                all_items_marked_read_this_turn.add(target_full_msg_id_str_opt)
                                any_content_generated_or_status_shown = True
                            else: 
                                current_item_content_parts.append(f"Message {target_full_msg_id_str_opt} (target: '{single_target_arg_str}') not found in capsule #{target_cap_num_id}.")
                                any_content_generated_or_status_shown = True
                        else: # Capsule not found (should be caught by permission check)
                            current_item_content_parts.append(f"Capsule #{target_cap_num_id} not found for message '{single_target_arg_str}'.")
                            any_content_generated_or_status_shown = True
                else:
                    # --- Reading a whole capsule ---
                    # No early exit based on capsule_id's read status. Always proceed to show message states.
                    capsule_to_read = capsule_manager.get_capsule(target_cap_num_id, capsule_type, lineage, include_deleted_messages=False) # type: ignore

                    if not self._check_action_permission(action_command, agent_data, room_context, capsule_to_read, target_numeric_id=target_cap_num_id):
                        current_item_content_parts.append(f"Read failed: Permission denied or capsule '{single_target_arg_str}' not found.")
                        any_content_generated_or_status_shown = True
                    elif capsule_to_read:
                        current_item_content_parts.append(f"**Capsule #{target_cap_num_id}: {capsule_to_read[consts.CAPSULE_TITLE_KEY]}**")
                        current_item_content_parts.append(f"Author: {capsule_to_read[consts.CAPSULE_AUTHOR_NAME_KEY]}, Created at Tick: {capsule_to_read[consts.CAPSULE_CREATED_AT_TICK_KEY]}")
                        if capsule_to_read.get(consts.CAPSULE_ABSTRACT_KEY): current_item_content_parts.append(f"Abstract: {capsule_to_read[consts.CAPSULE_ABSTRACT_KEY]}")
                        
                        messages = capsule_to_read.get(consts.CAPSULE_MESSAGES_KEY, [])
                        if not messages: 
                            current_item_content_parts.append("(This capsule has no messages.)")
                        else:
                            current_item_content_parts.append("\n**Messages:**")
                        
                        for msg in messages:
                            msg_id_full_internal = msg[consts.MESSAGE_ID_KEY]
                            msg_index_str_part = msg_id_full_internal.split('-')[-1]
                            user_friendly_msg_id_for_unread = f"{target_cap_num_id}-{msg_index_str_part}"

                            if agent_read_statuses.get(msg_id_full_internal, False):
                                current_item_content_parts.append(f"\n---\n**Message {msg_id_full_internal}** (already read. Use `/execute_action{{{consts.ACTION_CAPSULE_UNREAD} {user_friendly_msg_id_for_unread}}}` to show again.)")
                            else:
                                current_item_content_parts.append(f"\n---\n**Message {msg_id_full_internal}**\nAuthor: {msg[consts.MESSAGE_AUTHOR_NAME_KEY]} (Tick {msg[consts.MESSAGE_POSTED_AT_TICK_KEY]})")
                                if msg.get(consts.MESSAGE_TITLE_KEY): current_item_content_parts.append(f"Title: {msg[consts.MESSAGE_TITLE_KEY]}")
                                current_item_content_parts.append(f"Content:\n{msg[consts.MESSAGE_CONTENT_KEY]}")
                                all_items_marked_read_this_turn.add(msg_id_full_internal) # Mark NEWLY displayed message as read
                        
                        all_items_marked_read_this_turn.add(cap_full_id_str_for_capsule_itself) # Mark entire capsule as read
                        any_content_generated_or_status_shown = True
                    else: # Should be caught by permission check
                        current_item_content_parts.append(f"Capsule '{single_target_arg_str}' not found.")
                        any_content_generated_or_status_shown = True
                
                all_content_parts.extend(current_item_content_parts)
                processed_ids_for_log.append(single_target_arg_str)
            # End of loop for single_target_arg_str

            if all_content_parts:
                room_context.agent_manager.add_pending_notification(agent_data, "\n\n---\n\n".join(all_content_parts)) # type: ignore
            
            for item_id in all_items_marked_read_this_turn: 
                self._set_agent_read_status(agent_data, item_id, True, room_context)
            
            if processed_ids_for_log:
                actions_executed.append(f"Read command processed for: {', '.join(processed_ids_for_log)}." +
                                        (" Content/status sent to System Messages." if any_content_generated_or_status_shown else ""))
            elif not actions_executed:
                actions_executed.append("Read command could not be processed for the given IDs or no content to display.")
            return actions_executed, None

        # --- UPDATE ---
        elif action_command == consts.ACTION_CAPSULE_UPDATE:
            # ... (logic for update remains as previously defined) ...
            if not yaml_data: actions_executed.append("Update action requires YAML data."); return actions_executed, None
            # Use parse_single_target_id_arg for consistency, though update usually targets one item
            target_cap_num_id, _, target_full_msg_id_str_opt = parse_single_target_id_arg(action_args or "") 
            if target_cap_num_id is None : actions_executed.append(f"Invalid ID for update: '{action_args}'."); return actions_executed, None

            id_prefix_lineage = lineage 
            id_to_check_ownership = target_full_msg_id_str_opt if target_full_msg_id_str_opt else capsule_manager._get_capsule_file_prefix_and_full_id(capsule_type, target_cap_num_id, id_prefix_lineage)[1] # type: ignore
            
            cap_for_op = capsule_manager.get_capsule(target_cap_num_id, capsule_type, lineage, include_deleted_capsule=True, include_deleted_messages=True) # type: ignore
            if not self._check_action_permission(action_command, agent_data, room_context, cap_for_op, target_id_str=id_to_check_ownership, target_numeric_id=target_cap_num_id):
                actions_executed.append(f"Permission denied or item '{action_args}' not found for {action_command}."); return actions_executed, None

            success = False
            updated_message_content_for_mention = None
            message_id_for_mention_check = None
            capsule_title_for_mention_check = cap_for_op.get(consts.CAPSULE_TITLE_KEY, f"Capsule #{target_cap_num_id}") if cap_for_op else f"Capsule #{target_cap_num_id}"

            if target_full_msg_id_str_opt: 
                success = capsule_manager.update_message_content(target_cap_num_id, capsule_type, target_full_msg_id_str_opt, yaml_data, current_tick, lineage, room_context) # type: ignore
                if success and consts.YAML_CAPSULE_CONTENT in yaml_data:
                    updated_message_content_for_mention = str(yaml_data[consts.YAML_CAPSULE_CONTENT])
                    message_id_for_mention_check = target_full_msg_id_str_opt
            else: 
                success = capsule_manager.update_capsule_metadata(target_cap_num_id, capsule_type, yaml_data, current_tick, lineage, room_context) # type: ignore
            
            actions_executed.append(f"Item '{action_args}' {'updated' if success else 'update failed'}.")
            # Mention processing for updates would be handled by room-specific logic if needed
            return actions_executed, None
            
        # --- UNREAD ---
        elif action_command == consts.ACTION_CAPSULE_UNREAD:
            # ... (logic for unread remains as previously defined, ensure parse_single_target_id_arg is used if action_args can be comma-separated for unread)
            # Spec for unread says "ids" - implies comma separated is possible.
            if not action_args: 
                actions_executed.append(f"Usage: /execute_action{{{consts.ACTION_CAPSULE_UNREAD} <id1,id2-msg,...>}}")
                return actions_executed, None

            target_id_arg_strings_unread = [s.strip() for s in action_args.split(',') if s.strip()]
            if not target_id_arg_strings_unread:
                actions_executed.append("No valid IDs provided for unread.")
                return actions_executed, None
            
            # Expand ranges into individual IDs (same as read action)
            expanded_unread_ids = []
            for id_or_range in target_id_arg_strings_unread:
                expanded_ids, error_msg = parse_range_arg(id_or_range)
                if error_msg:
                    actions_executed.append(f"Unread error: {error_msg}")
                    return actions_executed, None
                expanded_unread_ids.extend(expanded_ids)
            
            processed_ids_for_unread_log = []
            any_successful_unread = False

            for single_target_arg_str_unread in expanded_unread_ids:
                target_cap_num_id, _, target_full_msg_id_str_opt = parse_single_target_id_arg(single_target_arg_str_unread)
                
                if target_cap_num_id is None: 
                    actions_executed.append(f"Invalid ID format for unread: '{single_target_arg_str_unread}'.")
                    processed_ids_for_unread_log.append(f"{single_target_arg_str_unread} (invalid)")
                    continue

                cap_to_check_for_unread = capsule_manager.get_capsule(target_cap_num_id, capsule_type, lineage, include_deleted_capsule=True) # type: ignore
                # Permission to unread: can you "see" it enough to know its ID? Read perm is proxy.
                # The item_to_check_perm is either the message ID or the capsule ID.
                item_id_for_perm_check = target_full_msg_id_str_opt if target_full_msg_id_str_opt \
                                         else (cap_to_check_for_unread.get(consts.CAPSULE_ID_KEY) if cap_to_check_for_unread else None)

                if not self._check_action_permission(consts.ACTION_CAPSULE_READ, agent_data, room_context, cap_to_check_for_unread, target_id_str=item_id_for_perm_check, target_numeric_id=target_cap_num_id):
                    actions_executed.append(f"Cannot unread: Item '{single_target_arg_str_unread}' not found or no permission to access."); 
                    processed_ids_for_unread_log.append(f"{single_target_arg_str_unread} (no access/found)")
                    continue
                
                if target_full_msg_id_str_opt: # Unreading a specific message
                    self._set_agent_read_status(agent_data, target_full_msg_id_str_opt, False, room_context)
                    if cap_to_check_for_unread: # Also unmark parent capsule
                         self._set_agent_read_status(agent_data, cap_to_check_for_unread[consts.CAPSULE_ID_KEY], False, room_context) # type: ignore
                    actions_executed.append(f"Message '{single_target_arg_str_unread}' marked as unread.")
                    any_successful_unread = True
                elif cap_to_check_for_unread : # Unreading a whole capsule
                    self._set_agent_read_status(agent_data, cap_to_check_for_unread[consts.CAPSULE_ID_KEY], False, room_context) # type: ignore
                    for msg in cap_to_check_for_unread.get(consts.CAPSULE_MESSAGES_KEY, []): # type: ignore
                        self._set_agent_read_status(agent_data, msg[consts.MESSAGE_ID_KEY], False, room_context)
                    actions_executed.append(f"Capsule '{single_target_arg_str_unread}' and its messages marked as unread.")
                    any_successful_unread = True
                else:  
                    actions_executed.append(f"Could not process unread for '{single_target_arg_str_unread}'.")
                processed_ids_for_unread_log.append(single_target_arg_str_unread)

            if not actions_executed and processed_ids_for_unread_log: # If only invalid items were processed by loop
                 actions_executed.append(f"Unread command processed for: {', '.join(processed_ids_for_unread_log)}." +
                                        (" Status updated." if any_successful_unread else " No items were unread."))
            elif not actions_executed: # Fallback if loop was empty
                 actions_executed.append("Unread command received no valid items to process.")

            return actions_executed, None

        # --- PIN, UNPIN, PAGE, DELETE, SEARCH ---
        # These generally target one item or a page, or a single search term.
        # If they need to support comma-separated values, their logic would also need similar loops.
        # For now, assume they operate on single `action_args` as per their typical usage.

        elif action_command in [consts.ACTION_CAPSULE_PIN, consts.ACTION_CAPSULE_UNPIN]:
            # Spec: "pin ids", "unpin ids" - implies comma separated
            if not action_args: 
                actions_executed.append(f"Usage: /execute_action{{{action_command} <capsule_id1,capsule_id2,...>}}")
                return actions_executed, None

            target_capsule_id_strs_pin = [s.strip() for s in action_args.split(',') if s.strip()]
            if not target_capsule_id_strs_pin:
                actions_executed.append(f"No valid capsule IDs provided for {action_command}.")
                return actions_executed, None

            pinned_ids = self._get_pinned_capsules_ids(agent_data, room_context).copy()
            changed_pin_status_count = 0

            for cap_id_str_pin in target_capsule_id_strs_pin:
                try:
                    target_numeric_id_pin = int(cap_id_str_pin)
                except ValueError:
                    actions_executed.append(f"Invalid capsule ID '{cap_id_str_pin}' for {action_command}, skipping.")
                    continue
                
                cap_to_pin = capsule_manager.get_capsule(target_numeric_id_pin, capsule_type, lineage) # type: ignore
                
                # Check if capsule exists
                if cap_to_pin is None:
                    actions_executed.append(f"Cannot {action_command} capsule #{target_numeric_id_pin}: Capsule not found.")
                    continue
                
                # Pin/unpin requires ability to "see" the capsule. Read permission is proxy.
                if not self._check_action_permission(consts.ACTION_CAPSULE_READ, agent_data, room_context, cap_to_pin, target_numeric_id=target_numeric_id_pin): 
                    actions_executed.append(f"Cannot {action_command} capsule #{target_numeric_id_pin}: Not found or no permission."); 
                    continue
                
                target_full_id_str = cap_to_pin[consts.CAPSULE_ID_KEY] # type: ignore
                if action_command == consts.ACTION_CAPSULE_PIN:
                    if target_full_id_str not in pinned_ids: 
                        pinned_ids.append(target_full_id_str)
                        actions_executed.append(f"Capsule #{target_numeric_id_pin} pinned.")
                        changed_pin_status_count+=1
                    else: actions_executed.append(f"Capsule #{target_numeric_id_pin} is already pinned.")
                else: # UNPIN
                    if target_full_id_str in pinned_ids: 
                        pinned_ids.remove(target_full_id_str)
                        actions_executed.append(f"Capsule #{target_numeric_id_pin} unpinned.")
                        changed_pin_status_count+=1
                    else: actions_executed.append(f"Capsule #{target_numeric_id_pin} was not pinned.")
            
            if changed_pin_status_count > 0:
                self._set_pinned_capsules_ids(agent_data, pinned_ids, room_context)
            if not actions_executed : actions_executed.append(f"No {action_command} operations performed.") # If all were invalid or no change
            return actions_executed, None

        elif action_command == consts.ACTION_CAPSULE_PAGE:
            # ... (logic for page remains as previously defined) ...
            if not action_args or not action_args.isdigit(): actions_executed.append(f"Usage: /execute_action{{{consts.ACTION_CAPSULE_PAGE} [number]}}"); return actions_executed, None
            page_num = int(action_args);
            if page_num <= 0: actions_executed.append("Page number must be positive."); return actions_executed, None
            self._set_current_page(agent_data, page_num, room_context); actions_executed.append(f"Navigated to page {page_num}.")
            return actions_executed, None

        elif action_command == consts.ACTION_CAPSULE_DELETE:
            # Delete likely targets one item at a time. If multiple, would need a loop.
            # Spec: /execute_action{delete id} or {delete id-msg_id} - singular.
            # ... (logic for delete remains as previously defined) ...
            target_cap_num_id, _, target_full_msg_id_str_opt = parse_single_target_id_arg(action_args or "")
            if target_cap_num_id is None : actions_executed.append(f"Invalid ID for delete: '{action_args}'."); return actions_executed, None
            
            id_prefix_lineage_del = lineage
            id_to_check_ownership_del = target_full_msg_id_str_opt if target_full_msg_id_str_opt else capsule_manager._get_capsule_file_prefix_and_full_id(capsule_type, target_cap_num_id, id_prefix_lineage_del)[1] # type: ignore
            
            cap_for_op_del = capsule_manager.get_capsule(target_cap_num_id, capsule_type, lineage, include_deleted_capsule=True, include_deleted_messages=True) # type: ignore
            if not self._check_action_permission(action_command, agent_data, room_context, cap_for_op_del, target_id_str=id_to_check_ownership_del, target_numeric_id=target_cap_num_id):
                actions_executed.append(f"Permission denied or item '{action_args}' not found for {action_command}."); return actions_executed, None
            success_del = False
            if target_full_msg_id_str_opt: success_del = capsule_manager.delete_message_from_capsule(target_cap_num_id, capsule_type, target_full_msg_id_str_opt, current_tick, lineage) # type: ignore
            else: success_del = capsule_manager.delete_capsule(target_cap_num_id, capsule_type, current_tick, lineage) # type: ignore
            actions_executed.append(f"Item '{action_args}' {'soft deleted' if success_del else 'delete failed'}.")
            return actions_executed, None


        elif action_command == consts.ACTION_CAPSULE_SEARCH:
            if not action_args: 
                actions_executed.append(f"Usage: /execute_action{{{consts.ACTION_CAPSULE_SEARCH} [tag]}}")
                return actions_executed, None

            search_tag = action_args.lower()
            current_agent_read_status = self._get_agent_read_status(agent_data, room_context)
            # list_capsules calls get_capsule_metadata, which includes 'total_message_count' 
            # and 'unread_message_count'
            all_caps_meta_unfiltered = capsule_manager.list_capsules( # type: ignore
                self._get_capsule_type(), 
                self._get_lineage_for_capsule_operations(agent_data, room_context), 
                agent_read_status=current_agent_read_status, 
                tag_filter=search_tag
            )

            # --- MODIFICATION START: Filter search results for mail ---
            if capsule_type == consts.CAPSULE_TYPE_MAIL and agent_name:
                authorized_search_results = []
                for meta in all_caps_meta_unfiltered:
                    is_author = agent_name == meta.get(consts.CAPSULE_AUTHOR_NAME_KEY)
                    is_recipient = agent_name in meta.get(consts.CAPSULE_RECIPIENTS_KEY, [])
                    if is_author or is_recipient:
                        authorized_search_results.append(meta)
                all_caps_meta_filtered = authorized_search_results
            else:
                all_caps_meta_filtered = all_caps_meta_unfiltered

            if all_caps_meta_filtered:
                results_header = f"Search results for tag '{search_tag}':"
                results_lines = []
                for cap in all_caps_meta_filtered:
                    unread_count = cap.get(consts.CAPSULE_UNREAD_MESSAGE_COUNT_KEY, 0)
                    total_messages = cap.get('total_message_count', 0) # From get_capsule_metadata
                    
                    status_text = ""
                    if total_messages == 0:
                        status_text = "(No Msgs)"
                    elif unread_count > 0:
                        status_text = f"({unread_count} unread)"
                    else: # unread_count is 0 and total_messages > 0
                        status_text = "(All Read)"
                    
                    capsule_numeric_id = cap.get(consts.CAPSULE_ID_KEY, "N/A").split('_')[-1]
                    if not capsule_numeric_id.isdigit() and '-' in cap.get(consts.CAPSULE_ID_KEY, ""): #Handles full ID like lineage_private_1
                         id_parts = cap.get(consts.CAPSULE_ID_KEY, "").split('_')
                         if len(id_parts) > 0:
                             potential_num_id = id_parts[-1]
                             if potential_num_id.isdigit():
                                 capsule_numeric_id = potential_num_id


                    results_lines.append(
                        f"- Capsule ID {capsule_numeric_id}: '{cap.get(consts.CAPSULE_TITLE_KEY, 'Untitled')}' {status_text}"
                    )
                
                room_context.agent_manager.add_pending_notification(agent_data, results_header + "\n" + "\n".join(results_lines)) # type: ignore
                actions_executed.append(f"Search results for tag '{search_tag}' sent to your System Messages.")
            else: 
                actions_executed.append(f"No capsules found with tag '{search_tag}'.")
            return actions_executed, None
            
        elif action_command == consts.ACTION_CAPSULE_MUTE:
            if not action_args:
                actions_executed.append("Mute requires a capsule ID. Example: `/execute_action{mute 5}`")
                return actions_executed, None
            
            try:
                capsule_id_to_mute = int(action_args.strip())
            except ValueError:
                actions_executed.append(f"Invalid capsule ID for mute: '{action_args}'. Must be a number.")
                return actions_executed, None
            
            # Check if capsule exists and permission to mute it (handled by room-specific permission check)
            capsule_to_mute = capsule_manager.get_capsule(capsule_id_to_mute, capsule_type, lineage)
            if not capsule_to_mute or not self._check_action_permission(action_command, agent_data, room_context, capsule_to_mute, target_numeric_id=capsule_id_to_mute):
                actions_executed.append(f"Cannot mute: Capsule #{capsule_id_to_mute} not found or mute not available in this room.")
                return actions_executed, None
            
            # Get or create muted capsules dict
            room_key = self._get_agent_room_data_key(room_context)
            muted_capsules = room_context.agent_manager.get_agent_room_state(
                agent_data, room_key, consts.AGENT_ROOM_STATE_MUTED_CAPSULES_KEY, default={}
            )
            
            capsule_full_id = capsule_to_mute[consts.CAPSULE_ID_KEY]
            muted_capsules[capsule_full_id] = True
            
            room_context.agent_manager.set_agent_room_state(
                agent_data, room_key, consts.AGENT_ROOM_STATE_MUTED_CAPSULES_KEY, muted_capsules
            )
            
            capsule_title = capsule_to_mute.get(consts.CAPSULE_TITLE_KEY, f"Capsule #{capsule_id_to_mute}")
            actions_executed.append(f"Muted capsule #{capsule_id_to_mute}: \"{capsule_title}\". You will no longer receive notifications for new replies. Use `/execute_action{{unmute {capsule_id_to_mute}}}` to turn notifications back on.")
            return actions_executed, None
            
        elif action_command == consts.ACTION_CAPSULE_UNMUTE:
            if not action_args:
                actions_executed.append("Unmute requires a capsule ID. Example: `/execute_action{unmute 5}`")
                return actions_executed, None
            
            try:
                capsule_id_to_unmute = int(action_args.strip())
            except ValueError:
                actions_executed.append(f"Invalid capsule ID for unmute: '{action_args}'. Must be a number.")
                return actions_executed, None
            
            # Check if capsule exists and permission to unmute it (handled by room-specific permission check)
            capsule_to_unmute = capsule_manager.get_capsule(capsule_id_to_unmute, capsule_type, lineage)
            if not capsule_to_unmute or not self._check_action_permission(action_command, agent_data, room_context, capsule_to_unmute, target_numeric_id=capsule_id_to_unmute):
                actions_executed.append(f"Cannot unmute: Capsule #{capsule_id_to_unmute} not found or unmute not available in this room.")
                return actions_executed, None
            
            # Get muted capsules dict
            room_key = self._get_agent_room_data_key(room_context)
            muted_capsules = room_context.agent_manager.get_agent_room_state(
                agent_data, room_key, consts.AGENT_ROOM_STATE_MUTED_CAPSULES_KEY, default={}
            )
            
            capsule_full_id = capsule_to_unmute[consts.CAPSULE_ID_KEY]
            
            if capsule_full_id in muted_capsules and muted_capsules[capsule_full_id]:
                muted_capsules[capsule_full_id] = False
                room_context.agent_manager.set_agent_room_state(
                    agent_data, room_key, consts.AGENT_ROOM_STATE_MUTED_CAPSULES_KEY, muted_capsules
                )
                capsule_title = capsule_to_unmute.get(consts.CAPSULE_TITLE_KEY, f"Capsule #{capsule_id_to_unmute}")
                actions_executed.append(f"Unmuted capsule #{capsule_id_to_unmute}: \"{capsule_title}\". You will now receive notifications for new replies.")
            else:
                actions_executed.append(f"Capsule #{capsule_id_to_unmute} was not muted.")
            return actions_executed, None
            
        else: # Fallback for unrecognized actions in this room
            if not actions_executed: # Only if this specific command wasn't handled by a more specific room overriding this.
                actions_executed.append(f"Action '{action_command}' is not recognized or implemented in this room.")
        
        return actions_executed, None