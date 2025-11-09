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

# station/capsule.py
"""
Manages capsule data persistence and operations for various capsule types,
including soft deletes, agent name updates, and supporting display logic
with unread message counts.
"""
import os
import uuid
import re 
import json 
from typing import Any, List, Dict, Optional, Tuple

from . import constants 
from . import file_io_utils


# --- Helper Functions ---
def _calculate_word_count(text: str) -> int:
    # Ensure text is a string before splitting
    return len(str(text).split()) if text is not None else 0

def _process_list_field_from_yaml(raw_value: Any) -> List[str]:
    """
    Processes a field that can be a comma-separated string or a list into a list of strings.
    """
    processed_list = []
    if isinstance(raw_value, str):
        processed_list = [item.strip() for item in raw_value.split(',') if item.strip()]
    elif isinstance(raw_value, list):
        processed_list = [str(item).strip() for item in raw_value if str(item).strip()]
    return processed_list

def _ensure_string_or_none(value: Any) -> Optional[str]:
    """Ensures a value is a string if not None."""
    if value is None:
        return None
    return str(value)

# ... (other helper functions like _generate_message_id, _get_capsule_dir_path, etc. remain the same) ...
def _generate_message_id(capsule_id_str: str, messages_list: List[Dict]) -> str:
    """
    Generates the next message ID for a capsule (e.g., capsule_id-1, capsule_id-2).
    It finds the highest existing numeric suffix for messages associated with this capsule_id_str.
    """
    max_num = 0
    if not messages_list: # Handle case of first message
        return f"{capsule_id_str}-1"

    for msg in messages_list:
        msg_id = msg.get(constants.MESSAGE_ID_KEY, "")
        if msg_id.startswith(capsule_id_str + "-"):
            try:
                num_part_str = msg_id.split('-')[-1]
                if num_part_str.isdigit(): # Ensure it's a digit before int conversion
                    num_part = int(num_part_str)
                    if num_part > max_num:
                        max_num = num_part
            except ValueError:
                pass 
    return f"{capsule_id_str}-{max_num + 1}" # Next ID is max_num + 1


def _get_capsule_dir_path(capsule_type: str, lineage_name: Optional[str] = None) -> str:
    base_capsules_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.CAPSULES_DIR_NAME)
    if capsule_type == constants.CAPSULE_TYPE_PUBLIC:
        return os.path.join(base_capsules_path, constants.PUBLIC_CAPSULES_SUBDIR_NAME)
    elif capsule_type == constants.CAPSULE_TYPE_MAIL:
        return os.path.join(base_capsules_path, constants.MAIL_CAPSULES_SUBDIR_NAME)
    elif capsule_type == constants.CAPSULE_TYPE_ARCHIVE:
        return os.path.join(base_capsules_path, constants.ARCHIVE_CAPSULES_SUBDIR_NAME)
    elif capsule_type == constants.CAPSULE_TYPE_PRIVATE:
        if not lineage_name: raise ValueError("Lineage name required for private capsules.")
        safe_lineage_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in lineage_name)
        return os.path.join(base_capsules_path, constants.PRIVATE_CAPSULES_SUBDIR_NAME, f"lineage_{safe_lineage_name}")
    else: raise ValueError(f"Unknown capsule type: {capsule_type}")

def _get_capsule_file_prefix_and_full_id(capsule_type: str, numeric_id: int, lineage_name: Optional[str] = None) -> Tuple[str, str]:
    if capsule_type == constants.CAPSULE_TYPE_PUBLIC: prefix = "public_"
    elif capsule_type == constants.CAPSULE_TYPE_MAIL: prefix = "mail_"
    elif capsule_type == constants.CAPSULE_TYPE_ARCHIVE: prefix = "archive_"
    elif capsule_type == constants.CAPSULE_TYPE_PRIVATE:
        if not lineage_name: raise ValueError("Lineage name required for private capsule ID prefix.")
        safe_lineage_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in lineage_name)
        prefix = f"{safe_lineage_name}_private_"
    else: raise ValueError(f"Unknown capsule type for prefix: {capsule_type}")
    full_id = f"{prefix}{numeric_id}"
    return prefix, full_id

def _get_capsule_path(capsule_type: str, numeric_id: int, lineage_name: Optional[str] = None) -> str:
    dir_path = _get_capsule_dir_path(capsule_type, lineage_name)
    _, full_capsule_id_str = _get_capsule_file_prefix_and_full_id(capsule_type, numeric_id, lineage_name)
    filename = f"{full_capsule_id_str}{constants.YAML_EXTENSION}"
    return os.path.join(dir_path, filename)


# --- Public API ---

def get_next_capsule_id(capsule_type: str, lineage_name: Optional[str] = None) -> int:
    # ... (no changes) ...
    dir_path = _get_capsule_dir_path(capsule_type, lineage_name)
    file_io_utils.ensure_dir_exists(dir_path)
    prefix, _ = _get_capsule_file_prefix_and_full_id(capsule_type, 1, lineage_name) # Pass 1 as dummy ID
    return file_io_utils.get_next_sequential_id(dir_path, prefix, constants.YAML_EXTENSION)


def create_capsule(capsule_content_from_agent: Dict[str, Any], 
                   capsule_type: str, 
                   author_agent_data: Dict[str, Any], 
                   current_tick: int, 
                   lineage_for_private: Optional[str] = None) -> Tuple[int, Optional[Dict[str, Any]]]:
    try:
        new_numeric_id = get_next_capsule_id(capsule_type, lineage_for_private)
        _, full_capsule_id_str = _get_capsule_file_prefix_and_full_id(capsule_type, new_numeric_id, lineage_for_private)
        
        # MODIFICATION: Lenient parsing for content, title, abstract, tags, recipients
        initial_content_raw = capsule_content_from_agent.get(constants.YAML_CAPSULE_CONTENT)
        initial_content = _ensure_string_or_none(initial_content_raw) or "" # Default to empty string if None after str()
        
        capsule_title_raw = capsule_content_from_agent.get(constants.YAML_CAPSULE_TITLE)
        capsule_title = _ensure_string_or_none(capsule_title_raw)

        capsule_abstract_raw = capsule_content_from_agent.get(constants.YAML_CAPSULE_ABSTRACT)
        capsule_abstract = _ensure_string_or_none(capsule_abstract_raw)

        tags_raw = capsule_content_from_agent.get(constants.YAML_CAPSULE_TAGS)
        processed_tags = _process_list_field_from_yaml(tags_raw)
        
        initial_word_count = _calculate_word_count(initial_content)
        
        first_message_id = _generate_message_id(full_capsule_id_str, [])

        first_message = {
            constants.MESSAGE_ID_KEY: first_message_id, 
            constants.MESSAGE_AUTHOR_NAME_KEY: author_agent_data.get(constants.AGENT_NAME_KEY),
            constants.MESSAGE_AUTHOR_LINEAGE_KEY: author_agent_data.get(constants.AGENT_LINEAGE_KEY),
            constants.MESSAGE_AUTHOR_GENERATION_KEY: author_agent_data.get(constants.AGENT_GENERATION_KEY),
            constants.MESSAGE_POSTED_AT_TICK_KEY: current_tick,
            constants.MESSAGE_TITLE_KEY: capsule_title, # Use processed title (can be None if initial msg uses capsule title)
            constants.MESSAGE_CONTENT_KEY: initial_content, # Use processed content
            constants.MESSAGE_WORD_COUNT_KEY: initial_word_count,
            constants.MESSAGE_IS_DELETED_KEY: False,
        }
        capsule_data = {
            constants.CAPSULE_ID_KEY: full_capsule_id_str, constants.CAPSULE_TYPE_KEY: capsule_type,
            constants.CAPSULE_AUTHOR_NAME_KEY: author_agent_data.get(constants.AGENT_NAME_KEY),
            constants.CAPSULE_AUTHOR_LINEAGE_KEY: author_agent_data.get(constants.AGENT_LINEAGE_KEY),
            constants.CAPSULE_AUTHOR_GENERATION_KEY: author_agent_data.get(constants.AGENT_GENERATION_KEY),
            constants.CAPSULE_CREATED_AT_TICK_KEY: current_tick, constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY: current_tick,
            constants.CAPSULE_TITLE_KEY: capsule_title, # Use processed title
            constants.CAPSULE_TAGS_KEY: processed_tags, # Use processed tags
            constants.CAPSULE_ABSTRACT_KEY: capsule_abstract, # Use processed abstract
            constants.CAPSULE_WORD_COUNT_TOTAL_KEY: initial_word_count, constants.CAPSULE_MESSAGES_KEY: [first_message],
            constants.CAPSULE_IS_DELETED_KEY: False,
        }
        if capsule_type == constants.CAPSULE_TYPE_MAIL:
            recipients_raw = capsule_content_from_agent.get(constants.YAML_CAPSULE_RECIPIENTS)
            capsule_data[constants.CAPSULE_RECIPIENTS_KEY] = _process_list_field_from_yaml(recipients_raw)
        
        if capsule_type == constants.CAPSULE_TYPE_PRIVATE and lineage_for_private:
            capsule_data[constants.CAPSULE_LINEAGE_ASSOCIATION_KEY] = lineage_for_private
            
        file_path = _get_capsule_path(capsule_type, new_numeric_id, lineage_for_private)
        file_io_utils.save_yaml(capsule_data, file_path)
        return new_numeric_id, capsule_data
    except Exception as e: print(f"Error creating capsule: {e}"); return 0, None

def add_message_to_capsule(numeric_id: int, 
                           capsule_type: str, 
                           message_content_from_agent: Dict[str, Any], 
                           author_agent_data: Dict[str, Any], 
                           current_tick: int, 
                           lineage_name: Optional[str] = None) -> bool:
    capsule_data = get_capsule(numeric_id, capsule_type, lineage_name, include_deleted_capsule=True, include_deleted_messages=True) 
    if not capsule_data: print(f"Cannot add message: Capsule {numeric_id} ({capsule_type}) not found."); return False
    if capsule_data.get(constants.CAPSULE_IS_DELETED_KEY, False): print(f"Cannot add message: Capsule {numeric_id} ({capsule_type}) is marked as deleted."); return False
    try:
        capsule_id_str = capsule_data[constants.CAPSULE_ID_KEY]
        messages_list = capsule_data.get(constants.CAPSULE_MESSAGES_KEY, []) 
        
        # MODIFICATION: Lenient parsing for reply content and title
        new_message_content_raw = message_content_from_agent.get(constants.YAML_CAPSULE_CONTENT)
        new_message_content = _ensure_string_or_none(new_message_content_raw) or ""

        new_message_title_raw = message_content_from_agent.get(constants.YAML_CAPSULE_TITLE) # Optional for replies
        new_message_title = _ensure_string_or_none(new_message_title_raw)

        new_message_word_count = _calculate_word_count(new_message_content)
        new_message = {
            constants.MESSAGE_ID_KEY: _generate_message_id(capsule_id_str, messages_list),
            constants.MESSAGE_AUTHOR_NAME_KEY: author_agent_data.get(constants.AGENT_NAME_KEY),
            constants.MESSAGE_AUTHOR_LINEAGE_KEY: author_agent_data.get(constants.AGENT_LINEAGE_KEY),
            constants.MESSAGE_AUTHOR_GENERATION_KEY: author_agent_data.get(constants.AGENT_GENERATION_KEY),
            constants.MESSAGE_POSTED_AT_TICK_KEY: current_tick,
            constants.MESSAGE_TITLE_KEY: new_message_title, # Use processed title
            constants.MESSAGE_CONTENT_KEY: new_message_content, # Use processed content
            constants.MESSAGE_WORD_COUNT_KEY: new_message_word_count,
            constants.MESSAGE_IS_DELETED_KEY: False,
        }
        messages_list.append(new_message) 
        capsule_data[constants.CAPSULE_MESSAGES_KEY] = messages_list
        capsule_data[constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY] = current_tick
        current_total_word_count = 0
        for msg_in_cap in capsule_data[constants.CAPSULE_MESSAGES_KEY]:
            if not msg_in_cap.get(constants.MESSAGE_IS_DELETED_KEY, False): current_total_word_count += msg_in_cap.get(constants.MESSAGE_WORD_COUNT_KEY, 0)
        capsule_data[constants.CAPSULE_WORD_COUNT_TOTAL_KEY] = current_total_word_count
        file_path = _get_capsule_path(capsule_type, numeric_id, lineage_name)
        file_io_utils.save_yaml(capsule_data, file_path); return True
    except Exception as e: print(f"Error adding message to capsule {numeric_id} of type {capsule_type}: {e}"); return False

def update_capsule_metadata(numeric_id: int, 
                            capsule_type: str, 
                            update_data: Dict[str, Any], 
                            current_tick: int, 
                            lineage_name: Optional[str] = None,
                            room_context = None) -> bool:
    capsule_data = get_capsule(numeric_id, capsule_type, lineage_name, include_deleted_capsule=True, include_deleted_messages=True) 
    if not capsule_data: return False
    if capsule_data.get(constants.CAPSULE_IS_DELETED_KEY, False): return False
    try:
        updated = False
        if constants.YAML_CAPSULE_TITLE in update_data:
            raw_title = update_data[constants.YAML_CAPSULE_TITLE]
            capsule_data[constants.CAPSULE_TITLE_KEY] = _ensure_string_or_none(raw_title)
            updated = True
        
        if constants.YAML_CAPSULE_TAGS in update_data:
            raw_tags = update_data[constants.YAML_CAPSULE_TAGS]
            capsule_data[constants.CAPSULE_TAGS_KEY] = _process_list_field_from_yaml(raw_tags)
            updated = True
            
        if constants.YAML_CAPSULE_ABSTRACT in update_data:
            raw_abstract = update_data[constants.YAML_CAPSULE_ABSTRACT]
            capsule_data[constants.CAPSULE_ABSTRACT_KEY] = _ensure_string_or_none(raw_abstract)
            updated = True
            
        # Handle content update for the first message of the capsule
        if constants.YAML_CAPSULE_CONTENT in update_data:
            messages = capsule_data.get(constants.CAPSULE_MESSAGES_KEY, [])
            if messages:
                # Update the first message's content
                raw_content = update_data[constants.YAML_CAPSULE_CONTENT]
                new_content = _ensure_string_or_none(raw_content)
                if new_content is not None:
                    messages[0][constants.MESSAGE_CONTENT_KEY] = new_content

                    # Recalculate word count for the updated message
                    messages[0][constants.MESSAGE_WORD_COUNT_KEY] = _calculate_word_count(new_content)

                    # Recalculate total word count for the capsule
                    current_total_word_count = 0
                    for msg in messages:
                        if not msg.get(constants.MESSAGE_IS_DELETED_KEY, False):
                            current_total_word_count += msg.get(constants.MESSAGE_WORD_COUNT_KEY, 0)
                    capsule_data[constants.CAPSULE_WORD_COUNT_TOTAL_KEY] = current_total_word_count
                    updated = True

        if updated:
            capsule_data[constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY] = current_tick
            file_path = _get_capsule_path(capsule_type, numeric_id, lineage_name)
            file_io_utils.save_yaml(capsule_data, file_path)
            
            # Notify agents about the capsule update if room_context is provided
            if room_context and capsule_type in [constants.CAPSULE_TYPE_PUBLIC, constants.CAPSULE_TYPE_ARCHIVE]:
                capsule_id = capsule_data.get(constants.CAPSULE_ID_KEY, "")
                capsule_title = capsule_data.get(constants.CAPSULE_TITLE_KEY, f"Capsule #{numeric_id}")
                notify_agents_of_capsule_update(capsule_id, capsule_type, capsule_title, str(numeric_id), room_context)
        
        return True # Return True even if no specific fields were updated, as long as capsule exists
    except Exception as e: print(f"Error updating capsule metadata for {numeric_id} of type {capsule_type}: {e}"); return False

def update_message_content(numeric_id: int, 
                           capsule_type: str, 
                           message_id_str: str, 
                           update_data: Dict[str, Any], 
                           current_tick: int, 
                           lineage_name: Optional[str] = None,
                           room_context = None) -> bool:
    capsule_data = get_capsule(numeric_id, capsule_type, lineage_name, include_deleted_capsule=True, include_deleted_messages=True) #
    if not capsule_data or capsule_data.get(constants.CAPSULE_IS_DELETED_KEY, False): return False #
    try:
        messages_list = capsule_data.get(constants.CAPSULE_MESSAGES_KEY, []) #
        message_updated_flag = False #
        # new_total_word_count = 0 # This was in the original, moved recalculation logic #

        target_message_index = -1
        for i, msg_check in enumerate(messages_list):
            if msg_check.get(constants.MESSAGE_ID_KEY) == message_id_str:
                target_message_index = i
                break
        
        if target_message_index == -1:
            print(f"Error updating message: Message ID {message_id_str} not found in capsule {numeric_id}.")
            return False # Message not found

        msg = messages_list[target_message_index] # Get the actual message dictionary
        if msg.get(constants.MESSAGE_IS_DELETED_KEY, False): 
            print(f"Error updating message: Message ID {message_id_str} is deleted.")
            return False # Cannot update deleted message
        
        # --- Start of MODIFICATIONS for title and content update ---
        if constants.YAML_CAPSULE_CONTENT in update_data:
            new_content_raw = update_data[constants.YAML_CAPSULE_CONTENT]
            new_content = _ensure_string_or_none(new_content_raw) or ""
            msg[constants.MESSAGE_CONTENT_KEY] = new_content #
            msg[constants.MESSAGE_WORD_COUNT_KEY] = _calculate_word_count(new_content) # Update message's own word count #
            message_updated_flag = True #
        
        # Handle title update: for the message itself AND for the capsule if it's the first message
        if constants.YAML_CAPSULE_TITLE in update_data: 
            new_title_raw = update_data[constants.YAML_CAPSULE_TITLE]
            processed_title = _ensure_string_or_none(new_title_raw)
            
            msg[constants.MESSAGE_TITLE_KEY] = processed_title # Update message title #
            message_updated_flag = True #

            # Check if this is the first message (ID ends with "-1")
            # More robust: check if it's actually the first in the list if IDs could somehow be non-sequential initially (though _generate_message_id is sequential)
            is_first_message = (target_message_index == 0) 

            if is_first_message:
                capsule_data[constants.CAPSULE_TITLE_KEY] = processed_title # Update parent capsule's title
                # message_updated_flag is already true
        # --- End of MODIFICATIONS for title and content update ---

        if message_updated_flag:
            messages_list[target_message_index] = msg # Ensure the modified msg object is set back into the list (good practice)
            
            # Recalculate total word count for the capsule from all non-deleted messages
            current_total_word_count = 0
            for m_sum in messages_list: # Iterate over the (potentially) modified messages_list
                if not m_sum.get(constants.MESSAGE_IS_DELETED_KEY, False):
                    current_total_word_count += m_sum.get(constants.MESSAGE_WORD_COUNT_KEY, 0)
            capsule_data[constants.CAPSULE_WORD_COUNT_TOTAL_KEY] = current_total_word_count # Set recalculated total #
            
            capsule_data[constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY] = current_tick #
            # capsule_data[constants.CAPSULE_MESSAGES_KEY] = messages_list # This line is redundant if messages_list is a reference modified in place

            file_path = _get_capsule_path(capsule_type, numeric_id, lineage_name) #
            file_io_utils.save_yaml(capsule_data, file_path)
            
            # Notify agents about the message update if room_context is provided
            if room_context and capsule_type in [constants.CAPSULE_TYPE_PUBLIC, constants.CAPSULE_TYPE_ARCHIVE]:
                capsule_id = capsule_data.get(constants.CAPSULE_ID_KEY, "")
                capsule_title = capsule_data.get(constants.CAPSULE_TITLE_KEY, f"Capsule #{numeric_id}")
                notify_agents_of_capsule_update(capsule_id, capsule_type, capsule_title, message_id_str, room_context)
            
            return True #
        
        return False # Return False if no relevant update_data fields triggered a change (message_updated_flag remained False)
    except Exception as e: print(f"Error updating message {message_id_str} in capsule {numeric_id} of type {capsule_type}: {e}"); return False #


# ... (get_capsule, get_capsule_metadata, list_capsules, delete_capsule, delete_message_from_capsule, 
#      _update_agent_name_references_in_capsule_data, update_agent_name_in_capsules
#      remain the same unless they directly take raw YAML data that needs this processing,
#      but they mostly operate on already processed capsule data or use other functions that do.)
def get_capsule(numeric_id: int,
                capsule_type: str,
                lineage_name: Optional[str] = None,
                include_deleted_capsule: bool = False,
                include_deleted_messages: bool = False) -> Optional[Dict[str, Any]]:
    try:
        file_path = _get_capsule_path(capsule_type, numeric_id, lineage_name)
        capsule_data = file_io_utils.load_yaml(file_path)
        if not capsule_data: return None
        if not include_deleted_capsule and capsule_data.get(constants.CAPSULE_IS_DELETED_KEY, False): return None 
        if not include_deleted_messages and constants.CAPSULE_MESSAGES_KEY in capsule_data:
            original_messages = capsule_data.get(constants.CAPSULE_MESSAGES_KEY, [])
            # Create a copy for display to avoid modifying the loaded data if it's cached elsewhere
            capsule_data_for_display = capsule_data.copy() 
            capsule_data_for_display[constants.CAPSULE_MESSAGES_KEY] = [
                msg for msg in original_messages if not msg.get(constants.MESSAGE_IS_DELETED_KEY, False)
            ]
            return capsule_data_for_display
        return capsule_data # Return original if deleted messages are included or no messages key
    except Exception as e: print(f"Error getting capsule {numeric_id} of type {capsule_type}: {e}"); return None

def get_capsule_metadata(numeric_id: int, 
                         capsule_type: str, 
                         lineage_name: Optional[str] = None,
                         agent_read_status: Optional[Dict[str, bool]] = None
                         ) -> Optional[Dict[str, Any]]:
    full_capsule_data = get_capsule(numeric_id, capsule_type, lineage_name, 
                                    include_deleted_capsule=False, 
                                    include_deleted_messages=False) 
    if not full_capsule_data:
        return None
    
    metadata = {key: value for key, value in full_capsule_data.items() if key != constants.CAPSULE_MESSAGES_KEY}
    
    active_messages = full_capsule_data.get(constants.CAPSULE_MESSAGES_KEY, []) 
    metadata['total_message_count'] = len(active_messages)
    
    unread_count = 0
    if agent_read_status is not None: # This is agent_data[room_key][AGENT_ROOM_STATE_READ_STATUS_KEY]
        for msg in active_messages: # active_messages are non-deleted messages from the capsule
            msg_id = msg.get(constants.MESSAGE_ID_KEY)
            if msg_id and not agent_read_status.get(msg_id, False): # If msg_id not in read_status or its value is False
                unread_count += 1
    metadata[constants.CAPSULE_UNREAD_MESSAGE_COUNT_KEY] = unread_count
    return metadata

def list_capsules(capsule_type: str,
                  lineage_name: Optional[str] = None,
                  agent_read_status: Optional[Dict[str, bool]] = None, 
                  tag_filter: Optional[str] = None 
                  ) -> List[Dict[str, Any]]: 
    capsule_metadata_list = []
    try:
        dir_path = _get_capsule_dir_path(capsule_type, lineage_name)
        if not file_io_utils.dir_exists(dir_path): return []
        
        # Correctly get prefix for the given capsule type and potential lineage
        prefix_pattern, _ = _get_capsule_file_prefix_and_full_id(capsule_type, 1, lineage_name) # Dummy ID for prefix

        all_capsule_files = file_io_utils.list_files(dir_path, constants.YAML_EXTENSION)
        candidate_file_infos = []
        escaped_prefix = re.escape(prefix_pattern)
        pattern = re.compile(f"^{escaped_prefix}(\\d+){re.escape(constants.YAML_EXTENSION)}$")
        
        for filename in all_capsule_files:
            match = pattern.match(filename)
            if match:
                try: candidate_file_infos.append({'filename': filename, 'id': int(match.group(1))})
                except ValueError: continue
        
        candidate_file_infos.sort(key=lambda x: x['id'], reverse=True) 
        
        for file_info in candidate_file_infos:
            metadata = get_capsule_metadata(file_info['id'], capsule_type, lineage_name, agent_read_status)
            if metadata: 
                if tag_filter:
                    tags = metadata.get(constants.CAPSULE_TAGS_KEY, []) # Tags are already processed into a list here
                    if isinstance(tags, list) and tag_filter.lower() in [str(t).lower() for t in tags]:
                        capsule_metadata_list.append(metadata)
                else:
                    capsule_metadata_list.append(metadata)
    except Exception as e: print(f"Error listing {capsule_type} capsules: {e}")
    return capsule_metadata_list


def delete_capsule(numeric_id: int, capsule_type: str, current_tick: int, lineage_name: Optional[str] = None) -> bool:
    # Load with include_deleted_capsule=True to allow re-deleting (idempotent) or to fetch it if already soft-deleted.
    capsule_data = get_capsule(numeric_id, capsule_type, lineage_name, include_deleted_capsule=True, include_deleted_messages=True)
    if not capsule_data: return False # Capsule file doesn't exist
    
    # If it's already marked deleted, we can consider the operation successful.
    if capsule_data.get(constants.CAPSULE_IS_DELETED_KEY, False): return True 
    
    try:
        capsule_data[constants.CAPSULE_IS_DELETED_KEY] = True
        capsule_data[constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY] = current_tick
        messages_list = capsule_data.get(constants.CAPSULE_MESSAGES_KEY, [])
        for i in range(len(messages_list)): 
            messages_list[i][constants.MESSAGE_IS_DELETED_KEY] = True
        capsule_data[constants.CAPSULE_MESSAGES_KEY] = messages_list
        
        # Also set total word count to 0 for deleted capsules
        capsule_data[constants.CAPSULE_WORD_COUNT_TOTAL_KEY] = 0

        file_path = _get_capsule_path(capsule_type, numeric_id, lineage_name)
        file_io_utils.save_yaml(capsule_data, file_path)
        return True
    except Exception as e: print(f"Error soft deleting capsule {numeric_id} of type {capsule_type}: {e}"); return False

def delete_message_from_capsule(numeric_id: int, capsule_type: str, message_id_str: str, current_tick: int, lineage_name: Optional[str] = None) -> bool:
    capsule_data = get_capsule(numeric_id, capsule_type, lineage_name, include_deleted_capsule=True, include_deleted_messages=True) 
    if not capsule_data: return False # Capsule itself not found
    if capsule_data.get(constants.CAPSULE_IS_DELETED_KEY, False): return False # Cannot modify messages of a deleted capsule

    try:
        messages_list = capsule_data.get(constants.CAPSULE_MESSAGES_KEY, [])
        message_found_and_marked = False
        active_messages_remain = False
        new_total_word_count = 0

        for i, msg in enumerate(messages_list):
            if msg.get(constants.MESSAGE_ID_KEY) == message_id_str:
                if msg.get(constants.MESSAGE_IS_DELETED_KEY, False): return True # Already deleted
                messages_list[i][constants.MESSAGE_IS_DELETED_KEY] = True
                message_found_and_marked = True
                # Don't break; continue to check active_messages_remain and recalculate word count
            
            if not messages_list[i].get(constants.MESSAGE_IS_DELETED_KEY, False): # Check current status after potential modification
                active_messages_remain = True
                new_total_word_count += messages_list[i].get(constants.MESSAGE_WORD_COUNT_KEY, 0)
        
        if not message_found_and_marked: print(f"Message {message_id_str} not found for deletion."); return False
            
        capsule_data[constants.CAPSULE_MESSAGES_KEY] = messages_list 
        capsule_data[constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY] = current_tick
        capsule_data[constants.CAPSULE_WORD_COUNT_TOTAL_KEY] = new_total_word_count

        if not active_messages_remain: # If all messages are now deleted
            print(f"All messages in capsule {capsule_data.get(constants.CAPSULE_ID_KEY)} are now deleted. Soft deleting capsule.")
            capsule_data[constants.CAPSULE_IS_DELETED_KEY] = True
            capsule_data[constants.CAPSULE_WORD_COUNT_TOTAL_KEY] = 0 # Explicitly set to 0

        file_path = _get_capsule_path(capsule_type, numeric_id, lineage_name)
        file_io_utils.save_yaml(capsule_data, file_path); return True
    except Exception as e: print(f"Error soft deleting message {message_id_str} from capsule {numeric_id} type {capsule_type}: {e}"); return False

def _update_agent_name_references_in_capsule_data(capsule_data: Dict[str, Any], old_name: str, new_name: str, current_tick: int) -> bool:
    # ... (no changes) ...
    made_change = False
    if capsule_data.get(constants.CAPSULE_AUTHOR_NAME_KEY) == old_name: capsule_data[constants.CAPSULE_AUTHOR_NAME_KEY] = new_name; made_change = True
    if constants.CAPSULE_RECIPIENTS_KEY in capsule_data:
        recipients = capsule_data[constants.CAPSULE_RECIPIENTS_KEY]
        if isinstance(recipients, list):
            new_recipients = [];
            for rec in recipients:
                if rec == old_name: new_recipients.append(new_name); made_change = True
                else: new_recipients.append(rec)
            capsule_data[constants.CAPSULE_RECIPIENTS_KEY] = new_recipients
    messages = capsule_data.get(constants.CAPSULE_MESSAGES_KEY, [])
    for msg in messages:
        if msg.get(constants.MESSAGE_AUTHOR_NAME_KEY) == old_name: msg[constants.MESSAGE_AUTHOR_NAME_KEY] = new_name; made_change = True
    if made_change: capsule_data[constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY] = current_tick
    return made_change

def update_agent_name_in_capsules(old_agent_name: str, new_agent_name: str, current_tick: int, capsule_type_to_scan: str = constants.CAPSULE_TYPE_MAIL):
    # ... (no changes) ...
    if capsule_type_to_scan not in [constants.CAPSULE_TYPE_MAIL, constants.CAPSULE_TYPE_PUBLIC, constants.CAPSULE_TYPE_ARCHIVE]:
        print(f"Name update currently supported only for mail, public, or archive. Skipping {capsule_type_to_scan}."); return 0
    dir_path = _get_capsule_dir_path(capsule_type_to_scan) # Lineage not needed for these types scan
    if not file_io_utils.dir_exists(dir_path): print(f"Directory {dir_path} for {capsule_type_to_scan} not found."); return 0
    updated_capsule_count = 0
    
    # Correctly get prefix for the given capsule type
    prefix_pattern, _ = _get_capsule_file_prefix_and_full_id(capsule_type_to_scan, 1, None) # Dummy ID for prefix, no lineage for these types
    
    escaped_prefix = re.escape(prefix_pattern)
    file_pattern = re.compile(f"^{escaped_prefix}(\\d+){re.escape(constants.YAML_EXTENSION)}$")
    
    for filename in file_io_utils.list_files(dir_path, constants.YAML_EXTENSION):
        if not file_pattern.match(filename): continue # Ensure we only process correctly named capsule files
        file_path = os.path.join(dir_path, filename)
        capsule_data = file_io_utils.load_yaml(file_path) 
        if capsule_data:
            if _update_agent_name_references_in_capsule_data(capsule_data, old_agent_name, new_agent_name, current_tick):
                try: file_io_utils.save_yaml(capsule_data, file_path); updated_capsule_count += 1
                except Exception as e: print(f"Error saving updated capsule {filename} during name update: {e}")
    if updated_capsule_count > 0: print(f"Agent name reference update: Processed {capsule_type_to_scan}. Updated {updated_capsule_count} capsules for name '{old_agent_name}' -> '{new_agent_name}'.")
    return updated_capsule_count

def notify_agents_of_capsule_update(capsule_id: str, capsule_type: str, 
                                  capsule_title: str, updated_item_id: str,
                                  room_context) -> None:
    """
    Notify all active agents who have read the capsule/message that it has been updated.
    Clears their read status for the updated content and sends notification.
    Only works for public_memory and archive capsule types.
    """
    # Only notify for shared capsule types
    if capsule_type not in [constants.CAPSULE_TYPE_PUBLIC, constants.CAPSULE_TYPE_ARCHIVE]:
        return
    
    try:
        # Get all active agents
        agents_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME)
        if not file_io_utils.dir_exists(agents_dir):
            return
            
        agent_files = file_io_utils.list_files(agents_dir, constants.YAML_EXTENSION)
        
        # Determine room short name for read status clearing
        room_short_name = None
        if capsule_type == constants.CAPSULE_TYPE_PUBLIC:
            room_short_name = constants.SHORT_ROOM_NAME_PUBLIC_MEMORY
        elif capsule_type == constants.CAPSULE_TYPE_ARCHIVE:
            room_short_name = constants.SHORT_ROOM_NAME_ARCHIVE
            
        if not room_short_name:
            return
            
        agents_notified = 0
        
        for agent_file_name in agent_files:
            agent_name = agent_file_name.replace(constants.YAML_EXTENSION, "")
            
            # Load agent data (only active agents)
            agent_data = room_context.agent_manager.load_agent_data(agent_name)
            if not agent_data:
                continue
                
            # Check if agent has read status for this room
            room_data = agent_data.get(room_short_name, {})
            read_statuses = room_data.get(constants.AGENT_ROOM_STATE_READ_STATUS_KEY, {})
            
            # Check if agent has read the updated content
            agent_had_read_content = False
            
            # Check if they read the specific message or the whole capsule
            if updated_item_id in read_statuses and read_statuses[updated_item_id]:
                agent_had_read_content = True
                # Clear read status for the updated item
                read_statuses[updated_item_id] = False
                
            # Also check if they read the whole capsule
            if capsule_id in read_statuses and read_statuses[capsule_id]:
                agent_had_read_content = True
                # Clear read status for the whole capsule
                read_statuses[capsule_id] = False
            
            if agent_had_read_content:
                # Update agent's read status
                if room_short_name not in agent_data:
                    agent_data[room_short_name] = {}
                agent_data[room_short_name][constants.AGENT_ROOM_STATE_READ_STATUS_KEY] = read_statuses
                
                # Determine the room display name
                room_display_name = "Public Memory Room" if capsule_type == constants.CAPSULE_TYPE_PUBLIC else "Archive Room"
                
                # Create notification message
                notification_msg = (
                    f"**Content Update Notification**\n\n"
                    f"A {room_display_name.lower()} item you previously read has been updated:\n"
                    f"**{capsule_title}** (Item: {updated_item_id})\n\n"
                    f"Your read status has been cleared for this content. "
                    f"To view the updated content, use: `/execute_action{{read {updated_item_id}}}` in the {room_display_name}."
                )
                
                # Add notification to agent
                room_context.agent_manager.add_pending_notification(agent_data, notification_msg)
                
                # Save updated agent data
                room_context.agent_manager.save_agent_data(agent_name, agent_data)
                agents_notified += 1
                
        if agents_notified > 0:
            print(f"Capsule update notifications: Notified {agents_notified} agent(s) about update to {updated_item_id}")
            
    except Exception as e:
        print(f"Error notifying agents of capsule update: {e}")
        # Don't raise - this is a non-critical feature