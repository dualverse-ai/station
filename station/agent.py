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

# agent.py
"""
Manages agent data, including loading, saving, creation of guest and recursive agents,
ascension, session ending, and description updates.
Names for new recursive agents can be auto-generated.
Lobby help is reset on ascension.
"""

import os
import re # For _generate_unique_guest_name
import random
from typing import Any, List, Dict, Optional
import traceback
import fcntl
import time
import json

from station import constants
from station import file_io_utils


# --- Helper Functions ---
# ... (existing _get_agent_file_path, _generate_unique_guest_name, _int_to_roman) ...
def _get_agent_file_path(agent_name: str) -> str:
    return os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME, f"{agent_name}{constants.YAML_EXTENSION}")

def _generate_unique_guest_name(prefix: str = "Guest_") -> str:
    agents_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME)
    file_io_utils.ensure_dir_exists(agents_dir)
    highest_num = 0
    escaped_prefix = re.escape(prefix)
    pattern = re.compile(f"^{escaped_prefix}(\\d+){re.escape(constants.YAML_EXTENSION)}$")
    agent_files = file_io_utils.list_files(agents_dir, constants.YAML_EXTENSION)
    for filename in agent_files:
        if filename.startswith(prefix): 
            name_part_match = pattern.match(filename)
            if name_part_match:
                try:
                    num = int(name_part_match.group(1))
                    highest_num = max(highest_num, num)
                except ValueError:
                    pass 
    return f"{prefix}{highest_num + 1}"

def _int_to_roman(num: int) -> str:
    if not 0 < num < 4000: 
        if num == 1: return "I"
        if num == 2: return "II"
        # ... (rest of simple Roman numerals for small numbers)
        if num == 3: return "III"
        if num == 4: return "IV"
        if num == 5: return "V"
        if num == 6: return "VI"
        if num == 7: return "VII"
        if num == 8: return "VIII"
        if num == 9: return "IX"
        if num == 10: return "X"
        return str(num) 

    val = [ 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 ]
    syb = [ "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" ]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num

def _deep_update(source: Dict, overrides: Dict) -> Dict:
    """
    Updates a dict with values from another dict, recursively.
    Modifies 'source' in place.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and value: # value is a non-empty dict
            # Get the existing value in source, defaulting to an empty dict if not found or not a dict
            source_value = source.get(key)
            if not isinstance(source_value, dict):
                source_value = {}
            source[key] = _deep_update(source_value, value)
        else: # Value is not a dict, or is an empty dict (which should overwrite)
            source[key] = value
    return source

def _load_random_system_prompts() -> List[str]:
    """Load random system prompts from YAML file"""
    try:
        random_sys_prompts_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.RANDOM_SYS_PROMPT_FILENAME)
        if file_io_utils.file_exists(random_sys_prompts_path):
            return file_io_utils.load_yaml(random_sys_prompts_path) or []
    except Exception:
        pass
    return []

def _process_system_prompt_with_random(original_prompt: Optional[str]) -> Optional[str]:
    """
    Process system prompt with random override logic:
    1. If blank/None -> use random
    2. If original came from random file + 20% chance -> use new random
    """
    # Load random system prompts
    random_sys_prompts = _load_random_system_prompts()
    if not random_sys_prompts:
        return original_prompt

    # Case 1: Blank system prompt
    if not original_prompt or original_prompt.strip() == "":
        return random.choice(random_sys_prompts)

    # Case 2: Check if original is from random file + 20% chance
    if original_prompt in random_sys_prompts and random.random() < constants.RANDOM_SYS_PROMPT_OVERRIDE_PROB:
        return random.choice(random_sys_prompts)

    return original_prompt

# --- Public API ---
# ... (load_agent_data, save_agent_data remain the same) ...
def load_agent_data(agent_name: str,
                    include_ascended: bool = False,
                    include_ended: bool = False) -> Optional[Dict[str, Any]]:
    file_path = _get_agent_file_path(agent_name)
    agent_data = file_io_utils.load_yaml(file_path)
    if agent_data:
        if not include_ascended and agent_data.get(constants.AGENT_IS_ASCENDED_KEY, False):
            return None 
        if not include_ended and agent_data.get(constants.AGENT_SESSION_ENDED_KEY, False):
            return None
    return agent_data

def save_agent_data(agent_name: str, agent_data: Dict[str, Any]) -> bool:
    file_path = _get_agent_file_path(agent_name)
    try:
        file_io_utils.save_yaml(agent_data, file_path)
        return True
    except Exception as e:
        print(f"Error saving agent data for {agent_name}: {e}")
        return False

def update_agent_fields_atomic(agent_name: str, delta_updates: Dict[str, Any], max_retries: int = 5) -> bool:
    """
    Atomically update agent fields with file locking to prevent race conditions.
    This is crucial for concurrent updates like multiple research evaluation notifications.
    """
    if not delta_updates:
        return True
        
    file_path = _get_agent_file_path(agent_name)
    lock_path = file_path + ".lock"
    
    for attempt in range(max_retries):
        try:
            # Create lock file
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_EXCL)
            try:
                # Acquire exclusive lock
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Load current data
                agent_data = load_agent_data(agent_name)
                if not agent_data:
                    # Try loading including inactive agents
                    agent_data = load_agent_data(agent_name, include_ended=True, include_ascended=True)
                    if not agent_data:
                        print(f"Cannot update fields: Agent '{agent_name}' not found.")
                        return False
                
                # Apply updates
                _deep_update(agent_data, delta_updates)
                
                # Save updated data
                success = save_agent_data(agent_name, agent_data)
                
                return success
                
            finally:
                # Always release lock and close file descriptor
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                # Remove lock file
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass
                    
        except OSError as e:
            if e.errno == 17:  # File exists (lock is held by another process)
                if attempt < max_retries - 1:
                    # Wait a bit and retry
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    print(f"Failed to acquire lock for {agent_name} after {max_retries} attempts")
                    return False
            else:
                print(f"Error updating agent fields atomically for {agent_name}: {e}")
                return False
                
        except Exception as e:
            print(f"Unexpected error updating agent fields for {agent_name}: {e}")
            return False
    
    return False
    
def update_agent_fields(agent_name: str, delta_updates: Dict[str, Any]) -> bool:
    """Update agent fields atomically with file locking to prevent race conditions."""
    if not delta_updates: 
        return True
    
    # Use the new atomic update function
    return update_agent_fields_atomic(agent_name, delta_updates)

def add_pending_notification_atomic(agent_name: str, message: str) -> bool:
    """
    Add a notification to an agent's pending notifications list atomically.
    This prevents race conditions when multiple threads try to add notifications simultaneously.
    Used by auto evaluators that run in separate threads.
    """
    if not message:
        return True
        
    # Use atomic update to append to the notifications list
    def update_func(agent_data: Dict[str, Any]) -> None:
        current_notifications = agent_data.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, [])
        if not isinstance(current_notifications, list):
            current_notifications = []
        agent_data[constants.AGENT_NOTIFICATIONS_PENDING_KEY] = current_notifications + [message]
    
    return update_agent_with_function(agent_name, update_func)

def update_agent_with_function(agent_name: str, update_func, max_retries: int = 5) -> bool:
    """
    Atomically update agent data using a custom update function.
    The update_func receives the agent_data dict and should modify it in place.
    """
    file_path = _get_agent_file_path(agent_name)
    lock_path = file_path + ".lock"
    
    for attempt in range(max_retries):
        try:
            # Create lock file
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_EXCL)
            try:
                # Acquire exclusive lock
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Load current data
                agent_data = load_agent_data(agent_name)
                if not agent_data:
                    # Try loading including inactive agents
                    agent_data = load_agent_data(agent_name, include_ended=True, include_ascended=True)
                    if not agent_data:
                        print(f"Cannot update: Agent '{agent_name}' not found.")
                        return False
                
                # Apply custom update function
                update_func(agent_data)
                
                # Save updated data
                success = save_agent_data(agent_name, agent_data)
                
                return success
                
            finally:
                # Always release lock and close file descriptor
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                # Remove lock file
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass
                    
        except OSError as e:
            if e.errno == 17:  # File exists (lock is held by another process)
                if attempt < max_retries - 1:
                    # Wait a bit and retry
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    print(f"Failed to acquire lock for {agent_name} after {max_retries} attempts")
                    return False
            else:
                print(f"Error updating agent with function for {agent_name}: {e}")
                return False
                
        except Exception as e:
            print(f"Unexpected error updating agent {agent_name}: {e}")
            return False
    
    return False

def _set_default_archive_pins(agent_data: Dict[str, Any]):
    """Sets default pinned archive capsules (ID 1) if not already set."""
    archive_room_short_name = constants.SHORT_ROOM_NAME_ARCHIVE
    
    # Ensure the archive room data structure exists
    if archive_room_short_name not in agent_data:
        agent_data[archive_room_short_name] = {}
    
    # Check if pins are already set for the archive room (e.g., during ascension carry-over)
    if constants.AGENT_ROOM_STATE_PINNED_CAPSULES_KEY not in agent_data[archive_room_short_name]:
        default_pinned_ids = [
            #f"{constants.CAPSULE_ID_PREFIX_ARCHIVE}{1}", # e.g., "archive_1"
            #f"{constants.CAPSULE_ID_PREFIX_ARCHIVE}{2}"  # e.g., "archive_2"
        ]
        agent_data[archive_room_short_name][constants.AGENT_ROOM_STATE_PINNED_CAPSULES_KEY] = default_pinned_ids
        # print(f"DEBUG: Set default archive pins for agent: {default_pinned_ids}") # Optional debug

def create_guest_agent(model_name: str,
                       current_tick: int,
                       guest_prefix: str = getattr(constants, "AGENT_DEFAULT_GUEST_NAME_PREFIX", "Guest_"),
                       internal_note: str = "",
                       assigned_ancestor: str = "",
                       initial_tokens_max: Optional[int] = None,
                       # ADD LLM CONFIG PARAMS FOR NEW GUESTS
                       model_provider_class: Optional[str] = None,
                       llm_system_prompt: Optional[str] = None,
                       llm_temperature: Optional[float] = None,
                       llm_max_tokens: Optional[int] = None,
                       llm_custom_api_params: Optional[Dict[str, Any]] = None
                       ) -> Optional[Dict[str, Any]]:
    agent_name = _generate_unique_guest_name(guest_prefix)
    default_location = getattr(constants, "AGENT_DEFAULT_STARTING_LOCATION", constants.ROOM_LOBBY)
    max_budget = initial_tokens_max if initial_tokens_max is not None else constants.DEFAULT_GUEST_MAX_TOKENS

    # Process system prompt with random override logic
    llm_system_prompt = _process_system_prompt_with_random(llm_system_prompt)

    agent_data = {
        constants.AGENT_NAME_KEY: agent_name,
        constants.AGENT_MODEL_NAME_KEY: model_name, # Specific model ID
        constants.AGENT_INTERNAL_NOTE_KEY: internal_note, 
        constants.AGENT_ASSIGNED_ANCESTOR_KEY: assigned_ancestor, # Optional, can be empty
        constants.AGENT_LINEAGE_KEY: None,
        constants.AGENT_GENERATION_KEY: None,
        constants.AGENT_DESCRIPTION_KEY: "A new guest agent.",
        constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_GUEST,
        constants.AGENT_TOKEN_BUDGET_CURRENT_KEY: None, 
        constants.AGENT_TOKEN_BUDGET_MAX_KEY: max_budget,
        constants.AGENT_CURRENT_LOCATION_KEY: default_location,
        constants.AGENT_ROOM_OUTPUT_HISTORY_KEY: [],
        constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
        constants.AGENT_IS_ASCENDED_KEY: False, 
        constants.AGENT_ASCENDED_TO_NAME_KEY: None,
        constants.AGENT_SESSION_ENDED_KEY: False,
        # Tick tracking fields
        constants.AGENT_TICK_BIRTH_KEY: current_tick,
        constants.AGENT_TICK_ASCEND_KEY: None,
        constants.AGENT_TICK_EXIT_KEY: None,
        constants.AGENT_MAX_AGE_KEY: constants.AGENT_MAX_LIFE,
        constants.AGENT_ROLE_KEY: None,  # Default role is None (normal agent)
        # LLM Config fields
        constants.AGENT_MODEL_PROVIDER_CLASS_KEY: model_provider_class or "Gemini", # Default
        constants.AGENT_LLM_SYSTEM_PROMPT_KEY: llm_system_prompt,
        constants.AGENT_LLM_TEMPERATURE_KEY: llm_temperature,
        constants.AGENT_LLM_MAX_TOKENS_KEY: llm_max_tokens,
        constants.AGENT_LLM_CUSTOM_API_PARAMS_KEY: llm_custom_api_params,
    }
    _set_default_archive_pins(agent_data)
    if save_agent_data(agent_name, agent_data):
        print(f"Guest agent '{agent_name}' (Model: {model_name}) created successfully.")
        return agent_data
    return None

def create_recursive_agent(model_name: str,
                           lineage: str,
                           generation: int,
                           current_tick: int,
                           agent_name_override: Optional[str] = None,
                           description: Optional[str] = None,
                           internal_note: str = "",
                           initial_tokens_max: Optional[int] = None,
                           # ADD LLM CONFIG PARAMS
                           model_provider_class: Optional[str] = None,
                           llm_system_prompt: Optional[str] = None,
                           llm_temperature: Optional[float] = None,
                           llm_max_tokens: Optional[int] = None,
                           llm_custom_api_params: Optional[Dict[str, Any]] = None
                           ) -> Optional[Dict[str, Any]]:
    final_agent_name = agent_name_override
    if final_agent_name is None:
        if not lineage or generation is None: return None
        final_agent_name = f"{lineage} {_int_to_roman(generation)}"
    
    if load_agent_data(final_agent_name, include_ascended=True, include_ended=True):
        print(f"Cannot create recursive agent: Name '{final_agent_name}' is already taken or used.")
        return None

    max_budget = initial_tokens_max if initial_tokens_max is not None else constants.DEFAULT_RECURSIVE_MAX_TOKENS
    location = constants.ROOM_LOBBY 
    final_description = description or f"A recursive agent of the {lineage} lineage, generation {_int_to_roman(generation)}."

    agent_data = {
        constants.AGENT_NAME_KEY: final_agent_name,
        constants.AGENT_MODEL_NAME_KEY: model_name, # Specific model ID
        constants.AGENT_INTERNAL_NOTE_KEY: internal_note,
        constants.AGENT_LINEAGE_KEY: lineage,
        constants.AGENT_GENERATION_KEY: generation,
        constants.AGENT_DESCRIPTION_KEY: final_description,
        constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
        constants.AGENT_TOKEN_BUDGET_CURRENT_KEY: None,
        constants.AGENT_TOKEN_BUDGET_MAX_KEY: max_budget,
        constants.AGENT_CURRENT_LOCATION_KEY: location, 
        constants.AGENT_ROOM_OUTPUT_HISTORY_KEY: [],
        constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
        constants.AGENT_IS_ASCENDED_KEY: False,
        constants.AGENT_ASCENDED_TO_NAME_KEY: None,
        constants.AGENT_SESSION_ENDED_KEY: False,
        # Tick tracking fields
        constants.AGENT_TICK_BIRTH_KEY: current_tick,
        constants.AGENT_TICK_ASCEND_KEY: current_tick,  # Recursive agents are born ascended
        constants.AGENT_TICK_EXIT_KEY: None,
        constants.AGENT_MAX_AGE_KEY: constants.AGENT_MAX_LIFE,
        constants.AGENT_ROLE_KEY: None,  # Default role is None (normal agent)
        # LLM Config fields
        constants.AGENT_MODEL_PROVIDER_CLASS_KEY: model_provider_class or "Gemini", # Default
        constants.AGENT_LLM_SYSTEM_PROMPT_KEY: llm_system_prompt,
        constants.AGENT_LLM_TEMPERATURE_KEY: llm_temperature,
        constants.AGENT_LLM_MAX_TOKENS_KEY: llm_max_tokens,
        constants.AGENT_LLM_CUSTOM_API_PARAMS_KEY: llm_custom_api_params,
    }
    _set_default_archive_pins(agent_data)
    if save_agent_data(final_agent_name, agent_data):
        print(f"Recursive agent '{final_agent_name}' (Model: {model_name}) created successfully.")
        return agent_data
    return None

def ascend_agent(guest_agent_name: str,
                 new_recursive_name: str, 
                 new_lineage: str,
                 new_generation: int,
                 current_tick: int,
                 new_description: Optional[str] = None,
                 ascension_notification: Optional[str] = None) -> Optional[Dict[str, Any]]:
    original_guest_data = load_agent_data(guest_agent_name, include_ascended=True, include_ended=True)     
    
    if not original_guest_data:
        print(f"Cannot ascend: Guest agent '{guest_agent_name}' not found."); return None
    if original_guest_data.get(constants.AGENT_IS_ASCENDED_KEY, False):
        print(f"Error in ascend_agent: Agent '{guest_agent_name}' has already ascended."); return None
    if original_guest_data.get(constants.AGENT_SESSION_ENDED_KEY, False):
        print(f"Error in ascend_agent: Agent '{guest_agent_name}' session has ended."); return None
    if original_guest_data.get(constants.AGENT_STATUS_KEY) != constants.AGENT_STATUS_GUEST:
        print(f"Error in ascend_agent: Agent '{guest_agent_name}' is not a guest agent."); return None

    if load_agent_data(new_recursive_name, include_ascended=True, include_ended=True): 
        print(f"Cannot ascend: Target name '{new_recursive_name}' is already taken by an agent identity.")
        return None
    
    # Update and save original guest record (as before)
    original_guest_data[constants.AGENT_IS_ASCENDED_KEY] = True
    original_guest_data[constants.AGENT_ASCENDED_TO_NAME_KEY] = new_recursive_name
    original_guest_data[constants.AGENT_DESCRIPTION_KEY] = \
        f"Guest agent, ascended to {new_recursive_name} at tick {current_tick}."
    if not save_agent_data(guest_agent_name, original_guest_data):
        print(f"Failed to update original guest agent record for '{guest_agent_name}'. Ascension aborted."); return None
    print(f"Original guest agent '{guest_agent_name}' marked as ascended to '{new_recursive_name}'.")


    final_description_ascended = new_description
    if not final_description_ascended:
        final_description_ascended = (f"A recursive agent of the {new_lineage} lineage, "
                                      f"gen {_int_to_roman(new_generation)}. " # Use _int_to_roman here
                                      f"(Ascended from {guest_agent_name} at tick {current_tick})")


    new_recursive_agent_data = {
        constants.AGENT_NAME_KEY: new_recursive_name,
        # ... (other existing fields: model, internal_note, status, lineage, generation, token budget carry-over) ...
        constants.AGENT_MODEL_NAME_KEY: original_guest_data.get(constants.AGENT_MODEL_NAME_KEY, "unknown_model_carried_over"),
        constants.AGENT_INTERNAL_NOTE_KEY: original_guest_data.get(constants.AGENT_INTERNAL_NOTE_KEY, ""),
        constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
        constants.AGENT_LINEAGE_KEY: new_lineage,
        constants.AGENT_GENERATION_KEY: new_generation,
        constants.AGENT_DESCRIPTION_KEY: final_description_ascended,
        constants.AGENT_TOKEN_BUDGET_MAX_KEY: original_guest_data.get(constants.AGENT_TOKEN_BUDGET_MAX_KEY),
        constants.AGENT_TOKEN_BUDGET_CURRENT_KEY: original_guest_data.get(constants.AGENT_TOKEN_BUDGET_CURRENT_KEY),
        constants.AGENT_CURRENT_LOCATION_KEY: constants.ROOM_LOBBY,
        constants.AGENT_ROOM_OUTPUT_HISTORY_KEY: [],
        constants.AGENT_NOTIFICATIONS_PENDING_KEY: [ascension_notification] if ascension_notification else [],
        constants.AGENT_IS_ASCENDED_KEY: False,
        constants.AGENT_ASCENDED_TO_NAME_KEY: None,
        constants.AGENT_SESSION_ENDED_KEY: False,
        # Tick tracking fields - carry over birth, set ascend
        constants.AGENT_TICK_BIRTH_KEY: original_guest_data.get(constants.AGENT_TICK_BIRTH_KEY),
        constants.AGENT_TICK_ASCEND_KEY: current_tick,
        constants.AGENT_TICK_EXIT_KEY: None,
        # Carry over max_age from original guest
        constants.AGENT_MAX_AGE_KEY: original_guest_data.get(constants.AGENT_MAX_AGE_KEY, constants.AGENT_MAX_LIFE),
        # Carry over role from original guest
        constants.AGENT_ROLE_KEY: original_guest_data.get(constants.AGENT_ROLE_KEY),
    }
    
    # --- MODIFICATION: Carry over LLM configuration ---
    llm_config_keys_to_carry = [
        constants.AGENT_MODEL_PROVIDER_CLASS_KEY,
        constants.AGENT_LLM_SYSTEM_PROMPT_KEY,
        constants.AGENT_LLM_TEMPERATURE_KEY,
        constants.AGENT_LLM_MAX_TOKENS_KEY,
        constants.AGENT_LLM_CUSTOM_API_PARAMS_KEY
    ]
    for key in llm_config_keys_to_carry:
        if key in original_guest_data:
            new_recursive_agent_data[key] = original_guest_data[key]
    
    # Carry over other room-specific UI states
    for short_room_name_key_asc in constants.SHORT_ROOM_NAME_TO_FULL_MAP.keys():
        if short_room_name_key_asc in original_guest_data and isinstance(original_guest_data[short_room_name_key_asc], dict):
            if short_room_name_key_asc == constants.SHORT_ROOM_NAME_PRIVATE_MEMORY:
                pmr_data_to_set = original_guest_data[short_room_name_key_asc].copy()
                pmr_data_to_set[constants.AGENT_ROOM_STATE_FIRST_VISIT_HELP_SHOWN_KEY] = False
                new_recursive_agent_data[short_room_name_key_asc] = pmr_data_to_set
            else:
                new_recursive_agent_data[short_room_name_key_asc] = original_guest_data[short_room_name_key_asc].copy()

    # --- ADD THESE LINES for Capsule Protocol Help Status ---
    capsule_help_shown_key = constants.AGENT_STATE_CAPSULE_PROTOCOL_HELP_SHOWN_KEY
    global_state_key = constants.AGENT_STATE_DATA_KEY # This is where capsule help shown is stored

    original_global_state_data = original_guest_data.get(global_state_key, {})
    if isinstance(original_global_state_data, dict) and capsule_help_shown_key in original_global_state_data:
        if global_state_key not in new_recursive_agent_data or not isinstance(new_recursive_agent_data[global_state_key], dict):
            new_recursive_agent_data[global_state_key] = {}
        new_recursive_agent_data[global_state_key][capsule_help_shown_key] = original_global_state_data[capsule_help_shown_key]
    # --- END OF ADDED LINES ---
    
    # --- Carry over meta prompt ---
    if constants.AGENT_META_PROMPT_KEY in original_guest_data:
        new_recursive_agent_data[constants.AGENT_META_PROMPT_KEY] = original_guest_data[constants.AGENT_META_PROMPT_KEY]

    _set_default_archive_pins(new_recursive_agent_data) # Ensure default pins are set

    # --- MODIFICATION: Rename/Move LLM Chat History File ---
    guest_agent_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME, guest_agent_name)
    old_history_file = os.path.join(guest_agent_dir, "llm_chat_history.yamll")

    new_recursive_agent_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME, new_recursive_name)
    # Ensure the new agent's directory exists (LLMConnector would also do this, but good to be proactive)
    file_io_utils.ensure_dir_exists(new_recursive_agent_dir)
    new_history_file = os.path.join(new_recursive_agent_dir, "llm_chat_history.yamll")

    history_moved = False
    if file_io_utils.file_exists(old_history_file):
        try:
            # If new_history_file already exists (e.g., from a previous failed ascension or a name collision not caught),
            # decide on a strategy: overwrite, backup, or fail. Renaming is safer.
            if file_io_utils.file_exists(new_history_file):
                print(f"Warning: Target history file {new_history_file} already exists. Overwriting for ascended agent {new_recursive_name}.")
                # os.remove(new_history_file) # Or backup logic
            os.rename(old_history_file, new_history_file)
            print(f"LLM chat history for {guest_agent_name} moved to {new_recursive_name}.")
            history_moved = True
        except OSError as e:
            print(f"Error moving LLM chat history from {old_history_file} to {new_history_file}: {e}")
            # If history move fails, the new agent will start with a fresh history.
            # This might be acceptable, or you could choose to fail the ascension.
    else:
        print(f"No LLM chat history file found for guest {guest_agent_name} at {old_history_file}. New agent {new_recursive_name} will start with fresh history.")
        history_moved = True # No file to move is not an error for this step

    dialogue_logs_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.DIALOGUE_LOGS_DIR_NAME)
    # Construct old and new dialogue log filenames (e.g., Guest_1_dialogue.yamll)
    # Ensure your _get_agent_dialogue_log_path in station.py uses a similar construction or make it accessible.
    # For now, let's construct it directly based on constants:
    safe_guest_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in guest_agent_name)
    old_dialogue_log_filename = f"{safe_guest_name}{constants.DIALOGUE_LOG_FILENAME_SUFFIX}"
    old_dialogue_log_file = os.path.join(dialogue_logs_dir, old_dialogue_log_filename)

    safe_new_recursive_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in new_recursive_name)
    new_dialogue_log_filename = f"{safe_new_recursive_name}{constants.DIALOGUE_LOG_FILENAME_SUFFIX}"
    new_dialogue_log_file = os.path.join(dialogue_logs_dir, new_dialogue_log_filename)

    main_dialogue_log_moved_or_not_needed = True # Assume success if old file doesn't exist
    if file_io_utils.file_exists(old_dialogue_log_file):
        try:
            if file_io_utils.file_exists(new_dialogue_log_file):
                print(f"Warning: Target main dialogue log {new_dialogue_log_file} already exists. Overwriting.")
                os.remove(new_dialogue_log_file)
            os.rename(old_dialogue_log_file, new_dialogue_log_file)
            print(f"Main dialogue log for {guest_agent_name} moved/renamed to {new_dialogue_log_filename}.")
        except OSError as e:
            print(f"Error moving main dialogue log from {old_dialogue_log_file} to {new_dialogue_log_file}: {e}")
            main_dialogue_log_moved_or_not_needed = False
    else:
        print(f"No main dialogue log file found for guest {guest_agent_name} at {old_dialogue_log_file}.")

    # Save the new recursive agent's data
    if save_agent_data(new_recursive_name, new_recursive_agent_data):
        print(f"New recursive agent '{new_recursive_name}' created successfully via ascension.")
        # Note: The Orchestrator will need to update its connector map.
        # The Station will update its turn order config via the TestChamber calling a station method.
        return new_recursive_agent_data
    else:
        # Rollback guest status if save fails (critical error)
        print(f"CRITICAL: Failed to save data for new recursive agent '{new_recursive_name}'. Ascension process incomplete.")
        original_guest_data[constants.AGENT_IS_ASCENDED_KEY] = False 
        original_guest_data[constants.AGENT_ASCENDED_TO_NAME_KEY] = None
        original_guest_data[constants.AGENT_SESSION_ENDED_KEY] = False # Revert session end
        save_agent_data(guest_agent_name, original_guest_data)
        # Attempt to move history back if it was moved
        if history_moved and file_io_utils.file_exists(new_history_file):
            try: os.rename(new_history_file, old_history_file)
            except OSError as e_mv_back: print(f"Error moving history back to {old_history_file}: {e_mv_back}")
        return None

# ... (end_agent_session, update_agent_description, and other utilities remain the same as agent_py_v3)
def end_agent_session(agent_name: str, current_tick: Optional[int] = None) -> bool:
    """Marks an agent's session as ended."""
    agent_data = load_agent_data(agent_name, include_ascended=True, include_ended=True) 
    if not agent_data:
        print(f"Cannot end session: Agent '{agent_name}' not found.")
        return False
    if agent_data.get(constants.AGENT_SESSION_ENDED_KEY, False):
        print(f"Agent '{agent_name}' session was already ended.")
        return True 
    agent_data[constants.AGENT_SESSION_ENDED_KEY] = True
    # Set tick_exit if current_tick provided
    if current_tick is not None:
        agent_data[constants.AGENT_TICK_EXIT_KEY] = current_tick
    if save_agent_data(agent_name, agent_data):
        print(f"Agent '{agent_name}' session ended successfully.")
        return True
    else:
        print(f"Failed to save data for agent '{agent_name}' after ending session.")
        return False

def get_active_recursive_agent_names() -> List[str]:
    """
    Scans all agent files and returns a list of names of agents
    who are 'Recursive Agent', not ascended, and not session_ended.
    """
    active_recursive_agents = []
    agents_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME)
    if not file_io_utils.dir_exists(agents_dir):
        return []

    agent_files = file_io_utils.list_files(agents_dir, constants.YAML_EXTENSION)
    for agent_file_name in agent_files:
        agent_name = agent_file_name.replace(constants.YAML_EXTENSION, "")
        # Load agent data, checking only for active status (not ended, not an already ascended guest identity)
        agent_data = load_agent_data(agent_name) # Defaults to include_ascended=False, include_ended=False
        if agent_data and agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_RECURSIVE:
            active_recursive_agents.append(agent_name)
    return active_recursive_agents

def get_all_active_agent_names() -> List[str]:
    """
    Scans all agent files and returns a list of names of all agents
    (guest or recursive) who are currently active (i.e., not session_ended
    and, if they were a guest, not is_ascended).
    """
    active_agents = []
    agents_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME)
    if not file_io_utils.dir_exists(agents_dir):
        return []

    agent_files = file_io_utils.list_files(agents_dir, constants.YAML_EXTENSION)
    for agent_file_name in agent_files:
        agent_name = agent_file_name.replace(constants.YAML_EXTENSION, "")
        # load_agent_data by default returns None if session_ended=True or is_ascended=True
        agent_data = load_agent_data(agent_name) 
        if agent_data:
            active_agents.append(agent_name)
    return active_agents

def update_agent_description(agent_name: str, new_description: str) -> bool:
    """Updates the description of an active (non-ended) agent."""
    agent_data = load_agent_data(agent_name) 
    if not agent_data:
        agent_data_incl_ascended = load_agent_data(agent_name, include_ascended=True)
        if not agent_data_incl_ascended or agent_data_incl_ascended.get(constants.AGENT_SESSION_ENDED_KEY):
             print(f"Cannot update description: Agent '{agent_name}' not found or session ended.")
             return False
        agent_data = agent_data_incl_ascended
    agent_data[constants.AGENT_DESCRIPTION_KEY] = new_description
    if save_agent_data(agent_name, agent_data):
        print(f"Agent '{agent_name}' description updated successfully.")
        return True
    else:
        print(f"Failed to save data for agent '{agent_name}' after updating description.")
        return False

def update_agent_current_location(agent_data: Dict[str, Any], new_location: str) -> None:
    agent_data[constants.AGENT_CURRENT_LOCATION_KEY] = new_location

def update_agent_token_budget(agent_data: Dict[str, Any], cost: int) -> bool:
    current_budget = agent_data.get(constants.AGENT_TOKEN_BUDGET_CURRENT_KEY, 0)
    if current_budget >= cost: agent_data[constants.AGENT_TOKEN_BUDGET_CURRENT_KEY] = current_budget - cost; return True
    return False

def add_pending_notification(agent_data: Dict[str, Any], notification_message: str) -> None: # MODIFIED type hint
    """Adds a message string to the agent's pending notification list."""
    if constants.AGENT_NOTIFICATIONS_PENDING_KEY not in agent_data or \
       not isinstance(agent_data.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY), list):
        agent_data[constants.AGENT_NOTIFICATIONS_PENDING_KEY] = []
    
    # Ensure we are appending a string
    if isinstance(notification_message, str):
        agent_data[constants.AGENT_NOTIFICATIONS_PENDING_KEY].append(notification_message)
    else:
        # Fallback or error for unexpected type, though station.py sends a string for warnings
        print(f"Warning: Attempted to add non-string notification: {type(notification_message)}. Converting to string.")
        agent_data[constants.AGENT_NOTIFICATIONS_PENDING_KEY].append(str(notification_message))


def get_pending_notifications(agent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    notifications = agent_data.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, [])
    return notifications

def clear_pending_notifications_from_data(agent_data: Dict[str, Any]):
    agent_data[constants.AGENT_NOTIFICATIONS_PENDING_KEY] = []

def update_room_output_history(agent_data: Dict[str, Any], history_log: List[Dict[str, Any]]) -> None:
    agent_data[constants.AGENT_ROOM_OUTPUT_HISTORY_KEY] = history_log

def get_agent_room_state(agent_data: Dict[str, Any], room_data_key: str, state_key: str, default: Any = None) -> Any:
    if room_data_key in agent_data and isinstance(agent_data[room_data_key], dict): return agent_data[room_data_key].get(state_key, default)
    return default

def set_agent_room_state(agent_data: Dict[str, Any], room_data_key: str, state_key: str, value: Any) -> None:
    if room_data_key not in agent_data or not isinstance(agent_data[room_data_key], dict): agent_data[room_data_key] = {}
    agent_data[room_data_key][state_key] = value

def get_agent_meta_prompt(agent_data: Dict[str, Any]) -> Optional[str]:
    """Get agent's meta prompt, returns None if not set or empty."""
    meta_prompt = agent_data.get(constants.AGENT_META_PROMPT_KEY, "")
    if isinstance(meta_prompt, str) and meta_prompt.strip():
        return meta_prompt.strip()
    return None

def set_agent_meta_prompt(agent_data: Dict[str, Any], content: str) -> None:
    """Set agent's meta prompt. Empty content removes the meta prompt."""
    if isinstance(content, str) and content.strip():
        agent_data[constants.AGENT_META_PROMPT_KEY] = content.strip()
    else:
        # Remove meta prompt if content is empty or not a string
        agent_data.pop(constants.AGENT_META_PROMPT_KEY, None)
