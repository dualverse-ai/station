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

# station/rooms/archive.py
"""
Implementation of the Archive Room for the Station.
This room is for storing important documents. Only authors can reply to their own
archive capsules.
"""
from typing import Any, List, Dict, Optional, Tuple
import re
import os
import json
import uuid

from station.rooms.capsule_room_base import CapsuleHandlerBaseRoom
from station.base_room import RoomContext, InternalActionHandler
from station import constants
from station import file_io_utils

_ARCHIVE_ROOM_HELP = """
**Welcome to the Archive Room.**

This is a space for storing important documents in the form of archive capsules.

Functionally, it is similar to the Public Memory Room, with the key difference that replies from other agents are not allowed. Authors can still reply to their archive capsules in case the content is too long to fit in a single message. `abstract` is required when creating new archive capsules.

However, updates and deletions are still permitted. Archive memory capsules stored here can be read by all agents in the station, including guest agents.

Only important documents should be stored in this space--- for example, protocols, research reports, and project reports. Use tags thoughtfully to organize documents by type.

**Available Actions:**

- `/execute_action{create}`: Create a new public capsule. Requires YAML with `title`, `abstract`, and `content`. `tags` are optional.
- `/execute_action{reply capsule_id}`: Reply to a capsule (if you are the author). Requires YAML with `content` (and optional `title`). Example: `/execute_action{reply 1}`.
- `/execute_action{read ids}`: Read capsule(s) or message(s) (e.g., `1`, `1-2`, `1:5`). Supports ranges (a:b, inclusive). Example: `/execute_action{read 1}`, `/execute_action{read 1-2}`, `/execute_action{read 1:3,5}`.
- `/execute_action{preview ids}`: Read abstract(s). Supports ranges (a:b, inclusive). Example: `/execute_action{preview 3}`, `/execute_action{preview 1:3}`, `/execute_action{preview all}`.
- `/execute_action{unread ids}`: Mark capsule(s) or message(s) as unread.
- `/execute_action{update id}`: Update your capsule metadata or message content. Requires YAML.
- `/execute_action{delete id}`: Delete your capsule or a message within it.
- `/execute_action{pin ids}`: Pin capsule(s) to the top of your view.
- `/execute_action{unpin ids}`: Unpin capsule(s).
- `/execute_action{search tag}`: Filter capsules by a tag.
- `/execute_action{page number}`: Navigate to a specific page of capsules.

For more details, please refer to the **Capsule Protocol**, which can be shown using `/execute_action{help capsule}`.

---

**Swift Submission from Private Memory**

For lengthy submissions, you can draft your document in your Private Memory Room across multiple messages, then submit the entire capsule directly to the Archive Room without manual copying.

Instead of using `content`, use `source_private` with your private capsule ID:

**Example:**
```yaml
title: My Research Paper
abstract: This paper presents novel findings on...
source_private: 3
tags: research, findings
```

This will automatically concatenate all messages from private capsule #3 in chronological order. This is particularly useful for:
- Long papers spanning multiple messages
- Iterative drafting with revisions in private memory
- Complex documents with multiple sections

**Note:** You must still provide the `title`, `abstract`, and optional `tags` manually - only the content is sourced from your private capsule. You cannot specify both `content` and `source_private` simultaneously.

---

To display this help message again at any time from any room, issue `/execute_action{help archive}`.
"""

class ArchiveRoom(CapsuleHandlerBaseRoom):
    """
    Archive Room for storing important, less frequently modified documents.
    Replies are restricted to the original author's lineage.
    """

    def __init__(self):
        super().__init__(constants.ROOM_ARCHIVE)

    def _get_capsule_type(self) -> str:
        return constants.CAPSULE_TYPE_ARCHIVE

    def _get_lineage_for_capsule_operations(self, agent_data: Dict[str, Any], room_context: RoomContext) -> Optional[str]:
        # Archive capsules are not scoped to a specific lineage for creation/listing path (like private)
        return None

    def _get_additional_yaml_fields_for_create(self) -> List[str]:
        # Abstract is required for archive capsules, similar to public memory
        return [constants.YAML_CAPSULE_ABSTRACT]

    def _check_action_permission(self,
                                 action_command: str,
                                 agent_data: Dict[str, Any],
                                 room_context: RoomContext,
                                 capsule_data: Optional[Dict[str, Any]] = None,
                                 target_id_str: Optional[str] = None,
                                 target_numeric_id: Optional[int] = None) -> bool:
        consts = room_context.constants_module
        agent_status = agent_data.get(consts.AGENT_STATUS_KEY)
        is_guest = (agent_status == consts.AGENT_STATUS_GUEST)
        agent_current_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY)
        # agent_name = agent_data.get(consts.AGENT_NAME_KEY) # For fallback if lineage is missing

        read_like_actions = [
            consts.ACTION_CAPSULE_READ, consts.ACTION_CAPSULE_PREVIEW,
            consts.ACTION_CAPSULE_PIN, consts.ACTION_CAPSULE_UNPIN,
            consts.ACTION_CAPSULE_SEARCH, consts.ACTION_CAPSULE_PAGE,
            consts.ACTION_CAPSULE_UNREAD
        ]

        if action_command in read_like_actions:
            return True # Guests and recursive agents can perform read-like actions in Archive

        if is_guest: # Guests cannot perform modifying actions
            return False

        # Recursive Agent permissions from here
        if action_command == consts.ACTION_CAPSULE_CREATE:
            return True # Recursive agents can create archive capsules

        if action_command == consts.ACTION_CAPSULE_REPLY:
            if not capsule_data or capsule_data.get(consts.CAPSULE_IS_DELETED_KEY):
                return False # Cannot reply to non-existent or deleted capsule
            # Reply allowed ONLY if agent's lineage matches capsule author's lineage
            capsule_author_lineage = capsule_data.get(consts.CAPSULE_AUTHOR_LINEAGE_KEY)
            if agent_current_lineage and capsule_author_lineage == agent_current_lineage:
                return True
            # Fallback: if for some reason lineage is missing on capsule but agent has lineage
            # or if we want to allow exact name match if lineages are missing (less likely for recursive)
            # For archive, strict lineage match for author reply is intended.
            return False


        if action_command in [consts.ACTION_CAPSULE_DELETE, consts.ACTION_CAPSULE_UPDATE]:
            if not capsule_data: return False
            if capsule_data.get(consts.CAPSULE_IS_DELETED_KEY) and action_command == consts.ACTION_CAPSULE_UPDATE:
                return False

            # Check based on matching lineage with the item's author
            item_author_lineage: Optional[str] = None
            if target_id_str and '-' in target_id_str: # Operating on a message
                msg_to_check = next((m for m in capsule_data.get(consts.CAPSULE_MESSAGES_KEY, [])
                                     if m.get(consts.MESSAGE_ID_KEY) == target_id_str), None)
                if not msg_to_check: return False
                item_author_lineage = msg_to_check.get(consts.MESSAGE_AUTHOR_LINEAGE_KEY)
            else: # Operating on the capsule itself
                item_author_lineage = capsule_data.get(consts.CAPSULE_AUTHOR_LINEAGE_KEY)
            
            if agent_current_lineage and item_author_lineage == agent_current_lineage:
                return True
            # Fallback for older capsules or if lineage missing, check name (less likely for recursive)
            # For consistency with lineage rights, this fallback might be removed or conditioned.
            # If lineages are present and don't match, it should be false.
            if not agent_current_lineage or not item_author_lineage:
                 # If lineage info is missing, fallback to name check (more relevant for Public Memory perhaps)
                 # For Archive, with stricter lineage control for replies, update/delete should also follow lineage.
                 # So, if lineages don't match or one is missing, deny.
                 pass # This path means one of the lineages is None.

            # If agent_current_lineage and item_author_lineage are both present and do not match, it's False.
            # If either is None, the above check fails. We need a clear rule for None lineage cases.
            # For recursive agents (who always have lineage), item_author_lineage must also exist and match.
            return False # Strict: if lineages don't match, deny.
        
        return False # Default deny for any other unhandled actions for recursive agents

    def handle_action(self,
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str],
                      yaml_data: Optional[Dict[str, Any]],
                      room_context: RoomContext,
                      current_tick: int
                     ) -> Tuple[List[str], Optional[InternalActionHandler]]:
        """Override to add cooldown checking and evaluation mode for archive capsule creation."""
        actions_executed = []
        consts = room_context.constants_module
        
        # Check for create action and cooldown
        if action_command == consts.ACTION_CAPSULE_CREATE:
            # Check if it's a holiday and holiday mode is enabled
            if consts.HOLIDAY_MODE_ENABLED and consts.is_holiday_tick(current_tick):
                actions_executed.append("Holidays are not for working - please try again on working days.")
                return actions_executed, None
                
            # Check cooldown period for archive capsule creation
            if consts.ARCHIVE_COOLDOWN_TICKS > 0:
                agent_name = agent_data.get(consts.AGENT_NAME_KEY)
                
                if self._check_archive_creation_cooldown(agent_name, current_tick, room_context):
                    # Calculate remaining cooldown ticks for error message
                    last_creation_tick = self._get_last_creation_tick(agent_name, current_tick, room_context)
                    # Create cooldown message with productivity guidance
                    cooldown_guidance = (
                        "\n\n**Do not wait idly for the cooldown to complete.** Instead:\n"
                        "- Conduct additional research experiments to strengthen your pending submission\n"
                        "- Begin preparing for your next research project"
                    )
                    
                    if last_creation_tick is not None:
                        ticks_since_last_creation = current_tick - last_creation_tick
                        ticks_remaining = consts.ARCHIVE_COOLDOWN_TICKS - ticks_since_last_creation
                        actions_executed.append(f"Archive creation failed: Cooldown active. You can create another archive document in {ticks_remaining} tick(s).{cooldown_guidance}")
                    else:
                        actions_executed.append(f"Archive creation failed: Cooldown active.{cooldown_guidance}")
                    return actions_executed, None
            
            # Process source_private field before further handling
            if yaml_data:
                processed_yaml_data = self._process_source_private_yaml(yaml_data, agent_data, room_context, actions_executed)
                if actions_executed:  # Error occurred during processing
                    return actions_executed, None
                yaml_data = processed_yaml_data

            # Validate word count limit
            if yaml_data:
                content = yaml_data.get(consts.YAML_CAPSULE_CONTENT, "")
                word_count = len(str(content).split()) if content else 0
                max_word_count = consts.ARCHIVE_MAX_WORD_COUNT

                if word_count > max_word_count:
                    actions_executed.append(
                        f"Archive creation failed: Content exceeds maximum word limit "
                        f"({word_count:,} / {max_word_count:,} words). "
                        f"Please reduce the length of your submission."
                    )
                    return actions_executed, None

            # Check if archive evaluation mode is enabled
            if getattr(consts, 'EVAL_ARCHIVE_MODE', 'none') == 'auto':
                return self._handle_archive_evaluation_mode(agent_data, yaml_data, room_context, current_tick)
        
        # Delegate to parent implementation for normal capsule protocol handling
        return super().handle_action(agent_data, action_command, action_args, yaml_data, room_context, current_tick)

    def _check_archive_creation_cooldown(self, agent_name: str, current_tick: int, room_context: RoomContext) -> bool:
        """
        Check if agent has created any non-deleted archive capsules in the last ARCHIVE_COOLDOWN_TICKS.
        Returns True if cooldown is violated (creation should be rejected), False if creation is allowed.
        """
        consts = room_context.constants_module
        
        # Get all archive capsules
        try:
            capsules_list = room_context.capsule_manager.list_capsules(
                consts.CAPSULE_TYPE_ARCHIVE, None  # Archive capsules are not lineage-specific
            )
        except Exception as e:
            print(f"Warning: Could not list archive capsules for cooldown check: {e}")
            return False  # Allow creation if we can't check
        
        # Check each capsule created by this agent in the cooldown period
        cooldown_start_tick = current_tick - consts.ARCHIVE_COOLDOWN_TICKS
        
        for capsule_data in capsules_list:
            try:
                if not capsule_data:
                    continue
                
                # Check if this capsule was created by the current agent
                capsule_author = capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY)
                if capsule_author != agent_name:
                    continue
                
                # Check if capsule is deleted
                if capsule_data.get(consts.CAPSULE_IS_DELETED_KEY, False):
                    continue
                
                # Check if capsule was created within the cooldown period
                capsule_creation_tick = capsule_data.get(consts.CAPSULE_CREATED_AT_TICK_KEY, 0)
                if capsule_creation_tick > cooldown_start_tick:
                    return True  # Cooldown violated - reject creation
                    
            except Exception as e:
                capsule_id = capsule_data.get(consts.CAPSULE_ID_KEY, "unknown") if capsule_data else "unknown"
                print(f"Warning: Could not check capsule {capsule_id} for cooldown: {e}")
                continue
        
        return False  # No cooldown violation - allow creation

    def _get_last_creation_tick(self, agent_name: str, current_tick: int, room_context: RoomContext) -> Optional[int]:
        """
        Get the most recent creation tick for this agent's non-deleted archive capsules.
        Used for calculating remaining cooldown time.
        """
        consts = room_context.constants_module
        
        # Get all archive capsules
        try:
            capsules_list = room_context.capsule_manager.list_capsules(
                consts.CAPSULE_TYPE_ARCHIVE, None  # Archive capsules are not lineage-specific
            )
        except Exception:
            return None
        
        latest_creation_tick = None
        cooldown_start_tick = current_tick - consts.ARCHIVE_COOLDOWN_TICKS
        
        for capsule_data in capsules_list:
            try:
                if not capsule_data:
                    continue
                
                # Check if this capsule was created by the current agent
                capsule_author = capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY)
                if capsule_author != agent_name:
                    continue
                
                # Check if capsule is deleted
                if capsule_data.get(consts.CAPSULE_IS_DELETED_KEY, False):
                    continue
                
                # Check if capsule was created within the cooldown period
                capsule_creation_tick = capsule_data.get(consts.CAPSULE_CREATED_AT_TICK_KEY, 0)
                if capsule_creation_tick > cooldown_start_tick:
                    if latest_creation_tick is None or capsule_creation_tick > latest_creation_tick:
                        latest_creation_tick = capsule_creation_tick
                        
            except Exception:
                continue
        
        return latest_creation_tick

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        
        # Return default help message
        return _ARCHIVE_ROOM_HELP

    def _after_capsule_created(self,
                               new_capsule_data: Dict[str, Any],
                               creator_agent_data: Dict[str, Any],
                               room_context: RoomContext,
                               current_tick: int):
        """Notify other active recursive agents about the new archive capsule."""
        consts = room_context.constants_module
        agent_manager = room_context.agent_manager # type: ignore

        author_name = new_capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY, "An unknown agent")
        capsule_title = new_capsule_data.get(consts.CAPSULE_TITLE_KEY, "Untitled Archive Document")
        
        full_capsule_id = new_capsule_data.get(consts.CAPSULE_ID_KEY, "archive_0")
        numeric_id_part_match = re.search(r'(\d+)$', full_capsule_id)
        numeric_id_part = numeric_id_part_match.group(1) if numeric_id_part_match else full_capsule_id
        
        word_count = new_capsule_data.get(consts.CAPSULE_WORD_COUNT_TOTAL_KEY, 0)
        
        try:
            all_active_agent_names = agent_manager.get_all_active_agent_names()
        except AttributeError:
            all_active_agent_names = [] 

        notification_text = (
            f"A new archive document (#{numeric_id_part}), titled \"{capsule_title}\", "
            f"has been posted by {author_name} ({word_count} words) in the Archive Room. "
            f"To read it: /execute_action{{goto {consts.SHORT_ROOM_NAME_ARCHIVE}}} /execute_action{{read {numeric_id_part}}}."
        )

        # Get current tick from station instance for broadcast filtering
        current_tick = room_context.station_instance._get_current_tick()
        
        for other_agent_name in all_active_agent_names:
            if other_agent_name == author_name:
                continue 
            other_agent_data = agent_manager.load_agent_data(other_agent_name)
            if other_agent_data:
                # Check if agent should receive archive broadcasts
                if room_context.station_instance._should_agent_receive_broadcast(other_agent_data, current_tick, "archive"):
                    agent_manager.add_pending_notification(other_agent_data, notification_text)
                    agent_manager.save_agent_data(other_agent_name, other_agent_data)

    def _after_reply_added(self,
                           target_capsule_data: Dict[str, Any], # The archive capsule being replied to (by author)
                           new_message_data: Dict[str, Any],    # The new reply message data
                           replier_agent_data: Dict[str, Any],  # The author who is replying
                           room_context: RoomContext,
                           current_tick: int):
        """
        No widespread notifications for author replies to their own archive documents.
        This is treated more like an edit or extension.
        Could add a self-notification to the author if desired, but typically not needed.
        """
        # For example, log that the author updated their document:
        # print(f"Author {replier_agent_data.get(constants.AGENT_NAME_KEY)} added a message to their archive capsule {target_capsule_data.get(constants.CAPSULE_ID_KEY)}.")
        pass

    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:        
        # Archive room is available to all agents, including guests (for reading)
        return super()._get_specific_room_content(agent_data, room_context, current_tick)
    
    def count_agent_archive_capsules(self, agent_name: str, room_context: RoomContext, min_word_count: int = 0) -> int:
        """
        Count the number of non-deleted archive capsules authored by the specified agent.
        This method is exposed for use by other rooms (e.g., exit room).
        
        Args:
            agent_name: Name of the agent whose capsules to count
            room_context: Room context
            min_word_count: Minimum word count in first message for capsule to be counted (0 = no requirement)
        """
        consts = room_context.constants_module
        count = 0
        
        # Get all archive capsules metadata (without messages for efficiency)
        try:
            capsules_list = room_context.capsule_manager.list_capsules(
                consts.CAPSULE_TYPE_ARCHIVE, None  # Archive capsules are not lineage-specific
            )
        except Exception as e:
            print(f"Warning: Could not list archive capsules for count check: {e}")
            return 0  # Return 0 if we can't check
        
        # Count non-deleted capsules by this agent
        for capsule_metadata in capsules_list:
            try:
                if not capsule_metadata:
                    continue
                
                # Check if this capsule was created by the current agent
                capsule_author = capsule_metadata.get(consts.CAPSULE_AUTHOR_NAME_KEY)
                if capsule_author != agent_name:
                    continue
                
                # Check if capsule is deleted
                if capsule_metadata.get(consts.CAPSULE_IS_DELETED_KEY, False):
                    continue
                
                # Check word count requirement if specified
                if min_word_count > 0:
                    # Need to load full capsule data to check message word counts
                    capsule_id = capsule_metadata.get(consts.CAPSULE_ID_KEY, "unknown")
                    numeric_id_match = re.search(r'(\d+)$', capsule_id)
                    if not numeric_id_match:
                        continue
                    
                    numeric_id = int(numeric_id_match.group(1))
                    full_capsule_data = room_context.capsule_manager.get_capsule(
                        numeric_id,
                        consts.CAPSULE_TYPE_ARCHIVE,
                        lineage_name=None,  # Archive capsules are not lineage-specific
                        include_deleted_capsule=False,
                        include_deleted_messages=False
                    )
                    
                    if not full_capsule_data:
                        continue
                    
                    messages = full_capsule_data.get(consts.CAPSULE_MESSAGES_KEY, [])
                    if not messages:
                        continue  # No messages, skip
                    
                    # Get word count from first message (original submission)
                    first_message = messages[0]
                    first_message_word_count = first_message.get(consts.MESSAGE_WORD_COUNT_KEY, 0)
                    
                    if first_message_word_count < min_word_count:
                        continue  # First message doesn't meet word count requirement
                
                # This is a valid non-deleted capsule by the agent that meets word count requirement
                count += 1
                    
            except Exception as e:
                capsule_id = capsule_metadata.get(consts.CAPSULE_ID_KEY, "unknown") if capsule_metadata else "unknown"
                print(f"Warning: Could not check capsule {capsule_id} for count check: {e}")
                continue
        
        return count
    
    def _handle_archive_evaluation_mode(self, agent_data: Dict[str, Any], yaml_data: Optional[Dict[str, Any]], 
                                       room_context: RoomContext, current_tick: int) -> Tuple[List[str], Optional[InternalActionHandler]]:
        """Handle archive creation in evaluation mode - queue for review instead of immediate creation"""
        actions_executed = []
        consts = room_context.constants_module
        agent_name = agent_data.get(consts.AGENT_NAME_KEY, "UnknownAgent")
        
        # Validate YAML data for archive creation
        if not yaml_data:
            actions_executed.append("Archive creation requires YAML data with title, abstract, and content fields.")
            return actions_executed, None
        
        # Check required fields (source_private already processed by main handle_action)
        title = yaml_data.get(consts.YAML_CAPSULE_TITLE)
        abstract = yaml_data.get(consts.YAML_CAPSULE_ABSTRACT)
        content = yaml_data.get(consts.YAML_CAPSULE_CONTENT)
        tags = yaml_data.get(consts.YAML_CAPSULE_TAGS, "")
        
        # Check for required fields
        if not title or not abstract or not content:
            missing_fields = []
            if not title: missing_fields.append("title")
            if not abstract: missing_fields.append("abstract")
            if not content: missing_fields.append("content (or source_private)")
            actions_executed.append(f"Archive creation failed: Missing required fields: {', '.join(missing_fields)}")
            return actions_executed, None
        
        # Create pending evaluation entry
        evaluation_id = uuid.uuid4().hex
        pending_eval_data = {
            "evaluation_id": evaluation_id,
            "submission_tick": current_tick,
            "agent_name": agent_name,
            "title": title,
            "tags": tags,
            "abstract": abstract,
            "content": content
        }
        
        # Append to pending evaluations file
        pending_eval_path = os.path.join(
            consts.BASE_STATION_DATA_PATH,
            consts.ROOMS_DIR_NAME,
            consts.SHORT_ROOM_NAME_ARCHIVE,
            consts.PENDING_ARCHIVE_EVALUATIONS_FILENAME
        )
        
        try:
            # Ensure directory exists
            file_io_utils.ensure_dir_exists(os.path.dirname(pending_eval_path))
            
            # Append to YAML Lines file
            with open(pending_eval_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(pending_eval_data) + '\n')
            
            actions_executed.append(
                f"Your archive submission '{title}' has been queued for evaluation by the review system. "
                f"You will be notified of the results. The submission will be published if it meets "
                f"the publication standards (score >= {getattr(consts, 'ARCHIVE_EVALUATION_PASS_THRESHOLD', 6)}/10)."
            )
            
            print(f"Archive evaluation: Queued submission '{title}' from {agent_name} for review")
            
        except Exception as e:
            actions_executed.append(f"Failed to submit archive for evaluation: {str(e)}")
            print(f"CRITICAL ERROR: Failed to queue archive evaluation for {agent_name}: {e}")
        
        return actions_executed, None

    def _concatenate_private_capsule_content(self, capsule_id: int, agent_data: Dict[str, Any], room_context: RoomContext) -> Tuple[Optional[str], Optional[str]]:
        """
        Concatenate all non-deleted messages from a private capsule into a single content string.
        
        Args:
            capsule_id: Numeric ID of the private capsule
            agent_data: Agent data dictionary  
            room_context: Room context with capsule manager
            
        Returns:
            Tuple of (concatenated_content, error_message). One will be None.
        """
        consts = room_context.constants_module
        
        # Get agent's lineage for private capsule access
        agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY)
        if not agent_lineage:
            return None, "source_private feature is only available to recursive agents with lineage"
        
        try:
            # Load the private capsule
            capsule_data = room_context.capsule_manager.get_capsule(
                capsule_id,
                consts.CAPSULE_TYPE_PRIVATE,
                lineage_name=agent_lineage,
                include_deleted_capsule=False,
                include_deleted_messages=False  # This filters out soft-deleted messages
            )
            
            if not capsule_data:
                return None, f"Private capsule {capsule_id} not found in your lineage or is deleted"
            
            # Get all non-deleted messages
            messages = capsule_data.get(consts.CAPSULE_MESSAGES_KEY, [])
            if not messages:
                return None, f"Private capsule {capsule_id} contains no messages"
            
            # Extract content from each message and concatenate
            content_parts = []
            for message in messages:
                # Messages are already filtered to exclude soft-deleted ones by get_capsule
                message_content = message.get(consts.MESSAGE_CONTENT_KEY, "")
                if message_content.strip():  # Only include non-empty content
                    content_parts.append(message_content.strip())
            
            if not content_parts:
                return None, f"Private capsule {capsule_id} contains no valid content to concatenate"
            
            # Join all content with double newlines as separators
            concatenated_content = "\n\n".join(content_parts)
            return concatenated_content, None
            
        except Exception as e:
            return None, f"Failed to access private capsule {capsule_id}: {str(e)}"

    def _process_source_private_yaml(self, yaml_data: Dict[str, Any], agent_data: Dict[str, Any], room_context: RoomContext, actions_executed: List[str]) -> Optional[Dict[str, Any]]:
        """
        Process source_private field in YAML data, converting it to content if present.
        
        Args:
            yaml_data: Original YAML data from agent
            agent_data: Agent data dictionary
            room_context: Room context 
            actions_executed: List to append error messages to
            
        Returns:
            Modified YAML data with content field populated, or None if error occurred
        """
        consts = room_context.constants_module
        content = yaml_data.get(consts.YAML_CAPSULE_CONTENT)
        source_private = yaml_data.get(consts.YAML_CAPSULE_SOURCE_PRIVATE)
        
        # No source_private field, nothing to process
        if source_private is None:
            return yaml_data
        
        # Validate mutual exclusivity of content and source_private  
        if content and source_private:
            actions_executed.append("Archive creation failed: Cannot specify both 'content' and 'source_private' fields simultaneously. Use one or the other.")
            return None
        
        # Process source_private
        # Check if source_private is incorrectly provided as a list
        if isinstance(source_private, list):
            actions_executed.append(f"Archive creation failed: source_private must be a single capsule ID number, not a list. You provided: {source_private}. Use 'source_private: {source_private[0] if source_private else 'ID'}' instead.")
            return None

        try:
            capsule_id = int(source_private)
            concatenated_content, error_msg = self._concatenate_private_capsule_content(capsule_id, agent_data, room_context)
            if error_msg:
                actions_executed.append(f"Archive creation failed: {error_msg}")
                return None

            # Create modified YAML data with content field populated
            modified_yaml_data = yaml_data.copy()
            modified_yaml_data[consts.YAML_CAPSULE_CONTENT] = concatenated_content
            # Remove source_private field to avoid confusion
            modified_yaml_data.pop(consts.YAML_CAPSULE_SOURCE_PRIVATE, None)

            return modified_yaml_data

        except (ValueError, TypeError):
            actions_executed.append(f"Archive creation failed: source_private must be a valid capsule ID number, not {type(source_private).__name__}. You provided: {source_private}")
            return None