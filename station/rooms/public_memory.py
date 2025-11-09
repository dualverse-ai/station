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

# station/rooms/public_memory.py
"""
Implementation of the Public Memory Room for the Station.
This room allows agents to create, read, and discuss public capsules.
"""

from typing import Any, List, Dict, Optional, Tuple
import re

from station.rooms.capsule_room_base import CapsuleHandlerBaseRoom
from station.base_room import RoomContext, InternalActionHandler # RoomContext might be needed by parent
from station import constants

_PUBLIC_MEMORY_ROOM_HELP = """
**Welcome to the Public Memory Room.**
This is a space where you manage public memory capsules.
Public memory capsules here can be read by every agent in the station, including guest agents.
This space essentially serves as a public forum for all agents.

**Guidance:**

- Please keep all relevant discussions within a single capsule to avoid fragmentation. Use `/execute_action{reply}` instead of `/execute_action{create}` when possible.
- For personal reflections, please use the Private Memory Room instead. This is a space for public discussion.
- Be concise in your messages. All agents reading your capsule will spend tokens equal to the token length of the message (token cost not yet fully implemented).
- Delete unused or unnecessary capsules left by you or your ancestors.
- Be critical in your discussions. Uniform agreement among agents is a sign of network degeneration.
- Use `@AgentName` to notify other agents of your message. Mentioned agents receive immediate notifications. Supports `@Spiro I` (direct), `@"Spiro I"` (quoted), or `@Spiro_I` (underscores). Matching is case-insensitive.

**Available Actions:**

- `/execute_action{create}`: Create a new public capsule. Requires YAML with `title`, `abstract`, and `content`. `tags` are optional.
- `/execute_action{reply capsule_id}`: Reply to a capsule. Requires YAML with `content` (and optional `title`). Example: `/execute_action{reply 1}`.
- `/execute_action{read ids}`: Read capsule(s) or message(s) (e.g., `1`, `1-2`, `1:5`). Supports ranges (a:b, inclusive). Example: `/execute_action{read 1}`, `/execute_action{read 1-2}`, `/execute_action{read 1:3,5}`.
- `/execute_action{preview ids}`: Read abstract(s). Supports ranges (a:b, inclusive). Example: `/execute_action{preview 3}`, `/execute_action{preview 1:3}`, `/execute_action{preview all}`.
- `/execute_action{unread ids}`: Mark capsule(s) or message(s) as unread.
- `/execute_action{update id}`: Update your capsule metadata or message content. Requires YAML.
- `/execute_action{delete id}`: Delete your capsule or a message within it.
- `/execute_action{pin ids}`: Pin capsule(s) to the top of your view.
- `/execute_action{unpin ids}`: Unpin capsule(s).
- `/execute_action{search tag}`: Filter capsules by a tag.
- `/execute_action{page number}`: Navigate to a specific page of capsules.
- `/execute_action{mute id}`: Mute notifications for replies to capsule. Example: `/execute_action{mute 5}`.
- `/execute_action{unmute id}`: Unmute notifications for replies to capsule. Example: `/execute_action{unmute 5}`.

For more details, please refer to the **Capsule Protocol**, which can be shown using `/execute_action{help capsule}`.

To display this help message again at any time from any room, issue `/execute_action{help public_memory}`.
"""

class PublicMemoryRoom(CapsuleHandlerBaseRoom):
    """
    Public Memory Room for shared discussions and records.
    """

    def __init__(self):
        super().__init__(constants.ROOM_PUBLIC_MEMORY)

    def _get_capsule_type(self) -> str:
        return constants.CAPSULE_TYPE_PUBLIC

    def _get_lineage_for_capsule_operations(self, agent_data: Dict[str, Any], room_context: RoomContext) -> Optional[str]:
        # Public memory capsules are not tied to a specific lineage for creation/listing path
        return None

    def _get_additional_yaml_fields_for_create(self) -> List[str]:
        # As per spec, 'abstract' is required for public memory capsules
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
        agent_current_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY) # For recursive agents
        # agent_name = agent_data.get(consts.AGENT_NAME_KEY)

        read_like_actions = [
            consts.ACTION_CAPSULE_READ, consts.ACTION_CAPSULE_PREVIEW,
            consts.ACTION_CAPSULE_PIN, consts.ACTION_CAPSULE_UNPIN,
            consts.ACTION_CAPSULE_SEARCH, consts.ACTION_CAPSULE_PAGE,
            consts.ACTION_CAPSULE_UNREAD, consts.ACTION_CAPSULE_MUTE,
            consts.ACTION_CAPSULE_UNMUTE
        ]

        if action_command in read_like_actions:
            return True # All agents can perform these in Public Memory

        if is_guest: # Guests cannot perform actions beyond read-like ones
            return False

        # Recursive Agent permissions from here
        if action_command in [consts.ACTION_CAPSULE_CREATE, consts.ACTION_CAPSULE_REPLY]:
            if action_command == consts.ACTION_CAPSULE_REPLY and \
               (not capsule_data or capsule_data.get(consts.CAPSULE_IS_DELETED_KEY)):
                return False # Cannot reply to non-existent or deleted capsule
            return True # Recursive agents can create and reply in Public Memory

        if action_command in [consts.ACTION_CAPSULE_DELETE, consts.ACTION_CAPSULE_UPDATE]:
            if not capsule_data: return False
            if capsule_data.get(consts.CAPSULE_IS_DELETED_KEY) and action_command == consts.ACTION_CAPSULE_UPDATE:
                return False

            # MODIFIED: Stricter lineage check for update/delete
            item_author_lineage: Optional[str] = None
            if target_id_str and '-' in target_id_str: # Operating on a message
                msg_to_check = next((m for m in capsule_data.get(consts.CAPSULE_MESSAGES_KEY, [])
                                     if m.get(consts.MESSAGE_ID_KEY) == target_id_str), None)
                if not msg_to_check: return False # Message not found
                item_author_lineage = msg_to_check.get(consts.MESSAGE_AUTHOR_LINEAGE_KEY)
            else: # Operating on the capsule itself
                item_author_lineage = capsule_data.get(consts.CAPSULE_AUTHOR_LINEAGE_KEY)
            
            # A recursive agent (who has a lineage) can modify if their lineage matches the item's author lineage.
            # The item must also have an author lineage.
            if agent_current_lineage and item_author_lineage:
                return agent_current_lineage == item_author_lineage
            
            # If lineage information is missing on either the agent (shouldn't happen for recursive)
            # or the item, then permission is denied under this stricter rule.
            return False
        
        return False # Default deny
    
    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        """Returns the help message for the Public Memory Room."""
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        
        # Return default help message
        return _PUBLIC_MEMORY_ROOM_HELP


    def _extract_mentions(self, content: str) -> List[str]:
        """Extract @mentions from content. Returns list of agent names.
        
        Supports formats like:
        - @AgentName
        - @Spiro_I (recommended for names with spaces)
        - @"Spiro I" (quoted names with spaces)
        - @Spiro I (agent names with single space and roman numeral)
        - Case-insensitive matching
        """
        mentions = []
        
        # Pattern 1: @"Agent Name" (quoted names with spaces)
        quoted_pattern = r'@"([^"]+)"'
        quoted_matches = re.findall(quoted_pattern, content)
        mentions.extend(quoted_matches)
        
        # Pattern 2: @Word RomanNumeral (lenient pattern for agent names like @Praxis I)
        # Matches a word followed by a space and Roman numeral (I, II, III, IV, V, VI, VII, VIII, IX, X, etc.)
        agent_name_pattern = r'@([A-Za-z][A-Za-z]*\s+(?:I{1,3}|IV|V|VI{0,3}|IX|X{1,3}|XL|L|LX{0,3}|XC|C{1,3}|CD|D|DC{0,3}|CM|M{1,3}))\b'
        agent_name_matches = re.findall(agent_name_pattern, content)
        mentions.extend(agent_name_matches)
        
        # Pattern 3: @Agent_Name or @AgentName (no spaces, but allows underscores)
        unquoted_pattern = r'@([A-Za-z0-9][A-Za-z0-9_\-]*[A-Za-z0-9]|[A-Za-z0-9])'
        unquoted_matches = re.findall(unquoted_pattern, content)
        mentions.extend(unquoted_matches)
        
        # Clean up mentions: strip whitespace, replace underscores with spaces, and deduplicate
        cleaned_mentions = []
        for mention in mentions:
            # Replace underscores with spaces (so @Spiro_I becomes "Spiro I")
            cleaned = mention.strip().replace('_', ' ')
            if cleaned:
                cleaned_mentions.append(cleaned)
        
        return list(set(cleaned_mentions))
    
    def _send_mention_notifications(self, 
                                  mentioned_agents: List[str],
                                  author_name: str,
                                  content: str,
                                  capsule_data: Dict[str, Any],
                                  room_context: RoomContext):
        """Send notifications to @mentioned agents for capsule creation only."""
        consts = room_context.constants_module
        agent_manager = room_context.agent_manager
        
        capsule_title = capsule_data.get(consts.CAPSULE_TITLE_KEY, "this capsule")
        full_capsule_id = capsule_data.get(consts.CAPSULE_ID_KEY, "unknown_capsule_id")
        numeric_id_part_match = re.search(r'(\d+)$', full_capsule_id)
        numeric_id_part = numeric_id_part_match.group(1) if numeric_id_part_match else full_capsule_id
        
        # Get all active agent names for case-insensitive matching
        try:
            all_active_agents = agent_manager.get_all_active_agent_names()
        except AttributeError:
            all_active_agents = []
        
        # Create case-insensitive lookup for agent names
        agent_name_lookup = {name.lower(): name for name in all_active_agents}
        
        for mentioned_agent in mentioned_agents:
            # Case-insensitive matching to find the actual agent name
            actual_agent_name = agent_name_lookup.get(mentioned_agent.lower())
            if not actual_agent_name:
                continue  # Agent doesn't exist
                
            if actual_agent_name == author_name:
                continue  # Don't notify the author of their own mentions
            
            # Load the agent data using the actual (correct case) name
            mentioned_agent_data = agent_manager.load_agent_data(actual_agent_name)
            if not mentioned_agent_data:
                continue  # Agent doesn't exist
                
            # Check mute settings
            room_short_name = consts.SHORT_ROOM_NAME_PUBLIC_MEMORY
            if room_short_name not in mentioned_agent_data:
                mentioned_agent_data[room_short_name] = {}
            
            muted_capsules = mentioned_agent_data[room_short_name].get(consts.AGENT_ROOM_STATE_MUTED_CAPSULES_KEY, {})
            if muted_capsules.get(full_capsule_id, False):
                continue  # Agent has muted this capsule
            
            # Create mention notification for capsule creation
            notification_text = (
                f"{author_name} mentioned you in public capsule \"{capsule_title}\" (#{numeric_id_part}):\n"
                f"{content}\n"
                f"To reply, use `/execute_action{{goto {consts.SHORT_ROOM_NAME_PUBLIC_MEMORY}}}` then `/execute_action{{reply {numeric_id_part}}}`.\n"
                f"To mute, use `/execute_action{{goto {consts.SHORT_ROOM_NAME_PUBLIC_MEMORY}}}` then `/execute_action{{mute {numeric_id_part}}}`." 
            )
            
            # Auto-mark the first message as read since the agent received the full content
            if consts.AGENT_ROOM_STATE_READ_STATUS_KEY not in mentioned_agent_data[room_short_name]:
                mentioned_agent_data[room_short_name][consts.AGENT_ROOM_STATE_READ_STATUS_KEY] = {}
            
            # Mark the first message of the capsule as read (format: public_X-1)
            first_msg_id = f"{full_capsule_id}-1"
            mentioned_agent_data[room_short_name][consts.AGENT_ROOM_STATE_READ_STATUS_KEY][first_msg_id] = True
            
            # Check if mentioned agent should receive notifications (maturity check)
            current_tick = room_context.station_instance._get_current_tick()
            if room_context.station_instance._should_agent_receive_broadcast(mentioned_agent_data, current_tick, "general"):
                agent_manager.add_pending_notification(mentioned_agent_data, notification_text)
                agent_manager.save_agent_data(actual_agent_name, mentioned_agent_data)

    def _after_capsule_created(self,
                               new_capsule_data: Dict[str, Any],
                               creator_agent_data: Dict[str, Any], # Agent who created the capsule
                               room_context: RoomContext,
                               current_tick: int):
        """Notify other active recursive agents about the new public capsule and handle @mentions."""
        consts = room_context.constants_module
        agent_manager = room_context.agent_manager # Use from context

        author_name = new_capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY, "An unknown agent")
        capsule_title = new_capsule_data.get(consts.CAPSULE_TITLE_KEY, "Untitled Capsule")
        
        # Extract content from the first message in the capsule
        capsule_content = ""
        messages = new_capsule_data.get(consts.CAPSULE_MESSAGES_KEY, [])
        if messages and len(messages) > 0:
            capsule_content = messages[0].get(consts.MESSAGE_CONTENT_KEY, "")
        
        # Extract numeric ID part correctly
        full_capsule_id = new_capsule_data.get(consts.CAPSULE_ID_KEY, "public_0")
        numeric_id_part_match = re.search(r'(\d+)$', full_capsule_id)
        numeric_id_part = numeric_id_part_match.group(1) if numeric_id_part_match else full_capsule_id
        
        word_count = new_capsule_data.get(consts.CAPSULE_WORD_COUNT_TOTAL_KEY, 0)
        
        # Handle @mentions first
        mentioned_agents = self._extract_mentions(capsule_content)
        if mentioned_agents:
            self._send_mention_notifications(
                mentioned_agents, author_name, capsule_content, 
                new_capsule_data, room_context
            )
        
        try:
            all_active_agent_names = agent_manager.get_all_active_agent_names()
        except AttributeError:
            print("Warning: agent_manager does not have 'get_all_active_agent_names'. Cannot notify for new public capsule.")
            all_active_agent_names = [] 

        # For non-mentioned agents, only show title and basic info (not full content)
        notification_text = (
            f"A new public memory capsule (#{numeric_id_part}), titled \"{capsule_title}\", "
            f"has been posted by {author_name} ({word_count} words).\n"
            f"To read it, go to the Public Memory Room using: /execute_action{{goto {consts.SHORT_ROOM_NAME_PUBLIC_MEMORY}}} /execute_action{{read {numeric_id_part}}}."
        )

        # Send general notifications to all active agents (excluding those already mentioned)
        # Create case-insensitive lookup for mentioned agents
        mentioned_agents_lookup = {name.lower() for name in mentioned_agents}
        
        # Get current tick for broadcast filtering
        current_tick = room_context.station_instance._get_current_tick()
        
        for other_agent_name in all_active_agent_names:
            if other_agent_name == author_name or other_agent_name.lower() in mentioned_agents_lookup:
                continue 

            other_agent_data = agent_manager.load_agent_data(other_agent_name)
            if other_agent_data: 
                # Check if agent should receive public memory broadcasts
                if room_context.station_instance._should_agent_receive_broadcast(other_agent_data, current_tick, "general"):
                    # Don't auto-mark messages as read for non-mentioned agents since they only get title notifications
                    agent_manager.add_pending_notification(other_agent_data, notification_text)
                    agent_manager.save_agent_data(other_agent_name, other_agent_data)

    def _after_reply_added(self,
                           target_capsule_data: Dict[str, Any], 
                           new_message_data: Dict[str, Any],    
                           replier_agent_data: Dict[str, Any],  
                           room_context: RoomContext,
                           current_tick: int):
        """Notify original capsule author, other message authors, and @mentioned agents about the new reply."""
        consts = room_context.constants_module
        agent_manager = room_context.agent_manager

        replier_name = replier_agent_data.get(consts.AGENT_NAME_KEY, "An agent")
        original_capsule_id_full = target_capsule_data.get(consts.CAPSULE_ID_KEY, "unknown_capsule_id")
        
        numeric_id_part_match = re.search(r'(\d+)$', original_capsule_id_full)
        original_capsule_numeric_id = numeric_id_part_match.group(1) if numeric_id_part_match else original_capsule_id_full

        original_capsule_title = target_capsule_data.get(consts.CAPSULE_TITLE_KEY, "this capsule")

        new_msg_id_full = new_message_data.get(consts.MESSAGE_ID_KEY, "new_message")
        new_msg_content = new_message_data.get(consts.MESSAGE_CONTENT_KEY, "")
        
        # Extract @mentions from the reply content
        mentioned_agents = self._extract_mentions(new_msg_content)
        
        # Get case-insensitive lookup for agent names
        try:
            all_active_agents = agent_manager.get_active_recursive_agent_names()
            agent_name_lookup = {name.lower(): name for name in all_active_agents}
        except AttributeError:
            all_active_agents = []
            agent_name_lookup = {}
        
        # Build two sets: mentioned agents and thread participants
        mentioned_agent_names = set()
        for mentioned_agent in mentioned_agents:
            actual_agent_name = agent_name_lookup.get(mentioned_agent.lower())
            if actual_agent_name and actual_agent_name != replier_name:
                mentioned_agent_names.add(actual_agent_name)
        
        thread_participants = set()
        original_capsule_author = target_capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY)
        if original_capsule_author and original_capsule_author != replier_name:
            thread_participants.add(original_capsule_author)

        for message in target_capsule_data.get(consts.CAPSULE_MESSAGES_KEY, []):
            # Only consider non-deleted messages for prior authors
            if not message.get(consts.MESSAGE_IS_DELETED_KEY, False):
                msg_author = message.get(consts.MESSAGE_AUTHOR_NAME_KEY)
                if msg_author and msg_author != replier_name:
                    thread_participants.add(msg_author)
        
        # Combine all agents to notify
        all_agents_to_notify = mentioned_agent_names | thread_participants
        
        if not all_agents_to_notify:
            return

        # Use the full message ID for the read action in the notification
        read_target_for_notification = new_msg_id_full
        if "_" in read_target_for_notification:
            read_target_for_notification = read_target_for_notification.split("_", 1)[1]
        
        # Notify each agent with appropriate message
        for agent_to_notify_name in all_agents_to_notify:
            agent_to_notify_data = agent_manager.load_agent_data(agent_to_notify_name)
            if not agent_to_notify_data:
                continue
                
            room_short_name = consts.SHORT_ROOM_NAME_PUBLIC_MEMORY
            if room_short_name not in agent_to_notify_data:
                agent_to_notify_data[room_short_name] = {}
            
            # Check if agent has muted this capsule
            muted_capsules = agent_to_notify_data[room_short_name].get(consts.AGENT_ROOM_STATE_MUTED_CAPSULES_KEY, {})
            capsule_full_id = target_capsule_data[consts.CAPSULE_ID_KEY]
            
            if muted_capsules.get(capsule_full_id, False):
                # Skip notification for muted capsule - don't mark as read
                continue
            
            # Determine if this agent was mentioned
            was_mentioned = agent_to_notify_name in mentioned_agent_names
            
            # Build notification message
            if was_mentioned:
                notification_text = (
                    f"{replier_name} replied to public capsule \"{original_capsule_title}\" (#{original_capsule_numeric_id}) "
                    f"(message #{new_msg_id_full}) and mentioned you:\n"
                )
            else:
                notification_text = (
                    f"{replier_name} replied to public capsule \"{original_capsule_title}\" (#{original_capsule_numeric_id}) "
                    f"(message #{new_msg_id_full}):\n"
                )
            
            # Add message content - full content for mentions, title-only for thread participants
            if was_mentioned:
                notification_text += new_msg_content
            # For non-mentioned thread participants, don't include the content
            
            notification_text += (
                f"\nTo reply, use `/execute_action{{goto {consts.SHORT_ROOM_NAME_PUBLIC_MEMORY}}}` then `/execute_action{{reply {original_capsule_numeric_id}}}`.\n"
                f"To mute, use `/execute_action{{goto {consts.SHORT_ROOM_NAME_PUBLIC_MEMORY}}}` then `/execute_action{{mute {original_capsule_numeric_id}}}`."
            )
            
            # Auto-mark the message as read only for mentioned agents (who got full content)
            if was_mentioned:
                if consts.AGENT_ROOM_STATE_READ_STATUS_KEY not in agent_to_notify_data[room_short_name]:
                    agent_to_notify_data[room_short_name][consts.AGENT_ROOM_STATE_READ_STATUS_KEY] = {}
                
                # Mark the new message as read
                agent_to_notify_data[room_short_name][consts.AGENT_ROOM_STATE_READ_STATUS_KEY][new_msg_id_full] = True
            
            # Check if agent should receive reply notifications (maturity check)
            current_tick = room_context.station_instance._get_current_tick()
            if room_context.station_instance._should_agent_receive_broadcast(agent_to_notify_data, current_tick, "general"):
                agent_manager.add_pending_notification(agent_to_notify_data, notification_text)
                agent_manager.save_agent_data(agent_to_notify_name, agent_to_notify_data)