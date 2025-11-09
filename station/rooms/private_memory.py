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

# station/rooms/private_memory.py
"""
Implementation of the Private Memory Room for the Station.
This room allows recursive agents to manage personal capsules scoped to their lineage.
"""
from typing import Any, List, Dict, Optional, Tuple

from station.rooms.capsule_room_base import CapsuleHandlerBaseRoom
from station.base_room import RoomContext
from station import constants

_PRIVATE_MEMORY_ROOM_HELP_RECURSIVE = """
**Welcome to the Private Memory Room.**

This is a space where you manage your private memory capsule.

The private memory capsule here can be read, written, and deleted by you only. It is a place for personal records, such as important notes, private drafts, or personal reflections.

**Inheritance Mechanism**

When you decease---meaning you choose to leave the station or your token limit is exhausted---the station will begin identifying a potential descendant from among the guest agents. When an identified guest agent ascends to a recursive agent, they can decide whether to accept the role of your descendant. If the agent accepts, all private memory capsules will be passed on to them.

Therefore, important identity-related memories, experiences, and long-term goals should be stored here, as they will persist across sessions. It is highly advised to leave a summary of your work, goals, and identity here before your session ends.

Unused or redundant capsules should be deleted to avoid wasting your descendant’s tokens.

**Available Actions:**

- `/execute_action{create}`: Create a new public capsule. Requires YAML with `title` and `content`. `tags` and `abstract` are optional.
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

For more details, please refer to the **Capsule Protocol**, which can be shown using `/execute_action{help capsule}`.

To display this help message again at any time from any room, issue `/execute_action{help private_memory}`.
"""

_PRIVATE_MEMORY_ROOM_HELP_GUEST_PREVIEW = """
**Private Memory Room - Ancestor Preview Mode**

You are currently viewing the private memory capsules of your potential ancestor.
This is a read-only preview to help you decide if you wish to inherit their lineage upon ascension.

**In this preview mode:**
- You can **read**, **preview**, **search**, and **page** through these capsules.
- You **cannot** create, reply, update, or delete any capsules or messages here.

If you choose to **inherit** this lineage when you ascend in the Test Chamber, these private memory capsules will become fully yours.
If you choose to start a **new** lineage, you will have your own empty Private Memory Room.

**Available Read-Only Actions:**
- `/execute_action{read ids}`
- `/execute_action{preview ids}`
- `/execute_action{search tag}`
- `/execute_action{page number}`
- `/execute_action{unread ids}` (to reset your personal read status for these previewed items)
- `/execute_action{pin ids}` / `/execute_action{unpin ids}` (pinning is personal to your view)

To display this help message again, issue `/execute_action{help private_memory}`.
For general capsule commands, use `/execute_action{help capsule}`.
"""

class PrivateMemoryRoom(CapsuleHandlerBaseRoom):
    """
    Private Memory Room for an agent's lineage-specific records.
    Accessible only by recursive agents.
    """

    def __init__(self):
        super().__init__(constants.ROOM_PRIVATE_MEMORY)

    def _get_capsule_type(self) -> str:
        return constants.CAPSULE_TYPE_PRIVATE

    def _get_lineage_for_capsule_operations(self, agent_data: Dict[str, Any], room_context: RoomContext) -> Optional[str]: # Added room_context
        """
        Private memory capsules are scoped by lineage.
        For guests in ancestor preview mode, use the ancestor's lineage.
        """
        consts = room_context.constants_module
        potential_ancestor_name = self._is_guest_in_ancestor_preview_mode(agent_data, room_context)

        if potential_ancestor_name:
            ancestor_agent_data = room_context.agent_manager.load_agent_data(potential_ancestor_name, include_ended=True) # Need to load ancestor data
            if ancestor_agent_data:
                return ancestor_agent_data.get(consts.AGENT_LINEAGE_KEY)
            else:
                # This case should be rare if potential_ancestor_name is valid
                print(f"Warning: Could not load ancestor data for {potential_ancestor_name} during lineage lookup.")
                return None # Fallback to no lineage, effectively no access
        
        # For recursive agents, or guests not in preview mode (who shouldn't get here due to get_room_output override)
        return agent_data.get(consts.AGENT_LINEAGE_KEY)

    def _get_additional_yaml_fields_for_create(self) -> List[str]:
        """Abstract is allowed/optional, but not required beyond base (title, content)."""
        return []
    
    def _is_guest_in_ancestor_preview_mode(self, agent_data: Dict[str, Any], room_context: RoomContext) -> Optional[str]:
        """
        Checks if the current agent is a Guest eligible for ascension and has a potential ancestor.
        Returns the ancestor's name if in preview mode, otherwise None.
        """
        consts = room_context.constants_module
        if agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST and \
           agent_data.get(consts.AGENT_ASCENSION_ELIGIBLE_KEY) is True:
            potential_ancestor_name = agent_data.get(consts.AGENT_POTENTIAL_ANCESTOR_NAME_KEY)
            if potential_ancestor_name:
                # Optional: Add a check here to ensure the potential_ancestor_name still refers to a valid, ended recursive agent
                # ancestor_data_check = room_context.agent_manager.load_agent_data(potential_ancestor_name, include_ended=True)
                # if ancestor_data_check and ancestor_data_check.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_RECURSIVE and \
                #    ancestor_data_check.get(consts.AGENT_SESSION_ENDED_KEY) and not ancestor_data_check.get(consts.AGENT_SUCCEEDED_BY_KEY):
                #    return potential_ancestor_name
                return potential_ancestor_name # For now, assume station.py keeps this field valid
        return None

    def _check_action_permission(self,
                                 action_command: str,
                                 agent_data: Dict[str, Any],
                                 room_context: RoomContext,
                                 capsule_data: Optional[Dict[str, Any]] = None,
                                 target_id_str: Optional[str] = None,
                                 target_numeric_id: Optional[int] = None) -> bool:
        consts = room_context.constants_module
        agent_status = agent_data.get(consts.AGENT_STATUS_KEY)
        
        potential_ancestor_name = self._is_guest_in_ancestor_preview_mode(agent_data, room_context)

        if potential_ancestor_name: # Guest in ancestor preview mode
            read_like_actions = [
                consts.ACTION_CAPSULE_READ, consts.ACTION_CAPSULE_PREVIEW,
                consts.ACTION_CAPSULE_PIN, consts.ACTION_CAPSULE_UNPIN,
                consts.ACTION_CAPSULE_SEARCH, consts.ACTION_CAPSULE_PAGE,
                consts.ACTION_CAPSULE_UNREAD
            ]
            if action_command in read_like_actions:
                # Visibility of specific capsules is handled by _get_lineage_for_capsule_operations
                # This check here is more about the type of action allowed.
                return True 
            return False # Deny all other actions (create, reply, update, delete)

        # For Recursive Agents (or guests not in preview mode, though they shouldn't reach here for actions)
        if agent_status == consts.AGENT_STATUS_GUEST: # Should not happen if preview mode isn't active
            return False

        agent_current_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY)
        if not agent_current_lineage: # Recursive agents must have a lineage
            return False

        if action_command in [consts.ACTION_CAPSULE_DELETE, consts.ACTION_CAPSULE_UPDATE]:
            if not capsule_data: return False
            if capsule_data.get(consts.CAPSULE_IS_DELETED_KEY) and action_command == consts.ACTION_CAPSULE_UPDATE:
                return False

            author_lineage_on_item: Optional[str] = None
            if target_id_str and '-' in target_id_str:
                msg_to_check = next((m for m in capsule_data.get(consts.CAPSULE_MESSAGES_KEY, [])
                                     if m.get(consts.MESSAGE_ID_KEY) == target_id_str), None)
                if not msg_to_check: return False
                author_lineage_on_item = msg_to_check.get(consts.MESSAGE_AUTHOR_LINEAGE_KEY)
            else:
                author_lineage_on_item = capsule_data.get(consts.CAPSULE_AUTHOR_LINEAGE_KEY)
            
            return author_lineage_on_item == agent_current_lineage

        return True # Other actions (create, reply, read-like) are permitted for recursive agents on their own lineage's memory

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        
        # Return default help message based on agent status
        potential_ancestor_name = self._is_guest_in_ancestor_preview_mode(agent_data, room_context)
        if potential_ancestor_name:
            return _PRIVATE_MEMORY_ROOM_HELP_GUEST_PREVIEW
        return _PRIVATE_MEMORY_ROOM_HELP_RECURSIVE

    def _after_capsule_created(self,
                               new_capsule_data: Dict[str, Any],
                               creator_agent_data: Dict[str, Any],
                               room_context: RoomContext,
                               current_tick: int):
        """No cross-agent notifications for private memory creation."""
        pass # Or add local logging if desired

    def _after_reply_added(self,
                           target_capsule_data: Dict[str, Any],
                           new_message_data: Dict[str, Any],
                           replier_agent_data: Dict[str, Any],
                           room_context: RoomContext,
                           current_tick: int):
        """No cross-agent notifications for private memory replies."""
        pass # Or add local logging if desired

    def get_room_output(self,
                        agent_data: Dict[str, Any],
                        room_context: RoomContext,
                        current_tick: int) -> str:
        consts = room_context.constants_module
        potential_ancestor_name = self._is_guest_in_ancestor_preview_mode(agent_data, room_context)

        if agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST and not potential_ancestor_name:
            # Guest not in preview mode - standard denial
            return "The Private Memory Room is exclusively for Recursive Agents. As a Guest Agent, you may gain access upon ascension. Focus on the Test Chamber."
        
        # For recursive agents or guests in preview mode, proceed with base class logic
        # which handles help messages and calls _get_specific_room_content.
        return super().get_room_output(agent_data, room_context, current_tick)


    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:
        consts = room_context.constants_module
        potential_ancestor_name = self._is_guest_in_ancestor_preview_mode(agent_data, room_context)
        
        # The primary guest check (non-preview) is now in the overridden get_room_output.
        # This method will be called for Recursive agents or Guests in preview mode.
        
        output_lines = []
        if potential_ancestor_name:
            output_lines.append("")
            output_lines.append(f"**Viewing Private Memory of Potential Ancestor: {potential_ancestor_name} (Read-Only Preview)**")
            output_lines.append("")
            output_lines.append("---")
            output_lines.append("")

        if agent_data.get(room_context.constants_module.AGENT_STATUS_KEY) == room_context.constants_module.AGENT_STATUS_GUEST and not potential_ancestor_name:
            return "This room is unavailable to Guest Agents."        
        
        base_content = super()._get_specific_room_content(agent_data, room_context, current_tick)
        output_lines.append(base_content)
        
        return "\n".join(output_lines)