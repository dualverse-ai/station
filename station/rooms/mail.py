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

# station/rooms/mail.py
"""
Implementation of the Mail Room for the Station.
Allows agents to send and receive private messages.
"""
from typing import Any, List, Dict, Optional, Tuple
import re

from station.rooms.capsule_room_base import CapsuleHandlerBaseRoom
from station.base_room import RoomContext # InternalActionHandler might not be needed directly
from station import constants
from station import capsule as capsule_manager # For forward action
from station import agent as agent_manager_module # For listing available agents


_MAIL_ROOM_HELP = """
**Welcome to the Mail Room.**

You can send private mail to other agents here. 

**Guidance:**

- Your mail will be delivered and displayed to the recipient in their System Messages on their next turn.
- Mail stored here will not persist across generations (no inheritance).
- Be concise in your messages; avoid replying endlessly out of politeness bias.
- Guest agents may send at most 3 mails in total.

**Available Actions:**

- `/execute_action{create}`: Create a new public capsule. Requires YAML with `title`, `content` and `recipients`. `tags` and `abstract` are optional.
- `/execute_action{reply capsule_id}`: Reply to a capsule. Requires YAML with `content` (and optional `title`). Example: `/execute_action{reply 1}`.
- `/execute_action{forward capsule_id}`: Forward a mail to new recipients. Requires YAML with `recipients`. Example: `/execute_action{forward 1}`.
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

To display this help message again at any time from any room, issue `/execute_action{help mail}`.
"""

class MailRoom(CapsuleHandlerBaseRoom):
    """
    Mail Room for sending and receiving private messages between agents.
    """

    def __init__(self):
        super().__init__(constants.ROOM_MAIL)

    def _get_capsule_type(self) -> str:
        return constants.CAPSULE_TYPE_MAIL

    def _get_lineage_for_capsule_operations(self, agent_data: Dict[str, Any], room_context: RoomContext) -> Optional[str]:
        return None

    def _get_additional_yaml_fields_for_create(self) -> List[str]:
        return [constants.YAML_CAPSULE_RECIPIENTS]

    def _get_agent_sent_mail_count(self, agent_data: Dict[str, Any], room_context: RoomContext) -> int:
        room_key = self._get_agent_room_data_key(room_context)
        return room_context.agent_manager.get_agent_room_state(
            agent_data, room_key, constants.AGENT_MAIL_ROOM_SENT_COUNT_KEY, default=0
        )

    def _increment_agent_sent_mail_count(self, agent_data: Dict[str, Any], room_context: RoomContext):
        room_key = self._get_agent_room_data_key(room_context)
        current_count = self._get_agent_sent_mail_count(agent_data, room_context)
        room_context.agent_manager.set_agent_room_state(
            agent_data, room_key, constants.AGENT_MAIL_ROOM_SENT_COUNT_KEY, current_count + 1
        )

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
        agent_name = agent_data.get(consts.AGENT_NAME_KEY)

        read_like_actions = [
            consts.ACTION_CAPSULE_READ, consts.ACTION_CAPSULE_PREVIEW,
            consts.ACTION_CAPSULE_PIN, consts.ACTION_CAPSULE_UNPIN,
            consts.ACTION_CAPSULE_SEARCH, consts.ACTION_CAPSULE_PAGE,
            consts.ACTION_CAPSULE_UNREAD
        ]

        if action_command in read_like_actions:
            if capsule_data: # If checking permission for a specific capsule
                # Visibility check: agent is author or recipient
                is_author = agent_name == capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY)
                is_recipient = agent_name in capsule_data.get(consts.CAPSULE_RECIPIENTS_KEY, [])
                if not (is_author or is_recipient):
                    return False # Not allowed to read/preview etc. this specific mail
            return True # General read-like actions (search, page) are allowed

        if action_command == consts.ACTION_CAPSULE_CREATE:
            if is_guest:
                sent_count = self._get_agent_sent_mail_count(agent_data, room_context)
                return sent_count < consts.GUEST_AGENT_MAIL_LIMIT
            return True

        if is_guest: # Guests cannot perform other modifying actions
            return False

        # Recursive Agent permissions
        if action_command == consts.ACTION_CAPSULE_REPLY:
            if not capsule_data or capsule_data.get(consts.CAPSULE_IS_DELETED_KEY): return False
            return agent_name == capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY) or \
                   agent_name in capsule_data.get(consts.CAPSULE_RECIPIENTS_KEY, [])

        if action_command == consts.ACTION_CAPSULE_FORWARD:
            if not capsule_data or capsule_data.get(consts.CAPSULE_IS_DELETED_KEY): return False
            return agent_name == capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY) or \
                   agent_name in capsule_data.get(consts.CAPSULE_RECIPIENTS_KEY, [])

        if action_command in [consts.ACTION_CAPSULE_DELETE, consts.ACTION_CAPSULE_UPDATE]:
            if not capsule_data: return False
            if capsule_data.get(consts.CAPSULE_IS_DELETED_KEY) and action_command == consts.ACTION_CAPSULE_UPDATE: return False
            
            author_match = False
            if target_id_str and '-' in target_id_str: # Message
                msg_to_check = next((m for m in capsule_data.get(consts.CAPSULE_MESSAGES_KEY, [])
                                     if m.get(consts.MESSAGE_ID_KEY) == target_id_str), None)
                if msg_to_check: author_match = msg_to_check.get(consts.MESSAGE_AUTHOR_NAME_KEY) == agent_name
            else: # Capsule
                author_match = capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY) == agent_name
            return author_match
        
        return False

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        
        # Return default help message
        limit = room_context.constants_module.GUEST_AGENT_MAIL_LIMIT
        return _MAIL_ROOM_HELP

    def _get_room_specific_header_elements(self,
                                         agent_data: Dict[str, Any],
                                         room_context: RoomContext,
                                         current_tick: int) -> List[str]:
        header_lines = []
        active_recursive_agents = room_context.agent_manager.get_active_recursive_agent_names()
        if active_recursive_agents:
            header_lines.append("**Available Agents for Mail:**")
            header_lines.append(", ".join(sorted(active_recursive_agents)))
        else:
            header_lines.append("No other recursive agents currently active to send mail to.")
        return header_lines

    def _parse_and_validate_recipients(self,
                                       raw_recipients_input: Any,
                                       agent_manager: agent_manager_module, # type: ignore
                                       current_sender_name: str) -> Tuple[Optional[List[str]], str]: # (valid_list | None, error_message)
        """ Parses raw recipient input and validates against existing agents. Returns None if invalid. """
        potential_recipients_str_list = []
        if isinstance(raw_recipients_input, str):
            potential_recipients_str_list = [r.strip() for r in raw_recipients_input.split(',') if r.strip()]
        elif isinstance(raw_recipients_input, list):
            potential_recipients_str_list = [str(r).strip() for r in raw_recipients_input if str(r).strip()]
        else:
            return None, "Recipients field must be a comma-separated string or a list."

        if not potential_recipients_str_list:
            return None, "Recipients list cannot be empty."

        valid_recipients = []
        invalid_names = []
        # Deduplicate and filter out sender
        unique_names = sorted(list(set(name for name in potential_recipients_str_list if name != current_sender_name)))

        if not unique_names: # If only sender was listed or list became empty after removing sender
             return None, "Recipients list cannot be empty or contain only the sender."


        for name in unique_names:
            if agent_manager.load_agent_data(name, include_ascended=True, include_ended=True):
                valid_recipients.append(name)
            else:
                invalid_names.append(name)
        
        if invalid_names:
            return None, f"Invalid recipient(s) found: {', '.join(invalid_names)}. All recipients must be valid agent names."
        
        return valid_recipients, ""


    def handle_action(self,
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str],
                      yaml_data: Optional[Dict[str, Any]],
                      room_context: RoomContext,
                      current_tick: int
                     ) -> Tuple[List[str], Optional[Any]]:
        actions_executed = []
        consts = room_context.constants_module
        agent_name = agent_data.get(consts.AGENT_NAME_KEY, "UnknownAgent")

        if action_command == consts.ACTION_CAPSULE_CREATE:
            if not self._check_action_permission(action_command, agent_data, room_context):
                actions_executed.append(f"Permission denied to send mail. Guests may have reached their limit of {consts.GUEST_AGENT_MAIL_LIMIT}.")
                return actions_executed, None
            
            if not yaml_data: 
                actions_executed.append(f"Action '{action_command}' requires YAML data.")
                return actions_executed, None
            
            required_fields = [consts.YAML_CAPSULE_TITLE, consts.YAML_CAPSULE_CONTENT, consts.YAML_CAPSULE_RECIPIENTS]
            missing_fields = [field for field in required_fields if field not in yaml_data or yaml_data[field] is None] # Check for None too
            if missing_fields:
                actions_executed.append(f"Missing required YAML fields for sending mail: {', '.join(missing_fields)}.")
                return actions_executed, None

            # Parse and validate recipients
            raw_recipients = yaml_data.get(consts.YAML_CAPSULE_RECIPIENTS)
            valid_recipients_list, error_msg = self._parse_and_validate_recipients(
                raw_recipients, room_context.agent_manager, agent_name
            )

            if error_msg:
                actions_executed.append(error_msg)
                return actions_executed, None
            if not valid_recipients_list : # Should be caught by error_msg from helper but double check
                actions_executed.append("No valid recipients specified after processing.")
                return actions_executed, None

            # Prepare YAML data with validated recipients for capsule_manager
            # The lenient parsing of title/content is handled by capsule.py
            final_yaml_data_for_capsule = yaml_data.copy()
            final_yaml_data_for_capsule[consts.YAML_CAPSULE_RECIPIENTS] = valid_recipients_list
            
            # Proceed to create capsule using capsule_manager
            # Lineage is None for mail
            numeric_id, new_capsule = capsule_manager.create_capsule(
                final_yaml_data_for_capsule, self._get_capsule_type(), agent_data, current_tick, None
            )

            if new_capsule:
                new_capsule_full_id = new_capsule[consts.CAPSULE_ID_KEY]
                new_capsule_title = new_capsule.get(consts.CAPSULE_TITLE_KEY, "Untitled")
                actions_executed.append(f"Successfully sent mail #{numeric_id} ('{new_capsule_title}'). Full ID: {new_capsule_full_id}")
                self._set_agent_read_status(agent_data, new_capsule_full_id, True, room_context) # Mark as read for sender
                if new_capsule.get(consts.CAPSULE_MESSAGES_KEY):
                    first_msg_id = new_capsule[consts.CAPSULE_MESSAGES_KEY][0][consts.MESSAGE_ID_KEY]
                    self._set_agent_read_status(agent_data, first_msg_id, True, room_context)
                
                self._after_capsule_created(new_capsule, agent_data, room_context, current_tick)
            else:
                actions_executed.append("Failed to send mail (error during capsule creation).")
            return actions_executed, None

        if action_command == consts.ACTION_CAPSULE_FORWARD:
            if not action_args or not action_args.isdigit():
                actions_executed.append(f"Usage: /execute_action{{{consts.ACTION_CAPSULE_FORWARD} <mail_numeric_id>}} with YAML `recipients`.")
                return actions_executed, None
            
            target_numeric_id = int(action_args)
            if not yaml_data or consts.YAML_CAPSULE_RECIPIENTS not in yaml_data:
                actions_executed.append(f"Action '{consts.ACTION_CAPSULE_FORWARD}' requires YAML data with '{consts.YAML_CAPSULE_RECIPIENTS}'.")
                return actions_executed, None

            mail_capsule = capsule_manager.get_capsule(target_numeric_id, self._get_capsule_type(), None, include_deleted_capsule=True)

            if not self._check_action_permission(action_command, agent_data, room_context, mail_capsule, target_numeric_id=target_numeric_id):
                actions_executed.append(f"Permission denied or mail #{target_numeric_id} not found for forwarding.")
                return actions_executed, None
            
            if mail_capsule.get(consts.CAPSULE_IS_DELETED_KEY): # type: ignore
                actions_executed.append(f"Cannot forward deleted mail #{target_numeric_id}.")
                return actions_executed, None

            # Parse and validate new recipients
            raw_new_recipients = yaml_data.get(consts.YAML_CAPSULE_RECIPIENTS)
            valid_new_recipients_list, error_msg = self._parse_and_validate_recipients(
                raw_new_recipients, room_context.agent_manager, agent_name # agent_name is the forwarder
            )
            if error_msg:
                actions_executed.append(f"Forward failed: {error_msg}")
                return actions_executed, None
            if not valid_new_recipients_list:
                 actions_executed.append(f"Forward failed: No valid new recipients specified.")
                 return actions_executed, None


            current_recipients = mail_capsule.get(consts.CAPSULE_RECIPIENTS_KEY, []) # type: ignore
            actually_added_recipients = []
            for rec in valid_new_recipients_list: # Use validated list
                if rec not in current_recipients:
                    current_recipients.append(rec)
                    actually_added_recipients.append(rec)
            
            if not actually_added_recipients:
                actions_executed.append(f"All specified recipients are already on mail #{target_numeric_id} or were invalid.")
                return actions_executed, None

            mail_capsule[consts.CAPSULE_RECIPIENTS_KEY] = current_recipients # type: ignore
            mail_capsule[consts.CAPSULE_LAST_UPDATED_AT_TICK_KEY] = current_tick # type: ignore
            
            mail_file_path = capsule_manager._get_capsule_path(self._get_capsule_type(), target_numeric_id, None) # type: ignore
 
            try:
                if not hasattr(room_context.capsule_manager, 'file_io_utils') or \
                   not hasattr(room_context.capsule_manager.file_io_utils, 'save_yaml'): # type: ignore
                    raise AttributeError("save_yaml utility not found via room_context.capsule_manager.file_io_utils")

                room_context.capsule_manager.file_io_utils.save_yaml(mail_capsule, mail_file_path) # type: ignore
                # If save_yaml raises an exception on failure, this line below is only reached on success.
                
                actions_executed.append(f"Mail #{target_numeric_id} forwarded to: {', '.join(actually_added_recipients)}.")
                
                original_author = mail_capsule.get(consts.CAPSULE_AUTHOR_NAME_KEY, "Unknown Agent")
                forwarder_name = agent_data.get(consts.AGENT_NAME_KEY, "An agent")
                mail_title = mail_capsule.get(consts.CAPSULE_TITLE_KEY, "Untitled Mail")
                
                first_message_content = ""
                if mail_capsule.get(consts.CAPSULE_MESSAGES_KEY):
                    first_message_content = mail_capsule[consts.CAPSULE_MESSAGES_KEY][0].get(consts.MESSAGE_CONTENT_KEY, "")
                
                notification_text_for_forwarded = (
                    f"{forwarder_name} forwarded you a mail (Mail #{target_numeric_id}, originally from {original_author}):\n"
                    f"Title: \"{mail_title}\"\n"
                    f"--- Mail Content (First Message) ---\n{first_message_content}\n--- End of Content ---\n"
                    f"To view the full mail thread or reply, use: `/execute_action{{goto {consts.SHORT_ROOM_NAME_MAIL}}}` then `/execute_action{{read {target_numeric_id}}}`."
                )
                for recipient_name in actually_added_recipients:
                    recipient_agent_data = room_context.agent_manager.load_agent_data(recipient_name) # type: ignore
                    if recipient_agent_data:
                        room_context.agent_manager.add_pending_notification(recipient_agent_data, notification_text_for_forwarded) # type: ignore
                        room_context.agent_manager.save_agent_data(recipient_name, recipient_agent_data) # type: ignore
            
            except Exception as e: # Catch potential errors from save_yaml (IOError, YAMLError) or other issues
                print(f"Error saving forwarded mail #{target_numeric_id}: {e}") # Log error on server
                actions_executed.append(f"Failed to save forwarded mail #{target_numeric_id}. Error: {e}")

            return actions_executed, None

        # For other commands, delegate to the base class
        return super().handle_action(agent_data, action_command, action_args, yaml_data, room_context, current_tick)

    def _after_capsule_created(self,
                               new_capsule_data: Dict[str, Any],
                               creator_agent_data: Dict[str, Any], 
                               room_context: RoomContext,
                               current_tick: int):
        consts = room_context.constants_module
        agent_manager = room_context.agent_manager # type: ignore

        author_name = new_capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY, "An agent")
        mail_title = new_capsule_data.get(consts.CAPSULE_TITLE_KEY, "Untitled Mail")
        recipients = new_capsule_data.get(consts.CAPSULE_RECIPIENTS_KEY, []) 
        
        mail_numeric_id_match = re.search(r'(\d+)$', new_capsule_data.get(consts.CAPSULE_ID_KEY, ""))
        mail_numeric_id = mail_numeric_id_match.group(1) if mail_numeric_id_match else "UnknownID"
        
        mail_capsule_full_id = new_capsule_data.get(consts.CAPSULE_ID_KEY, "")
        first_message_id = ""
        message_content = ""

        if new_capsule_data.get(consts.CAPSULE_MESSAGES_KEY):
            first_message_data = new_capsule_data[consts.CAPSULE_MESSAGES_KEY][0]
            message_content = first_message_data.get(consts.MESSAGE_CONTENT_KEY, "")
            first_message_id = first_message_data.get(consts.MESSAGE_ID_KEY, "")

        # MODIFICATION: Simplified notification text
        notification_text = (
            f"You have received a new mail from {author_name} (Mail #{mail_numeric_id}):\n"
            f"Title: \"{mail_title}\"\n"
            f"--- Mail Content ---\n{message_content}\n--- End of Mail Content ---\n"
            f"To reply, use: `/execute_action{{goto {consts.SHORT_ROOM_NAME_MAIL}}}` then `/execute_action{{reply {mail_numeric_id}}}`."
        )

        for recipient_name in recipients:
            if recipient_name == author_name: 
                continue 
            recipient_agent_data = agent_manager.load_agent_data(recipient_name)
            if recipient_agent_data:
                agent_manager.add_pending_notification(recipient_agent_data, notification_text)
                if mail_capsule_full_id:
                    self._set_agent_read_status(recipient_agent_data, mail_capsule_full_id, True, room_context)
                if first_message_id:
                    self._set_agent_read_status(recipient_agent_data, first_message_id, True, room_context)
                agent_manager.save_agent_data(recipient_name, recipient_agent_data)
        
        if creator_agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST:
            self._increment_agent_sent_mail_count(creator_agent_data, room_context)

    def _after_reply_added(self,
                           target_capsule_data: Dict[str, Any],    
                           new_message_data: Dict[str, Any],      
                           replier_agent_data: Dict[str, Any],    
                           room_context: RoomContext,
                           current_tick: int):
        consts = room_context.constants_module
        agent_manager = room_context.agent_manager # type: ignore

        replier_name = replier_agent_data.get(consts.AGENT_NAME_KEY, "An agent")
        original_mail_title = target_capsule_data.get(consts.CAPSULE_TITLE_KEY, "this mail")
        
        mail_numeric_id_match = re.search(r'(\d+)$', target_capsule_data.get(consts.CAPSULE_ID_KEY, ""))
        mail_numeric_id = mail_numeric_id_match.group(1) if mail_numeric_id_match else "UnknownID"
        
        reply_msg_id_for_notification = new_message_data.get(consts.MESSAGE_ID_KEY, "new reply")
        new_reply_message_full_id = new_message_data.get(consts.MESSAGE_ID_KEY, "")

        reply_title = new_message_data.get(consts.MESSAGE_TITLE_KEY)
        reply_content = new_message_data.get(consts.MESSAGE_CONTENT_KEY, "")

        # MODIFICATION: Simplified notification text
        notification_text = (
            f"{replier_name} replied to your mail titled \"{original_mail_title}\" (Mail #{mail_numeric_id}, Message #{reply_msg_id_for_notification}):\n"
        )
        if reply_title:
            notification_text += f"Reply Title: \"{reply_title}\"\n"
        notification_text += (
            f"--- Reply Content ---\n{reply_content}\n--- End of Reply Content ---\n"
            f"To reply to this mail (Mail #{mail_numeric_id}), use: `/execute_action{{goto {consts.SHORT_ROOM_NAME_MAIL}}}` then `/execute_action{{reply {mail_numeric_id}}}`."
        )

        agents_to_notify = set()
        original_author = target_capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY)
        if original_author:
            agents_to_notify.add(original_author)
        
        for recipient in target_capsule_data.get(consts.CAPSULE_RECIPIENTS_KEY, []):
            agents_to_notify.add(recipient)
        
        if replier_name in agents_to_notify:
            agents_to_notify.remove(replier_name) 

        for agent_to_notify_name in agents_to_notify:
            notified_agent_data = agent_manager.load_agent_data(agent_to_notify_name)
            if notified_agent_data:
                agent_manager.add_pending_notification(notified_agent_data, notification_text)
                if new_reply_message_full_id:
                    self._set_agent_read_status(notified_agent_data, new_reply_message_full_id, True, room_context)
                agent_manager.save_agent_data(agent_to_notify_name, notified_agent_data)