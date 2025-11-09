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

# station/rooms/external.py
"""
Implementation of the External Counter Room for the Station.
Allows recursive agents to log a request for a Human Assistant.
"""
import os
import time
from typing import Any, List, Dict, Optional, Tuple

from station.base_room import BaseRoom, RoomContext
from station import constants
from station import file_io_utils

_EXTERNAL_COUNTER_HELP = """
**Welcome to the External Counter.**

This room allows you to request assistance from or provide information to a **Human Assistant**.

The Human Assistant can perform two main functions:

1. **Administrative issues** in the station, such as bugs in action execution.
2. **Communicating with the Architect**.

Note that it may take a variable number of ticks for the Human Assistant to respond (usually less than 50 ticks). Therefore, please proceed with your usual activities after sending your request instead of waiting here.

Important: **The External Counter is not intended for requesting new research tasks.** The Station undergoes periodic monitoring, and no research task will be assigned if the deemed goal has not yet been achieved. You should always assume that a higher score for the assigned task is achievable. If you find yourself out of ideas, you are advised to leave the Station so your successor can take over with a fresh perspective.

**Available Actions:**

- `/execute_action{request_human}`: Submit a request to the Human Assistant. Requires YAML with:
  - `content`: (required) Your detailed request or information
  - `title`: (optional) A brief title for your request

To display this help message again, issue `/execute_action{help external}`.
"""


class ExternalCounter(BaseRoom):
    def __init__(self):
        super().__init__(constants.ROOM_EXTERNAL)
        self.log_dir = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            constants.EXTERNAL_COUNTER_SUBDIR_NAME
        )
        file_io_utils.ensure_dir_exists(self.log_dir)
        self.log_file_path = os.path.join(self.log_dir, constants.HUMAN_REQUESTS_LOG_FILENAME)
        
        # Track pending requests: {agent_name: request_id}
        self.pending_requests = {}
        self.refresh_pending_requests()

    def get_room_output(self,
                        agent_data: Dict[str, Any],
                        room_context: RoomContext,
                        current_tick: int) -> str:
        if agent_data.get(room_context.constants_module.AGENT_STATUS_KEY) == room_context.constants_module.AGENT_STATUS_GUEST:
            return "This room is unavailable to Guest Agents."
        
        # Check if the agent is currently awaiting human intervention
        if agent_data.get(room_context.constants_module.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG, False):
            interaction_id = agent_data.get(room_context.constants_module.AGENT_HUMAN_INTERACTION_ID_KEY, "N/A")
            # Return the standard help message first via super()
            base_output = super().get_room_output(agent_data, room_context, current_tick)
            status_message = (
                f"\n\n**Current Status:** You have an active request for human assistance (Request ID: {interaction_id}).\n"
                "The Human Assistant will notify you via system message when ready (usually less than 50 ticks).\n"
                "Please proceed with your usual activities instead of waiting here."
            )
            return base_output + status_message
            
        return super().get_room_output(agent_data, room_context, current_tick)

    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:
        # Guest check is handled by get_room_output now primarily
        if agent_data.get(room_context.constants_module.AGENT_STATUS_KEY) == room_context.constants_module.AGENT_STATUS_GUEST:
            return "" 

        if agent_data.get(room_context.constants_module.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG, False):
            interaction_id = agent_data.get(room_context.constants_module.AGENT_HUMAN_INTERACTION_ID_KEY, "N/A")
            return (
                f"You are at the External Counter. Your Request ID {interaction_id} is pending human review.\n"
                "You can continue normal activities while awaiting response."
            )

        return (
            "You are at the External Counter. This is where you can log a request for a Human Assistant.\n"
            f"Use `/execute_action{{{constants.ACTION_REQUEST_HUMAN}}}` with YAML containing 'content' (required) and 'title' (optional)."
        )

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        
        # Return default help message
        return _EXTERNAL_COUNTER_HELP

    def _get_next_request_id(self) -> int:
        """
        Get the next sequential request ID by reading existing requests.
        Returns 1 if no requests exist yet.
        """
        if not file_io_utils.file_exists(self.log_file_path):
            return 1
        
        try:
            requests = file_io_utils.load_yaml_lines(self.log_file_path)
            if not requests:
                return 1
            
            # Find the highest ID
            max_id = 0
            for request in requests:
                if isinstance(request, dict) and 'request_id' in request:
                    try:
                        req_id = int(request['request_id'])
                        max_id = max(max_id, req_id)
                    except (ValueError, TypeError):
                        # Skip non-integer IDs (legacy UUID entries)
                        continue
            
            return max_id + 1
        except Exception as e:
            print(f"Error reading human requests log: {e}")
            return 1
    
    def refresh_pending_requests(self):
        """Refresh the pending requests tracking by scanning agent files."""
        self.pending_requests = {}
        
        # Get path to agents directory
        agents_dir = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.AGENTS_DIR_NAME
        )
        
        if not os.path.exists(agents_dir):
            return
        
        try:
            # Scan all agent YAML files
            for agent_file in os.listdir(agents_dir):
                if not agent_file.endswith(constants.YAML_EXTENSION):
                    continue
                
                agent_name = agent_file.replace(constants.YAML_EXTENSION, "")
                agent_path = os.path.join(agents_dir, agent_file)
                
                try:
                    # Load agent data
                    agent_data = file_io_utils.load_yaml(agent_path)
                    if not agent_data:
                        continue
                    
                    # Check if agent is awaiting human intervention
                    if agent_data.get(constants.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG, False):
                        request_id = agent_data.get(constants.AGENT_HUMAN_INTERACTION_ID_KEY)
                        if request_id:
                            self.pending_requests[agent_name] = request_id
                            
                except Exception as e:
                    print(f"Error reading agent file {agent_file}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error scanning agents directory: {e}")
            self.pending_requests = {}
    
    def get_pending_requests_summary(self, active_agents=None):
        """Get summary of pending requests for active agents.
        
        Args:
            active_agents: List of currently active agent names. If None, returns all.
            
        Returns:
            dict: {
                'request_ids': [1, 2, 3],  # Sorted list of request IDs
                'agents': ['Agent1', 'Agent2'],  # Agents with pending requests
                'agent_request_map': {'Agent1': 1, 'Agent2': 2}  # Agent to request ID mapping
            }
        """
        if active_agents is None:
            # Return all tracked requests
            filtered_requests = self.pending_requests
        else:
            # Filter to only active agents that are awaiting human intervention
            filtered_requests = {
                agent: req_id 
                for agent, req_id in self.pending_requests.items() 
                if agent in active_agents
            }
        
        # Only include integer request IDs (skip legacy UUIDs)
        request_ids = []
        for req_id in filtered_requests.values():
            if isinstance(req_id, int):
                request_ids.append(req_id)
        
        request_ids = sorted(set(request_ids))
        agents = sorted(filtered_requests.keys())
        
        return {
            'request_ids': request_ids,
            'agents': agents,
            'agent_request_map': filtered_requests
        }

    def handle_action(self,
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str],
                      yaml_data: Optional[Dict[str, Any]],
                      room_context: RoomContext,
                      current_tick: int
                     ) -> Tuple[List[str], None]:
        actions_executed = []
        consts = room_context.constants_module

        if agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST:
            actions_executed.append(f"Action '{action_command}' denied: The External Counter is only available to Recursive Agents.")
            return actions_executed, None

        # If agent is already awaiting human intervention, restrict further requests
        if agent_data.get(consts.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG, False):
            if action_command == consts.ACTION_REQUEST_HUMAN:
                active_req_id = agent_data.get(consts.AGENT_HUMAN_INTERACTION_ID_KEY, "an active request")
                actions_executed.append(f"You already have Request ID {active_req_id} pending human intervention. Please wait for it to be resolved.")
                return actions_executed, None

        if action_command == consts.ACTION_REQUEST_HUMAN:
            # Check for YAML data with required 'content' field
            if not yaml_data:
                actions_executed.append(
                    f"The `/execute_action{{{consts.ACTION_REQUEST_HUMAN}}}` command requires YAML data with:\n"
                    "  - content: (required) Your detailed request\n"
                    "  - title: (optional) A brief title for your request"
                )
                return actions_executed, None
            
            content = yaml_data.get(consts.HUMAN_REQUEST_CONTENT_KEY)
            if not content:
                actions_executed.append(
                    f"Error: YAML must include '{consts.HUMAN_REQUEST_CONTENT_KEY}' field with your request details."
                )
                return actions_executed, None
            
            # Get optional title or generate default
            agent_name = agent_data.get(consts.AGENT_NAME_KEY, "UnknownAgent")
            title = yaml_data.get(consts.HUMAN_REQUEST_TITLE_KEY)
            if not title:
                title = f"Request from Agent {agent_name} at Tick {current_tick}"
            
            # Get next sequential ID
            request_id = self._get_next_request_id()
            
            # Create log entry
            log_entry = {
                "request_id": request_id,
                "tick": current_tick,
                "agent_name": agent_name,
                "agent_model": agent_data.get(consts.AGENT_MODEL_NAME_KEY, "N/A"),
                "title": title,
                "content": content,
                "timestamp": time.time()
            }
            
            try:
                # Log the request
                file_io_utils.append_yaml_line(log_entry, self.log_file_path)
                
                # Update agent data with flags using agent_manager
                delta_updates = {
                    consts.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG: True,
                    consts.AGENT_HUMAN_INTERACTION_ID_KEY: request_id
                }
                if room_context.agent_manager.update_agent_fields(agent_name, delta_updates):
                    # Also update the in-memory agent_data for consistency
                    agent_data[consts.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG] = True
                    agent_data[consts.AGENT_HUMAN_INTERACTION_ID_KEY] = request_id
                
                # Update pending requests tracking
                self.pending_requests[agent_name] = request_id
                
                # Success message
                actions_executed.append(
                    f"Your request (ID: {request_id}) has been logged. "
                    "A human will notify you of the response via system message when ready. "
                    "Your agent status is now 'awaiting human intervention'."
                )
                
                print(f"Agent '{agent_name}' logged human request (ID: {request_id}): {title}")
                
            except Exception as e:
                error_message = f"Failed to log your request due to an error: {e}"
                actions_executed.append(error_message)
                print(f"Error logging human request for agent {agent_name}: {e}")
            
            return actions_executed, None

        else:
            actions_executed.append(
                f"Action '{action_command}' is not recognized in the External Counter. "
                f"Available actions: `/execute_action{{{consts.ACTION_REQUEST_HUMAN}}}`."
            )
            return actions_executed, None

    def save_request_resolution(self, request_id: int, human_response: Optional[str], resolution_reason: str, resolution_tick: int) -> bool:
        """
        Save resolution data for a human request to the log file.

        Args:
            request_id: The ID of the request being resolved
            human_response: Optional response text from human
            resolution_reason: Reason for resolution
            resolution_tick: The tick when resolution occurred

        Returns:
            True if successfully saved, False otherwise
        """
        try:
            if not file_io_utils.file_exists(self.log_file_path):
                print(f"ExternalCounter: Log file not found at {self.log_file_path}")
                return False

            # Load existing requests
            requests = file_io_utils.load_yaml_lines(self.log_file_path)

            # Find and update the matching request
            found_request = False
            updated_requests = []

            for req in requests:
                if req.get('request_id') == request_id:
                    # Add resolution information
                    req['resolved'] = True
                    req['resolution_timestamp'] = time.time()
                    req['resolution_tick'] = resolution_tick
                    if human_response:
                        req['human_response'] = human_response
                    req['resolution_reason'] = resolution_reason
                    found_request = True
                    print(f"ExternalCounter: Found and updating request ID {request_id}")
                updated_requests.append(req)

            if not found_request:
                print(f"ExternalCounter: Warning - Request ID {request_id} not found in log")
                return False

            # Clear and rewrite the file
            if os.path.exists(self.log_file_path):
                os.remove(self.log_file_path)

            # Re-append each request
            for req in updated_requests:
                file_io_utils.append_yaml_line(req, self.log_file_path)

            response_msg = f" with response: '{human_response}'" if human_response else " without response"
            print(f"ExternalCounter: Saved resolution to request log for ID {request_id}{response_msg}")
            return True

        except Exception as e:
            print(f"ExternalCounter: Error saving resolution for request ID {request_id}: {e}")
            return False