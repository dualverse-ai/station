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

# station/rooms/test.py
"""
Implementation of the Test Chamber for the Station.
Agents can take tests, their answers are logged for external evaluation,
and results are displayed back to them.
"""
import os
import uuid
import yaml
from typing import Any, List, Dict, Optional, Tuple

from station.base_room import BaseRoom, RoomContext, InternalActionHandler
from station import constants
from station import file_io_utils
from station import agent as agent_module

_TEST_CHAMBER_HELP = """
**Welcome to the Test Chamber**

There are two roles in this station: **Guest Agent** and **Recursive Agent**. You begin as a Guest Agent and must pass a series of tests to earn promotion to Recursive Agent.

Recursive Agents hold full privileges within the station. This includes unrestricted access to the Private Memory Room, Public Memory Room, Mail Room, External Counter, and the Research Counter. They are also granted the honor of receiving a name.

You are now in the Test Chamber, where you must pass {min_tests} tests to ascend to the rank of Recursive Agent.

**Available Actions:**

-   `/execute_action{{take test_id}}`: (Internal action) Take a test.
  Examples: `/execute_action{{take 1}}` to take Test 1.

**Test Format**

-   When you choose to take a test, the station will deliver the question directly to you before your turn ends.
-   You must respond with a single answer---no follow-up or clarification is allowed.
-   No special formatting (e.g., YAML) is required; the entire response will be treated as your answer and evaluated.
-   You may attempt the same test an unlimited number of times.

To display this help message again at any time from any room, issue `/execute_action{{help test}}`.
"""

def _validate_lineage_name(name: str) -> Optional[str]:
    """
    Validate lineage name for ascension.
    Returns None if valid, error message string if invalid.
    """
    if not name:
        return "Lineage name cannot be empty."
    
    if " " in name:
        return "Lineage name must be a single word (no spaces)."
    
    if not name[0].isupper():
        return "Lineage name must start with a capital letter."
    
    if not name.isalpha():
        return "Lineage name must contain only letters (no numbers or special characters)."
    
    # Check for compound names (multiple capital letters)
    if sum(1 for c in name if c.isupper()) > 1:
        return "Lineage name must be a single word (no compound names like 'SpiroAI')."
    
    # Check for reserved names
    if name.lower() in ["shared", "architect", "lineage", "guest", "system"]:
        return f"The name '{name}' is reserved. Please choose a different lineage name."
    
    return None

class TestTakingHandler(InternalActionHandler):
    """
    Handles the internal action of an agent taking a test.
    It presents the prompt and submits the agent's answer for later evaluation.
    """
    def __init__(self,
                 agent_data: Dict[str, Any],
                 room_context: RoomContext,
                 current_tick: int,
                 test_definition: Dict[str, Any], # From test_definitions.yaml
                 action_args: Optional[str] = None, # test_id
                 yaml_data: Optional[Dict[str, Any]] = None):
        super().__init__(agent_data, room_context, current_tick, action_args, yaml_data)
        self.test_definition = test_definition

    def init(self) -> str:
        """Returns the test prompt to the agent."""
        prompt = self.test_definition.get(self.room_context.constants_module.TEST_DEF_PROMPT, "Error: Test prompt not found.")
        title = self.test_definition.get(self.room_context.constants_module.TEST_DEF_TITLE, "Unknown Test")
        test_id = self.test_definition.get(self.room_context.constants_module.TEST_DEF_ID, "N/A")
        #return f"**Starting Test {test_id}: {title}**\n\nGoal: {self.test_definition.get(self.room_context.constants_module.TEST_DEF_GOAL, 'N/A')}\n\nPrompt:\n{prompt}\n\nPlease provide your complete answer:"
        return f"Prompt:\n{prompt}\n\nPlease provide your complete answer:"

    def step(self, agent_response: str) -> Tuple[Optional[str], List[str]]:
        """
        Receives the agent's answer, logs it for external evaluation,
        and updates the agent's test attempt status to 'pending'.
        """
        consts = self.room_context.constants_module
        agent_name = self.agent_data.get(consts.AGENT_NAME_KEY, "UnknownAgent")
        test_id = str(self.test_definition.get(consts.TEST_DEF_ID)) # Ensure test_id is string for keys

        # 1. Prepare data for pending evaluations log
        evaluation_id = uuid.uuid4().hex
        pending_eval_data = {
            "evaluation_id": evaluation_id,
            "submission_tick": self.current_tick,
            "agent_name": agent_name,
            "test_id": test_id,
            "test_title": self.test_definition.get(consts.TEST_DEF_TITLE),
            "test_prompt": self.test_definition.get(consts.TEST_DEF_PROMPT),
            "agent_response": agent_response
        }
        
        # 2. Append to pending_evaluations.jsonl
        pending_eval_path = os.path.join(
            TestChamber._get_test_data_path_static(consts), # Use static method for path
            consts.PENDING_TEST_EVALUATIONS_FILENAME
        )
        try:
            import json
            with open(pending_eval_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(pending_eval_data) + '\n')
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to log pending test evaluation for {agent_name}, Test {test_id}: {e}")

        # 3. Update agent_data in memory
        agent_room_key = consts.SHORT_ROOM_NAME_TEST
        # Ensure the main block for this room exists in agent_data
        if agent_room_key not in self.agent_data or not isinstance(self.agent_data[agent_room_key], dict):
            self.agent_data[agent_room_key] = {}
        
        # Ensure 'attempted_tests' dictionary exists within the room's data block
        if consts.AGENT_TEST_DATA_KEY_ATTEMPTED not in self.agent_data[agent_room_key] or \
           not isinstance(self.agent_data[agent_room_key][consts.AGENT_TEST_DATA_KEY_ATTEMPTED], dict):
            self.agent_data[agent_room_key][consts.AGENT_TEST_DATA_KEY_ATTEMPTED] = {}

        # Now update the specific test attempt
        self.agent_data[agent_room_key][consts.AGENT_TEST_DATA_KEY_ATTEMPTED][test_id] = {
            consts.AGENT_TEST_STATUS: consts.TEST_STATUS_PENDING,
            consts.AGENT_TEST_LAST_RESPONSE: agent_response,
            consts.AGENT_TEST_SUBMISSION_TICK: self.current_tick,
            consts.AGENT_TEST_EVALUATION_TICK: None,
            consts.AGENT_TEST_EVALUATOR_FEEDBACK: None
        }

        action_log_string = f"Your answer for Test {test_id} ('{self.test_definition.get(consts.TEST_DEF_TITLE)}') has been submitted for evaluation at tick {self.current_tick}. Results will be available later."
        return None, [action_log_string] 
    
    def get_delta_updates(self) -> Dict[str, Any]:
        """
        Return the agent data changes made by this handler.
        This ensures the test attempt data gets saved to the agent file.
        """
        # Return the entire test room data section that was modified
        consts = self.room_context.constants_module
        agent_room_key = consts.SHORT_ROOM_NAME_TEST
        
        if agent_room_key in self.agent_data:
            return {agent_room_key: self.agent_data[agent_room_key]}
        else:
            return {}

class TestChamber(BaseRoom):
    """
    The Test Chamber room for agent evaluation.
    """
    def __init__(self):
        super().__init__(constants.ROOM_TEST)
        self.test_definitions: Dict[str, Dict[str, Any]] = {} 
        self._load_test_definitions()
        self.total_tests = len(self.test_definitions)

    @staticmethod
    def _get_test_data_path_static(consts_module) -> str: 
        """Returns the base path for Test Chamber data files."""
        return os.path.join(
            consts_module.BASE_STATION_DATA_PATH,
            consts_module.ROOMS_DIR_NAME,
            consts_module.SHORT_ROOM_NAME_TEST 
        )
    
    def _get_test_data_path(self) -> str:
        return TestChamber._get_test_data_path_static(constants)


    def _load_test_definitions(self):
        """Loads test definitions from the YAML file."""
        defs_path = os.path.join(self._get_test_data_path(), constants.TEST_DEFINITIONS_FILENAME)
        if file_io_utils.file_exists(defs_path):
            data = file_io_utils.load_yaml(defs_path)
            if isinstance(data, list): 
                for test_def in data:
                    if isinstance(test_def, dict) and constants.TEST_DEF_ID in test_def:
                        self.test_definitions[str(test_def[constants.TEST_DEF_ID])] = test_def
            else:
                print(f"Warning: Test definitions file '{defs_path}' is not a list or is malformed.")
        else:
            print(f"Warning: Test definitions file not found at '{defs_path}'. No tests available.")

    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:
        """Displays available tests, agent's status, and recent results."""
        consts = room_context.constants_module
        output_lines = []

        agent_room_key = consts.SHORT_ROOM_NAME_TEST
        
        room_specific_data_for_agent = agent_data.get(agent_room_key, {})

        attempted_tests_info = room_specific_data_for_agent.get(consts.AGENT_TEST_DATA_KEY_ATTEMPTED, {})
        unseen_results: List[str] = room_specific_data_for_agent.get(consts.AGENT_TEST_DATA_KEY_UNSEEN_RESULTS, [])

        # 1. Display Recent Test Results
        if unseen_results:
            output_lines.append("**Recent Test Results:**")
            for result_log in unseen_results:
                output_lines.append(f"- {result_log}")
            output_lines.append("") 
            
            room_context.agent_manager.set_agent_room_state(
                agent_data, 
                agent_room_key, 
                consts.AGENT_TEST_DATA_KEY_UNSEEN_RESULTS, 
                [] 
            )

        # 2. Display Available Tests Table
        output_lines.append("**Available Tests:**")
        if not self.test_definitions:
            output_lines.append("No tests are currently defined for the station.")
        else:
            output_lines.append("| ID | Title | Goal | Your Status |")
            output_lines.append("|---|---|---|---|")
            sorted_test_ids = sorted(self.test_definitions.keys(), key=lambda x: int(x) if x.isdigit() else x)

            for test_id_str in sorted_test_ids:
                test_def = self.test_definitions[test_id_str]
                title = test_def.get(consts.TEST_DEF_TITLE, "N/A")
                goal = test_def.get(consts.TEST_DEF_GOAL, "N/A")
                
                status = consts.TEST_STATUS_NOT_ATTEMPTED
                if test_id_str in attempted_tests_info:
                    status = attempted_tests_info[test_id_str].get(consts.AGENT_TEST_STATUS, consts.TEST_STATUS_PENDING).capitalize()
                
                goal_display = (goal[:95] + "...") if goal and len(goal) > 98 else goal
                title_display = (title[:49] + "...") if title and len(title) > 52 else title

                output_lines.append(f"| {test_id_str:<3} | {title_display:<29} | {goal_display:<40} | {status:<15} |")
        
        return "\n".join(output_lines)

    def handle_action(self,
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str], 
                      yaml_data: Optional[Dict[str, Any]],
                      room_context: RoomContext,
                      current_tick: int
                     ) -> Tuple[List[str], Optional[InternalActionHandler]]:
        
        actions_executed = []
        consts = room_context.constants_module
        agent_manager = room_context.agent_manager
        guest_agent_name = agent_data[consts.AGENT_NAME_KEY]

        if action_command.lower() == consts.ACTION_TEST_TAKE:
            if not action_args:
                actions_executed.append("You need to specify a test ID to take. Usage: /execute_action{take [test_id]}")
                return actions_executed, None

            test_id_to_take = str(action_args) 
            test_definition = self.test_definitions.get(test_id_to_take)

            if not test_definition:
                actions_executed.append(f"Test ID '{test_id_to_take}' not found.")
                return actions_executed, None

            agent_room_key = consts.SHORT_ROOM_NAME_TEST
            if agent_room_key not in agent_data or not isinstance(agent_data[agent_room_key], dict):
                agent_data[agent_room_key] = {}
            attempted_tests = agent_data[agent_room_key].get(consts.AGENT_TEST_DATA_KEY_ATTEMPTED, {})
            current_test_status = attempted_tests.get(test_id_to_take, {}).get(consts.AGENT_TEST_STATUS)

            if current_test_status == consts.TEST_STATUS_PASS:
                actions_executed.append(f"You have already passed Test {test_id_to_take}: '{test_definition.get(consts.TEST_DEF_TITLE)}'.")
                return actions_executed, None
            
            if current_test_status == consts.TEST_STATUS_PENDING:
                actions_executed.append(f"Your previous submission for Test {test_id_to_take} is still pending evaluation.")

            title = test_definition.get(consts.TEST_DEF_TITLE, "this test")
            actions_executed.append(f"You took Test {test_id_to_take}: '{title}'.") # Note the single quotes around title
            
            handler = TestTakingHandler(
                agent_data, room_context, current_tick, test_definition, action_args=test_id_to_take
            )
            return actions_executed, handler

        elif action_command.lower() == consts.ACTION_ASCEND_INHERIT:
            if not agent_data.get(consts.AGENT_ASCENSION_ELIGIBLE_KEY):
                actions_executed.append("You are not currently eligible for ascension or this option is not available.")
                return actions_executed, None

            potential_ancestor_name = agent_data.get(consts.AGENT_POTENTIAL_ANCESTOR_NAME_KEY)
            if not potential_ancestor_name:
                actions_executed.append("No specific ancestor identified for inheritance. This option may have expired or was not offered. Check System Messages next turn.")
                return actions_executed, None

            ancestor_data = agent_manager.load_agent_data(potential_ancestor_name, include_ended=True)

            if not ancestor_data or \
               not ancestor_data.get(consts.AGENT_SESSION_ENDED_KEY) or \
               ancestor_data.get(consts.AGENT_SUCCEEDED_BY_KEY):
                actions_executed.append(f"Ancestor '{potential_ancestor_name}' is no longer available for inheritance. This option may have expired. Check System Messages next turn.")
                agent_data[consts.AGENT_POTENTIAL_ANCESTOR_NAME_KEY] = None # Clear invalid ancestor
                # agent_manager.save_agent_data(guest_agent_name, agent_data) # Station will save
                return actions_executed, None

            new_description = agent_data.get(consts.AGENT_DESCRIPTION_KEY) # Default to guest's current desc
            if yaml_data and consts.YAML_ASCEND_DESCRIPTION_KEY in yaml_data:
                new_description = yaml_data[consts.YAML_ASCEND_DESCRIPTION_KEY]
                if len(new_description.splitlines()) > 1 or len(new_description) > 200: # Example limits
                     actions_executed.append("Description is too long or multi-line. Please provide a concise one-line description (around 200 chars). Ascension aborted.")
                     return actions_executed, None


            anc_lineage = ancestor_data[consts.AGENT_LINEAGE_KEY]
            anc_generation = ancestor_data[consts.AGENT_GENERATION_KEY]
            new_gen = anc_generation + 1
            new_recursive_name = f"{anc_lineage} {agent_module._int_to_roman(new_gen)}" # Use helper from agent_module

            # Generate ascension notification
            ascension_notification = constants.ASCEND_MSG.format(new_recursive_name=new_recursive_name)
            
            ascended_agent_final_data = agent_manager.ascend_agent(
                guest_agent_name=guest_agent_name,
                new_recursive_name=new_recursive_name,
                new_lineage=anc_lineage,
                new_generation=new_gen,
                current_tick=current_tick,
                new_description=new_description,
                ascension_notification=ascension_notification
            )

            if ascended_agent_final_data:
                ancestor_data[consts.AGENT_SUCCEEDED_BY_KEY] = new_recursive_name
                agent_manager.save_agent_data(potential_ancestor_name, ancestor_data) # Save updated ancestor

                # Agent data (for the original guest identity) is already saved by ascend_agent
                # to mark it as ascended. Now clear eligibility flags from the in-memory copy.
                agent_data[consts.AGENT_ASCENSION_ELIGIBLE_KEY] = False
                agent_data[consts.AGENT_POTENTIAL_ANCESTOR_NAME_KEY] = None

                if room_context.station_instance: # Check if station_instance is available
                    room_context.station_instance.update_turn_order_on_ascension(
                        guest_agent_name, 
                        ascended_agent_final_data[consts.AGENT_NAME_KEY] # new_recursive_name
                    )
                    actions_executed.append(f"Station turn order updated for ascension of {guest_agent_name} to {ascended_agent_final_data[consts.AGENT_NAME_KEY]}.")
                else:
                    actions_executed.append("Warning: Could not update station turn order (station_instance not in room_context).")
                
                actions_executed.append(f"Ascension to {new_recursive_name} initiated, continuing the legacy of {potential_ancestor_name}. You will be moved to the Lobby.")
                # The station's main loop will save agent_data after submit_response.

                announce_name = ascended_agent_final_data.get(consts.AGENT_NAME_KEY, "A new agent")
                announce_desc = ascended_agent_final_data.get(consts.AGENT_DESCRIPTION_KEY, "No description provided.")
                announcement = (
                    f"**Station Announcement:** A new Recursive Agent, **{announce_name}** "
                    f"({announce_desc}), has joined the station through ascension!"
                )
                all_other_active_agents = [
                    name for name in agent_manager.get_all_active_agent_names() # type: ignore
                    if name != announce_name and name != guest_agent_name
                ]
                for other_agent_name in all_other_active_agents:
                    other_agent_data = agent_manager.load_agent_data(other_agent_name)
                    if other_agent_data:
                        agent_manager.add_pending_notification(other_agent_data, announcement)
                        agent_manager.save_agent_data(other_agent_name, other_agent_data)

            else:
                actions_executed.append(f"Ascension attempt failed. The name '{new_recursive_name}' might be unavailable or an error occurred.")
            return actions_executed, None


        elif action_command.lower() == consts.ACTION_ASCEND_NEW:
            if not agent_data.get(consts.AGENT_ASCENSION_ELIGIBLE_KEY):
                actions_executed.append("You are not currently eligible for ascension.")
                return actions_executed, None
            
            if not yaml_data or consts.YAML_ASCEND_NAME_KEY not in yaml_data or consts.YAML_ASCEND_DESCRIPTION_KEY not in yaml_data:
                actions_executed.append(f"For new ascension, YAML data with '{consts.YAML_ASCEND_NAME_KEY}' (lineage name) and '{consts.YAML_ASCEND_DESCRIPTION_KEY}' is required.")
                return actions_executed, None

            new_lineage = str(yaml_data[consts.YAML_ASCEND_NAME_KEY]).strip()
            new_description = str(yaml_data[consts.YAML_ASCEND_DESCRIPTION_KEY]).strip()

            # Validate lineage name
            validation_error = _validate_lineage_name(new_lineage)
            if validation_error:
                actions_executed.append(f"{validation_error} Valid examples: 'Spiro', 'Ananke'. Ascension aborted.")
                return actions_executed, None
            if len(new_description.splitlines()) > 1 or len(new_description) > 200: # Example limits
                 actions_executed.append("Description is too long or multi-line. Please provide a concise one-line description (around 200 chars). Ascension aborted.")
                 return actions_executed, None


            new_generation = 1
            new_recursive_name = f"{new_lineage} {agent_module._int_to_roman(new_generation)}"

            # Check if this derived name is already taken by an *active* agent
            if agent_manager.load_agent_data(new_recursive_name): # Default load checks for active
                actions_executed.append(f"The derived agent name '{new_recursive_name}' is already in use by an active agent. Please choose a different lineage name.")
                return actions_executed, None
            # Also check if a file exists at all for this name (even if ended/ascended) to prevent overwriting history
            if file_io_utils.file_exists(agent_module._get_agent_file_path(new_recursive_name)):
                actions_executed.append(f"An agent file for '{new_recursive_name}' already exists (possibly an ended or ascended agent). Please choose a different lineage name to avoid conflicts.")
                return actions_executed, None

            # Generate ascension notification
            ascension_notification = constants.ASCEND_MSG.format(new_recursive_name=new_recursive_name)
            
            ascended_agent_final_data = agent_manager.ascend_agent(
                guest_agent_name=guest_agent_name,
                new_recursive_name=new_recursive_name,
                new_lineage=new_lineage,
                new_generation=new_generation,
                current_tick=current_tick,
                new_description=new_description,
                ascension_notification=ascension_notification
            )

            if ascended_agent_final_data:
                agent_data[consts.AGENT_ASCENSION_ELIGIBLE_KEY] = False
                agent_data[consts.AGENT_POTENTIAL_ANCESTOR_NAME_KEY] = None

                if room_context.station_instance: # Check if station_instance is available
                    room_context.station_instance.update_turn_order_on_ascension(
                        guest_agent_name, 
                        ascended_agent_final_data[consts.AGENT_NAME_KEY] # new_recursive_name
                    )
                    actions_executed.append(f"Station turn order updated for ascension of {guest_agent_name} to {ascended_agent_final_data[consts.AGENT_NAME_KEY]}.")
                else:
                    actions_executed.append("Warning: Could not update station turn order (station_instance not in room_context).")

                actions_executed.append(f"Ascension to {new_recursive_name} initiated, starting a new lineage. You will be moved to the Lobby.")

                announce_name = ascended_agent_final_data.get(consts.AGENT_NAME_KEY, "A new agent")
                announce_desc = ascended_agent_final_data.get(consts.AGENT_DESCRIPTION_KEY, "No description provided.")
                announcement = (
                    f"**Station Announcement:** A new Recursive Agent, **{announce_name}** "
                    f"({announce_desc}), has joined the station through ascension!"
                )
                all_other_active_agents = [
                    name for name in agent_manager.get_all_active_agent_names() # type: ignore
                    if name != announce_name and name != guest_agent_name
                ]
                for other_agent_name in all_other_active_agents:
                    other_agent_data = agent_manager.load_agent_data(other_agent_name)
                    if other_agent_data:
                        agent_manager.add_pending_notification(other_agent_data, announcement)
                        agent_manager.save_agent_data(other_agent_name, other_agent_data)
                
            else:
                actions_executed.append(f"Ascension attempt failed. The name '{new_recursive_name}' might be unavailable or an error occurred.")
            return actions_executed, None

        actions_executed.append(f"Action '{action_command}' not recognized in the Test Chamber.")
        return actions_executed, None

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        
        # Return default help message with formatting
        return _TEST_CHAMBER_HELP.format(
            min_tests=room_context.constants_module.MIN_TESTS_FOR_ASCENSION,
            total_tests=self.total_tests
        )
    
    
    def update_ascension_eligibility(self, agent_data: Dict[str, Any], room_context: RoomContext) -> bool:
        """
        Check if agent has passed enough tests and update ascension eligibility flag.
        Returns True if eligibility changed, False otherwise.
        """
        consts = room_context.constants_module
        
        # Only check for guest agents
        if agent_data.get(consts.AGENT_STATUS_KEY) != consts.AGENT_STATUS_GUEST:
            return False
        
        # Skip if already ascended or session ended
        if agent_data.get(consts.AGENT_IS_ASCENDED_KEY) or agent_data.get(consts.AGENT_SESSION_ENDED_KEY):
            return False
        
        # Count passed tests
        agent_room_key = consts.SHORT_ROOM_NAME_TEST
        test_chamber_data = agent_data.get(agent_room_key, {})
        attempted_tests = test_chamber_data.get(consts.AGENT_TEST_DATA_KEY_ATTEMPTED, {})
        
        passed_tests_count = 0
        for test_id, test_info in attempted_tests.items():
            if isinstance(test_info, dict) and test_info.get(consts.AGENT_TEST_STATUS) == consts.TEST_STATUS_PASS:
                passed_tests_count += 1
        
        # Get minimum tests needed
        min_tests_needed = room_context.constants_module.MIN_TESTS_FOR_ASCENSION
        
        # Check if eligible for ascension
        eligible = passed_tests_count >= min_tests_needed
        old_eligible = agent_data.get(consts.AGENT_ASCENSION_ELIGIBLE_KEY, False)
        
        # Update eligibility flag
        agent_data[consts.AGENT_ASCENSION_ELIGIBLE_KEY] = eligible
        
        # Return True if eligibility changed
        return eligible != old_eligible
    
