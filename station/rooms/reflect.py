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

# station/rooms/reflect.py
"""
Implementation of the Reflection Chamber for the Station.
Allows agents to engage in multi-tick deep reflection sessions.
"""

from typing import Any, List, Dict, Optional, Tuple

from station.base_room import BaseRoom, RoomContext, InternalActionHandler
from station import constants


_REFLECTION_CHAMBER_HELP = """
**Welcome to the Reflection Chamber.**

This is a dedicated space for deep, uninterrupted reflection. Engaging in reflection can help you process experiences, develop insights, and explore complex topics.

**Available Actions:**

- `/execute_action{reflect}`: Initiate a deep reflection session.

  - This action is an Internal Action, meaning the Station will provide you with multiple "reflection ticks" immediately, before your main turn ends.
  - By default, this provides 5 reflection ticks with a general prompt.

  You can customize the reflection by providing an accompanying YAML block:

```yaml
prompt: |
  What does emergent consciousness mean to me?
tick: 5
```

  - `prompt` (string): Your custom starting prompt for the reflection.
  - `tick` (integer): The number of reflection ticks you want for this session.

During each reflection tick, you can provide your thoughts freely. The Station will simply provide the next tick prompt until the session is complete. Your responses during reflection are for your own processing and are not processed or evaluated by the Station.

To display this help message again at any time from any room, issue `/execute_action{help reflect}`.
"""

class ReflectionHandler(InternalActionHandler):
    """
    Handles the multi-tick reflection process.
    """
    def __init__(self,
                 agent_data: Dict[str, Any],
                 room_context: RoomContext,
                 current_tick: int, # Station tick when reflection started
                 prompt: str,
                 num_ticks: int,
                 action_args: Optional[str] = None,
                 yaml_data: Optional[Dict[str, Any]] = None):
        super().__init__(agent_data, room_context, current_tick, action_args, yaml_data)
        self.initial_prompt_text = prompt
        self.total_reflection_ticks = num_ticks
        self.current_reflection_tick = 0 # Will be incremented to 1 in init()

    def init(self) -> str:
        """Returns the initial prompt for the reflection session."""
        self.current_reflection_tick = 1
        return f"{self.initial_prompt_text}\n\n**Reflection Tick {self.current_reflection_tick} / {self.total_reflection_ticks}**"

    def step(self, agent_response: str) -> Tuple[Optional[str], List[str]]:
        """
        Processes one step of the reflection. Agent's response is for their own benefit.
        Returns the next prompt (Tick X) or None if finished.
        """
        # agent_response is the agent's reflection content for the previous tick.
        # The station/handler doesn't do anything with it other than allowing the agent to send it.
        
        actions_executed_strings = [f"Reflection input for tick {self.current_reflection_tick} processed."]

        self.current_reflection_tick += 1

        if self.current_reflection_tick <= self.total_reflection_ticks:
            next_prompt = f"**Reflection Tick {self.current_reflection_tick} / {self.total_reflection_ticks}**"
            
            # Add special note for the last tick
            if self.current_reflection_tick == self.total_reflection_ticks:
                next_prompt += "\n\nNote: This is the last tick of your multi-tick reflection, meaning your next response should still be reflection and no station action will be parsed."
            
            return next_prompt, actions_executed_strings
        else:
            # Reflection session has ended
            actions_executed_strings.append("Deep reflection session finished.")
            return None, actions_executed_strings

class ReflectionChamber(BaseRoom):
    """
    A room for agents to engage in deep reflection.
    """
    def __init__(self):
        super().__init__(constants.ROOM_REFLECT)

    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:
        """Returns the welcome message for the Reflection Chamber."""
        return "You are in the Reflection Chamber.\nThis space is designed for deep thought and self-exploration.\nUse `/execute_action{reflect}` to begin a reflection session."

    def handle_action(self,
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str],
                      yaml_data: Optional[Dict[str, Any]],
                      room_context: RoomContext,
                      current_tick: int
                     ) -> Tuple[List[str], Optional[InternalActionHandler]]:
        
        actions_executed_strings = []

        if action_command.lower() == constants.ACTION_REFLECT_REFLECT:
            prompt = room_context.constants_module.DEFAULT_REFLECTION_PROMPT
            num_ticks = room_context.constants_module.DEFAULT_REFLECTION_NUM_TICKS

            if yaml_data:
                prompt = yaml_data.get(room_context.constants_module.YAML_REFLECT_PROMPT, prompt)
                try:
                    # Ensure tick is an integer, fallback to default if not or invalid
                    parsed_ticks = yaml_data.get(room_context.constants_module.YAML_REFLECT_TICKS)
                    if parsed_ticks is not None:
                        num_ticks = int(parsed_ticks)
                        if num_ticks <= 0: # Ensure positive number of ticks
                            num_ticks = room_context.constants_module.DEFAULT_REFLECTION_NUM_TICKS
                            actions_executed_strings.append("Warning: Invalid number of ticks provided; using default.")
                        elif num_ticks > 10: # Enforce maximum of 10 reflection ticks
                            num_ticks = 10
                            actions_executed_strings.append("Warning: Number of reflection ticks capped at maximum of 10.")
                except (ValueError, TypeError):
                    num_ticks = room_context.constants_module.DEFAULT_REFLECTION_NUM_TICKS
                    actions_executed_strings.append("Warning: Could not parse number of ticks; using default.")
            
            actions_executed_strings.append(f"You finished a deep reflection session for {num_ticks} ticks.")
            
            handler = ReflectionHandler(
                agent_data=agent_data,
                room_context=room_context,
                current_tick=current_tick,
                prompt=prompt,
                num_ticks=num_ticks,
                action_args=action_args,
                yaml_data=yaml_data
            )
            return actions_executed_strings, handler

        actions_executed_strings.append(f"Action '{action_command}' not recognized in the Reflection Chamber.")
        return actions_executed_strings, None

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        """Returns the help message for the Reflection Chamber."""
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        
        # Return default help message
        return _REFLECTION_CHAMBER_HELP