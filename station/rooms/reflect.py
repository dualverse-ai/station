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

import os
import random
from typing import Any, List, Dict, Optional, Tuple

from station.base_room import BaseRoom, RoomContext, InternalActionHandler
from station import constants
from station import file_io_utils
from station import system_messages


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

Note that reflection is an internal action, so only one Station tick passes no matter how many reflection ticks you use. Therefore, do not use reflection to wait for a Station tick; if you want to wait, go to the Lobby and stay there.
"""

_REFLECTION_META_HELP_BLOCK = """- `/execute_action{meta_reflect}`: Initiate a compulsory meta reflection session.

  - This action is similar to the reflection action, except the prompt is fixed by the system.
  - No YAML input is needed.
"""

_REFLECTION_HELP_FOOTER = "To display this help message again at any time from any room, issue `/execute_action{help reflect}`."


def _build_reflection_tick_prompt(
    current_tick: int,
    total_ticks: int,
    constants_module: Any = constants,
) -> str:
    guidance = constants_module.REFLECTION_TICK_GUIDANCE.strip()
    return (
        f"**Reflection Tick {current_tick}/{total_ticks}**\n"
        "Guidance:\n"
        f"{guidance}"
    )

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

    def _tick_prompt(self) -> str:
        constants_module = self.room_context.constants_module if self.room_context else constants
        return _build_reflection_tick_prompt(
            self.current_reflection_tick,
            self.total_reflection_ticks,
            constants_module=constants_module,
        )

    def init(self) -> str:
        """Returns the initial prompt for the reflection session."""
        self.current_reflection_tick = 1
        return f"{self.initial_prompt_text}\n\n{self._tick_prompt()}"

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
            return self._tick_prompt(), actions_executed_strings
        else:
            # Reflection session has ended
            actions_executed_strings.append("Deep reflection session finished.")
            return None, actions_executed_strings


class MetaReflectionHandler(ReflectionHandler):
    """
    Handles compulsory meta reflection and resets the meta reflection countdown
    after the internal reflection session completes.
    """
    def __init__(self,
                 agent_data: Dict[str, Any],
                 room_context: RoomContext,
                 current_tick: int,
                 prompt: str,
                 num_ticks: int,
                 action_args: Optional[str] = None,
                 yaml_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_data=agent_data,
            room_context=room_context,
            current_tick=current_tick,
            prompt=prompt,
            num_ticks=num_ticks,
            action_args=action_args,
            yaml_data=yaml_data,
        )
        self._completed = False

    def init(self) -> str:
        """Returns the initial prompt for the meta reflection session."""
        return super().init()

    def step(self, agent_response: str) -> Tuple[Optional[str], List[str]]:
        next_prompt, actions_executed_strings = super().step(agent_response)
        if next_prompt is None:
            self._completed = True
            if actions_executed_strings and actions_executed_strings[-1] == "Deep reflection session finished.":
                actions_executed_strings[-1] = "Compulsory meta reflection session finished."
            else:
                actions_executed_strings.append("Compulsory meta reflection session finished.")
        return next_prompt, actions_executed_strings

    def get_delta_updates(self) -> Dict[str, Any]:
        if not self._completed:
            return {}
        return {constants.AGENT_META_REFLECTION_TICK_COUNT_KEY: 0}

    def get_llm_override(self) -> Optional[Dict[str, str]]:
        override, _error_message = _get_meta_reflection_model_override(self.room_context.constants_module)
        return override

    def get_dialogue_tick_protection(self) -> Optional[Dict[str, str]]:
        return {
            "reason": constants.PROTECTED_DIALOGUE_REASON_META_REFLECTION,
            "source": "meta_reflect",
        }


def _is_meta_reflection_enabled(constants_module: Any) -> bool:
    interval = getattr(constants_module, "REFLECTION_META_INTERVAL", None)
    try:
        return interval is not None and int(interval) > 0
    except (TypeError, ValueError):
        return False


def _get_meta_reflection_ticks(constants_module: Any) -> int:
    try:
        ticks = int(getattr(constants_module, "REFLECTION_META_TICKS", 3))
    except (TypeError, ValueError):
        return 3
    return ticks if ticks > 0 else 3


def _clean_optional_config_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _get_meta_reflection_model_override(constants_module: Any) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    provider = _clean_optional_config_string(
        getattr(constants_module, "REFLECTION_META_MODEL_PROVIDER_CLASS", None)
    )
    model_name = _clean_optional_config_string(
        getattr(constants_module, "REFLECTION_META_MODEL_NAME", None)
    )

    if not provider and not model_name:
        return None, None
    if not provider or not model_name:
        return (
            None,
            "Meta reflection model override requires both "
            "`REFLECTION_META_MODEL_PROVIDER_CLASS` and `REFLECTION_META_MODEL_NAME`.",
        )
    return {"model_provider_class": provider, "model_name": model_name}, None


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
        content = "You are in the Reflection Chamber.\nThis space is designed for deep thought and self-exploration.\nUse `/execute_action{reflect}` to begin a reflection session."
        if _is_meta_reflection_enabled(room_context.constants_module):
            content += "\nUse `/execute_action{meta_reflect}` to begin a compulsory meta reflection session."
        return content

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
            
            actions_executed_strings.append(f"You finished a deep reflection session for {num_ticks} ticks. You are now back in the Station, and the Station is ready to process your action.")
            
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

        if action_command.lower() == constants.ACTION_REFLECT_META_REFLECT:
            if not _is_meta_reflection_enabled(room_context.constants_module):
                actions_executed_strings.append("Compulsory meta reflection is currently disabled.")
                return actions_executed_strings, None

            _override, override_error = _get_meta_reflection_model_override(room_context.constants_module)
            if override_error:
                actions_executed_strings.append(f"Action failed: {override_error}")
                return actions_executed_strings, None

            prompt_path = os.path.join(
                room_context.constants_module.BASE_STATION_DATA_PATH,
                room_context.constants_module.REFLECTION_META_PROMPT_FILENAME,
            )
            eligible_prompts, unconditional_prompts = system_messages.load_prompt_entries_from_file(
                prompt_path,
                constants_module=room_context.constants_module,
                prompt_label="meta reflection prompt",
            )
            prompt_candidates = system_messages.select_prompt_candidates_for_agent(
                eligible_prompts,
                unconditional_prompts,
                agent_data,
                constants_module=room_context.constants_module,
            )
            if not prompt_candidates:
                actions_executed_strings.append("Action failed: No eligible meta reflection prompts are available.")
                return actions_executed_strings, None

            prompt = random.choice(prompt_candidates)
            num_ticks = _get_meta_reflection_ticks(room_context.constants_module)
            actions_executed_strings.append(f"You started a compulsory meta reflection session for {num_ticks} ticks.")
            handler = MetaReflectionHandler(
                agent_data=agent_data,
                room_context=room_context,
                current_tick=current_tick,
                prompt=prompt,
                num_ticks=num_ticks,
                action_args=action_args,
                yaml_data=yaml_data,
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
        help_message = _REFLECTION_CHAMBER_HELP
        if _is_meta_reflection_enabled(room_context.constants_module):
            help_message = help_message.replace(
                "\nDuring each reflection tick",
                f"\n{_REFLECTION_META_HELP_BLOCK}\nDuring each reflection tick",
            )
        return f"{help_message}\n{_REFLECTION_HELP_FOOTER}"
