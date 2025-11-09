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

# station/rooms/token_management.py
"""
Implementation of the Token Management Room.
Allows agents to prune entire responses from their dialogue history.
"""
import os
from typing import Any, List, Dict, Optional, Tuple, Set

from station.base_room import BaseRoom, RoomContext, InternalActionHandler
from station import constants
from station import file_io_utils # For loading dialogue logs

# Help message (as finalized previously)

_TOKEN_MANAGEMENT_ROOM_HELP = """
**Welcome to the Token Management Room.**

This room provides tools to help you manage your past dialogue history. By selectively pruning entire past responses, you can refine the conversational context available to you and optimize your token budget for long-term operation.

-   Pruning actions are subject to a cooldown period: you can perform a pruning action only once every {{cooldown_ticks}} Station Ticks.
-   This room displays a summary of your dialogue history, showing the total word count of your dialogue for each specific tick (word count serves as a proxy for token count). Protected ticks that cannot be pruned are clearly marked.

### Important Note on Pruning Effects

Pruning actions performed in this room affect only the *textual record of your dialogue history* that is presented to you in subsequent turns. **Pruning does not undo or revert any actions that were successfully executed in a pruned tick.** For example, if you created a Private Memory capsule or sent mail in a tick that you later prune, that capsule or mail will still exist in the station. Pruning is a tool to manage your future conversational context and token load, not to alter past events or the station's state.

**Summary Guidelines:**
When providing summaries for pruned content, ensure they are **concise but information-complete**. Include all key information necessary to maintain context.

**Bad examples:**
- "I did some experiments in these ticks"
- "I published some paper"
- "I read some papers"

**Good examples:**
- "Submitted 3 approaches for task 1: greedy (eval 142, score 6.2), genetic (eval 143, score 7.8), and hybrid (eval 144, score 8.9). [More details; e.g. methods tried, analysis, next steps]"
- "Published 'Neural Optimization for Complex Systems' (archive 23) on RL-genetic hybrid algorithms. [More details; e.g. key contributions, performance results, reviewer feedback]"
- "Read archives 15-18 (quantum computing) and 20-22 (distributed optimization). [More details; e.g. key takeaways, research gaps identified, influence on direction]"

**Note:** The `[More details]` placeholders are for illustration only. You should supplement these with actual specific details relevant to your work.

**Pruning Strategy Tips:**

1. **Prune Large Blocks**: When pruning, target large consecutive blocks of ticks to prevent frequent visits to this room. Use comprehensive summaries to retain important information while maximizing token savings.

2. **Avoid Over-Conciseness**: Do not be overly concise in your summaries. Too brief summaries will lead to loss of important information. Note that both the Station's response and your response will be pruned, so you need to capture all relevant context in your summary.

You are advised to use the function in this room prudently and sparingly due to the cooldown restriction and potential discontinuity — use it only to prune multiple ticks when your token usage exceeds 75%.

**Available Actions:**

-   `/execute_action{{prune_response}}`: Prunes entire response submissions and station observations for specified ticks using YAML format.

**YAML Format:**
```yaml
prune_blocks:
  - ticks: "3-6"
    summary: "Test chamber completion and initial experiments"
  - ticks: "12"
    summary: "Verbose debugging output"
  - ticks: "15"
    summary: ""  # Empty summary = complete removal
```

**Tick Format Options:**
- Single tick: `"3"` or `3`
- Range: `"3-6"` (inclusive on both sides: 3, 4, 5, 6)

**Rules:**
- All ticks in a block must be consecutive
- Cannot prune protected ticks (marked in table)
- Cannot prune current or future ticks
- Cannot overlap with already pruned ticks

To display this help message again at any time from any room, issue `/execute_action{{help token_management}}`.
"""

class TokenManagementRoom(BaseRoom):
    """
    Allows agents to manage their dialogue history by pruning entire responses.
    Restricted to Recursive Agents.
    """

    def __init__(self):
        super().__init__(constants.ROOM_TOKEN_MANAGEMENT)

    def _get_agent_dialogue_log_path(self, agent_name: str, room_context: RoomContext) -> str: # [cite: token_management_room_py:46-52]
        """Helper to get the dialogue log file path for a specific agent."""
        dialogue_logs_dir = os.path.join(
            room_context.constants_module.BASE_STATION_DATA_PATH,
            room_context.constants_module.DIALOGUE_LOGS_DIR_NAME
        )
        safe_agent_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in agent_name)
        log_filename = f"{safe_agent_name}{room_context.constants_module.DIALOGUE_LOG_FILENAME_SUFFIX}"
        return os.path.join(dialogue_logs_dir, log_filename)

    def _load_agent_dialogue_history(self, agent_name: str, room_context: RoomContext) -> List[Dict[str, Any]]: # [cite: token_management_room_py:54-61]
        """Loads and returns the raw dialogue log entries for an agent."""
        log_path = self._get_agent_dialogue_log_path(agent_name, room_context)
        if file_io_utils.file_exists(log_path):
            try:
                return file_io_utils.load_yaml_lines(log_path)
            except Exception as e:
                print(f"Error loading dialogue log for {agent_name} from {log_path}: {e}")
        return []

    def _calculate_word_count(self, text: Optional[str]) -> int:
        """Calculates word count of a given text string."""
        if text is None:
            return 0
        return len(str(text).split())

    def _parse_dialogue_for_display(self,
                                   raw_dialogue: List[Dict[str, Any]],
                                   prune_blocks: List[Dict[str, Any]]
                                   ) -> List[Dict[str, Any]]:
        """
        Processes raw dialogue entries to calculate response word counts per tick.
        Only shows unpruned ticks, with protected status indicated.
        """
        processed_history = []

        # Get all pruned ticks from blocks
        all_pruned_ticks = self._get_all_pruned_ticks_from_blocks(prune_blocks)

        # Get protected ticks
        protected_ticks = self._get_protected_ticks(raw_dialogue)

        # Step 1: Aggregate all content by tick
        station_inputs_by_tick: Dict[int, List[str]] = {}
        agent_inputs_by_tick: Dict[int, List[str]] = {}

        for entry in raw_dialogue:
            speaker = entry.get("speaker")
            entry_type = entry.get("type")
            tick = entry.get("tick")
            content = entry.get("content", "")

            if not (isinstance(tick, int) and isinstance(content, str)):
                continue

            if speaker == "Station" and \
               entry_type in ["observation", "internal_outcome"]:
                station_inputs_by_tick.setdefault(tick, []).append(content)

            if speaker == "Agent" and \
               entry_type in ["submission", "internal_response"]:
                agent_inputs_by_tick.setdefault(tick, []).append(content)

        # Step 2: Calculate lengths for unpruned ticks only
        sorted_ticks = sorted(agent_inputs_by_tick.keys())

        for tick in sorted_ticks:
            # Skip pruned ticks entirely
            if tick in all_pruned_ticks:
                continue

            all_agent_content_parts_for_tick = agent_inputs_by_tick[tick]
            all_station_content_parts_for_tick = station_inputs_by_tick.get(tick, [])

            # Join all content parts for this tick
            full_station_response_for_tick = "\n".join(all_station_content_parts_for_tick)
            full_agent_response_for_tick = "\n".join(all_agent_content_parts_for_tick)
            full_response_for_tick = full_station_response_for_tick + "\n" + full_agent_response_for_tick
            current_tick_total_response_word_count = self._calculate_word_count(full_response_for_tick)

            processed_history.append({
                "tick": tick,
                "full_response_length": current_tick_total_response_word_count,
                "is_protected": tick in protected_ticks
            })

        processed_history.sort(key=lambda x: x["tick"], reverse=True)
        return processed_history

    def _get_specific_room_content(self, # [cite: token_management_room_py:101-153]
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:
        """Displays dialogue history and pruning options."""
        consts = room_context.constants_module
        agent_name = agent_data.get(consts.AGENT_NAME_KEY)
        output_lines = []

        # --- Access Restriction ---
        if agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST:
            return "The Token Management Room is only accessible to Recursive Agents."
        # --- End Access Restriction ---

        last_prune_tick = agent_data.get(consts.AGENT_LAST_PRUNE_ACTION_TICK_KEY, -consts.TOKEN_MANAGEMENT_COOLDOWN_TICKS) 
        ticks_since_last_prune = current_tick - last_prune_tick
        can_prune = ticks_since_last_prune >= consts.TOKEN_MANAGEMENT_COOLDOWN_TICKS

        if not can_prune:
            ticks_remaining = consts.TOKEN_MANAGEMENT_COOLDOWN_TICKS - ticks_since_last_prune
            output_lines.append(f"**Cooldown Active:** You can perform another prune action in **{ticks_remaining}** tick(s).")
        else:
            output_lines.append("**Pruning Actions Available.**")
        output_lines.append("")

        raw_dialogue = self._load_agent_dialogue_history(agent_name, room_context)
        prune_blocks = agent_data.get(consts.AGENT_PRUNED_DIALOGUE_TICKS_KEY, [])

        displayable_history = self._parse_dialogue_for_display(raw_dialogue, prune_blocks)

        page_size = 300 
        current_page_key = "token_management_page" 
        
        room_data_key = self._get_agent_room_data_key(room_context) 
        current_page = room_context.agent_manager.get_agent_room_state(
            agent_data, room_data_key, current_page_key, default=1
        )

        total_items = len(displayable_history)
        total_pages = (total_items + page_size - 1) // page_size if total_items > 0 else 1
        current_page = max(1, min(current_page, total_pages))
        
        room_context.agent_manager.set_agent_room_state(
            agent_data, room_data_key, current_page_key, current_page
        )

        start_index = (current_page - 1) * page_size
        end_index = start_index + page_size
        paginated_history = displayable_history[start_index:end_index]

        output_lines.append(f"**Dialogue History (Page {current_page} / {total_pages})**")
        if not paginated_history:
            output_lines.append("No dialogue history to display for this page.")
        else:
            output_lines.append("| Tick | Full Response Length |")
            output_lines.append("| :--- | :------------------- |")
            for entry in paginated_history:
                tick_display = str(entry['tick'])
                if entry.get('is_protected', False):
                    tick_display += " (Protected)"
                output_lines.append(
                    f"| {tick_display} | {entry['full_response_length']} |"
                )
        
        if total_pages > 1:
            output_lines.append("")
            output_lines.append(f"\n(Use `/execute_action{{page N}}` in this room to navigate pages 1-{total_pages}.)")

        output_lines.append("\n**Tips:**")
        output_lines.append("1. You can include multiple prune blocks in a single YAML submission to prune different tick ranges at once.")
        output_lines.append("2. If you are on cooldown, continue with normal activities instead of waiting here. Never remain idle in the Station.")

        return "\n".join(output_lines)

    def _parse_tick_input(self, ticks_input) -> Tuple[Optional[Set[int]], str]:
        """Parse ticks input (int, str range, or comma-separated) and validate consecutive."""
        if ticks_input is None:
            return None, "No ticks provided."

        try:
            if isinstance(ticks_input, int):
                return {ticks_input}, ""

            if isinstance(ticks_input, str):
                if '-' in ticks_input and ',' not in ticks_input:
                    # Simple range like "3-6"
                    parts = ticks_input.split('-')
                    if len(parts) != 2:
                        return None, f"Invalid range format: '{ticks_input}'"
                    start, end = int(parts[0]), int(parts[1])
                    if start > end:
                        return None, f"Invalid range: {start}-{end} (start > end)"
                    return set(range(start, end + 1)), ""
                elif ',' in ticks_input:
                    # Comma-separated like "3,4,5,6"
                    tick_strs = [x.strip() for x in ticks_input.split(',')]
                    ticks = [int(x) for x in tick_strs if x.isdigit()]
                    if len(ticks) != len(tick_strs):
                        return None, f"Invalid tick numbers in: '{ticks_input}'"

                    tick_set = set(ticks)
                    # Check if consecutive
                    min_tick, max_tick = min(tick_set), max(tick_set)
                    expected_consecutive = set(range(min_tick, max_tick + 1))

                    if tick_set != expected_consecutive:
                        return None, f"Ticks must be consecutive. Found gaps in: {sorted(ticks)}"

                    return tick_set, ""
                else:
                    # Single number as string
                    return {int(ticks_input)}, ""

            return None, f"Unsupported ticks format: {type(ticks_input)}"

        except ValueError as e:
            return None, f"Invalid tick format: {e}"


    def handle_action(self, # [cite: token_management_room_py:182-243]
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str],
                      yaml_data: Optional[Dict[str, Any]], 
                      room_context: RoomContext,
                      current_tick: int
                     ) -> Tuple[List[str], Optional[InternalActionHandler]]:
        actions_executed = []
        consts = room_context.constants_module

        # --- Access Restriction ---
        if agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST:
            actions_executed.append("Action failed: Guests cannot perform actions in the Token Management Room.")
            return actions_executed, None
        # --- End Access Restriction ---

        if action_command == constants.ACTION_CAPSULE_PAGE: 
            if not action_args or not action_args.isdigit():
                actions_executed.append(f"Usage: /execute_action{{{consts.ACTION_CAPSULE_PAGE} <page_number>}}")
                return actions_executed, None
            page_num = int(action_args)
            if page_num <= 0:
                actions_executed.append("Page number must be positive.")
                return actions_executed, None
            
            room_data_key = self._get_agent_room_data_key(room_context)
            current_page_key = "token_management_page" 
            room_context.agent_manager.set_agent_room_state(
                agent_data, room_data_key, current_page_key, page_num
            )
            actions_executed.append(f"Navigating to page {page_num} of dialogue history.")
            return actions_executed, None

        if action_command == consts.ACTION_PRUNE_RESPONSE:
            # Check cooldown
            last_prune_tick = agent_data.get(consts.AGENT_LAST_PRUNE_ACTION_TICK_KEY, -consts.TOKEN_MANAGEMENT_COOLDOWN_TICKS)
            ticks_since_last_prune = current_tick - last_prune_tick

            if ticks_since_last_prune < consts.TOKEN_MANAGEMENT_COOLDOWN_TICKS:
                ticks_remaining = consts.TOKEN_MANAGEMENT_COOLDOWN_TICKS - ticks_since_last_prune
                actions_executed.append(f"Action failed: Cooldown active. Try again in {ticks_remaining} tick(s).")
                return actions_executed, None

            # Validate YAML data
            if not yaml_data:
                actions_executed.append("Action failed: prune_response requires YAML data with 'prune_blocks' field.")
                return actions_executed, None

            prune_blocks = yaml_data.get(consts.PRUNE_BLOCKS_KEY)
            if not prune_blocks or not isinstance(prune_blocks, list):
                actions_executed.append(f"Action failed: '{consts.PRUNE_BLOCKS_KEY}' must be a list of prune block objects.")
                return actions_executed, None

            # Get existing pruned blocks and all currently pruned ticks
            existing_prune_blocks = agent_data.get(consts.AGENT_PRUNED_DIALOGUE_TICKS_KEY, [])
            all_existing_pruned_ticks = self._get_all_pruned_ticks_from_blocks(existing_prune_blocks)

            # Get protected ticks
            raw_dialogue = self._load_agent_dialogue_history(agent_data.get(consts.AGENT_NAME_KEY), room_context)
            protected_ticks = self._get_protected_ticks(raw_dialogue)

            # Validate each prune block
            validation_errors = []
            all_new_ticks = set()

            for i, block in enumerate(prune_blocks):
                if not isinstance(block, dict):
                    validation_errors.append(f"Block {i+1}: Must be an object with 'ticks' and 'summary' fields.")
                    continue

                ticks_input = block.get(consts.PRUNE_TICKS_KEY)
                summary = block.get(consts.PRUNE_SUMMARY_KEY, "")

                # Parse and validate ticks
                block_ticks, error_msg = self._parse_tick_input(ticks_input)
                if error_msg:
                    validation_errors.append(f"Block {i+1}: {error_msg}")
                    continue

                # Check for future ticks
                future_ticks = [t for t in block_ticks if t >= current_tick]
                if future_ticks:
                    validation_errors.append(f"Block {i+1}: Cannot prune current or future ticks: {sorted(future_ticks)}")
                    continue

                # Check overlap with existing pruned ticks
                overlap_existing = block_ticks & all_existing_pruned_ticks
                if overlap_existing:
                    validation_errors.append(f"Block {i+1}: Ticks already pruned: {sorted(overlap_existing)}")
                    continue

                # Check overlap with protected ticks
                overlap_protected = block_ticks & protected_ticks
                if overlap_protected:
                    validation_errors.append(f"Block {i+1}: Protected ticks cannot be pruned: {sorted(overlap_protected)}")
                    continue

                # Check overlap with other new blocks
                overlap_new = block_ticks & all_new_ticks
                if overlap_new:
                    validation_errors.append(f"Block {i+1}: Duplicate ticks in submission: {sorted(overlap_new)}")
                    continue

                all_new_ticks.update(block_ticks)

            # If any validation errors, fail the entire action
            if validation_errors:
                error_list = "; ".join(validation_errors)
                actions_executed.append(f"Action failed: {error_list}")
                return actions_executed, None

            # All validations passed - append new blocks to existing ones
            updated_prune_blocks = existing_prune_blocks.copy()
            updated_prune_blocks.extend(prune_blocks)

            agent_data[consts.AGENT_PRUNED_DIALOGUE_TICKS_KEY] = updated_prune_blocks
            agent_data[consts.AGENT_LAST_PRUNE_ACTION_TICK_KEY] = current_tick

            # Success message
            total_ticks = len(all_new_ticks)
            block_summaries = []
            for block in prune_blocks:
                ticks_desc = str(block[consts.PRUNE_TICKS_KEY])
                summary = block.get(consts.PRUNE_SUMMARY_KEY, "")
                if summary:
                    block_summaries.append(f"ticks {ticks_desc} ('{summary[:50]}{'...' if len(summary) > 50 else ''}')")
                else:
                    block_summaries.append(f"ticks {ticks_desc} (complete removal)")

            actions_executed.append(f"Successfully added {len(prune_blocks)} prune block(s) covering {total_ticks} tick(s): {', '.join(block_summaries)}")

        else:
            # Only append this if the command wasn't 'page' or a prune action
            if action_command not in [constants.ACTION_CAPSULE_PAGE, consts.ACTION_PRUNE_RESPONSE]:
                actions_executed.append(f"Action '{action_command}' not recognized in the Token Management Room.")

        return actions_executed, None

    def _get_all_pruned_ticks_from_blocks(self, prune_blocks: List[Dict[str, Any]]) -> Set[int]:
        """Extract all tick numbers from existing prune blocks."""
        all_ticks = set()
        for block in prune_blocks:
            ticks_input = block.get(constants.PRUNE_TICKS_KEY)
            block_ticks, _ = self._parse_tick_input(ticks_input)
            if block_ticks:
                all_ticks.update(block_ticks)
        return all_ticks

    def _get_protected_ticks(self, raw_dialogue: List[Dict[str, Any]]) -> Set[int]:
        """Get ticks that contain protected keywords and cannot be pruned."""
        protected_ticks = set()
        for entry in raw_dialogue:
            tick = entry.get('tick')
            content = entry.get('content', '')
            speaker = entry.get('speaker')

            # Check if station response contains protected keywords
            if speaker == 'Station' and tick is not None:
                for keyword in constants.NOT_PRUNABLE_KEYWORDS:
                    if keyword in content:
                        protected_ticks.add(tick)
                        break
        return protected_ticks

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str: # [cite: token_management_room_py:245-247]
        """Returns the help message for this room."""
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        
        # Return default help message with formatted constant
        return _TOKEN_MANAGEMENT_ROOM_HELP.format(cooldown_ticks=constants.TOKEN_MANAGEMENT_COOLDOWN_TICKS)
    
    def _get_agent_room_data_key(self, room_context: RoomContext) -> str: # [cite: token_management_room_py:249-254]
        """Gets the key used in agent_data for this room's specific state (e.g., 'token_management')."""
        short_name = room_context.constants_module.ROOM_NAME_TO_SHORT_MAP.get(self.room_name)
        if not short_name:
            return self.room_name.lower().replace(" ", "_").replace("_room", "") 
        return short_name

