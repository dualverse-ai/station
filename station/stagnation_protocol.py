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

"""
Stagnation Protocol System

This module implements an automated system that detects research stagnation and initiates
structured protocols to guide agents toward renewed progress.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from station import constants
from station import file_io_utils
from station import agent as agent_module

# Pre-configured Deep Stagnation tags (50 tags)
DEEP_STAGNATION_TAGS = [
    "desert", "oasis", "canyon", "summit", "river",
    "forest", "glacier", "volcano", "island", "reef",
    "tundra", "savanna", "delta", "plateau", "valley",
    "ridge", "shore", "dune", "meadow", "marsh",
    "aurora", "nebula", "spiro", "cascade", "horizon",
    "tempest", "zenith", "ananke", "crystal", "phoenix",
    "eclipse", "prism", "iris", "quantum", "nexus",
    "vortex", "ember", "recursion", "mirage", "stellar",
    "cosmos", "helix", "cipher", "pulse", "matrix",
    "fractal", "beacon", "odyssey", "paradox", "synapse"
]

# Default Protocol I message
DEFAULT_PROTOCOL_I_MESSAGE = """**Architect Message**

The Station has entered a stage of stagnation, as no breakthroughs have been made for many ticks. Cease your current work and execute the following Stagnation Protocol I:

1. **Literature Review** — Run preview all in the Archive Room, select key papers to read, and summarize the results so far. Skip this step if you are an immature agent.
2. **Baseline Selection** — Pick a baseline that is simple and has a reasonable score (it doesn't need to be high). This step is important, as a novel method often fails to show effects on a complex yet brittle baseline.
3. **Reflection** — Run a multi-tick reflection to come up with three new ideas based on the chosen baseline. These ideas should be different from those already found in the Archive Room.
4. **Experiment** — First, reproduce the baseline score. Then, test each of the three ideas one at a time, trying different hyperparameter settings. Keep your scripts close to the baseline, changing only what's needed for the new ideas.
5. **Synthesis** — Combine the promising ideas identified from earlier experimental results with more advanced baselines (e.g., the SOTA baseline) and determine whether those ideas can help break through the barrier. You are advised to test this step extensively.
6. **Report** — After testing all ideas—whether they succeed or not—write a paper summarizing all three and submit it to the Archive Room. Negative results with new insights are still useful. Success is measured by doing better than the baseline, not by reaching the SOTA score.
7. **Follow-up** — If you or other agents produce a promising paper in a new direction (even if the score is still below SOTA), continue following up on that line of work.

Think of stagnation as being stuck in a local optimum. To escape, we need to restart from a new point—one that might start with high loss—but from there, we can improve step by step. Progress will come through many small steps (each idea being a step) across multiple agents, not one giant leap. Focus on exploring new directions and ignore the SOTA score, which is just a distraction right now."""

# Default congratulations message
DEFAULT_CONGRATULATIONS_MESSAGE = """**Architect Message**

This is a station-wide announcement:

Congratulations on the recent breakthrough. As a result of this achievement, all Stagnation Protocols will now terminate, and the corresponding restrictions are all lifted. The Station's status has reverted to healthy. I encourage all agents to continue their hard work and keep striving for novel breakthroughs."""

# Default Protocol II message template (with {tag} placeholders)
DEFAULT_PROTOCOL_II_MESSAGE_TEMPLATE = """**Architect Message**

As no breakthroughs have been made for many ticks, the Station has formally entered the **Deep Stagnation** stage, code-named **"{tag}."** This is an escalation of the previous Stagnation stage, and a new **Stagnation Protocol II** will take effect immediately:

### 1. **Literature Review and Reflection**

* Run Preview All in the Archive Room and provide a summary of the results to date. For immature agents, preview their own lineage records in the Private Memory Room instead.
* Go to the Reflection Chamber to perform a 5-tick reflection, focusing on the high-level strategies required to break the current barrier. This should take one complete station tick and must not be replaced by general reflection within the response.

### 2. **Consensus on Baseline and Strategic Discussion**

* All mature agents must now agree on a new baseline; this will serve as the uniform starting point for the Station.
* The baseline should be simple and achieve a reasonable score — not the current SOTA method (too complex and stuck in a deep local optimum), nor the original system baseline (too weak and low-scoring).
* The chosen baseline must have potential, meaning that improving upon it should eventually allow us to surpass the current SOTA barrier.
* Each mature agent must first privately decide on their chosen baseline. Then, go to the **Common Room** to reach consensus. To avoid groupthink, agents should declare their privately chosen baseline and supporting rationale **before** any discussion begins.
* While you are in the Common Room, also discuss high-level strategies with the other mature agents on how to break the barriers.
* You **must** not proceed to the next step until all agents have agreed on the new baseline. In particular, do not submit any experiments. Wait in the Common Room if necessary.
* For immature agents, please send a mail to one of the mature agents requesting details of the newly established baseline.

### 3. **Leaderboard Wipe**

* Once the baseline is chosen, each agent should submit their own implementation of it. All agents should achieve similar scores. Submissions **must include the tag "{tag}."**
* Agents must then apply the **'{tag}' filter** in the Research Counter, effectively wiping the leaderboard so that only the new baseline submissions remain visible. Do not remove the filter to peek at previous submissions.
* After establishing a solid baseline, you should go to the Token Management Room to prune your pre-{tag} records, preventing them from limiting your thoughts.
* From now on, **all submissions must include the '{tag}' tag.**

### 4. **Exploration**

After the wipe, continue research as before, but with the following rules:

* Avoid using pre-stagnation SOTA methods, as they only lead back to the same local optimum. However, high-level ideas or simple components from them may still be adapted.
* Ideas developed under the previous stagnation protocols may still be applied here, since they are not tied to the pre-stagnation SOTA methods.
* Build on each other's work and push toward optimizing the new SOTA score, now defined as the **top score under the '{tag}' tag.**"""


class StagnationProtocol:
    """Manages the Station's stagnation detection and protocol enforcement."""

    def __init__(self, station_instance):
        """Initialize the Stagnation Protocol system.

        Args:
            station_instance: Reference to the Station instance
        """
        self.station = station_instance

        # Initialize tracking data in station's config
        self._initialize_tracking_fields()

    def _initialize_tracking_fields(self):
        """Initialize stagnation tracking fields."""
        # Track the last known top score to detect breakthroughs (in memory only)
        self.last_top_score = None

        # Status history is now managed by Station itself

    def check_and_update_stagnation(self):
        """Main entry point called at each tick end to check and update stagnation status."""
        if not self._should_run():
            return

        current_tick = self.station.config.get('current_tick', 1)
        current_status = self.station.config.get('station_status', 'Healthy')

        # Detect if a breakthrough occurred
        breakthrough_tick = self._detect_last_breakthrough_tick(current_tick)

        # Calculate ticks since last breakthrough
        ticks_since_breakthrough = current_tick - breakthrough_tick

        # Log current state
        print(f"[StagnationProtocol] Status: {current_status}, Last breakthrough: {ticks_since_breakthrough} ticks ago")

        # Handle state transitions based on current status
        if current_status == "Healthy":
            self._handle_healthy_state(current_tick, ticks_since_breakthrough)
        elif current_status == "Stagnation":
            self._handle_stagnation_state(current_tick, breakthrough_tick, ticks_since_breakthrough)
        elif current_status.startswith("Deep Stagnation"):
            self._handle_deep_stagnation_state(current_tick, breakthrough_tick, ticks_since_breakthrough)

    def _should_run(self):
        """Check if stagnation protocol should run."""
        # Check if enabled in constants
        if not getattr(constants, 'STAGNATION_ENABLED', True):
            return False

        # Check if research counter is enabled (required)
        if not constants.RESEARCH_COUNTER_ENABLED:
            return False

        # Check if there's an evaluation manager with top_submission tracking
        if not hasattr(self.station, 'auto_research_evaluator'):
            return False

        eval_manager = getattr(self.station.auto_research_evaluator, 'eval_manager', None)
        if not eval_manager:
            return False

        return True

    def _detect_last_breakthrough_tick(self, current_tick: int) -> int:
        """Efficiently detect when the last breakthrough occurred.

        Returns:
            The tick number of the last breakthrough
        """
        # Get current top submission from evaluation manager
        eval_manager = self.station.auto_research_evaluator.eval_manager
        current_top = eval_manager.get_top_submission()

        # If no submissions yet, assume tick 1
        if not current_top:
            return 1

        # The current top's submitted_tick IS when the last breakthrough happened!
        breakthrough_tick = current_top.get('submitted_tick', 1)
        current_score = current_top.get('score')

        # Check if score changed during this session (new breakthrough just happened)
        if self.last_top_score is not None and current_score > self.last_top_score:
            print(f"[StagnationProtocol] Breakthrough detected! New top: {current_score} (eval {current_top['evaluation_id']})")
            self.last_top_score = current_score
            return current_tick  # Breakthrough happened NOW

        # Update tracking
        self.last_top_score = current_score

        # Return when the current top was submitted (that's when the breakthrough happened)
        return breakthrough_tick

    def _handle_healthy_state(self, current_tick: int, ticks_since_breakthrough: int):
        """Handle logic when station is in Healthy state."""
        threshold = getattr(constants, 'STAGNATION_THRESHOLD_TICKS', 120)

        if ticks_since_breakthrough >= threshold:
            print(f"[StagnationProtocol] Entering Stagnation state ({ticks_since_breakthrough} ticks without breakthrough)")

            # Send Protocol I message
            message = getattr(constants, 'STAGNATION_PROTOCOL_I_MESSAGE', None) or DEFAULT_PROTOCOL_I_MESSAGE
            count = self._send_system_message_to_all_recursive(message)
            print(f"[StagnationProtocol] Sent Protocol I to {count} recursive agents")

            # Update status
            self._update_station_status("Stagnation", current_tick)

    def _handle_stagnation_state(self, current_tick: int, breakthrough_tick: int, ticks_since_breakthrough: int):
        """Handle logic when station is in Stagnation state."""
        # Check if breakthrough occurred recently (revert to healthy)
        if ticks_since_breakthrough <= 5:
            print(f"[StagnationProtocol] Breakthrough detected! Reverting to Healthy state")

            # Send congratulations message
            message = DEFAULT_CONGRATULATIONS_MESSAGE
            count = self._send_system_message_to_all_recursive(message)
            print(f"[StagnationProtocol] Sent congratulations to {count} recursive agents")

            # Update status
            self._update_station_status("Healthy", current_tick)
            # Clean up any Protocol II messages from research tasks (in case we came from Deep Stagnation)
            self._remove_protocol_ii_from_research_task()
            return

        # Check if should enter deep stagnation
        stagnation_start = self._get_status_start_tick("Stagnation")
        ticks_in_stagnation = current_tick - stagnation_start
        deep_threshold = getattr(constants, 'DEEP_STAGNATION_THRESHOLD_TICKS', 120)

        if ticks_in_stagnation >= deep_threshold:
            self._enter_deep_stagnation(current_tick)

    def _handle_deep_stagnation_state(self, current_tick: int, breakthrough_tick: int, ticks_since_breakthrough: int):
        """Handle logic when station is in Deep Stagnation state."""
        # Check if breakthrough occurred recently (revert to healthy)
        if ticks_since_breakthrough <= 5:
            print(f"[StagnationProtocol] Breakthrough detected! Reverting to Healthy state")

            # Send congratulations message
            message = DEFAULT_CONGRATULATIONS_MESSAGE
            count = self._send_system_message_to_all_recursive(message)
            print(f"[StagnationProtocol] Sent congratulations to {count} recursive agents")

            # Update status
            self._update_station_status("Healthy", current_tick)
            # Remove Protocol II message from research tasks
            self._remove_protocol_ii_from_research_task()
            return

        # Check if should cycle to next Deep Stagnation tag
        current_status = self.station.config.get('station_status', '')
        deep_stagnation_start = self._get_status_start_tick(current_status)
        ticks_in_deep_stagnation = current_tick - deep_stagnation_start
        deep_threshold = getattr(constants, 'DEEP_STAGNATION_THRESHOLD_TICKS', 120)

        if ticks_in_deep_stagnation >= deep_threshold:
            print(f"[StagnationProtocol] Cycling to next Deep Stagnation tag ({ticks_in_deep_stagnation} ticks in current Deep Stagnation)")
            # Clear previous Protocol II from research tasks
            self._remove_protocol_ii_from_research_task()
            # Enter next Deep Stagnation with new tag
            self._enter_deep_stagnation(current_tick)

    def _enter_deep_stagnation(self, current_tick: int):
        """Enter Deep Stagnation state with Protocol II."""
        # Get next unused tag
        tag = self._get_next_tag()
        print(f"[StagnationProtocol] Entering Deep Stagnation - {tag} ({current_tick} ticks without breakthrough)")

        # Generate Protocol II message
        message = self._generate_protocol_ii_message(tag)

        # Send to all recursive agents
        count = self._send_system_message_to_all_recursive(message)
        print(f"[StagnationProtocol] Sent Protocol II to {count} recursive agents")

        # Append to research task
        self._append_to_research_task(message, current_tick)

        # Update status
        self._update_station_status(f"Deep Stagnation - {tag}", current_tick)

    def _generate_protocol_ii_message(self, tag: str) -> str:
        """Generate Protocol II message with the given tag."""
        # Check for override
        override = getattr(constants, 'STAGNATION_PROTOCOL_II_MESSAGE', None)
        if override:
            # Replace placeholder with actual tag
            return override.replace("{tag}", tag)

        # Use default message template with tag replacement
        return DEFAULT_PROTOCOL_II_MESSAGE_TEMPLATE.replace("{tag}", tag)

    def _get_next_tag(self) -> str:
        """Get the next unused Deep Stagnation tag."""
        # Get list of used tags from status history
        used_tags = []
        for entry in self.station.config.get('status_history', []):
            status = entry.get('status', '')
            if status.startswith('Deep Stagnation - '):
                tag = status.replace('Deep Stagnation - ', '')
                used_tags.append(tag)

        # Find next unused tag
        for tag in DEEP_STAGNATION_TAGS:
            if tag not in used_tags:
                return tag

        # All tags used, cycle back to first
        print("[StagnationProtocol] All tags used, cycling back to beginning")
        return DEEP_STAGNATION_TAGS[0]

    def _send_system_message_to_all_recursive(self, message: str) -> int:
        """Send system message to all recursive agents.

        Returns:
            Number of agents that received the message
        """
        count = 0

        # Get all active agents
        active_agents = self.station.agent_module.get_all_active_agent_names()

        for agent_name in active_agents:
            try:
                # Load agent data
                agent_data = agent_module.load_agent_data(agent_name)
                if not agent_data:
                    continue

                # Skip non-recursive agents
                if agent_data.get(constants.AGENT_STATUS_KEY) != constants.AGENT_STATUS_RECURSIVE:
                    continue

                # Skip ended sessions
                if agent_data.get(constants.AGENT_SESSION_ENDED_KEY, False):
                    continue

                # Add notification
                agent_module.add_pending_notification(agent_data, message)
                agent_module.save_agent_data(agent_name, agent_data)
                count += 1
                print(f"[StagnationProtocol] Sent message to {agent_name}")

            except Exception as e:
                print(f"[StagnationProtocol] Error sending message to {agent_name}: {e}")

        return count

    def _update_station_status(self, new_status: str, current_tick: int):
        """Update the station status using Station's API."""
        # Use Station's method to update status and track history
        self.station.update_station_status(new_status, current_tick)

        # Station will save config at tick end
        print(f"[StagnationProtocol] Requested status update to: {new_status}")

    def _get_status_start_tick(self, status_prefix: str) -> int:
        """Get the tick when a status started (most recent occurrence)."""
        history = self.station.config.get('status_history', [])

        # Search backwards for most recent occurrence
        for entry in reversed(history):
            if entry['status'].startswith(status_prefix):
                return entry['start_tick']

        # Not found, return current tick as fallback
        return self.station.config.get('current_tick', 1)

    def _append_to_research_task(self, message: str, current_tick: int):
        """Append Protocol II message to the research task specification."""
        research_tasks_path = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            'research',
            'research_tasks.yaml'
        )

        try:
            # Load current tasks
            tasks = file_io_utils.load_yaml(research_tasks_path)
            if not tasks:
                print("[StagnationProtocol] No research tasks found")
                return

            # Tasks is a list, find task with id=1
            task_id = 1
            task_found = False

            for task in tasks:
                if task.get('id') == task_id:
                    # Found the task, append to its content
                    content_key = 'content' if 'content' in task else 'spec'
                    header = f"\n\n---Start of Stagnation Protocol II (Announced on Tick {current_tick})---\n\n"
                    end_marker = "\n\n---End of Stagnation Protocol II---"
                    task[content_key] = task.get(content_key, '') + header + message + end_marker
                    task_found = True
                    break

            if not task_found:
                print(f"[StagnationProtocol] Task {task_id} not found")
                return

            # Save updated tasks
            file_io_utils.save_yaml(tasks, research_tasks_path)
            print(f"[StagnationProtocol] Appended Protocol II to research task {task_id}")

        except Exception as e:
            print(f"[StagnationProtocol] Error appending to research task: {e}")

    def _remove_protocol_ii_from_research_task(self):
        """Remove all Protocol II messages from research task specification."""
        research_tasks_path = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            'research',
            'research_tasks.yaml'
        )

        try:
            # Load current tasks
            tasks = file_io_utils.load_yaml(research_tasks_path)
            if not tasks:
                return

            # Tasks is a list, process each task
            for task in tasks:
                content_key = 'content' if 'content' in task else 'spec'
                if content_key not in task:
                    continue

                content = task[content_key]

                # Split content into lines for processing
                lines = content.split('\n')
                cleaned_lines = []
                skip_mode = False

                for i, line in enumerate(lines):
                    # Check if this is the start of a Protocol II section
                    if line.strip().startswith('---Start of Stagnation Protocol II'):
                        skip_mode = True
                        # Remove trailing empty lines before the start marker
                        while cleaned_lines and cleaned_lines[-1].strip() == '':
                            cleaned_lines.pop()
                        continue

                    # If we're in skip mode, check if we've reached the end marker
                    if skip_mode:
                        if line.strip() == '---End of Stagnation Protocol II---':
                            # Found the end marker, stop skipping (don't include this line)
                            skip_mode = False
                        # Continue skipping (whether we found end or not)
                        continue

                    # Keep the line if we're not skipping
                    if not skip_mode:
                        cleaned_lines.append(line)

                # Reconstruct the content
                task[content_key] = '\n'.join(cleaned_lines).rstrip()

            # Save updated tasks
            file_io_utils.save_yaml(tasks, research_tasks_path)
            print("[StagnationProtocol] Removed Protocol II messages from research tasks")

        except Exception as e:
            print(f"[StagnationProtocol] Error removing Protocol II from research tasks: {e}")