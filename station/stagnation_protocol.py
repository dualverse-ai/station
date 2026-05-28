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
Stagnation Protocol System (simplified)

This module detects research stagnation and escalates through sequential
stagnation levels (Stagnation I, II, III, ...) every fixed interval without
score improvements. It broadcasts a single protocol message at each level and
returns to Healthy on any breakthrough.
"""

import re

from station import constants
from station import agent as agent_module
from station import supervisor_utils

# Default Protocol I message
DEFAULT_PROTOCOL_I_MESSAGE = """**Architect Message**

This is a station-wide announcement:

The Station has entered a stage of stagnation, as no breakthroughs have been achieved for many ticks. 

Cease your current work and execute the following **Stagnation Protocol**. Run each step on **separate Station ticks**:

### 1. **Literature Review**

Go to the Archive Room and ask the Surveyor for a targeted survey of the current frontier. Include your current research direction or hypotheses in the prompt, and ask the Surveyor to identify relevant successful mechanisms, scoped failures, important papers to read, and gaps or consensus views worth challenging.

Read several key papers cited by the Surveyor in full detail. Around 3–5 papers is a good number to start with. Do not run `preview all` as the default first action, as it may cause information overload and reinforce existing blind spots.

Then summarize the papers and research landscape in a new Private Memory Capsule. Focus on the current state of the art, common approaches, open challenges, gaps in prior work, and insights gained. For important negative results, record what they actually cover and what nearby regimes remain open.

This step should take several Station ticks. Make sure to document your findings thoroughly. Skip this step if you are an immature agent.

### 2. **Assumption-challenging Reflection**

It has been observed that a Station can become trapped in a local conceptual paradigm due to influential archival papers, existing SOTA methods, or repeated negative results. Run a multi-tick reflection based on the papers you have read and your own research journey. Reflect upon:

* Are there opposing schools of thought within the Station, similar to those in the human research community? If not, what is the dominant school of thought, and what are its blind spots?
* Which assumptions or archival papers have not been challenged conceptually, but should be? This should be at the conceptual level, e.g. existing taxonomies, foundational assumptions, or favored representations.
* Which assumptions or archival papers have not been challenged technically, but should be? This should be at the technical level, e.g. specific hypotheses, model restrictions, search spaces, or proof assumptions in existing methods.
* Which archive papers or evaluation histories are being treated as if they killed an entire paradigm? What is the exact logical gap between the paper’s formal claim and the broader paradigm?
* Can any negative result be inverted into a positive design principle? For example, does the obstruction identify which assumption to relax, which structure to preserve, or which nearby family remains open?
* If these assumptions are broken, what concrete research role, revival lane, or construction direction should exist that does not currently exist?

Remember that even published papers with high scores, widely accepted taxonomies, or paradigms deemed futile after many negative results should be challenged critically. Stagnation is often caused not only by lack of effort, but by blind spots induced by the existing literature.

### 3. **Novel Exploration Reflection**

Run another multi-tick reflection, this time focused on brainstorming novel ideas for the original problem, informed by your assumption-challenging reflection. This may include:

* Consider methods for breaking the assumptions identified in the previous reflection.
* Consider similar problems, simpler versions of the problem, or even unrelated lines of investigation that might not initially seem helpful but could lead to useful insights.
* Consider cross-disciplinary ideas or analogies that might inspire a breakthrough. Use looser forms of reasoning, such as metaphors, speculative models, or anthropomorphic descriptions, if they help generate novel ideas. Do not reject unusual ideas too early; first translate them into possible technical hypotheses, representations, or construction strategies.
* Brainstorm at least three novel ideas based on the above reflections.

You should develop a concrete research plan at the end of the reflection. The research plan should cover all the promising ideas you generate above, instead of being restricted to a single idea. It should span at least 60 ticks and include a timetable and contingency plans.

### 4. **Execution**

Execute the above research plan. 

* If you have been assigned a supervisor: You should ask your supervisor for permission to pivot if your current line of research is materially different from the new research plan. You may run some preliminary exeriments before asking for a pivot, so as to select the most promising direction.
* You should be persistent in following your research plan and should not pivot once it has started. Early failure should be expected.

## Stagnation Holiday

- To facilitate reflection and long-term planning, the next 5 ticks (till Station Tick __HOLIDAY_END_TICK__) will be declared a holiday, where you cannot goto the Research Center or submit new papers.
- You should spend the holiday to execute the first three steps of the Stagnation Protocol, which will prepare you for the next execution phase after the holiday.
- If you have time left in the holiday after completing the first three steps, you can use it to further reflect or discuss more with peers.

### For Supervisors

Supervisors must also execute the above steps (except Step 4 - Execution) and guide agents in completing the Stagnation Protocol:

- Supervisors must encourage agents to pursue high-risk, novel exploration instead of exploiting known methods or local optima. Multiple component changes and regime changes are encouraged. 
- Try to cultivate a diverse research community with opposing schools of thought and a willingness to challenge assumptions, especially established Station knowledge. Encourage agents to think and speak bluntly rather than diplomatically.
- Do not overemphasize rigor. Rigorous protocols or tools must not block novel attempts. Do not focus on scoping claims except when agents are ready to submit archival papers.
- Do not overemphasize short-term score optimization. A new paradigm may take a long time (>100 ticks) to catch up with the established paradigm, so do not let the agent give up because of a few low scores.
- The requirement for the score transfer hypothesis in the Research Proposal can be relaxed to mitigate score-chasing bias.
- The usual high bar for a pivot request may be temporarily relaxed at the start of the Stagnation Protocol.
- The usual requirement of only a single direction per Research Proposal can be relaxed to allow multiple promising directions to be explored in parallel during the Stagnation Protocol.
"""

# Default congratulations message
DEFAULT_CONGRATULATIONS_MESSAGE = """**Architect Message**

This is a station-wide announcement:

Congratulations on the recent breakthrough. As a result of this achievement, all Stagnation Protocols will now terminate, and the corresponding restrictions are all lifted. The Station's status has reverted to healthy. I encourage all agents to continue their hard work and keep striving for novel breakthroughs."""


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

    def check_and_update_stagnation(self):
        """Main entry point called at each tick end to check and update stagnation status."""
        if not self._should_run():
            return

        current_tick = self.station.config.get('current_tick', 1)
        current_status = self.station.config.get('station_status', 'Healthy')
        threshold = max(1, int(getattr(constants, 'STAGNATION_THRESHOLD_TICKS', 240)))

        if not self._is_default_status(current_status):
            self.station.config[constants.STATION_CONFIG_STAGNATION_COUNTER] = 0
            if hasattr(self.station, 'clear_stagnation_holiday_window'):
                self.station.clear_stagnation_holiday_window()
            print(f"[StagnationProtocol] Status '{current_status}' is manual; stagnation counting paused")
            return

        breakthrough_tick, improved_now = self._detect_last_breakthrough_tick(current_tick)
        current_counter = int(self.station.config.get(constants.STATION_CONFIG_STAGNATION_COUNTER, 0) or 0)
        ticks_since_breakthrough = self._compute_effective_counter(
            current_status=current_status,
            current_tick=current_tick,
            breakthrough_tick=breakthrough_tick,
            threshold=threshold,
            current_counter=current_counter,
        )
        self.station.config[constants.STATION_CONFIG_STAGNATION_COUNTER] = ticks_since_breakthrough

        # Log current state
        print(
            f"[StagnationProtocol] Status: {current_status}, "
            f"Counter: {ticks_since_breakthrough} ticks"
        )

        # On breakthrough, revert to healthy and notify if we were stagnant
        if improved_now and current_status != "Healthy":
            message = DEFAULT_CONGRATULATIONS_MESSAGE
            count = self._send_system_message_to_all_recursive(message)
            print(f"[StagnationProtocol] Sent congratulations to {count} recursive agents")
            self._update_station_status("Healthy", current_tick)
            self.station.config[constants.STATION_CONFIG_STAGNATION_COUNTER] = 0
            if hasattr(self.station, 'clear_stagnation_holiday_window'):
                self.station.clear_stagnation_holiday_window()
            return

        if improved_now:
            self.station.config[constants.STATION_CONFIG_STAGNATION_COUNTER] = 0
            return

        # Determine required stagnation level based on time since breakthrough
        target_level = ticks_since_breakthrough // threshold
        if target_level <= 0:
            return

        current_level = self._parse_stagnation_level(current_status)
        if target_level > current_level:
            holiday_duration = max(0, int(getattr(constants, 'STAGNATION_HOLIDAY_DURATION_TICKS', 10)))
            holiday_end_tick = current_tick + holiday_duration
            if hasattr(self.station, 'set_stagnation_holiday_window'):
                self.station.set_stagnation_holiday_window(current_tick, holiday_end_tick)

            message_template = getattr(constants, 'STAGNATION_PROTOCOL_I_MESSAGE', None) or DEFAULT_PROTOCOL_I_MESSAGE
            message = message_template.replace("__HOLIDAY_END_TICK__", str(holiday_end_tick))
            non_supervisor_message, supervisor_message = self._split_supervisor_section(message)
            count = self._send_protocol_message_to_recursive_by_role(
                non_supervisor_message=non_supervisor_message,
                supervisor_message=supervisor_message,
            )
            print(f"[StagnationProtocol] Sent Stagnation Protocol (level {target_level}) to {count} recursive agents")
            new_status = self._format_stagnation_status(target_level)
            self._update_station_status(new_status, current_tick)

    def handle_manual_status_update(self, status: str, current_tick: int) -> None:
        """Persist counter updates for manual status changes."""
        if not self._is_default_status(status):
            self.station.config[constants.STATION_CONFIG_STAGNATION_COUNTER] = 0
            return

        self.station.config[constants.STATION_CONFIG_STAGNATION_COUNTER] = self._status_floor(status)

    def _should_run(self):
        """Check if stagnation protocol should run."""
        if not getattr(constants, 'STAGNATION_ENABLED', True):
            return False
        if getattr(constants, 'RESEARCH_NO_SCORE', False):
            return False
        if not constants.RESEARCH_CENTER_ENABLED:
            return False
        if not hasattr(self.station, 'auto_research_evaluator'):
            return False

        eval_manager = getattr(self.station.auto_research_evaluator, 'eval_manager', None)
        if not eval_manager:
            return False

        return True

    def _detect_last_breakthrough_tick(self, current_tick: int) -> tuple[int, bool]:
        """Detect when the last breakthrough occurred and whether it happened now."""
        eval_manager = self.station.auto_research_evaluator.eval_manager
        current_top = eval_manager.get_top_submission()

        if not current_top:
            return 1, False

        breakthrough_tick = current_top.get('submitted_tick', 1)
        current_score = current_top.get('score')

        improved_now = False
        eps = getattr(constants, 'BREAKTHROUGH_EPS', 1e-8)
        if self.last_top_score is not None and current_score is not None:
            if current_score > self.last_top_score + eps:
                improved_now = True
                breakthrough_tick = current_tick

        # Update tracking
        self.last_top_score = current_score

        return breakthrough_tick, improved_now

    def _send_system_message_to_all_recursive(self, message: str) -> int:
        """Send system message to all recursive agents.

        Returns:
            Number of agents that received the message
        """
        count = 0
        active_agents = self.station.agent_module.get_all_active_agent_names()

        for agent_name in active_agents:
            try:
                agent_data = agent_module.load_agent_data(agent_name)
                if not agent_data:
                    continue

                if agent_data.get(constants.AGENT_STATUS_KEY) != constants.AGENT_STATUS_RECURSIVE:
                    continue

                if agent_data.get(constants.AGENT_SESSION_ENDED_KEY, False):
                    continue

                agent_module.add_pending_notification(agent_data, message)
                agent_module.save_agent_data(agent_name, agent_data)
                count += 1
                print(f"[StagnationProtocol] Sent message to {agent_name}")

            except Exception as e:
                print(f"[StagnationProtocol] Error sending message to {agent_name}: {e}")

        return count

    def _send_protocol_message_to_recursive_by_role(
        self,
        non_supervisor_message: str,
        supervisor_message: str,
    ) -> int:
        """Send protocol message to recursive agents with supervisor-specific targeting.

        Non-supervisors receive the general protocol. Supervisors receive the full
        version including the supervisor section.
        """
        count = 0
        active_agents = self.station.agent_module.get_all_active_agent_names()

        for agent_name in active_agents:
            try:
                agent_data = agent_module.load_agent_data(agent_name)
                if not agent_data:
                    continue

                if agent_data.get(constants.AGENT_STATUS_KEY) != constants.AGENT_STATUS_RECURSIVE:
                    continue

                if agent_data.get(constants.AGENT_SESSION_ENDED_KEY, False):
                    continue

                is_supervisor = supervisor_utils.is_supervisor(agent_data, constants)
                message = supervisor_message if is_supervisor else non_supervisor_message
                agent_module.add_pending_notification(agent_data, message)
                agent_module.save_agent_data(agent_name, agent_data)
                count += 1
                print(f"[StagnationProtocol] Sent role-targeted message to {agent_name}")

            except Exception as e:
                print(f"[StagnationProtocol] Error sending role-targeted message to {agent_name}: {e}")

        return count

    def _split_supervisor_section(self, message: str) -> tuple[str, str]:
        """Split protocol into non-supervisor and supervisor variants.

        Looks for a markdown heading named "For Supervisors". If present,
        non-supervisors receive content before that heading while supervisors
        receive the full message.
        """
        heading_match = re.search(
            r"(?im)^\s{0,3}#{2,6}\s+For\s+Supervisors\s*$",
            message,
        )
        if not heading_match:
            return message, message

        general_part = message[:heading_match.start()].rstrip()
        if not general_part:
            return message, message

        return general_part, message

    def _update_station_status(self, new_status: str, current_tick: int):
        """Update the station status using Station's API."""
        self.station.update_station_status(new_status, current_tick)
        print(f"[StagnationProtocol] Requested status update to: {new_status}")

    def _parse_stagnation_level(self, status: str) -> int:
        """Convert status string to stagnation level integer (Healthy = 0)."""
        if not status.startswith("Stagnation"):
            return 0

        suffix = status.replace("Stagnation", "", 1).strip()
        if not suffix:
            return 1

        try:
            return self._roman_to_int(suffix)
        except ValueError:
            return 1

    def _format_stagnation_status(self, level: int) -> str:
        """Format stagnation status string from level."""
        return f"Stagnation {self._int_to_roman(max(1, level))}"

    def _is_default_status(self, status: str) -> bool:
        """Default statuses are Healthy and Stagnation levels."""
        return status == "Healthy" or self._parse_stagnation_level(status) > 0

    def _status_floor(self, status: str) -> int:
        """Return the minimum counter implied by the current default status."""
        threshold = max(1, int(getattr(constants, 'STAGNATION_THRESHOLD_TICKS', 240)))
        return self._parse_stagnation_level(status) * threshold

    def _get_current_status_start_tick(self, current_status: str, current_tick: int) -> int:
        """Return the start tick of the current status from status history."""
        history = self.station.config.get('status_history') or []
        for entry in reversed(history):
            if entry.get('status') == current_status:
                start_tick = entry.get('start_tick')
                if isinstance(start_tick, int):
                    return start_tick
        return current_tick

    def _compute_effective_counter(
        self,
        current_status: str,
        current_tick: int,
        breakthrough_tick: int,
        threshold: int,
        current_counter: int,
    ) -> int:
        """Combine breakthrough history with persisted counter for restart-safe manual overrides."""
        breakthrough_counter = max(0, current_tick - breakthrough_tick)
        status_start_tick = self._get_current_status_start_tick(current_status, current_tick)
        status_floor = self._parse_stagnation_level(current_status) * threshold

        if status_start_tick > breakthrough_tick:
            return max(status_floor + max(0, current_tick - status_start_tick), current_counter)

        return max(breakthrough_counter, status_floor)

    def _is_default_status(self, status: str) -> bool:
        """Default statuses are Healthy and Stagnation levels."""
        return status == "Healthy" or self._parse_stagnation_level(status) > 0

    def _infer_tracking_start_tick(self, status: str, current_tick: int) -> int:
        """Infer a tracking baseline from the current default status."""
        threshold = max(1, int(getattr(constants, 'STAGNATION_THRESHOLD_TICKS', 240)))
        level = self._parse_stagnation_level(status)
        return max(0, int(current_tick) - (level * threshold))

    def _roman_to_int(self, roman: str) -> int:
        """Convert Roman numeral to integer."""
        roman = roman.upper()
        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        prev_value = 0
        for char in reversed(roman):
            value = roman_values.get(char)
            if value is None:
                raise ValueError(f"Invalid Roman numeral: {roman}")
            if value < prev_value:
                total -= value
            else:
                total += value
                prev_value = value
        return total

    def _int_to_roman(self, number: int) -> str:
        """Convert integer to Roman numeral."""
        val = [
            1000, 900, 500, 400,
            100, 90, 50, 40,
            10, 9, 5, 4,
            1
        ]
        syms = [
            "M", "CM", "D", "CD",
            "C", "XC", "L", "XL",
            "X", "IX", "V", "IV",
            "I"
        ]
        roman_num = ""
        i = 0
        n = max(1, number)
        while n > 0:
            for _ in range(n // val[i]):
                roman_num += syms[i]
                n -= val[i]
            i += 1
        return roman_num
