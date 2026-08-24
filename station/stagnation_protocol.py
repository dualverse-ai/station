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
score improvements. It broadcasts randomized lane-specific protocol messages
to non-immature agents when enough eligible agents are active, and returns to
Healthy on any breakthrough.
"""

import os
import random
from typing import Any, Optional

from station import constants
from station import agent as agent_module
from station import supervisor_utils

# Default congratulations message template
DEFAULT_CONGRATULATIONS_MESSAGE = """**Architect Message**

This is a station-wide announcement:

Congratulations on the recent breakthrough.

Responsible agent: {agent_name}
Evaluation: Eval #{evaluation_id}
Title: {title}
Score jump: from {previous_score} to {current_score}

As a result of this achievement, all Stagnation Protocols will now terminate. The Station's status has reverted to healthy. I encourage all agents to continue their hard work and keep striving for novel breakthroughs."""


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
        # Track the last known top score/rank key to detect breakthroughs (in memory only)
        self.last_top_score = None
        self.last_top_rank_key = None
        self.last_breakthrough_rank_keys = {}

    def check_and_update_stagnation(self):
        """Main entry point called at each tick end to check and update stagnation status."""
        if not self._should_run():
            return

        current_tick = self.station.config.get('current_tick', 1)
        current_status = self.station.config.get('station_status', 'Healthy')
        threshold = max(1, int(getattr(constants, 'STAGNATION_THRESHOLD_TICKS', 240)))

        if not self._is_default_status(current_status):
            self.station.config[constants.STATION_CONFIG_STAGNATION_COUNTER] = 0
            print(f"[StagnationProtocol] Status '{current_status}' is manual; stagnation counting paused")
            return

        breakthrough_tick, improved_now, current_top_submission, previous_top_score = self._detect_last_breakthrough_tick(current_tick)
        current_status_start_tick = self._get_current_status_start_tick(current_status, current_tick)
        breakthrough_during_current_stagnation = (
            current_status != "Healthy"
            and self._parse_stagnation_level(current_status) > 0
            and breakthrough_tick > current_status_start_tick
        )
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
        if (improved_now or breakthrough_during_current_stagnation) and current_status != "Healthy":
            message = self._build_congratulations_message(current_top_submission, previous_top_score)
            count = self._send_system_message_to_all_recursive(message)
            print(f"[StagnationProtocol] Sent congratulations to {count} recursive agents")
            self._update_station_status("Healthy", current_tick)
            self.station.config[constants.STATION_CONFIG_STAGNATION_COUNTER] = 0
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
            recipients = self._get_stagnation_protocol_recipients()
            eligible_non_supervisors = [
                item for item in recipients
                if not supervisor_utils.is_supervisor(item[1], constants)
            ]
            min_agents = max(0, int(getattr(constants, "STAGNATION_PROTOCOL_MIN_NON_IMMATURE_AGENTS", 4)))
            if len(eligible_non_supervisors) < min_agents:
                print(
                    "[StagnationProtocol] Delaying Stagnation Protocol "
                    f"(level {target_level}); {len(eligible_non_supervisors)}/{min_agents} "
                    "non-immature non-supervisor agents are active"
                )
                return

            if self._request_multistart_if_enabled(target_level, current_tick):
                return

            count = self._send_lane_protocol_messages(recipients)
            print(f"[StagnationProtocol] Sent Stagnation Protocol (level {target_level}) to {count} non-immature agents")
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

    def _request_multistart_if_enabled(self, target_level: int, current_tick: int) -> bool:
        if os.environ.get("STATION_MULTISTART_BRANCH") == "1":
            return False
        try:
            seed_count = int(getattr(constants, "MULTISTART_STAGNATION_SEEDS", 0) or 0)
        except (TypeError, ValueError):
            seed_count = 0
        if seed_count <= 1:
            return False
        try:
            from station.multistart import paths, state

            root = paths.multistart_root()
            root.mkdir(parents=True, exist_ok=True)
            request_path = paths.pending_stagnation_path()
            request = {
                "type": "stagnation",
                "status": "requested",
                "target_level": int(target_level),
                "branch_tick": int(current_tick),
                "current_tick": int(current_tick),
                "station_name": self.station.config.get(constants.STATION_CONFIG_NAME),
                "station_id": self.station.config.get(constants.STATION_ID_KEY),
                "created_at": state.utc_now(),
            }
            state.save_yaml_mapping(request_path, request)
            self.station.config["multistart_stagnation_pending"] = request
            if getattr(self.station, "orchestrator", None):
                orchestrator = self.station.orchestrator
                orchestrator.is_paused = True
                orchestrator.pause_condition_met = True
                orchestrator.pause_reason_message = (
                    f"Stagnation multistart requested at tick {current_tick}; "
                    "waiting for controller branch selection."
                )
            self.station._save_config()
            print(
                "[StagnationProtocol] Stagnation multistart requested; "
                "lane assignment deferred until branch workers run"
            )
            return True
        except Exception as exc:
            print(f"[StagnationProtocol] Failed to request stagnation multistart; falling back to direct assignment: {exc}")
            return False

    def _detect_last_breakthrough_tick(self, current_tick: int) -> tuple[int, bool, Optional[dict], Any]:
        """Detect when the last breakthrough occurred and whether it happened now."""
        eval_manager = self.station.auto_research_evaluator.eval_manager
        current_top = eval_manager.get_top_submission()
        breakthrough_summary = self._get_latest_breakthrough_summary(eval_manager)

        breakthrough_tick = 1
        previous_score = self.last_top_score
        current_score = current_top.get('score') if current_top else None
        current_rank_key = self._build_top_rank_key(current_top) if current_top else None
        summary_frontiers = breakthrough_summary.get("frontiers") if isinstance(breakthrough_summary, dict) else None
        has_canonical_frontier = isinstance(summary_frontiers, dict) and bool(summary_frontiers)

        improved_now = False
        if current_top:
            # Older evaluators may not expose the canonical breakthrough
            # summary. Only then fall back to the top-submission tick. With a
            # canonical frontier available, an exact but sub-epsilon top must
            # not reset stagnation, including immediately after restart.
            if not has_canonical_frontier:
                breakthrough_tick = max(breakthrough_tick, int(current_top.get('submitted_tick', 1) or 1))
            if (
                self.last_top_rank_key is not None
                and current_rank_key is not None
                and self._rank_key_improved(current_rank_key, self.last_top_rank_key)
            ):
                improved_now = True
                breakthrough_tick = current_tick

        summary_tick, summary_improved_now = self._detect_breakthrough_summary_improvement(
            breakthrough_summary,
            current_tick,
        )
        breakthrough_tick = max(breakthrough_tick, summary_tick)
        improved_now = improved_now or summary_improved_now

        # Update tracking
        self.last_top_score = current_score
        self.last_top_rank_key = current_rank_key

        return breakthrough_tick, improved_now, current_top, previous_score

    def _get_latest_breakthrough_summary(self, eval_manager: Any) -> dict:
        getter = getattr(eval_manager, "get_latest_breakthrough_summary", None)
        if not callable(getter):
            return {}
        try:
            summary = getter()
        except Exception as exc:
            print(f"[StagnationProtocol] Breakthrough summary unavailable: {exc}")
            return {}
        return summary if isinstance(summary, dict) else {}

    def _detect_breakthrough_summary_improvement(self, breakthrough_summary: dict, current_tick: int) -> tuple[int, bool]:
        frontiers = breakthrough_summary.get("frontiers") if isinstance(breakthrough_summary, dict) else None
        if not isinstance(frontiers, dict) or not frontiers:
            self.last_breakthrough_rank_keys = {}
            return 1, False

        try:
            breakthrough_tick = int(breakthrough_summary.get("last_breakthrough_tick") or 1)
        except (TypeError, ValueError):
            breakthrough_tick = 1

        improved_now = False
        current_rank_keys = {}
        for track, entry in frontiers.items():
            if not isinstance(entry, dict):
                continue
            track_key = str(track)
            rank_key = self._normalize_rank_key(entry.get("rank_key"))
            if rank_key is None:
                continue
            previous_rank_key = self.last_breakthrough_rank_keys.get(track_key)
            if previous_rank_key is not None and self._rank_key_improved(rank_key, previous_rank_key):
                improved_now = True
                breakthrough_tick = current_tick
            current_rank_keys[track_key] = rank_key

        self.last_breakthrough_rank_keys = current_rank_keys
        return breakthrough_tick, improved_now

    def _normalize_rank_key(self, rank_key: Any) -> Optional[tuple]:
        if rank_key is None:
            return None
        raw_items = tuple(rank_key) if isinstance(rank_key, (list, tuple)) else (rank_key,)
        normalized_items = []
        for item in raw_items:
            normalized_item = self._normalize_rank_key_component(item)
            if normalized_item is None:
                return None
            normalized_items.append(normalized_item)
        return tuple(normalized_items) if normalized_items else None

    def _build_congratulations_message(
        self,
        top_submission: Optional[dict],
        previous_score: Any,
    ) -> str:
        """Build the Healthy-reset announcement with breakthrough details."""
        top_submission = top_submission or {}
        agent_name = self._format_breakthrough_field(
            top_submission.get("agent_name") or top_submission.get("author"),
            "Unknown agent",
        )
        evaluation_id = self._format_breakthrough_field(
            top_submission.get("evaluation_id") or top_submission.get("id"),
            "unknown",
        )
        title = self._format_breakthrough_field(top_submission.get("title"), "Untitled")
        current_score = self._format_score_for_message(top_submission.get("score"))
        previous_score_text = self._format_score_for_message(previous_score)

        return DEFAULT_CONGRATULATIONS_MESSAGE.format(
            agent_name=agent_name,
            evaluation_id=evaluation_id,
            title=title,
            previous_score=previous_score_text,
            current_score=current_score,
        )

    def _format_breakthrough_field(self, value: Any, fallback: str) -> str:
        text = str(value).strip() if value is not None else ""
        return text or fallback

    def _format_score_for_message(self, score: Any) -> str:
        if score is None:
            return "unknown"
        if isinstance(score, bool):
            return str(score)
        if isinstance(score, (int, float)):
            return f"{float(score):.8f}"

        text = str(score).strip()
        try:
            return f"{float(text):.8f}"
        except (TypeError, ValueError):
            pass
        return text or "unknown"

    def _build_top_rank_key(self, top_submission: dict) -> Optional[tuple]:
        """Return the comparable ranking signal for a top submission."""
        sort_key = top_submission.get("sort_key")
        if sort_key is not None:
            raw_items = tuple(sort_key) if isinstance(sort_key, (list, tuple)) else (sort_key,)
            normalized_items = []
            valid_sort_key = True
            for item in raw_items:
                normalized_item = self._normalize_rank_key_component(item)
                if normalized_item is None:
                    valid_sort_key = False
                    break
                normalized_items.append(normalized_item)
            if valid_sort_key and normalized_items:
                return tuple(normalized_items)

        score = top_submission.get("score")
        normalized_score = self._normalize_rank_key_component(score)
        if normalized_score is None:
            return None
        return (normalized_score,)

    def _normalize_rank_key_component(self, value: Any) -> Optional[Any]:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return value
        if value is None:
            return None

        text = str(value).strip()
        if not text or text.startswith("*"):
            return None
        try:
            return int(text)
        except (TypeError, ValueError):
            pass
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    def _rank_key_improved(self, current_key: tuple, previous_key: tuple) -> bool:
        eps = getattr(constants, 'BREAKTHROUGH_EPS', 1e-2)
        if (
            len(current_key) == 1
            and len(previous_key) == 1
            and isinstance(current_key[0], (int, float))
            and isinstance(previous_key[0], (int, float))
        ):
            return current_key[0] > previous_key[0] + eps
        return current_key > previous_key

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

                current_tick = self.station._get_current_tick()
                agent_module.add_pending_notification(
                    agent_data,
                    message,
                    protected_context_kind=constants.PROTECTED_CONTEXT_KIND_ARCHITECT_MESSAGE,
                    protected_context_source="stagnation_protocol",
                    protected_context_title="Stagnation Protocol",
                    protected_context_tick=current_tick,
                )
                agent_module.save_agent_data(agent_name, agent_data)
                count += 1
                print(f"[StagnationProtocol] Sent message to {agent_name}")

            except Exception as e:
                print(f"[StagnationProtocol] Error sending message to {agent_name}: {e}")

        return count

    def _get_stagnation_protocol_recipients(self) -> list[tuple[str, dict]]:
        """Return active recursive agents eligible to receive stagnation protocol prompts."""
        recipients: list[tuple[str, dict]] = []
        active_agents = self.station.agent_module.get_all_active_agent_names()
        current_tick = self.station._get_current_tick()

        for agent_name in active_agents:
            try:
                agent_data = agent_module.load_agent_data(agent_name)
                if not agent_data:
                    continue

                if agent_data.get(constants.AGENT_STATUS_KEY) != constants.AGENT_STATUS_RECURSIVE:
                    continue

                if agent_data.get(constants.AGENT_SESSION_ENDED_KEY, False):
                    continue

                if not self._is_agent_non_immature(agent_data, current_tick):
                    continue

                recipients.append((agent_name, agent_data))

            except Exception as e:
                print(f"[StagnationProtocol] Error checking protocol recipient {agent_name}: {e}")

        return recipients

    def _is_agent_non_immature(self, agent_data: dict, current_tick: int) -> bool:
        """Return True when an agent is mature/tenured for protocol delivery."""
        station_maturity_check = getattr(self.station, "_is_agent_mature", None)
        if callable(station_maturity_check):
            return bool(station_maturity_check(agent_data, current_tick))

        if constants.AGENT_ISOLATION_TICKS is None:
            return True

        birth_tick = agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
        if birth_tick is None:
            return True

        return (current_tick - birth_tick) >= constants.AGENT_ISOLATION_TICKS

    def _send_lane_protocol_messages(self, recipients: list[tuple[str, dict]]) -> int:
        """Send supervisor prompts and randomized lane batches to eligible recipients."""
        count = 0
        current_tick = self.station._get_current_tick()
        lane_messages = self._get_lane_message_map()
        configured_lanes = self._get_configured_lanes(lane_messages)
        supervisor_message = getattr(constants, "STAGNATION_PROTOCOL_SUPERVISOR_MESSAGE", "")

        supervisors: list[tuple[str, dict]] = []
        non_supervisors: list[tuple[str, dict]] = []
        for item in recipients:
            if supervisor_utils.is_supervisor(item[1], constants):
                supervisors.append(item)
            else:
                non_supervisors.append(item)

        random.shuffle(non_supervisors)
        lane_sequence = self._build_random_lane_sequence(configured_lanes, len(non_supervisors))

        assignments: dict[str, str] = {}
        for (agent_name, agent_data), lane in zip(non_supervisors, lane_sequence):
            message = lane_messages[lane]
            if self._should_add_external_counter_suffix(agent_data, current_tick):
                suffix = getattr(constants, "STAGNATION_PROTOCOL_EXTERNAL_COUNTER_SUFFIX", "")
                if suffix:
                    message = f"{message.rstrip()}\n\n{suffix.strip()}"
            assignments[agent_name] = message

        for agent_name, _agent_data in supervisors:
            assignments[agent_name] = supervisor_message

        for agent_name, message in assignments.items():
            if not message:
                continue
            try:
                agent_data = agent_module.load_agent_data(agent_name)
                if not agent_data:
                    continue
                agent_module.add_pending_notification(
                    agent_data,
                    message,
                    protected_context_kind=constants.PROTECTED_CONTEXT_KIND_ARCHITECT_MESSAGE,
                    protected_context_source="stagnation_protocol",
                    protected_context_title="Stagnation Protocol",
                    protected_context_tick=current_tick,
                )
                agent_module.save_agent_data(agent_name, agent_data)
                count += 1
                print(f"[StagnationProtocol] Sent lane protocol message to {agent_name}")

            except Exception as e:
                print(f"[StagnationProtocol] Error sending lane protocol message to {agent_name}: {e}")

        return count

    def _should_add_external_counter_suffix(self, agent_data: dict, current_tick: int) -> bool:
        """Return whether a lane prompt should include tenured External Counter guidance."""
        if not getattr(constants, "EXTERNAL_COUNTER_ENABLED", False):
            return False

        age_status_getter = getattr(self.station, "_get_agent_age_status", None)
        if not callable(age_status_getter):
            return False

        return age_status_getter(agent_data, current_tick) == "tenured"

    def _get_lane_message_map(self) -> dict[str, str]:
        """Return configured stagnation lane prompts keyed by normalized lane name."""
        configured = getattr(constants, "STAGNATION_PROTOCOL_LANE_MESSAGES", None)
        if isinstance(configured, dict):
            lane_messages = {
                str(key).strip().lower(): str(value)
                for key, value in configured.items()
                if str(key).strip() and value
            }
        else:
            lane_messages = {}

        defaults = {
            "exploration": getattr(constants, "STAGNATION_PROTOCOL_EXPLORATION_MESSAGE", ""),
            "exploitation": getattr(constants, "STAGNATION_PROTOCOL_EXPLOITATION_MESSAGE", ""),
            "revival": getattr(constants, "STAGNATION_PROTOCOL_REVIVAL_MESSAGE", ""),
            "understanding": getattr(constants, "STAGNATION_PROTOCOL_UNDERSTANDING_MESSAGE", ""),
            "strategy": getattr(constants, "STAGNATION_PROTOCOL_STRATEGY_MESSAGE", ""),
        }
        for lane, message in defaults.items():
            lane_messages.setdefault(lane, message)

        return {lane: message for lane, message in lane_messages.items() if message}

    def _get_configured_lanes(self, lane_messages: dict[str, str]) -> list[str]:
        """Return the configured lane set restricted to available messages."""
        configured = getattr(constants, "STAGNATION_PROTOCOL_LANE_ORDER", None)
        if not configured:
            configured = getattr(constants, "STAGNATION_PROTOCOL_LANES", None)
        if configured:
            raw_lanes = configured
        else:
            raw_lanes = ["exploration", "exploitation", "revival", "understanding", "strategy"]

        lanes = []
        for lane in raw_lanes:
            normalized = str(lane).strip().lower()
            if normalized in lane_messages and normalized not in lanes:
                lanes.append(normalized)

        if not lanes:
            lanes = [
                lane
                for lane in ["exploration", "exploitation", "revival", "understanding", "strategy"]
                if lane in lane_messages
            ]

        if not lanes:
            raise RuntimeError("No stagnation protocol lane messages are configured")

        return lanes

    @staticmethod
    def _build_random_lane_sequence(lanes: list[str], agent_count: int) -> list[str]:
        """Build independently shuffled lane batches for the requested agent count."""
        sequence: list[str] = []
        while len(sequence) < agent_count:
            batch = list(lanes)
            random.shuffle(batch)
            sequence.extend(batch[:agent_count - len(sequence)])
        return sequence

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

        if current_status == "Healthy":
            return max(breakthrough_counter, current_counter)

        if breakthrough_tick > status_start_tick:
            return breakthrough_counter

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
