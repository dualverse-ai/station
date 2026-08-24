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

import time
import threading
import os
import traceback
import tempfile
import shutil
import copy
from contextlib import contextmanager
from queue import Queue
from typing import Dict, List, Optional, Any, Tuple

from station.station import Station
from station import constants
from station.constants import _load_config_overrides
from station import file_io_utils
from station import backup_utils
from station import context_compaction
from station import runtime_api_config
from station.system_messages import build_station_level_system_prompt
from station.llm_connectors import (
    BaseLLMConnector, create_llm_connector,
    LLMTransientAPIError, LLMPermanentAPIError, LLMSafetyBlockError, LLMConnectorError,
    LLMContextOverflowError, LLMCorruptedThoughtSignatureError
)
from station.llm_connectors.presets import build_model_preset_lookup
from station.base_room import LoggingInternalActionHandlerWrapper


DEPLOYMENT_LOG_MAX_BYTES = 256 * 1024 * 1024
DEPLOYMENT_LOG_CHECK_INTERVAL_TICKS = 100
DEPLOYMENT_LOG_FILENAMES = (
    "error.log",
    "nginx_error.log",
    "access.log",
    "nginx_access.log",
)


def truncate_oversized_deployment_logs(
    deployment_dir: str = "deployment",
    max_bytes: int = DEPLOYMENT_LOG_MAX_BYTES,
) -> List[str]:
    """Truncate oversized deployment logs in place so active writers stay valid."""
    truncated = []
    for filename in DEPLOYMENT_LOG_FILENAMES:
        path = os.path.join(deployment_dir, filename)
        try:
            if os.path.isfile(path) and os.path.getsize(path) > max_bytes:
                os.truncate(path, 0)
                truncated.append(path)
        except OSError as exc:
            print(f"Orchestrator: Could not inspect or truncate deployment log {path}: {exc}")
    return truncated


class Orchestrator:
    def __init__(self,
                 station_instance: Station,
                 auto_prepare_on_init: bool = True, # MODIFIED: Renamed and default to True
        log_event_queue: Optional[Queue] = None):
        # Load configuration overrides with verbose output at station initialization
        _load_config_overrides(verbose=True)
        runtime_api_config.validate_provider_backup_env_config()

        self.station = station_instance
        self.station.orchestrator = self
        self._last_deployment_log_check_tick: Optional[int] = None
        self._check_deployment_log_sizes(self.station._get_current_tick(), force=True)
        self.is_running: bool = False # True when the main_loop thread is active and processing
        self.is_prepared: bool = False # True when agent turn order loaded and connector configs validated
        self.orchestrator_thread: Optional[threading.Thread] = None
        self.agent_turn_order: List[str] = []
        self.current_tick_processed_agents: set[str] = set()
        self.agent_llm_connectors: Dict[str, BaseLLMConnector] = {}
        self._api_runtime_connector_lock = threading.RLock()

        # Durable temporal chat forks. Visible transcripts live under
        # station_data/temporal_chat; connector-only frozen history is hidden
        # below station_data/temporal_chat/.internal.
        self._temporal_chat_lock = threading.Lock()

        self.is_paused: bool = False
        self.pause_requested: bool = False
        self.pause_condition_met: bool = False
        self.pause_reason_message: str = ""
        self.pause_event = threading.Event()

        # Waiting state - automatically resumes when conditions resolve
        self.is_waiting: bool = False
        self.waiting_reasons: Dict[str, str] = {}
        self.wait_check_interval: float = 2.0  # seconds between auto-resume checks
        self._check_wait_conditions_before_next_tick: bool = False

        self.current_agent_index_in_turn_order: int = 0

        self.log_event_queue = log_event_queue
        self._parallel_action_commit_lock = threading.RLock()
        self.parallel_tick_runner = None
        self.research_submission_service = None
        self.archive_survey_submission_service = None

        self._cleanup_parallel_sync_on_startup()
        if constants.RESEARCH_CENTER_ENABLED and getattr(constants, "PARALLEL_RESEARCH_FAST_LANE_ENABLED", True):
            from station.eval_research.submission_service import ResearchSubmissionService

            self.research_submission_service = ResearchSubmissionService(
                self.station,
                log_event_func=self._push_log_event,
            )
            self.research_submission_service.start()
        if getattr(constants, "ARCHIVE_SURVEY_ENABLED", False) and getattr(constants, "PARALLEL_ARCHIVE_SURVEY_FAST_LANE_ENABLED", True):
            from station.eval_archive.surveyor import ArchiveSurveySubmissionService

            self.archive_survey_submission_service = ArchiveSurveySubmissionService(
                self.station,
                log_event_func=self._push_log_event,
            )
            self.archive_survey_submission_service.start()

        if auto_prepare_on_init:
            self.prepare_for_run()

        self._push_log_event("orchestrator_status", {"status": "initialized", "prepared": self.is_prepared, "message": "Orchestrator instance created."})

        # Restart stuck research evaluations before starting auto evaluator
        # This ensures any evaluations with unsent notifications from previous runs are requeued
        if constants.AUTO_EVAL_RESEARCH and constants.RESEARCH_CENTER_ENABLED:
            try:
                from station.eval_research import restart_stuck_evaluations
                restart_started_at = time.perf_counter()
                count = restart_stuck_evaluations(eval_manager=self.station.research_eval_manager)
                print(
                    "Orchestrator: restart_stuck_evaluations "
                    f"took {time.perf_counter() - restart_started_at:.3f}s"
                )
                if count > 0:
                    print(f"Orchestrator: Restarted {count} stuck research evaluation(s)")
                    self._push_log_event("orchestrator_info", {"message": f"Restarted {count} stuck research evaluation(s)"})
            except Exception as e:
                print(f"Orchestrator: Error restarting stuck evaluations: {e}")

        # Start auto research evaluator if enabled and Research Center room is enabled
        if constants.AUTO_EVAL_RESEARCH and constants.RESEARCH_CENTER_ENABLED:
            self.station.start_auto_research_evaluator(log_queue=self.log_event_queue)

        # Start auto external reporter if enabled and External Counter room is enabled
        if getattr(constants, "AUTO_EVAL_EXTERNAL_REPORT", False) and getattr(constants, "EXTERNAL_COUNTER_ENABLED", False):
            self.station.start_auto_external_reporter(log_queue=self.log_event_queue)

        # Initialize stagnation protocol if enabled
        if constants.STAGNATION_ENABLED and constants.RESEARCH_CENTER_ENABLED:
            self.station.init_stagnation_protocol()

        # Start auto archive evaluator if enabled
        if getattr(constants, 'EVAL_ARCHIVE_MODE', 'none') == 'auto':
            self.station.start_auto_archive_evaluator(log_queue=self.log_event_queue)

        # Start archive surveyor if enabled
        if getattr(constants, "ARCHIVE_SURVEY_ENABLED", False):
            self.station.start_auto_archive_surveyor(log_queue=self.log_event_queue)

        self._try_init_agents_and_launch()

        # AUTO_START: Auto-start orchestrator if enabled and there are active agents
        if constants.AUTO_START and auto_prepare_on_init and self.is_prepared and self.agent_turn_order:
            # Start in a separate thread to avoid blocking frontend initialization
            auto_start_thread = threading.Thread(target=self._auto_start_with_wait, daemon=True)
            auto_start_thread.start()

    def _push_log_event(self, event_type: str, data: Dict[str, Any]):
        if self.log_event_queue:
            log_message = {"event": event_type, "data": data, "timestamp": time.time()}
            try:
                self.log_event_queue.put_nowait(log_message)
            except Exception as e:
                print(f"Orchestrator: Error putting log event on queue: {e}")

    def _check_deployment_log_sizes(self, current_tick: int, force: bool = False) -> None:
        if not force:
            if current_tick % DEPLOYMENT_LOG_CHECK_INTERVAL_TICKS != 0:
                return
            if self._last_deployment_log_check_tick == current_tick:
                return

        self._last_deployment_log_check_tick = current_tick
        truncated = truncate_oversized_deployment_logs()
        if truncated:
            print(
                "Orchestrator: Truncated oversized deployment logs: "
                + ", ".join(truncated)
            )

    def _cleanup_parallel_sync_on_startup(self) -> None:
        try:
            from station.sync.parallel_state import ParallelTickState

            cleanup = ParallelTickState().cleanup_stale_run(
                station=self.station,
                eval_manager=getattr(self.station, "research_eval_manager", None),
            )
            if cleanup.get("had_stale_state"):
                message = (
                    "Recovered incomplete parallel tick before orchestrator preparation; "
                    f"rolled back provisional research evaluations: {cleanup.get('rolled_back_eval_ids', [])}; "
                    f"rolled back provisional archive surveys: {cleanup.get('rolled_back_survey_ids', [])}"
                )
                print(f"Orchestrator: {message}")
                self._push_log_event("parallel_tick_recovery", {"message": message})
        except Exception as exc:
            print(f"Orchestrator: parallel sync startup cleanup failed: {exc}")
            self._push_log_event(
                "orchestrator_error",
                {"message": f"parallel sync startup cleanup failed: {exc}", "trace": traceback.format_exc()},
            )

    def _auto_start_with_wait(self):
        """
        AUTO_START helper: Waits for pending background jobs to complete, then starts processing.
        Runs in a separate thread to avoid blocking frontend initialization.
        """
        print("Orchestrator: AUTO_START waiting for pending background jobs...")

        # Wait only when Research work has reached the configured tick-boundary limit.
        if constants.AUTO_EVAL_RESEARCH and constants.RESEARCH_CENTER_ENABLED:
            while True:
                if not self.station.should_wait_for_research_evaluations_at_tick_boundary():
                    break
                time.sleep(2)

        if getattr(constants, "AUTO_EVAL_EXTERNAL_REPORT", False) and getattr(constants, "EXTERNAL_COUNTER_ENABLED", False):
            while True:
                if self.station.has_failed_requeued_external_reports():
                    message = (
                        "AUTO_START stopped waiting because an External report already exhausted "
                        "its retry budget; the station remains paused for operator action."
                    )
                    print(f"Orchestrator: {message}")
                    self._push_log_event("orchestrator_info", {"message": message})
                    return
                if not self.station.has_pending_external_reports():
                    break
                time.sleep(2)

        if getattr(constants, "ARCHIVE_SURVEY_ENABLED", False):
            while True:
                if not self.station.has_pending_archive_surveys():
                    break
                time.sleep(2)

        # Check if processing loop has already been started (e.g., by user via frontend)
        if self.is_running:
            print("Orchestrator: AUTO_START skipped - processing loop already running")
            self._push_log_event("orchestrator_info", {"message": "AUTO_START skipped - processing loop already running"})
            return

        # Start processing loop
        print("Orchestrator: AUTO_START no pending evaluations, starting processing loop...")
        self._push_log_event("orchestrator_info", {"message": "AUTO_START starting processing loop"})
        self.start_processing_loop()

    def _init_agent_spawn_is_allowed(self, *, require_new_station: bool) -> bool:
        current_tick = self.station.config.get(constants.STATION_CONFIG_CURRENT_TICK, 0)
        if current_tick > 1:
            return False
        if self.agent_turn_order or self.station.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER):
            return False
        agents_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME)
        if os.path.isdir(agents_dir):
            active_agents = self.station.agent_module.get_all_active_agent_names()
            if active_agents:
                return False
        if require_new_station and not getattr(self.station, "is_new_station", False):
            init_agents_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.INIT_AGENTS_FILENAME)
            if not file_io_utils.file_exists(init_agents_path):
                return False
        return True

    def _spawn_init_agents_from_file(self) -> int:
        init_agents_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.INIT_AGENTS_FILENAME)
        if not file_io_utils.file_exists(init_agents_path):
            return 0

        init_agents = file_io_utils.load_yaml(init_agents_path)
        if not isinstance(init_agents, list) or not init_agents:
            print(f"Warning: init_agents.yaml at {init_agents_path} is empty or not a list.")
            return 0

        preset_lookup = build_model_preset_lookup()
        if not preset_lookup:
            print("Warning: No model presets available; init agents will be skipped.")
            return 0

        spawned = 0
        for display_name in init_agents:
            if not isinstance(display_name, str) or not display_name.strip():
                print(f"Warning: Skipping invalid init agent entry: {display_name}")
                continue

            preset = preset_lookup.get(display_name)
            if not preset:
                print(f"Warning: No model preset found for '{display_name}'.")
                continue

            model_provider_class = preset.get("model_provider_class")
            model_name = preset.get("model_name")
            if not model_provider_class or not model_name:
                print(f"Warning: Preset '{display_name}' missing provider or model name.")
                continue

            initial_tokens_max = preset.get("initial_tokens_max")
            if isinstance(initial_tokens_max, (int, float)):
                initial_tokens_max = int(initial_tokens_max)
            else:
                initial_tokens_max = None

            role_definition = preset.get("role_definition")
            if role_definition is None:
                role_definition = preset.get("llm_system_prompt")
            # Init-agent presets use blank role fields as "no explicit role",
            # allowing create_guest_agent() to sample the fresh-guest role pool.
            role_definition = role_definition or None

            success, msg = self.dynamic_add_agent_to_station(
                agent_type=constants.AGENT_STATUS_GUEST,
                model_provider_class=model_provider_class,
                model_name=model_name,
                initial_tokens_max=initial_tokens_max,
                role_definition=role_definition,
            )
            if success:
                spawned += 1
            else:
                print(f"Warning: Failed to spawn init agent '{display_name}': {msg}")

        if spawned and not self.is_prepared:
            self.prepare_for_run()

        return spawned

    def _try_init_agents_and_launch(self) -> None:
        if os.environ.get("STATION_MULTISTART_BRANCH") == "1":
            return
        if not self._init_agent_spawn_is_allowed(require_new_station=True):
            return

        spawned = self._spawn_init_agents_from_file()
        if (
            spawned
            and self.is_prepared
            and self.agent_turn_order
            and not self.is_running
        ):
            print("Orchestrator: Launching station after init agents spawn.")
            self.start_processing_loop()

    def try_init_agents_for_multistart_branch(self) -> int:
        if os.environ.get("STATION_MULTISTART_BRANCH") != "1":
            return 0
        if not self._init_agent_spawn_is_allowed(require_new_station=False):
            return 0
        spawned = self._spawn_init_agents_from_file()
        if spawned and self.is_prepared:
            self._load_agent_turn_order()
        return spawned

    @contextmanager
    def _agent_response_context(self, agent_name):
        """Context manager to handle agent waiting_station_response flag."""
        # Set flag before LLM call
        agent_data = self.station.agent_module.load_agent_data(agent_name)
        if agent_data:
            agent_data[constants.AGENT_WAITING_STATION_RESPONSE_KEY] = True
            self.station.agent_module.save_agent_data(agent_name, agent_data)
        
        try:
            yield
        finally:
            # Always clear flag, regardless of success or failure
            agent_data = self.station.agent_module.load_agent_data(agent_name)
            if agent_data:
                agent_data[constants.AGENT_WAITING_STATION_RESPONSE_KEY] = False
                self.station.agent_module.save_agent_data(agent_name, agent_data)

    def _load_agent_turn_order(self) -> bool:
        """
        Loads agent turn order from station config, verifies agent data files,
        and manages LLM connectors for agents entering/leaving the active roster.
        Updates self.agent_turn_order.
        Returns True if a non-empty, verifiable turn order is loaded, False otherwise.
        """
        previous_runtime_order_set = set(self.agent_turn_order)
        
        config_turn_order = list(self.station.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, []))
        
        verified_runtime_order: List[str] = []
        pruned_from_config: List[str] = []
        if config_turn_order:
            for agent_name_in_config in config_turn_order:
                # Check if agent file actually exists AND is loadable as an active agent
                # load_agent_data (default) returns None for ascended/ended agents.
                # If an agent ascended, their old guest identity won't load here.
                if self.station.agent_module.load_agent_data(agent_name_in_config):
                    verified_runtime_order.append(agent_name_in_config)
                else:
                    pruned_from_config.append(agent_name_in_config)
                    # Agent in config is not active (missing file, or ended/ascended)
                    print(f"Orchestrator: Agent '{agent_name_in_config}' from config is not active or file missing. Pruning from runtime order.")
                    self._push_log_event("orchestrator_warning", {"message": f"Agent '{agent_name_in_config}' from config pruned (inactive/missing)."})
        
        self.agent_turn_order = verified_runtime_order # Update to the new verified list
        new_runtime_order_set = set(self.agent_turn_order)
        
        # If agents were pruned, update the station config to reflect the cleaned list
        if config_turn_order and len(verified_runtime_order) < len(config_turn_order):
            self.station.config[constants.STATION_CONFIG_AGENT_TURN_ORDER] = verified_runtime_order
            self.station._save_config()
            print(f"Orchestrator: Updated station config with pruned agent list. New order: {verified_runtime_order}")
            self._push_log_event("orchestrator_info", {"message": f"Station config updated with pruned agent list"})

        agents_removed = previous_runtime_order_set - new_runtime_order_set
        for removed_agent_name in agents_removed:
            if removed_agent_name in self.agent_llm_connectors:
                connector = self.agent_llm_connectors.pop(removed_agent_name)
                if hasattr(connector, 'end_session_and_cleanup'):
                    connector.end_session_and_cleanup()
                self._push_log_event("connector_status", {"agent_name": removed_agent_name, "status": "removed_cleaned_up", "reason": "No longer in active turn order (e.g., ascended, ended)."})
                print(f"Orchestrator: Cleaned up connector for removed agent: {removed_agent_name}")

        agents_departed = sorted(agents_removed | set(pruned_from_config))
        if agents_departed:
            # Analyze departure reasons and handle based on user requirements
            ascended_agents = []
            other_departed_agents = []
            respawned_agents = []
            
            for agent_name in agents_departed:
                departure_reason = self.station.get_agent_departure_reason(agent_name)
                
                if departure_reason == 'ascended':
                    ascended_agents.append(agent_name)
                    # For ascended agents: no pausing, no respawning (per requirements)
                    self._push_log_event("orchestrator_info", {
                        "agent_name": agent_name,
                        "departure_reason": "ascension",
                        "action": "no_pause_no_respawn",
                        "message": f"Guest agent {agent_name} ascended - continuing without pause"
                    })
                    print(f"Orchestrator: Guest agent '{agent_name}' ascended. Continuing without pause.")
                    
                else:
                    other_departed_agents.append(agent_name)
                    
                    if constants.AUTO_RESPAWN:
                        # Attempt to respawn the agent
                        respawn_name = self.station.create_respawn_guest_agent(agent_name)
                        if respawn_name:
                            respawned_agents.append(respawn_name)
                            self._push_log_event("orchestrator_info", {
                                "original_agent": agent_name,
                                "respawned_agent": respawn_name,
                                "departure_reason": departure_reason,
                                "action": "respawned",
                                "message": f"Agent {agent_name} departed, respawned as {respawn_name}"
                            })
                            print(f"Orchestrator: Agent '{agent_name}' departed (reason: {departure_reason}), respawned as '{respawn_name}'.")
                        else:
                            self._push_log_event("orchestrator_error", {
                                "agent_name": agent_name,
                                "departure_reason": departure_reason,
                                "action": "respawn_failed",
                                "message": f"Failed to respawn agent {agent_name}"
                            })
                            print(f"Orchestrator: Failed to respawn agent '{agent_name}' after departure (reason: {departure_reason}).")
            
            # Determine if we should pause the station
            should_pause = False
            pause_reasons = []
            
            # If AUTO_RESPAWN is False and we have non-ascended departures, pause
            if not constants.AUTO_RESPAWN and other_departed_agents:
                should_pause = True
                pause_reasons.append(f"AUTO_RESPAWN disabled and {len(other_departed_agents)} agent(s) departed")
            
            # If AUTO_RESPAWN is True but some respawns failed, pause
            if constants.AUTO_RESPAWN and other_departed_agents:
                failed_respawns = len(other_departed_agents) - len(respawned_agents)
                if failed_respawns > 0:
                    should_pause = True
                    pause_reasons.append(f"{failed_respawns} agent(s) failed to respawn")
            
            if should_pause:
                self.is_paused = True
                self.pause_condition_met = True
                self.pause_reason_message = f"Agent departure issues: {'; '.join(pause_reasons)}. Station paused."
                self._push_log_event("orchestrator_status", {
                    "status": "paused_agent_departure",
                    "reason": self.pause_reason_message,
                    "departed_agents": agents_departed,
                    "runtime_removed_agents": sorted(agents_removed),
                    "config_pruned_agents": pruned_from_config,
                    "ascended_agents": ascended_agents,
                    "other_departed_agents": other_departed_agents,
                    "respawned_agents": respawned_agents
                })
                print(f"Orchestrator: {self.pause_reason_message}")
                
                # Save next agent index if pausing
                if self.station and self.is_prepared:
                    self.station.save_next_agent_index_to_config(self.current_agent_index_in_turn_order)
                    self._push_log_event("orchestrator_info", {"message": f"Saved next agent index {self.current_agent_index_in_turn_order} due to agent departure pause."})
            else:
                # No pause needed - log the successful handling
                self._push_log_event("orchestrator_status", {
                    "status": "agent_departure_handled",
                    "departed_agents": agents_departed,
                    "runtime_removed_agents": sorted(agents_removed),
                    "config_pruned_agents": pruned_from_config,
                    "ascended_agents": ascended_agents,
                    "respawned_agents": respawned_agents,
                    "message": "Agent departures handled successfully without pausing"
                })
                print(f"Orchestrator: Agent departures handled successfully. Ascended: {ascended_agents}, Respawned: {respawned_agents}. Continuing without pause.")
            
            # If we respawned agents, reload the turn order to include them
            if respawned_agents and not should_pause:
                config_turn_order = list(self.station.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, []))
                verified_runtime_order = []
                for agent_name_in_config in config_turn_order:
                    if self.station.agent_module.load_agent_data(agent_name_in_config):
                        verified_runtime_order.append(agent_name_in_config)
                self.agent_turn_order = verified_runtime_order
                new_runtime_order_set = set(self.agent_turn_order)
                print(f"Orchestrator: Turn order reloaded after respawning. New order: {self.agent_turn_order}")
                
                # Update config with the verified order after respawning
                if verified_runtime_order != config_turn_order:
                    self.station.config[constants.STATION_CONFIG_AGENT_TURN_ORDER] = verified_runtime_order
                    self.station._save_config()


        agents_added = new_runtime_order_set - previous_runtime_order_set
        for added_agent_name in agents_added:
            print(f"Orchestrator: Agent '{added_agent_name}' detected as new to runtime order. Connector will initialize on first LLM use.")
            self._push_log_event("orchestrator_info", {"message": f"Agent {added_agent_name} new to runtime order. Connector initialization deferred."})


        self._push_log_event("orchestrator_info", {"message": f"Runtime agent turn order updated: {self.agent_turn_order}"})
        
        if not self.agent_turn_order:
            print("Orchestrator: Warning - Agent turn order is empty after verification.")
        
        if not self.is_prepared or (self.is_prepared and self.current_agent_index_in_turn_order == 0 and not self.current_tick_processed_agents):
            loaded_next_index = self.station.get_next_agent_index_from_config()
            if self.agent_turn_order and 0 <= loaded_next_index < len(self.agent_turn_order):
                self.current_agent_index_in_turn_order = loaded_next_index
                print(f"Orchestrator: Setting next agent index from config: {loaded_next_index} ({self.agent_turn_order[loaded_next_index] if self.agent_turn_order else 'N/A'}).")
                self._push_log_event("orchestrator_info", {"message": f"Resuming from saved agent index {loaded_next_index}."})
            elif self.agent_turn_order: 
                print(f"Orchestrator: Saved agent index {loaded_next_index} out of bounds. Resetting to 0.")
                self.current_agent_index_in_turn_order = 0
                if self.is_prepared: self.station.save_next_agent_index_to_config(0) 
            else: 
                self.current_agent_index_in_turn_order = 0

        if not self.is_prepared and self.agent_turn_order:
            waiting_agent_index = None
            waiting_agent_name = None
            for idx, agent_name_in_order in enumerate(self.agent_turn_order):
                agent_data = self.station.agent_module.load_agent_data(agent_name_in_order)
                if agent_data and agent_data.get(constants.AGENT_WAITING_STATION_RESPONSE_KEY, False):
                    waiting_agent_index = idx
                    waiting_agent_name = agent_name_in_order
                    # Clear stale flag left behind by an interrupted response.
                    agent_data[constants.AGENT_WAITING_STATION_RESPONSE_KEY] = False
                    self.station.agent_module.save_agent_data(agent_name_in_order, agent_data)
                    break

            if waiting_agent_index is not None:
                self.current_agent_index_in_turn_order = waiting_agent_index
                self.station.save_next_agent_index_to_config(waiting_agent_index)
                print(f"Orchestrator: Resuming from interrupted agent {waiting_agent_name} at index {waiting_agent_index}.")
                self._push_log_event("orchestrator_info", {
                    "message": f"Resuming from interrupted agent {waiting_agent_name} at index {waiting_agent_index}."
                })
        
        if self.current_agent_index_in_turn_order >= len(self.agent_turn_order):
            self.current_agent_index_in_turn_order = 0
            
        return bool(self.agent_turn_order)
    
    def _trigger_pause_due_to_llm_error(self, agent_name: str, error: Exception, error_type: str = "LLM_TRANSIENT_ERROR"):
        """Sets orchestrator to paused state due to an LLM error and saves current agent index."""
        self.is_paused = True
        self.pause_condition_met = True # Or a more specific flag like self.llm_api_error_pause = True
        self.pause_reason_message = f"{error_type} for agent {agent_name}: {str(error)[:150]}. Station paused. Please investigate and resume."
        
        # Determine the index of the agent that failed, so we retry them on resume.
        # self.current_agent_index_in_turn_order should already be pointing to the agent
        # whose LLM call just failed, or if it was incremented, it's pointing to the next.
        # For retry, we want the index of the *current failing agent*.
        # Connector failures happen before the tick commit path advances any agent state.
        index_to_save = self.current_agent_index_in_turn_order 
        try:
            if self.agent_turn_order[self.current_agent_index_in_turn_order] != agent_name:
                # This case should be rare if index is managed correctly, but as a fallback:
                if agent_name in self.agent_turn_order:
                    index_to_save = self.agent_turn_order.index(agent_name)
        except IndexError:
            pass # Keep current_agent_index_in_turn_order if out of bounds

        if self.station and self.is_prepared:
            self.station.save_next_agent_index_to_config(index_to_save)
            self._push_log_event("orchestrator_info", {"message": f"Saved next agent index {index_to_save} (agent: {agent_name}) to config due to LLM error pause."})

        self._push_log_event("orchestrator_status", {
            "status": "paused_llm_error", 
            "agent_name": agent_name,
            "reason": self.pause_reason_message,
            "error_details": str(error), # Full error for detailed log
            "next_agent_index_on_resume": index_to_save
        })
        print(f"Orchestrator: {self.pause_reason_message}")

    def pause_due_to_research_issue(self, reason: str):
        """Pause the whole orchestrator for Research Center issues that require manual resume."""
        self.is_paused = True
        self.pause_condition_met = True
        self.pause_reason_message = (reason or "Research Center issue requires manual resume.").strip()

        index_to_save = self.current_agent_index_in_turn_order
        if self.station and self.is_prepared:
            self.station.save_next_agent_index_to_config(index_to_save)
            self._push_log_event(
                "orchestrator_info",
                {"message": f"Saved next agent index {index_to_save} due to Research Center pause."},
            )

        self._push_log_event(
            "orchestrator_status",
            {
                "status": "paused_research_issue",
                "reason": self.pause_reason_message,
                "next_agent_index_on_resume": index_to_save,
            },
        )
        print(f"Orchestrator: {self.pause_reason_message}")
    
    def initialize_connectors_for_active_agents(self) -> bool:
        # This method now iterates self.agent_turn_order which should be up-to-date
        # from a preceding _load_agent_turn_order() call if called by prepare_for_run.
        # Connector construction is intentionally lazy because building a connector
        # parses the agent's full LLM history.
        self._push_log_event("orchestrator_info", {"message": "Validating LLM connector configs for current turn order..."})

        if not self.agent_turn_order:
             self._push_log_event("orchestrator_info", {"message": "No agents in current turn order to validate connector configs for."})
             return True 

        all_successful = True
        for agent_name in self.agent_turn_order:
            if not self._validate_connector_config_for_agent(agent_name):
                all_successful = False
        
        msg_end = f"Connector config validation complete. Connectors will initialize lazily. All successful: {all_successful}."
        self._push_log_event("orchestrator_info", {"message": msg_end})
        # self.is_prepared should be set based on this in prepare_for_run
        return all_successful

    def _validate_connector_config_for_agent(self, agent_name: str) -> bool:
        agent_data = self.station.agent_module.load_agent_data(agent_name)
        if not agent_data:
            self._push_log_event("connector_error", {"agent_name": agent_name, "message": "Agent data not found for connector config validation."})
            return False
        model_provider_class = agent_data.get(constants.AGENT_MODEL_PROVIDER_CLASS_KEY)
        model_name_specific = agent_data.get(constants.AGENT_MODEL_NAME_KEY)
        if not model_provider_class or not model_name_specific:
            self._push_log_event("connector_error", {"agent_name": agent_name, "message": "Agent missing provider or model name."})
            return False
        agent_specific_data_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME, agent_name)
        try:
            file_io_utils.ensure_dir_exists(agent_specific_data_path)
        except Exception as e:
            self._push_log_event("connector_error", {"agent_name": agent_name, "message": f"Dir creation fail: {e}"})
            return False
        return True

    def initialize_connector_for_agent(self, agent_name: str, force_reinitialize: bool = False) -> bool:
        if not force_reinitialize and agent_name in self.agent_llm_connectors:
            return True 
        agent_data = self.station.agent_module.load_agent_data(agent_name)
        if not agent_data:
            self._push_log_event("connector_error", {"agent_name": agent_name, "message": "Agent data not found for connector init."})
            return False
        model_provider_class = agent_data.get(constants.AGENT_MODEL_PROVIDER_CLASS_KEY)
        model_name_specific = agent_data.get(constants.AGENT_MODEL_NAME_KEY)
        if not model_provider_class or not model_name_specific:
            self._push_log_event("connector_error", {"agent_name": agent_name, "message": f"Agent missing provider or model name."})
            return False
        agent_specific_data_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME, agent_name)
        try: file_io_utils.ensure_dir_exists(agent_specific_data_path)
        except Exception as e: self._push_log_event("connector_error", {"agent_name": agent_name, "message": f"Dir creation fail: {e}"}); return False
        raw_role_definition = self.station.agent_module.get_agent_role_definition(agent_data)
        system_prompt = build_station_level_system_prompt(agent_name, raw_role_definition)
        temperature = agent_data.get(constants.AGENT_LLM_TEMPERATURE_KEY, 1.0)
        temperature = float(temperature) if temperature is not None else 1.0
        max_tokens = agent_data.get(constants.AGENT_LLM_MAX_TOKENS_KEY); max_tokens = int(max_tokens) if max_tokens is not None else None
        custom_api_params = agent_data.get(constants.AGENT_LLM_CUSTOM_API_PARAMS_KEY)
        self._push_log_event("connector_status", {"agent_name": agent_name, "status": "initializing", "provider": model_provider_class, "model": model_name_specific})
        try:
            connector = create_llm_connector(
                model_class_name=model_provider_class, model_name=model_name_specific,
                agent_name=agent_name, agent_data_path=agent_specific_data_path,
                api_key=None, system_prompt=system_prompt,
                temperature=temperature, max_output_tokens=max_tokens,
                custom_api_params=custom_api_params
            )
            if connector: self.agent_llm_connectors[agent_name] = connector; self._push_log_event("connector_status", {"agent_name": agent_name, "status": "initialized_success"}); return True
            else: self._push_log_event("connector_error", {"agent_name": agent_name, "message": "Factory returned None."}); return False
        except Exception as e: self._push_log_event("connector_error", {"agent_name": agent_name, "message": f"Exception: {str(e)}"}); traceback.print_exc(); return False

    def _get_current_connector_for_agent(self, agent_name: str) -> Optional[BaseLLMConnector]:
        connector = self.agent_llm_connectors.get(agent_name)
        current_generation = runtime_api_config.get_generation()
        if connector and getattr(connector, "api_runtime_config_generation", -1) == current_generation:
            return connector

        with self._api_runtime_connector_lock:
            connector = self.agent_llm_connectors.get(agent_name)
            if connector and getattr(connector, "api_runtime_config_generation", -1) == current_generation:
                return connector
            if connector:
                self._push_log_event("connector_status", {
                    "agent_name": agent_name,
                    "status": "reinitializing",
                    "reason": "runtime_api_config_updated",
                })
            if not self.initialize_connector_for_agent(agent_name, force_reinitialize=True):
                return self.agent_llm_connectors.get(agent_name)
            return self.agent_llm_connectors.get(agent_name)

    def handle_runtime_api_config_updated(self, generation: int) -> None:
        self._push_log_event("orchestrator_info", {
            "message": f"Runtime API configuration updated to generation {generation}. New LLM requests will use refreshed connectors."
        })

    def _pause_for_unavailable_connector(self, agent_name: str, current_tick: int) -> None:
        """Pause the station when an agent connector cannot be constructed."""
        err_msg = f"SYSTEM_ERROR: No LLM connector for {agent_name}."
        self._push_log_event(
            "llm_event_error",
            {"agent_name": agent_name, "tick": current_tick, "error": err_msg},
        )
        self.station._log_dialogue_entry(agent_name, {
            "tick": current_tick,
            "speaker": "Station",
            "type": "llm_connector_error",
            "error": err_msg,
        })
        self._trigger_pause_due_to_llm_error(
            agent_name,
            LLMConnectorError(err_msg),
            "LLM_CONNECTOR_UNAVAILABLE",
        )

    def _check_automatic_wait_conditions(self) -> Tuple[bool, Dict[str, str]]:
        """Check for conditions that should trigger waiting state (auto-resumes when resolved)"""
        waiting_reasons = {}

        if (
            hasattr(self.station, 'should_wait_for_research_evaluations_at_tick_boundary')
            and self.station.should_wait_for_research_evaluations_at_tick_boundary()
        ):
            waiting_reasons['research_tick_boundary'] = "Research evaluations at tick limit"
        
        if hasattr(self.station, 'has_pending_archive_evaluations') and self.station.has_pending_archive_evaluations():
            waiting_reasons['pending_archives'] = "Pending archive evaluations"
        
        return bool(waiting_reasons), waiting_reasons

    def _check_automatic_pause_conditions(self) -> Tuple[bool, str]:
        """Check for conditions that should trigger pause state (manual resume required)"""
        pause_now = False
        reason = ""
        
        # Only check for human intervention if HUMAN_REQUEST_PAUSE is True
        if constants.HUMAN_REQUEST_PAUSE:
            agents_awaiting_human: List[str] = []
            if hasattr(self.station, 'get_agents_awaiting_human_intervention'):
                agents_awaiting_human = self.station.get_agents_awaiting_human_intervention() 
            else: 
                for agent_name_check in self.agent_turn_order:
                    agent_data = self.station.agent_module.load_agent_data(agent_name_check)
                    if agent_data and agent_data.get(constants.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG):
                        agents_awaiting_human.append(agent_name_check)
            
            if agents_awaiting_human:
                pause_now = True
                reason += f"Agent(s) {', '.join(agents_awaiting_human)} awaiting human intervention."
        
        self.pause_reason_message = reason.strip() if pause_now else ""
        return pause_now, self.pause_reason_message

    def _check_wait_conditions_resolved(self) -> bool:
        """Check if all current waiting conditions have been resolved"""
        if not self.waiting_reasons:
            return True
            
        # Check each waiting condition

        if 'research_tick_boundary' in self.waiting_reasons:
            if (
                hasattr(self.station, 'should_wait_for_research_evaluations_at_tick_boundary')
                and self.station.should_wait_for_research_evaluations_at_tick_boundary()
            ):
                return False

        if 'pending_archives' in self.waiting_reasons:
            if hasattr(self.station, 'has_pending_archive_evaluations') and self.station.has_pending_archive_evaluations():
                return False  # Still pending
        
        return True  # All conditions resolved

    def _enter_waiting_state(self, waiting_reasons: Dict[str, str]):
        """Enter waiting state with the given reasons"""
        self.is_waiting = True
        self.waiting_reasons = waiting_reasons.copy()
        reason_text = ", ".join(waiting_reasons.values())
        
        self._push_log_event("orchestrator_status", {
            "status": "waiting", 
            "reasons": waiting_reasons,
            "reason_text": reason_text
        })
        
        if self.station and self.is_prepared:
            self.station.save_next_agent_index_to_config(self.current_agent_index_in_turn_order)
            self._push_log_event("orchestrator_info", {
                "message": f"Saved next agent index {self.current_agent_index_in_turn_order} to config due to waiting state."
            })

    def _exit_waiting_state(self):
        """Exit waiting state and resume normal processing"""
        self.is_waiting = False
        old_reasons = self.waiting_reasons.copy()
        self.waiting_reasons.clear()
        
        self._push_log_event("orchestrator_status", {
            "status": "waiting_resolved",
            "resolved_reasons": old_reasons,
            "message": "Waiting conditions resolved, automatically resuming"
        })

    def maybe_run_context_compaction_after_turn(self, agent_name: str, current_tick: int) -> bool:
        agent_data = self.station.agent_module.load_agent_data(agent_name)
        if not self.station.should_compact_agent_context(agent_data):
            return False
        return self._run_context_compaction_maintenance(agent_name, current_tick)

    def _run_context_compaction_maintenance(self, agent_name: str, current_tick: int) -> bool:
        connector = self._get_current_connector_for_agent(agent_name)
        if not connector:
            self._push_log_event(
                "orchestrator_warning",
                {
                    "message": f"No connector found for context compaction for agent {agent_name}.",
                    "agent_name": agent_name,
                    "tick": current_tick,
                },
            )
            return False

        prompt = constants.CONTEXT_COMPACTION_PROMPT_TEMPLATE.format(agent_name=agent_name)
        try:
            self._push_log_event("llm_event", {
                "agent_name": agent_name,
                "tick": current_tick,
                "direction": "to_llm",
                "type": "context_compaction_prompt",
                "text_content": prompt,
                "full_length": len(prompt),
            })
            self.station._log_dialogue_entry(agent_name, {
                "tick": current_tick,
                "speaker": "Station",
                "type": "context_compaction_prompt",
                "content": prompt,
            })
            self._push_log_event(
                "context_compaction_event",
                {
                    "type": "prompt",
                    "agent_name": agent_name,
                    "tick": current_tick,
                },
            )

            try:
                response_text, thinking_text, _token_info = connector.send_message(prompt, current_tick)
            except (
                LLMTransientAPIError,
                LLMPermanentAPIError,
                LLMSafetyBlockError,
                LLMConnectorError,
                LLMContextOverflowError,
                LLMCorruptedThoughtSignatureError,
            ) as exc:
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick,
                    "speaker": "Station",
                    "type": "context_compaction_error",
                    "error": str(exc),
                })
                self._trigger_pause_due_to_llm_error(agent_name, exc, "LLM_CONTEXT_COMPACTION_ERROR")
                return False
            except Exception as exc:
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick,
                    "speaker": "Station",
                    "type": "context_compaction_error",
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                })
                self._trigger_pause_due_to_llm_error(agent_name, exc, "LLM_CONTEXT_COMPACTION_ERROR")
                return False

            if thinking_text:
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick,
                    "speaker": "AgentLLM",
                    "type": "thinking_block_context_compaction",
                    "agent_name": agent_name,
                    "content": thinking_text,
                })
            self.station._log_dialogue_entry(agent_name, {
                "tick": current_tick,
                "speaker": "Agent",
                "type": "context_compaction_response",
                "content": response_text,
            })
            self._push_log_event("llm_event", {
                "agent_name": agent_name,
                "tick": current_tick,
                "direction": "from_llm",
                "type": "context_compaction_response",
                "text_content": response_text,
                "thinking_text": thinking_text,
                "full_length": len(response_text or ""),
                "token_info": _token_info,
            })

            summary = context_compaction.normalize_summary_response(response_text)
            if not self.station.save_context_compaction_summary(agent_name, current_tick, summary):
                self._push_log_event(
                    "orchestrator_warning",
                        {
                            "message": f"Failed to save context compaction summary for {agent_name}.",
                            "agent_name": agent_name,
                            "tick": current_tick,
                        },
                )
                return False
            self.station._log_dialogue_entry(agent_name, {
                "tick": current_tick,
                "speaker": "Station",
                "type": "context_compaction_complete",
                "compacted_after_tick": current_tick,
            })
            self._push_log_event(
                "context_compaction_event",
                {
                    "type": "complete",
                    "agent_name": agent_name,
                    "tick": current_tick,
                },
            )
            return True
        finally:
            try:
                if hasattr(connector, "reload_session_from_disk"):
                    connector.reload_session_from_disk()
            except Exception as reload_exc:
                self._push_log_event(
                    "orchestrator_error",
                    {
                        "message": f"Failed to reload connector after context compaction for {agent_name}: {reload_exc}",
                        "agent_name": agent_name,
                        "tick": current_tick,
                        "trace": traceback.format_exc(),
                    },
                )
                self._trigger_pause_due_to_llm_error(
                    agent_name,
                    reload_exc,
                    "LLM_CONTEXT_COMPACTION_RELOAD_ERROR",
                )

    @staticmethod
    def _clean_llm_override_value(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _prepare_internal_action_connector_context(
        self,
        agent_name: str,
        handler_wrapper: LoggingInternalActionHandlerWrapper,
        current_tick: int,
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "connector": None,
            "override_active": False,
        }

        try:
            llm_override = handler_wrapper.get_llm_override()
        except Exception as exc:
            context["connector"] = None
            context["override_active"] = True
            context["error"] = LLMConnectorError(
                f"Failed to read internal action LLM override for {agent_name}: {exc}",
                original_exception=exc,
            )
            return context

        if not llm_override:
            context["connector"] = self._get_current_connector_for_agent(agent_name)
            return context

        provider = self._clean_llm_override_value(llm_override.get("model_provider_class"))
        model_name = self._clean_llm_override_value(llm_override.get("model_name"))
        context.update({
            "connector": None,
            "override_active": True,
            "override": {"model_provider_class": provider, "model_name": model_name},
        })

        if not provider or not model_name:
            context["error"] = LLMConnectorError(
                "Internal action LLM override requires both model_provider_class and model_name."
            )
            return context

        agent_data = self.station.agent_module.load_agent_data(agent_name, include_ended=True, include_ascended=True)
        if not agent_data:
            context["error"] = LLMConnectorError(f"Agent data not found for internal action LLM override: {agent_name}.")
            return context

        agent_history_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME, agent_name)
        base_history_file = os.path.join(agent_history_dir, "llm_chat_history.yamll")
        temp_dir_for_session: Optional[str] = None

        try:
            temp_dir_for_session = tempfile.mkdtemp(prefix="station_internal_llm_override_", dir="/tmp")
            seeded_history_file = os.path.join(temp_dir_for_session, "llm_chat_history.yamll")
            if os.path.exists(base_history_file):
                shutil.copy2(base_history_file, seeded_history_file)

            baseline_doc_count = len(file_io_utils.load_yaml_lines(seeded_history_file))

            raw_role_definition = self.station.agent_module.get_agent_role_definition(agent_data)
            system_prompt = build_station_level_system_prompt(agent_name, raw_role_definition)

            temperature = agent_data.get(constants.AGENT_LLM_TEMPERATURE_KEY, 1.0)
            temperature = float(temperature) if temperature is not None else 1.0
            max_tokens = agent_data.get(constants.AGENT_LLM_MAX_TOKENS_KEY)
            max_tokens = int(max_tokens) if max_tokens is not None else None

            agent_provider = self._clean_llm_override_value(agent_data.get(constants.AGENT_MODEL_PROVIDER_CLASS_KEY))
            custom_api_params = None
            if agent_provider and agent_provider.lower() == provider.lower():
                custom_api_params = agent_data.get(constants.AGENT_LLM_CUSTOM_API_PARAMS_KEY)

            override_connector = create_llm_connector(
                model_class_name=provider,
                model_name=model_name,
                agent_name=agent_name,
                agent_data_path=temp_dir_for_session,
                api_key=None,
                system_prompt=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                custom_api_params=custom_api_params,
            )
            if not override_connector:
                raise LLMConnectorError(
                    f"Factory returned no connector for internal action LLM override {provider}/{model_name}."
                )

            context.update({
                "connector": override_connector,
                "override_connector": override_connector,
                "temp_dir": temp_dir_for_session,
                "temp_history_file": seeded_history_file,
                "base_history_file": base_history_file,
                "baseline_doc_count": baseline_doc_count,
            })
            self._push_log_event("internal_action_event", {
                "agent_name": agent_name,
                "tick": current_tick,
                "status": "llm_override_start",
                "handler": type(handler_wrapper.actual_handler).__name__,
                "provider": provider,
                "model": model_name,
            })
            return context
        except Exception as exc:
            if temp_dir_for_session and os.path.isdir(temp_dir_for_session):
                try:
                    shutil.rmtree(temp_dir_for_session, ignore_errors=True)
                except Exception:
                    pass
            context["error"] = exc if isinstance(exc, LLMConnectorError) else LLMConnectorError(
                f"Failed to create internal action LLM override connector for {agent_name}: {exc}",
                original_exception=exc,
            )
            return context

    def _finalize_internal_action_connector_context(
        self,
        agent_name: str,
        connector_context: Dict[str, Any],
        current_tick: int,
    ) -> None:
        if not connector_context.get("override_active"):
            return

        provider = None
        model_name = None
        override = connector_context.get("override")
        if isinstance(override, dict):
            provider = override.get("model_provider_class")
            model_name = override.get("model_name")

        temp_connector = connector_context.get("override_connector")
        temp_dir = connector_context.get("temp_dir")

        try:
            temp_history_file = connector_context.get("temp_history_file")
            base_history_file = connector_context.get("base_history_file")
            baseline_doc_count = int(connector_context.get("baseline_doc_count") or 0)
            migrated_count = 0

            if isinstance(temp_history_file, str) and isinstance(base_history_file, str):
                temp_entries = file_io_utils.load_yaml_lines(temp_history_file)
                new_entries = temp_entries[baseline_doc_count:]
                for entry in new_entries:
                    file_io_utils.append_yaml_line(entry, base_history_file)
                    migrated_count += 1

            normal_connector = self.agent_llm_connectors.get(agent_name)
            if normal_connector:
                try:
                    normal_connector.reload_session_from_disk()
                    refreshed_tokens = normal_connector.get_current_total_session_tokens()
                    if refreshed_tokens is not None:
                        self.station.update_agent_token_budget(agent_name, refreshed_tokens)
                except Exception as reload_exc:
                    self._push_log_event("orchestrator_error", {
                        "message": f"Failed to reload original connector after internal action LLM override: {reload_exc}",
                        "agent_name": agent_name,
                        "tick": current_tick,
                        "trace": traceback.format_exc(),
                    })
                    self._trigger_pause_due_to_llm_error(
                        agent_name,
                        reload_exc,
                        "LLM_CONNECTOR_RELOAD_ERROR_INTERNAL",
                    )

            self._push_log_event("internal_action_event", {
                "agent_name": agent_name,
                "tick": current_tick,
                "status": "llm_override_end",
                "provider": provider,
                "model": model_name,
                "migrated_history_entries": migrated_count,
            })
        except Exception as migrate_exc:
            self._push_log_event("orchestrator_error", {
                "message": f"Failed to migrate internal action LLM override history: {migrate_exc}",
                "agent_name": agent_name,
                "tick": current_tick,
                "trace": traceback.format_exc(),
            })
            self._trigger_pause_due_to_llm_error(
                agent_name,
                migrate_exc,
                "LLM_HISTORY_MIGRATION_ERROR_INTERNAL",
            )
        finally:
            try:
                if temp_connector and hasattr(temp_connector, "end_session_and_cleanup"):
                    temp_connector.end_session_and_cleanup()
            except Exception:
                pass
            if isinstance(temp_dir, str) and os.path.isdir(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

    def run_single_tick(self):
        self._check_deployment_log_sizes(self.station._get_current_tick())
        if (
            self.research_submission_service is None
            and constants.RESEARCH_CENTER_ENABLED
            and getattr(constants, "PARALLEL_RESEARCH_FAST_LANE_ENABLED", True)
        ):
            from station.eval_research.submission_service import ResearchSubmissionService

            self.research_submission_service = ResearchSubmissionService(
                self.station,
                log_event_func=self._push_log_event,
            )
            self.research_submission_service.start()
        if (
            self.archive_survey_submission_service is None
            and getattr(constants, "ARCHIVE_SURVEY_ENABLED", False)
            and getattr(constants, "PARALLEL_ARCHIVE_SURVEY_FAST_LANE_ENABLED", True)
        ):
            from station.eval_archive.surveyor import ArchiveSurveySubmissionService

            self.archive_survey_submission_service = ArchiveSurveySubmissionService(
                self.station,
                log_event_func=self._push_log_event,
            )
            self.archive_survey_submission_service.start()
        if self.parallel_tick_runner is None:
            from station.sync.parallel_runner import ParallelTickRunner

            self.parallel_tick_runner = ParallelTickRunner(self)
        return self.parallel_tick_runner.run_single_tick()

    def main_loop(self):
        self._push_log_event("orchestrator_status", {"status": "main_loop_initiated"})
        self.is_running = True
        self.pause_event.clear() 

        try:
            while self.is_running:
                if self.is_paused:
                    self._push_log_event("orchestrator_status", {"status": "paused_wait_main_loop", "reason": self.get_pause_reason()})
                    self.pause_event.wait() 
                    self.pause_event.clear() 
                    if not self.is_running: break 
                    self._push_log_event("orchestrator_status", {"status": "resumed_main_loop"})
                elif self.is_waiting:
                    # Check if wait conditions are resolved
                    if self._check_wait_conditions_resolved():
                        self._exit_waiting_state()
                    else:
                        # Still waiting, sleep and check again
                        time.sleep(self.wait_check_interval)
                        continue

                if self._check_wait_conditions_before_next_tick:
                    self._check_wait_conditions_before_next_tick = False
                    should_wait, wait_reasons = self._check_automatic_wait_conditions()
                    if should_wait:
                        self._enter_waiting_state(wait_reasons)
                        time.sleep(self.wait_check_interval)
                        continue
                
                if not self.agent_turn_order:
                    self._push_log_event("orchestrator_info", {"message": "Agent turn order empty, re-validating configs/order."})
                    self.initialize_connectors_for_active_agents() 
                    if not self.agent_turn_order:
                        self._push_log_event("orchestrator_info", {"message": "Still no agents, sleeping."})
                        time.sleep(5) 
                        continue
                
                self.run_single_tick()
                
                if self.is_running and not self.is_paused: 
                    time.sleep(1) 
        except KeyboardInterrupt:
            self._push_log_event("orchestrator_status", {"status": "keyboard_interrupt"})
        except Exception as e:
            self._push_log_event("orchestrator_error", {"status": "main_loop_exception", "error": str(e), "trace": traceback.format_exc()})
            traceback.print_exc()
        finally:
            self.is_running = False 
            self._push_log_event("orchestrator_status", {"status": "main_loop_concluded"})

    def prepare_for_run(self) -> bool:
        if self.is_prepared and self.agent_turn_order: 
            self._push_log_event("orchestrator_info", {"message": "Orchestrator already prepared with agents."})
            return True
        
        self._push_log_event("orchestrator_info", {"message": "Preparing orchestrator for run..."})
        self.is_prepared = False 

        self._load_agent_turn_order() 
        # _load_agent_turn_order now sets current_agent_index_in_turn_order from config if not self.is_prepared
        
        if not self.agent_turn_order: 
            self._push_log_event("orchestrator_status", {"status": "prepared_no_agents", "prepared": True, "running": self.is_running, "message": "Preparation complete: No agents in (verified) turn order."})
            self.is_prepared = True 
            return True 

        if self.initialize_connectors_for_active_agents(): 
            self.is_prepared = True
            self._push_log_event("orchestrator_status", {"status": "prepared", "prepared": True, "running": self.is_running, "message": "Orchestrator prepared successfully with agents."})
            return True
        else:
            self.is_prepared = False 
            self._push_log_event("orchestrator_error", {"status": "prepare_failed", "prepared": False, "running": self.is_running, "message": "Orchestrator preparation failed (connector config issues)."})
            return False

    def start_processing_loop(self) -> bool:
        """
        Starts the main processing loop in a new thread if the orchestrator is prepared.
        """
        if self.is_running:
            msg = "Orchestrator processing loop is already running."
            print(f"Orchestrator: {msg}")
            self._push_log_event("orchestrator_control", {"action": "start_loop", "status": "already_running", "message": msg})
            return False # Or True if already running is considered a success for this call
        
        if not self.is_prepared:
            msg = "Orchestrator is not prepared (agent turn order/configs not validated). Please prepare first."
            print(f"Orchestrator: {msg}")
            self._push_log_event("orchestrator_control", {"action": "start_loop", "status": "not_prepared", "message": msg})
            return False

        self._push_log_event("orchestrator_control", {"action": "start_loop", "status": "attempting", "message": "Attempting to start processing loop..."})
        print("Orchestrator: Starting main loop in a new thread...")
        
        self.is_running = True 
        self.is_paused = False 
        self.pause_requested = False
        self.pause_condition_met = False
        self.pause_reason_message = ""
        self.pause_event.clear()
        
        # Clear waiting state
        self.is_waiting = False
        self.waiting_reasons.clear()
        
        self.orchestrator_thread = threading.Thread(target=self.main_loop, daemon=True)
        self.orchestrator_thread.start()
        self._push_log_event("orchestrator_status", {"status": "running", "prepared": self.is_prepared, "running": self.is_running, "message": "Orchestrator processing loop thread started."})
        return True

    # Orchestrator.start_orchestration can now be a convenience method
    def start_orchestration(self) -> bool:
        """Prepares and then starts the processing loop."""
        if not self.is_prepared:
            if not self.prepare_for_run():
                return False # Preparation failed
        return self.start_processing_loop()

    def stop_orchestration(self):
        if not self.is_running and (not self.orchestrator_thread or not self.orchestrator_thread.is_alive()):
            msg = "Orchestrator not currently running."
            print(f"Orchestrator: {msg}")
            self._push_log_event("orchestrator_control", {"action": "stop", "status": "not_running", "message": msg})
            if self.research_submission_service:
                self.research_submission_service.stop()
                self.research_submission_service = None
            if self.archive_survey_submission_service:
                self.archive_survey_submission_service.stop()
                self.archive_survey_submission_service = None
            return False
        
        msg_attempt = "Orchestrator stop requested."
        print(f"Orchestrator: {msg_attempt}")
        self._push_log_event("orchestrator_control", {"action": "stop", "status": "attempting", "message": msg_attempt})
        self.is_running = False

        # MODIFICATION: Save the next agent index before stopping
        if self.station and self.is_prepared: # Only save if prepared, otherwise index is not meaningful
            # If a tick completed fully, current_agent_index_in_turn_order would be 0 (for next tick).
            # If stopped mid-tick, it's the index of the agent that would have been next.
            self.station.save_next_agent_index_to_config(self.current_agent_index_in_turn_order)
            self._push_log_event("orchestrator_info", {"message": f"Saved next agent index {self.current_agent_index_in_turn_order} to config."})
        
        if self.is_paused: self.pause_event.set() # Wake up loop if paused
        
        # ... (thread join logic) ...
        final_message = "Orchestrator stopped." # Default
        if self.orchestrator_thread and self.orchestrator_thread.is_alive():
            self.orchestrator_thread.join(timeout=10) 
            if self.orchestrator_thread.is_alive(): final_message = "Orchestrator: Warning - Loop thread did not join."
            else: final_message = "Orchestrator: Loop thread joined. Orchestrator fully stopped."
        self._push_log_event("orchestrator_status", {"status": "stopped", "message": final_message})
        self.is_prepared = False # Orchestrator is no longer prepared once stopped

        if self.research_submission_service:
            self.research_submission_service.stop()
            self.research_submission_service = None
        if self.archive_survey_submission_service:
            self.archive_survey_submission_service.stop()
            self.archive_survey_submission_service = None
        
        # Stop auto research evaluator
        self.station.stop_auto_research_evaluator()

        # Stop auto external reporter
        self.station.stop_auto_external_reporter()

        # Stop archive surveyor
        self.station.stop_auto_archive_surveyor()
        
        return True

    def request_manual_pause(self):
        if not self.is_running:
            msg = "Orchestrator not running, cannot request pause."
            self._push_log_event("orchestrator_control", {"action": "manual_pause_request", "status": "not_running", "message": msg})
            return msg
        if self.is_paused:
            msg = "Orchestrator is already paused."
            self._push_log_event("orchestrator_control", {"action": "manual_pause_request", "status": "already_paused", "message": msg})
            return msg
        
        msg_req = "Manual pause requested. Will pause after current agent's turn or before next."
        print(f"Orchestrator: {msg_req}")
        self.pause_requested = True
        self._push_log_event("orchestrator_control", {"action": "manual_pause_request", "status": "pending", "message": msg_req})
        return msg_req

    def maybe_pause_after_configured_tick_end(self, ended_tick: int) -> bool:
        """Pause after a configured tick has fully completed.

        PAUSE_AFTER_TICK_END is interpreted as the tick that just ended, not the
        new current_tick written by Station.end_tick().
        """
        if not self.station:
            return False

        raw_target = getattr(constants, "PAUSE_AFTER_TICK_END", None)
        if raw_target is None or raw_target == "":
            return False

        try:
            target_tick = int(raw_target)
        except (TypeError, ValueError):
            self._push_log_event(
                "orchestrator_warning",
                {
                    "message": (
                        "Ignoring invalid PAUSE_AFTER_TICK_END: "
                        f"{raw_target!r}; expected an integer tick."
                    )
                },
            )
            return False

        if target_tick != int(ended_tick):
            return False

        self.is_paused = True
        self.pause_requested = False
        self.pause_condition_met = True
        self.pause_reason_message = (
            f"Configured pause after tick {ended_tick} completed. "
            "All agents for that tick have responded."
        )
        self.current_agent_index_in_turn_order = 0
        self.station.save_next_agent_index_to_config(0)
        self._push_log_event(
            "orchestrator_status",
            {
                "status": "paused_config_tick_end",
                "reason": self.pause_reason_message,
                "ended_tick": int(ended_tick),
                "current_tick": self.station._get_current_tick(),
                "next_agent_index": 0,
            },
        )
        print(f"Orchestrator: {self.pause_reason_message}")
        return True

    def cancel_pause_request(self):
        if not self.pause_requested:
            msg = "No pending pause request to cancel."
            self._push_log_event("orchestrator_control", {"action": "cancel_pause_request", "status": "no_request", "message": msg})
            return msg
        
        self.pause_requested = False
        msg = "Pending pause request has been cancelled."
        self._push_log_event("orchestrator_control", {"action": "cancel_pause_request", "status": "cancelled", "message": msg})
        return msg

    def resume_orchestration(self):
        if not self.is_running:
            msg = "Orchestrator not running, cannot resume."
            self._push_log_event("orchestrator_control", {"action": "resume_request", "status": "not_running", "message": msg})
            return msg
        if not self.is_paused and not self.is_waiting:
            msg = "Orchestrator is not currently paused or waiting."
            self._push_log_event("orchestrator_control", {"action": "resume_request", "status": "not_paused_or_waiting", "message": msg})
            return msg

        msg_resuming = "Orchestrator resuming..."
        print(f"Orchestrator: {msg_resuming}")
        self._check_wait_conditions_before_next_tick = True

        if constants.AUTO_EVAL_RESEARCH and constants.RESEARCH_CENTER_ENABLED:
            try:
                from station.eval_research import (
                    requeue_unfinished_instruction_evaluations,
                    reset_runtime_coder_counters,
                )

                reset_count = reset_runtime_coder_counters()
                if reset_count > 0:
                    self._push_log_event(
                        "orchestrator_info",
                        {"message": f"Reset runtime Research Center coder spawn counters for {reset_count} evaluation(s)."},
                    )
                requeued_retryable = requeue_unfinished_instruction_evaluations(
                    reason="Recovered after manual resume: unfinished instruction prompt requeued.",
                )
                if requeued_retryable > 0:
                    self._push_log_event(
                        "orchestrator_info",
                        {
                            "message": (
                                "Requeued "
                                f"{requeued_retryable} unfinished Research Center evaluation(s) after manual resume."
                            )
                        },
                    )
            except Exception as exc:
                print(f"Orchestrator: Error resetting Research Center coder spawn counters on resume: {exc}")
        
        # Clear both paused and waiting states
        self.is_paused = False
        self.pause_requested = False 
        self.pause_condition_met = False 
        self.pause_reason_message = ""
        
        if self.is_waiting:
            # Manual override of waiting state
            old_waiting_reasons = self.waiting_reasons.copy()
            self.is_waiting = False
            self.waiting_reasons.clear()
            self._push_log_event("orchestrator_status", {
                "status": "waiting_manually_overridden", 
                "overridden_reasons": old_waiting_reasons
            })
        
        self.pause_event.set() 
        self._push_log_event("orchestrator_status", {"status": "resuming", "message": msg_resuming})
        return msg_resuming
        
    def get_pause_reason(self) -> str:
        if self.is_paused:
            if self.pause_condition_met:
                return self.pause_reason_message or "Automatic condition met."
            return "Paused by user request or manually."
        return ""

    def add_agent_to_orchestrator(self, agent_name: str, reinitialize_connector: bool = False) -> bool:
        """
        Adds an agent to the orchestrator's turn order and initializes their LLM connector.
        Assumes agent_data file already exists and is configured with LLM details.
        Called by dynamic_add_agent_to_station after station.create_agent and config update.
        """
        self._push_log_event("agent_management", {"action": "add_to_orchestrator", "agent_name": agent_name, "status": "attempting"})
        if not self.station.agent_module.load_agent_data(agent_name): # Check if agent actually exists in station data
            msg = f"Agent {agent_name} data not found in station. Cannot add to orchestrator."
            self._push_log_event("agent_management", {"action": "add_to_orchestrator", "agent_name": agent_name, "status": "error", "message": msg})
            print(f"Orchestrator: {msg}")
            return False

        if not self.initialize_connector_for_agent(agent_name, force_reinitialize=reinitialize_connector):
            msg = f"Failed to initialize LLM connector for {agent_name}. Not fully added to orchestrator."
            self._push_log_event("agent_management", {"action": "add_to_orchestrator", "agent_name": agent_name, "status": "connector_fail", "message": msg})
            print(f"Orchestrator: {msg}")
            return False

        # Add to current runtime turn order if not present
        if agent_name not in self.agent_turn_order:
            self.agent_turn_order.append(agent_name)
            # If orchestrator is paused and new agent added before current_agent_index, adjust?
            # For simplicity, new agents are usually appended. If added mid-list, index might need care.
            # Current logic in _load_agent_turn_order handles index reset if list shrinks.
            msg_rt = f"Agent {agent_name} added to current runtime turn order."
            self._push_log_event("agent_management", {"action": "add_to_orchestrator", "agent_name": agent_name, "status": "added_to_runtime_order", "message": msg_rt})
            print(f"Orchestrator: {msg_rt}")
        
        # Ensure it's in the station's config (dynamic_add_agent_to_station should handle this primary save)
        station_config_order = list(self.station.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, []))
        if agent_name not in station_config_order:
            station_config_order.append(agent_name)
            self.station.config[constants.STATION_CONFIG_AGENT_TURN_ORDER] = station_config_order
            self.station._save_config() # Persist change
            self._push_log_event("agent_management", {"action": "add_to_orchestrator", "agent_name": agent_name, "status": "added_to_config_order"})
        
        return True


    def dynamic_add_agent_to_station(self, agent_type:str, model_provider_class: str, model_name: str,
                                     agent_name_override: Optional[str] = None,
                                     lineage: Optional[str] = None, generation: Optional[int] = None,
                                     initial_tokens_max: Optional[int] = None, internal_note: Optional[str] = None,
                                     assigned_ancestor: Optional[str] = None,
                                     role_definition: Optional[str] = None,
                                     llm_system_prompt: Optional[str] = None,
                                     llm_temperature: Optional[float] = None,
                                     llm_max_tokens: Optional[int] = None,
                                     llm_custom_api_params: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        # Allow adding agents while running - the station can handle it safely
        # if not self.is_paused and self.is_running:
        #     msg = "Orchestrator must be paused or stopped to add agents dynamically."
        #     self._push_log_event("agent_management", {"action": "dynamic_add_attempt", "status": "error_not_paused", "message": msg})
        #     return False, msg

        self._push_log_event("agent_management", {"action": "dynamic_add_attempt", "requested_name": agent_name_override or "auto-name"})

        # Keep backward compatibility with older callers and UI payload names.
        if role_definition is None and llm_system_prompt is not None:
            role_definition = llm_system_prompt
        
        created_agent_data, error_msg = self.station.create_agent(
            model_name=model_name, 
            agent_type=agent_type, agent_name=agent_name_override,
            lineage=lineage, generation=generation,
            initial_tokens_max=initial_tokens_max, internal_note=internal_note or "", 
            assigned_ancestor=assigned_ancestor or "",
            role_definition=role_definition,
        )
        if error_msg or not created_agent_data:
            full_err_msg = error_msg or "Failed to create agent in station."
            self._push_log_event("agent_management", {"action": "dynamic_add_fail_station_create", "error": full_err_msg})
            return False, full_err_msg
        
        new_agent_name = created_agent_data[constants.AGENT_NAME_KEY]

        # Reload data to ensure we have the definitive copy, then update and save
        current_agent_data = self.station.agent_module.load_agent_data(new_agent_name)
        if not current_agent_data: 
            err_reload = f"Failed to load newly created agent {new_agent_name} data for LLM config."
            self._push_log_event("agent_management", {"action": "dynamic_add_fail_load", "agent_name": new_agent_name, "error": err_reload})
            return False, err_reload

        current_agent_data[constants.AGENT_MODEL_PROVIDER_CLASS_KEY] = model_provider_class
        current_agent_data[constants.AGENT_MODEL_NAME_KEY] = model_name # Ensure this is set if create_agent didn't use it for this field
        if role_definition is not None:
            current_agent_data[constants.AGENT_ROLE_DEFINITION_KEY] = role_definition
        if llm_temperature is not None: current_agent_data[constants.AGENT_LLM_TEMPERATURE_KEY] = llm_temperature
        if llm_max_tokens is not None: current_agent_data[constants.AGENT_LLM_MAX_TOKENS_KEY] = llm_max_tokens
        if llm_custom_api_params is not None: current_agent_data[constants.AGENT_LLM_CUSTOM_API_PARAMS_KEY] = llm_custom_api_params
        
        if not self.station.agent_module.save_agent_data(new_agent_name, current_agent_data):
            err_save_llm_config = f"Failed to save LLM config for new agent {new_agent_name}."
            self._push_log_event("agent_management", {"action": "dynamic_add_fail_llm_config_save", "agent_name": new_agent_name, "error": err_save_llm_config})
            return False, err_save_llm_config
        
        self._push_log_event("agent_management", {"action": "dynamic_add_llm_config_saved", "agent_name": new_agent_name})

        if not self.add_agent_to_orchestrator(new_agent_name, reinitialize_connector=True):
            # Connector init failed, but agent exists in station.
            return False, f"Agent {new_agent_name} created in station, but LLM connector/orchestrator setup failed."
        
        final_msg = f"Agent {new_agent_name} added and connector initialized. Will be included in next turns if orchestrator is resumed/started."
        self._push_log_event("agent_management", {"action": "dynamic_add_success", "agent_name": new_agent_name, "message": final_msg})
        return True, final_msg

    def dynamic_end_agent_session_manually(self, agent_name_to_end: str) -> Tuple[bool, str]:
        self._push_log_event("agent_management", {"action": "dynamic_end_request", "agent_name": agent_name_to_end})
        
        agent_data = self.station.agent_module.load_agent_data(agent_name_to_end, include_ended=True, include_ascended=True)
        if not agent_data:
            msg = f"Agent '{agent_name_to_end}' not found in station records."
            self._push_log_event("agent_management", {"action": "dynamic_end_fail_not_found", "agent_name": agent_name_to_end, "error": msg})
            return False, msg

        if agent_data.get(constants.AGENT_SESSION_ENDED_KEY):
            msg = f"Agent '{agent_name_to_end}' session has already ended."
            return False, msg

        # Set the request flag
        success = self.station.update_specific_agent_fields(
            agent_name_to_end,
            {constants.AGENT_SESSION_END_REQUESTED_KEY: True}
        )

        if success:
            msg = f"Session end requested for agent '{agent_name_to_end}'. The session will be terminated at the start of their next turn."
            self._push_log_event("agent_management", {"action": "dynamic_end_request_success", "agent_name": agent_name_to_end, "message": msg})
            return True, msg
        else:
            msg = f"Failed to set session end request for agent '{agent_name_to_end}'."
            self._push_log_event("agent_management", {"action": "dynamic_end_request_fail", "agent_name": agent_name_to_end, "error": msg})
            return False, msg

    def remove_agent_from_orchestrator(self, agent_name: str):
        # Remove from runtime turn order
        if agent_name in self.agent_turn_order:
            # ... (index adjustment logic as before) ...
            current_idx_of_removed = -1
            try: current_idx_of_removed = self.agent_turn_order.index(agent_name)
            except ValueError: pass
            self.agent_turn_order = [name for name in self.agent_turn_order if name != agent_name]
            if current_idx_of_removed != -1 and current_idx_of_removed < self.current_agent_index_in_turn_order:
                self.current_agent_index_in_turn_order -= 1
            if self.current_agent_index_in_turn_order >= len(self.agent_turn_order):
                 self.current_agent_index_in_turn_order = 0 
            self._push_log_event("agent_management", {"action": "remove_from_runtime_order", "agent_name": agent_name})

        # Update and save the station's master config turn order (This part might be redundant if dynamic_end_agent_session_manually also calls station.end_agent_session which should update config)
        # However, if remove_agent_from_orchestrator is called directly for other reasons, this is needed.
        station_config_order = list(self.station.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, []))
        if agent_name in station_config_order:
            station_config_order = [name for name in station_config_order if name != agent_name]
            self.station.config[constants.STATION_CONFIG_AGENT_TURN_ORDER] = station_config_order
            self.station._save_config()
            self._push_log_event("agent_management", {"action": "remove_from_config_order", "agent_name": agent_name})

        if agent_name in self.agent_llm_connectors:
            connector = self.agent_llm_connectors.pop(agent_name)
            if hasattr(connector, 'end_session_and_cleanup'):
                connector.end_session_and_cleanup()
            self._push_log_event("agent_management", {"action": "connector_removed", "agent_name": agent_name})

    def resolve_human_intervention(self, agent_name: str, resolution_reason: str = "Intervention resolved by UI action.", human_response: Optional[str] = None, request_id: Optional[Any] = None) -> Tuple[bool, str]:
        """
        Clears the 'awaiting_human_intervention' flag for an agent and logs the event.
        Called after a human interaction is deemed complete or resolved.

        Args:
            agent_name: Name of the agent whose request is being resolved
            resolution_reason: Reason for resolution
            human_response: Optional response text from the human to be saved and sent to agent
            request_id: Optional request ID to resolve (defaults to most recent if omitted)
        """
        self._push_log_event("human_assist_event", {
            "type": "resolve_intervention_attempt",
            "agent_name": agent_name,
            "reason": resolution_reason,
            "has_response": human_response is not None,
            "request_id": request_id
        })

        # Load agent data, include ended in case the flag needs clearing on an already ended agent
        agent_data = self.station.agent_module.load_agent_data(agent_name, include_ended=True, include_ascended=True)
        if not agent_data:
            msg = f"Agent '{agent_name}' not found in station records. Cannot resolve human intervention."
            self._push_log_event("human_assist_event", {"type": "resolve_intervention_error", "agent_name": agent_name, "error": msg})
            print(f"Orchestrator: {msg}")
            return False, msg

        def _get_agent_request_ids(agent_data: Dict[str, Any]) -> List[Any]:
            request_ids = agent_data.get(constants.AGENT_HUMAN_INTERACTION_IDS_KEY)
            if isinstance(request_ids, list):
                return list(request_ids)
            legacy_id = agent_data.get(constants.AGENT_HUMAN_INTERACTION_ID_KEY)
            if legacy_id:
                return [legacy_id]
            return []

        request_ids = _get_agent_request_ids(agent_data)

        interaction_id = None
        if request_id is None:
            if request_ids:
                interaction_id = request_ids[-1]
        else:
            try:
                request_id_int = int(request_id)
            except (TypeError, ValueError):
                request_id_int = request_id
            if request_id_int in request_ids:
                interaction_id = request_id_int
            elif request_id in request_ids:
                interaction_id = request_id

        if interaction_id is None:
            msg = f"No matching pending request found for agent '{agent_name}'."
            self._push_log_event("human_assist_event", {"type": "resolve_intervention_error", "agent_name": agent_name, "error": msg})
            print(f"Orchestrator: {msg}")
            return False, msg

        print(f"Orchestrator: Resolving request - interaction_id={interaction_id}, human_response={'Yes' if human_response else 'No'}")

        # Save resolution to the log file via Administrative Counter
        if interaction_id:
            admin_room = self.station.rooms.get(constants.ROOM_ADMIN)
            if not admin_room:
                print(f"Orchestrator: Warning - Administrative Counter room not found")
            else:
                # Call the Administrative Counter's method to save the resolution
                current_tick = self.station._get_current_tick()
                save_success = admin_room.save_request_resolution(
                    request_id=interaction_id,
                    human_response=human_response,
                    resolution_reason=resolution_reason,
                    resolution_tick=current_tick
                )
                if not save_success:
                    print(f"Orchestrator: Warning - Failed to save resolution for request ID {interaction_id}")

        remaining_request_ids = [req_id for req_id in request_ids if req_id != interaction_id]
        delta_updates = {
            constants.AGENT_HUMAN_INTERACTION_IDS_KEY: remaining_request_ids
        }
        if remaining_request_ids:
            delta_updates[constants.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG] = True
            delta_updates[constants.AGENT_HUMAN_INTERACTION_ID_KEY] = remaining_request_ids[-1]
        else:
            delta_updates[constants.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG] = False
            delta_updates[constants.AGENT_HUMAN_INTERACTION_ID_KEY] = None # Clear the interaction ID

        if self.station.update_specific_agent_fields(agent_name, delta_updates):
            msg = f"Human intervention for agent '{agent_name}' (Request ID: {interaction_id or 'N/A'}) marked as resolved. Reason: {resolution_reason}"
            print(f"Orchestrator: {msg}")
            self._push_log_event("human_assist_event", {
                "type": "intervention_resolved",
                "agent_name": agent_name,
                "interaction_id": interaction_id,
                "request_id": interaction_id,
                "message": msg,
                "has_response": human_response is not None
            })

            # Add a system notification to the agent if they are still active and not the one resolving.
            # The agent might be the one triggering this via an action if the design allowed,
            # but typically human resolves via UI.
            active_agent_data = self.station.agent_module.load_agent_data(agent_name) # Check active status
            if active_agent_data: # Only send notification if agent is still active
                # Create notification with optional human response
                if human_response:
                    notification_msg = (
                        f"System: Your human assistance request (ID: {interaction_id or 'N/A'}) has been resolved.\n"
                        f"Human response:\n{human_response}"
                    )
                else:
                    notification_msg = f"System: Your human assistance request (ID: {interaction_id or 'N/A'}) has been marked as resolved by an assistant."

                self.station.agent_module.add_pending_notification(
                    active_agent_data,
                    notification_msg
                )
                self.station.agent_module.save_agent_data(agent_name, active_agent_data)
            
            # Update Administrative Counter tracking
            admin_room = self.station.rooms.get(constants.ROOM_ADMIN)
            if admin_room and hasattr(admin_room, 'refresh_pending_requests'):
                admin_room.refresh_pending_requests()
            
            # Check if this resolution clears an automatic pause condition for the orchestrator
            if self.is_paused and self.pause_condition_met:
                # Re-check all conditions. If this was the *only* condition, then pause_condition_met can be cleared.
                still_auto_paused_conditions, new_reason = self._check_automatic_pause_conditions()
                if not still_auto_paused_conditions:
                    self.pause_condition_met = False
                    self.pause_reason_message = "" # Cleared as no more auto-pause reasons
                    self._push_log_event("orchestrator_status", {
                        "status": "auto_pause_condition_cleared", 
                        "agent_name": agent_name, # Agent whose resolution might have cleared the condition
                        "message": "An automatic pause condition was cleared. Orchestrator may be resumable manually."
                    })
                    print(f"Orchestrator: Auto-pause condition potentially cleared by resolving intervention for {agent_name}.")
                else:
                    # Update reason if it changed (e.g., only pending tests remain)
                    self.pause_reason_message = new_reason
                    self._push_log_event("orchestrator_status", {
                        "status": "auto_pause_condition_updated", 
                        "reason": self.pause_reason_message
                    })


            return True, msg
        else:
            msg = f"Failed to update agent '{agent_name}'s status to resolve human intervention in their data file."
            self._push_log_event("human_assist_event", {"type": "resolve_intervention_error", "agent_name": agent_name, "error": msg})
            print(f"Orchestrator: {msg}")
            return False, msg

    def send_manual_message_to_agent_llm(self, agent_name: str, human_message: str, end_chat_after_this_message: bool) -> Tuple[bool, Dict[str, Any]]:
        self._push_log_event("human_assist_event", {
            "type": "manual_message_send_attempt", 
            "agent_name": agent_name, 
            "message_snippet": human_message[:100] + "..." if len(human_message) > 100 else human_message,
        })
        
        # Wait for agent to finish responding to station if currently processing
        wait_timeout = 60  # seconds
        wait_start_time = time.time()
        while time.time() - wait_start_time < wait_timeout:
            agent_data_check = self.station.agent_module.load_agent_data(agent_name, include_ended=True, include_ascended=True)
            if not agent_data_check:
                break  # Agent no longer exists
            if not agent_data_check.get(constants.AGENT_WAITING_STATION_RESPONSE_KEY, False):
                break  # Agent not waiting for response, safe to proceed
            time.sleep(0.5)  # Wait 500ms before checking again
        
        # Check if timeout occurred
        if time.time() - wait_start_time >= wait_timeout:
            msg = f"Timeout waiting for agent {agent_name} to finish responding to station. Please try again later."
            self._push_log_event("human_assist_event", {"type": "manual_message_timeout", "agent_name": agent_name, "error": msg})
            return False, {"error": msg}

        connector = self._get_current_connector_for_agent(agent_name)
        if not connector:
            print(f"Orchestrator: No active connector for {agent_name} during manual message. Attempting to create temporary one.")
            if not self.initialize_connector_for_agent(agent_name, force_reinitialize=True):
                msg = f"Failed to initialize LLM connector for agent {agent_name} for manual message."
                self._push_log_event("human_assist_event", {"type": "manual_message_error", "agent_name": agent_name, "error": msg})
                return False, {"error": msg}
            connector = self._get_current_connector_for_agent(agent_name)
            if not connector:
                msg = f"LLM connector for agent {agent_name} could not be established for manual message."
                self._push_log_event("human_assist_event", {"type": "manual_message_error", "agent_name": agent_name, "error": msg})
                return False, {"error": msg}

        current_tick = self.station._get_current_tick()
        
        interaction_id = "N/A (Manual Takeover)"
        agent_data_for_id = self.station.agent_module.load_agent_data(agent_name, include_ended=True, include_ascended=True)
        if agent_data_for_id and agent_data_for_id.get(constants.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG):
            request_ids = agent_data_for_id.get(constants.AGENT_HUMAN_INTERACTION_IDS_KEY)
            if isinstance(request_ids, list) and request_ids:
                interaction_id = request_ids[-1]
            else:
                interaction_id = agent_data_for_id.get(constants.AGENT_HUMAN_INTERACTION_ID_KEY, interaction_id)

        # Log human's message to the agent's main dialogue log
        self.station._log_dialogue_entry(agent_name, {
            "tick": current_tick, "speaker": "HumanAssistant", 
            "type": "manual_message_to_agent_llm", 
            "content": human_message, "interaction_id": interaction_id 
        })      

        # --- MODIFICATION START: Push SSE event for human's message immediately ---
        self._push_log_event("human_assist_event", {
            "type": "manual_message_human_part_sent", # Specific type for UI to pick up human's message
            "agent_name": agent_name,
            "tick": current_tick, 
            "text_content": human_message, # The human's message
            "interaction_id": interaction_id
        })
        # --- MODIFICATION END ---

        try:
            llm_response, thinking_text, token_info = connector.send_message(human_message, current_tick)

            total_tokens_in_session = token_info.get('total_tokens_in_session')
            if total_tokens_in_session is not None:
                can_agent_session_continue = self.station.update_agent_token_budget(agent_name, total_tokens_in_session)
                if not can_agent_session_continue:
                    self._push_log_event("orchestrator_warning", {
                        "message": f"Failed to persist token budget update during manual chat for agent {agent_name}.",
                        "agent_name": agent_name,
                        "tick": current_tick
                    })
                    self.station._log_dialogue_entry(agent_name, {
                        "tick": current_tick, "speaker": "Station", "type": "token_budget_update_warning_manual_chat",
                        "reason": "Token budget update could not be saved during manual chat."
                    })
  
            if thinking_text: 
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick, "speaker": "AgentLLM", 
                    "type": "thinking_block", 
                    "content": thinking_text, "interaction_id": interaction_id
                })
  
            self.station._log_dialogue_entry(agent_name, {
                "tick": current_tick, "speaker": "AgentLLM", 
                "type": "manual_llm_response_to_human", 
                "content": llm_response, "interaction_id": interaction_id, "token_info": token_info
            })
            self._push_log_event("human_assist_event", {
                "type": "manual_llm_response_received", 
                "agent_name": agent_name, 
                "tick": current_tick, # Added tick for context
                "text_content": llm_response, 
                "thinking_text": thinking_text, 
                "token_info": token_info
            })

            if end_chat_after_this_message:
                self.resolve_human_intervention(agent_name, "Human marked chat as ended after this manual message.")
            
            return True, {"llm_response": llm_response, "thinking_text": thinking_text, "token_info": token_info}
        except Exception as e:
            msg = f"Error sending manual message to LLM for {agent_name}: {e}"
            self._push_log_event("human_assist_event", {"type": "manual_message_error", "agent_name": agent_name, "error": msg, "trace": traceback.format_exc()})
            self.station._log_dialogue_entry(agent_name, {
                "tick": current_tick, "speaker": "Station",
                "type": "manual_message_llm_error",
                "error": msg, "interaction_id": interaction_id
            })
            return False, {"error": msg}

    def perform_final_chat_with_ended_agent(self, agent_name: str, human_message: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not self.station or not self.station.agent_module:
            return None, None, "Station or agent module not available in Orchestrator."

        agent_data = self.station.agent_module.load_agent_data(agent_name, include_ended=True, include_ascended=True)

        if not agent_data:
            return None, None, f"Agent '{agent_name}' not found."
        if not agent_data.get(constants.AGENT_SESSION_ENDED_KEY, False):
            return None, None, f"Agent '{agent_name}'s session has not officially ended. This function is for ended agents."

        model_provider_class = agent_data.get(constants.AGENT_MODEL_PROVIDER_CLASS_KEY)
        model_name_specific = agent_data.get(constants.AGENT_MODEL_NAME_KEY)

        if not model_provider_class or not model_name_specific:
            return None, None, f"Agent '{agent_name}' is missing LLM configuration for final chat."

        agent_specific_data_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME, agent_name)
        raw_role_definition = self.station.agent_module.get_agent_role_definition(agent_data)
        system_prompt = build_station_level_system_prompt(agent_name, raw_role_definition)
        temperature_str = agent_data.get(constants.AGENT_LLM_TEMPERATURE_KEY, "1.0")
        try: temperature = float(str(temperature_str))
        except ValueError: temperature = 1.0
        max_tokens_str = agent_data.get(constants.AGENT_LLM_MAX_TOKENS_KEY)
        max_tokens = None
        if max_tokens_str is not None:
            try: max_tokens = int(max_tokens_str)
            except ValueError: max_tokens = None
        custom_api_params = agent_data.get(constants.AGENT_LLM_CUSTOM_API_PARAMS_KEY)

        temp_connector = None
        current_tick_for_log = self.station._get_current_tick()

        # Log human's message to the agent's main dialogue log
        self.station._log_dialogue_entry(agent_name, {
            "tick": current_tick_for_log, "speaker": "HumanFinalChat",
            "type": "final_message_to_agent", "content": human_message,
            "note": "Interaction with ended agent session."
        })

        # --- MODIFICATION START: Push SSE event for human's message immediately ---
        self._push_log_event("final_chat_event", {
            "type": "human_message_sent", # Specific type for UI
            "agent_name": agent_name,
            "tick": current_tick_for_log, # Add tick
            "human_message": human_message,
            "status": "pending_agent_response"
        })
        # --- MODIFICATION END ---

        try:
            temp_connector = create_llm_connector(
                model_class_name=model_provider_class,
                model_name=model_name_specific,
                agent_name=agent_name,
                agent_data_path=agent_specific_data_path,
                api_key=None,
                system_prompt=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                custom_api_params=custom_api_params
            )

            if not temp_connector:
                # Log error for SSE if connector creation fails after logging human message
                self._push_log_event("final_chat_event", {
                    "type": "error_event", # General error type
                    "agent_name": agent_name,
                    "error_message": f"Failed to create temporary LLM connector for agent '{agent_name}'.",
                    "status": "error"
                })
                return None, None, f"Failed to create temporary LLM connector for agent '{agent_name}'."

            llm_response, thinking_text, token_info = temp_connector.send_message(human_message, current_tick_for_log)

            if thinking_text: 
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick_for_log, "speaker": f"{agent_name} (FinalThinking)",
                    "type": "thinking_block", "content": thinking_text,
                    "note": "Interaction with ended agent session."
                })

            self.station._log_dialogue_entry(agent_name, {
                "tick": current_tick_for_log, "speaker": f"{agent_name} (FinalResponse)",
                "type": "final_agent_response_to_human", "content": llm_response,
                "token_info": token_info, "note": "Interaction with ended agent session."
            })
            
            self._push_log_event("final_chat_event", {
                "type": "agent_response_received", # Specific type for UI
                "agent_name": agent_name,
                "tick": current_tick_for_log, # Add tick
                # "human_message": human_message, # Already sent in previous event for live display
                "llm_response": llm_response, 
                "thinking_text": thinking_text, 
                "status": "success", 
                "token_info": token_info
            })

            return llm_response, thinking_text, None 

        except LLMTransientAPIError as e:
            error_msg = f"Transient API error during final chat with {agent_name} (after retries): {e}"
            self.station._log_dialogue_entry(agent_name, {"tick": current_tick_for_log, "speaker": "Station", "type": "final_chat_error", "error": error_msg, "note": "Interaction with ended agent session."})
            self._push_log_event("final_chat_event", {"agent_name": agent_name, "type": "error_event", "status": "error", "error_message": error_msg})
            return None, None, error_msg
        except (LLMPermanentAPIError, LLMSafetyBlockError, LLMConnectorError) as e:
            error_msg = f"LLM Connector error during final chat with {agent_name}: {e}"
            self.station._log_dialogue_entry(agent_name, {"tick": current_tick_for_log, "speaker": "Station", "type": "final_chat_error", "error": error_msg, "note": "Interaction with ended agent session."})
            self._push_log_event("final_chat_event", {"agent_name": agent_name, "type": "error_event", "status": "error", "error_message": error_msg})
            return None, None, error_msg
        except Exception as e:
            detailed_error = traceback.format_exc()
            print(f"Orchestrator: Unexpected error during final chat with {agent_name}: {detailed_error}")
            error_msg = f"Unexpected error during final chat with {agent_name}: {str(e)}"
            self.station._log_dialogue_entry(agent_name, {"tick": current_tick_for_log, "speaker": "Station", "type": "final_chat_error", "error": error_msg, "details": detailed_error, "note": "Interaction with ended agent session."})
            self._push_log_event("final_chat_event", {"agent_name": agent_name, "type": "error_event", "status": "error", "error_message": error_msg, "details": detailed_error})
            return None, None, error_msg
        finally:
            if temp_connector:
                if hasattr(temp_connector, 'end_session_and_cleanup'): 
                    temp_connector.end_session_and_cleanup()
                if agent_name in self.agent_llm_connectors and self.agent_llm_connectors[agent_name] == temp_connector:
                    del self.agent_llm_connectors[agent_name]

    def _temporal_chat_timestamp(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _temporal_chat_root_dir(self) -> str:
        return os.path.join(constants.BASE_STATION_DATA_PATH, constants.TEMPORAL_CHAT_DIR_NAME)

    def _temporal_chat_safe_agent_name(self, agent_name: str) -> str:
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("Agent name is required.")
        stripped = agent_name.strip()
        if os.path.isabs(stripped) or os.path.basename(stripped) != stripped:
            raise ValueError("Invalid agent name for temporal chat.")
        altsep = os.path.altsep
        if altsep and altsep in stripped:
            raise ValueError("Invalid agent name for temporal chat.")
        return stripped

    def _temporal_chat_transcript_path(self, agent_name: str) -> str:
        safe_name = self._temporal_chat_safe_agent_name(agent_name)
        return os.path.join(
            self._temporal_chat_root_dir(),
            f"{safe_name}{constants.YAML_EXTENSION}",
        )

    def _temporal_chat_internal_dir(self, agent_name: str) -> str:
        safe_name = self._temporal_chat_safe_agent_name(agent_name)
        return os.path.join(
            self._temporal_chat_root_dir(),
            constants.TEMPORAL_CHAT_INTERNAL_DIR_NAME,
            safe_name,
        )

    def _temporal_chat_internal_history_path(self, agent_name: str) -> str:
        return os.path.join(self._temporal_chat_internal_dir(agent_name), "llm_chat_history.yamll")

    def _temporal_chat_internal_meta_path(self, agent_name: str) -> str:
        return os.path.join(self._temporal_chat_internal_dir(agent_name), "meta.yaml")

    def _load_temporal_chat_record(self, agent_name: str) -> Optional[Dict[str, Any]]:
        path = self._temporal_chat_transcript_path(agent_name)
        record = file_io_utils.load_yaml(path)
        return record if isinstance(record, dict) else None

    def _load_temporal_chat_internal_meta(self, agent_name: str) -> Optional[Dict[str, Any]]:
        path = self._temporal_chat_internal_meta_path(agent_name)
        meta = file_io_utils.load_yaml(path)
        return meta if isinstance(meta, dict) else None

    def _build_temporal_chat_public_state(
        self,
        agent_name: str,
        record: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not record:
            return {
                "exists": False,
                "agent_name": agent_name,
                "base_tick": None,
                "history_start_tick": None,
                "created_at": None,
                "updated_at": None,
                "messages": [],
            }

        messages: List[Dict[str, Any]] = []
        for raw_msg in record.get("messages", []) or []:
            if not isinstance(raw_msg, dict):
                continue
            role = raw_msg.get("role")
            if role == "model":
                role = "assistant"
            if role not in ("user", "assistant"):
                continue
            content = raw_msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            msg = {
                "role": role,
                "content": content,
            }
            thinking_content = raw_msg.get("thinking_content")
            if isinstance(thinking_content, str) and thinking_content.strip():
                msg["thinking_content"] = thinking_content
            if raw_msg.get("tick") is not None:
                msg["tick"] = raw_msg.get("tick")
            if raw_msg.get("created_at") is not None:
                msg["created_at"] = raw_msg.get("created_at")
            messages.append(msg)

        return {
            "exists": True,
            "agent_name": record.get("agent_name") or agent_name,
            "base_tick": record.get("base_tick"),
            "history_start_tick": record.get("history_start_tick"),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "messages": messages,
        }

    def _validate_temporal_chat_agent(self, agent_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not self.station or not self.station.agent_module:
            return None, "Station or agent module not available in Orchestrator."
        try:
            self._temporal_chat_safe_agent_name(agent_name)
        except ValueError as e:
            return None, str(e)
        agent_data = self.station.agent_module.load_agent_data(agent_name, include_ended=True, include_ascended=True)
        if not agent_data:
            return None, f"Agent '{agent_name}' not found."
        if agent_data.get(constants.AGENT_IS_ASCENDED_KEY, False):
            return None, f"Agent '{agent_name}' is ascended; temporal chat is not supported for ascended agents."
        return agent_data, None

    @staticmethod
    def _coerce_temporal_chat_tick(value: Any) -> Optional[int]:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            numeric = text[1:] if text[:1] in ("-", "+") else text
            if not numeric.isdigit():
                return None
            try:
                return int(text)
            except (TypeError, ValueError):
                return None
        return None

    def _resolve_temporal_chat_base_tick(self, requested_base_tick: Any = None) -> Tuple[Optional[int], Optional[str]]:
        current_tick = self._coerce_temporal_chat_tick(self.station._get_current_tick() if self.station else 0)
        if current_tick is None:
            current_tick = 0

        if requested_base_tick is None or (isinstance(requested_base_tick, str) and not requested_base_tick.strip()):
            return current_tick, None

        base_tick = self._coerce_temporal_chat_tick(requested_base_tick)
        if base_tick is None:
            return None, "Invalid branch tick: enter a whole-number station tick, or leave it blank for current."
        if base_tick < 0:
            return None, "Invalid branch tick: tick cannot be negative."
        if base_tick > current_tick:
            return None, f"Invalid branch tick: tick {base_tick} is in the future. Current station tick is {current_tick}."
        return base_tick, None

    def _load_temporal_chat_seed_history(
        self,
        agent_name: str,
        base_tick: int,
        history_start_tick: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        agent_history_file = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.AGENTS_DIR_NAME,
            agent_name,
            "llm_chat_history.yamll",
        )
        if not os.path.exists(agent_history_file):
            return []

        history_entries = file_io_utils.load_yaml_lines(agent_history_file)
        selected_entries: List[Dict[str, Any]] = []
        for entry in history_entries:
            if not isinstance(entry, dict):
                continue
            entry_tick = self._coerce_temporal_chat_tick(entry.get("tick"))
            if entry_tick is None:
                if history_start_tick is None:
                    selected_entries.append(copy.deepcopy(entry))
                continue
            if history_start_tick is not None and entry_tick < history_start_tick:
                continue
            if entry_tick <= base_tick:
                selected_entries.append(copy.deepcopy(entry))
        return selected_entries

    def _build_temporal_chat_model_snapshot(
        self,
        agent_name: str,
        agent_data: Dict[str, Any],
        base_tick: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        model_provider_class = agent_data.get(constants.AGENT_MODEL_PROVIDER_CLASS_KEY)
        model_name_specific = agent_data.get(constants.AGENT_MODEL_NAME_KEY)
        if not model_provider_class or not model_name_specific:
            return None, f"Agent '{agent_name}' is missing LLM configuration for temporal chat."

        # Freeze the actual runtime system prompt, including the Station wrapper.
        role_definition = self.station.agent_module.get_agent_role_definition(agent_data)
        system_prompt = build_station_level_system_prompt(agent_name, role_definition)
        temperature_str = agent_data.get(constants.AGENT_LLM_TEMPERATURE_KEY, "1.0")
        try:
            temperature = float(str(temperature_str))
        except ValueError:
            temperature = 1.0

        max_tokens_str = agent_data.get(constants.AGENT_LLM_MAX_TOKENS_KEY)
        max_tokens = None
        if max_tokens_str is not None:
            try:
                max_tokens = int(max_tokens_str)
            except (TypeError, ValueError):
                max_tokens = None

        custom_api_params = agent_data.get(constants.AGENT_LLM_CUSTOM_API_PARAMS_KEY)
        history_start_tick = self.station.agent_module.latest_context_compaction_anchor_at_or_before(
            agent_data,
            base_tick,
        )
        return {
            "model_provider_class": model_provider_class,
            "model_name": model_name_specific,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "custom_api_params": copy.deepcopy(custom_api_params),
            "system_prompt": system_prompt,
            "history_start_tick": history_start_tick,
        }, None

    def _initialize_temporal_chat_fork(
        self,
        agent_name: str,
        agent_data: Dict[str, Any],
        requested_base_tick: Any = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Freeze an agent context at a branch tick and reset the visible transcript."""
        base_tick, error_msg = self._resolve_temporal_chat_base_tick(requested_base_tick)
        if error_msg:
            return None, error_msg
        if base_tick is None:
            return None, "Invalid branch tick."

        model_snapshot, error_msg = self._build_temporal_chat_model_snapshot(agent_name, agent_data, base_tick)
        if error_msg:
            return None, error_msg

        root_dir = self._temporal_chat_root_dir()
        transcript_path = self._temporal_chat_transcript_path(agent_name)
        internal_dir = self._temporal_chat_internal_dir(agent_name)
        internal_history_path = self._temporal_chat_internal_history_path(agent_name)
        internal_meta_path = self._temporal_chat_internal_meta_path(agent_name)

        try:
            if os.path.isdir(internal_dir):
                shutil.rmtree(internal_dir, ignore_errors=True)
            if os.path.isfile(transcript_path):
                os.remove(transcript_path)
            file_io_utils.ensure_dir_exists(root_dir)
            file_io_utils.ensure_dir_exists(internal_dir)

            history_entries = self._load_temporal_chat_seed_history(
                agent_name,
                base_tick,
                model_snapshot.get("history_start_tick"),
            )
            if history_entries:
                for entry in history_entries:
                    file_io_utils.append_yaml_line(entry, internal_history_path)
            else:
                file_io_utils.save_text("", internal_history_path)

            now = self._temporal_chat_timestamp()
            internal_meta = {
                "schema_version": 2,
                "agent_name": agent_name,
                "base_tick": base_tick,
                "created_at": now,
                "model": model_snapshot,
            }
            file_io_utils.save_yaml(internal_meta, internal_meta_path)

            record = {
                "schema_version": 2,
                "agent_name": agent_name,
                "base_tick": base_tick,
                "history_start_tick": model_snapshot.get("history_start_tick"),
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }
            file_io_utils.save_yaml(record, transcript_path)
            return record, None
        except Exception as e:
            detailed_error = traceback.format_exc()
            print(f"Orchestrator: Failed to initialize temporal chat fork for {agent_name}: {detailed_error}")
            return None, f"Failed to initialize temporal chat for '{agent_name}': {e}"

    def _create_temporal_chat_connector(
        self,
        agent_name: str,
        record: Dict[str, Any],
    ) -> Tuple[Optional[BaseLLMConnector], Optional[str]]:
        internal_history_path = self._temporal_chat_internal_history_path(agent_name)
        if not os.path.isfile(internal_history_path):
            return None, "Temporal chat context is missing. Click Branch to create a new frozen context."

        meta = self._load_temporal_chat_internal_meta(agent_name)
        if not meta:
            return None, "Temporal chat metadata is missing. Click Branch to create a new frozen context."
        model_snapshot = meta.get("model")
        if not isinstance(model_snapshot, dict):
            return None, "Temporal chat metadata is invalid. Click Branch to create a new frozen context."

        model_provider_class = model_snapshot.get("model_provider_class")
        model_name_specific = model_snapshot.get("model_name")
        if not model_provider_class or not model_name_specific:
            return None, "Temporal chat metadata is missing model configuration. Click Branch to create a new frozen context."

        try:
            temperature = float(model_snapshot.get("temperature", 1.0))
        except (TypeError, ValueError):
            temperature = 1.0

        try:
            connector = create_llm_connector(
                model_class_name=model_provider_class,
                model_name=model_name_specific,
                agent_name=agent_name,
                agent_data_path=self._temporal_chat_internal_dir(agent_name),
                api_key=None,
                system_prompt=model_snapshot.get("system_prompt"),
                temperature=temperature,
                max_output_tokens=model_snapshot.get("max_tokens"),
                custom_api_params=model_snapshot.get("custom_api_params"),
            )
        except Exception as e:
            return None, f"Failed to create LLM connector for temporal chat with agent '{agent_name}': {e}"
        if not connector:
            return None, f"Failed to create LLM connector for temporal chat with agent '{agent_name}'."

        connector.persist_to_disk = True
        connector._skip_agent_data_sync = True
        connector.system_prompt = model_snapshot.get("system_prompt")
        connector._last_known_system_prompt = model_snapshot.get("system_prompt")
        connector.agent_prune_blocks = []
        connector._last_known_prune_blocks = []
        connector.context_history_start_tick = model_snapshot.get("history_start_tick")
        connector._last_known_context_history_start_tick = model_snapshot.get("history_start_tick")
        try:
            connector._initialize_chat_session()
        except Exception as e:
            return None, f"Failed to load frozen temporal chat context for '{agent_name}': {e}"
        return connector, None

    def get_temporal_chat_state(self, agent_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Return the persisted visible temporal chat transcript for an agent."""
        _agent_data, error_msg = self._validate_temporal_chat_agent(agent_name)
        if error_msg:
            return None, error_msg
        record = self._load_temporal_chat_record(agent_name)
        return self._build_temporal_chat_public_state(agent_name, record), None

    def refresh_temporal_chat(
        self,
        agent_name: str,
        base_tick: Any = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Discard an agent's temporal fork and freeze a fresh context at a branch tick."""
        with self._temporal_chat_lock:
            agent_data, error_msg = self._validate_temporal_chat_agent(agent_name)
            if error_msg:
                return None, error_msg
            record, error_msg = self._initialize_temporal_chat_fork(agent_name, agent_data, requested_base_tick=base_tick)
            if error_msg:
                return None, error_msg
            return self._build_temporal_chat_public_state(agent_name, record), None

    def perform_temporal_chat_with_agent(
        self,
        agent_name: str,
        user_message: str,
        base_tick: Any = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """Send one message in a persisted temporal chat fork."""
        if not isinstance(user_message, str) or not user_message.strip():
            return None, None, None, "user_message is required and cannot be empty."

        with self._temporal_chat_lock:
            agent_data, error_msg = self._validate_temporal_chat_agent(agent_name)
            if error_msg:
                return None, None, None, error_msg

            record = self._load_temporal_chat_record(agent_name)
            if not record:
                record, error_msg = self._initialize_temporal_chat_fork(
                    agent_name,
                    agent_data,
                    requested_base_tick=base_tick,
                )
                if error_msg:
                    return None, None, None, error_msg

            connector, error_msg = self._create_temporal_chat_connector(agent_name, record)
            if error_msg:
                return None, None, self._build_temporal_chat_public_state(agent_name, record), error_msg

            current_tick = self.station._get_current_tick()
            stripped_message = user_message.strip()
            now = self._temporal_chat_timestamp()
            messages = record.get("messages")
            if not isinstance(messages, list):
                messages = []
            messages.append({
                "role": "user",
                "content": stripped_message,
                "tick": current_tick,
                "created_at": now,
            })
            record["messages"] = messages
            record["updated_at"] = now
            try:
                file_io_utils.save_yaml(record, self._temporal_chat_transcript_path(agent_name))
            except Exception as e:
                return None, None, None, f"Temporal chat user message save failed: {e}"

            try:
                llm_response, thinking_text, _token_info = connector.send_message(stripped_message, current_tick)
            except LLMTransientAPIError as e:
                return None, None, self._build_temporal_chat_public_state(agent_name, record), (
                    f"Transient API error during temporal chat with {agent_name} (after retries): {e}"
                )
            except (LLMPermanentAPIError, LLMSafetyBlockError, LLMConnectorError) as e:
                return None, None, self._build_temporal_chat_public_state(agent_name, record), (
                    f"LLM connector error during temporal chat with {agent_name}: {e}"
                )
            except Exception as e:
                detailed_error = traceback.format_exc()
                print(f"Orchestrator: Unexpected error during temporal chat with {agent_name}: {detailed_error}")
                return None, None, self._build_temporal_chat_public_state(agent_name, record), (
                    f"Unexpected error during temporal chat with {agent_name}: {str(e)}"
                )
            finally:
                try:
                    if connector and hasattr(connector, "end_session_and_cleanup"):
                        connector.end_session_and_cleanup()
                except Exception:
                    pass

            now = self._temporal_chat_timestamp()
            assistant_message = {
                "role": "assistant",
                "content": llm_response or "",
                "tick": current_tick,
                "created_at": now,
            }
            if thinking_text:
                assistant_message["thinking_content"] = thinking_text
            messages.append(assistant_message)
            record["messages"] = messages
            record["updated_at"] = now
            try:
                file_io_utils.save_yaml(record, self._temporal_chat_transcript_path(agent_name))
            except Exception as e:
                return llm_response, thinking_text, None, f"Temporal chat response generated but transcript save failed: {e}"

            return llm_response, thinking_text, self._build_temporal_chat_public_state(agent_name, record), None

    def create_manual_backup(self) -> Tuple[bool, str]:
        """
        Create a manual backup of the station data.
        Can be called at any time, even mid-tick.
        
        Returns:
            Tuple[bool, str]: (success, message or backup_path)
            
        Raises:
            Exception: If backup creation fails (halts orchestrator)
        """
        current_tick = self.station._get_current_tick() if self.station else 0
        backup_path = backup_utils.create_backup(current_tick, "manual", self.station)
        
        self._push_log_event("backup_event", {
            "type": "manual_backup_success",
            "tick": current_tick,
            "backup_path": backup_path,
            "message": f"Manual backup created at tick {current_tick}"
        })
        return True, backup_path

if __name__ == "__main__":
    print("--- Orchestrator Test Script (Full LLM Integration & Persistent History) ---")

    # Ensure base station_data directory and necessary subdirectories exist
    file_io_utils.ensure_dir_exists(constants.BASE_STATION_DATA_PATH) # type: ignore
    agents_base_dir_main = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME)
    file_io_utils.ensure_dir_exists(agents_base_dir_main) # type: ignore
    # ... (ensure other necessary directories: dialogue_logs, rooms subdirs) ...
    file_io_utils.ensure_dir_exists(os.path.join(constants.BASE_STATION_DATA_PATH, constants.DIALOGUE_LOGS_DIR_NAME)) 
    rooms_base_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.ROOMS_DIR_NAME)
    file_io_utils.ensure_dir_exists(rooms_base_dir) 
    for room_subdir_const in [constants.ADMIN_COUNTER_SUBDIR_NAME, constants.MISC_ROOM_SUBDIR_NAME,
                              constants.COMMON_ROOM_SUBDIR_NAME]:
        file_io_utils.ensure_dir_exists(os.path.join(rooms_base_dir, room_subdir_const))


    station_instance: Optional[Station] = None
    try:
        runtime_api_config.validate_provider_backup_env_config()
        station_instance = Station()
    except Exception as e:
        print(f"CRITICAL: Failed to initialize Station: {e}"); traceback.print_exc(); exit(1)

    # Initialize orchestrator but don't auto-init connectors yet, __main__ will drive it.
    orchestrator = Orchestrator(station_instance, auto_initialize_connectors=False) 

    print("\n--- Configuring and Adding Test Agents for API Mode ---")
    if not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY not set. LLM calls will fail if not using simulated responses.")
    
    # Define configurations for agents you want to test
    # The keys here ("AgentConfig1", "AgentConfig2") are just for iterating the configs.
    # The actual agent names will be generated by create_agent for guests, or specified for recursive.
    agents_to_create_and_configure = {
        "ConfigForGuest1": { # Conceptual name for this configuration block
            constants.AGENT_TYPE_KEY: constants.AGENT_STATUS_GUEST, # Use your defined constant
            # For guests, agent_name_override passed to dynamic_add_agent_to_station will be ignored for naming,
            # but useful if you want to try to load an existing agent with that specific name first.
            # If creating, create_agent will generate "Guest_X".
            "desired_agent_name_if_recursive_or_check": "LiveOrchGuest1", # Name to use if recursive / or for logging
            constants.AGENT_MODEL_PROVIDER_CLASS_KEY: "Gemini",
            constants.AGENT_MODEL_NAME_KEY: "gemini-1.5-flash-latest",
            constants.AGENT_LLM_TEMPERATURE_KEY: 1.0, # Using 1.0 as per your setting
            constants.AGENT_LLM_MAX_TOKENS_KEY: 512,
            constants.AGENT_ROLE_DEFINITION_KEY: "You are an inquisitive Guest agent in the Station. Your first goal is to read the Codex preface."
        },
        "ConfigForGuest2": {
            constants.AGENT_TYPE_KEY: constants.AGENT_STATUS_GUEST,
            "desired_agent_name_if_recursive_or_check": "LiveOrchGuest2",
            constants.AGENT_MODEL_PROVIDER_CLASS_KEY: "Gemini",
            constants.AGENT_MODEL_NAME_KEY: "gemini-1.5-flash-latest",
            constants.AGENT_LLM_TEMPERATURE_KEY: 1.0,
            constants.AGENT_LLM_MAX_TOKENS_KEY: 512,
            constants.AGENT_ROLE_DEFINITION_KEY: "You are another Guest agent. After the first agent acts, go to the public memory and read capsule 1."
        }
    }

    for config_id, config_data in agents_to_create_and_configure.items():
        print(f"\nProcessing configuration: {config_id}")
        
        agent_type_to_create = config_data[constants.AGENT_TYPE_KEY]
        # For guests, create_agent will generate the name. We don't pass a name override for creation.
        # For recursive, we might pass the desired name from config.
        agent_name_override_for_creation = None
        if agent_type_to_create == constants.AGENT_STATUS_RECURSIVE:
            agent_name_override_for_creation = config_data.get("desired_agent_name_if_recursive_or_check")
            # Ensure lineage and generation are provided in config_data if recursive and auto-naming
            if not agent_name_override_for_creation:
                if not config_data.get(constants.AGENT_LINEAGE_KEY) or config_data.get(constants.AGENT_GENERATION_KEY) is None:
                    print(f"Skipping {config_id}: Recursive agent needs name_override or lineage+generation.")
                    continue
        
        # Orchestrator's dynamic_add_agent_to_station handles creation if needed,
        # then LLM config update, then connector initialization.
        # It's important that dynamic_add_agent_to_station uses the *actual name*
        # that results from station.create_agent (especially for guests).
        
        # The dynamic_add_agent_to_station method now internally handles calling station.create_agent
        # and then configuring the LLM details on the *actually created/loaded* agent data.
        success, message = orchestrator.dynamic_add_agent_to_station(
            agent_type=agent_type_to_create,
            model_provider_class=config_data[constants.AGENT_MODEL_PROVIDER_CLASS_KEY],
            model_name=config_data[constants.AGENT_MODEL_NAME_KEY],
            agent_name_override=agent_name_override_for_creation, # For recursive, or None for guests
            lineage=config_data.get(constants.AGENT_LINEAGE_KEY), # For recursive
            generation=config_data.get(constants.AGENT_GENERATION_KEY), # For recursive
            initial_tokens_max=config_data.get(constants.AGENT_TOKEN_BUDGET_MAX_KEY), # Use the right constant
            internal_note=config_data.get(constants.AGENT_INTERNAL_NOTE_KEY),
            role_definition=(
                config_data.get(constants.AGENT_ROLE_DEFINITION_KEY)
                if config_data.get(constants.AGENT_ROLE_DEFINITION_KEY) is not None
                else config_data.get(constants.LEGACY_AGENT_LLM_SYSTEM_PROMPT_KEY)
            ),
            llm_temperature=config_data.get(constants.AGENT_LLM_TEMPERATURE_KEY),
            llm_max_tokens=config_data.get(constants.AGENT_LLM_MAX_TOKENS_KEY)
        )
        
        if success:
            print(f"Orchestrator processing for {config_id} (agent likely {message.split(' ')[1]} or similar): {message}")
        else:
            print(f"Failed orchestrator processing for {config_id}: {message}")

    orchestrator._load_agent_turn_order() # Ensure orchestrator's list is fresh from config

    if not orchestrator.agent_llm_connectors:
        print("Orchestrator: No LLM connectors were initialized. Check agent configurations in YAML files and API key (GOOGLE_API_KEY).")
        # Optionally exit if critical: exit(1) 
    if not orchestrator.agent_turn_order:
        print("Orchestrator: No agents in turn order after setup. Exiting test.")
    else:
        print("\n--- Starting Orchestrator with Real LLM Calls ---")
        orchestrator.start_orchestration()
        num_ticks_to_simulate = 2 
        print(f"\nOrchestrator running. Simulating for {num_ticks_to_simulate} ticks...")
        # ... (rest of the __main__ loop for simulation and shutdown as before) ...
        try:
            for i in range(num_ticks_to_simulate):
                num_agents_in_loop = len(orchestrator.agent_turn_order) if orchestrator.agent_turn_order else 1
                wait_time = (num_agents_in_loop * 15) + 5 # Increased wait time for real LLM calls
                current_sim_tick = station_instance._get_current_tick() if station_instance else -1
                print(f"[Test Main Thread] Waiting ~{wait_time}s for Orchestrator Tick {current_sim_tick} to complete...")
                
                start_wait_time = time.time()
                while time.time() - start_wait_time < wait_time:
                    if not orchestrator.is_running: break
                    if orchestrator.is_paused and orchestrator.pause_condition_met:
                        # ... (handling for auto-pause if you implement that test) ...
                        break 
                    time.sleep(1)

                if not orchestrator.is_running:
                    print("[Test Main Thread] Orchestrator stopped during tick simulation.")
                    break
                print(f"[Test Main Thread] Tick {current_sim_tick} processing period ended.")
        except KeyboardInterrupt:
            print("\n[Test Main Thread] KeyboardInterrupt detected.")
        finally:
            print("\n[Test Main Thread] Requesting orchestrator stop...")
            orchestrator.stop_orchestration()
            print("--- Orchestrator Test Script Finished ---")
