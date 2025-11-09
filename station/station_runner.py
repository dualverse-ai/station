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
from contextlib import contextmanager
from queue import Queue, Empty as QueueEmpty
from typing import Dict, List, Optional, Any, Tuple

from station.station import Station
from station import constants
from station.constants import _load_config_overrides
from station import file_io_utils
from station import backup_utils
from station.llm_connectors import (
    BaseLLMConnector, create_llm_connector,
    LLMTransientAPIError, LLMPermanentAPIError, LLMSafetyBlockError, LLMConnectorError, LLMContextOverflowError
)
from station.base_room import InternalActionHandler, LoggingInternalActionHandlerWrapper


class Orchestrator:
    def __init__(self,
                 station_instance: Station,
                 auto_prepare_on_init: bool = True, # MODIFIED: Renamed and default to True
                 log_event_queue: Optional[Queue] = None):
        # Load configuration overrides with verbose output at station initialization
        _load_config_overrides(verbose=True)

        self.station = station_instance
        self.is_running: bool = False # True when the main_loop thread is active and processing
        self.is_prepared: bool = False # True when agent turn order loaded and connectors initialized
        self.orchestrator_thread: Optional[threading.Thread] = None
        self.agent_turn_order: List[str] = []
        self.current_tick_processed_agents: set[str] = set()
        self.agent_llm_connectors: Dict[str, BaseLLMConnector] = {}

        self.is_paused: bool = False
        self.pause_requested: bool = False
        self.pause_condition_met: bool = False
        self.pause_reason_message: str = ""
        self.pause_event = threading.Event()

        # Waiting state - automatically resumes when conditions resolve
        self.is_waiting: bool = False
        self.waiting_reasons: Dict[str, str] = {}
        self.wait_check_interval: float = 2.0  # seconds between auto-resume checks

        self.current_agent_index_in_turn_order: int = 0

        self.log_event_queue = log_event_queue

        if auto_prepare_on_init:
            self.prepare_for_run()

        self._push_log_event("orchestrator_status", {"status": "initialized", "prepared": self.is_prepared, "message": "Orchestrator instance created."})

        # Restart stuck research evaluations before starting auto evaluator
        # This ensures any evaluations with unsent notifications from previous runs are requeued
        if constants.AUTO_EVAL_RESEARCH and constants.RESEARCH_COUNTER_ENABLED:
            try:
                from station.eval_research import restart_stuck_evaluations
                count = restart_stuck_evaluations()
                if count > 0:
                    print(f"Orchestrator: Restarted {count} stuck research evaluation(s)")
                    self._push_log_event("orchestrator_info", {"message": f"Restarted {count} stuck research evaluation(s)"})
            except Exception as e:
                print(f"Orchestrator: Error restarting stuck evaluations: {e}")

        # Start auto evaluator if enabled
        if constants.AUTO_EVAL_TEST:
            self.station.start_auto_evaluator(log_queue=self.log_event_queue)

        # Start auto research evaluator if enabled and Research Counter room is enabled
        if constants.AUTO_EVAL_RESEARCH and constants.RESEARCH_COUNTER_ENABLED:
            self.station.start_auto_research_evaluator(log_queue=self.log_event_queue)

        # Initialize stagnation protocol if enabled
        if constants.STAGNATION_ENABLED and constants.RESEARCH_COUNTER_ENABLED:
            self.station.init_stagnation_protocol()

        # Start auto archive evaluator if enabled
        if getattr(constants, 'EVAL_ARCHIVE_MODE', 'none') == 'auto':
            self.station.start_auto_archive_evaluator(log_queue=self.log_event_queue)

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

    def _auto_start_with_wait(self):
        """
        AUTO_START helper: Waits for pending evaluations to complete, then starts processing.
        Runs in a separate thread to avoid blocking frontend initialization.
        """
        print("Orchestrator: AUTO_START waiting for pending evaluations...")

        # Wait for any running/pending research evaluations to complete
        if constants.AUTO_EVAL_RESEARCH and constants.RESEARCH_COUNTER_ENABLED:
            while True:
                if not self.station.has_pending_research_evaluations():
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
        if config_turn_order:
            for agent_name_in_config in config_turn_order:
                # Check if agent file actually exists AND is loadable as an active agent
                # load_agent_data (default) returns None for ascended/ended agents.
                # If an agent ascended, their old guest identity won't load here.
                if self.station.agent_module.load_agent_data(agent_name_in_config):
                    verified_runtime_order.append(agent_name_in_config)
                else:
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

        if agents_removed:
            # Analyze departure reasons and handle based on user requirements
            ascended_agents = []
            other_departed_agents = []
            respawned_agents = []
            
            for agent_name in agents_removed:
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
                    "departed_agents": list(agents_removed),
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
                    "departed_agents": list(agents_removed),
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
            print(f"Orchestrator: Agent '{added_agent_name}' detected as new to runtime order. Initializing connector...")
            self._push_log_event("orchestrator_info", {"message": f"Agent {added_agent_name} new to runtime order. Initializing connector."})
            # --- MODIFICATION: Immediately initialize connector for newly added agents ---
            if not self.initialize_connector_for_agent(added_agent_name, force_reinitialize=True): # Force reinitialize if it somehow existed but wasn't active
                print(f"Orchestrator: CRITICAL - Failed to initialize connector for newly added agent {added_agent_name}. This agent may not function correctly.")
                self._push_log_event("connector_error", {"agent_name": added_agent_name, "message": "Failed to initialize connector upon being added to turn order."})
            # --- END OF MODIFICATION ---


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
        # The call to _get_llm_response happens before current_agent_index_in_turn_order is incremented for the turn.
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
    
    def initialize_connectors_for_active_agents(self) -> bool:
        # This method now iterates self.agent_turn_order which should be up-to-date
        # from a preceding _load_agent_turn_order() call if called by prepare_for_run.
        self._push_log_event("orchestrator_info", {"message": "Initializing/Verifying LLM connectors for current turn order..."})

        if not self.agent_turn_order:
             self._push_log_event("orchestrator_info", {"message": "No agents in current turn order to initialize connectors for."})
             return True 

        all_successful = True
        newly_initialized_count = 0
        for agent_name in self.agent_turn_order:
            was_present_before = agent_name in self.agent_llm_connectors
            # Force reinitialize if it wasn't present, to ensure it gets set up.
            # If it was present, initialize_connector_for_agent won't re-init unless force_reinitialize=True.
            if not self.initialize_connector_for_agent(agent_name, force_reinitialize=not was_present_before):
                all_successful = False
            elif not was_present_before and agent_name in self.agent_llm_connectors:
                newly_initialized_count +=1
        
        msg_end = f"Connector initialization/verification complete. {newly_initialized_count} new/reinitialized. All successful: {all_successful}."
        self._push_log_event("orchestrator_info", {"message": msg_end})
        # self.is_prepared should be set based on this in prepare_for_run
        return all_successful

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
        system_prompt = agent_data.get(constants.AGENT_LLM_SYSTEM_PROMPT_KEY, f"You are Agent {agent_name}.")
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

    def _get_llm_response(self, agent_name: str, observation: str, current_tick: int) -> Tuple[Optional[str], bool]:
        connector = self.agent_llm_connectors.get(agent_name)
        if not connector:
            err_msg = f"SYSTEM_ERROR: No LLM connector for {agent_name}."
            self._push_log_event("llm_event_error", {"agent_name": agent_name, "tick": current_tick, "error": err_msg})
            # --- MODIFICATION START: Log error to main dialogue log ---
            self.station._log_dialogue_entry(agent_name, {
                "tick": current_tick,
                "speaker": "Station",
                "type": "llm_connector_error",
                "error": err_msg
            })
            # --- MODIFICATION END ---
            return None, False 

        obs_snippet = observation.replace('\n', ' ')[:200] + "..."
        self._push_log_event("llm_event", {"agent_name": agent_name, "tick": current_tick, "direction": "to_llm", "type": "observation", "text_content": observation, "full_length": len(observation)})
        
        with self._agent_response_context(agent_name):
            try:
                # --- MODIFICATION START: Unpack thinking_text ---
                response_text, thinking_text, token_info = connector.send_message(observation, current_tick)
                # --- MODIFICATION END ---

                # --- MODIFICATION START: Log thinking_text to main dialogue log if present ---
                if thinking_text:
                    self.station._log_dialogue_entry(agent_name, {
                        "tick": current_tick,
                        "speaker": "AgentLLM", # Or a more specific speaker like "AgentLLMThinking"
                        "type": "thinking_block",
                        "agent_name": agent_name, # Ensure agent_name is part of the log data
                        "content": thinking_text 
                    })
                # --- MODIFICATION END ---
                
                resp_snippet = response_text.replace('\n', ' ')[:150] + "..."
                self._push_log_event("llm_event", {
                    "agent_name": agent_name, "tick": current_tick, "direction": "from_llm", "type": "response", 
                    "text_content": response_text, 
                    "thinking_text": thinking_text, # MODIFIED: Include thinking_text in SSE event
                    "full_length": len(response_text), 
                    "token_info": token_info
                })
                total_tokens_in_session = token_info.get('total_tokens_in_session')
                if total_tokens_in_session is not None:
                    can_continue = self.station.update_agent_token_budget(agent_name, total_tokens_in_session)
                    if not can_continue: 
                        self._push_log_event("agent_event", {"type": "session_ended_tokens", "agent_name": agent_name, "tick": current_tick, "reason": "Token budget exhausted."})
                        # --- MODIFICATION START: Log session end to main dialogue log ---
                        self.station._log_dialogue_entry(agent_name, {
                            "tick": current_tick,
                            "speaker": "Station",
                            "type": "session_end_tokens",
                            "reason": "Token budget exhausted."
                        })
                        # --- MODIFICATION END ---
                        return response_text, False # response_text might still be useful for a final display
                    else: 
                        adata = self.station.agent_module.load_agent_data(agent_name); 
                        cur = adata.get(constants.AGENT_TOKEN_BUDGET_CURRENT_KEY) if adata else "N/A"; 
                        mx = adata.get(constants.AGENT_TOKEN_BUDGET_MAX_KEY) if adata else "N/A"
                        self._push_log_event("agent_event", {"type": "token_budget_updated", "agent_name": agent_name, "tick": current_tick, "current_used": cur, "max_budget":mx})
                return response_text, True

            except LLMTransientAPIError as e:
                err_msg = f"Transient API Error for {agent_name}: {e}. Orchestrator will pause."
                print(f"Orchestrator: {err_msg}")
                self._push_log_event("llm_event_error", {"agent_name": agent_name, "tick": current_tick, "error_type": "LLMTransientAPIError", "error": str(e), "original_exception": str(e.original_exception)})
                # --- MODIFICATION START: Log error to main dialogue log ---
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick, "speaker": "Station", "type": "llm_api_error_transient",
                    "error": str(e), "original_exception": str(e.original_exception)
                })
                # --- MODIFICATION END ---
                self._trigger_pause_due_to_llm_error(agent_name, e, "LLM_TRANSIENT_API_ERROR")
                return None, True 
            
            except LLMSafetyBlockError as e:
                err_msg = f"LLM Safety Block for {agent_name}: {e}. Block Reason: {e.block_reason}."
                print(f"Orchestrator: {err_msg}")
                self._push_log_event("llm_event_error", {"agent_name": agent_name, "tick": current_tick, "error_type": "LLMSafetyBlockError", "error": str(e), "block_reason": e.block_reason, "prompt_feedback": str(e.prompt_feedback)})
                # --- MODIFICATION START: Log safety block to main dialogue log ---
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick, "speaker": "Station", "type": "llm_safety_block",
                    "error": str(e), "block_reason": str(e.block_reason), "prompt_feedback": str(e.prompt_feedback)
                })
                # --- MODIFICATION END ---
                # Pause orchestrator for safety blocks to allow human intervention
                self._trigger_pause_due_to_llm_error(agent_name, e, "LLM_SAFETY_BLOCK")
                return f"SYSTEM_ERROR: LLM response blocked by safety filters. Orchestrator paused for human intervention.", True

            except LLMContextOverflowError as e:
                err_msg = f"Context window overflow for {agent_name}: {e}. Agent session terminated."
                print(f"Orchestrator: {err_msg}")
                self._push_log_event("agent_event", {"type": "session_ended_context_overflow", "agent_name": agent_name, "tick": current_tick, "reason": "Context window overflow."})
                # --- MODIFICATION START: Log session end to main dialogue log ---
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick,
                    "speaker": "Station",
                    "type": "session_end_context_overflow",
                    "reason": "Context window overflow.",
                    "error": str(e)
                })
                # --- MODIFICATION END ---
                # Context overflow should terminate the agent session with proper broadcast
                critical_notification = f"CRITICAL: Your input exceeded the model's context window. Your session is being terminated."
                self.station._terminate_agent_session_with_broadcast(agent_name, "context window overflow", critical_notification)
                return f"CRITICAL: Context window overflow for {agent_name}. Session terminated.", False

            except LLMPermanentAPIError as e:
                err_msg = f"Permanent API Error for {agent_name}: {e}. This agent's LLM connector may be misconfigured or disabled."
                print(f"Orchestrator: {err_msg}")
                self._push_log_event("llm_event_error", {"agent_name": agent_name, "tick": current_tick, "error_type": "LLMPermanentAPIError", "error": str(e), "original_exception": str(e.original_exception)})
                # --- MODIFICATION START: Log error to main dialogue log ---
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick, "speaker": "Station", "type": "llm_api_error_permanent",
                    "error": str(e), "original_exception": str(e.original_exception)
                })
                # --- MODIFICATION END ---
                # Pause orchestrator for permanent API errors to allow human intervention
                self._trigger_pause_due_to_llm_error(agent_name, e, "LLM_PERMANENT_API_ERROR")
                return f"SYSTEM_ERROR: Permanent LLM API Error for {agent_name}. Orchestrator paused for human intervention.", True
            
            except Exception as e: 
                err_msg = f"Unexpected error getting LLM response for {agent_name}: {e}"
                print(f"Orchestrator: {err_msg}")
                self._push_log_event("llm_event_error", {"agent_name": agent_name, "tick": current_tick, "error_type": "UnknownConnectorError", "error": err_msg, "trace": traceback.format_exc()})
                # --- MODIFICATION START: Log error to main dialogue log ---
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick, "speaker": "Station", "type": "llm_connector_error_unknown",
                    "error": err_msg, "trace": traceback.format_exc()
                })
                # --- MODIFICATION END ---
                self._trigger_pause_due_to_llm_error(agent_name, e, "LLM_CONNECTOR_ERROR")
                return f"SYSTEM_ERROR: Unexpected LLM Connector Error for {agent_name}.", True

    def _handle_real_internal_action_loop(self, 
                                          agent_name: str, 
                                          handler_wrapper: LoggingInternalActionHandlerWrapper, 
                                          initial_prompt: str, 
                                          current_tick: int):
        self._push_log_event("internal_action_event", {
            "agent_name": agent_name, "tick": current_tick, "status": "start", 
            "handler": type(handler_wrapper.actual_handler).__name__, 
            "text_content": initial_prompt 
        })
        
        connector = self.agent_llm_connectors.get(agent_name)
        if not connector:
            err_msg = f"LLM connector missing for agent {agent_name}. Internal action cannot proceed."
            self._push_log_event("internal_action_event", {"agent_name": agent_name, "tick": current_tick, "status": "error", "message": err_msg})
            # --- MODIFICATION START: Log error to main dialogue log ---
            self.station._log_dialogue_entry(agent_name, {
                "tick": current_tick, "internal_step": handler_wrapper.internal_step_count,
                "speaker": "Station", "type": "internal_action_llm_connector_error", 
                "handler": type(handler_wrapper.actual_handler).__name__, "error": err_msg
            })
            # --- MODIFICATION END ---
            return

        current_internal_prompt = initial_prompt
        max_internal_steps = 50 
        loop_step_count = 0 

        while current_internal_prompt and loop_step_count < max_internal_steps and self.is_running:
            loop_step_count += 1
            
            self._push_log_event("llm_event", {
                "agent_name": agent_name, "tick": current_tick, "internal_loop_step": loop_step_count, 
                "direction": "to_llm", "type": "internal_prompt", 
                "text_content": current_internal_prompt 
            })
            
            try:
                # --- MODIFICATION START: Unpack internal_thinking_text ---
                llm_internal_response_text, internal_thinking_text, token_info = connector.send_message(current_internal_prompt, current_tick)
                # --- MODIFICATION END ---
            except LLMTransientAPIError as e:
                err_msg = f"Transient API Error during internal action for {agent_name}: {e}. Pausing orchestrator."
                print(f"Orchestrator: {err_msg}")
                self._push_log_event("llm_event_error", {"agent_name": agent_name, "tick": current_tick, "internal_loop_step": loop_step_count, "error_type": "LLMTransientAPIError_Internal", "error": str(e)})
                # --- MODIFICATION START: Log error to main dialogue log ---
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick, "internal_step": handler_wrapper.internal_step_count,
                    "speaker": "Station", "type": "internal_action_api_error_transient",
                    "handler": type(handler_wrapper.actual_handler).__name__, "error": str(e)
                })
                # --- MODIFICATION END ---
                self._trigger_pause_due_to_llm_error(agent_name, e, "LLM_TRANSIENT_API_ERROR_INTERNAL")
                return 
            except LLMSafetyBlockError as e:
                err_msg = f"LLM Safety Block during internal action for {agent_name}: {e}."
                print(f"Orchestrator: {err_msg}")
                self._push_log_event("llm_event_error", {"agent_name": agent_name, "tick": current_tick, "internal_loop_step": loop_step_count, "error_type": "LLMSafetyBlockError_Internal", "error": str(e)})
                # --- MODIFICATION START: Log safety block to main dialogue log ---
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick, "internal_step": handler_wrapper.internal_step_count,
                    "speaker": "Station", "type": "internal_action_llm_safety_block",
                    "handler": type(handler_wrapper.actual_handler).__name__, "error": str(e),
                    "block_reason": str(e.block_reason), "prompt_feedback": str(e.prompt_feedback)
                })
                # --- MODIFICATION END ---
                llm_internal_response_text = f"SYSTEM_ERROR: LLM response blocked by safety filters. Reason: {e.block_reason}."
                internal_thinking_text = None # No thinking if blocked
            except LLMContextOverflowError as e: 
                err_msg = f"Context window overflow during internal action for {agent_name}: {e}. Agent session terminated."
                print(f"Orchestrator: {err_msg}")
                self._push_log_event("agent_event", {"type": "session_ended_context_overflow_internal", "agent_name": agent_name, "tick": current_tick, "reason": "Context window overflow during internal action."})
                # --- MODIFICATION START: Log session end to main dialogue log ---
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick, "internal_step": handler_wrapper.internal_step_count,
                    "speaker": "Station", "type": "session_end_context_overflow_internal",
                    "handler": type(handler_wrapper.actual_handler).__name__, "reason": "Context window overflow during internal action.", "error": str(e)
                })
                # --- MODIFICATION END ---
                # Context overflow during internal action should terminate the agent session with proper broadcast
                critical_notification = f"CRITICAL: Your input exceeded the model's context window during an internal action. Your session is being terminated."
                self.station._terminate_agent_session_with_broadcast(agent_name, "context window overflow", critical_notification)
                return 
            except (LLMPermanentAPIError, LLMConnectorError) as e: 
                err_msg = f"Permanent/Connector Error during internal action for {agent_name}: {e}. Aborting internal action."
                print(f"Orchestrator: {err_msg}")
                self._push_log_event("llm_event_error", {"agent_name": agent_name, "tick": current_tick, "internal_loop_step": loop_step_count, "error_type": type(e).__name__ + "_Internal", "error": str(e)})
                # --- MODIFICATION START: Log error to main dialogue log ---
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick, "internal_step": handler_wrapper.internal_step_count,
                    "speaker": "Station", "type": "internal_action_api_error_permanent",
                    "handler": type(handler_wrapper.actual_handler).__name__, "error": str(e)
                })
                # --- MODIFICATION END ---
                self._trigger_pause_due_to_llm_error(agent_name, e, "LLM_CONNECTOR_ERROR_INTERNAL")
                handler_wrapper.step(f"SYSTEM_ERROR: LLM connection failed permanently: {e}") 
                return 

            # --- MODIFICATION START: Log internal_thinking_text to main dialogue log if present ---
            if internal_thinking_text:
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick,
                    "internal_step": handler_wrapper.internal_step_count, # Add step for context
                    "speaker": "AgentLLM", # Or "AgentLLMThinkingInternal"
                    "type": "thinking_block_internal",
                    "agent_name": agent_name,
                    "handler": type(handler_wrapper.actual_handler).__name__,
                    "content": internal_thinking_text
                })
            # --- MODIFICATION END ---

            total_tokens_in_session = token_info.get('total_tokens_in_session')
            can_continue_session = True
            if total_tokens_in_session is not None:
                can_continue_session = self.station.update_agent_token_budget(agent_name, total_tokens_in_session)
                if not can_continue_session:
                    self._push_log_event("agent_event", {"type": "session_ended_tokens_internal", "agent_name": agent_name, "tick": current_tick, "reason": "Token budget exhausted during internal action."})
                    # --- MODIFICATION START: Log session end to main dialogue log ---
                    self.station._log_dialogue_entry(agent_name, {
                        "tick": current_tick, "internal_step": handler_wrapper.internal_step_count,
                        "speaker": "Station", "type": "internal_action_session_end_tokens",
                        "handler": type(handler_wrapper.actual_handler).__name__, "reason": "Token budget exhausted."
                    })
                    # --- MODIFICATION END ---
                    handler_wrapper.step(f"SYSTEM_NOTE: Token budget exhausted. Internal action terminated for {agent_name}.") 
                    break 
            
            self._push_log_event("llm_event", {
                "agent_name": agent_name, "tick": current_tick, "internal_loop_step": loop_step_count, 
                "direction": "from_llm", "type": "internal_response", 
                "text_content": llm_internal_response_text,
                "thinking_text": internal_thinking_text, # MODIFIED: Include thinking_text in SSE
                "token_info": token_info
            })
            
            next_prompt, executed_strings_in_step = handler_wrapper.step(llm_internal_response_text)
            # The LoggingInternalActionHandlerWrapper already logs the "agent_response" (llm_internal_response_text)
            # and the "next_internal_prompt" or "internal_completion" to the main dialogue log.
            # We have added separate logging for the thinking_block above.
            
            delta_updates = handler_wrapper.get_delta_updates()
            if delta_updates:
                if self.station.update_specific_agent_fields(agent_name, delta_updates):
                    self._push_log_event("internal_action_event", {"agent_name": agent_name, "tick": current_tick, "status": "delta_applied", "step": handler_wrapper.internal_step_count, "updates": list(delta_updates.keys())})
            
            if executed_strings_in_step:
                 self._push_log_event("internal_action_event", {"agent_name": agent_name, "tick": current_tick, "status": "step_executed_strings", "step": handler_wrapper.internal_step_count, "log": executed_strings_in_step})

            if next_prompt is None:
                self._push_log_event("internal_action_event", {"agent_name": agent_name, "tick": current_tick, "status": "end", "handler": type(handler_wrapper.actual_handler).__name__, "final_log": executed_strings_in_step})
                break 
            current_internal_prompt = next_prompt
        
        if loop_step_count >= max_internal_steps:
            self._push_log_event("internal_action_event", {"agent_name": agent_name, "tick": current_tick, "status": "max_steps_reached", "handler": type(handler_wrapper.actual_handler).__name__})
            # --- MODIFICATION START: Log max steps reached to main dialogue log ---
            self.station._log_dialogue_entry(agent_name, {
                "tick": current_tick, "internal_step": handler_wrapper.internal_step_count,
                "speaker": "Station", "type": "internal_action_max_steps",
                "handler": type(handler_wrapper.actual_handler).__name__
            })
            # --- MODIFICATION END ---

    def _check_automatic_wait_conditions(self) -> Tuple[bool, Dict[str, str]]:
        """Check for conditions that should trigger waiting state (auto-resumes when resolved)"""
        waiting_reasons = {}
        
        if hasattr(self.station, 'has_pending_test_evaluations') and self.station.has_pending_test_evaluations():
            waiting_reasons['pending_tests'] = "Pending test evaluations"
        
        # Note: We do NOT wait for research evaluations here anymore.
        # Research evaluations run in parallel and only cause waiting at tick boundaries
        # via should_wait_for_research_evaluations_at_tick_boundary()
        
        if hasattr(self.station, 'has_pending_archive_evaluations') and self.station.has_pending_archive_evaluations():
            waiting_reasons['pending_archives'] = "Pending archive evaluations"
        
        # Check for active Claude Code debugging sessions
        # if constants.CLAUDE_CODE_DEBUG_ENABLED and hasattr(self.station, 'has_pending_claude_code_sessions') and self.station.has_pending_claude_code_sessions():
        #    waiting_reasons['claude_code_debug'] = "Active Claude Code debugging sessions"
        
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
        if 'pending_tests' in self.waiting_reasons:
            if hasattr(self.station, 'has_pending_test_evaluations') and self.station.has_pending_test_evaluations():
                return False  # Still pending

        # Note: 'pending_research' is never added to waiting_reasons (research evals run in parallel)
        # Research waiting happens via should_wait_for_research_evaluations_at_tick_boundary() instead

        if 'pending_archives' in self.waiting_reasons:
            if hasattr(self.station, 'has_pending_archive_evaluations') and self.station.has_pending_archive_evaluations():
                return False  # Still pending
        
        if 'claude_code_debug' in self.waiting_reasons:
            if constants.CLAUDE_CODE_DEBUG_ENABLED and hasattr(self.station, 'has_pending_claude_code_sessions') and self.station.has_pending_claude_code_sessions():
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

    def _refresh_connector_and_update_tokens_after_turn(self, agent_name: str, current_tick: int):
        """
        Called after an agent's turn processing (submit_response + internal_actions) completes.
        Forces the connector to refresh based on latest pruning info and updates the
        agent's token budget with the recalculated session token count.
        """
        connector = self.agent_llm_connectors.get(agent_name)
        if not connector:
            self._push_log_event("orchestrator_warning", {
                "message": f"No connector found for agent {agent_name} during post-turn token refresh.",
                "agent_name": agent_name, "tick": current_tick
            })
            return

        try:
            print(f"Orchestrator ({agent_name}, Tick {current_tick}): Forcing connector history refresh and token recalculation post-turn.")
            new_token_count = connector.force_refresh_and_get_current_session_tokens()

            if new_token_count is not None:
                agent_data_for_token_update = self.station.agent_module.load_agent_data(agent_name)
                if agent_data_for_token_update:
                    old_token_count = agent_data_for_token_update.get(constants.AGENT_TOKEN_BUDGET_CURRENT_KEY)

                    # Use update_agent_token_budget() to ensure termination check is performed
                    can_continue = self.station.update_agent_token_budget(agent_name, new_token_count)

                    if can_continue:
                        self._push_log_event("agent_event", {
                            "type": "token_budget_recalculated_post_turn",
                            "agent_name": agent_name,
                            "tick": current_tick,
                            "old_token_count": old_token_count,
                            "new_token_count": new_token_count,
                            "reason": "Post-turn refresh, possibly due to pruning."
                        })
                        print(f"Orchestrator ({agent_name}, Tick {current_tick}): Token budget updated to {new_token_count} after post-turn refresh.")
                    else:
                        # Agent was terminated due to budget exhaustion during post-turn refresh
                        self._push_log_event("agent_event", {
                            "type": "session_ended_tokens_post_refresh",
                            "agent_name": agent_name,
                            "tick": current_tick,
                            "old_token_count": old_token_count,
                            "new_token_count": new_token_count,
                            "reason": "Token budget exhausted during post-turn token recalculation."
                        })
                        print(f"Orchestrator ({agent_name}, Tick {current_tick}): Agent terminated due to token budget exhaustion during post-turn refresh ({new_token_count} tokens).")
                else:
                    self._push_log_event("orchestrator_warning", {
                        "message": f"Could not load agent data for {agent_name} to save recalculated token count.",
                        "agent_name": agent_name, "tick": current_tick
                    })
            else:
                self._push_log_event("orchestrator_warning", {
                    "message": f"Connector for {agent_name} returned None for recalculated token count.",
                    "agent_name": agent_name, "tick": current_tick
                })
        except Exception as e:
            self._push_log_event("orchestrator_error", {
                "message": f"Error during post-turn token refresh for {agent_name}: {str(e)}",
                "agent_name": agent_name, "tick": current_tick, "trace": traceback.format_exc()
            })
            print(f"Orchestrator ({agent_name}, Tick {current_tick}): Exception during force_refresh_and_get_current_session_tokens: {e}")


    def run_single_tick(self):
        if not self.is_running: return False
        current_tick = self.station._get_current_tick()
        self._push_log_event("tick_event", {"type": "prepare", "tick": current_tick})

        # Check for holiday mode at tick start - move agents out of Research Counter
        if (self.current_agent_index_in_turn_order == 0 and
            constants.HOLIDAY_MODE_ENABLED and
            constants.is_holiday_tick(current_tick)):
            for agent_name in self.agent_turn_order:
                agent_data = self.station.agent_module.load_agent_data(agent_name)
                if agent_data and agent_data.get(constants.AGENT_CURRENT_LOCATION_KEY) == constants.ROOM_RESEARCH_COUNTER:
                    # Move agent to lobby
                    self.station.agent_module.update_agent_current_location(agent_data, constants.ROOM_LOBBY)
                    # Add notification
                    holiday_msg = "The Research Counter is closed during holidays. You have been automatically moved to the Lobby."
                    self.station.agent_module.add_pending_notification(agent_data, holiday_msg)
                    self.station.agent_module.save_agent_data(agent_name, agent_data)
                    self._push_log_event("holiday_event", {"agent": agent_name, "action": "moved_from_research", "tick": current_tick})

        if not self.is_prepared: # Should ideally be prepared before run_single_tick is called by main_loop
            print("Orchestrator: FATAL - run_single_tick called while not prepared. Stopping.")
            self._push_log_event("orchestrator_error", {"message": "run_single_tick called without preparation."})
            self.is_running = False
            return False

        # _load_agent_turn_order is now primarily for updating the list and handling removed/added agents' connectors.
        # The current_agent_index_in_turn_order should persist across calls to run_single_tick unless a tick completes.
        self._load_agent_turn_order() 

        if not self.agent_turn_order:
            self._push_log_event("tick_event", {"type": "skip_empty_order", "tick": current_tick})
            self.station.end_tick(); new_tick_after_empty = self.station._get_current_tick()
            self._push_log_event("tick_event", {"type": "end_after_empty", "ended_tick": current_tick, "next_tick": new_tick_after_empty})
            return True 

        self._push_log_event("tick_event", {"type": "start", "tick": current_tick, "turn_order": self.agent_turn_order, "start_index": self.current_agent_index_in_turn_order})
        all_agents_processed_successfully_this_tick = True

        while self.current_agent_index_in_turn_order < len(self.agent_turn_order):
            if not self.is_running: return False 

            if self.is_paused: 
                self._push_log_event("orchestrator_status", {"status": "paused_wait", "reason": self.get_pause_reason()})
                self.pause_event.wait()
                self.pause_event.clear()
                if not self.is_running: return False 
                self._push_log_event("orchestrator_status", {"status": "resumed_in_tick"})
            
            # Check pause request first - it should override waiting state
            if self.pause_requested:
                self.is_paused = True
                self.pause_requested = False
                if self.is_waiting:
                    # Transitioning from waiting to paused
                    self.pause_reason_message = "Manual pause request received (was waiting)."
                    self._push_log_event("orchestrator_status", {"status": "paused_from_waiting", "reason": self.pause_reason_message, "next_agent_index": self.current_agent_index_in_turn_order, "waiting_reasons": self.waiting_reasons})
                else:
                    self.pause_reason_message = "Manual pause request received."
                    self._push_log_event("orchestrator_status", {"status": "paused", "reason": self.pause_reason_message, "next_agent_index": self.current_agent_index_in_turn_order})
                
                if self.station and self.is_prepared:
                    self.station.save_next_agent_index_to_config(self.current_agent_index_in_turn_order)
                    self._push_log_event("orchestrator_info", {"message": f"Saved next agent index {self.current_agent_index_in_turn_order} to config due to manual pause."})
                return True
            
            if self.is_waiting:
                # Check if wait conditions are resolved
                if self._check_wait_conditions_resolved():
                    self._exit_waiting_state()
                else:
                    # Still waiting, skip processing
                    return True 

            agent_name = self.agent_turn_order[self.current_agent_index_in_turn_order]
            self._push_log_event("agent_event", {"type": "turn_start", "agent_name": agent_name, "tick": current_tick})
            
            agent_data_for_turn = self.station.agent_module.load_agent_data(agent_name)
            if not agent_data_for_turn:
                self._push_log_event("agent_event", {"type": "turn_skip_inactive", "agent_name": agent_name, "tick": current_tick})
                self.current_tick_processed_agents.add(agent_name)
                self.current_agent_index_in_turn_order += 1
                continue
            
            # Check agent life limit
            if not self.station._check_agent_life_limit(agent_name, current_tick):
                self._push_log_event("agent_event", {"type": "turn_ended_life_limit", "agent_name": agent_name, "tick": current_tick})
                self.current_tick_processed_agents.add(agent_name)
                self.current_agent_index_in_turn_order += 1
                all_agents_processed_successfully_this_tick = False
                continue
            
            # Check and notify if agent just reached maturity
            self.station._check_and_notify_maturity(agent_name, agent_data_for_turn, current_tick)

            # Check if a session end has been requested for this agent
            if agent_data_for_turn.get(constants.AGENT_SESSION_END_REQUESTED_KEY):
                self._push_log_event("agent_event", {"type": "session_end_by_request", "agent_name": agent_name, "tick": current_tick})
                self.station.end_agent_session(agent_name)
                self.remove_agent_from_orchestrator(agent_name)
                # The agent is now removed from the live turn order, so we continue to the next index,
                # which will now point to the next agent in the modified list.
                # No need to increment the index here as the list has shifted.
                continue
            
            # NOTE: Removed turn skipping for agents awaiting human intervention
            # Agents can continue working while waiting for human assistance

            observation_markdown, obs_error = self.station.request_status(agent_name)
            if obs_error or not observation_markdown:
                self._push_log_event("agent_event", {"type": "turn_skip_obs_error", "agent_name": agent_name, "tick": current_tick, "error": obs_error})
                self.current_tick_processed_agents.add(agent_name)
                self.current_agent_index_in_turn_order += 1
                continue

            # MODIFICATION: _get_llm_response now returns (text, can_continue_bool)
            llm_response_text, can_agent_session_continue = self._get_llm_response(agent_name, observation_markdown, current_tick)

            if self.is_paused: # _get_llm_response might have triggered a pause via _trigger_pause_due_to_llm_error
                print(f"Orchestrator: Paused after LLM API error for {agent_name}. Turn will be retried upon resume.")
                # Do NOT increment current_agent_index_in_turn_order
                return True # Signal main_loop to wait on pause_event

            if llm_response_text is None and not can_agent_session_continue: # Critical connector error
                 self._push_log_event("agent_event", {"type": "turn_skip_connector_fail", "agent_name": agent_name, "tick": current_tick})
                 self.current_tick_processed_agents.add(agent_name); self.current_agent_index_in_turn_order += 1; all_agents_processed_successfully_this_tick = False; continue
            
            if not can_agent_session_continue: # Session ended due to tokens or other critical LLM issue
                self._push_log_event("agent_event", {"type": "turn_ended_by_llm_response_handler", "agent_name": agent_name, "tick": current_tick, "reason": "Token budget or critical LLM error."})
                # Agent session is already marked ended by station.update_agent_token_budget
                # It will be pruned from turn order on next tick's _load_agent_turn_order
                self.current_tick_processed_agents.add(agent_name)
                self.current_agent_index_in_turn_order += 1
                all_agents_processed_successfully_this_tick = False # This agent's turn ended prematurely
                continue # Move to next agent
            handler_wrapper, actions_executed_summary, submit_error = self.station.submit_response(agent_name, llm_response_text) # type: ignore
            
            if submit_error:
                self._push_log_event("agent_event", {"type": "submit_error", "agent_name": agent_name, "tick": current_tick, "error": submit_error})
            if actions_executed_summary:
                 self._push_log_event("agent_event", {"type": "actions_executed", "agent_name": agent_name, "tick": current_tick, "summary": actions_executed_summary})
            
            if handler_wrapper:
                initial_prompt = handler_wrapper.init() 
                self._handle_real_internal_action_loop(agent_name, handler_wrapper, initial_prompt, current_tick)
                if self.is_paused: 
                    self._push_log_event("orchestrator_status", {"status": "paused_during_internal", "agent_name": agent_name})
                    return True 

            self._refresh_connector_and_update_tokens_after_turn(agent_name, current_tick)           

            self.current_tick_processed_agents.add(agent_name)
            self._push_log_event("agent_event", {"type": "turn_end", "agent_name": agent_name, "tick": current_tick})
            self.current_agent_index_in_turn_order += 1
        
        if not self.is_running: return False

        if self.current_agent_index_in_turn_order >= len(self.agent_turn_order):
            self._push_log_event("tick_event", {"type": "all_agents_processed", "tick": current_tick})
            self.current_agent_index_in_turn_order = 0
            self.current_tick_processed_agents.clear()

            if self.station and self.is_prepared and not self.is_paused: # Only save if not already paused for another reason
                self.station.save_next_agent_index_to_config(self.current_agent_index_in_turn_order) # Should save 0
                #self._push_log_event("orchestrator_info", {"message": f"Saved next agent index {self.current_agent_index_in_turn_order} to config after full tick completion."})            

            # Check for pause conditions first (manual resume required) - human intervention overrides waiting
            should_auto_pause, auto_pause_reason = self._check_automatic_pause_conditions()
            if should_auto_pause:
                self.is_paused = True
                self.pause_condition_met = True
                self._push_log_event("orchestrator_status", {"status": "paused_auto_condition", "tick": current_tick, "reason": self.pause_reason_message})

                if self.station and self.is_prepared:
                    self.station.save_next_agent_index_to_config(self.current_agent_index_in_turn_order) # Saves 0
                    self._push_log_event("orchestrator_info", {"message": f"Saved next agent index {self.current_agent_index_in_turn_order} to config due to auto-pause at end of tick."})
            else:
                # Check for waiting conditions (auto-resume) only if not pausing
                should_wait, wait_reasons = self._check_automatic_wait_conditions()
                if should_wait:
                    self._enter_waiting_state(wait_reasons)
            
            # Check if we need to wait for research evaluations at tick boundary
            if hasattr(self.station, 'should_wait_for_research_evaluations_at_tick_boundary'):
                if self.station.should_wait_for_research_evaluations_at_tick_boundary():
                    # Enter formal waiting state so it shows in the web interface
                    self._enter_waiting_state({'research_tick_boundary': 'Research evaluations at tick limit'})
                    
                    self._push_log_event("orchestrator_status", {
                        "status": "waiting_for_research_at_tick_boundary",
                        "tick": current_tick,
                        "message": "Waiting for research evaluations that have reached their tick limit"
                    })
                    
                    # Wait for evaluations to complete or timeout
                    while self.station.should_wait_for_research_evaluations_at_tick_boundary() and self.is_running:
                        time.sleep(1)  # Check every second
                        # Keep the waiting state active
                        if not self.is_waiting:
                            self._enter_waiting_state({'research_tick_boundary': 'Research evaluations at tick limit'})
                    
                    # Exit waiting state
                    if self.is_waiting:
                        self._exit_waiting_state()
                    
                    self._push_log_event("orchestrator_status", {
                        "status": "research_wait_resolved",
                        "tick": current_tick,
                        "message": "Research evaluations at tick boundary have completed"
                    })
            
            new_station_tick = self.station.end_tick()
            self._push_log_event("tick_event", {"type": "end", "ended_tick": current_tick, "next_tick": new_station_tick, "auto_paused": self.is_paused})
            
            # Create automatic backup if enabled and at backup interval
            if backup_utils.should_create_automatic_backup(current_tick):
                backup_path = backup_utils.create_backup(current_tick, "automatic", self.station)
                self._push_log_event("backup_event", {
                    "type": "automatic_backup_success",
                    "tick": current_tick,
                    "backup_path": backup_path,
                    "message": f"Automatic backup created at tick {current_tick}"
                })

            # Check and update stagnation status
            self.station.check_stagnation()

            return True
        
        if self.is_paused: 
            self._push_log_event("orchestrator_status", {"status": "paused_mid_tick", "tick": current_tick, "next_agent_index": self.current_agent_index_in_turn_order})
            return True

        return False 

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
                
                if not self.agent_turn_order:
                    self._push_log_event("orchestrator_info", {"message": "Agent turn order empty, re-initializing connectors/order."})
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
            self._push_log_event("orchestrator_error", {"status": "prepare_failed", "prepared": False, "running": self.is_running, "message": "Orchestrator preparation failed (connector issues)."})
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
            msg = "Orchestrator is not prepared (agent connectors not initialized). Please prepare first."
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
        
        # Stop auto evaluator
        self.station.stop_auto_evaluator()
        
        # Stop auto research evaluator
        self.station.stop_auto_research_evaluator()
        
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
        
        created_agent_data, error_msg = self.station.create_agent(
            model_name=model_name, 
            agent_type=agent_type, agent_name=agent_name_override,
            lineage=lineage, generation=generation,
            initial_tokens_max=initial_tokens_max, internal_note=internal_note or "", 
            assigned_ancestor=assigned_ancestor or "",
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
        if llm_system_prompt: current_agent_data[constants.AGENT_LLM_SYSTEM_PROMPT_KEY] = llm_system_prompt
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

    def resolve_human_intervention(self, agent_name: str, resolution_reason: str = "Intervention resolved by UI action.", human_response: Optional[str] = None) -> Tuple[bool, str]:
        """
        Clears the 'awaiting_human_intervention' flag for an agent and logs the event.
        Called after a human interaction is deemed complete or resolved.

        Args:
            agent_name: Name of the agent whose request is being resolved
            resolution_reason: Reason for resolution
            human_response: Optional response text from the human to be saved and sent to agent
        """
        self._push_log_event("human_assist_event", {
            "type": "resolve_intervention_attempt",
            "agent_name": agent_name,
            "reason": resolution_reason,
            "has_response": human_response is not None
        })

        # Load agent data, include ended in case the flag needs clearing on an already ended agent
        agent_data = self.station.agent_module.load_agent_data(agent_name, include_ended=True, include_ascended=True)
        if not agent_data:
            msg = f"Agent '{agent_name}' not found in station records. Cannot resolve human intervention."
            self._push_log_event("human_assist_event", {"type": "resolve_intervention_error", "agent_name": agent_name, "error": msg})
            print(f"Orchestrator: {msg}")
            return False, msg

        interaction_id = agent_data.get(constants.AGENT_HUMAN_INTERACTION_ID_KEY) # Get ID before clearing
        print(f"Orchestrator: Resolving request - interaction_id={interaction_id}, human_response={'Yes' if human_response else 'No'}")

        # Save resolution to the log file via External Counter
        if interaction_id:
            external_room = self.station.rooms.get(constants.ROOM_EXTERNAL)
            if not external_room:
                print(f"Orchestrator: Warning - External Counter room not found")
            else:
                # Call the External Counter's method to save the resolution
                current_tick = self.station._get_current_tick()
                save_success = external_room.save_request_resolution(
                    request_id=interaction_id,
                    human_response=human_response,
                    resolution_reason=resolution_reason,
                    resolution_tick=current_tick
                )
                if not save_success:
                    print(f"Orchestrator: Warning - Failed to save resolution for request ID {interaction_id}")

        delta_updates = {
            constants.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG: False,
            constants.AGENT_HUMAN_INTERACTION_ID_KEY: None # Clear the interaction ID
        }

        if self.station.update_specific_agent_fields(agent_name, delta_updates):
            msg = f"Human intervention for agent '{agent_name}' (Request ID: {interaction_id or 'N/A'}) marked as resolved. Reason: {resolution_reason}"
            print(f"Orchestrator: {msg}")
            self._push_log_event("human_assist_event", {
                "type": "intervention_resolved",
                "agent_name": agent_name,
                "interaction_id": interaction_id,
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
            
            # Update External Counter tracking
            external_room = self.station.rooms.get(constants.ROOM_EXTERNAL)
            if external_room and hasattr(external_room, 'refresh_pending_requests'):
                external_room.refresh_pending_requests()
            
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

        connector = self.agent_llm_connectors.get(agent_name)
        if not connector:
            print(f"Orchestrator: No active connector for {agent_name} during manual message. Attempting to create temporary one.")
            if not self.initialize_connector_for_agent(agent_name, force_reinitialize=True):
                msg = f"Failed to initialize LLM connector for agent {agent_name} for manual message."
                self._push_log_event("human_assist_event", {"type": "manual_message_error", "agent_name": agent_name, "error": msg})
                return False, {"error": msg}
            connector = self.agent_llm_connectors.get(agent_name)
            if not connector:
                msg = f"LLM connector for agent {agent_name} could not be established for manual message."
                self._push_log_event("human_assist_event", {"type": "manual_message_error", "agent_name": agent_name, "error": msg})
                return False, {"error": msg}

        current_tick = self.station._get_current_tick()
        
        interaction_id = "N/A (Manual Takeover)"
        agent_data_for_id = self.station.agent_module.load_agent_data(agent_name, include_ended=True, include_ascended=True)
        if agent_data_for_id and agent_data_for_id.get(constants.AGENT_AWAITING_HUMAN_INTERVENTION_FLAG):
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
                    self._push_log_event("agent_event", {"type": "session_ended_tokens_manual_chat", "agent_name": agent_name, "tick": current_tick, "reason": "Token budget exhausted during manual chat."})
                    self.station._log_dialogue_entry(agent_name, {
                        "tick": current_tick, "speaker": "Station", "type": "session_end_tokens_manual_chat",
                        "reason": "Token budget exhausted during manual chat."
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
        system_prompt = agent_data.get(constants.AGENT_LLM_SYSTEM_PROMPT_KEY)
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
    for room_subdir_const in [constants.CODEX_ROOM_SUBDIR_NAME, constants.SHORT_ROOM_NAME_TEST, 
                              constants.EXTERNAL_COUNTER_SUBDIR_NAME, constants.MISC_ROOM_SUBDIR_NAME,
                              constants.COMMON_ROOM_SUBDIR_NAME]:
        file_io_utils.ensure_dir_exists(os.path.join(rooms_base_dir, room_subdir_const))


    station_instance: Optional[Station] = None
    try:
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
            constants.AGENT_LLM_SYSTEM_PROMPT_KEY: "You are an inquisitive Guest agent in the Station. Your first goal is to read the Codex preface."
        },
        "ConfigForGuest2": {
            constants.AGENT_TYPE_KEY: constants.AGENT_STATUS_GUEST,
            "desired_agent_name_if_recursive_or_check": "LiveOrchGuest2",
            constants.AGENT_MODEL_PROVIDER_CLASS_KEY: "Gemini",
            constants.AGENT_MODEL_NAME_KEY: "gemini-1.5-flash-latest",
            constants.AGENT_LLM_TEMPERATURE_KEY: 1.0,
            constants.AGENT_LLM_MAX_TOKENS_KEY: 512,
            constants.AGENT_LLM_SYSTEM_PROMPT_KEY: "You are another Guest agent. After the first agent acts, go to the public memory and read capsule 1."
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
            llm_system_prompt=config_data.get(constants.AGENT_LLM_SYSTEM_PROMPT_KEY),
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