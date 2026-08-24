"""Concurrent tick runner.

Each tick has two phases:

1. Prepare every agent observation before any agent action is committed.
2. Send all initial observations to LLM providers concurrently. Supported
   fast-lane actions detected in those responses are submitted immediately via
   single-writer services, while all normal station mutations are committed
   later in a deterministic centralized order.

Internal action loops then run per agent, with only the mutating handler step
serialized by the orchestrator's parallel commit lock.
"""

from __future__ import annotations

import copy
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from station import backup_utils, constants, tick_timing
from station.base_room import LoggingInternalActionHandlerWrapper
from station.llm_connectors import (
    LLMCorruptedThoughtSignatureError,
    LLMConnectorError,
    LLMContextOverflowError,
    LLMPermanentAPIError,
    LLMSafetyBlockError,
    LLMTransientAPIError,
)
from station.sync.parallel_state import ParallelTickState, build_parallel_action_op_id


@dataclass
class StagedLLMTurn:
    prompt: str
    response: str
    thinking_text: Optional[str]
    token_info: Dict[str, Any]
    api_metadata: Optional[Dict[str, Any]] = None
    thought_signature: Optional[str] = None


@dataclass
class InitialLLMResult:
    agent_name: str
    response_text: Optional[str]
    can_continue: bool
    staged_turn: Optional[StagedLLMTurn] = None
    precommitted_actions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    latest_token_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class PreparedTurn:
    agent_name: str
    observation: str
    agent_data: Dict[str, Any]


class ParallelTickRunner:
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.station = orchestrator.station
        self.state_store = ParallelTickState()
        self._state_lock = threading.Lock()

    def run_single_tick(self) -> bool:
        if not self.orchestrator.is_running:
            return False

        current_tick = self.station._get_current_tick()
        self.orchestrator.current_agent_index_in_turn_order = 0
        tick_timing.record_tick_start(current_tick)
        self.orchestrator._push_log_event(
            "tick_event",
            {"type": "prepare", "tick": current_tick},
        )

        self._cleanup_stale_parallel_state()

        if self.orchestrator.pause_requested:
            self.orchestrator.is_paused = True
            self.orchestrator.pause_requested = False
            self.orchestrator.pause_reason_message = "Manual pause request received."
            self.station.save_next_agent_index_to_config(0)
            self.orchestrator._push_log_event(
                "orchestrator_status",
                {"status": "paused", "reason": self.orchestrator.pause_reason_message, "next_agent_index": 0},
            )
            return True

        if self.orchestrator.is_waiting:
            if self.orchestrator._check_wait_conditions_resolved():
                self.orchestrator._exit_waiting_state()
            else:
                return True

        if not self.orchestrator.is_prepared:
            print("Orchestrator: FATAL - parallel run_single_tick called while not prepared. Stopping.")
            self.orchestrator._push_log_event("orchestrator_error", {"message": "parallel run_single_tick called without preparation."})
            self.orchestrator.is_running = False
            return False

        self.orchestrator._load_agent_turn_order()
        self.orchestrator.current_agent_index_in_turn_order = 0

        if not self.orchestrator.agent_turn_order:
            self.orchestrator._push_log_event("tick_event", {"type": "skip_empty_order", "tick": current_tick})
            self.station.end_tick()
            new_tick_after_empty = self.station._get_current_tick()
            tick_timing.record_tick_end(
                current_tick,
                new_tick_after_empty,
                metadata={"empty_turn_order": True},
            )
            self.orchestrator._push_log_event(
                "tick_event",
                {"type": "end_after_empty", "ended_tick": current_tick, "next_tick": new_tick_after_empty},
            )
            return True

        turn_order = list(self.orchestrator.agent_turn_order)
        state = self.state_store.begin_tick(
            current_tick,
            turn_order,
            eval_manager=getattr(self.station, "research_eval_manager", None),
        )

        self._move_agents_out_of_research_on_holiday(current_tick, turn_order)

        self.orchestrator._push_log_event(
            "tick_event",
            {
                "type": "start",
                "tick": current_tick,
                "turn_order": turn_order,
                "start_index": 0,
            },
        )

        internal_handlers: List[Tuple[str, LoggingInternalActionHandlerWrapper, str]] = []

        with tick_timing.time_phase(
            current_tick,
            "prepare_all_station_responses",
            metadata={"agent_count": len(turn_order)},
        ):
            prepared_turns, all_success, life_limit_handlers = self._prepare_observations(current_tick, turn_order, state)
            internal_handlers.extend(life_limit_handlers)
        if not self.orchestrator.is_running:
            return False

        if not prepared_turns and not internal_handlers:
            finished = self._finish_tick_boundary(current_tick)
            if finished:
                self._complete_tick_state(state)
            return finished

        initial_results: List[InitialLLMResult] = []
        if prepared_turns:
            with tick_timing.time_phase(
                current_tick,
                "wait_agent_responses_parallel",
                metadata={"agent_count": len(prepared_turns)},
            ):
                initial_results = self._collect_initial_llm_responses(current_tick, prepared_turns, state)

        if self.orchestrator.is_paused:
            self._abort_parallel_tick(state, turn_order, reset_connectors=True)
            return True

        initial_result_by_agent = {result.agent_name: result for result in initial_results}

        if prepared_turns:
            with tick_timing.time_phase(
                current_tick,
                "commit_initial_responses",
                metadata={"agent_count": len(prepared_turns)},
            ):
                for turn in prepared_turns:
                    if not self.orchestrator.is_running:
                        return False
                    result = initial_result_by_agent.get(turn.agent_name)
                    if result is None:
                        all_success = False
                        continue

                    # Connector history is persisted before Station mutates agent
                    # state. This matters for
                    # actions such as ascension, where the agent module moves the
                    # current identity's history file to the new recursive identity.
                    if result.staged_turn:
                        self._append_staged_turns(turn.agent_name, [result.staged_turn], current_tick)
                        with self._state_lock:
                            self.state_store.mark_agent_history_flushed(state, turn.agent_name)

                    if result.response_text is None and not result.can_continue:
                        self.orchestrator._push_log_event(
                            "agent_event",
                            {"type": "turn_skip_connector_fail", "agent_name": turn.agent_name, "tick": current_tick},
                        )
                        all_success = False
                        continue
                    if not result.can_continue:
                        self.orchestrator._push_log_event(
                            "agent_event",
                            {
                                "type": "turn_ended_by_llm_response_handler",
                                "agent_name": turn.agent_name,
                                "tick": current_tick,
                                "reason": "Token budget or critical LLM error.",
                            },
                        )
                        all_success = False
                        continue

                    with tick_timing.time_phase(
                        current_tick,
                        "commit_agent_response",
                        metadata={"agent_name": turn.agent_name},
                    ):
                        handler_wrapper, actions_summary, submit_error = self.station.submit_response(
                            turn.agent_name,
                            result.response_text or "",
                            precommitted_action_results=result.precommitted_actions,
                        )
                    if submit_error:
                        self.orchestrator._push_log_event(
                            "agent_event",
                            {"type": "submit_error", "agent_name": turn.agent_name, "tick": current_tick, "error": submit_error},
                        )
                    if actions_summary:
                        self.orchestrator._push_log_event(
                            "agent_event",
                            {"type": "actions_executed", "agent_name": turn.agent_name, "tick": current_tick, "summary": actions_summary},
                        )

                    latest_token_info = result.latest_token_info
                    if handler_wrapper:
                        initial_prompt = handler_wrapper.init()
                        internal_handlers.append((turn.agent_name, handler_wrapper, initial_prompt))
                    else:
                        self._update_token_budget_from_info(turn.agent_name, latest_token_info, current_tick)

                    self.orchestrator.current_tick_processed_agents.add(turn.agent_name)
                    self.orchestrator._push_log_event("agent_event", {"type": "turn_committed", "agent_name": turn.agent_name, "tick": current_tick})
                    with self._state_lock:
                        self.state_store.mark_agent_committed(state, turn.agent_name)

        if internal_handlers:
            for agent_name, handler, _prompt in internal_handlers:
                with self._state_lock:
                    self.state_store.mark_internal_action_started(
                        state,
                        agent_name,
                        type(handler.actual_handler).__name__,
                    )
            with tick_timing.time_phase(
                current_tick,
                "internal_actions_parallel",
                metadata={"agent_count": len(internal_handlers)},
            ):
                internal_token_info = self._run_internal_handlers_parallel(current_tick, internal_handlers, state)
            for agent_name, token_info in internal_token_info.items():
                self._update_token_budget_from_info(agent_name, token_info, current_tick)

        compaction_agents: List[str] = []
        for turn in prepared_turns:
            if turn.agent_name not in compaction_agents:
                compaction_agents.append(turn.agent_name)
        for agent_name, _handler, _prompt in internal_handlers:
            if agent_name not in compaction_agents:
                compaction_agents.append(agent_name)
        maybe_compact = getattr(self.orchestrator, "maybe_run_context_compaction_after_turn", None)
        for agent_name in compaction_agents:
            if callable(maybe_compact):
                maybe_compact(agent_name, current_tick)
            if self.orchestrator.is_paused:
                self.station.save_next_agent_index_to_config(0)
                return True

        if self.orchestrator.is_paused:
            self.station.save_next_agent_index_to_config(0)
            return True

        self.orchestrator.current_agent_index_in_turn_order = 0
        self.orchestrator.current_tick_processed_agents.clear()
        self.station.save_next_agent_index_to_config(0)

        if not all_success:
            self.orchestrator._push_log_event(
                "tick_event",
                {"type": "completed_with_agent_errors", "tick": current_tick},
            )

        finished = self._finish_tick_boundary(current_tick)
        if finished:
            self._complete_tick_state(state)
        return finished

    def _cleanup_stale_parallel_state(self) -> None:
        cleanup = self.state_store.cleanup_stale_run(
            station=self.station,
            eval_manager=getattr(self.station, "research_eval_manager", None),
        )
        if cleanup.get("had_stale_state"):
            self.orchestrator._push_log_event(
                "parallel_tick_recovery",
                {
                    "message": "Recovered incomplete parallel tick; retrying from tick boundary.",
                    "rolled_back_eval_ids": cleanup.get("rolled_back_eval_ids", []),
                    "rolled_back_history_agents": cleanup.get("rolled_back_history_agents", []),
                },
            )

    def _move_agents_out_of_research_on_holiday(self, current_tick: int, turn_order: List[str]) -> None:
        if not self.station.is_holiday_tick(current_tick):
            return
        for agent_name in turn_order:
            agent_data = self.station.agent_module.load_agent_data(agent_name)
            if agent_data and agent_data.get(constants.AGENT_CURRENT_LOCATION_KEY) == constants.ROOM_RESEARCH_CENTER:
                self.station.agent_module.update_agent_current_location(agent_data, constants.ROOM_LOBBY)
                holiday_msg = "The Research Center is closed during holidays. You have been automatically moved to the Lobby."
                self.station.agent_module.add_pending_notification(agent_data, holiday_msg)
                self.station.agent_module.save_agent_data(agent_name, agent_data)
                self.orchestrator._push_log_event(
                    "holiday_event",
                    {"agent": agent_name, "action": "moved_from_research", "tick": current_tick},
                )

    def _prepare_observations(
        self,
        current_tick: int,
        turn_order: List[str],
        state: Dict[str, Any],
    ) -> Tuple[List[PreparedTurn], bool, List[Tuple[str, LoggingInternalActionHandlerWrapper, str]]]:
        prepared_turns: List[PreparedTurn] = []
        internal_handlers: List[Tuple[str, LoggingInternalActionHandlerWrapper, str]] = []
        all_success = True

        for agent_name in turn_order:
            self.orchestrator._push_log_event("agent_event", {"type": "turn_start", "agent_name": agent_name, "tick": current_tick})
            agent_data = self.station.agent_module.load_agent_data(agent_name)
            if not agent_data:
                self.orchestrator._push_log_event("agent_event", {"type": "turn_skip_inactive", "agent_name": agent_name, "tick": current_tick})
                self.orchestrator.current_tick_processed_agents.add(agent_name)
                continue

            # Manual end is an administrative/emergency path and does not run
            # the normal descendant role prompt.
            if agent_data.get(constants.AGENT_SESSION_END_REQUESTED_KEY):
                self.orchestrator._push_log_event("agent_event", {"type": "session_end_by_request", "agent_name": agent_name, "tick": current_tick})
                self.station.end_agent_session(agent_name)
                self.orchestrator.remove_agent_from_orchestrator(agent_name)
                continue

            can_continue_life, life_limit_handler = self.station._check_agent_life_limit(agent_name, current_tick)
            if not can_continue_life:
                self.orchestrator._push_log_event("agent_event", {"type": "turn_ended_life_limit", "agent_name": agent_name, "tick": current_tick})
                if life_limit_handler:
                    initial_prompt = life_limit_handler.init()
                    internal_handlers.append((agent_name, life_limit_handler, initial_prompt))
                self.orchestrator.current_tick_processed_agents.add(agent_name)
                all_success = False
                continue

            self.station._check_and_notify_maturity(agent_name, agent_data, current_tick)

            with tick_timing.time_phase(
                current_tick,
                "prepare_station_response",
                metadata={"agent_name": agent_name},
            ):
                connector = self.orchestrator.agent_llm_connectors.get(agent_name)
                if connector:
                    connector.sync_state()

                observation_markdown, obs_error = self.station.request_status(agent_name)
            if obs_error or not observation_markdown:
                self.orchestrator._push_log_event(
                    "agent_event",
                    {"type": "turn_skip_obs_error", "agent_name": agent_name, "tick": current_tick, "error": obs_error},
                )
                self.orchestrator.current_tick_processed_agents.add(agent_name)
                continue

            with self._state_lock:
                path = self.state_store.write_agent_text(
                    state,
                    agent_name=agent_name,
                    filename="observation.md",
                    content=observation_markdown,
                )
                self.state_store.mark_agent_observation(state, agent_name, path)

            latest_agent_data = self.station.agent_module.load_agent_data(agent_name) or agent_data
            prepared_turns.append(
                PreparedTurn(
                    agent_name=agent_name,
                    observation=observation_markdown,
                    agent_data=copy.deepcopy(latest_agent_data),
                )
            )

        return prepared_turns, all_success, internal_handlers

    def _collect_initial_llm_responses(
        self,
        current_tick: int,
        prepared_turns: List[PreparedTurn],
        state: Dict[str, Any],
    ) -> List[InitialLLMResult]:
        max_workers = max(1, len(prepared_turns))
        results: List[InitialLLMResult] = []
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="station-parallel-agent") as executor:
            future_map = {
                executor.submit(self._run_initial_llm_for_agent, current_tick, turn, state): turn.agent_name
                for turn in prepared_turns
            }
            for future in as_completed(future_map):
                agent_name = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    trace = traceback.format_exc()
                    self.orchestrator._push_log_event(
                        "orchestrator_error",
                        {"message": f"Parallel LLM worker failed for {agent_name}: {exc}", "trace": trace},
                    )
                    result = InitialLLMResult(
                        agent_name=agent_name,
                        response_text=None,
                        can_continue=False,
                        error=str(exc),
                    )
                results.append(result)
        return results

    def _run_initial_llm_for_agent(
        self,
        current_tick: int,
        turn: PreparedTurn,
        state: Dict[str, Any],
    ) -> InitialLLMResult:
        with tick_timing.time_phase(
            current_tick,
            "wait_agent_response",
            metadata={"agent_name": turn.agent_name},
        ):
            response_text, can_continue, staged_turn, token_info = self._send_llm_staged(
                agent_name=turn.agent_name,
                prompt=turn.observation,
                current_tick=current_tick,
                prompt_type="observation",
                internal_loop_step=None,
            )

        precommitted: Dict[str, Dict[str, Any]] = {}
        if response_text and can_continue and not self.orchestrator.is_paused:
            with self._state_lock:
                path = self.state_store.write_agent_text(
                    state,
                    agent_name=turn.agent_name,
                    filename="response.md",
                    content=response_text,
                )
                self.state_store.mark_agent_response(state, turn.agent_name, path)
            precommitted = self._precommit_fast_lane_actions(
                current_tick=current_tick,
                turn=turn,
                response_text=response_text,
                state=state,
            )

        return InitialLLMResult(
            agent_name=turn.agent_name,
            response_text=response_text,
            can_continue=can_continue,
            staged_turn=staged_turn,
            precommitted_actions=precommitted,
            latest_token_info=token_info,
        )

    def _send_llm_staged(
        self,
        *,
        agent_name: str,
        prompt: str,
        current_tick: int,
        prompt_type: str,
        internal_loop_step: Optional[int],
        connector: Any = None,
        persist_to_disk: bool = False,
    ) -> Tuple[Optional[str], bool, Optional[StagedLLMTurn], Optional[Dict[str, Any]]]:
        connector = connector or self.orchestrator._get_current_connector_for_agent(agent_name)
        if not connector:
            self.orchestrator._pause_for_unavailable_connector(agent_name, current_tick)
            return None, False, None, None

        # Stream prompt text; web_interface sanitizes it to the selected dashboard agent.
        event_payload = {
            "agent_name": agent_name,
            "tick": current_tick,
            "direction": "to_llm",
            "type": prompt_type,
            "text_content": prompt,
            "full_length": len(prompt),
        }
        if internal_loop_step is not None:
            event_payload["internal_loop_step"] = internal_loop_step
        self.orchestrator._push_log_event("llm_event", event_payload)

        old_persist = getattr(connector, "persist_to_disk", True)
        connector.persist_to_disk = bool(persist_to_disk)
        if hasattr(connector, "_last_api_metadata"):
            connector._last_api_metadata = None
        try:
            with self.orchestrator._agent_response_context(agent_name):
                response_text, thinking_text, token_info = connector.send_message(prompt, current_tick)
        except LLMTransientAPIError as exc:
            self._handle_llm_exception(agent_name, current_tick, exc, internal_loop_step)
            return None, True, None, None
        except LLMSafetyBlockError as exc:
            if internal_loop_step is None:
                self._handle_llm_exception(agent_name, current_tick, exc, internal_loop_step)
                return "SYSTEM_ERROR: LLM response blocked by safety filters. Orchestrator paused for human intervention.", True, None, None
            response_text = f"SYSTEM_ERROR: LLM response blocked by safety filters. Reason: {exc.block_reason}."
            thinking_text = None
            token_info = {}
            self._log_safety_block(agent_name, current_tick, exc, internal_loop_step)
        except LLMContextOverflowError as exc:
            self._handle_llm_exception(agent_name, current_tick, exc, internal_loop_step)
            return None, True, None, None
        except LLMCorruptedThoughtSignatureError as exc:
            self._handle_llm_exception(agent_name, current_tick, exc, internal_loop_step)
            return f"SYSTEM_ERROR: Corrupted Gemini thought signature for {agent_name}. Orchestrator paused for manual review.", True, None, None
        except (LLMPermanentAPIError, LLMConnectorError) as exc:
            self._handle_llm_exception(agent_name, current_tick, exc, internal_loop_step)
            return f"SYSTEM_ERROR: Permanent LLM API Error for {agent_name}. Orchestrator paused for human intervention.", True, None, None
        except Exception as exc:
            self._handle_llm_exception(agent_name, current_tick, exc, internal_loop_step)
            return f"SYSTEM_ERROR: Unexpected LLM Connector Error for {agent_name}.", True, None, None
        finally:
            connector.persist_to_disk = old_persist

        if thinking_text:
            log_entry = {
                "tick": current_tick,
                "speaker": "AgentLLM",
                "type": "thinking_block" if internal_loop_step is None else "thinking_block_internal",
                "agent_name": agent_name,
                "content": thinking_text,
            }
            if internal_loop_step is not None:
                log_entry["internal_step"] = internal_loop_step
            self.station._log_dialogue_entry(agent_name, log_entry)

        response_event = {
            "agent_name": agent_name,
            "tick": current_tick,
            "direction": "from_llm",
            "type": "response" if internal_loop_step is None else "internal_response",
            "text_content": response_text,
            "thinking_text": thinking_text,
            "full_length": len(response_text or ""),
            "token_info": token_info,
        }
        if internal_loop_step is not None:
            response_event["internal_loop_step"] = internal_loop_step
        self.orchestrator._push_log_event("llm_event", response_event)

        staged_turn = StagedLLMTurn(
            prompt=prompt,
            response=response_text or "",
            thinking_text=thinking_text,
            token_info=token_info or {},
            api_metadata=getattr(connector, "_last_api_metadata", None),
            thought_signature=getattr(connector, "_last_model_turn_thought_signature", None),
        )
        return response_text, True, staged_turn, token_info

    def _handle_llm_exception(
        self,
        agent_name: str,
        current_tick: int,
        exc: BaseException,
        internal_loop_step: Optional[int],
    ) -> None:
        error_type = type(exc).__name__
        self.orchestrator._push_log_event(
            "llm_event_error",
            {
                "agent_name": agent_name,
                "tick": current_tick,
                "internal_loop_step": internal_loop_step,
                "error_type": error_type,
                "error": str(exc),
                "trace": traceback.format_exc(),
            },
        )

        if isinstance(exc, LLMContextOverflowError):
            event_type = "llm_context_overflow_manual_pause"
            reason = "LLM_CONTEXT_OVERFLOW"
            if internal_loop_step is not None:
                event_type = "internal_action_context_overflow_manual_pause"
                reason = "LLM_CONTEXT_OVERFLOW_INTERNAL"
            self.station._log_dialogue_entry(agent_name, {
                "tick": current_tick,
                "speaker": "Station",
                "type": event_type,
                "reason": "Context overflow persisted after connector retries. Station paused for manual check.",
                "error": str(exc),
            })
            # Old behavior intentionally disabled: do not terminate or respawn an agent
            # just because an external API returned a context-overflow error.
            # critical_notification = "CRITICAL: Your input exceeded the model's context window. Your session is being terminated."
            # if internal_loop_step is not None:
            #     critical_notification = "CRITICAL: Your input exceeded the model's context window during an internal action. Your session is being terminated."
            # self.station._terminate_agent_session_with_broadcast(agent_name, "context window overflow", critical_notification)
            self.orchestrator._trigger_pause_due_to_llm_error(agent_name, exc, reason)
            return

        if isinstance(exc, LLMCorruptedThoughtSignatureError):
            event_type = "llm_corrupted_thought_signature_manual_pause"
            reason = "LLM_CORRUPTED_THOUGHT_SIGNATURE"
            if internal_loop_step is not None:
                event_type = "internal_action_corrupted_thought_signature_manual_pause"
                reason = "LLM_CORRUPTED_THOUGHT_SIGNATURE_INTERNAL"
            self.station._log_dialogue_entry(agent_name, {
                "tick": current_tick,
                "speaker": "Station",
                "type": event_type,
                "reason": "Gemini rejected a persisted thought signature. Station paused without provider fallback or retry.",
                "error": str(exc),
            })
            self.orchestrator._trigger_pause_due_to_llm_error(agent_name, exc, reason)
            return

        if isinstance(exc, LLMTransientAPIError):
            reason = "LLM_TRANSIENT_API_ERROR_INTERNAL" if internal_loop_step is not None else "LLM_TRANSIENT_API_ERROR"
            self.orchestrator._trigger_pause_due_to_llm_error(agent_name, exc, reason)
            return

        if isinstance(exc, LLMSafetyBlockError):
            self._log_safety_block(agent_name, current_tick, exc, internal_loop_step)
            reason = "LLM_SAFETY_BLOCK_INTERNAL" if internal_loop_step is not None else "LLM_SAFETY_BLOCK"
            self.orchestrator._trigger_pause_due_to_llm_error(agent_name, exc, reason)
            return

        if isinstance(exc, (LLMPermanentAPIError, LLMConnectorError)):
            reason = "LLM_CONNECTOR_ERROR_INTERNAL" if internal_loop_step is not None else "LLM_PERMANENT_API_ERROR"
            self.orchestrator._trigger_pause_due_to_llm_error(agent_name, exc, reason)
            return

        reason = "LLM_CONNECTOR_ERROR_INTERNAL" if internal_loop_step is not None else "LLM_CONNECTOR_ERROR"
        self.orchestrator._trigger_pause_due_to_llm_error(agent_name, exc, reason)

    def _log_safety_block(
        self,
        agent_name: str,
        current_tick: int,
        exc: LLMSafetyBlockError,
        internal_loop_step: Optional[int],
    ) -> None:
        entry = {
            "tick": current_tick,
            "speaker": "Station",
            "type": "llm_safety_block" if internal_loop_step is None else "internal_action_llm_safety_block",
            "error": str(exc),
            "block_reason": str(exc.block_reason),
            "prompt_feedback": str(exc.prompt_feedback),
        }
        if internal_loop_step is not None:
            entry["internal_step"] = internal_loop_step
        self.station._log_dialogue_entry(agent_name, entry)

    def _precommit_fast_lane_actions(
        self,
        *,
        current_tick: int,
        turn: PreparedTurn,
        response_text: str,
        state: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        research_service = getattr(self.orchestrator, "research_submission_service", None)
        archive_survey_service = getattr(self.orchestrator, "archive_survey_submission_service", None)

        actions = self._filtered_actions_for_commit(response_text)
        simulated_location = turn.agent_data.get(constants.AGENT_CURRENT_LOCATION_KEY)
        simulated_agent_data = copy.deepcopy(turn.agent_data)
        precommitted: Dict[str, Dict[str, Any]] = {}

        for action_index, parsed_action in enumerate(actions, start=1):
            command = parsed_action.command
            if command in [constants.ACTION_GO, constants.ACTION_GO_TO] or (
                command == constants.ACTION_EXIT_TERMINATE and simulated_location != constants.ROOM_EXIT
            ):
                target_room_short_name = (
                    constants.SHORT_ROOM_NAME_EXIT
                    if command == constants.ACTION_EXIT_TERMINATE
                    else parsed_action.args
                )
                simulated_location = self._simulate_navigation(
                    simulated_agent_data,
                    simulated_location,
                    target_room_short_name,
                    current_tick,
                )
                simulated_agent_data[constants.AGENT_CURRENT_LOCATION_KEY] = simulated_location
                continue

            op_id = build_parallel_action_op_id(current_tick, turn.agent_name, action_index)
            simulated_agent_data[constants.AGENT_CURRENT_LOCATION_KEY] = simulated_location
            if (
                command == constants.ACTION_RESEARCH_SUBMIT
                and simulated_location == constants.ROOM_RESEARCH_CENTER
                and research_service is not None
                and getattr(constants, "PARALLEL_RESEARCH_FAST_LANE_ENABLED", True)
            ):
                result = research_service.submit_and_wait(
                    agent_data=simulated_agent_data,
                    yaml_data=parsed_action.yaml_data,
                    current_tick=current_tick,
                    run_id=str(state.get("run_id") or ""),
                    op_id=op_id,
                )
                precommitted[op_id] = {
                    "action": "research_submit",
                    "accepted": result.accepted,
                    "messages": result.messages,
                    "eval_id": result.eval_id,
                    "notification": result.notification,
                    "error": result.error,
                }
                if result.eval_id:
                    with self._state_lock:
                        self.state_store.record_fast_lane_evaluation(
                            state,
                            eval_id=result.eval_id,
                            agent_name=turn.agent_name,
                            op_id=op_id,
                        )
                if result.accepted:
                    try:
                        from station.rooms.research_center import ResearchCenter

                        cooldown_ticks = ResearchCenter._get_submission_cooldown_ticks(
                            simulated_agent_data,
                            constants,
                        )
                    except Exception:
                        cooldown_ticks = 0
                    if cooldown_ticks > 0:
                        simulated_agent_data[constants.AGENT_LAST_RESEARCH_SUBMISSION_TICK_KEY] = current_tick
                continue

            if (
                command == getattr(constants, "ACTION_ARCHIVE_SURVEY", "survey")
                and simulated_location == constants.ROOM_ARCHIVE
                and archive_survey_service is not None
                and getattr(constants, "ARCHIVE_SURVEY_ENABLED", False)
                and getattr(constants, "PARALLEL_ARCHIVE_SURVEY_FAST_LANE_ENABLED", True)
            ):
                result = archive_survey_service.submit_and_wait(
                    agent_data=simulated_agent_data,
                    yaml_data=parsed_action.yaml_data,
                    current_tick=current_tick,
                    run_id=str(state.get("run_id") or ""),
                    op_id=op_id,
                )
                precommitted[op_id] = {
                    "action": "archive_survey",
                    "accepted": result.accepted,
                    "messages": result.messages,
                    "survey_id": result.survey_id,
                    "notification": result.notification,
                    "error": result.error,
                }
                if result.survey_id:
                    with self._state_lock:
                        self.state_store.record_fast_lane_survey(
                            state,
                            survey_id=result.survey_id,
                            agent_name=turn.agent_name,
                            op_id=op_id,
                        )

        return precommitted

    def _filtered_actions_for_commit(self, response_text: str) -> List[Any]:
        parsed_actions = self.station.action_parser.parse(response_text)
        max_actions = getattr(constants, "MAX_ACTIONS_PER_TURN", 10)
        actions = parsed_actions[:max_actions]
        has_ascension = any(
            action.command in [constants.ACTION_ASCEND_INHERIT, constants.ACTION_ASCEND_NEW]
            for action in actions
        )
        if not has_ascension:
            return actions
        allowed = {
            constants.ACTION_GO,
            constants.ACTION_GO_TO,
            constants.ACTION_ASCEND_INHERIT,
            constants.ACTION_ASCEND_NEW,
        }
        return [action for action in actions if action.command in allowed]

    def _simulate_navigation(
        self,
        agent_data: Dict[str, Any],
        current_location: Optional[str],
        target_room_short_name: Optional[str],
        current_tick: int,
    ) -> Optional[str]:
        target_room_full_name = constants.SHORT_ROOM_NAME_TO_FULL_MAP.get(target_room_short_name)
        if not target_room_full_name or target_room_full_name not in self.station.rooms:
            return current_location
        if (
            target_room_full_name in [
                constants.ROOM_ARCHIVE,
                constants.ROOM_PUBLIC_MEMORY,
                constants.ROOM_MAIL,
                constants.ROOM_COMMON,
                constants.ROOM_EXTERNAL_COUNTER,
            ]
            and not self.station._is_agent_mature(agent_data, current_tick)
        ):
            return current_location
        if target_room_full_name == constants.ROOM_RESEARCH_CENTER and self.station.is_holiday_tick(current_tick):
            return current_location
        return target_room_full_name

    def _append_staged_turns(self, agent_name: str, turns: List[StagedLLMTurn], current_tick: int) -> None:
        connector = self.orchestrator.agent_llm_connectors.get(agent_name)
        if not connector:
            return
        old_persist = getattr(connector, "persist_to_disk", True)
        connector.persist_to_disk = True
        try:
            for turn in turns:
                connector._append_turn_to_history_file(current_tick, "user", turn.prompt, None, None)
                append_kwargs: Dict[str, Any] = {}
                if turn.api_metadata:
                    append_kwargs["api_metadata"] = turn.api_metadata
                if turn.thought_signature is not None:
                    append_kwargs["thought_signature"] = turn.thought_signature
                try:
                    connector._append_turn_to_history_file(
                        current_tick,
                        "model",
                        turn.response,
                        turn.thinking_text,
                        turn.token_info,
                        **append_kwargs,
                    )
                except TypeError:
                    no_signature_kwargs = dict(append_kwargs)
                    no_signature_kwargs.pop("thought_signature", None)
                    try:
                        connector._append_turn_to_history_file(
                            current_tick,
                            "model",
                            turn.response,
                            turn.thinking_text,
                            turn.token_info,
                            **no_signature_kwargs,
                        )
                    except TypeError:
                        no_metadata_kwargs = dict(append_kwargs)
                        no_metadata_kwargs.pop("api_metadata", None)
                        try:
                            connector._append_turn_to_history_file(
                                current_tick,
                                "model",
                                turn.response,
                                turn.thinking_text,
                                turn.token_info,
                                **no_metadata_kwargs,
                            )
                        except TypeError:
                            connector._append_turn_to_history_file(
                                current_tick,
                                "model",
                                turn.response,
                                turn.thinking_text,
                                turn.token_info,
                            )
            if (
                getattr(connector, "_needs_reload_after_staged_history_flush", False)
                and hasattr(connector, "reload_session_from_disk")
            ):
                connector.reload_session_from_disk()
                connector._needs_reload_after_staged_history_flush = False
        finally:
            connector.persist_to_disk = old_persist

    def _run_internal_handlers_parallel(
        self,
        current_tick: int,
        handlers: List[Tuple[str, LoggingInternalActionHandlerWrapper, str]],
        state: Dict[str, Any],
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        latest_token_info: Dict[str, Optional[Dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(handlers)), thread_name_prefix="station-parallel-internal") as executor:
            future_map = {
                executor.submit(self._handle_internal_action_loop_parallel, agent_name, handler, prompt, current_tick): agent_name
                for agent_name, handler, prompt in handlers
            }
            for future in as_completed(future_map):
                agent_name = future_map[future]
                try:
                    latest_token_info[agent_name] = future.result()
                    with self._state_lock:
                        self.state_store.mark_internal_action_completed(state, agent_name)
                except Exception as exc:
                    with self._state_lock:
                        self.state_store.mark_internal_action_failed(state, agent_name, str(exc))
                    self.orchestrator._push_log_event(
                        "orchestrator_error",
                        {
                            "message": f"Parallel internal action worker failed for {agent_name}: {exc}",
                            "trace": traceback.format_exc(),
                        },
                    )
                    latest_token_info[agent_name] = None
        return latest_token_info

    def _handle_internal_action_loop_parallel(
        self,
        agent_name: str,
        handler_wrapper: LoggingInternalActionHandlerWrapper,
        initial_prompt: str,
        current_tick: int,
    ) -> Optional[Dict[str, Any]]:
        self.orchestrator._push_log_event(
            "internal_action_event",
            {
                "agent_name": agent_name,
                "tick": current_tick,
                "status": "start",
                "handler": type(handler_wrapper.actual_handler).__name__,
                "text_content": initial_prompt,
            },
        )

        connector_context = self.orchestrator._prepare_internal_action_connector_context(agent_name, handler_wrapper, current_tick)
        connector = connector_context.get("connector")
        connector_error = connector_context.get("error")
        if not connector:
            err_msg = (
                f"LLM connector unavailable for agent {agent_name}. Internal action cannot proceed: {connector_error}"
                if connector_error
                else f"LLM connector missing for agent {agent_name}. Internal action cannot proceed."
            )
            self.orchestrator._push_log_event(
                "internal_action_event",
                {"agent_name": agent_name, "tick": current_tick, "status": "error", "message": err_msg},
            )
            return None

        current_internal_prompt = initial_prompt
        max_internal_steps = 50
        loop_step_count = 0
        latest_token_info: Optional[Dict[str, Any]] = None
        staged_turns_to_flush: List[StagedLLMTurn] = []
        override_active = bool(connector_context.get("override_active"))

        try:
            while current_internal_prompt and loop_step_count < max_internal_steps and self.orchestrator.is_running:
                loop_step_count += 1
                with tick_timing.time_phase(
                    current_tick,
                    "wait_internal_agent_response",
                    metadata={"agent_name": agent_name, "internal_loop_step": loop_step_count},
                ):
                    response_text, can_continue, staged_turn, token_info = self._send_llm_staged(
                        agent_name=agent_name,
                        prompt=current_internal_prompt,
                        current_tick=current_tick,
                        prompt_type="internal_prompt",
                        internal_loop_step=loop_step_count,
                        connector=connector,
                        persist_to_disk=override_active,
                    )
                latest_token_info = token_info or latest_token_info
                if not can_continue or response_text is None:
                    break

                # Persist the internal LLM turn before handler.step mutates any
                # room/agent state, preserving the required history-before-mutation ordering.
                if staged_turn and not override_active:
                    staged_turns_to_flush.append(staged_turn)
                    self._append_staged_turns(agent_name, staged_turns_to_flush, current_tick)
                    staged_turns_to_flush = []

                with self.orchestrator._parallel_action_commit_lock:
                    with tick_timing.time_phase(
                        current_tick,
                        "commit_internal_action_step",
                        metadata={"agent_name": agent_name, "internal_loop_step": loop_step_count},
                    ):
                        next_prompt, executed_strings = handler_wrapper.step(response_text)
                        delta_updates = handler_wrapper.get_delta_updates()
                        if delta_updates:
                            if self.station.update_specific_agent_fields(agent_name, delta_updates):
                                self.orchestrator._push_log_event(
                                    "internal_action_event",
                                    {
                                        "agent_name": agent_name,
                                        "tick": current_tick,
                                        "status": "delta_applied",
                                        "step": handler_wrapper.internal_step_count,
                                        "updates": list(delta_updates.keys()),
                                    },
                                )

                if executed_strings:
                    self.orchestrator._push_log_event(
                        "internal_action_event",
                        {
                            "agent_name": agent_name,
                            "tick": current_tick,
                            "status": "step_executed_strings",
                            "step": handler_wrapper.internal_step_count,
                            "log": executed_strings,
                        },
                    )

                if next_prompt is None:
                    self.orchestrator._push_log_event(
                        "internal_action_event",
                        {
                            "agent_name": agent_name,
                            "tick": current_tick,
                            "status": "end",
                            "handler": type(handler_wrapper.actual_handler).__name__,
                            "final_log": executed_strings,
                        },
                    )
                    break
                current_internal_prompt = next_prompt

            if loop_step_count >= max_internal_steps:
                self.orchestrator._push_log_event(
                    "internal_action_event",
                    {
                        "agent_name": agent_name,
                        "tick": current_tick,
                        "status": "max_steps_reached",
                        "handler": type(handler_wrapper.actual_handler).__name__,
                    },
                )
                self.station._log_dialogue_entry(agent_name, {
                    "tick": current_tick,
                    "internal_step": handler_wrapper.internal_step_count,
                    "speaker": "Station",
                    "type": "internal_action_max_steps",
                    "handler": type(handler_wrapper.actual_handler).__name__,
                })
        finally:
            self.orchestrator._finalize_internal_action_connector_context(agent_name, connector_context, current_tick)

        return latest_token_info

    def _update_token_budget_from_info(
        self,
        agent_name: str,
        token_info: Optional[Dict[str, Any]],
        current_tick: int,
    ) -> None:
        if not token_info:
            return
        total_tokens = token_info.get("total_tokens_in_session")
        if total_tokens is None:
            return
        can_continue = self.station.update_agent_token_budget(agent_name, total_tokens)
        if not can_continue:
            self.orchestrator._push_log_event(
                "orchestrator_warning",
                {
                    "message": f"Failed to persist token budget update for agent {agent_name}.",
                    "agent_name": agent_name,
                    "tick": current_tick,
                },
            )
            return
        agent_data = self.station.agent_module.load_agent_data(agent_name)
        current_used = agent_data.get(constants.AGENT_TOKEN_BUDGET_CURRENT_KEY) if agent_data else "N/A"
        max_budget = agent_data.get(constants.AGENT_TOKEN_BUDGET_MAX_KEY) if agent_data else "N/A"
        self.orchestrator._push_log_event(
            "agent_event",
            {
                "type": "token_budget_updated",
                "agent_name": agent_name,
                "tick": current_tick,
                "current_used": current_used,
                "max_budget": max_budget,
            },
        )

    def _complete_tick_state(self, state: Dict[str, Any]) -> None:
        with self._state_lock:
            self.state_store.mark_completed(state)

    def _abort_parallel_tick(self, state: Dict[str, Any], turn_order: List[str], *, reset_connectors: bool) -> None:
        self.state_store.rollback_provisional_research(
            getattr(self.station, "research_eval_manager", None),
            state,
        )
        state["status"] = "aborted"
        state["aborted_timestamp"] = time.time()
        try:
            self.state_store.save_current(state)
        finally:
            self.state_store.clear_current()
        if reset_connectors:
            for agent_name in turn_order:
                try:
                    self.orchestrator.initialize_connector_for_agent(agent_name, force_reinitialize=True)
                except Exception as exc:
                    self.orchestrator._push_log_event(
                        "connector_error",
                        {"agent_name": agent_name, "message": f"Failed to reset connector after parallel abort: {exc}"},
                    )
        self.station.save_next_agent_index_to_config(0)

    def _finish_tick_boundary(self, current_tick: int) -> bool:
        self.orchestrator._push_log_event("tick_event", {"type": "all_agents_processed", "tick": current_tick})
        self.orchestrator.current_agent_index_in_turn_order = 0
        self.orchestrator.current_tick_processed_agents.clear()

        should_auto_pause, _auto_pause_reason = self.orchestrator._check_automatic_pause_conditions()
        if should_auto_pause:
            self.orchestrator.is_paused = True
            self.orchestrator.pause_condition_met = True
            self.orchestrator._push_log_event(
                "orchestrator_status",
                {"status": "paused_auto_condition", "tick": current_tick, "reason": self.orchestrator.pause_reason_message},
            )
            self.station.save_next_agent_index_to_config(0)
        else:
            should_wait, wait_reasons = self.orchestrator._check_automatic_wait_conditions()
            if should_wait:
                self.orchestrator._enter_waiting_state(wait_reasons)

        self._wait_for_tick_boundary_work(current_tick)

        new_station_tick = self.station.end_tick()
        tick_timing.record_tick_end(
            current_tick,
            new_station_tick,
            metadata={"auto_paused": self.orchestrator.is_paused},
        )
        self.orchestrator._push_log_event(
            "tick_event",
            {"type": "end", "ended_tick": current_tick, "next_tick": new_station_tick, "auto_paused": self.orchestrator.is_paused},
        )

        if backup_utils.should_create_automatic_backup(current_tick):
            backup_path = backup_utils.create_backup(current_tick, "automatic", self.station)
            self.orchestrator._push_log_event(
                "backup_event",
                {
                    "type": "automatic_backup_success",
                    "tick": current_tick,
                    "backup_path": backup_path,
                    "message": f"Automatic backup created at tick {current_tick}",
                },
            )

        if self.orchestrator.maybe_pause_after_configured_tick_end(current_tick):
            return True

        self.station.check_stagnation()
        return True

    def _wait_for_tick_boundary_work(self, current_tick: int) -> None:
        if hasattr(self.station, "should_wait_for_research_evaluations_at_tick_boundary"):
            if self.station.should_wait_for_research_evaluations_at_tick_boundary():
                self.orchestrator._enter_waiting_state({"research_tick_boundary": "Research evaluations at tick limit"})
                self.orchestrator._push_log_event(
                    "orchestrator_status",
                    {
                        "status": "waiting_for_research_at_tick_boundary",
                        "tick": current_tick,
                        "message": "Waiting for research evaluations that have reached their tick limit",
                    },
                )
                with tick_timing.time_phase(
                    current_tick,
                    "wait_research_tick_boundary",
                ):
                    while self.station.should_wait_for_research_evaluations_at_tick_boundary() and self.orchestrator.is_running:
                        time.sleep(1)
                        if not self.orchestrator.is_waiting:
                            self.orchestrator._enter_waiting_state({"research_tick_boundary": "Research evaluations at tick limit"})
                if self.orchestrator.is_waiting:
                    self.orchestrator._exit_waiting_state()
                self.orchestrator._push_log_event(
                    "orchestrator_status",
                    {
                        "status": "research_wait_resolved",
                        "tick": current_tick,
                        "message": "Research evaluations at tick boundary have completed",
                    },
                )

        if hasattr(self.station, "should_wait_for_external_reports_at_tick_boundary"):
            if self.station.should_wait_for_external_reports_at_tick_boundary():
                self.orchestrator._enter_waiting_state({"external_tick_boundary": "External reports at tick limit"})
                self.orchestrator._push_log_event(
                    "orchestrator_status",
                    {
                        "status": "waiting_for_external_at_tick_boundary",
                        "tick": current_tick,
                        "message": "Waiting for external reports that have reached their tick limit",
                    },
                )
                with tick_timing.time_phase(
                    current_tick,
                    "wait_external_tick_boundary",
                ):
                    while self.station.should_wait_for_external_reports_at_tick_boundary() and self.orchestrator.is_running:
                        time.sleep(1)
                        if not self.orchestrator.is_waiting:
                            self.orchestrator._enter_waiting_state({"external_tick_boundary": "External reports at tick limit"})
                if self.orchestrator.is_waiting:
                    self.orchestrator._exit_waiting_state()
                self.orchestrator._push_log_event(
                    "orchestrator_status",
                    {
                        "status": "external_wait_resolved",
                        "tick": current_tick,
                        "message": "External reports at tick boundary have completed",
                    },
                )

        if hasattr(self.station, "should_wait_for_archive_surveys_at_tick_boundary"):
            if self.station.should_wait_for_archive_surveys_at_tick_boundary():
                self.orchestrator._enter_waiting_state({"archive_survey_tick_boundary": "Archive surveys at tick limit"})
                self.orchestrator._push_log_event(
                    "orchestrator_status",
                    {
                        "status": "waiting_for_archive_survey_at_tick_boundary",
                        "tick": current_tick,
                        "message": "Waiting for archive surveys that have reached their tick limit",
                    },
                )
                with tick_timing.time_phase(
                    current_tick,
                    "wait_archive_survey_tick_boundary",
                ):
                    while self.station.should_wait_for_archive_surveys_at_tick_boundary() and self.orchestrator.is_running:
                        time.sleep(1)
                        if not self.orchestrator.is_waiting:
                            self.orchestrator._enter_waiting_state({"archive_survey_tick_boundary": "Archive surveys at tick limit"})
                if self.orchestrator.is_waiting:
                    self.orchestrator._exit_waiting_state()
                self.orchestrator._push_log_event(
                    "orchestrator_status",
                    {
                        "status": "archive_survey_wait_resolved",
                        "tick": current_tick,
                        "message": "Archive surveys at tick boundary have completed",
                    },
                )
