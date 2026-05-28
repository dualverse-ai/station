import shutil
import tempfile
import unittest
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from station import constants
from station import agent as agent_module
from station import file_io_utils
from station.base_room import RoomContext
from station.eval_research.evaluation_manager import EvaluationManager
from station.eval_research.runtime_paths import ensure_runtime_layout
from station.eval_research.submission_service import ResearchSubmissionService
from station.llm_connectors.base import BaseLLMConnector, LLMContextOverflowError
from station.station import Station
from station.sync.parallel_runner import ParallelTickRunner, StagedLLMTurn
from station.sync.parallel_state import ParallelTickState, build_parallel_action_op_id
from station.sync.parallel_status import build_parallel_tick_status


class TempStationDataTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_parallel_test_", dir="/tmp")
        self._saved_constants = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_CENTER_ENABLED": constants.RESEARCH_CENTER_ENABLED,
            "TOKEN_MANAGEMENT_ROOM_ENABLED": constants.TOKEN_MANAGEMENT_ROOM_ENABLED,
            "EXTERNAL_COUNTER_ENABLED": constants.EXTERNAL_COUNTER_ENABLED,
            "THEORY_ROOM_ENABLED": constants.THEORY_ROOM_ENABLED,
            "MAZE_ENABLED": constants.MAZE_ENABLED,
            "AGENT_ISOLATION_TICKS": constants.AGENT_ISOLATION_TICKS,
        }
        constants.BASE_STATION_DATA_PATH = self.tmpdir
        constants.RESEARCH_CENTER_ENABLED = True
        constants.TOKEN_MANAGEMENT_ROOM_ENABLED = False
        constants.EXTERNAL_COUNTER_ENABLED = False
        constants.THEORY_ROOM_ENABLED = False
        constants.MAZE_ENABLED = False
        constants.AGENT_ISOLATION_TICKS = None

    def tearDown(self):
        for key, value in self._saved_constants.items():
            setattr(constants, key, value)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def new_eval_manager(self):
        paths = ensure_runtime_layout()
        return EvaluationManager(paths.evaluations_dir)


class WakeRecorder:
    def __init__(self):
        self.reasons = []

    def wake(self, reason=""):
        self.reasons.append(reason)


class FakeStation:
    def __init__(self, eval_manager):
        self.research_eval_manager = eval_manager
        self.auto_research_evaluator = WakeRecorder()

    def is_holiday_tick(self, tick):
        return False


class ParallelResearchSyncTests(TempStationDataTestCase):
    def test_atomic_instruction_creation_allocates_unique_ids_under_threads(self):
        manager = self.new_eval_manager()

        def submit(index):
            eval_data, active_ids = manager.create_instruction_evaluation_atomic(
                author=f"agent-{index}",
                title=f"title-{index}",
                content=f"instruction-{index}",
                tick=7,
                tags=["tag"],
                abstract="abstract",
                lineage="lineage",
                max_active_for_author=10,
            )
            self.assertEqual([], active_ids)
            return eval_data["id"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(submit, range(20)))

        self.assertEqual(20, len(ids))
        self.assertEqual(20, len(set(ids)))
        self.assertEqual([str(i) for i in range(1, 21)], sorted(ids, key=int))

    def test_atomic_instruction_creation_enforces_active_author_limit(self):
        manager = self.new_eval_manager()

        first, active_ids = manager.create_instruction_evaluation_atomic(
            author="Agent A",
            title="first",
            content="instruction",
            tick=1,
            max_active_for_author=1,
        )
        second, blocked_active_ids = manager.create_instruction_evaluation_atomic(
            author="Agent A",
            title="second",
            content="instruction",
            tick=1,
            max_active_for_author=1,
        )

        self.assertIsNotNone(first)
        self.assertEqual([], active_ids)
        self.assertIsNone(second)
        self.assertEqual([first["id"]], blocked_active_ids)

    def test_fast_lane_submission_creates_provisional_eval_and_wakes_evaluator(self):
        manager = self.new_eval_manager()
        station = FakeStation(manager)
        service = ResearchSubmissionService(station)
        agent_data = {
            constants.AGENT_NAME_KEY: "Agent A",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_LINEAGE_KEY: "Lineage",
        }
        yaml_data = {
            constants.YAML_CAPSULE_TITLE: "Fast submit",
            constants.YAML_CAPSULE_TAGS: "alpha, beta",
            constants.YAML_CAPSULE_ABSTRACT: "Short abstract.",
            "instruction": "Run the experiment.",
        }

        try:
            result = service.submit_and_wait(
                agent_data=agent_data,
                yaml_data=yaml_data,
                current_tick=5,
                run_id="run-1",
                op_id="op-1",
                timeout=5,
            )
        finally:
            service.stop()

        self.assertTrue(result.accepted)
        self.assertEqual("1", result.eval_id)
        self.assertIn("Evaluation ID: 1", result.messages[0])
        self.assertTrue(station.auto_research_evaluator.reasons)

        eval_data = manager.get_evaluation("1")
        self.assertEqual("provisional", eval_data["parallel_commit_status"])
        self.assertEqual("run-1", eval_data["parallel_tick"]["run_id"])
        self.assertEqual("op-1", eval_data["parallel_tick"]["op_id"])

    def test_parallel_recovery_deletes_only_provisional_fast_lane_evals(self):
        manager = self.new_eval_manager()
        state_store = ParallelTickState(self.tmpdir)
        state = state_store.begin_tick(3, ["Agent A"], manager)

        provisional, _ = manager.create_instruction_evaluation_atomic(
            author="Agent A",
            title="provisional",
            content="instruction",
            tick=3,
            max_active_for_author=10,
            extra_metadata={
                "parallel_commit_status": "provisional",
                "parallel_tick": {"run_id": state["run_id"], "op_id": "op-1"},
            },
        )
        committed, _ = manager.create_instruction_evaluation_atomic(
            author="Agent B",
            title="committed",
            content="instruction",
            tick=3,
            max_active_for_author=10,
            extra_metadata={
                "parallel_commit_status": "committed",
                "parallel_tick": {"run_id": state["run_id"], "op_id": "op-2"},
            },
        )
        state_store.record_fast_lane_evaluation(
            state,
            eval_id=provisional["id"],
            agent_name="Agent A",
            op_id="op-1",
        )

        rolled_back = state_store.rollback_provisional_research(manager, state)

        self.assertEqual([provisional["id"]], rolled_back)
        self.assertIsNone(manager.get_evaluation(provisional["id"]))
        self.assertIsNotNone(manager.get_evaluation(committed["id"]))

    def test_parallel_tick_status_reports_waiting_and_pending_agents(self):
        status = build_parallel_tick_status({
            "status": "running",
            "tick": 9,
            "run_id": "run-1",
            "turn_order": ["Agent A", "Agent B", "Agent C", "Agent D"],
            "agents": {
                "Agent A": {"observation_prepared": True},
                "Agent B": {"observation_prepared": True, "response_received": True},
                "Agent C": {
                    "observation_prepared": True,
                    "response_received": True,
                    "actions_committed": True,
                },
            },
        })

        self.assertEqual(9, status["tick"])
        self.assertEqual(["Agent D"], status["preparing_station_response"])
        self.assertEqual(["Agent A"], status["waiting_for_response"])
        self.assertEqual(["Agent B"], status["response_received_pending_commit"])
        self.assertEqual(["Agent C"], status["committed"])
        self.assertEqual({
            "total": 4,
            "observation_prepared": 3,
            "response_received": 2,
            "pending_commit": 1,
            "committed": 1,
            "internal_action_running": 0,
        }, status["counts"])
        self.assertEqual([], status["internal_action_running"])

    def test_parallel_tick_status_reports_internal_actions_after_commit(self):
        status = build_parallel_tick_status({
            "status": "running",
            "tick": 10,
            "run_id": "run-2",
            "turn_order": ["Agent A", "Agent B"],
            "agents": {
                "Agent A": {
                    "observation_prepared": True,
                    "response_received": True,
                    "actions_committed": True,
                },
                "Agent B": {
                    "observation_prepared": True,
                    "response_received": True,
                    "actions_committed": True,
                },
            },
            "internal_actions": {
                "Agent B": {
                    "status": "running",
                    "handler": "ResearchSubmitHandler",
                    "started_timestamp": 123.0,
                }
            },
        })

        self.assertEqual(["Agent B"], status["internal_action_running"])
        self.assertEqual("ResearchSubmitHandler", status["internal_action_details"]["Agent B"]["handler"])
        self.assertEqual(1, status["counts"]["internal_action_running"])

    def test_parallel_recovery_removes_uncommitted_flushed_history(self):
        state_store = ParallelTickState(self.tmpdir)
        state = state_store.begin_tick(7, ["Agent A", "Agent B"], None)
        state_store.mark_agent_history_flushed(state, "Agent A")
        state_store.mark_agent_history_flushed(state, "Agent B")
        state_store.mark_agent_committed(state, "Agent B")

        for agent_name in ["Agent A", "Agent B"]:
            history_path = os.path.join(
                self.tmpdir,
                constants.AGENTS_DIR_NAME,
                agent_name,
                "llm_chat_history.yamll",
            )
            file_io_utils.append_yaml_line({"tick": 6, "role": "user", "parts": [{"text": "previous"}]}, history_path)
            file_io_utils.append_yaml_line({"tick": 7, "role": "user", "parts": [{"text": "staged prompt"}]}, history_path)
            file_io_utils.append_yaml_line({"tick": 7, "role": "model", "parts": [{"text": "staged response"}]}, history_path)

        rolled_back = state_store.rollback_uncommitted_history(object(), state)

        self.assertEqual(["Agent A"], rolled_back)
        agent_a_entries = file_io_utils.load_yaml_lines(
            os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME, "Agent A", "llm_chat_history.yamll")
        )
        agent_b_entries = file_io_utils.load_yaml_lines(
            os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME, "Agent B", "llm_chat_history.yamll")
        )
        self.assertEqual([6], [entry["tick"] for entry in agent_a_entries])
        self.assertEqual([6, 7, 7], [entry["tick"] for entry in agent_b_entries])

    def test_submit_response_consumes_precommitted_research_without_duplicate_eval(self):
        from station import agent as agent_module
        from station.station import Station

        station = Station()
        station.config[constants.STATION_CONFIG_CURRENT_TICK] = 5
        station._save_config()

        agent_name = "Agent A"
        agent_module.save_agent_data(
            agent_name,
            {
                constants.AGENT_NAME_KEY: agent_name,
                constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                constants.AGENT_LINEAGE_KEY: "Lineage",
                constants.AGENT_CURRENT_LOCATION_KEY: constants.ROOM_RESEARCH_CENTER,
                constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
                constants.AGENT_SHOWN_NOTIFICATIONS_KEY: [],
                constants.AGENT_ROOM_OUTPUT_HISTORY_KEY: [],
                constants.AGENT_SESSION_ENDED_KEY: False,
                constants.AGENT_IS_ASCENDED_KEY: False,
            },
        )

        eval_data, _ = station.research_eval_manager.create_instruction_evaluation_atomic(
            author=agent_name,
            title="precommitted",
            content="instruction",
            tick=5,
            max_active_for_author=10,
            extra_metadata={
                "parallel_commit_status": "provisional",
                "parallel_tick": {"run_id": "run-1", "op_id": "op-1"},
            },
        )
        response = """Submitting now.

/execute_action{submit}
```yaml
title: precommitted
tags: alpha
abstract: Short abstract.
instruction: Run the experiment.
```
"""
        op_id = build_parallel_action_op_id(5, agent_name, 1)
        handler, actions, error = station.submit_response(
            agent_name,
            response,
            precommitted_action_results={
                op_id: {
                    "accepted": True,
                    "messages": ["Experiment 'precommitted' submitted. Evaluation ID: 1."],
                    "eval_id": eval_data["id"],
                    "notification": "Your experiment request has been queued for the coder. Evaluation ID: 1.",
                }
            },
        )

        self.assertIsNone(handler)
        self.assertIsNone(error)
        self.assertIn("Evaluation ID: 1", " ".join(actions))
        self.assertEqual(["1"], station.research_eval_manager.get_all_evaluation_ids())
        self.assertEqual(
            "committed",
            station.research_eval_manager.get_evaluation("1")["parallel_commit_status"],
        )

        saved_agent = agent_module.load_agent_data(agent_name)
        self.assertIn(
            "Your experiment request has been queued for the coder. Evaluation ID: 1.",
            saved_agent[constants.AGENT_NOTIFICATIONS_PENDING_KEY],
        )

    def test_staged_history_flush_preserves_optional_thought_signature(self):
        class FakeConnector:
            def __init__(self):
                self.persist_to_disk = False
                self.calls = []
                self.reload_calls = 0

            def _append_turn_to_history_file(
                self,
                tick,
                role,
                text,
                thinking_text=None,
                token_info=None,
                api_metadata=None,
                thought_signature=None,
            ):
                self.calls.append(
                    {
                        "tick": tick,
                        "role": role,
                        "text": text,
                        "thinking_text": thinking_text,
                        "token_info": token_info,
                        "thought_signature": thought_signature,
                    }
                )

            def reload_session_from_disk(self):
                self.reload_calls += 1

        connector = FakeConnector()
        fake_orchestrator = type(
            "FakeOrchestrator",
            (),
            {
                "station": object(),
                "agent_llm_connectors": {"Agent A": connector},
            },
        )()
        runner = ParallelTickRunner(fake_orchestrator)

        runner._append_staged_turns(
            "Agent A",
            [
                StagedLLMTurn(
                    prompt="prompt",
                    response="response",
                    thinking_text="thinking",
                    token_info={"total_tokens_in_session": 10},
                    thought_signature="signature-1",
                )
            ],
            9,
        )

        self.assertEqual(2, len(connector.calls))
        self.assertEqual("user", connector.calls[0]["role"])
        self.assertIsNone(connector.calls[0]["thought_signature"])
        self.assertEqual("model", connector.calls[1]["role"])
        self.assertEqual("signature-1", connector.calls[1]["thought_signature"])
        self.assertEqual(0, connector.reload_calls)
        self.assertFalse(connector.persist_to_disk)

    def test_staged_history_flush_reloads_only_when_connector_requests_it(self):
        class FakeConnector:
            def __init__(self):
                self.persist_to_disk = False
                self._needs_reload_after_staged_history_flush = True
                self.calls = []
                self.reload_calls = 0

            def _append_turn_to_history_file(
                self,
                tick,
                role,
                text,
                thinking_text=None,
                token_info=None,
                api_metadata=None,
                thought_signature=None,
            ):
                self.calls.append((tick, role, text))

            def reload_session_from_disk(self):
                self.reload_calls += 1

        connector = FakeConnector()
        fake_orchestrator = type(
            "FakeOrchestrator",
            (),
            {
                "station": object(),
                "agent_llm_connectors": {"Agent A": connector},
            },
        )()
        runner = ParallelTickRunner(fake_orchestrator)

        runner._append_staged_turns(
            "Agent A",
            [
                StagedLLMTurn(
                    prompt="prompt",
                    response="response",
                    thinking_text=None,
                    token_info={},
                )
            ],
            9,
        )

        self.assertEqual(1, connector.reload_calls)
        self.assertFalse(connector._needs_reload_after_staged_history_flush)
        self.assertFalse(connector.persist_to_disk)

    def test_parallel_context_overflow_pauses_instead_of_terminating_agent(self):
        class FakeStationForOverflow:
            def __init__(self):
                self.dialogue_entries = []
                self.terminations = []

            def _log_dialogue_entry(self, agent_name, entry):
                self.dialogue_entries.append((agent_name, entry))

            def _terminate_agent_session_with_broadcast(self, *args, **kwargs):
                self.terminations.append((args, kwargs))

        class FakeOrchestratorForOverflow:
            def __init__(self, station):
                self.station = station
                self.pause_calls = []

            def _push_log_event(self, event_type, payload):
                pass

            def _trigger_pause_due_to_llm_error(self, agent_name, error, error_type):
                self.pause_calls.append((agent_name, str(error), error_type))

        station = FakeStationForOverflow()
        orchestrator = FakeOrchestratorForOverflow(station)
        runner = ParallelTickRunner(orchestrator)

        runner._handle_llm_exception(
            "Agent A",
            7,
            LLMContextOverflowError("input exceeds context"),
            internal_loop_step=None,
        )

        self.assertEqual([], station.terminations)
        self.assertEqual(
            [("Agent A", "input exceeds context", "LLM_CONTEXT_OVERFLOW")],
            orchestrator.pause_calls,
        )
        self.assertEqual("llm_context_overflow_manual_pause", station.dialogue_entries[0][1]["type"])

    def _minimal_station_for_request_status(self, agent_data):
        agent_module.save_agent_data(agent_data[constants.AGENT_NAME_KEY], agent_data)

        class QuietRoom:
            def build_ascension_system_message(self, _agent_data, _room_context):
                return None

            def get_room_output(self, _agent_data, _room_context, _current_tick):
                return ""

        station = object.__new__(Station)
        station.agent_module = agent_module
        station.capsule_module = None
        station.config = {
            constants.STATION_CONFIG_CURRENT_TICK: 10,
            constants.STATION_CONFIG_STATION_STATUS: "Testing",
        }
        station.rooms = {constants.ROOM_LOBBY: QuietRoom()}
        station.room_context = RoomContext(
            agent_manager=agent_module,
            capsule_manager=None,
            notification_manager=None,
            constants_module=constants,
            station_instance=station,
        )
        station._get_current_tick = lambda: 10
        station.is_holiday_tick = lambda _tick: False
        station._check_and_apply_inactivity_warning = lambda data, _tick: data
        station._check_and_apply_meta_reflection_warning = lambda data, _tick: data
        station._check_and_apply_life_warnings = lambda data, _tick: data
        station._check_and_apply_supervisor_report_reminder = lambda data, _tick: data
        station._get_agent_age_status = lambda _data, _tick: None
        station._log_dialogue_entry = lambda _agent_name, _entry: None
        return station

    def test_request_status_hides_missing_or_stale_token_budget_lines(self):
        base_agent = {
            constants.AGENT_NAME_KEY: "Agent A",
            constants.AGENT_DESCRIPTION_KEY: "desc",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_TOKEN_BUDGET_MAX_KEY: 100,
            constants.AGENT_CURRENT_LOCATION_KEY: constants.ROOM_LOBBY,
            constants.AGENT_ROOM_OUTPUT_HISTORY_KEY: [],
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
            constants.AGENT_SESSION_ENDED_KEY: False,
            constants.AGENT_IS_ASCENDED_KEY: False,
        }

        missing_count_station = self._minimal_station_for_request_status({
            **base_agent,
            constants.AGENT_TOKEN_BUDGET_CURRENT_KEY: None,
        })
        missing_observation, missing_error = missing_count_station.request_status("Agent A")
        self.assertIsNone(missing_error)
        self.assertNotIn("Agent Token Budget", missing_observation)

        stale_station = self._minimal_station_for_request_status({
            **base_agent,
            constants.AGENT_TOKEN_BUDGET_CURRENT_KEY: 95,
            constants.AGENT_TOKEN_BUDGET_CURRENT_STALE_KEY: True,
            constants.AGENT_TOKEN_BUDGET_STALE_REASON_KEY: constants.TOKEN_BUDGET_STALE_REASON_PROVIDER_COUNT_UNAVAILABLE_AFTER_PRUNE,
        })
        stale_observation, stale_error = stale_station.request_status("Agent A")
        self.assertIsNone(stale_error)
        self.assertNotIn("Agent Token Budget", stale_observation)
        saved = agent_module.load_agent_data("Agent A")
        self.assertEqual([], saved.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, []))

    def test_fresh_token_update_clears_stale_budget_flags(self):
        agent_module.save_agent_data(
            "Agent A",
            {
                constants.AGENT_NAME_KEY: "Agent A",
                constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                constants.AGENT_TOKEN_BUDGET_CURRENT_KEY: 95,
                constants.AGENT_TOKEN_BUDGET_MAX_KEY: 100,
                constants.AGENT_TOKEN_BUDGET_CURRENT_STALE_KEY: True,
                constants.AGENT_TOKEN_BUDGET_STALE_REASON_KEY: constants.TOKEN_BUDGET_STALE_REASON_PROVIDER_COUNT_UNAVAILABLE_AFTER_PRUNE,
                constants.AGENT_LAST_PRUNE_ACTION_TICK_KEY: 9,
            },
        )
        station = object.__new__(Station)
        station.agent_module = agent_module
        station._get_current_tick = lambda: 10

        updated, effective_max = station._update_agent_token_usage_only("Agent A", 12)

        self.assertEqual(12, updated[constants.AGENT_TOKEN_BUDGET_CURRENT_KEY])
        self.assertEqual(100, effective_max)
        self.assertNotIn(constants.AGENT_TOKEN_BUDGET_CURRENT_STALE_KEY, updated)
        self.assertNotIn(constants.AGENT_TOKEN_BUDGET_STALE_REASON_KEY, updated)

    def test_parallel_prepare_marks_unavailable_post_prune_count_stale(self):
        agent_name = "Agent A"
        agent_module.save_agent_data(
            agent_name,
            {
                constants.AGENT_NAME_KEY: agent_name,
                constants.AGENT_DESCRIPTION_KEY: "desc",
                constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                constants.AGENT_TOKEN_BUDGET_CURRENT_KEY: 265477,
                constants.AGENT_TOKEN_BUDGET_MAX_KEY: 300000,
                constants.AGENT_CURRENT_LOCATION_KEY: constants.ROOM_LOBBY,
                constants.AGENT_ROOM_OUTPUT_HISTORY_KEY: [],
                constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
                constants.AGENT_PRUNED_DIALOGUE_TICKS_KEY: [
                    {
                        constants.PRUNE_TICKS_KEY: "1-2",
                        constants.PRUNE_SUMMARY_KEY: "summary",
                        constants.PRUNE_PRUNED_AT_TICK_KEY: 9,
                    }
                ],
                constants.AGENT_LAST_PRUNE_ACTION_TICK_KEY: 9,
                constants.AGENT_SESSION_ENDED_KEY: False,
                constants.AGENT_IS_ASCENDED_KEY: False,
            },
        )

        class UncountableConnector(BaseLLMConnector):
            def __init__(self):
                self.initialized = 0
                super().__init__(
                    model_name="fake",
                    agent_name=agent_name,
                    agent_data_path=os.path.join(self_tmpdir, constants.AGENTS_DIR_NAME, agent_name),
                    api_key="fake",
                    system_prompt="system",
                )
                self._last_known_prune_blocks = []

            def _load_history_from_file(self):
                return []

            def _append_turn_to_history_file(self, *args, **kwargs):
                pass

            def _initialize_chat_session(self):
                self.initialized += 1

            def _send_message_implementation(self, user_prompt, current_tick, attempt_number=0):
                return "ok", None, {}

            def get_chat_history(self):
                return []

            def get_current_total_session_tokens(self):
                return 265477

            def _can_count_current_session_tokens_authoritatively(self):
                return False

        self_tmpdir = self.tmpdir
        station = self._minimal_station_for_request_status(agent_module.load_agent_data(agent_name))
        connector = UncountableConnector()

        class FakeOrchestrator:
            def __init__(self):
                self.station = station
                self.agent_llm_connectors = {agent_name: connector}
                self.current_tick_processed_agents = set()
                self.events = []

            def _push_log_event(self, event_type, payload):
                self.events.append((event_type, payload))

        runner = ParallelTickRunner(FakeOrchestrator())
        state = {}

        prepared, all_success, internal_handlers = runner._prepare_observations(10, [agent_name], state)

        self.assertTrue(all_success)
        self.assertEqual([], internal_handlers)
        self.assertEqual(1, len(prepared))
        self.assertNotIn("Agent Token Budget", prepared[0].observation)
        updated_agent = agent_module.load_agent_data(agent_name)
        self.assertTrue(updated_agent[constants.AGENT_TOKEN_BUDGET_CURRENT_STALE_KEY])
        self.assertEqual(
            constants.TOKEN_BUDGET_STALE_REASON_PROVIDER_COUNT_UNAVAILABLE_AFTER_PRUNE,
            updated_agent[constants.AGENT_TOKEN_BUDGET_STALE_REASON_KEY],
        )
        self.assertEqual(265477, updated_agent[constants.AGENT_TOKEN_BUDGET_CURRENT_KEY])

    def test_initial_staged_turn_flushes_before_ascension_commit(self):
        guest_name = "Guest_1"
        new_name = "Noether I"
        old_history = os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME, guest_name, "llm_chat_history.yamll")
        new_history = os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME, new_name, "llm_chat_history.yamll")

        class FileBackedConnector:
            def __init__(self, history_path):
                self.history_path = history_path
                self.persist_to_disk = False
                self.reload_calls = 0

            def _append_turn_to_history_file(
                self,
                tick,
                role,
                text,
                thinking_text=None,
                token_info=None,
                api_metadata=None,
                thought_signature=None,
            ):
                if not self.persist_to_disk:
                    return
                os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
                with open(self.history_path, "a", encoding="utf-8") as handle:
                    handle.write(f"tick={tick} role={role} text={text}\n")

            def reload_session_from_disk(self):
                self.reload_calls += 1

        class AscensionLikeStation:
            def __init__(self):
                self.history_present_before_submit = False
                self.submitted_response = None
                self.research_eval_manager = None

            def _get_current_tick(self):
                return 0

            def is_holiday_tick(self, tick):
                return False

            def save_next_agent_index_to_config(self, index):
                self.saved_index = index

            def submit_response(self, agent_name, response_text, precommitted_action_results=None):
                self.submitted_response = response_text
                self.history_present_before_submit = os.path.exists(old_history)
                if self.history_present_before_submit:
                    os.makedirs(os.path.dirname(new_history), exist_ok=True)
                    os.rename(old_history, new_history)
                return None, ["Ascended"], None

        class FakeOrchestrator:
            def __init__(self, station, connector):
                self.station = station
                self.is_running = True
                self.is_prepared = True
                self.pause_requested = False
                self.is_waiting = False
                self.is_paused = False
                self.agent_turn_order = [guest_name]
                self.agent_llm_connectors = {guest_name: connector}
                self.current_agent_index_in_turn_order = 0
                self.current_tick_processed_agents = set()
                self.events = []

            def _push_log_event(self, event_type, payload):
                self.events.append((event_type, payload))

            def _load_agent_turn_order(self):
                return None

        station = AscensionLikeStation()
        connector = FileBackedConnector(old_history)
        runner = ParallelTickRunner(FakeOrchestrator(station, connector))
        runner._cleanup_stale_parallel_state = lambda: None
        runner._move_agents_out_of_research_on_holiday = lambda current_tick, turn_order: None
        runner._prepare_observations = lambda current_tick, turn_order, state: (
            [type("Prepared", (), {"agent_name": guest_name, "observation": "observation", "agent_data": {}})()],
            True,
            [],
        )
        runner._collect_initial_llm_responses = lambda current_tick, prepared_turns, state: [
            type(
                "Result",
                (),
                {
                    "agent_name": guest_name,
                    "response_text": "/execute_action{ascend_new}",
                    "can_continue": True,
                    "staged_turn": StagedLLMTurn(
                        prompt="observation before ascend",
                        response="/execute_action{ascend_new}",
                        thinking_text=None,
                        token_info={"total_tokens_in_session": 123},
                    ),
                    "precommitted_actions": {},
                    "latest_token_info": {"total_tokens_in_session": 123},
                },
            )()
        ]
        runner._update_token_budget_from_info = lambda agent_name, token_info, current_tick: None
        runner._finish_tick_boundary = lambda current_tick: True
        runner._complete_tick_state = lambda state: None

        self.assertTrue(runner.run_single_tick())

        self.assertTrue(station.history_present_before_submit)
        self.assertTrue(os.path.exists(new_history))
        self.assertFalse(os.path.exists(old_history))
        with open(new_history, "r", encoding="utf-8") as handle:
            history_text = handle.read()
        self.assertIn("observation before ascend", history_text)
        self.assertIn("/execute_action{ascend_new}", history_text)

    def test_internal_staged_turn_flushes_before_handler_step(self):
        events = []
        staged_turn = StagedLLMTurn(
            prompt="internal prompt",
            response="internal response",
            thinking_text=None,
            token_info={"total_tokens_in_session": 12},
        )

        class FakeStation:
            def update_specific_agent_fields(self, agent_name, delta_updates):
                return True

            def _log_dialogue_entry(self, agent_name, entry):
                pass

        class FakeOrchestrator:
            def __init__(self):
                self.station = FakeStation()
                self.is_running = True
                self.is_paused = False
                self._parallel_action_commit_lock = threading.Lock()

            def _prepare_internal_action_connector_context(self, agent_name, handler_wrapper, current_tick):
                return {"connector": object(), "override_active": False}

            def _finalize_internal_action_connector_context(self, agent_name, connector_context, current_tick):
                pass

            def _push_log_event(self, event_type, payload):
                pass

        class FakeHandler:
            actual_handler = object()
            internal_step_count = 0

            def step(self, response_text):
                events.append("step")
                return None, ["done"]

            def get_delta_updates(self):
                return {}

        runner = ParallelTickRunner(FakeOrchestrator())
        runner._send_llm_staged = lambda **kwargs: (
            "internal response",
            True,
            staged_turn,
            {"total_tokens_in_session": 12},
        )
        runner._append_staged_turns = lambda agent_name, turns, current_tick: events.append("flush")

        runner._handle_internal_action_loop_parallel("Agent A", FakeHandler(), "internal prompt", 4)

        self.assertEqual(["flush", "step"], events)

    def test_archive_reviewer_reload_preserves_explicit_system_prompt(self):
        tmpdir = self.tmpdir

        class FakeConnector(BaseLLMConnector):
            def __init__(self, agent_name, system_prompt):
                self.initialized_count = 0
                super().__init__(
                    model_name="fake-model",
                    agent_name=agent_name,
                    agent_data_path=os.path.join(tmpdir, "reviewer_history"),
                    api_key="fake",
                    system_prompt=system_prompt,
                )

            def _load_history_from_file(self):
                return []

            def _append_turn_to_history_file(
                self,
                tick,
                role,
                text,
                thinking_text=None,
                token_info=None,
                api_metadata=None,
            ):
                pass

            def _initialize_chat_session(self):
                self.initialized_count += 1

            def _send_message_implementation(self, user_prompt, current_tick, attempt_number=0):
                return "ok", None, {}

            def get_chat_history(self):
                return []

            def get_current_total_session_tokens(self):
                return 0

        reviewer_prompt = "explicit reviewer system prompt"
        agent_module.save_agent_data(
            "AutoArchiveEvaluator",
            {
                constants.AGENT_NAME_KEY: "AutoArchiveEvaluator",
                constants.AGENT_ROLE_DEFINITION_KEY: "wrong station role prompt",
                constants.AGENT_SESSION_ENDED_KEY: False,
                constants.AGENT_IS_ASCENDED_KEY: False,
            },
        )

        connector = FakeConnector("AutoArchiveEvaluator", reviewer_prompt)
        connector.reload_session_from_disk()

        self.assertEqual(reviewer_prompt, connector.system_prompt)
        self.assertEqual(1, connector.initialized_count)

    def test_normal_agent_reload_picks_up_role_prompt_changes(self):
        tmpdir = self.tmpdir

        class FakeConnector(BaseLLMConnector):
            def __init__(self, agent_name, system_prompt):
                self.initialized_count = 0
                super().__init__(
                    model_name="fake-model",
                    agent_name=agent_name,
                    agent_data_path=os.path.join(tmpdir, "normal_history"),
                    api_key="fake",
                    system_prompt=system_prompt,
                )

            def _load_history_from_file(self):
                return []

            def _append_turn_to_history_file(
                self,
                tick,
                role,
                text,
                thinking_text=None,
                token_info=None,
                api_metadata=None,
            ):
                pass

            def _initialize_chat_session(self):
                self.initialized_count += 1

            def _send_message_implementation(self, user_prompt, current_tick, attempt_number=0):
                return "ok", None, {}

            def get_chat_history(self):
                return []

            def get_current_total_session_tokens(self):
                return 0

        agent_module.save_agent_data(
            "Agent A",
            {
                constants.AGENT_NAME_KEY: "Agent A",
                constants.AGENT_ROLE_DEFINITION_KEY: "new supervisor role prompt",
                constants.AGENT_SESSION_ENDED_KEY: False,
                constants.AGENT_IS_ASCENDED_KEY: False,
            },
        )

        connector = FakeConnector("Agent A", "old prompt")
        connector.reload_session_from_disk()

        self.assertIn("new supervisor role prompt", connector.system_prompt)
        self.assertNotEqual("old prompt", connector.system_prompt)


if __name__ == "__main__":
    unittest.main()
