import shutil
import tempfile
import unittest
import os
import threading
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from station import backup_utils, constants, research_storage
from station import agent as agent_module
from station import file_io_utils
from station.action_parser import ActionParser
from station.base_room import BaseRoom, RoomContext
from station.eval_research.evaluation_manager import EvaluationManager
from station.eval_research.runtime_paths import ensure_runtime_layout
from station.eval_research.submission_service import ResearchSubmissionService
from station.llm_connectors.base import BaseLLMConnector, LLMContextOverflowError
from station.station import Station
from station.sync.parallel_runner import ParallelTickRunner, PreparedTurn, StagedLLMTurn
from station.sync.parallel_state import ParallelTickState, build_parallel_action_op_id
from station.sync.parallel_status import build_parallel_tick_status


class TempStationDataTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_parallel_test_", dir="/tmp")
        self._saved_constants = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "BACKUP_BASE_DIR": constants.BACKUP_BASE_DIR,
            "RESEARCH_CENTER_ENABLED": constants.RESEARCH_CENTER_ENABLED,
            "EXTERNAL_COUNTER_ENABLED": constants.EXTERNAL_COUNTER_ENABLED,
            "MAZE_ENABLED": constants.MAZE_ENABLED,
            "AGENT_ISOLATION_TICKS": constants.AGENT_ISOLATION_TICKS,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
        }
        constants.BASE_STATION_DATA_PATH = self.tmpdir
        constants.BACKUP_BASE_DIR = os.path.join(self.tmpdir, "_backup")
        constants.RESEARCH_CENTER_ENABLED = True
        constants.EXTERNAL_COUNTER_ENABLED = False
        constants.MAZE_ENABLED = False
        constants.AGENT_ISOLATION_TICKS = None
        constants.RESEARCH_STORAGE_BASE_PATH = None

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
    def test_environment_research_storage_base_overrides_constant_value(self):
        constants.RESEARCH_STORAGE_BASE_PATH = str(os.path.join(self.tmpdir, "yaml_storage"))
        env_storage = os.path.join(self.tmpdir, "env_storage")

        with mock.patch.dict(
            os.environ,
            {research_storage.BASE_PATH_ENV: env_storage},
        ):
            paths = ensure_runtime_layout()

        self.assertTrue(os.path.islink(paths.storage_root))
        self.assertTrue(research_storage.path_is_within(Path(paths.storage_real_root), Path(env_storage)))
        marker = research_storage.read_allocation_marker(Path(paths.storage_real_root))
        self.assertEqual("live", marker["kind"])

    def test_changing_research_storage_base_relocates_existing_managed_link(self):
        first_base = os.path.join(self.tmpdir, "first_storage")
        second_base = os.path.join(self.tmpdir, "second_storage")

        with mock.patch.dict(os.environ, {research_storage.BASE_PATH_ENV: first_base}):
            first = ensure_runtime_layout()
        first_target = Path(first.storage_real_root)
        (first_target / "artifact.txt").write_text("preserved", encoding="utf-8")

        with mock.patch.dict(os.environ, {research_storage.BASE_PATH_ENV: second_base}):
            second = ensure_runtime_layout()

        second_target = Path(second.storage_real_root)
        self.assertTrue(research_storage.path_is_within(second_target, Path(second_base)))
        self.assertEqual("preserved", (second_target / "artifact.txt").read_text(encoding="utf-8"))
        self.assertFalse(first_target.exists())
        self.assertFalse(research_storage.allocation_marker_path(first_target).exists())
        self.assertEqual("live", research_storage.read_allocation_marker(second_target)["kind"])

    def _minimal_orchestrator_for_turn_order(self, station):
        from station.station_runner import Orchestrator

        orchestrator = object.__new__(Orchestrator)
        orchestrator.station = station
        orchestrator.agent_turn_order = []
        orchestrator.current_tick_processed_agents = set()
        orchestrator.agent_llm_connectors = {}
        orchestrator.is_prepared = False
        orchestrator.is_paused = False
        orchestrator.pause_condition_met = False
        orchestrator.pause_reason_message = ""
        orchestrator.current_agent_index_in_turn_order = 0
        orchestrator.events = []
        orchestrator._push_log_event = lambda event_type, payload: orchestrator.events.append((event_type, payload))
        orchestrator.initialize_connector_for_agent = lambda *_args, **_kwargs: True
        return orchestrator

    def _minimal_station_for_turn_order(self, turn_order):
        station = object.__new__(Station)
        station.agent_module = agent_module
        station.config = {
            constants.STATION_CONFIG_CURRENT_TICK: 10,
            constants.STATION_CONFIG_AGENT_TURN_ORDER: list(turn_order),
        }
        station._save_config = lambda: None
        station.get_next_agent_index_from_config = lambda: 0
        station.save_next_agent_index_to_config = lambda index: station.config.__setitem__(
            constants.STATION_CONFIG_NEXT_AGENT_INDEX, index
        )
        station._get_current_tick = lambda: 10
        return station

    def test_restart_pruned_ended_agent_is_respawned(self):
        original = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="Aletheia",
            generation=1,
            current_tick=1,
            model_provider_class="OpenAI",
        )
        self.assertIsNotNone(original)
        original[constants.AGENT_SESSION_ENDED_KEY] = True
        agent_module.save_agent_data(original[constants.AGENT_NAME_KEY], original)

        station = self._minimal_station_for_turn_order([original[constants.AGENT_NAME_KEY]])
        station.get_agent_departure_reason = Station.get_agent_departure_reason.__get__(station, Station)
        station.create_respawn_guest_agent = Station.create_respawn_guest_agent.__get__(station, Station)
        orchestrator = self._minimal_orchestrator_for_turn_order(station)

        self.assertFalse(orchestrator.agent_turn_order)
        self.assertTrue(orchestrator._load_agent_turn_order())

        self.assertEqual(1, len(orchestrator.agent_turn_order))
        respawned_name = orchestrator.agent_turn_order[0]
        self.assertNotEqual(original[constants.AGENT_NAME_KEY], respawned_name)
        respawned = agent_module.load_agent_data(respawned_name)
        self.assertIsNotNone(respawned)
        self.assertEqual(constants.AGENT_STATUS_GUEST, respawned[constants.AGENT_STATUS_KEY])
        handled_events = [payload for event_type, payload in orchestrator.events if payload.get("status") == "agent_departure_handled"]
        self.assertTrue(handled_events)
        self.assertEqual([original[constants.AGENT_NAME_KEY]], handled_events[-1]["config_pruned_agents"])

    def test_restart_pruned_ascended_guest_is_not_respawned(self):
        guest = agent_module.create_guest_agent(
            model_name="test-model",
            current_tick=1,
            model_provider_class="OpenAI",
        )
        self.assertIsNotNone(guest)
        guest[constants.AGENT_IS_ASCENDED_KEY] = True
        guest[constants.AGENT_ASCENDED_TO_NAME_KEY] = "Aletheia I"
        agent_module.save_agent_data(guest[constants.AGENT_NAME_KEY], guest)

        station = self._minimal_station_for_turn_order([guest[constants.AGENT_NAME_KEY]])
        station.get_agent_departure_reason = Station.get_agent_departure_reason.__get__(station, Station)
        station.create_respawn_guest_agent = lambda _name: self.fail("ascended guest should not respawn")
        orchestrator = self._minimal_orchestrator_for_turn_order(station)

        self.assertFalse(orchestrator._load_agent_turn_order())
        self.assertEqual([], station.config[constants.STATION_CONFIG_AGENT_TURN_ORDER])
        handled_events = [payload for event_type, payload in orchestrator.events if payload.get("status") == "agent_departure_handled"]
        self.assertTrue(handled_events)
        self.assertEqual([guest[constants.AGENT_NAME_KEY]], handled_events[-1]["ascended_agents"])

    def test_restart_pruned_ended_agent_pauses_when_auto_respawn_disabled(self):
        original = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="Aletheia",
            generation=1,
            current_tick=1,
            model_provider_class="OpenAI",
        )
        self.assertIsNotNone(original)
        original[constants.AGENT_SESSION_ENDED_KEY] = True
        agent_module.save_agent_data(original[constants.AGENT_NAME_KEY], original)

        station = self._minimal_station_for_turn_order([original[constants.AGENT_NAME_KEY]])
        station.get_agent_departure_reason = Station.get_agent_departure_reason.__get__(station, Station)
        station.create_respawn_guest_agent = lambda _name: self.fail("AUTO_RESPAWN disabled")
        orchestrator = self._minimal_orchestrator_for_turn_order(station)

        old_auto_respawn = constants.AUTO_RESPAWN
        constants.AUTO_RESPAWN = False
        try:
            self.assertFalse(orchestrator._load_agent_turn_order())
        finally:
            constants.AUTO_RESPAWN = old_auto_respawn

        self.assertTrue(orchestrator.is_paused)
        self.assertTrue(orchestrator.pause_condition_met)
        self.assertIn("AUTO_RESPAWN disabled", orchestrator.pause_reason_message)

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

    def test_atomic_instruction_creation_allows_configured_active_author_limit(self):
        manager = self.new_eval_manager()

        first, active_ids = manager.create_instruction_evaluation_atomic(
            author="Agent A",
            title="first",
            content="instruction",
            tick=1,
            max_active_for_author=2,
        )
        second, second_active_ids = manager.create_instruction_evaluation_atomic(
            author="Agent A",
            title="second",
            content="instruction",
            tick=1,
            max_active_for_author=2,
        )
        third, blocked_active_ids = manager.create_instruction_evaluation_atomic(
            author="Agent A",
            title="third",
            content="instruction",
            tick=1,
            max_active_for_author=2,
        )

        self.assertIsNotNone(first)
        self.assertEqual([], active_ids)
        self.assertIsNotNone(second)
        self.assertIsInstance(second_active_ids, list)
        self.assertIsNone(third)
        self.assertEqual([first["id"], second["id"]], blocked_active_ids)

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

    def test_fast_lane_precommit_updates_simulated_cooldown_between_same_turn_submits(self):
        manager = self.new_eval_manager()
        station = FakeStation(manager)
        station.action_parser = ActionParser()
        service = ResearchSubmissionService(station)

        class FakeOrchestrator:
            def __init__(self):
                self.station = station
                self.research_submission_service = service
                self.archive_survey_submission_service = None

            def _push_log_event(self, _event_type, _payload):
                pass

        old_cooldown = constants.RESEARCH_SUBMISSION_COOLDOWN_TICKS
        old_limit = constants.RESEARCH_MAX_CONCURRENT_SUBMISSIONS
        constants.RESEARCH_SUBMISSION_COOLDOWN_TICKS = 10
        constants.RESEARCH_MAX_CONCURRENT_SUBMISSIONS = 2
        try:
            service.start()
            runner = ParallelTickRunner(FakeOrchestrator())
            turn = PreparedTurn(
                agent_name="Agent A",
                observation="",
                agent_data={
                    constants.AGENT_NAME_KEY: "Agent A",
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_LINEAGE_KEY: "Lineage",
                    constants.AGENT_CURRENT_LOCATION_KEY: constants.ROOM_RESEARCH_CENTER,
                },
            )
            response_text = """/execute_action{submit}
```yaml
title: First
tags: alpha
abstract: First abstract.
instruction: Run first.
```

/execute_action{submit}
```yaml
title: Second
tags: beta
abstract: Second abstract.
instruction: Run second.
```
"""
            precommitted = runner._precommit_fast_lane_actions(
                current_tick=5,
                turn=turn,
                response_text=response_text,
                state={"run_id": "run-1"},
            )
        finally:
            service.stop()
            constants.RESEARCH_SUBMISSION_COOLDOWN_TICKS = old_cooldown
            constants.RESEARCH_MAX_CONCURRENT_SUBMISSIONS = old_limit

        first = precommitted["tick:5:agent:Agent A:action:1"]
        second = precommitted["tick:5:agent:Agent A:action:2"]
        self.assertTrue(first["accepted"])
        self.assertEqual("1", first["eval_id"])
        self.assertFalse(second["accepted"])
        self.assertEqual("cooldown", second["error"])
        self.assertIn("cooldown active", second["messages"][0])
        self.assertEqual(["1"], manager.get_active_eval_ids_for_author("Agent A"))

    def test_fast_lane_precommit_accepts_two_same_turn_submits_when_limit_allows(self):
        manager = self.new_eval_manager()
        station = FakeStation(manager)
        station.action_parser = ActionParser()
        service = ResearchSubmissionService(station)

        class FakeOrchestrator:
            def __init__(self):
                self.station = station
                self.research_submission_service = service
                self.archive_survey_submission_service = None

            def _push_log_event(self, _event_type, _payload):
                pass

        old_cooldown = constants.RESEARCH_SUBMISSION_COOLDOWN_TICKS
        old_limit = constants.RESEARCH_MAX_CONCURRENT_SUBMISSIONS
        constants.RESEARCH_SUBMISSION_COOLDOWN_TICKS = 0
        constants.RESEARCH_MAX_CONCURRENT_SUBMISSIONS = 2
        try:
            service.start()
            runner = ParallelTickRunner(FakeOrchestrator())
            turn = PreparedTurn(
                agent_name="Agent A",
                observation="",
                agent_data={
                    constants.AGENT_NAME_KEY: "Agent A",
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_LINEAGE_KEY: "Lineage",
                    constants.AGENT_CURRENT_LOCATION_KEY: constants.ROOM_RESEARCH_CENTER,
                },
            )
            response_text = """/execute_action{submit}
```yaml
title: First
tags: alpha
abstract: First abstract.
instruction: Run first.
```

/execute_action{submit}
```yaml
title: Second
tags: beta
abstract: Second abstract.
instruction: Run second.
```
"""
            precommitted = runner._precommit_fast_lane_actions(
                current_tick=5,
                turn=turn,
                response_text=response_text,
                state={"run_id": "run-1"},
            )
        finally:
            service.stop()
            constants.RESEARCH_SUBMISSION_COOLDOWN_TICKS = old_cooldown
            constants.RESEARCH_MAX_CONCURRENT_SUBMISSIONS = old_limit

        first = precommitted["tick:5:agent:Agent A:action:1"]
        second = precommitted["tick:5:agent:Agent A:action:2"]
        self.assertTrue(first["accepted"])
        self.assertEqual("1", first["eval_id"])
        self.assertTrue(second["accepted"])
        self.assertEqual("2", second["eval_id"])
        self.assertEqual(["1", "2"], manager.get_active_eval_ids_for_author("Agent A"))

    def test_parallel_recovery_reserves_rolled_back_fast_lane_eval_ids(self):
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
        tombstone = manager.get_evaluation(provisional["id"])
        self.assertEqual("rolled_back", tombstone["parallel_commit_status"])
        self.assertEqual("blocked", tombstone["status"])
        self.assertTrue(tombstone["notification"]["suppressed"])
        self.assertIsNotNone(manager.get_evaluation(committed["id"]))

        replacement, _ = manager.create_instruction_evaluation_atomic(
            author="Agent C",
            title="replacement",
            content="instruction",
            tick=3,
            max_active_for_author=10,
        )
        self.assertEqual("3", replacement["id"])

        notifications = []
        manager.set_notification_callback(lambda author, message: notifications.append((author, message)))
        manager.finalize_evaluation(provisional["id"], "stale coder report", final_status="completed")
        self.assertFalse(manager.send_notification_if_pending(provisional["id"]))
        self.assertEqual([], notifications)
        self.assertEqual("rolled_back", manager.get_evaluation(provisional["id"])["parallel_commit_status"])

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

    def test_parallel_snapshot_cleanup_removes_dirs_after_retention_window(self):
        state_store = ParallelTickState(self.tmpdir)
        state_store.ensure_layout()
        old_dir = os.path.join(state_store.run_dir, "tick_10_oldrun")
        keep_dir = os.path.join(state_store.run_dir, "tick_11_keeprun")
        active_dir = os.path.join(state_store.run_dir, "tick_10_activerun")
        invalid_dir = os.path.join(state_store.run_dir, "misc")
        for path in [old_dir, keep_dir, active_dir, invalid_dir]:
            file_io_utils.ensure_dir_exists(path)

        removed = state_store.cleanup_completed_run_dirs(
            20,
            retention_ticks=10,
            exclude_run_id="activerun",
        )

        self.assertEqual([old_dir], removed)
        self.assertFalse(os.path.exists(old_dir))
        self.assertTrue(os.path.isdir(keep_dir))
        self.assertTrue(os.path.isdir(active_dir))
        self.assertTrue(os.path.isdir(invalid_dir))

    def test_backup_skips_parallel_sync_directory(self):
        sync_dir = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.PARALLEL_TICK_STATE_DIR_NAME,
        )
        agents_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME)

        self.assertTrue(backup_utils._should_skip_backup_dir(sync_dir, constants.BACKUP_BASE_DIR))
        self.assertFalse(backup_utils._should_skip_backup_dir(agents_dir, constants.BACKUP_BASE_DIR))

    def test_backup_records_admin_symlink_without_following_branch_data(self):
        file_io_utils.save_yaml(
            {"station_id": "station-id", "current_tick": 1},
            os.path.join(constants.BASE_STATION_DATA_PATH, constants.STATION_CONFIG_FILENAME),
        )
        file_io_utils.save_text("live", os.path.join(constants.BASE_STATION_DATA_PATH, "live.txt"))
        admin_dir = os.path.join(constants.BASE_STATION_DATA_PATH, "admin")
        file_io_utils.ensure_dir_exists(admin_dir)

        with tempfile.TemporaryDirectory(prefix="station_branch_target_", dir="/tmp") as branch_dir:
            file_io_utils.save_text("branch", os.path.join(branch_dir, "branch_marker.txt"))
            os.symlink(branch_dir, os.path.join(admin_dir, "station_data_s1"), target_is_directory=True)

            manifest_path = backup_utils.create_backup(1, "test")

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        file_paths = {entry["path"] for entry in manifest["files"]}
        symlinks = {entry["path"]: entry["target"] for entry in manifest.get("symlinks", [])}

        self.assertIn("live.txt", file_paths)
        self.assertNotIn("admin/station_data_s1/branch_marker.txt", file_paths)
        self.assertEqual(branch_dir, symlinks.get("admin/station_data_s1"))

        with tempfile.TemporaryDirectory(prefix="station_restore_target_", dir="/tmp") as restore_parent:
            restore_dir = os.path.join(restore_parent, "station_data_restored")
            self.assertTrue(backup_utils.restore_backup("station-id", 1, restore_dir))
            restored_link = os.path.join(restore_dir, "admin", "station_data_s1")
            self.assertTrue(os.path.islink(restored_link))
            self.assertEqual(branch_dir, os.readlink(restored_link))
            self.assertFalse(os.path.exists(os.path.join(restored_link, "branch_marker.txt")))

    def test_restore_backup_subtree_restores_multistart_job_only(self):
        file_io_utils.save_yaml(
            {"station_id": "station-id", "current_tick": 1},
            os.path.join(constants.BASE_STATION_DATA_PATH, constants.STATION_CONFIG_FILENAME),
        )
        file_io_utils.save_text("live", os.path.join(constants.BASE_STATION_DATA_PATH, "live.txt"))
        job_dir = os.path.join(constants.BASE_STATION_DATA_PATH, "multistart", "501_job")
        file_io_utils.save_text("state", os.path.join(job_dir, "state.yaml"))
        file_io_utils.save_text("seed", os.path.join(job_dir, "station_data_s1", "station_config.yaml"))
        admin_dir = os.path.join(job_dir, "admin")
        file_io_utils.ensure_dir_exists(admin_dir)
        os.symlink(os.path.join("..", "station_data_s1"), os.path.join(admin_dir, "station_data_s1"), target_is_directory=True)

        backup_utils.create_backup(1, "test")

        with tempfile.TemporaryDirectory(prefix="station_restore_target_", dir="/tmp") as restore_parent:
            restore_dir = os.path.join(restore_parent, "multistart_501_job")
            self.assertTrue(backup_utils.restore_backup_subtree("station-id", 1, "multistart/501_job", restore_dir))
            self.assertTrue(os.path.isfile(os.path.join(restore_dir, "state.yaml")))
            restored_config = os.path.join(restore_dir, "station_data_s1", "station_config.yaml")
            self.assertEqual("seed", file_io_utils.load_text(restored_config))
            self.assertFalse(os.path.exists(os.path.join(restore_dir, "live.txt")))
            restored_link = os.path.join(restore_dir, "admin", "station_data_s1")
            self.assertTrue(os.path.islink(restored_link))
            self.assertEqual(os.path.join("..", "station_data_s1"), os.readlink(restored_link))

    def test_same_tick_normal_backup_moves_existing_snapshot_aside(self):
        file_io_utils.save_yaml(
            {"station_id": "station-id", "current_tick": 1},
            os.path.join(constants.BASE_STATION_DATA_PATH, constants.STATION_CONFIG_FILENAME),
        )
        file_io_utils.save_text("before", os.path.join(constants.BASE_STATION_DATA_PATH, "marker.txt"))

        first_manifest_path = backup_utils.create_backup(1, "manual")
        file_io_utils.save_text("after", os.path.join(constants.BASE_STATION_DATA_PATH, "marker.txt"))
        second_manifest_path = backup_utils.create_backup(1, "automatic")

        self.assertEqual(first_manifest_path, second_manifest_path)
        self.assertEqual("tick_1.json", os.path.basename(first_manifest_path))
        moved_manifests = [
            name for name in os.listdir(os.path.dirname(first_manifest_path))
            if name.startswith("tick_1_backup_") and name.endswith(".json")
        ]
        self.assertEqual(1, len(moved_manifests))

        with open(os.path.join(os.path.dirname(first_manifest_path), moved_manifests[0]), "r", encoding="utf-8") as handle:
            moved_manifest = json.load(handle)
        with open(first_manifest_path, "r", encoding="utf-8") as handle:
            second_manifest = json.load(handle)

        def marker_hash(manifest):
            for entry in manifest["files"]:
                if entry["path"] == "marker.txt":
                    return entry["hash"]
            return None

        self.assertIsNotNone(marker_hash(moved_manifest))
        self.assertIsNotNone(marker_hash(second_manifest))
        self.assertNotEqual(marker_hash(moved_manifest), marker_hash(second_manifest))

        with tempfile.TemporaryDirectory(prefix="station_restore_target_", dir="/tmp") as restore_parent:
            restore_dir = os.path.join(restore_parent, "station_data_restored")
            self.assertTrue(backup_utils.restore_backup("station-id", 1, restore_dir))
            self.assertEqual("after", file_io_utils.load_text(os.path.join(restore_dir, "marker.txt")))

    def test_multistart_restore_prefers_named_snapshot_over_normal_tick_snapshot(self):
        file_io_utils.save_yaml(
            {"station_id": "station-id", "current_tick": 1},
            os.path.join(constants.BASE_STATION_DATA_PATH, constants.STATION_CONFIG_FILENAME),
        )
        job_dir = os.path.join(constants.BASE_STATION_DATA_PATH, "multistart", "501_job")
        seed_config = os.path.join(job_dir, "station_data_s1", "station_config.yaml")
        file_io_utils.save_text("full seed", seed_config)
        named_manifest = backup_utils.create_backup(
            1,
            "multistart",
            snapshot_suffix="multistart_501_job",
        )
        self.assertEqual("tick_1_multistart_501_job.json", os.path.basename(named_manifest))

        shutil.rmtree(os.path.join(job_dir, "station_data_s1"))
        file_io_utils.save_text("pruned", os.path.join(job_dir, "station_data_s1.pruned.yaml"))
        backup_utils.create_backup(1, "automatic")

        with tempfile.TemporaryDirectory(prefix="station_restore_target_", dir="/tmp") as restore_parent:
            restore_dir = os.path.join(restore_parent, "multistart_501_job")
            self.assertTrue(backup_utils.restore_backup_subtree("station-id", 1, "multistart/501_job", restore_dir))
            self.assertEqual("full seed", file_io_utils.load_text(os.path.join(restore_dir, "station_data_s1", "station_config.yaml")))
            self.assertFalse(os.path.exists(os.path.join(restore_dir, "station_data_s1.pruned.yaml")))

    def test_full_restore_can_select_named_multistart_snapshot(self):
        file_io_utils.save_yaml(
            {"station_id": "station-id", "current_tick": 1},
            os.path.join(constants.BASE_STATION_DATA_PATH, constants.STATION_CONFIG_FILENAME),
        )
        marker_path = os.path.join(constants.BASE_STATION_DATA_PATH, "marker.txt")
        file_io_utils.save_text("selected branch at tick 1", marker_path)
        backup_utils.create_backup(
            1,
            "multistart",
            snapshot_suffix="multistart_501_job",
        )
        file_io_utils.save_text("later automatic state", marker_path)
        backup_utils.create_backup(1, "automatic")

        with tempfile.TemporaryDirectory(prefix="station_restore_target_", dir="/tmp") as restore_parent:
            restore_dir = os.path.join(restore_parent, "station_data_restored")
            self.assertTrue(
                backup_utils.restore_backup(
                    "station-id",
                    1,
                    restore_dir,
                    snapshot_suffix="multistart_501_job",
                )
            )
            self.assertEqual(
                "selected branch at tick 1",
                file_io_utils.load_text(os.path.join(restore_dir, "marker.txt")),
            )

    def test_backup_follows_research_storage_root_symlink(self):
        file_io_utils.save_yaml(
            {"station_id": "station-id", "current_tick": 1},
            os.path.join(constants.BASE_STATION_DATA_PATH, constants.STATION_CONFIG_FILENAME),
        )
        research_dir = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            constants.SHORT_ROOM_NAME_RESEARCH,
        )
        file_io_utils.ensure_dir_exists(research_dir)

        with tempfile.TemporaryDirectory(prefix="station_shared_storage_", dir="/tmp") as storage_dir:
            stored_file = os.path.join(storage_dir, "lineages", "alpha", "data", "result.txt")
            file_io_utils.save_text("shared", stored_file)
            os.symlink(storage_dir, os.path.join(storage_dir, "loop"), target_is_directory=True)
            deceptive_research_dir = os.path.join(
                storage_dir,
                "nested",
                "station_data_s9",
                constants.ROOMS_DIR_NAME,
                constants.SHORT_ROOM_NAME_RESEARCH,
            )
            file_io_utils.ensure_dir_exists(deceptive_research_dir)
            os.symlink(
                storage_dir,
                os.path.join(deceptive_research_dir, constants.RESEARCH_STORAGE_DIR),
                target_is_directory=True,
            )
            os.symlink(
                storage_dir,
                os.path.join(research_dir, constants.RESEARCH_STORAGE_DIR),
                target_is_directory=True,
            )

            manifest_path = backup_utils.create_backup(1, "test")

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        file_paths = {entry["path"] for entry in manifest["files"]}
        symlink_paths = {entry["path"] for entry in manifest.get("symlinks", [])}

        storage_relative = "/".join([
            constants.ROOMS_DIR_NAME,
            constants.SHORT_ROOM_NAME_RESEARCH,
            constants.RESEARCH_STORAGE_DIR,
        ])
        self.assertIn(f"{storage_relative}/lineages/alpha/data/result.txt", file_paths)
        self.assertNotIn(storage_relative, symlink_paths)
        self.assertIn(f"{storage_relative}/loop", symlink_paths)
        self.assertIn(
            f"{storage_relative}/nested/station_data_s9/{storage_relative}",
            symlink_paths,
        )

    def test_backup_follows_archived_multistart_branch_research_storage_symlink(self):
        file_io_utils.save_yaml(
            {"station_id": "station-id", "current_tick": 1},
            os.path.join(constants.BASE_STATION_DATA_PATH, constants.STATION_CONFIG_FILENAME),
        )
        branch_research = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            "multistart",
            "501_job",
            "station_data_s1",
            constants.ROOMS_DIR_NAME,
            constants.SHORT_ROOM_NAME_RESEARCH,
        )
        file_io_utils.ensure_dir_exists(branch_research)
        file_io_utils.save_yaml(
            {"job_id": "job", "status": "finalizing"},
            os.path.join(constants.BASE_STATION_DATA_PATH, "multistart", "501_job", "state.yaml"),
        )

        with tempfile.TemporaryDirectory(prefix="station_remote_seed_storage_", dir="/tmp") as storage_dir:
            stored_file = os.path.join(storage_dir, "lineages", "alpha", "result.txt")
            file_io_utils.save_text("remote branch", stored_file)
            os.symlink(
                storage_dir,
                os.path.join(branch_research, constants.RESEARCH_STORAGE_DIR),
                target_is_directory=True,
            )

            manifest_path = backup_utils.create_backup(
                1,
                "multistart",
                snapshot_suffix="multistart_501_job",
            )

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected = (
            "multistart/501_job/station_data_s1/rooms/research/storage/"
            "lineages/alpha/result.txt"
        )
        self.assertIn(expected, {entry["path"] for entry in manifest["files"]})
        self.assertNotIn(
            "multistart/501_job/station_data_s1/rooms/research/storage",
            {entry["path"] for entry in manifest.get("symlinks", [])},
        )

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

    def test_unavailable_connector_pauses_parallel_send(self):
        from station.station_runner import Orchestrator

        class FakeStationForMissingConnector:
            def __init__(self):
                self.dialogue_entries = []

            def _log_dialogue_entry(self, agent_name, entry):
                self.dialogue_entries.append((agent_name, entry))

        class FakeOrchestratorForMissingConnector:
            _pause_for_unavailable_connector = Orchestrator._pause_for_unavailable_connector

            def __init__(self, station):
                self.station = station
                self.pause_calls = []

            def _get_current_connector_for_agent(self, agent_name):
                return None

            def _push_log_event(self, event_type, payload):
                pass

            def _trigger_pause_due_to_llm_error(self, agent_name, error, error_type):
                self.pause_calls.append((agent_name, str(error), error_type))

        station = FakeStationForMissingConnector()
        orchestrator = FakeOrchestratorForMissingConnector(station)

        parallel_result = ParallelTickRunner(orchestrator)._send_llm_staged(
            agent_name="Agent A",
            prompt="observation",
            current_tick=7,
            prompt_type="observation",
            internal_loop_step=None,
        )

        self.assertEqual((None, False, None, None), parallel_result)
        self.assertEqual(
            [
                ("Agent A", "SYSTEM_ERROR: No LLM connector for Agent A.", "LLM_CONNECTOR_UNAVAILABLE"),
            ],
            orchestrator.pause_calls,
        )
        self.assertEqual(
            ["llm_connector_error"],
            [entry[1]["type"] for entry in station.dialogue_entries],
        )

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
            constants.AGENT_TOKEN_BUDGET_STALE_REASON_KEY: constants.TOKEN_BUDGET_STALE_REASON_PROVIDER_COUNT_UNAVAILABLE_AFTER_CONTEXT_FILTER,
        })
        stale_observation, stale_error = stale_station.request_status("Agent A")
        self.assertIsNone(stale_error)
        self.assertNotIn("Agent Token Budget", stale_observation)
        saved = agent_module.load_agent_data("Agent A")
        self.assertEqual([], saved.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, []))

    def test_request_status_reuses_pending_observation_until_response_commit(self):
        agent_name = "Agent A"
        agent_module.save_agent_data(
            agent_name,
            {
                constants.AGENT_NAME_KEY: agent_name,
                constants.AGENT_DESCRIPTION_KEY: "desc",
                constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                constants.AGENT_TOKEN_BUDGET_CURRENT_KEY: 10,
                constants.AGENT_TOKEN_BUDGET_MAX_KEY: 100,
                constants.AGENT_CURRENT_LOCATION_KEY: constants.ROOM_LOBBY,
                constants.AGENT_ROOM_OUTPUT_HISTORY_KEY: [],
                constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
                constants.AGENT_SHOWN_NOTIFICATIONS_KEY: [],
                constants.AGENT_SESSION_ENDED_KEY: False,
                constants.AGENT_IS_ASCENDED_KEY: False,
            },
        )

        class RetryHelpRoom(BaseRoom):
            def __init__(self):
                super().__init__(constants.ROOM_LOBBY)

            def build_ascension_system_message(self, _agent_data, _room_context):
                return None

            def _get_specific_room_content(self, _agent_data, _room_context, _current_tick):
                return "Lobby body after help."

            def handle_action(self, *_args, **_kwargs):
                return [], None

            def get_help_message(self, _agent_data, _room_context):
                return "Retry-visible lobby help."

        class NoActionParser:
            def parse(self, _response_text):
                return []

        station = object.__new__(Station)
        station.agent_module = agent_module
        station.capsule_module = None
        station.config = {
            constants.STATION_CONFIG_CURRENT_TICK: 10,
            constants.STATION_CONFIG_STATION_STATUS: "Testing",
        }
        room = RetryHelpRoom()
        station.rooms = {constants.ROOM_LOBBY: room}
        station.room_context = RoomContext(
            agent_manager=agent_module,
            capsule_manager=None,
            notification_manager=None,
            constants_module=constants,
            station_instance=station,
        )
        station.action_parser = NoActionParser()
        station._get_current_tick = lambda: 10
        station.is_holiday_tick = lambda _tick: False
        station._check_and_apply_inactivity_warning = lambda data, _tick: data
        station._check_and_apply_meta_reflection_warning = lambda data, _tick: data
        station._check_and_apply_life_warnings = lambda data, _tick: data
        station._check_and_apply_supervisor_report_reminder = lambda data, _tick: data
        station._get_agent_age_status = lambda _data, _tick: None
        station._is_agent_inactive = lambda _data: False
        station._update_meta_reflection_tick_count = lambda data, _tick: data
        station._log_dialogue_entry = lambda _agent_name, _entry: None

        first_observation, first_error = station.request_status(agent_name)
        self.assertIsNone(first_error)
        self.assertIn("Retry-visible lobby help.", first_observation)

        saved_after_first_render = agent_module.load_agent_data(agent_name)
        self.assertTrue(
            saved_after_first_render[constants.SHORT_ROOM_NAME_LOBBY][
                constants.AGENT_ROOM_STATE_FIRST_VISIT_HELP_SHOWN_KEY
            ]
        )
        self.assertEqual(
            10,
            saved_after_first_render[constants.AGENT_PENDING_OBSERVATION_TICK_KEY],
        )

        second_observation, second_error = station.request_status(agent_name)
        self.assertIsNone(second_error)
        self.assertEqual(first_observation, second_observation)
        self.assertIn("Retry-visible lobby help.", second_observation)

        station.submit_response(agent_name, "No action this turn.", 10)
        saved_after_commit = agent_module.load_agent_data(agent_name)
        self.assertNotIn(constants.AGENT_PENDING_OBSERVATION_TICK_KEY, saved_after_commit)
        self.assertNotIn(constants.AGENT_PENDING_OBSERVATION_TEXT_KEY, saved_after_commit)

        third_observation, third_error = station.request_status(agent_name)
        self.assertIsNone(third_error)
        self.assertNotIn("Retry-visible lobby help.", third_observation)

    def test_fresh_token_update_clears_stale_budget_flags(self):
        agent_module.save_agent_data(
            "Agent A",
            {
                constants.AGENT_NAME_KEY: "Agent A",
                constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                constants.AGENT_TOKEN_BUDGET_CURRENT_KEY: 95,
                constants.AGENT_TOKEN_BUDGET_MAX_KEY: 100,
                constants.AGENT_TOKEN_BUDGET_CURRENT_STALE_KEY: True,
                constants.AGENT_TOKEN_BUDGET_STALE_REASON_KEY: constants.TOKEN_BUDGET_STALE_REASON_PROVIDER_COUNT_UNAVAILABLE_AFTER_CONTEXT_FILTER,
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

    def test_parallel_prepare_marks_unavailable_service_filter_count_stale(self):
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
            constants.TOKEN_BUDGET_STALE_REASON_PROVIDER_COUNT_UNAVAILABLE_AFTER_CONTEXT_FILTER,
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
