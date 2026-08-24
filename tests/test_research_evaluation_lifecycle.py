import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from station import constants
from station.eval_research.auto_evaluator import AutoResearchEvaluator
from station.eval_research.evaluation_manager import EvaluationManager
from station.eval_research.restart_evaluations import (
    requeue_instruction_evaluations,
    requeue_unfinished_instruction_evaluations,
    restart_stuck_evaluations,
)
from station.station import Station
from station.station_runner import Orchestrator


class ResearchEvaluationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self._station_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._station_tmp.cleanup)
        root = Path(self._station_tmp.name)
        station_data = root / "station_data"
        index_dir = root / "index"
        station_data.mkdir()
        index_dir.mkdir()

        env_patcher = mock.patch.dict(os.environ, {"STATION_INDEX_DB_DIR": str(index_dir)}, clear=False)
        base_path_patcher = mock.patch.object(constants, "BASE_STATION_DATA_PATH", str(station_data))
        env_patcher.start()
        base_path_patcher.start()
        self.addCleanup(env_patcher.stop)
        self.addCleanup(base_path_patcher.stop)

    def test_missing_required_research_evaluator_raises_at_tick_boundary(self):
        station = Station.__new__(Station)
        station.auto_research_evaluator = None

        with mock.patch.object(constants, "AUTO_EVAL_RESEARCH", True), \
             mock.patch.object(constants, "RESEARCH_CENTER_ENABLED", True):
            with self.assertRaisesRegex(RuntimeError, "missing or stopped"):
                station.should_wait_for_research_evaluations_at_tick_boundary()

    def test_research_evaluator_start_failure_raises(self):
        evaluator = SimpleNamespace(is_running=False, start_evaluation_loop=lambda: False)
        station = Station.__new__(Station)
        station.auto_research_evaluator = evaluator

        with mock.patch.object(constants, "AUTO_EVAL_RESEARCH", True):
            with self.assertRaisesRegex(RuntimeError, "failed to start"):
                station.start_auto_research_evaluator()

        self.assertIsNone(station.auto_research_evaluator)

    def test_orchestrator_resume_wait_uses_research_tick_boundary_not_any_pending_eval(self):
        class FakeStation:
            def __init__(self, boundary_wait):
                self.boundary_wait = boundary_wait

            def should_wait_for_research_evaluations_at_tick_boundary(self):
                return self.boundary_wait

            def has_pending_research_evaluations(self):
                return True

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.station = FakeStation(boundary_wait=False)
        orchestrator.waiting_reasons = {}

        should_wait, reasons = Orchestrator._check_automatic_wait_conditions(orchestrator)
        self.assertFalse(should_wait)
        self.assertEqual(reasons, {})

        orchestrator.station.boundary_wait = True
        should_wait, reasons = Orchestrator._check_automatic_wait_conditions(orchestrator)
        self.assertTrue(should_wait)
        self.assertEqual(reasons, {"research_tick_boundary": "Research evaluations at tick limit"})

        orchestrator.waiting_reasons = reasons
        self.assertFalse(Orchestrator._check_wait_conditions_resolved(orchestrator))
        orchestrator.station.boundary_wait = False
        self.assertTrue(Orchestrator._check_wait_conditions_resolved(orchestrator))

    def test_main_loop_does_not_pre_wait_for_research_between_normal_ticks(self):
        class FakeStation:
            def should_wait_for_research_evaluations_at_tick_boundary(self):
                return True

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.station = FakeStation()
        orchestrator.is_running = False
        orchestrator.is_paused = False
        orchestrator.is_waiting = False
        orchestrator.waiting_reasons = {}
        orchestrator.wait_check_interval = 0
        orchestrator.pause_event = threading.Event()
        orchestrator.agent_turn_order = ["Agent A"]
        orchestrator._check_wait_conditions_before_next_tick = False
        orchestrator.events = []

        orchestrator._push_log_event = lambda event_type, payload: orchestrator.events.append((event_type, payload))
        orchestrator._check_wait_conditions_resolved = lambda: True
        orchestrator._exit_waiting_state = lambda: None
        orchestrator._enter_waiting_state = lambda reasons: orchestrator.events.append(("entered_wait", reasons))
        orchestrator.initialize_connectors_for_active_agents = lambda: None

        def run_single_tick():
            orchestrator.events.append(("tick", {}))
            orchestrator.is_running = False
            return True

        orchestrator.run_single_tick = run_single_tick

        Orchestrator.main_loop(orchestrator)

        self.assertIn(("tick", {}), orchestrator.events)
        self.assertNotIn(("entered_wait", {"research_tick_boundary": "Research evaluations at tick limit"}), orchestrator.events)

    def test_main_loop_resume_guard_pre_waits_for_research_boundary_once(self):
        class FakeStation:
            def should_wait_for_research_evaluations_at_tick_boundary(self):
                return True

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.station = FakeStation()
        orchestrator.is_running = False
        orchestrator.is_paused = False
        orchestrator.is_waiting = False
        orchestrator.waiting_reasons = {}
        orchestrator.wait_check_interval = 0
        orchestrator.pause_event = threading.Event()
        orchestrator.agent_turn_order = ["Agent A"]
        orchestrator._check_wait_conditions_before_next_tick = True
        orchestrator.events = []

        orchestrator._push_log_event = lambda event_type, payload: orchestrator.events.append((event_type, payload))
        orchestrator._check_wait_conditions_resolved = lambda: True
        orchestrator._exit_waiting_state = lambda: None

        def enter_waiting_state(reasons):
            orchestrator.events.append(("entered_wait", reasons))
            orchestrator.is_running = False

        orchestrator._enter_waiting_state = enter_waiting_state
        orchestrator.initialize_connectors_for_active_agents = lambda: None
        orchestrator.run_single_tick = lambda: orchestrator.events.append(("tick", {}))

        Orchestrator.main_loop(orchestrator)

        self.assertIn(("entered_wait", {"research_tick_boundary": "Research evaluations at tick limit"}), orchestrator.events)
        self.assertNotIn(("tick", {}), orchestrator.events)
        self.assertFalse(orchestrator._check_wait_conditions_before_next_tick)

    def test_restart_requeues_no_report_resume_limit_blocked_eval_and_resets_resume_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evaluations_dir = root / "evaluations"
            run_requests_dir = root / "run_requests"
            coder_sessions_dir = root / "coder_sessions"
            evaluations_dir.mkdir()
            run_requests_dir.mkdir()
            coder_sessions_dir.mkdir()

            eval_manager = EvaluationManager(str(evaluations_dir))
            eval_manager.create_evaluation(
                eval_id="29",
                author="Confluence I",
                title="Resume limit regression",
                content="Run one experiment.",
                tick=14,
                tags=["regression"],
                abstract="No-report resume-limit failures should be retried on restart.",
                lineage="confluence",
            )
            eval_manager.update_evaluation(
                "29",
                lambda record: (
                    record.update({"status": "blocked", "final": {}}),
                    record.setdefault("coder", {}).update(
                        {
                            "active": False,
                            "status": "blocked",
                            "spawn_count": 1,
                            "resume_count": 10,
                            "max_resumes": 10,
                            "session_id": "codex_29_spawn_1_deadbeef",
                            "resume_token": "resume-token",
                            "failure_category": "resume_limit_exceeded",
                            "last_error": "resume limit exceeded",
                        }
                    ),
                ),
            )

            paths = SimpleNamespace(
                evaluations_dir=str(evaluations_dir),
                run_requests_dir=str(run_requests_dir),
                coder_sessions_dir=str(coder_sessions_dir),
                research_root=str(root),
            )

            recovered = restart_stuck_evaluations(eval_manager=eval_manager, paths=paths)
            record = eval_manager.get_evaluation("29")

            self.assertEqual(recovered, 1)
            self.assertEqual(record["status"], "queued")
            self.assertEqual(record["coder"]["status"], "queued")
            self.assertFalse(record["coder"]["active"])
            self.assertEqual(record["coder"]["spawn_count"], 0)
            self.assertEqual(record["coder"]["resume_count"], 0)
            self.assertEqual(record["coder"]["max_resumes"], constants.RESEARCH_CODER_MAX_RESUMES)
            self.assertIsNone(record["coder"]["session_id"])
            self.assertIsNone(record["coder"]["resume_token"])

    def test_restart_requeues_failed_no_report_instruction_eval(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evaluations_dir = root / "evaluations"
            run_requests_dir = root / "run_requests"
            coder_sessions_dir = root / "coder_sessions"
            evaluations_dir.mkdir()
            run_requests_dir.mkdir()
            coder_sessions_dir.mkdir()

            eval_manager = EvaluationManager(str(evaluations_dir))
            eval_manager.create_evaluation(
                eval_id="30",
                author="Confluence I",
                title="Failed no-report regression",
                content="Run one experiment.",
                tick=14,
                tags=["regression"],
                abstract="No-report failed records should be retried on restart.",
                lineage="confluence",
            )
            eval_manager.update_evaluation(
                "30",
                lambda record: (
                    record.update({"status": "failed", "final": {}}),
                    record.setdefault("coder", {}).update(
                        {
                            "active": False,
                            "status": "failed",
                            "spawn_count": 2,
                            "resume_count": 4,
                            "failure_category": "launch_error",
                            "last_error": "launcher failed",
                        }
                    ),
                ),
            )

            paths = SimpleNamespace(
                evaluations_dir=str(evaluations_dir),
                run_requests_dir=str(run_requests_dir),
                coder_sessions_dir=str(coder_sessions_dir),
                research_root=str(root),
            )

            recovered = restart_stuck_evaluations(eval_manager=eval_manager, paths=paths)
            record = eval_manager.get_evaluation("30")

            self.assertEqual(recovered, 1)
            self.assertEqual(record["status"], "queued")
            self.assertEqual(record["coder"]["status"], "queued")
            self.assertEqual(record["coder"]["spawn_count"], 0)
            self.assertEqual(record["coder"]["resume_count"], 0)

    def test_unfinished_requeue_uses_index_without_listing_evaluation_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evaluations_dir = root / "evaluations"
            run_requests_dir = root / "run_requests"
            coder_sessions_dir = root / "coder_sessions"
            evaluations_dir.mkdir()
            run_requests_dir.mkdir()
            coder_sessions_dir.mkdir()

            eval_manager = EvaluationManager(str(evaluations_dir))
            eval_manager.create_evaluation(
                eval_id="31",
                author="Confluence I",
                title="Indexed failed no-report regression",
                content="Run one experiment.",
                tick=14,
                tags=["regression"],
                abstract="No-report failed records should be discovered from the index.",
                lineage="confluence",
            )
            eval_manager.update_evaluation(
                "31",
                lambda record: (
                    record.update({"status": "failed", "final": {}}),
                    record.setdefault("coder", {}).update(
                        {
                            "active": False,
                            "status": "failed",
                            "spawn_count": 2,
                            "resume_count": 4,
                            "failure_category": "launch_error",
                            "last_error": "launcher failed",
                        }
                    ),
                ),
            )

            paths = SimpleNamespace(
                evaluations_dir=str(evaluations_dir),
                run_requests_dir=str(run_requests_dir),
                coder_sessions_dir=str(coder_sessions_dir),
                research_root=str(root),
            )

            real_listdir = __import__("os").listdir

            def guarded_listdir(path):
                if Path(path) == evaluations_dir:
                    raise AssertionError("restart recovery must not list evaluation YAML files")
                return real_listdir(path)

            with mock.patch("station.eval_research.restart_evaluations.os.listdir", side_effect=guarded_listdir):
                recovered = requeue_unfinished_instruction_evaluations(
                    reason="indexed restart discovery test",
                    include_active_statuses=False,
                    eval_manager=eval_manager,
                    paths=paths,
                )

            record = eval_manager.get_evaluation("31")
            self.assertEqual(recovered, 1)
            self.assertEqual(record["status"], "queued")
            self.assertEqual(record["coder"]["status"], "queued")

    def test_shutdown_requeue_skips_evaluation_manager_when_no_running_instruction_evals(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evaluations_dir = root / "evaluations"
            run_requests_dir = root / "run_requests"
            coder_sessions_dir = root / "coder_sessions"
            evaluations_dir.mkdir()
            run_requests_dir.mkdir()
            coder_sessions_dir.mkdir()

            giant_payload = "x" * (1024 * 1024)
            (evaluations_dir / "293.yaml").write_text(
                "\n".join(
                    [
                        "schema_version: 2",
                        "id: '293'",
                        "title: giant completed eval",
                        "instruction: |-",
                        "  do something",
                        "status: completed",
                        "final:",
                        f"  stdout: '{giant_payload}'",
                    ]
                ),
                encoding="utf-8",
            )

            paths = SimpleNamespace(
                evaluations_dir=str(evaluations_dir),
                run_requests_dir=str(run_requests_dir),
                coder_sessions_dir=str(coder_sessions_dir),
                research_root=str(root),
            )

            with mock.patch("station.eval_research.restart_evaluations.EvaluationManager") as manager_cls:
                count = requeue_instruction_evaluations(
                    reason="shutdown test",
                    kill_running_coders=False,
                    paths=paths,
                )

            self.assertEqual(count, 0)
            manager_cls.assert_not_called()

    def test_run_request_survives_attempt_start_metadata_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            request_path = root / "64_attempt_1.yaml"
            request_path.write_text("eval_id: '64'\nattempt: 1\n", encoding="utf-8")

            class FailingEvalManager:
                def get_evaluation(self, eval_id):
                    return {
                        "id": str(eval_id),
                        "instruction": "Run one official attempt.",
                        "attempts": [
                            {
                                "attempt": 1,
                                "submission_path": str(root / "64.py"),
                            }
                        ],
                    }

                def mark_attempt_running(self, *_args, **_kwargs):
                    raise TimeoutError("index lock unavailable")

            evaluator = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
            evaluator.eval_manager = FailingEvalManager()
            evaluator.station = SimpleNamespace(_get_current_tick=lambda: 7)

            with self.assertRaises(TimeoutError):
                AutoResearchEvaluator._execute_run_request(evaluator, "64", 1, str(request_path))

            self.assertTrue(request_path.exists())


if __name__ == "__main__":
    unittest.main()
