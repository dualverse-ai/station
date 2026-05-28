from __future__ import annotations

import os
import tempfile
import types
import unittest
from unittest import mock

from station import constants
from station.base_room import RoomContext
from station.eval_research.auto_evaluator import AutoResearchEvaluator
from station.eval_research.evaluation_manager import EvaluationManager, build_evaluation_previews
from station.eval_research.runtime_paths import ensure_runtime_layout
from station.lineage_evolution import LineageEvolutionManager
from station.rooms.research_center import ResearchCenter
from station.station import Station
from station.supervisor_utils import build_current_station_sota_lines


class _AgentManagerStub:
    def __init__(self):
        self.notifications = []

    def add_pending_notification(self, agent_data, message):
        self.notifications.append(message)


class ResearchCenterInterfacesTest(unittest.TestCase):
    def test_attempt_logs_include_running_banner_and_informative_completion_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                evaluator = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
                evaluator.paths = paths
                evaluator.timeout = 900

                banner = AutoResearchEvaluator._initialize_attempt_logs(
                    evaluator,
                    "99",
                    1,
                    "function",
                    cpu_ids=[0, 1],
                    gpu_ids=None,
                )
                stdout_text, stderr_text = AutoResearchEvaluator._write_attempt_logs(
                    evaluator,
                    "99",
                    "worker output\n",
                    "",
                    1.5,
                    {"Message": "ok", "metric": "7"},
                    safe_to_resubmit=True,
                    prelude_stdout=banner,
                )

                with open(os.path.join(paths.stdout_dir, "99.log"), "r", encoding="utf-8") as handle:
                    persisted_stdout = handle.read()

            self.assertIn("ATTEMPT_RUNNING", persisted_stdout)
            self.assertIn("Official evaluation attempt 1 for evaluation 99 has started.", persisted_stdout)
            self.assertIn("Stdout and stderr will be appended here as they are produced by the worker.", persisted_stdout)
            self.assertIn("Some submissions buffer their own output, so a quiet log does not by itself mean the evaluation is stalled.", persisted_stdout)
            self.assertIn("worker output", persisted_stdout)
            self.assertIn(
                "Official evaluation finished. The score and secondary metrics below are authoritative.",
                persisted_stdout,
            )
            self.assertIn("ATTEMPT_COMPLETE", persisted_stdout)
            self.assertIn("PRIMARY_SCORE: 1.5", persisted_stdout)
            self.assertEqual(stdout_text, persisted_stdout)
            self.assertEqual(stderr_text, "")

    def test_public_payload_and_top_submission_work_for_new_eval_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")
            callback_updates = []

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)
                eval_manager.set_top_submission_callback(lambda submission: callback_updates.append(submission))

                eval_manager.create_evaluation(
                    eval_id="21",
                    author="Veritas I",
                    title="Canonical interface test",
                    content="Implement the canonical interface flow.",
                    tick=10,
                    tags=["api"],
                    abstract="Exercise public evaluation accessors.",
                    lineage="veritas",
                )
                eval_manager.register_attempt("21", "print('hello world')\n", "storage/submission/21.py")
                eval_manager.complete_attempt(
                    "21",
                    1,
                    success=True,
                    score=2.5,
                    stdout="hello world\n",
                    stderr="warning\n",
                    details={"Message": "verified", "metric": "7"},
                    sort_key=(2.5,),
                    status="completed",
                )
                eval_manager.finalize_evaluation(
                    "21",
                    "# Coder Report\n\n## Summary\nDone.\n\n## Final Status\nsuccess\n",
                )

                summary = eval_manager.get_result_summary("21")
                payload = eval_manager.get_submission_payload("21")
                top_submission = eval_manager.get_top_submission()
                preview = build_evaluation_previews(["21"], paths.evaluations_dir)[0]

            self.assertEqual(summary["author"], "Veritas I")
            self.assertTrue(summary["success"])
            self.assertEqual(summary["score"], 2.5)
            self.assertEqual(payload["code"], "print('hello world')\n")
            self.assertIn("STDOUT:\nhello world", payload["logs"])
            self.assertIn("STDERR:\nwarning", payload["logs"])
            self.assertEqual(payload["instruction"], "Implement the canonical interface flow.")
            self.assertEqual(top_submission["evaluation_id"], "21")
            self.assertEqual(top_submission["score"], 2.5)
            self.assertTrue(callback_updates)
            self.assertEqual(callback_updates[-1]["evaluation_id"], "21")
            self.assertEqual(callback_updates[-1]["score"], 2.5)
            self.assertEqual(top_submission["tags"], ["api"])
            self.assertEqual(top_submission["abstract"], "Exercise public evaluation accessors.")
            self.assertIn("Canonical interface test", preview)
            self.assertIn("Author: Veritas I", preview)
            self.assertIn("Abstract: Exercise public evaluation accessors.", preview)

    def test_supervisor_fallback_and_stats_use_top_submission_without_display_lookup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="22",
                    author="Veritas II",
                    title="SOTA fallback test",
                    content="Exercise top submission consumers.",
                    tick=11,
                    tags=["fallback", "supervisor"],
                    abstract="Ensure fallback consumers retain Research Center context.",
                    lineage="veritas",
                )
                eval_manager.register_attempt("22", "print('fallback')\n", "storage/submission/22.py")
                eval_manager.complete_attempt(
                    "22",
                    1,
                    success=True,
                    score=3.0,
                    stdout="fallback\n",
                    stderr="",
                    details={"Message": "ok", "metric": "9"},
                    sort_key=(3.0,),
                    status="completed",
                )
                eval_manager.finalize_evaluation("22", "# Coder Report\n\n## Final Status\nsuccess\n")

                top_submission = eval_manager.get_top_submission()
                stats = eval_manager.get_evaluation_statistics()
                sota_lines = build_current_station_sota_lines(top_submission, None, constants, current_tick=20)

            self.assertEqual(stats["top_submission"]["evaluation_id"], "22")
            self.assertEqual(stats["top_submission"]["tags"], ["fallback", "supervisor"])
            self.assertIn("- Title: SOTA fallback test", sota_lines)
            self.assertIn("- Author: Veritas II", sota_lines)
            self.assertIn("- Tags: [fallback, supervisor]", sota_lines)

    def test_tick_boundary_wait_uses_running_eval_without_breaking_top_submission(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_EVAL_MAX_TICK", 2):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="40",
                    author="Veritas III",
                    title="Completed control",
                    content="Produce an initial scored result.",
                    tick=5,
                    tags=["control"],
                    abstract="Scored control evaluation.",
                    lineage="veritas",
                )
                eval_manager.register_attempt("40", "print('control')\n", "storage/submission/40.py")
                eval_manager.complete_attempt(
                    "40",
                    1,
                    success=True,
                    score=4.0,
                    stdout="control\n",
                    stderr="",
                    details={"Message": "ok"},
                    sort_key=(4.0,),
                    status="completed",
                )
                eval_manager.finalize_evaluation("40", "# Coder Report\n\n## Final Status\nsuccess\n")

                eval_manager.create_evaluation(
                    eval_id="41",
                    author="Logos III",
                    title="Running evaluation",
                    content="Stay active across the tick boundary.",
                    tick=9,
                    tags=["running"],
                    abstract="Active evaluation for wait testing.",
                    lineage="logos",
                )
                def mark_running(record):
                    record["status"] = "running"
                    record.setdefault("coder", {})["status"] = "coder_running"
                    record["coder"]["active"] = True

                eval_manager.update_evaluation("41", mark_running)

                stats = eval_manager.get_evaluation_statistics()
                should_wait = eval_manager.should_wait_at_tick(current_tick=10)

            self.assertTrue(should_wait)
            self.assertEqual(stats["top_submission"]["evaluation_id"], "40")
            self.assertEqual(stats["running_count"], 1)
            self.assertEqual(stats["running_evaluations"][0]["evaluation_id"], "41")

    def test_statistics_and_wait_checks_use_sqlite_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_EVAL_MAX_TICK", 2):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="50",
                    author="Cache I",
                    title="Completed cache control",
                    content="Produce one completed result.",
                    tick=4,
                    tags=["cache"],
                    abstract="Completed cache control.",
                    lineage="cache",
                )
                eval_manager.register_attempt("50", "print('done')\n", "storage/submission/50.py")
                eval_manager.complete_attempt(
                    "50",
                    1,
                    success=True,
                    score=7.0,
                    stdout="done\n",
                    stderr="",
                    details={"Message": "ok"},
                    sort_key=(7.0,),
                    status="completed",
                )
                eval_manager.finalize_evaluation("50", "# Coder Report\n\n## Final Status\nsuccess\n")

                eval_manager.create_evaluation(
                    eval_id="51",
                    author="Cache II",
                    title="Queued cache case",
                    content="Stay queued for cache access.",
                    tick=10,
                    tags=["cache", "queued"],
                    abstract="Queued cache case.",
                    lineage="cache",
                )

                with mock.patch.object(
                    eval_manager,
                    "_load_evaluation_any",
                    side_effect=AssertionError("statistics path should not reload YAML"),
                ):
                    stats = eval_manager.get_evaluation_statistics()
                    active_ids = eval_manager.get_active_eval_ids_for_author("Cache II")
                    should_wait = eval_manager.should_wait_at_tick(current_tick=11)

            self.assertEqual(stats["top_submission"]["evaluation_id"], "50")
            self.assertEqual(stats["running_count"], 0)
            self.assertEqual(stats["queued_count"], 1)
            self.assertEqual(stats["queued_evaluations"][0]["evaluation_id"], "51")
            self.assertEqual(active_ids, ["51"])
            self.assertTrue(should_wait)

    def test_final_status_success_is_normalized_to_completed_and_does_not_block_tick(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_EVAL_MAX_TICK", 2):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="43",
                    author="Veritas V",
                    title="Status normalization test",
                    content="Finish cleanly.",
                    tick=10,
                    tags=["status"],
                    abstract="Final status success should normalize to completed.",
                    lineage="veritas",
                )
                eval_manager.register_attempt("43", "print('ok')\n", "storage/submission/43.py")
                eval_manager.complete_attempt(
                    "43",
                    1,
                    success=True,
                    score=6.0,
                    stdout="ok\n",
                    stderr="",
                    details={"Message": "ok"},
                    sort_key=(6.0,),
                    status="completed",
                )
                eval_manager.finalize_evaluation(
                    "43",
                    "# Coder Report\n\n## Final Status\nsuccess\n",
                )

                record = eval_manager.get_evaluation("43")
                should_wait = eval_manager.should_wait_at_tick(current_tick=11)

            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["coder"]["status"], "completed")
            self.assertEqual(record["final"]["status"], "completed")
            self.assertFalse(should_wait)

    def test_complete_attempt_does_not_clobber_terminal_status_after_finalize(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_EVAL_MAX_TICK", 2):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="44",
                    author="Veritas VI",
                    title="Late completion race test",
                    content="Reproduce finalize-before-complete race.",
                    tick=10,
                    tags=["race"],
                    abstract="Late complete_attempt must not revert terminal status.",
                    lineage="veritas",
                )
                eval_manager.register_attempt("44", "print('a1')\n", "storage/submission/44.py")
                eval_manager.complete_attempt(
                    "44",
                    1,
                    success=True,
                    score=0.0,
                    stdout="a1\n",
                    stderr="",
                    details={"Message": "ok"},
                    sort_key=(0.0,),
                    status="completed",
                )
                eval_manager.register_attempt("44", "print('a2')\n", "storage/submission/44.py")
                eval_manager.mark_attempt_running("44", 2)
                eval_manager.update_evaluation(
                    "44",
                    lambda record: record.setdefault("coder", {}).update({"active": False, "status": "partial"}),
                )
                eval_manager.finalize_evaluation(
                    "44",
                    "# Coder Report\n\n## Final Status\npartial\n",
                    final_status="partial",
                )
                eval_manager.complete_attempt(
                    "44",
                    2,
                    success=True,
                    score=0.0,
                    stdout="a2\n",
                    stderr="",
                    details={"Message": "ok"},
                    sort_key=(0.0,),
                    status="completed",
                )

                record = eval_manager.get_evaluation("44")
                should_wait = eval_manager.should_wait_at_tick(current_tick=11)

            self.assertEqual(record["status"], "partial")
            self.assertEqual(record["final"]["status"], "partial")
            self.assertEqual(record["coder"]["status"], "partial")
            self.assertFalse(should_wait)

    def test_active_elapsed_uses_coder_start_even_while_attempt_is_running(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="45",
                    author="Veritas VII",
                    title="Elapsed timer source test",
                    content="Use coder start time while running.",
                    tick=10,
                    tags=["elapsed"],
                    abstract="Attempt start must not reset displayed elapsed time.",
                    lineage="veritas",
                )
                eval_manager.update_evaluation(
                    "45",
                    lambda record: record.setdefault("coder", {}).update(
                        {
                            "active": True,
                            "status": "running",
                            "started_timestamp": 100.0,
                        }
                    ),
                )
                eval_manager.register_attempt("45", "print('ok')\n", "storage/submission/45.py")
                eval_manager.update_evaluation(
                    "45",
                    lambda record: record["attempts"][-1].update(
                        {
                            "status": "running",
                            "started_timestamp": 250.0,
                        }
                    ),
                )
                def mark_attempt_running(record):
                    record["status"] = "running"
                    record.setdefault("coder", {})["status"] = "attempt_running"
                    record["coder"]["active"] = True

                eval_manager.update_evaluation("45", mark_attempt_running)

                with mock.patch("station.eval_research.evaluation_manager.time.time", return_value=400.0):
                    active = eval_manager.get_active_evaluations()

            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["start_timestamp"], 100.0)
            self.assertEqual(active[0]["elapsed_seconds"], 300)

    def test_pending_notification_and_queued_indexes_update_without_full_rescan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="60",
                    author="Queue I",
                    title="Queued instruction index",
                    content="Stay queued.",
                    tick=1,
                    tags=["queue"],
                    abstract="Queued instruction index.",
                    lineage="queue",
                )
                self.assertEqual(eval_manager.get_queued_instruction_eval_ids(), ["60"])

                eval_manager.set_coder_spawned("60", session_id="codex_60_spawn_1_deadbeef", pid=1234)
                self.assertEqual(eval_manager.get_queued_instruction_eval_ids(), [])

                eval_manager.create_evaluation(
                    eval_id="61",
                    author="Queue II",
                    title="Pending notification index",
                    content="Finish without callback.",
                    tick=2,
                    tags=["notify"],
                    abstract="Pending notification index.",
                    lineage="queue",
                )
                eval_manager.register_attempt("61", "print('ok')\n", "storage/submission/61.py")
                eval_manager.complete_attempt(
                    "61",
                    1,
                    success=True,
                    score=1.0,
                    stdout="ok\n",
                    stderr="",
                    details={"Message": "ok"},
                    sort_key=(1.0,),
                    status="completed",
                )
                eval_manager.finalize_evaluation("61", "# Coder Report\n\n## Final Status\nsuccess\n")
                self.assertEqual(eval_manager.get_pending_notification_eval_ids(), ["61"])

                eval_manager.update_evaluation(
                    "61",
                    lambda record: record.setdefault("notification", {}).update({"sent": True}),
                )
                self.assertEqual(eval_manager.get_pending_notification_eval_ids(), [])

    def test_indexes_rebuild_from_disk_after_manager_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_EVAL_MAX_TICK", 2):
                paths = ensure_runtime_layout()
                writer = EvaluationManager(paths.evaluations_dir)

                writer.create_evaluation(
                    eval_id="70",
                    author="Restart I",
                    title="Completed before restart",
                    content="Completed before restart.",
                    tick=3,
                    tags=["restart"],
                    abstract="Completed before restart.",
                    lineage="restart",
                )
                writer.register_attempt("70", "print('done')\n", "storage/submission/70.py")
                writer.complete_attempt(
                    "70",
                    1,
                    success=True,
                    score=9.0,
                    stdout="done\n",
                    stderr="",
                    details={"Message": "ok"},
                    sort_key=(9.0,),
                    status="completed",
                )
                writer.finalize_evaluation("70", "# Coder Report\n\n## Final Status\nsuccess\n")

                writer.create_evaluation(
                    eval_id="71",
                    author="Restart II",
                    title="Queued before restart",
                    content="Queued before restart.",
                    tick=10,
                    tags=["restart", "queued"],
                    abstract="Queued before restart.",
                    lineage="restart",
                )

                restarted = EvaluationManager(paths.evaluations_dir)

                with mock.patch.object(
                    restarted,
                    "_load_evaluation_any",
                    side_effect=AssertionError("restart-rebuilt indexes should satisfy hot reads"),
                ):
                    stats = restarted.get_evaluation_statistics()
                    top = restarted.get_top_submission()
                    should_wait = restarted.should_wait_at_tick(current_tick=11)
                    queued_ids = restarted.get_queued_instruction_eval_ids()

            self.assertEqual(top["evaluation_id"], "70")
            self.assertEqual(stats["top_submission"]["evaluation_id"], "70")
            self.assertEqual(stats["running_count"], 0)
            self.assertEqual(stats["queued_count"], 1)
            self.assertEqual(stats["queued_evaluations"][0]["evaluation_id"], "71")
            self.assertTrue(should_wait)
            self.assertEqual(queued_ids, ["71"])

    def test_evaluation_statistics_split_running_and_queued_experiments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="72",
                    author="Queue Split I",
                    title="Queued experiment",
                    content="Queued experiment.",
                    tick=10,
                    tags=["queue"],
                    abstract="Queued experiment.",
                    lineage="split",
                )
                eval_manager.create_evaluation(
                    eval_id="73",
                    author="Queue Split II",
                    title="Running experiment",
                    content="Running experiment.",
                    tick=11,
                    tags=["running"],
                    abstract="Running experiment.",
                    lineage="split",
                )
                eval_manager.update_evaluation(
                    "73",
                    lambda record: record.setdefault("coder", {}).update(
                        {
                            "active": True,
                            "status": "running",
                            "started_timestamp": 100.0,
                        }
                    ),
                )
                def mark_running(record):
                    record["status"] = "running"
                    record.setdefault("coder", {})["status"] = "coder_running"
                    record["coder"]["active"] = True

                eval_manager.update_evaluation("73", mark_running)

                with mock.patch("station.eval_research.evaluation_manager.time.time", return_value=160.0):
                    stats = eval_manager.get_evaluation_statistics()

            self.assertEqual(stats["queued_count"], 1)
            self.assertEqual(stats["queued_evaluations"][0]["evaluation_id"], "72")
            self.assertEqual(stats["queued_evaluations"][0]["status"], "queued")
            self.assertEqual(stats["running_count"], 1)
            self.assertEqual(stats["running_evaluations"][0]["evaluation_id"], "73")
            self.assertEqual(stats["running_evaluations"][0]["status"], "coder_running")

    def test_deferred_finalization_stays_running_without_consuming_coder_slot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="76",
                    author="Deferred I",
                    title="Deferred finalization",
                    content="Keep running until finalization is ready.",
                    tick=12,
                    tags=["deferred"],
                    abstract="Coder slot should be released before finalization.",
                    lineage="deferred",
                )
                eval_manager.register_attempt("76", "print('ok')\n", "storage/submission/76.py")
                eval_manager.mark_attempt_running("76", 1)
                eval_manager.set_coder_exited(
                    "76",
                    exit_code=0,
                    keep_running=True,
                    running_substate="attempt_running",
                )
                eval_manager.complete_attempt(
                    "76",
                    1,
                    success=True,
                    score=1.0,
                    stdout="ok\n",
                    stderr="",
                    details={"Message": "ok"},
                    sort_key=(1.0,),
                    status="completed",
                )

                record = eval_manager.get_evaluation("76")
                stats = eval_manager.get_evaluation_statistics()
                active_ids = eval_manager.get_active_eval_ids_for_author("Deferred I")

            self.assertEqual(record["status"], "running")
            self.assertEqual(record["coder"]["status"], "attempt_running")
            self.assertFalse(record["coder"]["active"])
            self.assertEqual(eval_manager.get_active_coder_count(), 0)
            self.assertEqual(stats["running_count"], 0)
            self.assertEqual(active_ids, ["76"])

    def test_terminal_transition_unlocks_submission_stops_tick_wait_and_sends_notification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")
            notifications = []

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_EVAL_MAX_TICK", 2):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)
                eval_manager.set_notification_callback(lambda author, message: notifications.append((author, message)))

                eval_manager.create_evaluation(
                    eval_id="79",
                    author="Terminal I",
                    title="Exact terminal transition",
                    content="Stay blocked until report and official eval are both done.",
                    tick=20,
                    tags=["terminal"],
                    abstract="Terminal transition should be atomic from the agent perspective.",
                    lineage="terminal",
                )
                eval_manager.register_attempt("79", "print('ok')\n", "storage/submission/79.py")
                eval_manager.mark_attempt_running("79", 1, start_tick=20)
                eval_manager.set_coder_exited(
                    "79",
                    exit_code=0,
                    keep_running=True,
                    running_substate="attempt_running",
                )

                active_before = eval_manager.get_active_eval_ids_for_author("Terminal I")
                self.assertEqual(active_before, ["79"])
                self.assertTrue(eval_manager.should_wait_at_tick(current_tick=21))

                eval_manager.complete_attempt(
                    "79",
                    1,
                    success=True,
                    score=2.0,
                    stdout="ok\n",
                    stderr="",
                    details={"Message": "ok"},
                    sort_key=(2.0,),
                    status="completed",
                )
                eval_manager.finalize_evaluation("79", "# Coder Report\n\n## Final Status\nsuccess\n")

                record = eval_manager.get_evaluation("79")
                active_after = eval_manager.get_active_eval_ids_for_author("Terminal I")
                should_wait_after = eval_manager.should_wait_at_tick(current_tick=21)

            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["final"]["status"], "completed")
            self.assertFalse(record["coder"]["active"])
            self.assertEqual(active_after, [])
            self.assertFalse(should_wait_after)
            self.assertEqual(len(notifications), 1)
            self.assertEqual(notifications[0][0], "Terminal I")
            self.assertIn("Your research submission 'Exact terminal transition' (ID: 79) has completed.", notifications[0][1])

    def test_manager_refreshes_indexes_after_external_requeue_script_style_update(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                live_manager = EvaluationManager(paths.evaluations_dir)
                external_manager = EvaluationManager(paths.evaluations_dir)

                live_manager.create_evaluation(
                    eval_id="80",
                    author="Refresh I",
                    title="External refresh target",
                    content="This eval will be externally requeued.",
                    tick=22,
                    tags=["refresh"],
                    abstract="Live manager should pick up external queue rewrites.",
                    lineage="refresh",
                )
                live_manager.update_evaluation(
                    "80",
                    lambda record: record.update({"status": "running"}),
                )
                live_manager.update_evaluation(
                    "80",
                    lambda record: record.setdefault("coder", {}).update(
                        {
                            "active": True,
                            "status": "coder_running",
                            "active_pid": 9999,
                        }
                    ),
                )

                self.assertEqual(live_manager.get_active_eval_ids_for_author("Refresh I"), ["80"])
                self.assertEqual(live_manager.get_queued_instruction_eval_ids(), [])

                external_manager.update_evaluation(
                    "80",
                    lambda record: (
                        record.update({"status": "queued"}),
                        record.setdefault("coder", {}).update(
                            {
                                "active": False,
                                "active_pid": None,
                                "status": "queued",
                                "last_error": "Manually requeued via external script.",
                            }
                        ),
                    ),
                )

                queued_ids = live_manager.get_queued_instruction_eval_ids()
                active_after = live_manager.get_active_eval_ids_for_author("Refresh I")
                refreshed = live_manager.get_evaluation("80")

            self.assertEqual(queued_ids, ["80"])
            self.assertEqual(active_after, ["80"])
            self.assertEqual(refreshed["status"], "queued")
            self.assertEqual(refreshed["coder"]["status"], "queued")
            self.assertIn("external script", refreshed["coder"]["last_error"])

    def test_launch_gate_uses_active_coder_slots_not_all_running_evals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="77",
                    author="Deferred II",
                    title="Settling eval",
                    content="Still nonterminal but coder already exited.",
                    tick=13,
                    tags=["deferred"],
                    abstract="Should not consume a coder slot.",
                    lineage="deferred",
                )
                eval_manager.update_evaluation(
                    "77",
                    lambda record: record.update({"status": "running"}),
                )
                eval_manager.update_evaluation(
                    "77",
                    lambda record: record.setdefault("coder", {}).update(
                        {
                            "active": False,
                            "status": "attempt_running",
                        }
                    ),
                )

                eval_manager.create_evaluation(
                    eval_id="78",
                    author="Deferred III",
                    title="Queued eval",
                    content="Should still launch because no coder slot is occupied.",
                    tick=14,
                    tags=["queued"],
                    abstract="Queued eval.",
                    lineage="deferred",
                )

                paused_reasons = []
                launched_ids = []
                evaluator = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
                evaluator.eval_manager = eval_manager
                evaluator.max_parallel_workers = 1
                evaluator.coder_manager = types.SimpleNamespace(
                    active_sessions={},
                    launch_for_evaluation=lambda eval_data: launched_ids.append(str(eval_data.get("id"))) or True,
                )
                evaluator._pause_orchestrator = lambda reason: paused_reasons.append(reason)
                evaluator._push_log_event = lambda *_args, **_kwargs: None

                AutoResearchEvaluator._launch_queued_coders(evaluator)

            self.assertEqual(launched_ids, ["78"])
            self.assertEqual(paused_reasons, [])

    def test_attempt_running_with_live_coder_consumes_coder_launch_slot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="81",
                    author="Pipeline I",
                    title="GPU attempt already submitted",
                    content="The coder is polling the official attempt logs.",
                    tick=15,
                    tags=["pipeline"],
                    abstract="Attempt-running should not block coder preparation for queued work.",
                    lineage="pipeline",
                )
                eval_manager.register_attempt("81", "print('queued')\n", "storage/submission/81.py")
                eval_manager.update_evaluation(
                    "81",
                    lambda record: record.setdefault("coder", {}).update(
                        {
                            "active": True,
                            "status": "attempt_running",
                            "active_pid": 12345,
                        }
                    ),
                )

                eval_manager.create_evaluation(
                    eval_id="82",
                    author="Pipeline II",
                    title="Queued coder work",
                    content="This should launch while the previous attempt waits for GPUs.",
                    tick=16,
                    tags=["pipeline"],
                    abstract="Queued coder work.",
                    lineage="pipeline",
                )

                paused_reasons = []
                launched_ids = []
                evaluator = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
                evaluator.eval_manager = eval_manager
                evaluator.max_parallel_workers = 1
                evaluator.coder_manager = types.SimpleNamespace(
                    active_sessions={},
                    launch_for_evaluation=lambda eval_data: launched_ids.append(str(eval_data.get("id"))) or True,
                )
                evaluator._pause_orchestrator = lambda reason: paused_reasons.append(reason)
                evaluator._push_log_event = lambda *_args, **_kwargs: None

                active_coder_count = eval_manager.get_active_coder_count()
                stats = eval_manager.get_evaluation_statistics()
                AutoResearchEvaluator._launch_queued_coders(evaluator)

            self.assertEqual(active_coder_count, 1)
            self.assertEqual(stats["running_count"], 1)
            self.assertEqual(stats["running_evaluations"][0]["evaluation_id"], "81")
            self.assertEqual(stats["running_evaluations"][0]["status"], "attempt_queued")
            self.assertEqual(launched_ids, [])
            self.assertEqual(paused_reasons, [])

    def test_running_attempt_without_live_coder_still_appears_in_running_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="83",
                    author="Pipeline III",
                    title="GPU attempt still running after coder exit",
                    content="The official attempt continues after the coder process exits.",
                    tick=17,
                    tags=["pipeline"],
                    abstract="Running attempt should remain visible in dashboard jobs.",
                    lineage="pipeline",
                )
                eval_manager.register_attempt("83", "print('running')\n", "storage/submission/83.py")
                eval_manager.mark_attempt_running("83", 1)
                eval_manager.update_evaluation(
                    "83",
                    lambda record: record.setdefault("coder", {}).update(
                        {
                            "active": False,
                            "status": "attempt_running",
                            "active_pid": None,
                        }
                    ),
                )

                record = eval_manager.get_evaluation("83")
                active_coder_count = eval_manager.get_active_coder_count()
                stats = eval_manager.get_evaluation_statistics()

            self.assertEqual(record["status"], "running")
            self.assertFalse(record["coder"]["active"])
            self.assertEqual(record["attempts"][-1]["status"], "running")
            self.assertEqual(active_coder_count, 0)
            self.assertEqual(stats["running_count"], 1)
            self.assertEqual(stats["running_evaluations"][0]["evaluation_id"], "83")
            self.assertEqual(stats["running_evaluations"][0]["status"], "attempt_running")

    def test_station_statistics_fallback_reads_research_stats_without_live_evaluator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_CENTER_ENABLED", True):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)
                eval_manager.create_evaluation(
                    eval_id="74",
                    author="Fallback I",
                    title="Queued fallback experiment",
                    content="Still queued on disk.",
                    tick=12,
                    tags=["queue"],
                    abstract="Queued fallback experiment.",
                    lineage="fallback",
                )
                eval_manager.create_evaluation(
                    eval_id="75",
                    author="Fallback II",
                    title="Completed fallback result",
                    content="Completed fallback result.",
                    tick=8,
                    tags=["complete"],
                    abstract="Completed fallback result.",
                    lineage="fallback",
                )
                eval_manager.register_attempt("75", "print('ok')\n", "storage/submission/75.py")
                eval_manager.complete_attempt(
                    "75",
                    1,
                    success=True,
                    score=8.0,
                    stdout="ok\n",
                    stderr="",
                    details={"Message": "ok"},
                    sort_key=(8.0,),
                    status="completed",
                )
                eval_manager.finalize_evaluation("75", "# Coder Report\n\n## Final Status\nsuccess\n")

                station_stub = types.SimpleNamespace(
                    rooms={},
                    config={constants.STATION_CONFIG_AGENT_TURN_ORDER: []},
                    agent_module=types.SimpleNamespace(load_agent_data=lambda *_args, **_kwargs: None),
                    auto_research_evaluator=None,
                )
                stats = Station.get_station_statistics(station_stub)

            self.assertEqual(stats["queued_experiments_count"], 1)
            self.assertEqual(stats["queued_experiments"][0]["evaluation_id"], "74")
            self.assertEqual(stats["running_experiments_count"], 0)
            self.assertEqual(stats["top_research_submission"]["evaluation_id"], "75")

    def test_notification_and_review_formatting_diverge_as_intended(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")
            notifications = []
            long_stdout = "abcde" * 5

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_EVAL_LOG_MAX_CHARS", 12):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)
                eval_manager.set_notification_callback(lambda author, message: notifications.append((author, message)))

                eval_manager.create_evaluation(
                    eval_id="42",
                    author="Veritas IV",
                    title="Formatting contract test",
                    content="Build the formatting contract case.",
                    tick=12,
                    tags=["format"],
                    abstract="This abstract should appear in review only.",
                    lineage="veritas",
                )
                eval_manager.register_attempt("42", "print('format')\n", "storage/submission/42.py")
                eval_manager.complete_attempt(
                    "42",
                    1,
                    success=True,
                    score=5.5,
                    stdout=long_stdout,
                    stderr="",
                    details={"Message": "formatted", "metric": "11"},
                    sort_key=(5.5,),
                    status="completed",
                )
                eval_manager.finalize_evaluation(
                    "42",
                    "# Coder Report\n\n## Summary\nFormatting test.\n\n## Final Status\nsuccess\n",
                )

                review = eval_manager.get_review_info("42")

            self.assertEqual(notifications[0][0], "Veritas IV")
            notification_message = notifications[0][1]
            self.assertIn("**Score:** 5.5", notification_message)
            self.assertIn("**metric:** 11", notification_message)
            self.assertLess(notification_message.index("**Score:** 5.5"), notification_message.index("**Coder Report:**"))
            self.assertLess(notification_message.index("**Coder Report:**"), notification_message.index("**Final Stdout:**"))
            self.assertNotIn("**Instruction Prompt:**", notification_message)
            self.assertNotIn("**Abstract:**", notification_message)
            self.assertIn("[... truncated after 12 characters]", notification_message)

            self.assertIsNotNone(review)
            review_message = review["message"]
            self.assertIn("**Abstract:** This abstract should appear in review only.", review_message)
            self.assertIn("**Instruction Prompt:**", review_message)
            self.assertLess(review_message.index("**Instruction Prompt:**"), review_message.index("**Score:** 5.5"))
            self.assertLess(review_message.index("**Score:** 5.5"), review_message.index("**Coder Report:**"))
            self.assertIn("[... truncated after 12 characters]", review_message)

    def test_lineage_evolution_reads_research_results_through_evaluation_manager(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="30",
                    author="Veritas I",
                    title="First result",
                    content="Baseline run",
                    tick=3,
                    tags=["baseline"],
                    abstract="First result",
                    lineage="veritas",
                )
                eval_manager.register_attempt("30", "print('v1')\n", "storage/submission/30.py")
                eval_manager.complete_attempt(
                    "30",
                    1,
                    success=True,
                    score=1.0,
                    stdout="v1\n",
                    stderr="",
                    details={"Message": "ok"},
                    sort_key=(1.0,),
                    status="completed",
                )
                eval_manager.finalize_evaluation("30", "# Coder Report\n\n## Final Status\nsuccess\n")

                eval_manager.create_evaluation(
                    eval_id="31",
                    author="Logos II",
                    title="Improved result",
                    content="Improved run",
                    tick=4,
                    tags=["improved"],
                    abstract="Improved result",
                    lineage="logos",
                )
                eval_manager.register_attempt("31", "print('v2')\n", "storage/submission/31.py")
                eval_manager.complete_attempt(
                    "31",
                    1,
                    success=True,
                    score=2.0,
                    stdout="v2\n",
                    stderr="",
                    details={"Message": "better"},
                    sort_key=(2.0,),
                    status="completed",
                )
                eval_manager.finalize_evaluation("31", "# Coder Report\n\n## Final Status\nsuccess\n")

                lineage_manager = LineageEvolutionManager(types.SimpleNamespace())
                breakthroughs = lineage_manager._load_all_breakthroughs()

            self.assertEqual(breakthroughs.get("Veritas"), 1)
            self.assertEqual(breakthroughs.get("Logos"), 1)

    def test_storage_list_does_not_consume_read_cooldown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                os.makedirs(paths.shared_storage, exist_ok=True)
                with open(os.path.join(paths.shared_storage, "note.txt"), "w", encoding="utf-8") as handle:
                    handle.write("hello from shared")

                room = ResearchCenter()
                agent_data = {
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_NAME_KEY: "Tester I",
                    constants.AGENT_LINEAGE_KEY: "tester",
                    constants.SHORT_ROOM_NAME_RESEARCH: {},
                }
                agent_manager = _AgentManagerStub()
                room_context = RoomContext(
                    agent_manager=agent_manager,
                    capsule_manager=None,
                    notification_manager=None,
                    constants_module=constants,
                    station_instance=types.SimpleNamespace(),
                )

                result_list_t10, _ = room.handle_action(agent_data, "storage", "list shared", None, room_context, current_tick=10)
                state_after_list = room._get_room_state(agent_data, constants).copy()
                result_read_t10, _ = room.handle_action(agent_data, "read", "shared/note.txt", None, room_context, current_tick=10)
                state_after_read = room._get_room_state(agent_data, constants).copy()
                result_list_t11, _ = room.handle_action(agent_data, "storage", "list shared", None, room_context, current_tick=11)
                result_read_t11, _ = room.handle_action(agent_data, "read", "shared/note.txt", None, room_context, current_tick=11)

            self.assertEqual(result_list_t10, ["Directory listing for storage/shared sent to your System Messages."])
            self.assertEqual(state_after_list, {})
            self.assertEqual(result_read_t10, ["File content for storage/shared/note.txt sent to your System Messages."])
            self.assertEqual(state_after_read[constants.AGENT_RESEARCH_READ_WINDOW_TICK_KEY], 10)
            self.assertEqual(state_after_read[constants.AGENT_RESEARCH_READ_COOLDOWN_UNTIL_TICK_KEY], 16)
            self.assertEqual(result_list_t11, ["Directory listing for storage/shared sent to your System Messages."])
            self.assertEqual(result_read_t11, ["Read cooldown active. `read` and `read_code` will be available again in 5 tick(s)."])

    def test_system_evaluator_symlink_reads_are_allowed_and_do_not_consume_cooldown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                os.makedirs(paths.evaluators_dir, exist_ok=True)
                evaluator_source = os.path.join(paths.evaluators_dir, "evaluator.py")
                with open(evaluator_source, "w", encoding="utf-8") as handle:
                    handle.write("def construct_hadamard():\n    return 'ok'\n")

                room = ResearchCenter()
                agent_data = {
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_NAME_KEY: "Tester I",
                    constants.AGENT_LINEAGE_KEY: "tester",
                    constants.SHORT_ROOM_NAME_RESEARCH: {},
                }
                agent_manager = _AgentManagerStub()
                room_context = RoomContext(
                    agent_manager=agent_manager,
                    capsule_manager=None,
                    notification_manager=None,
                    constants_module=constants,
                    station_instance=types.SimpleNamespace(),
                )

                result_eval, _ = room.handle_action(agent_data, "read", "system/evaluator.py", None, room_context, current_tick=10)
                room_state = room._get_room_state(agent_data, constants).copy()

            self.assertEqual(result_eval, ["File content for storage/system/evaluator.py sent to your System Messages."])
            self.assertEqual(room_state, {})
            self.assertEqual(len(agent_manager.notifications), 1)
            self.assertIn("construct_hadamard", agent_manager.notifications[0])

    def test_station_uses_one_shared_research_eval_manager(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_CENTER_ENABLED", True), \
                 mock.patch.object(constants, "AUTO_EVAL_RESEARCH", False):
                station = Station()
                room = station.rooms[constants.ROOM_RESEARCH_CENTER]
                evaluator = AutoResearchEvaluator(
                    station_instance=station,
                    enabled=False,
                    eval_manager=station.research_eval_manager,
                )

            self.assertIsNotNone(station.research_eval_manager)
            self.assertIs(room.eval_manager, station.research_eval_manager)
            self.assertIs(evaluator.eval_manager, station.research_eval_manager)

    def test_shared_eval_manager_immediately_clears_room_active_lock_after_finalize(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_CENTER_ENABLED", True), \
                 mock.patch.object(constants, "AUTO_EVAL_RESEARCH", False), \
                 mock.patch.object(constants, "RESEARCH_SUBMISSION_COOLDOWN_TICKS", 0):
                station = Station()
                room = station.rooms[constants.ROOM_RESEARCH_CENTER]
                eval_manager = station.research_eval_manager
                agent_data = {
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_NAME_KEY: "Noether I",
                    constants.AGENT_LINEAGE_KEY: "noether",
                    constants.SHORT_ROOM_NAME_RESEARCH: {},
                }

                eval_manager.create_evaluation(
                    eval_id="90",
                    author="Noether I",
                    title="Stale lock regression",
                    content="Run one experiment.",
                    tick=10,
                    tags=["regression"],
                    abstract="Reproduce stale active lock.",
                    lineage="noether",
                )
                active_message = room._build_submission_status_message(agent_data, constants, current_tick=10)

                eval_manager.register_attempt("90", "print('ok')\n", "storage/submission/90.py")
                eval_manager.complete_attempt(
                    "90",
                    1,
                    success=True,
                    score=1.0,
                    stdout="ok\n",
                    stderr="",
                    details={"Message": "ok"},
                    sort_key=(1.0,),
                    status="completed",
                )
                eval_manager.finalize_evaluation("90", "# Coder Report\n\n## Final Status\nsuccess\n")
                ready_message = room._build_submission_status_message(agent_data, constants, current_tick=11)

            self.assertIn("**Active Experiment:**", active_message)
            self.assertIn("90", active_message)
            self.assertEqual(ready_message, "**Submission Ready:** You may submit one focused experiment.")


if __name__ == "__main__":
    unittest.main()
