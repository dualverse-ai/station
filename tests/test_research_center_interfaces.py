from __future__ import annotations

import os
import json
import time
import tempfile
import types
import unittest
from unittest import mock

from station import constants
from station import file_io_utils
from station.base_room import RoomContext
from station.eval_research.cpu_coordinator import CPUCoordinator
from station.eval_research.gpu_coordinator import GPUCoordinator
from station.eval_research.auto_evaluator import (
    AutoResearchEvaluator,
    _managed_gpu_ids,
    _resource_coordination_station_id,
)
from station.eval_research.evaluation_manager import (
    STDOUT_STDERR_HIDDEN_NOTICE,
    EvaluationManager,
    build_evaluation_previews,
)
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
    def test_storage_read_uses_outer_fence_longer_than_embedded_fences(self):
        room = ResearchCenter.__new__(ResearchCenter)
        content = "# Survey\n\n```text\ninside\n```\n\nTail"

        rendered = room._format_storage_read("storage/system/survey.md", content, 1)

        self.assertIn("\n\n````\n# Survey", rendered)
        self.assertIn("```text\ninside\n```", rendered)
        self.assertTrue(rendered.endswith("Tail\n````"))

    def test_storage_read_keeps_triple_fence_for_plain_content(self):
        room = ResearchCenter.__new__(ResearchCenter)

        rendered = room._format_storage_read("storage/shared/note.txt", "plain text", 1)

        self.assertTrue(rendered.endswith("```\nplain text\n```"))

    def test_gpu_num_autodetects_available_gpu_ids(self):
        with mock.patch.object(constants, "RESEARCH_EVAL_GPU_NUM", 2), \
             mock.patch.object(constants, "RESEARCH_EVAL_USE_DIFF_GPU", False), \
             mock.patch.object(constants, "RESEARCH_EVAL_AVAILABLE_GPUS", []), \
             mock.patch("station.eval_research.auto_evaluator._detect_gpu_ids_with_nvidia_smi", return_value=[0, 1, 2]):
            self.assertEqual(_managed_gpu_ids(), [0, 1, 2])

    def test_gpu_num_raises_clear_error_when_autodetect_fails(self):
        with mock.patch.object(constants, "RESEARCH_EVAL_GPU_NUM", 1), \
             mock.patch.object(constants, "RESEARCH_EVAL_USE_DIFF_GPU", False), \
             mock.patch.object(constants, "RESEARCH_EVAL_AVAILABLE_GPUS", []), \
             mock.patch("station.eval_research.auto_evaluator._detect_gpu_ids_with_nvidia_smi", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "Set RESEARCH_EVAL_AVAILABLE_GPUS explicitly"):
                _managed_gpu_ids()

    def test_multistart_resource_coordination_id_includes_seed(self):
        station = mock.Mock(station_id="station-uuid")
        with mock.patch.dict(
            os.environ,
            {"STATION_MULTISTART_BRANCH": "1", "STATION_MULTISTART_SEED": "2"},
        ):
            self.assertEqual(
                "station-uuid:s2",
                _resource_coordination_station_id(station),
            )

    def test_multistart_resource_coordination_requires_seed(self):
        station = mock.Mock(station_id="station-uuid")
        with mock.patch.dict(os.environ, {"STATION_MULTISTART_BRANCH": "1"}):
            os.environ.pop("STATION_MULTISTART_SEED", None)
            with self.assertRaisesRegex(RuntimeError, "STATION_MULTISTART_SEED"):
                _resource_coordination_station_id(station)

    def test_cpu_coordination_keeps_same_eval_isolated_between_multistart_seeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            coord_path = os.path.join(temp_dir, "cpu.json")
            seed_one = CPUCoordinator(
                coord_file_path=coord_path,
                available_cpus=[0, 1],
                station_id="station-uuid:s1",
                expiry_seconds=10,
            )
            self.assertEqual([0], seed_one.allocate("101", count=1))

            seed_two = CPUCoordinator(
                coord_file_path=coord_path,
                available_cpus=[0, 1],
                station_id="station-uuid:s2",
                expiry_seconds=10,
            )
            self.assertEqual([1], seed_two.allocate("101", count=1))

            with open(coord_path, "r", encoding="utf-8") as handle:
                allocations = json.load(handle)["allocations"]
            self.assertEqual([0], allocations["station-uuid:s1:101"]["cpus"])
            self.assertEqual([1], allocations["station-uuid:s2:101"]["cpus"])

    def test_gpu_coordination_keeps_same_eval_isolated_between_multistart_seeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            coord_path = os.path.join(temp_dir, "gpu.json")
            seed_one = GPUCoordinator(
                coord_file_path=coord_path,
                available_gpus=[0, 1],
                station_id="station-uuid:s1",
                expiry_seconds=10,
            )
            self.assertEqual([0], seed_one.allocate("201", count=1))

            seed_two = GPUCoordinator(
                coord_file_path=coord_path,
                available_gpus=[0, 1],
                station_id="station-uuid:s2",
                expiry_seconds=10,
            )
            self.assertEqual([1], seed_two.allocate("201", count=1))

            with open(coord_path, "r", encoding="utf-8") as handle:
                allocations = json.load(handle)["allocations"]
            self.assertEqual([0], allocations["station-uuid:s1:201"]["gpus"])
            self.assertEqual([1], allocations["station-uuid:s2:201"]["gpus"])

    def test_cpu_coordination_records_expiry_and_reclaims_expired_foreign_allocation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            coord_path = os.path.join(temp_dir, "cpu.json")
            first = CPUCoordinator(
                coord_file_path=coord_path,
                available_cpus=[0, 1],
                station_id="station-a",
                expiry_seconds=0.01,
            )
            self.assertEqual(first.allocate("101", count=1), [0])
            with open(coord_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            allocation = data["allocations"]["station-a:101"]
            self.assertIn("expires_at", allocation)
            self.assertEqual(allocation["timeout_seconds"], 0.005)

            time.sleep(0.02)
            second = CPUCoordinator(
                coord_file_path=coord_path,
                available_cpus=[0, 1],
                station_id="station-b",
                expiry_seconds=10,
            )
            self.assertEqual(second.allocate("102", count=2), [0, 1])
            with open(coord_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertNotIn("station-a:101", data["allocations"])
            self.assertEqual(data["allocations"]["station-b:102"]["cpus"], [0, 1])

    def test_gpu_coordination_records_expiry_and_reclaims_expired_foreign_allocation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            coord_path = os.path.join(temp_dir, "gpu.json")
            first = GPUCoordinator(
                coord_file_path=coord_path,
                available_gpus=[0, 1],
                station_id="station-a",
                expiry_seconds=0.01,
            )
            self.assertEqual(first.allocate("201", count=1), [0])
            with open(coord_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            allocation = data["allocations"]["station-a:201"]
            self.assertIn("expires_at", allocation)
            self.assertEqual(allocation["timeout_seconds"], 0.005)

            time.sleep(0.02)
            second = GPUCoordinator(
                coord_file_path=coord_path,
                available_gpus=[0, 1],
                station_id="station-b",
                expiry_seconds=10,
            )
            self.assertEqual(second.allocate("202", count=2), [0, 1])
            with open(coord_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertNotIn("station-a:201", data["allocations"])
            self.assertEqual(data["allocations"]["station-b:202"]["gpus"], [0, 1])

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
                "Official evaluation finished. The attempt status, score, and secondary metrics below are authoritative.",
                persisted_stdout,
            )
            self.assertIn("ATTEMPT_COMPLETE", persisted_stdout)
            self.assertIn("ATTEMPT_STATUS: completed", persisted_stdout)
            self.assertIn("PRIMARY_SCORE: 1.5", persisted_stdout)
            self.assertEqual(stdout_text, persisted_stdout)
            self.assertEqual(stderr_text, "")

    def test_cpu_only_run_request_skips_gpu_allocation_when_gpu_managed(self):
        class PendingFuture:
            def done(self):
                return False

        class RecordingThreadPool:
            def __init__(self):
                self.calls = []

            def submit(self, fn, *args):
                self.calls.append((fn, args))
                return PendingFuture()

        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_EVAL_GPU_NUM", 1), \
                 mock.patch.object(constants, "RESEARCH_EVAL_USE_DIFF_GPU", False), \
                 mock.patch.object(constants, "RESEARCH_EVAL_AVAILABLE_GPUS", [0, 1]), \
                 mock.patch.object(constants, "RESEARCH_EVAL_CPU_NUM", None):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)
                eval_manager.create_evaluation(
                    eval_id="91",
                    author="Veritas I",
                    title="CPU-only dispatch",
                    content="Run without GPU.",
                    tick=10,
                    tags=["cpu"],
                    abstract="CPU-only dispatch should not allocate GPU.",
                    lineage="veritas",
                )
                eval_manager.register_attempt(
                    "91",
                    "print('cpu only')\n",
                    "storage/submission/91.py",
                    cpu_only=True,
                )
                request_path = os.path.join(paths.run_requests_dir, "91_attempt_1.yaml")
                file_io_utils.save_yaml(
                    {"eval_id": "91", "attempt": 1, "cpu_only": True},
                    request_path,
                    sort_keys=False,
                )

                thread_pool = RecordingThreadPool()
                evaluator = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
                evaluator.thread_pool = thread_pool
                evaluator.active_futures = {}
                evaluator.attempt_worker_max = 32
                evaluator.run_requests_dir = paths.run_requests_dir
                evaluator.eval_manager = eval_manager
                evaluator._allocate_gpu = lambda _eval_id: self.fail("CPU-only dispatch allocated a GPU")
                evaluator._report_exception = lambda *args, **kwargs: None

                AutoResearchEvaluator._dispatch_run_requests(evaluator)

            self.assertEqual(len(thread_pool.calls), 1)
            _, args = thread_pool.calls[0]
            self.assertEqual(args[0], "91")
            self.assertEqual(args[1], 1)
            self.assertEqual(args[3], [])
            self.assertIsNone(args[4])
            self.assertEqual(evaluator.active_futures["91"]["gpu_ids"], [])

    def test_public_payload_and_top_submission_work_for_new_eval_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")
            callback_updates = []

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_SCORE_DISPLAY_PRECISION", 2):
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
            self.assertIn("Score: 2.50 | metric: 7\nMessage:\nverified", preview)

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

    def test_completed_attempt_with_na_score_finalizes_as_completed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_EVAL_MAX_TICK", 2):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="45",
                    author="Veritas VI",
                    title="Diagnostic completion test",
                    content="Run a diagnostic-only submission.",
                    tick=10,
                    tags=["diagnostic"],
                    abstract="Completed diagnostic attempts may be non-scorable.",
                    lineage="veritas",
                )
                eval_manager.register_attempt("45", "print('diagnostic')\n", "storage/submission/45.py")
                eval_manager.complete_attempt(
                    "45",
                    1,
                    success=True,
                    score=constants.RESEARCH_SCORE_NA,
                    stdout="diagnostic\nATTEMPT_COMPLETE\nATTEMPT_STATUS: completed\n",
                    stderr="",
                    details={"Message": "diagnostic completed"},
                    sort_key=None,
                    status="completed",
                )
                eval_manager.finalize_evaluation(
                    "45",
                    "# Coder Report\n\n## Final Status\n",
                )

                record = eval_manager.get_evaluation("45")
                should_wait = eval_manager.should_wait_at_tick(current_tick=11)

            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["coder"]["status"], "completed")
            self.assertEqual(record["final"]["status"], "completed")
            self.assertEqual(record["final"]["primary_score"], constants.RESEARCH_SCORE_NA)
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

    def test_stale_attempt_running_after_inactive_coder_requeues_after_timeout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="84",
                    author="Pipeline IV",
                    title="Stale post-coder attempt",
                    content="The evaluator worker vanished after the coder exited.",
                    tick=18,
                    tags=["pipeline"],
                    abstract="Stale attempt should be requeued.",
                    lineage="pipeline",
                )
                eval_manager.register_attempt("84", "print('running')\n", "storage/submission/84.py")
                eval_manager.mark_attempt_running("84", 1)
                old_started = time.time() - 120.0
                eval_manager.update_evaluation(
                    "84",
                    lambda record: (
                        record.setdefault("coder", {}).update(
                            {
                                "active": False,
                                "status": "attempt_running",
                                "active_pid": None,
                            }
                        ),
                        record["attempts"][-1].update({"started_timestamp": old_started}),
                    ),
                )

                events = []
                evaluator = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
                evaluator.eval_manager = eval_manager
                evaluator.coder_manager = types.SimpleNamespace(active_sessions={})
                evaluator.active_futures = {}
                evaluator.paths = paths
                evaluator.timeout = 60
                evaluator._list_run_request_paths = lambda: []
                evaluator._push_log_event = lambda event_type, data: events.append((event_type, data))

                AutoResearchEvaluator._recover_stale_coder_states(evaluator)
                record = eval_manager.get_evaluation("84")

            self.assertEqual(record["status"], "queued")
            self.assertEqual(record["coder"]["status"], "queued")
            self.assertFalse(record["coder"]["active"])
            self.assertEqual(record["attempts"][-1]["status"], "abandoned")
            self.assertIn("inactive", record["attempts"][-1]["error"])
            self.assertEqual(events[0][0], "auto_research_recovered_stale_coder")
            self.assertFalse(events[0][1]["coder_active"])

    def test_inactive_coder_attempt_running_before_timeout_is_not_requeued(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)

                eval_manager.create_evaluation(
                    eval_id="85",
                    author="Pipeline V",
                    title="Attempt still within timeout",
                    content="The official attempt may still be running.",
                    tick=19,
                    tags=["pipeline"],
                    abstract="Fresh attempt should remain running.",
                    lineage="pipeline",
                )
                eval_manager.register_attempt("85", "print('running')\n", "storage/submission/85.py")
                eval_manager.mark_attempt_running("85", 1)
                eval_manager.update_evaluation(
                    "85",
                    lambda record: record.setdefault("coder", {}).update(
                        {
                            "active": False,
                            "status": "attempt_running",
                            "active_pid": None,
                        }
                    ),
                )

                events = []
                evaluator = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
                evaluator.eval_manager = eval_manager
                evaluator.coder_manager = types.SimpleNamespace(active_sessions={})
                evaluator.active_futures = {}
                evaluator.paths = paths
                evaluator.timeout = 3600
                evaluator._list_run_request_paths = lambda: []
                evaluator._push_log_event = lambda event_type, data: events.append((event_type, data))

                AutoResearchEvaluator._recover_stale_coder_states(evaluator)
                record = eval_manager.get_evaluation("85")

            self.assertEqual(record["status"], "running")
            self.assertEqual(record["coder"]["status"], "attempt_running")
            self.assertEqual(record["attempts"][-1]["status"], "running")
            self.assertEqual(events, [])

    def test_station_statistics_without_live_evaluator_avoids_research_index(self):
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
                with mock.patch.object(
                    EvaluationManager,
                    "get_evaluation_statistics",
                    side_effect=AssertionError("station statistics must not query the Research index"),
                ):
                    stats = Station.get_station_statistics(station_stub)

            self.assertEqual(stats["queued_experiments_count"], 0)
            self.assertEqual(stats["queued_experiments"], [])
            self.assertEqual(stats["running_experiments_count"], 0)
            self.assertIsNone(stats["top_research_submission"])

    def test_station_statistics_exposes_cached_breakthrough_age(self):
        station_stub = types.SimpleNamespace(
            rooms={},
            config={
                constants.STATION_CONFIG_CURRENT_TICK: 200,
                constants.STATION_CONFIG_AGENT_TURN_ORDER: [],
                constants.STATION_CONFIG_STAGNATION_COUNTER: 186,
                constants.STATION_CONFIG_TOP_EVALUATION_ID: "75",
                constants.STATION_CONFIG_TOP_TITLE: "Cached top",
                constants.STATION_CONFIG_TOP_SCORE: 8.0,
                constants.STATION_CONFIG_TOP_SORT_KEY: [8.0],
                constants.STATION_CONFIG_TOP_TICK: 14,
                constants.STATION_CONFIG_TOP_AGENT_NAME: "Fallback II",
            },
            agent_module=types.SimpleNamespace(load_agent_data=lambda *_args, **_kwargs: None),
            auto_research_evaluator=None,
        )

        with mock.patch(
            "station.eval_research.evaluation_index.get_top_submission",
            side_effect=AssertionError("dashboard statistics must use station config"),
        ):
            stats = Station.get_station_statistics(station_stub)

        self.assertEqual(186, stats["ticks_since_last_breakthrough"])
        self.assertEqual("75", stats["top_research_submission"]["evaluation_id"])

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
            self.assertLess(notification_message.index("**Coder Report:**"), notification_message.index("**Stdout/Stderr:**"))
            self.assertNotIn("**Instruction Prompt:**", notification_message)
            self.assertNotIn("**Abstract:**", notification_message)
            self.assertIn(STDOUT_STDERR_HIDDEN_NOTICE, notification_message)
            self.assertNotIn(long_stdout, notification_message)
            self.assertNotIn("[... truncated after 12 characters]", notification_message)

            self.assertIsNotNone(review)
            review_message = review["message"]
            self.assertIn("**Abstract:** This abstract should appear in review only.", review_message)
            self.assertIn("**Instruction Prompt:**", review_message)
            self.assertLess(review_message.index("**Instruction Prompt:**"), review_message.index("**Score:** 5.5"))
            self.assertLess(review_message.index("**Score:** 5.5"), review_message.index("**Coder Report:**"))
            self.assertIn(STDOUT_STDERR_HIDDEN_NOTICE, review_message)
            self.assertNotIn(long_stdout, review_message)
            self.assertNotIn("[... truncated after 12 characters]", review_message)

    def test_evaluation_table_is_lineage_local_but_direct_review_can_cross_lineage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""):
                paths = ensure_runtime_layout()
                eval_manager = EvaluationManager(paths.evaluations_dir)
                room = ResearchCenter(eval_manager=eval_manager, paths=paths)

                eval_manager.create_evaluation(
                    eval_id="30",
                    author="Veritas I",
                    title="Own lineage method",
                    content="Run own lineage method.",
                    tick=10,
                    tags=["own"],
                    abstract="Own lineage abstract.",
                    lineage="veritas",
                )
                eval_manager.register_attempt("30", "print('own')\n", "storage/submission/30.py")
                eval_manager.complete_attempt(
                    "30",
                    1,
                    success=True,
                    score=1.0,
                    stdout="own\n",
                    stderr="",
                    details={"Message": "own"},
                    sort_key=(1.0,),
                    status="completed",
                )
                eval_manager.finalize_evaluation("30", "# Coder Report\n\nown report\n")

                eval_manager.create_evaluation(
                    eval_id="31",
                    author="Logos I",
                    title="Other lineage method",
                    content="Run other lineage method.",
                    tick=11,
                    tags=["other"],
                    abstract="Other lineage abstract.",
                    lineage="logos",
                )
                eval_manager.register_attempt("31", "print('other')\n", "storage/submission/31.py")
                eval_manager.complete_attempt(
                    "31",
                    1,
                    success=True,
                    score=2.0,
                    stdout="other\n",
                    stderr="",
                    details={"Message": "other"},
                    sort_key=(2.0,),
                    status="completed",
                )
                eval_manager.finalize_evaluation("31", "# Coder Report\n\nother report\n")

                agent_data = {
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_NAME_KEY: "Veritas II",
                    constants.AGENT_LINEAGE_KEY: "veritas",
                    constants.AGENT_TICK_BIRTH_KEY: 0,
                    constants.SHORT_ROOM_NAME_RESEARCH: {},
                }
                supervisor_data = {
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_NAME_KEY: "Supervisor",
                    constants.AGENT_LINEAGE_KEY: "supervisor",
                    constants.AGENT_ROLE_KEY: constants.ROLE_SUPERVISOR,
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

                lineage_table = room._get_specific_room_content(agent_data, room_context, current_tick=120)
                supervisor_table = room._get_specific_room_content(supervisor_data, room_context, current_tick=120)
                help_text = room.get_help_message(agent_data, room_context)
                immature_review_result, _ = room.handle_action(agent_data, "review", "31", None, room_context, current_tick=10)
                review_result, _ = room.handle_action(agent_data, "review", "31", None, room_context, current_tick=120)

            self.assertIn("Own lineage method", lineage_table)
            self.assertNotIn("Other lineage method", lineage_table)
            self.assertIn("Own lineage method", supervisor_table)
            self.assertIn("Other lineage method", supervisor_table)
            self.assertIn("Agents only see their own lineage's evaluations", help_text)
            self.assertIn("by evaluation ID", help_text)
            self.assertIn("Do not run broad review attempts", help_text)
            self.assertNotIn("Non-supervisor", help_text)
            self.assertNotIn("station policy permits", help_text)
            self.assertEqual(immature_review_result, ["Immature agents cannot review other-lineage evaluations."])
            self.assertEqual(review_result, ["Evaluation 31 details sent to your System Messages."])
            self.assertTrue(any("Other lineage method" in message for message in agent_manager.notifications))

    def test_lineage_evolution_reads_research_results_through_index(self):
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

                eval_manager.create_evaluation(
                    eval_id="32",
                    author="Veritas II",
                    title="Dimension progress",
                    content="Improve a secondary track",
                    tick=5,
                    tags=["progress"],
                    abstract="Progress result",
                    lineage="veritas",
                )
                eval_manager.register_attempt("32", "print('v3')\n", "storage/submission/32.py")
                eval_manager.complete_attempt(
                    "32",
                    1,
                    success=True,
                    score=1.5,
                    stdout="v3\n",
                    stderr="",
                    details={"Message": "track progress"},
                    sort_key=(1.5,),
                    progress_records=[
                        {
                            "track": "dimension:d4",
                            "rank_key": [-0.7],
                            "value": 0.7,
                        }
                    ],
                    status="completed",
                )
                eval_manager.finalize_evaluation("32", "# Coder Report\n\n## Final Status\nsuccess\n")

                eval_manager.get_result_summary = mock.Mock(side_effect=AssertionError("should use index"))
                eval_manager.get_all_evaluation_ids = mock.Mock(side_effect=AssertionError("should use index"))
                lineage_manager = LineageEvolutionManager(types.SimpleNamespace(), eval_manager=eval_manager)
                breakthroughs = lineage_manager._load_all_breakthroughs()

            self.assertEqual(breakthroughs.get("Veritas"), 2)
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

    def test_immature_agent_cannot_access_shared_storage(self):
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
                    constants.AGENT_TICK_BIRTH_KEY: 0,
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

                result_list, _ = room.handle_action(agent_data, "storage", "list shared", None, room_context, current_tick=10)
                result_read, _ = room.handle_action(agent_data, "read", "shared/note.txt", None, room_context, current_tick=10)
                result_list_mature, _ = room.handle_action(agent_data, "storage", "list shared", None, room_context, current_tick=40)
                help_text = room.get_help_message(agent_data, room_context)

            expected = [
                "Immature agents cannot access `storage/shared`. Use your lineage storage instead, for example `storage/tester/...`, until you mature."
            ]
            self.assertEqual(result_list, expected)
            self.assertEqual(result_read, expected)
            self.assertEqual(result_list_mature, ["Directory listing for storage/shared sent to your System Messages."])
            self.assertIn("Immature agents cannot access `storage/shared`", help_text)
            self.assertIn("across the lineage storage directories your coder can access", help_text)
            self.assertIn(
                "Avoid saving any file larger than 1 GB unless it is necessary.",
                help_text,
            )

    def test_immature_agent_cannot_access_other_lineage_storage_even_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS", True):
                paths = ensure_runtime_layout()
                other_lineage_dir = os.path.join(paths.lineages_root, "other")
                os.makedirs(other_lineage_dir, exist_ok=True)
                with open(os.path.join(other_lineage_dir, "note.txt"), "w", encoding="utf-8") as handle:
                    handle.write("hello from other lineage")

                room = ResearchCenter()
                agent_data = {
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_NAME_KEY: "Tester I",
                    constants.AGENT_LINEAGE_KEY: "tester",
                    constants.AGENT_TICK_BIRTH_KEY: 0,
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

                result_list_immature, _ = room.handle_action(agent_data, "storage", "list other", None, room_context, current_tick=10)
                result_read_immature, _ = room.handle_action(agent_data, "read", "lineages/other/note.txt", None, room_context, current_tick=10)
                result_info_immature, _ = room.handle_action(agent_data, "storage", "info", None, room_context, current_tick=10)
                result_list_mature, _ = room.handle_action(agent_data, "storage", "list other", None, room_context, current_tick=40)
                result_read_mature, _ = room.handle_action(agent_data, "read", "lineages/other/note.txt", None, room_context, current_tick=40)

            expected = [
                "Immature agents cannot access other lineage storage. Use your lineage storage instead, for example `storage/tester/...`, until you mature."
            ]
            self.assertEqual(result_list_immature, expected)
            self.assertEqual(result_read_immature, expected)
            self.assertEqual(result_info_immature, ["Storage information sent to your System Messages."])
            self.assertEqual(result_list_mature, ["Directory listing for storage/other sent to your System Messages."])
            self.assertEqual(result_read_mature, ["File content for storage/other/note.txt sent to your System Messages."])
            self.assertTrue(any("Cross-lineage reads: Disabled until maturity" in message for message in agent_manager.notifications))

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
                 mock.patch.object(constants, "AUTO_EVAL_RESEARCH", False), \
                 mock.patch.object(constants, "RESEARCH_EVAL_GPU_NUM", None), \
                 mock.patch.object(constants, "RESEARCH_EVAL_USE_DIFF_GPU", False):
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
                 mock.patch.object(constants, "RESEARCH_SUBMISSION_COOLDOWN_TICKS", 0), \
                 mock.patch.object(constants, "RESEARCH_MAX_CONCURRENT_SUBMISSIONS", 1):
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

            self.assertIn("**Active Experiment Limit:**", active_message)
            self.assertIn("90", active_message)
            self.assertEqual(ready_message, "**Submission Ready:** You may submit one focused experiment.")

    def test_research_submit_status_and_help_reflect_configured_active_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = os.path.join(temp_dir, "station-data")

            with mock.patch.object(constants, "BASE_STATION_DATA_PATH", data_root), \
                 mock.patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", ""), \
                 mock.patch.object(constants, "RESEARCH_CENTER_ENABLED", True), \
                 mock.patch.object(constants, "AUTO_EVAL_RESEARCH", False), \
                 mock.patch.object(constants, "RESEARCH_SUBMISSION_COOLDOWN_TICKS", 0), \
                 mock.patch.object(constants, "RESEARCH_MAX_CONCURRENT_SUBMISSIONS", 2):
                station = Station()
                room = station.rooms[constants.ROOM_RESEARCH_CENTER]
                eval_manager = station.research_eval_manager
                agent_data = {
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_NAME_KEY: "Curie I",
                    constants.AGENT_LINEAGE_KEY: "curie",
                    constants.SHORT_ROOM_NAME_RESEARCH: {},
                }

                help_text = room.get_help_message(
                    agent_data,
                    RoomContext(None, None, None, constants, station),
                )
                eval_manager.create_evaluation(
                    eval_id="91",
                    author="Curie I",
                    title="First active",
                    content="Run one experiment.",
                    tick=10,
                    tags=["limit"],
                    abstract="First active experiment.",
                    lineage="curie",
                )
                one_active_message = room._build_submission_status_message(agent_data, constants, current_tick=10)
                eval_manager.create_evaluation(
                    eval_id="92",
                    author="Curie I",
                    title="Second active",
                    content="Run another experiment.",
                    tick=10,
                    tags=["limit"],
                    abstract="Second active experiment.",
                    lineage="curie",
                )
                limit_message = room._build_submission_status_message(agent_data, constants, current_tick=10)

            self.assertIn("at most **2 active evaluations**", help_text)
            self.assertIn("Concurrent submissions are useful for exploration", help_text)
            self.assertIn("**Submission Ready:**", one_active_message)
            self.assertIn("Active experiments: 1/2 (91)", one_active_message)
            self.assertIn("**Active Experiment Limit:**", limit_message)
            self.assertIn("2/2", limit_message)
            self.assertIn("91, 92", limit_message)

    def test_refresh_replaces_registry_after_evaluator_loads(self):
        old_registry = object()
        evaluator = mock.Mock()
        new_registry = mock.Mock()
        new_registry.get_evaluator.return_value = evaluator
        service = object.__new__(AutoResearchEvaluator)
        service.task_registry = old_registry

        with mock.patch(
            "station.eval_research.auto_evaluator.ResearchTaskRegistry",
            return_value=new_registry,
        ):
            loaded = service.refresh_task_registry()

        self.assertIs(evaluator, loaded)
        self.assertIs(new_registry, service.task_registry)

    def test_refresh_keeps_old_registry_when_evaluator_does_not_load(self):
        old_registry = object()
        new_registry = mock.Mock()
        new_registry.get_evaluator.return_value = None
        service = object.__new__(AutoResearchEvaluator)
        service.task_registry = old_registry

        with mock.patch(
            "station.eval_research.auto_evaluator.ResearchTaskRegistry",
            return_value=new_registry,
        ):
            with self.assertRaisesRegex(RuntimeError, "could not be loaded"):
                service.refresh_task_registry()

        self.assertIs(old_registry, service.task_registry)

    def test_refresh_endpoint_returns_before_background_reload(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        app_path = os.path.join(repo_root, "web_interface", "app.py")
        with open(app_path, "r", encoding="utf-8") as handle:
            app_source = handle.read()
        route_start = app_source.index("@app.route('/api/station/research_evaluator/refresh'")
        route_end = app_source.index("@app.route('/api/station/api_runtime_config'", route_start)
        route_source = app_source[route_start:route_end]
        worker_start = app_source.index("def _run_research_evaluator_refresh")
        worker_source = app_source[worker_start:route_start]

        self.assertIn("@auth_required", route_source)
        self.assertIn("methods=['GET', 'POST']", route_source)
        self.assertIn("threading.Thread", route_source)
        self.assertIn('"status": "requested"', route_source)
        self.assertIn("}), 202", route_source)
        self.assertNotIn("while ", route_source)
        self.assertLess(worker_source.index("has_pending_or_running"), worker_source.index("refresh_task_registry"))
        self.assertIn('status="completed"', worker_source)

    def test_auto_transition_detects_and_retargets_live_dimension_and_count(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        script_path = os.path.join(
            repo_root,
            "example",
            "research_alpha_evolve",
            "kissing_margin",
            "auto_phase2_transition.sh",
        )
        with open(script_path, "r", encoding="utf-8") as handle:
            script_source = handle.read()

        self.assertIn("replace_d.detect_current_values", script_source)
        self.assertIn('details.get("N") != target_count', script_source)
        self.assertIn("artifact_dimension != dimension", script_source)
        self.assertIn("replace_d.update_evaluator", script_source)
        self.assertIn("replace_d.update_task_text", script_source)


if __name__ == "__main__":
    unittest.main()
