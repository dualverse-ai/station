import contextlib
import io
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import filelock

from station import constants
from station import file_io_utils
from station import index_paths
from station.eval_research.evaluation_manager import (
    EvaluationManager,
    get_evaluation_code_info,
    get_evaluation_review_info,
    get_evaluation_submission_payload,
)
from station.eval_research.coder_manager import ActiveCoderSession, ResearchCoderManager
from station.eval_research.base_evaluator import ResearchTaskEvaluator
from station.eval_research.auto_evaluator import AutoResearchEvaluator
from station.eval_research.runtime_paths import (
    SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME,
    SUBMIT_EVAL_CLI_SNAPSHOT_FILENAME,
    build_runtime_paths,
    ensure_runtime_layout,
    load_task_spec_markdown,
)
from station.eval_research.submit_eval_cli import submit_eval
from station.rooms.research_center import ResearchCenter
from scripts.migrate.migrate_research_eval_artifacts import migrate, needs_migration
from scripts.set_scores_na import process_evaluation as set_scores_na_process_evaluation
from scripts.void_eval import void_evaluation as void_evaluation_record


class ResearchEvaluationArtifactTests(unittest.TestCase):
    def _make_manager(self, root: str) -> EvaluationManager:
        eval_dir = Path(root) / "evaluations"
        file_io_utils.ensure_dir_exists(str(eval_dir))
        return EvaluationManager(str(eval_dir))

    def _create_eval_with_submission(self, manager: EvaluationManager, root: str, eval_id: str = "204") -> str:
        manager.create_evaluation(
            eval_id=eval_id,
            author="Ariadne II",
            title="Artifact-backed evaluation",
            content="Run the experiment.",
            tick=131,
            tags=["n50", "native-c"],
            abstract="Artifact backed metadata.",
            lineage="ariadne",
        )
        submission_path = Path(root) / constants.RESEARCH_STORAGE_DIR / "submission" / f"{eval_id}.py"
        file_io_utils.save_text("def solution_batch():\n    return {}\n", str(submission_path))
        return str(submission_path)

    def test_research_coder_kills_idle_codex_transcript(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "CLI_WORKER_TRANSCRIPT_IDLE_TIMEOUT_SECONDS": constants.CLI_WORKER_TRANSCRIPT_IDLE_TIMEOUT_SECONDS,
        }

        class DummyStation:
            station_id = "idle-test"

        class DummyProcess:
            pid = 4321

            def poll(self):
                return None

        class DummyHandle:
            def close(self):
                pass

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                constants.BASE_STATION_DATA_PATH = str(Path(temp_dir) / "station_data")
                constants.RESEARCH_STORAGE_BASE_PATH = None
                constants.CLI_WORKER_TRANSCRIPT_IDLE_TIMEOUT_SECONDS = 1800
                paths = ensure_runtime_layout(constants)
                manager = EvaluationManager(paths.evaluations_dir)
                manager.create_evaluation(
                    eval_id="204",
                    author="Ariadne II",
                    title="Idle transcript",
                    content="Run it.",
                    tick=1,
                    tags=[],
                    abstract="Idle transcript test.",
                    lineage="ariadne",
                )
                coder = ResearchCoderManager(DummyStation(), manager, paths=paths)
                run_dir = Path(paths.coder_sessions_dir) / "codex_204_spawn_1_deadbeef"
                file_io_utils.ensure_dir_exists(str(run_dir))
                transcript_path = run_dir / "transcript.jsonl"
                stderr_path = run_dir / "stderr.txt"
                file_io_utils.save_text('{"type":"thread.started"}\n', str(transcript_path))
                file_io_utils.save_text("", str(stderr_path))
                session = ActiveCoderSession(
                    eval_id="204",
                    session_id="codex_204_spawn_1_deadbeef",
                    run_dir=str(run_dir),
                    backend="codex",
                    transcript_format="jsonl",
                    process=DummyProcess(),
                    transcript_handle=DummyHandle(),
                    stderr_handle=DummyHandle(),
                    report_path=str(Path(paths.reports_dir) / "204.md"),
                    prompt_path=str(run_dir / "prompt.txt"),
                    command=["codex", "exec"],
                    debug_dir=None,
                    dialogue_log_path=None,
                    transcript_path=str(transcript_path),
                    stderr_path=str(stderr_path),
                    last_message_path=str(run_dir / "last_message.txt"),
                    last_transcript_size=os.path.getsize(transcript_path),
                    last_transcript_growth_timestamp=time.time() - 1900,
                )
                coder.active_sessions["204"] = session

                with mock.patch("station.eval_research.coder_manager.os.killpg") as killpg:
                    coder._check_codex_transcript_idle_timeouts()

                killpg.assert_called_once_with(DummyProcess.pid, signal.SIGTERM)
                self.assertTrue(session.transcript_idle_timeout_triggered)
                self.assertIn("did not grow", session.transcript_idle_timeout_reason)
                failure = coder._classify_no_report_failure(session, returncode=-15)
                self.assertEqual(failure.category, "codex_transcript_idle_timeout")
                updated = manager.get_evaluation("204")
                self.assertEqual(updated["coder"]["failure_category"], "codex_transcript_idle_timeout")
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_fresh_respawn_lists_previous_session_artifacts_in_new_prompt(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
        }

        class DummyStation:
            station_id = "respawn-artifact-test"

        class DummyProcess:
            pid = 5678

            def poll(self):
                return None

        class FakeBackend:
            supports_resume = False
            extra_allowed_roots = None

            def prepare_launch(
                self,
                *,
                executable,
                workspace_root,
                run_dir,
                model_name,
                storage_root,
                prompt,
                extra_allowed_roots=None,
            ):
                self.extra_allowed_roots = list(extra_allowed_roots or [])
                return SimpleNamespace(
                    command=[executable, "-c", "pass"],
                    stdin_text=None,
                    transcript_path=os.path.join(run_dir, "transcript.jsonl"),
                    stderr_path=os.path.join(run_dir, "stderr.txt"),
                    last_message_path=os.path.join(run_dir, "last_message.txt"),
                    transcript_format="jsonl",
                    env_overrides={},
                )

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                constants.BASE_STATION_DATA_PATH = str(Path(temp_dir) / "station_data")
                constants.RESEARCH_STORAGE_BASE_PATH = None
                paths = ensure_runtime_layout(constants)
                manager = EvaluationManager(paths.evaluations_dir)
                manager.create_evaluation(
                    eval_id="301",
                    author="Ariadne II",
                    title="Respawn artifact links",
                    content="Continue useful prior work if it exists.",
                    tick=1,
                    tags=[],
                    abstract="Respawn artifact link test.",
                    lineage="ariadne",
                )

                old_session_id = "codex_301_spawn_1_deadbeef"
                old_session_dir = Path(paths.coder_sessions_dir) / old_session_id
                file_io_utils.ensure_dir_exists(str(old_session_dir))
                file_io_utils.save_text("old prompt", str(old_session_dir / "prompt.txt"))
                file_io_utils.save_text('{"type":"thread.started"}\n', str(old_session_dir / "transcript.jsonl"))

                duplicate_spawn_session_id = "codex_301_spawn_1_8a9c7d1e"
                duplicate_spawn_session_dir = Path(paths.coder_sessions_dir) / duplicate_spawn_session_id
                file_io_utils.ensure_dir_exists(str(duplicate_spawn_session_dir))
                file_io_utils.save_text("second old prompt", str(duplicate_spawn_session_dir / "prompt.txt"))

                other_eval_session_dir = Path(paths.coder_sessions_dir) / "codex_302_spawn_1_ignored"
                file_io_utils.ensure_dir_exists(str(other_eval_session_dir))

                def mark_for_respawn(record):
                    record["status"] = "queued"
                    coder = record.setdefault("coder", {})
                    coder["active"] = False
                    coder["status"] = "queued"
                    coder["spawn_count"] = 1
                    coder["session_id"] = old_session_id
                    coder["failure_category"] = "report_missing"
                    coder["last_error"] = "Coder exited without producing storage/report/301.md"

                manager.update_evaluation("301", mark_for_respawn)
                coder = ResearchCoderManager(DummyStation(), manager, paths=paths)
                fake_backend = FakeBackend()

                with mock.patch(
                    "station.eval_research.coder_manager.get_cli_worker_backend",
                    return_value=fake_backend,
                ), mock.patch.object(
                    ResearchCoderManager,
                    "_detect_backend_executable",
                    return_value=sys.executable,
                ), mock.patch(
                    "station.eval_research.coder_manager.subprocess.Popen",
                    return_value=DummyProcess(),
                ):
                    self.assertTrue(coder.launch_for_evaluation(manager.get_evaluation("301")))

                updated = manager.get_evaluation("301")
                index_dir = os.path.dirname(index_paths.get_station_index_database_path(constants.BASE_STATION_DATA_PATH))
                self.assertIn(os.path.realpath(index_dir), [os.path.realpath(path) for path in fake_backend.extra_allowed_roots])
                new_session_id = updated["coder"]["session_id"]
                self.assertNotEqual(new_session_id, old_session_id)
                self.assertEqual(updated["coder"]["spawn_count"], 2)

                new_session_dir = Path(paths.coder_sessions_dir) / new_session_id
                self.assertFalse((new_session_dir / "previous_sessions").exists())

                prompt_text = file_io_utils.load_text(str(new_session_dir / "prompt.txt"))
                self.assertIn("Previous respawn context", prompt_text)
                self.assertIn("crashed, timed out, or exited without writing the final report", prompt_text)
                self.assertIn(f"coder_sessions/{duplicate_spawn_session_id}/", prompt_text)
                self.assertIn(f"coder_sessions/{old_session_id}/", prompt_text)
                self.assertNotIn("coder_sessions/codex_302_spawn_1_ignored/", prompt_text)
                self.assertIn("storage/submission/301.py", prompt_text)
                self.assertIn("Coder exited without producing storage/report/301.md", prompt_text)

                active_session = coder.active_sessions.pop("301")
                active_session.transcript_handle.close()
                active_session.stderr_handle.close()
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_retryable_infra_failure_schedules_resume_backoff(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_CODER_RESUME_BACKOFF_SECONDS": constants.RESEARCH_CODER_RESUME_BACKOFF_SECONDS,
        }

        class DummyStation:
            station_id = "resume-backoff-test"

        class DummyProcess:
            pid = 6789

            def poll(self):
                return 1

        class DummyHandle:
            def close(self):
                pass

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                constants.BASE_STATION_DATA_PATH = str(Path(temp_dir) / "station_data")
                constants.RESEARCH_STORAGE_BASE_PATH = None
                constants.RESEARCH_CODER_RESUME_BACKOFF_SECONDS = [123, 456]
                paths = ensure_runtime_layout(constants)
                manager = EvaluationManager(paths.evaluations_dir)
                manager.create_evaluation(
                    eval_id="401",
                    author="Ariadne II",
                    title="Resume backoff",
                    content="Run it.",
                    tick=1,
                    tags=[],
                    abstract="Resume backoff test.",
                    lineage="ariadne",
                )
                run_dir = Path(paths.coder_sessions_dir) / "codex_401_spawn_1_deadbeef"
                file_io_utils.ensure_dir_exists(str(run_dir))
                transcript_path = run_dir / "transcript.jsonl"
                stderr_path = run_dir / "stderr.txt"
                file_io_utils.save_text(
                    '{"type":"thread.started","thread_id":"resume-token-401"}\n'
                    '{"type":"error","message":"rate limit exceeded"}\n',
                    str(transcript_path),
                )
                file_io_utils.save_text("", str(stderr_path))

                coder = ResearchCoderManager(DummyStation(), manager, paths=paths)
                coder.active_sessions["401"] = ActiveCoderSession(
                    eval_id="401",
                    session_id="codex_401_spawn_1_deadbeef",
                    run_dir=str(run_dir),
                    backend="codex",
                    transcript_format="jsonl",
                    process=DummyProcess(),
                    transcript_handle=DummyHandle(),
                    stderr_handle=DummyHandle(),
                    report_path=str(Path(paths.reports_dir) / "401.md"),
                    prompt_path=str(run_dir / "prompt.txt"),
                    command=["codex", "exec"],
                    debug_dir=None,
                    dialogue_log_path=None,
                    transcript_path=str(transcript_path),
                    stderr_path=str(stderr_path),
                    last_message_path=str(run_dir / "last_message.txt"),
                    last_transcript_size=0,
                    last_transcript_growth_timestamp=time.time(),
                )

                before_poll = time.time()
                coder.poll()
                after_poll = time.time()

                updated = manager.get_evaluation("401")
                coder_state = updated["coder"]
                self.assertEqual(updated["status"], "running")
                self.assertEqual(coder_state["status"], "pending_resume")
                self.assertTrue(coder_state["active"])
                self.assertEqual(coder_state["failure_category"], "infra_transient")
                self.assertEqual(coder_state["resume_token"], "resume-token-401")
                self.assertEqual(coder_state["resume_delay_seconds"], 123)
                self.assertGreaterEqual(coder_state["next_resume_timestamp"], before_poll + 123)
                self.assertLessEqual(coder_state["next_resume_timestamp"], after_poll + 123)
                self.assertNotIn("401", coder.active_sessions)
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_resuming_coder_launch_waits_for_backoff_timestamp(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
        }

        class DummyCoderManager:
            active_sessions = {}

            def __init__(self):
                self.launched_ids = []

            def get_resume_backoff_remaining_seconds(self, coder_state):
                return ResearchCoderManager.get_resume_backoff_remaining_seconds(coder_state)

            def launch_for_evaluation(self, eval_data):
                self.launched_ids.append(str(eval_data.get("id")))
                return True

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                constants.BASE_STATION_DATA_PATH = str(Path(temp_dir) / "station_data")
                constants.RESEARCH_STORAGE_BASE_PATH = None
                paths = ensure_runtime_layout(constants)
                manager = EvaluationManager(paths.evaluations_dir)
                manager.create_evaluation(
                    eval_id="402",
                    author="Ariadne II",
                    title="Deferred resume",
                    content="Run it.",
                    tick=1,
                    tags=[],
                    abstract="Deferred resume test.",
                    lineage="ariadne",
                )

                def mark_pending_resume(record):
                    record["status"] = "running"
                    coder = record.setdefault("coder", {})
                    coder["active"] = True
                    coder["status"] = "pending_resume"
                    coder["resume_token"] = "resume-token-402"
                    coder["next_resume_timestamp"] = time.time() + 600
                    coder["resume_delay_seconds"] = 600

                manager.update_evaluation("402", mark_pending_resume)

                evaluator = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
                evaluator.eval_manager = manager
                evaluator.coder_manager = DummyCoderManager()
                evaluator._pause_orchestrator = lambda reason: None
                evaluator._report_exception = lambda **kwargs: None

                AutoResearchEvaluator._launch_resuming_coders(evaluator)
                self.assertEqual(evaluator.coder_manager.launched_ids, [])

                manager.update_evaluation(
                    "402",
                    lambda record: record.setdefault("coder", {}).update(
                        {"next_resume_timestamp": time.time() - 1}
                    ),
                )
                AutoResearchEvaluator._launch_resuming_coders(evaluator)
                self.assertEqual(evaluator.coder_manager.launched_ids, ["402"])
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_resume_backoff_uses_capped_configured_schedule(self):
        original_schedule = constants.RESEARCH_CODER_RESUME_BACKOFF_SECONDS
        try:
            constants.RESEARCH_CODER_RESUME_BACKOFF_SECONDS = [300, 600, 1200, 2400, 3600]
            self.assertEqual(ResearchCoderManager.get_resume_backoff_delay_seconds(0), 300)
            self.assertEqual(ResearchCoderManager.get_resume_backoff_delay_seconds(1), 600)
            self.assertEqual(ResearchCoderManager.get_resume_backoff_delay_seconds(2), 1200)
            self.assertEqual(ResearchCoderManager.get_resume_backoff_delay_seconds(3), 2400)
            self.assertEqual(ResearchCoderManager.get_resume_backoff_delay_seconds(4), 3600)
            self.assertEqual(ResearchCoderManager.get_resume_backoff_delay_seconds(20), 3600)
            self.assertEqual(
                ResearchCoderManager.get_resume_backoff_remaining_seconds(
                    {"next_resume_timestamp": 1300},
                    now=1000,
                ),
                300,
            )
        finally:
            constants.RESEARCH_CODER_RESUME_BACKOFF_SECONDS = original_schedule

    def test_changed_eval_refreshes_across_manager_instances_without_full_rescan(self):
        with tempfile.TemporaryDirectory() as root:
            manager_a = self._make_manager(root)
            manager_b = self._make_manager(root)

            submission_path = self._create_eval_with_submission(manager_a, root)
            with mock.patch.object(manager_b, "_load_evaluation_any", wraps=manager_b._load_evaluation_any) as load_eval:
                self.assertEqual(manager_b.get_all_evaluation_ids(), ["204"])
                self.assertEqual(manager_b.get_queued_instruction_eval_ids(), ["204"])
                self.assertEqual([call.args[0] for call in load_eval.call_args_list], [])

                load_eval.reset_mock()
                manager_a.register_attempt("204", submission_path)
                manager_a.complete_attempt("204", 1, True, 55.0, "stdout\n", "", {"Message": "scored"}, status="completed")
                manager_a.finalize_evaluation("204", "report\n", final_status="partial")

                compact = {item[constants.EVALUATION_ID_KEY]: item for item in manager_b.get_compact_display_infos()}
                self.assertEqual(compact["204"][constants.EVALUATION_SCORE_KEY], 55.0)
                self.assertEqual(manager_b.get_top_submission()["evaluation_id"], "204")
                self.assertEqual([call.args[0] for call in load_eval.call_args_list], [])

            index_path = Path(root) / "evaluations" / constants.RESEARCH_EVALUATION_INDEX_FILENAME
            self.assertFalse(index_path.exists())

    def test_statistics_use_cached_indexes_when_eval_dir_is_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self._make_manager(root)
            self._create_eval_with_submission(manager, root, eval_id="205")

            initial_stats = manager.get_evaluation_statistics()
            self.assertEqual(initial_stats["queued_count"], 1)

            with mock.patch.object(
                manager,
                "_load_evaluation_any",
                side_effect=AssertionError("dashboard stats should use the SQLite evaluation index"),
            ):
                stats = manager.get_evaluation_statistics()

            self.assertEqual(stats["queued_count"], 1)
            self.assertEqual(stats["queued_evaluations"][0]["evaluation_id"], "205")

    def test_lazy_manager_targeted_read_and_update_do_not_scan_all_evals(self):
        with tempfile.TemporaryDirectory() as root:
            writer = self._make_manager(root)
            for eval_id in ("201", "202", "203"):
                self._create_eval_with_submission(writer, root, eval_id=eval_id)

            lazy_manager = EvaluationManager(str(Path(root) / "evaluations"), preload=False)
            with mock.patch.object(lazy_manager, "_load_evaluation_any", wraps=lazy_manager._load_evaluation_any) as load_eval:
                record = lazy_manager.get_evaluation("202")
                self.assertEqual(record["id"], "202")

                lazy_manager.update_evaluation("202", lambda data: data.update({"title": "Updated by script"}))
                updated = lazy_manager.get_evaluation("202")

            self.assertEqual(updated["title"], "Updated by script")
            self.assertEqual([call.args[0] for call in load_eval.call_args_list], ["202"])

    def test_evaluation_lock_timeout_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self._make_manager(root)
            self._create_eval_with_submission(manager, root)

            class BusyLock:
                def __init__(self, *_args, **_kwargs):
                    pass

                def __enter__(self):
                    raise filelock.Timeout("busy")

                def __exit__(self, *_args):
                    return False

            output = io.StringIO()
            with mock.patch("station.eval_research.evaluation_manager.filelock.FileLock", BusyLock):
                with contextlib.redirect_stdout(output):
                    with self.assertRaises(filelock.Timeout):
                        with manager._file_lock("204", timeout=0.0):
                            pass

            self.assertIn("failed to acquire evaluation lock", output.getvalue())
            self.assertIn("204", output.getvalue())

    def test_score_maintenance_script_updates_evaluation_cache(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self._make_manager(root)
            submission_path = self._create_eval_with_submission(manager, root)
            manager.register_attempt("204", submission_path)
            manager.complete_attempt("204", 1, True, 55.0, "stdout\n", "", {"Message": "scored"}, status="completed")
            manager.finalize_evaluation("204", "report\n", final_status="completed")

            updated, changes = set_scores_na_process_evaluation(manager, "204", rewrite_messages=True)

            self.assertTrue(updated)
            self.assertGreater(changes, 0)
            record = manager.get_evaluation("204")
            self.assertEqual(record["final"]["primary_score"], constants.RESEARCH_SCORE_NA)
            display_info = manager.get_display_info("204")
            self.assertEqual(display_info[constants.EVALUATION_SCORE_KEY], constants.RESEARCH_SCORE_NA)
            self.assertIsNone(manager.get_top_submission())
            self.assertFalse((Path(root) / "evaluations" / constants.RESEARCH_EVALUATION_INDEX_FILENAME).exists())

    def test_void_eval_script_updates_evaluation_cache(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self._make_manager(root)
            submission_path = self._create_eval_with_submission(manager, root)
            manager.register_attempt("204", submission_path)

            with mock.patch("scripts.void_eval.add_pending_notification_atomic", return_value=True) as add_notification:
                self.assertTrue(void_evaluation_record(manager, "204", "bad run", dry_run=False))

            add_notification.assert_called_once()
            record = manager.get_evaluation("204")
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["final"]["primary_score"], constants.RESEARCH_SCORE_NA)
            self.assertTrue(record["notification"]["sent"])
            display_info = manager.get_display_info("204")
            self.assertEqual(display_info[constants.EVALUATION_STATUS_KEY], "failed")
            self.assertNotIn("204", manager.get_running_instruction_eval_ids())
            self.assertFalse((Path(root) / "evaluations" / constants.RESEARCH_EVALUATION_INDEX_FILENAME).exists())

    def test_register_complete_finalize_do_not_embed_large_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self._make_manager(root)
            submission_path = self._create_eval_with_submission(manager, root)

            data = manager.get_evaluation("204")
            self.assertEqual(data["artifacts"]["submission"], "storage/submission/204.py")
            attempt = manager.register_attempt("204", submission_path)
            self.assertEqual(attempt, 1)
            manager.complete_attempt(
                "204",
                1,
                success=True,
                score=55.172413793103445,
                stdout="official stdout\nATTEMPT_COMPLETE\nfooter\n",
                stderr="",
                details={"Message": "scored", "ValidCount": ["16", 16]},
                status="completed",
            )
            manager.finalize_evaluation("204", "# Coder Report\n\nFinal report.\n", final_status="partial")

            data = manager.get_evaluation("204")
            attempt_data = data["attempts"][0]
            final = data["final"]
            for blob_key in ("submission_snapshot", "stdout", "stdout_visible", "stderr", "coder_report"):
                self.assertNotIn(blob_key, attempt_data)
                self.assertNotIn(blob_key, final)

            self.assertEqual(data["artifacts"]["submission"], "storage/submission/204.py")
            self.assertEqual(final["artifacts"]["stdout"], "storage/stdout/204.log")
            self.assertEqual(file_io_utils.load_text(str(Path(root) / "storage" / "stdout" / "204.log")), "official stdout\nATTEMPT_COMPLETE\nfooter\n")
            self.assertEqual(file_io_utils.load_text(str(Path(root) / "storage" / "report" / "204.md")), "# Coder Report\n\nFinal report.\n")

    def test_generated_submit_script_uses_artifact_snapshot_for_unmigrated_metadata(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
        }
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                paths = ensure_runtime_layout(constants)
                file_io_utils.save_yaml(
                    {
                        "schema_version": 2,
                        "id": "7",
                        "author": "Aletheia I",
                        "lineage": "aletheia",
                        "title": "Old schema",
                        "tags": ["old"],
                        "abstract": "Old schema attempt.",
                        "instruction": "Run.",
                        "submitted_tick": 1,
                        "status": "running",
                        "coder": {"max_attempts": 5},
                        "attempts": [],
                        "final": {},
                    },
                    str(Path(paths.evaluations_dir) / "7.yaml"),
                    sort_keys=False,
                )
                file_io_utils.save_text("def solution_batch():\n    return {}\n", str(Path(paths.submissions_dir) / "7.py"))

                snapshot_path = Path(paths.research_root) / SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME / SUBMIT_EVAL_CLI_SNAPSHOT_FILENAME
                self.assertTrue(snapshot_path.exists())
                script_text = file_io_utils.load_text(paths.submit_script_path)
                self.assertIn("_internal/submit_eval_cli_snapshot.py", script_text)
                self.assertNotIn("station.eval_research.submit_eval_cli", script_text)

                result = subprocess.run(
                    [paths.submit_script_path, "7"],
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
                data = file_io_utils.load_yaml(str(Path(paths.evaluations_dir) / "7.yaml"))
                self.assertEqual(len(data["attempts"]), 1)
                attempt = data["attempts"][0]
                self.assertEqual(data["artifacts"]["submission"], "storage/submission/7.py")
                self.assertNotIn("submission_snapshot", attempt)
                self.assertNotIn("stdout", attempt)
                self.assertNotIn("stdout_visible", attempt)
                self.assertNotIn("stderr", attempt)
                self.assertEqual(attempt["stdout_path"], "storage/stdout/7.log")
                self.assertEqual(attempt["stderr_path"], "storage/stderr/7.log")
                stdout = file_io_utils.load_text(str(Path(paths.stdout_dir) / "7.log"))
                self.assertIn("ATTEMPT_QUEUED", stdout)
                self.assertTrue((Path(paths.run_requests_dir) / "7_attempt_1.yaml").exists())
                manager = EvaluationManager(paths.evaluations_dir)
                self.assertIn("7", manager.get_running_instruction_eval_ids())
                active_ids = {item["evaluation_id"] for item in manager.get_active_evaluations()}
                self.assertIn("7", active_ids)
                self.assertFalse(Path(paths.evaluation_index_path).exists())
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_submit_cli_uses_artifact_attempt_when_baseline_is_migrated(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
        }
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                paths = build_runtime_paths(constants)
                manager = EvaluationManager(paths.evaluations_dir)
                manager.create_evaluation(
                    eval_id="1",
                    author="System",
                    title="Baseline",
                    content="",
                    tick=0,
                    tags=["baseline"],
                    abstract="Baseline.",
                    lineage="system",
                )
                manager.create_evaluation(
                    eval_id="8",
                    author="Aletheia I",
                    title="New schema",
                    content="Run.",
                    tick=1,
                    tags=["new"],
                    abstract="New schema attempt.",
                    lineage="aletheia",
                )
                file_io_utils.save_text("def solution_batch():\n    return {}\n", str(Path(paths.submissions_dir) / "8.py"))

                self.assertEqual(submit_eval("8"), 0)
                data = file_io_utils.load_yaml(str(Path(paths.evaluations_dir) / "8.yaml"))
                self.assertEqual(len(data["attempts"]), 1)
                attempt = data["attempts"][0]
                self.assertNotIn("submission_snapshot", attempt)
                self.assertEqual(attempt["stdout_path"], "storage/stdout/8.log")
                self.assertEqual(attempt["stderr_path"], "storage/stderr/8.log")
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_review_code_and_submission_payload_read_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self._make_manager(root)
            submission_path = self._create_eval_with_submission(manager, root)
            manager.register_attempt("204", submission_path)
            manager.complete_attempt(
                "204",
                1,
                success=True,
                score=55.0,
                stdout="visible stdout\nATTEMPT_COMPLETE\nmachine footer\n",
                stderr="stderr text\n",
                details={"Message": "scored"},
                status="completed",
            )
            manager.finalize_evaluation("204", "report text\n", final_status="completed")

            eval_dir = str(Path(root) / "evaluations")
            review = get_evaluation_review_info("204", eval_dir)
            code = get_evaluation_code_info("204", eval_dir)
            payload = get_evaluation_submission_payload("204", eval_dir)

            self.assertEqual(review["status"], "completed")
            self.assertIn("report text", review["message"])
            self.assertIn("visible stdout", review["message"])
            self.assertNotIn("machine footer", review["message"])
            self.assertEqual(code["status"], "completed")
            self.assertIn("def solution_batch", code["code"])
            self.assertIn("STDOUT:\nvisible stdout", payload["logs"])
            self.assertIn("STDERR:\nstderr text", payload["logs"])

    def test_read_code_and_payload_stay_pending_before_final(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self._make_manager(root)
            submission_path = self._create_eval_with_submission(manager, root)
            manager.register_attempt("204", submission_path)

            eval_dir = str(Path(root) / "evaluations")
            code = get_evaluation_code_info("204", eval_dir)
            payload = get_evaluation_submission_payload("204", eval_dir)

            self.assertEqual(code["status"], "pending")
            self.assertEqual(payload["code"], "Code not available")
            self.assertEqual(payload["logs"], "Logs not available")

    def test_system_baseline_seeding_uses_artifacts(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_EVAL_CPU_NUM": constants.RESEARCH_EVAL_CPU_NUM,
            "RESEARCH_EVAL_USE_DIFF_GPU": constants.RESEARCH_EVAL_USE_DIFF_GPU,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
        }

        class DummyAgentModule:
            @staticmethod
            def add_pending_notification_atomic(author, message):
                return True

        class DummyStation:
            station_id = "artifact-test"
            agent_module = DummyAgentModule()

            def sync_top_research_submission_config(self, submission):
                self.top_submission = submission

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_EVAL_CPU_NUM = None
                constants.RESEARCH_EVAL_USE_DIFF_GPU = False
                constants.RESEARCH_STORAGE_BASE_PATH = None

                paths = build_runtime_paths(constants)
                file_io_utils.save_text(
                    "id: 1\nauthor: System\ntitle: Baseline\nabstract: Baseline artifact test.\ntags: [baseline]\ncontent: |\n  def solution_batch():\n      return {}\n",
                    paths.baseline_path,
                )

                with contextlib.redirect_stdout(io.StringIO()):
                    from station.eval_research.auto_evaluator import AutoResearchEvaluator

                    evaluator = AutoResearchEvaluator(DummyStation(), enabled=False)
                    evaluator._seed_system_baseline_if_needed()

                data = evaluator.eval_manager.get_evaluation("1")
                self.assertTrue(data["system_baseline"])
                self.assertEqual(data["artifacts"]["submission"], "storage/submission/1.py")
                self.assertEqual(data["attempts"][0]["submission_path"], str(Path(paths.submissions_dir) / "1.py"))
                self.assertNotIn("submission_snapshot", data["attempts"][0])
                self.assertEqual(
                    file_io_utils.load_text(str(Path(paths.submissions_dir) / "1.py")),
                    "def solution_batch():\n    return {}\n",
                )

                evaluator.eval_manager.complete_attempt(
                    "1",
                    1,
                    success=True,
                    score=1.0,
                    stdout="baseline stdout\n",
                    stderr="",
                    details={"Message": "baseline scored"},
                    status="completed",
                )
                evaluator._finalize_system_baseline("1", "def solution_batch():\n    return {}\n")
                finalized = evaluator.eval_manager.get_evaluation("1")
                final = finalized["final"]
                for blob_key in ("submission_snapshot", "stdout", "stdout_visible", "stderr", "coder_report"):
                    self.assertNotIn(blob_key, final)
                self.assertEqual(final["artifacts"]["report"], "storage/report/1.md")
                self.assertIn("This is a system baseline", file_io_utils.load_text(str(Path(paths.reports_dir) / "1.md")))
                self.assertEqual(file_io_utils.load_text(str(Path(paths.stdout_dir) / "1.log")), "baseline stdout\n")
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_coder_prompt_uses_storage_tmp_for_sage_workspace(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
        }

        class DummyStation:
            station_id = "prompt-test"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                paths = ensure_runtime_layout(constants)
                manager = EvaluationManager(paths.evaluations_dir)
                coder = ResearchCoderManager(DummyStation(), manager, paths=paths)

                prompt = coder._build_prompt(
                    {
                        "id": "123",
                        "lineage": "axiom",
                        "instruction": "Use Sage.",
                        "coder": {"max_attempts": 5},
                    }
                )

                self.assertTrue((Path(paths.tmp_storage) / "axiom").is_dir())
                self.assertIn("Disposable lineage scratch root: `storage/tmp/axiom`", prompt)
                self.assertIn("storage/tmp/axiom/eval_123", prompt)
                self.assertIn("storage/tmp/axiom/eval_123/sage", prompt)
                self.assertIn("temporary test scripts, probes, generated intermediates, caches, and mutable workspaces", prompt)
                self.assertIn('os.environ["HOME"] = "storage/tmp/axiom/eval_123/sage/home"', prompt)
                self.assertIn('os.environ["DOT_SAGE"] = "storage/tmp/axiom/eval_123/sage/dot_sage"', prompt)
                self.assertIn('os.environ["SAGE_DOTDIR"] = "storage/tmp/axiom/eval_123/sage/dot_sage"', prompt)
                self.assertIn('os.environ["XDG_CACHE_HOME"] = "storage/tmp/axiom/eval_123/sage/xdg_cache"', prompt)
                self.assertIn("Do not put Sage/CAS cache or workspace directories under `storage/axiom`", prompt)
                self.assertIn("Do not fork a Python child after importing Sage in the parent", prompt)
                self.assertIn("local probe behavior is not authoritative unless it runs the same submitted snapshot", prompt)
                self.assertNotIn("storage/axiom/data/sage", prompt)
                self.assertNotIn("agents cannot read", prompt)
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_research_task_spec_strips_coder_only_content_for_agents_and_keeps_it_for_coder_prompt(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
        }

        class DummyStation:
            station_id = "task-spec-test"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                paths = ensure_runtime_layout(constants)
                task_spec_path = Path(paths.task_spec_path)
                file_io_utils.save_text(
                    "Visible line\n__CODER_ONLY_BEGIN__\nHidden line\n__CODER_ONLY_END__\nVisible tail\n",
                    str(task_spec_path),
                )

                public_task_spec = load_task_spec_markdown(constants)
                self.assertIn("Visible line", public_task_spec)
                self.assertIn("Visible tail", public_task_spec)
                self.assertNotIn("Hidden line", public_task_spec)
                self.assertNotIn("__CODER_ONLY_BEGIN__", public_task_spec)

                manager = EvaluationManager(paths.evaluations_dir)
                coder = ResearchCoderManager(DummyStation(), manager, paths=paths)
                prompt = coder._build_prompt(
                    {
                        "id": "124",
                        "lineage": "axiom",
                        "instruction": "Use the hidden block.",
                        "coder": {"max_attempts": 5},
                    }
                )

                self.assertIn("Hidden line", prompt)
                self.assertIn("__CODER_ONLY_BEGIN__", prompt)
                self.assertIn("__CODER_ONLY_END__", prompt)
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_python_sandbox_exposes_author_storage_tmp_lineage(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_EVAL_PYTHON_CONDA_ENV": constants.RESEARCH_EVAL_PYTHON_CONDA_ENV,
        }

        class DummyAgentModule:
            @staticmethod
            def load_agent_data(author):
                return {constants.AGENT_LINEAGE_KEY: "axiom"}

        class DummyStation:
            station_id = "sandbox-test"
            agent_module = DummyAgentModule()

            def sync_top_research_submission_config(self, submission):
                pass

        class TmpStorageEvaluator(ResearchTaskEvaluator):
            def evaluate_submission(self, result, eval_id: str = None, author: str = None):
                return True, 1.0, {}

            def get_expected_function_name(self):
                return "solution_batch"

            def get_task_description(self):
                return "tmp storage sandbox smoke test"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                constants.RESEARCH_EVAL_PYTHON_CONDA_ENV = "station"
                paths = ensure_runtime_layout(constants)

                evaluator = AutoResearchEvaluator(DummyStation(), enabled=False)
                evaluator.timeout = 10
                eval_entry = {
                    constants.EVALUATION_ID_KEY: "321",
                    constants.EVALUATION_AUTHOR_KEY: "Axiom I",
                    constants.EVALUATION_CONTENT_KEY: (
                        "import os\n"
                        "def solution_batch():\n"
                        "    path = 'storage/tmp/axiom/eval_321/sage/home'\n"
                        "    os.makedirs(path, exist_ok=True)\n"
                        "    with open(os.path.join(path, 'marker.txt'), 'w', encoding='utf-8') as handle:\n"
                        "        handle.write('ok')\n"
                        "    return {'ok': True}\n"
                    ),
                }

                result = evaluator._execute_submission(eval_entry, TmpStorageEvaluator())

                self.assertTrue(result["success"], result.get("logs") or result.get("error"))
                self.assertEqual(
                    file_io_utils.load_text(str(Path(paths.tmp_storage) / "axiom" / "eval_321" / "sage" / "home" / "marker.txt")),
                    "ok",
                )
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_research_storage_resolver_does_not_create_unknown_lineages(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
        }
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                paths = ensure_runtime_layout(constants)
                room = ResearchCenter(eval_manager=EvaluationManager(paths.evaluations_dir), paths=paths)
                agent_data = {constants.AGENT_LINEAGE_KEY: "Axiom"}

                ok, resolved, display, consumes = room._resolve_storage_target(agent_data, "storage/109", constants)

                self.assertFalse(ok)
                self.assertIn("Lineage storage not found", resolved)
                self.assertIsNone(display)
                self.assertFalse(consumes)
                self.assertFalse((Path(paths.lineages_root) / "109").exists())

                ok, resolved, display, consumes = room._resolve_storage_target(agent_data, "storage/stdout", constants)

                self.assertFalse(ok)
                self.assertIn("not an agent-readable lineage storage path", resolved)
                self.assertIsNone(display)
                self.assertFalse(consumes)
                self.assertFalse((Path(paths.lineages_root) / "stdout").exists())
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_research_storage_resolver_creates_own_lineage_only(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
        }
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                paths = ensure_runtime_layout(constants)
                room = ResearchCenter(eval_manager=EvaluationManager(paths.evaluations_dir), paths=paths)
                agent_data = {constants.AGENT_LINEAGE_KEY: "Axiom"}

                ok, resolved, display, consumes = room._resolve_storage_target(agent_data, "storage/axiom", constants)

                self.assertTrue(ok)
                self.assertEqual(os.path.realpath(resolved), os.path.realpath(str(Path(paths.lineages_root) / "axiom")))
                self.assertEqual(display, "storage/axiom")
                self.assertTrue(consumes)
                self.assertTrue((Path(paths.lineages_root) / "axiom").is_dir())
                self.assertTrue((Path(paths.storage_real_root) / "axiom").is_symlink())
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_research_center_init_cleans_only_empty_invalid_lineage_dirs(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
        }
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                paths = ensure_runtime_layout(constants)

                invalid_empty = ["unknown", "stdout", "109", "114,", "206,", "1-40"]
                for name in invalid_empty:
                    file_io_utils.ensure_dir_exists(str(Path(paths.lineages_root) / name / "data"))
                    try:
                        (Path(paths.storage_real_root) / name).symlink_to(Path("lineages") / name)
                    except FileExistsError:
                        pass

                valid_empty = Path(paths.lineages_root) / "axiom"
                file_io_utils.ensure_dir_exists(str(valid_empty / "data"))
                protected_invalid = Path(paths.lineages_root) / "stderr"
                file_io_utils.ensure_dir_exists(str(protected_invalid / "data"))
                file_io_utils.save_text("keep", str(protected_invalid / "data" / "note.txt"))

                ResearchCenter(eval_manager=EvaluationManager(paths.evaluations_dir), paths=paths)

                for name in invalid_empty:
                    self.assertFalse((Path(paths.lineages_root) / name).exists(), name)
                    if name not in constants.RESEARCH_STORAGE_RESERVED_NAMES:
                        self.assertFalse((Path(paths.storage_real_root) / name).exists(), name)

                self.assertTrue(valid_empty.is_dir())
                self.assertTrue((valid_empty / "data").is_dir())
                self.assertTrue(protected_invalid.is_dir())
                self.assertEqual(file_io_utils.load_text(str(protected_invalid / "data" / "note.txt")), "keep")
                for root_dir in ["submission", "stdout", "stderr", "report", "tmp", "shared", "system", "lineages"]:
                    self.assertTrue((Path(paths.storage_real_root) / root_dir).exists(), root_dir)
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_compact_index_uses_normal_yaml_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self._make_manager(root)
            submission_path = self._create_eval_with_submission(manager, root)
            manager.register_attempt("204", submission_path)
            manager.complete_attempt("204", 1, True, 55.0, "stdout\n", "", {"Message": "scored"}, status="completed")
            manager.finalize_evaluation("204", "report\n", final_status="partial")
            manager.refresh_from_disk()

            compact = {item[constants.EVALUATION_ID_KEY]: item for item in manager.get_compact_display_infos()}
            self.assertEqual(compact["204"][constants.EVALUATION_SCORE_KEY], 55.0)
            self.assertEqual(compact["204"][constants.EVALUATION_TAGS_KEY], ["n50", "native-c"])
            self.assertEqual(manager.get_top_submission()["evaluation_id"], "204")

    def test_migration_writes_artifacts_and_strips_inline_blobs(self):
        original_data_path = constants.BASE_STATION_DATA_PATH
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                research_root = data_root / constants.ROOMS_DIR_NAME / constants.SHORT_ROOM_NAME_RESEARCH
                eval_dir = research_root / constants.RESEARCH_EVALUATIONS_SUBDIR_NAME
                file_io_utils.ensure_dir_exists(str(eval_dir))
                file_io_utils.save_yaml(
                    {
                        "schema_version": 2,
                        "id": "9",
                        "author": "Aletheia I",
                        "lineage": "aletheia",
                        "title": "Old inline eval",
                        "tags": ["old"],
                        "abstract": "Old inline blobs.",
                        "instruction": "Run.",
                        "submitted_tick": 1,
                        "status": "completed",
                        "attempts": [
                            {
                                "attempt": 1,
                                "status": "completed",
                                "submission_path": "storage/submission/9.py",
                                "submission_snapshot": "print('old')\n",
                                "stdout": "old stdout\n",
                                "stdout_visible": "old stdout\n",
                                "stderr": "",
                                "primary_score": 1.0,
                            }
                        ],
                        "final": {
                            "status": "completed",
                            "attempt": 1,
                            "primary_score": 1.0,
                            constants.EVALUATION_DETAILS_KEY: {"Message": "ok"},
                            "sort_key": None,
                            "submission_snapshot": "print('old')\n",
                            "stdout": "old stdout\n",
                            "stdout_visible": "old stdout\n",
                            "stderr": "",
                            "coder_report": "old report\n",
                        },
                        "notification": {"sent": True},
                    },
                    str(eval_dir / "9.yaml"),
                    sort_keys=False,
                )
                file_io_utils.save_yaml(
                    {
                        "schema_version": 2,
                        "id": "10",
                        "author": "Aletheia I",
                        "lineage": "aletheia",
                        "title": "Attempt only eval",
                        "tags": ["old"],
                        "abstract": "Old attempt blobs.",
                        "instruction": "Run.",
                        "submitted_tick": 1,
                        "status": "running",
                        "attempts": [
                            {
                                "attempt": 1,
                                "status": "failed",
                                "submission_path": "storage/submission/10.py",
                                "submission_snapshot": "print('attempt old')\n",
                                "stdout": "attempt stdout\n",
                                "stdout_visible": "attempt stdout\n",
                                "stderr": "attempt stderr\n",
                                "primary_score": "N/A",
                            }
                        ],
                        "notification": {"sent": False},
                    },
                    str(eval_dir / "10.yaml"),
                    sort_keys=False,
                )
                file_io_utils.save_yaml(
                    {
                        "schema_version": 2,
                        "id": "11",
                        "author": "Aletheia I",
                        "lineage": "aletheia",
                        "title": "Empty final code eval",
                        "tags": ["old"],
                        "abstract": "Final has no code snapshot.",
                        "instruction": "Run.",
                        "submitted_tick": 1,
                        "status": "blocked",
                        "artifacts": {"submission": "storage/submission/11.py"},
                        "attempts": [
                            {
                                "attempt": 1,
                                "status": "failed",
                                "submission_path": "storage/submission/11.py",
                                "stdout": "attempt stdout\n",
                                "stdout_visible": "attempt stdout\n",
                                "stderr": "",
                                "primary_score": "N/A",
                            }
                        ],
                        "final": {
                            "status": "blocked",
                            "attempt": 1,
                            "primary_score": "N/A",
                            constants.EVALUATION_DETAILS_KEY: "",
                            "sort_key": None,
                            "submission_snapshot": "",
                            "stdout": "attempt stdout\n",
                            "stdout_visible": "attempt stdout\n",
                            "stderr": "",
                            "coder_report": "blocked report\n",
                        },
                        "notification": {"sent": True},
                    },
                    str(eval_dir / "11.yaml"),
                    sort_keys=False,
                )
                file_io_utils.save_text("print('artifact exists')\n", str(research_root / "storage" / "submission" / "11.py"))
                file_io_utils.save_yaml(
                    {
                        "schema_version": 2,
                        "id": "12",
                        "author": "Aletheia I",
                        "lineage": "aletheia",
                        "title": "Empty final stderr eval",
                        "tags": ["old"],
                        "abstract": "Final stderr is empty after an earlier failed attempt.",
                        "instruction": "Run.",
                        "submitted_tick": 1,
                        "status": "partial",
                        "attempts": [
                            {
                                "attempt": 1,
                                "status": "timeout",
                                "submission_path": "storage/submission/12.py",
                                "submission_snapshot": "print('old')\n",
                                "stdout": "old stdout\n",
                                "stdout_visible": "old stdout\n",
                                "stderr": "stale timeout stderr\n",
                                "primary_score": "N/A",
                            }
                        ],
                        "final": {
                            "status": "partial",
                            "attempt": 2,
                            "primary_score": 0.0,
                            constants.EVALUATION_DETAILS_KEY: {"Message": "partial"},
                            "sort_key": None,
                            "submission_snapshot": "print('final')\n",
                            "stdout": "final stdout\n",
                            "stdout_visible": "final stdout\n",
                            "stderr": "",
                            "coder_report": "final report\n",
                        },
                        "notification": {"sent": True},
                    },
                    str(eval_dir / "12.yaml"),
                    sort_keys=False,
                )
                file_io_utils.save_text("", str(research_root / "storage" / "stderr" / "12.log"))
                file_io_utils.save_yaml(
                    {
                        "schema_version": 2,
                        "id": "13",
                        "author": "Aletheia I",
                        "lineage": "aletheia",
                        "title": "Trailing newline stderr eval",
                        "tags": ["old"],
                        "abstract": "Existing artifact has one trailing newline.",
                        "instruction": "Run.",
                        "submitted_tick": 1,
                        "status": "completed",
                        "attempts": [],
                        "final": {
                            "status": "completed",
                            "attempt": 1,
                            "primary_score": 1.0,
                            constants.EVALUATION_DETAILS_KEY: {"Message": "ok"},
                            "sort_key": None,
                            "submission_snapshot": "print('newline')\n",
                            "stdout": "newline stdout\n",
                            "stderr": "warning line",
                            "coder_report": "newline report\n",
                        },
                        "notification": {"sent": True},
                    },
                    str(eval_dir / "13.yaml"),
                    sort_keys=False,
                )
                file_io_utils.save_text("warning line\n", str(research_root / "storage" / "stderr" / "13.log"))

                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(migrate(apply=True, data_root=str(data_root)), 0)
                migrated = file_io_utils.load_yaml(str(eval_dir / "9.yaml"))
                final = migrated["final"]
                for blob_key in ("submission_snapshot", "stdout", "stdout_visible", "stderr", "coder_report"):
                    self.assertNotIn(blob_key, migrated["attempts"][0])
                    self.assertNotIn(blob_key, final)
                self.assertEqual(migrated["artifacts"]["submission"], "storage/submission/9.py")
                self.assertEqual(file_io_utils.load_text(str(research_root / "storage" / "submission" / "9.py")), "print('old')\n")
                self.assertEqual(file_io_utils.load_text(str(research_root / "storage" / "stdout" / "9.log")), "old stdout\n")
                self.assertEqual(file_io_utils.load_text(str(research_root / "storage" / "report" / "9.md")), "old report\n")

                attempt_only = file_io_utils.load_yaml(str(eval_dir / "10.yaml"))
                for blob_key in ("submission_snapshot", "stdout", "stdout_visible", "stderr", "coder_report"):
                    self.assertNotIn(blob_key, attempt_only["attempts"][0])
                self.assertEqual(file_io_utils.load_text(str(research_root / "storage" / "submission" / "10.py")), "print('attempt old')\n")
                self.assertEqual(file_io_utils.load_text(str(research_root / "storage" / "stdout" / "10.log")), "attempt stdout\n")
                self.assertEqual(file_io_utils.load_text(str(research_root / "storage" / "stderr" / "10.log")), "attempt stderr\n")

                empty_code = file_io_utils.load_yaml(str(eval_dir / "11.yaml"))
                self.assertNotIn("submission", empty_code["final"]["artifacts"])
                self.assertEqual(get_evaluation_code_info("11", str(eval_dir))["status"], "pending")
                self.assertEqual(get_evaluation_submission_payload("11", str(eval_dir))["code"], "Code not available")
                self.assertIn("attempt stdout", get_evaluation_submission_payload("11", str(eval_dir))["logs"])

                empty_final_stderr = file_io_utils.load_yaml(str(eval_dir / "12.yaml"))
                self.assertEqual(empty_final_stderr["final"]["artifacts"]["stderr"], "storage/stderr/12.log")
                self.assertEqual(file_io_utils.load_text(str(research_root / "storage" / "stderr" / "12.log")), "")
                self.assertEqual(file_io_utils.load_text(str(research_root / "storage" / "stdout" / "12.log")), "final stdout\n")
                self.assertEqual(file_io_utils.load_text(str(research_root / "storage" / "report" / "12.md")), "final report\n")

                trailing_newline = file_io_utils.load_yaml(str(eval_dir / "13.yaml"))
                self.assertEqual(trailing_newline["final"]["artifacts"]["stderr"], "storage/stderr/13.log")
                self.assertEqual(file_io_utils.load_text(str(research_root / "storage" / "stderr" / "13.log")), "warning line\n")
        finally:
            constants.BASE_STATION_DATA_PATH = original_data_path

    def test_migration_check_uses_sql_index_for_all_candidates(self):
        original_data_path = constants.BASE_STATION_DATA_PATH
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                research_root = data_root / constants.ROOMS_DIR_NAME / constants.SHORT_ROOM_NAME_RESEARCH
                eval_dir = research_root / constants.RESEARCH_EVALUATIONS_SUBDIR_NAME
                file_io_utils.ensure_dir_exists(str(eval_dir))
                file_io_utils.save_yaml(
                    {
                        "schema_version": 2,
                        "id": "1",
                        "author": "System",
                        "title": "Already migrated",
                        "status": "completed",
                        "artifacts": {
                            "submission": "storage/submission/1.py",
                            "stdout": "storage/stdout/1.log",
                            "stderr": "storage/stderr/1.log",
                            "report": "storage/report/1.md",
                        },
                        "attempts": [],
                    },
                    str(eval_dir / "1.yaml"),
                    sort_keys=False,
                )
                file_io_utils.save_yaml(
                    {
                        "schema_version": 2,
                        "id": "2",
                        "author": "Aletheia I",
                        "title": "Still legacy",
                        "status": "completed",
                        "attempts": [],
                        "final": {
                            "status": "completed",
                            "attempt": 1,
                            "submission_snapshot": "print('legacy')\n",
                            "stdout": "legacy stdout\n",
                            "stderr": "",
                            "coder_report": "legacy report\n",
                        },
                    },
                    str(eval_dir / "2.yaml"),
                    sort_keys=False,
                )

                self.assertTrue(needs_migration(data_root=str(data_root)))
                with mock.patch(
                    "station.eval_research.evaluation_index._iter_evaluation_files",
                    side_effect=AssertionError("unexpected evaluation YAML rescan"),
                ):
                    self.assertTrue(needs_migration(data_root=str(data_root)))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(migrate(apply=True, data_root=str(data_root)), 0)
                self.assertFalse(needs_migration(data_root=str(data_root)))
        finally:
            constants.BASE_STATION_DATA_PATH = original_data_path

    def test_migration_script_preflights_before_apply(self):
        original_data_path = constants.BASE_STATION_DATA_PATH
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                research_root = data_root / constants.ROOMS_DIR_NAME / constants.SHORT_ROOM_NAME_RESEARCH
                eval_dir = research_root / constants.RESEARCH_EVALUATIONS_SUBDIR_NAME
                script_path = Path(__file__).resolve().parents[1] / "scripts" / "migrate" / "migrate_research_eval_artifacts.py"
                file_io_utils.ensure_dir_exists(str(eval_dir))
                file_io_utils.save_yaml(
                    {
                        "schema_version": 2,
                        "id": "15",
                        "author": "Aletheia I",
                        "title": "Preflight script migration",
                        "status": "completed",
                        "attempts": [],
                        "final": {
                            "status": "completed",
                            "attempt": 1,
                            "submission_snapshot": "print('preflight')\n",
                            "stdout": "preflight stdout\n",
                            "stderr": "",
                            "coder_report": "preflight report\n",
                        },
                    },
                    str(eval_dir / "15.yaml"),
                    sort_keys=False,
                )

                result = subprocess.run(
                    [sys.executable, str(script_path), "--data-root", str(data_root)],
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )

                self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
                self.assertLess(result.stdout.index("DRY-RUN:"), result.stdout.index("APPLY:"))
                self.assertEqual(
                    file_io_utils.load_text(str(research_root / "storage" / "submission" / "15.py")),
                    "print('preflight')\n",
                )
        finally:
            constants.BASE_STATION_DATA_PATH = original_data_path

    def test_migration_preserves_existing_artifact_on_inline_mismatch(self):
        original_data_path = constants.BASE_STATION_DATA_PATH
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                research_root = data_root / constants.ROOMS_DIR_NAME / constants.SHORT_ROOM_NAME_RESEARCH
                eval_dir = research_root / constants.RESEARCH_EVALUATIONS_SUBDIR_NAME
                artifact_path = research_root / "storage" / "submission" / "14.py"
                file_io_utils.ensure_dir_exists(str(eval_dir))
                file_io_utils.save_yaml(
                    {
                        "schema_version": 2,
                        "id": "14",
                        "author": "Aletheia I",
                        "title": "Mismatched artifact",
                        "status": "completed",
                        "attempts": [],
                        "final": {
                            "status": "completed",
                            "attempt": 1,
                            "submission_snapshot": "print('inline')\n",
                            "stdout": "final stdout\n",
                            "stderr": "",
                            "coder_report": "final report\n",
                        },
                    },
                    str(eval_dir / "14.yaml"),
                    sort_keys=False,
                )
                file_io_utils.save_text("print('artifact')\n", str(artifact_path))

                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(migrate(apply=True, data_root=str(data_root)), 0)

                migrated = file_io_utils.load_yaml(str(eval_dir / "14.yaml"))
                self.assertNotIn("submission_snapshot", migrated["final"])
                self.assertEqual(migrated["final"]["artifacts"]["submission"], "storage/submission/14.py")
                self.assertEqual(file_io_utils.load_text(str(artifact_path)), "print('artifact')\n")
                self.assertIn("artifacts_existing_with_inline_mismatch=1", output.getvalue())
        finally:
            constants.BASE_STATION_DATA_PATH = original_data_path


if __name__ == "__main__":
    unittest.main()
