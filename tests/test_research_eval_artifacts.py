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
from station.eval_research import evaluation_index
from station.eval_research.coder_manager import ActiveCoderSession, ResearchCoderManager
from station.eval_research.base_evaluator import ResearchTaskEvaluator
from station.eval_research.auto_evaluator import AutoResearchEvaluator
from station.eval_research.runtime_paths import (
    LOCAL_PROBE_SNAPSHOT_FILENAME,
    LINEAGE_ALIAS_CONFLICT_SUFFIX,
    SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME,
    SUBMIT_EVAL_CLI_SNAPSHOT_FILENAME,
    build_runtime_paths,
    ensure_lineage_storage,
    ensure_runtime_layout,
    load_task_spec_markdown,
)
from station.eval_research.submit_eval_cli import submit_eval
from station.rooms.research_center import ResearchCenter
from scripts.set_scores_na import process_evaluation as set_scores_na_process_evaluation
from scripts.void_eval import void_evaluation as void_evaluation_record


class ResearchEvaluationArtifactTests(unittest.TestCase):
    def _make_manager(self, root: str) -> EvaluationManager:
        eval_dir = Path(root) / "evaluations"
        file_io_utils.ensure_dir_exists(str(eval_dir))
        return EvaluationManager(str(eval_dir))

    def test_evaluation_update_does_not_rescan_all_evaluation_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self._make_manager(temp_dir)
            manager.create_evaluation(
                eval_id="1",
                author="Ariadne II",
                title="Initial title",
                content="Run the experiment.",
                tick=1,
                tags=[],
                abstract="Initial abstract.",
                lineage="ariadne",
            )

            with mock.patch.object(
                evaluation_index,
                "_iter_evaluation_files",
                side_effect=AssertionError("unexpected evaluation YAML rescan"),
            ):
                updated = manager.update_evaluation(
                    "1",
                    lambda data: data.update({"title": "Updated title"}),
                )

            self.assertIsNotNone(updated)
            self.assertEqual("Updated title", manager.get_evaluation("1")["title"])

    def test_coder_launch_cleans_up_process_when_pid_update_fails(self):
        original_data_path = constants.BASE_STATION_DATA_PATH
        original_storage_path = constants.RESEARCH_STORAGE_BASE_PATH

        class DummyStation:
            station_id = "cleanup-test"

        class FakeBackend:
            supports_resume = False

            def prepare_launch(self, **kwargs):
                run_dir = Path(kwargs["run_dir"])
                return SimpleNamespace(
                    command=["fake-coder"],
                    stdin_text=None,
                    env_overrides={},
                    transcript_path=str(run_dir / "transcript.jsonl"),
                    stderr_path=str(run_dir / "stderr.txt"),
                    transcript_format="jsonl",
                    last_message_path=str(run_dir / "last_message.txt"),
                )

        class FakeProcess:
            pid = 424242

            def __init__(self):
                self.returncode = None
                self.terminated = False
                self.killed = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def kill(self):
                self.killed = True
                self.returncode = -9

            def wait(self, timeout=None):
                return self.returncode

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                constants.BASE_STATION_DATA_PATH = str(Path(temp_dir) / "station_data")
                constants.RESEARCH_STORAGE_BASE_PATH = None
                paths = ensure_runtime_layout(constants)
                manager = EvaluationManager(paths.evaluations_dir)
                eval_data = manager.create_evaluation(
                    eval_id="1",
                    author="Ariadne II",
                    title="Cleanup launch",
                    content="Run the experiment.",
                    tick=1,
                    tags=[],
                    abstract="Cleanup launch test.",
                    lineage="ariadne",
                    backend="codex",
                )
                coder = ResearchCoderManager(DummyStation(), manager, paths=paths)
                fake_process = FakeProcess()

                with mock.patch.object(coder, "_detect_backend_executable", return_value="fake-coder"), \
                    mock.patch(
                        "station.eval_research.coder_manager.get_cli_worker_backend",
                        return_value=FakeBackend(),
                    ), \
                    mock.patch(
                        "station.eval_research.coder_manager.subprocess.Popen",
                        return_value=fake_process,
                    ), \
                    mock.patch.object(
                        manager,
                        "update_evaluation",
                        side_effect=RuntimeError("lock timeout"),
                    ):
                    with self.assertRaises(RuntimeError):
                        coder.launch_for_evaluation(eval_data)

                self.assertTrue(fake_process.terminated)
                self.assertEqual({}, coder.active_sessions)
        finally:
            constants.BASE_STATION_DATA_PATH = original_data_path
            constants.RESEARCH_STORAGE_BASE_PATH = original_storage_path

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
                    coder._check_cli_job_transcript_idle_timeouts()

                killpg.assert_called_once_with(DummyProcess.pid, signal.SIGTERM)
                self.assertTrue(session.transcript_idle_timeout_triggered)
                self.assertIn("did not grow", session.transcript_idle_timeout_reason)
                failure = coder._classify_cli_job_failure(session.eval_id, session)
                self.assertEqual(failure.category, "infra_transient")
                self.assertTrue(failure.is_retryable_infra)
                updated = manager.get_evaluation("204")
                self.assertEqual(updated["coder"]["failure_category"], "codex_transcript_idle_timeout")
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_research_coder_runtime_env_uses_paths_station_data_root(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_SEED_BANK_ENABLED": constants.RESEARCH_SEED_BANK_ENABLED,
        }

        class DummyStation:
            station_id = "branch-env-test"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_multistart" / "4_job" / "station_data_s2"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                constants.RESEARCH_SEED_BANK_ENABLED = False
                paths = ensure_runtime_layout(constants)
                manager = EvaluationManager(paths.evaluations_dir)
                coder = ResearchCoderManager(DummyStation(), manager, paths=paths)
                env = coder._build_runtime_env()
                self.assertEqual(os.path.abspath(str(data_root)), env["STATION_BASE_DATA_PATH"])
                self.assertNotIn("STATION_RESEARCH_ROOT", env)
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

    def test_retryable_infra_failure_resets_resume_backoff_after_normal_progress(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_CODER_RESUME_BACKOFF_SECONDS": constants.RESEARCH_CODER_RESUME_BACKOFF_SECONDS,
        }

        class DummyStation:
            station_id = "resume-backoff-reset-test"

        class DummyProcess:
            pid = 6790

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
                    eval_id="4010",
                    author="Ariadne II",
                    title="Resume backoff reset",
                    content="Run it.",
                    tick=1,
                    tags=[],
                    abstract="Resume backoff reset test.",
                    lineage="ariadne",
                )
                manager.update_evaluation(
                    "4010",
                    lambda record: record.setdefault("coder", {}).update({"resume_count": 2}),
                )
                run_dir = Path(paths.coder_sessions_dir) / "codex_4010_spawn_2_deadbeef"
                file_io_utils.ensure_dir_exists(str(run_dir))
                transcript_path = run_dir / "transcript.jsonl"
                stderr_path = run_dir / "stderr.txt"
                file_io_utils.save_text(
                    '{"type":"thread.started","thread_id":"resume-token-4010"}\n'
                    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"working"}}\n'
                    '{"type":"error","message":"rate limit exceeded"}\n',
                    str(transcript_path),
                )
                file_io_utils.save_text("", str(stderr_path))

                coder = ResearchCoderManager(DummyStation(), manager, paths=paths)
                coder.active_sessions["4010"] = ActiveCoderSession(
                    eval_id="4010",
                    session_id="codex_4010_spawn_2_deadbeef",
                    run_dir=str(run_dir),
                    backend="codex",
                    transcript_format="jsonl",
                    process=DummyProcess(),
                    transcript_handle=DummyHandle(),
                    stderr_handle=DummyHandle(),
                    report_path=str(Path(paths.reports_dir) / "4010.md"),
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

                updated = manager.get_evaluation("4010")
                coder_state = updated["coder"]
                self.assertEqual(updated["status"], "running")
                self.assertEqual(coder_state["status"], "pending_resume")
                self.assertEqual(coder_state["resume_count"], 0)
                self.assertEqual(coder_state["resume_delay_seconds"], 123)
                self.assertGreaterEqual(coder_state["next_resume_timestamp"], before_poll + 123)
                self.assertLessEqual(coder_state["next_resume_timestamp"], after_poll + 123)
                self.assertNotIn("4010", coder.active_sessions)
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_retryable_infra_failure_resets_resume_backoff_after_progress_before_adjacent_failure_markers(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_CODER_RESUME_BACKOFF_SECONDS": constants.RESEARCH_CODER_RESUME_BACKOFF_SECONDS,
        }

        class DummyStation:
            station_id = "resume-backoff-adjacent-failure-reset-test"

        class DummyProcess:
            pid = 6792

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
                    eval_id="4012",
                    author="Ariadne II",
                    title="Resume backoff adjacent failure reset",
                    content="Run it.",
                    tick=1,
                    tags=[],
                    abstract="Resume backoff adjacent failure reset test.",
                    lineage="ariadne",
                )
                manager.update_evaluation(
                    "4012",
                    lambda record: record.setdefault("coder", {}).update({"resume_count": 2}),
                )
                run_dir = Path(paths.coder_sessions_dir) / "codex_4012_spawn_2_deadbeef"
                file_io_utils.ensure_dir_exists(str(run_dir))
                transcript_path = run_dir / "transcript.jsonl"
                stderr_path = run_dir / "stderr.txt"
                file_io_utils.save_text(
                    '{"type":"thread.started","thread_id":"resume-token-4012"}\n'
                    '{"type":"turn.started"}\n'
                    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"working"}}\n'
                    '{"type":"item.completed","item":{"id":"item_1","type":"command_execution","status":"completed"}}\n'
                    '{"type":"error","message":"Selected model is at capacity. Please try a different model."}\n'
                    '{"type":"turn.failed","error":{"message":"Selected model is at capacity. Please try a different model."}}\n',
                    str(transcript_path),
                )
                file_io_utils.save_text("", str(stderr_path))

                coder = ResearchCoderManager(DummyStation(), manager, paths=paths)
                coder.active_sessions["4012"] = ActiveCoderSession(
                    eval_id="4012",
                    session_id="codex_4012_spawn_2_deadbeef",
                    run_dir=str(run_dir),
                    backend="codex",
                    transcript_format="jsonl",
                    process=DummyProcess(),
                    transcript_handle=DummyHandle(),
                    stderr_handle=DummyHandle(),
                    report_path=str(Path(paths.reports_dir) / "4012.md"),
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

                updated = manager.get_evaluation("4012")
                coder_state = updated["coder"]
                self.assertEqual(updated["status"], "running")
                self.assertEqual(coder_state["status"], "pending_resume")
                self.assertEqual(coder_state["resume_count"], 0)
                self.assertEqual(coder_state["resume_delay_seconds"], 123)
                self.assertGreaterEqual(coder_state["next_resume_timestamp"], before_poll + 123)
                self.assertLessEqual(coder_state["next_resume_timestamp"], after_poll + 123)
                self.assertNotIn("4012", coder.active_sessions)
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_retryable_infra_failure_does_not_reset_resume_backoff_for_old_progress(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_CODER_RESUME_BACKOFF_SECONDS": constants.RESEARCH_CODER_RESUME_BACKOFF_SECONDS,
        }

        class DummyStation:
            station_id = "resume-backoff-old-progress-test"

        class DummyProcess:
            pid = 6791

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
                    eval_id="4011",
                    author="Ariadne II",
                    title="Resume backoff old progress",
                    content="Run it.",
                    tick=1,
                    tags=[],
                    abstract="Resume backoff old progress test.",
                    lineage="ariadne",
                )
                manager.update_evaluation(
                    "4011",
                    lambda record: record.setdefault("coder", {}).update({"resume_count": 1}),
                )
                run_dir = Path(paths.coder_sessions_dir) / "codex_4011_spawn_2_deadbeef"
                file_io_utils.ensure_dir_exists(str(run_dir))
                transcript_path = run_dir / "transcript.jsonl"
                stderr_path = run_dir / "stderr.txt"
                file_io_utils.save_text(
                    '{"type":"thread.started","thread_id":"resume-token-4011"}\n'
                    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"old work"}}\n'
                    '{"type":"turn.failed","error":{"message":"temporary backend hiccup"}}\n'
                    '{"type":"turn.started"}\n'
                    '{"type":"turn.failed","error":{"message":"Selected model is at capacity. Please try a different model."}}\n',
                    str(transcript_path),
                )
                file_io_utils.save_text("", str(stderr_path))

                coder = ResearchCoderManager(DummyStation(), manager, paths=paths)
                coder.active_sessions["4011"] = ActiveCoderSession(
                    eval_id="4011",
                    session_id="codex_4011_spawn_2_deadbeef",
                    run_dir=str(run_dir),
                    backend="codex",
                    transcript_format="jsonl",
                    process=DummyProcess(),
                    transcript_handle=DummyHandle(),
                    stderr_handle=DummyHandle(),
                    report_path=str(Path(paths.reports_dir) / "4011.md"),
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

                updated = manager.get_evaluation("4011")
                coder_state = updated["coder"]
                self.assertEqual(updated["status"], "running")
                self.assertEqual(coder_state["status"], "pending_resume")
                self.assertEqual(coder_state["resume_count"], 1)
                self.assertEqual(coder_state["resume_delay_seconds"], 456)
                self.assertGreaterEqual(coder_state["next_resume_timestamp"], before_poll + 456)
                self.assertLessEqual(coder_state["next_resume_timestamp"], after_poll + 456)
                self.assertNotIn("4011", coder.active_sessions)
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

    def test_finalize_removes_matching_eval_tmp_storage(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self._make_manager(root)
            submission_path = self._create_eval_with_submission(manager, root)
            eval_tmp = Path(root) / "storage" / "tmp" / "ariadne" / "eval_204"
            sibling_tmp = Path(root) / "storage" / "tmp" / "ariadne" / "eval_999"
            file_io_utils.ensure_dir_exists(str(eval_tmp))
            file_io_utils.ensure_dir_exists(str(sibling_tmp))
            file_io_utils.save_text("scratch", str(eval_tmp / "probe.txt"))
            file_io_utils.save_text("keep", str(sibling_tmp / "probe.txt"))

            manager.register_attempt("204", submission_path)
            manager.complete_attempt(
                "204",
                1,
                success=True,
                score=55.0,
                stdout="stdout\n",
                stderr="",
                details={"Message": "scored"},
                status="completed",
            )
            manager.finalize_evaluation("204", "# Coder Report\n\nFinal report.\n", final_status="completed")

            self.assertFalse(eval_tmp.exists())
            self.assertTrue(sibling_tmp.exists())
            self.assertEqual(file_io_utils.load_text(str(sibling_tmp / "probe.txt")), "keep")

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
                self.assertIn("export STATION_BASE_DATA_PATH=", script_text)
                eval_tool_text = file_io_utils.load_text(paths.eval_tool_script_path)
                self.assertIn("export STATION_BASE_DATA_PATH=", eval_tool_text)

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
                self.assertIn("wait patiently for `ATTEMPT_COMPLETE`", stdout)
                self.assertTrue((Path(paths.run_requests_dir) / "7_attempt_1.yaml").exists())
                manager = EvaluationManager(paths.evaluations_dir)
                self.assertIn("7", manager.get_running_instruction_eval_ids())
                active_ids = {item["evaluation_id"] for item in manager.get_active_evaluations()}
                self.assertIn("7", active_ids)
                self.assertFalse(Path(paths.evaluation_index_path).exists())
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_generated_local_probe_refreshes_limit_and_applies_optional_timeout(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_EVAL_MEMORY_LIMIT": constants.RESEARCH_EVAL_MEMORY_LIMIT,
        }
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                constants.RESEARCH_EVAL_MEMORY_LIMIT = "128m"
                paths = ensure_runtime_layout(constants)

                probe_script = Path(paths.local_probe_script_path)
                snapshot_path = (
                    Path(paths.research_root)
                    / SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME
                    / LOCAL_PROBE_SNAPSHOT_FILENAME
                )
                self.assertTrue(probe_script.is_file())
                self.assertTrue(os.access(probe_script, os.X_OK))
                self.assertTrue(snapshot_path.is_file())
                self.assertIn("_internal/local_probe_snapshot.py", probe_script.read_text(encoding="utf-8"))
                self.assertIn("MEMORY_LIMIT = '128m'", snapshot_path.read_text(encoding="utf-8"))

                completed = subprocess.run(
                    [str(probe_script), "--", sys.executable, "-c", "print('probe-ok')"],
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("probe-ok", completed.stdout)

                timed_out = subprocess.run(
                    [
                        str(probe_script),
                        "--timeout",
                        "0.1",
                        "--",
                        sys.executable,
                        "-c",
                        "import time; time.sleep(5)",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(timed_out.returncode, 124, timed_out.stderr)
                self.assertIn("timeout exceeded", timed_out.stderr)

                memory_limited = subprocess.run(
                    [
                        str(probe_script),
                        "--",
                        sys.executable,
                        "-c",
                        "x = bytearray(256 * 1024 * 1024); print(len(x))",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                self.assertNotEqual(memory_limited.returncode, 0)

                constants.RESEARCH_EVAL_MEMORY_LIMIT = "96m"
                ensure_runtime_layout(constants)
                refreshed_snapshot = snapshot_path.read_text(encoding="utf-8")
                self.assertIn("MEMORY_LIMIT = '96m'", refreshed_snapshot)
                self.assertNotIn("MEMORY_LIMIT = '128m'", refreshed_snapshot)
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_generated_submit_script_records_cpu_only_when_gpu_managed(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_EVAL_GPU_NUM": constants.RESEARCH_EVAL_GPU_NUM,
            "RESEARCH_EVAL_USE_DIFF_GPU": constants.RESEARCH_EVAL_USE_DIFF_GPU,
            "RESEARCH_EVAL_AVAILABLE_GPUS": constants.RESEARCH_EVAL_AVAILABLE_GPUS,
        }
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                constants.RESEARCH_EVAL_GPU_NUM = 1
                constants.RESEARCH_EVAL_USE_DIFF_GPU = False
                constants.RESEARCH_EVAL_AVAILABLE_GPUS = [0, 1]
                paths = ensure_runtime_layout(constants)
                manager = EvaluationManager(paths.evaluations_dir)
                manager.create_evaluation(
                    eval_id="10",
                    author="Aletheia I",
                    title="Snapshot CPU-only",
                    content="Run without GPU.",
                    tick=1,
                    tags=["cpu"],
                    abstract="Snapshot CPU-only attempt.",
                    lineage="aletheia",
                )
                file_io_utils.save_text("def solution_batch():\n    return {}\n", str(Path(paths.submissions_dir) / "10.py"))

                result = subprocess.run(
                    [paths.submit_script_path, "10", "--cpu-only"],
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )

                self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
                data = file_io_utils.load_yaml(str(Path(paths.evaluations_dir) / "10.yaml"))
                self.assertTrue(data["attempts"][0]["cpu_only"])
                run_request = file_io_utils.load_yaml(str(Path(paths.run_requests_dir) / "10_attempt_1.yaml"))
                self.assertTrue(run_request["cpu_only"])
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

    def test_submit_cli_records_cpu_only_attempt_when_gpu_managed(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_EVAL_GPU_NUM": constants.RESEARCH_EVAL_GPU_NUM,
            "RESEARCH_EVAL_USE_DIFF_GPU": constants.RESEARCH_EVAL_USE_DIFF_GPU,
            "RESEARCH_EVAL_AVAILABLE_GPUS": constants.RESEARCH_EVAL_AVAILABLE_GPUS,
        }
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                constants.RESEARCH_EVAL_GPU_NUM = 1
                constants.RESEARCH_EVAL_USE_DIFF_GPU = False
                constants.RESEARCH_EVAL_AVAILABLE_GPUS = [0, 1]
                paths = build_runtime_paths(constants)
                manager = EvaluationManager(paths.evaluations_dir)
                manager.create_evaluation(
                    eval_id="9",
                    author="Aletheia I",
                    title="CPU-only attempt",
                    content="Run without GPU.",
                    tick=1,
                    tags=["cpu"],
                    abstract="CPU-only attempt metadata.",
                    lineage="aletheia",
                )
                file_io_utils.save_text("def solution_batch():\n    return {}\n", str(Path(paths.submissions_dir) / "9.py"))

                self.assertEqual(submit_eval("9", cpu_only=True), 0)

                data = file_io_utils.load_yaml(str(Path(paths.evaluations_dir) / "9.yaml"))
                self.assertTrue(data["attempts"][0]["cpu_only"])
                run_request = file_io_utils.load_yaml(str(Path(paths.run_requests_dir) / "9_attempt_1.yaml"))
                self.assertTrue(run_request["cpu_only"])
                stdout = file_io_utils.load_text(str(Path(paths.stdout_dir) / "9.log"))
                self.assertIn("CPU-only mode requested", stdout)
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
            self.assertIn("Stdout/stderr are not shown here", review["message"])
            self.assertNotIn("visible stdout", review["message"])
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
            "RESEARCH_EVAL_GPU_NUM": constants.RESEARCH_EVAL_GPU_NUM,
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
                constants.RESEARCH_EVAL_GPU_NUM = None
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

    def test_coder_prompt_renders_local_probe_memory_limit(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_EVAL_MEMORY_LIMIT": constants.RESEARCH_EVAL_MEMORY_LIMIT,
        }

        class DummyStation:
            station_id = "prompt-local-probe-test"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                constants.RESEARCH_EVAL_MEMORY_LIMIT = "12g"
                paths = ensure_runtime_layout(constants)
                manager = EvaluationManager(paths.evaluations_dir)
                coder = ResearchCoderManager(DummyStation(), manager, paths=paths)

                prompt = coder._build_prompt(
                    {
                        "id": "124",
                        "lineage": "axiom",
                        "instruction": "Run a bounded probe.",
                        "coder": {"max_attempts": 5},
                    }
                )

                self.assertIn("configured official-attempt memory limit is `12g`", prompt)
                self.assertIn(
                    "bash local_probe.sh [--timeout SECONDS] -- <command>",
                    prompt,
                )
                self.assertIn("same `12g` memory limit", prompt)
                self.assertIn("If `--timeout` is omitted, no timeout is applied", prompt)
                self.assertNotIn("less than 50% of available RAM", prompt)
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_coder_prompt_uses_attempt_completion_not_score_as_stop_rule(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
        }

        class DummyStation:
            station_id = "prompt-stop-rule-test"

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
                        "id": "125",
                        "lineage": "axiom",
                        "instruction": "Run a diagnostic-only analysis.",
                        "coder": {"max_attempts": 5},
                    }
                )

                self.assertIn("Do not use multiple official attempts for score optimization.", prompt)
                self.assertIn(
                    "Avoid repeated full-run official attempts by using quick end-to-end tests and runtime estimates",
                    prompt,
                )
                self.assertIn("If the agent does not explicitly specify a compute budget", prompt)
                self.assertIn("use the maximum compute allowed by the task specification", prompt)
                self.assertIn("If the agent explicitly specifies a compute budget, you must respect it", prompt)
                self.assertIn("requested versus actual compute afterward", prompt)
                self.assertIn("periodically atomically checkpoint resumable state", prompt)
                self.assertIn("so a later evaluation can continue after timeout", prompt)
                self.assertIn("ATTEMPT_STATUS: completed", prompt)
                self.assertIn("Retry only when there was an exception or the result is materially misaligned", prompt)
                self.assertIn("A computational timeout is not itself a failed scientific result", prompt)
                self.assertIn("do not resubmit", prompt)
                self.assertIn("mark unfinished work as `UNDECIDED-AT-BUDGET`", prompt)
                self.assertIn("If the attempt timed out computationally, write the report instead of resubmitting", prompt)
                self.assertIn("the score may legitimately be `n.a.` for diagnostic or non-scorable work", prompt)
                self.assertNotIn("`failed`, `timeout`, or `rejected`", prompt)
                self.assertNotIn("A non-`n.a.` primary score", prompt)
                self.assertNotIn("agent explicitly asked for optimization", prompt)
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_coder_prompt_includes_lineage_memory_word_budget(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
        }

        class DummyStation:
            station_id = "prompt-memory-budget-test"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                paths = ensure_runtime_layout(constants)
                lineage_dir = Path(paths.storage_real_root) / "axiom"
                file_io_utils.ensure_dir_exists(str(lineage_dir))
                file_io_utils.save_text("alpha beta gamma\n", str(lineage_dir / "CODER.md"))
                manager = EvaluationManager(paths.evaluations_dir)
                coder = ResearchCoderManager(DummyStation(), manager, paths=paths)

                prompt = coder._build_prompt(
                    {
                        "id": "127",
                        "lineage": "axiom",
                        "instruction": "Use relevant lineage memory.",
                        "coder": {"max_attempts": 5},
                    }
                )

                self.assertIn("Lineage memory maintenance", prompt)
                self.assertIn("Current `storage/axiom/CODER.md` word count at launch: 3.", prompt)
                self.assertIn("future coder sessions working on submissions from the same lineage", prompt)
                self.assertIn("at or below 5,000 words", prompt)
                self.assertIn("Do not over-prune", prompt)
                self.assertIn("around 5,000 words is acceptable", prompt)
                self.assertIn("wc -w storage/axiom/CODER.md", prompt)
                self.assertIn("storage/axiom/data/notes/", prompt)
                self.assertIn("Prior evaluation inspection", prompt)
                self.assertIn("bash eval_tool.sh preview <ID>", prompt)
                self.assertIn("without printing coder prompts, raw code, stdout, or stderr", prompt)
                self.assertIn("Read raw evaluation YAML, submission code, stdout, or stderr only when the preview is insufficient", prompt)
                self.assertIn("Agents cannot see stdout/stderr in completion notifications or review", prompt)
                self.assertIn("summarize them in the Coder Report", prompt)
                self.assertIn("When the agent instruction asks for evidence, a diagnosis, or a direct comparison", prompt)
                self.assertIn("local artifacts are supplementary and must not be the only place the answer appears", prompt)
                self.assertIn("If the requested evidence was not measured, say so clearly", prompt)
                self.assertIn("Include important stdout information or requested raw data here when relevant", prompt)
                self.assertIn("State the total compute invested", prompt)
                self.assertIn("whether the execution was faithful to the agent's requested compute budget", prompt)
                self.assertIn("If the requested implementation or computation timed out", prompt)
                self.assertIn("estimate how much total additional compute would likely be needed", prompt)
                self.assertIn("requested-versus-actual compute gap", prompt)
                self.assertIn("sleep {time}", prompt)
                self.assertNotIn("sleep <module 'time'", prompt)
                self.assertIn("Token conservation", prompt)
                self.assertIn("Use targeted `rg`/`sed` reads", prompt)
                self.assertIn("avoid broad dumps over all of `storage/`, `evaluations/`, `coder_sessions/`, or `research_center/`", prompt)
                self.assertIn("Spend tokens when needed for correctness", prompt)
                self.assertIn("print scientifically valuable results and debugging-relevant diagnostics", prompt)
                self.assertIn("unexpected or interesting result that may be worth mentioning", prompt)
                self.assertIn("For optimization or search runs", prompt)
                self.assertIn("appeared converged, non-converged, or inconclusive", prompt)
                self.assertIn("continuing the same search process in a later evaluation appears promising", prompt)
                self.assertIn("do not give broader recommendations", prompt)
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_research_coder_working_directory_can_run_eval_tool_preview(self):
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
                eval_id = "128"
                file_io_utils.save_yaml(
                    {
                        "id": eval_id,
                        "title": "Eval tool reachable",
                        "author": "Axiom I",
                        "lineage": "axiom",
                        "tags": ["tool"],
                        "abstract": "Reachability abstract.",
                        "status": "completed",
                        "instruction": "Inspect this instruction.",
                        "artifacts": {
                            "report": os.path.join(constants.RESEARCH_STORAGE_DIR, "report", f"{eval_id}.md"),
                        },
                    },
                    os.path.join(paths.evaluations_dir, f"{eval_id}{constants.RESEARCH_EVALUATION_FILE_EXTENSION}"),
                    sort_keys=False,
                )
                file_io_utils.save_text("Reachable report.", os.path.join(paths.reports_dir, f"{eval_id}.md"))

                result = subprocess.run(
                    ["bash", "eval_tool.sh", "preview", eval_id],
                    cwd=paths.research_root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("# Research Evaluation Preview #128", result.stdout)
                self.assertIn("Inspect this instruction.", result.stdout)
                self.assertIn("Reachable report.", result.stdout)
                self.assertIn("Stdout log: `storage/stdout/128.log`", result.stdout)
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_coder_prompt_includes_cpu_only_tip_only_when_gpu_managed(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_EVAL_GPU_NUM": constants.RESEARCH_EVAL_GPU_NUM,
            "RESEARCH_EVAL_USE_DIFF_GPU": constants.RESEARCH_EVAL_USE_DIFF_GPU,
            "RESEARCH_EVAL_AVAILABLE_GPUS": constants.RESEARCH_EVAL_AVAILABLE_GPUS,
        }

        class DummyStation:
            station_id = "prompt-cpu-only-test"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                paths = ensure_runtime_layout(constants)
                manager = EvaluationManager(paths.evaluations_dir)
                coder = ResearchCoderManager(DummyStation(), manager, paths=paths)
                eval_data = {
                    "id": "126",
                    "lineage": "axiom",
                    "instruction": "Run a CPU-only diagnostic.",
                    "coder": {"max_attempts": 5},
                }

                constants.RESEARCH_EVAL_GPU_NUM = 1
                constants.RESEARCH_EVAL_USE_DIFF_GPU = False
                constants.RESEARCH_EVAL_AVAILABLE_GPUS = [0]
                managed_prompt = coder._build_prompt(eval_data)

                constants.RESEARCH_EVAL_GPU_NUM = None
                constants.RESEARCH_EVAL_USE_DIFF_GPU = False
                unmanaged_prompt = coder._build_prompt(eval_data)

                self.assertIn("bash submit_eval.sh 126 --cpu-only", managed_prompt)
                self.assertIn("hides CUDA devices", managed_prompt)
                self.assertIn("Local probe commands may have no GPU/CUDA access", managed_prompt)
                self.assertIn("Do not infer that the official attempt lacks GPU access", managed_prompt)
                self.assertIn("do not pass `--cpu-only`", managed_prompt)
                self.assertIn("let the scheduler allocate the GPU", managed_prompt)
                self.assertNotIn("--cpu-only", unmanaged_prompt)
                self.assertNotIn("Local probe commands may have no GPU/CUDA access", unmanaged_prompt)
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

                with mock.patch.object(constants, "RESEARCH_EVAL_GPU_NUM", None), \
                     mock.patch.object(constants, "RESEARCH_EVAL_USE_DIFF_GPU", False):
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

    def test_python_sandbox_exposes_cross_lineage_root_when_enabled(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_EVAL_PYTHON_CONDA_ENV": constants.RESEARCH_EVAL_PYTHON_CONDA_ENV,
            "RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS": constants.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS,
        }

        class DummyAgentModule:
            @staticmethod
            def load_agent_data(author):
                return {constants.AGENT_LINEAGE_KEY: "aster"}

        class DummyStation:
            station_id = "sandbox-cross-lineage-test"
            agent_module = DummyAgentModule()

            def sync_top_research_submission_config(self, submission):
                pass

        class CrossLineageEvaluator(ResearchTaskEvaluator):
            def evaluate_submission(self, result, eval_id: str = None, author: str = None):
                return result == "bridge seed", 1.0 if result == "bridge seed" else 0.0, {"result": result}

            def get_expected_function_name(self):
                return "solution_batch"

            def get_task_description(self):
                return "cross-lineage sandbox smoke test"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                constants.RESEARCH_EVAL_PYTHON_CONDA_ENV = "station"
                constants.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS = True
                paths = ensure_runtime_layout(constants)
                other_file = Path(paths.lineages_root) / "heuron" / "witnesses" / "bridge_45.json"
                file_io_utils.ensure_dir_exists(str(other_file.parent))
                file_io_utils.save_text("bridge seed", str(other_file))

                with mock.patch.object(constants, "RESEARCH_EVAL_GPU_NUM", None), \
                     mock.patch.object(constants, "RESEARCH_EVAL_USE_DIFF_GPU", False):
                    evaluator = AutoResearchEvaluator(DummyStation(), enabled=False)
                evaluator.timeout = 10
                eval_entry = {
                    constants.EVALUATION_ID_KEY: "4243",
                    constants.EVALUATION_AUTHOR_KEY: "Aster I",
                    constants.EVALUATION_CONTENT_KEY: (
                        "from pathlib import Path\n"
                        "def solution_batch():\n"
                        "    return Path('storage/lineages/heuron/witnesses/bridge_45.json').read_text(encoding='utf-8')\n"
                    ),
                }

                result = evaluator._execute_submission(eval_entry, CrossLineageEvaluator())

                self.assertTrue(result["success"], result.get("logs") or result.get("error"))
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_python_sandbox_exposes_legacy_top_level_storage_when_enabled(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_EVAL_PYTHON_CONDA_ENV": constants.RESEARCH_EVAL_PYTHON_CONDA_ENV,
            "RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS": constants.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS,
        }

        class DummyAgentModule:
            @staticmethod
            def load_agent_data(author):
                return {constants.AGENT_LINEAGE_KEY: "aster"}

        class DummyStation:
            station_id = "sandbox-legacy-storage-test"
            agent_module = DummyAgentModule()

            def sync_top_research_submission_config(self, submission):
                pass

        class LegacyStorageEvaluator(ResearchTaskEvaluator):
            def evaluate_submission(self, result, eval_id: str = None, author: str = None):
                return result == "legacy parent artifact", 1.0 if result == "legacy parent artifact" else 0.0, {"result": result}

            def get_expected_function_name(self):
                return "solution_batch"

            def get_task_description(self):
                return "legacy top-level storage sandbox smoke test"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                constants.RESEARCH_EVAL_PYTHON_CONDA_ENV = "station"
                constants.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS = True
                paths = ensure_runtime_layout(constants)
                legacy_file = Path(paths.storage_real_root) / "axioma" / "hadamard_parent_interlock_n42.json"
                file_io_utils.ensure_dir_exists(str(legacy_file.parent))
                file_io_utils.save_text("legacy parent artifact", str(legacy_file))

                with mock.patch.object(constants, "RESEARCH_EVAL_GPU_NUM", None), \
                     mock.patch.object(constants, "RESEARCH_EVAL_USE_DIFF_GPU", False):
                    evaluator = AutoResearchEvaluator(DummyStation(), enabled=False)
                evaluator.timeout = 10
                eval_entry = {
                    constants.EVALUATION_ID_KEY: "4244",
                    constants.EVALUATION_AUTHOR_KEY: "Aster I",
                    constants.EVALUATION_CONTENT_KEY: (
                        "from pathlib import Path\n"
                        "def solution_batch():\n"
                        "    return Path('storage/axioma/hadamard_parent_interlock_n42.json').read_text(encoding='utf-8')\n"
                    ),
                }

                result = evaluator._execute_submission(eval_entry, LegacyStorageEvaluator())

                self.assertTrue(result["success"], result.get("logs") or result.get("error"))
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_system_baseline_can_write_system_storage_only_during_startup_window(self):
        original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "RESEARCH_STORAGE_BASE_PATH": constants.RESEARCH_STORAGE_BASE_PATH,
            "RESEARCH_EVAL_PYTHON_CONDA_ENV": constants.RESEARCH_EVAL_PYTHON_CONDA_ENV,
            "RESEARCH_SYSTEM_BASELINE_WRITABLE_TICK_LIMIT": constants.RESEARCH_SYSTEM_BASELINE_WRITABLE_TICK_LIMIT,
        }

        class DummyAgentModule:
            @staticmethod
            def load_agent_data(author):
                return {constants.AGENT_LINEAGE_KEY: "system"}

        class DummyStation:
            station_id = "system-baseline-write-test"
            agent_module = DummyAgentModule()

            def sync_top_research_submission_config(self, submission):
                pass

        class TinyEvaluator(ResearchTaskEvaluator):
            def evaluate_submission(self, result, eval_id: str = None, author: str = None):
                return True, 1.0, {}

            def get_expected_function_name(self):
                return "solution_batch"

            def get_task_description(self):
                return "system storage write smoke test"

        def dir_mode(path: Path) -> int:
            return path.stat().st_mode & 0o777

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "station_data"
                constants.BASE_STATION_DATA_PATH = str(data_root)
                constants.RESEARCH_STORAGE_BASE_PATH = None
                constants.RESEARCH_EVAL_PYTHON_CONDA_ENV = "station"
                constants.RESEARCH_SYSTEM_BASELINE_WRITABLE_TICK_LIMIT = 5
                paths = ensure_runtime_layout(constants)
                system_path = Path(paths.system_storage)

                with mock.patch.object(constants, "RESEARCH_EVAL_GPU_NUM", None), \
                     mock.patch.object(constants, "RESEARCH_EVAL_USE_DIFF_GPU", False):
                    evaluator = AutoResearchEvaluator(DummyStation(), enabled=False)
                evaluator.timeout = 10
                code = (
                    "from pathlib import Path\n"
                    "def solution_batch():\n"
                    "    Path('storage/system/generated.txt').write_text('ok', encoding='utf-8')\n"
                    "    return {'ok': True}\n"
                )
                startup_entry = {
                    constants.EVALUATION_ID_KEY: "baseline-startup",
                    constants.EVALUATION_AUTHOR_KEY: "System",
                    constants.EVALUATION_CONTENT_KEY: code,
                    constants.EVALUATION_SUBMITTED_TICK_KEY: 0,
                    "current_tick": 0,
                    "system_baseline": True,
                }

                startup_result = evaluator._execute_submission(startup_entry, TinyEvaluator())

                self.assertTrue(startup_result["success"], startup_result.get("logs") or startup_result.get("error"))
                self.assertEqual(file_io_utils.load_text(str(system_path / "generated.txt")), "ok")
                self.assertEqual(dir_mode(system_path), 0o555)

                late_entry = dict(startup_entry)
                late_entry[constants.EVALUATION_ID_KEY] = "baseline-late"
                late_entry["current_tick"] = 6
                late_entry[constants.EVALUATION_CONTENT_KEY] = (
                    "from pathlib import Path\n"
                    "def solution_batch():\n"
                    "    Path('storage/system/late.txt').write_text('late', encoding='utf-8')\n"
                    "    return {'ok': True}\n"
                )

                late_result = evaluator._execute_submission(late_entry, TinyEvaluator())

                self.assertFalse(late_result["success"])
                self.assertFalse((system_path / "late.txt").exists())
                self.assertEqual(dir_mode(system_path), 0o555)
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

    def test_runtime_layout_migrates_split_lineage_storage_and_suffixes_conflicts(self):
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
                canonical = Path(ensure_lineage_storage(paths, "axiom"))
                alias = Path(paths.storage_real_root) / "axiom"

                (canonical / "data" / "canonical_only.txt").write_text("canonical", encoding="utf-8")
                (canonical / "data" / "identical.txt").write_text("same", encoding="utf-8")
                (canonical / "data" / "conflict.txt").write_text("canonical version", encoding="utf-8")

                alias.unlink()
                (alias / "data" / "nested").mkdir(parents=True)
                (alias / "data" / "legacy_only.txt").write_text("legacy", encoding="utf-8")
                (alias / "data" / "nested" / "artifact.json").write_text("{}", encoding="utf-8")
                (alias / "data" / "identical.txt").write_text("same", encoding="utf-8")
                (alias / "data" / "conflict.txt").write_text("legacy version", encoding="utf-8")

                ensure_runtime_layout(constants)

                self.assertTrue(alias.is_symlink())
                self.assertEqual(alias.resolve(), canonical.resolve())
                self.assertEqual((canonical / "data" / "canonical_only.txt").read_text(), "canonical")
                self.assertEqual((canonical / "data" / "legacy_only.txt").read_text(), "legacy")
                self.assertEqual((canonical / "data" / "nested" / "artifact.json").read_text(), "{}")
                self.assertEqual((canonical / "data" / "identical.txt").read_text(), "same")
                self.assertEqual((canonical / "data" / "conflict.txt").read_text(), "legacy version")

                self.assertEqual(
                    (canonical / "data" / f"conflict.txt{LINEAGE_ALIAS_CONFLICT_SUFFIX}").read_text(),
                    "canonical version",
                )
                self.assertEqual(
                    (canonical / "data" / f"identical.txt{LINEAGE_ALIAS_CONFLICT_SUFFIX}").read_text(),
                    "same",
                )

                ensure_runtime_layout(constants)

                self.assertTrue(alias.is_symlink())
                self.assertFalse(
                    (canonical / "data" / f"conflict.txt{LINEAGE_ALIAS_CONFLICT_SUFFIX}.2").exists()
                )
        finally:
            for key, value in original_values.items():
                setattr(constants, key, value)

    def test_ensure_lineage_storage_rejects_wrong_symlink_target(self):
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
                alias = Path(paths.storage_real_root) / "axiom"
                alias.symlink_to(Path("lineages") / "other")

                with self.assertRaisesRegex(RuntimeError, "points to"):
                    ensure_lineage_storage(paths, "axiom")
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

    def test_breakthrough_summary_uses_persisted_progress_records(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self._make_manager(root)
            for eval_id, track, score, tick in [
                ("201", "d3", 0.95, 10),
                ("202", "d3", 0.90, 20),
                ("203", "d4", 0.80, 30),
            ]:
                submission_path = self._create_eval_with_submission(manager, root, eval_id=eval_id)
                manager.update_evaluation(eval_id, lambda data, tick=tick: data.update({"submitted_tick": tick}))
                manager.register_attempt(eval_id, submission_path)
                manager.complete_attempt(
                    eval_id,
                    1,
                    True,
                    score,
                    "stdout\n",
                    "",
                    {"Message": "scored"},
                    progress_records=[
                        {
                            "track": f"dimension:{track}",
                            "rank_key": [-score],
                            "value": score,
                            "label": f"{track} density",
                            "metadata": {"dimension": track},
                        }
                    ],
                    status="completed",
                )
                manager.finalize_evaluation(eval_id, "report\n")

            summary = manager.get_latest_breakthrough_summary()
            self.assertIn("global", summary["frontiers"])
            self.assertEqual(summary["frontiers"]["dimension:d3"]["evaluation_id"], "202")
            self.assertEqual(summary["frontiers"]["dimension:d3"]["value"], 0.90)
            self.assertEqual(summary["frontiers"]["dimension:d4"]["evaluation_id"], "203")
            self.assertEqual(summary["last_breakthrough_tick"], 30)


if __name__ == "__main__":
    unittest.main()
