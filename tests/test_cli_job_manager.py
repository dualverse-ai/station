import io
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from station import constants
from station.eval_archive.surveyor import AutoArchiveSurveyor
from station.eval_research.coder_manager import ResearchAuditManager, ResearchCoderManager
from station.eval_research.evaluation_manager import EvaluationManager
from station.eval_research.runtime_paths import ensure_runtime_layout
from station.multistart.admin import AdminSelectionManager
from station.workers.cli import CodexCliWorkerBackend
from station.workers.job_manager import CliJobLaunchDecision, CliJobLaunchSpec, CliJobManager
from web_interface.archive_survey_service import WebArchiveSurveyService


class CliJobManagerTests(unittest.TestCase):
    def test_all_missing_output_cli_exits_are_retryable_by_default(self):
        class Manager(CliJobManager):
            pass

        session = types.SimpleNamespace(transcript_idle_timeout_reason=None)
        failure = Manager()._classify_cli_job_failure("17", session)

        self.assertEqual("infra_transient", failure.category)
        self.assertTrue(failure.is_retryable_infra)
        self.assertIn("required completion artifact", failure.reason.lower())

    def test_transcript_idle_timeout_is_retryable(self):
        session = types.SimpleNamespace(
            transcript_idle_timeout_reason="Codex transcript did not grow.",
        )

        failure = CliJobManager()._classify_cli_job_failure("18", session)

        self.assertEqual("infra_transient", failure.category)
        self.assertTrue(failure.is_retryable_infra)
        self.assertEqual("Codex transcript did not grow.", failure.reason)

    def test_coder_surveyor_and_admin_share_one_job_manager(self):
        worker_classes = (
            ResearchCoderManager,
            ResearchAuditManager,
            AutoArchiveSurveyor,
            AdminSelectionManager,
            WebArchiveSurveyService,
        )
        for worker_class in worker_classes:
            with self.subTest(worker_class=worker_class.__name__):
                self.assertTrue(issubclass(worker_class, CliJobManager))
                self.assertIs(
                    worker_class._classify_cli_job_failure,
                    CliJobManager._classify_cli_job_failure,
                )
                self.assertTrue(worker_class.fresh_attempt_after_resume_exhaustion)

    def test_shared_job_manager_applies_provider_only_policy_to_codex_workers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = CliJobLaunchSpec(
                executable="codex",
                run_dir=temp_dir,
                backend="codex",
                model_name=None,
                workspace_root=temp_dir,
                storage_root=temp_dir,
                prompt="prompt",
                env={"CODEX_HOME": temp_dir},
            )
            decision = CliJobLaunchDecision(
                mode="fresh",
                spawn_count=1,
                resume_count=0,
                resume_token=None,
            )

            prepared = CliJobManager._prepare_cli_job_launch(spec, decision)

        self.assertIn('web_search="disabled"', prepared.command)
        self.assertIn("features.network_proxy.enabled=true", prepared.command)
        self.assertTrue(any(
            value.startswith("features.network_proxy.domains=")
            and '"api.openai.com" = "allow"' in value
            for value in prepared.command
        ))

    def test_codex_without_dedicated_override_preserves_existing_launch_behavior(self):
        backend = CodexCliWorkerBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = backend.prepare_launch(
                executable="codex",
                run_dir=temp_dir,
                model_name="gpt-5.5",
                storage_root=temp_dir,
                workspace_root=temp_dir,
                prompt="prompt",
                runtime_env={"CODEX_HOME": temp_dir},
            )

        self.assertFalse(any(value == 'model_provider="alt"' for value in prepared.command))
        self.assertFalse(any(value.startswith("model_providers.alt=") for value in prepared.command))
        self.assertEqual(
            {"OPENAI_BASE_URL": None, "OPENAI_API_KEY": None},
            prepared.env_overrides,
        )

    def test_codex_dedicated_override_configures_alt_provider_for_fresh_and_resume(self):
        backend = CodexCliWorkerBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_env = {
                "CODEX_HOME": temp_dir,
                "CODEX_API_KEY": "codex-key",
                "CODEX_BASE_URL": "https://codex.example/v1",
            }
            fresh = backend.prepare_launch(
                executable="codex",
                run_dir=temp_dir,
                model_name="gpt-5.5",
                storage_root=temp_dir,
                workspace_root=temp_dir,
                prompt="prompt",
                runtime_env=runtime_env,
            )
            resumed = backend.prepare_resume_launch(
                executable="codex",
                run_dir=temp_dir,
                model_name="gpt-5.5",
                storage_root=temp_dir,
                workspace_root=temp_dir,
                resume_token="thread-id",
                prompt="resume",
                runtime_env=runtime_env,
            )

        expected_provider = (
            'model_providers.alt={ name="Alt", base_url="https://codex.example/v1", '
            'env_key="CODEX_API_KEY", wire_api="responses" }'
        )
        for prepared in (fresh, resumed):
            self.assertIn('model_provider="alt"', prepared.command)
            self.assertIn(expected_provider, prepared.command)
            self.assertFalse(any("supports_websockets" in value for value in prepared.command))
            self.assertEqual(
                {
                    "OPENAI_BASE_URL": None,
                    "OPENAI_API_KEY": None,
                    "CODEX_ACCESS_TOKEN": None,
                    "CODEX_AUTH": None,
                },
                prepared.env_overrides,
            )
            self.assertLess(prepared.command.index('model_provider="alt"'), prepared.command.index("exec"))

    def test_codex_dedicated_override_requires_key_and_base_url_together(self):
        backend = CodexCliWorkerBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            base_kwargs = {
                "executable": "codex",
                "run_dir": temp_dir,
                "model_name": None,
                "storage_root": temp_dir,
                "workspace_root": temp_dir,
                "prompt": "prompt",
            }
            for runtime_env, missing_name in (
                ({"CODEX_HOME": temp_dir, "CODEX_API_KEY": "codex-key"}, "CODEX_BASE_URL"),
                ({"CODEX_HOME": temp_dir, "CODEX_BASE_URL": "https://codex.example/v1"}, "CODEX_API_KEY"),
            ):
                with self.subTest(missing_name=missing_name):
                    with self.assertRaisesRegex(ValueError, missing_name):
                        backend.prepare_launch(**base_kwargs, runtime_env=runtime_env)

    def test_coder_resume_budget_exhaustion_schedules_fresh_session(self):
        original_data_path = constants.BASE_STATION_DATA_PATH
        original_storage_path = constants.RESEARCH_STORAGE_BASE_PATH
        pauses = []

        class DummyStation:
            station_id = "shared-manager-resume-exhaustion"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                constants.BASE_STATION_DATA_PATH = str(Path(temp_dir) / "station_data")
                constants.RESEARCH_STORAGE_BASE_PATH = None
                paths = ensure_runtime_layout(constants)
                manager = EvaluationManager(paths.evaluations_dir)
                manager.create_evaluation(
                    eval_id="701",
                    author="Ariadne II",
                    title="Resume exhaustion",
                    content="Run it.",
                    tick=1,
                    tags=[],
                    abstract="Resume exhaustion characterization.",
                    lineage="ariadne",
                )

                def mark_pending_resume(record):
                    record["status"] = "running"
                    coder = record.setdefault("coder", {})
                    coder.update(
                        {
                            "active": True,
                            "active_pid": None,
                            "status": "pending_resume",
                            "spawn_count": 1,
                            "resume_count": 2,
                            "max_resumes": 2,
                            "resume_token": "thread-701",
                        }
                    )

                manager.update_evaluation("701", mark_pending_resume)
                coder_manager = ResearchCoderManager(
                    DummyStation(),
                    manager,
                    paths=paths,
                    pause_callback=pauses.append,
                )

                self.assertFalse(coder_manager.launch_for_evaluation(manager.get_evaluation("701")))
                updated = manager.get_evaluation("701")
                self.assertEqual("queued", updated["status"])
                self.assertEqual("retry", updated["coder"]["status"])
                self.assertEqual("resume_limit_exceeded", updated["coder"]["failure_category"])
                self.assertEqual(0, updated["coder"]["resume_count"])
                self.assertIsNone(updated["coder"]["resume_token"])
                self.assertEqual([], pauses)
        finally:
            constants.BASE_STATION_DATA_PATH = original_data_path
            constants.RESEARCH_STORAGE_BASE_PATH = original_storage_path

    def test_failed_completion_hook_keeps_finished_session_retryable(self):
        class Manager(CliJobManager):
            def _check_cli_job_transcript_idle_timeouts(self):
                pass

            def _cli_job_completion_ready(self, job_id, session):
                return True

            def _on_cli_job_completed(self, job_id, session, returncode):
                raise RuntimeError("persistence failed")

        session = types.SimpleNamespace(
            process=types.SimpleNamespace(poll=lambda: 0),
            transcript_handle=io.StringIO(),
            stderr_handle=io.StringIO(),
        )
        manager = Manager()
        manager.active_sessions["1"] = session

        with self.assertRaisesRegex(RuntimeError, "persistence failed"):
            manager.poll_cli_jobs()

        self.assertIs(session, manager.active_sessions["1"])
        self.assertTrue(session.transcript_handle.closed)
        self.assertTrue(session.stderr_handle.closed)

    def test_pid_persistence_abort_escalates_after_terminate_timeout(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("codex", 5), 0]
        session = types.SimpleNamespace(process=process)
        manager = CliJobManager()
        manager._terminate_cli_job_process = mock.Mock()

        manager._abort_cli_job_process(session)

        self.assertEqual(
            [mock.call(session, force=False), mock.call(session, force=True)],
            manager._terminate_cli_job_process.call_args_list,
        )
        self.assertEqual([mock.call(timeout=5), mock.call(timeout=5)], process.wait.call_args_list)

    def test_shutdown_hook_runs_after_signal_before_handles_close(self):
        events = []

        class Handle:
            def close(self):
                events.append("close")

        session = types.SimpleNamespace(transcript_handle=Handle(), stderr_handle=Handle())
        manager = CliJobManager()
        manager.active_sessions["1"] = session
        manager._terminate_cli_job_process = lambda active, force=False: events.append("signal")

        manager.stop_active_cli_jobs(
            after_terminate=lambda job_id, active: events.append("requeue")
        )

        self.assertEqual(["signal", "requeue", "close", "close"], events)
        self.assertFalse(manager.active_sessions)


if __name__ == "__main__":
    unittest.main()
