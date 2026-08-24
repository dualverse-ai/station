import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from station import constants, dashboard_statistics, file_io_utils, runtime_api_config
from station.eval_external import pending_queue
from station.eval_external.auto_external_reporter import AutoExternalReporter
from station.eval_research.coder_helpers import (
    build_coder_access_metadata,
    is_coder_internet_allowed,
    render_coder_internet_policy,
)
from station.eval_research.coder_manager import ResearchCoderManager
from station.eval_research.evaluation_manager import EvaluationManager
from station.eval_research.runtime_paths import ensure_runtime_layout
from station.multistart.branch_worker import _quiescent
from station.rooms.external_counter import _EXTERNAL_COUNTER_HELP
from station.station import Station
from station.station_runner import Orchestrator
from station.system_messages import build_station_level_system_prompt, build_tenured_reached_message
from station.workers.cli import CodexCliWorkerBackend, get_codex_provider_domains


class _AgentModuleStub:
    def __init__(self):
        self.notifications = []

    def add_pending_notification_atomic(self, author, message):
        self.notifications.append((author, message))


class _TransientStatusError(Exception):
    status_code = 504
    body = {"retry_after": 12}
    response = SimpleNamespace(headers={})


class ExternalReporterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base_path_patch = patch.object(constants, "BASE_STATION_DATA_PATH", self.temp_dir.name)
        self.base_path_patch.start()
        self.addCleanup(self.base_path_patch.stop)

        self.agent_module = _AgentModuleStub()
        self.station = SimpleNamespace(
            agent_module=self.agent_module,
            _get_current_tick=lambda: 17,
        )
        self.reporter = AutoExternalReporter(self.station, enabled=True)
        self.addCleanup(AutoExternalReporter._active_instances.pop, id(self.station), None)

    @staticmethod
    def _report(report_id="1", status=None):
        return {
            constants.EXTERNAL_REPORT_ID_KEY: report_id,
            constants.EXTERNAL_REPORT_TITLE_KEY: "Test report",
            constants.EXTERNAL_REPORT_AUTHOR_KEY: "Ada",
            constants.EXTERNAL_REPORT_PROMPT_KEY: "Find sources.",
            constants.EXTERNAL_REPORT_SUBMITTED_TICK_KEY: 15,
            constants.EXTERNAL_REPORT_STATUS_KEY: status or constants.EXTERNAL_REPORT_STATUS_PENDING,
        }

    def _save_report(self, report):
        report_id = report[constants.EXTERNAL_REPORT_ID_KEY]
        file_io_utils.save_yaml(
            report,
            os.path.join(self.reporter.reports_dir, f"report_{report_id}.yaml"),
        )

    def test_streaming_request_has_no_max_tool_calls(self):
        events = [
            SimpleNamespace(type="response.output_text.delta", delta="hello "),
            SimpleNamespace(type="response.output_text.delta", delta="world"),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(output_text="ignored")),
        ]
        responses = SimpleNamespace(create=Mock(return_value=events))
        client = SimpleNamespace(base_url="https://api.openai.com/v1", responses=responses)

        result = self.reporter._request_report(client, "prompt")

        self.assertEqual(result, "hello world")
        kwargs = responses.create.call_args.kwargs
        self.assertTrue(kwargs["stream"])
        self.assertNotIn("max_tool_calls", kwargs)
        self.assertEqual(kwargs["input"], "prompt")

    def test_transient_failure_retries_then_completes(self):
        report = self._report()
        self._save_report(report)

        with (
            patch.object(self.reporter, "_build_openai_client", return_value=object()),
            patch.object(
                self.reporter,
                "_request_report",
                side_effect=[_TransientStatusError("gateway timeout"), "completed report"],
            ) as request_report,
            patch("station.eval_external.auto_external_reporter.time.sleep") as sleep,
        ):
            self.reporter._run_report(report)

        self.assertEqual(request_report.call_count, 2)
        sleep.assert_called_once_with(60.0)
        persisted = self.reporter._load_report("1")
        self.assertEqual(persisted[constants.EXTERNAL_REPORT_STATUS_KEY], constants.EXTERNAL_REPORT_STATUS_COMPLETED)
        self.assertEqual(persisted[constants.EXTERNAL_REPORT_CONTENT_KEY], "completed report")
        self.assertEqual(len(self.agent_module.notifications), 1)

    def test_retry_delay_matches_llm_connector_schedule(self):
        no_fallback = {}
        self.assertEqual(
            [self.reporter._retry_delay_seconds(attempt, no_fallback) for attempt in range(6)],
            [60.0, 60.0, 120.0, 120.0, 240.0, 240.0],
        )
        self.assertEqual(
            self.reporter._retry_delay_seconds(0, {"provider_fallback_enabled": True}),
            0.0,
        )
        self.assertEqual(
            self.reporter._retry_delay_seconds(
                5,
                {
                    "provider_fallback_enabled": True,
                    "cycle_wrapped": True,
                    "cycle_count": 1,
                },
            ),
            60.0,
        )

    def test_recovery_requeues_running_and_orphaned_pending_reports(self):
        running = self._report("1", constants.EXTERNAL_REPORT_STATUS_RUNNING)
        running[constants.EXTERNAL_REPORT_START_TICK_KEY] = 16
        pending = self._report("2", constants.EXTERNAL_REPORT_STATUS_PENDING)
        self._save_report(running)
        self._save_report(pending)

        self.reporter._recover_interrupted_reports()
        self.reporter._recover_interrupted_reports()

        recovered_running = self.reporter._load_report("1")
        self.assertEqual(
            recovered_running[constants.EXTERNAL_REPORT_STATUS_KEY],
            constants.EXTERNAL_REPORT_STATUS_PENDING,
        )
        self.assertNotIn(constants.EXTERNAL_REPORT_START_TICK_KEY, recovered_running)
        queued = pending_queue.load(self.reporter.pending_reports_file)
        self.assertEqual({entry[constants.EXTERNAL_REPORT_ID_KEY] for entry in queued}, {"1", "2"})

    def test_job_statistics_include_persisted_and_queued_reports_without_duplicates(self):
        running = self._report("1", constants.EXTERNAL_REPORT_STATUS_RUNNING)
        pending = self._report("2", constants.EXTERNAL_REPORT_STATUS_PENDING)
        self._save_report(running)
        self._save_report(pending)
        self.assertTrue(pending_queue.append(self.reporter.pending_reports_file, pending, constants.EXTERNAL_REPORT_ID_KEY))

        stats = self.reporter.get_job_statistics()

        self.assertEqual(["1"], [job[constants.EXTERNAL_REPORT_ID_KEY] for job in stats["running_jobs"]])
        self.assertEqual(["2"], [job[constants.EXTERNAL_REPORT_ID_KEY] for job in stats["queued_jobs"]])
        self.assertEqual("external_report", stats["running_jobs"][0]["job_type"])

    def test_lightweight_job_statistics_normalize_fields_without_history_scan(self):
        running = self._report("1", constants.EXTERNAL_REPORT_STATUS_RUNNING)
        running[constants.EXTERNAL_REPORT_REQUEUE_COUNT_KEY] = 1
        queued = self._report("2", constants.EXTERNAL_REPORT_STATUS_PENDING)
        self._save_report(running)
        self._save_report(queued)
        self.reporter.active_futures["1"] = object()
        self.assertTrue(pending_queue.append(
            self.reporter.pending_reports_file,
            queued,
            constants.EXTERNAL_REPORT_ID_KEY,
        ))

        with patch.object(
            self.reporter,
            "_load_all_reports",
            side_effect=AssertionError("dashboard refresh must not scan report history"),
        ):
            stats = self.reporter.get_lightweight_job_statistics()

        running_job = stats["running_jobs"][0]
        self.assertEqual(running_job["evaluation_id"], "1")
        self.assertEqual(running_job["agent_name"], "Ada")
        self.assertEqual(running_job["execution_source"], "external_surveyor")
        self.assertFalse(running_job["drainable"])

        queued_job = stats["queued_jobs"][0]
        self.assertEqual(queued_job["evaluation_id"], "2")
        self.assertEqual(queued_job["agent_name"], "Ada")
        self.assertTrue(queued_job["drainable"])
        self.assertNotIn(constants.EXTERNAL_REPORT_PROMPT_KEY, queued_job)
        self.assertTrue(self.reporter.has_failed_requeued())
        self.assertTrue(self.reporter.has_drainable_pending_or_running())

    def test_auto_start_does_not_wait_forever_on_failed_requeued_report(self):
        orchestrator = object.__new__(Orchestrator)
        orchestrator.station = SimpleNamespace(
            has_failed_requeued_external_reports=Mock(return_value=True),
            has_pending_external_reports=Mock(return_value=True),
        )
        orchestrator.is_running = False
        orchestrator._push_log_event = Mock()
        orchestrator.start_processing_loop = Mock()

        with (
            patch.object(constants, "AUTO_EVAL_RESEARCH", False),
            patch.object(constants, "RESEARCH_CENTER_ENABLED", False),
            patch.object(constants, "AUTO_EVAL_EXTERNAL_REPORT", True),
            patch.object(constants, "EXTERNAL_COUNTER_ENABLED", True),
            patch.object(constants, "ARCHIVE_SURVEY_ENABLED", False),
        ):
            orchestrator._auto_start_with_wait()

        orchestrator.station.has_pending_external_reports.assert_not_called()
        orchestrator.start_processing_loop.assert_not_called()

    def test_multistart_quiescence_ignores_failed_requeued_external_report(self):
        station = SimpleNamespace(
            has_pending_research_evaluations=lambda: False,
            has_pending_coder_sessions=lambda: False,
            has_pending_external_reports=lambda: True,
            has_drainable_external_reports=lambda: False,
            has_pending_archive_surveys=lambda: False,
        )

        self.assertTrue(_quiescent(station))

    def test_dashboard_drainable_counts_exclude_failed_external_retries(self):
        stats = {
            "running_jobs": [
                {"job_type": "research"},
                {"job_type": "external_report", "drainable": False},
            ],
            "queued_jobs": [
                {"job_type": "external_report", "drainable": False},
                {"job_type": "archive_survey"},
            ],
        }

        dashboard_statistics._update_drainable_job_counts(stats)

        self.assertEqual(stats["drainable_running_jobs_count"], 1)
        self.assertEqual(stats["drainable_queued_jobs_count"], 1)

    def test_retry_exhaustion_requeues_and_pauses_without_notifying(self):
        report = self._report()
        self._save_report(report)
        orchestrator = SimpleNamespace(pause_due_to_research_issue=Mock())
        self.station.orchestrator = orchestrator
        self.reporter.max_retries = 5
        failures = [_TransientStatusError("gateway timeout") for _ in range(6)]

        with (
            patch.object(self.reporter, "_build_openai_client", return_value=object()),
            patch.object(self.reporter, "_request_report", side_effect=failures) as request_report,
            patch("station.eval_external.auto_external_reporter.time.sleep") as sleep,
            patch("station.eval_external.auto_external_reporter.time.time", return_value=1000.0),
        ):
            self.reporter._run_report(report)

        self.assertEqual(request_report.call_count, 6)
        self.assertEqual(sleep.call_count, 5)
        persisted = self.reporter._load_report("1")
        self.assertEqual(persisted[constants.EXTERNAL_REPORT_STATUS_KEY], constants.EXTERNAL_REPORT_STATUS_PENDING)
        self.assertEqual(persisted[constants.EXTERNAL_REPORT_REQUEUE_COUNT_KEY], 1)
        self.assertEqual(persisted[constants.EXTERNAL_REPORT_NEXT_RETRY_AT_KEY], 1120.0)
        self.assertIn("gateway timeout", persisted[constants.EXTERNAL_REPORT_ERROR_KEY])
        queued = pending_queue.load(self.reporter.pending_reports_file)
        self.assertEqual([entry[constants.EXTERNAL_REPORT_ID_KEY] for entry in queued], ["1"])
        orchestrator.pause_due_to_research_issue.assert_called_once()
        self.assertIn("failed and is being requeued", orchestrator.pause_due_to_research_issue.call_args.args[0])
        self.assertEqual(self.agent_module.notifications, [])

    def test_requeued_report_waits_until_next_retry_time(self):
        report = self._report()
        report[constants.EXTERNAL_REPORT_NEXT_RETRY_AT_KEY] = 1120.0
        self._save_report(report)
        self.assertTrue(pending_queue.append(
            self.reporter.pending_reports_file,
            report,
            constants.EXTERNAL_REPORT_ID_KEY,
        ))

        with patch("station.eval_external.auto_external_reporter.time.time", return_value=1119.0):
            self.assertEqual(self.reporter._get_queue_items(), [])
        with patch("station.eval_external.auto_external_reporter.time.time", return_value=1120.0):
            self.assertEqual(
                [entry[constants.EXTERNAL_REPORT_ID_KEY] for entry in self.reporter._get_queue_items()],
                ["1"],
            )

    def test_every_failure_retries_then_requeues_and_pauses_without_notifying(self):
        report = self._report()
        self._save_report(report)
        orchestrator = SimpleNamespace(pause_due_to_research_issue=Mock())
        self.station.orchestrator = orchestrator
        self.reporter.max_retries = 5

        with (
            patch.object(self.reporter, "_build_openai_client", return_value=object()),
            patch.object(
                self.reporter,
                "_request_report",
                side_effect=[ValueError("invalid request") for _ in range(6)],
            ) as request_report,
            patch("station.eval_external.auto_external_reporter.time.sleep") as sleep,
            patch("station.eval_external.auto_external_reporter.time.time", return_value=1000.0),
        ):
            self.reporter._run_report(report)

        self.assertEqual(request_report.call_count, 6)
        self.assertEqual(sleep.call_count, 5)
        persisted = self.reporter._load_report("1")
        self.assertEqual(persisted[constants.EXTERNAL_REPORT_STATUS_KEY], constants.EXTERNAL_REPORT_STATUS_PENDING)
        self.assertEqual(persisted[constants.EXTERNAL_REPORT_REQUEUE_COUNT_KEY], 1)
        self.assertEqual(persisted[constants.EXTERNAL_REPORT_NEXT_RETRY_AT_KEY], 1120.0)
        queued = pending_queue.load(self.reporter.pending_reports_file)
        self.assertEqual([entry[constants.EXTERNAL_REPORT_ID_KEY] for entry in queued], ["1"])
        orchestrator.pause_due_to_research_issue.assert_called_once()
        self.assertIn("failed and is being requeued", orchestrator.pause_due_to_research_issue.call_args.args[0])
        self.assertEqual(self.agent_module.notifications, [])

    def test_requeue_persistence_failure_still_pauses_without_notifying(self):
        report = self._report()
        self._save_report(report)
        orchestrator = SimpleNamespace(pause_due_to_research_issue=Mock())
        self.station.orchestrator = orchestrator
        self.reporter.max_retries = 0

        with (
            patch.object(self.reporter, "_build_openai_client", return_value=object()),
            patch.object(self.reporter, "_request_report", side_effect=ValueError("invalid request")),
            patch.object(self.reporter, "_requeue_report_after_failure", return_value=False),
        ):
            self.reporter._run_report(report)

        persisted = self.reporter._load_report("1")
        self.assertNotEqual(persisted[constants.EXTERNAL_REPORT_STATUS_KEY], constants.EXTERNAL_REPORT_STATUS_FAILED)
        orchestrator.pause_due_to_research_issue.assert_called_once()
        self.assertEqual(self.agent_module.notifications, [])

    def test_openai_environment_is_fallback_for_external_counter(self):
        external_env_values = {
            "EXTERNAL_OPENAI_API_KEY": None,
            "EXTERNAL_OPENAI_BASE_URL": None,
            "EXTERNAL_HTTP_PROXY": None,
            "EXTERNAL_HTTPS_PROXY": None,
        }
        snapshot = {
            "env": {
                "OPENAI_API_KEY": "standard-key",
                "OPENAI_BASE_URL": "https://provider.example.test/v1",
                "OPENAI_HTTP_PROXY": None,
                "OPENAI_HTTPS_PROXY": None,
            },
            "provider_endpoint": {"index": 0, "name": "base", "configured": True},
        }
        fake_http_client = object()
        with (
            patch("station.eval_external.auto_external_reporter.runtime_api_config.get_env_values", return_value=external_env_values),
            patch("station.eval_external.auto_external_reporter.runtime_api_config.get_config_snapshot", return_value=snapshot) as get_snapshot,
            patch("station.eval_external.auto_external_reporter.httpx.Client", return_value=fake_http_client) as http_client,
            patch("station.eval_external.auto_external_reporter.OpenAI") as openai_client,
        ):
            self.reporter._build_openai_client()

        get_snapshot.assert_called_once_with(self.reporter._OPENAI_ENV_NAMES, provider_id="openai")
        http_client.assert_called_once_with(
            proxy=None,
            timeout=constants.EXTERNAL_REPORT_TIMEOUT_SECONDS,
            trust_env=False,
        )
        kwargs = openai_client.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "standard-key")
        self.assertEqual(kwargs["base_url"], "https://provider.example.test/v1")
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertIs(kwargs["http_client"], fake_http_client)

    def test_external_override_does_not_inherit_openai_proxy(self):
        env_values = {
            "EXTERNAL_OPENAI_API_KEY": None,
            "EXTERNAL_OPENAI_BASE_URL": "https://external.example.test/v1",
            "EXTERNAL_HTTP_PROXY": None,
            "EXTERNAL_HTTPS_PROXY": None,
            "OPENAI_API_KEY": "standard-key",
            "OPENAI_BASE_URL": "https://openai.example.test/v1",
            "OPENAI_HTTP_PROXY": "http://openai-proxy.example.test",
            "OPENAI_HTTPS_PROXY": "http://openai-proxy.example.test",
        }
        fake_http_client = object()
        with (
            patch("station.eval_external.auto_external_reporter.runtime_api_config.get_env_values", return_value=env_values),
            patch("station.eval_external.auto_external_reporter.httpx.Client", return_value=fake_http_client) as http_client,
            patch("station.eval_external.auto_external_reporter.OpenAI") as openai_client,
        ):
            self.reporter._build_openai_client()

        http_client.assert_called_once_with(
            proxy=None,
            timeout=constants.EXTERNAL_REPORT_TIMEOUT_SECONDS,
            trust_env=False,
        )
        self.assertEqual(openai_client.call_args.kwargs["base_url"], "https://external.example.test/v1")
        self.assertIs(openai_client.call_args.kwargs["http_client"], fake_http_client)

    def test_no_external_override_uses_openai_proxy(self):
        external_env_values = {
            "EXTERNAL_OPENAI_API_KEY": None,
            "EXTERNAL_OPENAI_BASE_URL": None,
            "EXTERNAL_HTTP_PROXY": None,
            "EXTERNAL_HTTPS_PROXY": None,
        }
        snapshot = {
            "env": {
                "OPENAI_API_KEY": "standard-key",
                "OPENAI_BASE_URL": "https://openai.example.test/v1",
                "OPENAI_HTTP_PROXY": "http://openai-proxy.example.test",
                "OPENAI_HTTPS_PROXY": "http://openai-proxy.example.test",
            },
            "provider_endpoint": {"index": 0, "name": "base", "configured": True},
        }
        fake_http_client = object()
        with (
            patch("station.eval_external.auto_external_reporter.runtime_api_config.get_env_values", return_value=external_env_values),
            patch("station.eval_external.auto_external_reporter.runtime_api_config.get_config_snapshot", return_value=snapshot),
            patch("station.eval_external.auto_external_reporter.httpx.Client", return_value=fake_http_client) as http_client,
            patch("station.eval_external.auto_external_reporter.OpenAI") as openai_client,
        ):
            self.reporter._build_openai_client()

        http_client.assert_called_once_with(
            proxy="http://openai-proxy.example.test",
            timeout=constants.EXTERNAL_REPORT_TIMEOUT_SECONDS,
            trust_env=False,
        )
        self.assertIs(openai_client.call_args.kwargs["http_client"], fake_http_client)

    def test_openai_backup_rotates_for_deterministic_errors_after_retrying_base_once(self):
        report = self._report()
        self._save_report(report)
        base_snapshot = {
            "env": {
                "OPENAI_API_KEY": "base-key",
                "OPENAI_BASE_URL": "https://base.example.test/v1",
                "OPENAI_HTTP_PROXY": None,
                "OPENAI_HTTPS_PROXY": None,
            },
            "external_override": False,
            "provider_endpoint": {"index": 0, "name": "base", "configured": True},
        }
        backup_snapshot = {
            "env": {
                "OPENAI_API_KEY": "backup-key",
                "OPENAI_BASE_URL": "https://backup.example.test/v1",
                "OPENAI_HTTP_PROXY": None,
                "OPENAI_HTTPS_PROXY": None,
            },
            "provider_endpoint": {"index": 1, "name": "backup_1", "configured": True},
            "provider_endpoint_cycle_wrapped": False,
        }

        with (
            patch.object(self.reporter, "_get_openai_runtime_snapshot", return_value=base_snapshot),
            patch.object(self.reporter, "_build_openai_client", side_effect=lambda snapshot: snapshot) as build_client,
            patch.object(
                self.reporter,
                "_request_report",
                side_effect=[ValueError("first"), ValueError("second"), "completed report"],
            ),
            patch.object(
                runtime_api_config,
                "advance_provider_fallback_after_failure",
                side_effect=[
                    {
                        "handled": True,
                        "failure_streak": 1,
                        "retry_same_endpoint": True,
                        "retry_snapshot": None,
                        "cycle_wrapped": False,
                    },
                    {
                        "handled": True,
                        "failure_streak": 0,
                        "retry_same_endpoint": False,
                        "retry_snapshot": backup_snapshot,
                        "cycle_wrapped": False,
                    },
                ],
            ) as advance_fallback,
            patch.object(runtime_api_config, "record_provider_success") as record_success,
            patch("station.eval_external.auto_external_reporter.time.sleep") as sleep,
        ):
            self.reporter._run_report(report)

        self.assertEqual(
            [call.args[0]["provider_endpoint"]["index"] for call in build_client.call_args_list],
            [0, 0, 1],
        )
        self.assertEqual(advance_fallback.call_count, 2)
        self.assertEqual(advance_fallback.call_args_list[0].args, ("openai", 0, 0, self.reporter._OPENAI_ENV_NAMES))
        self.assertEqual(advance_fallback.call_args_list[1].args, ("openai", 0, 1, self.reporter._OPENAI_ENV_NAMES))
        record_success.assert_called_once_with("openai", 1)
        sleep.assert_not_called()
        persisted = self.reporter._load_report("1")
        self.assertEqual(persisted[constants.EXTERNAL_REPORT_STATUS_KEY], constants.EXTERNAL_REPORT_STATUS_COMPLETED)
        self.assertNotIn(constants.EXTERNAL_REPORT_ERROR_KEY, persisted)
        self.assertNotIn(constants.EXTERNAL_REPORT_REQUEUE_COUNT_KEY, persisted)
        self.assertEqual(
            self.agent_module.notifications,
            [("Ada", 'Your external report request "Test report" (ID: 1) has completed.\n\n'
              "**External Report:**\ncompleted report")],
        )

    def test_pending_queue_serializes_concurrent_appends(self):
        queue_path = os.path.join(self.temp_dir.name, "pending.yamll")

        def append_entry(report_id):
            return pending_queue.append(queue_path, {"id": str(report_id)}, "id")

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(append_entry, range(20)))

        self.assertTrue(all(results))
        self.assertFalse(append_entry(3))
        entries = pending_queue.load(queue_path)
        self.assertEqual(len(entries), 20)
        self.assertEqual({entry["id"] for entry in entries}, {str(i) for i in range(20)})


class ExternalAccessTests(unittest.TestCase):
    def test_external_counter_allows_tenured_agents_and_supervisors(self):
        station = object.__new__(Station)
        recursive = {
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_TICK_BIRTH_KEY: 0,
        }
        supervisor = {
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_TICK_BIRTH_KEY: 99,
            constants.AGENT_ROLE_KEY: constants.ROLE_SUPERVISOR,
        }
        guest = {
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_GUEST,
            constants.AGENT_TICK_BIRTH_KEY: 0,
        }
        with (
            patch.object(constants, "AGENT_ISOLATION_TICKS", 40),
            patch.object(constants, "MIN_AGENT_AGE_BEFORE_LEAVE", 100),
        ):
            self.assertFalse(station._is_agent_external_counter_allowed(recursive, 99))
            self.assertTrue(station._is_agent_external_counter_allowed(recursive, 100))
            self.assertTrue(station._is_agent_external_counter_allowed(supervisor, 99))
            self.assertFalse(station._is_agent_external_counter_allowed(guest, 100))

    def test_tenure_message_mentions_external_counter_only_when_enabled(self):
        with (
            patch.object(constants, "MIN_ARCHIVE_BEFORE_LEAVE", 0),
            patch.object(constants, "EXTERNAL_COUNTER_ENABLED", True),
        ):
            enabled = build_tenured_reached_message(constants)
        with (
            patch.object(constants, "MIN_ARCHIVE_BEFORE_LEAVE", 0),
            patch.object(constants, "EXTERNAL_COUNTER_ENABLED", False),
        ):
            disabled = build_tenured_reached_message(constants)

        self.assertIn("access the **External Counter**", enabled)
        self.assertIn("Research Center coders for your submissions can now access external websites", enabled)
        self.assertNotIn("External Counter", disabled)

    def test_supervisor_prompt_mentions_external_counter_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            agents_dir = os.path.join(temp_dir, constants.AGENTS_DIR_NAME)
            file_io_utils.ensure_dir_exists(agents_dir)
            file_io_utils.save_yaml(
                {
                    constants.AGENT_NAME_KEY: "Supervisor",
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_ROLE_KEY: constants.ROLE_SUPERVISOR,
                },
                os.path.join(agents_dir, f"Supervisor{constants.YAML_EXTENSION}"),
            )
            with (
                patch.object(constants, "BASE_STATION_DATA_PATH", temp_dir),
                patch.object(constants, "EXTERNAL_COUNTER_ENABLED", True),
            ):
                enabled = build_station_level_system_prompt("Supervisor", "ignored")
            with (
                patch.object(constants, "BASE_STATION_DATA_PATH", temp_dir),
                patch.object(constants, "EXTERNAL_COUNTER_ENABLED", False),
            ):
                disabled = build_station_level_system_prompt("Supervisor", "ignored")

        self.assertIn("You may access it at any time", enabled)
        self.assertIn("promising and novel relative to external literature", enabled)
        self.assertIn("refer tenured agents to relevant External Reports", enabled)
        self.assertIn("cite a relevant External Report from the External Counter", enabled)
        self.assertNotIn("You may access it at any time", disabled)
        self.assertNotIn("cite a relevant External Report from the External Counter", disabled)
        self.assertLess(enabled.index("* **Question Room**"), enabled.index("* **External Counter**"))
        self.assertLess(enabled.index("* **External Counter**"), enabled.index("* **Lifecycle**"))
        self.assertLess(enabled.index("### 3. Strategic Awareness"), enabled.index("promising and novel relative to external literature"))
        self.assertLess(enabled.index("promising and novel relative to external literature"), enabled.index("## Structure of Each Regular Meeting"))
        self.assertLess(enabled.index("### Research Proposal Requirement"), enabled.index("cite a relevant External Report from the External Counter"))

    def test_help_uses_external_surveyor_and_explains_boundaries(self):
        self.assertIn("External Surveyor", _EXTERNAL_COUNTER_HELP)
        self.assertIn("You are encouraged to do a `{preview all}`", _EXTERNAL_COUNTER_HELP)
        self.assertNotIn("special agent called Surveyor", _EXTERNAL_COUNTER_HELP)
        self.assertIn("not citable as External #IDs", _EXTERNAL_COUNTER_HELP)
        self.assertIn("cannot access Research Center storage", _EXTERNAL_COUNTER_HELP)
        self.assertNotIn("not considered a verified source", _EXTERNAL_COUNTER_HELP)


class CoderInternetAccessTests(unittest.TestCase):
    def test_tenured_submission_snapshots_internet_access(self):
        agent = {
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_TICK_BIRTH_KEY: 0,
        }
        with (
            patch.object(constants, "AGENT_ISOLATION_TICKS", 40),
            patch.object(constants, "MIN_AGENT_AGE_BEFORE_LEAVE", 100),
            patch.object(constants, "EXTERNAL_COUNTER_ENABLED", True),
        ):
            mature = build_coder_access_metadata(agent, 99, constants)
            tenured = build_coder_access_metadata(agent, 100, constants)

        self.assertFalse(mature["internet_access_allowed"])
        self.assertTrue(tenured["is_tenured"])
        self.assertTrue(tenured["internet_access_allowed"])

    def test_external_counter_must_be_enabled_at_submission(self):
        agent = {
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_TICK_BIRTH_KEY: 0,
        }
        with (
            patch.object(constants, "AGENT_ISOLATION_TICKS", 40),
            patch.object(constants, "MIN_AGENT_AGE_BEFORE_LEAVE", 100),
            patch.object(constants, "EXTERNAL_COUNTER_ENABLED", False),
        ):
            access = build_coder_access_metadata(agent, 100, constants)
        self.assertEqual(
            access,
            {
                "phase": "mature",
                "submitted_tick": 100,
                "agent_birth_tick": 0,
                "agent_age_ticks": 100,
                "maturity_threshold_ticks": 40,
                "reason": "age_at_or_above_isolation_threshold",
            },
        )

    def test_guest_does_not_gain_internet_access_from_age(self):
        guest = {
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_GUEST,
            constants.AGENT_TICK_BIRTH_KEY: 0,
        }
        with (
            patch.object(constants, "MIN_AGENT_AGE_BEFORE_LEAVE", 100),
            patch.object(constants, "EXTERNAL_COUNTER_ENABLED", True),
        ):
            access = build_coder_access_metadata(guest, 100, constants)

        self.assertFalse(access["is_tenured"])
        self.assertFalse(access["internet_access_allowed"])

    def test_internet_policy_requires_explicit_agent_request(self):
        enabled_eval = {"coder_access": {"internet_access_allowed": True}}
        disabled_eval = {"coder_access": {"internet_access_allowed": False}}

        enabled = render_coder_internet_policy(enabled_eval)
        disabled = render_coder_internet_policy(disabled_eval)

        self.assertTrue(is_coder_internet_allowed(enabled_eval))
        self.assertIn("only when the agent's instruction explicitly requests", enabled)
        self.assertIn("do not access external websites", enabled)
        self.assertNotIn("Save requested external artifacts", enabled)
        self.assertNotIn("Comprehensive literature surveys", enabled)
        self.assertEqual(disabled, "")
        with patch.object(constants, "EXTERNAL_COUNTER_ENABLED", False):
            self.assertFalse(is_coder_internet_allowed(enabled_eval, constants))
            self.assertEqual(render_coder_internet_policy(enabled_eval, constants), "")

    def test_codex_launch_enables_network_only_when_allowed(self):
        backend = CodexCliWorkerBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "config.toml"), "w", encoding="utf-8") as handle:
                handle.write(
                    'openai_base_url = "https://top-level.example/v1"\n'
                    '[model_providers.Custom]\n'
                    'base_url = "https://custom-provider.example/v1"\n'
                    '[model_providers.Secondary]\n'
                    'base_url = "secondary.example/api"\n'
                    '[model_providers.Invalid]\n'
                    'base_url = "://missing-host"\n'
                )
            runtime_env = {
                "CODEX_HOME": temp_dir,
                "CODEX_API_KEY": "runtime-key",
                "CODEX_BASE_URL": "https://runtime.example/v1",
                "OPENAI_BASE_URL": "https://effective.example/v1",
            }
            online = backend.prepare_launch(
                executable="codex",
                run_dir=temp_dir,
                model_name=None,
                storage_root=os.path.join(temp_dir, "storage"),
                workspace_root=temp_dir,
                prompt="prompt",
                network_access=True,
                runtime_env=runtime_env,
            )
            offline = backend.prepare_launch(
                executable="codex",
                run_dir=temp_dir,
                model_name=None,
                storage_root=os.path.join(temp_dir, "storage"),
                workspace_root=temp_dir,
                prompt="prompt",
                network_access=False,
                runtime_env=runtime_env,
            )

        self.assertIn("--search", online.command)
        self.assertIn("sandbox_workspace_write.network_access=true", online.command)
        self.assertNotIn("danger-full-access", online.command)
        self.assertNotIn('web_search="disabled"', online.command)
        self.assertNotIn("features.network_proxy.enabled=true", online.command)
        self.assertNotIn("--search", offline.command)
        self.assertIn('web_search="disabled"', offline.command)
        self.assertIn("sandbox_workspace_write.network_access=true", offline.command)
        self.assertIn("features.network_proxy.enabled=true", offline.command)
        domain_config = next(
            value for value in offline.command
            if value.startswith("features.network_proxy.domains=")
        )
        for domain in (
            "api.openai.com",
            "custom-provider.example",
            "effective.example",
            "runtime.example",
            "secondary.example",
            "top-level.example",
        ):
            self.assertIn(f'"{domain}" = "allow"', domain_config)
        self.assertNotIn("missing-host", domain_config)

    def test_codex_provider_domains_include_all_configured_providers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "config.toml"), "w", encoding="utf-8") as handle:
                handle.write(
                    '[model_providers.OpenAI]\n'
                    'base_url = "https://api.openai.com/v1"\n'
                    '[model_providers.One]\n'
                    'base_url = "https://one.example/v1"\n'
                    '[model_providers.Two]\n'
                    'base_url = "two.example:8443/v1"\n'
                )

            domains = get_codex_provider_domains({"CODEX_HOME": temp_dir})

        self.assertEqual(domains, ["api.openai.com", "one.example", "two.example"])

    def test_codex_resume_preserves_network_permission(self):
        backend = CodexCliWorkerBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            online = backend.prepare_resume_launch(
                executable="codex",
                run_dir=temp_dir,
                model_name=None,
                storage_root=os.path.join(temp_dir, "storage"),
                workspace_root=temp_dir,
                resume_token="thread-id",
                prompt="resume",
                network_access=True,
            )
            offline = backend.prepare_resume_launch(
                executable="codex",
                run_dir=temp_dir,
                model_name=None,
                storage_root=os.path.join(temp_dir, "storage"),
                workspace_root=temp_dir,
                resume_token="thread-id",
                prompt="resume",
                network_access=False,
                runtime_env={"CODEX_HOME": temp_dir},
            )

        self.assertIn("--search", online.command)
        self.assertIn("sandbox_workspace_write.network_access=true", online.command)
        self.assertIn('web_search="disabled"', offline.command)
        self.assertIn("features.network_proxy.enabled=true", offline.command)

    def test_offline_and_online_coder_prompts_are_internally_consistent(self):
        class DummyStation:
            station_id = "coder-internet-prompt-test"

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(constants, "BASE_STATION_DATA_PATH", os.path.join(temp_dir, "station_data")),
            patch.object(constants, "RESEARCH_STORAGE_BASE_PATH", None),
        ):
            paths = ensure_runtime_layout(constants)
            manager = EvaluationManager(paths.evaluations_dir)
            coder = ResearchCoderManager(DummyStation(), manager, paths=paths)
            base_eval = {
                "id": "501",
                "lineage": "axiom",
                "instruction": "Run the requested experiment.",
                "coder": {"max_attempts": 5},
            }
            offline = coder._build_prompt(base_eval)
            with patch.object(constants, "EXTERNAL_COUNTER_ENABLED", True):
                online = coder._build_prompt({
                    **base_eval,
                    "coder_access": {
                        "phase": "mature",
                        "internet_access_allowed": True,
                    },
                })

        self.assertIn("- No internet access by default", offline)
        self.assertIn("external web/network access", offline)
        self.assertNotIn("- Internet access: enabled", offline)
        self.assertNotIn("This coder session has internet access", offline)

        self.assertIn("- Internet access: enabled", online)
        self.assertIn("This coder session has internet access", online)
        self.assertIn("only when the agent's instruction explicitly requests", online)
        self.assertNotIn("No internet access", online)
        self.assertNotIn("external web/network access", online)


if __name__ == "__main__":
    unittest.main()
