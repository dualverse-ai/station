import unittest
import tempfile
import io
import sqlite3
from pathlib import Path
from unittest import mock

from station import constants
from station.eval_archive.survey_worker import (
    ActiveSurveySession,
    DEFAULT_ARCHIVE_SURVEY_PROMPT_TEMPLATE,
    build_archive_survey_prompt,
)
from station.workers.job_manager import CliJobManager, CliJobSession
from station import file_io_utils, index_paths
from web_interface.archive_utils import build_archive_detail_payload_from_capsule
from web_interface.archive_survey_service import (
    WEB_ARCHIVE_SURVEY_PROMPT_OVERRIDES,
    WebArchiveSurveyService,
    WebArchiveSurveyStore,
    build_web_archive_survey_templates,
)


class ArchiveDashboardPayloadTests(unittest.TestCase):
    def test_archive_detail_payload_includes_reviewer_and_other_replies(self):
        capsule_data = {
            constants.CAPSULE_ID_KEY: "archive_7",
            constants.CAPSULE_TITLE_KEY: "Paper Title",
            constants.CAPSULE_AUTHOR_NAME_KEY: "Axiom I",
            constants.CAPSULE_CREATED_AT_TICK_KEY: 42,
            constants.CAPSULE_ABSTRACT_KEY: "A concise abstract.",
            constants.CAPSULE_MESSAGES_KEY: [
                {
                    constants.MESSAGE_ID_KEY: "archive_7-1",
                    constants.MESSAGE_AUTHOR_NAME_KEY: "Axiom I",
                    constants.MESSAGE_POSTED_AT_TICK_KEY: 42,
                    constants.MESSAGE_TITLE_KEY: "Paper Title",
                    constants.MESSAGE_CONTENT_KEY: "Manuscript body.",
                    constants.MESSAGE_IS_DELETED_KEY: False,
                },
                {
                    constants.MESSAGE_ID_KEY: "archive_7-2",
                    constants.MESSAGE_AUTHOR_NAME_KEY: "Archive Review System",
                    constants.MESSAGE_POSTED_AT_TICK_KEY: 43,
                    constants.MESSAGE_CONTENT_KEY: (
                        "**Reviewer Evaluation**\n\n"
                        "**Score:** 8/10\n\n"
                        "**Reviewer Comments:**\nUseful result."
                    ),
                    constants.MESSAGE_IS_DELETED_KEY: False,
                },
                {
                    constants.MESSAGE_ID_KEY: "archive_7-3",
                    constants.MESSAGE_AUTHOR_NAME_KEY: "Peer Agent",
                    constants.MESSAGE_POSTED_AT_TICK_KEY: 44,
                    constants.MESSAGE_CONTENT_KEY: "Additional capsule discussion.",
                    constants.MESSAGE_IS_DELETED_KEY: False,
                },
                {
                    constants.MESSAGE_ID_KEY: "archive_7-4",
                    constants.MESSAGE_AUTHOR_NAME_KEY: "Archive Review System",
                    constants.MESSAGE_POSTED_AT_TICK_KEY: 45,
                    constants.MESSAGE_CONTENT_KEY: "Deleted reply.",
                    constants.MESSAGE_IS_DELETED_KEY: True,
                },
            ],
        }

        payload = build_archive_detail_payload_from_capsule(capsule_data)

        self.assertEqual(8.0, payload["reviewer_score"])
        self.assertEqual(2, payload["reply_count"])
        paper_markdown = payload["paper_markdown"]
        self.assertIn("Manuscript body.", paper_markdown)
        self.assertIn("## Capsule Replies", paper_markdown)
        self.assertIn("### Reply 1 by Archive Review System at Tick 43", paper_markdown)
        self.assertIn("**Reviewer Evaluation**", paper_markdown)
        self.assertIn("### Reply 2 by Peer Agent at Tick 44", paper_markdown)
        self.assertIn("Additional capsule discussion.", paper_markdown)
        self.assertNotIn("Deleted reply.", paper_markdown)

    def test_web_survey_store_is_persistent_and_uses_separate_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "station_data"
            store = WebArchiveSurveyStore(str(base_path))

            self.assertNotEqual(
                Path(store.paths.index_db_path),
                Path(index_paths.get_station_index_database_path(str(base_path))),
            )
            self.assertIn("web_interface/archive_surveyor", store.paths.root)
            self.assertTrue(Path(store.paths.question_link).is_symlink())

            record = store.create(
                owner="human",
                prompt="Explain the construction and suggest a new experiment.",
                selected_archive_ids=[7, 3, 7],
                source_tick=42,
                task_spec_snapshot="# Task",
                archive_preview_snapshot="Archive #3 and Archive #7",
                question_preview_snapshot="Question #11",
            )
            self.assertEqual("queued", record["status"])
            self.assertEqual([3, 7], record["selected_archive_ids"])
            self.assertEqual("Question #11", record["question_preview_snapshot"])
            rows = store.list("human")
            self.assertEqual(1, len(rows))
            self.assertEqual(record["id"], rows[0]["id"])
            self.assertFalse(rows[0]["has_report"])

            report_text = "# Human report\n\n$E=mc^2$"
            file_io_utils.save_text(report_text, store._report_path(record["id"]))
            detail = store.get(record["id"], owner="human")
            self.assertEqual(report_text, detail["report_markdown"])
            self.assertTrue(store.list("human")[0]["has_report"])

            reopened_store = WebArchiveSurveyStore(str(base_path))
            self.assertEqual(record["id"], reopened_store.list("human")[0]["id"])
            reopened_store.delete(record["id"], "human")
            self.assertEqual([], reopened_store.list("human"))

    def test_dashboard_survey_prompt_is_expert_facing_and_allows_ideas(self):
        self.assertIn("$requester_description", DEFAULT_ARCHIVE_SURVEY_PROMPT_TEMPLATE.template)
        self.assertIn("$role_boundary", DEFAULT_ARCHIVE_SURVEY_PROMPT_TEMPLATE.template)
        self.assertIn("$requester_prompt", DEFAULT_ARCHIVE_SURVEY_PROMPT_TEMPLATE.template)
        prompt = build_archive_survey_prompt(
            survey_id="W3",
            report_basename="web_3",
            requester_prompt="Propose ways forward.",
            workspace_root="/tmp/web-surveyor",
            task_spec="# Task",
            archive_preview="Archive #1",
            question_room_access=True,
            question_preview="Question #11",
            prompt_overrides={
                **WEB_ARCHIVE_SURVEY_PROMPT_OVERRIDES,
                "surveyor_identity": "You are the Archive Surveyor for dashboard survey request #W3.",
            },
        )

        self.assertIn("external human expert", prompt)
        self.assertIn("reader will usually not open cited papers", prompt)
        self.assertIn("Make the report self-contained", prompt)
        self.assertIn("Question #11", prompt)
        self.assertIn("Question Room discussions", prompt)
        self.assertIn("Question #ID-message", prompt)
        self.assertIn("Define Station-specific or newly introduced terms", prompt)
        self.assertIn("Common field terminology", prompt)
        self.assertIn("You may brainstorm, propose, compare, and critique", prompt)
        self.assertIn("# Archive Survey Report #W3", prompt)
        self.assertNotIn("Human Archive Survey", prompt)
        self.assertIn("reports/web_3.draft.md", prompt)
        self.assertIn("mv reports/web_3.draft.md reports/web_3.md", prompt)
        self.assertNotIn("Do not propose any new idea", prompt)
        template_ids = {item["id"] for item in build_web_archive_survey_templates()}
        self.assertIn("brainstorm", template_ids)

    def test_web_survey_worker_uses_snapshot_and_completes_outside_station_queue(self):
        class FinishedProcess:
            pid = 43210

            @staticmethod
            def poll():
                return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "station_data"
            service = WebArchiveSurveyService(str(base_path))
            self.assertIsInstance(service, CliJobManager)
            source_index = Path(index_paths.get_station_index_database_path(str(base_path)))
            source_index.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(source_index) as connection:
                connection.execute("CREATE TABLE source_marker(value TEXT)")
                connection.execute("INSERT INTO source_marker(value) VALUES('snapshot')")

            record = service.store.create(
                owner="human",
                prompt="Explain the evidence and propose a direction.",
                selected_archive_ids=[1],
                source_tick=9,
                task_spec_snapshot="# Task",
                archive_preview_snapshot="Archive #1",
                question_preview_snapshot="Question #9",
            )
            captured = {}

            def fake_start(session_id, spec, prepared):
                captured.update({"prompt": spec.prompt, "index_db_path": spec.env["STATION_INDEX_DB_PATH"]})
                return CliJobSession(
                    session_id=session_id,
                    run_dir=spec.run_dir,
                    backend=spec.backend,
                    transcript_format="jsonl",
                    process=FinishedProcess(),
                    transcript_handle=io.StringIO(),
                    stderr_handle=io.StringIO(),
                    prompt_path=str(Path(spec.run_dir) / "prompt.txt"),
                    command=["codex"],
                    transcript_path=str(Path(spec.run_dir) / "transcript.jsonl"),
                    stderr_path=str(Path(spec.run_dir) / "stderr.txt"),
                    last_message_path=None,
                )

            with mock.patch("web_interface.archive_survey_service.detect_cli_worker_executable", return_value="/usr/bin/codex"), \
                 mock.patch("web_interface.archive_survey_service.CliJobManager._prepare_cli_job_launch", return_value=object()), \
                 mock.patch("web_interface.archive_survey_service.CliJobManager._start_cli_job_process", side_effect=fake_start):
                service.launch_cli_job(str(record["id"]))

            running = service.store.get(record["id"], owner="human", include_report=False)
            self.assertEqual("running", running["status"])
            self.assertIn("external human expert", captured["prompt"])
            self.assertIn("Archive #1", captured["prompt"])
            self.assertIn("Question #9", captured["prompt"])
            self.assertIn("question_room/", captured["prompt"])
            self.assertIn("selected these Archive papers for special attention", captured["prompt"])
            snapshot_path = Path(captured["index_db_path"])
            self.assertTrue(snapshot_path.is_file())
            self.assertNotEqual(source_index, snapshot_path)
            with sqlite3.connect(snapshot_path) as connection:
                self.assertEqual("snapshot", connection.execute("SELECT value FROM source_marker").fetchone()[0])

            file_io_utils.save_text("# Finished human report", service.store._report_path(record["id"]))
            service.poll_cli_jobs()
            completed = service.store.get(record["id"], owner="human")
            self.assertEqual("completed", completed["status"])
            self.assertEqual("# Finished human report", completed["report_markdown"])
            self.assertFalse(snapshot_path.exists())
            station_survey_queue = base_path / "rooms" / "archive" / "surveyor" / "pending_archive_surveys.yamll"
            self.assertFalse(station_survey_queue.exists())

    def test_web_survey_shutdown_requeues_an_interrupted_request(self):
        class RunningProcess:
            pid = 43210

            def __init__(self):
                self.return_code = None

            def poll(self):
                return self.return_code

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "station_data"
            service = WebArchiveSurveyService(str(base_path))
            record = service.store.create(
                owner="human",
                prompt="Summarize the archive.",
                selected_archive_ids=[],
                source_tick=12,
                task_spec_snapshot="# Task",
                archive_preview_snapshot="Archive #1",
                question_preview_snapshot="Question #1",
            )

            def mark_running(record_local):
                record_local["status"] = "running"
                record_local["session"]["status"] = "running"
                record_local["session"]["active"] = True
                record_local["session"]["active_pid"] = 43210

            service.store.update(record["id"], mark_running)
            process = RunningProcess()
            active = ActiveSurveySession(
                survey_id=str(record["id"]),
                session_id="web_1_codex_spawn_1_test",
                run_dir=str(base_path / "session"),
                backend="codex",
                transcript_format="jsonl",
                process=process,
                transcript_handle=io.StringIO(),
                stderr_handle=io.StringIO(),
                prompt_path=str(base_path / "prompt.txt"),
                command=["codex"],
                transcript_path=str(base_path / "transcript.jsonl"),
                stderr_path=str(base_path / "stderr.txt"),
                last_message_path=None,
                report_path=service.store._report_path(record["id"]),
                draft_path=service.store._draft_path(record["id"]),
            )
            service.active_sessions[str(record["id"])] = active
            service._running = True

            def terminate(_session, force=False):
                process.return_code = -9 if force else -15

            with mock.patch.object(service, "_terminate_cli_job_process", side_effect=terminate):
                service.stop()

            requeued = service.store.get(record["id"], owner="human", include_report=False)
            self.assertEqual("queued", requeued["status"])
            self.assertFalse(requeued["session"]["active"])
            self.assertIsNone(requeued["session"]["active_pid"])
            self.assertIn("safely requeued", requeued["session"]["last_error"])

    def test_web_survey_transient_failure_uses_shared_same_session_resume(self):
        class FailedProcess:
            pid = 43210

            @staticmethod
            def poll():
                return 1

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "station_data"
            service = WebArchiveSurveyService(str(base_path))
            record = service.store.create(
                owner="human",
                prompt="Summarize the archive.",
                selected_archive_ids=[],
                source_tick=12,
                task_spec_snapshot="# Task",
                archive_preview_snapshot="Archive #1",
                question_preview_snapshot="Question #1",
            )

            def mark_running(record_local):
                record_local["status"] = "running"
                session = record_local["session"]
                session.update(
                    {
                        "status": "running",
                        "active": True,
                        "active_pid": 43210,
                        "spawn_count": 1,
                    }
                )

            service.store.update(record["id"], mark_running)
            run_dir = base_path / "session"
            run_dir.mkdir(parents=True)
            transcript_path = run_dir / "transcript.jsonl"
            transcript_path.write_text(
                '{"type":"thread.started","thread_id":"web-thread-1"}\n'
                '{"type":"error","message":"503 Service Unavailable"}\n',
                encoding="utf-8",
            )
            stderr_path = run_dir / "stderr.txt"
            stderr_path.write_text("", encoding="utf-8")
            service.active_sessions[str(record["id"])] = ActiveSurveySession(
                survey_id=str(record["id"]),
                session_id="web_1_codex_spawn_1_test",
                run_dir=str(run_dir),
                backend="codex",
                transcript_format="jsonl",
                process=FailedProcess(),
                transcript_handle=io.StringIO(),
                stderr_handle=io.StringIO(),
                prompt_path=str(run_dir / "prompt.txt"),
                command=["codex"],
                transcript_path=str(transcript_path),
                stderr_path=str(stderr_path),
                last_message_path=None,
                report_path=service.store._report_path(record["id"]),
                draft_path=service.store._draft_path(record["id"]),
            )

            with mock.patch.object(constants, "RESEARCH_CODER_RESUME_BACKOFF_SECONDS", [0]):
                service.poll_cli_jobs()

            pending = service.store.get(record["id"], owner="human", include_report=False)
            self.assertEqual("running", pending["status"])
            self.assertEqual("pending_resume", pending["session"]["status"])
            self.assertEqual("web-thread-1", pending["session"]["resume_token"])
            self.assertEqual(0, pending["session"]["resume_count"])
            self.assertFalse(service.active_sessions)


if __name__ == "__main__":
    unittest.main()
