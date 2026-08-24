import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from station import constants
from station.station import Station
from station.workers.cli import check_cli_worker_transcript_growth_timeout
from station.eval_research.evaluation_helpers import (
    PERSISTED_STDOUT_TRUNCATION_MARKER,
    append_stdout_with_limit,
    find_conda_python,
    resolve_conda_env,
    setup_conda_env,
    truncate_persisted_stdout,
)
from station.eval_research.runtime_paths import strip_task_spec_coder_only_sections
from web_interface.task_spec_utils import (
    MAX_TASK_SPEC_BYTES,
    TaskSpecConflictError,
    get_task_spec_snapshot,
    save_task_spec_snapshot,
)


class ResearchRuntimeHelpersTests(unittest.TestCase):
    def test_find_conda_python_prefers_active_prefix_without_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix = Path(tmp_dir) / "envs" / "station"
            python_path = prefix / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.write_text("#!/bin/sh\n", encoding="utf-8")
            os.chmod(python_path, 0o755)

            env = {
                "CONDA_DEFAULT_ENV": "station",
                "CONDA_PREFIX": str(prefix),
            }

            with mock.patch("station.eval_research.evaluation_helpers.subprocess.run") as run_mock:
                resolved = find_conda_python("station", env)

            self.assertEqual(resolved, str(python_path))
            self.assertEqual(env["PATH"].split(os.pathsep)[0], str(python_path.parent))
            self.assertEqual(env["CONDA_PREFIX"], str(prefix))
            self.assertEqual(env["CONDA_DEFAULT_ENV"], "station")
            run_mock.assert_not_called()

    def test_resolve_conda_env_uses_same_path_normalization_for_wrappers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix = Path(tmp_dir) / "envs" / "station"
            python_path = prefix / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.write_text("#!/bin/sh\n", encoding="utf-8")
            os.chmod(python_path, 0o755)

            env = {
                "STATION_PYTHON_EXECUTABLE": str(python_path),
                "PATH": os.pathsep.join(["/usr/bin", str(python_path.parent)]),
            }

            resolved = resolve_conda_env("station", env)

            self.assertEqual(resolved, str(python_path))
            self.assertEqual(env["PATH"].split(os.pathsep), [str(python_path.parent), "/usr/bin"])
            self.assertEqual(env["CONDA_PREFIX"], str(prefix))

            env = {
                "STATION_PYTHON_EXECUTABLE": str(python_path),
                "PATH": "/usr/bin",
            }

            self.assertTrue(setup_conda_env("station", env))
            self.assertEqual(env["PATH"].split(os.pathsep)[0], str(python_path.parent))

    def test_truncate_persisted_stdout_adds_marker(self):
        with mock.patch("station.eval_research.evaluation_helpers.constants.RESEARCH_EVAL_PERSISTED_STDOUT_MAX_CHARS", 20):
            truncated = truncate_persisted_stdout("abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(truncated, PERSISTED_STDOUT_TRUNCATION_MARKER[:20])

    def test_append_stdout_limit_is_per_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout_path = Path(tmp_dir) / "stdout.log"
            stderr_path = Path(tmp_dir) / "stderr.log"
            with mock.patch("station.eval_research.evaluation_helpers.constants.RESEARCH_EVAL_PERSISTED_STDOUT_MAX_CHARS", 64):
                append_stdout_with_limit(str(stdout_path), "a" * 200)
                append_stdout_with_limit(str(stderr_path), "b" * 200)
            stdout_text = stdout_path.read_text(encoding="utf-8")
            stderr_text = stderr_path.read_text(encoding="utf-8")
            self.assertIn(PERSISTED_STDOUT_TRUNCATION_MARKER, stdout_text)
            self.assertIn(PERSISTED_STDOUT_TRUNCATION_MARKER, stderr_text)
            self.assertNotEqual(stdout_text, stderr_text)

    def test_strip_task_spec_coder_only_sections_removes_hidden_block(self):
        task_spec = (
            "Visible intro\n"
            "__CODER_ONLY_BEGIN__\n"
            "Hidden coder-only instructions\n"
            "__CODER_ONLY_END__\n"
            "Visible tail\n"
        )

        stripped = strip_task_spec_coder_only_sections(task_spec)

        self.assertNotIn("Hidden coder-only instructions", stripped)
        self.assertNotIn("__CODER_ONLY_BEGIN__", stripped)
        self.assertNotIn("__CODER_ONLY_END__", stripped)
        self.assertIn("Visible intro", stripped)
        self.assertIn("Visible tail", stripped)

    def test_cli_worker_transcript_idle_check_is_codex_specific(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            transcript_path = Path(tmp_dir) / "transcript.jsonl"
            transcript_path.write_text('{"type":"thread.started"}\n', encoding="utf-8")
            size = transcript_path.stat().st_size
            with mock.patch.object(constants, "CLI_WORKER_TRANSCRIPT_IDLE_TIMEOUT_SECONDS", 1800):
                claude_result = check_cli_worker_transcript_growth_timeout(
                    backend="claude",
                    transcript_path=str(transcript_path),
                    last_size=size,
                    last_growth_timestamp=100.0,
                    now=2000.0,
                )
                self.assertFalse(claude_result.applies)
                self.assertFalse(claude_result.timed_out)

                codex_result = check_cli_worker_transcript_growth_timeout(
                    backend="codex",
                    transcript_path=str(transcript_path),
                    last_size=size,
                    last_growth_timestamp=100.0,
                    now=2000.0,
                )
                self.assertTrue(codex_result.applies)
                self.assertTrue(codex_result.timed_out)

                transcript_path.write_text('{"type":"thread.started"}\n{"type":"turn.started"}\n', encoding="utf-8")
                growth_result = check_cli_worker_transcript_growth_timeout(
                    backend="codex",
                    transcript_path=str(transcript_path),
                    last_size=size,
                    last_growth_timestamp=100.0,
                    now=2000.0,
                )
                self.assertTrue(growth_result.applies)
                self.assertFalse(growth_result.timed_out)
                self.assertEqual(growth_result.last_growth_timestamp, 2000.0)


class TaskSpecDashboardPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base_path = Path(self.temp_dir.name)
        self.task_path = (
            self.base_path
            / constants.ROOMS_DIR_NAME
            / constants.SHORT_ROOM_NAME_RESEARCH
            / constants.RESEARCH_TASK_SPEC_FILENAME
        )
        self.task_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_path_patch = mock.patch.object(
            constants,
            "BASE_STATION_DATA_PATH",
            str(self.base_path),
        )
        self.base_path_patch.start()
        self.addCleanup(self.base_path_patch.stop)

    def test_snapshot_reads_markdown_and_reports_revision(self):
        content = "# Task\n\nLet $x=1$.\n"
        self.task_path.write_text(content, encoding="utf-8")

        snapshot = get_task_spec_snapshot()

        self.assertEqual(snapshot["raw_markdown"], content)
        self.assertEqual(
            snapshot["revision"],
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            snapshot["relative_path"],
            str(Path(constants.ROOMS_DIR_NAME) / constants.SHORT_ROOM_NAME_RESEARCH / constants.RESEARCH_TASK_SPEC_FILENAME),
        )
        self.assertIsNotNone(snapshot["modified_at_ns"])

    def test_save_normalizes_line_endings_and_trailing_newline(self):
        original = "# Old task\n"
        self.task_path.write_text(original, encoding="utf-8")
        revision = get_task_spec_snapshot()["revision"]

        saved = save_task_spec_snapshot(
            "# New task\r\n\r\n$$x^2$$",
            expected_revision=revision,
        )

        expected = "# New task\n\n$$x^2$$\n"
        self.assertEqual(self.task_path.read_text(encoding="utf-8"), expected)
        self.assertEqual(saved["raw_markdown"], expected)
        self.assertEqual(saved["revision"], hashlib.sha256(expected.encode("utf-8")).hexdigest())

    def test_stale_revision_cannot_overwrite_newer_content(self):
        self.task_path.write_text("# First\n", encoding="utf-8")
        stale_revision = get_task_spec_snapshot()["revision"]
        self.task_path.write_text("# Updated elsewhere\n", encoding="utf-8")

        with self.assertRaises(TaskSpecConflictError):
            save_task_spec_snapshot("# Stale edit", expected_revision=stale_revision)

        self.assertEqual(self.task_path.read_text(encoding="utf-8"), "# Updated elsewhere\n")

    def test_invalid_content_is_rejected_without_changing_file(self):
        original = "# Keep me\n"
        self.task_path.write_text(original, encoding="utf-8")
        revision = get_task_spec_snapshot()["revision"]

        invalid_values = ["   \n", "bad\x00text", "x" * (MAX_TASK_SPEC_BYTES + 1)]
        for invalid in invalid_values:
            with self.subTest(length=len(invalid)):
                with self.assertRaises(ValueError):
                    save_task_spec_snapshot(invalid, expected_revision=revision)
                self.assertEqual(self.task_path.read_text(encoding="utf-8"), original)


class DialogueTimestampPersistenceTests(unittest.TestCase):
    def test_dialogue_logger_adds_timestamp_without_mutating_caller_entry(self):
        station = Station.__new__(Station)
        station._get_agent_dialogue_log_path = lambda _agent_name: "/tmp/dialogue.yamll"
        original_entry = {"tick": 17, "speaker": "Station", "type": "observation"}

        with mock.patch("station.station.time.time", return_value=1234.5), mock.patch(
            "station.station.file_io_utils.append_yaml_line"
        ) as append_mock:
            station._log_dialogue_entry("Axiom I", original_entry)

        persisted_entry, persisted_path = append_mock.call_args.args
        self.assertEqual(persisted_path, "/tmp/dialogue.yamll")
        self.assertEqual(persisted_entry["timestamp"], 1234.5)
        self.assertNotIn("timestamp", original_entry)

    def test_dialogue_logger_preserves_explicit_timestamp(self):
        station = Station.__new__(Station)
        station._get_agent_dialogue_log_path = lambda _agent_name: "/tmp/dialogue.yamll"

        with mock.patch("station.station.time.time", return_value=9999.0), mock.patch(
            "station.station.file_io_utils.append_yaml_line"
        ) as append_mock:
            station._log_dialogue_entry(
                "Axiom I",
                {"tick": 17, "speaker": "Agent", "timestamp": 4321.25},
            )

        persisted_entry = append_mock.call_args.args[0]
        self.assertEqual(persisted_entry["timestamp"], 4321.25)


if __name__ == "__main__":
    unittest.main()
