import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from station import constants
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


if __name__ == "__main__":
    unittest.main()
