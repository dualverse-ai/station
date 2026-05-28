import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from station.eval_research.auto_evaluator import AutoResearchEvaluator
from station.eval_research.restart_evaluations import requeue_instruction_evaluations


class ResearchEvaluationLifecycleTests(unittest.TestCase):
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
