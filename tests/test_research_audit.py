import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from station import constants
from station.eval_research.auto_evaluator import AutoResearchEvaluator
from station.eval_research.coder_manager import ResearchCoderManager
from station.eval_research.evaluation_manager import EvaluationManager
from station.eval_research.restart_evaluations import requeue_instruction_evaluations
from station.eval_research.runtime_paths import ensure_runtime_layout
from station.eval_research.submit_audit_cli import submit_audit


class ResearchAuditCliTests(unittest.TestCase):
    def _manager(self, root):
        consts = SimpleNamespace(**{key: getattr(constants, key) for key in dir(constants) if key.isupper()})
        consts.BASE_STATION_DATA_PATH = str(root / "station_data")
        paths = ensure_runtime_layout(consts)
        manager = EvaluationManager(paths.evaluations_dir)
        manager.create_evaluation("17", "author", "title", "instruction", 1, lineage="lineage")
        manager.register_attempt("17", str(Path(paths.submissions_dir) / "17.py"))
        manager.complete_attempt("17", 1, True, 1, "ATTEMPT_COMPLETE", "", details="ok")
        report_path = Path(paths.reports_dir) / "17.md"
        report_path.write_text("# Coder Report\n\n## Final Status\n`success`\n", encoding="utf-8")
        return ResearchCoderManager(SimpleNamespace(), manager, paths=paths), manager, report_path

    def test_requires_report_and_writes_atomic_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "storage" / "audit"
            audit.mkdir(parents=True)
            self.assertEqual(submit_audit("17", "pass", research_root=str(root)), 1)
            (audit / "17.md").write_text("## Independent Audit\nclear\n", encoding="utf-8")
            self.assertEqual(submit_audit("17", "pass", research_root=str(root)), 0)
            self.assertEqual((audit / "17.verdict").read_text(encoding="utf-8").strip(), "pass")
            self.assertFalse((audit / "17.verdict.tmp").exists())

    def test_original_auditor_prompt_requirements_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            prompt = coder.audit_manager._build_audit_prompt(manager.get_evaluation("17"))
            for requirement in (
                "audit the submitted code, official result, and Coder Report",
                "Do not modify any evaluation files or perform a new official submission",
                "incomplete searches described as exhaustive",
                "timeouts treated as completed work",
                "honestly reported timeout",
                "UNDECIDED-AT-BUDGET",
                "pass with that qualification",
                "mark the audit as `pass`",
                "Mark the audit as `fail` only when there is a material scientific or implementation error",
                "give a detailed and actionable account of the failure",
                "Write your report to `storage/audit/17.md`",
                "bash submit_audit.sh 17 pass",
                "bash submit_audit.sh 17 fail",
            ):
                self.assertIn(requirement, prompt)

    def test_last_audit_prompt_is_agent_facing_and_concise(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            manager.update_evaluation("17", lambda r: r.setdefault("audit", {}).update({
                "repair_round": 1,
                "status": "running",
            }))
            prompt = coder.audit_manager._build_audit_prompt(manager.get_evaluation("17"))
            self.assertIn("This is the final configured audit", prompt)
            self.assertIn("will be delivered directly to the agent", prompt)
            self.assertIn("whether pass or fail", prompt)
            self.assertIn("in both cases, please be concise and use at most two paragraphs", prompt)
            self.assertNotIn("state the material issue, its scientific impact", prompt)
            self.assertNotIn("give a detailed and actionable account of the failure", prompt)

    def test_rejects_invalid_verdict_without_touching_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "storage" / "audit"
            audit.mkdir(parents=True)
            (audit / "2.md").write_text("report", encoding="utf-8")
            (audit / "2.verdict").write_text("pass\n", encoding="utf-8")
            self.assertEqual(submit_audit("2", "maybe", research_root=str(root)), 2)
            self.assertEqual((audit / "2.verdict").read_text(encoding="utf-8"), "pass\n")

    def test_pass_appends_report_and_finalizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, report_path = self._manager(Path(tmp))
            coder._handle_audit_verdict("17", "pass", "Clear; minor qualification: citation is terse.", None)
            self.assertEqual(manager.get_evaluation("17")["status"], "completed")
            self.assertIn("Independent Audit", report_path.read_text(encoding="utf-8"))

    def test_fail_requests_repair_then_exhaustion_is_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, report_path = self._manager(Path(tmp))
            manager.update_evaluation("17", lambda r: (
                r.setdefault("coder", {}).update({"spawn_count": 2}),
                r.setdefault("audit", {}).update({"spawn_count": 2}),
            ))
            coder._handle_audit_verdict("17", "fail", "Material omission: fix the search and rerun.", None)
            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "queued")
            self.assertEqual(record["audit"]["repair_round"], 1)
            self.assertEqual(record["audit"]["spawn_count"], 0)
            self.assertEqual(record["coder"]["spawn_count"], 0)
            final_audit_path = Path(coder.paths.audit_dir) / "17.md"
            final_audit_path.write_text("Still materially incomplete.\n", encoding="utf-8")
            manager.update_evaluation(
                "17",
                lambda r: r.setdefault("audit", {}).update({
                    "last_report_path": "storage/audit/17.md",
                }),
            )
            coder._handle_audit_verdict("17", "fail", "Still materially incomplete.", None)
            self.assertEqual(manager.get_evaluation("17")["status"], "partial")
            self.assertIn("Independent Audit Failure", report_path.read_text(encoding="utf-8"))
            notification = manager.get_notification_message("17")
            self.assertIn("Still materially incomplete.", notification)
            self.assertNotIn("Material omission: fix the search and rerun.", notification)
            self.assertIn("Final Independent Audit", notification)

    def test_failed_audit_grants_one_attempt_when_budget_is_exhausted(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            submission_path = str(Path(coder.paths.submissions_dir) / "17.py")
            for attempt_number in range(2, 6):
                self.assertEqual(manager.register_attempt("17", submission_path), attempt_number)
                manager.complete_attempt(
                    "17",
                    attempt_number,
                    True,
                    1,
                    "ATTEMPT_COMPLETE",
                    "",
                    details="ok",
                )

            coder._handle_audit_verdict("17", "fail", "Material omission: rerun the missing check.", None)
            record = manager.get_evaluation("17")
            self.assertEqual(len(record["attempts"]), 5)
            self.assertEqual(record["coder"]["max_attempts"], 6)

            prompt = coder._build_resume_prompt(record)
            self.assertIn("5 used, 1 remaining out of 6", prompt)
            self.assertIn("Audit rounds do not reset this budget", prompt)

    def test_audit_repair_resumes_completed_codex_thread_with_new_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, report_path = self._manager(Path(tmp))
            run_dir = Path(coder.paths.coder_sessions_dir) / "codex_17_spawn_1_deadbeef"
            run_dir.mkdir(parents=True)
            transcript_path = run_dir / "transcript.jsonl"
            transcript_path.write_text(
                '{"type":"thread.started","thread_id":"thread-17"}\n',
                encoding="utf-8",
            )
            session = SimpleNamespace(
                session_id=run_dir.name,
                report_path=str(report_path),
                backend="codex",
                transcript_path=str(transcript_path),
            )
            with mock.patch.object(coder.audit_manager, "launch_for_evaluation", return_value=True):
                coder._on_cli_job_completed("17", session, 0)
            self.assertEqual(manager.get_evaluation("17")["coder"]["resume_token"], "thread-17")

            audit_report = "Material omission: compute the missing exact check."
            (Path(coder.paths.audit_dir) / "17.md").write_text(audit_report, encoding="utf-8")
            coder._handle_audit_verdict("17", "fail", audit_report, "audit-session")
            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["coder"]["status"], "pending_resume")
            self.assertTrue(record["coder"]["active"])
            self.assertEqual(record["coder"]["resume_token"], "thread-17")
            self.assertEqual(record["coder"]["spawn_count"], 0)
            self.assertEqual(record["coder"]["resume_count"], 0)

            prompt = coder._build_resume_prompt(record)
            self.assertIn("Resume the same Research Center coder session", prompt)
            self.assertIn(audit_report, prompt)
            self.assertIn("do not restart the implementation from scratch", prompt)
            self.assertIn("This is your last coder repair round", prompt)
            self.assertIn("failure will be communicated directly to the submitting agent", prompt)
            self.assertIn("without further coder repair", prompt)
            self.assertIn("fix every material problem raised in the auditor report", prompt)
            self.assertNotIn("A computational timeout is not itself", prompt)

            state = coder._load_cli_job_state("17")
            decision = coder._next_cli_job_launch("17", state)
            self.assertIsNotNone(decision)
            self.assertTrue(decision.is_resume)
            self.assertEqual(decision.resume_token, "thread-17")
            session_id = coder._format_cli_job_session_id("17", state, decision)
            claimed = coder._claim_cli_job_launch("17", session_id, decision)
            spec = coder._build_cli_job_launch_spec("17", session_id, decision, claimed)
            prepared = coder._prepare_cli_job_launch(spec, decision, coder._get_cli_job_backend)
            self.assertEqual(spec.prompt, prompt)
            self.assertEqual(prepared.stdin_text, prompt)
            self.assertIn("resume", prepared.command)
            self.assertIn("thread-17", prepared.command)

    def test_fresh_repair_fallback_keeps_audit_and_previous_session_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            old_session = Path(coder.paths.coder_sessions_dir) / "claude_17_spawn_1_deadbeef"
            old_session.mkdir(parents=True)
            (old_session / "prompt.txt").write_text("original prompt\n", encoding="utf-8")
            audit_report = "Material omission: verify the missing boundary case."
            (Path(coder.paths.audit_dir) / "17.md").write_text(audit_report, encoding="utf-8")
            manager.update_evaluation("17", lambda r: r.setdefault("coder", {}).update({
                "backend": "claude",
                "spawn_count": 2,
                "resume_token": None,
            }))
            coder._handle_audit_verdict("17", "fail", audit_report, "audit-session")

            state = coder._load_cli_job_state("17")
            decision = coder._next_cli_job_launch("17", state)
            self.assertIsNotNone(decision)
            self.assertFalse(decision.is_resume)
            self.assertEqual(decision.spawn_count, 1)
            session_id = coder._format_cli_job_session_id("17", state, decision)
            claimed = coder._claim_cli_job_launch("17", session_id, decision)
            with mock.patch.object(coder, "_detect_backend_executable", return_value="claude"):
                spec = coder._build_cli_job_launch_spec("17", session_id, decision, claimed)
            self.assertIn(audit_report, spec.prompt)
            self.assertIn("Independent audit repair context", spec.prompt)
            self.assertIn(f"coder_sessions/{old_session.name}/", spec.prompt)
            self.assertIn("Inspect useful prior artifacts", spec.prompt)

    def test_audit_waits_for_official_attempt_to_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, report_path = self._manager(Path(tmp))
            manager.update_evaluation("17", lambda r: r["attempts"][-1].update({"status": "running"}))
            session = SimpleNamespace(
                session_id="coder-session",
                report_path=str(report_path),
                backend="codex",
                transcript_path=str(Path(coder.paths.coder_sessions_dir) / "missing.jsonl"),
            )
            with mock.patch.object(coder.audit_manager, "launch_for_evaluation", return_value=True) as launch:
                coder._on_cli_job_completed("17", session, 0)
                self.assertFalse(launch.called)
                self.assertIn("17", coder.pending_report_finalizations)
                self.assertEqual(manager.get_evaluation("17")["coder"]["status"], "attempt_running")

                manager.update_evaluation("17", lambda r: r["attempts"][-1].update({"status": "completed"}))
                coder._finalize_pending_reports_if_ready()

            record = manager.get_evaluation("17")
            self.assertTrue(launch.called)
            self.assertNotIn("17", coder.pending_report_finalizations)
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["coder"]["status"], "audit_running")
            self.assertFalse(record.get("final"))

    def test_audit_launch_removes_stale_round_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            audit_dir = Path(coder.paths.audit_dir)
            report_path = audit_dir / "17.md"
            verdict_path = audit_dir / "17.verdict"
            tmp_verdict_path = audit_dir / "17.verdict.tmp"
            report_path.write_text("previous audit", encoding="utf-8")
            verdict_path.write_text("fail\n", encoding="utf-8")
            tmp_verdict_path.write_text("pass\n", encoding="utf-8")
            prompt = coder.audit_manager._build_audit_prompt(manager.get_evaluation("17"))
            self.assertIn("previous audit", prompt)

            coder.audit_manager._before_cli_job_process_launch("17", None, None, None)
            self.assertFalse(report_path.exists())
            self.assertFalse(verdict_path.exists())
            self.assertFalse(tmp_verdict_path.exists())

    def test_repair_report_must_be_replaced_before_reaudit(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, report_path = self._manager(Path(tmp))
            coder._handle_audit_verdict("17", "fail", "Repair required.", None)
            baseline = manager.get_evaluation("17")["audit"]["repair_report_baseline_mtime_ns"]
            session = SimpleNamespace(report_path=str(report_path))
            self.assertFalse(coder._cli_job_completion_ready("17", session))
            report_path.write_text("# Coder Report\n\nrepair complete\n", encoding="utf-8")
            if report_path.stat().st_mtime_ns <= baseline:
                os.utime(report_path, ns=(baseline + 1, baseline + 1))
            self.assertTrue(coder._cli_job_completion_ready("17", session))

    def test_restart_resets_interrupted_auditor_spawn_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            manager.update_evaluation("17", lambda r: r.update({
                "status": "running",
                "coder": dict(r.get("coder", {}) or {}, active=False, status="audit_running"),
                "audit": dict(
                    r.get("audit", {}) or {},
                    active=True,
                    active_pid=999,
                    status="running",
                    spawn_count=3,
                ),
            }))
            recovered = requeue_instruction_evaluations(
                eval_ids=["17"],
                reason="station restart",
                eval_manager=manager,
                paths=coder.paths,
            )
            self.assertEqual(recovered, 1)
            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["coder"]["status"], "audit_running")
            self.assertEqual(record["audit"]["status"], "retry")
            self.assertEqual(record["audit"]["spawn_count"], 0)

    def test_restart_with_completed_report_requeues_only_initial_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            manager.update_evaluation("17", lambda r: r.update({
                "status": "running",
                "coder": dict(
                    r.get("coder", {}) or {},
                    active=False,
                    active_pid=None,
                    status="attempt_running",
                    spawn_count=3,
                ),
            }))
            recovered = requeue_instruction_evaluations(
                eval_ids=["17"],
                reason="station restart",
                eval_manager=manager,
                paths=coder.paths,
            )
            self.assertEqual(recovered, 1)
            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["coder"]["status"], "audit_running")
            self.assertEqual(record["coder"]["spawn_count"], 0)
            self.assertEqual(record["audit"]["status"], "retry")
            self.assertEqual(record["audit"]["spawn_count"], 0)
            self.assertFalse(record.get("final"))
            coder_state = coder._load_cli_job_state("17")
            audit_state = coder.audit_manager._load_cli_job_state("17")
            self.assertFalse(coder_state.fresh_launch_eligible)
            self.assertTrue(audit_state.fresh_launch_eligible)

    def test_restart_resumes_interrupted_repair_coder_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, report_path = self._manager(Path(tmp))
            baseline = report_path.stat().st_mtime_ns
            manager.update_evaluation("17", lambda r: r.update({
                "status": "running",
                "coder": dict(
                    r.get("coder", {}) or {},
                    active=True,
                    active_pid=999,
                    status="resuming",
                    spawn_count=3,
                    resume_count=4,
                    resume_token="thread-17",
                ),
                "audit": dict(
                    r.get("audit", {}) or {},
                    active=False,
                    status="repairing",
                    repair_round=1,
                    repair_report_baseline_mtime_ns=baseline,
                ),
            }))
            recovered = requeue_instruction_evaluations(
                eval_ids=["17"],
                reason="station restart",
                eval_manager=manager,
                paths=coder.paths,
            )
            self.assertEqual(recovered, 1)
            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["coder"]["status"], "pending_resume")
            self.assertTrue(record["coder"]["active"])
            self.assertEqual(record["coder"]["resume_token"], "thread-17")
            self.assertEqual(record["coder"]["spawn_count"], 0)
            self.assertEqual(record["coder"]["resume_count"], 0)

    def test_restart_does_not_reaudit_replaced_report_from_interrupted_repair_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, report_path = self._manager(Path(tmp))
            coder._handle_audit_verdict("17", "fail", "Repair required.", None)
            baseline = manager.get_evaluation("17")["audit"]["repair_report_baseline_mtime_ns"]
            report_path.write_text("# Repaired report written before attempt settled\n", encoding="utf-8")
            if report_path.stat().st_mtime_ns <= baseline:
                os.utime(report_path, ns=(baseline + 1, baseline + 1))
            manager.register_attempt("17", str(Path(coder.paths.submissions_dir) / "17.py"))
            manager.update_evaluation("17", lambda r: r.update({
                "status": "running",
                "coder": dict(
                    r.get("coder", {}) or {},
                    active=True,
                    active_pid=999,
                    status="attempt_running",
                    resume_token="thread-17",
                ),
            }))

            recovered = requeue_instruction_evaluations(
                eval_ids=["17"],
                reason="station restart",
                eval_manager=manager,
                paths=coder.paths,
            )

            self.assertEqual(recovered, 1)
            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["audit"]["status"], "repairing")
            self.assertEqual(record["coder"]["status"], "pending_resume")
            self.assertEqual(record["attempts"][-1]["status"], "abandoned")
            self.assertIn("station restart", record["attempts"][-1]["error"])
            advanced_baseline = record["audit"]["repair_report_baseline_mtime_ns"]
            self.assertGreater(advanced_baseline, baseline)

            # A second restart before the repair coder relaunches must not treat
            # the pre-interruption report as a completed repair.
            recovered = requeue_instruction_evaluations(
                eval_ids=["17"],
                reason="second station restart",
                eval_manager=manager,
                paths=coder.paths,
            )
            self.assertEqual(recovered, 1)
            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["audit"]["status"], "repairing")
            self.assertEqual(record["coder"]["status"], "pending_resume")
            self.assertEqual(record["audit"]["repair_report_baseline_mtime_ns"], advanced_baseline)

    def test_restart_recovers_legacy_transient_auditor_block_by_resuming_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            session_id = "codex_17_spawn_1_rate429"
            run_dir = Path(coder.paths.coder_sessions_dir) / f"audit_{session_id}"
            run_dir.mkdir(parents=True)
            (run_dir / "transcript.jsonl").write_text(
                '{"type":"thread.started","thread_id":"audit-thread-recovered"}\n'
                '{"type":"turn.failed","error":{"message":"429 Too Many Requests"}}\n',
                encoding="utf-8",
            )
            manager.update_evaluation("17", lambda r: r.update({
                "status": "running",
                "coder": dict(r.get("coder", {}) or {}, active=False, status="audit_running"),
                "audit": dict(
                    r.get("audit", {}) or {},
                    active=False,
                    status="blocked",
                    session_id=session_id,
                    last_error=(
                        "Auditor could not produce a verdict within the current audit round's retry limit: "
                        "Transient auditor backend/provider failure: too many requests."
                    ),
                ),
            }))

            recovered = requeue_instruction_evaluations(
                eval_ids=["17"],
                reason="station restart",
                eval_manager=manager,
                paths=coder.paths,
            )

            self.assertEqual(recovered, 1)
            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["coder"]["status"], "audit_running")
            self.assertEqual(record["audit"]["status"], "pending_resume")
            self.assertEqual(record["audit"]["resume_token"], "audit-thread-recovered")
            self.assertEqual(record["audit"]["max_spawns"], constants.RESEARCH_CODER_MAX_SPAWNS)
            self.assertEqual(record["audit"]["max_resumes"], constants.RESEARCH_CODER_MAX_RESUMES)

    def test_recovery_does_not_skip_an_interrupted_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, report_path = self._manager(Path(tmp))
            coder._handle_audit_verdict("17", "fail", "Material omission: rerun the experiment.", None)
            record = manager.get_evaluation("17")
            baseline = record["audit"]["repair_report_baseline_mtime_ns"]

            evaluator = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
            evaluator.eval_manager = manager
            evaluator.coder_manager = SimpleNamespace(active_sessions={}, audit_manager=SimpleNamespace(active_sessions={}))
            evaluator.active_futures = {}
            evaluator.paths = coder.paths
            evaluator.timeout = 900
            evaluator._has_run_request_for_eval = lambda _eval_id: False
            evaluator._pid_exists = lambda _pid: False

            # The old report is still present, but the repair coder has not
            # replaced it. Recovery must requeue the repair coder.
            manager.update_evaluation("17", lambda r: r.update({
                "status": "running",
                "coder": dict(r.get("coder", {}) or {}, active=True, active_pid=999, status="coder_running"),
            }))
            evaluator._recover_stale_coder_states()
            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "queued")
            self.assertEqual(record["audit"]["status"], "repairing")
            self.assertEqual(record["audit"]["repair_report_baseline_mtime_ns"], baseline)

            # A replaced report is evidence that repair completed before the
            # interruption; recovery may proceed directly to a fresh audit.
            report_path.write_text(report_path.read_text(encoding="utf-8") + "\nrepair complete\n", encoding="utf-8")
            report_path.touch()
            if report_path.stat().st_mtime_ns <= baseline:
                os.utime(report_path, ns=(baseline + 1, baseline + 1))
            manager.update_evaluation("17", lambda r: r.update({
                "status": "running",
                "coder": dict(r.get("coder", {}) or {}, active=True, active_pid=999, status="coder_running"),
            }))
            evaluator._recover_stale_coder_states()
            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["audit"]["status"], "queued")
            self.assertEqual(record["coder"]["status"], "audit_running")

    def test_runtime_recovery_does_not_reaudit_during_interrupted_repair_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, report_path = self._manager(Path(tmp))
            coder._handle_audit_verdict("17", "fail", "Repair required.", None)
            baseline = manager.get_evaluation("17")["audit"]["repair_report_baseline_mtime_ns"]
            report_path.write_text("# Repaired report written before attempt settled\n", encoding="utf-8")
            if report_path.stat().st_mtime_ns <= baseline:
                os.utime(report_path, ns=(baseline + 1, baseline + 1))
            manager.register_attempt("17", str(Path(coder.paths.submissions_dir) / "17.py"))
            manager.update_evaluation("17", lambda r: r.update({
                "status": "running",
                "coder": dict(
                    r.get("coder", {}) or {},
                    active=True,
                    active_pid=999,
                    status="attempt_running",
                    resume_token="thread-17",
                ),
            }))
            evaluator = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
            evaluator.eval_manager = manager
            evaluator.coder_manager = SimpleNamespace(
                active_sessions={},
                audit_manager=SimpleNamespace(active_sessions={}),
            )
            evaluator.active_futures = {}
            evaluator.paths = coder.paths
            evaluator.timeout = 900
            evaluator._has_run_request_for_eval = lambda _eval_id: False
            evaluator._pid_exists = lambda _pid: False

            evaluator._recover_stale_coder_states()

            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["audit"]["status"], "repairing")
            self.assertEqual(record["coder"]["status"], "pending_resume")
            self.assertEqual(record["attempts"][-1]["status"], "abandoned")
            self.assertIn("interrupted repair attempt", record["attempts"][-1]["error"])
            advanced_baseline = record["audit"]["repair_report_baseline_mtime_ns"]
            self.assertGreater(advanced_baseline, baseline)

            evaluator._recover_stale_coder_states()
            record = manager.get_evaluation("17")
            self.assertEqual(record["audit"]["status"], "repairing")
            self.assertEqual(record["coder"]["status"], "pending_resume")
            self.assertEqual(record["audit"]["repair_report_baseline_mtime_ns"], advanced_baseline)

    def test_runtime_recovery_finishes_persisted_auditor_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            audit_report = "The implementation and conclusions are supported."
            (Path(coder.paths.audit_dir) / "17.md").write_text(audit_report, encoding="utf-8")
            (Path(coder.paths.audit_dir) / "17.verdict").write_text("pass\n", encoding="utf-8")
            manager.update_evaluation("17", lambda r: r.update({
                "status": "running",
                "coder": dict(r.get("coder", {}) or {}, active=False, status="audit_running"),
                "audit": dict(r.get("audit", {}) or {}, active=False, status="pass"),
            }))
            evaluator = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
            evaluator.eval_manager = manager
            evaluator.coder_manager = coder
            evaluator.active_futures = {}
            evaluator.paths = coder.paths
            evaluator.timeout = 900
            evaluator._has_run_request_for_eval = lambda _eval_id: False
            evaluator._pid_exists = lambda _pid: False
            evaluator._push_log_event = lambda *_args, **_kwargs: None

            evaluator._recover_stale_coder_states()

            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["audit"]["status"], "pass")
            self.assertTrue(record.get("final"))

    def test_runtime_recovery_preserves_blocked_auditor(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            manager.update_evaluation("17", lambda r: r.update({
                "status": "running",
                "coder": dict(r.get("coder", {}) or {}, active=False, status="audit_running"),
                "audit": dict(
                    r.get("audit", {}) or {},
                    active=False,
                    status="blocked",
                    last_error="Auditor infrastructure retry limit exhausted.",
                ),
            }))
            evaluator = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
            evaluator.eval_manager = manager
            evaluator.coder_manager = coder
            evaluator.active_futures = {}
            evaluator.paths = coder.paths
            evaluator.timeout = 900
            evaluator._has_run_request_for_eval = lambda _eval_id: False
            evaluator._pid_exists = lambda _pid: False
            evaluator._push_log_event = lambda *_args, **_kwargs: None

            evaluator._recover_stale_coder_states()

            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["audit"]["status"], "blocked")
            self.assertFalse(record.get("final"))

    def test_disabled_flag_uses_pre_audit_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, report_path = self._manager(Path(tmp))
            session = SimpleNamespace(session_id="coder-session", report_path=str(report_path))
            with mock.patch.object(constants, "RESEARCH_CODER_AUDIT_ENABLED", False):
                coder._on_cli_job_completed("17", session, 0)
            record = manager.get_evaluation("17")
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["audit"]["status"], "not_started")

    def test_audit_idle_timeout_hooks_update_audit_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            session = SimpleNamespace(session_id="audit-session")
            reason = coder.audit_manager._cli_job_idle_timeout_reason("17", session, 61, 60)
            self.assertIn("independent audit", reason)
            coder.audit_manager._on_cli_job_idle_timeout("17", session, reason, 61, 60)
            record = manager.get_evaluation("17")
            self.assertEqual(record["audit"]["last_error"], reason)

    def test_transient_auditor_failure_uses_coder_resume_budget_and_backoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            session = SimpleNamespace(session_id="audit-session")
            failure = SimpleNamespace(
                category="infra_transient",
                resume_token="audit-thread-17",
                resume_count=0,
                delay_seconds=10,
                next_resume_timestamp=1234.0,
                reason="Transient auditor backend/provider failure: too many requests.",
                returncode=1,
            )
            coder.audit_manager._schedule_cli_job_resume("17", session, failure)
            record = manager.get_evaluation("17")
            audit = record["audit"]
            self.assertEqual(audit["status"], "pending_resume")
            self.assertEqual(audit["resume_token"], "audit-thread-17")
            self.assertEqual(audit["next_resume_timestamp"], 1234.0)

            manager.update_evaluation(
                "17",
                lambda r: r.setdefault("audit", {}).update({"next_resume_timestamp": None}),
            )
            state = coder.audit_manager._load_cli_job_state("17")
            self.assertEqual(state.max_spawns, constants.RESEARCH_CODER_MAX_SPAWNS)
            self.assertEqual(state.max_resumes, constants.RESEARCH_CODER_MAX_RESUMES)
            decision = coder.audit_manager._next_cli_job_launch("17", state)
            self.assertIsNotNone(decision)
            self.assertTrue(decision.is_resume)
            self.assertEqual(decision.resume_token, "audit-thread-17")

            claimed = coder.audit_manager._claim_cli_job_launch("17", "audit-resume", decision)
            self.assertIsNotNone(claimed)
            audit = manager.get_evaluation("17")["audit"]
            self.assertEqual(audit["spawn_count"], 0)
            self.assertEqual(audit["resume_count"], 1)
            self.assertEqual(audit["status"], "resuming")

    def test_auditor_resume_budget_exhaustion_schedules_fresh_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            manager.update_evaluation("17", lambda record: record.update({
                "status": "running",
                "audit": dict(
                    record.get("audit", {}) or {},
                    active=False,
                    status="pending_resume",
                    spawn_count=1,
                    resume_count=2,
                    max_resumes=2,
                    resume_token="audit-thread-exhausted",
                ),
            }))

            self.assertFalse(coder.audit_manager.launch_for_evaluation(manager.get_evaluation("17")))

            record = manager.get_evaluation("17")
            self.assertEqual("running", record["status"])
            self.assertEqual("retry", record["audit"]["status"])
            self.assertEqual("resume_limit_exceeded", record["audit"]["failure_category"])
            self.assertEqual(0, record["audit"]["resume_count"])
            self.assertIsNone(record["audit"]["resume_token"])

    def test_shared_cli_worker_resumes_auditor_after_429(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            run_dir = Path(coder.paths.coder_sessions_dir) / "audit_rate_limit"
            run_dir.mkdir(parents=True)
            transcript_path = run_dir / "transcript.jsonl"
            transcript_path.write_text(
                '{"type":"thread.started","thread_id":"audit-thread-429"}\n'
                '{"type":"error","message":"exceeded retry limit, last status: 429 Too Many Requests"}\n'
                '{"type":"turn.failed","error":{"message":"exceeded retry limit, last status: 429 Too Many Requests"}}\n',
                encoding="utf-8",
            )
            stderr_path = run_dir / "stderr.txt"
            stderr_path.write_text("", encoding="utf-8")
            session = SimpleNamespace(
                session_id="audit-rate-limit",
                backend="codex",
                process=SimpleNamespace(poll=lambda: 1),
                transcript_handle=mock.Mock(),
                stderr_handle=mock.Mock(),
                transcript_path=str(transcript_path),
                stderr_path=str(stderr_path),
                report_path=str(Path(coder.paths.audit_dir) / "17.md"),
                verdict_path=str(Path(coder.paths.audit_dir) / "17.verdict"),
                last_transcript_size=0,
                last_transcript_growth_timestamp=0.0,
                transcript_idle_timeout_triggered=False,
                transcript_idle_timeout_reason=None,
            )
            manager.update_evaluation("17", lambda r: r.update({
                "status": "running",
                "coder": dict(r.get("coder", {}) or {}, active=False, status="audit_running"),
                "audit": dict(
                    r.get("audit", {}) or {},
                    active=True,
                    status="running",
                    spawn_count=1,
                    resume_count=0,
                    started_timestamp=0,
                ),
            }))
            coder.audit_manager.active_sessions["17"] = session

            coder.audit_manager.poll_cli_jobs()

            audit = manager.get_evaluation("17")["audit"]
            self.assertEqual(audit["status"], "pending_resume")
            self.assertEqual(audit["resume_token"], "audit-thread-429")
            self.assertEqual(audit["resume_count"], 0)
            self.assertEqual(audit["resume_delay_seconds"], 10)
            self.assertGreater(audit["next_resume_timestamp"], 0)
            self.assertNotIn("17", coder.audit_manager.active_sessions)

    def test_shared_cli_worker_retries_unclassified_auditor_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, manager, _report_path = self._manager(Path(tmp))
            run_dir = Path(coder.paths.coder_sessions_dir) / "audit_unknown_failure"
            run_dir.mkdir(parents=True)
            transcript_path = run_dir / "transcript.jsonl"
            transcript_path.write_text(
                '{"type":"thread.started","thread_id":"audit-thread-unknown"}\n'
                '{"type":"turn.failed","error":{"message":"model configuration rejected"}}\n',
                encoding="utf-8",
            )
            stderr_path = run_dir / "stderr.txt"
            stderr_path.write_text("", encoding="utf-8")
            session = SimpleNamespace(
                session_id="audit-unknown-failure",
                backend="codex",
                process=SimpleNamespace(poll=lambda: 1),
                transcript_handle=mock.Mock(),
                stderr_handle=mock.Mock(),
                transcript_path=str(transcript_path),
                stderr_path=str(stderr_path),
                report_path=str(Path(coder.paths.audit_dir) / "17.md"),
                verdict_path=str(Path(coder.paths.audit_dir) / "17.verdict"),
                last_transcript_size=0,
                last_transcript_growth_timestamp=0.0,
                transcript_idle_timeout_triggered=False,
                transcript_idle_timeout_reason=None,
            )
            manager.update_evaluation("17", lambda r: r.update({
                "status": "running",
                "coder": dict(r.get("coder", {}) or {}, active=False, status="audit_running"),
                "audit": dict(
                    r.get("audit", {}) or {},
                    active=True,
                    status="running",
                    spawn_count=1,
                    resume_count=0,
                    started_timestamp=0,
                ),
            }))
            coder.audit_manager.active_sessions["17"] = session

            coder.audit_manager.poll_cli_jobs()

            audit = manager.get_evaluation("17")["audit"]
            self.assertEqual("pending_resume", audit["status"])
            self.assertEqual("infra_transient", audit["failure_category"])
            self.assertEqual("audit-thread-unknown", audit["resume_token"])
            self.assertNotIn("17", coder.audit_manager.active_sessions)

    def test_disabled_flag_does_not_install_audit_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            consts = SimpleNamespace(**{key: getattr(constants, key) for key in dir(constants) if key.isupper()})
            consts.BASE_STATION_DATA_PATH = str(Path(tmp) / "station_data")
            consts.RESEARCH_CODER_AUDIT_ENABLED = False
            paths = ensure_runtime_layout(consts)
            self.assertFalse(Path(paths.audit_dir).exists())
            self.assertFalse((Path(paths.research_root) / "submit_audit.sh").exists())

    def test_audit_submit_script_uses_startup_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder, _manager, _report_path = self._manager(Path(tmp))
            research_root = Path(coder.paths.research_root)
            script_text = (research_root / "submit_audit.sh").read_text(encoding="utf-8")
            snapshot_path = research_root / "_internal" / "submit_audit_cli_snapshot.py"
            self.assertTrue(snapshot_path.is_file())
            self.assertIn("_internal/submit_audit_cli_snapshot.py", script_text)
            self.assertNotIn("station.eval_research.submit_audit_cli", script_text)
            self.assertIn("STATION_RESEARCH_ROOT", script_text)
            self.assertIn("def submit_audit", snapshot_path.read_text(encoding="utf-8"))
            audit_report = Path(coder.paths.audit_dir) / "17.md"
            audit_report.write_text("snapshot audit report\n", encoding="utf-8")
            result = subprocess.run(
                [str(research_root / "submit_audit.sh"), "17", "pass"],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            self.assertEqual(
                (Path(coder.paths.audit_dir) / "17.verdict").read_text(encoding="utf-8").strip(),
                "pass",
            )


if __name__ == "__main__":
    unittest.main()
