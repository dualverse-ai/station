"""
Research Center coder launcher and session monitor.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from station import constants
from station import file_io_utils
from station import index_paths
from .coder_helpers import (
    get_coder_access_phase,
    is_coder_internet_allowed,
    render_coder_internet_policy,
    render_coder_access_policy,
    render_import_guidance,
    render_workflow_inspection_step,
)
from station.workers.cli import (
    apply_codex_proxy_overrides,
    build_cli_worker_runtime_env,
    candidate_js_bin_dirs,
    detect_cli_worker_executable,
    get_cli_worker_backend,
)
from station.workers.job_manager import (
    CliJobFailure,
    CliJobLaunchDecision,
    CliJobLaunchSpec,
    CliJobManager,
    CliJobSession,
    CliJobState,
)
from .evaluation_manager import EvaluationManager
from .runtime_paths import (
    ResearchRuntimePaths,
    detect_station_python_executable,
    ensure_lineage_storage,
    ensure_runtime_layout,
    load_task_spec_markdown_for_audience,
)


@dataclass(kw_only=True)
class ActiveCoderSession(CliJobSession):
    eval_id: str
    report_path: str
    debug_dir: Optional[str]
    dialogue_log_path: Optional[str]


@dataclass(kw_only=True)
class ActiveAuditSession(CliJobSession):
    eval_id: str
    report_path: str
    verdict_path: str


def _research_coder_resume_backoff_schedule():
    return constants.RESEARCH_CODER_RESUME_BACKOFF_SECONDS


class ResearchAuditManager(CliJobManager):
    """Small second CLI worker used to independently audit a completed coder report."""

    fresh_attempt_after_resume_exhaustion = True

    def __init__(self, owner: "ResearchCoderManager"):
        super().__init__()
        self.owner = owner
        self.eval_manager = owner.eval_manager
        self.paths = owner.paths
        self.on_verdict = owner._handle_audit_verdict

    def launch_for_evaluation(self, eval_data: Dict[str, Any]) -> bool:
        return self.launch_cli_job(str(eval_data.get("id")))

    @classmethod
    def _resume_backoff_schedule(cls):
        return _research_coder_resume_backoff_schedule()

    def _load_cli_job_state(self, job_id: str) -> CliJobState:
        record = self.eval_manager.get_evaluation(job_id) or {}
        audit = record.get("audit", {}) or {}
        status = str(record.get("status", "")).lower()
        audit_status = str(audit.get("status", "not_started")).lower()
        return CliJobState(
            backend=str((record.get("coder", {}) or {}).get("backend") or constants.RESEARCH_CODER_BACKEND).lower(),
            spawn_count=int(audit.get("spawn_count", 0) or 0),
            resume_count=int(audit.get("resume_count", 0) or 0),
            max_spawns=int(audit.get("max_spawns", constants.RESEARCH_CODER_MAX_SPAWNS)),
            max_resumes=int(audit.get("max_resumes", constants.RESEARCH_CODER_MAX_RESUMES)),
            resume_token=str(audit.get("resume_token") or "").strip() or None,
            next_resume_timestamp=audit.get("next_resume_timestamp"),
            fresh_launch_eligible=(status == "running" and audit_status in {"not_started", "queued", "retry"}),
            resume_launch_eligible=(status == "running" and audit_status in {"pending_resume", "resuming"}),
        )

    def _claim_cli_job_launch(self, job_id, session_id, decision):
        claimed = None
        def mutator(record):
            nonlocal claimed
            if str(record.get("status", "")).lower() != "running" or record.get("final"):
                return
            audit = record.setdefault("audit", {})
            if audit.get("active") or str(audit.get("status", "not_started")).lower() not in {
                "not_started", "queued", "retry", "pending_resume", "resuming",
            }:
                return
            is_resume = decision.is_resume
            audit.update({
                "status": "resuming" if is_resume else "running", "active": True, "session_id": session_id,
                "active_pid": None,
                "started_timestamp": (
                    audit.get("started_timestamp") or time.time()
                    if is_resume else time.time()
                ),
                "last_error": None,
                "completed_timestamp": None,
                "spawn_count": int(audit.get("spawn_count", 0) or 0) + (0 if is_resume else 1),
                "resume_count": int(audit.get("resume_count", 0) or 0) + (1 if is_resume else 0),
                "max_spawns": int(audit.get("max_spawns", constants.RESEARCH_CODER_MAX_SPAWNS)),
                "max_resumes": int(audit.get("max_resumes", constants.RESEARCH_CODER_MAX_RESUMES)),
            })
            claimed = record
        self.eval_manager.update_evaluation(job_id, mutator)
        return claimed

    def _build_audit_prompt(self, record: Dict[str, Any]) -> str:
        eval_id = str(record.get("id"))
        def read(path: str, limit: int = 200000) -> str:
            if not path or not file_io_utils.file_exists(path):
                return "(not available)"
            text = file_io_utils.load_text(path) or ""
            return text if len(text) <= limit else text[:limit] + "\n[truncated]"
        attempts = record.get("attempts") or []
        latest = attempts[-1] if attempts else {}
        coder_report_path = os.path.join(self.paths.reports_dir, f"{eval_id}.md")
        submission_path = os.path.join(self.paths.submissions_dir, f"{eval_id}.py")
        stdout_path = os.path.join(self.paths.stdout_dir, f"{eval_id}.log")
        stderr_path = os.path.join(self.paths.stderr_dir, f"{eval_id}.log")
        task_spec = load_task_spec_markdown_for_audience(constants, include_coder_only=True).strip() or "(not available)"
        previous = read(os.path.join(self.paths.audit_dir, f"{eval_id}.md"))
        audit = record.get("audit", {}) or {}
        failed_round = int(audit.get("repair_round", 0) or 0)
        max_rounds = max(1, int(getattr(constants, "RESEARCH_CODER_AUDIT_MAX_ROUNDS", 2) or 1))
        is_last_audit = failed_round + 1 >= max_rounds
        final_audit_guidance = (
            """
- This is the final configured audit. The original coder has exhausted the available repair attempts, so your report—whether pass or fail—will be delivered directly to the agent. As such, in both cases, please be concise and use at most two paragraphs.
"""
            if is_last_audit
            else
            """
- If failing, give a detailed and actionable account of the failure: identify the relevant code or report claim, explain why it is wrong or unsupported, describe its scientific impact, and state what the coder must fix. The original coder will receive your report directly.
"""
        )
        return f'''You are the independent auditor for Research Center evaluation `{eval_id}`.

Your sole job is to audit the submitted code, official result, and Coder Report. Do not modify any evaluation files or perform a new official submission.

Audit standards:
- Check whether the implementation faithfully follows the agent instruction.
- Judge the requested scientific work against the agent instruction. The task specification also defines the official evaluator, but an instruction may explicitly request an auxiliary analysis and use a format control whose score is irrelevant. In that case, do not fail merely because the auxiliary result is not itself a score-winning submission; verify the analysis and the control as requested.
- Check whether the code and official output support the Coder Report's scientific conclusions.
- Look for material errors such as omitted cases, incomplete searches described as exhaustive, unsafe pruning, invalid controls or invariants, incorrect formulas, timeouts treated as completed work, or conclusions contradicted by the code or official output.
- An honestly reported timeout or `UNDECIDED-AT-BUDGET` result is not an audit failure by itself. If the coder does not overclaim and states that another evaluation is needed, pass with that qualification. Fail only if the timeout is misrepresented as completed work or was caused by a material implementation error.
- Distinguish material scientific errors from minor reporting problems.
- If the underlying result remains scientifically valid and the problem is only wording, presentation, or scope that can be clarified without rerunning the experiment, mark the audit as `pass`. Briefly explain the qualification so the agent knows the precise limitation.
- Mark the audit as `fail` only when there is a material scientific or implementation error that requires the coder to change the code, rerun the experiment, substantially revise the conclusion, or provide missing decisive evidence.
{final_audit_guidance.rstrip()}
- If passing, write one short paragraph stating either that the evaluation is clear or listing any minor qualifications you noticed.

Write your report to `storage/audit/{eval_id}.md`, then submit exactly one verdict:
`bash submit_audit.sh {eval_id} pass` or `bash submit_audit.sh {eval_id} fail`.

## Research Task Specification
{task_spec}
## Agent Instruction
{record.get("instruction", "")}
## Submitted Code
{read(submission_path)}
## Official Attempt Status
{latest.get("status", "(not available)")}
## Official Score
{latest.get("primary_score", "(not available)")}
## Secondary Metrics
{latest.get("evaluation_details", "(not available)")}
## Official Stdout
{read(stdout_path)}
## Official Stderr
{read(stderr_path)}
## Coder Report
{read(coder_report_path)}
## Previous Auditor Report
{previous}
'''

    def _build_cli_job_launch_spec(self, job_id, session_id, decision, claimed):
        coder = claimed.get("coder", {}) or {}
        backend = str(coder.get("backend") or constants.RESEARCH_CODER_BACKEND).lower()
        env = self.owner._build_runtime_env()
        if backend == "codex":
            apply_codex_proxy_overrides(env)
        prompt = self._build_audit_prompt(claimed)
        run_dir = os.path.abspath(os.path.join(self.paths.coder_sessions_dir, f"audit_{session_id}"))
        return CliJobLaunchSpec(
            executable=self.owner._detect_backend_executable(backend, env), run_dir=run_dir,
            backend=backend, model_name=coder.get("model_name"),
            workspace_root=os.path.abspath(self.paths.research_root), storage_root=self.paths.storage_root,
            prompt=prompt, env=env,
            extra_allowed_roots=tuple(self.owner._extra_allowed_roots()),
            network_access=is_coder_internet_allowed(claimed, constants),
        )

    def _before_cli_job_process_launch(self, job_id, decision, prepared, spec):
        # The prompt already contains the previous report. Remove round-global
        # artifacts so an interrupted worker cannot satisfy this launch with a
        # stale report or verdict from an earlier audit round.
        for suffix in (".md", ".verdict", ".verdict.tmp"):
            try:
                os.remove(os.path.join(self.paths.audit_dir, f"{job_id}{suffix}"))
            except FileNotFoundError:
                pass

    def _make_active_cli_job_session(self, job_id, base_session, decision, claimed, launch_metadata):
        return ActiveAuditSession(
            **vars(base_session), eval_id=job_id,
            report_path=os.path.abspath(os.path.join(self.paths.audit_dir, f"{job_id}.md")),
            verdict_path=os.path.abspath(os.path.join(self.paths.audit_dir, f"{job_id}.verdict")),
        )

    def _mark_cli_job_pid(self, job_id, pid):
        self.eval_manager.update_evaluation(job_id, lambda r: r.setdefault("audit", {}).update({"active_pid": pid}))

    def _cli_job_completion_ready(self, job_id, session):
        if not (file_io_utils.file_exists(session.report_path) and file_io_utils.file_exists(session.verdict_path)):
            return False
        return file_io_utils.load_text(session.verdict_path).strip().lower() in {"pass", "fail"}

    def _on_cli_job_completed(self, job_id, session, returncode):
        report = file_io_utils.load_text(session.report_path) or ""
        verdict = (file_io_utils.load_text(session.verdict_path) or "").strip().lower()
        self.eval_manager.update_evaluation(job_id, lambda r: r.setdefault("audit", {}).update({
            "active": False, "active_pid": None, "status": verdict, "last_verdict": verdict,
            "last_report_path": os.path.relpath(session.report_path, self.paths.research_root),
            "completed_timestamp": time.time(), "last_error": None,
        }))
        self.on_verdict(job_id, verdict, report, session.session_id)

    def _on_cli_job_no_report_exit(self, job_id, session, returncode):
        self.eval_manager.update_evaluation(job_id, lambda r: r.setdefault("audit", {}).update({
            "active": False, "active_pid": None, "status": "retry",
            "last_error": f"Auditor exited without a valid report/verdict (returncode={returncode}).",
        }))

    def _schedule_cli_job_fresh_attempt(self, job_id, session, failure):
        self.eval_manager.update_evaluation(job_id, lambda r: r.setdefault("audit", {}).update({
            "active": False, "active_pid": None, "status": "retry", "last_error": failure.reason,
            "failure_category": failure.category, "resume_token": None, "resume_count": 0,
            "resume_delay_seconds": 0, "next_resume_timestamp": None,
        }))

    def _schedule_cli_job_resume(self, job_id, session, failure):
        def mutator(record):
            audit = record.setdefault("audit", {})
            audit.update({
                "active": False,
                "active_pid": None,
                "status": "pending_resume",
                "completed_timestamp": time.time(),
                "failure_category": failure.category,
                "resume_token": failure.resume_token or audit.get("resume_token"),
                "resume_count": failure.resume_count,
                "resume_delay_seconds": failure.delay_seconds,
                "next_resume_timestamp": failure.next_resume_timestamp,
                "last_error": failure.reason,
            })
            record["status"] = "running"

        self.eval_manager.update_evaluation(job_id, mutator)
        self.owner._push_log_event(
            "research_coder_audit_retryable_infra_failure",
            {
                "eval_id": job_id,
                "session_id": getattr(session, "session_id", None),
                "returncode": failure.returncode,
                "reason": failure.reason,
                "resume_delay_seconds": failure.delay_seconds,
                "next_resume_timestamp": failure.next_resume_timestamp,
            },
        )
        delay_suffix = f" after {failure.delay_seconds}s" if failure.delay_seconds > 0 else ""
        print(
            f"ResearchAuditManager: Scheduling same-session resume for evaluation {job_id} "
            f"(session={session.session_id}, reason={failure.reason}){delay_suffix}"
        )

    def _on_cli_job_attempts_exhausted(self, job_id, session, failure):
        state = self._load_cli_job_state(job_id)
        if failure.resume_budget_exhausted:
            reason = (
                f"Research Center auditor for evaluation {job_id} exceeded the configured resume limit "
                f"of {state.max_resumes} without producing a verdict. Manual intervention is required."
            )
        else:
            reason = (
                f"Research Center auditor for evaluation {job_id} exited without a verdict after "
                f"{state.spawn_count} total spawns. Manual intervention is required."
            )
        self.eval_manager.update_evaluation(job_id, lambda r: r.setdefault("audit", {}).update({
            "active": False, "active_pid": None, "status": "blocked", "last_error": reason,
            "failure_category": "resume_limit_exceeded" if failure.resume_budget_exhausted else failure.category,
            "resume_token": failure.resume_token or None,
        }))
        self.owner._pause_station(reason)

    def _before_cli_job_poll(self):
        timeout = int(getattr(constants, "RESEARCH_CODER_AUDIT_TIMEOUT_SECONDS", 0) or 0)
        if timeout <= 0:
            return
        now = time.time()
        for eval_id, session in list(self.active_sessions.items()):
            record = self.eval_manager.get_evaluation(eval_id) or {}
            started = float((record.get("audit", {}) or {}).get("started_timestamp") or 0)
            if started and now - started >= timeout:
                self._terminate_cli_job_process(session, force=False)
                self.eval_manager.update_evaluation(eval_id, lambda r: r.setdefault("audit", {}).update({
                    "last_error": f"Auditor exceeded its {timeout}-second timeout.",
                }))

    def _cli_job_idle_timeout_reason(self, job_id, session, idle_seconds, timeout_seconds):
        return (
            f"Codex CLI transcript for independent audit of evaluation {job_id} did not grow for "
            f"{idle_seconds} seconds, exceeding the configured CLI worker transcript idle timeout "
            f"of {timeout_seconds} seconds."
        )

    def _on_cli_job_idle_timeout(self, job_id, session, reason, idle_seconds, timeout_seconds):
        self.eval_manager.update_evaluation(
            job_id,
            lambda record: record.setdefault("audit", {}).update({"last_error": reason}),
        )
        self.owner._push_log_event(
            "research_coder_audit_codex_transcript_idle_timeout",
            {
                "eval_id": job_id,
                "session_id": session.session_id,
                "idle_seconds": idle_seconds,
                "timeout_seconds": timeout_seconds,
            },
        )


class ResearchCoderManager(CliJobManager):
    fresh_attempt_after_resume_exhaustion = True

    def __init__(
        self,
        station_instance,
        eval_manager: EvaluationManager,
        paths: Optional[ResearchRuntimePaths] = None,
        log_queue=None,
        pause_callback: Optional[Callable[[str], None]] = None,
    ):
        super().__init__()
        self.station = station_instance
        self.eval_manager = eval_manager
        self.paths = paths or ensure_runtime_layout()
        if not hasattr(self.paths, "audit_dir"):
            self.paths.audit_dir = os.path.join(self.paths.storage_root, getattr(constants, "RESEARCH_AUDIT_SUBDIR_NAME", "audit"))
        if bool(getattr(constants, "RESEARCH_CODER_AUDIT_ENABLED", True)):
            file_io_utils.ensure_dir_exists(self.paths.audit_dir)
        self.log_queue = log_queue
        self.pause_callback = pause_callback
        self.pending_report_finalizations: Dict[str, Dict[str, Any]] = {}
        self.audit_manager = ResearchAuditManager(self)

    def has_active_sessions(self) -> bool:
        audit_active = bool(
            getattr(constants, "RESEARCH_CODER_AUDIT_ENABLED", True)
            and self.audit_manager.active_sessions
        )
        return bool(self.active_sessions or audit_active)

    def _push_log_event(self, event_type: str, data: Dict[str, Any]):
        if self.log_queue is None:
            return
        try:
            self.log_queue.put_nowait({"event": event_type, "data": data, "timestamp": time.time()})
        except Exception as exc:
            print(f"ResearchCoderManager: Failed to queue log event: {exc}")

    def _pause_station(self, reason: str):
        print(f"ResearchCoderManager: {reason}")
        self._push_log_event("research_coder_pause", {"reason": reason})
        if self.pause_callback:
            self.pause_callback(reason)

    def _extra_allowed_roots(self) -> List[str]:
        index_db_path = index_paths.get_station_index_database_path(constants.BASE_STATION_DATA_PATH)
        return [os.path.dirname(index_db_path)]

    @classmethod
    def _resume_backoff_schedule(cls) -> Any:
        return _research_coder_resume_backoff_schedule()

    def _get_debug_dir(self, eval_id: str) -> Optional[str]:
        if not getattr(constants, "RESEARCH_CODER_DEBUG_DUMP_ENABLED", False):
            return None
        debug_dir = os.path.join(self.paths.debug_tmp_root, str(eval_id))
        file_io_utils.ensure_dir_exists(debug_dir)
        return debug_dir

    def _append_dialogue_log(self, dialogue_log_path: Optional[str], heading: str, body: str):
        if not dialogue_log_path:
            return
        existing = file_io_utils.load_text(dialogue_log_path) if file_io_utils.file_exists(dialogue_log_path) else ""
        section = f"\n=== {heading} ===\n{body.rstrip()}\n" if body else f"\n=== {heading} ===\n"
        file_io_utils.save_text((existing or "") + section, dialogue_log_path)

    def _write_debug_session_header(
        self,
        eval_id: str,
        session_id: str,
        backend: str,
        command: List[str],
        prompt: str,
        transcript_path: str,
        transcript_format: str,
        last_message_path: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        debug_dir = self._get_debug_dir(eval_id)
        if not debug_dir:
            return None, None

        dialogue_log_path = os.path.join(debug_dir, "dialogue.log")
        body = "\n".join(
            [
                f"eval_id: {eval_id}",
                f"session_id: {session_id}",
                f"backend: {backend}",
                f"started_at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
                f"transcript_path: {transcript_path}",
                f"transcript_format: {transcript_format}",
                f"last_message_path: {last_message_path}",
                "",
                "Command:",
                " ".join(command),
                "",
                "Prompt:",
                prompt,
            ]
        )
        self._append_dialogue_log(dialogue_log_path, f"SESSION {session_id} START", body)
        return debug_dir, dialogue_log_path

    def _write_debug_session_footer(self, session: ActiveCoderSession, returncode: Optional[int]):
        if not session.dialogue_log_path:
            return

        transcript_text = (
            file_io_utils.load_text(session.transcript_path)
            if file_io_utils.file_exists(session.transcript_path)
            else ""
        )
        stderr_text = file_io_utils.load_text(session.stderr_path) if file_io_utils.file_exists(session.stderr_path) else ""
        last_message = (
            file_io_utils.load_text(session.last_message_path)
            if session.last_message_path and file_io_utils.file_exists(session.last_message_path)
            else ""
        )
        report_text = (
            file_io_utils.load_text(session.report_path)
            if file_io_utils.file_exists(session.report_path)
            else ""
        )

        footer_body = "\n".join(
            [
                f"returncode: {returncode}",
                "",
                f"transcript ({session.transcript_format}):",
                transcript_text or "(empty)",
                "",
                "stderr:",
                stderr_text or "(empty)",
                "",
                "last_message:",
                last_message or "(empty)",
                "",
                "report:",
                report_text or "(empty)",
            ]
        )
        self._append_dialogue_log(session.dialogue_log_path, f"SESSION {session.session_id} END", footer_body)

    def _detect_backend_executable(self, backend: str, env: Dict[str, str]) -> str:
        return detect_cli_worker_executable(backend, env)

    def _get_cli_job_backend(self, backend: str):
        return get_cli_worker_backend(backend)

    def _candidate_js_bin_dirs(self) -> List[str]:
        return candidate_js_bin_dirs()

    def _build_runtime_env(self) -> Dict[str, str]:
        env = build_cli_worker_runtime_env(constants.RESEARCH_EVAL_PYTHON_CONDA_ENV)
        research_root = os.path.abspath(self.paths.research_root)
        station_data_root = os.path.abspath(os.path.join(research_root, os.pardir, os.pardir))
        env["STATION_BASE_DATA_PATH"] = station_data_root
        if bool(getattr(constants, "RESEARCH_SEED_BANK_ENABLED", False)):
            env["STATION_RESEARCH_ROOT"] = research_root
        return env

    def _render_recent_attempts(self, lineage: str, eval_id: str) -> str:
        recent = self.eval_manager.get_recent_lineage_attempt_summaries(lineage, limit=5, exclude_eval_id=eval_id)
        if not recent:
            return f"No prior completed evaluations from lineage `{lineage}`."
        lines = []
        for item in recent:
            lines.append(f"- Eval #{item['id']} | score: {item['score']} | abstract: {item['abstract']}")
        return "\n".join(lines)

    def _ensure_lineage_tmp_storage(self, lineage: str) -> str:
        tmp_lineage_path = os.path.join(self.paths.tmp_storage, (lineage or "unknown").lower())
        file_io_utils.ensure_dir_exists(tmp_lineage_path)
        return tmp_lineage_path

    def _find_previous_session_dirs(self, eval_id: str, current_session_id: str) -> List[str]:
        if not os.path.isdir(self.paths.coder_sessions_dir):
            return []

        pattern = re.compile(
            rf"^[A-Za-z0-9_-]+_{re.escape(str(eval_id))}_spawn_([0-9]+)_[A-Za-z0-9]+$"
        )
        session_dirs: List[tuple[int, str, str]] = []
        for name in os.listdir(self.paths.coder_sessions_dir):
            if name == current_session_id:
                continue
            match = pattern.match(name)
            if not match:
                continue
            full_path = os.path.join(self.paths.coder_sessions_dir, name)
            if not os.path.isdir(full_path):
                continue
            try:
                spawn_number = int(match.group(1))
            except (TypeError, ValueError):
                spawn_number = 0
            session_dirs.append((spawn_number, name, full_path))

        session_dirs.sort(key=lambda item: (item[0], item[1]))
        return [full_path for _spawn_number, _name, full_path in session_dirs]

    def _render_previous_session_artifact_block(
        self,
        eval_id: str,
        previous_session_dirs: Optional[List[str]],
        previous_failure_reason: Optional[str],
    ) -> str:
        if not previous_session_dirs:
            return ""

        relative_paths = [
            os.path.relpath(session_dir, self.paths.research_root).rstrip("/") + "/"
            for session_dir in previous_session_dirs
        ]
        path_lines = "\n".join(f"- `{path}`" for path in relative_paths)
        failure_line = ""
        if previous_failure_reason:
            failure_line = f"- Last recorded failure: {previous_failure_reason}\n"
        audit_repair = str(previous_failure_reason or "").startswith("Independent audit failed")
        context_line = (
            "- This is a fresh fallback because the completed coder conversation could not be resumed for the audit repair."
            if audit_repair
            else "- This is a fresh respawn after an earlier coder session crashed, timed out, or exited without writing the final report."
        )

        return f"""Previous respawn context
{context_line}
{failure_line}- Prior session artifacts are available at:
{path_lines}
- Inspect useful prior artifacts such as `prompt.txt`, `transcript.jsonl` or `stdout.txt`, `stderr.txt`, and `last_message.txt` before restarting from scratch.
- Existing files such as `storage/submission/{eval_id}.py` and disposable scratch under `storage/tmp/...` may also contain useful work from the earlier session. Reuse them only after verifying they still match the agent instruction.
"""

    def _build_resume_prompt(self, eval_data: Dict[str, Any]) -> str:
        eval_id = str(eval_data.get("id"))
        lineage = str(eval_data.get("lineage", "unknown")).lower()
        attempts_used = len(eval_data.get("attempts") or [])
        coder_state = eval_data.get("coder", {}) or {}
        try:
            max_attempts = int(
                coder_state.get("max_attempts", constants.RESEARCH_CODER_MAX_ATTEMPTS)
            )
        except (TypeError, ValueError):
            max_attempts = int(constants.RESEARCH_CODER_MAX_ATTEMPTS)
        attempts_remaining = max(0, max_attempts - attempts_used)
        audit = eval_data.get("audit", {}) or {}
        if str(audit.get("status", "")).lower() == "repairing":
            audit_path = os.path.join(self.paths.audit_dir, f"{eval_id}.md")
            audit_report = (
                file_io_utils.load_text(audit_path)
                if file_io_utils.file_exists(audit_path)
                else "(auditor report is unavailable)"
            )
            failed_round = max(1, int(audit.get("repair_round", 1) or 1))
            max_rounds = max(1, int(getattr(constants, "RESEARCH_CODER_AUDIT_MAX_ROUNDS", 2) or 1))
            final_repair_note = (
                "This is your last coder repair round. If the audit still fails after this repair, "
                "the failure will be communicated directly to the submitting agent without further "
                "coder repair. Make sure to fix every material problem raised in the auditor report."
                if failed_round + 1 >= max_rounds
                else ""
            )
            return f"""Resume the same Research Center coder session for evaluation {eval_id} after the independent auditor identified a material problem.

The original research task, agent instruction, access policy, execution rules, and prior conversation remain in force. Continue from the existing workspace, submission, artifacts, and Coder Report; do not restart the implementation from scratch. {final_repair_note}

Official attempt budget for this evaluation: {attempts_used} used, {attempts_remaining} remaining out of {max_attempts}. Audit rounds do not reset this budget. Use the remaining official attempt only when the audit finding requires changed code or decisive new evidence.

## Independent Auditor Report

{audit_report}

## Required Repair

- Address every material issue identified by the auditor. Preserve prior work that remains valid.
- Reinspect the current submission, official stdout/stderr, persistent artifacts, and Coder Report.
- Do not merely rewrite the report when the auditor identified a code, implementation, or evidence problem.
- Run a new official attempt only when changed code or missing decisive evidence requires it. Do not invent a different experiment.
- Replace `storage/report/{eval_id}.md` with the corrected Coder Report.
- Update `storage/{lineage}/CODER.md` only with durable, reusable information.
- Finish the repair completely so the independent auditor can perform the final audit.
"""
        tmp_workspace = f"storage/tmp/{lineage}/eval_{eval_id}"
        return f"""Resume the same Research Center coding session for evaluation {eval_id}.

Your previous Codex session was interrupted by a transient backend/provider failure before you wrote `storage/report/{eval_id}.md`.

Continue from the existing workspace state instead of restarting the experiment from scratch.
- Reuse the files already written under `storage/{lineage}` and `storage/submission/{eval_id}.py` if they are still correct.
- Use disposable scratch such as `{tmp_workspace}` for temporary scripts, probes, caches, and mutable workspace files that do not need to persist.
- Use `{tmp_workspace}/sage` for Sage/CAS caches; do not put caches under persistent lineage storage.
- Continue the same single experiment only.
- If the official attempt has not been launched yet, launch it.
- If the official attempt already finished, inspect the final stdout/stderr and write `storage/report/{eval_id}.md`.

Finish by writing `storage/report/{eval_id}.md`."""

    def _build_prompt(
        self,
        eval_data: Dict[str, Any],
        previous_session_dirs: Optional[List[str]] = None,
        previous_failure_reason: Optional[str] = None,
    ) -> str:
        lineage = str(eval_data.get("lineage", "unknown")).lower()
        lineage_root = ensure_lineage_storage(self.paths, lineage)
        self._ensure_lineage_tmp_storage(lineage)
        coder_md_path = os.path.join(lineage_root, "CODER.md")
        coder_md = file_io_utils.load_text(coder_md_path) if file_io_utils.file_exists(coder_md_path) else ""
        coder_md_word_count = len(coder_md.split())
        task_spec = (
            load_task_spec_markdown_for_audience(constants, include_coder_only=True).strip()
            or "No research task spec is available."
        )
        recent_attempts = self._render_recent_attempts(lineage, str(eval_data.get("id")))
        python_bin = detect_station_python_executable(self._build_runtime_env())
        eval_id = str(eval_data.get("id"))
        attempts_used = len(eval_data.get("attempts") or [])
        coder_state = eval_data.get("coder", {}) or {}
        try:
            max_attempts = int(
                coder_state.get("max_attempts", constants.RESEARCH_CODER_MAX_ATTEMPTS)
            )
        except (TypeError, ValueError):
            max_attempts = int(constants.RESEARCH_CODER_MAX_ATTEMPTS)
        attempts_remaining = max(0, max_attempts - attempts_used)
        attempt_budget_line = (
            f"- Official attempts for this evaluation: {attempts_used} used, "
            f"{attempts_remaining} remaining out of {max_attempts}; audit rounds do not reset this budget.\n"
        )
        tmp_workspace = f"storage/tmp/{lineage}/eval_{eval_id}"
        sage_workspace = f"{tmp_workspace}/sage"
        sage_home = f"{sage_workspace}/home"
        sage_dotdir = f"{sage_workspace}/dot_sage"
        sage_xdg_cache = f"{sage_workspace}/xdg_cache"
        access_phase = get_coder_access_phase(eval_data)
        access_policy = render_coder_access_policy(eval_data, lineage, eval_id, constants)
        internet_access_allowed = is_coder_internet_allowed(eval_data, constants)
        internet_policy = render_coder_internet_policy(eval_data, constants)
        internet_context_line = (
            "- Internet access: enabled"
            if internet_access_allowed
            else "- No internet access by default"
        )
        internet_policy_section = f"\n{internet_policy}" if internet_policy else ""
        restricted_request_kinds = (
            "hacking, penetration, exploit, malware, credential theft, unauthorized-access code"
            if internet_access_allowed
            else "hacking, penetration, exploit, malware, credential theft, external web/network access, unauthorized-access code"
        )
        import_guidance = render_import_guidance(access_phase, lineage)
        workflow_inspection_step = render_workflow_inspection_step(access_phase)
        previous_session_artifact_block = self._render_previous_session_artifact_block(
            eval_id,
            previous_session_dirs,
            previous_failure_reason,
        )
        audit = eval_data.get("audit", {}) or {}
        audit_repair_block = ""
        if str(audit.get("status", "")).lower() == "repairing" or int(audit.get("repair_round", 0) or 0) > 0:
            audit_path = os.path.join(self.paths.audit_dir, f"{eval_id}.md")
            audit_report = file_io_utils.load_text(audit_path) if file_io_utils.file_exists(audit_path) else ""
            audit_repair_block = f"""Independent audit repair context
- This is an audit repair after an independent audit found a material scientific problem.
- Read `storage/audit/{eval_id}.md` and fix the specific issue. Do not merely rewrite the report; change and rerun the submission when the report requests it.
- Previous audit report:
{audit_report}

"""
        timeout_seconds = int(getattr(constants, "RESEARCH_CODER_TIMEOUT_SECONDS", 0) or 0)
        timing_block = ""
        if timeout_seconds > 0:
            coder_state = eval_data.get("coder", {}) or {}
            try:
                started_timestamp = float(coder_state.get("started_timestamp") or time.time())
            except (TypeError, ValueError):
                started_timestamp = time.time()
            deadline_timestamp = started_timestamp + timeout_seconds
            timing_block = (
                "Session timing\n"
                f"- The session started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_timestamp))}.\n"
                f"- The report deadline is {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(deadline_timestamp))} ({timeout_seconds} seconds after session start).\n"
                "- The session will be automatically terminated if the final report is not submitted within 15 minutes after the deadline.\n"
                "- Track elapsed time against the deadline.\n"
                "- If you have to finalize the report because of the time limit, state that you have insufficient time to carry out the whole task within one evaluation and ask the agent to resubmit another evaluation so you can continue."
            )
            timing_block = f"{timing_block}\n\n"
        gpu_management_enabled = constants.RESEARCH_EVAL_GPU_NUM is not None or bool(constants.RESEARCH_EVAL_USE_DIFF_GPU)
        evaluation_memory_limit = str(constants.RESEARCH_EVAL_MEMORY_LIMIT or "not configured")
        cpu_only_submission_tip = ""
        if gpu_management_enabled:
            cpu_only_submission_tip = (
                "- Local probe commands may have no GPU/CUDA access even when official attempts can receive a station-managed GPU. Do not infer that the official attempt lacks GPU access from local `torch.cuda.is_available() == False`, empty local JAX CUDA devices, or a failed local `nvidia-smi` check.\n"
                f"- If the agent instruction requests GPU/CUDA/JAX/PyTorch GPU use, GPU-scale search, or says results are only meaningful on GPU, do not pass `--cpu-only`. Start the official attempt with `bash submit_eval.sh {eval_id}` and let the scheduler allocate the GPU.\n"
                f"- If the submitted script does not need GPU resources and the agent did not request GPU access, start the official attempt with `bash submit_eval.sh {eval_id} --cpu-only`. This tells the scheduler not to reserve a GPU and hides CUDA devices in the sandbox.\n"
                f"- Use the normal `bash submit_eval.sh {eval_id}` command when GPU acceleration is needed, requested, or when you are unsure.\n"
            )
        seed_bank_guidance = ""
        if bool(getattr(constants, "RESEARCH_SEED_BANK_ENABLED", False)):
            max_seed_candidates = max(
                1, int(getattr(constants, "RESEARCH_SEED_BANK_MAX_CANDIDATES", 64))
            )
            if access_phase == "immature":
                seed_bank_direct_access = f"""- You may directly read an underlying NPZ artifact when needed, for example when the agent explicitly requests low-level artifact access. Because this session was requested by an immature agent from lineage `{lineage}`, first obtain the `SeedRecord` through the helper, then read only the NPZ members named by that returned record's `descriptor` from `bank.bank_root / record.artifact_path`. Do not browse arbitrary Seed Bank artifacts or read other members from the NPZ."""
                seed_bank_scope = f"""- This session was requested by an immature agent from lineage `{lineage}`. The helper automatically restricts every query to seeds submitted by lineage `{lineage}`."""
            else:
                seed_bank_direct_access = """- You may directly read underlying NPZ artifacts when needed, for example when the agent explicitly requests low-level artifact access. Resolve a selected record with `bank.bank_root / record.artifact_path` and keep access read-only."""
                seed_bank_scope = """- This session was requested by a mature agent. The helper automatically queries the entire station-wide Seed Bank."""
            seed_bank_guidance = f"""Seed Bank (read-only)
- Station automatically scores and stores valid candidates returned by this task, up to {max_seed_candidates} candidates per official attempt. Do not write to the Seed Bank or its SQLite/NPZ files directly.
- {max_seed_candidates} is a maximum, not a target. Return only candidates that are independently useful for future continuation. Return fewer candidates or `None` when appropriate; never add tiny perturbations, rescalings, repeated checkpoints, or other filler merely to fill the batch.
- Open the frozen helper with:
  ```python
  import sys
  sys.path.insert(0, "storage/system")
  from seed_bank import SeedBank
  bank = SeedBank.open()
  ```
- Get focused signatures, arguments, and copyable examples with `python storage/system/seed_bank.py --help TOPIC` or `print(SeedBank.help("TOPIC"))`. Available topics are `top`, `sample`, `from_evaluations`, `load`, `load_population`, `iter_all`, `iter_batches`, `rank`, `rank_metadata`, `distinct`, `filters`, `metrics`, `metadata`, and `summary`. Read only the topics needed for this experiment.
- The helper supports indexed filtering and ordering, bounded-batch loading, uniform sampling, custom reranking, exact-content deduplication, per-evaluation caps, and caller-defined structural diversity selection.
- Exact-content deduplication removes only numerically identical stored content. It does not establish that candidates belong to different structural or optimization basins.
- If the agent defines structural features, invariances, a distance or clustering rule, and a threshold, implement that definition faithfully. Do not substitute or invent another structural-distance rule.
- If the agent requests "distinct" seeds without defining structural distinctness, apply only exact-content deduplication. In the Coder Report, explicitly warn that the loaded seeds may still be variations from the same optimization basin; do not describe them as structurally diverse.
- When the agent requests structural diversity selection, report the candidate-pool size, retained size, score range, nearest-neighbor distance statistics, and concentration by source evaluation and lineage.
- For large banks, prefer indexed filters, bounded `top` pools, per-evaluation caps, `rank_metadata`, or bounded-batch iteration. Numerical reranking or structural comparison over a large pool is linear in the loaded seed data.
{seed_bank_direct_access}
{seed_bank_scope}

"""

        return f"""You are the Research Center Coder for evaluation {eval_id}.

This is a non-interactive coding section. You cannot ask follow-up questions. Make reasonable assumptions, complete the work, and finish by writing `storage/report/{eval_id}.md`.

Your job is to faithfully implement the agent's instruction for the assigned research task, execute it, inspect the result, debug if needed, and leave a final Coder Report.

Primary goal
- Make the agent's requested experiment or analysis run faithfully.
- If the agent does not explicitly specify a compute budget, use the maximum compute allowed by the task specification within the single official attempt instead of choosing a cheap arbitrary run. Do not impose a short per-process timeout merely to collect a quick diagnostic unless the task requires it or the constraint is unavoidable.
- If the agent explicitly specifies a compute budget, you must respect it. If an implementation, environment, or safety constraint makes any deviation inevitable, state the deviation and its reason to the agent upfront, and report the requested versus actual compute afterward.
- Avoid repeated full-run official attempts by using quick end-to-end tests and runtime estimates to catch preventable errors beforehand. Do not use multiple official attempts for score optimization.
- A completed official attempt generally means the implementation succeeded well enough to report back, even when the primary score is `n.a.`, absent, or poor. Retry only when there was an exception or the result is materially misaligned with the agent's instruction.

Context for this evaluation
- Lineage: {lineage}
- Working directory: {os.path.abspath(self.paths.research_root)}
- Python executable: {python_bin}
{attempt_budget_line}- Max attempts: {max_attempts}
{internet_context_line}

{timing_block}{access_policy}{internet_policy_section}

Paths for this evaluation
- Submission: `storage/submission/{eval_id}.py`
- Stdout log: `storage/stdout/{eval_id}.log`
- Stderr log: `storage/stderr/{eval_id}.log`
- Final report: `storage/report/{eval_id}.md`
- Persistent lineage memory: `storage/{lineage}/CODER.md`
- Disposable lineage scratch root: `storage/tmp/{lineage}`

   {previous_session_artifact_block}{audit_repair_block}
{seed_bank_guidance}{import_guidance}
- Do not rely on `__file__`-relative path resolution to find `storage/...`.
- The evaluator copies the submission into a temporary sandbox file such as `run.py` before import, so `__file__` will not point to `storage/submission/{eval_id}.py`.
- Correct path assumptions should be based on the working directory's `storage/...` view or another robust path that does not depend on `__file__`.

Persistent file I/O
- Submission code may directly read and write persistent files only in locations allowed by the Research access and filesystem policy above.
- If the agent asks to write or emit files and does not specify a path, save them under `storage/{lineage}/data/`.
- If reusable code is created, prefer placing it in lineage library files for future submissions.
- For long-running optimization or search, when practical, periodically atomically checkpoint resumable state and the best-so-far result under `storage/{lineage}/data/`; document the path and resume procedure in the Coder Report and `CODER.md` so a later evaluation can continue after timeout.
- Use `storage/tmp/{lineage}` for disposable per-evaluation work that agents do not need to inspect, such as temporary test scripts, probes, generated intermediates, caches, and mutable workspaces. Prefer this per-evaluation workspace: `{tmp_workspace}`.
- Do not put disposable work under `storage/{lineage}` or `storage/{lineage}/data`; those are persistent, agent-visible lineage storage.

Lineage memory maintenance
- Current `storage/{lineage}/CODER.md` word count at launch: {coder_md_word_count}.
- This file will be read by future coder sessions working on submissions from the same lineage, so keep it concise, current, and reusable.
- Before finishing, keep `storage/{lineage}/CODER.md` at or below 5,000 words.
- Do not over-prune: around 5,000 words is acceptable when the file contains useful durable context that prevents future sessions from rediscovering the same facts.
- Check the current count with: `wc -w storage/{lineage}/CODER.md`.
- Prune stale, duplicated, outdated, or overly detailed experiment notes.
- Preserve durable reusable facts: important file paths, current best artifacts, validated pitfalls, interfaces, and conclusions that prevent repeated failed work.
- If long details are still useful, move them to a separate markdown file under `storage/{lineage}/data/notes/` and leave only a short pointer in `CODER.md`.

Prior evaluation inspection
- If you need to inspect a prior evaluation by ID, run `bash eval_tool.sh preview <ID>` first.
- The preview shows evaluation metadata, abstract, the agent instruction, Coder Report, and stdout path without printing coder prompts, raw code, stdout, or stderr.
- Read raw evaluation YAML, submission code, stdout, or stderr only when the preview is insufficient for the specific technical detail you need.

Token conservation
- Prefer `bash eval_tool.sh preview <ID>` before reading raw evaluation YAML, code, stdout, or stderr.
- Use targeted `rg`/`sed` reads; avoid broad dumps over all of `storage/`, `evaluations/`, `coder_sessions/`, or `research_center/`.
- When searching common terms, add specific paths, filenames, eval IDs, or method names so output stays relevant.
- Read stdout/stderr only when the preview and Coder Report are insufficient; start with targeted `rg` or `tail`.
- Spend tokens when needed for correctness, but avoid printing irrelevant or duplicative context. Keep stdout concise: print scientifically valuable results and debugging-relevant diagnostics, not large unneeded dumps.

Execution and submission rules
- You may use `{python_bin}` for local testing or probing if needed. Local tests are advisory only. The official attempt is started only through `bash submit_eval.sh {eval_id}`.
{cpu_only_submission_tip}- Direct Python execution is reserved for lightweight code such as debugging, sanity checks, and local testing, not computationally intensive code.
- For computationally intensive code, you must use the official submit path. Tasks will usually allow sandbox submission for computationally intensive processes.
- The configured official-attempt memory limit is `{evaluation_memory_limit}`. Direct local commands do not inherit this limit.
- Run any local probe that may use substantial memory through `bash local_probe.sh [--timeout SECONDS] -- <command>`. The wrapper enforces the same `{evaluation_memory_limit}` memory limit. If `--timeout` is omitted, no timeout is applied.
- Only one execution attempt may be active at a time. Never submit a new attempt while the previous one is still running.
- Multi-attempt is for debugging only, not for cross-attempt optimization.
- A completed attempt is not a debugging failure just because the primary score is `n.a.`, absent, or poor. Finalize unless there was an exception, validation error, wrong submission, malformed output, or material mismatch with the agent's instruction.
- A computational timeout is not itself a failed scientific result. Unless it was caused by a specific implementation error that you corrected, do not resubmit; report the completed evidence, mark unfinished work as `UNDECIDED-AT-BUDGET`, and state that another evaluation is needed.
- Even if the agent says "optimize", that optimization must stay within a single submission, such as an optimization algorithm inside the submitted code, not across multiple official attempts based on prior attempt results.
- Do not invent fallback experiments or substitute another submission unless the agent explicitly requested that fallback. For example, do not scan other submissions and fall back to a higher-scored valid submission when the assigned instruction fails or performs poorly.
- Any `/execute_action{{...}}` strings that appear in task specs, reviews, or historical notes are for in-station agents, not for you.

Safety guidelines
- If the agent requests {restricted_request_kinds}, or scanning files outside the allowed Research Center storage/read-only surfaces, reject the request in `storage/report/{eval_id}.md` without running an experiment.
- You may run C/C++ code when appropriate: emit a self-contained C source file from Python, compile it into a shared library with `gcc -O3 -fopenmp -shared -fPIC` and fall back to non-OpenMP if needed, then call it from Python with `ctypes`. Keep the C block self-contained instead of patching C fragments by exact string match.

Sage/CAS runtime notes
- Before `import sage.all`, create this per-evaluation Sage/CAS workspace and its subdirectories if they do not exist: `{sage_workspace}`.
- Use these exact paths before importing Sage:
  ```python
  import os
  for path in ["{sage_home}", "{sage_dotdir}", "{sage_xdg_cache}"]:
      os.makedirs(path, exist_ok=True)
  os.environ["HOME"] = "{sage_home}"
  os.environ["DOT_SAGE"] = "{sage_dotdir}"
  os.environ["SAGE_DOTDIR"] = "{sage_dotdir}"
  os.environ["XDG_CACHE_HOME"] = "{sage_xdg_cache}"
  ```
- Do not put Sage/CAS cache or workspace directories under `storage/{lineage}` or `storage/{lineage}/data`. If Sage produces durable artifacts that matter for future submissions, copy only those selected files into `storage/{lineage}/data/` and document them in `storage/{lineage}/CODER.md`.
- Record reusable Sage conventions and important durable artifacts in `storage/{lineage}/CODER.md`, but do not record per-evaluation cache directories as lineage memory. Keep mutable Sage cache directories per-evaluation to avoid concurrent cache races.
- Run expensive Sage calls such as `solve`, `ideal.dimension`, `variety(ring=QQ)`, `gens`, `integral_points`, and Heegner routines in fresh subprocesses with parent-enforced wall-clock timeouts; do not rely on `signal.alarm` alone.
- Do not fork a Python child after importing Sage in the parent. Start a fresh Python/Sage subprocess that imports Sage inside the subprocess, because fork-after-Sage can deadlock or hang differently in the official evaluator than in local probes.
- Sage/Singular wrapper calls can fail or hang, and local probe behavior is not authoritative unless it runs the same submitted snapshot under the same timeout wrapper. Use exact manual elimination, direct Singular subprocesses, lower-level Singular dimension hooks, or scoped fallbacks when needed.
- Exact-verify every candidate against the original equations. Treat Sage timeouts, skipped branches, or partial symbolic solver outputs as inconclusive, not as proofs of absence.

Log polling and wait procedure
- The system clears the stdout and stderr logs at the start of each attempt.
- While an attempt is active, poll both logs using `sleep {{time}}` with capped backoff intervals of 30s, 60s, 120s, 240s, 480s, then 600s for all later polls.
- When a new attempt starts, first run `sleep 30`, then inspect both logs, then continue with the next interval in that sequence until the attempt finishes.
- Sleep may be capped at 30 seconds. Before and after each sleep, print the current time and the cumulative elapsed time so you can verify the actual wait duration before polling the logs.
- The system appends a clear completion footer when an attempt has finished and it is safe to either try again or finalize. Be patient and wait for that footer before taking the next step.
- The system also provides the latest attempt status, primary score, and secondary metrics after completion. Treat `ATTEMPT_STATUS: completed` as the run-completion signal; the score may legitimately be `n.a.` for diagnostic or non-scorable work.

Final report requirements
- Do not write the final Coder Report until you have inspected the final stdout, final stderr, final attempt status, final primary score, and final secondary metrics.
- Agents cannot see stdout/stderr in completion notifications or review. If stdout/stderr contain important results, tables, data, or diagnostics requested by the agent, summarize them in the Coder Report. Do not rely on stdout/stderr alone.
- When the agent instruction asks for evidence, a diagnosis, or a direct comparison, answer it explicitly in the Coder Report. Include the decisive numeric evidence and its interpretation in the report itself; JSON, CSV, stdout, and other local artifacts are supplementary and must not be the only place the answer appears. If the requested evidence was not measured, say so clearly rather than implying a conclusion.
- Always write a Coder Report, even if the task is blocked, partially implemented, or unsuccessful.
- If the main scientific goal was not achieved but the attempt completed faithfully, run a bounded lightweight diagnosis of the final result and artifacts when useful. Focus on scientifically useful interpretation beyond the score/log headline, such as promising partial objects, near misses, obstruction patterns, reusable diagnostics, or why the attempt failed. This should be done proactively even if the agent did not ask for it, but it must remain quick, must not start a new official attempt, must not change the submitted result, and must not become a separate research experiment.
- If the requested implementation or computation timed out, was interrupted, or was materially cut short, estimate how much total additional compute would likely be needed to finish the requested work instead of reporting only `timeout`. Base the estimate on observed phase timings, completed versus incomplete instances, progress indicators, resource use, and the remaining requested coverage. Give a bounded estimate or explain why no reliable estimate is possible, and make the requested-versus-actual compute gap explicit so the agent can decide how many further evaluations are justified.
- If the code is implemented faithfully and `ATTEMPT_STATUS: completed` appears with no exception or material mismatch with the agent's instruction, even if the score is `n.a.` or poor, it is usually sufficient to return the result to the agent and let the agent decide the next step.

Reporting constraints for multiple experiments
- If the agent asks for multiple experiments, such as trying method A then method B, or sweeping hyperparameters, only execute the first experiment.
- In that case, state clearly in the Coder Report that you are only allowed to run one experiment per submission, and that multiple experiments should be broken into multiple submissions.

Workflow
1. Read the research task specification.
2. Read `storage/{lineage}/CODER.md` if it exists.
3. Read the agent instruction carefully.
{workflow_inspection_step}
5. Implement the requested method.
6. Write or update `storage/submission/{eval_id}.py` and any helper files you need.
7. Start an execution attempt using:
   `bash submit_eval.sh {eval_id}`
8. Poll `storage/stdout/{eval_id}.log` and `storage/stderr/{eval_id}.log` until the explicit completion footer appears, using `sleep {{time}}` with intervals of 30s, 60s, 120s, 240s, 480s, and then 600s repeatedly.
9. Retry only after an exception, rejection, validation/malformed-output problem, or a specific implementation error that you corrected. If the attempt timed out computationally, write the report instead of resubmitting.
10. Post-experiment diagnosis: after the final official attempt has completed and no debugging retry is needed, perform the bounded lightweight diagnosis described in the final report requirements when useful.
11. When finished, write `storage/report/{eval_id}.md`.
12. At the end, update `storage/{lineage}/CODER.md` with reusable libraries, important file locations, special instructions, pitfalls, and any context that will matter for future submissions from this lineage. Remove stale or misleading old notes if needed.

Required format for `storage/report/{eval_id}.md`

# Coder Report

## Summary
What you implemented and the final outcome.

## Final Result
Final primary score, final secondary metrics, and brief interpretation of important stdout/stderr information.
Include important stdout information or requested raw data here when relevant.
State the total compute invested: official wall-clock time, configured attempt budget, CPU/GPU resources when available, and the number or scope of major attempted instances. State whether the execution was faithful to the agent's requested compute budget and experimental coverage. If it was not faithful, give the requested-versus-actual comparison and the reason for the deviation.

## Files Changed
List major file changes in a few lines. Do not include a raw diff.

## Implementation Details
Briefly sketch what the implementation does.
If the algorithm has major hyperparameters, list the important ones and the final values used.
If there are no meaningful hyperparameters, you may keep this section short.

## Faithfulness And Confidence
State whether the instruction was implemented faithfully, partially faithfully, or not faithfully, and give a brief confidence estimate.

## Major Deviations
List any material deviations, ambiguity resolutions, environment limitations, or missing dependencies.

## Post-Experiment Diagnosis
State what lightweight post-run diagnosis you performed and what it revealed. If none was useful or applicable, state `None`.
For optimization or search runs, state whether the search appeared converged, non-converged, or inconclusive, and whether continuing the same search process in a later evaluation appears promising. Cite concrete evidence such as late-run gains, termination reason, budget limits, or variation across runs.
Otherwise, keep this section to plain observations and do not give broader recommendations.

## Miscellaneous
State any unexpected or interesting result that may be worth mentioning, especially observations that could deserve follow-up investigation even if the agent did not ask for them. Keep this neutral and do not give recommendations.
Put any additional comments here that do not fit the sections above.
If the agent instruction asked direct questions, answer them here.
If there is nothing additional to say, this section may be short or left effectively blank.

## Final Status
State one of: `success`, `partial`, `failed`, `blocked`.

Use the context below.

=== RESEARCH TASK SPEC ===
{task_spec}

=== LINEAGE MEMORY: storage/{lineage}/CODER.md ===
{coder_md or "(empty)"}

=== MOST RECENT 5 COMPLETED EVALUATIONS FROM THIS LINEAGE ===
{recent_attempts}

=== AGENT INSTRUCTION ===
{eval_data.get("instruction", "")}
"""

    def launch_for_evaluation(self, eval_data: Dict[str, Any]) -> bool:
        return self.launch_cli_job(str(eval_data.get("id")))

    def _load_cli_job_state(self, job_id: str) -> CliJobState:
        eval_data = self.eval_manager.get_evaluation(job_id) or {}
        coder = eval_data.get("coder", {}) or {}
        top_status = str(eval_data.get("status", "")).strip().lower()
        worker_status = str(coder.get("status", "")).strip().lower()
        return CliJobState(
            backend=str(coder.get("backend") or constants.RESEARCH_CODER_BACKEND).lower(),
            spawn_count=int(coder.get("spawn_count", 0)),
            resume_count=int(coder.get("resume_count", 0)),
            max_spawns=int(coder.get("max_spawns", constants.RESEARCH_CODER_MAX_SPAWNS)),
            max_resumes=int(coder.get("max_resumes", constants.RESEARCH_CODER_MAX_RESUMES)),
            resume_token=str(coder.get("resume_token") or "").strip() or None,
            next_resume_timestamp=coder.get("next_resume_timestamp"),
            fresh_launch_eligible=top_status == "queued",
            resume_launch_eligible=(
                top_status == "running" and worker_status in {"pending_resume", "resuming"}
            ),
        )

    def _claim_cli_job_launch(
        self,
        job_id: str,
        session_id: str,
        decision: CliJobLaunchDecision,
    ) -> Optional[Dict[str, Any]]:
        eval_data = self.eval_manager.get_evaluation(job_id) or {}
        coder = eval_data.get("coder", {}) or {}
        previous_failure_reason = str(coder.get("last_error") or "").strip()
        claimed = self.eval_manager.claim_coder_launch(
            job_id,
            session_id=session_id,
            backend=str(coder.get("backend") or constants.RESEARCH_CODER_BACKEND).lower(),
            model_name=coder.get("model_name"),
            preserve_started_timestamp=decision.is_resume,
            substate="resuming" if decision.is_resume else "coder_running",
            increment_spawn_count=not decision.is_resume,
            increment_resume_count=decision.is_resume,
        )
        if claimed is not None:
            claimed["_previous_coder_failure_reason"] = previous_failure_reason
        return claimed

    def _build_cli_job_launch_spec(
        self,
        job_id: str,
        session_id: str,
        decision: CliJobLaunchDecision,
        claimed: Dict[str, Any],
    ) -> CliJobLaunchSpec:
        coder = claimed.get("coder", {}) or {}
        backend = str(coder.get("backend") or constants.RESEARCH_CODER_BACKEND).lower()
        lineage = str(claimed.get("lineage", "unknown")).lower()
        ensure_lineage_storage(self.paths, lineage)
        self._ensure_lineage_tmp_storage(lineage)
        env = self._build_runtime_env()
        if backend == "codex":
            apply_codex_proxy_overrides(env)
        if bool(getattr(constants, "RESEARCH_SEED_BANK_ENABLED", False)):
            env["STATION_LINEAGE"] = lineage
            env["STATION_ACCESS_PHASE"] = get_coder_access_phase(claimed)

        audit_status = str((claimed.get("audit", {}) or {}).get("status", "")).lower()
        previous_session_dirs = (
            self._find_previous_session_dirs(job_id, session_id)
            if not decision.is_resume and (decision.spawn_count > 1 or audit_status == "repairing")
            else []
        )
        previous_failure_reason = str(claimed.get("_previous_coder_failure_reason") or "").strip()
        prompt = (
            self._build_resume_prompt(claimed)
            if decision.is_resume
            else self._build_prompt(
                claimed,
                previous_session_dirs=previous_session_dirs,
                previous_failure_reason=previous_failure_reason,
            )
        )
        return CliJobLaunchSpec(
            executable=self._detect_backend_executable(backend, env),
            run_dir=os.path.abspath(os.path.join(self.paths.coder_sessions_dir, session_id)),
            backend=backend,
            model_name=coder.get("model_name"),
            workspace_root=os.path.abspath(self.paths.research_root),
            storage_root=self.paths.storage_root,
            prompt=prompt,
            env=env,
            extra_allowed_roots=tuple(self._extra_allowed_roots()),
            network_access=is_coder_internet_allowed(claimed, constants),
        )

    def _before_cli_job_process_launch(self, job_id, decision, prepared, spec):
        return self._write_debug_session_header(
            eval_id=job_id,
            session_id=os.path.basename(spec.run_dir),
            backend=spec.backend,
            command=prepared.command,
            prompt=spec.prompt,
            transcript_path=prepared.transcript_path,
            transcript_format=prepared.transcript_format,
            last_message_path=prepared.last_message_path,
        )

    def _make_active_cli_job_session(self, job_id, base_session, decision, claimed, launch_metadata):
        debug_dir, dialogue_log_path = launch_metadata
        return ActiveCoderSession(
            **vars(base_session),
            eval_id=job_id,
            report_path=os.path.abspath(os.path.join(self.paths.reports_dir, f"{job_id}.md")),
            debug_dir=debug_dir,
            dialogue_log_path=dialogue_log_path,
        )

    def _mark_cli_job_pid(self, job_id: str, pid: int) -> None:
        self.eval_manager.update_evaluation(
            job_id,
            lambda record: record.setdefault("coder", {}).update({"active_pid": pid}),
        )

    def _on_cli_job_started(self, job_id, session, decision):
        self._push_log_event(
            "research_coder_started",
            {
                "eval_id": job_id,
                "session_id": session.session_id,
                "backend": session.backend,
                "pid": session.process.pid,
                "launch_mode": decision.mode,
            },
        )
        print(
            f"ResearchCoderManager: Started coder for evaluation {job_id} "
            f"(session={session.session_id}, pid={session.process.pid}, mode={decision.mode})"
        )

    def _on_cli_job_resume_deferred(self, job_id, state, remaining_seconds):
        self._push_log_event(
            "research_coder_resume_deferred",
            {
                "eval_id": job_id,
                "remaining_seconds": remaining_seconds,
                "next_resume_timestamp": state.next_resume_timestamp,
            },
        )

    def _parse_final_status(self, report_text: str) -> Optional[str]:
        match = re.search(r"## Final Status\s+`?([A-Za-z]+)`?", report_text, flags=re.IGNORECASE)
        if not match:
            return None
        status = match.group(1).strip().lower()
        if status in {"success", "partial", "failed", "blocked"}:
            return status
        return None

    def _latest_attempt_is_finalized(self, eval_data: Dict[str, Any]) -> bool:
        attempts = eval_data.get("attempts") or []
        if not attempts:
            return True
        latest_attempt = attempts[-1] or {}
        latest_attempt_status = str(latest_attempt.get("status", "")).strip().lower()
        if latest_attempt_status in {"queued", "running"}:
            return False
        return True

    def _finalize_pending_reports_if_ready(self):
        for eval_id, payload in list(self.pending_report_finalizations.items()):
            eval_data = self.eval_manager.get_evaluation(eval_id) or {}
            if not self._latest_attempt_is_finalized(eval_data):
                continue
            if self._audit_is_pending(eval_data):
                self._begin_audit(
                    eval_id,
                    session_id=payload.get("session_id"),
                    returncode=payload.get("returncode"),
                    resume_token=payload.get("resume_token"),
                )
            else:
                self.eval_manager.finalize_evaluation(
                    eval_id,
                    payload.get("report_text") or "",
                    final_status=payload.get("final_status"),
                )
                self._push_log_event(
                    "research_coder_completed",
                    {
                        "eval_id": eval_id,
                        "session_id": payload.get("session_id"),
                        "returncode": payload.get("returncode"),
                        "deferred_until_attempt_complete": True,
                    },
                )
            self.pending_report_finalizations.pop(eval_id, None)

    def _audit_is_pending(self, eval_data: Dict[str, Any]) -> bool:
        if not bool(getattr(constants, "RESEARCH_CODER_AUDIT_ENABLED", True)):
            return False
        audit = eval_data.get("audit", {}) or {}
        return (
            str(audit.get("status", "not_started")).lower() not in {"pass", "blocked"}
            and not audit.get("active")
        )

    def _extract_session_resume_token(self, session: ActiveCoderSession) -> Optional[str]:
        try:
            backend_runner = self._get_cli_job_backend(session.backend)
            return backend_runner.extract_resume_token(session.transcript_path)
        except Exception:
            return None

    def _begin_audit(
        self,
        eval_id: str,
        *,
        session_id: Optional[str],
        returncode: Optional[Any],
        resume_token: Optional[str],
    ) -> None:
        self._append_archived_audits(eval_id)
        self.eval_manager.update_evaluation(eval_id, lambda record: record.setdefault("coder", {}).update({
            "active": False,
            "active_pid": None,
            "status": "audit_running",
            "completed_timestamp": time.time(),
            "exit_code": returncode,
            "resume_token": resume_token,
        }))
        self.eval_manager.update_evaluation(
            eval_id,
            lambda record: record.setdefault("audit", {}).update({"status": "queued", "active": False}),
        )
        try:
            self.audit_manager.launch_for_evaluation(self.eval_manager.get_evaluation(eval_id) or {})
        except Exception as exc:
            self.eval_manager.update_evaluation(eval_id, lambda record: record.setdefault("audit", {}).update({
                "status": "retry",
                "active": False,
                "active_pid": None,
                "last_error": f"Auditor launch failed: {exc}",
            }))
        self._push_log_event(
            "research_coder_audit_started",
            {"eval_id": eval_id, "session_id": session_id},
        )

    def _cli_job_missing_report_reason(self, job_id: str, session: ActiveCoderSession) -> str:
        return f"Coder exited without producing storage/report/{job_id}.md"

    def _on_cli_job_process_exited(self, job_id, session, returncode):
        self._write_debug_session_footer(session, returncode)
        print(
            f"ResearchCoderManager: Coder process exited for evaluation {job_id} "
            f"(session={session.session_id}, returncode={returncode})"
        )

    def _cli_job_completion_ready(self, job_id: str, session: ActiveCoderSession) -> bool:
        if not file_io_utils.file_exists(session.report_path):
            return False
        eval_data = self.eval_manager.get_evaluation(job_id) or {}
        audit = eval_data.get("audit", {}) or {}
        if str(audit.get("status", "")).lower() != "repairing":
            return True
        baseline = int(audit.get("repair_report_baseline_mtime_ns", 0) or 0)
        if not baseline:
            return True
        try:
            return os.stat(session.report_path).st_mtime_ns > baseline
        except OSError:
            return False

    def _on_cli_job_completed(self, job_id, session, returncode):
        report_text = file_io_utils.load_text(session.report_path) or ""
        final_status = self._parse_final_status(report_text)
        eval_data = self.eval_manager.get_evaluation(job_id) or {}
        resume_token = self._extract_session_resume_token(session)
        if not self._latest_attempt_is_finalized(eval_data):
            self.eval_manager.set_coder_exited(
                job_id,
                exit_code=returncode,
                keep_running=True,
                running_substate="attempt_running",
            )
            if resume_token:
                self.eval_manager.update_evaluation(
                    job_id,
                    lambda record: record.setdefault("coder", {}).update({"resume_token": resume_token}),
                )
            self.pending_report_finalizations[job_id] = {
                "report_text": report_text,
                "final_status": final_status,
                "session_id": session.session_id,
                "returncode": returncode,
                "resume_token": resume_token,
            }
            return
        if self._audit_is_pending(eval_data):
            self._begin_audit(
                job_id,
                session_id=session.session_id,
                returncode=returncode,
                resume_token=resume_token,
            )
            return
        if self._latest_attempt_is_finalized(eval_data):
            self.eval_manager.finalize_evaluation(job_id, report_text, final_status=final_status)
            self._push_log_event(
                "research_coder_completed",
                {"eval_id": job_id, "session_id": session.session_id, "returncode": returncode},
            )
            return

    def _append_audit_report(self, eval_id: str, report_text: str, verdict: str) -> str:
        report_path = os.path.join(self.paths.reports_dir, f"{eval_id}.md")
        existing = file_io_utils.load_text(report_path) if file_io_utils.file_exists(report_path) else ""
        heading = "## Independent Audit" if verdict == "pass" else "## Independent Audit Failure"
        section = f"\n\n{heading}\n\n{report_text.strip()}\n"
        combined = existing.rstrip() + section
        file_io_utils.save_text(combined, report_path)
        return combined

    def _append_archived_audits(self, eval_id: str) -> None:
        report_path = os.path.join(self.paths.reports_dir, f"{eval_id}.md")
        existing = file_io_utils.load_text(report_path) if file_io_utils.file_exists(report_path) else ""
        names = sorted(
            name for name in os.listdir(self.paths.audit_dir)
            if name.startswith(f"{eval_id}.round_") and name.endswith(".md")
        )
        for name in names:
            text = file_io_utils.load_text(os.path.join(self.paths.audit_dir, name)) or ""
            if text.strip() and text.strip() not in existing:
                existing = existing.rstrip() + f"\n\n## Independent Audit ({name[:-3]})\n\n{text.strip()}\n"
        if existing:
            file_io_utils.save_text(existing, report_path)

    def _handle_audit_verdict(self, eval_id: str, verdict: str, report_text: str, session_id: Optional[str]):
        eval_data = self.eval_manager.get_evaluation(eval_id) or {}
        if verdict == "pass":
            combined = self._append_audit_report(eval_id, report_text, verdict)
            self.eval_manager.finalize_evaluation(
                eval_id,
                combined,
                final_status=self._parse_final_status(combined),
                notification_report_mode="latest_audit",
            )
            self._push_log_event("research_coder_audit_passed", {"eval_id": eval_id, "session_id": session_id})
            return
        if verdict == "fail":
            audit = eval_data.get("audit", {}) or {}
            round_number = int(audit.get("repair_round", 0) or 0)
            combined = self._append_audit_report(eval_id, report_text, verdict)
            max_rounds = max(1, int(getattr(constants, "RESEARCH_CODER_AUDIT_MAX_ROUNDS", 2) or 1))
            current_audit_round = round_number + 1
            if current_audit_round < max_rounds:
                file_io_utils.save_text(
                    report_text.strip() + "\n",
                    os.path.join(self.paths.audit_dir, f"{eval_id}.round_{round_number}.md"),
                )
                report_path = os.path.join(self.paths.reports_dir, f"{eval_id}.md")
                try:
                    repair_report_baseline_mtime_ns = os.stat(report_path).st_mtime_ns
                except OSError:
                    repair_report_baseline_mtime_ns = 0
                coder_state = eval_data.get("coder", {}) or {}
                backend = str(coder_state.get("backend") or constants.RESEARCH_CODER_BACKEND).lower()
                resume_token = str(coder_state.get("resume_token") or "").strip()
                try:
                    can_resume = bool(resume_token and self._get_cli_job_backend(backend).supports_resume)
                except Exception:
                    can_resume = False
                attempts_used = len(eval_data.get("attempts") or [])
                try:
                    configured_max_attempts = int(
                        coder_state.get("max_attempts", constants.RESEARCH_CODER_MAX_ATTEMPTS)
                    )
                except (TypeError, ValueError):
                    configured_max_attempts = int(constants.RESEARCH_CODER_MAX_ATTEMPTS)
                # A failed non-final audit is a new repair opportunity. Preserve
                # the normal budget, but guarantee one official attempt when
                # the coder had already consumed the prior budget.
                repair_max_attempts = max(configured_max_attempts, attempts_used + 1)
                def mutator(record):
                    coder = record.setdefault("coder", {})
                    coder.update({
                        "active": can_resume,
                        "active_pid": None,
                        "status": "pending_resume" if can_resume else "queued",
                        "spawn_count": 0,
                        "resume_count": 0,
                        "next_resume_timestamp": None,
                        "resume_delay_seconds": None,
                        "started_timestamp": None,
                        "completed_timestamp": None,
                        "exit_code": None,
                        "failure_category": None,
                        "resume_token": resume_token if can_resume else None,
                        "last_error": "Independent audit failed; repair required.",
                        "max_attempts": repair_max_attempts,
                    })
                    record["status"] = "running" if can_resume else "queued"
                    audit_record = record.setdefault("audit", {})
                    audit_record.update({
                        "status": "repairing",
                        "active": False,
                        "active_pid": None,
                        "spawn_count": 0,
                        "resume_count": 0,
                        "next_resume_timestamp": None,
                        "resume_delay_seconds": None,
                        "resume_token": None,
                        "failure_category": None,
                        "session_id": None,
                        "repair_round": round_number + 1,
                        "last_verdict": "fail",
                        "repair_report_baseline_mtime_ns": repair_report_baseline_mtime_ns,
                    })
                self.eval_manager.update_evaluation(eval_id, mutator)
                self._push_log_event("research_coder_audit_repair_requested", {
                    "eval_id": eval_id,
                    "repair_round": round_number + 1,
                    "resume_previous_session": can_resume,
                    "attempts_used": attempts_used,
                    "max_attempts": repair_max_attempts,
                    "attempts_remaining": max(0, repair_max_attempts - attempts_used),
                })
                return
            self.eval_manager.finalize_evaluation(
                eval_id,
                combined,
                final_status="partial",
                notification_report_mode="latest_audit",
            )
            self._push_log_event("research_coder_audit_exhausted", {"eval_id": eval_id, "repair_round": round_number})
            return
        # Infrastructure failure: leave the evaluation safely visible for manual recovery.
        reason = report_text or "Auditor failed before producing a verdict."
        self.eval_manager.update_evaluation(eval_id, lambda r: r.setdefault("audit", {}).update({"status": "blocked", "active": False, "last_error": reason}))
        self._pause_station(f"Research auditor blocked evaluation {eval_id}: {reason}")

    def _on_cli_job_no_report_exit(self, job_id, session, returncode):
        self.eval_manager.set_coder_exited(job_id, exit_code=returncode, keep_running=False)

    def _schedule_cli_job_resume(self, job_id, session, failure):
        def mutator(record: Dict[str, Any]):
            coder = record.setdefault("coder", {})
            coder["active"] = True
            coder["active_pid"] = None
            coder["completed_timestamp"] = time.time()
            coder["exit_code"] = failure.returncode
            coder["failure_category"] = failure.category
            coder["resume_token"] = failure.resume_token or coder.get("resume_token")
            coder["resume_count"] = failure.resume_count
            coder["resume_delay_seconds"] = failure.delay_seconds
            coder["next_resume_timestamp"] = failure.next_resume_timestamp
            coder["last_error"] = failure.reason
            coder["status"] = "pending_resume"
            record["status"] = "running"

        self.eval_manager.update_evaluation(job_id, mutator)
        self._push_log_event(
            "research_coder_retryable_infra_failure",
            {
                "eval_id": job_id,
                "session_id": session.session_id,
                "returncode": failure.returncode,
                "reason": failure.reason,
                "resume_delay_seconds": failure.delay_seconds,
                "next_resume_timestamp": failure.next_resume_timestamp,
            },
        )
        delay_suffix = f" after {failure.delay_seconds}s" if failure.delay_seconds > 0 else ""
        print(
            f"ResearchCoderManager: Scheduling same-session resume for evaluation {job_id} "
            f"(session={session.session_id}, reason={failure.reason}){delay_suffix}"
        )

    def _schedule_cli_job_fresh_attempt(self, job_id, session, failure):
        def mutator(record: Dict[str, Any]):
            coder = record.setdefault("coder", {})
            coder["active"] = False
            coder["active_pid"] = None
            coder["status"] = "retry"
            coder["failure_category"] = failure.category
            coder["resume_token"] = None
            coder["resume_count"] = 0
            coder["resume_delay_seconds"] = 0
            coder["next_resume_timestamp"] = None
            coder["last_error"] = failure.reason
            record["status"] = "queued"

        self.eval_manager.update_evaluation(job_id, mutator)
        self._push_log_event(
            "research_coder_respawn_scheduled",
            {
                "eval_id": job_id,
                "session_id": getattr(session, "session_id", None),
                "returncode": failure.returncode,
            },
        )

    def _on_cli_job_attempts_exhausted(
        self,
        job_id,
        session,
        failure,
    ):
        state = self._load_cli_job_state(job_id)
        if failure.resume_budget_exhausted:
            reason = (
                f"Research Center coder for evaluation {job_id} exceeded the configured resume limit "
                f"of {state.max_resumes} without producing storage/report/{job_id}.md. "
                "Manual intervention is required."
            )
        else:
            reason = (
                f"Research Center coder for evaluation {job_id} exited without a report after "
                f"{state.spawn_count} total spawns. Manual intervention is required."
            )

        def mutator(record: Dict[str, Any]):
            coder = record.setdefault("coder", {})
            coder["active"] = False
            coder["active_pid"] = None
            coder["failure_category"] = (
                "resume_limit_exceeded"
                if failure.resume_budget_exhausted
                else failure.category
            )
            coder["resume_token"] = failure.resume_token or None
            coder["last_error"] = reason
            if failure.resume_budget_exhausted:
                coder["status"] = "blocked"
            record["status"] = "blocked"

        self.eval_manager.update_evaluation(job_id, mutator)
        self._pause_station(reason)

    def poll(self):
        self.poll_cli_jobs()
        if not bool(getattr(constants, "RESEARCH_CODER_AUDIT_ENABLED", True)):
            return
        self.audit_manager.poll_cli_jobs()
        for eval_id in self.eval_manager.get_running_instruction_eval_ids():
            if eval_id in self.audit_manager.active_sessions:
                continue
            record = self.eval_manager.get_evaluation(eval_id) or {}
            audit = record.get("audit", {}) or {}
            if str(audit.get("status", "")).lower() in {"queued", "retry", "pending_resume"} and not audit.get("active"):
                try:
                    self.audit_manager.launch_for_evaluation(record)
                except Exception as exc:
                    self.eval_manager.update_evaluation(eval_id, lambda r: r.setdefault("audit", {}).update({
                        "status": "retry", "active": False, "last_error": f"Auditor launch failed: {exc}",
                    }))

    def _before_cli_job_poll(self) -> None:
        self._check_session_timeouts()

    def _after_cli_job_watchdogs(self) -> None:
        self._finalize_pending_reports_if_ready()

    def _check_session_timeouts(self):
        report_deadline_seconds = int(getattr(constants, "RESEARCH_CODER_TIMEOUT_SECONDS", 0) or 0)
        if report_deadline_seconds <= 0:
            return
        grace_period_seconds = 15 * 60
        actual_timeout_seconds = report_deadline_seconds + grace_period_seconds

        now = time.time()
        for eval_id, session in list(self.active_sessions.items()):
            eval_data = self.eval_manager.get_evaluation(eval_id) or {}
            coder = eval_data.get("coder") or {}
            started_timestamp = float(coder.get("started_timestamp") or 0)
            if started_timestamp <= 0:
                continue
            if now - started_timestamp < actual_timeout_seconds:
                continue

            self._terminate_cli_job_process(session, force=False)
            reason = (
                f"Research Center coder for evaluation {eval_id} exceeded the configured report deadline "
                f"of {report_deadline_seconds} seconds plus a 15-minute grace period."
            )
            self._append_dialogue_log(session.dialogue_log_path, "TIMEOUT", reason)
            self._push_log_event(
                "research_coder_timeout",
                {
                    "eval_id": eval_id,
                    "timeout_seconds": actual_timeout_seconds,
                    "deadline_seconds": report_deadline_seconds,
                    "grace_seconds": grace_period_seconds,
                },
            )

    def _cli_job_idle_timeout_reason(self, job_id, session, idle_seconds, timeout_seconds):
        return (
            f"Codex CLI transcript for evaluation {job_id} did not grow for "
            f"{idle_seconds} seconds, exceeding the configured CLI worker "
            f"transcript idle timeout of {timeout_seconds} seconds."
        )

    def _on_cli_job_idle_timeout(self, job_id, session, reason, idle_seconds, timeout_seconds):
        self._append_dialogue_log(session.dialogue_log_path, "CODEX TRANSCRIPT IDLE TIMEOUT", reason)
        self._push_log_event(
            "research_coder_codex_transcript_idle_timeout",
            {
                "eval_id": job_id,
                "session_id": session.session_id,
                "idle_seconds": idle_seconds,
                "timeout_seconds": timeout_seconds,
            },
        )

        def mutator(record: Dict[str, Any]):
            coder = record.setdefault("coder", {})
            coder["failure_category"] = "codex_transcript_idle_timeout"
            coder["last_error"] = reason

        self.eval_manager.update_evaluation(job_id, mutator)

    def shutdown(self, reason: str) -> int:
        for session in list(self.active_sessions.values()):
            self._append_dialogue_log(session.dialogue_log_path, "SHUTDOWN", reason)
        self.stop_active_cli_jobs(force=False)
        self.audit_manager.stop_active_cli_jobs(force=False)

        from .restart_evaluations import requeue_instruction_evaluations

        return requeue_instruction_evaluations(
            reason=reason,
            kill_running_coders=True,
            eval_manager=self.eval_manager,
            paths=self.paths,
        )
