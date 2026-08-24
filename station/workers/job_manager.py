"""Shared subprocess, resume, retry, watchdog, and interruption loop."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from station import file_io_utils
from station.workers.cli import (
    CodexCliWorkerBackend,
    check_cli_worker_transcript_growth_timeout,
    get_cli_worker_backend,
)


@dataclass(frozen=True)
class CliJobState:
    backend: str
    spawn_count: int
    resume_count: int
    max_spawns: int
    max_resumes: int
    resume_token: Optional[str]
    next_resume_timestamp: Optional[float]
    fresh_launch_eligible: bool
    resume_launch_eligible: bool


@dataclass(frozen=True)
class CliJobLaunchDecision:
    mode: str
    spawn_count: int
    resume_count: int
    resume_token: Optional[str]

    @property
    def is_resume(self) -> bool:
        return self.mode == "resume"


@dataclass(frozen=True)
class CliJobLaunchSpec:
    executable: str
    run_dir: str
    backend: str
    model_name: Optional[str]
    workspace_root: str
    storage_root: str
    prompt: str
    env: Dict[str, str]
    extra_allowed_roots: tuple[str, ...] = ()
    network_access: bool = False


@dataclass(kw_only=True)
class CliJobSession:
    session_id: str
    run_dir: str
    backend: str
    transcript_format: str
    process: subprocess.Popen
    transcript_handle: Any
    stderr_handle: Any
    prompt_path: str
    command: List[str]
    transcript_path: str
    stderr_path: str
    last_message_path: Optional[str]
    last_transcript_size: int = 0
    last_transcript_growth_timestamp: float = 0.0
    transcript_idle_timeout_triggered: bool = False
    transcript_idle_timeout_reason: Optional[str] = None


@dataclass
class CliJobFailure:
    category: str
    is_retryable_infra: bool
    reason: str
    resume_token: Optional[str] = None
    resume_count: int = 0
    delay_seconds: int = 0
    next_resume_timestamp: Optional[float] = None
    returncode: Optional[int] = None
    resume_budget_exhausted: bool = False


class CliJobManager:
    """One shared subprocess/resume/fresh-attempt state machine."""
    fresh_attempt_after_resume_exhaustion = False

    def __init__(self) -> None:
        self.active_sessions: Dict[str, Any] = {}

    def has_active_sessions(self) -> bool:
        return bool(self.active_sessions)

    @staticmethod
    def _parse_resume_backoff_schedule(raw_schedule: Any) -> List[int]:
        if isinstance(raw_schedule, str):
            raw_values = [value for value in re.split(r"[,\s]+", raw_schedule.strip()) if value]
        else:
            try:
                raw_values = list(raw_schedule)
            except TypeError:
                raw_values = [raw_schedule] if isinstance(raw_schedule, (int, float)) else []
        delays: List[int] = []
        for value in raw_values:
            try:
                delays.append(max(0, int(float(value))))
            except (TypeError, ValueError):
                continue
        return delays

    @classmethod
    def get_resume_backoff_delay_seconds(cls, completed_resume_count: Any) -> int:
        schedule = cls._parse_resume_backoff_schedule(cls._resume_backoff_schedule())
        if not schedule:
            return 0
        try:
            resume_count = max(0, int(completed_resume_count or 0))
        except (TypeError, ValueError):
            resume_count = 0
        return schedule[min(resume_count, len(schedule) - 1)]

    @staticmethod
    def get_resume_backoff_remaining_seconds(worker_state, now: Optional[float] = None) -> float:
        raw_timestamp = (worker_state.next_resume_timestamp if isinstance(worker_state, CliJobState)
                         else (worker_state or {}).get("next_resume_timestamp"))
        try:
            next_resume_timestamp = float(raw_timestamp or 0)
        except (TypeError, ValueError):
            next_resume_timestamp = 0.0
        if next_resume_timestamp <= 0:
            return 0.0
        return max(0.0, next_resume_timestamp - (time.time() if now is None else float(now)))

    def _noop_cli_job_hook(self, *args):
        pass
    _before_cli_job_poll = _after_cli_job_watchdogs = _after_cli_job_poll = _noop_cli_job_hook
    _on_cli_job_process_exited = _on_cli_job_no_report_exit = _noop_cli_job_hook
    _on_cli_job_started = _on_cli_job_resume_deferred = _noop_cli_job_hook
    _before_cli_job_process_launch = _noop_cli_job_hook

    def _make_active_cli_job_session(self, job_id, base_session, decision, claimed, launch_metadata):
        return base_session

    def _get_cli_job_backend(self, backend):
        return get_cli_worker_backend(backend)

    def _format_cli_job_session_id(self, job_id, state, decision) -> str:
        return f"{state.backend}_{job_id}_spawn_{decision.spawn_count}_{uuid.uuid4().hex[:8]}"

    def _next_cli_job_launch(self, job_id: str, state: CliJobState) -> Optional[CliJobLaunchDecision]:
        if state.resume_launch_eligible:
            backend_runner = self._get_cli_job_backend(state.backend)
            resume_token = str(state.resume_token or "").strip()
            if resume_token and bool(getattr(backend_runner, "supports_resume", False)):
                if state.resume_count >= state.max_resumes:
                    if self.fresh_attempt_after_resume_exhaustion:
                        failure = CliJobFailure("resume_limit_exceeded", False,
                            "Same-session resume budget exhausted; starting a fresh attempt.", resume_token)
                        self._schedule_cli_job_fresh_attempt(job_id, None, failure)
                    else:
                        self._on_cli_job_attempts_exhausted(job_id, None, CliJobFailure(
                            "resume_limit_exceeded", False, "Same-session resume budget exhausted.",
                            resume_token, resume_budget_exhausted=True,
                        ))
                    return None
                remaining_seconds = self.get_resume_backoff_remaining_seconds(state)
                if remaining_seconds > 0:
                    self._on_cli_job_resume_deferred(job_id, state, remaining_seconds)
                    return None
                return CliJobLaunchDecision("resume", max(1, state.spawn_count),
                                            state.resume_count + 1, resume_token)
            if self.fresh_attempt_after_resume_exhaustion and state.spawn_count < state.max_spawns:
                failure = CliJobFailure("resume_unavailable", False,
                    "Same-session resume is unavailable; starting a fresh attempt.", resume_token or None)
                self._schedule_cli_job_fresh_attempt(job_id, None, failure)
                return None

        if not state.fresh_launch_eligible:
            return None
        if state.spawn_count >= state.max_spawns:
            failure = CliJobFailure("report_missing", False, "Fresh attempt budget exhausted.",
                                    state.resume_token)
            self._on_cli_job_attempts_exhausted(job_id, None, failure)
            return None
        return CliJobLaunchDecision("fresh", state.spawn_count + 1, 0, None)

    def launch_cli_job(self, job_id: str) -> bool:
        job_id = str(job_id)
        if job_id in self.active_sessions:
            return False
        state = self._load_cli_job_state(job_id)
        decision = self._next_cli_job_launch(job_id, state)
        if decision is None:
            return False

        session_id = self._format_cli_job_session_id(job_id, state, decision)
        claimed = self._claim_cli_job_launch(job_id, session_id, decision)
        if claimed is None:
            return False
        spec = self._build_cli_job_launch_spec(job_id, session_id, decision, claimed)
        prepared = self._prepare_cli_job_launch(spec, decision, self._get_cli_job_backend)
        launch_metadata = self._before_cli_job_process_launch(job_id, decision, prepared, spec)
        base_session = self._start_cli_job_process(session_id, spec, prepared)
        active_session = self._make_active_cli_job_session(
            job_id, base_session, decision, claimed, launch_metadata)
        try:
            self._mark_cli_job_pid(job_id, active_session.process.pid)
        except Exception:
            self._abort_cli_job_process(active_session)
            self._close_cli_job_session_handles(active_session)
            raise
        self.active_sessions[job_id] = active_session
        self._on_cli_job_started(job_id, active_session, decision)
        return True

    @staticmethod
    def _prepare_cli_job_launch(spec, decision, backend_getter=get_cli_worker_backend):
        backend_runner = backend_getter(spec.backend)
        kwargs = {
            "executable": spec.executable,
            "workspace_root": os.path.abspath(spec.workspace_root),
            "run_dir": os.path.abspath(spec.run_dir),
            "model_name": spec.model_name,
            "storage_root": spec.storage_root,
            "prompt": spec.prompt,
            "extra_allowed_roots": list(spec.extra_allowed_roots),
        }
        if isinstance(backend_runner, CodexCliWorkerBackend):
            kwargs["runtime_env"] = spec.env
        if spec.network_access:
            kwargs["network_access"] = True
        if decision.is_resume:
            kwargs["resume_token"] = str(decision.resume_token or "")
            return backend_runner.prepare_resume_launch(**kwargs)
        return backend_runner.prepare_launch(**kwargs)

    @staticmethod
    def _start_cli_job_process(session_id, spec, prepared, popen_factory=None):
        popen_factory = popen_factory or subprocess.Popen
        run_dir = os.path.abspath(spec.run_dir)
        file_io_utils.ensure_dir_exists(run_dir)
        prompt_path = os.path.join(run_dir, "prompt.txt")
        file_io_utils.save_text(spec.prompt, prompt_path)

        launch_env = dict(spec.env)
        for key, value in (prepared.env_overrides or {}).items():
            if value is None:
                launch_env.pop(key, None)
            else:
                launch_env[key] = value

        transcript_handle = open(prepared.transcript_path, "w", encoding="utf-8")
        stderr_handle = open(prepared.stderr_path, "w", encoding="utf-8")
        try:
            popen_kwargs = {
                "cwd": os.path.abspath(spec.workspace_root),
                "env": launch_env,
                "stdout": transcript_handle,
                "stderr": stderr_handle,
                "text": True,
                "start_new_session": True,
            }
            has_stdin = prepared.stdin_text is not None
            if has_stdin:
                popen_kwargs["stdin"] = subprocess.PIPE
            process = popen_factory(prepared.command, **popen_kwargs)
            if has_stdin:
                assert process.stdin is not None
                process.stdin.write(prepared.stdin_text)
                process.stdin.close()
        except Exception:
            transcript_handle.close()
            stderr_handle.close()
            raise

        return CliJobSession(
            session_id=session_id,
            run_dir=run_dir,
            backend=spec.backend,
            transcript_format=prepared.transcript_format,
            process=process,
            transcript_handle=transcript_handle,
            stderr_handle=stderr_handle,
            prompt_path=prompt_path,
            command=prepared.command,
            transcript_path=prepared.transcript_path,
            stderr_path=prepared.stderr_path,
            last_message_path=prepared.last_message_path,
            last_transcript_growth_timestamp=time.time(),
        )

    def poll_cli_jobs(self) -> None:
        self._before_cli_job_poll()
        self._check_cli_job_transcript_idle_timeouts()
        self._after_cli_job_watchdogs()
        finished: List[str] = []
        for job_id, session in list(self.active_sessions.items()):
            returncode = session.process.poll()
            if returncode is None:
                continue
            finished.append(job_id)
            self._close_cli_job_session_handles(session)
            self._on_cli_job_process_exited(job_id, session, returncode)

            if self._cli_job_completion_ready(job_id, session):
                self._on_cli_job_completed(job_id, session, returncode)
                continue

            self._on_cli_job_no_report_exit(job_id, session, returncode)
            state = self._load_cli_job_state(job_id)
            failure = self._classify_cli_job_failure(job_id, session)
            backend_runner = self._get_cli_job_backend(session.backend)
            resume_token = backend_runner.extract_resume_token(session.transcript_path)
            failure.resume_token = resume_token
            failure.returncode = returncode
            can_resume = (failure.is_retryable_infra and resume_token
                          and bool(getattr(backend_runner, "supports_resume", False)))
            if can_resume:
                resume_count = state.resume_count
                if self._transcript_has_normal_progress(session.transcript_path) and resume_count > 0:
                    resume_count = 0
                if resume_count < state.max_resumes:
                    failure.resume_count = resume_count
                    failure.delay_seconds = self.get_resume_backoff_delay_seconds(resume_count)
                    if failure.delay_seconds > 0:
                        failure.next_resume_timestamp = time.time() + failure.delay_seconds
                    self._schedule_cli_job_resume(job_id, session, failure)
                    continue
                if self.fresh_attempt_after_resume_exhaustion and state.spawn_count < state.max_spawns:
                    self._schedule_cli_job_fresh_attempt(job_id, session, failure)
                else:
                    failure.resume_budget_exhausted = True
                    self._on_cli_job_attempts_exhausted(job_id, session, failure)
                continue

            if state.spawn_count < state.max_spawns:
                self._schedule_cli_job_fresh_attempt(job_id, session, failure)
            else:
                self._on_cli_job_attempts_exhausted(job_id, session, failure)
        for job_id in finished:
            self.active_sessions.pop(job_id, None)
        self._after_cli_job_poll()

    def _classify_cli_job_failure(self, job_id: str, session: Any) -> CliJobFailure:
        reason = (
            session.transcript_idle_timeout_reason
            or self._cli_job_missing_report_reason(job_id, session)
        )
        return CliJobFailure(
            "infra_transient",
            True,
            reason,
        )

    def _cli_job_missing_report_reason(self, job_id: str, session: Any) -> str:
        return "CLI worker exited without producing its required completion artifact."

    @staticmethod
    def _read_transcript_events(transcript_path: str):
        if not transcript_path or not file_io_utils.file_exists(transcript_path):
            return
        try:
            transcript_text = file_io_utils.load_text(transcript_path) or ""
        except Exception:
            return
        for raw_line in transcript_text.splitlines():
            line = raw_line.strip()
            if not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

    @classmethod
    def _transcript_has_normal_progress(cls, transcript_path: str) -> bool:
        progress_since_last_failure = progress_before_latest_failure = False
        saw_failure_marker = in_failure_marker_run = False
        for payload in cls._read_transcript_events(transcript_path):
            event_type = str(payload.get("type", "")).strip().lower()
            if event_type in {"turn.failed", "error"}:
                if not in_failure_marker_run:
                    progress_before_latest_failure = progress_since_last_failure
                    progress_since_last_failure = False
                saw_failure_marker = True
                in_failure_marker_run = True
                continue
            in_failure_marker_run = False
            if event_type != "item.completed":
                continue
            item = payload.get("item")
            if isinstance(item, dict) and str(item.get("type", "")).strip().lower() in {
                "agent_message", "command_execution"}:
                progress_since_last_failure = True
        return progress_before_latest_failure if saw_failure_marker else progress_since_last_failure

    def _check_cli_job_transcript_idle_timeouts(self) -> None:
        for job_id, session in list(self.active_sessions.items()):
            if session.transcript_idle_timeout_triggered:
                continue
            result = check_cli_worker_transcript_growth_timeout(
                backend=session.backend,
                transcript_path=session.transcript_path,
                last_size=session.last_transcript_size,
                last_growth_timestamp=session.last_transcript_growth_timestamp,
            )
            if not result.applies:
                continue
            session.last_transcript_size = result.current_size
            session.last_transcript_growth_timestamp = result.last_growth_timestamp
            if not result.timed_out:
                continue
            reason = self._cli_job_idle_timeout_reason(
                job_id, session, int(result.idle_seconds), int(result.timeout_seconds))
            session.transcript_idle_timeout_triggered = True
            session.transcript_idle_timeout_reason = reason
            self._terminate_cli_job_process(session, force=False)
            self._on_cli_job_idle_timeout(
                job_id, session, reason, result.idle_seconds, result.timeout_seconds)

    def _terminate_cli_job_process(self, session: Any, *, force: bool = False) -> None:
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.killpg(session.process.pid, sig)
            return
        except Exception as exc:
            print(f"{type(self).__name__}: Failed to signal process group "
                  f"for session {session.session_id}: {exc}")
        try:
            (session.process.kill if force else session.process.terminate)()
        except Exception as exc:
            print(f"{type(self).__name__}: Failed to signal process "
                  f"for session {session.session_id}: {exc}")

    def _abort_cli_job_process(self, session: Any) -> None:
        if session.process.poll() is not None:
            return
        self._terminate_cli_job_process(session, force=False)
        try:
            session.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._terminate_cli_job_process(session, force=True)
            session.process.wait(timeout=5)
        except (AttributeError, TypeError):
            pass

    @staticmethod
    def _close_cli_job_session_handles(session: Any) -> None:
        for handle in (session.transcript_handle, session.stderr_handle):
            try:
                handle.close()
            except Exception:
                pass

    def stop_active_cli_jobs(self, *, force: bool = False, after_terminate=None) -> None:
        for job_id, session in list(self.active_sessions.items()):
            self._terminate_cli_job_process(session, force=force)
            if after_terminate:
                after_terminate(job_id, session)
            self._close_cli_job_session_handles(session)
        self.active_sessions.clear()
