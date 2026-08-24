# Copyright 2025 DualverseAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Startup recovery helpers for Research Center evaluations.

The new coder-based flow stores one YAML file per evaluation. On restart we:
- preserve terminal evaluations so notifications can be resent later
- requeue active instruction-level evaluations
- optionally terminate matching active coder processes owned by this station
"""

from __future__ import annotations

import os
import signal
import time
from typing import Iterable, Optional

from station import constants
from station import file_io_utils
from station import agent as agent_module
from station.eval_research import evaluation_index
from station.eval_research.evaluation_manager import (
    ACTIVE_EVALUATION_STATUSES,
    EvaluationManager,
    _normalize_evaluation_status,
)
from station.eval_research.runtime_paths import ensure_submit_runtime_layout
from station.workers.cli import get_cli_worker_backend


def _ensure_recovery_paths(paths=None):
    return paths or ensure_submit_runtime_layout()


def _pid_exists(pid: Optional[int]) -> bool:
    if pid in (None, "", 0):
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _read_proc_text(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            data = handle.read()
        return data.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _read_proc_environ(pid: int) -> str:
    return _read_proc_text(f"/proc/{pid}/environ")


def _pid_matches_station(pid: Optional[int], research_root: str, coder_sessions_dir: str) -> bool:
    if not _pid_exists(pid):
        return False

    pid_int = int(pid)
    try:
        proc_cwd = os.path.realpath(os.readlink(f"/proc/{pid_int}/cwd"))
    except Exception:
        proc_cwd = ""
    proc_cmdline = _read_proc_text(f"/proc/{pid_int}/cmdline")

    research_root = os.path.realpath(research_root)
    coder_sessions_dir = os.path.realpath(coder_sessions_dir)

    if proc_cwd == research_root:
        return True
    if research_root and research_root in proc_cmdline:
        return True
    if coder_sessions_dir and coder_sessions_dir in proc_cmdline:
        return True
    return False


def _wrapper_pid_matches_station(pid: Optional[int], research_root: str) -> bool:
    if not _pid_exists(pid):
        return False
    pid_int = int(pid)
    proc_cmdline = _read_proc_text(f"/proc/{pid_int}/cmdline")
    if "__sandbox_execution_wrapper__.py" not in proc_cmdline:
        return False
    proc_environ = _read_proc_environ(pid_int)
    expected = f"STATION_RESEARCH_ROOT={os.path.realpath(research_root)}"
    return expected in proc_environ


def _terminate_pid(pid: Optional[int], *, kill_process_group: bool = True) -> bool:
    if not _pid_exists(pid):
        return False

    pid_int = int(pid)
    terminated = False

    try:
        if kill_process_group:
            os.killpg(pid_int, signal.SIGTERM)
        else:
            os.kill(pid_int, signal.SIGTERM)
        terminated = True
    except Exception:
        try:
            os.kill(pid_int, signal.SIGTERM)
            terminated = True
        except Exception:
            return False

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if not _pid_exists(pid_int):
            return True
        time.sleep(0.1)

    try:
        if kill_process_group:
            os.killpg(pid_int, signal.SIGKILL)
        else:
            os.kill(pid_int, signal.SIGKILL)
        terminated = True
    except Exception as exc:
        print(f"restart_evaluations: Failed to send SIGKILL to pid {pid_int}: {exc}")
        try:
            os.kill(pid_int, signal.SIGKILL)
            terminated = True
        except Exception as fallback_exc:
            print(f"restart_evaluations: Fallback SIGKILL also failed for pid {pid_int}: {fallback_exc}")

    return terminated


def _terminate_station_wrapper_processes(research_root: str) -> int:
    started_at = time.perf_counter()
    killed = 0
    proc_root = "/proc"
    if not os.path.isdir(proc_root):
        return killed
    for name in os.listdir(proc_root):
        if not name.isdigit():
            continue
        pid = int(name)
        if not _wrapper_pid_matches_station(pid, research_root):
            continue
        if _terminate_pid(pid, kill_process_group=True):
            killed += 1
    print(
        "restart_evaluations: wrapper process scan/terminate "
        f"killed={killed} in {time.perf_counter() - started_at:.3f}s"
    )
    return killed


def _remove_run_requests(paths, eval_id: str) -> int:
    removed = 0
    if not os.path.isdir(paths.run_requests_dir):
        return removed
    prefix = f"{eval_id}_attempt_"
    suffix = ".yaml"
    for name in os.listdir(paths.run_requests_dir):
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        path = os.path.join(paths.run_requests_dir, name)
        try:
            os.remove(path)
            removed += 1
        except FileNotFoundError:
            continue
        except Exception as exc:
            print(f"restart_evaluations: Failed to remove stale run-request {path}: {exc}")
    return removed


def _is_no_report_terminal_requeueable(eval_data: dict) -> bool:
    coder = eval_data.get("coder", {}) or {}
    status = str(eval_data.get("status", "")).strip().lower()
    if status not in {"failed", "blocked", "partial"}:
        return False
    if eval_data.get("final"):
        return False
    if eval_data.get("submission_mode") == "direct" or eval_data.get("system_baseline"):
        return False
    return not bool(coder.get("active"))


def _coder_report_path(eval_data: dict, eval_manager: EvaluationManager) -> Optional[str]:
    evaluations_dir = getattr(eval_manager, "evaluations_dir", "")
    if not evaluations_dir:
        return None
    return os.path.join(
        os.path.dirname(os.path.abspath(evaluations_dir)),
        "storage",
        "report",
        f"{eval_data.get('id')}.md",
    )


def _repair_report_is_new(eval_data: dict, eval_manager: EvaluationManager) -> bool:
    """Tell restart recovery whether a repair coder replaced the old report."""
    audit = eval_data.get("audit", {}) or {}
    baseline = int(audit.get("repair_report_baseline_mtime_ns", 0) or 0)
    report_path = _coder_report_path(eval_data, eval_manager)
    if not baseline or not report_path:
        return False
    try:
        return os.stat(report_path).st_mtime_ns > baseline
    except OSError:
        return False


def _coder_report_exists(eval_data: dict, eval_manager: EvaluationManager) -> bool:
    report_path = _coder_report_path(eval_data, eval_manager)
    if not report_path:
        return False
    return file_io_utils.file_exists(report_path) and bool((file_io_utils.load_text(report_path) or "").strip())


def _abandon_interrupted_repair_attempt(
    eval_data: dict,
    reason: str,
    eval_manager: EvaluationManager,
) -> bool:
    """Abandon a dead repair attempt and require a subsequent report rewrite."""
    attempts = eval_data.get("attempts") or []
    if not attempts:
        return False
    latest_attempt = attempts[-1]
    if str(latest_attempt.get("status", "")).strip().lower() not in {"queued", "running"}:
        return False

    latest_attempt["status"] = "abandoned"
    latest_attempt["completed_timestamp"] = latest_attempt.get("completed_timestamp") or time.time()
    latest_attempt["error"] = reason

    report_path = _coder_report_path(eval_data, eval_manager)
    if report_path:
        try:
            report_mtime_ns = os.stat(report_path).st_mtime_ns
        except OSError:
            report_mtime_ns = 0
        if report_mtime_ns:
            audit = eval_data.setdefault("audit", {})
            previous_baseline = int(audit.get("repair_report_baseline_mtime_ns", 0) or 0)
            audit["repair_report_baseline_mtime_ns"] = max(previous_baseline, report_mtime_ns)
    return True


def _extract_audit_resume_token(eval_data: dict, eval_manager: EvaluationManager) -> Optional[str]:
    audit = eval_data.get("audit", {}) or {}
    persisted = str(audit.get("resume_token") or "").strip()
    if persisted:
        return persisted
    session_id = str(audit.get("session_id") or "").strip()
    if not session_id:
        return None
    backend = str((eval_data.get("coder", {}) or {}).get("backend") or constants.RESEARCH_CODER_BACKEND).lower()
    backend_runner = get_cli_worker_backend(backend)
    if not bool(getattr(backend_runner, "supports_resume", False)):
        return None
    research_root = os.path.dirname(os.path.abspath(eval_manager.evaluations_dir))
    transcript_path = os.path.join(
        research_root,
        "coder_sessions",
        f"audit_{session_id}",
        backend_runner.transcript_filename,
    )
    return backend_runner.extract_resume_token(transcript_path)


def _is_retryable_audit_infrastructure_block(audit: dict) -> bool:
    if str(audit.get("status", "")).strip().lower() != "blocked":
        return False
    if str(audit.get("failure_category", "")).strip().lower() == "infra_transient":
        return True
    return "transient auditor backend/provider failure" in str(audit.get("last_error") or "").lower()


def _requeue_eval_record(
    eval_manager: EvaluationManager,
    eval_id: str,
    reason: str,
    *,
    allow_retryable_blocked: bool = False,
    force_reopen_terminal: bool = False,
) -> bool:
    eval_id = str(eval_id)
    eval_data = eval_manager.get_evaluation(eval_id)
    if not isinstance(eval_data, dict):
        return False
    if "instruction" not in eval_data:
        return False
    if eval_data.get("final") and not force_reopen_terminal:
        return False
    if eval_data.get("submission_mode") == "direct" or eval_data.get("system_baseline"):
        return False

    status = _normalize_evaluation_status(eval_data.get("status")) or "queued"
    if force_reopen_terminal:
        pass
    elif status not in ACTIVE_EVALUATION_STATUSES and status != "queued":
        if not (allow_retryable_blocked and _is_no_report_terminal_requeueable(eval_data)):
            return False
    elif status == "blocked" and not (allow_retryable_blocked and _is_no_report_terminal_requeueable(eval_data)):
        return False

    def mutator(record):
        audit_record = record.setdefault("audit", {})
        coder_current = record.setdefault("coder", {})
        audit_stage = str(audit_record.get("status", "")).strip().lower()
        attempts = record.get("attempts") or []
        latest_attempt_status = str((attempts[-1] if attempts else {}).get("status", "")).strip().lower()
        if (
            bool(getattr(constants, "RESEARCH_CODER_AUDIT_ENABLED", True))
            and _is_retryable_audit_infrastructure_block(audit_record)
            and attempts
        ):
            audit_resume_token = _extract_audit_resume_token(record, eval_manager)
            coder_current.update({"active": False, "active_pid": None, "status": "audit_running"})
            audit_record.update({
                "active": False,
                "active_pid": None,
                "status": "pending_resume" if audit_resume_token else "retry",
                "spawn_count": 0,
                "resume_count": 0,
                "max_spawns": constants.RESEARCH_CODER_MAX_SPAWNS,
                "max_resumes": constants.RESEARCH_CODER_MAX_RESUMES,
                "next_resume_timestamp": None,
                "resume_delay_seconds": None,
                "resume_token": audit_resume_token,
                "started_timestamp": None,
                "completed_timestamp": None,
                "failure_category": "infra_transient",
                "last_error": reason,
            })
            record["status"] = "running"
            return
        # A completed report recovered after its official attempt settled must
        # proceed to audit, not relaunch the already-finished coder.
        if (
            bool(getattr(constants, "RESEARCH_CODER_AUDIT_ENABLED", True))
            and audit_stage in {"", "not_started"}
            and attempts
            and latest_attempt_status not in {"queued", "running"}
            and _coder_report_exists(record, eval_manager)
        ):
            coder_current.update({
                "active": False,
                "active_pid": None,
                "status": "audit_running",
                "spawn_count": 0,
                "resume_count": 0,
            })
            audit_record.update({
                "active": False,
                "active_pid": None,
                "status": "retry",
                "spawn_count": 0,
                "resume_count": 0,
                "next_resume_timestamp": None,
                "resume_delay_seconds": None,
                "resume_token": None,
                "started_timestamp": None,
                "completed_timestamp": None,
                "last_error": reason,
            })
            record["status"] = "running"
            return
        # An interrupted repair coder must resume independently. A replacement
        # report is ready for re-audit only after its official attempt settles.
        if bool(getattr(constants, "RESEARCH_CODER_AUDIT_ENABLED", True)) and audit_stage == "repairing" and record.get("attempts"):
            repair_attempt_interrupted = _abandon_interrupted_repair_attempt(
                record,
                reason,
                eval_manager,
            )
            if not repair_attempt_interrupted and _repair_report_is_new(record, eval_manager):
                coder_current.update({
                    "active": False,
                    "active_pid": None,
                    "status": "audit_running",
                    "spawn_count": 0,
                    "resume_count": 0,
                })
                audit_record.update({
                    "active": False,
                    "active_pid": None,
                    "status": "retry",
                    "spawn_count": 0,
                    "resume_count": 0,
                    "next_resume_timestamp": None,
                    "resume_delay_seconds": None,
                    "resume_token": None,
                    "started_timestamp": None,
                    "completed_timestamp": None,
                    "last_error": reason,
                })
                record["status"] = "running"
            else:
                resume_token = str(coder_current.get("resume_token") or "").strip()
                coder_current.update({
                    "active": bool(resume_token),
                    "active_pid": None,
                    "status": "pending_resume" if resume_token else "queued",
                    "spawn_count": 0,
                    "resume_count": 0,
                    "next_resume_timestamp": None,
                    "resume_delay_seconds": None,
                    "started_timestamp": None,
                    "completed_timestamp": None,
                    "last_error": reason,
                })
                record["status"] = "running" if resume_token else "queued"
            return
        if (
            bool(getattr(constants, "RESEARCH_CODER_AUDIT_ENABLED", True))
            and not coder_current.get("active")
            and audit_stage in {"running", "retry", "queued", "pending_resume", "resuming"}
            and record.get("attempts")
        ):
            audit_resume_token = str(audit_record.get("resume_token") or "").strip()
            coder_current.update({"active": False, "active_pid": None, "status": "audit_running"})
            audit_record.update({
                "active": False,
                "active_pid": None,
                "status": "pending_resume" if audit_resume_token else "retry",
                "spawn_count": 0,
                "resume_count": 0,
                "next_resume_timestamp": None,
                "resume_delay_seconds": None,
                "resume_token": audit_resume_token or None,
                "started_timestamp": None,
                "completed_timestamp": None,
                "last_error": reason,
            })
            record["status"] = "running"
            return
        coder_record = record.setdefault("coder", {})
        coder_record["active"] = False
        coder_record["active_pid"] = None
        coder_record["status"] = "queued"
        coder_record["spawn_count"] = 0
        coder_record["resume_count"] = 0
        coder_record["max_resumes"] = constants.RESEARCH_CODER_MAX_RESUMES
        coder_record["failure_category"] = None
        coder_record["session_id"] = None
        coder_record["resume_token"] = None
        coder_record["started_timestamp"] = None
        coder_record["completed_timestamp"] = None
        coder_record["exit_code"] = None
        coder_record["last_error"] = reason
        record["status"] = "queued"
        if force_reopen_terminal:
            record["final"] = {}
            notification = record.setdefault("notification", {})
            notification["sent"] = False
            notification["sent_timestamp"] = None
            notification["message"] = None
        if attempts:
            latest = attempts[-1]
            latest_status = str(latest.get("status", ""))
            if latest_status in {"queued", "running"}:
                latest["status"] = "abandoned"
                latest["completed_timestamp"] = latest.get("completed_timestamp") or time.time()
                latest["error"] = reason

    updated = eval_manager.update_evaluation(eval_id, mutator)
    return bool(updated)


def reset_runtime_coder_counters(
    *,
    eval_ids: Optional[Iterable[str]] = None,
    eval_manager: Optional[EvaluationManager] = None,
    paths=None,
) -> int:
    paths = _ensure_recovery_paths(paths)
    eval_manager = eval_manager or EvaluationManager(paths.evaluations_dir)
    reset_count = 0

    if eval_ids is None:
        target_ids = eval_manager.get_unfinished_instruction_eval_ids()
    else:
        target_ids = [str(eval_id) for eval_id in eval_ids]

    for eval_id in target_ids:
        eval_data = eval_manager.get_evaluation(eval_id)
        if not isinstance(eval_data, dict):
            continue
        if "instruction" not in eval_data:
            continue
        if eval_data.get("final"):
            continue
        if eval_data.get("submission_mode") == "direct" or eval_data.get("system_baseline"):
            continue

        def mutator(record):
            coder_record = record.setdefault("coder", {})
            coder_record["spawn_count"] = 0
            coder_record["resume_count"] = 0
            coder_record["max_resumes"] = constants.RESEARCH_CODER_MAX_RESUMES
            audit_record = record.setdefault("audit", {})
            if str(audit_record.get("status", "")).strip().lower() in {
                "queued", "retry", "running", "pending_resume", "resuming",
            }:
                audit_record["spawn_count"] = 0
                audit_record["resume_count"] = 0

        updated = eval_manager.update_evaluation(str(eval_id), mutator)
        if updated is not None:
            reset_count += 1

    return reset_count


def requeue_instruction_evaluations(
    *,
    eval_ids: Optional[Iterable[str]] = None,
    reason: str,
    kill_running_coders: bool = False,
    force_reopen_terminal: bool = False,
    eval_manager: Optional[EvaluationManager] = None,
    paths=None,
) -> int:
    started_at = time.perf_counter()
    paths = _ensure_recovery_paths(paths)
    target_ids: list[str]
    if eval_ids is None:
        if eval_manager is not None:
            target_ids = eval_manager.get_running_instruction_eval_ids()
        else:
            target_ids = evaluation_index.get_running_instruction_eval_ids(paths.evaluations_dir)
        if not target_ids:
            if kill_running_coders:
                _terminate_station_wrapper_processes(paths.research_root)
            print(
                "restart_evaluations: requeue_instruction_evaluations "
                f"targeted=0 recovered=0 in {time.perf_counter() - started_at:.3f}s"
            )
            return 0
    else:
        target_ids = [str(eval_id) for eval_id in eval_ids]
    eval_manager = eval_manager or EvaluationManager(paths.evaluations_dir)
    recovered = 0

    for eval_id in target_ids:
        eval_data = eval_manager.get_evaluation(eval_id)
        if not isinstance(eval_data, dict) or "instruction" not in eval_data:
            continue
        if eval_data.get("final") and not force_reopen_terminal:
            continue
        if eval_data.get("submission_mode") == "direct" or eval_data.get("system_baseline"):
            continue

        status = _normalize_evaluation_status(eval_data.get("status")) or "queued"
        if eval_ids is None and status != "running":
            continue

        coder = eval_data.get("coder", {}) or {}
        pid = coder.get("active_pid")
        if kill_running_coders and _pid_matches_station(pid, paths.research_root, paths.coder_sessions_dir):
            _terminate_pid(pid, kill_process_group=True)
        audit_pid = (eval_data.get("audit", {}) or {}).get("active_pid")
        if kill_running_coders and _pid_matches_station(audit_pid, paths.research_root, paths.coder_sessions_dir):
            _terminate_pid(audit_pid, kill_process_group=True)

        notification_message = None
        author = None
        if force_reopen_terminal:
            notification = eval_data.get("notification") or {}
            notification_message = notification.get("message")
            author = eval_data.get("author")

        _remove_run_requests(paths, str(eval_id))
        if _requeue_eval_record(
            eval_manager,
            str(eval_id),
            reason,
            force_reopen_terminal=force_reopen_terminal,
        ):
            if force_reopen_terminal and author and notification_message:
                agent_module.remove_pending_notification_atomic(str(author), str(notification_message))
            recovered += 1

    if kill_running_coders:
        _terminate_station_wrapper_processes(paths.research_root)

    print(
        "restart_evaluations: requeue_instruction_evaluations "
        f"targeted={len(target_ids)} recovered={recovered} "
        f"in {time.perf_counter() - started_at:.3f}s"
    )
    return recovered


def requeue_unfinished_instruction_evaluations(
    *,
    reason: str,
    include_active_statuses: bool = True,
    eval_manager: Optional[EvaluationManager] = None,
    paths=None,
) -> int:
    started_at = time.perf_counter()
    paths = _ensure_recovery_paths(paths)
    eval_manager = eval_manager or EvaluationManager(paths.evaluations_dir)
    recovered = 0

    target_ids = set(
        eval_manager.get_unfinished_requeue_candidate_instruction_eval_ids(
            include_active_statuses=include_active_statuses,
        )
    )

    for eval_id in sorted(target_ids, key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value))):
        eval_data = eval_manager.get_evaluation(eval_id)
        if not isinstance(eval_data, dict) or "instruction" not in eval_data:
            continue
        if eval_data.get("final"):
            continue
        if eval_data.get("submission_mode") == "direct" or eval_data.get("system_baseline"):
            continue

        status = _normalize_evaluation_status(eval_data.get("status")) or "queued"
        if status in {"failed", "blocked", "partial"}:
            if not _is_no_report_terminal_requeueable(eval_data):
                continue
        elif status not in ACTIVE_EVALUATION_STATUSES and status != "queued":
            continue
        elif not include_active_statuses:
            continue

        coder = eval_data.get("coder", {}) or {}
        if bool(coder.get("active")):
            continue
        attempts = eval_data.get("attempts") or []
        latest_attempt = attempts[-1] if attempts else {}
        latest_attempt_status = str(latest_attempt.get("status", "")).strip().lower()
        if latest_attempt_status == "running":
            continue

        _remove_run_requests(paths, str(eval_id))
        if _requeue_eval_record(
            eval_manager,
            str(eval_id),
            reason,
            allow_retryable_blocked=True,
        ):
            recovered += 1

    print(
        "restart_evaluations: requeue_unfinished_instruction_evaluations "
        f"targeted={len(target_ids)} recovered={recovered} "
        f"in {time.perf_counter() - started_at:.3f}s"
    )
    return recovered


def restart_stuck_evaluations(*, eval_manager: Optional[EvaluationManager] = None, paths=None) -> int:
    started_at = time.perf_counter()
    paths = _ensure_recovery_paths(paths)
    eval_manager = eval_manager or EvaluationManager(paths.evaluations_dir)
    phase_started_at = time.perf_counter()
    recovered = requeue_instruction_evaluations(
        reason="Recovered after restart: instruction prompt requeued.",
        kill_running_coders=True,
        eval_manager=eval_manager,
        paths=paths,
    )
    print(
        "restart_evaluations: restart_stuck_evaluations running-phase "
        f"took {time.perf_counter() - phase_started_at:.3f}s"
    )
    phase_started_at = time.perf_counter()
    recovered += requeue_unfinished_instruction_evaluations(
        reason="Recovered after restart: unfinished instruction prompt requeued.",
        include_active_statuses=False,
        eval_manager=eval_manager,
        paths=paths,
    )
    print(
        "restart_evaluations: restart_stuck_evaluations unfinished-phase "
        f"took {time.perf_counter() - phase_started_at:.3f}s"
    )
    phase_started_at = time.perf_counter()
    reset_runtime_coder_counters(eval_manager=eval_manager, paths=paths)
    print(
        "restart_evaluations: restart_stuck_evaluations reset-counters phase "
        f"took {time.perf_counter() - phase_started_at:.3f}s"
    )
    print(
        "restart_evaluations: restart_stuck_evaluations completed "
        f"recovered={recovered} in {time.perf_counter() - started_at:.3f}s"
    )
    return recovered


__all__ = [
    "restart_stuck_evaluations",
    "requeue_instruction_evaluations",
    "requeue_unfinished_instruction_evaluations",
    "reset_runtime_coder_counters",
]
