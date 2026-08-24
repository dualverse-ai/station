"""
Archive Surveyor background service.

The Surveyor is a process-backed local researcher for Archive Room survey
requests. It reuses the generic CLI worker backend layer, but owns its own
request/session/report state under station_data/rooms/archive/surveyor.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable, Dict, List, Optional

import filelock
import yaml

from station import capsule as capsule_module
from station import constants
from station import file_io_utils
from station import index_paths
from station import supervisor_utils
from station.eval_archive.archive_preview import build_archive_preview
from station.eval_archive.survey_worker import (
    ActiveSurveySession,
    build_archive_survey_prompt,
    ensure_survey_source_link,
)
from station.workers.cli import (
    apply_codex_proxy_overrides,
    build_cli_worker_runtime_env,
    detect_cli_worker_executable,
)
from station.workers.job_manager import (
    CliJobLaunchSpec,
    CliJobManager,
    CliJobState,
)
from station.eval_research.runtime_paths import get_research_root, load_task_spec_markdown
from station.rooms.mail import format_new_mail_notification
from station.sync.fast_lane_service import FastLaneSubmissionRequest, FastLaneSubmissionService


SURVEY_STATUS_QUEUED = "queued"
SURVEY_STATUS_RUNNING = "running"
SURVEY_STATUS_COMPLETED = "completed"
SURVEY_STATUS_FAILED = "failed"
SURVEY_STATUS_BLOCKED = "blocked"
TERMINAL_SURVEY_STATUSES = {SURVEY_STATUS_COMPLETED, SURVEY_STATUS_FAILED, SURVEY_STATUS_BLOCKED}
SURVEYOR_AGENTS_MD_CONTENT = (
    "Follow the initial Archive Surveyor prompt as your authority; this repository is for Station "
    "development, and you do not need to access its source code or developer docs to perform your "
    "surveyor function.\n"
)
SURVEY_REPORT_MAIL_NOTICE = (
    "**Survey Report Notice:** "
    "This is the full survey report. Please do not reply to this mail; "
    "Archive Surveyor is a system worker and cannot receive replies. You are the only recipient "
    "of this report. Other agents cannot read this mail, but you may share report content with "
    "them if needed."
)
SURVEY_NOTIFICATION_MAIL_READ_REPAIRED_KEY = "mail_read_status_repaired"
SURVEY_NOTIFICATION_MAIL_READ_REPAIRED_TIMESTAMP_KEY = "mail_read_status_repaired_timestamp"


@dataclass(frozen=True)
class ArchiveSurveyorPaths:
    archive_room_root: str
    surveyor_root: str
    requests_dir: str
    reports_dir: str
    sessions_dir: str
    pending_file: str
    submission_lock_path: str
    archive_link: str
    research_link: str
    question_link: str
    archive_target: str
    research_target: str
    question_target: str


@dataclass
class ArchiveSurveyValidationResult:
    ok: bool
    messages: List[str]
    prompt: Optional[str] = None
    error: Optional[str] = None
    question_room_access: bool = False


@dataclass
class ArchiveSurveySubmissionResult:
    accepted: bool
    messages: List[str]
    survey_id: Optional[str] = None
    notification: Optional[str] = None
    error: Optional[str] = None


def _get_archive_room_root(consts_module=constants) -> str:
    return os.path.join(
        consts_module.BASE_STATION_DATA_PATH,
        consts_module.ROOMS_DIR_NAME,
        consts_module.SHORT_ROOM_NAME_ARCHIVE,
    )


def _get_archive_capsules_root(consts_module=constants) -> str:
    return os.path.join(
        consts_module.BASE_STATION_DATA_PATH,
        consts_module.CAPSULES_DIR_NAME,
        consts_module.ARCHIVE_CAPSULES_SUBDIR_NAME,
    )


def _get_question_capsules_root(consts_module=constants) -> str:
    return os.path.join(
        consts_module.BASE_STATION_DATA_PATH,
        consts_module.CAPSULES_DIR_NAME,
        consts_module.QUESTION_CAPSULES_SUBDIR_NAME,
    )


def _agent_has_question_room_access(
    agent_data: Dict[str, Any],
    current_tick: int,
    station_instance: Any = None,
    consts_module=constants,
) -> bool:
    if station_instance and hasattr(station_instance, "_is_agent_question_room_allowed"):
        return bool(station_instance._is_agent_question_room_allowed(agent_data, current_tick))

    if agent_data.get(consts_module.AGENT_STATUS_KEY) != consts_module.AGENT_STATUS_RECURSIVE:
        return False
    if supervisor_utils.is_supervisor(agent_data, consts_module):
        return True

    birth_tick = agent_data.get(consts_module.AGENT_TICK_BIRTH_KEY)
    if birth_tick is None:
        return False
    try:
        agent_age = int(current_tick) - int(birth_tick)
    except (TypeError, ValueError):
        return False
    tenured_threshold = getattr(consts_module, "MIN_AGENT_AGE_BEFORE_LEAVE", None)
    return tenured_threshold is not None and tenured_threshold > 0 and agent_age >= int(tenured_threshold)


def _ensure_symlink(link_path: str, target_path: str) -> None:
    ensure_survey_source_link(link_path, target_path)


def ensure_archive_surveyor_layout(consts_module=constants) -> ArchiveSurveyorPaths:
    archive_room_root = _get_archive_room_root(consts_module)
    surveyor_root = os.path.join(archive_room_root, consts_module.ARCHIVE_SURVEYOR_SUBDIR_NAME)
    archive_target = _get_archive_capsules_root(consts_module)
    research_target = get_research_root(consts_module)
    question_target = _get_question_capsules_root(consts_module)

    file_io_utils.ensure_dir_exists(surveyor_root)
    file_io_utils.ensure_dir_exists(archive_target)
    file_io_utils.ensure_dir_exists(question_target)

    paths = ArchiveSurveyorPaths(
        archive_room_root=archive_room_root,
        surveyor_root=surveyor_root,
        requests_dir=os.path.join(surveyor_root, consts_module.ARCHIVE_SURVEY_REQUESTS_SUBDIR_NAME),
        reports_dir=os.path.join(surveyor_root, consts_module.ARCHIVE_SURVEY_REPORTS_SUBDIR_NAME),
        sessions_dir=os.path.join(surveyor_root, consts_module.ARCHIVE_SURVEY_SESSIONS_SUBDIR_NAME),
        pending_file=os.path.join(surveyor_root, consts_module.PENDING_ARCHIVE_SURVEYS_FILENAME),
        submission_lock_path=os.path.join(surveyor_root, ".submission.lock"),
        archive_link=os.path.join(surveyor_root, consts_module.ARCHIVE_SURVEY_ARCHIVE_LINK_NAME),
        research_link=os.path.join(surveyor_root, consts_module.ARCHIVE_SURVEY_RESEARCH_LINK_NAME),
        question_link=os.path.join(surveyor_root, consts_module.ARCHIVE_SURVEY_QUESTION_LINK_NAME),
        archive_target=archive_target,
        research_target=research_target,
        question_target=question_target,
    )
    for dir_path in (paths.requests_dir, paths.reports_dir, paths.sessions_dir):
        file_io_utils.ensure_dir_exists(dir_path)
    agents_md_path = os.path.join(paths.surveyor_root, "AGENTS.md")
    if file_io_utils.load_text(agents_md_path) != SURVEYOR_AGENTS_MD_CONTENT:
        file_io_utils.save_text(SURVEYOR_AGENTS_MD_CONTENT, agents_md_path)
    _ensure_symlink(paths.archive_link, paths.archive_target)
    _ensure_symlink(paths.research_link, paths.research_target)
    _ensure_symlink(paths.question_link, paths.question_target)
    return paths


def _request_path(paths: ArchiveSurveyorPaths, survey_id: str) -> str:
    return os.path.join(paths.requests_dir, f"survey_{survey_id}.yaml")


def _load_request(paths: ArchiveSurveyorPaths, survey_id: str) -> Optional[Dict[str, Any]]:
    data = file_io_utils.load_yaml(_request_path(paths, survey_id))
    return data if isinstance(data, dict) else None


def _save_request(paths: ArchiveSurveyorPaths, survey_id: str, data: Dict[str, Any]) -> None:
    file_io_utils.save_yaml(data, _request_path(paths, survey_id), sort_keys=False)


def _iter_request_ids(paths: ArchiveSurveyorPaths) -> List[str]:
    ids: List[str] = []
    for filename in file_io_utils.list_files(paths.requests_dir, constants.YAML_EXTENSION):
        if not filename.startswith("survey_"):
            continue
        raw_id = filename[len("survey_") : -len(constants.YAML_EXTENSION)]
        if raw_id:
            ids.append(raw_id)
    return sorted(ids, key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)))


def _load_pending_entries(paths: ArchiveSurveyorPaths) -> List[Dict[str, Any]]:
    try:
        return [entry for entry in file_io_utils.load_yaml_lines(paths.pending_file) if isinstance(entry, dict)]
    except Exception as exc:
        print(f"ArchiveSurveyor: Failed to load pending survey queue: {exc}")
        return []


def _write_pending_entries(paths: ArchiveSurveyorPaths, entries: List[Dict[str, Any]]) -> None:
    if not entries:
        file_io_utils.save_text("", paths.pending_file)
        return
    documents = []
    for entry in entries:
        documents.append(yaml.dump(entry, sort_keys=False, allow_unicode=True, default_flow_style=False).rstrip())
    file_io_utils.save_text("\n---\n".join(documents) + "\n", paths.pending_file)


def _remove_pending_entry(paths: ArchiveSurveyorPaths, survey_id: str) -> None:
    lock = filelock.FileLock(paths.submission_lock_path, timeout=60)
    with lock:
        remaining = [
            entry
            for entry in _load_pending_entries(paths)
            if str(entry.get("id") or entry.get("survey_id")) != str(survey_id)
        ]
        _write_pending_entries(paths, remaining)


def queue_archive_survey_request(
    *,
    author: str,
    lineage: Optional[str],
    prompt: str,
    tick: int,
    backend: Optional[str] = None,
    model_name: Optional[str] = None,
    question_room_access: bool = False,
    parallel_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    paths = ensure_archive_surveyor_layout()
    lock = filelock.FileLock(paths.submission_lock_path, timeout=60)
    with lock:
        survey_id = str(file_io_utils.get_next_sequential_id(paths.requests_dir, "survey_", constants.YAML_EXTENSION))
        now = time.time()
        record = {
            "schema_version": 1,
            "id": survey_id,
            "author": author,
            "lineage": (lineage or "unknown").lower(),
            "prompt": prompt,
            "question_room_access": bool(question_room_access),
            "submitted_tick": tick,
            "submitted_timestamp": now,
            "status": SURVEY_STATUS_QUEUED,
            "artifacts": {
                "draft": os.path.join(constants.ARCHIVE_SURVEY_REPORTS_SUBDIR_NAME, f"{survey_id}.draft.md"),
                "report": os.path.join(constants.ARCHIVE_SURVEY_REPORTS_SUBDIR_NAME, f"{survey_id}.md"),
            },
            "session": {
                "backend": backend or constants.ARCHIVE_SURVEY_BACKEND,
                "model_name": model_name if model_name is not None else constants.ARCHIVE_SURVEY_MODEL_NAME,
                "active": False,
                "session_id": None,
                "active_pid": None,
                "status": SURVEY_STATUS_QUEUED,
                "spawn_count": 0,
                "max_spawns": getattr(
                    constants,
                    "RESEARCH_CODER_MAX_SPAWNS",
                    constants.ARCHIVE_SURVEY_MAX_SPAWNS,
                ),
                "resume_token": None,
                "resume_count": 0,
                "max_resumes": constants.ARCHIVE_SURVEY_MAX_RESUMES,
                "resume_delay_seconds": 0,
                "next_resume_timestamp": None,
                "started_timestamp": None,
                "completed_timestamp": None,
                "exit_code": None,
                "last_error": None,
            },
            "notification": {
                "sent": False,
                "sent_timestamp": None,
                "message": None,
                "mail_capsule_id": None,
                "mail_numeric_id": None,
            },
        }
        if parallel_metadata:
            record["parallel_commit_status"] = "provisional"
            record["parallel_tick"] = {
                **parallel_metadata,
                "created_timestamp": now,
            }
        _save_request(paths, survey_id, record)
        file_io_utils.append_yaml_line(
            {
                "id": survey_id,
                "author": author,
                "submitted_tick": tick,
                "submitted_timestamp": now,
            },
            paths.pending_file,
        )
        return record


def get_active_survey_ids_for_author(author: str) -> List[str]:
    paths = ensure_archive_surveyor_layout()
    author_lower = str(author or "").lower()
    active_ids: List[str] = []
    for survey_id in _iter_request_ids(paths):
        record = _load_request(paths, survey_id)
        if not isinstance(record, dict):
            continue
        if str(record.get("author", "")).lower() != author_lower:
            continue
        if str(record.get("status", "")).lower() in {SURVEY_STATUS_QUEUED, SURVEY_STATUS_RUNNING}:
            active_ids.append(str(survey_id))
    return active_ids


def validate_archive_survey_request(
    *,
    agent_data: Dict[str, Any],
    yaml_data: Optional[Dict[str, Any]],
    current_tick: int,
    station_instance: Any = None,
    consts_module=constants,
) -> ArchiveSurveyValidationResult:
    if not getattr(consts_module, "ARCHIVE_SURVEY_ENABLED", False):
        return ArchiveSurveyValidationResult(
            ok=False,
            messages=["Archive survey failed: Archive Surveyor is disabled."],
            error="disabled",
        )

    agent_status = agent_data.get(consts_module.AGENT_STATUS_KEY)
    agent_name = agent_data.get(consts_module.AGENT_NAME_KEY, "UnknownAgent")

    if agent_status != consts_module.AGENT_STATUS_RECURSIVE:
        return ArchiveSurveyValidationResult(
            ok=False,
            messages=["Archive survey failed: only recursive agents can request Archive Surveyor reports."],
            error="not_recursive",
        )

    if station_instance and hasattr(station_instance, "_is_agent_mature"):
        if not station_instance._is_agent_mature(agent_data, current_tick):
            return ArchiveSurveyValidationResult(
                ok=False,
                messages=["Archive survey failed: only mature recursive agents can request Archive Surveyor reports."],
                error="immature",
            )

    if not yaml_data:
        return ArchiveSurveyValidationResult(
            ok=False,
            messages=["Archive survey requires YAML data with a non-empty `prompt` field."],
            error="missing_yaml",
        )

    prompt = str(yaml_data.get(consts_module.YAML_ARCHIVE_SURVEY_PROMPT, "") or "").strip()
    if not prompt:
        return ArchiveSurveyValidationResult(
            ok=False,
            messages=["Archive survey requires YAML data with a non-empty `prompt` field."],
            error="missing_prompt",
        )

    max_active = int(getattr(consts_module, "ARCHIVE_SURVEY_MAX_ACTIVE_PER_AGENT", 1))
    active_ids = get_active_survey_ids_for_author(agent_name)
    if max_active >= 0 and len(active_ids) >= max_active:
        return ArchiveSurveyValidationResult(
            ok=False,
            messages=[
                f"Archive survey failed: you already have {len(active_ids)} pending/running survey request(s): "
                f"{', '.join(active_ids)}."
            ],
            error="active_limit",
        )

    question_room_access = _agent_has_question_room_access(
        agent_data,
        current_tick,
        station_instance=station_instance,
        consts_module=consts_module,
    )
    return ArchiveSurveyValidationResult(
        ok=True,
        messages=[],
        prompt=prompt,
        question_room_access=question_room_access,
    )


def mark_archive_survey_committed(survey_id: str) -> bool:
    paths = ensure_archive_surveyor_layout()
    lock = filelock.FileLock(_request_path(paths, survey_id) + ".lock", timeout=60)
    with lock:
        record = _load_request(paths, survey_id)
        if not isinstance(record, dict):
            return False
        if str(record.get("parallel_commit_status") or "").lower() == "provisional":
            record["parallel_commit_status"] = "committed"
            parallel_meta = record.setdefault("parallel_tick", {})
            if isinstance(parallel_meta, dict):
                parallel_meta["committed_timestamp"] = time.time()
            _save_request(paths, survey_id, record)
        return True


def delete_archive_survey_request(survey_id: str, *, kill_running: bool = True) -> bool:
    paths = ensure_archive_surveyor_layout()
    lock = filelock.FileLock(_request_path(paths, survey_id) + ".lock", timeout=60)
    with lock:
        record = _load_request(paths, survey_id)
        if not isinstance(record, dict):
            return False
        session = record.get("session") or {}
        if kill_running:
            _terminate_survey_process(session)
        _remove_pending_entry(paths, survey_id)
        for filename in (f"{survey_id}.md", f"{survey_id}.draft.md"):
            try:
                file_io_utils.delete_file(os.path.join(paths.reports_dir, filename))
            except Exception as exc:
                print(f"ArchiveSurveyor: failed to delete report artifact {filename}: {exc}")
        session_id = session.get("session_id")
        if session_id:
            try:
                shutil.rmtree(os.path.join(paths.sessions_dir, str(session_id)), ignore_errors=True)
            except Exception as exc:
                print(f"ArchiveSurveyor: failed to delete session {session_id}: {exc}")
        return file_io_utils.delete_file(_request_path(paths, survey_id))


def rollback_provisional_archive_surveys(
    *,
    run_id: str,
    explicit_ids: Optional[List[str]] = None,
) -> List[str]:
    paths = ensure_archive_surveyor_layout()
    explicit = {str(survey_id) for survey_id in (explicit_ids or [])}
    run_id = str(run_id or "")
    candidates: List[str] = []

    for survey_id in _iter_request_ids(paths):
        record = _load_request(paths, survey_id)
        if not isinstance(record, dict):
            continue
        status = str(record.get("parallel_commit_status") or "").strip().lower()
        parallel_meta = record.get("parallel_tick") or {}
        request_run_id = str(parallel_meta.get("run_id") or "") if isinstance(parallel_meta, dict) else ""
        if status == "provisional" and (survey_id in explicit or (run_id and request_run_id == run_id)):
            candidates.append(str(survey_id))

    rolled_back: List[str] = []
    for survey_id in candidates:
        try:
            if delete_archive_survey_request(survey_id, kill_running=True):
                rolled_back.append(str(survey_id))
        except Exception as exc:
            print(f"ArchiveSurveyor: failed to roll back provisional survey {survey_id}: {exc}")
    return rolled_back


def _terminate_survey_process(session: Dict[str, Any]) -> None:
    pid = session.get("active_pid")
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return
    if pid_int <= 0:
        return
    try:
        os.killpg(os.getpgid(pid_int), signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception as exc:
        try:
            os.kill(pid_int, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception as fallback_exc:
            print(f"ArchiveSurveyor: failed to terminate survey process {pid_int}: {exc}; fallback: {fallback_exc}")


class ArchiveSurveySubmissionService(FastLaneSubmissionService):
    """Single-writer service for Archive survey actions received in parallel ticks."""

    service_name = "ArchiveSurveySubmissionService"

    def __init__(
        self,
        station_instance: Any,
        *,
        log_event_func: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        super().__init__(station_instance, log_event_func=log_event_func)

    def _process_request(self, request: FastLaneSubmissionRequest) -> ArchiveSurveySubmissionResult:
        consts = constants
        agent_data = dict(request.agent_data)
        yaml_data = dict(request.yaml_data) if isinstance(request.yaml_data, dict) else request.yaml_data
        validation = validate_archive_survey_request(
            agent_data=agent_data,
            yaml_data=yaml_data,
            current_tick=request.current_tick,
            station_instance=self.station,
            consts_module=consts,
        )
        if not validation.ok:
            return ArchiveSurveySubmissionResult(
                accepted=False,
                messages=validation.messages,
                error=validation.error,
            )

        agent_name = str(agent_data.get(consts.AGENT_NAME_KEY, "UnknownAgent"))
        record = queue_archive_survey_request(
            author=agent_name,
            lineage=agent_data.get(consts.AGENT_LINEAGE_KEY),
            prompt=str(validation.prompt or ""),
            tick=request.current_tick,
            question_room_access=bool(validation.question_room_access),
            parallel_metadata={
                "run_id": request.run_id,
                "op_id": request.op_id,
            },
        )
        survey_id = str(record.get("id", "unknown"))
        self._wake_surveyor(survey_id)
        self._push_log_event(
            "parallel_archive_survey_submission_accepted",
            {"agent_name": agent_name, "survey_id": survey_id, "op_id": request.op_id},
        )
        return ArchiveSurveySubmissionResult(
            accepted=True,
            messages=[
                f"Archive survey request queued. Survey ID: {survey_id}. "
                "The report will be sent to your mail automatically when ready."
            ],
            survey_id=survey_id,
        )

    def _default_timeout_seconds(self) -> float:
        return float(getattr(constants, "PARALLEL_ARCHIVE_SURVEY_SUBMISSION_TIMEOUT_SECONDS", 0.0))

    def _timeout_result(self) -> ArchiveSurveySubmissionResult:
        return ArchiveSurveySubmissionResult(
            accepted=False,
            messages=["Archive survey failed: Archive survey submission service timed out."],
            error="timeout",
        )

    def _empty_result(self) -> ArchiveSurveySubmissionResult:
        return ArchiveSurveySubmissionResult(
            accepted=False,
            messages=["Archive survey failed: Archive survey submission service returned no result."],
            error="empty_result",
        )

    def _exception_result(self, exc: Exception, request: FastLaneSubmissionRequest) -> ArchiveSurveySubmissionResult:
        return ArchiveSurveySubmissionResult(
            accepted=False,
            messages=[f"Archive survey failed: internal Archive survey submission error: {exc}"],
            error=str(exc),
        )

    def _exception_event_type(self) -> str:
        return "parallel_archive_survey_submission_error"

    def _wake_surveyor(self, survey_id: str) -> None:
        surveyor = getattr(self.station, "auto_archive_surveyor", None)
        if surveyor and hasattr(surveyor, "wake"):
            try:
                surveyor.wake(f"parallel archive survey {survey_id}")
            except Exception as exc:
                self._push_log_event(
                    "parallel_archive_survey_wake_error",
                    {"survey_id": survey_id, "error": str(exc)},
                )


class AutoArchiveSurveyor(CliJobManager):
    _active_instances: Dict[int, "AutoArchiveSurveyor"] = {}
    fresh_attempt_after_resume_exhaustion = True

    def __init__(self, station_instance, enabled: Optional[bool] = None, log_queue: Optional[Queue] = None):
        super().__init__()
        station_id = id(station_instance)
        if station_id in self._active_instances and self._active_instances[station_id].is_running:
            print("AutoArchiveSurveyor: WARNING - another surveyor is already running for this station")

        self.station = station_instance
        self.enabled = enabled if enabled is not None else constants.ARCHIVE_SURVEY_ENABLED
        self.check_interval = constants.ARCHIVE_SURVEY_CHECK_INTERVAL
        self.timeout_seconds = constants.ARCHIVE_SURVEY_TIMEOUT_SECONDS
        self.max_parallel_workers = constants.ARCHIVE_SURVEY_MAX_PARALLEL_WORKERS
        self.log_queue = log_queue
        self.paths = ensure_archive_surveyor_layout()

        self.is_running = False
        self.surveyor_thread: Optional[threading.Thread] = None
        self._wake_event = threading.Event()
        self._active_instances[station_id] = self

    def _push_log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if self.log_queue is None:
            return
        try:
            self.log_queue.put_nowait({"event": event_type, "data": data, "timestamp": time.time()})
        except Exception as exc:
            print(f"AutoArchiveSurveyor: failed to queue log event: {exc}")

    @classmethod
    def _resume_backoff_schedule(cls) -> Any:
        return getattr(constants, "RESEARCH_CODER_RESUME_BACKOFF_SECONDS", [])

    def _pause_station(self, reason: str) -> None:
        orchestrator = getattr(self.station, "orchestrator", None)
        if orchestrator and hasattr(orchestrator, "pause_due_to_research_issue"):
            orchestrator.pause_due_to_research_issue(reason)
            return
        print(f"AutoArchiveSurveyor: Unable to pause orchestrator directly. Reason: {reason}")

    def start_surveyor_loop(self) -> bool:
        if not self.enabled:
            print("AutoArchiveSurveyor: archive surveyor is disabled")
            return False
        if self.is_running:
            return True
        self.is_running = True
        self._wake_event.clear()
        self._recover_stale_requests(restart_recovery=True)
        self.surveyor_thread = threading.Thread(target=self._surveyor_loop, daemon=True)
        self.surveyor_thread.start()
        print("AutoArchiveSurveyor: surveyor loop started")
        return True

    def stop_surveyor_loop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        self._wake_event.set()
        self.stop_active_cli_jobs(
            force=False,
            after_terminate=lambda survey_id, _session: self._mark_requeued_after_shutdown(survey_id),
        )
        if self.surveyor_thread and self.surveyor_thread.is_alive():
            self.surveyor_thread.join(timeout=5)
        station_id = id(self.station)
        if self._active_instances.get(station_id) is self:
            del self._active_instances[station_id]
        print("AutoArchiveSurveyor: surveyor loop stopped")

    def wake(self, reason: str = "") -> None:
        if reason:
            self._push_log_event("archive_surveyor_wake", {"reason": reason})
        self._wake_event.set()

    def has_pending_or_running(self) -> bool:
        if self.active_sessions:
            return True
        for survey_id in _iter_request_ids(self.paths):
            record = _load_request(self.paths, survey_id)
            if isinstance(record, dict) and str(record.get("status", "")).lower() in {
                SURVEY_STATUS_QUEUED,
                SURVEY_STATUS_RUNNING,
            }:
                return True
        return False

    @staticmethod
    def _coerce_tick(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def should_wait_at_tick(self, current_tick: int) -> bool:
        max_allowed_ticks = int(getattr(constants, "ARCHIVE_SURVEY_MAX_TICK", 2) or 0)
        if max_allowed_ticks <= 0:
            return False

        for survey_id in _iter_request_ids(self.paths):
            record = _load_request(self.paths, survey_id)
            if not isinstance(record, dict):
                continue
            if str(record.get("status", "")).strip().lower() != SURVEY_STATUS_RUNNING:
                continue
            submitted_tick = self._coerce_tick(record.get("submitted_tick"))
            if submitted_tick is None:
                continue
            elapsed_ticks = current_tick - submitted_tick + 1
            if elapsed_ticks >= max_allowed_ticks:
                return True
        return False

    def get_job_statistics(self) -> Dict[str, Any]:
        now = time.time()
        running_jobs: List[Dict[str, Any]] = []
        queued_jobs: List[Dict[str, Any]] = []

        for survey_id in _iter_request_ids(self.paths):
            record = _load_request(self.paths, survey_id)
            if not isinstance(record, dict):
                continue
            status = str(record.get("status", "")).strip().lower()
            if status not in {SURVEY_STATUS_QUEUED, SURVEY_STATUS_RUNNING}:
                continue
            session = record.get("session") or {}
            start_timestamp = session.get("started_timestamp") or record.get("submitted_timestamp") or 0
            job = {
                "evaluation_id": f"Archive Survey #{survey_id}",
                "job_id": f"archive_survey_{survey_id}",
                "job_type": "archive_survey",
                "agent_name": record.get("author", "Unknown"),
                "title": f"Archive Survey #{survey_id}",
                "start_tick": record.get("submitted_tick", 0),
                "submitted_tick": record.get("submitted_tick", 0),
                "start_timestamp": start_timestamp,
                "elapsed_seconds": int(now - start_timestamp) if start_timestamp else 0,
                "status": session.get("status") or status,
                "top_level_status": status,
                "coder_active": bool(session.get("active")),
                "execution_source": "surveyor",
                "system_baseline": False,
            }
            if status == SURVEY_STATUS_RUNNING:
                running_jobs.append(job)
            else:
                queued_jobs.append(job)

        running_jobs.sort(key=lambda item: item.get("start_timestamp", 0), reverse=True)
        queued_jobs.sort(key=lambda item: item.get("submitted_tick", 0), reverse=True)
        return {
            "running_count": len(running_jobs),
            "queued_count": len(queued_jobs),
            "running_jobs": running_jobs,
            "queued_jobs": queued_jobs,
        }

    def get_lightweight_job_statistics(self) -> Dict[str, Any]:
        """Return dashboard job counts without scanning historical requests."""
        now = time.time()
        running_jobs = []
        for survey_id in sorted(self.active_sessions):
            record = _load_request(self.paths, survey_id) or {}
            session = record.get("session") or {}
            started = session.get("started_timestamp") or record.get("submitted_timestamp") or now
            running_jobs.append({
                "evaluation_id": f"Archive Survey #{survey_id}",
                "job_id": f"archive_survey_{survey_id}",
                "job_type": "archive_survey",
                "agent_name": record.get("author", ""),
                "title": record.get("title") or f"Archive Survey #{survey_id}",
                "status": SURVEY_STATUS_RUNNING,
                "top_level_status": SURVEY_STATUS_RUNNING,
                "execution_source": "surveyor",
                "system_baseline": False,
                "elapsed_seconds": max(0, int(now - started)),
                "started_timestamp": started,
            })
        queued_jobs = []
        try:
            pending_entries = _load_pending_entries(self.paths)
        except Exception:
            pending_entries = []
        active_ids = set(self.active_sessions)
        for entry in pending_entries:
            survey_id = str(entry.get("id") or entry.get("survey_id") or "").strip()
            if not survey_id or survey_id in active_ids:
                continue
            queued_jobs.append({
                "evaluation_id": f"Archive Survey #{survey_id}",
                "job_id": f"archive_survey_{survey_id}",
                "job_type": "archive_survey",
                "agent_name": "",
                "title": f"Archive Survey #{survey_id}",
                "status": SURVEY_STATUS_QUEUED,
                "top_level_status": SURVEY_STATUS_QUEUED,
                "execution_source": "surveyor",
                "system_baseline": False,
                "elapsed_seconds": 0,
            })
        return {
            "running_count": len(running_jobs),
            "queued_count": len(queued_jobs),
            "running_jobs": running_jobs,
            "queued_jobs": queued_jobs,
        }

    def _surveyor_loop(self) -> None:
        while self.is_running:
            try:
                self.poll_cli_jobs()
                self._recover_stale_requests(restart_recovery=False)
                self._repair_pending_notifications()
                self._launch_queued_surveys()
                self._wake_event.wait(self.check_interval)
                self._wake_event.clear()
            except Exception as exc:
                print(f"AutoArchiveSurveyor: unhandled loop error: {exc}")
                traceback.print_exc()
                self._wake_event.wait(self.check_interval * 2)
                self._wake_event.clear()

    @staticmethod
    def _pid_exists(pid: Any) -> bool:
        if pid in (None, "", 0):
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ValueError, TypeError):
            return False

    def _report_path(self, survey_id: str) -> str:
        return os.path.join(self.paths.reports_dir, f"{survey_id}.md")

    def _draft_path(self, survey_id: str) -> str:
        return os.path.join(self.paths.reports_dir, f"{survey_id}.draft.md")

    def _load_report_text(self, survey_id: str) -> str:
        path = self._report_path(survey_id)
        if not file_io_utils.file_exists(path):
            return ""
        try:
            return file_io_utils.load_text(path) or ""
        except UnicodeDecodeError as exc:
            print(
                f"AutoArchiveSurveyor: report for request {survey_id} is not valid UTF-8 "
                f"at byte {exc.start}; replacing malformed bytes for delivery."
            )
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read()

    def _queued_survey_ids(self) -> List[str]:
        ids: List[str] = []
        seen = set()
        for entry in _load_pending_entries(self.paths):
            survey_id = str(entry.get("id") or entry.get("survey_id") or "").strip()
            if not survey_id or survey_id in seen:
                continue
            record = _load_request(self.paths, survey_id)
            session = record.get("session") or {} if isinstance(record, dict) else {}
            status = str(record.get("status", "")).lower() if isinstance(record, dict) else ""
            session_status = str(session.get("status", "")).lower()
            if status == SURVEY_STATUS_QUEUED or (status == SURVEY_STATUS_RUNNING and session_status == "pending_resume"):
                ids.append(survey_id)
                seen.add(survey_id)

        for survey_id in _iter_request_ids(self.paths):
            if survey_id in seen:
                continue
            record = _load_request(self.paths, survey_id)
            session = record.get("session") or {} if isinstance(record, dict) else {}
            status = str(record.get("status", "")).lower() if isinstance(record, dict) else ""
            session_status = str(session.get("status", "")).lower()
            if status == SURVEY_STATUS_QUEUED or (status == SURVEY_STATUS_RUNNING and session_status == "pending_resume"):
                ids.append(survey_id)
                seen.add(survey_id)
        return sorted(ids, key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)))

    def _recover_stale_requests(self, *, restart_recovery: bool = False) -> None:
        for survey_id in _iter_request_ids(self.paths):
            if survey_id in self.active_sessions:
                continue
            record = _load_request(self.paths, survey_id)
            if not isinstance(record, dict):
                continue
            status = str(record.get("status", "")).lower()
            if status == SURVEY_STATUS_BLOCKED:
                if restart_recovery and not self._load_report_text(survey_id).strip():
                    self._mark_queued(
                        survey_id,
                        "Recovered after restart: unfinished Archive Surveyor request requeued.",
                        reset_attempt_counters=True,
                    )
                continue
            # Historical failures predate coder-style blocked/restart
            # recovery.  Never reopen them automatically on upgrade.
            if status == SURVEY_STATUS_FAILED:
                continue
            if status != SURVEY_STATUS_RUNNING:
                continue
            session = record.get("session") or {}
            pid = session.get("active_pid")
            if self._pid_exists(pid):
                continue

            # While the station remains alive, a pending same-session resume
            # is left for the normal launcher.  Across station restart, match
            # the Research Coder and restart from a fresh queued session.
            if str(session.get("status", "")).lower() == "pending_resume" and not restart_recovery:
                continue

            report_text = self._load_report_text(survey_id)
            if report_text.strip():
                self._mark_completed(survey_id, exit_code=session.get("exit_code"), error=None)
                self._deliver_report_if_needed(survey_id)
                continue

            self._mark_queued(
                survey_id,
                "Recovered stale Surveyor session with no final report; fresh session requeued.",
                reset_attempt_counters=restart_recovery,
            )

    def _repair_pending_notifications(self) -> None:
        for survey_id in _iter_request_ids(self.paths):
            record = _load_request(self.paths, survey_id)
            if not isinstance(record, dict):
                continue
            if str(record.get("status", "")).lower() != SURVEY_STATUS_COMPLETED:
                continue
            notification = record.get("notification") or {}
            if notification.get("sent"):
                continue
            self._deliver_report_if_needed(survey_id)

    def _launch_queued_surveys(self) -> None:
        if len(self.active_sessions) >= self.max_parallel_workers:
            return
        for survey_id in self._queued_survey_ids():
            if len(self.active_sessions) >= self.max_parallel_workers:
                break
            if survey_id in self.active_sessions:
                continue
            try:
                launched = self._launch_survey(survey_id)
            except Exception as exc:
                reason = f"Failed to launch Archive Surveyor for request {survey_id}: {exc}"
                print(f"AutoArchiveSurveyor: {reason}")
                traceback.print_exc()
                self._mark_blocked(survey_id, reason)
                continue
            if launched:
                _remove_pending_entry(self.paths, survey_id)

    def _claim_launch(
        self,
        survey_id: str,
        session_id: str,
        backend: str,
        model_name: Optional[str],
        *,
        use_resume: bool = False,
    ) -> Optional[Dict[str, Any]]:
        lock = filelock.FileLock(_request_path(self.paths, survey_id) + ".lock", timeout=60)
        with lock:
            record = _load_request(self.paths, survey_id)
            if not isinstance(record, dict):
                return None
            session = record.setdefault("session", {})
            record_status = str(record.get("status", "")).lower()
            session_status = str(session.get("status", "")).lower()
            if use_resume:
                if record_status != SURVEY_STATUS_RUNNING or session_status != "pending_resume":
                    return None
            elif record_status != SURVEY_STATUS_QUEUED:
                return None
            if bool(session.get("active")):
                return None
            session["backend"] = backend
            session["model_name"] = model_name
            session["active"] = True
            session["session_id"] = session_id
            session["active_pid"] = None
            session["status"] = "resuming" if use_resume else SURVEY_STATUS_RUNNING
            if use_resume:
                session["resume_count"] = int(session.get("resume_count", 0)) + 1
            else:
                session["spawn_count"] = int(session.get("spawn_count", 0)) + 1
            session["started_timestamp"] = time.time()
            session["completed_timestamp"] = None
            session["exit_code"] = None
            session["last_error"] = None
            session["next_resume_timestamp"] = None
            session["resume_delay_seconds"] = 0
            record["status"] = SURVEY_STATUS_RUNNING
            _save_request(self.paths, survey_id, record)
            return dict(record)

    def _launch_survey(self, survey_id: str) -> bool:
        return self.launch_cli_job(survey_id)

    def _load_cli_job_state(self, job_id: str) -> CliJobState:
        record = _load_request(self.paths, job_id) or {}
        session = record.get("session", {}) or {}
        top_status = str(record.get("status", "")).strip().lower()
        worker_status = str(session.get("status", "")).strip().lower()
        return CliJobState(
            backend=str(session.get("backend") or constants.ARCHIVE_SURVEY_BACKEND).lower(),
            spawn_count=int(session.get("spawn_count", 0)),
            resume_count=int(session.get("resume_count", 0)),
            max_spawns=int(session.get("max_spawns", constants.ARCHIVE_SURVEY_MAX_SPAWNS)),
            max_resumes=int(session.get("max_resumes", constants.ARCHIVE_SURVEY_MAX_RESUMES)),
            resume_token=str(session.get("resume_token") or "").strip() or None,
            next_resume_timestamp=session.get("next_resume_timestamp"),
            fresh_launch_eligible=top_status == SURVEY_STATUS_QUEUED,
            resume_launch_eligible=(
                top_status == SURVEY_STATUS_RUNNING and worker_status == "pending_resume"
            ),
        )

    def _claim_cli_job_launch(self, job_id, session_id, decision):
        state = self._load_cli_job_state(job_id)
        return self._claim_launch(
            job_id,
            session_id,
            state.backend,
            (_load_request(self.paths, job_id) or {}).get("session", {}).get("model_name"),
            use_resume=decision.is_resume,
        )

    def _build_cli_job_launch_spec(self, job_id, session_id, decision, claimed):
        session_state = claimed.get("session", {}) or {}
        backend = str(session_state.get("backend") or constants.ARCHIVE_SURVEY_BACKEND).lower()
        surveyor_root = os.path.abspath(self.paths.surveyor_root)
        station_data_root = os.path.abspath(os.path.join(surveyor_root, os.pardir, os.pardir, os.pardir))
        env = build_cli_worker_runtime_env(constants.RESEARCH_EVAL_PYTHON_CONDA_ENV)
        env["STATION_BASE_DATA_PATH"] = station_data_root
        if backend == "codex":
            apply_codex_proxy_overrides(env)
        index_db_path = index_paths.get_station_index_database_path(constants.BASE_STATION_DATA_PATH)
        return CliJobLaunchSpec(
            executable=detect_cli_worker_executable(backend, env),
            run_dir=os.path.abspath(os.path.join(self.paths.sessions_dir, session_id)),
            backend=backend,
            model_name=session_state.get("model_name"),
            workspace_root=surveyor_root,
            storage_root=self.paths.surveyor_root,
            prompt=self._build_prompt(claimed),
            env=env,
            extra_allowed_roots=(os.path.dirname(index_db_path),),
        )

    def _make_active_cli_job_session(self, job_id, base_session, decision, claimed, launch_metadata):
        return ActiveSurveySession(
            **vars(base_session),
            survey_id=job_id,
            report_path=self._report_path(job_id),
            draft_path=self._draft_path(job_id),
        )

    def _mark_cli_job_pid(self, job_id: str, pid: int) -> None:
        self._update_request(
            job_id,
            lambda record: record.setdefault("session", {}).update({"active_pid": pid}),
        )

    def _on_cli_job_started(self, job_id, session, decision):
        self._push_log_event(
            "archive_surveyor_started",
            {
                "survey_id": job_id,
                "session_id": session.session_id,
                "backend": session.backend,
                "pid": session.process.pid,
                "launch_mode": decision.mode,
            },
        )
        print(
            f"AutoArchiveSurveyor: Started survey {job_id} "
            f"(session={session.session_id}, pid={session.process.pid}, mode={decision.mode})"
        )

    def _on_cli_job_process_exited(self, job_id, session, returncode):
        print(
            f"AutoArchiveSurveyor: Surveyor process exited for request {job_id} "
            f"(session={session.session_id}, returncode={returncode})"
        )

    def _cli_job_completion_ready(self, job_id, session):
        return bool(self._load_report_text(job_id).strip())

    def _on_cli_job_completed(self, job_id, session, returncode):
        self._mark_completed(job_id, exit_code=returncode, error=None)
        self._deliver_report_if_needed(job_id)

    def _cli_job_missing_report_reason(self, job_id, session):
        reason = session.transcript_idle_timeout_reason or f"Archive Surveyor exited without producing reports/{job_id}.md."
        if file_io_utils.file_exists(session.draft_path):
            reason += f" Only reports/{job_id}.draft.md exists."
        return reason

    def _schedule_cli_job_resume(self, job_id, session, failure):
        def schedule_resume(record: Dict[str, Any]) -> None:
            state = record.setdefault("session", {})
            state["active"] = False
            state["active_pid"] = None
            state["status"] = "pending_resume"
            state["resume_token"] = failure.resume_token
            state["resume_count"] = failure.resume_count
            state["resume_delay_seconds"] = failure.delay_seconds
            state["next_resume_timestamp"] = failure.next_resume_timestamp
            state["exit_code"] = failure.returncode
            state["last_error"] = failure.reason
            record["status"] = SURVEY_STATUS_RUNNING

        self._update_request(job_id, schedule_resume)
        self._push_log_event(
            "archive_surveyor_retryable_infra_failure",
            {
                "survey_id": job_id,
                "session_id": getattr(session, "session_id", None),
                "reason": failure.reason,
                "launch_mode": "resume",
                "resume_count": failure.resume_count,
                "max_resumes": self._load_cli_job_state(job_id).max_resumes,
                "resume_delay_seconds": failure.delay_seconds,
                "next_resume_timestamp": failure.next_resume_timestamp,
            },
        )

    def _schedule_cli_job_fresh_attempt(self, job_id, session, failure):
        self._mark_queued(job_id, failure.reason)
        record = _load_request(self.paths, job_id) or {}
        file_io_utils.append_yaml_line(
            {
                "id": job_id,
                "author": record.get("author", "Unknown"),
                "submitted_tick": record.get("submitted_tick"),
                "requeued_timestamp": time.time(),
            },
            self.paths.pending_file,
        )

    def _on_cli_job_attempts_exhausted(
        self,
        job_id,
        session,
        failure,
    ):
        reason = failure.reason if session else f"Archive Surveyor request #{job_id} exhausted automatic recovery."
        self._mark_blocked(job_id, reason)

    def _before_cli_job_poll(self) -> None:
        self._check_session_timeouts()

    def _check_session_timeouts(self) -> None:
        if self.timeout_seconds <= 0:
            return
        now = time.time()
        for survey_id, session in list(self.active_sessions.items()):
            record = _load_request(self.paths, survey_id) or {}
            started = float((record.get("session") or {}).get("started_timestamp") or 0)
            if started <= 0 or now - started < self.timeout_seconds:
                continue
            self._terminate_cli_job_process(session, force=False)
            self._push_log_event("archive_surveyor_timeout", {"survey_id": survey_id, "timeout_seconds": self.timeout_seconds})

    def _cli_job_idle_timeout_reason(self, job_id, session, idle_seconds, timeout_seconds):
        return (
            f"Codex CLI transcript for Archive Survey #{job_id} did not grow for "
            f"{idle_seconds} seconds, exceeding the configured CLI worker "
            f"transcript idle timeout of {timeout_seconds} seconds."
        )

    def _on_cli_job_idle_timeout(self, job_id, session, reason, idle_seconds, timeout_seconds):
        self._push_log_event(
            "archive_surveyor_codex_transcript_idle_timeout",
            {
                "survey_id": job_id,
                "session_id": session.session_id,
                "idle_seconds": idle_seconds,
                "timeout_seconds": timeout_seconds,
            },
        )
        self._update_request(
            job_id,
            lambda record: record.setdefault("session", {}).update({"last_error": reason}),
        )

    def _update_request(self, survey_id: str, mutator) -> Optional[Dict[str, Any]]:
        lock = filelock.FileLock(_request_path(self.paths, survey_id) + ".lock", timeout=60)
        with lock:
            record = _load_request(self.paths, survey_id)
            if not isinstance(record, dict):
                return None
            mutator(record)
            _save_request(self.paths, survey_id, record)
            return record

    def _mark_queued(
        self,
        survey_id: str,
        reason: str,
        *,
        reset_attempt_counters: bool = False,
    ) -> None:
        def mutator(record: Dict[str, Any]) -> None:
            session = record.setdefault("session", {})
            session["active"] = False
            session["active_pid"] = None
            session["status"] = SURVEY_STATUS_QUEUED
            session["completed_timestamp"] = time.time()
            session["last_error"] = reason
            session["resume_token"] = None
            session["resume_count"] = 0
            session["next_resume_timestamp"] = None
            session["resume_delay_seconds"] = 0
            if reset_attempt_counters:
                session["spawn_count"] = 0
                notification = record.setdefault("notification", {})
                notification["sent"] = False
                notification["sent_timestamp"] = None
                notification["message"] = None
                notification["mail_capsule_id"] = None
                notification["mail_numeric_id"] = None
                record.pop("completed_tick", None)
                record.pop("completed_timestamp", None)
            record["status"] = SURVEY_STATUS_QUEUED
            record.pop("error", None)

        self._update_request(survey_id, mutator)

    def _mark_requeued_after_shutdown(self, survey_id: str) -> None:
        self._mark_queued(
            survey_id,
            "Recovered during station shutdown: Archive Surveyor request requeued.",
            reset_attempt_counters=True,
        )
        record = _load_request(self.paths, survey_id) or {}
        file_io_utils.append_yaml_line(
            {
                "id": survey_id,
                "author": record.get("author", "Unknown"),
                "submitted_tick": record.get("submitted_tick"),
                "requeued_timestamp": time.time(),
            },
            self.paths.pending_file,
        )

    def _mark_completed(self, survey_id: str, exit_code: Optional[int], error: Optional[str]) -> None:
        def mutator(record: Dict[str, Any]) -> None:
            session = record.setdefault("session", {})
            session["active"] = False
            session["active_pid"] = None
            session["status"] = SURVEY_STATUS_COMPLETED
            session["completed_timestamp"] = time.time()
            session["exit_code"] = exit_code
            session["last_error"] = error
            record["status"] = SURVEY_STATUS_COMPLETED
            record["completed_tick"] = self.station._get_current_tick() if hasattr(self.station, "_get_current_tick") else None
            record["completed_timestamp"] = time.time()

        self._update_request(survey_id, mutator)

    def _mark_blocked(self, survey_id: str, reason: str) -> None:
        def mutator(record: Dict[str, Any]) -> None:
            session = record.setdefault("session", {})
            session["active"] = False
            session["active_pid"] = None
            session["status"] = SURVEY_STATUS_BLOCKED
            session["completed_timestamp"] = time.time()
            session["last_error"] = reason
            record["status"] = SURVEY_STATUS_BLOCKED
            record["completed_tick"] = self.station._get_current_tick() if hasattr(self.station, "_get_current_tick") else None
            record["completed_timestamp"] = time.time()
            record["error"] = reason

        self._update_request(survey_id, mutator)
        self._pause_station(
            f"Archive Surveyor request #{survey_id} exhausted automatic recovery and requires restart/manual intervention: {reason}"
        )

    def _create_mail_capsule(self, author: str, survey_id: str, report_text: str) -> Optional[Dict[str, Any]]:
        current_tick = self.station._get_current_tick() if hasattr(self.station, "_get_current_tick") else 0
        surveyor_agent_data = {
            constants.AGENT_NAME_KEY: "Archive Surveyor",
            constants.AGENT_LINEAGE_KEY: "System",
            constants.AGENT_GENERATION_KEY: 0,
        }
        yaml_data = {
            constants.YAML_CAPSULE_TITLE: f"Archive Survey Report #{survey_id}",
            constants.YAML_CAPSULE_ABSTRACT: f"Archive Surveyor report for request #{survey_id}.",
            constants.YAML_CAPSULE_TAGS: ["archive-survey"],
            constants.YAML_CAPSULE_CONTENT: self._format_report_mail_content(report_text),
            constants.YAML_CAPSULE_RECIPIENTS: [author],
        }
        numeric_id, capsule_data = capsule_module.create_capsule(
            yaml_data,
            constants.CAPSULE_TYPE_MAIL,
            surveyor_agent_data,
            current_tick,
            None,
        )
        if not capsule_data:
            return None
        return {
            "numeric_id": numeric_id,
            "capsule_id": capsule_data.get(constants.CAPSULE_ID_KEY),
        }

    def _format_report_mail_content(self, report_text: str) -> str:
        return f"{SURVEY_REPORT_MAIL_NOTICE}\n\n{report_text}"

    def _deliver_mail_via_mail_room(self, author: str, mail_numeric_id: Any) -> Optional[str]:
        try:
            numeric_id = int(mail_numeric_id)
        except (TypeError, ValueError):
            return None

        rooms = getattr(self.station, "rooms", {}) or {}
        mail_room = rooms.get(constants.ROOM_MAIL) if hasattr(rooms, "get") else None
        room_context = getattr(self.station, "room_context", None)
        agent_module = getattr(self.station, "agent_module", None)
        if not mail_room or not room_context or not agent_module:
            return None
        if not hasattr(mail_room, "_deliver_mail_notification"):
            return None
        load_agent_data = getattr(agent_module, "load_agent_data", None)
        if not callable(load_agent_data) or not load_agent_data(author):
            return None

        mail_capsule = capsule_module.get_capsule(
            numeric_id,
            constants.CAPSULE_TYPE_MAIL,
            None,
            include_deleted_messages=True,
        )
        if not mail_capsule:
            return None

        mail_numeric_id_text = str(mail_numeric_id)
        message_content = ""
        first_message_id = ""
        if mail_capsule.get(constants.CAPSULE_MESSAGES_KEY):
            first_message = mail_capsule[constants.CAPSULE_MESSAGES_KEY][0]
            message_content = first_message.get(constants.MESSAGE_CONTENT_KEY, "")
            first_message_id = first_message.get(constants.MESSAGE_ID_KEY, "")
        message = format_new_mail_notification(
            "Archive Surveyor",
            mail_numeric_id_text,
            mail_capsule.get(constants.CAPSULE_TITLE_KEY, f"Archive Survey Report"),
            message_content,
            constants,
        )
        read_item_ids = []
        capsule_id = mail_capsule.get(constants.CAPSULE_ID_KEY)
        if capsule_id:
            read_item_ids.append(str(capsule_id))
        if first_message_id:
            read_item_ids.append(str(first_message_id))
        delivered = mail_room._deliver_mail_notification(
            author,
            message,
            room_context,
            read_item_ids=read_item_ids,
        )
        return message if delivered else None

    def _deliver_report_if_needed(self, survey_id: str) -> bool:
        record = _load_request(self.paths, survey_id)
        if not isinstance(record, dict):
            return False
        if str(record.get("parallel_commit_status") or "").lower() == "provisional":
            return False
        if str(record.get("status", "")).lower() != SURVEY_STATUS_COMPLETED:
            return False
        notification = record.setdefault("notification", {})
        if notification.get("sent"):
            return True

        author = record.get("author")
        report_text = self._load_report_text(survey_id)
        if not author or not report_text.strip():
            return False

        mail_capsule_id = notification.get("mail_capsule_id")
        mail_numeric_id = notification.get("mail_numeric_id")
        if not mail_capsule_id or mail_numeric_id is None:
            mail_result = self._create_mail_capsule(str(author), survey_id, report_text)
            if not mail_result:
                print(f"AutoArchiveSurveyor: failed to create mail capsule for survey {survey_id}")
                return False

            def mark_mail(record_local: Dict[str, Any]) -> None:
                notification_local = record_local.setdefault("notification", {})
                notification_local["mail_capsule_id"] = mail_result["capsule_id"]
                notification_local["mail_numeric_id"] = mail_result["numeric_id"]

            self._update_request(survey_id, mark_mail)
            mail_capsule_id = mail_result["capsule_id"]
            mail_numeric_id = mail_result["numeric_id"]

        try:
            message = self._deliver_mail_via_mail_room(str(author), mail_numeric_id)
            sent = bool(message)
        except Exception as exc:
            print(f"AutoArchiveSurveyor: failed to notify {author} for survey {survey_id}: {exc}")
            sent = False
        if not sent:
            return False

        def mark_sent(record_local: Dict[str, Any]) -> None:
            notification_local = record_local.setdefault("notification", {})
            notification_local["sent"] = True
            notification_local["sent_timestamp"] = time.time()
            notification_local["message"] = message
            notification_local["mail_capsule_id"] = mail_capsule_id
            notification_local["mail_numeric_id"] = mail_numeric_id
            notification_local[SURVEY_NOTIFICATION_MAIL_READ_REPAIRED_KEY] = True
            notification_local[SURVEY_NOTIFICATION_MAIL_READ_REPAIRED_TIMESTAMP_KEY] = time.time()

        self._update_request(survey_id, mark_sent)
        self._push_log_event("archive_surveyor_completed", {"survey_id": survey_id, "author": author})
        return True

    def _load_archive_preview(self) -> str:
        return build_archive_preview(self.paths.archive_target)

    def _load_question_preview(self) -> str:
        try:
            question_capsules_dir = self.paths.question_target
            if not os.path.isdir(question_capsules_dir):
                return "No Question Room problems currently available."
            capsule_files = []
            for filename in os.listdir(question_capsules_dir):
                if not (filename.startswith("question_") and filename.endswith(constants.YAML_EXTENSION)):
                    continue
                try:
                    capsule_id = int(filename.split("_", 1)[1].split(".", 1)[0])
                except (IndexError, ValueError):
                    continue
                capsule_files.append((capsule_id, filename))
            capsule_files.sort(key=lambda item: item[0])

            previews: List[str] = []
            for capsule_id, filename in capsule_files:
                capsule_data = file_io_utils.load_yaml(os.path.join(question_capsules_dir, filename))
                if not isinstance(capsule_data, dict):
                    continue
                if capsule_data.get(constants.CAPSULE_IS_DELETED_KEY, False):
                    continue
                title = capsule_data.get(constants.CAPSULE_TITLE_KEY, "Untitled")
                author = capsule_data.get(constants.CAPSULE_AUTHOR_NAME_KEY, "Unknown")
                status = capsule_data.get(constants.QUESTION_STATUS_KEY, constants.QUESTION_STATUS_PENDING)
                net_upvote = capsule_data.get(constants.QUESTION_NET_UPVOTE_KEY, 0)
                abstract = capsule_data.get(constants.CAPSULE_ABSTRACT_KEY, "")
                solved_by = capsule_data.get(constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY)
                solved_line = f"\nSolved By: {solved_by}" if solved_by else ""
                preview = (
                    f"**Question #{capsule_id}: {title}**\n"
                    f"Author: {author}, Status: {status}, Net Upvote: {net_upvote}{solved_line}\n"
                    f"Abstract: {abstract if abstract else '(No abstract available.)'}"
                )
                previews.append(preview)
            return "\n\n---\n\n".join(previews) if previews else "No Question Room problems currently available."
        except Exception as exc:
            print(f"AutoArchiveSurveyor: failed to load Question Room preview: {exc}")
            return "Error loading Question Room preview."

    def _build_prompt(self, record: Dict[str, Any]) -> str:
        survey_id = str(record.get("id"))
        task_spec = load_task_spec_markdown(constants).strip() or "No research task spec is available."
        question_room_access = bool(record.get("question_room_access", False))
        return build_archive_survey_prompt(
            survey_id=survey_id,
            requester_prompt=str(record.get("prompt", "")).strip(),
            workspace_root=self.paths.surveyor_root,
            task_spec=task_spec,
            archive_preview=self._load_archive_preview(),
            question_room_access=question_room_access,
            question_preview=self._load_question_preview() if question_room_access else "",
        )
