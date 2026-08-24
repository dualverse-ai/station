"""Persistent, dashboard-owned Archive Surveyor service.

This subsystem intentionally does not use Station's Archive Surveyor queue,
tick waiting, agent notifications, Running Jobs, or SQLite database.  It owns
its persistence and worker lease under ``station_data/web_interface`` while
reusing the shared CLI job engine.
"""

from __future__ import annotations

import glob
import os
import shutil
import sqlite3
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import filelock

from station import constants, file_io_utils, index_paths
from station.eval_archive.survey_worker import (
    ActiveSurveySession,
    build_archive_survey_prompt,
    ensure_survey_source_link,
)
from station.eval_research.runtime_paths import get_research_root
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


WEB_SURVEY_STATUS_QUEUED = "queued"
WEB_SURVEY_STATUS_RUNNING = "running"
WEB_SURVEY_STATUS_COMPLETED = "completed"
WEB_SURVEY_STATUS_FAILED = "failed"

_WEB_SURVEY_ROOT_PARTS = ("web_interface", "archive_surveyor")
_WEB_SURVEY_INDEX_FILENAME = "web_archive_surveys.sqlite3"
_WEB_SURVEY_MAX_ACTIVE_PER_USER = 2
_WEB_SURVEY_MAX_PROMPT_CHARS = 30000
_WEB_SURVEY_LIST_LIMIT = 1000
_WEB_SURVEY_SCHEMA_VERSION = 2
_WEB_SURVEY_AGENTS_MD = (
    "Follow the initial Archive Survey prompt as your authority. This workspace contains "
    "read-only Station research sources and your report directory; do not inspect Station source "
    "code or developer documentation.\n"
)
_WEB_SURVEY_AUDIENCE = "human_expert"


# Dashboard-facing additions to the default Station Surveyor prompt. Keeping this
# override here leaves the normal agent prompt and policy entirely owned by the
# Station Surveyor module.
WEB_ARCHIVE_SURVEY_PROMPT_OVERRIDES = {
    "requester_description": (
        "You are a PhD-level research surveyor answering an external human expert who is reviewing "
        "the accumulated work of Station, a multi-agent research environment. Use Station-local "
        "evidence and your own technical judgment to answer the human's question."
    ),
    "role_boundary": """Audience and reasoning contract:
- Write for a technically sophisticated human who may know the research field but does not know Station's internal vocabulary.
- Avoid internal Station shorthand where ordinary field language is clearer. Define Station-specific or newly introduced terms on first use. Common field terminology that an expert should know does not need elementary explanation.
- Explain what an `Archive #ID` paper, an `Eval #ID` experiment, or a `Question #ID` thread establishes before relying on it. Translate agent names, ticks, capsules, scores, compiler nicknames, and internal workflow labels into their scientific meaning when they matter.
- Assume the reader will usually not open cited papers, evaluations, or Question Room threads. Make the report self-contained by including the definitions, formulas, assumptions, experimental setup, results, qualifications, and reasoning needed to understand every important conclusion. Citations provide traceability; they are not a substitute for explanation.
- You may brainstorm, propose, compare, and critique new research ideas, hypotheses, experiments, mechanisms, or next directions when the request calls for it. Clearly separate evidence-backed findings from your own suggestions and label speculative proposals as such.
- Prefer a coherent expert narrative over a dump of internal logs. Preserve exact citations so the reader can trace every Station-local claim.""",
    "request_noun": "human expert request",
    "requester_term": "human expert",
    "general_request_guidance": (
        "For a broad landscape request, the archive is usually sufficient; use evaluation-level scanning "
        "when the request asks for specific technical evidence or when archive evidence is clearly too sparse."
    ),
    "review_checks": "completeness, citation accuracy, accessibility to the intended human audience, and formatting",
    "guidelines_intro": (
        "Your overall goal is to help an external human expert understand and critically use the accumulated "
        "knowledge of Station."
    ),
    "analysis_guideline": (
        "Do your own analysis and integration rather than merely retrieving records. Analyze common themes, "
        "duplicate risk, assumptions, evidence gaps, underexplored regimes, and areas where existing work "
        "appears weak or saturated."
    ),
    "idea_guidelines": """- When useful, turn negative results into design principles and propose technically plausible ways forward.
- If brainstorming, give the scientific rationale, assumptions, expected information gain, and main failure mode of each serious proposal rather than listing shallow ideas.
- Do not treat Station conventions as universal field conventions. Explain local objectives, score meanings, constructions, and constraints before drawing conclusions.""",
    "novelty_guideline": (
        '"Novel" means novel with respect to Station archive papers and Station evaluation records unless '
        "you explicitly state a broader literature claim."
    ),
    "focus_guideline": "Stay focused on what the human asked and be clear about uncertainty.",
    "idea_rules": (
        "- You may offer new ideas and recommendations, but distinguish them explicitly from claims "
        "supported by Station evidence."
    ),
    "evidence_distinction_rule": (
        "Distinguish evidence-backed claims from hypotheses, guesses, and your own suggestions."
    ),
    "report_title": "# Archive Survey Report #{survey_id}",
    "request_restatement": "Briefly restate the request in scientifically clear language.",
    "executive_summary_instruction": "Give a concise answer to the main question.",
    "main_content_examples": (
        "prior work, duplicate-risk analysis, evidence gaps, frontier summary, assumptions to challenge, "
        "or proposed research ideas"
    ),
    "main_content_prompt_qualifier": "the human's prompt",
    "limitations_instruction": "State missing evidence, scope limits, or uncertainty.",
    "request_label": "HUMAN EXPERT REQUEST",
    "question_work_cycle": (
        "\n5. If the human expert asks about open problems, solved problems, pending questions, or "
        "Question Room discussions, scan the Question Room Preview below and read relevant full question "
        "files with commands such as `cat question_room/question_15.yaml`."
    ),
}


class WebArchiveSurveyError(RuntimeError):
    pass


class WebArchiveSurveyBusyError(WebArchiveSurveyError):
    pass


class WebArchiveSurveyNotFoundError(WebArchiveSurveyError):
    pass


@dataclass(frozen=True)
class WebArchiveSurveyPaths:
    root: str
    requests_dir: str
    reports_dir: str
    sessions_dir: str
    sources_dir: str
    index_dir: str
    index_db_path: str
    submission_lock_path: str
    worker_lock_path: str
    archive_link: str
    research_link: str
    question_link: str
    archive_target: str
    research_target: str
    question_target: str


def build_web_archive_survey_templates() -> List[Dict[str, str]]:
    return [
        {
            "id": "open_question",
            "label": "Open Question",
            "prompt": (
                "Answer the following question using relevant Station archive papers, Research Center "
                "evaluations, and Question Room discussions. Explain Station-specific terms for an external "
                "human expert, separate established evidence from speculation, and cite Archive #ID, Eval #ID, "
                "and Question #ID when relevant.\n\n"
                "Question:\n"
            ),
        },
        {
            "id": "landscape",
            "label": "Research Landscape",
            "prompt": (
                "Survey the current research landscape represented by Station. Identify the main "
                "scientific approaches, strongest results, recurring failure mechanisms, disagreements, "
                "and important open gaps, including relevant Question Room discussions. Write for an external "
                "expert and cite the supporting Archive #ID, Eval #ID, and Question #ID records."
            ),
        },
        {
            "id": "related_work",
            "label": "Find Related Work",
            "prompt": (
                "Find and synthesize Station archive papers and evaluations related to the topic below. "
                "Explain what has been established, what was only diagnostic or partial, and what remains "
                "uncertain. Define local Station terminology when it first appears.\n\nTopic:\n"
            ),
        },
        {
            "id": "compare_papers",
            "label": "Compare Papers",
            "prompt": (
                "Compare the Station archive papers identified below by ID or title. Explain their goals, "
                "methods, evidence, assumptions, overlaps, differences, and unresolved issues for an "
                "external human expert. Cite each paper as Archive #ID and use Eval #ID evidence when it "
                "materially clarifies the comparison.\n\nPapers to compare:\n"
            ),
        },
        {
            "id": "brainstorm",
            "label": "Evidence + New Ideas",
            "prompt": (
                "Using Station archive papers and evaluations as evidence, assess the direction below and "
                "then propose a small set of technically serious new ideas or experiments. Clearly label "
                "which conclusions are supported by Station evidence and which are your own proposals. "
                "For each proposal, explain its rationale, assumptions, expected information gain, and "
                "main failure mode.\n\nDirection:\n"
            ),
        },
    ]


def _clean_prompt(value: Any) -> str:
    prompt = str(value or "").strip()
    if not prompt:
        raise ValueError("Survey request cannot be empty.")
    if len(prompt) > _WEB_SURVEY_MAX_PROMPT_CHARS:
        raise ValueError(f"Survey request is too long; maximum {_WEB_SURVEY_MAX_PROMPT_CHARS} characters.")
    return prompt


def _clean_owner(value: Any) -> str:
    owner = str(value or "dashboard").strip()
    return owner[:160] or "dashboard"


def _clean_selected_archive_ids(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    selected: List[int] = []
    for raw_id in value:
        try:
            numeric_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if numeric_id > 0 and numeric_id not in selected:
            selected.append(numeric_id)
    return sorted(selected)[:200]


def _prompt_preview(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def ensure_web_archive_survey_layout(base_data_path: Optional[str] = None) -> WebArchiveSurveyPaths:
    base_root = os.path.abspath(base_data_path or constants.BASE_STATION_DATA_PATH)
    root = os.path.join(base_root, *_WEB_SURVEY_ROOT_PARTS)
    archive_target = os.path.join(
        base_root,
        constants.CAPSULES_DIR_NAME,
        constants.ARCHIVE_CAPSULES_SUBDIR_NAME,
    )
    research_target = get_research_root(constants) if base_data_path is None else os.path.join(
        base_root,
        constants.ROOMS_DIR_NAME,
        constants.SHORT_ROOM_NAME_RESEARCH,
    )
    question_target = os.path.join(
        base_root,
        constants.CAPSULES_DIR_NAME,
        constants.QUESTION_CAPSULES_SUBDIR_NAME,
    )
    paths = WebArchiveSurveyPaths(
        root=root,
        requests_dir=os.path.join(root, "requests"),
        reports_dir=os.path.join(root, "reports"),
        sessions_dir=os.path.join(root, "sessions"),
        sources_dir=os.path.join(root, "sources"),
        index_dir=os.path.join(root, "index"),
        index_db_path=os.path.join(root, "index", _WEB_SURVEY_INDEX_FILENAME),
        submission_lock_path=os.path.join(root, ".submission.lock"),
        worker_lock_path=os.path.join(root, ".worker.lock"),
        archive_link=os.path.join(root, "archive_papers"),
        research_link=os.path.join(root, "research_center"),
        question_link=os.path.join(root, "question_room"),
        archive_target=archive_target,
        research_target=research_target,
        question_target=question_target,
    )
    for directory in (
        paths.root,
        paths.requests_dir,
        paths.reports_dir,
        paths.sessions_dir,
        paths.sources_dir,
        paths.index_dir,
    ):
        file_io_utils.ensure_dir_exists(directory)
    agents_path = os.path.join(paths.root, "AGENTS.md")
    if file_io_utils.load_text(agents_path) != _WEB_SURVEY_AGENTS_MD:
        file_io_utils.save_text(_WEB_SURVEY_AGENTS_MD, agents_path)
    ensure_survey_source_link(paths.archive_link, paths.archive_target, log_prefix="WebArchiveSurveyor")
    ensure_survey_source_link(paths.research_link, paths.research_target, log_prefix="WebArchiveSurveyor")
    ensure_survey_source_link(paths.question_link, paths.question_target, log_prefix="WebArchiveSurveyor")
    return paths


class WebArchiveSurveyStore:
    """YAML/report source of truth with a separate SQLite list/queue index."""

    def __init__(self, base_data_path: Optional[str] = None):
        self.base_data_path = os.path.abspath(base_data_path or constants.BASE_STATION_DATA_PATH)
        self.paths = ensure_web_archive_survey_layout(self.base_data_path)
        index_existed = os.path.isfile(self.paths.index_db_path)
        self._initialize_index()
        if not index_existed:
            self._rebuild_index_from_requests()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.paths.index_db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize_index(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS surveys (
                    id INTEGER PRIMARY KEY,
                    owner TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prompt_preview TEXT NOT NULL,
                    selected_archive_ids TEXT NOT NULL DEFAULT '',
                    source_tick INTEGER,
                    submitted_timestamp REAL NOT NULL,
                    started_timestamp REAL,
                    completed_timestamp REAL,
                    updated_timestamp REAL NOT NULL,
                    report_path TEXT,
                    error TEXT,
                    active_pid INTEGER,
                    session_id TEXT,
                    spawn_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_web_surveys_owner_submitted
                    ON surveys(owner, submitted_timestamp DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_web_surveys_status_submitted
                    ON surveys(status, submitted_timestamp ASC, id ASC);
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(_WEB_SURVEY_SCHEMA_VERSION),),
            )

    def _request_path(self, survey_id: Any) -> str:
        return os.path.join(self.paths.requests_dir, f"web_survey_{int(survey_id)}.yaml")

    def _report_path(self, survey_id: Any) -> str:
        return os.path.join(self.paths.reports_dir, f"web_{int(survey_id)}.md")

    def _draft_path(self, survey_id: Any) -> str:
        return os.path.join(self.paths.reports_dir, f"web_{int(survey_id)}.draft.md")

    def _load_record(self, survey_id: Any) -> Optional[Dict[str, Any]]:
        data = file_io_utils.load_yaml(self._request_path(survey_id))
        return data if isinstance(data, dict) else None

    def _save_record(self, record: Dict[str, Any]) -> None:
        file_io_utils.save_yaml(record, self._request_path(record["id"]), sort_keys=False)

    @staticmethod
    def _selected_ids_text(record: Dict[str, Any]) -> str:
        return ",".join(str(value) for value in _clean_selected_archive_ids(record.get("selected_archive_ids")))

    def _index_values(self, record: Dict[str, Any]) -> tuple[Any, ...]:
        session = record.get("session") if isinstance(record.get("session"), dict) else {}
        return (
            int(record["id"]),
            _clean_owner(record.get("owner")),
            str(record.get("status") or WEB_SURVEY_STATUS_QUEUED),
            _prompt_preview(record.get("prompt")),
            self._selected_ids_text(record),
            record.get("source_tick"),
            float(record.get("submitted_timestamp") or time.time()),
            session.get("started_timestamp"),
            record.get("completed_timestamp"),
            float(record.get("updated_timestamp") or time.time()),
            os.path.relpath(self._report_path(record["id"]), self.paths.root),
            record.get("error") or session.get("last_error"),
            session.get("active_pid"),
            session.get("session_id"),
            int(session.get("spawn_count") or 0),
        )

    def _upsert_index(self, connection: sqlite3.Connection, record: Dict[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO surveys(
                id, owner, status, prompt_preview, selected_archive_ids, source_tick,
                submitted_timestamp, started_timestamp, completed_timestamp,
                updated_timestamp, report_path, error, active_pid, session_id, spawn_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                owner=excluded.owner,
                status=excluded.status,
                prompt_preview=excluded.prompt_preview,
                selected_archive_ids=excluded.selected_archive_ids,
                source_tick=excluded.source_tick,
                submitted_timestamp=excluded.submitted_timestamp,
                started_timestamp=excluded.started_timestamp,
                completed_timestamp=excluded.completed_timestamp,
                updated_timestamp=excluded.updated_timestamp,
                report_path=excluded.report_path,
                error=excluded.error,
                active_pid=excluded.active_pid,
                session_id=excluded.session_id,
                spawn_count=excluded.spawn_count
            """,
            self._index_values(record),
        )

    def _rebuild_index_from_requests(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM surveys")
            for filename in file_io_utils.list_files(self.paths.requests_dir, constants.YAML_EXTENSION):
                if not filename.startswith("web_survey_"):
                    continue
                record = file_io_utils.load_yaml(os.path.join(self.paths.requests_dir, filename))
                if isinstance(record, dict) and str(record.get("id") or "").isdigit():
                    self._upsert_index(connection, record)
            next_id = int(connection.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM surveys").fetchone()[0])
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('next_survey_id', ?)",
                (str(next_id),),
            )

    def create(
        self,
        *,
        owner: str,
        prompt: Any,
        selected_archive_ids: Any,
        source_tick: Optional[int],
        task_spec_snapshot: str,
        archive_preview_snapshot: str,
        question_preview_snapshot: str = "",
    ) -> Dict[str, Any]:
        cleaned_prompt = _clean_prompt(prompt)
        cleaned_owner = _clean_owner(owner)
        selected_ids = _clean_selected_archive_ids(selected_archive_ids)
        lock = filelock.FileLock(self.paths.submission_lock_path, timeout=60)
        with lock, self._connect() as connection:
            active_count = connection.execute(
                "SELECT COUNT(*) FROM surveys WHERE owner=? AND status IN (?, ?)",
                (cleaned_owner, WEB_SURVEY_STATUS_QUEUED, WEB_SURVEY_STATUS_RUNNING),
            ).fetchone()[0]
            if int(active_count) >= _WEB_SURVEY_MAX_ACTIVE_PER_USER:
                raise WebArchiveSurveyBusyError(
                    f"You already have {active_count} pending or running web survey request(s)."
                )
            next_id_row = connection.execute(
                "SELECT value FROM metadata WHERE key='next_survey_id'"
            ).fetchone()
            next_id = int(next_id_row[0]) if next_id_row else int(
                connection.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM surveys").fetchone()[0]
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('next_survey_id', ?)",
                (str(next_id + 1),),
            )
            now = time.time()
            record = {
                "schema_version": _WEB_SURVEY_SCHEMA_VERSION,
                "id": next_id,
                "owner": cleaned_owner,
                "audience": _WEB_SURVEY_AUDIENCE,
                "prompt": cleaned_prompt,
                "selected_archive_ids": selected_ids,
                "source_tick": source_tick,
                "task_spec_snapshot": str(task_spec_snapshot or "").strip() or "No research task spec is available.",
                "archive_preview_snapshot": str(archive_preview_snapshot or "").strip() or "No archive papers currently available.",
                "question_preview_snapshot": str(question_preview_snapshot or "").strip() or "No Question Room problems currently available.",
                "submitted_timestamp": now,
                "updated_timestamp": now,
                "completed_timestamp": None,
                "status": WEB_SURVEY_STATUS_QUEUED,
                "error": None,
                "session": {
                    "backend": constants.ARCHIVE_SURVEY_BACKEND,
                    "model_name": constants.ARCHIVE_SURVEY_MODEL_NAME,
                    "active": False,
                    "active_pid": None,
                    "session_id": None,
                    "status": WEB_SURVEY_STATUS_QUEUED,
                    "spawn_count": 0,
                    "max_spawns": constants.ARCHIVE_SURVEY_MAX_SPAWNS,
                    "resume_count": 0,
                    "max_resumes": constants.ARCHIVE_SURVEY_MAX_RESUMES,
                    "resume_token": None,
                    "resume_delay_seconds": 0,
                    "next_resume_timestamp": None,
                    "started_timestamp": None,
                    "completed_timestamp": None,
                    "exit_code": None,
                    "last_error": None,
                },
            }
            self._save_record(record)
            self._upsert_index(connection, record)
            return record

    def update(self, survey_id: Any, mutator) -> Dict[str, Any]:
        survey_id = int(survey_id)
        lock = filelock.FileLock(self._request_path(survey_id) + ".lock", timeout=60)
        with lock:
            record = self._load_record(survey_id)
            if not record:
                raise WebArchiveSurveyNotFoundError(f"Web survey #{survey_id} was not found.")
            mutator(record)
            record["updated_timestamp"] = time.time()
            self._save_record(record)
            with self._connect() as connection:
                self._upsert_index(connection, record)
            return record

    def list(self, owner: str, limit: int = _WEB_SURVEY_LIST_LIMIT) -> List[Dict[str, Any]]:
        effective_limit = max(1, min(int(limit or _WEB_SURVEY_LIST_LIMIT), _WEB_SURVEY_LIST_LIMIT))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM surveys WHERE owner=?
                ORDER BY submitted_timestamp DESC, id DESC LIMIT ?
                """,
                (_clean_owner(owner), effective_limit),
            ).fetchall()
        return [self._summary_from_row(row) for row in rows]

    def list_by_status(self, statuses: List[str], limit: int = _WEB_SURVEY_LIST_LIMIT) -> List[Dict[str, Any]]:
        cleaned = [str(status).strip().lower() for status in statuses if str(status).strip()]
        if not cleaned:
            return []
        placeholders = ",".join("?" for _ in cleaned)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM surveys WHERE status IN ({placeholders}) ORDER BY submitted_timestamp ASC, id ASC LIMIT ?",
                (*cleaned, max(1, int(limit))),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, survey_id: Any, owner: Optional[str] = None, include_report: bool = True) -> Dict[str, Any]:
        survey_id = int(survey_id)
        record = self._load_record(survey_id)
        if not record or (owner is not None and _clean_owner(record.get("owner")) != _clean_owner(owner)):
            raise WebArchiveSurveyNotFoundError(f"Web survey #{survey_id} was not found.")
        payload = dict(record)
        payload["report_markdown"] = ""
        if include_report and file_io_utils.file_exists(self._report_path(survey_id)):
            payload["report_markdown"] = file_io_utils.load_text(self._report_path(survey_id)) or ""
        return payload

    def delete(self, survey_id: Any, owner: str) -> None:
        survey_id = int(survey_id)
        lock = filelock.FileLock(self._request_path(survey_id) + ".lock", timeout=60)
        with lock, self._connect() as connection:
            record = self._load_record(survey_id)
            if not record or _clean_owner(record.get("owner")) != _clean_owner(owner):
                raise WebArchiveSurveyNotFoundError(f"Web survey #{survey_id} was not found.")
            if str(record.get("status") or "").lower() == WEB_SURVEY_STATUS_RUNNING:
                raise WebArchiveSurveyBusyError("A running survey cannot be removed. Wait for it to finish first.")
            connection.execute("DELETE FROM surveys WHERE id=? AND owner=?", (survey_id, _clean_owner(owner)))
            for path in (self._request_path(survey_id), self._report_path(survey_id), self._draft_path(survey_id)):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            source_root = os.path.join(self.paths.sources_dir, str(survey_id))
            if os.path.isdir(source_root):
                shutil.rmtree(source_root)
            for session_path in glob.glob(os.path.join(self.paths.sessions_dir, f"web_{survey_id}_*")):
                if os.path.isdir(session_path):
                    shutil.rmtree(session_path)

    def _summary_from_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        selected = []
        for value in str(row["selected_archive_ids"] or "").split(","):
            if value.isdigit():
                selected.append(int(value))
        return {
            "id": int(row["id"]),
            "status": row["status"],
            "prompt_preview": row["prompt_preview"],
            "selected_archive_ids": selected,
            "source_tick": row["source_tick"],
            "submitted_timestamp": row["submitted_timestamp"],
            "started_timestamp": row["started_timestamp"],
            "completed_timestamp": row["completed_timestamp"],
            "updated_timestamp": row["updated_timestamp"],
            "has_report": file_io_utils.file_exists(self._report_path(row["id"])),
            "error": row["error"],
            "spawn_count": row["spawn_count"],
        }


class WebArchiveSurveyService(CliJobManager):
    """Leader-locked background worker for dashboard survey requests."""
    fresh_attempt_after_resume_exhaustion = True

    def __init__(self, base_data_path: Optional[str] = None):
        super().__init__()
        self.store = WebArchiveSurveyStore(base_data_path)
        self.paths = self.store.paths
        self.max_workers = max(1, int(constants.WEB_ARCHIVE_SURVEY_MAX_PARALLEL_WORKERS or 1))
        self.timeout_seconds = float(constants.ARCHIVE_SURVEY_TIMEOUT_SECONDS or 0)
        self.check_interval = max(1.0, float(constants.ARCHIVE_SURVEY_CHECK_INTERVAL or 5.0))
        self._wake_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._leader_lock = filelock.FileLock(self.paths.worker_lock_path, thread_local=False)
        self._leader_acquired = False
        self._running = False
        self._start_guard = threading.Lock()

    def ensure_worker_started(self) -> bool:
        if not constants.ARCHIVE_SURVEY_ENABLED:
            return False
        with self._start_guard:
            if self._running and self._thread and self._thread.is_alive():
                return True
            try:
                self._leader_lock.acquire(timeout=0)
            except filelock.Timeout:
                return False
            self._leader_acquired = True
            self._running = True
            self._wake_event.clear()
            self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="web-archive-surveyor")
            self._thread.start()
            print("WebArchiveSurveyor: worker loop started")
            return True

    def wake(self) -> None:
        self._wake_event.set()

    def stop(self) -> None:
        with self._start_guard:
            if not self._running:
                return
            self._running = False
            self._wake_event.set()

            def finish_shutdown(survey_id, session):
                deadline = time.monotonic() + 5.0
                while session.process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
                if session.process.poll() is None:
                    self._terminate_cli_job_process(session, force=True)
                report_text = (
                    file_io_utils.load_text(session.report_path)
                    if file_io_utils.file_exists(session.report_path)
                    else ""
                )
                if (report_text or "").strip():
                    self._mark_completed(survey_id, exit_code=session.process.poll())
                else:
                    self._requeue_after_shutdown(survey_id)

            self.stop_active_cli_jobs(force=False, after_terminate=finish_shutdown)
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
            if self._leader_acquired:
                try:
                    self._leader_lock.release()
                except Exception:
                    pass
            self._leader_acquired = False
            print("WebArchiveSurveyor: worker loop stopped")

    def submit(
        self,
        *,
        owner: str,
        prompt: Any,
        selected_archive_ids: Any,
        source_tick: Optional[int],
        task_spec_snapshot: str,
        archive_preview_snapshot: str,
        question_preview_snapshot: str = "",
    ) -> Dict[str, Any]:
        record = self.store.create(
            owner=owner,
            prompt=prompt,
            selected_archive_ids=selected_archive_ids,
            source_tick=source_tick,
            task_spec_snapshot=task_spec_snapshot,
            archive_preview_snapshot=archive_preview_snapshot,
            question_preview_snapshot=question_preview_snapshot,
        )
        self.ensure_worker_started()
        self.wake()
        return record

    def _worker_loop(self) -> None:
        while self._running:
            try:
                self.poll_cli_jobs()
                self._recover_stale_requests()
                self._launch_queued_requests()
            except Exception as exc:
                print(f"WebArchiveSurveyor: worker loop error: {exc}")
                traceback.print_exc()
            self._wake_event.wait(self.check_interval)
            self._wake_event.clear()

    @staticmethod
    def _pid_exists(value: Any) -> bool:
        try:
            pid = int(value)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _recover_stale_requests(self) -> None:
        for row in self.store.list_by_status([WEB_SURVEY_STATUS_RUNNING]):
            survey_id = str(row["id"])
            if survey_id in self.active_sessions:
                continue
            record = self.store.get(survey_id, include_report=False)
            report_path = self.store._report_path(survey_id)
            if file_io_utils.file_exists(report_path) and (file_io_utils.load_text(report_path) or "").strip():
                self._mark_completed(survey_id, exit_code=(record.get("session") or {}).get("exit_code"))
                continue
            session = record.get("session") or {}
            if str(session.get("status") or "").lower() == "pending_resume":
                continue
            if self._pid_exists(session.get("active_pid")):
                continue
            if int(session.get("spawn_count") or 0) < int(session.get("max_spawns") or constants.ARCHIVE_SURVEY_MAX_SPAWNS):
                self._mark_queued(survey_id, "Recovered an interrupted web Surveyor session.")
            else:
                self._mark_failed(survey_id, "Web Archive Surveyor exhausted its retry limit without a final report.")

    def _launch_queued_requests(self) -> None:
        available = self.max_workers - len(self.active_sessions)
        if available <= 0:
            return
        rows = self.store.list_by_status(
            [WEB_SURVEY_STATUS_RUNNING, WEB_SURVEY_STATUS_QUEUED],
            limit=_WEB_SURVEY_LIST_LIMIT,
        )
        for row in rows:
            if len(self.active_sessions) >= self.max_workers:
                break
            survey_id = str(row["id"])
            if survey_id in self.active_sessions:
                continue
            try:
                self.launch_cli_job(survey_id)
            except WebArchiveSurveyNotFoundError:
                continue
            except Exception as exc:
                print(f"WebArchiveSurveyor: failed to launch survey {survey_id}: {exc}")
                traceback.print_exc()
                try:
                    record = self.store.get(survey_id, include_report=False)
                except WebArchiveSurveyNotFoundError:
                    continue
                spawn_count = int((record.get("session") or {}).get("spawn_count") or 0)
                max_spawns = int((record.get("session") or {}).get("max_spawns") or constants.ARCHIVE_SURVEY_MAX_SPAWNS)
                if spawn_count < max_spawns:
                    self._mark_queued(survey_id, str(exc))
                else:
                    self._mark_failed(survey_id, str(exc))

    def _snapshot_station_index(self, survey_id: str) -> str:
        source_path = index_paths.get_station_index_database_path(self.store.base_data_path)
        if not os.path.isfile(source_path):
            raise WebArchiveSurveyError("Station SQLite index is unavailable; rebuild the Station index first.")
        snapshot_root = os.path.join(self.paths.sources_dir, str(survey_id))
        file_io_utils.ensure_dir_exists(snapshot_root)
        snapshot_path = os.path.join(snapshot_root, "station_index.sqlite3")
        try:
            os.unlink(snapshot_path)
        except FileNotFoundError:
            pass
        source_uri = f"file:{quote(os.path.abspath(source_path))}?mode=ro"
        with sqlite3.connect(source_uri, uri=True, timeout=30) as source, sqlite3.connect(snapshot_path) as target:
            source.backup(target)
        return snapshot_path

    def _update_cli_job_record(self, job_id, session_updates, record_updates=None):
        def mutator(record):
            record.setdefault("session", {}).update(session_updates)
            record.update(record_updates or {})

        return self.store.update(job_id, mutator)

    @classmethod
    def _resume_backoff_schedule(cls) -> Any:
        return getattr(constants, "RESEARCH_CODER_RESUME_BACKOFF_SECONDS", [])

    def _load_cli_job_state(self, job_id: str) -> CliJobState:
        record = self.store.get(job_id, include_report=False)
        session = record.get("session") or {}
        top_status = str(record.get("status") or "").lower()
        session_status = str(session.get("status") or "").lower()
        return CliJobState(
            backend=str(session.get("backend") or constants.ARCHIVE_SURVEY_BACKEND).lower(),
            spawn_count=int(session.get("spawn_count") or 0),
            resume_count=int(session.get("resume_count") or 0),
            max_spawns=int(session.get("max_spawns") or constants.ARCHIVE_SURVEY_MAX_SPAWNS),
            max_resumes=int(session.get("max_resumes") or constants.ARCHIVE_SURVEY_MAX_RESUMES),
            resume_token=str(session.get("resume_token") or "").strip() or None,
            next_resume_timestamp=session.get("next_resume_timestamp"),
            fresh_launch_eligible=top_status == WEB_SURVEY_STATUS_QUEUED,
            resume_launch_eligible=(
                top_status == WEB_SURVEY_STATUS_RUNNING and session_status == "pending_resume"
            ),
        )

    def _format_cli_job_session_id(self, job_id, state, decision) -> str:
        return f"web_{job_id}_{state.backend}_spawn_{decision.spawn_count}_{uuid.uuid4().hex[:8]}"

    def _claim_cli_job_launch(self, job_id, session_id, decision) -> Dict[str, Any]:
        def mutator(record: Dict[str, Any]) -> None:
            top_status = str(record.get("status") or "").lower()
            session = record.setdefault("session", {})
            session_status = str(session.get("status") or "").lower()
            eligible = (
                top_status == WEB_SURVEY_STATUS_RUNNING and session_status == "pending_resume"
                if decision.is_resume
                else top_status == WEB_SURVEY_STATUS_QUEUED
            )
            if not eligible or bool(session.get("active")):
                raise WebArchiveSurveyBusyError(f"Web survey #{job_id} is not launchable.")
            session["active"] = True
            session["active_pid"] = None
            session["session_id"] = session_id
            session["status"] = "resuming" if decision.is_resume else WEB_SURVEY_STATUS_RUNNING
            session["spawn_count"] = decision.spawn_count
            session["resume_count"] = decision.resume_count
            session["resume_token"] = decision.resume_token
            session["resume_delay_seconds"] = 0
            session["next_resume_timestamp"] = None
            session["started_timestamp"] = time.time()
            session["completed_timestamp"] = None
            session["exit_code"] = None
            session["last_error"] = None
            record["status"] = WEB_SURVEY_STATUS_RUNNING
            record["error"] = None

        return self.store.update(job_id, mutator)

    def _build_cli_job_launch_spec(self, job_id, session_id, decision, claimed):
        session_state = claimed.get("session") or {}
        backend = str(session_state.get("backend") or constants.ARCHIVE_SURVEY_BACKEND).lower()
        model_name = session_state.get("model_name")
        snapshot_path = os.path.join(self.paths.sources_dir, str(job_id), "station_index.sqlite3")
        if not decision.is_resume or not os.path.isfile(snapshot_path):
            snapshot_path = self._snapshot_station_index(job_id)
        selected_ids = _clean_selected_archive_ids(claimed.get("selected_archive_ids"))
        selected_context = ""
        if selected_ids:
            selected_context = (
                "The dashboard user selected these Archive papers for special attention, without limiting the "
                "survey to them: "
                + ", ".join(f"Archive #{archive_id}" for archive_id in selected_ids)
                + ".\n\n"
            )
        prompt_overrides = dict(WEB_ARCHIVE_SURVEY_PROMPT_OVERRIDES)
        prompt_overrides["surveyor_identity"] = f"You are the Archive Surveyor for dashboard survey request #W{job_id}."
        prompt = build_archive_survey_prompt(
            survey_id=f"W{job_id}",
            report_basename=f"web_{job_id}",
            requester_prompt=selected_context + str(claimed.get("prompt") or "").strip(),
            workspace_root=self.paths.root,
            task_spec=str(claimed.get("task_spec_snapshot") or "").strip(),
            archive_preview=str(claimed.get("archive_preview_snapshot") or "").strip(),
            question_room_access=True,
            question_preview=str(
                claimed.get("question_preview_snapshot")
                or "Question Room preview was not captured for this older request; inspect question_room/ directly when relevant."
            ).strip(),
            prompt_overrides=prompt_overrides,
        )
        env = build_cli_worker_runtime_env(constants.RESEARCH_EVAL_PYTHON_CONDA_ENV)
        env["STATION_BASE_DATA_PATH"] = os.path.abspath(self.store.base_data_path)
        env["STATION_INDEX_DB_PATH"] = os.path.abspath(snapshot_path)
        env.pop("STATION_INDEX_DB_DIR", None)
        if backend == "codex":
            apply_codex_proxy_overrides(env)
        return CliJobLaunchSpec(
            executable=detect_cli_worker_executable(backend, env),
            run_dir=os.path.join(self.paths.sessions_dir, session_id),
            backend=backend,
            model_name=model_name,
            workspace_root=self.paths.root,
            storage_root=self.paths.root,
            prompt=prompt,
            env=env,
        )

    def _make_active_cli_job_session(self, job_id, base_session, decision, claimed, launch_metadata):
        return ActiveSurveySession(
            **vars(base_session),
            survey_id=job_id,
            report_path=self.store._report_path(job_id),
            draft_path=self.store._draft_path(job_id),
        )

    def _mark_cli_job_pid(self, job_id: str, pid: int) -> None:
        self._update_cli_job_record(job_id, {"active_pid": pid})

    def _on_cli_job_started(self, job_id, session, decision):
        print(
            f"WebArchiveSurveyor: started survey {job_id} "
            f"(pid={session.process.pid}, mode={decision.mode})"
        )

    def _cli_job_completion_ready(self, job_id, session):
        report_text = (
            file_io_utils.load_text(session.report_path)
            if file_io_utils.file_exists(session.report_path)
            else ""
        )
        return bool((report_text or "").strip())

    def _on_cli_job_completed(self, job_id, session, returncode):
        self._mark_completed(job_id, exit_code=returncode)

    def _cli_job_transient_failure_reason(self, job_id: str, pattern: str) -> str:
        return f"Transient Web Archive Surveyor backend/provider failure: {pattern}."

    def _cli_job_missing_report_reason(self, job_id, session):
        reason = session.transcript_idle_timeout_reason or (
            f"Web Archive Surveyor exited without producing reports/web_{job_id}.md."
        )
        if file_io_utils.file_exists(session.draft_path):
            reason += " A draft report exists but was not finalized."
        return reason

    def _schedule_cli_job_resume(self, job_id, session, failure):
        self._update_cli_job_record(
            job_id,
            {
                "active": False,
                "active_pid": None,
                "status": "pending_resume",
                "resume_token": failure.resume_token,
                "resume_count": failure.resume_count,
                "resume_delay_seconds": failure.delay_seconds,
                "next_resume_timestamp": failure.next_resume_timestamp,
                "exit_code": failure.returncode,
                "last_error": failure.reason,
                "failure_category": failure.category,
            },
            {"status": WEB_SURVEY_STATUS_RUNNING, "error": failure.reason},
        )

    def _schedule_cli_job_fresh_attempt(self, job_id, session, failure):
        self._mark_queued(job_id, failure.reason, failure_category=failure.category)

    def _on_cli_job_attempts_exhausted(self, job_id, session, failure):
        state = self._load_cli_job_state(job_id)
        reason = (
            f"Web Archive Surveyor exhausted {state.spawn_count} fresh spawn(s) and its "
            f"same-session resume budget without a final report. Last error: {failure.reason}"
        )
        self._mark_failed(job_id, reason)

    def _before_cli_job_poll(self) -> None:
        self._check_session_timeouts()

    def _check_session_timeouts(self) -> None:
        if self.timeout_seconds <= 0:
            return
        now = time.time()
        for survey_id, session in list(self.active_sessions.items()):
            record = self.store.get(survey_id, include_report=False)
            started = float((record.get("session") or {}).get("started_timestamp") or 0)
            if started > 0 and now - started >= self.timeout_seconds:
                self._terminate_cli_job_process(session, force=False)

    def _cli_job_idle_timeout_reason(self, job_id, session, idle_seconds, timeout_seconds):
        return (
            f"Web Archive Surveyor transcript for request #{job_id} did not grow for "
            f"{idle_seconds} seconds, exceeding the configured timeout of {timeout_seconds} seconds."
        )

    def _on_cli_job_idle_timeout(self, job_id, session, reason, idle_seconds, timeout_seconds):
        self._update_cli_job_record(
            job_id,
            {"last_error": reason, "failure_category": "codex_transcript_idle_timeout"},
        )

    def _mark_queued(
        self,
        survey_id: str,
        reason: str,
        *,
        failure_category: Optional[str] = None,
    ) -> None:
        self._update_cli_job_record(
            survey_id,
            {
                "active": False,
                "active_pid": None,
                "status": WEB_SURVEY_STATUS_QUEUED,
                "completed_timestamp": time.time(),
                "last_error": reason,
                "failure_category": failure_category,
                "resume_count": 0,
                "resume_token": None,
                "resume_delay_seconds": 0,
                "next_resume_timestamp": None,
            },
            {"status": WEB_SURVEY_STATUS_QUEUED, "error": reason},
        )

    def _requeue_after_shutdown(self, survey_id: str) -> None:
        self._mark_queued(survey_id, "Web dashboard worker stopped; request safely requeued.")

    def _mark_completed(self, survey_id: str, exit_code: Optional[int]) -> None:
        completed_timestamp = time.time()
        self._update_cli_job_record(
            survey_id,
            {
                "active": False,
                "active_pid": None,
                "status": WEB_SURVEY_STATUS_COMPLETED,
                "completed_timestamp": completed_timestamp,
                "exit_code": exit_code,
                "last_error": None,
            },
            {
                "status": WEB_SURVEY_STATUS_COMPLETED,
                "completed_timestamp": completed_timestamp,
                "error": None,
            },
        )
        self._cleanup_source_snapshot(survey_id)

    def _mark_failed(self, survey_id: str, reason: str) -> None:
        completed_timestamp = time.time()
        self._update_cli_job_record(
            survey_id,
            {
                "active": False,
                "active_pid": None,
                "status": WEB_SURVEY_STATUS_FAILED,
                "completed_timestamp": completed_timestamp,
                "last_error": reason,
            },
            {
                "status": WEB_SURVEY_STATUS_FAILED,
                "completed_timestamp": completed_timestamp,
                "error": reason,
            },
        )
        self._cleanup_source_snapshot(survey_id)

    def _cleanup_source_snapshot(self, survey_id: str) -> None:
        source_root = os.path.join(self.paths.sources_dir, str(survey_id))
        if os.path.isdir(source_root):
            shutil.rmtree(source_root)
