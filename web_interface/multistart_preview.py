"""Side-effect-free read model for the fixed seed-1 multistart dashboard preview."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from station import agent_summary, constants, file_io_utils, tick_timing
from station.multistart import paths, state, waiting
from station.station_config import top_submission_from_config


PREVIEW_SEED = 1
_RUNNING_BRANCH_STATUSES = {"running", "interviewing", "waiting_quiescent"}
_ALLOWED_MUTATION_PATHS = {"/login", "/api/multistart/pause", "/api/multistart/resume"}
_MAX_CAPSULE_ROWS = 5000


@dataclass(frozen=True)
class PreviewContext:
    repo: Path
    job_path: Path
    data_root: Path
    job: dict[str, Any]
    detail: dict[str, Any]
    branch: dict[str, Any]
    config: dict[str, Any]


def request_allowed(method: str, path: str) -> bool:
    """Return whether an HTTP request is safe in read-only preview mode."""
    return str(method).upper() in {"GET", "HEAD", "OPTIONS"} or str(path) in _ALLOWED_MUTATION_PATHS


def get_preview_context(repo: Path | None = None, seed: int = PREVIEW_SEED) -> PreviewContext | None:
    """Resolve one fixed branch without changing global Station paths."""
    repo_path = (repo or paths.repo_root()).resolve()
    job = waiting.active_job(repo_path)
    if not job:
        return None

    job_path_text = str(job.get("job_dir") or "").strip()
    if not job_path_text:
        return None
    job_path = Path(job_path_text).resolve()
    detail = state.load_job_state(job_path)
    branches = detail.get("branches") if isinstance(detail.get("branches"), list) else []
    branch = next(
        (
            dict(item)
            for item in branches
            if isinstance(item, dict) and _safe_int(item.get("seed")) == int(seed)
        ),
        None,
    )
    if branch is None:
        return None

    data_root = Path(str(branch.get("data_root") or state.branch_dir(job_path, seed))).resolve()
    try:
        data_root.relative_to(job_path)
    except ValueError:
        return None
    if not data_root.is_dir():
        return None

    return PreviewContext(
        repo=repo_path,
        job_path=job_path,
        data_root=data_root,
        job=dict(job),
        detail=dict(detail),
        branch=branch,
        config=state.read_station_config(data_root),
    )


def dashboard_context(repo: Path | None = None) -> dict[str, Any] | None:
    context = get_preview_context(repo)
    if context is None:
        return None
    branch = _branch_summary(context)
    counts = _branch_counts(context.detail)
    seed_count = _safe_int(context.job.get("seed_count")) or _safe_int(context.detail.get("seed_count")) or 0
    return {
        "active": True,
        "seed": PREVIEW_SEED,
        "mode": context.job.get("mode") or context.detail.get("mode"),
        "job_id": context.job.get("job_id") or context.detail.get("job_id"),
        "status": context.detail.get("status") or context.job.get("status"),
        "control": state.job_control(context.detail),
        "seed_count": seed_count,
        "completed_count": counts.get("completed", 0),
        "failed_count": counts.get("failed", 0),
        "branch": branch,
    }


def station_config(repo: Path | None = None) -> dict[str, Any] | None:
    context = get_preview_context(repo)
    if context is None:
        return None
    config = context.config
    return {
        "station_status": config.get(constants.STATION_CONFIG_STATION_STATUS, "Unknown"),
        "station_name": config.get(constants.STATION_CONFIG_NAME, ""),
        "station_description": config.get(constants.STATION_CONFIG_DESCRIPTION, ""),
        "station_id": config.get(constants.STATION_ID_KEY, "Unknown"),
        "read_only": True,
        "preview_seed": PREVIEW_SEED,
    }


def orchestrator_status(repo: Path | None = None) -> dict[str, Any] | None:
    context = get_preview_context(repo)
    if context is None:
        return None
    branch = _branch_summary(context)
    status_text = str(branch.get("status") or "pending").lower()
    turn_order = context.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, [])
    if not isinstance(turn_order, list):
        turn_order = []
    counts = _branch_counts(context.detail)
    seed_count = _safe_int(context.job.get("seed_count")) or _safe_int(context.detail.get("seed_count")) or 0
    paused = status_text == "paused" or state.job_paused(context.detail)
    return {
        "is_prepared": bool(turn_order),
        "is_running": status_text in _RUNNING_BRANCH_STATUSES or paused,
        "is_paused": paused,
        "is_waiting": status_text in {"interviewing", "waiting_quiescent"},
        "waiting_reasons": ({"multistart": branch.get("note")} if status_text in {"interviewing", "waiting_quiescent"} else {}),
        "pause_requested": state.job_paused(context.detail),
        "pause_condition_met": paused,
        "pause_reason": branch.get("note") if paused else "",
        "current_tick": branch.get("current_tick", -1),
        "station_status": context.config.get(constants.STATION_CONFIG_STATION_STATUS, "Unknown"),
        "turn_order": [str(name) for name in turn_order],
        "parallel_tick_status": None,
        "agents_awaiting_human": _agents_awaiting_human(context.data_root, turn_order),
        "read_only": True,
        "preview_seed": PREVIEW_SEED,
        "branch_status": status_text,
        "target_tick": branch.get("target_tick"),
        "multistart": {
            "active": True,
            "mode": context.job.get("mode") or context.detail.get("mode"),
            "job_id": context.job.get("job_id") or context.detail.get("job_id"),
            "status": context.detail.get("status") or context.job.get("status"),
            "control": state.job_control(context.detail),
            "seed_count": seed_count,
            "completed_count": counts.get("completed", 0),
            "failed_count": counts.get("failed", 0),
            "branch": branch,
        },
    }


def agents(repo: Path | None = None) -> list[dict[str, Any]] | None:
    context = get_preview_context(repo)
    if context is None:
        return None
    agents_dir = context.data_root / constants.AGENTS_DIR_NAME
    if not agents_dir.is_dir():
        return []
    return agent_summary.get_all_agents_summary(base_path=str(context.data_root))


def dialogue_log_path(agent_name: str, repo: Path | None = None) -> Path | None:
    context = get_preview_context(repo)
    if context is None:
        return None
    if agent_name == "Reviewer":
        return (
            context.data_root
            / constants.ROOMS_DIR_NAME
            / constants.SHORT_ROOM_NAME_ARCHIVE
            / "llm_chat_history.yamll"
        )
    safe_name = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in agent_name)
    return context.data_root / constants.DIALOGUE_LOGS_DIR_NAME / f"{safe_name}{constants.DIALOGUE_LOG_FILENAME_SUFFIX}"


def statistics(repo: Path | None = None) -> dict[str, Any] | None:
    context = get_preview_context(repo)
    if context is None:
        return None
    current_tick = _safe_int(context.config.get(constants.STATION_CONFIG_CURRENT_TICK), 0) or 0
    evaluation_stats = _research_statistics(context.data_root)
    pending_requests = _pending_human_requests(
        context.data_root,
        context.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, []),
    )
    return {
        "pending_human_requests": pending_requests,
        "current_tick": current_tick,
        "ticks_since_last_breakthrough": _safe_int(
            context.config.get(constants.STATION_CONFIG_STAGNATION_COUNTER)
        ),
        "top_research_submission": top_submission_from_config(context.config),
        "running_experiments_count": len(evaluation_stats["running"]),
        "running_experiments": evaluation_stats["running"],
        "queued_experiments_count": len(evaluation_stats["queued"]),
        "queued_experiments": evaluation_stats["queued"],
        "running_jobs_count": len(evaluation_stats["running"]),
        "running_jobs": evaluation_stats["running"],
        "queued_jobs_count": len(evaluation_stats["queued"]),
        "queued_jobs": evaluation_stats["queued"],
        "tick_timing": tick_timing.get_timing_summary(
            base_path=str(context.data_root),
            current_tick=current_tick,
        ),
        "pending_research_evaluations": bool(evaluation_stats["running"] or evaluation_stats["queued"]),
        "pending_coder_sessions": bool(evaluation_stats["running"]),
        "pending_external_reports": False,
        "pending_archive_surveys": False,
        "pending_archive_evaluations": False,
        "read_only": True,
        "preview_seed": PREVIEW_SEED,
    }


def handoff_statistics() -> dict[str, Any]:
    """Return a drain-safe empty view after the preview job has been cleared."""
    return {
        "pending_human_requests": {
            "request_ids": [],
            "agents": [],
            "agent_request_map": {},
        },
        "current_tick": -1,
        "top_research_submission": None,
        "running_experiments_count": 0,
        "running_experiments": [],
        "queued_experiments_count": 0,
        "queued_experiments": [],
        "running_jobs_count": 0,
        "running_jobs": [],
        "queued_jobs_count": 0,
        "queued_jobs": [],
        "tick_timing": {},
        "pending_research_evaluations": False,
        "pending_coder_sessions": False,
        "pending_external_reports": False,
        "pending_archive_surveys": False,
        "pending_archive_evaluations": False,
        "read_only": True,
        "preview_seed": PREVIEW_SEED,
        "handoff_pending": True,
    }


def task_spec_snapshot(repo: Path | None = None) -> dict[str, Any] | None:
    context = get_preview_context(repo)
    if context is None:
        return None
    task_path = (
        context.data_root
        / constants.ROOMS_DIR_NAME
        / constants.SHORT_ROOM_NAME_RESEARCH
        / constants.RESEARCH_TASK_SPEC_FILENAME
    )
    content = file_io_utils.load_text(str(task_path)) or ""
    try:
        modified_at_ns: Optional[str] = str(task_path.stat().st_mtime_ns)
    except OSError:
        modified_at_ns = None
    return {
        "raw_markdown": content,
        "revision": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "relative_path": str(task_path.relative_to(context.data_root)),
        "modified_at_ns": modified_at_ns,
        "read_only": True,
    }


class ReadOnlyCapsuleView:
    """Minimal capsule API backed by query-only SQLite and one-file detail reads."""

    def __init__(self, data_root: Path):
        self.data_root = data_root

    def list_capsules(self, capsule_type: str, _lineage_name: Any = None) -> list[dict[str, Any]]:
        rows, _total = self.list_capsules_page(
            capsule_type,
            None,
            page=1,
            page_size=_MAX_CAPSULE_ROWS,
            sort_by="numeric_id",
            sort_direction="desc",
        )
        return rows

    def list_capsules_page(
        self,
        capsule_type: str,
        _lineage_name: Any = None,
        *,
        page: int = 1,
        page_size: int = 100,
        sort_by: str = "numeric_id",
        sort_direction: str = "desc",
        **_kwargs: Any,
    ) -> tuple[list[dict[str, Any]], int]:
        db_path = _station_db_path(self.data_root)
        if not db_path.is_file():
            return [], 0
        sort_columns = {
            "numeric_id": "numeric_id",
            "title": "title COLLATE NOCASE",
            "author": "author_name COLLATE NOCASE",
            "created_at_tick": "created_at_tick",
            "last_updated_at_tick": "last_updated_at_tick",
            "question_status": "question_status COLLATE NOCASE",
            "question_net_upvote": "question_net_upvote",
            "message_count": "total_message_count",
        }
        sort_expression = sort_columns.get(str(sort_by), "numeric_id")
        direction = "ASC" if str(sort_direction).lower() == "asc" else "DESC"
        safe_page = max(1, int(page or 1))
        safe_size = max(1, min(_MAX_CAPSULE_ROWS, int(page_size or 100)))
        try:
            with _read_only_db(db_path) as conn:
                total = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM capsule_metadata WHERE capsule_type = ? AND lineage_key = '' AND is_deleted = 0",
                        (capsule_type,),
                    ).fetchone()[0]
                )
                rows = conn.execute(
                    f"""
                    SELECT * FROM capsule_metadata
                    WHERE capsule_type = ? AND lineage_key = '' AND is_deleted = 0
                    ORDER BY {sort_expression} {direction}, numeric_id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (capsule_type, safe_size, (safe_page - 1) * safe_size),
                ).fetchall()
        except (OSError, sqlite3.Error):
            return [], 0
        return [_capsule_metadata(row) for row in rows], total

    def get_capsule(
        self,
        numeric_id: int,
        capsule_type: str,
        _lineage_name: Any = None,
        *,
        include_deleted_capsule: bool = False,
        include_deleted_messages: bool = False,
    ) -> dict[str, Any] | None:
        subdir = {
            constants.CAPSULE_TYPE_ARCHIVE: constants.ARCHIVE_CAPSULES_SUBDIR_NAME,
            constants.CAPSULE_TYPE_QUESTION: constants.QUESTION_CAPSULES_SUBDIR_NAME,
        }.get(capsule_type)
        prefix = {
            constants.CAPSULE_TYPE_ARCHIVE: "archive_",
            constants.CAPSULE_TYPE_QUESTION: "question_",
        }.get(capsule_type)
        if not subdir or not prefix:
            return None
        path = self.data_root / constants.CAPSULES_DIR_NAME / subdir / f"{prefix}{int(numeric_id)}{constants.YAML_EXTENSION}"
        data = file_io_utils.load_yaml(str(path))
        if not isinstance(data, dict):
            return None
        if not include_deleted_capsule and data.get(constants.CAPSULE_IS_DELETED_KEY, False):
            return None
        if not include_deleted_messages:
            data = dict(data)
            data[constants.CAPSULE_MESSAGES_KEY] = [
                item
                for item in data.get(constants.CAPSULE_MESSAGES_KEY, [])
                if isinstance(item, dict) and not item.get(constants.MESSAGE_IS_DELETED_KEY, False)
            ]
        return data


def capsule_view(repo: Path | None = None) -> ReadOnlyCapsuleView | None:
    context = get_preview_context(repo)
    return ReadOnlyCapsuleView(context.data_root) if context else None


def _branch_summary(context: PreviewContext) -> dict[str, Any]:
    return waiting._branch_public_status(
        context.job_path,
        context.branch,
        context.detail.get("selected_seed"),
        context.job.get("branch_tick") or context.detail.get("branch_tick"),
        context.job.get("roll_ticks") or context.detail.get("roll_ticks"),
    )


def _branch_counts(detail: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for branch in detail.get("branches") or []:
        if isinstance(branch, dict):
            key = str(branch.get("status") or "pending")
            counts[key] = counts.get(key, 0) + 1
    return counts


def _agents_awaiting_human(data_root: Path, turn_order: Any) -> list[str]:
    names = turn_order if isinstance(turn_order, list) else []
    return [
        str(name)
        for name in names
        if agent_summary.agent_has_human_intervention_request(
            agent_summary.get_agent_human_intervention_fields(str(name), base_path=str(data_root))
        )
    ]


def _pending_human_requests(data_root: Path, turn_order: Any) -> dict[str, Any]:
    names = turn_order if isinstance(turn_order, list) else []
    request_ids: list[Any] = []
    agents: list[str] = []
    mapping: dict[str, list[Any]] = {}
    for name in names:
        agent_name = str(name)
        fields = agent_summary.get_agent_human_intervention_fields(agent_name, base_path=str(data_root))
        if not agent_summary.agent_has_human_intervention_request(fields):
            continue
        raw_ids = fields.get(constants.AGENT_HUMAN_INTERACTION_IDS_KEY)
        ids = list(raw_ids) if isinstance(raw_ids, list) else []
        single_id = fields.get(constants.AGENT_HUMAN_INTERACTION_ID_KEY)
        if single_id is not None and single_id not in ids:
            ids.append(single_id)
        agents.append(agent_name)
        mapping[agent_name] = ids
        request_ids.extend(item for item in ids if item not in request_ids)
    return {"request_ids": request_ids, "agents": agents, "agent_request_map": mapping}


def _research_statistics(data_root: Path) -> dict[str, Any]:
    db_path = _station_db_path(data_root)
    evaluations_dir = data_root / constants.ROOMS_DIR_NAME / constants.SHORT_ROOM_NAME_RESEARCH / constants.RESEARCH_EVALUATIONS_SUBDIR_NAME
    result: dict[str, Any] = {"running": [], "queued": []}
    if not db_path.is_file() or not evaluations_dir.is_dir():
        return result
    scope = str(evaluations_dir.resolve())
    try:
        with _read_only_db(db_path) as conn:
            rows = conn.execute(
                """
                SELECT eval_id, author, title, submitted_tick, start_timestamp, display_status,
                       top_level_status, latest_attempt_status, coder_active, execution_source, system_baseline
                FROM research_evaluations
                WHERE evaluations_dir = ? AND is_active = 1
                ORDER BY start_timestamp DESC, eval_id_num DESC, eval_id DESC
                """,
                (scope,),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return result
    now = time.time()
    for row in rows:
        item = {
            "evaluation_id": str(row["eval_id"]),
            "agent_name": row["author"],
            "title": row["title"] or "",
            "start_tick": row["submitted_tick"] or 0,
            "submitted_tick": row["submitted_tick"] or 0,
            "start_timestamp": row["start_timestamp"] or 0,
            "elapsed_seconds": max(0, int(now - float(row["start_timestamp"] or now))),
            "status": row["display_status"] or row["top_level_status"] or "queued",
            "top_level_status": row["top_level_status"] or "queued",
            "latest_attempt_status": row["latest_attempt_status"] or "",
            "coder_active": bool(row["coder_active"]),
            "execution_source": row["execution_source"] or "coder",
            "system_baseline": bool(row["system_baseline"]),
        }
        if str(item["top_level_status"]).lower() == "queued":
            result["queued"].append(item)
        elif str(item["top_level_status"]).lower() == "running":
            result["running"].append(item)
    return result


def _capsule_metadata(row: sqlite3.Row) -> dict[str, Any]:
    payload = {
        constants.CAPSULE_ID_KEY: row["capsule_id"],
        constants.CAPSULE_TYPE_KEY: row["capsule_type"],
        constants.CAPSULE_AUTHOR_NAME_KEY: row["author_name"],
        constants.CAPSULE_AUTHOR_LINEAGE_KEY: row["author_lineage"],
        constants.CAPSULE_AUTHOR_GENERATION_KEY: row["author_generation"],
        constants.CAPSULE_CREATED_AT_TICK_KEY: row["created_at_tick"],
        constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY: row["last_updated_at_tick"],
        constants.CAPSULE_TITLE_KEY: row["title"],
        constants.CAPSULE_ABSTRACT_KEY: row["abstract"],
        constants.CAPSULE_WORD_COUNT_TOTAL_KEY: row["word_count_total"] or 0,
        constants.CAPSULE_IS_DELETED_KEY: bool(row["is_deleted"]),
        "total_message_count": row["total_message_count"] or 0,
        "reviewer_score": row["reviewer_score"],
    }
    if row["capsule_type"] == constants.CAPSULE_TYPE_QUESTION:
        payload[constants.QUESTION_STATUS_KEY] = row["question_status"] or constants.QUESTION_STATUS_PENDING
        payload[constants.QUESTION_NET_UPVOTE_KEY] = row["question_net_upvote"] or 0
        payload[constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY] = row["question_solved_by_message_id"]
    return payload


def _station_db_path(data_root: Path) -> Path:
    return data_root / constants.STATION_INDEX_DIR_NAME / constants.STATION_INDEX_DB_FILENAME


@contextmanager
def _read_only_db(path: Path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.2)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA busy_timeout = 200")
        yield conn
    finally:
        conn.close()


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
