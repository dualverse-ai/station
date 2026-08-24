from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from station import constants
from station.multistart import paths, state


TERMINAL_STATUSES = {"complete", "cancelled"}


def active_job(repo: Path | None = None) -> dict[str, Any] | None:
    payload = state.load_current_job(repo)
    if not payload:
        return None
    status = str(payload.get("status") or "").lower()
    if status in TERMINAL_STATUSES:
        return None
    return payload


def waiting_mode_active(repo: Path | None = None) -> bool:
    repo_path = repo or paths.repo_root()
    return (
        active_job(repo_path) is not None
        or paths.pending_init_path(repo_path).is_file()
        or paths.pending_stagnation_path(repo_path).is_file()
    )


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_score(score: Any) -> str:
    if score is None or score == "":
        return "-"
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return f"{float(score):.8g}"
    return str(score)


def _path_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _evaluation_summary(data_root: Path) -> dict[str, Any]:
    evaluations_dir = data_root / "rooms" / "research" / "evaluations"
    summary: dict[str, Any] = {
        "total": 0,
        "running": 0,
        "queued": 0,
        "completed": 0,
        "failed": 0,
        "active_coders": 0,
        "active_evaluation_ids": [],
        "active_coder_pids": [],
    }
    if not evaluations_dir.is_dir():
        return summary

    indexed = _evaluation_summary_from_index(data_root, evaluations_dir)
    if indexed is not None:
        return indexed

    return summary


def _evaluation_summary_from_index(data_root: Path, evaluations_dir: Path) -> dict[str, Any] | None:
    db_path = data_root / "index" / constants.STATION_INDEX_DB_FILENAME
    if not db_path.is_file():
        return None

    scope = str(evaluations_dir.resolve())
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.2)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA query_only = ON")
            counts = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN top_level_status = 'queued' OR status = 'queued' THEN 1 ELSE 0 END) AS queued,
                    SUM(CASE WHEN top_level_status = 'running' OR status = 'running' THEN 1 ELSE 0 END) AS running,
                    SUM(CASE WHEN top_level_status IN ('completed', 'success') OR status IN ('completed', 'success') THEN 1 ELSE 0 END) AS completed,
                    SUM(CASE WHEN top_level_status IN ('failed', 'blocked', 'partial', 'cancelled', 'canceled')
                              OR status IN ('failed', 'blocked', 'partial', 'cancelled', 'canceled') THEN 1 ELSE 0 END) AS failed,
                    SUM(CASE WHEN active_coder = 1 OR coder_active = 1 THEN 1 ELSE 0 END) AS active_coders
                FROM research_evaluations
                WHERE evaluations_dir = ?
                """,
                (scope,),
            ).fetchone()
            if counts is None:
                return None

            active_rows = conn.execute(
                """
                SELECT eval_id, active_coder, coder_active
                FROM research_evaluations
                WHERE evaluations_dir = ?
                  AND (is_active = 1 OR active_coder = 1 OR coder_active = 1)
                ORDER BY start_timestamp DESC, eval_id_num DESC, eval_id DESC
                LIMIT 12
                """,
                (scope,),
            ).fetchall()
            active_ids = [str(row["eval_id"]) for row in active_rows]
            active_coder_ids = [
                str(row["eval_id"])
                for row in active_rows
                if int(row["active_coder"] or 0) or int(row["coder_active"] or 0)
            ]
            return {
                "total": int(counts["total"] or 0),
                "running": int(counts["running"] or 0),
                "queued": int(counts["queued"] or 0),
                "completed": int(counts["completed"] or 0),
                "failed": int(counts["failed"] or 0),
                "active_coders": int(counts["active_coders"] or 0),
                "active_evaluation_ids": active_ids,
                "active_coder_pids": [],
            }
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return None


def _branch_note(branch: dict[str, Any], evals: dict[str, Any]) -> str:
    status_text = str(branch.get("status") or "pending")
    if status_text == "copy_pending":
        return "waiting for parallel branch copy"
    if status_text == "copying":
        return "copying branch data and Research storage"
    if status_text == "copy_failed":
        return str(branch.get("error") or "branch copy failed; retry pending")
    if status_text == "pending":
        return "-"
    if status_text == "completed":
        return "complete"
    if status_text == "failed":
        return str(branch.get("error") or "failed")
    if status_text == "paused":
        return str(branch.get("pause_reason") or "paused")
    if status_text == "waiting_quiescent":
        return "waiting for background work"
    if status_text == "interviewing":
        return "interviewing agents"
    if evals.get("active_coders"):
        return f"{evals['active_coders']} coder(s) active"
    if evals.get("running"):
        return f"{evals['running']} evaluation(s) running"
    if branch.get("target_tick") is not None:
        return f"rolling to tick {branch.get('target_tick')}"
    return status_text


def _branch_public_status(
    job_path: Path,
    branch: dict[str, Any],
    selected_seed: Any = None,
    job_branch_tick: Any = None,
    job_roll_ticks: Any = None,
) -> dict[str, Any]:
    seed = _safe_int(branch.get("seed"), 0) or 0
    data_root = Path(str(branch.get("data_root") or state.branch_dir(job_path, seed)))
    config_path = data_root / "station_config.yaml"
    config = state.read_station_config(data_root)
    evals = _evaluation_summary(data_root)

    config_tick = _safe_int(config.get("current_tick"))
    branch_tick = _safe_int(branch.get("current_tick"))
    current_tick = max([tick for tick in (config_tick, branch_tick) if tick is not None], default=None)
    target_tick = _safe_int(branch.get("target_tick"))
    start_tick = _safe_int(branch.get("start_tick"))
    if start_tick is None:
        start_tick = _safe_int(job_branch_tick)
    original_start_tick = _safe_int(job_branch_tick)
    total_branch_ticks = _safe_int(job_roll_ticks)
    done_branch_ticks = None
    if current_tick is not None and original_start_tick is not None:
        done_branch_ticks = max(0, current_tick - original_start_tick)
    if total_branch_ticks is None and target_tick is not None and original_start_tick is not None:
        total_branch_ticks = max(0, target_tick - original_start_tick)
    progress_percent = None
    if done_branch_ticks is not None and total_branch_ticks is not None and total_branch_ticks > 0:
        progress_percent = round((done_branch_ticks / total_branch_ticks) * 100)
    elif current_tick is not None and target_tick is not None:
        if start_tick is not None and target_tick > start_tick:
            progress_percent = round(((current_tick - start_tick) / (target_tick - start_tick)) * 100)
        elif target_tick > 0:
            progress_percent = round((current_tick / target_tick) * 100)
    if progress_percent is not None:
        progress_percent = max(0, min(100, progress_percent))
    if done_branch_ticks is not None and total_branch_ticks is not None:
        done_branch_ticks = max(0, min(total_branch_ticks, done_branch_ticks))

    station_name = config.get("station_name") or ""
    last_tick_timestamp = _path_mtime(config_path)
    last_tick_age_seconds = None
    if last_tick_timestamp is not None:
        last_tick_age_seconds = max(0, int(time.time() - last_tick_timestamp))
    top_score = config.get("top_score", branch.get("top_score"))
    top_sort_key = config.get("top_sort_key", branch.get("top_sort_key"))
    return {
        "seed": seed,
        "selected": selected_seed is not None and str(selected_seed) == str(seed),
        "status": branch.get("status") or "pending",
        "data_dir": data_root.name,
        "station_name": station_name,
        "station_label": station_name or f"Seed {seed}",
        "station_id": config.get("station_id") or "",
        "current_tick": current_tick,
        "last_tick_timestamp": last_tick_timestamp,
        "last_tick_age_seconds": last_tick_age_seconds,
        "start_tick": start_tick,
        "target_tick": target_tick,
        "progress_done_ticks": done_branch_ticks,
        "progress_total_ticks": total_branch_ticks,
        "progress_percent": progress_percent,
        "top_evaluation_id": config.get("top_evaluation_id") or branch.get("top_evaluation_id") or "-",
        "top_tick": config.get("top_tick") or "-",
        "top_score": top_score,
        "top_sort_key": top_sort_key,
        "top_score_display": _format_score(top_score),
        "evaluations": evals,
        "pid": branch.get("pid"),
        "attempts": branch.get("attempts", 0),
        "note": _branch_note(branch, evals),
        "log_path": branch.get("log_path"),
        "error": branch.get("error"),
        "copy_status": branch.get("copy_status"),
        "copy_started_at": branch.get("copy_started_at"),
        "copy_completed_at": branch.get("copy_completed_at"),
    }


def _job_stage(
    status_text: str,
    counts: dict[str, int],
    seed_count: Any,
    selected_seed: Any,
    control: str,
) -> dict[str, str]:
    normalized = str(status_text or "").lower()
    try:
        total = int(seed_count or 0)
    except (TypeError, ValueError):
        total = 0
    completed = int(counts.get("completed") or 0)
    failed = int(counts.get("failed") or 0)

    if normalized == "creating":
        copied = int(counts.get("copy_complete") or 0)
        copying = int(counts.get("copying") or 0)
        failed_copies = int(counts.get("copy_failed") or 0)
        note = f"{copied}/{total} seed copies complete"
        if copying:
            note += f"; {copying} copying in parallel"
        if failed_copies:
            note += f"; {failed_copies} retrying after failure"
        return {"stage": "copying branch data", "stage_note": note}

    if normalized == "selecting":
        return {"stage": "admin running", "stage_note": "administrator selection is running"}
    if normalized == "finalizing":
        return {"stage": "finalizing", "stage_note": f"selected seed {selected_seed}" if selected_seed else "installing selected seed"}
    if normalized == "failed":
        return {"stage": "halted", "stage_note": "manual intervention required"}
    if failed:
        return {"stage": "halted", "stage_note": f"{failed} branch(es) failed"}
    paused = int(counts.get("paused") or 0)
    running = int(counts.get("running") or 0)
    if control == state.CONTROL_PAUSED and not running and not paused:
        pending = int(counts.get("pending") or 0)
        return {"stage": "paused", "stage_note": f"{pending} branch(es) queued; launch paused" if pending else "branch rolling paused"}
    if paused and not running:
        return {"stage": "paused", "stage_note": f"{paused} branch(es) paused"}
    if paused:
        return {"stage": "pausing", "stage_note": f"{paused} paused; {running} still running"}
    if total > 0 and completed >= total:
        return {"stage": "pending admin", "stage_note": "all branches complete; waiting for administrator selection"}
    if counts.get("interviewing"):
        return {"stage": "interviewing", "stage_note": f"{counts['interviewing']} branch(es) interviewing"}
    if counts.get("waiting_quiescent"):
        return {"stage": "waiting for jobs", "stage_note": f"{counts['waiting_quiescent']} branch(es) draining background work"}
    if counts.get("running"):
        return {"stage": "rolling branches", "stage_note": f"{counts['running']} branch(es) running"}
    if counts.get("pending"):
        return {"stage": "pending branches", "stage_note": f"{counts['pending']} branch(es) queued"}
    return {"stage": normalized or "pending", "stage_note": "-"}


def public_status(repo: Path | None = None) -> dict[str, Any]:
    repo_path = repo or paths.repo_root()
    active = active_job(repo_path)
    pending_init = state.load_yaml_mapping(paths.pending_init_path(repo_path)) if active is None else {}
    pending_stagnation = (
        state.load_yaml_mapping(paths.pending_stagnation_path(repo_path))
        if active is None and not pending_init
        else {}
    )
    pending_request = pending_init or pending_stagnation
    job = active or pending_request
    job_path_text = str(job.get("job_dir") or "")
    detail = state.load_job_state(Path(job_path_text)) if job_path_text else {}
    job_path = Path(job_path_text) if job_path_text else Path()
    raw_branches = detail.get("branches") if isinstance(detail.get("branches"), list) else []
    job_branch_tick = job.get("branch_tick") or detail.get("branch_tick")
    job_roll_ticks = job.get("roll_ticks") or detail.get("roll_ticks")
    branches = [
        _branch_public_status(job_path, branch, detail.get("selected_seed"), job_branch_tick, job_roll_ticks)
        for branch in raw_branches
        if isinstance(branch, dict)
    ]
    counts = {
        "copy_pending": 0,
        "copying": 0,
        "copy_failed": 0,
        "copy_complete": 0,
        "pending": 0,
        "running": 0,
        "paused": 0,
        "waiting_quiescent": 0,
        "interviewing": 0,
        "completed": 0,
        "failed": 0,
    }
    for branch in branches:
        status_text = str(branch.get("status") or "pending")
        counts[status_text] = counts.get(status_text, 0) + 1
        if branch.get("copy_status") == "complete":
            counts["copy_complete"] += 1
    active_coders = sum(int(branch.get("evaluations", {}).get("active_coders") or 0) for branch in branches)
    status_text = detail.get("status") or job.get("status")
    seed_count = job.get("seed_count") or detail.get("seed_count")
    selected_seed = detail.get("selected_seed")
    control = state.job_control(detail) if detail else state.CONTROL_RUNNING
    if pending_request:
        blocked = str(status_text or "") == "blocked_disk_space"
        mode_label = "stagnation" if pending_stagnation else "init"
        stage = {
            "stage": "waiting for disk space" if blocked else f"preparing {mode_label} multistart",
            "stage_note": job.get("message") or f"The controller is preparing the {mode_label} multistart job.",
        }
    else:
        stage = _job_stage(str(status_text or ""), counts, seed_count, selected_seed, control)
    return {
        "active": bool(job),
        "mode": job.get("mode") or detail.get("mode"),
        "status": status_text,
        "control": control,
        **stage,
        "job_id": job.get("job_id") or detail.get("job_id"),
        "branch_tick": job.get("branch_tick") or detail.get("branch_tick"),
        "job_dir": job_path_text,
        "station_name": detail.get("station_name"),
        "origin_station_id": detail.get("origin_station_id"),
        "seed_count": seed_count,
        "max_parallel": job.get("max_parallel") or detail.get("max_parallel"),
        "roll_ticks": job.get("roll_ticks") or detail.get("roll_ticks"),
        "selected_seed": selected_seed,
        "created_at": detail.get("created_at") or job.get("created_at"),
        "updated_at": detail.get("updated_at") or job.get("last_checked_at"),
        "message": job.get("message"),
        "disk_space": job.get("disk_space"),
        "counts": counts,
        "active_coders": active_coders,
        "branches": branches,
    }
