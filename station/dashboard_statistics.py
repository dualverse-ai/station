"""Fast control-plane statistics for the web dashboard.

This module intentionally avoids SQLite indexes and historical directory
scans. It reads station config, in-memory service state, pending queue files,
and only the authoritative records for currently active jobs.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, Set, Tuple

from station import agent_summary, constants, file_io_utils, tick_timing
from station.station_config import top_submission_from_config


def build_station_statistics(station: Any) -> Dict[str, Any]:
    """Build the dashboard statistics payload without historical index reads."""
    current_tick_getter = getattr(station, "_get_current_tick", None)
    current_tick = (
        current_tick_getter()
        if callable(current_tick_getter)
        else station.config.get(constants.STATION_CONFIG_CURRENT_TICK, 0)
    )
    stats = _empty_statistics(current_tick)
    _add_human_requests(stats, station)
    _add_research_statistics(stats, station)
    _add_service_jobs(stats, station, "auto_archive_surveyor", "ARCHIVE_SURVEY_ENABLED")
    _add_service_jobs(stats, station, "auto_external_reporter", "EXTERNAL_COUNTER_ENABLED")
    return stats


def _empty_statistics(current_tick: Any) -> Dict[str, Any]:
    return {
        "pending_human_requests": {
            "request_ids": [],
            "agents": [],
            "agent_request_map": {},
        },
        "current_tick": current_tick,
        "ticks_since_last_breakthrough": None,
        "top_research_submission": None,
        "running_experiments_count": 0,
        "running_experiments": [],
        "queued_experiments_count": 0,
        "queued_experiments": [],
        "running_jobs_count": 0,
        "running_jobs": [],
        "queued_jobs_count": 0,
        "queued_jobs": [],
        "drainable_running_jobs_count": 0,
        "drainable_queued_jobs_count": 0,
        "tick_timing": tick_timing.get_timing_summary(),
    }


def _add_human_requests(stats: Dict[str, Any], station: Any) -> None:
    if constants.ROOM_ADMIN not in station.rooms:
        return
    active_agents = station.config.get(constants.STATION_CONFIG_AGENT_TURN_ORDER, [])
    base_path = station.room_context.constants_module.BASE_STATION_DATA_PATH
    agents_awaiting = []
    for agent_name in active_agents:
        fields = agent_summary.get_agent_human_intervention_fields(agent_name, base_path=base_path)
        if agent_summary.agent_has_human_intervention_request(fields):
            agents_awaiting.append(agent_name)
    stats["pending_human_requests"] = station.rooms[
        constants.ROOM_ADMIN
    ].get_pending_requests_summary(active_agents=agents_awaiting)


def _add_research_statistics(stats: Dict[str, Any], station: Any) -> None:
    if not constants.RESEARCH_CENTER_ENABLED:
        return
    if not constants.RESEARCH_NO_SCORE:
        stats["top_research_submission"] = top_submission_from_config(station.config)
        stats["ticks_since_last_breakthrough"] = _nonnegative_int(
            station.config.get(constants.STATION_CONFIG_STAGNATION_COUNTER)
        )

    evaluator = getattr(station, "auto_research_evaluator", None)
    if evaluator is None:
        return
    running_jobs, queued_jobs = _research_jobs(evaluator)
    stats["running_experiments"] = running_jobs
    stats["running_experiments_count"] = len(running_jobs)
    stats["queued_experiments"] = queued_jobs
    stats["queued_experiments_count"] = len(queued_jobs)
    stats["running_jobs"] = list(running_jobs)
    stats["running_jobs_count"] = len(running_jobs)
    stats["queued_jobs"] = list(queued_jobs)
    stats["queued_jobs_count"] = len(queued_jobs)
    _update_drainable_job_counts(stats)


def _nonnegative_int(value: Any) -> int | None:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _research_jobs(evaluator: Any) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    running_ids = _mapping_keys(getattr(evaluator, "active_futures", {}))
    coder_manager = getattr(evaluator, "coder_manager", None)
    running_ids.update(_mapping_keys(getattr(coder_manager, "active_sessions", {})))
    evaluations_dir = getattr(evaluator, "evaluations_dir", "")
    now = time.time()
    running_jobs = [
        {
            "evaluation_id": str(eval_id),
            **_research_job_metadata(evaluations_dir, eval_id, now),
            "status": "running",
            "top_level_status": "running",
            "execution_source": "coder",
            "system_baseline": False,
        }
        for eval_id in sorted(running_ids)
    ]

    queued_jobs = []
    for name in _run_request_names(getattr(evaluator, "run_requests_dir", "")):
        eval_id = str(name).split("_attempt_", 1)[0]
        if eval_id and eval_id not in running_ids:
            queued_jobs.append({
                "evaluation_id": eval_id,
                "title": "",
                "agent_name": "",
                "status": "queued",
                "top_level_status": "queued",
                "execution_source": "coder",
                "system_baseline": False,
            })
    return running_jobs, queued_jobs


def _mapping_keys(value: Any) -> Set[str]:
    try:
        return {str(item) for item in value.keys()}
    except (AttributeError, RuntimeError):
        return set()


def _research_job_metadata(evaluations_dir: str, eval_id: Any, now: float) -> Dict[str, Any]:
    record = None
    if evaluations_dir:
        try:
            record = file_io_utils.load_yaml(os.path.join(evaluations_dir, f"{eval_id}.yaml"))
        except (OSError, TypeError, ValueError):
            pass
    if not isinstance(record, dict):
        return {"title": "", "agent_name": "", "start_timestamp": 0, "elapsed_seconds": 0}

    coder = record.get("coder") or {}
    started_timestamp = coder.get("started_timestamp") or record.get("submitted_timestamp") or 0
    try:
        elapsed_seconds = max(0, int(now - float(started_timestamp))) if started_timestamp else 0
    except (TypeError, ValueError):
        elapsed_seconds = 0
    return {
        "title": record.get("title") or "",
        "agent_name": record.get("author") or record.get("agent_name") or "",
        "start_timestamp": started_timestamp,
        "elapsed_seconds": elapsed_seconds,
    }


def _run_request_names(run_requests_dir: str) -> Iterable[str]:
    if not run_requests_dir or not os.path.isdir(run_requests_dir):
        return ()
    try:
        return tuple(sorted(name for name in os.listdir(run_requests_dir) if name.endswith(".yaml")))
    except OSError:
        return ()


def _add_service_jobs(
    stats: Dict[str, Any],
    station: Any,
    service_attribute: str,
    enabled_constant: str,
) -> None:
    service = getattr(station, service_attribute, None)
    if not getattr(constants, enabled_constant, False) or service is None:
        return
    getter = getattr(service, "get_lightweight_job_statistics", None)
    service_stats = getter() if callable(getter) else {}
    stats["running_jobs"].extend(service_stats.get("running_jobs", []))
    stats["queued_jobs"].extend(service_stats.get("queued_jobs", []))
    stats["running_jobs_count"] = len(stats["running_jobs"])
    stats["queued_jobs_count"] = len(stats["queued_jobs"])
    _update_drainable_job_counts(stats)


def _update_drainable_job_counts(stats: Dict[str, Any]) -> None:
    stats["drainable_running_jobs_count"] = sum(
        job.get("drainable", True) is not False
        for job in stats["running_jobs"]
    )
    stats["drainable_queued_jobs_count"] = sum(
        job.get("drainable", True) is not False
        for job in stats["queued_jobs"]
    )
