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

"""Append-only timing records for station ticks and major tick phases."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from station import constants, file_io_utils

_TIMING_LOCK = threading.RLock()
_RECENT_TICK_STATE_LIMIT = 100


def get_timing_file_path(base_path: Optional[str] = None) -> str:
    """Return the persistent tick timing log path."""
    root = base_path or constants.BASE_STATION_DATA_PATH
    return os.path.join(root, constants.TICK_TIMING_FILENAME)


def get_timing_state_file_path(base_path: Optional[str] = None) -> str:
    """Return the small hot-read timing state path."""
    root = base_path or constants.BASE_STATION_DATA_PATH
    return os.path.join(root, constants.TICK_TIMING_STATE_FILENAME)


def _utc_now(timestamp: Optional[float] = None) -> tuple[float, str]:
    ts = time.time() if timestamp is None else float(timestamp)
    return ts, datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _append_event(event: Dict[str, Any], *, base_path: Optional[str] = None) -> None:
    event.setdefault("recorded_timestamp", time.time())
    event.setdefault("recorded_at", datetime.fromtimestamp(event["recorded_timestamp"], tz=timezone.utc).isoformat())
    with _TIMING_LOCK:
        file_io_utils.append_yaml_line(event, get_timing_file_path(base_path))


def record_tick_start(
    tick: int,
    sync_mode: str,
    *,
    turn_order: Optional[List[str]] = None,
    base_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record the wall-clock start time of a station tick."""
    with _TIMING_LOCK:
        if _find_latest_tick_start_timestamp(tick, base_path=base_path) is not None:
            return
    started_timestamp, started_at = _utc_now()
    event: Dict[str, Any] = {
        "event": "tick_start",
        "tick": tick,
        "sync_mode": sync_mode,
        "started_timestamp": started_timestamp,
        "started_at": started_at,
    }
    if turn_order is not None:
        event["turn_order_count"] = len(turn_order)
    if metadata:
        event["metadata"] = dict(metadata)
    _append_event(event, base_path=base_path)
    _save_latest_timing_state(event, base_path=base_path)


def record_tick_end(
    tick: int,
    next_tick: int,
    sync_mode: str,
    *,
    base_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record the wall-clock end time and duration of a completed station tick."""
    ended_timestamp, ended_at = _utc_now()
    started_timestamp = _find_latest_tick_start_timestamp(tick, base_path=base_path)
    event: Dict[str, Any] = {
        "event": "tick_end",
        "tick": tick,
        "next_tick": next_tick,
        "sync_mode": sync_mode,
        "ended_timestamp": ended_timestamp,
        "ended_at": ended_at,
    }
    if started_timestamp is not None:
        event["duration_seconds"] = max(0.0, ended_timestamp - started_timestamp)
    if metadata:
        event["metadata"] = dict(metadata)
    _append_event(event, base_path=base_path)
    _save_latest_timing_state(event, base_path=base_path)


def record_phase(
    tick: int,
    phase: str,
    sync_mode: str,
    duration_seconds: float,
    *,
    started_timestamp: Optional[float] = None,
    ended_timestamp: Optional[float] = None,
    base_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record elapsed wall time for a named tick phase."""
    end_ts = time.time() if ended_timestamp is None else float(ended_timestamp)
    start_ts = end_ts - duration_seconds if started_timestamp is None else float(started_timestamp)
    event: Dict[str, Any] = {
        "event": "phase",
        "tick": tick,
        "sync_mode": sync_mode,
        "phase": phase,
        "duration_seconds": max(0.0, float(duration_seconds)),
        "started_timestamp": start_ts,
        "started_at": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
        "ended_timestamp": end_ts,
        "ended_at": datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
    }
    if metadata:
        event["metadata"] = dict(metadata)
    _append_event(event, base_path=base_path)


@contextmanager
def time_phase(
    tick: int,
    phase: str,
    sync_mode: str,
    *,
    base_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    """Context manager that records a phase duration even if the block raises."""
    started_timestamp = time.time()
    try:
        yield
    finally:
        ended_timestamp = time.time()
        record_phase(
            tick,
            phase,
            sync_mode,
            ended_timestamp - started_timestamp,
            started_timestamp=started_timestamp,
            ended_timestamp=ended_timestamp,
            base_path=base_path,
            metadata=metadata,
        )


def get_timing_summary(*, base_path: Optional[str] = None, current_tick: Optional[int] = None) -> Dict[str, Any]:
    """Return timing data for dashboard display.

    The dashboard intentionally computes elapsed time from
    latest_tick_started_timestamp on the client so stale backend polls still
    age visually.
    """
    state = _load_latest_timing_state(base_path=base_path)
    latest_start = _get_latest_tick_start_event_from_state(state, current_tick=current_tick)
    averages = _compute_timing_averages_from_state(state)

    payload: Dict[str, Any] = {
        "latest_tick": latest_start.get("tick") if latest_start else None,
        "latest_tick_started_timestamp": latest_start.get("started_timestamp") if latest_start else None,
        "latest_tick_started_at": latest_start.get("started_at") if latest_start else None,
        "average_last_10_tick_seconds": averages["average_last_10_tick_seconds"],
        "average_last_50_tick_seconds": averages["average_last_50_tick_seconds"],
        "average_last_100_tick_seconds": averages["average_last_100_tick_seconds"],
        "completed_tick_count": averages["completed_tick_count"],
    }
    return payload


def _save_latest_timing_state(event: Dict[str, Any], *, base_path: Optional[str] = None) -> None:
    state_path = get_timing_state_file_path(base_path)
    previous = _load_latest_timing_state(base_path=base_path)
    state = dict(previous) if isinstance(previous, dict) else {}

    if event.get("event") == "tick_start":
        state.update({
            "latest_tick": event.get("tick"),
            "latest_tick_started_timestamp": event.get("started_timestamp"),
            "latest_tick_started_at": event.get("started_at"),
            "latest_tick_sync_mode": event.get("sync_mode"),
        })
    elif event.get("event") == "tick_end":
        recent_ticks = _updated_recent_ticks(state, event)
        state.update({
            "latest_completed_tick": event.get("tick"),
            "latest_next_tick": event.get("next_tick"),
            "latest_tick_ended_timestamp": event.get("ended_timestamp"),
            "latest_tick_ended_at": event.get("ended_at"),
            "latest_tick_duration_seconds": event.get("duration_seconds"),
            "recent_ticks": recent_ticks,
        })
    else:
        return

    state["schema_version"] = 1
    state["updated_timestamp"] = time.time()
    state["updated_at"] = datetime.fromtimestamp(state["updated_timestamp"], tz=timezone.utc).isoformat()
    file_io_utils.save_yaml(state, state_path)


def _load_latest_timing_state(*, base_path: Optional[str] = None) -> Dict[str, Any]:
    state_path = get_timing_state_file_path(base_path)
    try:
        state = file_io_utils.load_yaml(state_path)
    except Exception:
        return {}
    return state if isinstance(state, dict) else {}


def _updated_recent_ticks(state: Dict[str, Any], event: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_recent_ticks = state.get("recent_ticks")
    recent_ticks = [dict(item) for item in raw_recent_ticks if isinstance(item, dict)] if isinstance(raw_recent_ticks, list) else []
    event_tick = event.get("tick")
    existing_record = next(
        (item for item in recent_ticks if item.get("tick") == event_tick or str(item.get("tick")) == str(event_tick)),
        {},
    )
    recent_ticks = [item for item in recent_ticks if item.get("tick") != event_tick and str(item.get("tick")) != str(event_tick)]

    latest_tick = state.get("latest_tick")
    latest_matches_event = latest_tick == event_tick or str(latest_tick) == str(event_tick)
    started_timestamp = _as_optional_float(
        state.get("latest_tick_started_timestamp") if latest_matches_event else existing_record.get("started_timestamp")
    )
    ended_timestamp = _as_optional_float(event.get("ended_timestamp"))
    duration = _as_optional_float(event.get("duration_seconds"))
    if duration is None and started_timestamp is not None and ended_timestamp is not None:
        duration = max(0.0, ended_timestamp - started_timestamp)

    recent_ticks.append({
        "tick": event_tick,
        "next_tick": event.get("next_tick"),
        "sync_mode": event.get("sync_mode") or existing_record.get("sync_mode") or state.get("latest_tick_sync_mode"),
        "started_timestamp": started_timestamp,
        "started_at": state.get("latest_tick_started_at") if latest_matches_event else existing_record.get("started_at"),
        "ended_timestamp": ended_timestamp,
        "ended_at": event.get("ended_at"),
        "duration_seconds": duration,
    })
    return recent_ticks[-_RECENT_TICK_STATE_LIMIT:]


def _compute_timing_averages_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    recent_ticks = state.get("recent_ticks")
    if not isinstance(recent_ticks, list):
        recent_ticks = []

    completed_durations = [
        duration
        for duration in (_as_optional_float(item.get("duration_seconds")) for item in recent_ticks if isinstance(item, dict))
        if duration is not None
    ]

    return {
        "average_last_10_tick_seconds": _average_window(completed_durations, 10),
        "average_last_50_tick_seconds": _average_window(completed_durations, 50),
        "average_last_100_tick_seconds": _average_window(completed_durations, 100),
        "completed_tick_count": len(completed_durations),
    }


def _get_latest_tick_start_event_from_state(state: Dict[str, Any], *, current_tick: Optional[int] = None) -> Optional[Dict[str, Any]]:
    if not state:
        return None

    latest_tick = state.get("latest_tick")
    if current_tick is not None and latest_tick not in {current_tick, str(current_tick)}:
        return None
    started_timestamp = _as_optional_float(state.get("latest_tick_started_timestamp"))
    if started_timestamp is None:
        return None
    return {
        "event": "tick_start",
        "tick": latest_tick,
        "started_timestamp": started_timestamp,
        "started_at": state.get("latest_tick_started_at"),
        "sync_mode": state.get("latest_tick_sync_mode"),
    }


def _find_latest_tick_start_timestamp(tick: int, *, base_path: Optional[str] = None) -> Optional[float]:
    state = _load_latest_timing_state(base_path=base_path)
    latest_event = _get_latest_tick_start_event_from_state(state, current_tick=tick)
    if latest_event is not None:
        return _as_optional_float(latest_event.get("started_timestamp"))

    recent_ticks = state.get("recent_ticks")
    if isinstance(recent_ticks, list):
        for item in reversed(recent_ticks):
            if not isinstance(item, dict):
                continue
            if item.get("tick") == tick or str(item.get("tick")) == str(tick):
                return _as_optional_float(item.get("started_timestamp"))
    return None


def _average(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _average_window(values: List[float], window_size: int) -> Optional[float]:
    if len(values) < window_size:
        return None
    return _average(values[-window_size:])


def _as_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
