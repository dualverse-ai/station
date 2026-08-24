"""Bounded, cursor-based broadcast buffer for dashboard live events."""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class EventCursorState:
    cursor: int
    dropped_count: int = 0
    reset: bool = False


@dataclass(frozen=True)
class EventBatch:
    events: List[Tuple[int, Dict[str, Any]]]
    cursor: int
    dropped_count: int = 0
    reset: bool = False


class DashboardEventBroker:
    """Queue-compatible producer surface with independent cursor readers.

    The buffer is intentionally transient. A new browser starts at the latest
    sequence instead of replaying everything produced while it was absent.
    Existing browsers may catch up a small bounded window after a short
    disconnect.
    """

    def __init__(self, max_events: int = 50, max_replay: int = 50):
        self.max_events = max(1, int(max_events))
        self.max_replay = max(1, min(int(max_replay), self.max_events))
        self._events: Deque[Tuple[int, Dict[str, Any]]] = deque(maxlen=self.max_events)
        self._next_sequence = 1
        self._condition = threading.Condition()
        self.epoch = uuid.uuid4().hex

    def put(self, event: Dict[str, Any], block: bool = True, timeout: Optional[float] = None) -> None:
        del block, timeout
        self._publish(event)

    def put_nowait(self, event: Dict[str, Any]) -> None:
        self._publish(event)

    def _publish(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            raise TypeError("Dashboard events must be dictionaries.")
        with self._condition:
            sequence = self._next_sequence
            self._next_sequence += 1
            self._events.append((sequence, event))
            self._condition.notify_all()

    @property
    def buffered_count(self) -> int:
        with self._condition:
            return len(self._events)

    @staticmethod
    def _coerce_cursor(cursor: Any) -> Optional[int]:
        if cursor is None or str(cursor).strip().lower() in {"", "latest"}:
            return None
        try:
            return max(0, int(cursor))
        except (TypeError, ValueError):
            return None

    def _normalize_cursor_locked(self, cursor: Any) -> EventCursorState:
        latest = self._next_sequence - 1
        requested = self._coerce_cursor(cursor)
        if requested is None:
            return EventCursorState(cursor=latest)
        if requested > latest:
            return EventCursorState(cursor=latest, reset=True)

        oldest_available = self._events[0][0] if self._events else latest + 1
        replay_floor = max(0, oldest_available - 1, latest - self.max_replay)
        if requested < replay_floor:
            return EventCursorState(
                cursor=replay_floor,
                dropped_count=replay_floor - requested,
                reset=True,
            )
        return EventCursorState(cursor=requested)

    def open_cursor(self, cursor: Any = None) -> EventCursorState:
        with self._condition:
            return self._normalize_cursor_locked(cursor)

    def read_after(
        self,
        cursor: Any,
        *,
        limit: int = 50,
        wait_timeout: float = 0.0,
    ) -> EventBatch:
        safe_limit = max(1, min(int(limit), self.max_replay))
        safe_timeout = max(0.0, float(wait_timeout))
        with self._condition:
            state = self._normalize_cursor_locked(cursor)
            if safe_timeout > 0 and self._next_sequence - 1 <= state.cursor:
                self._condition.wait_for(
                    lambda: self._next_sequence - 1 > state.cursor,
                    timeout=safe_timeout,
                )
                refreshed = self._normalize_cursor_locked(state.cursor)
                state = EventCursorState(
                    cursor=refreshed.cursor,
                    dropped_count=state.dropped_count + refreshed.dropped_count,
                    reset=state.reset or refreshed.reset,
                )

            events: List[Tuple[int, Dict[str, Any]]] = []
            for sequence, event in self._events:
                if sequence <= state.cursor:
                    continue
                events.append((sequence, event))
                if len(events) >= safe_limit:
                    break

            next_cursor = events[-1][0] if events else state.cursor
            return EventBatch(
                events=events,
                cursor=next_cursor,
                dropped_count=state.dropped_count,
                reset=state.reset,
            )
