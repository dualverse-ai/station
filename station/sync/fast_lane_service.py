"""Generic single-writer service for parallel tick fast-lane actions."""

from __future__ import annotations

import copy
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, Dict, Optional


@dataclass
class FastLaneSubmissionRequest:
    agent_data: Dict[str, Any]
    yaml_data: Optional[Dict[str, Any]]
    current_tick: int
    run_id: str
    op_id: str
    result: Optional[Any] = None
    done: threading.Event = field(default_factory=threading.Event)


class FastLaneSubmissionService(ABC):
    """Single-writer worker shared by parallel fast-lane action services."""

    service_name = "FastLaneSubmissionService"

    def __init__(
        self,
        station_instance: Any,
        *,
        log_event_func: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.station = station_instance
        self.log_event_func = log_event_func
        self._queue: Queue[FastLaneSubmissionRequest] = Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker_loop,
                name=self.service_name,
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

    def submit_and_wait(
        self,
        *,
        agent_data: Dict[str, Any],
        yaml_data: Optional[Dict[str, Any]],
        current_tick: int,
        run_id: str,
        op_id: str,
        timeout: Optional[float] = None,
    ) -> Any:
        self.start()
        request = FastLaneSubmissionRequest(
            agent_data=copy.deepcopy(agent_data),
            yaml_data=copy.deepcopy(yaml_data),
            current_tick=int(current_tick),
            run_id=str(run_id),
            op_id=str(op_id),
        )
        self._queue.put(request)
        wait_timeout = timeout if timeout is not None else self._default_timeout_seconds()
        if wait_timeout <= 0:
            request.done.wait()
        elif not request.done.wait(wait_timeout):
            return self._timeout_result()
        return request.result or self._empty_result()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                request = self._queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                request.result = self._process_request(request)
            except Exception as exc:
                request.result = self._exception_result(exc, request)
                self._push_log_event(
                    self._exception_event_type(),
                    {"error": str(exc), "op_id": request.op_id},
                )
            finally:
                request.done.set()
                self._queue.task_done()

    @abstractmethod
    def _process_request(self, request: FastLaneSubmissionRequest) -> Any:
        pass

    @abstractmethod
    def _timeout_result(self) -> Any:
        pass

    @abstractmethod
    def _empty_result(self) -> Any:
        pass

    @abstractmethod
    def _exception_result(self, exc: Exception, request: FastLaneSubmissionRequest) -> Any:
        pass

    def _default_timeout_seconds(self) -> float:
        return 0.0

    def _exception_event_type(self) -> str:
        return "parallel_fast_lane_submission_error"

    def _push_log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if not self.log_event_func:
            return
        try:
            self.log_event_func(event_type, data)
        except Exception:
            pass
