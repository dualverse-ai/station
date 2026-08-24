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
Research Center auto evaluator for the coder-based workflow.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

from station import constants
from station import file_io_utils
from .base_evaluator import ResearchTaskEvaluator
from .breakthroughs import normalize_progress_records
from .coder_manager import ResearchCoderManager
from .evaluation_helpers import save_stdout_with_limit
from .evaluation_manager import EvaluationManager, extract_secondary_metrics_for_display_info
from .gpu_coordinator import GPUCoordinator
from .cpu_coordinator import CPUCoordinator
from .runtime_paths import ensure_runtime_layout, ensure_task_spec_markdown, load_baseline_definitions
from .task_registry import ResearchTaskRegistry


class FatalResearchNotificationError(RuntimeError):
    """Raised when Research Center notification delivery fails and must not be retried automatically."""


def _gpu_management_enabled() -> bool:
    return constants.RESEARCH_EVAL_GPU_NUM is not None or bool(constants.RESEARCH_EVAL_USE_DIFF_GPU)


def _resource_coordination_station_id(station_instance: Any) -> str:
    """Return a resource owner ID unique to one live Station process lineage."""
    station_id = str(getattr(station_instance, "station_id", None) or "unknown")
    if os.environ.get("STATION_MULTISTART_BRANCH") != "1":
        return station_id
    seed = str(os.environ.get("STATION_MULTISTART_SEED") or "").strip()
    if not seed:
        raise RuntimeError(
            "Multistart Research resource coordination requires STATION_MULTISTART_SEED"
        )
    return f"{station_id}:s{seed}"


def _gpu_count_per_task() -> int:
    configured_count = (
        constants.RESEARCH_EVAL_GPU_NUM
        if constants.RESEARCH_EVAL_GPU_NUM is not None
        else constants.RESEARCH_EVAL_GPUS_PER_TASK
    )
    try:
        count = int(configured_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "GPU count must be a positive integer. Set RESEARCH_EVAL_GPU_NUM, "
            "or use legacy RESEARCH_EVAL_GPUS_PER_TASK with RESEARCH_EVAL_USE_DIFF_GPU."
        ) from exc
    if count <= 0:
        raise ValueError(
            "GPU count must be a positive integer. Set RESEARCH_EVAL_GPU_NUM, "
            "or use legacy RESEARCH_EVAL_GPUS_PER_TASK with RESEARCH_EVAL_USE_DIFF_GPU."
        )
    return count


def _detect_gpu_ids_with_nvidia_smi() -> List[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    gpu_ids: List[int] = []
    for line in result.stdout.splitlines():
        match = re.search(r"\d+", line)
        if match:
            gpu_ids.append(int(match.group(0)))
    return gpu_ids


def _parse_gpu_list(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        if value:
            return [int(gpu) for gpu in value]
        return _detect_gpu_ids_with_nvidia_smi()
    if isinstance(value, str):
        if not value.strip():
            return _detect_gpu_ids_with_nvidia_smi()
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if value is None:
        return _detect_gpu_ids_with_nvidia_smi()
    return [int(value)]


def _managed_gpu_ids() -> List[int]:
    if not _gpu_management_enabled():
        return []
    gpu_ids = _parse_gpu_list(constants.RESEARCH_EVAL_AVAILABLE_GPUS)
    if not gpu_ids:
        raise RuntimeError(
            "GPU management is enabled, but no usable GPUs could be auto-detected. "
            "Set RESEARCH_EVAL_AVAILABLE_GPUS explicitly, for example [0, 1, 2, 3], "
            "or disable GPU management."
        )
    per_task = _gpu_count_per_task()
    if len(gpu_ids) < per_task:
        raise RuntimeError(
            f"RESEARCH_EVAL_GPU_NUM is {per_task}, but only {len(gpu_ids)} GPU(s) are available: {gpu_ids}."
        )
    return gpu_ids


class AutoResearchEvaluator:
    _active_instances: Dict[int, "AutoResearchEvaluator"] = {}

    def __init__(
        self,
        station_instance,
        enabled: bool = None,
        log_queue: Optional[Queue] = None,
        eval_manager: Optional[EvaluationManager] = None,
    ):
        init_started_at = time.perf_counter()
        station_id = id(station_instance)
        if station_id in self._active_instances and self._active_instances[station_id].is_running:
            print("AutoResearchEvaluator: WARNING - Another evaluator instance is already running for this station")

        self.station = station_instance
        self.enabled = enabled if enabled is not None else constants.AUTO_EVAL_RESEARCH
        self.check_interval = constants.RESEARCH_EVAL_CHECK_INTERVAL
        self.timeout = constants.RESEARCH_EVAL_TIMEOUT
        self.memory_limit = constants.RESEARCH_EVAL_MEMORY_LIMIT
        self.cpu_limit = constants.RESEARCH_EVAL_CPU_LIMIT
        self.max_parallel_workers = max(1, int(constants.RESEARCH_EVAL_MAX_PARALLEL_WORKERS or 1))
        # Official attempts are a subset of live coder workflows. Basing this
        # executor on host CPU count created a large pool unrelated to the
        # configured station concurrency or CPUs allocated per evaluation.
        self.attempt_worker_max = self.max_parallel_workers
        self.log_queue = log_queue

        phase_started_at = time.perf_counter()
        self.paths = ensure_runtime_layout()
        self.seed_bank_store = None
        if bool(getattr(constants, "RESEARCH_SEED_BANK_ENABLED", False)):
            from .seed_bank import ensure_seed_bank_layout

            self.seed_bank_store = ensure_seed_bank_layout(self.paths, constants)
        ensure_task_spec_markdown()
        print(
            "AutoResearchEvaluator: startup ensure_runtime_layout/task_spec "
            f"took {time.perf_counter() - phase_started_at:.3f}s"
        )
        self.research_room_path = self.paths.research_root
        self.evaluations_dir = self.paths.evaluations_dir
        self.run_requests_dir = self.paths.run_requests_dir

        phase_started_at = time.perf_counter()
        self.task_registry = ResearchTaskRegistry()
        print(
            "AutoResearchEvaluator: startup ResearchTaskRegistry init "
            f"took {time.perf_counter() - phase_started_at:.3f}s"
        )

        self.is_running = False
        self.evaluation_thread: Optional[threading.Thread] = None
        self._wake_event = threading.Event()
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.active_futures: Dict[str, Dict[str, Any]] = {}

        runtime_station_id = _resource_coordination_station_id(station_instance)
        resource_expiry_seconds = max(float(self.timeout) * 2, 300.0)
        phase_started_at = time.perf_counter()
        self.gpu_coordinator = GPUCoordinator(
            coord_file_path=constants.RESEARCH_EVAL_GPU_COORD_FILE,
            available_gpus=_managed_gpu_ids(),
            station_id=runtime_station_id,
            expiry_seconds=resource_expiry_seconds,
        )
        print(
            "AutoResearchEvaluator: startup GPUCoordinator init "
            f"took {time.perf_counter() - phase_started_at:.3f}s"
        )
        self.cpu_coordinator = None
        if constants.RESEARCH_EVAL_CPU_NUM is not None:
            available_cpus = self._parse_cpu_list(constants.RESEARCH_EVAL_AVAILABLE_CPUS)
            phase_started_at = time.perf_counter()
            self.cpu_coordinator = CPUCoordinator(
                coord_file_path=constants.RESEARCH_EVAL_CPU_COORD_FILE,
                available_cpus=available_cpus,
                station_id=runtime_station_id,
                expiry_seconds=resource_expiry_seconds,
            )
            print(
                "AutoResearchEvaluator: startup CPUCoordinator init "
                f"took {time.perf_counter() - phase_started_at:.3f}s"
            )

        if self.enabled:
            phase_started_at = time.perf_counter()
            self._check_python_sandbox_setup()
            print(
                "AutoResearchEvaluator: startup python sandbox check "
                f"took {time.perf_counter() - phase_started_at:.3f}s"
            )

        phase_started_at = time.perf_counter()
        self.eval_manager = eval_manager or EvaluationManager(self.evaluations_dir)
        print(
            "AutoResearchEvaluator: startup EvaluationManager init "
            f"took {time.perf_counter() - phase_started_at:.3f}s"
        )
        self.eval_manager.set_notification_callback(self._send_notification)
        self.eval_manager.set_top_submission_callback(self.station.sync_top_research_submission_config)
        self.station.sync_top_research_submission_config(self.eval_manager.get_top_submission())

        phase_started_at = time.perf_counter()
        self.coder_manager = ResearchCoderManager(
            station_instance=self.station,
            eval_manager=self.eval_manager,
            paths=self.paths,
            log_queue=self.log_queue,
            pause_callback=self._pause_orchestrator,
        )
        print(
            "AutoResearchEvaluator: startup ResearchCoderManager init "
            f"took {time.perf_counter() - phase_started_at:.3f}s"
        )

        self._active_instances[station_id] = self
        print(
            "AutoResearchEvaluator: __init__ completed "
            f"in {time.perf_counter() - init_started_at:.3f}s"
        )

    def _push_log_event(self, event_type: str, data: Dict[str, Any]):
        if self.log_queue is None:
            return
        try:
            self.log_queue.put_nowait({"event": event_type, "data": data, "timestamp": time.time()})
        except Exception as exc:
            print(f"AutoResearchEvaluator: Error putting log event on queue: {exc}")

    def _report_exception(
        self,
        *,
        context: str,
        exc: BaseException,
        event_type: str = "auto_research_error",
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        trace = traceback.format_exc()
        print(f"AutoResearchEvaluator: {context}: {exc}\n{trace}")
        payload: Dict[str, Any] = {
            "error": f"{context}: {exc}",
            "trace": trace,
        }
        if extra_data:
            payload.update(extra_data)
        self._push_log_event(event_type, payload)

    def _send_notification(self, author: str, message: str):
        if author.lower() == "system":
            return
        success = self.station.agent_module.add_pending_notification_atomic(author, message)
        if not success:
            raise FatalResearchNotificationError(
                f"Research notification delivery failed for {author}: "
                "add_pending_notification_atomic returned False."
            )

    def _pause_orchestrator(self, reason: str):
        orchestrator = getattr(self.station, "orchestrator", None)
        if orchestrator and hasattr(orchestrator, "pause_due_to_research_issue"):
            orchestrator.pause_due_to_research_issue(reason)
            return
        print(f"AutoResearchEvaluator: Unable to pause orchestrator directly. Reason: {reason}")

    def _check_python_sandbox_setup(self) -> bool:
        started_at = time.perf_counter()
        try:
            from .evaluation_helpers import resolve_conda_env

            env = os.environ.copy()
            resolve_started_at = time.perf_counter()
            python_path = resolve_conda_env(constants.RESEARCH_EVAL_PYTHON_CONDA_ENV, env)
            print(
                "AutoResearchEvaluator: startup resolve_conda_env "
                f"took {time.perf_counter() - resolve_started_at:.3f}s"
            )
            if not python_path:
                print(
                    f"AutoResearchEvaluator: Warning - could not resolve python for conda env "
                    f"'{constants.RESEARCH_EVAL_PYTHON_CONDA_ENV}'"
                )
            sandbox_base = constants.RESEARCH_EVAL_SANDBOX_BASE_DIR
            file_io_utils.ensure_dir_exists(sandbox_base)
            result = os.access(sandbox_base, os.W_OK)
            print(
                "AutoResearchEvaluator: startup _check_python_sandbox_setup "
                f"completed in {time.perf_counter() - started_at:.3f}s"
            )
            return result
        except Exception as exc:
            print(f"AutoResearchEvaluator: Python sandbox setup check failed: {exc}")
            print(
                "AutoResearchEvaluator: startup _check_python_sandbox_setup "
                f"failed after {time.perf_counter() - started_at:.3f}s"
            )
            return False

    @staticmethod
    def _parse_cpu_list(value: Any) -> List[int]:
        if value is None:
            count = os.cpu_count() or 0
            return list(range(count))
        if isinstance(value, (list, tuple)):
            if not value:
                count = os.cpu_count() or 0
                return list(range(count))
            return [int(cpu) for cpu in value]
        if isinstance(value, str):
            if not value.strip():
                count = os.cpu_count() or 0
                return list(range(count))
            items: List[int] = []
            for part in [chunk.strip() for chunk in value.split(",") if chunk.strip()]:
                if "-" in part:
                    start_str, end_str = part.split("-", 1)
                    start = int(start_str.strip())
                    end = int(end_str.strip())
                    if end < start:
                        start, end = end, start
                    items.extend(range(start, end + 1))
                else:
                    items.append(int(part))
            deduped: List[int] = []
            seen = set()
            for cpu in items:
                if cpu not in seen:
                    deduped.append(cpu)
                    seen.add(cpu)
            return deduped
        return []

    def _allocate_gpu(self, eval_id: str) -> Optional[List[int]]:
        if not _gpu_management_enabled():
            return None
        return self.gpu_coordinator.allocate(eval_id, count=_gpu_count_per_task())

    def _deallocate_gpu(self, eval_id: str):
        if _gpu_management_enabled():
            self.gpu_coordinator.deallocate(eval_id)

    def _cleanup_gpu_allocations(self):
        if _gpu_management_enabled():
            self.gpu_coordinator.cleanup_stale_allocations(
                stale_run_seconds=constants.RESEARCH_EVAL_GPU_STALE_RUN_HOURS * 3600
            )

    def _allocate_cpu(self, eval_id: str) -> Optional[List[int]]:
        if constants.RESEARCH_EVAL_CPU_NUM is None or not self.cpu_coordinator:
            return None
        return self.cpu_coordinator.allocate(eval_id, count=constants.RESEARCH_EVAL_CPU_NUM)

    def _deallocate_cpu(self, eval_id: str):
        if constants.RESEARCH_EVAL_CPU_NUM is not None and self.cpu_coordinator:
            self.cpu_coordinator.deallocate(eval_id)

    def _cleanup_cpu_allocations(self):
        if constants.RESEARCH_EVAL_CPU_NUM is not None and self.cpu_coordinator:
            self.cpu_coordinator.cleanup_stale_allocations(
                stale_run_seconds=constants.RESEARCH_EVAL_CPU_STALE_RUN_HOURS * 3600
            )

    def start_evaluation_loop(self) -> bool:
        started_at = time.perf_counter()
        if not self.enabled or self.is_running:
            return False

        if not self._check_python_sandbox_setup():
            print(
                "AutoResearchEvaluator: start_evaluation_loop aborted after "
                f"{time.perf_counter() - started_at:.3f}s"
            )
            return False

        phase_started_at = time.perf_counter()
        self._seed_system_baseline_if_needed()
        print(
            "AutoResearchEvaluator: startup baseline seeding/check "
            f"took {time.perf_counter() - phase_started_at:.3f}s"
        )

        self.is_running = True
        self._wake_event.clear()
        phase_started_at = time.perf_counter()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.attempt_worker_max)
        self.evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.evaluation_thread.start()
        print(
            "AutoResearchEvaluator: startup thread pool + evaluation thread "
            f"took {time.perf_counter() - phase_started_at:.3f}s"
        )
        print(
            "AutoResearchEvaluator: start_evaluation_loop completed "
            f"in {time.perf_counter() - started_at:.3f}s"
        )
        return True

    def wake(self, reason: str = "") -> None:
        """Wake the evaluator loop before the normal polling interval expires."""
        if reason:
            self._push_log_event("auto_research_wake", {"reason": reason})
        self._wake_event.set()

    def refresh_task_registry(self) -> ResearchTaskEvaluator:
        """Reload the active evaluator after all Research Center work has drained."""
        new_registry = ResearchTaskRegistry()
        evaluator = new_registry.get_evaluator()
        if evaluator is None:
            raise RuntimeError("The active Research Center evaluator could not be loaded.")
        evaluator.get_task_description()
        self.task_registry = new_registry
        return evaluator

    def stop_evaluation_loop(self):
        started_at = time.perf_counter()
        if not self.is_running:
            return
        self.is_running = False
        self._wake_event.set()

        for metadata in self.active_futures.values():
            future = metadata.get("future")
            if future and not future.done():
                future.cancel()

        if self.thread_pool:
            phase_started_at = time.perf_counter()
            self.thread_pool.shutdown(wait=False, cancel_futures=True)
            self.thread_pool = None
            print(
                "AutoResearchEvaluator: shutdown thread pool cleanup "
                f"took {time.perf_counter() - phase_started_at:.3f}s"
            )

        phase_started_at = time.perf_counter()
        self._kill_orphaned_evaluation_processes()
        print(
            "AutoResearchEvaluator: shutdown pre-requeue wrapper cleanup "
            f"took {time.perf_counter() - phase_started_at:.3f}s"
        )

        try:
            phase_started_at = time.perf_counter()
            requeued = self.coder_manager.shutdown("Recovered during station shutdown: instruction prompt requeued.")
            print(
                "AutoResearchEvaluator: shutdown coder_manager.shutdown "
                f"took {time.perf_counter() - phase_started_at:.3f}s"
            )
            if requeued:
                self._push_log_event(
                    "auto_research_shutdown_requeue",
                    {"count": requeued, "message": f"Requeued {requeued} active instruction prompt(s) during shutdown."},
                )
        except Exception as exc:
            self._push_log_event(
                "auto_research_error",
                {"error": f"Failed to requeue active instruction prompts during shutdown: {exc}"},
            )

        phase_started_at = time.perf_counter()
        self._kill_orphaned_evaluation_processes()
        print(
            "AutoResearchEvaluator: shutdown post-requeue wrapper cleanup "
            f"took {time.perf_counter() - phase_started_at:.3f}s"
        )

        station_id = id(self.station)
        if self._active_instances.get(station_id) is self:
            del self._active_instances[station_id]
        print(
            "AutoResearchEvaluator: stop_evaluation_loop completed "
            f"in {time.perf_counter() - started_at:.3f}s"
        )

    def _kill_orphaned_evaluation_processes(self):
        try:
            from .restart_evaluations import _terminate_station_wrapper_processes

            _terminate_station_wrapper_processes(self.paths.research_root)
        except Exception as exc:
            self._report_exception(
                context="Failed to terminate orphaned Research Center evaluation processes",
                exc=exc,
            )

    def has_active_coder_sessions(self) -> bool:
        return self.coder_manager.has_active_sessions()

    def has_pending_or_running(self) -> bool:
        if self.coder_manager.has_active_sessions():
            return True
        if self.active_futures:
            return True
        if self._list_run_request_paths():
            return True
        return bool(self.eval_manager.get_active_evaluations())

    def _list_run_request_paths(self) -> List[str]:
        if not os.path.exists(self.run_requests_dir):
            return []
        paths = [
            os.path.join(self.run_requests_dir, name)
            for name in os.listdir(self.run_requests_dir)
            if name.endswith(".yaml")
        ]
        paths.sort(key=lambda path: os.path.getmtime(path))
        return paths

    def _load_run_request(self, request_path: str) -> Optional[Dict[str, Any]]:
        data = file_io_utils.load_yaml(request_path)
        return data if isinstance(data, dict) else None

    def _evaluation_loop(self):
        while self.is_running:
            try:
                self.coder_manager.poll()
                self._recover_stale_coder_states()
                self._cleanup_gpu_allocations()
                self._cleanup_cpu_allocations()
                self._check_completed_futures()
                self._repair_terminal_notifications()
                self._launch_resuming_coders()
                self._launch_queued_coders()
                self._dispatch_run_requests()

                if constants.AUTO_START:
                    should_restart, running_evals = self._check_all_running_exceeded_double_timeout()
                    if should_restart:
                        timeout_details = [
                            f"ID {info['evaluation_id']}: {info['title']} ({info['agent_name']}, {info['elapsed_seconds']}s)"
                            for info in running_evals
                        ]
                        message = (
                            f"AUTO_START: All {len(running_evals)} running attempt(s) exceeded "
                            f"{self.timeout * 2}s. Restarting station.\nEvaluations: {', '.join(timeout_details)}"
                        )
                        print(message)
                        self._push_log_event("auto_research_restart", {"reason": message})
                        try:
                            subprocess.run(["./start.sh", "--force"], capture_output=True, text=True, timeout=120, cwd=os.getcwd())
                        except Exception:
                            traceback.print_exc()
                        self.is_running = False
                        return

                self._wake_event.wait(self.check_interval)
                self._wake_event.clear()
            except FatalResearchNotificationError as exc:
                self._report_exception(
                    context="Fatal Research Center notification delivery failure",
                    exc=exc,
                )
                self._pause_orchestrator(str(exc))
                self.is_running = False
                return
            except Exception as exc:
                self._report_exception(
                    context="Unhandled exception in research evaluation loop",
                    exc=exc,
                )
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

    def _has_run_request_for_eval(self, eval_id: str) -> bool:
        prefix = f"{eval_id}_attempt_"
        for request_path in self._list_run_request_paths():
            if os.path.basename(request_path).startswith(prefix):
                return True
        return False

    @staticmethod
    def _attempt_elapsed_seconds(attempt: Dict[str, Any]) -> float:
        started = attempt.get("started_timestamp") or attempt.get("requested_timestamp")
        try:
            started_float = float(started)
        except (TypeError, ValueError):
            return 0.0
        if started_float <= 0:
            return 0.0
        return max(0.0, time.time() - started_float)

    def _recover_stale_coder_states(self):
        for eval_id in self.eval_manager.get_running_instruction_eval_ids():
            audit_enabled = bool(getattr(constants, "RESEARCH_CODER_AUDIT_ENABLED", True))
            audit_manager = getattr(self.coder_manager, "audit_manager", None)
            if eval_id in self.coder_manager.active_sessions or (
                audit_enabled and audit_manager is not None and eval_id in audit_manager.active_sessions
            ):
                continue
            if eval_id in self.active_futures:
                continue
            if self._has_run_request_for_eval(eval_id):
                continue

            eval_data = self.eval_manager.get_evaluation(eval_id)
            if not isinstance(eval_data, dict) or eval_data.get("final"):
                continue

            if audit_enabled:
                audit = eval_data.get("audit", {}) or {}
                audit_status = str(audit.get("status", "")).strip().lower()
                audit_pid = audit.get("active_pid")
                audit_report_path = os.path.join(self.paths.audit_dir, f"{eval_id}.md")
                audit_verdict_path = os.path.join(self.paths.audit_dir, f"{eval_id}.verdict")
                if audit_status in {"pass", "fail"}:
                    verdict = (
                        (file_io_utils.load_text(audit_verdict_path) or "").strip().lower()
                        if os.path.exists(audit_verdict_path)
                        else ""
                    )
                    if verdict in {"pass", "fail"} and os.path.exists(audit_report_path):
                        self.coder_manager._handle_audit_verdict(
                            eval_id,
                            verdict,
                            file_io_utils.load_text(audit_report_path) or "",
                            audit.get("session_id"),
                        )
                    else:
                        self.eval_manager.update_evaluation(
                            eval_id,
                            lambda record: record.setdefault("audit", {}).update({
                                "active": False,
                                "active_pid": None,
                                "status": "retry",
                                "last_error": "Recovered incomplete persisted auditor verdict; auditor will be relaunched.",
                            }),
                        )
                    continue
                if audit_status == "blocked":
                    continue
                if bool(audit.get("active")):
                    if self._pid_exists(audit_pid):
                        continue
                    if os.path.exists(audit_report_path) and os.path.exists(audit_verdict_path):
                        verdict = (file_io_utils.load_text(audit_verdict_path) or "").strip().lower()
                        if verdict in {"pass", "fail"}:
                            self.coder_manager._handle_audit_verdict(
                                eval_id, verdict, file_io_utils.load_text(audit_report_path) or "", audit.get("session_id")
                            )
                            continue
                    self.eval_manager.update_evaluation(eval_id, lambda record: record.setdefault("audit", {}).update({
                        "active": False, "active_pid": None, "status": "retry",
                        "last_error": "Recovered stale auditor process; auditor will be relaunched.",
                    }))
                    continue
                if audit_status == "repairing":
                    coder_state = eval_data.get("coder", {}) or {}
                    if bool(coder_state.get("active")) and self._pid_exists(coder_state.get("active_pid")):
                        continue
                    attempts = eval_data.get("attempts") or []
                    latest_attempt = attempts[-1] if attempts else {}
                    latest_attempt_status = str(latest_attempt.get("status", "")).strip().lower()
                    repair_attempt_interrupted = latest_attempt_status in {"queued", "running"}
                    # The report from before the repair is still present. Only
                    # advance to re-audit when the repair coder replaced it and
                    # the corresponding official attempt reached a terminal state.
                    report_path = os.path.join(self.paths.reports_dir, f"{eval_id}.md")
                    try:
                        report_mtime_ns = os.stat(report_path).st_mtime_ns
                    except OSError:
                        report_mtime_ns = 0
                    baseline_mtime_ns = int(audit.get("repair_report_baseline_mtime_ns", 0) or 0)
                    if baseline_mtime_ns:
                        repair_report_ready = report_mtime_ns > baseline_mtime_ns
                    else:
                        started_timestamp = float(coder_state.get("started_timestamp") or 0)
                        repair_report_ready = bool(
                            report_mtime_ns and started_timestamp and report_mtime_ns > int(started_timestamp * 1e9)
                        )
                    if repair_report_ready and not repair_attempt_interrupted:
                        self.eval_manager.update_evaluation(eval_id, lambda record: record.update({
                            "status": "running",
                            "coder": dict(record.get("coder", {}) or {}, active=False, active_pid=None, status="audit_running"),
                            "audit": dict(record.get("audit", {}) or {}, status="queued", active=False),
                        }))
                    else:
                        resume_token = str(coder_state.get("resume_token") or "").strip()
                        from .restart_evaluations import _abandon_interrupted_repair_attempt

                        def recover_repair(record):
                            interruption_reason = (
                                "Recovered interrupted repair attempt: no live evaluator future, "
                                "run request, or repair coder process remained."
                            )
                            attempt_abandoned = _abandon_interrupted_repair_attempt(
                                record,
                                interruption_reason,
                                self.eval_manager,
                            )
                            record.update({
                                "status": "running" if resume_token else "queued",
                                "coder": dict(
                                    record.get("coder", {}) or {},
                                    active=bool(resume_token),
                                    active_pid=None,
                                    status="pending_resume" if resume_token else "queued",
                                    next_resume_timestamp=None,
                                    resume_delay_seconds=None,
                                    last_error=(
                                        "Recovered interrupted repair attempt."
                                        if attempt_abandoned
                                        else (record.get("coder", {}) or {}).get("last_error")
                                    ),
                                ),
                            })
                        self.eval_manager.update_evaluation(eval_id, recover_repair)
                    continue
                if audit_status in {"queued", "retry", "running", "pending_resume", "resuming"} and not bool((eval_data.get("coder", {}) or {}).get("active")):
                    # An unfinished audit is independently restartable; it must not
                    # cause the coder attempt to be rerun.
                    continue

            coder = eval_data.get("coder", {}) or {}
            coder_status = str(coder.get("status", "")).strip().lower()
            if coder_status in {"pending_resume", "resuming"} and str(coder.get("resume_token", "")).strip():
                continue

            attempts = eval_data.get("attempts") or []
            latest_attempt = attempts[-1] if attempts else {}
            latest_attempt_status = str(latest_attempt.get("status", "")).strip().lower()
            report_path = os.path.join(self.paths.reports_dir, f"{eval_id}.md")

            coder_active = bool(coder.get("active"))
            pid = coder.get("active_pid")
            if coder_active and self._pid_exists(pid):
                continue

            if (
                not coder_active
                and coder_status == "attempt_running"
                and latest_attempt_status in {"queued", "running"}
                and self._attempt_elapsed_seconds(latest_attempt) < float(self.timeout)
            ):
                continue

            if os.path.exists(report_path) and latest_attempt_status not in {"queued", "running"}:
                report_text = file_io_utils.load_text(report_path) or ""
                if audit_enabled and audit_status in {"", "not_started"}:
                    resume_token = str(coder.get("resume_token") or "").strip() or None
                    if not resume_token:
                        session_id = str(coder.get("session_id") or "").strip()
                        transcript_path = os.path.join(
                            self.paths.coder_sessions_dir,
                            session_id,
                            "transcript.jsonl",
                        )
                        try:
                            backend = str(coder.get("backend") or constants.RESEARCH_CODER_BACKEND).lower()
                            resume_token = self.coder_manager._get_cli_job_backend(backend).extract_resume_token(
                                transcript_path
                            )
                        except Exception:
                            resume_token = None
                    self.eval_manager.update_evaluation(eval_id, lambda record: record.update({
                        "status": "running",
                        "coder": dict(
                            record.get("coder", {}) or {},
                            active=False,
                            active_pid=None,
                            status="audit_running",
                            resume_token=resume_token,
                        ),
                        "audit": dict(
                            record.get("audit", {}) or {},
                            status="queued",
                            active=False,
                            active_pid=None,
                        ),
                    }))
                    self._push_log_event(
                        "auto_research_recovered_stale_coder_report_for_audit",
                        {"eval_id": eval_id, "session_id": coder.get("session_id")},
                    )
                    continue
                if audit_enabled:
                    continue
                final_status = self.coder_manager._parse_final_status(report_text)
                self.eval_manager.finalize_evaluation(eval_id, report_text, final_status=final_status)
                self._push_log_event(
                    "auto_research_recovered_stale_coder_report",
                    {"eval_id": eval_id, "session_id": coder.get("session_id")},
                )
                continue

            if coder_active:
                reason = (
                    "Recovered stale coder state: coder session was marked active, "
                    "but no live coder process, active future, or run-request remained."
                )
            else:
                reason = (
                    "Recovered stale attempt state: coder was inactive and no active future "
                    "or run-request remained after the attempt timeout."
                )

            def mutator(record: Dict[str, Any]):
                coder_record = record.setdefault("coder", {})
                coder_record["active"] = False
                coder_record["active_pid"] = None
                coder_record["completed_timestamp"] = time.time()
                coder_record["last_error"] = reason
                coder_record["failure_category"] = "stale_session_recovered"
                coder_record["status"] = "queued"
                record["status"] = "queued"

                attempts_local = record.get("attempts") or []
                if attempts_local:
                    latest_local = attempts_local[-1]
                    latest_status_local = str(latest_local.get("status", "")).strip().lower()
                    if latest_status_local in {"queued", "running"}:
                        latest_local["status"] = "abandoned"
                        latest_local["completed_timestamp"] = latest_local.get("completed_timestamp") or time.time()
                        latest_local["error"] = reason

            self.eval_manager.update_evaluation(eval_id, mutator)
            self._push_log_event(
                "auto_research_recovered_stale_coder",
                {
                    "eval_id": eval_id,
                    "session_id": coder.get("session_id"),
                    "active_pid": pid,
                    "coder_active": coder_active,
                },
            )

    def _repair_terminal_notifications(self):
        for eval_id in self.eval_manager.get_pending_notification_eval_ids():
            try:
                self.eval_manager.send_notification_if_pending(eval_id)
            except Exception as exc:
                self._push_log_event(
                    "auto_research_error",
                    {"error": f"Failed to resend pending notification for eval {eval_id}: {exc}"},
                )

    def _launch_queued_coders(self):
        if self.eval_manager.get_active_coder_count() >= self.max_parallel_workers:
            return

        for eval_id in self.eval_manager.get_queued_instruction_eval_ids():
            if self.eval_manager.get_active_coder_count() >= self.max_parallel_workers:
                break
            if eval_id in self.coder_manager.active_sessions:
                continue

            eval_data = self.eval_manager.get_evaluation(eval_id)
            if not isinstance(eval_data, dict) or "instruction" not in eval_data:
                continue
            if eval_data.get("final"):
                continue
            if str(eval_data.get("status", "queued")) != "queued":
                continue
            if eval_data.get("submission_mode") == "direct" or eval_data.get("system_baseline"):
                continue
            try:
                launched = self.coder_manager.launch_for_evaluation(eval_data)
            except Exception as exc:
                reason = f"Failed to launch Research Center coder for evaluation {eval_id}: {exc}"

                def mutator(record: Dict[str, Any]):
                    coder = record.setdefault("coder", {})
                    coder["active"] = False
                    coder["active_pid"] = None
                    coder["status"] = "blocked"
                    coder["failure_category"] = "launch_error"
                    coder["last_error"] = reason
                    record["status"] = "blocked"

                self.eval_manager.update_evaluation(eval_id, mutator)
                self._report_exception(
                    context=f"Research Center coder launch failed for evaluation {eval_id}",
                    exc=exc,
                    event_type="auto_research_coder_launch_failed",
                    extra_data={"eval_id": eval_id},
                )
                self._pause_orchestrator(reason)
                return

            if not launched:
                continue

    def _launch_resuming_coders(self):
        for eval_id in self.eval_manager.get_resuming_instruction_eval_ids():
            if eval_id in self.coder_manager.active_sessions:
                continue

            eval_data = self.eval_manager.get_evaluation(eval_id)
            if not isinstance(eval_data, dict) or "instruction" not in eval_data:
                continue
            if eval_data.get("final"):
                continue
            if str(eval_data.get("status", "queued")).strip().lower() != "running":
                continue
            coder = eval_data.get("coder", {}) or {}
            if str(coder.get("status", "")).strip().lower() not in {"pending_resume", "resuming"}:
                continue
            if not str(coder.get("resume_token", "")).strip():
                continue
            if self.coder_manager.get_resume_backoff_remaining_seconds(coder) > 0:
                continue

            try:
                launched = self.coder_manager.launch_for_evaluation(eval_data)
            except Exception as exc:
                reason = f"Failed to resume Research Center coder for evaluation {eval_id}: {exc}"

                def mutator(record: Dict[str, Any]):
                    coder_record = record.setdefault("coder", {})
                    coder_record["active"] = False
                    coder_record["active_pid"] = None
                    coder_record["status"] = "blocked"
                    coder_record["failure_category"] = "launch_error"
                    coder_record["last_error"] = reason
                    record["status"] = "blocked"

                self.eval_manager.update_evaluation(eval_id, mutator)
                self._report_exception(
                    context=f"Research Center coder resume failed for evaluation {eval_id}",
                    exc=exc,
                    event_type="auto_research_coder_resume_failed",
                    extra_data={"eval_id": eval_id},
                )
                self._pause_orchestrator(reason)
                return

            if launched:
                return

    def _dispatch_run_requests(self):
        if not self.thread_pool:
            return

        for request_path in self._list_run_request_paths():
            if len(self.active_futures) >= self.attempt_worker_max:
                return

            request = self._load_run_request(request_path)
            if not request:
                try:
                    os.rename(request_path, request_path + ".invalid")
                except Exception as exc:
                    self._report_exception(
                        context=f"Failed to quarantine invalid run-request file {request_path}",
                        exc=exc,
                    )
                continue

            eval_id = str(request.get("eval_id"))
            attempt_number = int(request.get("attempt", 0))
            if not eval_id or attempt_number <= 0:
                try:
                    os.rename(request_path, request_path + ".invalid")
                except Exception as exc:
                    self._report_exception(
                        context=f"Failed to quarantine malformed run-request file {request_path}",
                        exc=exc,
                    )
                continue

            if eval_id in self.active_futures:
                continue

            eval_data = self.eval_manager.get_evaluation(eval_id)
            if not isinstance(eval_data, dict) or "instruction" not in eval_data:
                try:
                    os.rename(request_path, request_path + ".orphaned")
                except Exception as exc:
                    self._report_exception(
                        context=f"Failed to quarantine orphaned run-request file {request_path}",
                        exc=exc,
                    )
                continue

            cpu_ids = None
            if constants.RESEARCH_EVAL_CPU_NUM is not None:
                cpu_ids = self._allocate_cpu(eval_id)
                if cpu_ids is None:
                    continue

            gpu_ids = None
            gpu_management_enabled = _gpu_management_enabled()
            cpu_only = gpu_management_enabled and bool(request.get("cpu_only", False))
            if gpu_management_enabled:
                if cpu_only:
                    gpu_ids = []
                else:
                    gpu_ids = self._allocate_gpu(eval_id)
                    if gpu_ids is None:
                        if cpu_ids is not None:
                            self._deallocate_cpu(eval_id)
                        continue

            future = self.thread_pool.submit(self._execute_run_request, eval_id, attempt_number, request_path, gpu_ids, cpu_ids)
            self.active_futures[eval_id] = {
                "future": future,
                "gpu_ids": gpu_ids,
                "cpu_ids": cpu_ids,
                "attempt": attempt_number,
            }

    def _check_completed_futures(self):
        completed: List[str] = []
        for eval_id, metadata in list(self.active_futures.items()):
            future: Future = metadata["future"]
            if not future.done():
                continue
            completed.append(eval_id)
            try:
                future.result()
            except Exception as exc:
                self._report_exception(
                    context=f"Run-request worker failed for evaluation {eval_id}",
                    exc=exc,
                    extra_data={"eval_id": eval_id},
                )

        for eval_id in completed:
            metadata = self.active_futures.pop(eval_id, {})
            if metadata.get("gpu_ids"):
                self._deallocate_gpu(eval_id)
            if metadata.get("cpu_ids") is not None:
                self._deallocate_cpu(eval_id)

    def _write_attempt_logs(
        self,
        eval_id: str,
        stdout_text: str,
        stderr_text: str,
        score: Any,
        details: Any,
        safe_to_resubmit: bool,
        prelude_stdout: str = "",
        attempt_status: str = "completed",
    ):
        stdout_path = os.path.join(self.paths.stdout_dir, f"{eval_id}.log")
        stderr_path = os.path.join(self.paths.stderr_dir, f"{eval_id}.log")
        normalized_status = str(attempt_status or "completed").strip().lower()
        if normalized_status == "completed":
            status_message = (
                "The submission has run to completion. You may now submit the next attempt or write the final Coder Report."
            )
        else:
            status_message = (
                f"The attempt has settled with status '{normalized_status}'. Inspect the failure details "
                "before deciding whether to retry or finalize the report."
            )

        _, metrics_dict = extract_secondary_metrics_for_display_info({constants.EVALUATION_DETAILS_KEY: details})
        footer_lines = [
            "",
            "Official evaluation finished. The attempt status, score, and secondary metrics below are authoritative. Inspect them before deciding whether to resubmit or finalize the report.",
            "ATTEMPT_COMPLETE",
            f"ATTEMPT_STATUS: {normalized_status}",
            status_message,
            f"PRIMARY_SCORE: {score}",
            f"SECONDARY_METRICS: {json.dumps(metrics_dict, sort_keys=True)}",
            f"SAFE_TO_RESUBMIT: {'true' if safe_to_resubmit else 'false'}",
        ]
        stdout_sections: List[str] = []
        if prelude_stdout.strip():
            stdout_sections.append(prelude_stdout.rstrip())
        if stdout_text.rstrip():
            stdout_sections.append(stdout_text.rstrip())
        stdout_sections.append("\n".join(footer_lines).lstrip("\n"))
        combined_stdout = "\n\n".join(section for section in stdout_sections if section).lstrip("\n") + "\n"
        save_stdout_with_limit(stdout_path, combined_stdout)
        file_io_utils.save_text((stderr_text or "").rstrip() + ("\n" if stderr_text else ""), stderr_path)
        return combined_stdout, stderr_text or ""

    def _build_attempt_start_banner(
        self,
        eval_id: str,
        attempt_number: int,
        execution_mode: str,
        gpu_ids: Optional[List[int]] = None,
        cpu_ids: Optional[List[int]] = None,
    ) -> str:
        resource_parts: List[str] = []
        if cpu_ids is not None:
            resource_parts.append(f"CPUs={cpu_ids}")
        if gpu_ids == []:
            resource_parts.append("GPUs=disabled")
        elif gpu_ids is not None:
            resource_parts.append(f"GPUs={gpu_ids}")
        if not resource_parts:
            resource_parts.append("default resources")

        lines = [
            "ATTEMPT_RUNNING",
            f"Official evaluation attempt {attempt_number} for evaluation {eval_id} has started.",
            f"Execution mode: {execution_mode}",
            f"Allocated resources: {', '.join(resource_parts)}",
            f"Timeout: {self.timeout} seconds",
            "Stdout and stderr will be appended here as they are produced by the worker.",
            "Some submissions buffer their own output, so a quiet log does not by itself mean the evaluation is stalled.",
        ]
        return "\n".join(lines).rstrip() + "\n"

    def _initialize_attempt_logs(
        self,
        eval_id: str,
        attempt_number: int,
        execution_mode: str,
        gpu_ids: Optional[List[int]] = None,
        cpu_ids: Optional[List[int]] = None,
    ) -> str:
        banner = self._build_attempt_start_banner(eval_id, attempt_number, execution_mode, gpu_ids=gpu_ids, cpu_ids=cpu_ids)
        stdout_path = os.path.join(self.paths.stdout_dir, f"{eval_id}.log")
        stderr_path = os.path.join(self.paths.stderr_dir, f"{eval_id}.log")
        save_stdout_with_limit(stdout_path, banner)
        file_io_utils.save_text("", stderr_path)
        return banner

    def _get_evaluator(self) -> ResearchTaskEvaluator:
        evaluator = self.task_registry.get_evaluator()
        if evaluator:
            return evaluator
        raise RuntimeError("No Research Center evaluator is available.")

    def _execute_run_request(
        self,
        eval_id: str,
        attempt_number: int,
        request_path: str,
        gpu_ids: Optional[List[int]] = None,
        cpu_ids: Optional[List[int]] = None,
    ):
        eval_data = self.eval_manager.get_evaluation(eval_id)
        if not isinstance(eval_data, dict) or "instruction" not in eval_data:
            if os.path.exists(request_path):
                os.remove(request_path)
            return

        attempts = eval_data.get("attempts") or []
        attempt = next((item for item in attempts if int(item.get("attempt", -1)) == int(attempt_number)), None)
        if not attempt:
            if os.path.exists(request_path):
                os.remove(request_path)
            return

        self.eval_manager.mark_attempt_running(eval_id, attempt_number, start_tick=getattr(self.station, "_get_current_tick", lambda: 0)())
        evaluator = self._get_evaluator()
        execution_mode = evaluator.get_execution_mode() if hasattr(evaluator, "get_execution_mode") else "function"
        start_banner = self._initialize_attempt_logs(
            eval_id,
            attempt_number,
            execution_mode,
            gpu_ids=gpu_ids,
            cpu_ids=cpu_ids,
        )
        if os.path.exists(request_path):
            os.remove(request_path)
        submission_path = str(attempt.get("submission_path") or "").strip()
        content = file_io_utils.load_text(submission_path) if submission_path and file_io_utils.file_exists(submission_path) else ""
        author = eval_data.get("author", "Unknown")
        title = eval_data.get("title", "Untitled")
        is_system_baseline = bool(eval_data.get("system_baseline"))
        current_tick = getattr(self.station, "_get_current_tick", lambda: 0)()

        temp_eval_entry = {
            constants.EVALUATION_ID_KEY: eval_id,
            constants.EVALUATION_CONTENT_KEY: content,
            constants.EVALUATION_AUTHOR_KEY: author,
            constants.EVALUATION_SUBMITTED_TICK_KEY: eval_data.get(constants.EVALUATION_SUBMITTED_TICK_KEY, 0),
            "current_tick": current_tick,
            "system_baseline": is_system_baseline,
        }
        if bool(getattr(constants, "RESEARCH_SEED_BANK_ENABLED", False)):
            temp_eval_entry["lineage"] = str(eval_data.get("lineage", "unknown")).lower()
            temp_eval_entry["coder_access_phase"] = str(
                (eval_data.get("coder_access") or {}).get("phase") or "mature"
            ).lower()

        try:
            if hasattr(evaluator, "validate_submission_code"):
                is_valid, error_msg = evaluator.validate_submission_code(content, author, self.station.agent_module)
                if not is_valid:
                    stdout_text, stderr_text = self._write_attempt_logs(
                        eval_id,
                        "",
                        error_msg,
                        constants.RESEARCH_SCORE_NA,
                        "",
                        safe_to_resubmit=(attempt_number < int(eval_data.get("coder", {}).get("max_attempts", constants.RESEARCH_CODER_MAX_ATTEMPTS))),
                        prelude_stdout=start_banner,
                        attempt_status="failed",
                    )
                    self.eval_manager.complete_attempt(
                        eval_id,
                        attempt_number,
                        success=False,
                        score=constants.RESEARCH_SCORE_NA,
                        stdout=stdout_text,
                        stderr=stderr_text,
                        details="",
                        error=error_msg,
                        status="failed",
                    )
                    if is_system_baseline:
                        self._finalize_system_baseline(eval_id, content)
                    return

            execution_result = self._execute_submission(
                eval_entry=temp_eval_entry,
                evaluator=evaluator,
                gpu_ids=gpu_ids,
                cpu_ids=cpu_ids,
                live_stdout_path=os.path.join(self.paths.stdout_dir, f"{eval_id}.log"),
                live_stderr_path=os.path.join(self.paths.stderr_dir, f"{eval_id}.log"),
            )
            stdout_raw = execution_result.get("stdout", "")
            stderr_raw = execution_result.get("stderr", "")
            safe_to_resubmit = attempt_number < int(
                eval_data.get("coder", {}).get("max_attempts", constants.RESEARCH_CODER_MAX_ATTEMPTS)
            )

            if not execution_result.get("success"):
                error_msg = execution_result.get("error", "Execution failed")
                attempt_status = "timeout" if "timed out after" in error_msg.lower() else "failed"
                stdout_text, stderr_text = self._write_attempt_logs(
                    eval_id,
                    stdout_raw,
                    stderr_raw,
                    constants.RESEARCH_SCORE_NA,
                    "",
                    safe_to_resubmit=safe_to_resubmit,
                    prelude_stdout=start_banner,
                    attempt_status=attempt_status,
                )
                self.eval_manager.complete_attempt(
                    eval_id,
                    attempt_number,
                    success=False,
                    score=constants.RESEARCH_SCORE_NA,
                    stdout=stdout_text,
                    stderr=stderr_text,
                    details="",
                    error=error_msg,
                    status=attempt_status,
                )
                if is_system_baseline:
                    self._finalize_system_baseline(eval_id, content)
                return

            algorithm_result = execution_result["result"]
            if bool(getattr(constants, "RESEARCH_SEED_BANK_ENABLED", False)):
                from .seed_bank import validate_and_rank_seed_batch

                if algorithm_result is None:
                    success = True
                    score = constants.RESEARCH_SCORE_NA
                    details = {
                        "Message": (
                            "The submission intentionally returned None. The attempt completed "
                            "as non-scorable and no candidate was added to the Seed Bank."
                        ),
                        "SeedBankCandidates": 0,
                    }
                    sort_key = None
                else:
                    try:
                        seed_batch = evaluator.evaluate_seed_batch_with_formatting(algorithm_result, eval_id, author)
                    except TypeError:
                        seed_batch = evaluator.evaluate_seed_batch_with_formatting(algorithm_result, eval_id)
                    try:
                        ranked_seed_batch = validate_and_rank_seed_batch(seed_batch, constants)
                    except ValueError as exc:
                        success = False
                        score = constants.RESEARCH_SCORE_NA
                        details = {"Message": f"Seed batch evaluation failed: {exc}"}
                        sort_key = (float("-inf"),)
                    else:
                        winner_index = ranked_seed_batch.winner_index
                        runner_up_index = ranked_seed_batch.runner_up_index
                        if self.seed_bank_store is None:
                            raise RuntimeError("Seed Bank is enabled but its Station store was not initialized.")
                        try:
                            self.seed_bank_store.save_batch(
                                eval_id=eval_id,
                                attempt_number=attempt_number,
                                lineage=str(eval_data.get("lineage", "unknown")),
                                author=str(author),
                                ranked=ranked_seed_batch,
                            )
                        except (TypeError, ValueError) as exc:
                            success = False
                            score = constants.RESEARCH_SCORE_NA
                            details = {"Message": f"Seed batch persistence rejected: {exc}"}
                            sort_key = (float("-inf"),)
                        else:
                            success = True
                            score = float(seed_batch.scores[winner_index])
                            sort_key = tuple(seed_batch.sort_keys[winner_index])
                            winner_details = seed_batch.details[winner_index]
                            details = dict(winner_details) if isinstance(winner_details, dict) else {
                                "Message": str(winner_details)
                            }
                            batch_size = len(seed_batch.seeds)
                            valid_count = len(ranked_seed_batch.ranked_indices)
                            runner_up_text = "none"
                            if runner_up_index is not None:
                                runner_up_text = (
                                    f"batch index {runner_up_index}, "
                                    f"score {float(seed_batch.scores[runner_up_index]):.12g}"
                                )
                            original_message = str(details.get("Message", "")).strip()
                            summary = (
                                f"Seed batch: {batch_size} candidates, {valid_count} valid; "
                                f"winner batch index {winner_index}; runner-up {runner_up_text}."
                            )
                            details["Message"] = f"{original_message} {summary}".strip()
                            details["BatchSize"] = batch_size
                            details["ValidCandidates"] = valid_count
                            details["WinnerBatchIndex"] = winner_index
                            details["RunnerUpBatchIndex"] = runner_up_index
                            details["RunnerUpScore"] = (
                                None if runner_up_index is None else float(seed_batch.scores[runner_up_index])
                            )
            else:
                try:
                    eval_result = evaluator.evaluate_submission_with_formatting(algorithm_result, eval_id, author)
                except TypeError:
                    eval_result = evaluator.evaluate_submission_with_formatting(algorithm_result, eval_id)

                if len(eval_result) == 4:
                    success, score, details, sort_key = eval_result
                else:
                    success, score, details = eval_result
                    sort_key = None

            if isinstance(details, dict):
                error_message = details.get("Message", "Evaluation failed")
            else:
                error_message = str(details)
            progress_records = self._get_progress_records(
                evaluator=evaluator,
                result=algorithm_result,
                success=bool(success),
                score=score,
                details=details,
                sort_key=sort_key,
                eval_id=eval_id,
                author=author,
            )

            stdout_text, stderr_text = self._write_attempt_logs(
                eval_id,
                stdout_raw,
                stderr_raw,
                score if success else constants.RESEARCH_SCORE_NA,
                details,
                safe_to_resubmit=safe_to_resubmit,
                prelude_stdout=start_banner,
                attempt_status="completed" if success else "failed",
            )
            self.eval_manager.complete_attempt(
                eval_id,
                attempt_number,
                success=bool(success),
                score=score if success else constants.RESEARCH_SCORE_NA,
                stdout=stdout_text,
                stderr=stderr_text,
                details=details,
                error=None if success else f"Evaluation failed: {error_message}",
                sort_key=sort_key,
                status="completed" if success else "failed",
                progress_records=progress_records if success else [],
            )
            if is_system_baseline:
                self._finalize_system_baseline(eval_id, content)
        except Exception as exc:
            error_text = f"Evaluation system error: {exc}"
            stdout_text, stderr_text = self._write_attempt_logs(
                eval_id,
                "",
                traceback.format_exc(),
                constants.RESEARCH_SCORE_NA,
                "",
                safe_to_resubmit=(attempt_number < int(eval_data.get("coder", {}).get("max_attempts", constants.RESEARCH_CODER_MAX_ATTEMPTS))),
                prelude_stdout=start_banner,
                attempt_status="failed",
            )
            self.eval_manager.complete_attempt(
                eval_id,
                attempt_number,
                success=False,
                score=constants.RESEARCH_SCORE_NA,
                stdout=stdout_text,
                stderr=stderr_text,
                details="",
                error=error_text,
                status="failed",
            )
            if is_system_baseline:
                self._finalize_system_baseline(eval_id, content)

    def _execute_submission(
        self,
        eval_entry: Dict[str, Any],
        evaluator: ResearchTaskEvaluator,
        gpu_ids: Optional[List[int]] = None,
        cpu_ids: Optional[List[int]] = None,
        live_stdout_path: Optional[str] = None,
        live_stderr_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._execute_submission_in_python_sandbox(
            eval_entry,
            evaluator,
            gpu_ids,
            cpu_ids,
            live_stdout_path=live_stdout_path,
            live_stderr_path=live_stderr_path,
        )

    def _get_progress_records(
        self,
        *,
        evaluator: ResearchTaskEvaluator,
        result: Any,
        success: bool,
        score: Any,
        details: Any,
        sort_key: Any,
        eval_id: str,
        author: str,
    ) -> List[Dict[str, Any]]:
        getter = getattr(evaluator, "get_progress_records", None)
        if not callable(getter):
            return []
        try:
            raw_records = getter(
                result=result,
                success=success,
                score=score,
                details=details,
                sort_key=sort_key,
                eval_id=eval_id,
                author=author,
            )
        except Exception as exc:
            print(f"AutoResearchEvaluator: progress record hook failed eval_id={eval_id!r}: {exc}")
            return []
        return normalize_progress_records(raw_records)

    def _system_baseline_report(self, raw_code: str) -> str:
        return (
            "This is a system baseline. The raw code is:\n"
            f"```python\n{raw_code.rstrip()}\n```"
        )

    def _finalize_system_baseline(self, eval_id: str, raw_code: str):
        report = self._system_baseline_report(raw_code)
        self.eval_manager.finalize_evaluation(eval_id, report)

    def _seed_system_baseline_if_needed(self):
        for baseline in load_baseline_definitions(constants):
            eval_id = str(baseline.get("id") or "1")
            if self.eval_manager.get_evaluation(eval_id):
                continue

            raw_code = str(baseline.get("content") or "").rstrip()
            if not raw_code:
                continue

            title = str(baseline.get("title") or "System baseline")
            abstract = str(baseline.get("abstract") or "")
            tags = baseline.get("tags") or ["baseline"]
            if isinstance(tags, str):
                tags = [item.strip() for item in tags.split(",") if item.strip()]
            elif not isinstance(tags, list):
                tags = ["baseline"]

            submitted_tick = int(baseline.get("submitted_tick") or 0)
            self.eval_manager.create_evaluation(
                eval_id=eval_id,
                author=str(baseline.get("author") or "System"),
                title=title,
                content="",
                tick=submitted_tick,
                tags=tags,
                abstract=abstract,
                lineage="system",
            )

            def mutator(record: Dict[str, Any]):
                record["system_baseline"] = True
                record["submission_mode"] = "direct"
                coder = record.setdefault("coder", {})
                coder["backend"] = "system"
                coder["model_name"] = None
                coder["max_attempts"] = 1
                coder["max_spawns"] = 1
                coder["status"] = "queued"

            self.eval_manager.update_evaluation(eval_id, mutator)

            stdout_path = os.path.join(self.paths.stdout_dir, f"{eval_id}.log")
            stderr_path = os.path.join(self.paths.stderr_dir, f"{eval_id}.log")
            save_stdout_with_limit(stdout_path, "")
            file_io_utils.save_text("", stderr_path)

            source_path = os.path.join(self.paths.submissions_dir, f"{eval_id}.py")
            file_io_utils.save_text(raw_code.rstrip() + "\n", source_path)
            attempt_number = self.eval_manager.register_attempt(eval_id, source_path)
            if attempt_number is None:
                continue

            run_request = {
                "eval_id": eval_id,
                "attempt": int(attempt_number),
                "created_timestamp": time.time(),
                "source": "baseline",
            }
            run_request_path = os.path.join(self.paths.run_requests_dir, f"{eval_id}_attempt_{attempt_number}.yaml")
            file_io_utils.save_yaml(run_request, run_request_path, sort_keys=False)
            self._push_log_event("research_baseline_seeded", {"eval_id": eval_id, "source_path": source_path})

    def has_pending_claude_sessions(self) -> bool:
        return self.has_active_coder_sessions()

    def has_active_claude_sessions(self) -> bool:
        return self.has_active_coder_sessions()

    def _check_all_running_exceeded_double_timeout(self) -> Tuple[bool, List[Dict[str, Any]]]:
        running_evals = [
            info
            for info in self.eval_manager.get_active_evaluations()
            if info.get("status") == "attempt_running"
        ]
        if not running_evals:
            return False, []

        threshold = self.timeout * 2
        all_exceeded = all((info.get("elapsed_seconds", 0) >= threshold) for info in running_evals)
        return all_exceeded, running_evals

    from .executor_sandbox import _execute_submission_in_python_sandbox
