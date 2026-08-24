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

import os
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Any

import httpx
import openai
from openai import OpenAI

from station import constants
from station import file_io_utils
from station import runtime_api_config
from station.eval_external import pending_queue


class ExternalReportTransientError(RuntimeError):
    pass


class AutoExternalReporter:
    """
    Background processor for External Counter submissions.
    """

    _active_instances = {}
    _OPENAI_ENV_NAMES = (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_HTTP_PROXY",
        "OPENAI_HTTPS_PROXY",
    )
    _EXTERNAL_ENV_NAMES = (
        "EXTERNAL_OPENAI_API_KEY",
        "EXTERNAL_OPENAI_BASE_URL",
        "EXTERNAL_HTTP_PROXY",
        "EXTERNAL_HTTPS_PROXY",
    )

    def __init__(self, station_instance, enabled: bool = None, log_queue=None):
        station_id = id(station_instance)
        if station_id in self._active_instances and self._active_instances[station_id].is_running:
            print("AutoExternalReporter: WARNING - Another reporter instance already exists for this station")

        self.station = station_instance
        self.enabled = enabled if enabled is not None else constants.AUTO_EVAL_EXTERNAL_REPORT
        self.check_interval = constants.EXTERNAL_REPORT_CHECK_INTERVAL
        self.timeout_seconds = constants.EXTERNAL_REPORT_TIMEOUT_SECONDS
        self.model_name = constants.EXTERNAL_REPORT_MODEL_NAME
        self.max_retries = constants.EXTERNAL_REPORT_MAX_RETRIES
        self.retry_max_delay = constants.EXTERNAL_REPORT_RETRY_MAX_DELAY_SECONDS
        self.streaming = constants.EXTERNAL_REPORT_STREAMING
        self.prompt_template = constants.EXTERNAL_REPORT_PROMPT_TEMPLATE
        self.max_parallel_workers = constants.EXTERNAL_REPORT_MAX_PARALLEL_WORKERS
        self.log_queue = log_queue

        self.is_running = False
        self.reporter_thread: Optional[threading.Thread] = None
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.active_futures: Dict[str, Future] = {}

        self.external_room_path = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            constants.SHORT_ROOM_NAME_EXTERNAL
        )
        self.pending_reports_file = os.path.join(
            self.external_room_path,
            constants.PENDING_EXTERNAL_REPORTS_FILENAME
        )
        self.reports_dir = os.path.join(
            self.external_room_path,
            constants.EXTERNAL_REPORTS_SUBDIR_NAME
        )

        file_io_utils.ensure_dir_exists(self.reports_dir)

        self._active_instances[station_id] = self

    def _push_log_event(self, event_type: str, data: Dict[str, Any]):
        if self.log_queue:
            self.log_queue.put_nowait({"event": event_type, "data": data, "timestamp": time.time()})

    def start_reporter_loop(self) -> bool:
        if not self.enabled:
            print("AutoExternalReporter: External reporting is disabled")
            return False
        if self.is_running:
            print("AutoExternalReporter: Reporter loop already running")
            return True

        self._recover_interrupted_reports()
        self.is_running = True
        if not self.thread_pool:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_parallel_workers)
        self.reporter_thread = threading.Thread(target=self._reporter_loop, daemon=True)
        self.reporter_thread.start()
        print("AutoExternalReporter: Reporter loop started")
        return True

    def stop_reporter_loop(self, wait: bool = True):
        self.is_running = False
        thread_pool = self.thread_pool
        self.thread_pool = None
        if thread_pool:
            thread_pool.shutdown(wait=wait)
        print("AutoExternalReporter: Reporter loop stopped")

    def has_pending_or_running(self) -> bool:
        stats = self.get_lightweight_job_statistics()
        return bool(stats["running_jobs"] or stats["queued_jobs"])

    def has_drainable_pending_or_running(self) -> bool:
        stats = self.get_lightweight_job_statistics()
        return any(
            job.get("drainable", True) is not False
            for job in stats["running_jobs"] + stats["queued_jobs"]
        )

    def has_failed_requeued(self) -> bool:
        stats = self.get_lightweight_job_statistics()
        return any(
            job.get("drainable", True) is False
            for job in stats["running_jobs"] + stats["queued_jobs"]
        )

    def get_job_statistics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return persisted queued/running External Counter jobs for station statistics."""
        reports_by_id = {}
        running_jobs = []
        queued_jobs = []

        for report in self._load_all_reports():
            report_id = report.get(constants.EXTERNAL_REPORT_ID_KEY)
            if report_id is None:
                continue
            report_id = str(report_id)
            reports_by_id[report_id] = report
            status = report.get(constants.EXTERNAL_REPORT_STATUS_KEY)
            job = dict(report)
            job["job_type"] = "external_report"
            job["drainable"] = not bool(
                report.get(constants.EXTERNAL_REPORT_REQUEUE_COUNT_KEY)
            )
            if status == constants.EXTERNAL_REPORT_STATUS_RUNNING:
                running_jobs.append(job)
            elif status == constants.EXTERNAL_REPORT_STATUS_PENDING:
                queued_jobs.append(job)

        queued_ids = {str(job.get(constants.EXTERNAL_REPORT_ID_KEY)) for job in queued_jobs}
        for queued_report in self._load_pending_reports():
            report_id = queued_report.get(constants.EXTERNAL_REPORT_ID_KEY)
            if report_id is None:
                continue
            report_id = str(report_id)
            persisted = reports_by_id.get(report_id)
            status = (persisted or queued_report).get(constants.EXTERNAL_REPORT_STATUS_KEY)
            if status != constants.EXTERNAL_REPORT_STATUS_PENDING or report_id in queued_ids:
                continue
            job = dict(persisted or queued_report)
            job["job_type"] = "external_report"
            job["drainable"] = not bool(
                job.get(constants.EXTERNAL_REPORT_REQUEUE_COUNT_KEY)
            )
            queued_jobs.append(job)
            queued_ids.add(report_id)

        return {"running_jobs": running_jobs, "queued_jobs": queued_jobs}

    def get_lightweight_job_statistics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return dashboard jobs without scanning completed report files."""
        running_jobs = []
        for report_id in sorted(self.active_futures):
            report = self._load_report(str(report_id)) or {}
            running_jobs.append({
                constants.EXTERNAL_REPORT_ID_KEY: str(report_id),
                "evaluation_id": str(report_id),
                constants.EXTERNAL_REPORT_STATUS_KEY: constants.EXTERNAL_REPORT_STATUS_RUNNING,
                constants.EXTERNAL_REPORT_TITLE_KEY: report.get(constants.EXTERNAL_REPORT_TITLE_KEY, ""),
                constants.EXTERNAL_REPORT_AUTHOR_KEY: report.get(constants.EXTERNAL_REPORT_AUTHOR_KEY, ""),
                "agent_name": report.get(constants.EXTERNAL_REPORT_AUTHOR_KEY, ""),
                "job_type": "external_report",
                "execution_source": "external_surveyor",
                "drainable": not bool(report.get(constants.EXTERNAL_REPORT_REQUEUE_COUNT_KEY)),
            })

        queued_jobs = []
        try:
            pending_reports = self._load_pending_reports()
        except Exception:
            pending_reports = []
        active_ids = set(self.active_futures)
        for report in pending_reports:
            report_id = str(report.get(constants.EXTERNAL_REPORT_ID_KEY) or "").strip()
            if not report_id or report_id in active_ids:
                continue
            queued_jobs.append({
                constants.EXTERNAL_REPORT_ID_KEY: report_id,
                "evaluation_id": report_id,
                constants.EXTERNAL_REPORT_STATUS_KEY: constants.EXTERNAL_REPORT_STATUS_PENDING,
                constants.EXTERNAL_REPORT_TITLE_KEY: report.get(constants.EXTERNAL_REPORT_TITLE_KEY, ""),
                constants.EXTERNAL_REPORT_AUTHOR_KEY: report.get(constants.EXTERNAL_REPORT_AUTHOR_KEY, ""),
                "agent_name": report.get(constants.EXTERNAL_REPORT_AUTHOR_KEY, ""),
                "job_type": "external_report",
                "execution_source": "external_surveyor",
                "drainable": not bool(report.get(constants.EXTERNAL_REPORT_REQUEUE_COUNT_KEY)),
            })
        return {"running_jobs": running_jobs, "queued_jobs": queued_jobs}

    def should_wait_at_tick(self, current_tick: int) -> bool:
        max_allowed_ticks = constants.EXTERNAL_REPORT_MAX_TICK

        for report in self._load_all_reports():
            status = report.get(constants.EXTERNAL_REPORT_STATUS_KEY)
            if status not in (constants.EXTERNAL_REPORT_STATUS_PENDING, constants.EXTERNAL_REPORT_STATUS_RUNNING):
                continue
            submitted_tick = report.get(constants.EXTERNAL_REPORT_SUBMITTED_TICK_KEY)
            if submitted_tick is None:
                continue
            elapsed_ticks = current_tick - submitted_tick + 1
            if elapsed_ticks >= max_allowed_ticks:
                return True

        pending_reports = self._load_pending_reports()
        for report in pending_reports:
            report_id = report.get(constants.EXTERNAL_REPORT_ID_KEY)
            if report_id:
                report_entry = self._load_report(report_id)
                if report_entry:
                    status = report_entry.get(constants.EXTERNAL_REPORT_STATUS_KEY)
                    if status in (
                        constants.EXTERNAL_REPORT_STATUS_COMPLETED,
                        constants.EXTERNAL_REPORT_STATUS_FAILED,
                    ):
                        continue
            submitted_tick = report.get(constants.EXTERNAL_REPORT_SUBMITTED_TICK_KEY)
            if submitted_tick is None:
                continue
            elapsed_ticks = current_tick - submitted_tick + 1
            if elapsed_ticks >= max_allowed_ticks:
                return True

        return False

    def _reporter_loop(self):
        while self.is_running:
            try:
                self._check_completed_futures()
                queue_items = self._get_queue_items()

                for item in queue_items:
                    if not self.is_running:
                        break

                    report_id = str(item.get(constants.EXTERNAL_REPORT_ID_KEY))
                    if report_id in self.active_futures:
                        continue

                    if len(self.active_futures) >= self.max_parallel_workers:
                        break

                    if not self._pop_from_queue(item):
                        continue

                    title = item.get(constants.EXTERNAL_REPORT_TITLE_KEY, "Untitled")
                    author = item.get(constants.EXTERNAL_REPORT_AUTHOR_KEY, "Unknown")
                    print(f"AutoExternalReporter: Starting external report {report_id} for {author} - {title}")
                    self._push_log_event("auto_external_event", {
                        "report_id": report_id,
                        "author": author,
                        "title": title,
                        "status": "starting",
                    })
                    self._mark_report_running(report_id)
                    future = self.thread_pool.submit(self._run_report, item)
                    self.active_futures[report_id] = future
            except Exception as e:
                print(f"AutoExternalReporter: Loop error: {e}")
                traceback.print_exc()
            time.sleep(self.check_interval)

    def _check_completed_futures(self):
        if not self.active_futures:
            return
        completed_ids = []
        for report_id, future in self.active_futures.items():
            if future.done():
                completed_ids.append(report_id)
                try:
                    future.result()
                except Exception as e:
                    print(f"AutoExternalReporter: Report {report_id} completed with error: {e}")
        for report_id in completed_ids:
            del self.active_futures[report_id]

    def _get_queue_items(self) -> List[Dict[str, Any]]:
        items = []
        now = time.time()
        pending_reports = self._load_pending_reports()
        for entry in pending_reports:
            report_id = entry.get(constants.EXTERNAL_REPORT_ID_KEY)
            if not report_id:
                continue
            report_entry = self._load_report(report_id)
            if report_entry:
                status = report_entry.get(constants.EXTERNAL_REPORT_STATUS_KEY)
                if status in (
                    constants.EXTERNAL_REPORT_STATUS_RUNNING,
                    constants.EXTERNAL_REPORT_STATUS_COMPLETED,
                    constants.EXTERNAL_REPORT_STATUS_FAILED,
                ):
                    continue
            try:
                next_retry_at = float(entry.get(constants.EXTERNAL_REPORT_NEXT_RETRY_AT_KEY) or 0.0)
            except (TypeError, ValueError):
                next_retry_at = 0.0
            if next_retry_at > now:
                continue
            items.append(entry)
        return items

    def _load_pending_reports(self) -> List[Dict[str, Any]]:
        try:
            return pending_queue.load(self.pending_reports_file)
        except Exception as e:
            print(f"AutoExternalReporter: Failed to load pending reports: {e}")
            return []

    def _load_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        report_path = os.path.join(self.reports_dir, f"report_{report_id}.yaml")
        return file_io_utils.load_yaml(report_path)

    def _load_all_reports(self) -> List[Dict[str, Any]]:
        reports = []
        for filename in file_io_utils.list_files(self.reports_dir, constants.YAML_EXTENSION):
            if not filename.startswith("report_"):
                continue
            report_path = os.path.join(self.reports_dir, filename)
            report_data = file_io_utils.load_yaml(report_path)
            if isinstance(report_data, dict):
                reports.append(report_data)
        return reports

    def _pop_from_queue(self, item: Dict[str, Any]) -> bool:
        report_id = item.get(constants.EXTERNAL_REPORT_ID_KEY)
        if not report_id:
            return False
        return self._remove_from_pending_reports(str(report_id))

    def _remove_from_pending_reports(self, report_id: str) -> bool:
        try:
            return pending_queue.remove(
                self.pending_reports_file,
                report_id,
                constants.EXTERNAL_REPORT_ID_KEY,
            )
        except Exception as e:
            print(f"AutoExternalReporter: Failed to remove report {report_id}: {e}")
            return False

    def _recover_interrupted_reports(self) -> None:
        recovered = 0
        for report in self._load_all_reports():
            status = report.get(constants.EXTERNAL_REPORT_STATUS_KEY)
            if status not in (
                constants.EXTERNAL_REPORT_STATUS_PENDING,
                constants.EXTERNAL_REPORT_STATUS_RUNNING,
            ):
                continue
            report_id = report.get(constants.EXTERNAL_REPORT_ID_KEY)
            if report_id is None:
                continue
            if status == constants.EXTERNAL_REPORT_STATUS_RUNNING:
                report[constants.EXTERNAL_REPORT_STATUS_KEY] = constants.EXTERNAL_REPORT_STATUS_PENDING
                report.pop(constants.EXTERNAL_REPORT_START_TICK_KEY, None)
                file_io_utils.save_yaml(
                    report,
                    os.path.join(self.reports_dir, f"report_{report_id}.yaml"),
                )
            if pending_queue.append(
                self.pending_reports_file,
                report,
                constants.EXTERNAL_REPORT_ID_KEY,
            ):
                recovered += 1
        if recovered:
            print(f"AutoExternalReporter: Recovered {recovered} interrupted external report(s)")

    def _mark_report_running(self, report_id: str):
        report_entry = self._load_report(report_id)
        if not report_entry:
            return
        report_entry[constants.EXTERNAL_REPORT_STATUS_KEY] = constants.EXTERNAL_REPORT_STATUS_RUNNING
        report_entry[constants.EXTERNAL_REPORT_START_TICK_KEY] = self.station._get_current_tick()
        report_entry.pop(constants.EXTERNAL_REPORT_NEXT_RETRY_AT_KEY, None)
        file_io_utils.save_yaml(report_entry, os.path.join(self.reports_dir, f"report_{report_id}.yaml"))

    def _run_report(self, item: Dict[str, Any]):
        report_id = str(item.get(constants.EXTERNAL_REPORT_ID_KEY))
        author = item.get(constants.EXTERNAL_REPORT_AUTHOR_KEY, "Unknown")
        title = item.get(constants.EXTERNAL_REPORT_TITLE_KEY, "Untitled")
        prompt = item.get(constants.EXTERNAL_REPORT_PROMPT_KEY, "")

        report_prompt = self.prompt_template.format(query=prompt)
        runtime_snapshot = self._get_openai_runtime_snapshot()
        provider_retry_state: Dict[str, Any] = {}
        for attempt in range(self.max_retries + 1):
            try:
                client = self._build_openai_client(runtime_snapshot)
                report_text = self._request_report(client, report_prompt)
                if not report_text.strip():
                    raise ValueError("Empty report returned from external research.")

                self._record_openai_provider_success(runtime_snapshot)
                self._update_report_completed(report_id, report_text)
                print(f"AutoExternalReporter: Completed external report {report_id} for {author} - {title}")
                self._push_log_event("auto_external_event", {
                    "report_id": report_id,
                    "author": author,
                    "title": title,
                    "status": "completed",
                })
                self._send_report_notification(author, report_id, title, report_text)
                return
            except Exception as e:
                if attempt < self.max_retries:
                    runtime_snapshot = self._advance_openai_provider_fallback(
                        runtime_snapshot,
                        provider_retry_state,
                        e,
                    )
                    delay = self._retry_delay_seconds(attempt, provider_retry_state)
                    retry_timing = f"in {delay:.1f}s" if delay > 0 else "immediately"
                    print(
                        f"AutoExternalReporter: Failure for report {report_id}; "
                        f"retrying {retry_timing} ({attempt + 1}/{self.max_retries}): "
                        f"{type(e).__name__}: {e}"
                    )
                    self._push_log_event("auto_external_event", {
                        "report_id": report_id,
                        "author": author,
                        "title": title,
                        "status": "retrying",
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                    })
                    if delay > 0:
                        time.sleep(delay)
                    continue

                error_message = f"{type(e).__name__}: {e}"
                self._record_openai_provider_failure(runtime_snapshot)

                self._handle_report_failure(report_id, author, title, error_message)
                return

    @staticmethod
    def _provider_endpoint(runtime_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = runtime_snapshot.get("provider_endpoint")
        return endpoint if isinstance(endpoint, dict) else {}

    def _uses_openai_provider_fallback(self, runtime_snapshot: Dict[str, Any]) -> bool:
        if runtime_snapshot.get("external_override"):
            return False
        return bool(self._provider_endpoint(runtime_snapshot).get("configured"))

    def _record_openai_provider_success(self, runtime_snapshot: Dict[str, Any]) -> None:
        if not self._uses_openai_provider_fallback(runtime_snapshot):
            return
        endpoint = self._provider_endpoint(runtime_snapshot)
        runtime_api_config.record_provider_success("openai", endpoint.get("index"))

    def _record_openai_provider_failure(self, runtime_snapshot: Dict[str, Any]) -> None:
        if not self._uses_openai_provider_fallback(runtime_snapshot):
            return
        endpoint = self._provider_endpoint(runtime_snapshot)
        runtime_api_config.record_provider_failure(
            "openai",
            endpoint.get("index"),
            promote_default=False,
        )

    def _advance_openai_provider_fallback(
        self,
        runtime_snapshot: Dict[str, Any],
        retry_state: Dict[str, Any],
        error: Exception,
    ) -> Dict[str, Any]:
        """Retry one OpenAI endpoint twice, then rotate to the next configured endpoint."""
        fallback_enabled = self._uses_openai_provider_fallback(runtime_snapshot)
        retry_state["provider_fallback_enabled"] = fallback_enabled
        retry_state["cycle_wrapped"] = False
        if not fallback_enabled:
            return runtime_snapshot

        endpoint = self._provider_endpoint(runtime_snapshot)
        endpoint_index = endpoint.get("index")
        if retry_state.get("endpoint_index") != endpoint_index:
            retry_state["endpoint_index"] = endpoint_index
            retry_state["failure_streak"] = 0
        decision = runtime_api_config.advance_provider_fallback_after_failure(
            "openai",
            endpoint_index,
            int(retry_state.get("failure_streak", 0)),
            self._OPENAI_ENV_NAMES,
        )
        retry_state["failure_streak"] = int(decision.get("failure_streak", 0))
        if not decision.get("handled"):
            return runtime_snapshot

        if decision.get("retry_same_endpoint"):
            print(
                "AutoExternalReporter: OpenAI endpoint "
                f"{endpoint.get('name', endpoint_index)} (index={endpoint_index}) failed "
                f"({type(error).__name__}: {error}); retrying the same endpoint once."
            )
            return runtime_snapshot

        retry_snapshot = decision["retry_snapshot"]
        retry_snapshot["external_override"] = False
        retry_endpoint = self._provider_endpoint(retry_snapshot)
        print(
            "AutoExternalReporter: OpenAI endpoint "
            f"{endpoint.get('name', endpoint_index)} (index={endpoint_index}) failed "
            f"({type(error).__name__}: {error}); switching to "
            f"{retry_endpoint.get('name', retry_endpoint.get('index'))} "
            f"(index={retry_endpoint.get('index')}, "
            f"base_url={retry_snapshot.get('env', {}).get('OPENAI_BASE_URL') or 'provider_default'})."
        )
        retry_state["endpoint_index"] = retry_endpoint.get("index")
        retry_state["failure_streak"] = 0
        if decision.get("cycle_wrapped"):
            retry_state["cycle_wrapped"] = True
            retry_state["cycle_count"] = int(retry_state.get("cycle_count", 0)) + 1
        return retry_snapshot

    def _request_report(self, client: OpenAI, report_prompt: str) -> str:
        request_params = {
            "model": self.model_name,
            "input": report_prompt,
            "tools": [{"type": "web_search_preview"}],
            "stream": self.streaming,
        }

        response = client.responses.create(**request_params)
        if not self.streaming:
            return response.output_text or ""

        text_parts = []
        completed_response = None
        for event in response:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                text_parts.append(getattr(event, "delta", ""))
            elif event_type == "response.completed":
                completed_response = getattr(event, "response", None)
            elif event_type in {"response.failed", "response.incomplete", "error"}:
                error = getattr(getattr(event, "response", None), "error", None) or getattr(event, "error", None)
                raise ExternalReportTransientError(f"External report stream failed: {error or event_type}")

        report_text = "".join(text_parts)
        if not report_text and completed_response is not None:
            report_text = getattr(completed_response, "output_text", "") or ""
        return report_text

    @staticmethod
    def _retry_delay_seconds(attempt: int, retry_state: Dict[str, Any]) -> float:
        if retry_state.get("provider_fallback_enabled"):
            if not retry_state.get("cycle_wrapped"):
                return 0.0
            attempt_number = int(retry_state.get("cycle_count", 1))
        else:
            attempt_number = attempt + 1

        doubling_level = min((max(attempt_number, 1) - 1) // 2, 4)
        return float(constants.LLM_RETRY_DELAY_SECONDS * (2 ** doubling_level))

    def _pause_orchestrator(self, reason: str) -> None:
        orchestrator = getattr(self.station, "orchestrator", None)
        if orchestrator and hasattr(orchestrator, "pause_due_to_research_issue"):
            orchestrator.pause_due_to_research_issue(reason)
            return
        print(f"AutoExternalReporter: Unable to pause orchestrator directly. Reason: {reason}")

    def _get_openai_runtime_snapshot(self) -> Dict[str, Any]:
        env_values = runtime_api_config.get_env_values(self._EXTERNAL_ENV_NAMES)
        external_override = any(env_values.get(name) not in (None, "") for name in self._EXTERNAL_ENV_NAMES)
        if external_override:
            env_values.update(runtime_api_config.get_env_values(self._OPENAI_ENV_NAMES))
            return {
                "env": env_values,
                "external_override": True,
                "provider_endpoint": {},
            }

        snapshot = runtime_api_config.get_config_snapshot(
            self._OPENAI_ENV_NAMES,
            provider_id="openai",
        )
        snapshot["external_override"] = False
        return snapshot

    def _build_openai_client(self, runtime_snapshot: Optional[Dict[str, Any]] = None) -> OpenAI:
        """
        Build an OpenAI client, optionally using EXTERNAL_* overrides.
        Uses the shared OpenAI provider endpoint when overrides are not set.
        """
        runtime_snapshot = runtime_snapshot or self._get_openai_runtime_snapshot()
        env_values = runtime_snapshot.get("env", {})
        external_override = bool(runtime_snapshot.get("external_override"))
        api_key = env_values.get("EXTERNAL_OPENAI_API_KEY") or env_values.get("OPENAI_API_KEY") or None
        base_url = env_values.get("EXTERNAL_OPENAI_BASE_URL") or env_values.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        if external_override:
            http_proxy = env_values.get("EXTERNAL_HTTP_PROXY") or None
            https_proxy = env_values.get("EXTERNAL_HTTPS_PROXY") or None
        else:
            http_proxy = env_values.get("OPENAI_HTTP_PROXY") or None
            https_proxy = env_values.get("OPENAI_HTTPS_PROXY") or None

        if http_proxy and https_proxy and http_proxy != https_proxy:
            transport_http = httpx.HTTPTransport(proxy=http_proxy, trust_env=False)
            transport_https = httpx.HTTPTransport(proxy=https_proxy, trust_env=False)
            http_client = httpx.Client(
                mounts={"http://": transport_http, "https://": transport_https},
                timeout=self.timeout_seconds,
                trust_env=False,
            )
        else:
            proxy_url = https_proxy or http_proxy
            http_client = httpx.Client(
                proxy=proxy_url,
                timeout=self.timeout_seconds,
                trust_env=False,
            )
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
            http_client=http_client,
        )

    def _update_report_completed(self, report_id: str, report_text: str):
        report_entry = self._load_report(report_id) or {}
        report_entry[constants.EXTERNAL_REPORT_STATUS_KEY] = constants.EXTERNAL_REPORT_STATUS_COMPLETED
        report_entry[constants.EXTERNAL_REPORT_COMPLETED_TICK_KEY] = self.station._get_current_tick()
        report_entry[constants.EXTERNAL_REPORT_CONTENT_KEY] = report_text
        report_entry.pop(constants.EXTERNAL_REPORT_ERROR_KEY, None)
        report_entry.pop(constants.EXTERNAL_REPORT_REQUEUE_COUNT_KEY, None)
        report_entry.pop(constants.EXTERNAL_REPORT_NEXT_RETRY_AT_KEY, None)
        report_path = os.path.join(self.reports_dir, f"report_{report_id}.yaml")
        file_io_utils.save_yaml(report_entry, report_path)

    def _handle_report_failure(
        self,
        report_id: str,
        author: str,
        title: str,
        error_message: str,
    ) -> None:
        traceback.print_exc()
        self._pause_orchestrator(
            f'External report "{title}" (ID: {report_id}) failed and is being requeued: '
            f"{error_message}"
        )
        requeued = self._requeue_report_after_failure(report_id, error_message)
        status = "requeued" if requeued else "failed_to_requeue"
        self._push_log_event("auto_external_error", {
            "report_id": report_id,
            "author": author,
            "title": title,
            "status": status,
            "error": error_message,
            "trace": traceback.format_exc(),
        })

        if requeued:
            print(
                f"AutoExternalReporter: Requeued external report {report_id} and pausing station: "
                f"{error_message}"
            )
        else:
            print(
                f"AutoExternalReporter: Failed to requeue external report {report_id}; "
                f"pausing station: {error_message}"
            )

    def _requeue_report_after_failure(self, report_id: str, error_message: str) -> bool:
        report_entry = self._load_report(report_id) or {}
        try:
            requeue_count = int(report_entry.get(constants.EXTERNAL_REPORT_REQUEUE_COUNT_KEY) or 0) + 1
        except (TypeError, ValueError):
            requeue_count = 1
        delay = min(
            self.retry_max_delay * (2 ** min(requeue_count - 1, 4)),
            constants.EXTERNAL_REPORT_REQUEUE_MAX_DELAY_SECONDS,
        )
        report_entry[constants.EXTERNAL_REPORT_STATUS_KEY] = constants.EXTERNAL_REPORT_STATUS_PENDING
        report_entry[constants.EXTERNAL_REPORT_ERROR_KEY] = error_message
        report_entry[constants.EXTERNAL_REPORT_REQUEUE_COUNT_KEY] = requeue_count
        report_entry[constants.EXTERNAL_REPORT_NEXT_RETRY_AT_KEY] = time.time() + delay
        report_entry.pop(constants.EXTERNAL_REPORT_START_TICK_KEY, None)
        report_entry.pop(constants.EXTERNAL_REPORT_COMPLETED_TICK_KEY, None)
        report_path = os.path.join(self.reports_dir, f"report_{report_id}.yaml")
        try:
            file_io_utils.save_yaml(report_entry, report_path)
            pending_queue.append(
                self.pending_reports_file,
                report_entry,
                constants.EXTERNAL_REPORT_ID_KEY,
            )
            return True
        except Exception as e:
            print(f"AutoExternalReporter: Failed to requeue report {report_id}: {e}")
            return False

    def _send_report_notification(self, author: str, report_id: str, title: str, report_text: str):
        message = (
            f"Your external report request \"{title}\" (ID: {report_id}) has completed.\n\n"
            f"**External Report:**\n"
            f"{report_text}"
        )
        try:
            self.station.agent_module.add_pending_notification_atomic(author, message)
        except Exception as e:
            print(f"AutoExternalReporter: Failed to send notification to {author}: {e}")
