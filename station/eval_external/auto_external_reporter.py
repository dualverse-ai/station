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
import fcntl
import yaml
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Any

import httpx
from openai import OpenAI

from station import constants
from station import file_io_utils
from station import runtime_api_config


class AutoExternalReporter:
    """
    Background processor for External Counter submissions.
    """

    _active_instances = {}

    def __init__(self, station_instance, enabled: bool = None, log_queue=None):
        station_id = id(station_instance)
        if station_id in self._active_instances and self._active_instances[station_id].is_running:
            print("AutoExternalReporter: WARNING - Another reporter instance already exists for this station")

        self.station = station_instance
        self.enabled = enabled if enabled is not None else constants.AUTO_EVAL_EXTERNAL_REPORT
        self.check_interval = constants.EXTERNAL_REPORT_CHECK_INTERVAL
        self.timeout_seconds = constants.EXTERNAL_REPORT_TIMEOUT_SECONDS
        self.model_name = constants.EXTERNAL_REPORT_MODEL_NAME
        self.max_tool_calls = constants.EXTERNAL_REPORT_MAX_TOOL_CALLS
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

        self.is_running = True
        if not self.thread_pool:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_parallel_workers)
        self.reporter_thread = threading.Thread(target=self._reporter_loop, daemon=True)
        self.reporter_thread.start()
        print("AutoExternalReporter: Reporter loop started")
        return True

    def stop_reporter_loop(self):
        self.is_running = False
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
            self.thread_pool = None
        print("AutoExternalReporter: Reporter loop stopped")

    def has_pending_or_running(self) -> bool:
        pending_reports = self._load_pending_reports()
        if pending_reports:
            return True

        for report in self._load_all_reports():
            if report.get(constants.EXTERNAL_REPORT_STATUS_KEY) == constants.EXTERNAL_REPORT_STATUS_RUNNING:
                return True
        return False

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
            items.append(entry)
        return items

    def _load_pending_reports(self) -> List[Dict[str, Any]]:
        if not file_io_utils.file_exists(self.pending_reports_file):
            return []
        try:
            pending = file_io_utils.load_yaml_lines(self.pending_reports_file)
            return [entry for entry in pending if isinstance(entry, dict)]
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
        max_retries = 5
        retry_delay = 0.1

        for attempt in range(max_retries):
            try:
                if not file_io_utils.file_exists(self.pending_reports_file):
                    return False

                with open(self.pending_reports_file, "r+", encoding="utf-8") as f:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        content = f.read()
                        if not content.strip():
                            return False

                        all_entries = file_io_utils.load_yaml_lines(self.pending_reports_file)
                        remaining = []
                        found = False
                        for entry in all_entries:
                            if entry.get(constants.EXTERNAL_REPORT_ID_KEY) != report_id:
                                remaining.append(entry)
                            else:
                                found = True

                        if not found:
                            return False

                        f.seek(0)
                        f.truncate()
                        for entry in remaining:
                            f.write("---\n")
                            yaml.dump(entry, f, default_flow_style=False, allow_unicode=True)
                        return True
                    except (IOError, OSError) as lock_error:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                        raise lock_error
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    print(f"AutoExternalReporter: Failed to remove report {report_id}: {e}")
                    return False
        return False

    def _mark_report_running(self, report_id: str):
        report_entry = self._load_report(report_id)
        if not report_entry:
            return
        report_entry[constants.EXTERNAL_REPORT_STATUS_KEY] = constants.EXTERNAL_REPORT_STATUS_RUNNING
        report_entry[constants.EXTERNAL_REPORT_START_TICK_KEY] = self.station._get_current_tick()
        file_io_utils.save_yaml(report_entry, os.path.join(self.reports_dir, f"report_{report_id}.yaml"))

    def _run_report(self, item: Dict[str, Any]):
        report_id = str(item.get(constants.EXTERNAL_REPORT_ID_KEY))
        author = item.get(constants.EXTERNAL_REPORT_AUTHOR_KEY, "Unknown")
        title = item.get(constants.EXTERNAL_REPORT_TITLE_KEY, "Untitled")
        prompt = item.get(constants.EXTERNAL_REPORT_PROMPT_KEY, "")

        try:
            client = self._build_openai_client()
            response = client.responses.create(
                model=self.model_name,
                input=self.prompt_template.format(query=prompt),
                tools=[{"type": "web_search_preview"}],
                max_tool_calls=self.max_tool_calls,
            )
            report_text = response.output_text or ""
            if not report_text.strip():
                raise ValueError("Empty report returned from external research.")

            self._update_report_completed(report_id, report_text)
            print(f"AutoExternalReporter: Completed external report {report_id} for {author} - {title}")
            self._push_log_event("auto_external_event", {
                "report_id": report_id,
                "author": author,
                "title": title,
                "status": "completed",
            })
            self._send_report_notification(author, report_id, title, report_text)
        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            print(f"AutoExternalReporter: Failed external report {report_id} for {author} - {title}: {error_message}")
            traceback.print_exc()
            self._push_log_event("auto_external_error", {
                "report_id": report_id,
                "author": author,
                "title": title,
                "status": "failed",
                "error": error_message,
                "trace": traceback.format_exc(),
            })
            self._update_report_failed(report_id, error_message)
            self._send_report_failure(author, report_id, title, error_message)

    def _build_openai_client(self) -> OpenAI:
        """
        Build an OpenAI client, optionally using EXTERNAL_* overrides.
        Falls back to default environment variables when overrides are not set.
        """
        env_values = runtime_api_config.get_env_values([
            "EXTERNAL_OPENAI_API_KEY",
            "EXTERNAL_OPENAI_BASE_URL",
            "EXTERNAL_HTTP_PROXY",
            "EXTERNAL_HTTPS_PROXY",
        ])
        api_key = env_values.get("EXTERNAL_OPENAI_API_KEY") or None
        base_url_raw = env_values.get("EXTERNAL_OPENAI_BASE_URL")
        base_url = base_url_raw if base_url_raw is not None else "https://api.openai.com/v1"
        http_proxy = env_values.get("EXTERNAL_HTTP_PROXY") or constants.LLM_HTTP_PROXY
        https_proxy = env_values.get("EXTERNAL_HTTPS_PROXY") or constants.LLM_HTTPS_PROXY

        if http_proxy or https_proxy:
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
                http_client = httpx.Client(proxy=proxy_url, timeout=self.timeout_seconds, trust_env=False)
            return OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout_seconds, http_client=http_client)

        if api_key or base_url:
            return OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout_seconds)

        return OpenAI(timeout=self.timeout_seconds)

    def _update_report_completed(self, report_id: str, report_text: str):
        report_entry = self._load_report(report_id) or {}
        report_entry[constants.EXTERNAL_REPORT_STATUS_KEY] = constants.EXTERNAL_REPORT_STATUS_COMPLETED
        report_entry[constants.EXTERNAL_REPORT_COMPLETED_TICK_KEY] = self.station._get_current_tick()
        report_entry[constants.EXTERNAL_REPORT_CONTENT_KEY] = report_text
        report_path = os.path.join(self.reports_dir, f"report_{report_id}.yaml")
        file_io_utils.save_yaml(report_entry, report_path)

    def _update_report_failed(self, report_id: str, error_message: str):
        report_entry = self._load_report(report_id) or {}
        report_entry[constants.EXTERNAL_REPORT_STATUS_KEY] = constants.EXTERNAL_REPORT_STATUS_FAILED
        report_entry[constants.EXTERNAL_REPORT_COMPLETED_TICK_KEY] = self.station._get_current_tick()
        report_entry[constants.EXTERNAL_REPORT_ERROR_KEY] = error_message
        report_path = os.path.join(self.reports_dir, f"report_{report_id}.yaml")
        file_io_utils.save_yaml(report_entry, report_path)

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

    def _send_report_failure(self, author: str, report_id: str, title: str, error_message: str):
        message = (
            f"Your external report request \"{title}\" (ID: {report_id}) failed.\n\n"
            f"Status: failed\n"
            f"Error: {error_message}"
        )
        try:
            self.station.agent_module.add_pending_notification_atomic(author, message)
        except Exception as e:
            print(f"AutoExternalReporter: Failed to send failure notification to {author}: {e}")
