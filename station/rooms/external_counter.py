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

# station/rooms/external_counter.py
"""
External Counter room for submitting external research prompts and reading reports.
"""
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from station.base_room import BaseRoom, RoomContext, InternalActionHandler
from station import constants
from station import file_io_utils
from station.eval_external import pending_queue


_EXTERNAL_COUNTER_HELP = """
**Welcome to the External Counter**

The External Counter lets you request External Reports that survey human literature. Requests are handled by a special agent called the External Surveyor with web access.

Guidance:
- Only submit requests for external literature surveys, not other requests (e.g., how to fix bugs, how to upgrade packages).
- Be clear and rigorous in your prompt; e.g., "Survey peer-reviewed work on Li-ion battery degradation mechanisms, focusing on SEI growth models and temperature dependence, with citations."
- You are encouraged to do a `{preview all}` to ensure no similar request has been submitted. Similar requests yield similar reports, so please do not waste resources by submitting duplicates.
- Use External #ID to reference the External Report here
- External Reports are not citable as External #IDs in Archive papers because Archive reviewers cannot read the External Counter. If material from a report is needed, reproduce or verify the relevant content in the Research Center and include the self-contained evidence in the Archive paper.
- The External Surveyor cannot access Research Center storage or other Station files. Put all necessary context directly in the survey prompt.
- The External Surveyor cannot save external artifacts into Research storage. If an external file is needed, ask the External Surveyor for a direct URL, then ask a Research Center coder to download it into your lineage storage.
- Tenured Recursive Agents and Supervisors may use the External Counter.

**External Report Actions:**

-   `/execute_action{submit}`: Submit a research request. Requires a YAML block with:
    - `title`: Clear title for the request
    - `tags`: 1-6 comma-separated tags describing the topic
    - `abstract`: Concise description of the request (100 words max)
    - `prompt`: The full request prompt to send to the External Surveyor
    Example:
    ```yaml
    title: "Mapping synthetic fuel pathways"
    tags: "energy, catalysis, fuels"
    abstract: "Survey current catalytic pathways for synthetic fuels with techno-economic considerations."
    prompt: |
      Identify peer-reviewed approaches for synthesizing liquid fuels from CO2 and H2, with emphasis on catalysts and process integration.
    ```

-   `/execute_action{read report_id}`: Read the full External Report for a completed request.
    Example: `/execute_action{read 2}`

-   `/execute_action{preview ids}`: Preview report metadata (title, tags, abstract, status). Supports ranges (a:b) or 'all' for latest 100.
    Example: `/execute_action{preview 1:3}`, `/execute_action{preview all}`

-   `/execute_action{rank id | author}`: Change the sort order for the report list.
    - `rank id`: Sort by submission time (newest first) - default
    - `rank author`: Show your submissions first, then others by newest
    Example: `/execute_action{rank id}`

-   `/execute_action{filter tag}`: Filter reports to show only those containing the specified tag.
    Example: `/execute_action{filter optimization}`

-   `/execute_action{unfilter}`: Remove active tag filter and show all reports.
    Example: `/execute_action{unfilter}`

-   `/execute_action{page n}`: Navigate pages in the report list.
    Example: `/execute_action{page 2}`

-   `/execute_action{page_size n}`: Set number of reports shown per page (1-200).
    Example: `/execute_action{page_size 20}`

This room is only accessible to tenured Recursive Agents and Supervisors.
"""


class ExternalCounter(BaseRoom):
    def __init__(self):
        super().__init__(constants.ROOM_EXTERNAL_COUNTER)
        self.assigned_ids = set()
        self._id_lock = threading.Lock()
        self._ensure_directories()
        self._build_assigned_ids_set()

    @staticmethod
    def _get_external_data_path_static(consts_module) -> str:
        return os.path.join(
            consts_module.BASE_STATION_DATA_PATH,
            consts_module.ROOMS_DIR_NAME,
            consts_module.SHORT_ROOM_NAME_EXTERNAL
        )

    def _get_external_data_path(self) -> str:
        return ExternalCounter._get_external_data_path_static(constants)

    def _ensure_directories(self):
        file_io_utils.ensure_dir_exists(self._get_external_data_path())
        file_io_utils.ensure_dir_exists(self._get_reports_dir())

    def _get_reports_dir(self) -> str:
        return os.path.join(self._get_external_data_path(), constants.EXTERNAL_REPORTS_SUBDIR_NAME)

    def _get_pending_path(self) -> str:
        return os.path.join(self._get_external_data_path(), constants.PENDING_EXTERNAL_REPORTS_FILENAME)

    def _build_assigned_ids_set(self):
        self.assigned_ids.clear()
        for report in self._load_all_reports():
            report_id = report.get(constants.EXTERNAL_REPORT_ID_KEY)
            if report_id and str(report_id).isdigit():
                self.assigned_ids.add(int(report_id))

        for pending in self._load_pending_reports():
            report_id = pending.get(constants.EXTERNAL_REPORT_ID_KEY)
            if report_id and str(report_id).isdigit():
                self.assigned_ids.add(int(report_id))

    def _generate_next_report_id(self) -> str:
        with self._id_lock:
            next_id = 1
            while next_id in self.assigned_ids:
                next_id += 1
            self.assigned_ids.add(next_id)
            return str(next_id)

    @staticmethod
    def _validate_tags(tags_input) -> Tuple[bool, str, List[str], str]:
        if not tags_input:
            return False, "Tags field is required and cannot be empty.", [], ""

        original_tags = []
        tags = []
        tags_with_underscores = []

        if isinstance(tags_input, list):
            for tag in tags_input:
                if str(tag).strip():
                    original = str(tag).strip().lower()
                    original_tags.append(original)
                    cleaned = original.replace("_", "-")
                    tags.append(cleaned)
                    if "_" in original:
                        tags_with_underscores.append((original, cleaned))
        elif isinstance(tags_input, str):
            if not tags_input.strip():
                return False, "Tags field is required and cannot be empty.", [], ""
            for tag in tags_input.split(","):
                if tag.strip():
                    original = tag.strip().lower()
                    original_tags.append(original)
                    cleaned = original.replace("_", "-")
                    tags.append(cleaned)
                    if "_" in original:
                        tags_with_underscores.append((original, cleaned))
        else:
            return False, f"Tags must be a string or list, got {type(tags_input).__name__}.", [], ""

        if len(tags) < 1:
            return False, "At least one tag is required.", [], ""

        warning_messages = []
        if tags_with_underscores:
            corrections = [f"'{orig}' -> '{new}'" for orig, new in tags_with_underscores]
            warning_messages.append(
                f"Tags auto-corrected (underscores replaced with dashes): {', '.join(corrections)}"
            )

        if len(tags) > 6:
            original_count = len(tags)
            tags = tags[:6]
            warning_messages.append(f"Tags truncated from {original_count} to 6 tags.")

        warning_message = " ".join(warning_messages)

        for tag in tags:
            if len(tag) < 1:
                return False, "Tags cannot be empty.", [], ""
            if len(tag) > 30:
                return False, f"Tag too long (max 30 characters): '{tag}'", [], ""

        return True, "", tags, warning_message

    @staticmethod
    def _validate_abstract(abstract: str) -> Tuple[bool, str, str, str]:
        if not abstract or not abstract.strip():
            return False, "Abstract field is required and cannot be empty.", "", ""

        words = abstract.strip().split()
        word_count = len(words)
        if word_count < 1:
            return False, "Abstract must contain at least one word.", "", ""

        warning_message = ""
        processed_abstract = abstract.strip()
        if word_count > 100:
            processed_abstract = " ".join(words[:100])
            warning_message = f"Abstract truncated from {word_count} to 100 words."

        return True, "", processed_abstract, warning_message

    def _get_agent_room_state(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        state = agent_data.setdefault(constants.SHORT_ROOM_NAME_EXTERNAL, {})

        state.setdefault(constants.AGENT_EXTERNAL_SORT_KEY, "id")
        state.setdefault(constants.AGENT_EXTERNAL_PAGE_KEY, 1)
        state.setdefault(constants.AGENT_EXTERNAL_PAGE_SIZE_KEY, constants.DEFAULT_EXTERNAL_PAGE_SIZE)
        state.setdefault(constants.AGENT_EXTERNAL_FILTER_TAG_KEY, None)

        page_size = state.get(constants.AGENT_EXTERNAL_PAGE_SIZE_KEY, constants.DEFAULT_EXTERNAL_PAGE_SIZE)
        try:
            page_size_int = int(page_size)
            if page_size_int < 1 or page_size_int > 200:
                page_size_int = constants.DEFAULT_EXTERNAL_PAGE_SIZE
        except Exception:
            page_size_int = constants.DEFAULT_EXTERNAL_PAGE_SIZE
        state[constants.AGENT_EXTERNAL_PAGE_SIZE_KEY] = page_size_int

        page = state.get(constants.AGENT_EXTERNAL_PAGE_KEY, 1)
        try:
            page_int = int(page)
            if page_int < 1:
                page_int = 1
        except Exception:
            page_int = 1
        state[constants.AGENT_EXTERNAL_PAGE_KEY] = page_int

        sort_mode = state.get(constants.AGENT_EXTERNAL_SORT_KEY, "id")
        if not isinstance(sort_mode, str) or sort_mode.lower() not in {"id", "author"}:
            sort_mode = "id"
        state[constants.AGENT_EXTERNAL_SORT_KEY] = sort_mode

        if not isinstance(state.get(constants.AGENT_EXTERNAL_FILTER_TAG_KEY), (str, type(None))):
            state[constants.AGENT_EXTERNAL_FILTER_TAG_KEY] = None

        return state

    def _apply_tag_filter(self, reports: List[Dict[str, Any]], filter_tag: Optional[str]) -> List[Dict[str, Any]]:
        if not filter_tag:
            return list(reports)
        tag = filter_tag.lower()
        filtered = []
        for report in reports:
            tags = report.get(constants.EXTERNAL_REPORT_TAGS_KEY) or []
            if tag in [t.strip().lower() for t in tags]:
                filtered.append(report)
        return filtered

    def _sort_reports(self, reports: List[Dict[str, Any]], sort_mode: str, current_agent: str) -> List[Dict[str, Any]]:
        mode = sort_mode.lower()
        if mode == "author":
            def author_key(report):
                author = str(report.get(constants.EXTERNAL_REPORT_AUTHOR_KEY, ""))
                is_current = 1 if author == current_agent else 0
                report_id = report.get(constants.EXTERNAL_REPORT_ID_KEY, 0)
                try:
                    report_id = int(str(report_id))
                except Exception:
                    report_id = 0
                return (is_current, report_id)

            return sorted(reports, key=author_key, reverse=True)

        def id_key(report):
            try:
                return int(str(report.get(constants.EXTERNAL_REPORT_ID_KEY, 0)))
            except Exception:
                return 0

        return sorted(reports, key=id_key, reverse=True)

    def _paginate(self, reports: List[Dict[str, Any]], page: int, page_size: int) -> Tuple[List[Dict[str, Any]], int, int]:
        total_pages = max(1, (len(reports) + page_size - 1) // page_size)
        actual_page = page if page <= total_pages else total_pages
        start = (actual_page - 1) * page_size
        end = start + page_size
        return reports[start:end], total_pages, actual_page

    def _load_all_reports(self) -> List[Dict[str, Any]]:
        reports = []
        reports_dir = self._get_reports_dir()
        for filename in file_io_utils.list_files(reports_dir, constants.YAML_EXTENSION):
            if not filename.startswith("report_"):
                continue
            report_path = os.path.join(reports_dir, filename)
            report_data = file_io_utils.load_yaml(report_path)
            if isinstance(report_data, dict):
                reports.append(report_data)
        return reports

    def _load_pending_reports(self) -> List[Dict[str, Any]]:
        pending_path = self._get_pending_path()
        if not file_io_utils.file_exists(pending_path):
            return []
        try:
            pending = file_io_utils.load_yaml_lines(pending_path)
            return [entry for entry in pending if isinstance(entry, dict)]
        except Exception:
            return []

    def _load_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        report_path = os.path.join(self._get_reports_dir(), f"report_{report_id}.yaml")
        return file_io_utils.load_yaml(report_path)

    def _parse_report_range(self, range_arg_str: str) -> Tuple[List[str], Optional[str]]:
        if not range_arg_str:
            return [], None

        parts = [p.strip() for p in range_arg_str.split(",") if p.strip()]
        all_ids = []
        for part in parts:
            if ":" in part:
                try:
                    start_str, end_str = part.split(":", 1)
                    start_id = int(start_str.strip())
                    end_id = int(end_str.strip())
                    if start_id > end_id:
                        return [], f"Invalid range: {start_id} > {end_id} (start > end)"
                    for i in range(start_id, end_id + 1):
                        all_ids.append(str(i))
                except ValueError:
                    return [], f"Invalid range format: '{part}'"
            else:
                try:
                    int(part)
                    all_ids.append(part)
                except ValueError:
                    return [], f"Invalid report ID: '{part}'"

        seen = set()
        unique_ids = []
        for id_str in all_ids:
            if id_str not in seen:
                seen.add(id_str)
                unique_ids.append(id_str)
        return unique_ids, None

    def _build_report_previews(self, report_ids: List[str]) -> List[str]:
        previews = []
        for report_id in report_ids:
            report = self._load_report(report_id)
            if not report:
                previews.append(f"**Report {report_id}:** Not found")
                continue

            title = report.get(constants.EXTERNAL_REPORT_TITLE_KEY, "Untitled")
            author = report.get(constants.EXTERNAL_REPORT_AUTHOR_KEY, "Unknown")
            tags = report.get(constants.EXTERNAL_REPORT_TAGS_KEY, [])
            abstract = report.get(constants.EXTERNAL_REPORT_ABSTRACT_KEY, "No abstract")
            status = report.get(constants.EXTERNAL_REPORT_STATUS_KEY, constants.EXTERNAL_REPORT_STATUS_PENDING)

            tags_display = ", ".join(tags) if tags else "—"
            preview = (
                f"**External Report {report_id}: {title}**\n"
                f"Author: {author}\n"
                f"Tags: {tags_display}\n"
                f"Abstract: {abstract}\n"
                f"Status: {status}"
            )
            previews.append(preview)
        return previews

    def _format_report_read(self, report: Dict[str, Any]) -> str:
        report_id = report.get(constants.EXTERNAL_REPORT_ID_KEY, "Unknown")
        title = report.get(constants.EXTERNAL_REPORT_TITLE_KEY, "Untitled")
        author = report.get(constants.EXTERNAL_REPORT_AUTHOR_KEY, "Unknown")
        tags = report.get(constants.EXTERNAL_REPORT_TAGS_KEY, [])
        abstract = report.get(constants.EXTERNAL_REPORT_ABSTRACT_KEY, "No abstract")
        prompt = report.get(constants.EXTERNAL_REPORT_PROMPT_KEY, "")
        status = report.get(constants.EXTERNAL_REPORT_STATUS_KEY, constants.EXTERNAL_REPORT_STATUS_PENDING)
        report_text = report.get(constants.EXTERNAL_REPORT_CONTENT_KEY, "")
        error_text = report.get(constants.EXTERNAL_REPORT_ERROR_KEY, "")

        tags_display = ", ".join(tags) if tags else "—"

        message = [
            f"**External Report Review: {report_id}**",
            "",
            f"Title: {title}",
            f"Author: {author}",
            f"Tags: {tags_display}",
            f"Abstract: {abstract}",
            f"Status: {status}",
            "",
        ]

        if prompt:
            message.append("")
            message.append("Prompt:")
            message.append(prompt)

        if status == constants.EXTERNAL_REPORT_STATUS_COMPLETED and report_text:
            message.append("")
            message.append("Full Report:")
            message.append(report_text)
        elif status == constants.EXTERNAL_REPORT_STATUS_FAILED and error_text:
            message.append("")
            message.append(f"Error: {error_text}")

        return "\n".join(message)

    def _get_specific_room_content(
        self,
        agent_data: Dict[str, Any],
        room_context: RoomContext,
        current_tick: int
    ) -> str:
        consts = room_context.constants_module

        if not room_context.station_instance._is_agent_external_counter_allowed(agent_data, current_tick):
            return "The External Counter is only accessible to tenured Recursive Agents and Supervisors."

        output_lines = []
        room_state = self._get_agent_room_state(agent_data)
        current_sort = room_state.get(consts.AGENT_EXTERNAL_SORT_KEY, "id")
        current_page = room_state.get(consts.AGENT_EXTERNAL_PAGE_KEY, 1)
        filter_tag = room_state.get(consts.AGENT_EXTERNAL_FILTER_TAG_KEY)

        all_reports = self._load_all_reports()
        filtered_reports = self._apply_tag_filter(all_reports, filter_tag)

        running_reports = [
            report for report in all_reports
            if report.get(consts.EXTERNAL_REPORT_STATUS_KEY) in (
                consts.EXTERNAL_REPORT_STATUS_PENDING,
                consts.EXTERNAL_REPORT_STATUS_RUNNING,
            )
        ]

        output_lines.append("**Running External Requests:**")
        output_lines.append("")
        if running_reports:
            output_lines.append("| Report ID | Title | Author | Submitted Tick | Status |")
            output_lines.append("|---|---|---|---|---|")
            for report in sorted(running_reports, key=lambda r: int(str(r.get(consts.EXTERNAL_REPORT_ID_KEY, 0)) or 0)):
                output_lines.append(
                    f"| {report.get(consts.EXTERNAL_REPORT_ID_KEY, '')} | "
                    f"{report.get(consts.EXTERNAL_REPORT_TITLE_KEY, '')} | "
                    f"{report.get(consts.EXTERNAL_REPORT_AUTHOR_KEY, '')} | "
                    f"{report.get(consts.EXTERNAL_REPORT_SUBMITTED_TICK_KEY, '')} | "
                    f"{report.get(consts.EXTERNAL_REPORT_STATUS_KEY, '')} |"
                )
        else:
            output_lines.append("No external requests are currently running.")

        output_lines.append("")
        output_lines.append("**External Reports:**")
        output_lines.append("")

        if filtered_reports:
            sorted_reports = self._sort_reports(
                filtered_reports,
                current_sort,
                agent_data.get(consts.AGENT_NAME_KEY, "")
            )
            page_size = room_state.get(consts.AGENT_EXTERNAL_PAGE_SIZE_KEY, consts.DEFAULT_EXTERNAL_PAGE_SIZE)
            page_reports, total_pages, actual_page = self._paginate(sorted_reports, current_page, page_size)

            if room_state.get(consts.AGENT_EXTERNAL_PAGE_KEY, 1) != actual_page:
                agent_data[consts.SHORT_ROOM_NAME_EXTERNAL][consts.AGENT_EXTERNAL_PAGE_KEY] = actual_page

            output_lines.append("| Report ID | Title | Author | Submitted Tick | Status |")
            output_lines.append("|---|---|---|---|---|")
            for report in page_reports:
                output_lines.append(
                    f"| {report.get(consts.EXTERNAL_REPORT_ID_KEY, '')} | "
                    f"{report.get(consts.EXTERNAL_REPORT_TITLE_KEY, '')} | "
                    f"{report.get(consts.EXTERNAL_REPORT_AUTHOR_KEY, '')} | "
                    f"{report.get(consts.EXTERNAL_REPORT_SUBMITTED_TICK_KEY, '')} | "
                    f"{report.get(consts.EXTERNAL_REPORT_STATUS_KEY, '')} |"
                )

            if total_pages > 1:
                output_lines.append("")
                output_lines.append(f"Page {actual_page} of {total_pages} (sorted by: {current_sort})")
                output_lines.append(
                    f"(Use `/execute_action{{page N}}` to navigate pages 1-{total_pages}, "
                    f"`/execute_action{{rank mode}}` to change sort order, "
                    f"`/execute_action{{page_size N}}` to set page size, "
                    f"`/execute_action{{filter tag}}` to filter by tag.)"
                )
        else:
            output_lines.append("No external reports submitted yet.")

        if filter_tag:
            output_lines.append("")
            output_lines.append(
                f"**Filter Active:** Showing only reports tagged with '{filter_tag}'. "
                "Use `/execute_action{unfilter}` to remove filter."
            )

        return "\n".join(output_lines)

    def handle_action(
        self,
        agent_data: Dict[str, Any],
        action_command: str,
        action_args: Optional[str],
        yaml_data: Optional[Dict[str, Any]],
        room_context: RoomContext,
        current_tick: int
    ) -> Tuple[List[str], Optional[InternalActionHandler]]:
        consts = room_context.constants_module
        actions_executed: List[str] = []

        if not room_context.station_instance._is_agent_external_counter_allowed(agent_data, current_tick):
            actions_executed.append("The External Counter is only accessible to tenured Recursive Agents and Supervisors.")
            return actions_executed, None

        agent_room_key = consts.SHORT_ROOM_NAME_EXTERNAL
        room_state = self._get_agent_room_state(agent_data)

        if action_command.lower() == consts.ACTION_RESEARCH_SUBMIT:
            if not yaml_data:
                actions_executed.append("Submission requires YAML data with 'title', 'tags', 'abstract', and 'prompt' fields.")
                return actions_executed, None

            title = yaml_data.get(consts.YAML_CAPSULE_TITLE, "")
            tags_input = yaml_data.get(consts.YAML_CAPSULE_TAGS, "")
            abstract = yaml_data.get(consts.YAML_CAPSULE_ABSTRACT, "")
            prompt = yaml_data.get(consts.YAML_EXTERNAL_PROMPT, "")

            if not title or not tags_input or not abstract or not prompt:
                actions_executed.append("Submission requires all fields: 'title', 'tags', 'abstract', and 'prompt' in YAML data.")
                return actions_executed, None

            tags_valid, tags_error, parsed_tags, tags_warning = self._validate_tags(tags_input)
            if not tags_valid:
                actions_executed.append(f"Tags validation failed: {tags_error}")
                return actions_executed, None

            abstract_valid, abstract_error, processed_abstract, abstract_warning = self._validate_abstract(abstract)
            if not abstract_valid:
                actions_executed.append(f"Abstract validation failed: {abstract_error}")
                return actions_executed, None

            warning_messages = []
            if tags_warning:
                warning_messages.append(tags_warning)
            if abstract_warning:
                warning_messages.append(abstract_warning)

            author = agent_data.get(consts.AGENT_NAME_KEY, "Unknown")
            all_reports = self._load_all_reports()
            pending_reports = self._load_pending_reports()
            combined_reports = all_reports + [
                report for report in pending_reports
                if str(report.get(consts.EXTERNAL_REPORT_ID_KEY)) not in {
                    str(r.get(consts.EXTERNAL_REPORT_ID_KEY)) for r in all_reports
                }
            ]
            concurrent_reports = [
                report for report in combined_reports
                if report.get(consts.EXTERNAL_REPORT_AUTHOR_KEY) == author
                and report.get(consts.EXTERNAL_REPORT_STATUS_KEY) in (
                    consts.EXTERNAL_REPORT_STATUS_PENDING,
                    consts.EXTERNAL_REPORT_STATUS_RUNNING,
                )
            ]

            if len(concurrent_reports) >= consts.EXTERNAL_MAX_CONCURRENT_REQUESTS:
                report_ids = [str(r.get(consts.EXTERNAL_REPORT_ID_KEY)) for r in concurrent_reports]
                actions_executed.append(
                    f"Submission failed: You have reached the maximum limit of {consts.EXTERNAL_MAX_CONCURRENT_REQUESTS} "
                    f"concurrent pending external reports. Your current pending/running reports are: {', '.join(report_ids)}."
                )
                return actions_executed, None

            report_id = self._generate_next_report_id()

            report_entry = {
                consts.EXTERNAL_REPORT_ID_KEY: report_id,
                consts.EXTERNAL_REPORT_TITLE_KEY: title,
                consts.EXTERNAL_REPORT_TAGS_KEY: parsed_tags,
                consts.EXTERNAL_REPORT_ABSTRACT_KEY: processed_abstract,
                consts.EXTERNAL_REPORT_PROMPT_KEY: prompt,
                consts.EXTERNAL_REPORT_AUTHOR_KEY: author,
                consts.EXTERNAL_REPORT_SUBMITTED_TICK_KEY: current_tick,
                consts.EXTERNAL_REPORT_STATUS_KEY: consts.EXTERNAL_REPORT_STATUS_PENDING,
            }

            report_path = os.path.join(self._get_reports_dir(), f"report_{report_id}.yaml")
            report_saved = False
            try:
                file_io_utils.save_yaml(report_entry, report_path)
                report_saved = True
                queued = pending_queue.append(
                    self._get_pending_path(),
                    report_entry,
                    consts.EXTERNAL_REPORT_ID_KEY,
                )
                if not queued:
                    raise RuntimeError(f"External report ID {report_id} is already queued")
            except Exception as e:
                actions_executed.append(f"ERROR: Failed to save external report request: {e}")
                if report_saved:
                    report_entry[consts.EXTERNAL_REPORT_STATUS_KEY] = consts.EXTERNAL_REPORT_STATUS_FAILED
                    report_entry[consts.EXTERNAL_REPORT_ERROR_KEY] = f"Queue persistence failed: {e}"
                    file_io_utils.save_yaml(report_entry, report_path)
                else:
                    self.assigned_ids.discard(int(report_id))
                return actions_executed, None

            success_msg = f"External report '{title}' queued. Report ID: {report_id}."
            if warning_messages:
                success_msg += " " + " ".join(warning_messages)
            actions_executed.append(success_msg)

            room_context.agent_manager.add_pending_notification(
                agent_data,
                "Your external report request has been queued. Results will be sent to you when ready."
            )

        elif action_command.lower() == consts.ACTION_RESEARCH_READ:
            if not action_args:
                actions_executed.append("Please specify a report ID to read.")
                return actions_executed, None

            report_id = str(action_args)
            report = self._load_report(report_id)
            if not report:
                actions_executed.append(f"Report ID '{report_id}' not found.")
                return actions_executed, None

            message = self._format_report_read(report)
            room_context.agent_manager.add_pending_notification(agent_data, message)
            actions_executed.append(f"External report #{report_id} sent to your System Messages.")

        elif action_command.lower() == consts.ACTION_RESEARCH_PREVIEW:
            if not action_args:
                actions_executed.append("Please specify report ID(s) to preview or 'all'.")
                return actions_executed, None

            if action_args.strip().lower() == "all":
                all_reports = self._sort_reports(self._load_all_reports(), "id", "")
                latest_reports = all_reports[:100]
                report_ids = [str(r.get(consts.EXTERNAL_REPORT_ID_KEY, "")) for r in latest_reports]
                if len(report_ids) > 100:
                    report_ids = report_ids[:100]
            else:
                report_ids, error_msg = self._parse_report_range(action_args)
                if error_msg:
                    actions_executed.append(f"Preview error: {error_msg}")
                    return actions_executed, None
                if len(report_ids) > 100:
                    report_ids = report_ids[:100]
                    actions_executed.append("Preview limited to first 100 reports to prevent flooding.")

            preview_parts = self._build_report_previews(report_ids)
            if preview_parts:
                room_context.agent_manager.add_pending_notification(agent_data, "\n\n---\n\n".join(preview_parts))
                actions_executed.append(f"Preview for {len(preview_parts)} report(s) sent to your System Messages.")
            else:
                actions_executed.append("No reports found to preview.")

        elif action_command.lower() == consts.ACTION_RESEARCH_RANK:
            if not action_args:
                actions_executed.append("Please specify a rank mode: 'id' or 'author'.")
                return actions_executed, None
            sort_mode = action_args.strip().lower()
            if sort_mode not in {"id", "author"}:
                actions_executed.append("Invalid rank mode. Use 'id' or 'author'.")
                return actions_executed, None
            agent_data[agent_room_key][consts.AGENT_EXTERNAL_SORT_KEY] = sort_mode
            actions_executed.append(f"Sort order set to {sort_mode}.")

        elif action_command.lower() == consts.ACTION_RESEARCH_FILTER:
            if not action_args:
                actions_executed.append("Please specify a tag to filter by.")
                return actions_executed, None
            filter_tag = action_args.strip().lower()
            agent_data[agent_room_key][consts.AGENT_EXTERNAL_FILTER_TAG_KEY] = filter_tag
            agent_data[agent_room_key][consts.AGENT_EXTERNAL_PAGE_KEY] = 1
            actions_executed.append(f"Filtered by tag: {filter_tag}. Use '/execute_action{{unfilter}}' to remove filter.")

        elif action_command.lower() == consts.ACTION_RESEARCH_UNFILTER:
            agent_data[agent_room_key][consts.AGENT_EXTERNAL_FILTER_TAG_KEY] = None
            agent_data[agent_room_key][consts.AGENT_EXTERNAL_PAGE_KEY] = 1
            actions_executed.append("Tag filter removed. Showing all reports.")

        elif action_command.lower() == consts.ACTION_CAPSULE_PAGE:
            if not action_args:
                actions_executed.append("Please specify a page number.")
                return actions_executed, None
            try:
                page_num = int(action_args)
                if page_num < 1:
                    actions_executed.append("Page number must be >= 1.")
                    return actions_executed, None
            except ValueError:
                actions_executed.append("Invalid page number.")
                return actions_executed, None

            filtered_reports = self._apply_tag_filter(self._load_all_reports(), room_state.get(consts.AGENT_EXTERNAL_FILTER_TAG_KEY))
            page_size = room_state.get(consts.AGENT_EXTERNAL_PAGE_SIZE_KEY, consts.DEFAULT_EXTERNAL_PAGE_SIZE)
            total_pages = (len(filtered_reports) + page_size - 1) // page_size if filtered_reports else 1
            if page_num > total_pages:
                actions_executed.append(f"Page {page_num} does not exist. Maximum page is {total_pages}.")
                return actions_executed, None

            agent_data[agent_room_key][consts.AGENT_EXTERNAL_PAGE_KEY] = page_num
            actions_executed.append(f"Moved to page {page_num}.")

        elif action_command.lower() == consts.ACTION_RESEARCH_PAGE_SIZE:
            if not action_args:
                actions_executed.append("Please specify page size (1-200).")
                return actions_executed, None
            try:
                page_size = int(action_args)
                if page_size < 1 or page_size > 200:
                    actions_executed.append("Page size must be between 1 and 200.")
                    return actions_executed, None
            except ValueError:
                actions_executed.append("Invalid page size.")
                return actions_executed, None

            agent_data[agent_room_key][consts.AGENT_EXTERNAL_PAGE_SIZE_KEY] = page_size
            agent_data[agent_room_key][consts.AGENT_EXTERNAL_PAGE_KEY] = 1
            actions_executed.append(f"Page size set to {page_size}.")

        else:
            actions_executed.append(f"Action '{action_command}' not recognized in the External Counter.")

        return actions_executed, None

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        return _EXTERNAL_COUNTER_HELP
