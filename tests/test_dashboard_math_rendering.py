import unittest
from pathlib import Path


class DashboardRenderingTests(unittest.TestCase):
    def setUp(self):
        self.template = Path("web_interface/templates/dashboard.html").read_text(encoding="utf-8")
        self.javascript = Path("web_interface/static/js/dashboard.js").read_text(encoding="utf-8")
        self.styles = Path("web_interface/static/css/style.css").read_text(encoding="utf-8")
        self.app_source = Path("web_interface/app.py").read_text(encoding="utf-8")
        self.start_source = Path("start.sh").read_text(encoding="utf-8")
        self.stop_source = Path("stop.sh").read_text(encoding="utf-8")

    def test_dashboard_loads_pinned_katex_marked_extension(self):
        marked_script = "marked@12.0.2/lib/marked.umd.js"
        yaml_script = "js-yaml@4.1.0/dist/js-yaml.min.js"
        katex_script = "katex@0.16.22/dist/katex.min.js"
        extension_script = "marked-katex-extension@5.1.5/lib/index.umd.js"

        self.assertIn(marked_script, self.template)
        self.assertIn(yaml_script, self.template)
        self.assertIn("katex@0.16.22/dist/katex.min.css", self.template)
        self.assertIn(katex_script, self.template)
        self.assertIn(extension_script, self.template)
        self.assertLess(self.template.index(katex_script), self.template.index(extension_script))

    def test_shared_markdown_renderer_normalizes_station_latex_delimiters(self):
        self.assertIn("markedKatexFactory", self.javascript)
        self.assertIn("normalizeLatexDelimitersForMarked", self.javascript)
        self.assertIn("dashboardMathRenderingEnabled", self.javascript)
        self.assertIn("throwOnError: false", self.javascript)
        self.assertIn("nonStandard: true", self.javascript)
        self.assertIn("output: 'html'", self.javascript)
        self.assertIn("STATION_PROTECTED_CODE", self.javascript)

    def test_display_math_can_scroll_inside_dashboard_panels(self):
        self.assertIn(".markdown-content-host .katex-display", self.styles)
        self.assertIn("overflow-x: auto", self.styles)
        self.assertIn("min-width: max-content", self.styles)

    def test_shared_markdown_host_has_complete_typography(self):
        self.assertIn("Canonical rendered Markdown typography", self.styles)
        self.assertIn(".markdown-content-host h1", self.styles)
        self.assertIn("font-size: 1.65em", self.styles)
        self.assertIn(".markdown-content-host ul li::before", self.styles)
        self.assertIn("list-style: disc outside", self.styles)
        self.assertIn("content: none !important", self.styles)
        self.assertIn(".markdown-content-host p", self.styles)
        self.assertIn("margin: 0.65rem 0", self.styles)
        self.assertIn(".markdown-content-host table", self.styles)
        self.assertIn("white-space: pre !important", self.styles)

    def test_theme_selector_uses_industry_light_name_without_changing_value(self):
        self.assertIn('<option value="light">Light</option>', self.template)
        self.assertNotIn('<option value="light">White</option>', self.template)

    def test_more_tools_contains_task_spec_editor(self):
        self.assertIn('id="open-research-task-spec-modal-button"', self.template)
        self.assertIn('id="research-task-spec-modal"', self.template)
        self.assertIn('id="task-spec-preview-pane"', self.template)
        self.assertIn('id="task-spec-raw-editor"', self.template)
        self.assertIn("Preview renders Markdown and LaTeX", self.template)

    def test_task_editor_uses_shared_renderer_and_revision_checked_api(self):
        self.assertIn("renderMarkdownForDashboard(rawMarkdown", self.javascript)
        self.assertIn("'/api/station/research_task_spec', 'GET'", self.javascript)
        self.assertIn("'/api/station/research_task_spec', 'PUT'", self.javascript)
        self.assertIn("expected_revision: taskSpecState.revision", self.javascript)
        self.assertIn("taskSpecRawEditor.addEventListener('input'", self.javascript)

    def test_task_editor_backend_routes_are_authenticated(self):
        route = "@app.route('/api/station/research_task_spec'"
        self.assertEqual(self.app_source.count(route), 2)
        self.assertIn("def get_research_task_spec", self.app_source)
        self.assertIn("def update_research_task_spec", self.app_source)
        self.assertGreaterEqual(self.app_source.count("@auth_required"), 2)

    def test_task_editor_has_responsive_full_height_layout(self):
        self.assertIn(".task-spec-modal-content", self.styles)
        self.assertIn("height: 90vh", self.styles)
        self.assertIn("#task-spec-raw-editor", self.styles)

    def test_task_preview_uses_canonical_dialogue_markdown_host(self):
        preview_marker = 'id="task-spec-preview-pane" class="task-spec-pane markdown-content-host"'
        self.assertIn(preview_marker, self.template)

    def test_live_updates_use_one_cursor_handler_and_bounded_broadcast_backend(self):
        self.assertIn("DashboardEventBroker", self.app_source)
        self.assertIn("orchestrator_event_broker.read_after", self.app_source)
        self.assertIn('"X-Accel-Buffering": "no"', self.app_source)
        self.assertNotIn("orchestrator_log_queue", self.app_source)
        self.assertIn("function processLiveEvent(eventData)", self.javascript)
        self.assertIn("liveEventCursorParam()", self.javascript)
        self.assertIn("data.events) ? data.events : []).forEach(processLiveEvent)", self.javascript)
        self.assertIn("processLiveEvent(JSON.parse(event.data))", self.javascript)
        self.assertIn("document.addEventListener('visibilitychange'", self.javascript)
        self.assertNotIn("SSE live log stream connected.", self.javascript)
        self.assertNotIn("getOrchestratorStatus();\n            fetchRecentEvents();", self.javascript)
        self.assertIn("isRefreshingAgentSelector", self.javascript)
        self.assertIn("agentSelectorDashboard.value === currentSelectedAgentForDialogueView", self.javascript)
        visibility_start = self.javascript.index("document.addEventListener('visibilitychange'")
        visibility_end = self.javascript.index("if (copySystemPromptButton)", visibility_start)
        visibility_handler = self.javascript[visibility_start:visibility_end]
        self.assertNotIn("handleAgentDialogueViewChange", visibility_handler)
        self.assertNotIn("liveEventCursor = null", visibility_handler)

    def test_dialogue_history_load_is_bounded_retryable_and_staged(self):
        self.assertIn("const HISTORY_REQUEST_TIMEOUT_MS = 60000", self.javascript)
        self.assertIn("const HISTORY_RENDER_CHUNK_SIZE = 12", self.javascript)
        self.assertIn("const HISTORY_RENDER_CHUNK_TEXT_BUDGET = 24000", self.javascript)
        self.assertIn("async function fetchHistoryWithRetry", self.javascript)
        self.assertIn("new AbortController()", self.javascript)
        self.assertIn("await renderHistoryEntries", self.javascript)
        self.assertIn("window.requestAnimationFrame", self.javascript)
        self.assertIn("requestId !== historyLoadSequence", self.javascript)
        self.assertIn("bufferedLiveEventsDuringHistory", self.javascript)
        self.assertIn(".forEach(appendLiveEventToAgentDialogue)", self.javascript)
        self.assertIn("latestHistoryTimestamp", self.javascript)
        self.assertIn("eventTimestamp > loadedHistoryTimestampCutoff", self.javascript)
        self.assertIn("liveDialogueFingerprint", self.javascript)
        self.assertIn("renderedAgentDialogueFingerprints.has(dialogueFingerprint)", self.javascript)
        self.assertIn("bubbleWrapper.dataset.dialogueFingerprint", self.javascript)
        self.assertIn("latestTickAttemptView", self.javascript)
        self.assertIn("find(attempt => attempt.hasAgentResponse)", self.javascript)
        self.assertIn("pendingLiveStationObservations", self.javascript)
        self.assertIn("removeRenderedDialogueTick", self.javascript)
        self.assertIn("data-dialogue-tick", self.javascript)
        self.assertIn("renderedAgentDialogueFingerprints.delete(existingFingerprint)", self.javascript)
        self.assertIn("stickToBottom: !loadFullHistory && historyWindowMode === 'recent'", self.javascript)
        self.assertIn("renderedTextInChunk >= HISTORY_RENDER_CHUNK_TEXT_BUDGET", self.javascript)
        render_start = self.javascript.index("async function renderHistoryEntries")
        render_end = self.javascript.index("function cancelActiveHistoryRequest", render_start)
        render_history_source = self.javascript[render_start:render_end]
        self.assertEqual(1, render_history_source.count("agentSpecificBubbleLog.appendChild(renderFragment)"))

    def test_production_transport_leaves_capacity_for_history_requests(self):
        self.assertIn("GUNICORN_THREADS=${GUNICORN_THREADS:-8}", self.start_source)
        self.assertEqual(2, self.start_source.count('--threads "$GUNICORN_THREADS"'))
        self.assertIn("gzip_types application/json text/plain text/css application/javascript", self.start_source)
        self.assertNotIn("text/event-stream", self.start_source)
        self.assertIn("wait_timeout=5", self.app_source)

    def test_more_tools_are_grouped_by_function(self):
        expected_groups = ["Agent Controls", "Research", "Communication", "Administration"]
        positions = [self.template.index(f">{label}</div>") for label in expected_groups]
        self.assertEqual(positions, sorted(positions))
        self.assertGreaterEqual(self.template.count('class="more-tools-group"'), 4)
        self.assertIn(".more-tools-group + .more-tools-group", self.styles)
        send_system_position = self.template.index('id="open-send-system-message-modal-button"')
        intervene_position = self.template.index('id="open-direct-message-modal-button"')
        admin_position = self.template.index(">Administration</div>")
        research_label_position = self.template.index(">Research</div>")
        surveyor_position = self.template.index('id="open-web-archive-surveys-modal-button"')
        task_spec_position = self.template.index('id="open-research-task-spec-modal-button"')
        self.assertLess(send_system_position, intervene_position)
        self.assertLess(intervene_position, admin_position)
        self.assertLess(research_label_position, surveyor_position)
        self.assertLess(surveyor_position, task_spec_position)

    def test_dashboard_archive_surveyor_has_isolated_friendly_frontend(self):
        expected_ids = [
            "open-web-archive-surveys-modal-button",
            "web-archive-surveys-modal",
            "close-web-archive-surveys-modal-button",
            "open-web-archive-survey-request-button",
            "web-archive-survey-request-modal",
            "web-archive-survey-prompt",
            "submit-web-archive-survey-button",
            "web-archive-surveys-table-body",
            "remove-web-archive-survey-button",
            "copy-web-archive-survey-markdown-button",
        ]
        for element_id in expected_ids:
            self.assertIn(f'id="{element_id}"', self.template)
        self.assertIn("/api/web/archive-surveys", self.javascript)
        self.assertNotIn("data-remove-web-survey-id", self.javascript)
        self.assertNotIn("scheduleWebArchiveSurveyPolling", self.javascript)
        self.assertIn("webArchiveSurveysLoaded", self.javascript)
        self.assertIn("archive-survey-row.is-clickable", self.javascript)
        self.assertIn("archive-papers-row archive-survey-row", self.javascript)
        self.assertIn(".archive-survey-row.is-disabled", self.styles)
        self.assertIn("Survey report Markdown copied to clipboard", self.javascript)
        self.assertIn("Submitted Tick", self.template)
        for row_select_id in (
            "archive-row-count-select",
            "web-archive-survey-row-count-select",
            "question-page-size-select",
        ):
            self.assertIn(f'id="{row_select_id}"', self.template)
        for pagination_id in (
            "archive-previous-page-button",
            "archive-next-page-button",
            "archive-page-summary",
            "web-archive-survey-previous-page-button",
            "web-archive-survey-next-page-button",
            "web-archive-survey-page-summary",
        ):
            self.assertIn(f'id="{pagination_id}"', self.template)
        self.assertGreaterEqual(self.template.count('<option value="100" selected>100</option>'), 3)
        self.assertIn(".table-row-count-control", self.styles)
        self.assertIn("justify-self: end", self.styles)
        self.assertIn("let archivePage = 1", self.javascript)
        self.assertIn("let webArchiveSurveyPage = 1", self.javascript)
        self.assertIn("key: 'accepted_tick', direction: 'desc'", self.javascript)
        self.assertIn("sortBy: 'authored_tick'", self.javascript)
        self.assertIn("sortDirection: 'desc'", self.javascript)
        self.assertIn("archivePapersCache.slice(pageStart, pageStart + archiveRowCount)", self.javascript)
        self.assertIn("webArchiveSurveys.slice(pageStart, pageStart + webArchiveSurveyRowCount)", self.javascript)
        self.assertNotIn("archivePapersCache.slice(0, archiveRowCount)", self.javascript)
        self.assertNotIn("webArchiveSurveys.slice(0, webArchiveSurveyRowCount)", self.javascript)
        self.assertIn("survey.source_tick", self.javascript)
        self.assertIn("Archive ID: #${numericId}", self.javascript)
        self.assertIn("Survey ID: W${surveyId}", self.javascript)
        self.assertIn(">Surveyor</button>", self.template)
        self.assertNotIn("archive-human-surveyor-tab-button", self.template)
        self.assertNotIn("archive-modal-tabs", self.template)
        self.assertNotIn("Human Surveyor", self.template)
        self.assertNotIn("archive-paper-select-checkbox", self.template)
        self.assertNotIn("archive-paper-select-checkbox", self.javascript)
        self.assertNotIn("selected_archive_ids:", self.javascript)
        self.assertIn('id="web-archive-survey-report-shell" class="archive-papers-pane', self.template)
        self.assertNotIn('id="web-archive-survey-request"', self.template)
        self.assertIn("marked complete, but its report content is unavailable", self.javascript)
        self.assertNotIn("Report ready — click to open", self.javascript)
        self.assertIn("Report is running. Select Refresh to check again.", self.javascript)
        self.assertIn("Select a paper to read the full publication.", self.template)
        self.assertIn("Select a question to read the agent discussion.", self.template)
        self.assertNotIn("Indexed activity view", self.template)
        self.assertNotIn("Scrollable table", self.template)
        self.assertIn("Back to Surveys", self.template)
        self.assertNotIn("Previous reports", self.template)
        self.assertNotIn("archive-survey-history-shell", self.template)
        self.assertNotIn("archive-survey-table", self.template)
        self.assertIn(".archive-survey-request-modal-content", self.styles)
        self.assertEqual(self.template.count("archive-surveyor-header-button"), 2)
        self.assertGreaterEqual(self.template.count("archive-survey-action-button"), 8)
        self.assertIn("font-size: 0.875rem", self.styles)
        remove_start = self.styles.index(".archive-survey-remove-button {")
        remove_end = self.styles.index("}", remove_start)
        self.assertNotIn("font-size", self.styles[remove_start:remove_end])
        self.assertIn("justify-content: center", self.styles)
        self.assertIn(".archive-survey-status.status-running::before", self.styles)
        self.assertIn(".archive-survey-status.status-running", self.styles)

    def test_modal_close_icon_is_optically_centered(self):
        close_start = self.styles.index(".close-button {")
        close_end = self.styles.index("}", close_start)
        close_block = self.styles[close_start:close_end]
        self.assertIn("align-items: center", close_block)
        self.assertIn("justify-content: center", close_block)
        self.assertIn("line-height: 1", close_block)

    def test_archive_family_modal_buttons_center_their_labels(self):
        self.assertIn(
            ".archive-papers-modal-content button:not(.archive-sort-button):not(.question-sort-button)",
            self.styles,
        )
        self.assertIn(".archive-survey-request-modal-content button", self.styles)
        self.assertIn("justify-content: center", self.styles)
        self.assertIn("text-align: center", self.styles)
        self.assertIn('class="archive-detail-actions flex gap-3"', self.template)
        self.assertIn(".archive-detail-actions button", self.styles)
        self.assertIn("white-space: nowrap", self.styles)
        self.assertIn("#archive-paper-detail-title", self.styles)
        self.assertIn("overflow-wrap: anywhere", self.styles)

    def test_web_archive_survey_routes_use_separate_service(self):
        self.assertIn("WebArchiveSurveyService", self.app_source)
        self.assertIn("def web_archive_survey_submit_route", self.app_source)
        self.assertIn("def web_archive_survey_delete_route", self.app_source)
        self.assertIn("build_question_survey_preview", self.app_source)
        self.assertIn("Failed to start or recover web Archive Surveyor", self.app_source)
        self.assertNotIn("queue_archive_survey_request(", self.app_source)

    def test_safe_restart_drains_and_force_restart_requeues_web_surveys(self):
        self.assertIn("read_web_archive_survey_counts", self.stop_source)
        self.assertIn("wait_for_station_jobs_to_finish", self.stop_source)
        self.assertIn('"running_jobs"', self.stop_source)
        self.assertIn('"drainable_running_jobs_count"', self.stop_source)
        self.assertIn('"drainable_queued_jobs_count"', self.stop_source)
        self.assertIn("including Research, Archive Surveyor, and External Counter requests", self.stop_source)
        self.assertIn("web Surveyor running", self.stop_source)
        self.assertIn("Active External Counter and web Surveyor requests may be requeued", self.stop_source)

    def test_external_jobs_use_external_dashboard_fields(self):
        self.assertIn("exp.job_type === 'external_report'", self.javascript)
        self.assertIn("`External #${reportId}", self.javascript)
        self.assertIn("external surveyor", self.javascript)

    def test_external_failure_restart_is_backward_compatible(self):
        self.assertIn("SKIP_EXTERNAL_RETRY_DRAIN", self.stop_source)
        self.assertIn('job.get("job_type") == "external_report"', self.stop_source)
        self.assertIn("EXTERNAL_REPORT_STATUS_PENDING", self.start_source)
        self.assertIn("EXTERNAL_REPORT_STATUS_RUNNING", self.start_source)
        self.assertIn("pending_queue.remove", self.start_source)

    def test_fragmented_thinking_normalizer_is_used_across_dashboard_surfaces(self):
        self.assertIn("function normalizeThinkingWhitespace", self.javascript)
        self.assertIn("newlineDensity >= 0.055", self.javascript)
        self.assertIn("averageLineLength <= 24", self.javascript)
        self.assertGreaterEqual(self.javascript.count("normalizeThinkingWhitespace("), 5)

    def test_thinking_text_has_readable_line_height(self):
        self.assertIn(".thinking-block .whitespace-pre-wrap", self.styles)
        self.assertIn("line-height: 1.45", self.styles)

    def test_thinking_is_collapsible_and_closed_by_default(self):
        self.assertIn("function renderCollapsibleThinking", self.javascript)
        self.assertIn('<details class="thinking-block', self.javascript)
        self.assertNotIn('<details class="thinking-block border-l-4 border-sky-700 bg-slate-800/50 mb-2 text-slate-400 text-xs rounded" open>', self.javascript)
        self.assertIn("content: 'Show thinking'", self.styles)
        self.assertIn("content: 'Hide thinking'", self.styles)
        self.assertGreaterEqual(self.javascript.count("renderCollapsibleThinking("), 4)

    def test_station_action_yaml_blocks_render_as_markdown_cards(self):
        self.assertIn("function enhanceStationYamlBlocks", self.javascript)
        self.assertIn("function enhanceStationBareYamlActions", self.javascript)
        self.assertIn("STATIONBAREYAMLCODE", self.javascript)
        self.assertIn("window.jsyaml.FAILSAFE_SCHEMA", self.javascript)
        self.assertIn("meta: 'Agent Meta Prompt'", self.javascript)
        self.assertIn("submit: 'Research Submission'", self.javascript)
        self.assertIn("reply: 'Capsule Reply'", self.javascript)
        self.assertIn("['prompt', 'Prompt']", self.javascript)
        self.assertIn("'description', 'prompt'", self.javascript)
        self.assertIn("enhanceYamlCards: false", self.javascript)
        self.assertNotIn("View raw YAML", self.javascript)
        self.assertIn(".station-yaml-card", self.styles)
        self.assertGreaterEqual(self.javascript.count("</article>`.trim();"), 2)
        self.assertIn("function protectDashboardEmbeddedHtml", self.javascript)
        self.assertIn("function restoreDashboardEmbeddedHtml", self.javascript)
        self.assertIn("STATIONEMBEDDEDHTML", self.javascript)
        self.assertIn("enhanceStationBareYamlActions(markdownSource, embeddedHtml)", self.javascript)
        self.assertIn('<details class="station-yaml-instructions" open>', self.javascript)
        self.assertIn('class="station-yaml-copy-source" hidden', self.javascript)
        self.assertIn("copySource.value", self.javascript)

    def test_station_response_meta_prompt_field_renders_as_markdown_card(self):
        self.assertIn("function enhanceAgentMetaPromptFields", self.javascript)
        self.assertIn("Agent Meta Prompt:", self.javascript)
        self.assertIn("renderAgentMetaPromptCard", self.javascript)
        self.assertNotIn("View raw prompt", self.javascript)
        self.assertIn("data-copy-kind=\"Meta prompt\"", self.javascript)
        self.assertIn(".station-meta-prompt-card", self.styles)

    def test_loose_markdown_lists_keep_marker_and_text_on_one_line(self):
        self.assertIn(".markdown-content-host li > p:first-child", self.styles)
        self.assertIn("display: inline", self.styles)

    def test_historical_dialogue_does_not_invent_render_time(self):
        self.assertIn("function formatEventTime", self.javascript)
        self.assertIn("timestamp: logEntry.timestamp ?? null", self.javascript)
        self.assertNotIn("200000 - logEntry.tick*1000", self.javascript)
        self.assertIn("const timeSuffix = timeStr ? ` @ ${timeStr}` : ''", self.javascript)

    def test_legacy_storage_reads_normalize_nested_markdown_fences(self):
        self.assertIn("function repairStationCodeFences", self.javascript)
        self.assertIn("(?:Storage|Code) Read", self.javascript)
        self.assertIn("const nestedFences = content.match(/`{3,}/g) || []", self.javascript)
        self.assertIn("const extension = prefix.match", self.javascript)
        self.assertIn("function normalizeStorageReadFences", self.javascript)
        self.assertIn("function enhanceStorageReadBlocks", self.javascript)
        self.assertIn("const closingFences = Array.from", self.javascript)
        self.assertIn("const safeFence = '`'.repeat", self.javascript)
        self.assertIn('<details class="station-storage-read">', self.javascript)
        self.assertNotIn('<details class="station-storage-read" open>', self.javascript)
        self.assertIn("content: 'Show file'", self.styles)
        self.assertIn("content: 'Hide file'", self.styles)
        self.assertLess(
            self.javascript.index("repairStationCodeFences(markdownText)"),
            self.javascript.index("normalizeStorageReadFences(repairedReadSource)"),
        )
        self.assertLess(
            self.javascript.index("normalizeStorageReadFences(repairedReadSource)"),
            self.javascript.index("normalizeLatexDelimitersForMarked(normalizedActionSource)"),
        )

    def test_truncated_storage_read_closes_before_system_message_notice(self):
        self.assertIn("SYSTEM_MESSAGES_TRUNCATION_NOTICE", self.javascript)
        self.assertIn("const truncatedAtNotice", self.javascript)
        self.assertIn("if (!closingFences.length && !truncatedAtNotice) continue", self.javascript)
        self.assertIn("const closingPrefix = content.endsWith('\\n') ? '' : '\\n'", self.javascript)

    def test_station_action_commands_have_dedicated_styles(self):
        self.assertIn("function normalizeStationActionCommands", self.javascript)
        self.assertIn("/\\/execute_action\\{[^}\\n]+\\}/g", self.javascript)
        self.assertIn("function styleStationActionCommandHtml", self.javascript)
        self.assertIn("category = 'navigation'", self.javascript)
        self.assertIn("category = 'read'", self.javascript)
        self.assertIn("category = 'write'", self.javascript)
        self.assertIn("const actionBodyHtml", self.javascript)
        self.assertIn("code.station-action-command", self.styles)
        self.assertIn("code.station-action-navigation", self.styles)
        self.assertIn("code.station-action-read", self.styles)
        self.assertIn("code.station-action-write", self.styles)
        self.assertIn("content: 'ACTION'", self.styles)
        self.assertNotIn("content: 'STATION ACTION'", self.styles)


if __name__ == "__main__":
    unittest.main()
