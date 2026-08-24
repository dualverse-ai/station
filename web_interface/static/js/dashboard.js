/*
 * Copyright 2025 DualverseAI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// static/js/dashboard.js
document.addEventListener('DOMContentLoaded', () => {
    const operationMode = document.body.dataset.operationMode;
    const isMultistartPreview = operationMode === 'multistart_preview';
    // --- DOM Elements ---
    let stationTickDashboard = document.getElementById('station-tick-dashboard');
    let stationVersionDashboard = document.getElementById('station-version-dashboard');
    let stationStatusDashboard = null; // Will be created dynamically in header
    let cachedStationStatus = "Unknown"; // Track current station status string value
    let latestTickStartedTimestampMs = null;
    let latestTickTimingTooltip = '';
    const orchestratorStatusDisplay = document.getElementById('orchestrator-status-display');
    const orchestratorStatusDot = document.getElementById('orchestrator-status-dot');
    const orchestratorPauseReasonDisplay = document.getElementById('orchestrator-pause-reason-display');
    const orchestratorNextAgentDisplay = document.getElementById('orchestrator-next-agent-display');

    
    const startLoopButton = document.getElementById('start-loop-button'); 
    const pauseOrchestratorButton = document.getElementById('pause-orchestrator-button');
    const resumeOrchestratorButton = document.getElementById('resume-orchestrator-button');
    const stopOrchestratorButton = document.getElementById('stop-orchestrator-button');

    const agentSelectorDashboard = document.getElementById('agent-selector-dashboard');
    const createApiAgentModalButton = document.getElementById('create-api-agent-modal-button');
    const endApiAgentSessionButton = document.getElementById('end-api-agent-session-button');    
    
    const manualInteractionSidebarSection = document.getElementById('manual-interaction-sidebar-section');
    const manualInteractionTitle = document.getElementById('manual-interaction-title');
    const manualInteractionSelectedAgentDisplay = document.getElementById('manual-interaction-selected-agent-display');
    
    const openDirectMessageModalButton = document.getElementById('open-direct-message-modal-button');
    const resolveHumanInterventionButton = document.getElementById('resolve-human-intervention-button');

    const directMessageModal = document.getElementById('direct-message-modal');
    const closeDirectMessageModalButton = document.getElementById('close-direct-message-modal-button');
    const directMessageModalTitle = document.getElementById('direct-message-modal-title');
    const directMessageAgentSelector = document.getElementById('direct-message-agent-selector');
    const directMessageModalDescription = document.getElementById('direct-message-modal-description');
    const directMessageInput = document.getElementById('direct-message-input');
    const confirmSendDirectMessageButton = document.getElementById('confirm-send-direct-message-button');
    const directMessageResponseArea = document.getElementById('direct-message-response-area');
    const directMessageLlmResponseContent = document.getElementById('direct-message-llm-response-content');

    // Human Request Resolution Modal Elements
    const resolveRequestModal = document.getElementById('resolve-request-modal');
    const closeResolveRequestModalButton = document.getElementById('close-resolve-request-modal-button');
    const confirmResolveRequestButton = document.getElementById('confirm-resolve-request-button');
    const requestIdDisplay = document.getElementById('request-id-display');
    const requestTickDisplay = document.getElementById('request-tick-display');
    const requestAgentDisplay = document.getElementById('request-agent-display');
    const requestModelDisplay = document.getElementById('request-model-display');
    const requestTitleDisplay = document.getElementById('request-title-display');
    const requestContentDisplay = document.getElementById('request-content-display');
    const requestSelectorWrapper = document.getElementById('request-selector-wrapper');
    const requestSelector = document.getElementById('request-selector');
    const resolveResponseInput = document.getElementById('resolve-response-input');
    const resolveRequestStatus = document.getElementById('resolve-request-status');

    const openSendSystemMessageModalButton = document.getElementById('open-send-system-message-modal-button');
    const sendSystemMessageModal = document.getElementById('send-system-message-modal');
    const closeSendSystemMessageModalButton = document.getElementById('close-send-system-message-modal-button');
    const systemMessageAgentSelector = document.getElementById('system-message-agent-selector');
    const systemMessageContent = document.getElementById('system-message-content');
    const systemMessageArchitectCheckbox = document.getElementById('system-message-architect-checkbox');
    const confirmSendSystemMessageButton = document.getElementById('confirm-send-system-message-button');

    const openTemporalChatModalButton = document.getElementById('open-temporal-chat-modal-button');
    const temporalChatModal = document.getElementById('temporal-chat-modal');
    const closeTemporalChatModalButton = document.getElementById('close-temporal-chat-modal-button');
    const branchTemporalChatButton = document.getElementById('branch-temporal-chat-button');
    const copyTemporalChatButton = document.getElementById('copy-temporal-chat-button');
    const copyTemporalChatWithoutThinkingButton = document.getElementById('copy-temporal-chat-without-thinking-button');
    const temporalChatAgentSelector = document.getElementById('temporal-chat-agent-selector');
    const temporalChatBranchTickInput = document.getElementById('temporal-chat-branch-tick-input');
    const temporalChatBaseTick = document.getElementById('temporal-chat-base-tick');
    const temporalChatTranscript = document.getElementById('temporal-chat-transcript');
    const temporalChatInput = document.getElementById('temporal-chat-input');
    const confirmTemporalChatSendButton = document.getElementById('confirm-temporal-chat-send-button');

    const openArchivePapersModalButton = document.getElementById('open-archive-papers-modal-button');
    const archivePapersModal = document.getElementById('archive-papers-modal');
    const closeArchivePapersModalButton = document.getElementById('close-archive-papers-modal-button');
    const copyAllArchiveAbstractsButton = document.getElementById('copy-all-archive-abstracts-button');
    const archivePapersListView = document.getElementById('archive-papers-list-view');
    const archivePaperDetailView = document.getElementById('archive-paper-detail-view');
    const archivePapersTableBody = document.getElementById('archive-papers-table-body');
    const archivePaperDetailTitle = document.getElementById('archive-paper-detail-title');
    const archivePaperDetailMeta = document.getElementById('archive-paper-detail-meta');
    const archivePaperMarkdownContainer = document.getElementById('archive-paper-markdown-container');
    const archiveRowCountSelect = document.getElementById('archive-row-count-select');
    const archivePreviousPageButton = document.getElementById('archive-previous-page-button');
    const archiveNextPageButton = document.getElementById('archive-next-page-button');
    const archivePageSummary = document.getElementById('archive-page-summary');
    const backToArchivePapersListButton = document.getElementById('back-to-archive-papers-list-button');
    const copyArchivePaperMarkdownButton = document.getElementById('copy-archive-paper-markdown-button');
    const openWebArchiveSurveysModalButton = document.getElementById('open-web-archive-surveys-modal-button');
    const webArchiveSurveysModal = document.getElementById('web-archive-surveys-modal');
    const closeWebArchiveSurveysModalButton = document.getElementById('close-web-archive-surveys-modal-button');
    const webArchiveSurveysListView = document.getElementById('web-archive-surveys-list-view');
    const webArchiveSurveyRequestModal = document.getElementById('web-archive-survey-request-modal');
    const openWebArchiveSurveyRequestButton = document.getElementById('open-web-archive-survey-request-button');
    const closeWebArchiveSurveyRequestModalButton = document.getElementById('close-web-archive-survey-request-modal-button');
    const cancelWebArchiveSurveyRequestButton = document.getElementById('cancel-web-archive-survey-request-button');
    const webArchiveSurveyTemplateButtons = document.getElementById('web-archive-survey-template-buttons');
    const webArchiveSurveyPrompt = document.getElementById('web-archive-survey-prompt');
    const clearWebArchiveSurveyPromptButton = document.getElementById('clear-web-archive-survey-prompt-button');
    const submitWebArchiveSurveyButton = document.getElementById('submit-web-archive-survey-button');
    const refreshWebArchiveSurveysButton = document.getElementById('refresh-web-archive-surveys-button');
    const webArchiveSurveysTableBody = document.getElementById('web-archive-surveys-table-body');
    const webArchiveSurveyRowCountSelect = document.getElementById('web-archive-survey-row-count-select');
    const webArchiveSurveyPreviousPageButton = document.getElementById('web-archive-survey-previous-page-button');
    const webArchiveSurveyNextPageButton = document.getElementById('web-archive-survey-next-page-button');
    const webArchiveSurveyPageSummary = document.getElementById('web-archive-survey-page-summary');
    const webArchiveSurveyReportShell = document.getElementById('web-archive-survey-report-shell');
    const webArchiveSurveyReportTitle = document.getElementById('web-archive-survey-report-title');
    const webArchiveSurveyReportMeta = document.getElementById('web-archive-survey-report-meta');
    const webArchiveSurveyReport = document.getElementById('web-archive-survey-report');
    const removeWebArchiveSurveyButton = document.getElementById('remove-web-archive-survey-button');
    const copyWebArchiveSurveyMarkdownButton = document.getElementById('copy-web-archive-survey-markdown-button');
    const closeWebArchiveSurveyReportButton = document.getElementById('close-web-archive-survey-report-button');

    const openQuestionRoomModalButton = document.getElementById('open-question-room-modal-button');
    const questionRoomModal = document.getElementById('question-room-modal');
    const closeQuestionRoomModalButton = document.getElementById('close-question-room-modal-button');
    const questionRoomListView = document.getElementById('question-room-list-view');
    const questionRoomDetailView = document.getElementById('question-room-detail-view');
    const questionRoomTableBody = document.getElementById('question-room-table-body');
    const questionPageSizeSelect = document.getElementById('question-page-size-select');
    const questionPreviousPageButton = document.getElementById('question-previous-page-button');
    const questionNextPageButton = document.getElementById('question-next-page-button');
    const questionPageSummary = document.getElementById('question-page-summary');
    const questionRoomDetailTitle = document.getElementById('question-room-detail-title');
    const questionRoomDetailStatus = document.getElementById('question-room-detail-status');
    const questionRoomDetailMeta = document.getElementById('question-room-detail-meta');
    const questionRoomThread = document.getElementById('question-room-thread');
    const backToQuestionRoomListButton = document.getElementById('back-to-question-room-list-button');

    const openSpeakCommonRoomModalButton = document.getElementById('open-speak-common-room-modal-button');
    const speakCommonRoomModal = document.getElementById('speak-common-room-modal');
    const closeSpeakCommonRoomModalButton = document.getElementById('close-speak-common-room-modal-button');
    const commonRoomSpeakerName = document.getElementById('common-room-speaker-name');
    const commonRoomMessageContent = document.getElementById('common-room-message-content');
    const confirmSpeakCommonRoomButton = document.getElementById('confirm-speak-common-room-button');

    const updateStationConfigButton = document.getElementById('update-station-config-button');
    const updateStationConfigModal = document.getElementById('update-station-config-modal');
    const closeUpdateStationConfigModalButton = document.getElementById('close-update-station-config-modal-button');
    const updateStationConfigForm = document.getElementById('update-station-config-form');
    const updateStationId = document.getElementById('update-station-id');
    const updateStationStatus = document.getElementById('update-station-status');
    const updateStationName = document.getElementById('update-station-name');
    const updateStationDescription = document.getElementById('update-station-description');
    const confirmUpdateStationConfigButton = document.getElementById('confirm-update-station-config-button');
    const currentStationStatus = document.getElementById('current-station-status');
    const currentStationName = document.getElementById('current-station-name');
    const currentStationDescription = document.getElementById('current-station-description');
    const openResearchTaskSpecModalButton = document.getElementById('open-research-task-spec-modal-button');
    const researchTaskSpecModal = document.getElementById('research-task-spec-modal');
    const closeResearchTaskSpecModalButton = document.getElementById('close-research-task-spec-modal-button');
    const taskSpecPreviewTab = document.getElementById('task-spec-preview-tab');
    const taskSpecEditTab = document.getElementById('task-spec-edit-tab');
    const taskSpecPreviewPane = document.getElementById('task-spec-preview-pane');
    const taskSpecEditPane = document.getElementById('task-spec-edit-pane');
    const taskSpecRawEditor = document.getElementById('task-spec-raw-editor');
    const taskSpecPath = document.getElementById('task-spec-path');
    const taskSpecRevision = document.getElementById('task-spec-revision');
    const taskSpecDirtyIndicator = document.getElementById('task-spec-dirty-indicator');
    const reloadTaskSpecButton = document.getElementById('reload-task-spec-button');
    const saveTaskSpecButton = document.getElementById('save-task-spec-button');
    const openApiRuntimeConfigModalButton = document.getElementById('open-api-runtime-config-modal-button');
    const apiRuntimeConfigModal = document.getElementById('api-runtime-config-modal');
    const closeApiRuntimeConfigModalButton = document.getElementById('close-api-runtime-config-modal-button');
    const apiRuntimeConfigForm = document.getElementById('api-runtime-config-form');
    const apiRuntimeTargetSelect = document.getElementById('api-runtime-target-select');
    const apiRuntimeConfigBody = document.getElementById('api-runtime-config-body');
    const saveApiRuntimeConfigButton = document.getElementById('save-api-runtime-config-button');
    const moreToolsButton = document.getElementById('more-tools-button');
    const moreToolsPopover = document.getElementById('more-tools-popover');

    const globalNotificationBubbleLog = document.getElementById('global-notification-bubble-log');
    const agentSpecificBubbleLog = document.getElementById('agent-specific-bubble-log');
    const dashboardMetricsGrid = document.querySelector('.dashboard-metrics-grid');
    const logViewTitle = document.getElementById('log-view-title');
    const logModeFullButton = document.getElementById('log-mode-full-button');
    const logModeCollapsedButton = document.getElementById('log-mode-collapsed-button');
    const toggleAgentLogButton = document.getElementById('toggle-agent-log-button');
    const toggleStationLogButton = document.getElementById('toggle-station-log-button');
    const historyWindowSelector = document.getElementById('history-window-selector');

    const clearLogButton = document.getElementById('clear-log-button');
    const loadFullHistoryButton = document.getElementById('load-full-history-button');
    const copySystemPromptButton = document.getElementById('copy-system-prompt-button');
    const dashboardStatusMessages = document.getElementById('dashboard-status-messages');

    const createApiAgentModal = document.getElementById('create-api-agent-modal');
    const closeApiModalButton = document.getElementById('close-api-modal-button');
    const createApiAgentForm = document.getElementById('create-api-agent-form');
    const apiAgentTypeSelect = document.getElementById('api-agent-type');
    const apiRecursiveFieldsDiv = document.getElementById('api-recursive-fields');
    const apiModelPreset = document.getElementById('api-model-preset');
    const MODEL_PRESETS = (window.STATION_CONFIG && Array.isArray(window.STATION_CONFIG.modelPresets))
        ? window.STATION_CONFIG.modelPresets
        : [];

    // Chat: persisted per-agent fork under station_data/temporal_chat.
    const temporalChatByAgent = new Map(); // agent_name -> [{role:'user'|'assistant', content:string}]
    const temporalChatBaseTickByAgent = new Map(); // agent_name -> station tick used for the frozen fork
    const temporalChatExistsByAgent = new Map(); // agent_name -> boolean
    const temporalChatQueueByAgent = new Map(); // agent_name -> [string]
    const temporalChatProcessingByAgent = new Map(); // agent_name -> boolean
    const temporalChatBranchingByAgent = new Map(); // agent_name -> boolean
    let archivePapersCache = [];
    let archiveAllAbstractsMarkdown = '';
    let archiveSelectedPaper = null;
    let archiveRowCount = 100;
    let archivePage = 1;
    let archiveSortState = { key: 'accepted_tick', direction: 'desc' };
    let webArchiveSurveyTemplates = [];
    let webArchiveSurveys = [];
    let webArchiveSurveysLoaded = false;
    let webArchiveSelectedSurvey = null;
    let webArchiveSurveyRowCount = 100;
    let webArchiveSurveyPage = 1;
    let questionListState = {
        page: 1,
        pageSize: 100,
        total: 0,
        totalPages: 1,
        sortBy: 'authored_tick',
        sortDirection: 'desc'
    };
    let questionListRequestSequence = 0;
    let taskSpecState = {
        rawMarkdown: '',
        savedRawMarkdown: '',
        revision: '',
        relativePath: '',
        dirty: false,
        loaded: false,
        mode: 'preview'
    };

    let sseSource = null;
    let pollingFallbackInterval = null;
    let sseReconnectTimer = null;
    let pollingRequestInFlight = false;
    let liveEventCursor = null;
    let liveStreamEpoch = null;
    let liveTransportMode = 'idle';
    let liveUiRefreshTimer = null;
    let liveAgentRefreshNeeded = false;
    let currentSelectedAgentForDialogueView = "all"; 
    let isLoadingHistoryForAgent = null;
    let isRefreshingAgentSelector = false;
    let orchestratorState = { is_prepared: false, is_running: false, is_paused: false, agents_awaiting_human: [], turn_order: [] };
    let resolveRequestCache = new Map();
    let fullAgentListCache = [];
    let fullDialogueHistoryCache = {};
    let isDirectMessageInProgress = false; // Track if a direct message is currently being sent 
    const LOG_DISPLAY_MODE_STORAGE_KEY = 'station.dashboard.logDisplayMode';
    const HISTORY_WINDOW_STORAGE_KEY = 'station.dashboard.historyWindow';
    const LOG_MODES = ['full', 'collapsed'];
    let logDisplayMode = localStorage.getItem(LOG_DISPLAY_MODE_STORAGE_KEY) || 'full';
    if (!LOG_MODES.includes(logDisplayMode)) {
        logDisplayMode = 'full';
    }
    let historyWindowMode = localStorage.getItem(HISTORY_WINDOW_STORAGE_KEY) || 'recent';
    if (!['recent', 'earliest'].includes(historyWindowMode)) {
        historyWindowMode = 'recent';
    }
    if (isMultistartPreview) historyWindowMode = 'recent';
    let logMessageSequence = 0;
    const expandedMessageIds = new Set();
    const collapsedSourceKeys = new Set();
    const renderedAgentDialogueFingerprints = new Set();
    const pendingLiveStationObservations = new Map();
    const multistartPreviewDialogueSignatures = new Map();
    const multistartPreviewDialogueMtimes = new Map();
    let multistartPreviewDialogueRefreshInFlight = false;
    const backgroundHistoryLoads = new Map();
    const HISTORY_WINDOW_TICKS = 50;
    const HISTORY_REQUEST_TIMEOUT_MS = 60000;
    const HISTORY_RENDER_CHUNK_SIZE = 12;
    const HISTORY_RENDER_CHUNK_TEXT_BUDGET = 24000;
    let activeFloatingTooltipTarget = null;
    let dashboardFloatingTooltip = null;
    let dashboardTooltipHideTimer = null;
    let historyLoadSequence = 0;
    let activeHistoryAbortController = null;
    let bufferedLiveEventsDuringHistory = [];
    let dashboardStatusTimer = null;

    // --- Helper Functions ---
    function hideDashboardStatus() {
        if (!dashboardStatusMessages) return;
        if (dashboardStatusTimer) {
            clearTimeout(dashboardStatusTimer);
            dashboardStatusTimer = null;
        }
        dashboardStatusMessages.classList.remove('opacity-100');
        dashboardStatusMessages.classList.add('opacity-0', 'is-hidden');
        dashboardStatusMessages.setAttribute('aria-hidden', 'true');
        dashboardStatusMessages.tabIndex = -1;
    }

    function showDashboardStatus(message, type = 'info', duration = 4000) {
        if (!dashboardStatusMessages) { console.error("dashboardStatusMessages element not found"); return; }
        if (dashboardStatusTimer) {
            clearTimeout(dashboardStatusTimer);
            dashboardStatusTimer = null;
        }
        dashboardStatusMessages.textContent = message;
        dashboardStatusMessages.className = 'dashboard-status-overlay p-3 glass-surface rounded-md text-sm border transition-all duration-300 opacity-100';
        dashboardStatusMessages.setAttribute('aria-hidden', 'false');
        dashboardStatusMessages.tabIndex = 0;
        const typeClasses = {
            error: ['bg-red-700', 'text-red-100', 'border-red-600'],
            success: ['bg-emerald-700', 'text-emerald-100', 'border-emerald-600'],
            info: ['bg-sky-700', 'text-sky-100', 'border-sky-600']
        };
        dashboardStatusMessages.classList.add(...(typeClasses[type] || typeClasses.info));
        if (duration > 0) {
            dashboardStatusTimer = setTimeout(() => {
                hideDashboardStatus();
            }, duration);
        }
    }

    function getDashboardFloatingTooltip() {
        if (!dashboardFloatingTooltip) {
            dashboardFloatingTooltip = document.createElement('div');
            dashboardFloatingTooltip.className = 'dashboard-floating-tooltip';
            dashboardFloatingTooltip.setAttribute('role', 'tooltip');
            dashboardFloatingTooltip.addEventListener('pointerover', () => {
                if (dashboardTooltipHideTimer) {
                    clearTimeout(dashboardTooltipHideTimer);
                    dashboardTooltipHideTimer = null;
                }
            });
            dashboardFloatingTooltip.addEventListener('pointerout', (event) => {
                if (event.relatedTarget instanceof Node && dashboardFloatingTooltip.contains(event.relatedTarget)) return;
                hideDashboardFloatingTooltip();
            });
            document.body.appendChild(dashboardFloatingTooltip);
        }
        return dashboardFloatingTooltip;
    }

    function positionDashboardFloatingTooltip(target) {
        if (!target || !dashboardFloatingTooltip || !dashboardFloatingTooltip.classList.contains('is-visible')) return;

        const targetRect = target.getBoundingClientRect();
        const tooltipWidth = dashboardFloatingTooltip.offsetWidth || 320;
        const tooltipHeight = dashboardFloatingTooltip.offsetHeight || 180;
        const viewportPadding = 16;
        const gap = 2;
        const preferSidePlacement = window.innerWidth > 820;

        let left = preferSidePlacement ? targetRect.right - gap : targetRect.left;
        let top = preferSidePlacement
            ? targetRect.top + (targetRect.height - tooltipHeight) / 2
            : targetRect.bottom + 8;

        if (preferSidePlacement && left + tooltipWidth > window.innerWidth - viewportPadding) {
            left = targetRect.left - tooltipWidth - gap;
        }
        if (!preferSidePlacement && left + tooltipWidth > window.innerWidth - viewportPadding) {
            left = window.innerWidth - tooltipWidth - viewportPadding;
        }
        if (top + tooltipHeight > window.innerHeight - viewportPadding) {
            top = preferSidePlacement
                ? window.innerHeight - tooltipHeight - viewportPadding
                : targetRect.top - tooltipHeight - 8;
        }

        dashboardFloatingTooltip.style.left = `${Math.max(viewportPadding, left)}px`;
        dashboardFloatingTooltip.style.top = `${Math.max(viewportPadding, top)}px`;
    }

    function showDashboardFloatingTooltip(target) {
        if (!target) return;
        const tooltipText = String(target.getAttribute('data-tooltip') || '').trim();
        if (!tooltipText) return;
        if (dashboardTooltipHideTimer) {
            clearTimeout(dashboardTooltipHideTimer);
            dashboardTooltipHideTimer = null;
        }

        activeFloatingTooltipTarget = target;
        const tooltip = getDashboardFloatingTooltip();
        tooltip.textContent = tooltipText;
        tooltip.classList.add('is-visible');
        positionDashboardFloatingTooltip(target);
    }

    function hideDashboardFloatingTooltip(target = null) {
        if (target && target !== activeFloatingTooltipTarget) return;
        if (dashboardTooltipHideTimer) clearTimeout(dashboardTooltipHideTimer);
        dashboardTooltipHideTimer = setTimeout(() => {
            if (dashboardFloatingTooltip && dashboardFloatingTooltip.matches(':hover')) return;
            if (dashboardFloatingTooltip) {
                dashboardFloatingTooltip.classList.remove('is-visible');
            }
            activeFloatingTooltipTarget = null;
            dashboardTooltipHideTimer = null;
        }, 180);
    }

    function findTooltipTarget(eventTarget) {
        if (!(eventTarget instanceof Element)) return null;
        const target = eventTarget.closest('.dashboard-metric-card[data-tooltip]');
        if (!target || !String(target.getAttribute('data-tooltip') || '').trim()) return null;
        return target;
    }

    if (dashboardMetricsGrid) {
        dashboardMetricsGrid.addEventListener('pointerover', (event) => {
            const target = findTooltipTarget(event.target);
            if (!target) return;
            showDashboardFloatingTooltip(target);
        });

        dashboardMetricsGrid.addEventListener('pointerout', (event) => {
            const target = findTooltipTarget(event.target);
            if (!target) return;
            if (event.relatedTarget instanceof Node && target.contains(event.relatedTarget)) return;
            hideDashboardFloatingTooltip(target);
        });

        dashboardMetricsGrid.addEventListener('focusin', (event) => {
            const target = findTooltipTarget(event.target);
            if (target) showDashboardFloatingTooltip(target);
        });

        dashboardMetricsGrid.addEventListener('focusout', (event) => {
            const target = findTooltipTarget(event.target);
            if (target) hideDashboardFloatingTooltip(target);
        });
    }

    window.addEventListener('resize', () => {
        positionDashboardFloatingTooltip(activeFloatingTooltipTarget);
    });

    function isMoreToolsOpen() {
        return Boolean(moreToolsPopover && moreToolsPopover.classList.contains('is-open'));
    }

    function positionMoreToolsPopover() {
        if (!moreToolsButton || !moreToolsPopover || !isMoreToolsOpen()) return;

        const buttonRect = moreToolsButton.getBoundingClientRect();
        const popoverWidth = moreToolsPopover.offsetWidth || 272;
        const popoverHeight = moreToolsPopover.offsetHeight || 280;
        const viewportPadding = 16;
        const gap = 12;
        const preferSidePlacement = window.innerWidth > 820;

        let left = preferSidePlacement ? buttonRect.right + gap : buttonRect.left;
        let top = preferSidePlacement ? buttonRect.top : buttonRect.bottom + 8;

        if (preferSidePlacement && left + popoverWidth > window.innerWidth - viewportPadding) {
            left = buttonRect.left - popoverWidth - gap;
        }
        if (!preferSidePlacement && left + popoverWidth > window.innerWidth - viewportPadding) {
            left = window.innerWidth - popoverWidth - viewportPadding;
        }
        if (top + popoverHeight > window.innerHeight - viewportPadding) {
            top = preferSidePlacement
                ? window.innerHeight - popoverHeight - viewportPadding
                : buttonRect.top - popoverHeight - 8;
        }

        moreToolsPopover.style.left = `${Math.max(viewportPadding, left)}px`;
        moreToolsPopover.style.top = `${Math.max(viewportPadding, top)}px`;
    }

    function setMoreToolsOpen(open) {
        if (!moreToolsButton || !moreToolsPopover) return;
        moreToolsPopover.classList.toggle('is-open', open);
        moreToolsButton.setAttribute('aria-expanded', open ? 'true' : 'false');
        if (open) {
            positionMoreToolsPopover();
        }
    }

    if (moreToolsButton && moreToolsPopover) {
        document.body.appendChild(moreToolsPopover);

        moreToolsButton.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            setMoreToolsOpen(!isMoreToolsOpen());
        });

        moreToolsPopover.addEventListener('click', (event) => {
            if (event.target instanceof Element && event.target.closest('button')) {
                setMoreToolsOpen(false);
            }
        });

        document.addEventListener('click', (event) => {
            if (!isMoreToolsOpen()) return;
            if (moreToolsButton.contains(event.target) || moreToolsPopover.contains(event.target)) return;
            setMoreToolsOpen(false);
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && isMoreToolsOpen()) {
                setMoreToolsOpen(false);
                moreToolsButton.focus();
            }
        });

        window.addEventListener('resize', positionMoreToolsPopover);
    }

    let dashboardMathRenderingEnabled = false;
    const SYSTEM_MESSAGES_TRUNCATION_NOTICE = '[System Messages exceeded the length limit; please do not issue too many actions within one tick.]';

    function configureDashboardMathRendering() {
        const markedKatexFactory = window.markedKatex && (window.markedKatex.default || window.markedKatex);
        if (!window.marked || typeof window.marked.use !== 'function' || typeof markedKatexFactory !== 'function' || !window.katex) {
            console.warn('KaTeX Markdown extension unavailable. Formula rendering will use the plain Markdown fallback.');
            return;
        }
        try {
            window.marked.use(markedKatexFactory({
                throwOnError: false,
                nonStandard: true,
                strict: 'ignore',
                trust: false,
                output: 'html'
            }));
            dashboardMathRenderingEnabled = true;
        } catch (error) {
            console.error('Failed to initialize KaTeX Markdown rendering:', error);
        }
    }

    function normalizeLatexDelimitersForMarked(markdownText) {
        let source = String(markdownText || '');
        const protectedCode = [];
        const protectCode = (segment) => {
            const placeholder = `@@STATION_PROTECTED_CODE_${protectedCode.length}@@`;
            protectedCode.push(segment);
            return placeholder;
        };

        // Marked consumes the backslashes in \(...\) and \[...\] before the
        // KaTeX extension sees them. Protect code first, then translate only
        // prose math delimiters to the extension's native dollar delimiters.
        source = source.replace(/(^|\n)([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?\n\2\3[ \t]*(?=\n|$)/g, protectCode);
        source = source.replace(/(`+)([^`\n]*?)\1/g, protectCode);
        source = source.replace(
            /\\\[([\s\S]*?)\\\]/g,
            (_match, formula) => `$$${String(formula).replace(/\r?\n[ \t]*/g, ' ').trim()}$$`
        );
        source = source.replace(/\\\(([\s\S]*?)\\\)/g, (_match, formula) => `$${formula}$`);
        source = source.replace(/@@STATION_PROTECTED_CODE_(\d+)@@/g, (_match, index) => protectedCode[Number(index)] || '');
        return source;
    }

    function repairStationCodeFences(markdownText) {
        const source = String(markdownText || '');
        const readBlockPattern = /(^### Message \d+[^\n]*\n+\*\*(?:Storage|Code) Read:\*\*[^\n]*\n+)(`{3,})([^\n]*)\n([\s\S]*?)\n`{3,}[ \t]*(?=\n+(?:### Message \d+[^\n]*\n|---\n+## Actions Detected))/gm;
        return source.replace(readBlockPattern, (_match, prefix, _openingFence, info, content) => {
            const nestedFences = content.match(/`{3,}/g) || [];
            const fenceWidth = Math.max(3, ...nestedFences.map(fence => fence.length)) + 1;
            const safeFence = '`'.repeat(fenceWidth);
            const extension = prefix.match(/`[^`]+\.([A-Za-z0-9]+)`/)?.[1]?.toLowerCase();
            const languages = {
                json: 'json',
                yaml: 'yaml',
                yml: 'yaml',
                py: 'python',
                md: 'markdown',
                c: 'c',
                cc: 'cpp',
                cpp: 'cpp',
                h: 'c',
                hpp: 'cpp',
                js: 'javascript',
                ts: 'typescript',
                sh: 'bash',
            };
            const language = info.trim() || languages[extension] || '';
            return `${prefix}${safeFence}${language}\n${content}\n${safeFence}`;
        });
    }

    function normalizeStorageReadFences(markdownText) {
        const source = String(markdownText || '');
        const storageHeaderPattern = /\*\*Storage Read:\*\*[^\n]*\n[ \t]*\n[ \t]*(`{3,})[^\n]*\n/g;
        const boundaryPatterns = [
            /\n### Message \d+[ \t]*(?:\n|$)/g,
            /\n---\n\n## Actions Detected(?: \(Last Turn\))?/g,
            /\n---\n\n## Room Output History/g,
            /\n---\n\n## Current State/g,
            /\n\[Use `\/execute_action\{read /g,
        ];
        let result = '';
        let cursor = 0;
        let headerMatch;

        while ((headerMatch = storageHeaderPattern.exec(source)) !== null) {
            const contentStart = storageHeaderPattern.lastIndex;
            let sectionEnd = source.length;
            boundaryPatterns.forEach(pattern => {
                pattern.lastIndex = contentStart;
                const boundaryMatch = pattern.exec(source);
                if (boundaryMatch && boundaryMatch.index < sectionEnd) sectionEnd = boundaryMatch.index;
            });

            const noticeIndex = source.indexOf(SYSTEM_MESSAGES_TRUNCATION_NOTICE, contentStart);
            let truncationBoundary = -1;
            if (noticeIndex >= 0) {
                truncationBoundary = noticeIndex;
                while (truncationBoundary > contentStart && /[\r\n]/.test(source[truncationBoundary - 1])) {
                    truncationBoundary -= 1;
                }
                if (truncationBoundary < sectionEnd) sectionEnd = truncationBoundary;
            }

            const section = source.slice(contentStart, sectionEnd);
            const closingFencePattern = /^(`{3,})[ \t]*$/gm;
            const closingFences = Array.from(section.matchAll(closingFencePattern));
            const truncatedAtNotice = truncationBoundary >= 0 && sectionEnd === truncationBoundary;
            if (!closingFences.length && !truncatedAtNotice) continue;

            const closingFence = closingFences.length ? closingFences[closingFences.length - 1] : null;
            const closingStart = closingFence ? contentStart + closingFence.index : sectionEnd;
            const closingEnd = closingFence ? closingStart + closingFence[0].length : sectionEnd;
            const content = source.slice(contentStart, closingStart);
            const backtickRuns = content.match(/`+/g) || [];
            const longestRun = backtickRuns.reduce((longest, run) => Math.max(longest, run.length), 0);
            const safeFence = '`'.repeat(Math.max(3, longestRun + 1));
            const openingStart = headerMatch.index + headerMatch[0].lastIndexOf(headerMatch[1]);
            const closingPrefix = content.endsWith('\n') ? '' : '\n';

            result += source.slice(cursor, openingStart);
            result += `${safeFence}\n${content}${closingPrefix}${safeFence}`;
            cursor = closingEnd;
            storageHeaderPattern.lastIndex = closingEnd;
        }

        return cursor ? result + source.slice(cursor) : source;
    }

    function enhanceStorageReadBlocks(markdownSource, embeddedHtml) {
        const source = String(markdownSource || '');
        const storageReadPattern = /\*\*Storage Read:\*\*[ \t]*([^\n]*)\r?\n[ \t]*\r?\n[ \t]*(`{3,})[^\n]*\r?\n([\s\S]*?)\r?\n[ \t]*\2[ \t]*(?=\n|$)/g;
        return source.replace(storageReadPattern, (_match, rawPath, _fence, content) => {
            const path = String(rawPath || '').trim().replace(/^`|`$/g, '');
            const panel = `
                <details class="station-storage-read">
                    <summary class="station-storage-read-summary">
                        <span class="station-storage-read-label">Storage Read</span>
                        <code>${escapeHtml(path || 'file')}</code>
                        <span class="station-storage-read-toggle" aria-hidden="true"></span>
                    </summary>
                    <pre class="station-storage-read-content"><code>${escapeHtml(content)}</code></pre>
                </details>
            `.trim();
            return protectDashboardEmbeddedHtml(panel, embeddedHtml);
        });
    }

    function normalizeStationActionCommands(markdownText) {
        let source = String(markdownText || '');
        const protectedCode = [];
        const protectCode = (segment) => {
            const placeholder = `@@STATION_ACTION_CODE_${protectedCode.length}@@`;
            protectedCode.push(segment);
            return placeholder;
        };

        source = source.replace(/(^|\n)([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?\n\2\3[ \t]*(?=\n|$)/g, protectCode);
        source = source.replace(/(`+)([^`\n]*?)\1/g, protectCode);
        source = source.replace(
            /\/execute_action\{[^}\n]+\}/g,
            command => `\`${command}\``
        );
        source = source.replace(/@@STATION_ACTION_CODE_(\d+)@@/g, (_match, index) => protectedCode[Number(index)] || '');
        return source;
    }

    function styleStationActionCommandHtml(renderedHtml) {
        return String(renderedHtml || '').replace(
            /<code>(\/execute_action\{[\s\S]*?\})<\/code>/g,
            (_match, commandHtml) => {
                const verbMatch = commandHtml.match(/^\/execute_action\{\s*([a-zA-Z_]+)/);
                const verb = verbMatch ? verbMatch[1].toLowerCase() : 'action';
                const actionBodyMatch = commandHtml.match(/^\/execute_action\{([\s\S]*)\}$/);
                const actionBodyHtml = actionBodyMatch ? actionBodyMatch[1].trim() : commandHtml;
                const navigationActions = new Set(['goto']);
                const readActions = new Set(['read', 'read_task', 'read_code', 'review', 'preview', 'rank', 'filter', 'unfilter', 'storage', 'page', 'page_size', 'help']);
                const writeActions = new Set(['submit', 'create', 'reply', 'update', 'forward', 'meta', 'speak', 'survey', 'reflect', 'request_human']);
                let category = 'control';
                if (navigationActions.has(verb)) category = 'navigation';
                else if (readActions.has(verb)) category = 'read';
                else if (writeActions.has(verb)) category = 'write';
                return `<code class="station-action-command station-action-${category}" data-station-action-verb="${verb}">${actionBodyHtml}</code>`;
            }
        );
    }

    configureDashboardMathRendering();

    const STATION_YAML_CARD_MAX_CHARS = 120000;
    const STATION_YAML_ACTION_LABELS = {
        submit: 'Research Submission',
        reply: 'Capsule Reply',
        create: 'Capsule Creation',
        meta: 'Agent Meta Prompt',
        speak: 'Common Room Post',
        reflect: 'Reflection',
        survey: 'Archive Survey Request',
        request_human: 'Human Assistance Request',
    };
    const STATION_YAML_MARKDOWN_FIELDS = [
        ['abstract', 'Abstract'],
        ['content', 'Content'],
        ['message', 'Message'],
        ['description', 'Description'],
        ['prompt', 'Prompt'],
    ];

    function getStationYamlActionHint(markdownSource, fenceOffset) {
        const context = markdownSource.slice(Math.max(0, fenceOffset - 320), fenceOffset);
        const match = context.match(/(?:^|\n)\s*`?\/execute_action\{([^}\n]+)\}`?\s*$/i);
        if (!match) return '';
        return String(match[1] || '').trim().split(/\s+/)[0].toLowerCase();
    }

    function parseStationYamlCard(rawYaml, actionHint) {
        if (!window.jsyaml || typeof window.jsyaml.load !== 'function') return null;
        if (!rawYaml || rawYaml.length > STATION_YAML_CARD_MAX_CHARS) return null;
        try {
            const parsed = window.jsyaml.load(rawYaml, {
                schema: window.jsyaml.FAILSAFE_SCHEMA,
                json: false,
            });
            if (!parsed || Object.prototype.toString.call(parsed) !== '[object Object]') return null;

            const keys = Object.keys(parsed);
            const hasRecognizedText = ['title', 'content', 'abstract', 'instruction', 'message', 'description', 'prompt']
                .some(key => typeof parsed[key] === 'string');
            const hasKnownAction = Boolean(STATION_YAML_ACTION_LABELS[actionHint]);
            const resemblesResearchSubmission = ['title', 'abstract', 'instruction'].every(key => typeof parsed[key] === 'string');
            const resemblesCapsule = typeof parsed.title === 'string' && typeof parsed.content === 'string';
            if (!keys.length || !hasRecognizedText || (!hasKnownAction && !resemblesResearchSubmission && !resemblesCapsule)) {
                return null;
            }
            return parsed;
        } catch (_error) {
            return null;
        }
    }

    function renderDashboardMarkdownInline(value) {
        const textValue = String(value || '');
        if (!window.marked || typeof window.marked.parseInline !== 'function') return escapeHtml(textValue);
        try {
            const source = dashboardMathRenderingEnabled
                ? normalizeLatexDelimitersForMarked(textValue)
                : textValue;
            return window.marked.parseInline(source, { gfm: true, breaks: false });
        } catch (_error) {
            return escapeHtml(textValue);
        }
    }

    function renderStationYamlCard(parsed, rawYaml, actionHint) {
        const actionLabel = STATION_YAML_ACTION_LABELS[actionHint]
            || (typeof parsed.instruction === 'string' ? 'Research Submission' : 'Structured Station Action');
        const title = typeof parsed.title === 'string' && parsed.title.trim()
            ? parsed.title.trim()
            : actionLabel;
        const eyebrow = title === actionLabel ? 'Station Action' : actionLabel;
        const tags = Array.isArray(parsed.tags)
            ? parsed.tags.map(tag => String(tag).trim()).filter(Boolean)
            : String(parsed.tags || '').split(',').map(tag => tag.trim()).filter(Boolean);

        const tagHtml = tags.length
            ? `<div class="station-yaml-tags">${tags.map(tag => `<span>${escapeHtml(tag)}</span>`).join('')}</div>`
            : '';
        const fieldsHtml = STATION_YAML_MARKDOWN_FIELDS.map(([key, label]) => {
            if (typeof parsed[key] !== 'string' || !parsed[key].trim()) return '';
            return `
                <section class="station-yaml-field">
                    <div class="station-yaml-field-label">${label}</div>
                    <div class="station-yaml-field-markdown markdown-content-host">${renderMarkdownForDashboard(parsed[key], { enhanceYamlCards: false })}</div>
                </section>`;
        }).join('');
        const instructionHtml = typeof parsed.instruction === 'string' && parsed.instruction.trim()
            ? `
                <details class="station-yaml-instructions" open>
                    <summary>Full instructions</summary>
                    <div class="station-yaml-field-markdown markdown-content-host">${renderMarkdownForDashboard(parsed.instruction, { enhanceYamlCards: false })}</div>
                </details>`
            : '';

        return `
            <article class="station-yaml-card" data-station-action="${escapeHtml(actionHint || 'structured')}">
                <header class="station-yaml-card-header">
                    <div class="station-yaml-card-heading">
                        <div class="station-yaml-card-eyebrow">${escapeHtml(eyebrow)}</div>
                        <div class="station-yaml-card-title">${renderDashboardMarkdownInline(title)}</div>
                    </div>
                    <button type="button" class="station-yaml-copy-button" data-copy-kind="YAML">Copy YAML</button>
                </header>
                ${tagHtml}
                <div class="station-yaml-card-body">${fieldsHtml}${instructionHtml}</div>
                <textarea class="station-yaml-copy-source" hidden readonly aria-hidden="true">${escapeHtml(rawYaml)}</textarea>
            </article>`.trim();
    }

    function renderAgentMetaPromptCard(rawPrompt) {
        return `
            <article class="station-yaml-card station-meta-prompt-card" data-station-action="agent-meta-prompt">
                <header class="station-yaml-card-header">
                    <div class="station-yaml-card-heading">
                        <div class="station-yaml-card-eyebrow">Agent State</div>
                        <div class="station-yaml-card-title">Agent Meta Prompt</div>
                    </div>
                    <button type="button" class="station-yaml-copy-button" data-copy-kind="Meta prompt">Copy Prompt</button>
                </header>
                <div class="station-yaml-card-body">
                    <div class="station-yaml-field-markdown markdown-content-host">${renderMarkdownForDashboard(rawPrompt, { enhanceYamlCards: false })}</div>
                </div>
                <textarea class="station-yaml-copy-source" hidden readonly aria-hidden="true">${escapeHtml(rawPrompt)}</textarea>
            </article>`.trim();
    }

    function protectDashboardEmbeddedHtml(html, embeddedHtml) {
        const index = embeddedHtml.length;
        embeddedHtml.push(html);
        return `\n\nSTATIONEMBEDDEDHTML${index}TOKEN\n\n`;
    }

    function restoreDashboardEmbeddedHtml(renderedHtml, embeddedHtml) {
        let restored = String(renderedHtml || '');
        embeddedHtml.forEach((html, index) => {
            const token = `STATIONEMBEDDEDHTML${index}TOKEN`;
            restored = restored.replace(new RegExp(`<p>\\s*${token}\\s*</p>`, 'g'), html);
            restored = restored.split(token).join(html);
        });
        return restored;
    }

    function enhanceAgentMetaPromptFields(markdownSource, embeddedHtml) {
        const source = String(markdownSource || '');
        const metaPromptPattern = /(^|\n)[ \t]*[-*+][ \t]+Agent Meta Prompt:[ \t]*\r?\n[ \t]*```(?:markdown|md)?[ \t]*\r?\n([\s\S]*?)\r?\n[ \t]*```(?=\n|$)/gi;
        return source.replace(metaPromptPattern, (fullMatch, leadingNewline, rawPrompt) => {
            if (!rawPrompt.trim() || rawPrompt.length > STATION_YAML_CARD_MAX_CHARS) return fullMatch;
            return `${leadingNewline}${protectDashboardEmbeddedHtml(renderAgentMetaPromptCard(rawPrompt), embeddedHtml)}\n`;
        });
    }

    function enhanceStationYamlBlocks(markdownSource, embeddedHtml) {
        const source = String(markdownSource || '');
        const yamlFencePattern = /(^|\n)[ \t]*```ya?ml[ \t]*\r?\n([\s\S]*?)\r?\n[ \t]*```(?=\n|$)/gi;
        return source.replace(yamlFencePattern, (fullMatch, leadingNewline, rawYaml, fenceOffset) => {
            const actionHint = getStationYamlActionHint(source, fenceOffset);
            const parsed = parseStationYamlCard(rawYaml, actionHint);
            if (!parsed) return fullMatch;
            return `${leadingNewline}${protectDashboardEmbeddedHtml(renderStationYamlCard(parsed, rawYaml, actionHint), embeddedHtml)}\n`;
        });
    }

    function enhanceStationBareYamlActions(markdownSource, embeddedHtml) {
        let source = String(markdownSource || '');
        const protectedCode = [];
        source = source.replace(
            /(^|\n)([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?\n\2\3[ \t]*(?=\n|$)/g,
            segment => {
                const placeholder = `STATIONBAREYAMLCODE${protectedCode.length}TOKEN`;
                protectedCode.push(segment);
                return placeholder;
            }
        );

        const actionLinePattern = /(^|\n)[ \t]*`?\/execute_action\{([^}\n]+)\}`?[ \t]*\r?\n/g;
        let result = '';
        let cursor = 0;
        let actionMatch;

        while ((actionMatch = actionLinePattern.exec(source)) !== null) {
            if (actionMatch.index < cursor) continue;
            const actionHint = String(actionMatch[2] || '').trim().split(/\s+/)[0].toLowerCase();
            if (!STATION_YAML_ACTION_LABELS[actionHint]) continue;

            let payloadStart = actionLinePattern.lastIndex;
            const blankPrefix = source.slice(payloadStart).match(/^(?:[ \t]*\r?\n)*/);
            payloadStart += blankPrefix ? blankPrefix[0].length : 0;
            const remaining = source.slice(payloadStart);
            if (!/^[A-Za-z_][A-Za-z0-9_-]*[ \t]*:/.test(remaining)) continue;

            const lines = remaining.match(/.*(?:\r?\n|$)/g) || [];
            let consumed = 0;
            let payloadEnd = -1;
            for (let index = 0; index < lines.length; index += 1) {
                consumed += lines[index].length;
                const nextLine = index + 1 < lines.length ? lines[index + 1] : '';
                const nextText = nextLine.replace(/\r?\n$/, '');
                const nextContinuesYaml = !nextLine
                    || !nextText.trim()
                    || /^[ \t]/.test(nextText)
                    || /^[A-Za-z_][A-Za-z0-9_-]*[ \t]*:/.test(nextText)
                    || /^-[ \t]+/.test(nextText)
                    || /^#/.test(nextText);
                if (nextContinuesYaml && nextLine) continue;

                const rawYaml = remaining.slice(0, consumed).trimEnd();
                const parsed = parseStationYamlCard(rawYaml, actionHint);
                if (parsed) payloadEnd = payloadStart + rawYaml.length;
                break;
            }

            if (payloadEnd < payloadStart) continue;
            const rawYaml = source.slice(payloadStart, payloadEnd);
            const parsed = parseStationYamlCard(rawYaml, actionHint);
            if (!parsed) continue;
            result += source.slice(cursor, payloadStart);
            result += `${protectDashboardEmbeddedHtml(renderStationYamlCard(parsed, rawYaml, actionHint), embeddedHtml)}\n`;
            cursor = payloadEnd;
            actionLinePattern.lastIndex = payloadEnd;
        }

        if (cursor) source = result + source.slice(cursor);
        source = source.replace(/STATIONBAREYAMLCODE(\d+)TOKEN/g, (_match, index) => protectedCode[Number(index)] || '');
        return source;
    }

    function renderMarkdownForDashboard(markdownText, options = {}) {
        if (window.marked && typeof window.marked.parse === 'function') {
            try {
                const embeddedHtml = [];
                const repairedReadSource = repairStationCodeFences(markdownText);
                const normalizedStorageSource = normalizeStorageReadFences(repairedReadSource);
                const normalizedActionSource = normalizeStationActionCommands(normalizedStorageSource);
                let markdownSource = dashboardMathRenderingEnabled
                    ? normalizeLatexDelimitersForMarked(normalizedActionSource)
                    : normalizedActionSource;
                if (options.enhanceYamlCards !== false) {
                    markdownSource = enhanceStorageReadBlocks(markdownSource, embeddedHtml);
                    markdownSource = enhanceAgentMetaPromptFields(markdownSource, embeddedHtml);
                    markdownSource = enhanceStationYamlBlocks(markdownSource, embeddedHtml);
                    markdownSource = enhanceStationBareYamlActions(markdownSource, embeddedHtml);
                }
                let renderedHtml = marked.parse(markdownSource, { gfm: true, breaks: true, pedantic: false, smartypants: false });
                renderedHtml = restoreDashboardEmbeddedHtml(renderedHtml, embeddedHtml);
                return styleStationActionCommandHtml(renderedHtml);
            } 
            catch (e) { 
                console.error("Markdown parsing error:", e, "Input:", markdownText); 
                const pre = document.createElement('pre'); pre.style.whiteSpace = 'pre-wrap';
                pre.textContent = String(markdownText || ""); return pre.outerHTML;
            }
        }
        console.warn("marked.js not loaded. Displaying raw text.");
        const pre = document.createElement('pre'); pre.style.whiteSpace = 'pre-wrap';
        pre.textContent = String(markdownText || ""); return pre.outerHTML;
    }
    
    function escapeHtml(unsafe) {
        if (unsafe === null || typeof unsafe === 'undefined') return '';
        return String(unsafe).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    function formatEventTime(timestamp) {
        const numericTimestamp = Number(timestamp);
        if (!Number.isFinite(numericTimestamp) || numericTimestamp <= 0) return '';
        const eventDate = new Date(numericTimestamp * 1000);
        if (Number.isNaN(eventDate.getTime())) return '';
        return eventDate.toLocaleTimeString();
    }

    function normalizeThinkingWhitespace(thinkingText) {
        const text = String(thinkingText || '').replace(/\r\n?/g, '\n').trim();
        if (!text) return '';

        const lines = text.split('\n');
        const nonEmptyLines = lines.filter(line => line.trim());
        const newlineDensity = (lines.length - 1) / Math.max(1, text.length);
        const averageLineLength = nonEmptyLines.length
            ? nonEmptyLines.reduce((total, line) => total + line.trim().length, 0) / nonEmptyLines.length
            : text.length;
        const shortLineRatio = nonEmptyLines.length
            ? nonEmptyLines.filter(line => line.trim().length <= 4).length / nonEmptyLines.length
            : 0;
        const looksTokenFragmented = lines.length >= 9
            && newlineDensity >= 0.055
            && (averageLineLength <= 24 || shortLineRatio >= 0.3);

        if (!looksTokenFragmented) return text;

        return text
            .split(/\n{2,}/)
            .map(paragraph => {
                const fragments = paragraph.split('\n');
                let joined = fragments[0] ? fragments[0].trim() : '';
                for (let index = 1; index < fragments.length; index += 1) {
                    const previousFragment = fragments[index - 1];
                    const currentFragment = fragments[index];
                    const currentText = currentFragment.trim();
                    if (!currentText) continue;
                    const hasWordBoundary = /[ \t]$/.test(previousFragment) || /^[ \t]/.test(currentFragment);
                    joined += `${hasWordBoundary ? ' ' : ''}${currentText}`;
                }
                return joined.replace(/[ \t]{2,}/g, ' ').trim();
            })
            .filter(Boolean)
            .join('\n\n');
    }

    function renderCollapsibleThinking(thinkingText, label = 'Agent Thinking') {
        const content = normalizeThinkingWhitespace(thinkingText);
        if (!content) return '';
        return `
            <details class="thinking-block border-l-4 border-sky-700 bg-slate-800/50 mb-2 text-slate-400 text-xs rounded">
                <summary class="thinking-summary">
                    <span class="thinking-summary-label">${escapeHtml(label)}</span>
                    <span class="thinking-toggle-text" aria-hidden="true"></span>
                </summary>
                <div class="thinking-content whitespace-pre-wrap">${escapeHtml(content)}</div>
            </details>
        `;
    }

    function formatAgentListForStatus(agentNames, maxVisible = 5) {
        if (!Array.isArray(agentNames) || agentNames.length === 0) return 'none';
        const visibleNames = agentNames.slice(0, maxVisible).join(', ');
        const hiddenCount = agentNames.length - maxVisible;
        return hiddenCount > 0 ? `${visibleNames} +${hiddenCount} more` : visibleNames;
    }

    function formatParallelTickStatus(parallelTickStatus) {
        if (!parallelTickStatus) return '';
        if (parallelTickStatus.error) {
            return `Parallel status unavailable: ${parallelTickStatus.error}`;
        }
        if (!parallelTickStatus.active) return '';

        const counts = parallelTickStatus.counts || {};
        const preparing = parallelTickStatus.preparing_station_response || [];
        const waiting = parallelTickStatus.waiting_for_response || [];
        const pendingCommit = parallelTickStatus.response_received_pending_commit || [];
        const internalRunning = parallelTickStatus.internal_action_running || [];
        const total = Number.isFinite(Number(counts.total)) ? Number(counts.total) : 0;
        const prepared = Number.isFinite(Number(counts.observation_prepared)) ? Number(counts.observation_prepared) : 0;
        const responsesReceived = Number.isFinite(Number(counts.response_received)) ? Number(counts.response_received) : 0;
        const committed = Number.isFinite(Number(counts.committed)) ? Number(counts.committed) : 0;

        if (preparing.length > 0) return `Preparing station responses (${prepared}/${total} ready)`;
        if (waiting.length > 0) return `Waiting for ${formatAgentListForStatus(waiting, 4)} (${responsesReceived}/${total} done)`;
        if (pendingCommit.length > 0) return `Committing ${pendingCommit.length} response(s) (${committed}/${total} done)`;
        if (internalRunning.length > 0) return `Waiting for internal action: ${formatAgentListForStatus(internalRunning, 4)}`;
        if (total > 0) return `Committed ${committed}/${total} responses`;
        return 'No active agents';
    }

    function setTooltipIfPresent(elementId, text) {
        const element = document.getElementById(elementId);
        if (!element) return;
        const tooltipText = String(text || '').trim();
        if (tooltipText) {
            element.setAttribute('data-tooltip', tooltipText);
            element.setAttribute('tabindex', '0');
        } else {
            element.removeAttribute('data-tooltip');
            element.removeAttribute('tabindex');
        }
    }

    function formatMinutesFromSeconds(seconds) {
        if (seconds === null || typeof seconds === 'undefined' || seconds === '') return 'N/A';
        const numericSeconds = Number(seconds);
        if (!Number.isFinite(numericSeconds)) return 'N/A';
        if (numericSeconds > 0 && numericSeconds < 60) return '<1m';
        return `${Math.round(Math.max(0, numericSeconds) / 60)}m`;
    }

    function formatElapsedTickMinutes(elapsedMs) {
        if (!Number.isFinite(elapsedMs)) return 'N/A';
        return `${Math.floor(Math.max(0, elapsedMs) / 60000)}m`;
    }

    function interpolateColor(start, end, ratio) {
        const clamped = Math.max(0, Math.min(1, ratio));
        const values = start.map((value, index) => Math.round(value + (end[index] - value) * clamped));
        return `rgb(${values[0]}, ${values[1]}, ${values[2]})`;
    }

    function getTickAgeColor(elapsedMinutes) {
        const green = [34, 197, 94];
        const yellow = [234, 179, 8];
        const red = [239, 68, 68];
        if (!Number.isFinite(elapsedMinutes)) return '';
        if (elapsedMinutes < 30) return `rgb(${green.join(', ')})`;
        if (elapsedMinutes < 60) return interpolateColor(green, yellow, (elapsedMinutes - 30) / 30);
        if (elapsedMinutes < 120) return interpolateColor(yellow, red, (elapsedMinutes - 60) / 60);
        return `rgb(${red.join(', ')})`;
    }

    function getTopScoreAgeColor(tickAge) {
        const green = [34, 197, 94];
        const yellow = [234, 179, 8];
        const red = [239, 68, 68];
        if (!Number.isFinite(tickAge)) return '';
        if (tickAge <= 300) return `rgb(${green.join(', ')})`;
        if (tickAge < 600) return interpolateColor(green, yellow, (tickAge - 300) / 300);
        if (tickAge < 900) return interpolateColor(yellow, red, (tickAge - 600) / 300);
        return `rgb(${red.join(', ')})`;
    }

    function renderTimeSinceLastTick() {
        const valueElement = document.getElementById('queued-experiments-count');
        const detailsElement = document.getElementById('queued-experiments-details');
        if (!valueElement) return;

        if (!Number.isFinite(latestTickStartedTimestampMs)) {
            valueElement.textContent = 'N/A';
            valueElement.style.color = '';
            valueElement.style.removeProperty('--tick-age-color');
            valueElement.classList.remove('tick-age-colored');
            if (detailsElement) detailsElement.classList.add('hidden');
            setTooltipIfPresent('queued-experiments-metric-card', latestTickTimingTooltip || 'No tick timing record is available.');
            return;
        }

        const elapsedMs = Math.max(0, Date.now() - latestTickStartedTimestampMs);
        const elapsedMinutes = elapsedMs / 60000;
        const color = getTickAgeColor(elapsedMinutes);
        valueElement.textContent = formatElapsedTickMinutes(elapsedMs);
        valueElement.style.setProperty('--tick-age-color', color);
        valueElement.style.color = color;
        valueElement.classList.add('tick-age-colored');
        if (detailsElement) {
            detailsElement.textContent = '';
            detailsElement.classList.add('hidden');
        }
        setTooltipIfPresent('queued-experiments-metric-card', latestTickTimingTooltip);
    }

    function updateTickTimingFromStats(tickTiming) {
        const timing = tickTiming || {};
        const startedSeconds = Number(timing.latest_tick_started_timestamp);
        if (Number.isFinite(startedSeconds) && startedSeconds > 0) {
            latestTickStartedTimestampMs = startedSeconds * 1000;
        } else {
            latestTickStartedTimestampMs = null;
        }

        latestTickTimingTooltip = [
            `Average time per last 10 ticks: ${formatMinutesFromSeconds(timing.average_last_10_tick_seconds)}`,
            `Average time per last 50 ticks: ${formatMinutesFromSeconds(timing.average_last_50_tick_seconds)}`,
            `Average time per last 100 ticks: ${formatMinutesFromSeconds(timing.average_last_100_tick_seconds)}`
        ].join('\n');

        renderTimeSinceLastTick();
    }

    function _getTemporalChatMessages(agentName) {
        if (!temporalChatByAgent.has(agentName)) temporalChatByAgent.set(agentName, []);
        return temporalChatByAgent.get(agentName);
    }

    function _getTemporalChatQueue(agentName) {
        if (!temporalChatQueueByAgent.has(agentName)) temporalChatQueueByAgent.set(agentName, []);
        return temporalChatQueueByAgent.get(agentName);
    }

    function _isTemporalChatProcessing(agentName) {
        return !!temporalChatProcessingByAgent.get(agentName);
    }

    function _isTemporalChatBranching(agentName) {
        return !!temporalChatBranchingByAgent.get(agentName);
    }

    function _updateCachedTemporalChatMeta(agentName, chatState) {
        if (!agentName || !chatState) return;
        const agentInfo = fullAgentListCache.find(a => a && a.name === agentName);
        if (!agentInfo) return;
        agentInfo.temporal_chat_exists = !!chatState.exists;
        agentInfo.temporal_chat_base_tick = typeof chatState.base_tick === 'undefined' ? null : chatState.base_tick;
        agentInfo.temporal_chat_updated_at = chatState.updated_at || null;
        if (temporalChatAgentSelector) {
            Array.from(temporalChatAgentSelector.options || []).forEach(option => {
                if (option.value === agentName) {
                    option.textContent = formatAgentDisplayName(agentInfo, { includeBranchMarker: true });
                }
            });
        }
    }

    function _applyTemporalChatState(agentName, chatState) {
        if (!agentName || !chatState) return;
        const rawMessages = Array.isArray(chatState.messages) ? chatState.messages : [];
        temporalChatByAgent.set(agentName, rawMessages
            .filter(m => m && (m.role === 'user' || m.role === 'assistant' || m.role === 'model'))
            .map(m => ({
                role: m.role === 'model' ? 'assistant' : m.role,
                content: String(m.content || ''),
                thinking_content: String(m.thinking_content || ''),
                tick: m.tick
            })));
        temporalChatBaseTickByAgent.set(agentName, typeof chatState.base_tick === 'undefined' ? null : chatState.base_tick);
        temporalChatExistsByAgent.set(agentName, !!chatState.exists);
        _updateCachedTemporalChatMeta(agentName, chatState);
    }

    function _updateTemporalChatBaseTick(agentName) {
        if (!temporalChatBaseTick) return;
        if (!agentName) {
            temporalChatBaseTick.textContent = 'Dialogue (not started):';
            return;
        }
        const baseTick = temporalChatBaseTickByAgent.get(agentName);
        const exists = !!temporalChatExistsByAgent.get(agentName);
        temporalChatBaseTick.textContent = exists && baseTick !== null && typeof baseTick !== 'undefined'
            ? `Dialogue (branched from tick ${baseTick}):`
            : 'Dialogue (not started):';
    }

    function _getDashboardCurrentTick() {
        if (!stationTickDashboard) return null;
        const parsed = parseInt(String(stationTickDashboard.textContent || '').trim(), 10);
        return Number.isFinite(parsed) ? parsed : null;
    }

    function _parseTemporalChatBranchTickInput() {
        if (!temporalChatBranchTickInput) return { ok: true, value: null };
        const raw = String(temporalChatBranchTickInput.value || '').trim();
        if (!raw) return { ok: true, value: null };
        if (!/^\d+$/.test(raw)) {
            return { ok: false, error: 'Branch tick must be a whole non-negative number.' };
        }
        const branchTick = parseInt(raw, 10);
        const currentTick = _getDashboardCurrentTick();
        if (currentTick !== null && branchTick > currentTick) {
            return { ok: false, error: `Branch tick cannot be in the future. Current station tick is ${currentTick}.` };
        }
        return { ok: true, value: branchTick };
    }

    async function _loadTemporalChatState(agentName) {
        if (!agentName) return null;
        const result = await fetchApi(`/api/agent/${encodeURIComponent(agentName)}/temporal_chat`, 'GET');
        if (result && result.success && result.chat) {
            _applyTemporalChatState(agentName, result.chat);
            return result.chat;
        }
        throw new Error((result && (result.error || result.message)) || 'Failed to load chat.');
    }

    function _updateTemporalChatSendButtonState(agentName) {
        if (confirmTemporalChatSendButton) {
            const q = agentName ? _getTemporalChatQueue(agentName) : [];
            const qLen = q ? q.length : 0;
            const busy = agentName ? _isTemporalChatProcessing(agentName) : false;
            const branching = agentName ? _isTemporalChatBranching(agentName) : false;
            if (branching) {
                confirmTemporalChatSendButton.textContent = 'Send';
            } else if (qLen > 0) {
                confirmTemporalChatSendButton.textContent = `Send (Queued: ${qLen})`;
            } else if (busy) {
                confirmTemporalChatSendButton.textContent = 'Sending...';
            } else {
                confirmTemporalChatSendButton.textContent = 'Send';
            }
            confirmTemporalChatSendButton.disabled = branching; // sending still allows enqueue
        }

        if (branchTemporalChatButton) {
            const qLen = agentName ? _getTemporalChatQueue(agentName).length : 0;
            const exists = agentName ? !!temporalChatExistsByAgent.get(agentName) : false;
            const branching = agentName ? _isTemporalChatBranching(agentName) : false;
            branchTemporalChatButton.textContent = branching ? 'Branching...' : (exists ? 'Branch Again' : 'Branch');
            branchTemporalChatButton.disabled = !agentName || _isTemporalChatProcessing(agentName) || qLen > 0;
        }

        if (temporalChatBranchTickInput) {
            const qLen = agentName ? _getTemporalChatQueue(agentName).length : 0;
            temporalChatBranchTickInput.disabled = !agentName || _isTemporalChatProcessing(agentName) || qLen > 0;
        }
        _updateTemporalChatBaseTick(agentName);
    }

    function _renderTemporalChatTranscript(agentName) {
        if (!temporalChatTranscript) return;
        const msgs = agentName ? _getTemporalChatMessages(agentName) : [];
        if (!msgs || msgs.length === 0) {
            temporalChatTranscript.innerHTML = '<div class="text-slate-500 text-xs">No messages yet.</div>';
            _updateTemporalChatSendButtonState(agentName);
            return;
        }

        temporalChatTranscript.innerHTML = '';
        msgs.forEach(m => {
            const role = m.role === 'assistant' ? 'Agent' : 'You';
            const tickText = typeof m.tick !== 'undefined' && m.tick !== null ? ` (Tick ${escapeHtml(String(m.tick))})` : '';
            const headerHtml = `<strong>[${role}${tickText}]</strong>`;
            const content = String(m.content || "");
            const thinkingContent = normalizeThinkingWhitespace(m.thinking_content);
            const thinkingHtml = renderCollapsibleThinking(thinkingContent);
            const rendered = renderMarkdownForDashboard(content);
            const bubbleContentHtml = `${headerHtml}${thinkingHtml}<div class="mt-1 text-sm markdown-content-host">${rendered}</div>`;
            const rawTextForCopy = thinkingContent.trim()
                ? `Thinking:\n${thinkingContent}\n\nResponse:\n${content}`
                : content;
            const bubbleStyle = m.role === 'assistant' ? 'chat-bubble-agent' : 'chat-bubble-station';
            temporalChatTranscript.appendChild(createLogBubble(bubbleContentHtml, rawTextForCopy, bubbleStyle, {
                isCollapsible: false,
                sourceType: m.role === 'assistant' ? 'agent' : 'station'
            }));
        });
        temporalChatTranscript.scrollTop = temporalChatTranscript.scrollHeight;
        _updateTemporalChatSendButtonState(agentName);
    }

    function _formatTemporalChatForCopy(agentName, includeThinking = true) {
        // Copy only the visible temporal fork, not the hidden frozen station dialogue context.
        const msgs = agentName ? _getTemporalChatMessages(agentName).filter(m => m && (m.role === 'user' || m.role === 'assistant')) : [];
        if (!msgs || msgs.length === 0) return '';
        return msgs.map(m => {
            const role = m.role === 'assistant' ? 'Agent' : 'You';
            const thinkingContent = normalizeThinkingWhitespace(m.thinking_content);
            const thinkingText = includeThinking && thinkingContent ? `Thinking:\n${thinkingContent}\n\n` : '';
            return `${role}:\n${thinkingText}${String(m.content || '').trim()}\n`;
        }).join('\n');
    }

    function _closeTemporalChatModal() {
        hideModalWithoutReset(temporalChatModal, true);
    }

    function formatArchiveScore(score) {
        if (score === null || typeof score === 'undefined' || score === '') return 'N/A';
        const numericScore = Number(score);
        return Number.isFinite(numericScore) ? `${numericScore}/10` : 'N/A';
    }

    function formatTopResearchScore(score) {
        if (score === null || typeof score === 'undefined') return 'N/A';
        if (typeof score !== 'number' && typeof score !== 'string') return String(score);

        const rawScore = String(score).trim();
        if (!rawScore) return 'N/A';

        const numericPattern = /^[+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?$/;
        if (!numericPattern.test(rawScore)) return rawScore;

        const numericScore = Number(rawScore);
        if (!Number.isFinite(numericScore)) return rawScore;
        if (Number.isInteger(numericScore)) return rawScore;

        const decimalPlaces = 8;
        const factor = 10 ** decimalPlaces;
        const truncatedScore = Math.trunc(numericScore * factor) / factor;
        return truncatedScore.toFixed(decimalPlaces);
    }

    function getArchiveSortValue(paper, key) {
        const value = paper ? paper[key] : null;
        if (key === 'numeric_id' || key === 'accepted_tick' || key === 'word_count') {
            const numericValue = Number(value);
            return Number.isFinite(numericValue) ? numericValue : -Infinity;
        }
        if (key === 'reviewer_score') {
            const numericValue = Number(value);
            return Number.isFinite(numericValue) ? numericValue : -Infinity;
        }
        return String(value || '').toLowerCase();
    }

    function sortArchivePapers() {
        const { key, direction } = archiveSortState;
        const multiplier = direction === 'asc' ? 1 : -1;
        archivePapersCache.sort((a, b) => {
            const aValue = getArchiveSortValue(a, key);
            const bValue = getArchiveSortValue(b, key);
            if (aValue < bValue) return -1 * multiplier;
            if (aValue > bValue) return 1 * multiplier;
            return (Number(b.numeric_id) || 0) - (Number(a.numeric_id) || 0);
        });
    }

    function updateArchiveSortButtons() {
        document.querySelectorAll('.archive-sort-button').forEach(button => {
            const key = button.getAttribute('data-sort-key');
            const label = (button.textContent || '').replace(/\s[↑↓]$/, '');
            if (key === archiveSortState.key) {
                button.textContent = `${label} ${archiveSortState.direction === 'asc' ? '↑' : '↓'}`;
            } else {
                button.textContent = label;
            }
        });
    }

    function setArchiveListLoadingState(message) {
        if (!archivePapersTableBody) return;
        archivePapersTableBody.innerHTML = `<tr><td colspan="6" class="archive-empty-state">${escapeHtml(message)}</td></tr>`;
    }

    function updateArchivePagination() {
        const total = archivePapersCache.length;
        const totalPages = Math.max(1, Math.ceil(total / archiveRowCount));
        archivePage = Math.min(Math.max(1, archivePage), totalPages);
        if (archivePageSummary) {
            archivePageSummary.textContent = `Page ${archivePage} of ${totalPages} · ${total} ${total === 1 ? 'paper' : 'papers'}`;
        }
        if (archivePreviousPageButton) archivePreviousPageButton.disabled = archivePage <= 1;
        if (archiveNextPageButton) archiveNextPageButton.disabled = archivePage >= totalPages;
    }

    function renderArchivePapersTable() {
        if (!archivePapersTableBody) return;
        if (!Array.isArray(archivePapersCache) || archivePapersCache.length === 0) {
            setArchiveListLoadingState('No archive papers available.');
            updateArchiveSortButtons();
            updateArchivePagination();
            return;
        }

        sortArchivePapers();
        updateArchiveSortButtons();
        updateArchivePagination();

        const pageStart = (archivePage - 1) * archiveRowCount;
        const rows = archivePapersCache.slice(pageStart, pageStart + archiveRowCount).map(paper => {
            const title = escapeHtml(paper.title || 'Untitled');
            const author = escapeHtml(paper.author || 'Unknown');
            const acceptedTick = escapeHtml(String(paper.accepted_tick ?? 'N/A'));
            const reviewerScore = formatArchiveScore(paper.reviewer_score);
            const abstract = escapeHtml(String(paper.abstract || ''));
            const abstractPreview = abstract.length > 220 ? `${abstract.slice(0, 220)}...` : abstract;
            const scoreClass = reviewerScore === 'N/A' ? 'archive-reviewer-score na' : 'archive-reviewer-score';
            return `
                <tr class="archive-papers-row" data-archive-id="${paper.numeric_id}">
                    <td>${paper.numeric_id}</td>
                    <td class="archive-paper-title-cell">
                        <strong>${title}</strong>
                        <span>${abstractPreview || 'No abstract available.'}</span>
                    </td>
                    <td>${author}</td>
                    <td>${acceptedTick}</td>
                    <td><span class="${scoreClass}">${escapeHtml(reviewerScore)}</span></td>
                    <td>${escapeHtml(String(paper.word_count ?? 0))}</td>
                </tr>
            `;
        });

        archivePapersTableBody.innerHTML = rows.join('');
        archivePapersTableBody.querySelectorAll('.archive-papers-row').forEach(row => {
            row.addEventListener('click', () => {
                const numericId = Number(row.getAttribute('data-archive-id'));
                if (Number.isFinite(numericId)) {
                    void openArchivePaperDetail(numericId);
                }
            });
        });
    }

    function showArchiveListView() {
        if (archivePapersListView) archivePapersListView.classList.remove('hidden');
        if (archivePaperDetailView) archivePaperDetailView.classList.add('hidden');
    }

    function showArchiveDetailView() {
        if (archivePapersListView) archivePapersListView.classList.add('hidden');
        if (archivePaperDetailView) archivePaperDetailView.classList.remove('hidden');
    }

    function formatWebSurveyTimestamp(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric) || numeric <= 0) return '—';
        return new Date(numeric * 1000).toLocaleString();
    }

    function setWebArchiveSurveyTableState(message) {
        if (!webArchiveSurveysTableBody) return;
        webArchiveSurveysTableBody.innerHTML = `<tr><td colspan="5" class="archive-empty-state">${escapeHtml(message)}</td></tr>`;
    }

    function sortWebArchiveSurveys() {
        webArchiveSurveys.sort((a, b) => {
            const tickDifference = (Number(b.source_tick) || 0) - (Number(a.source_tick) || 0);
            if (tickDifference !== 0) return tickDifference;
            const timeDifference = (Number(b.submitted_timestamp) || 0) - (Number(a.submitted_timestamp) || 0);
            if (timeDifference !== 0) return timeDifference;
            return (Number(b.id) || 0) - (Number(a.id) || 0);
        });
    }

    function updateWebArchiveSurveyPagination() {
        const total = webArchiveSurveys.length;
        const totalPages = Math.max(1, Math.ceil(total / webArchiveSurveyRowCount));
        webArchiveSurveyPage = Math.min(Math.max(1, webArchiveSurveyPage), totalPages);
        if (webArchiveSurveyPageSummary) {
            webArchiveSurveyPageSummary.textContent = `Page ${webArchiveSurveyPage} of ${totalPages} · ${total} ${total === 1 ? 'report' : 'reports'}`;
        }
        if (webArchiveSurveyPreviousPageButton) webArchiveSurveyPreviousPageButton.disabled = webArchiveSurveyPage <= 1;
        if (webArchiveSurveyNextPageButton) webArchiveSurveyNextPageButton.disabled = webArchiveSurveyPage >= totalPages;
    }

    function renderWebArchiveSurveyTemplates() {
        if (!webArchiveSurveyTemplateButtons) return;
        webArchiveSurveyTemplateButtons.innerHTML = webArchiveSurveyTemplates.map(template => `
            <button type="button" class="archive-survey-template-button" data-template-id="${escapeHtml(template.id)}">
                ${escapeHtml(template.label || template.id)}
            </button>
        `).join('');
        webArchiveSurveyTemplateButtons.querySelectorAll('.archive-survey-template-button').forEach(button => {
            button.addEventListener('click', () => {
                const template = webArchiveSurveyTemplates.find(item => item.id === button.getAttribute('data-template-id'));
                if (!template || !webArchiveSurveyPrompt) return;
                webArchiveSurveyPrompt.value = String(template.prompt || '');
                webArchiveSurveyPrompt.focus();
            });
        });
    }

    function renderWebArchiveSurveysTable() {
        if (!webArchiveSurveysTableBody) return;
        if (!webArchiveSurveys.length) {
            setWebArchiveSurveyTableState('No survey reports yet. Submit a request to create the first one.');
            updateWebArchiveSurveyPagination();
            return;
        }
        sortWebArchiveSurveys();
        updateWebArchiveSurveyPagination();
        const pageStart = (webArchiveSurveyPage - 1) * webArchiveSurveyRowCount;
        webArchiveSurveysTableBody.innerHTML = webArchiveSurveys.slice(pageStart, pageStart + webArchiveSurveyRowCount).map(survey => {
            const status = String(survey.status || 'queued').toLowerCase();
            const isClickable = status === 'completed' && Boolean(survey.has_report);
            return `
                <tr class="archive-papers-row archive-survey-row ${isClickable ? 'is-clickable' : 'is-disabled'}" data-web-survey-id="${escapeHtml(String(survey.id))}">
                    <td>W${escapeHtml(String(survey.id))}</td>
                    <td><span class="archive-survey-status status-${escapeHtml(status)}">${escapeHtml(status)}</span></td>
                    <td class="archive-paper-title-cell">
                        <strong>${escapeHtml(survey.prompt_preview || 'Survey request')}</strong>
                        ${isClickable ? '' : `<span>${survey.error ? escapeHtml(String(survey.error)) : (status === 'running' ? 'Report is running. Select Refresh to check again.' : 'Waiting to start. Select Refresh to check again.')}</span>`}
                    </td>
                    <td>${survey.source_tick === null || typeof survey.source_tick === 'undefined' ? '—' : `Tick ${escapeHtml(String(survey.source_tick))}`}</td>
                    <td>${escapeHtml(formatWebSurveyTimestamp(survey.submitted_timestamp))}</td>
                </tr>
            `;
        }).join('');
        webArchiveSurveysTableBody.querySelectorAll('.archive-survey-row.is-clickable').forEach(row => {
            row.addEventListener('click', () => {
                const surveyId = Number(row.getAttribute('data-web-survey-id'));
                if (Number.isFinite(surveyId)) void openWebArchiveSurvey(surveyId);
            });
        });
    }

    async function loadWebArchiveSurveyTemplates() {
        const result = await fetchApi('/api/web/archive-surveys/templates', 'GET');
        webArchiveSurveyTemplates = Array.isArray(result.templates) ? result.templates : [];
        renderWebArchiveSurveyTemplates();
    }

    async function loadWebArchiveSurveys({ quiet = false } = {}) {
        if (!quiet) setWebArchiveSurveyTableState('Loading survey history...');
        const result = await fetchApi('/api/web/archive-surveys', 'GET');
        webArchiveSurveys = Array.isArray(result.surveys) ? result.surveys : [];
        webArchiveSurveysLoaded = true;
        renderWebArchiveSurveysTable();
        const selectedId = webArchiveSelectedSurvey ? Number(webArchiveSelectedSurvey.id) : null;
        if (selectedId && webArchiveSurveys.some(item => Number(item.id) === selectedId)) {
            const summary = webArchiveSurveys.find(item => Number(item.id) === selectedId);
            if (summary && ['queued', 'running'].includes(String(summary.status || '').toLowerCase())) {
                await openWebArchiveSurvey(selectedId, { quiet: true });
            }
        }
    }

    function showWebArchiveSurveysListView() {
        if (webArchiveSurveysListView) webArchiveSurveysListView.classList.remove('hidden');
        webArchiveSelectedSurvey = null;
        if (webArchiveSurveyReportShell) webArchiveSurveyReportShell.classList.add('hidden');
    }

    async function openWebArchiveSurveysModal() {
        if (!webArchiveSurveysModal) return;
        showWebArchiveSurveysListView();
        webArchiveSurveysModal.style.display = 'block';
        try {
            if (!webArchiveSurveysLoaded) await loadWebArchiveSurveys();
            else renderWebArchiveSurveysTable();
        } catch (_error) {
            setWebArchiveSurveyTableState('Failed to load survey history.');
        }
    }

    async function openWebArchiveSurveyRequestModal() {
        if (!webArchiveSurveyRequestModal) return;
        webArchiveSurveyRequestModal.style.display = 'block';
        try {
            if (!webArchiveSurveyTemplates.length) await loadWebArchiveSurveyTemplates();
        } catch (_error) {
            showDashboardStatus('Failed to load survey templates. You can still write a custom request.', 'error');
        }
        if (webArchiveSurveyPrompt) webArchiveSurveyPrompt.focus();
    }

    function closeWebArchiveSurveyRequestModal(preserveDraft = true) {
        if (!webArchiveSurveyRequestModal) return;
        if (preserveDraft && webArchiveSurveyPrompt && webArchiveSurveyPrompt.value.trim()) {
            webArchiveSurveyRequestModal.dataset.reopenWithDraft = 'true';
        } else {
            delete webArchiveSurveyRequestModal.dataset.reopenWithDraft;
        }
        webArchiveSurveyRequestModal.style.display = 'none';
    }

    async function submitWebArchiveSurvey() {
        if (!webArchiveSurveyPrompt || !submitWebArchiveSurveyButton) return;
        const prompt = webArchiveSurveyPrompt.value.trim();
        if (!prompt) {
            showDashboardStatus('Survey request cannot be empty.', 'error');
            webArchiveSurveyPrompt.focus();
            return;
        }
        submitWebArchiveSurveyButton.disabled = true;
        showDashboardStatus('Submitting survey request...', 'info', 0);
        try {
            const result = await fetchApi('/api/web/archive-surveys', 'POST', {
                prompt
            });
            if (result.survey) {
                webArchiveSurveys = [result.survey, ...webArchiveSurveys.filter(item => Number(item.id) !== Number(result.survey.id))];
                webArchiveSurveysLoaded = true;
                webArchiveSurveyPage = 1;
                renderWebArchiveSurveysTable();
            }
            webArchiveSurveyPrompt.value = '';
            closeWebArchiveSurveyRequestModal(false);
            showDashboardStatus(result.message || 'Survey request submitted.', 'success');
        } finally {
            submitWebArchiveSurveyButton.disabled = false;
        }
    }

    async function openWebArchiveSurvey(surveyId, { quiet = false } = {}) {
        if (!quiet) showDashboardStatus(`Loading Archive Survey W${surveyId}...`, 'info', 1500);
        const result = await fetchApi(`/api/web/archive-surveys/${encodeURIComponent(surveyId)}`, 'GET');
        const survey = result.survey || {};
        const report = String(survey.report_markdown || '').trim();
        if (!report) {
            showDashboardStatus(`Survey W${surveyId} is marked complete, but its report content is unavailable. Press Refresh and try again.`, 'error');
            return;
        }
        webArchiveSelectedSurvey = survey;
        const status = String(survey.status || 'queued').toLowerCase();
        if (webArchiveSurveyReportTitle) webArchiveSurveyReportTitle.textContent = `Archive Survey W${surveyId}`;
        if (webArchiveSurveyReportMeta) {
            const parts = [
                `Survey ID: W${surveyId}`,
                `Status: ${status}`,
                `Submitted Tick: ${survey.source_tick === null || typeof survey.source_tick === 'undefined' ? 'N/A' : survey.source_tick}`,
                `Submitted Time: ${formatWebSurveyTimestamp(survey.submitted_timestamp)}`
            ];
            webArchiveSurveyReportMeta.textContent = parts.join(' | ');
        }
        if (webArchiveSurveyReport) {
            webArchiveSurveyReport.innerHTML = renderMarkdownForDashboard(report);
            webArchiveSurveyReport.scrollTop = 0;
        }
        if (copyWebArchiveSurveyMarkdownButton) copyWebArchiveSurveyMarkdownButton.disabled = false;
        if (removeWebArchiveSurveyButton) {
            removeWebArchiveSurveyButton.disabled = status === 'running';
            removeWebArchiveSurveyButton.title = status === 'running' ? 'Running surveys cannot be removed' : 'Remove this survey request and its stored report';
        }
        if (webArchiveSurveysListView) webArchiveSurveysListView.classList.add('hidden');
        if (webArchiveSurveyReportShell) webArchiveSurveyReportShell.classList.remove('hidden');
    }

    async function removeWebArchiveSurvey(surveyId) {
        if (!window.confirm(`Remove Archive Survey W${surveyId} and its stored report?`)) return;
        const result = await fetchApi(`/api/web/archive-surveys/${encodeURIComponent(surveyId)}`, 'DELETE');
        webArchiveSurveys = webArchiveSurveys.filter(item => Number(item.id) !== Number(surveyId));
        if (webArchiveSelectedSurvey && Number(webArchiveSelectedSurvey.id) === Number(surveyId)) {
            webArchiveSelectedSurvey = null;
            showWebArchiveSurveysListView();
        }
        renderWebArchiveSurveysTable();
        showDashboardStatus(result.message || `Archive Survey W${surveyId} removed.`, 'success');
    }

    async function copyTextToClipboard(text, successMessage) {
        try {
            await navigator.clipboard.writeText(text);
            showDashboardStatus(successMessage, 'success');
        } catch (e) {
            showDashboardStatus('Failed to copy to clipboard.', 'error');
        }
    }

    document.addEventListener('click', (event) => {
        if (!(event.target instanceof Element)) return;
        const copyButton = event.target.closest('.station-yaml-copy-button');
        if (!copyButton) return;
        const card = copyButton.closest('.station-yaml-card');
        const copySource = card ? card.querySelector('.station-yaml-copy-source') : null;
        if (!copySource) return;
        const copyKind = copyButton.dataset.copyKind || 'Content';
        void copyTextToClipboard(copySource.value || '', `${copyKind} copied to clipboard.`);
    });

    function updateCopySystemPromptButtonState() {
        if (!copySystemPromptButton) return;
        const selectedAgent = agentSelectorDashboard ? agentSelectorDashboard.value : 'all';
        copySystemPromptButton.disabled = !selectedAgent || selectedAgent === 'all';
    }

    function updateHistoryWindowSelector() {
        if (!historyWindowSelector) return;
        historyWindowSelector.value = historyWindowMode;
        historyWindowSelector.disabled = isMultistartPreview ||
            (currentSelectedAgentForDialogueView !== "all" && !!isLoadingHistoryForAgent);
    }

    function showHistoryWindowSelector(show) {
        if (!historyWindowSelector) return;
        historyWindowSelector.classList.toggle('hidden', !show);
    }

    function showLoadFullHistoryButton(show, text = 'Load Full History') {
        if (!loadFullHistoryButton) return;
        loadFullHistoryButton.textContent = text;
        loadFullHistoryButton.classList.toggle('hidden', isMultistartPreview || !show);
    }

    async function loadArchivePapers() {
        setArchiveListLoadingState('Loading archive papers...');
        const result = await fetchApi('/api/archive/papers', 'GET');
        archivePapersCache = Array.isArray(result.capsules) ? result.capsules : [];
        archiveAllAbstractsMarkdown = String(result.all_abstracts_markdown || '');
        archivePage = 1;
        renderArchivePapersTable();
    }

    async function openArchivePaperDetail(numericId) {
        showDashboardStatus(`Loading archive paper #${numericId}...`, 'info', 2000);
        const result = await fetchApi(`/api/archive/papers/${numericId}`, 'GET');
        archiveSelectedPaper = result;
        if (archivePaperDetailTitle) {
            archivePaperDetailTitle.textContent = result.title || `Archive Paper #${numericId}`;
        }
        if (archivePaperDetailMeta) {
            const metaParts = [
                `Archive ID: #${numericId}`,
                `Author: ${result.author || 'Unknown'}`,
                `Accepted Tick: ${result.accepted_tick ?? 'N/A'}`,
                `Reviewer Score: ${formatArchiveScore(result.reviewer_score)}`
            ];
            archivePaperDetailMeta.textContent = metaParts.join(' | ');
        }
        if (archivePaperMarkdownContainer) {
            archivePaperMarkdownContainer.innerHTML = renderMarkdownForDashboard(String(result.paper_markdown || ''));
            archivePaperMarkdownContainer.scrollTop = 0;
        }
        showArchiveDetailView();
    }

    async function openArchivePapersModal() {
        if (!archivePapersModal) return;
        archiveSelectedPaper = null;
        showArchiveListView();
        archivePapersModal.style.display = 'block';
        try {
            await loadArchivePapers();
        } catch (error) {
            setArchiveListLoadingState('Failed to load archive papers.');
        }
    }

    function setQuestionListLoadingState(message) {
        if (!questionRoomTableBody) return;
        questionRoomTableBody.innerHTML = `<tr><td colspan="7" class="archive-empty-state">${escapeHtml(message)}</td></tr>`;
    }

    function questionStatusClass(status) {
        const normalized = String(status || 'pending').toLowerCase();
        return ['pending', 'open', 'redacted', 'solved', 'retired'].includes(normalized) ? `status-${normalized}` : 'status-pending';
    }

    function updateQuestionSortButtons() {
        document.querySelectorAll('.question-sort-button').forEach(button => {
            const key = button.getAttribute('data-sort-key');
            const label = (button.textContent || '').replace(/\s[↑↓]$/, '');
            button.textContent = key === questionListState.sortBy
                ? `${label} ${questionListState.sortDirection === 'asc' ? '↑' : '↓'}`
                : label;
        });
    }

    function updateQuestionPagination() {
        if (questionPageSummary) {
            const noun = questionListState.total === 1 ? 'question' : 'questions';
            questionPageSummary.textContent = `Page ${questionListState.page} of ${questionListState.totalPages} · ${questionListState.total} ${noun}`;
        }
        if (questionPreviousPageButton) questionPreviousPageButton.disabled = questionListState.page <= 1;
        if (questionNextPageButton) questionNextPageButton.disabled = questionListState.page >= questionListState.totalPages;
    }

    function renderQuestionRoomTable(questions) {
        if (!questionRoomTableBody) return;
        updateQuestionSortButtons();
        updateQuestionPagination();
        if (!Array.isArray(questions) || questions.length === 0) {
            setQuestionListLoadingState('No questions are available.');
            return;
        }

        questionRoomTableBody.innerHTML = questions.map(question => {
            const title = escapeHtml(question.title || 'Untitled');
            const author = escapeHtml(question.author || 'Unknown');
            const abstract = String(question.abstract || '');
            const abstractPreviewRaw = abstract.length > 220 ? `${abstract.slice(0, 220)}...` : abstract;
            const status = String(question.status || 'pending').toLowerCase();
            return `
                <tr class="question-room-row" data-question-id="${question.numeric_id}">
                    <td>${question.numeric_id}</td>
                    <td class="archive-paper-title-cell">
                        <strong>${title}</strong>
                        <span>${escapeHtml(abstractPreviewRaw || 'No abstract available.')}</span>
                    </td>
                    <td>${author}</td>
                    <td>${escapeHtml(String(question.authored_tick ?? 'N/A'))}</td>
                    <td><span class="question-status-badge ${questionStatusClass(status)}">${escapeHtml(status)}</span></td>
                    <td><span class="question-net-upvote">${escapeHtml(String(question.net_upvote ?? 0))}</span></td>
                    <td>${escapeHtml(String(question.reply_count ?? 0))}</td>
                </tr>
            `;
        }).join('');

        questionRoomTableBody.querySelectorAll('.question-room-row').forEach(row => {
            row.addEventListener('click', () => {
                const numericId = Number(row.getAttribute('data-question-id'));
                if (Number.isFinite(numericId)) void openQuestionRoomDetail(numericId);
            });
        });
    }

    function showQuestionRoomListView() {
        if (questionRoomListView) questionRoomListView.classList.remove('hidden');
        if (questionRoomDetailView) questionRoomDetailView.classList.add('hidden');
    }

    function showQuestionRoomDetailView() {
        if (questionRoomListView) questionRoomListView.classList.add('hidden');
        if (questionRoomDetailView) questionRoomDetailView.classList.remove('hidden');
    }

    async function loadQuestionRoomPage() {
        const requestSequence = ++questionListRequestSequence;
        setQuestionListLoadingState('Loading questions...');
        const params = new URLSearchParams({
            page: String(questionListState.page),
            page_size: String(questionListState.pageSize),
            sort_by: questionListState.sortBy,
            sort_direction: questionListState.sortDirection
        });
        const result = await fetchApi(`/api/questions?${params.toString()}`, 'GET');
        if (requestSequence !== questionListRequestSequence) return;

        questionListState.total = Number(result.total) || 0;
        questionListState.totalPages = Math.max(1, Number(result.total_pages) || 1);
        if (questionListState.page > questionListState.totalPages) {
            questionListState.page = questionListState.totalPages;
            await loadQuestionRoomPage();
            return;
        }
        questionListState.page = Math.max(1, Number(result.page) || questionListState.page);
        renderQuestionRoomTable(Array.isArray(result.questions) ? result.questions : []);
    }

    function renderQuestionRoomThread(question) {
        if (!questionRoomThread) return;
        const messages = Array.isArray(question.messages) ? question.messages : [];
        const sections = [];
        if (question.abstract) {
            sections.push(`<div class="question-thread-abstract"><strong>Abstract</strong><div class="mt-1">${escapeHtml(String(question.abstract))}</div></div>`);
        }
        if (question.status === 'solved' && question.solved_by_message_id && !messages.some(message => message.is_accepted)) {
            sections.push('<div class="question-thread-abstract">The accepted solution is no longer available in the active thread.</div>');
        }
        messages.forEach((message, index) => {
            const kind = message.is_question ? 'Question' : `Reply ${index}`;
            const postAction = message.is_question ? 'posted the question' : 'posted a reply';
            const acceptedBadge = message.is_accepted
                ? '<span class="question-accepted-badge">Accepted Solution</span>'
                : '';
            const optionalTitle = message.title
                ? `<span class="question-thread-message-title">${escapeHtml(String(message.title))}</span>`
                : '';
            const solutionVoteMeta = message.is_question
                ? ''
                : `<span class="question-thread-message-meta">Solution net upvote: ${escapeHtml(String(message.solution_net_upvote ?? 0))}</span>`;
            sections.push(`
                <article class="question-thread-message${message.is_accepted ? ' is-accepted' : ''}">
                    <div class="question-thread-message-header">
                        <div class="question-thread-byline">
                            <strong class="question-thread-author">${escapeHtml(message.author || 'Unknown')}</strong>
                            <span class="question-thread-posted">${postAction} at Tick ${escapeHtml(String(message.posted_tick ?? 'N/A'))}</span>
                            ${acceptedBadge}
                        </div>
                        <div class="question-thread-message-details">
                            <span class="question-message-kind">${escapeHtml(kind)}</span>
                            ${optionalTitle}
                            <span class="question-thread-message-meta">${escapeHtml(String(message.message_id || ''))}</span>
                            ${solutionVoteMeta}
                        </div>
                    </div>
                    <div class="markdown-content-host">${renderMarkdownForDashboard(String(message.content || ''))}</div>
                </article>
            `);
        });
        if (sections.length === 0) sections.push('<div class="archive-empty-state">This question has no active messages.</div>');
        questionRoomThread.innerHTML = sections.join('');
        questionRoomThread.scrollTop = 0;
    }

    async function openQuestionRoomDetail(numericId) {
        showDashboardStatus(`Loading Question #${numericId}...`, 'info', 2000);
        const result = await fetchApi(`/api/questions/${numericId}`, 'GET');
        if (questionRoomDetailTitle) questionRoomDetailTitle.textContent = result.title || `Question #${numericId}`;
        if (questionRoomDetailStatus) {
            const status = String(result.status || 'pending').toLowerCase();
            questionRoomDetailStatus.textContent = status;
            questionRoomDetailStatus.className = `question-status-badge ${questionStatusClass(status)}`;
        }
        if (questionRoomDetailMeta) {
            questionRoomDetailMeta.textContent = [
                `Question #${numericId}`,
                `Author: ${result.author || 'Unknown'}`,
                `Authored Tick: ${result.authored_tick ?? 'N/A'}`,
                `Last Activity: ${result.last_activity_tick ?? 'N/A'}`,
                `Net Upvote: ${result.net_upvote ?? 0}`
            ].join(' | ');
        }
        renderQuestionRoomThread(result);
        showQuestionRoomDetailView();
    }

    async function openQuestionRoomModal() {
        if (!questionRoomModal) return;
        showQuestionRoomListView();
        questionRoomModal.style.display = 'block';
        try {
            await loadQuestionRoomPage();
        } catch (error) {
            setQuestionListLoadingState('Could not load questions.');
        }
    }

    function formatAgentDisplayName(agent, options = {}) {
        const modelName = agent.model_name || "Unknown Model";
        const supervisorLabel = agent.is_supervisor ? " [Supervisor]" : "";
        const branchLabel = options.includeBranchMarker && agent.temporal_chat_exists && agent.temporal_chat_base_tick !== null && typeof agent.temporal_chat_base_tick !== 'undefined'
            ? `; Branched from Tick ${agent.temporal_chat_base_tick}`
            : "";
        
        // Handle ended/ascended agents
        let suffix = '';
        if (agent.status.startsWith("Session Ended")) {
            suffix = '; ended';
        } else if (agent.status.startsWith("Ascended")) {
            suffix = '; ascended';
        }
        
        return `${agent.name} (${modelName}${suffix}${branchLabel})${supervisorLabel}`;
    }

    function getCurrentLogContainer() {
        return currentSelectedAgentForDialogueView === "all" ? globalNotificationBubbleLog : agentSpecificBubbleLog;
    }

    function getPlainTextPreview(rawText, maxChars = 220) {
        const text = String(rawText || '')
            .replace(/\r\n/g, '\n')
            .replace(/[ \t]+/g, ' ')
            .replace(/\n{3,}/g, '\n\n')
            .trim();
        if (!text) return '';
        if (text.length <= maxChars) return text;
        return `${text.slice(0, maxChars).trimEnd()}...`;
    }

    function shouldMakeMessageCollapsible(eventType, rawTextForCopy, bubbleStyle) {
        if (bubbleStyle && bubbleStyle.includes('chat-bubble-error')) return false;
        if (['error', 'stream_error', 'connector_error'].includes(eventType)) return false;

        const rawText = String(rawTextForCopy || '');
        if (!rawText.trim()) return false;

        const lineCount = rawText.split(/\r\n|\r|\n/).length;
        return rawText.length > 700 || lineCount > 8;
    }

    function getSourceKey(sourceType) {
        return `${currentSelectedAgentForDialogueView || 'all'}:${sourceType}`;
    }

    function updateLogModeButtons() {
        const buttons = [logModeFullButton, logModeCollapsedButton].filter(Boolean);
        buttons.forEach(button => {
            const isActive = button.dataset.logMode === logDisplayMode;
            button.classList.toggle('log-display-mode-button-active', isActive);
        });
    }

    function applyLogDisplayMode() {
        [globalNotificationBubbleLog, agentSpecificBubbleLog].filter(Boolean).forEach(container => {
            container.classList.remove('log-mode-full', 'log-mode-collapsed');
            container.classList.add(`log-mode-${logDisplayMode}`);
        });
        updateLogModeButtons();
        updateSourceToggleButtons();
    }

    function setCurrentLogExpandedState(expanded) {
        const container = getCurrentLogContainer();
        if (!container) return;
        container.querySelectorAll('.log-message-collapsible').forEach(bubble => {
            setLogMessageExpanded(bubble, expanded);
        });
    }

    function setLogDisplayMode(mode) {
        if (!LOG_MODES.includes(mode)) return;
        logDisplayMode = mode;
        localStorage.setItem(LOG_DISPLAY_MODE_STORAGE_KEY, mode);
        ['agent', 'station'].forEach(sourceType => {
            const sourceKey = getSourceKey(sourceType);
            if (mode === 'full') collapsedSourceKeys.delete(sourceKey);
            else collapsedSourceKeys.add(sourceKey);
        });
        applyLogDisplayMode();
        setCurrentLogExpandedState(mode === 'full');
    }

    function syncLogMessageToggleText(bubbleWrapper) {
        if (!bubbleWrapper) return;
        const toggle = bubbleWrapper.querySelector('.log-message-toggle');
        if (!toggle) return;
        const isExpanded = bubbleWrapper.classList.contains('log-message-expanded');
        toggle.textContent = isExpanded ? 'Collapse' : 'Expand';
        toggle.setAttribute('aria-expanded', isExpanded ? 'true' : 'false');
    }

    function setLogMessageExpanded(bubbleWrapper, expanded) {
        if (!bubbleWrapper || !bubbleWrapper.classList.contains('log-message-collapsible')) return;
        const messageId = bubbleWrapper.dataset.messageId;
        bubbleWrapper.classList.toggle('log-message-expanded', expanded);
        if (messageId) {
            if (expanded) expandedMessageIds.add(messageId);
            else expandedMessageIds.delete(messageId);
        }
        syncLogMessageToggleText(bubbleWrapper);
    }

    function setCurrentLogSourceCollapsed(sourceType, collapsed) {
        const container = getCurrentLogContainer();
        if (!container) return;
        const sourceKey = getSourceKey(sourceType);
        if (collapsed) collapsedSourceKeys.add(sourceKey);
        else collapsedSourceKeys.delete(sourceKey);

        container.querySelectorAll(`.log-message-collapsible[data-source-type="${sourceType}"]`).forEach(bubble => {
            setLogMessageExpanded(bubble, !collapsed);
        });
        updateSourceToggleButtons();
    }

    function isCurrentLogSourceCollapsed(sourceType) {
        return collapsedSourceKeys.has(getSourceKey(sourceType));
    }

    function updateSourceToggleButtons() {
        if (toggleAgentLogButton) {
            const collapsed = isCurrentLogSourceCollapsed('agent');
            toggleAgentLogButton.textContent = collapsed ? 'Expand Agent' : 'Collapse Agent';
        }
        if (toggleStationLogButton) {
            const collapsed = isCurrentLogSourceCollapsed('station');
            toggleStationLogButton.textContent = collapsed ? 'Expand Station' : 'Collapse Station';
        }
    }

    function createLogBubble(bubbleContentHtml, rawTextForCopy, bubbleStyle, options = {}) {
        const bubbleWrapper = document.createElement('div');
        bubbleWrapper.classList.add('chat-bubble', 'text-sm', ...bubbleStyle.split(' '));
        bubbleWrapper.classList.add('log-message-shell');

        const eventType = options.eventType || '';
        const isCollapsible = options.isCollapsible !== undefined
            ? options.isCollapsible
            : shouldMakeMessageCollapsible(eventType, rawTextForCopy, bubbleStyle);
        const messageId = options.messageId || `log-message-${++logMessageSequence}`;
        bubbleWrapper.dataset.messageId = messageId;
        const sourceType = options.sourceType || 'station';
        bubbleWrapper.dataset.sourceType = sourceType;
        if (options.historyKey) {
            bubbleWrapper.dataset.historyKey = options.historyKey;
        }

        if (isCollapsible) {
            bubbleWrapper.classList.add('log-message-collapsible');
            const sourceCollapsed = collapsedSourceKeys.has(getSourceKey(sourceType));
            if (expandedMessageIds.has(messageId) || (logDisplayMode === 'full' && !sourceCollapsed)) {
                bubbleWrapper.classList.add('log-message-expanded');
            }

            const preview = getPlainTextPreview(options.previewText || rawTextForCopy, 220);
            let headerHtml = options.headerHtml || '';
            let bodyHtml = bubbleContentHtml;
            if (!headerHtml) {
                const leadingStrongMatch = String(bubbleContentHtml).match(/^\s*(<strong>[\s\S]*?<\/strong>)([\s\S]*)$/);
                if (leadingStrongMatch) {
                    headerHtml = leadingStrongMatch[1];
                    bodyHtml = leadingStrongMatch[2].trimStart();
                }
            }
            bubbleWrapper.innerHTML = `
                <div class="log-message-header">
                    <div class="log-message-title">${headerHtml}</div>
                    <button type="button" class="log-message-toggle">Expand</button>
                </div>
                <div class="log-message-preview">${escapeHtml(preview || '(empty message)')}</div>
                <div class="log-message-body">${bodyHtml}</div>
            `;
            syncLogMessageToggleText(bubbleWrapper);
        } else {
            bubbleWrapper.innerHTML = bubbleContentHtml;
        }

        if (rawTextForCopy && typeof rawTextForCopy === 'string' && rawTextForCopy.trim()) {
            const copyButton = document.createElement('button');
            copyButton.textContent = 'Copy';
            copyButton.classList.add('copy-btn');
            copyButton.onclick = (e) => {
                e.stopPropagation();
                navigator.clipboard.writeText(rawTextForCopy).then(() => {
                    copyButton.textContent = 'Copied!';
                    setTimeout(() => { copyButton.textContent = 'Copy'; }, 1500);
                }).catch(err => {
                    console.error('Failed to copy text: ', err);
                    showDashboardStatus('Failed to copy text.', 'error');
                });
            };
            bubbleWrapper.appendChild(copyButton);
        }
        return bubbleWrapper;
    }

    function addMessageToGlobalNotificationBubbleLog(eventData) {
        if (!globalNotificationBubbleLog) { console.error("globalNotificationBubbleLog element not found"); return; }

        const { event: eventType, data, timestamp } = eventData;
        let displayThisEvent = false;
        let bubbleContentHtml = "";
        let rawTextForCopy = "";
        const timeStr = formatEventTime(timestamp);
        const timeSuffix = timeStr ? ` @ ${timeStr}` : '';
        let titlePrefix = `<strong>[${eventType.replace(/_/g, ' ').toUpperCase()}${timeSuffix}]</strong>`;
        let bubbleStyle = 'chat-bubble-system'; 

        switch (eventType) {
            case 'orchestrator_status':
            case 'orchestrator_control':
            case 'orchestrator_info':
            case 'stream_status':
            case 'system_message': 
                displayThisEvent = true;
                bubbleContentHtml = `${titlePrefix} ${escapeHtml(data.message || JSON.stringify(data))}`;
                rawTextForCopy = data.message || JSON.stringify(data);
                break;
            case 'tick_event':
                displayThisEvent = true;
                bubbleStyle = 'text-indigo-300 chat-bubble-system'; 
                let tickMsg = `Tick ${data.tick} - Event: ${data.type}`;
                if(data.turn_order) tickMsg += ` | Order: ${data.turn_order.join(', ')}`;
                if(data.next_tick) tickMsg += ` | Next Tick: ${data.next_tick}`;
                if(data.reason) tickMsg += ` | Reason: ${escapeHtml(data.reason)}`;
                if(data.auto_paused !== undefined) tickMsg += ` | Auto-Paused: ${data.auto_paused}`;
                bubbleContentHtml = `${titlePrefix} ${escapeHtml(tickMsg)}`;
                rawTextForCopy = tickMsg;
                break;
            case 'agent_event':
                if (['token_budget_updated', 'session_ended_tokens', 'session_ended_tokens_internal', 'turn_start', 'turn_end', 'session_ended_by_llm_response_handler'].includes(data.type)) {
                    displayThisEvent = true;
                    bubbleStyle = 'text-purple-300 chat-bubble-system';
                    let agentEventMsg = `Agent: <strong>${escapeHtml(data.agent_name)}</strong> (Tick ${data.tick}) - ${data.type.replace(/_/g, ' ')}`;
                    if (data.type === 'token_budget_updated' && data.current_used !== undefined && data.max_budget !== undefined) {
                        agentEventMsg += ` | Budget: ${data.current_used} / ${data.max_budget}`;
                    }
                    if (data.reason) agentEventMsg += ` | Reason: ${escapeHtml(data.reason)}`;
                    bubbleContentHtml = `${titlePrefix} ${agentEventMsg}`;
                    rawTextForCopy = agentEventMsg;
                }
                break;
            case 'llm_event': 
                if (data.token_info) {
                    displayThisEvent = true;
                    bubbleStyle = 'text-sky-300 chat-bubble-system'; 
                    const agentNameHtml = `<strong>${escapeHtml(data.agent_name)}</strong>`;
                    const tickText = escapeHtml(String(data.tick || 'N/A'));
                    let tokenDetails = "";
                    const ti = data.token_info;
                    const formatTokenValue = (value) => (value !== null && value !== undefined ? value : 'N/A');
                    tokenDetails += ` Session Total: ${formatTokenValue(ti.total_tokens_in_session)}.`;
                    tokenDetails += ` Last Prompt: ${formatTokenValue(ti.last_exchange_prompt_tokens)}.`;
                    tokenDetails += ` Last Completion: ${formatTokenValue(ti.last_exchange_completion_tokens)}.`;
                    tokenDetails += ` Cached: ${formatTokenValue(ti.last_exchange_cached_tokens)}.`;
                    tokenDetails += ` Thoughts: ${formatTokenValue(ti.last_exchange_thoughts_tokens)}.`;
                    bubbleContentHtml = `${titlePrefix} Agent: ${agentNameHtml} (Tick ${tickText}) - LLM Token Usage:${escapeHtml(tokenDetails)}`;
                    rawTextForCopy = `Agent: ${data.agent_name} (Tick ${data.tick || 'N/A'}) - LLM Token Usage:${tokenDetails}`;
                }
                break;
            case 'connector_status':
            case 'connector_error':
                displayThisEvent = true;
                bubbleStyle = eventType === 'connector_error' ? 'chat-bubble-error' : 'chat-bubble-system';
                bubbleContentHtml = `${titlePrefix} Agent: ${escapeHtml(data.agent_name)} - ${escapeHtml(data.status || data.message)}`;
                rawTextForCopy = `Connector for ${data.agent_name}: ${data.status || data.message}`;
                break;
            case 'human_assist_event': 
                 if (!['manual_message_to_llm', 'manual_llm_response_received', 'manual_message_send_attempt'].includes(data.type)) {
                    displayThisEvent = true;
                    bubbleStyle = 'text-fuchsia-300 chat-bubble-system';
                    let assistMsg = `Agent: ${escapeHtml(data.agent_name)} - ${data.type.replace(/_/g, ' ')}`;
                    if(data.message) assistMsg += `: ${escapeHtml(data.message)}`;
                    if(data.error) assistMsg += ` ERROR: ${escapeHtml(data.error)}`;
                    bubbleContentHtml = `${titlePrefix} ${assistMsg}`;
                    rawTextForCopy = assistMsg;
                }
                break;
            case 'error':
            case 'stream_error':
                displayThisEvent = true;
                bubbleStyle = 'chat-bubble-error';
                bubbleContentHtml = `${titlePrefix} ${escapeHtml(data.message || JSON.stringify(data))}`;
                rawTextForCopy = `ERROR: ${data.message || JSON.stringify(data)}`;
                break;
            // --- MODIFICATION START: Add case for final_chat_event notifications ---
            case 'final_chat_event':
                 if (data.status === 'success') {
                    displayThisEvent = true;
                    bubbleStyle = 'text-teal-300 chat-bubble-system';
                    let finalChatMsg = `Final chat with Agent: <strong>${escapeHtml(data.agent_name)}</strong> successful.`;
                    if (data.token_info) {
                         const ti = data.token_info;
                         const formatTokenValue = (value) => (value !== null && value !== undefined ? value : 'N/A');
                         finalChatMsg += ` Tokens - Prompt: ${formatTokenValue(ti.last_exchange_prompt_tokens)}, Completion: ${formatTokenValue(ti.last_exchange_completion_tokens)}.`;
                    }
                    bubbleContentHtml = `${titlePrefix} ${finalChatMsg}`;
                    rawTextForCopy = finalChatMsg;
                 } else if (data.status === 'error') {
                    displayThisEvent = true;
                    bubbleStyle = 'chat-bubble-error';
                    bubbleContentHtml = `${titlePrefix} Error in final chat with Agent: <strong>${escapeHtml(data.agent_name)}</strong> - ${escapeHtml(data.error_message || "Unknown error")}`;
                    rawTextForCopy = `Error in final chat with ${data.agent_name}: ${data.error_message}`;
                 }
                break;
            // --- MODIFICATION END ---
            // --- Auto Evaluation Events ---
            case 'auto_eval_status':
            case 'auto_eval_event':
            case 'auto_eval_error':
                displayThisEvent = true;
                if (eventType === 'auto_eval_error') {
                    bubbleStyle = 'chat-bubble-error';
                    bubbleContentHtml = `${titlePrefix} Auto-Eval Error: ${escapeHtml(data.error || data.message)}`;
                    rawTextForCopy = `Auto-Eval Error: ${data.error || data.message}`;
                } else {
                    bubbleStyle = 'text-violet-300 chat-bubble-system';
                    let evalMsg = `Auto-Evaluator: ${data.status || 'event'}`;
                    if (data.agent_name && data.test_id) {
                        evalMsg += ` - Agent: ${escapeHtml(data.agent_name)}, Test: ${data.test_id}`;
                        if (data.result !== undefined) {
                            evalMsg += `, Result: ${data.result ? 'PASS' : 'FAIL'}`;
                        }
                        if (data.comment) {
                            evalMsg += `, Comment: ${escapeHtml(data.comment)}`;
                        }
                    } else if (data.pending_count) {
                        evalMsg += ` - Processing ${data.pending_count} tests`;
                    }
                    if (data.message) {
                        evalMsg += ` - ${escapeHtml(data.message)}`;
                    }
                    bubbleContentHtml = `${titlePrefix} ${evalMsg}`;
                    rawTextForCopy = evalMsg;
                }
                break;
        }

        if (displayThisEvent) {
            const bubbleWrapper = createLogBubble(bubbleContentHtml, rawTextForCopy, bubbleStyle, {
                eventType,
                messageId: `global:${eventType}:${timestamp || Date.now()}:${++logMessageSequence}`,
                sourceType: 'station'
            });
            
            const initialMsg = globalNotificationBubbleLog.querySelector('.initial-log-message');
            if (initialMsg) {
                initialMsg.remove();
            }
            globalNotificationBubbleLog.insertBefore(bubbleWrapper, globalNotificationBubbleLog.firstChild);

            const MAX_GLOBAL_LOG_BUBBLES = 200;
            if (globalNotificationBubbleLog.children.length > MAX_GLOBAL_LOG_BUBBLES) {
                globalNotificationBubbleLog.removeChild(globalNotificationBubbleLog.lastChild);
            }
        }
    }
    
    // --- addMessageToAgentBubbleLog (with consistent thinking block logic from previous step) ---
    function addMessageToAgentBubbleLog(eventData, isHistorical = false, historicalAppendTarget = null) {
        if (!agentSpecificBubbleLog) { console.error("agentSpecificBubbleLog element not found"); return; }

        let { event: eventType, data, timestamp } = eventData; 
        
        // Bug Fix: If historical entry has no type, assign a default based on content
        if (isHistorical && !eventType) {
            if (data.speaker && (data.speaker === 'Station' || data.speaker.startsWith('Station'))) {
                eventType = 'observation';
            } else if (data.text_content || data.content) {
                eventType = 'submission';
            } else {
                eventType = 'historical_log'; // Fallback for unknown historical entries
            }
            eventData.event = eventType; // Persist the change for later logic
        }

        const dialogueFingerprint = liveDialogueFingerprint(eventData);
        if (dialogueFingerprint && renderedAgentDialogueFingerprints.has(dialogueFingerprint)) return;
        
        const agentDialogueEventTypes = [
            'observation', 'submission', 'internal_prompt', 'internal_response', 
            'context_compaction_prompt', 'context_compaction_response',
            'llm_event', 
            'submission_outcome', 'internal_action_event', 'internal_outcome', 'internal_completion',
            'manual_message_to_agent_llm', 'manual_llm_response_to_human', // Effective types after live-event mapping
            'final_message_to_agent', 'final_agent_response_to_human',
            'system_message', 
            'error',
            'agent_event', // Include agent_event for general agent activities
            'connector_status', 'connector_error', // Include connector events
            'human_assist_event', 'final_chat_event', // Include interaction events
            'historical_log' // Include our fallback type
        ];

        let isRelevantDialogueEvent = agentDialogueEventTypes.includes(eventType);

        // If top-level eventType is generic (like llm_event), check inner type
        if (eventType === 'llm_event' && data.type) {
            isRelevantDialogueEvent = [
                'observation',
                'submission',
                'internal_prompt',
                'internal_response',
                'response',
                'context_compaction_prompt',
                'context_compaction_response',
            ].includes(data.type);
        } else if (eventType === 'agent_event') {
            isRelevantDialogueEvent = ['actions_executed', 'submit_error'].includes(data.type);
        }
        
        if (!isRelevantDialogueEvent) {
             if(!isHistorical && eventType !== 'thinking_block' && eventType !== 'thinking_block_internal') {
                // console.warn(`Skipping live event in agent log due to unrecognized type: ${eventType}`, data);
             } else if (isHistorical && eventType !== 'thinking_block' && eventType !== 'thinking_block_internal') {
                console.warn(`Skipping historical event in agent log due to unrecognized type: ${eventType}`, data);
             }
             return; 
        }


        if (agentSpecificBubbleLog.childElementCount === 1 && agentSpecificBubbleLog.firstChild && agentSpecificBubbleLog.firstChild.classList && agentSpecificBubbleLog.firstChild.classList.contains('initial-log-message')) {
            agentSpecificBubbleLog.innerHTML = ''; 
        }
        
        let bubbleContentHtml = "";
        let rawTextForCopy = JSON.stringify(data, null, 2); 
        const timeStr = formatEventTime(timestamp);
        const timeSuffix = timeStr ? ` @ ${timeStr}` : '';
        let titlePrefix = `<strong>[${eventType.replace(/_/g, ' ').toUpperCase()}${timeSuffix}]</strong>`;
        let bubbleStyle = 'chat-bubble-system text-xs';
        let sourceType = 'station';

        let thinkingHtml = "";
        const thinkingContent = normalizeThinkingWhitespace(data.thinking_text || data.historical_thinking_text);
        if (thinkingContent && thinkingContent.trim() !== "") {
            thinkingHtml = renderCollapsibleThinking(thinkingContent);
             rawTextForCopy = `Thinking:\n${thinkingContent}\n\nResponse:\n${(data.text_content || data.content || "")}`;
        }
        

        switch (eventType) {
            case 'system_message': 
                const systemMessageContent = String(data.message || data.content || "");
                titlePrefix = `<strong>[SYSTEM${timeSuffix}] (Agent: ${escapeHtml(data.agent_name || currentSelectedAgentForDialogueView)})</strong>`;
                bubbleContentHtml = `${titlePrefix} <div class="mt-1 text-sm markdown-content-host">${renderMarkdownForDashboard(systemMessageContent)}</div>`;
                rawTextForCopy = systemMessageContent;
                break;
            case 'agent_event': 
                 if (['actions_executed', 'submit_error'].includes(data.type)) {
                    bubbleStyle = 'chat-bubble-system text-purple-300';
                    let agentEventHeader = `<strong>[AGENT EVENT${timeSuffix}] (Agent: ${escapeHtml(data.agent_name)}, Tick ${escapeHtml(String(data.tick))})</strong> - ${escapeHtml(data.type.replace(/_/g, ' '))}`;
                    let agentEventContent = "";
                    if(data.summary && Array.isArray(data.summary)) {
                         agentEventContent = data.summary.map(s => `- ${s}`).join('\n'); 
                         rawTextForCopy = `Agent: ${data.agent_name} (Tick ${data.tick}) - ${data.type.replace('_', ' ')}\nSummary:\n${data.summary.join('\n')}`;
                    } else if (data.error) { 
                        agentEventContent = `ERROR: ${data.error}`; 
                        rawTextForCopy = `Agent Event Error for ${data.agent_name}: ${data.error}`;
                    }
                    bubbleContentHtml = `${agentEventHeader}<div class="mt-1 text-sm markdown-content-host">${renderMarkdownForDashboard(agentEventContent)}</div>`;
                } else { return; } 
                break;
            
            case 'llm_event': 
            case 'observation': 
            case 'submission':  
            case 'internal_prompt': 
            case 'internal_response':
            case 'context_compaction_prompt':
            case 'context_compaction_response':
            case 'manual_message_to_agent_llm':
            case 'manual_llm_response_to_human':
            case 'final_message_to_agent':
            case 'final_agent_response_to_human':
            case 'historical_log':
                const hasFullDialogueContent =
                    (data.text_content !== undefined && data.text_content !== null && String(data.text_content) !== "") ||
                    (data.content !== undefined && data.content !== null && String(data.content) !== "");
                let dialogueMarkdownContent = "N/A (Content missing for Markdown rendering)";
                if (hasFullDialogueContent) {
                    dialogueMarkdownContent = String(data.text_content || data.content);
                } else if (data.content_omitted) {
                    dialogueMarkdownContent = "Content omitted from this live stream. Reload this agent's history to view the saved message.";
                }
                if (!thinkingHtml) { 
                    rawTextForCopy = dialogueMarkdownContent;
                }

                let speaker = data.speaker || "Unknown";
                let direction = data.direction; 

                if (isHistorical || !direction) { 
                    if (
                        ['observation', 'internal_prompt', 'context_compaction_prompt', 'llm_event', 'historical_log'].includes(eventType)
                        && (data.type === 'internal_prompt' || data.type === 'context_compaction_prompt' || speaker === 'Station')
                    ) {
                        speaker = data.speaker || "Station"; direction = 'to_llm';
                    } else if (
                        ['submission', 'internal_response', 'context_compaction_response', 'llm_event', 'historical_log'].includes(eventType)
                        && (data.type === 'internal_response' || data.type === 'context_compaction_response' || data.type === 'response')
                    ) {
                        speaker = data.speaker || (data.agent_name || "Agent"); direction = 'from_llm';
                    } else if (eventType === 'manual_message_to_agent_llm') {
                        speaker = data.speaker || 'HumanAssistant'; direction = 'to_llm';
                    } else if (eventType === 'manual_llm_response_to_human') {
                        speaker = data.speaker || (data.agent_name || "Agent") + " (LLM)"; direction = 'from_llm';
                    } else if (eventType === 'final_message_to_agent') {
                        speaker = data.speaker || 'HumanFinalChat'; direction = 'to_llm';
                    } else if (eventType === 'final_agent_response_to_human') {
                        speaker = data.speaker || `${data.agent_name || 'Agent'} (FinalResponse)`; direction = 'from_llm';
                    } else { // Default historical log direction
                        direction = (speaker === 'Station' || speaker.startsWith('Station')) ? 'to_llm' : 'from_llm';
                    }
                } else if (eventType === 'llm_event') { 
                     speaker = (data.direction === 'to_llm' ? 'Station' : (data.agent_name || 'AgentLLM'));
                     if (data.type === 'internal_prompt') speaker = 'Station (Internal)';
                     if (data.type === 'internal_response') speaker = `${data.agent_name || 'Agent'} (Internal LLM)`;
                     if (data.type === 'context_compaction_prompt') speaker = 'Station (Context Compaction)';
                     if (data.type === 'context_compaction_response') speaker = `${data.agent_name || 'Agent'} (Context Summary)`;
                }

                let headerText = `<strong>[${escapeHtml(speaker)}${timeSuffix}] (Tick ${escapeHtml(String(data.tick || 'N/A'))})</strong>`;
                if (
                    eventType === 'llm_event'
                    && ['internal_prompt', 'context_compaction_prompt', 'context_compaction_response'].includes(data.type)
                ) {
                     headerText += ` - ${escapeHtml(data.type.replace(/_/g, ' '))}`;
                }
                
                bubbleContentHtml = `${headerText}${thinkingHtml}<div class="mt-1 text-sm markdown-content-host">${renderMarkdownForDashboard(dialogueMarkdownContent)}</div>`;
                
                bubbleStyle = (direction === 'to_llm') ? 'chat-bubble-station' : 'chat-bubble-agent';
                sourceType = (direction === 'to_llm') ? 'station' : 'agent';
                break;

            case 'submission_outcome': 
                 bubbleStyle = 'chat-bubble-system';
                 titlePrefix = `<strong>[SUBMISSION OUTCOME${timeSuffix}] (Agent: ${escapeHtml(data.agent_name || 'N/A')}, Tick ${escapeHtml(String(data.tick || 'N/A'))})</strong>`;
                 let outcomeTextForMarkdown = `Actions Executed:\n${(data.actions_executed_summary || data.summary || []).map(s => `- ${s}`).join('\n')}`;
                 if(data.error) outcomeTextForMarkdown += `\nError: ${data.error}`;
                 
                 let outcomeRendered = renderMarkdownForDashboard(outcomeTextForMarkdown);

                 if(data.internal_action_initiated) {
                     const handlerClass = data.internal_action_initiated.handler_class || "Unknown Handler";
                     const initialPromptForSnippet = data.internal_action_initiated.text_content || data.internal_action_initiated.initial_prompt || "N/A";
                     const snippet = initialPromptForSnippet.substring(0, 70) + (initialPromptForSnippet.length > 70 ? "..." : "");
                     outcomeRendered += `<p class="text-xs mt-1 pl-4">Internal Action Started: ${escapeHtml(handlerClass)} - Prompt (snippet): ${escapeHtml(snippet)}</p>`;
                 }
                 bubbleContentHtml = `${titlePrefix} <div class="mt-1 text-sm markdown-content-host">${outcomeRendered}</div>`;
                 rawTextForCopy = outcomeTextForMarkdown + (data.internal_action_initiated ? `\nInternal action snippet: ${data.internal_action_initiated.initial_prompt ? data.internal_action_initiated.initial_prompt.substring(0,70) : 'N/A'}` : '');
                break;
            
            case 'internal_action_event': 
                 bubbleStyle = 'chat-bubble-system text-amber-300'; 
                 titlePrefix = `<strong>[INTERNAL ACTION${timeSuffix}] (Agent: ${escapeHtml(data.agent_name)}, Tick ${escapeHtml(String(data.tick))})</strong>`;
                 let internalStatusMsg = `Status: ${escapeHtml(data.status.toUpperCase())}`;
                 if(data.handler) internalStatusMsg += ` | Handler: ${escapeHtml(data.handler)}. `;
                 
                 let internalContentDetailMarkdown = "";
                 let liveInternalThinkingHtml = "";
                 const normalizedInternalThinking = normalizeThinkingWhitespace(data.thinking_text);
                 if (!isHistorical && normalizedInternalThinking) {
                    liveInternalThinkingHtml = renderCollapsibleThinking(
                        normalizedInternalThinking,
                        'Agent Thinking (Internal)'
                    );
                 }

                 if(data.status === 'start' && data.text_content) { 
                     internalContentDetailMarkdown = data.text_content; 
                     rawTextForCopy = `Internal Action Start for ${data.agent_name}. Handler: ${data.handler}. Initial Prompt: ${data.text_content}`;
                 } else { 
                    let plainTextDetails = "";
                    if(data.message) plainTextDetails += data.message;
                    if(data.updates && Array.isArray(data.updates)) plainTextDetails += ` | Delta updates: ${data.updates.join(', ')}. `;
                    if(data.log && Array.isArray(data.log)) plainTextDetails += ` | Step log: ${data.log.join('; ')}. `;
                    internalContentDetailMarkdown = plainTextDetails; 
                    rawTextForCopy = plainTextDetails || JSON.stringify(data); 
                 }
                 bubbleContentHtml = `${titlePrefix} ${escapeHtml(internalStatusMsg)} ${liveInternalThinkingHtml || thinkingHtml} <div class="mt-1 text-sm markdown-content-host">${renderMarkdownForDashboard(internalContentDetailMarkdown)}</div>`;
                break;
            
            case 'internal_outcome': 
            case 'internal_completion': 
                let internalOutcomeMarkdown = String(data.next_prompt || data.completion_message || "N/A");
                rawTextForCopy = internalOutcomeMarkdown;
                bubbleStyle = 'chat-bubble-station'; 
                let internalOutcomeHeader = `<strong>[STATION (Internal)${timeSuffix}] (Agent: ${escapeHtml(data.agent_name)}, Tick ${escapeHtml(String(data.tick))})</strong>`;
                if (data.handler) internalOutcomeHeader += ` - Handler: ${escapeHtml(data.handler)}`;
                
                let internalOutcomeBodyRendered = `${thinkingHtml}<div class="mt-1 text-sm markdown-content-host">${renderMarkdownForDashboard(internalOutcomeMarkdown)}</div>`;
                
                if (data.actions_executed_in_step && data.actions_executed_in_step.length > 0) {
                    const actionsListMarkdown = data.actions_executed_in_step.map(s => `- ${s}`).join('\n'); 
                    internalOutcomeBodyRendered += `<div class="text-xs pl-4 mt-1 markdown-content-host"><strong>Step Log:</strong>${renderMarkdownForDashboard(actionsListMarkdown)}</div>`;
                }
                bubbleContentHtml = `${internalOutcomeHeader}${internalOutcomeBodyRendered}`;
                break;
            case 'error': 
                bubbleStyle = 'chat-bubble-error';
                let errorMessageContent = String(data.message || data.error || JSON.stringify(data));
                titlePrefix = `<strong>[ERROR${timeSuffix}] (Agent: ${escapeHtml(data.agent_name || currentSelectedAgentForDialogueView)})</strong>`;
                bubbleContentHtml = `${titlePrefix} <div class="mt-1 text-sm markdown-content-host">${renderMarkdownForDashboard(errorMessageContent)}</div>`;
                rawTextForCopy = `ERROR: ${errorMessageContent}`;
                break;
            default: 
                console.warn(`Unhandled eventType in addMessageToAgentBubbleLog: ${eventType}`, eventData);
                return; 
        }
        
        const bubbleWrapper = createLogBubble(bubbleContentHtml, rawTextForCopy, bubbleStyle, {
            eventType,
            messageId: `agent:${currentSelectedAgentForDialogueView}:${eventType}:${timestamp || Date.now()}:${++logMessageSequence}`,
            sourceType,
            historyKey: eventData.historyKey
        });
        if (data.tick !== undefined && data.tick !== null) {
            bubbleWrapper.dataset.dialogueTick = String(data.tick);
        }
        if (dialogueFingerprint) {
            renderedAgentDialogueFingerprints.add(dialogueFingerprint);
            bubbleWrapper.dataset.dialogueFingerprint = dialogueFingerprint;
        }
        
        const appendTarget = (isHistorical && historicalAppendTarget) ? historicalAppendTarget : agentSpecificBubbleLog;
        appendTarget.appendChild(bubbleWrapper);

        if (isHistorical && appendTarget === agentSpecificBubbleLog && agentSpecificBubbleLog.lastChild === bubbleWrapper) {
            agentSpecificBubbleLog.scrollTop = agentSpecificBubbleLog.scrollHeight;
        } else if (!isHistorical) {
            if (agentSpecificBubbleLog.scrollHeight - agentSpecificBubbleLog.scrollTop < agentSpecificBubbleLog.clientHeight + 350) {
                agentSpecificBubbleLog.scrollTop = agentSpecificBubbleLog.scrollHeight;
            }
        }
    }

    async function fetchApi(endpoint, method = 'GET', body = null) {
        // Use endpoint directly (no path prefix needed)
        const fullEndpoint = endpoint;
        
        const options = { method, headers: { 'Content-Type': 'application/json', } };
        if (body) options.body = JSON.stringify(body);
        try {
            const response = await fetch(fullEndpoint, options);
            if (!response.ok) {
                let errorData;
                try { errorData = await response.json(); } 
                catch (e) { errorData = { message: `HTTP error! Status: ${response.status} ${response.statusText}` }; }
                throw new Error(errorData.message || errorData.error || `Request failed: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`API call to ${endpoint} failed:`, error);
            showDashboardStatus(`Error: ${error.message || 'Network request failed or non-JSON response.'}`, 'error');
            throw error; 
        }
    }

    function applyMultistartPreviewReadOnlyState() {
        if (!isMultistartPreview) return;
        const lockedIds = [
            'start-loop-button',
            'pause-orchestrator-button',
            'resume-orchestrator-button',
            'stop-orchestrator-button',
            'create-api-agent-modal-button',
            'end-api-agent-session-button',
            'resolve-human-intervention-button',
            'open-temporal-chat-modal-button',
            'open-direct-message-modal-button',
            'open-web-archive-surveys-modal-button',
            'open-web-archive-survey-request-button',
            'open-speak-common-room-modal-button',
            'open-send-system-message-modal-button',
            'update-station-config-button',
            'open-api-runtime-config-modal-button',
            'create-backup-button',
            'copy-system-prompt-button',
            'task-spec-edit-tab',
            'save-task-spec-button',
        ];
        lockedIds.forEach((id) => {
            const element = document.getElementById(id);
            if (!element) return;
            element.disabled = true;
            element.title = 'Unavailable in the read-only Seed 1 multistart preview.';
        });
        if (resolveHumanInterventionButton) resolveHumanInterventionButton.classList.add('hidden');
        if (taskSpecRawEditor) taskSpecRawEditor.readOnly = true;
        if (startLoopButton) startLoopButton.textContent = 'Managed by Multistart';
    }

    function updateMultistartPreviewBanner(status) {
        if (!isMultistartPreview) return;
        const detail = document.getElementById('multistart-preview-detail');
        const multistart = status?.multistart || {};
        const branch = multistart.branch || {};
        if (!detail) return;
        const seedStatus = branch.status || status?.branch_status || 'pending';
        const currentTick = branch.current_tick ?? status?.current_tick ?? '-';
        const targetTick = branch.target_tick ?? status?.target_tick ?? '-';
        const completed = multistart.completed_count ?? 0;
        const total = multistart.seed_count ?? '?';
        detail.textContent = ` Seed 1 ${seedStatus}, tick ${currentTick} of ${targetTick}. ` +
            `${completed}/${total} branches completed. This branch may not be selected.`;
    }

    function multistartBranchStatusLabel(rawStatus) {
        const normalized = String(rawStatus || 'pending').trim().toLowerCase();
        const labels = {
            copy_pending: 'Waiting for Copy',
            copying: 'Copying Branch Data',
            copy_failed: 'Copy Retry Pending',
            pending: 'Waiting to Launch',
            running: 'Running',
            paused: 'Paused',
            interviewing: 'Interviewing Agents',
            waiting_quiescent: 'Draining Background Jobs',
            completed: 'Completed',
            failed: 'Failed',
        };
        return labels[normalized] || String(rawStatus || 'Pending');
    }

    function updateOrchestratorControlButtons(status) {
        if (!status) { console.warn("updateOrchestratorControlButtons called with null status"); return; }
        orchestratorState = status; 

        if (status.read_only || isMultistartPreview) {
            applyMultistartPreviewReadOnlyState();
            return;
        }

        const isPrepared = status.is_prepared;
        const isRunning = status.is_running;
        const isPaused = status.is_paused;
        const isWaiting = status.is_waiting;
        const pauseRequested = status.pause_requested;

        // Handle the Pause/Cancel Pause button logic
        if (pauseOrchestratorButton) {
            if (pauseRequested && isRunning && !isPaused) {
                // A pause has been requested but not yet activated
                pauseOrchestratorButton.textContent = 'Cancel Pause';
                pauseOrchestratorButton.classList.remove('bg-amber-600', 'hover:bg-amber-500');
                pauseOrchestratorButton.classList.add('bg-sky-600', 'hover:bg-sky-500');
                pauseOrchestratorButton.disabled = false;
            } else {
                // Normal state
                pauseOrchestratorButton.textContent = 'Pause';
                pauseOrchestratorButton.classList.remove('bg-sky-600', 'hover:bg-sky-500');
                pauseOrchestratorButton.classList.add('bg-amber-600', 'hover:bg-amber-500');
                pauseOrchestratorButton.disabled = !isRunning || isPaused;
            }
        }

        if(startLoopButton) startLoopButton.disabled = !isPrepared || isRunning;
        if(resumeOrchestratorButton) resumeOrchestratorButton.disabled = !isRunning || (!isPaused && !isWaiting);
        if(stopOrchestratorButton) stopOrchestratorButton.disabled = !isRunning && !isPrepared;
        
        // Create agent can work anytime since it doesn't conflict with running orchestrator
        if(createApiAgentModalButton) createApiAgentModalButton.disabled = false;

        if (openTemporalChatModalButton) {
            openTemporalChatModalButton.disabled = false;
        }
        
        // End agent session can be requested at any time, unless the agent's session has already ended.
        if (endApiAgentSessionButton) {
            const selectedAgentName = agentSelectorDashboard ? agentSelectorDashboard.value : "all";
            const selectedAgentData = fullAgentListCache.find(a => a.name === selectedAgentName);
            let isEnded = false;
            let endRequested = false;
            if (selectedAgentData) {
                isEnded = selectedAgentData.status.startsWith("Session Ended") || selectedAgentData.status.startsWith("Ascended");
                endRequested = selectedAgentData.session_end_requested;
            }

            if (endRequested) {
                endApiAgentSessionButton.disabled = true;
                endApiAgentSessionButton.textContent = "End Requested";
            } else {
                endApiAgentSessionButton.disabled = (selectedAgentName === "all") || isEnded;
                endApiAgentSessionButton.textContent = "End Agent";
            }
        }
        
        // Update Intervene tool. Target selection now happens inside the modal.
        if (openDirectMessageModalButton) {
            openDirectMessageModalButton.disabled = isDirectMessageInProgress || buildDirectMessageTargetOptions().length === 0;
        }

        // Update Resolve Request button visibility (only show when selected agent has pending request)
        if (resolveHumanInterventionButton) {
            const selectedAgentName = agentSelectorDashboard ? agentSelectorDashboard.value : "all";
            const isSpecificAgentSelected = selectedAgentName && selectedAgentName !== "all";
            const selectedAgentHasPendingRequest = isSpecificAgentSelected && 
                status.agents_awaiting_human && 
                status.agents_awaiting_human.includes(selectedAgentName);
            
            resolveHumanInterventionButton.classList.toggle('hidden', !selectedAgentHasPendingRequest);
            // No need to check orchestrator state - button can be used even when running
            resolveHumanInterventionButton.disabled = false;
        }
    }

    async function getStationVersion() {
        try {
            const data = await fetchApi('/api/station/version');
            if (data.success && data.version) {
                if (stationVersionDashboard) {
                    stationVersionDashboard.textContent = data.version;
                }
            }
        } catch (error) {
            console.error('Error fetching station version:', error);
            if (stationVersionDashboard) {
                stationVersionDashboard.textContent = 'Error';
            }
        }
    }

    let stationStatisticsRequest = null;
    async function updateStationStatistics() {
        if (stationStatisticsRequest) return stationStatisticsRequest;
        stationStatisticsRequest = (async () => {
        try {
            const data = await fetchApi('/api/station/statistics');
            if (data.success && data.statistics) {
                const stats = data.statistics;

                const formatJobLine = (exp) => {
                    if (exp.job_type === 'external_report') {
                        const reportId = exp.id ?? exp.evaluation_id ?? '?';
                        const author = exp.author || exp.agent_name || 'Unknown';
                        const status = exp.status || 'unknown';
                        return `External #${reportId}: ${exp.title || 'Untitled'} (${author}, ${status}, external surveyor)`;
                    }
                    const elapsedMin = Math.floor((exp.elapsed_seconds || 0) / 60);
                    const elapsedSec = (exp.elapsed_seconds || 0) % 60;
                    const status = exp.status || 'unknown';
                    const executionSource = (exp.job_type === 'archive_survey')
                        ? 'archive surveyor'
                        : exp.system_baseline
                        ? 'system baseline'
                        : ((exp.execution_source || 'coder') === 'direct' ? 'direct evaluator' : 'coder');
                    const idLabel = exp.job_type === 'archive_survey'
                        ? exp.evaluation_id
                        : `Eval ${exp.evaluation_id}`;
                    return `${idLabel}: ${exp.title} (${exp.agent_name}, ${status}, ${executionSource}, ${elapsedMin}m${elapsedSec}s)`;
                };
                // Update pending human requests
                const pendingRequests = stats.pending_human_requests;
                const requestCount = pendingRequests.request_ids.length;
                const countElement = document.getElementById('pending-requests-count');
                const detailsElement = document.getElementById('pending-requests-details');
                
                if (countElement) {
                    countElement.textContent = requestCount > 0 ? requestCount : '0';
                }
                
                if (detailsElement && requestCount > 0) {
                    const requestIds = pendingRequests.request_ids.join(', ');
                    const agents = pendingRequests.agents.join(', ');
                    detailsElement.innerHTML = `IDs: ${requestIds}<br>Agents: ${agents}`;
                    detailsElement.classList.remove('hidden');
                    setTooltipIfPresent('pending-requests-metric-card', `IDs: ${requestIds}\nAgents: ${agents}`);
                } else if (detailsElement) {
                    detailsElement.classList.add('hidden');
                    setTooltipIfPresent('pending-requests-metric-card', '');
                }
                
                // Update jobs
                const runningCount = stats.running_jobs_count ?? stats.running_experiments_count ?? 0;
                const runningJobs = stats.running_jobs || stats.running_experiments || [];
                const queuedCount = stats.queued_jobs_count ?? stats.queued_experiments_count ?? 0;
                const queuedJobs = stats.queued_jobs || stats.queued_experiments || [];
                const totalJobsCount = runningCount + queuedCount;
                const countElementRunning = document.getElementById('running-experiments-count');
                const detailsElementRunning = document.getElementById('running-experiments-details');
                
                if (countElementRunning) {
                    countElementRunning.textContent = totalJobsCount > 0 ? totalJobsCount : '0';
                }
                
                if (detailsElementRunning && totalJobsCount > 0) {
                    const detailSections = [];
                    if (runningJobs.length > 0) {
                        detailSections.push(`Running: ${runningJobs.map(formatJobLine).join('<br>')}`);
                    }
                    if (queuedJobs.length > 0) {
                        detailSections.push(`Queued: ${queuedJobs.map(formatJobLine).join('<br>')}`);
                    }
                    const detailsHtml = detailSections.join('<br>');
                    detailsElementRunning.innerHTML = detailsHtml;
                    detailsElementRunning.classList.remove('hidden');
                    const tooltipSections = [];
                    tooltipSections.push(`Running Jobs (${runningCount})`);
                    tooltipSections.push(runningJobs.length > 0 ? runningJobs.map(formatJobLine).join('\n') : 'None');
                    tooltipSections.push(`Queued Jobs (${queuedCount})`);
                    tooltipSections.push(queuedJobs.length > 0 ? queuedJobs.map(formatJobLine).join('\n') : 'None');
                    setTooltipIfPresent('running-experiments-metric-card', tooltipSections.join('\n'));
                } else if (detailsElementRunning) {
                    detailsElementRunning.classList.add('hidden');
                    setTooltipIfPresent('running-experiments-metric-card', '');
                }

                updateTickTimingFromStats(stats.tick_timing);
                
                // Update top research submission
                const topSubmission = stats.top_research_submission;
                const scoreElement = document.getElementById('top-research-score');
                const detailsElementResearch = document.getElementById('top-research-details');
                
                // Check for both undefined and null to handle no-score mode
                if (scoreElement && topSubmission && topSubmission !== null) {
                    scoreElement.textContent = formatTopResearchScore(topSubmission.score);
                    const currentTick = Number(stats.current_tick);
                    const topSubmittedTick = Number(topSubmission.submitted_tick);
                    const topScoreTickAge = Number.isFinite(currentTick) && Number.isFinite(topSubmittedTick)
                        ? Math.max(0, currentTick - topSubmittedTick)
                        : null;
                    const breakthroughAgeRaw = stats.ticks_since_last_breakthrough;
                    const breakthroughAgeNumber = Number(breakthroughAgeRaw);
                    const breakthroughTickAge = breakthroughAgeRaw !== null
                        && breakthroughAgeRaw !== undefined
                        && Number.isFinite(breakthroughAgeNumber)
                        ? Math.max(0, breakthroughAgeNumber)
                        : null;
                    const topScoreAgeColor = getTopScoreAgeColor(topScoreTickAge);
                    if (topScoreAgeColor) {
                        scoreElement.style.setProperty('--top-score-age-color', topScoreAgeColor);
                        scoreElement.style.color = topScoreAgeColor;
                        scoreElement.classList.add('top-score-age-colored');
                    } else {
                        scoreElement.style.removeProperty('--top-score-age-color');
                        scoreElement.style.color = '';
                        scoreElement.classList.remove('top-score-age-colored');
                    }
                    if (detailsElementResearch) {
                        detailsElementResearch.innerHTML = 
                            `ID: ${topSubmission.evaluation_id}<br>` +
                            `Title: ${topSubmission.title}<br>` +
                            `Agent: ${topSubmission.agent_name}<br>` +
                            `Tick: ${topSubmission.submitted_tick}`;
                        detailsElementResearch.classList.remove('hidden');
                    }
                    setTooltipIfPresent(
                        'top-research-metric-card',
                        `Score: ${topSubmission.score}\n` +
                        `ID: ${topSubmission.evaluation_id}\n` +
                        `Title: ${topSubmission.title || ''}\n` +
                        `Agent: ${topSubmission.agent_name || ''}\n` +
                        `Tick: ${topSubmission.submitted_tick ?? ''}\n` +
                        `Ticks since last top score: ${topScoreTickAge === null ? 'N/A' : topScoreTickAge}\n` +
                        `Ticks since last breakthrough: ${breakthroughTickAge === null ? 'N/A' : breakthroughTickAge}`
                    );
                } else if (scoreElement) {
                    scoreElement.textContent = 'N/A';
                    scoreElement.style.removeProperty('--top-score-age-color');
                    scoreElement.style.color = '';
                    scoreElement.classList.remove('top-score-age-colored');
                    if (detailsElementResearch) {
                        detailsElementResearch.classList.add('hidden');
                    }
                    setTooltipIfPresent('top-research-metric-card', 'No scored research submission is available.');
                }
            }
        } catch (error) {
            console.error('Error fetching station statistics:', error);
        } finally {
            stationStatisticsRequest = null;
        }
        })();
        return stationStatisticsRequest;
    }

    let orchestratorStatusRequest = null;
    async function getOrchestratorStatus() {
        if (orchestratorStatusRequest) return orchestratorStatusRequest;
        orchestratorStatusRequest = (async () => {
        try {
            const data = await fetchApi('/api/orchestrator/status');
            if (data.success && data.status) {
                const status = data.status;
                const previewBranch = status.multistart?.branch || {};
                const previewBranchStatus = previewBranch.status || status.branch_status || 'pending';
                if(orchestratorStatusDot) orchestratorStatusDot.className = 'status-dot'; 
                if (status.read_only) {
                    const normalizedBranchStatus = String(previewBranchStatus).toLowerCase();
                    if (orchestratorStatusDisplay) {
                        orchestratorStatusDisplay.textContent = multistartBranchStatusLabel(previewBranchStatus);
                    }
                    if (orchestratorStatusDot) {
                        if (normalizedBranchStatus === 'failed') orchestratorStatusDot.classList.add('status-error');
                        else if (normalizedBranchStatus === 'paused') orchestratorStatusDot.classList.add('status-paused');
                        else if (['interviewing', 'waiting_quiescent', 'pending'].includes(normalizedBranchStatus)) orchestratorStatusDot.classList.add('status-waiting');
                        else if (normalizedBranchStatus === 'completed') orchestratorStatusDot.classList.add('status-prepared');
                        else orchestratorStatusDot.classList.add('status-running');
                    }
                } else if (status.is_running) {
                    if (status.is_waiting) {
                        // Waiting state (running but waiting for conditions to resolve)
                        if(orchestratorStatusDisplay) orchestratorStatusDisplay.textContent = 'Waiting';
                        if(orchestratorStatusDot) orchestratorStatusDot.classList.add('status-waiting');
                    } else if (status.is_paused) {
                        if(orchestratorStatusDisplay) orchestratorStatusDisplay.textContent = 'Paused';
                        if(orchestratorStatusDot) orchestratorStatusDot.classList.add('status-paused');
                    } else {
                        if(orchestratorStatusDisplay) orchestratorStatusDisplay.textContent = 'Running';
                        if(orchestratorStatusDot) orchestratorStatusDot.classList.add('status-running');
                    }
                } else if (status.is_prepared) {
                    if(orchestratorStatusDisplay) orchestratorStatusDisplay.textContent = 'Prepared (Idle)';
                    if(orchestratorStatusDot) orchestratorStatusDot.classList.add('status-prepared');
                }
                else {
                    if(orchestratorStatusDisplay) orchestratorStatusDisplay.textContent = 'Idle / Stopped';
                    if(orchestratorStatusDot) orchestratorStatusDot.classList.add('status-idle');
                }
                
                // Update pause/wait reason display
                let reasonText = "";
                if (status.read_only) {
                    reasonText = previewBranch.note || 'Read-only Seed 1 candidate; this branch may not be selected.';
                } else if (status.is_waiting) {
                    reasonText = Object.values(status.waiting_reasons || {}).join(', ');
                } else if (status.is_paused) {
                    reasonText = status.pause_reason || "Manual Pause";
                }
                if(orchestratorPauseReasonDisplay) orchestratorPauseReasonDisplay.textContent = reasonText;
                
                // Update next agent display
                if(orchestratorNextAgentDisplay) {
                    const parallelStatusText = formatParallelTickStatus(status.parallel_tick_status);
                    if (status.read_only) {
                        const currentTick = previewBranch.current_tick ?? status.current_tick ?? '-';
                        const targetTick = previewBranch.target_tick ?? status.target_tick ?? '-';
                        orchestratorNextAgentDisplay.textContent =
                            `Seed 1: ${multistartBranchStatusLabel(previewBranchStatus)} · tick ${currentTick}/${targetTick}`;
                    } else if (status.is_running && !status.is_paused && parallelStatusText) {
                        orchestratorNextAgentDisplay.textContent = parallelStatusText;
                    } else if (status.is_running && !status.is_paused && !status.is_waiting) {
                        orchestratorNextAgentDisplay.textContent = "Preparing tick status...";
                    } else if (status.is_waiting) {
                        orchestratorNextAgentDisplay.textContent = "Waiting for conditions to resolve...";
                    } else if (status.is_paused) {
                        orchestratorNextAgentDisplay.textContent = "Paused...";
                    } else if (status.is_prepared) {
                        orchestratorNextAgentDisplay.textContent = "Ready to run.";
                    } else {
                        orchestratorNextAgentDisplay.textContent = "Not running or no agents.";
                    }
                }
                
                if(stationTickDashboard) stationTickDashboard.textContent = status.current_tick !== -1 ? status.current_tick : "N/A";

                // Update station status if it changed
                if (status.station_status && status.station_status !== cachedStationStatus) {
                    cachedStationStatus = status.station_status;
                    updateStationStatusInHeader(cachedStationStatus);
                }

                updateOrchestratorControlButtons(status);
                updateMultistartPreviewBanner(status);
            } else {
                showDashboardStatus(data.error || 'Failed to get orchestrator status.', 'error');
                if(orchestratorStatusDisplay) orchestratorStatusDisplay.textContent = "Error";
                if(orchestratorStatusDot) orchestratorStatusDot.className = 'status-dot status-error';
                if (isMultistartPreview) {
                    window.setTimeout(() => window.location.replace('/multistart'), 1000);
                }
            }
            return data;
        } catch (error) { 
            if(orchestratorStatusDisplay) orchestratorStatusDisplay.textContent = "Offline";
            if(orchestratorStatusDot) orchestratorStatusDot.className = 'status-dot status-error';
            return null;
        } finally {
            orchestratorStatusRequest = null;
        }
        })();
        return orchestratorStatusRequest;
    }

    async function fetchAgentsForDashboard(orchestratorStatusData = null) {
         try {
            const data = await fetchApi('/api/agents');
            const statusData = orchestratorStatusData || await fetchApi('/api/orchestrator/status');
            
            if (data.success && agentSelectorDashboard) {
                fullAgentListCache = data.agents;
                const currentTurnOrder = statusData?.success ? statusData.status.turn_order : [];
                isRefreshingAgentSelector = true;
                try {
                    const currentVal = agentSelectorDashboard.value;
                    agentSelectorDashboard.innerHTML = '<option value="all">Global Notifications</option><option value="Reviewer">Reviewer</option>';

                    let activeAgentsInOrder = [];
                    let otherAgents = [];

                    currentTurnOrder.forEach(nameInOrder => {
                        const agentInfo = fullAgentListCache.find(a => a.name === nameInOrder);
                        if (agentInfo) activeAgentsInOrder.push(agentInfo);
                    });

                    fullAgentListCache.forEach(agentInfo => {
                        if (!currentTurnOrder.includes(agentInfo.name)) otherAgents.push(agentInfo);
                    });

                    otherAgents.sort((a, b) => {
                        if (a.status.startsWith("Ascended") && !b.status.startsWith("Ascended")) return 1;
                        if (!a.status.startsWith("Ascended") && b.status.startsWith("Ascended")) return -1;
                        if (a.status.startsWith("Session Ended") && !b.status.startsWith("Session Ended")) return 1;
                        if (!a.status.startsWith("Session Ended") && b.status.startsWith("Session Ended")) return -1;
                        return a.name.localeCompare(b.name);
                    });

                    activeAgentsInOrder.forEach(agent => {
                        const option = document.createElement('option');
                        option.value = agent.name;
                        option.textContent = formatAgentDisplayName(agent);
                        agentSelectorDashboard.appendChild(option);
                    });

                    if (activeAgentsInOrder.length > 0 && otherAgents.length > 0) {
                        const separator = document.createElement('option');
                        separator.disabled = true;
                        separator.textContent = '--- Other Agents ---';
                        agentSelectorDashboard.appendChild(separator);
                    }

                    otherAgents.forEach(agent => {
                        const option = document.createElement('option');
                        option.value = agent.name;
                        option.textContent = formatAgentDisplayName(agent);
                        agentSelectorDashboard.appendChild(option);
                    });

                    if (Array.from(agentSelectorDashboard.options).some(opt => opt.value === currentVal)) {
                        agentSelectorDashboard.value = currentVal;
                    } else {
                        agentSelectorDashboard.value = "all";
                    }
                } finally {
                    isRefreshingAgentSelector = false;
                }
                updateCopySystemPromptButtonState();
            }
        } catch (error) { console.error("Error fetching and populating agents:", error); }
    }

    function buildHistoryApiUrl(agentName, options = {}) {
        const params = new URLSearchParams();
        if (options.full) {
            params.set('full', 'true');
        } else {
            params.set('window', options.window || historyWindowMode);
            params.set('ticks', String(options.ticks || HISTORY_WINDOW_TICKS));
        }
        if (options.modifiedAtNs) params.set('modified_at_ns', String(options.modifiedAtNs));
        const query = params.toString();
        return `/api/agent_dialogue_history/${encodeURIComponent(agentName)}${query ? `?${query}` : ''}`;
    }

    function captureAgentLogScrollAnchor() {
        if (!agentSpecificBubbleLog) return null;
        const containerRect = agentSpecificBubbleLog.getBoundingClientRect();
        const bubbles = Array.from(agentSpecificBubbleLog.querySelectorAll('.chat-bubble[data-history-key]'));
        for (const bubble of bubbles) {
            const rect = bubble.getBoundingClientRect();
            if (rect.bottom >= containerRect.top) {
                return {
                    historyKey: bubble.dataset.historyKey,
                    offsetTop: rect.top - containerRect.top,
                    fallbackScrollTop: agentSpecificBubbleLog.scrollTop,
                };
            }
        }
        return { historyKey: null, offsetTop: 0, fallbackScrollTop: agentSpecificBubbleLog.scrollTop };
    }

    function restoreAgentLogScrollAnchor(anchor) {
        if (!agentSpecificBubbleLog || !anchor) return;
        if (!anchor.historyKey) {
            agentSpecificBubbleLog.scrollTop = anchor.fallbackScrollTop || 0;
            return;
        }
        const escapedKey = (window.CSS && typeof window.CSS.escape === 'function')
            ? window.CSS.escape(anchor.historyKey)
            : String(anchor.historyKey).replace(/["\\]/g, '\\$&');
        const bubble = agentSpecificBubbleLog.querySelector(`.chat-bubble[data-history-key="${escapedKey}"]`);
        if (!bubble) {
            agentSpecificBubbleLog.scrollTop = anchor.fallbackScrollTop || 0;
            return;
        }
        const containerRect = agentSpecificBubbleLog.getBoundingClientRect();
        const rect = bubble.getBoundingClientRect();
        agentSpecificBubbleLog.scrollTop += (rect.top - containerRect.top) - anchor.offsetTop;
    }

    function captureCurrentSourceCollapseState() {
        return {
            agent: isCurrentLogSourceCollapsed('agent'),
            station: isCurrentLogSourceCollapsed('station'),
        };
    }

    function restoreCurrentSourceCollapseState(state) {
        if (!state) return;
        setCurrentLogSourceCollapsed('agent', !!state.agent);
        setCurrentLogSourceCollapsed('station', !!state.station);
    }

    function getHistoryEntryKey(logEntry) {
        const stableParts = [
            logEntry.tick ?? 'no_tick',
            logEntry.type || 'historical_log',
            logEntry.speaker || '',
            logEntry.interaction_id || '',
            logEntry.internal_step ?? '',
            String(logEntry.content || logEntry.text_content || logEntry.next_prompt || logEntry.completion_message || '').slice(0, 80),
        ];
        return stableParts.join('|');
    }

    function isHistoricalThinkingEntry(entryType) {
        return entryType === 'thinking_block' ||
            entryType === 'thinking_block_internal' ||
            entryType === 'thinking_block_context_compaction' ||
            entryType === 'manual_llm_thinking_for_human';
    }

    function isStationSpeaker(speaker) {
        return !!speaker && (speaker === 'Station' || String(speaker).startsWith('Station'));
    }

    function isHistoricalThinkingTarget(logEntry, pendingThinking) {
        if (!pendingThinking) return false;
        const entryType = logEntry.type || 'historical_log';
        const targetTypes = new Set([
            'submission',
            'internal_response',
            'context_compaction_response',
            'manual_llm_response_to_human',
            'final_agent_response_to_human',
        ]);
        if (!targetTypes.has(entryType)) return false;
        if (isStationSpeaker(logEntry.speaker)) return false;

        if (pendingThinking.tick !== undefined && pendingThinking.tick !== null &&
            logEntry.tick !== undefined && logEntry.tick !== null &&
            String(pendingThinking.tick) !== String(logEntry.tick)) {
            return false;
        }

        const pendingAgent = pendingThinking.agent_name || currentSelectedAgentForDialogueView;
        const entryAgent = logEntry.agent_name || currentSelectedAgentForDialogueView;
        return !pendingAgent || !entryAgent || pendingAgent === entryAgent;
    }

    function historyRenderWasSuperseded(options = {}) {
        return (options.requestId !== undefined && options.requestId !== historyLoadSequence) ||
            (options.agentName && options.agentName !== currentSelectedAgentForDialogueView);
    }

    function latestHistoryTimestamp(historyEntries) {
        return historyEntries.reduce((latest, entry) => {
            const timestamp = Number(entry && entry.timestamp);
            return Number.isFinite(timestamp) ? Math.max(latest, timestamp) : latest;
        }, Number.NEGATIVE_INFINITY);
    }

    function multistartPreviewDialogueSignature(historyEntries) {
        if (!Array.isArray(historyEntries) || historyEntries.length === 0) return 'empty';
        const tail = historyEntries.slice(-4).map(entry => {
            const content = String(
                entry?.content || entry?.text_content || entry?.next_prompt || entry?.completion_message || ''
            );
            return `${getHistoryEntryKey(entry || {})}|${content.length}|${entry?.timestamp ?? ''}`;
        });
        return `${historyEntries.length}|${tail.join('||')}`;
    }

    function dialogueEntryFingerprint(tick, type, internalStep, content) {
        if (content === undefined || content === null) return null;
        return JSON.stringify([
            tick ?? '',
            type || '',
            internalStep ?? '',
            String(content),
        ]);
    }

    function liveDialogueFingerprint(eventData) {
        const data = eventData && typeof eventData.data === 'object' ? eventData.data : {};
        let type = data.type || eventData?.event || '';
        let content = data.text_content;

        if (eventData?.event === 'llm_event' && type === 'response') {
            type = 'submission';
        } else if (eventData?.event === 'human_assist_event') {
            if (type === 'manual_message_human_part_sent') type = 'manual_message_to_agent_llm';
            if (type === 'manual_llm_response_received') type = 'manual_llm_response_to_human';
        } else if (eventData?.event === 'final_chat_event') {
            if (type === 'human_message_sent') {
                type = 'final_message_to_agent';
                content = data.human_message;
            } else if (type === 'agent_response_received') {
                type = 'final_agent_response_to_human';
                content = data.llm_response;
            }
        }

        return dialogueEntryFingerprint(
            data.tick,
            type,
            data.internal_loop_step,
            content,
        );
    }

    function latestTickAttemptView(historyEntries) {
        const numericTicks = historyEntries
            .map(entry => Number(entry && entry.tick))
            .filter(Number.isFinite);
        if (numericTicks.length === 0) return { entries: historyEntries, pendingObservation: null };

        const latestTickKey = String(Math.max(...numericTicks));
        const observationIndexes = [];
        historyEntries.forEach((entry, index) => {
            const isLatestTick = String(entry && entry.tick) === latestTickKey;
            const isStationObservation = entry?.type === 'observation' && isStationSpeaker(entry?.speaker);
            if (isLatestTick && isStationObservation) observationIndexes.push(index);
        });
        if (observationIndexes.length === 0) return { entries: historyEntries, pendingObservation: null };

        const attempts = observationIndexes.map((startIndex, attemptIndex) => {
            const boundedEnd = observationIndexes[attemptIndex + 1] ?? historyEntries.length;
            const hasAgentResponse = historyEntries
                .slice(startIndex, boundedEnd)
                .some(entry => entry?.type === 'submission');
            return { startIndex, endIndex: boundedEnd, hasAgentResponse };
        });

        const latestAttempt = attempts[attempts.length - 1];
        const selectedAttempt = [...attempts].reverse().find(attempt => attempt.hasAgentResponse) || latestAttempt;
        const pendingObservation = latestAttempt.hasAgentResponse
            ? null
            : historyEntries[latestAttempt.startIndex];
        const entries = historyEntries.filter((entry, index) => {
            if (String(entry && entry.tick) !== latestTickKey) return true;
            return index >= selectedAttempt.startIndex && index < selectedAttempt.endIndex;
        });
        return { entries, pendingObservation };
    }

    function historyObservationToLiveEvent(logEntry, agentName) {
        if (!logEntry) return null;
        return {
            event: 'llm_event',
            data: {
                agent_name: agentName,
                tick: logEntry.tick,
                direction: 'to_llm',
                type: 'observation',
                text_content: logEntry.content || logEntry.text_content || '',
                full_length: String(logEntry.content || logEntry.text_content || '').length,
            },
            timestamp: logEntry.timestamp ?? null,
        };
    }

    function yieldToBrowserRender() {
        return new Promise(resolve => {
            if (document.visibilityState === 'visible' && typeof window.requestAnimationFrame === 'function') {
                window.requestAnimationFrame(() => resolve());
            } else {
                setTimeout(resolve, 0);
            }
        });
    }

    async function renderHistoryEntries(historyToDisplay, options = {}) {
        let pendingHistoricalThinking = null;
        const renderFragment = document.createDocumentFragment();
        let entriesInChunk = 0;
        let renderedTextInChunk = 0;

        for (const logEntry of historyToDisplay) {
            if (historyRenderWasSuperseded(options)) return false;
            const entryType = logEntry.type || 'historical_log';

            if (isHistoricalThinkingEntry(entryType)) {
                pendingHistoricalThinking = {
                    text: String(logEntry.content || ""),
                    tick: logEntry.tick,
                    agent_name: logEntry.agent_name || currentSelectedAgentForDialogueView,
                };
                continue;
            }

            const attachedHistoricalThinking = isHistoricalThinkingTarget(logEntry, pendingHistoricalThinking)
                ? pendingHistoricalThinking.text
                : null;
            pendingHistoricalThinking = null;

            const sseLikeEventData = {
                event: entryType,
                historyKey: getHistoryEntryKey(logEntry),
                data: {
                    agent_name: logEntry.agent_name || currentSelectedAgentForDialogueView,
                    tick: logEntry.tick,
                    speaker: logEntry.speaker,
                    text_content: logEntry.text_content || logEntry.content,
                    content: logEntry.content,
                    actions_executed_summary: logEntry.actions_executed_summary,
                    error: logEntry.error,
                    internal_action_initiated: logEntry.internal_action_initiated,
                    status: logEntry.status,
                    handler: logEntry.handler,
                    direction: logEntry.direction,
                    type: logEntry.type,
                    snippet: logEntry.snippet,
                    full_length: logEntry.full_length,
                    interaction_id: logEntry.interaction_id,
                    next_prompt: logEntry.next_prompt,
                    actions_executed_in_step: logEntry.actions_executed_in_step,
                    completion_message: logEntry.completion_message,
                    token_info: logEntry.token_info,
                    historical_thinking_text: attachedHistoricalThinking,
                    ...(logEntry.data && typeof logEntry.data === 'object' ? logEntry.data : {})
                },
                timestamp: logEntry.timestamp ?? null
            };

            if (logEntry.content && !sseLikeEventData.data.text_content) {
                sseLikeEventData.data.text_content = logEntry.content;
            }
            if (logEntry.next_prompt && entryType === 'internal_outcome' && !sseLikeEventData.data.text_content) {
                sseLikeEventData.data.text_content = logEntry.next_prompt;
            }
            addMessageToAgentBubbleLog(sseLikeEventData, true, renderFragment);
            entriesInChunk += 1;
            renderedTextInChunk += String(
                logEntry.content || logEntry.text_content || logEntry.next_prompt || logEntry.completion_message || ''
            ).length + String(attachedHistoricalThinking || '').length;

            if (entriesInChunk >= HISTORY_RENDER_CHUNK_SIZE ||
                renderedTextInChunk >= HISTORY_RENDER_CHUNK_TEXT_BUDGET) {
                entriesInChunk = 0;
                renderedTextInChunk = 0;
                await yieldToBrowserRender();
            }
        }

        if (historyRenderWasSuperseded(options)) return false;
        if (agentSpecificBubbleLog && options.replaceExisting) {
            agentSpecificBubbleLog.replaceChildren(renderFragment);
        } else if (agentSpecificBubbleLog && renderFragment.childNodes.length > 0) {
            // Commit the completed history in one DOM update so the transcript
            // appears at its final position instead of visibly growing chunk by chunk.
            agentSpecificBubbleLog.appendChild(renderFragment);
        }
        if (agentSpecificBubbleLog && options.stickToBottom) {
            agentSpecificBubbleLog.scrollTop = agentSpecificBubbleLog.scrollHeight;
        }
        return true;
    }

    function cancelActiveHistoryRequest() {
        if (!activeHistoryAbortController) return;
        activeHistoryAbortController.abort();
        activeHistoryAbortController = null;
    }

    async function fetchHistoryWithRetry(apiUrl, requestId, agentName) {
        let lastError = null;
        for (let attempt = 1; attempt <= 2; attempt += 1) {
            if (requestId !== historyLoadSequence) throw new Error('History request superseded.');
            const controller = new AbortController();
            activeHistoryAbortController = controller;
            let timedOut = false;
            const timeoutId = setTimeout(() => {
                timedOut = true;
                controller.abort();
            }, HISTORY_REQUEST_TIMEOUT_MS);

            try {
                const response = await fetch(apiUrl, {
                    method: 'GET',
                    headers: { 'Accept': 'application/json' },
                    signal: controller.signal,
                });
                if (!response.ok) {
                    let message = `History request failed: ${response.status}`;
                    try {
                        const errorData = await response.json();
                        message = errorData.error || errorData.message || message;
                    } catch (_error) {}
                    throw new Error(message);
                }
                return await response.json();
            } catch (error) {
                if (requestId !== historyLoadSequence) throw error;
                lastError = timedOut
                    ? new Error(`History request timed out after ${HISTORY_REQUEST_TIMEOUT_MS / 1000} seconds.`)
                    : error;
                if (attempt < 2) {
                    showDashboardStatus(`History for ${agentName} is taking longer than expected. Retrying once...`, 'info', 0);
                }
            } finally {
                clearTimeout(timeoutId);
                if (activeHistoryAbortController === controller) {
                    activeHistoryAbortController = null;
                }
            }
        }
        throw lastError || new Error('History request failed.');
    }

    async function handleAgentDialogueViewChange(loadFullHistory = false) {
        if (!agentSelectorDashboard || !globalNotificationBubbleLog || !agentSpecificBubbleLog || !logViewTitle) {
            console.error("One or more essential log view DOM elements are missing.");
            return;
        }
        
        const requestId = ++historyLoadSequence;
        cancelActiveHistoryRequest();
        bufferedLiveEventsDuringHistory = [];
        renderedAgentDialogueFingerprints.clear();
        pendingLiveStationObservations.clear();
        let loadedHistoryTimestampCutoff = Number.NEGATIVE_INFINITY;
        const newSelectedAgent = agentSelectorDashboard.value;
        isLoadingHistoryForAgent = (newSelectedAgent !== "all") ? newSelectedAgent : null; 
        currentSelectedAgentForDialogueView = newSelectedAgent;
        updateHistoryWindowSelector();

        showLoadFullHistoryButton(false);

        if (currentSelectedAgentForDialogueView === "all") {
            globalNotificationBubbleLog.style.display = "flex";
            agentSpecificBubbleLog.style.display = "none";
            logViewTitle.textContent = "Global Notifications";
            isLoadingHistoryForAgent = null; 
            hideDashboardStatus();
            updateHistoryWindowSelector();
            showHistoryWindowSelector(true);
        } else {
            globalNotificationBubbleLog.style.display = "none";
            agentSpecificBubbleLog.style.display = "flex"; 
            logViewTitle.textContent = `Dialogue: ${currentSelectedAgentForDialogueView}`;
            
            agentSpecificBubbleLog.innerHTML = '';
            const initialMessageDiv = document.createElement('div');
            initialMessageDiv.classList.add('chat-bubble', 'chat-bubble-system', 'initial-log-message');
            initialMessageDiv.textContent = `Loading history for ${currentSelectedAgentForDialogueView}...`;
            agentSpecificBubbleLog.appendChild(initialMessageDiv);

            showDashboardStatus(`Loading ${historyWindowMode} ${HISTORY_WINDOW_TICKS} tick history for ${currentSelectedAgentForDialogueView}...`, 'info', 0);
            try {
                const apiUrl = buildHistoryApiUrl(currentSelectedAgentForDialogueView, { full: loadFullHistory });
                const apiData = await fetchHistoryWithRetry(apiUrl, requestId, newSelectedAgent);
                
                if (requestId !== historyLoadSequence || currentSelectedAgentForDialogueView !== newSelectedAgent) return;

                if (agentSpecificBubbleLog.firstChild && agentSpecificBubbleLog.firstChild.classList.contains('initial-log-message')) {
                    agentSpecificBubbleLog.innerHTML = ''; 
                }

                let historyToDisplay = [];
                
                if (apiData.success && apiData.history) {
                    if (isMultistartPreview) {
                        multistartPreviewDialogueSignatures.set(
                            newSelectedAgent,
                            multistartPreviewDialogueSignature(apiData.history),
                        );
                        if (apiData.source_modified_at_ns) {
                            multistartPreviewDialogueMtimes.set(newSelectedAgent, apiData.source_modified_at_ns);
                        }
                    }
                    if (loadFullHistory) {
                        fullDialogueHistoryCache[currentSelectedAgentForDialogueView] = apiData.history;
                    } else {
                        delete fullDialogueHistoryCache[currentSelectedAgentForDialogueView];
                    }
                    if (loadFullHistory) {
                        historyToDisplay = apiData.history;
                    } else {
                        const latestTickView = latestTickAttemptView(apiData.history);
                        historyToDisplay = latestTickView.entries;
                        const pendingObservationEvent = historyObservationToLiveEvent(
                            latestTickView.pendingObservation,
                            newSelectedAgent,
                        );
                        if (pendingObservationEvent) {
                            pendingLiveStationObservations.set(
                                String(pendingObservationEvent.data.tick),
                                pendingObservationEvent,
                            );
                        }
                    }
                    loadedHistoryTimestampCutoff = latestHistoryTimestamp(apiData.history);
                    
                    if (apiData.history.length === 0) {
                        addMessageToAgentBubbleLog({event: "system_message", data: {message: `No dialogue history found for ${currentSelectedAgentForDialogueView}. New live messages for this agent will be appended.`}, timestamp: Date.now()/1000}, true);
                    } else {
                        const isPartialWindow = !!(apiData.range && apiData.range.is_partial);
                        if (apiData.is_truncated || isPartialWindow) {
                            showHistoryWindowSelector(false);
                            showLoadFullHistoryButton(true, 'Load Full History');
                            addMessageToAgentBubbleLog({
                                event: "system_message", 
                                data: { message: `Showing ${historyWindowMode} ${HISTORY_WINDOW_TICKS} tick dialogue history. Click 'Load Full History' to load the full history in the background.` },
                                timestamp: Date.now()/1000
                            }, true);
                        }
                        
                        const rendered = await renderHistoryEntries(historyToDisplay, {
                            requestId,
                            agentName: newSelectedAgent,
                            stickToBottom: !loadFullHistory && historyWindowMode === 'recent',
                        });
                        if (!rendered) return;
                        if (!loadFullHistory && historyWindowMode === 'earliest') {
                            agentSpecificBubbleLog.scrollTop = 0;
                        }
                    }
                    const rangeText = apiData.range && apiData.range.mode !== 'full'
                        ? ` (${apiData.range.mode} ${apiData.range.ticks} ticks)`
                        : '';
                    showDashboardStatus(`History loaded for ${currentSelectedAgentForDialogueView}${rangeText} (${historyToDisplay.length} entries).`, 'success');
                } else {
                    showDashboardStatus(apiData.error || `Failed to load history for ${currentSelectedAgentForDialogueView}.`, 'error');
                     addMessageToAgentBubbleLog({event: "error", data: {message: `Failed to load history for ${currentSelectedAgentForDialogueView}. ${apiData.error || ''}`}, timestamp: Date.now()/1000}, true);
                }
            } catch (error) {
                if (requestId !== historyLoadSequence) return;
                console.error(`Error in handleAgentDialogueViewChange for ${newSelectedAgent}:`, error);
                agentSpecificBubbleLog.innerHTML = '';
                addMessageToAgentBubbleLog({
                    event: "error",
                    data: { message: `Could not load history for ${newSelectedAgent}. ${error.message || ''}` },
                    timestamp: Date.now() / 1000,
                }, true);
                showDashboardStatus(`Could not load history for ${newSelectedAgent}. Please try selecting the agent again.`, 'error');
            }
            finally {
                if (requestId === historyLoadSequence && isLoadingHistoryForAgent === newSelectedAgent) {
                    isLoadingHistoryForAgent = null;
                    updateHistoryWindowSelector();
                    const bufferedEvents = bufferedLiveEventsDuringHistory;
                    bufferedLiveEventsDuringHistory = [];
                    bufferedEvents
                        .filter(eventData => {
                            const eventTimestamp = Number(eventData && eventData.timestamp);
                            return !Number.isFinite(loadedHistoryTimestampCutoff) ||
                                !Number.isFinite(eventTimestamp) ||
                                eventTimestamp > loadedHistoryTimestampCutoff;
                        })
                        .forEach(appendLiveEventToAgentDialogue);
                }
            }
        }
        updateSourceToggleButtons();
        getOrchestratorStatus(); 
    }

    async function refreshMultistartPreviewDialogue() {
        if (!isMultistartPreview || multistartPreviewDialogueRefreshInFlight) return;
        if (document.visibilityState !== 'visible') return;
        const agentName = currentSelectedAgentForDialogueView;
        if (!agentName || agentName === 'all' || isLoadingHistoryForAgent) return;
        if (!agentSpecificBubbleLog || agentSpecificBubbleLog.style.display === 'none') return;

        multistartPreviewDialogueRefreshInFlight = true;
        try {
            const response = await fetch(
                buildHistoryApiUrl(agentName, {
                    window: 'recent',
                    ticks: HISTORY_WINDOW_TICKS,
                    modifiedAtNs: multistartPreviewDialogueMtimes.get(agentName),
                }),
                {method: 'GET', headers: {'Accept': 'application/json'}, cache: 'no-store'},
            );
            if (!response.ok) return;
            const apiData = await response.json();
            if (apiData.unchanged) return;
            if (!apiData.success || !Array.isArray(apiData.history)) return;
            if (currentSelectedAgentForDialogueView !== agentName) return;

            const signature = multistartPreviewDialogueSignature(apiData.history);
            if (multistartPreviewDialogueSignatures.get(agentName) === signature) return;

            const scrollAnchor = captureAgentLogScrollAnchor();
            const distanceFromBottom = agentSpecificBubbleLog.scrollHeight -
                agentSpecificBubbleLog.scrollTop - agentSpecificBubbleLog.clientHeight;
            const stickToBottom = distanceFromBottom < 120;
            const sourceCollapseState = captureCurrentSourceCollapseState();
            const latestTickView = latestTickAttemptView(apiData.history);
            const renderRequestId = historyLoadSequence;

            renderedAgentDialogueFingerprints.clear();
            pendingLiveStationObservations.clear();
            const pendingObservationEvent = historyObservationToLiveEvent(
                latestTickView.pendingObservation,
                agentName,
            );
            if (pendingObservationEvent) {
                pendingLiveStationObservations.set(
                    String(pendingObservationEvent.data.tick),
                    pendingObservationEvent,
                );
            }

            const rendered = await renderHistoryEntries(latestTickView.entries, {
                requestId: renderRequestId,
                agentName,
                replaceExisting: true,
                stickToBottom,
            });
            if (!rendered || currentSelectedAgentForDialogueView !== agentName) return;
            restoreCurrentSourceCollapseState(sourceCollapseState);
            if (!stickToBottom) restoreAgentLogScrollAnchor(scrollAnchor);
            multistartPreviewDialogueSignatures.set(agentName, signature);
            if (apiData.source_modified_at_ns) {
                multistartPreviewDialogueMtimes.set(agentName, apiData.source_modified_at_ns);
            }
        } catch (error) {
            console.debug('Seed 1 dialogue refresh failed; retrying on the next poll.', error);
        } finally {
            multistartPreviewDialogueRefreshInFlight = false;
        }
    }

    async function loadFullHistoryInBackground() {
        if (!currentSelectedAgentForDialogueView || currentSelectedAgentForDialogueView === "all") return;
        const agentName = currentSelectedAgentForDialogueView;
        const cachedFullHistory = fullDialogueHistoryCache[agentName];
        if (Array.isArray(cachedFullHistory) && cachedFullHistory.length > 0) {
            const renderRequestId = historyLoadSequence;
            const scrollAnchor = captureAgentLogScrollAnchor();
            const sourceCollapseState = captureCurrentSourceCollapseState();
            renderedAgentDialogueFingerprints.clear();
            pendingLiveStationObservations.clear();
            agentSpecificBubbleLog.innerHTML = '';
            const rendered = await renderHistoryEntries(cachedFullHistory, {
                requestId: renderRequestId,
                agentName,
            });
            if (!rendered) return;
            restoreCurrentSourceCollapseState(sourceCollapseState);
            restoreAgentLogScrollAnchor(scrollAnchor);
            showLoadFullHistoryButton(false, 'Load Full History');
            showHistoryWindowSelector(false);
            showDashboardStatus(`Showing full history for ${agentName} (${cachedFullHistory.length} entries).`, 'success');
            return;
        }
        if (backgroundHistoryLoads.get(agentName)) {
            showDashboardStatus(`Full history is already loading for ${agentName}.`, 'info');
            return;
        }

        backgroundHistoryLoads.set(agentName, true);
        if (loadFullHistoryButton) {
            loadFullHistoryButton.disabled = true;
            loadFullHistoryButton.textContent = 'Loading Full...';
        }
        showDashboardStatus(`Loading full history for ${agentName} in the background...`, 'info', 0);

        try {
            const apiData = await fetchApi(buildHistoryApiUrl(agentName, { full: true }));
            if (!apiData.success || !apiData.history) {
                showDashboardStatus(apiData.error || `Failed to load full history for ${agentName}.`, 'error');
                return;
            }

            fullDialogueHistoryCache[agentName] = apiData.history;
            if (currentSelectedAgentForDialogueView === agentName) {
                const renderRequestId = historyLoadSequence;
                const scrollAnchor = captureAgentLogScrollAnchor();
                const sourceCollapseState = captureCurrentSourceCollapseState();
                renderedAgentDialogueFingerprints.clear();
                pendingLiveStationObservations.clear();
                agentSpecificBubbleLog.innerHTML = '';
                const rendered = await renderHistoryEntries(apiData.history, {
                    requestId: renderRequestId,
                    agentName,
                });
                if (!rendered) return;
                restoreCurrentSourceCollapseState(sourceCollapseState);
                restoreAgentLogScrollAnchor(scrollAnchor);
                showLoadFullHistoryButton(false, 'Load Full History');
                showHistoryWindowSelector(false);
            }
            showDashboardStatus(`Full history loaded for ${agentName} (${apiData.history.length} entries).`, 'success');
        } catch (error) {
            console.error(`Error loading full history for ${agentName}:`, error);
        } finally {
            backgroundHistoryLoads.delete(agentName);
            if (loadFullHistoryButton) {
                loadFullHistoryButton.disabled = false;
                if (!loadFullHistoryButton.classList.contains('hidden')) {
                    loadFullHistoryButton.textContent = 'Load Full History';
                }
            }
        }
    }
    
    function scheduleLiveUiRefresh(refreshAgents = false) {
        liveAgentRefreshNeeded = liveAgentRefreshNeeded || refreshAgents;
        if (liveUiRefreshTimer) return;
        liveUiRefreshTimer = setTimeout(async () => {
            liveUiRefreshTimer = null;
            const orchestratorStatusData = await getOrchestratorStatus();
            updateStationStatistics();
            if (liveAgentRefreshNeeded) fetchAgentsForDashboard(orchestratorStatusData);
            liveAgentRefreshNeeded = false;
        }, 150);
    }

    function applyLiveStreamIdentity(epoch, sequence, forceCursor = false) {
        const nextEpoch = typeof epoch === 'string' && epoch ? epoch : null;
        if (nextEpoch && nextEpoch !== liveStreamEpoch) {
            liveStreamEpoch = nextEpoch;
            liveEventCursor = null;
            forceCursor = true;
        }
        const nextSequence = Number(sequence);
        if (!Number.isFinite(nextSequence) || nextSequence < 0) return;
        if (forceCursor || liveEventCursor === null || nextSequence > liveEventCursor) {
            liveEventCursor = nextSequence;
        }
    }

    function showLiveGapNotice(droppedCount) {
        if ((Number(droppedCount) || 0) <= 0) return;
        showDashboardStatus('Live updates resumed. Older transient notifications were skipped; saved agent history remains available.', 'info');
    }

    function removeRenderedDialogueTick(tick) {
        const tickKey = String(tick);
        agentSpecificBubbleLog
            .querySelectorAll('.chat-bubble[data-dialogue-tick]')
            .forEach(existingBubble => {
                if (existingBubble.dataset.dialogueTick !== tickKey) return;
                const existingFingerprint = existingBubble.dataset.dialogueFingerprint;
                if (existingFingerprint) renderedAgentDialogueFingerprints.delete(existingFingerprint);
                existingBubble.remove();
            });
    }

    function appendLiveEventToAgentDialogue(eventData) {
        const topLevelEventType = eventData.event;
        const ssePayload = eventData.data && typeof eventData.data === 'object' ? eventData.data : {};
        const timestamp = eventData.timestamp;
        if (currentSelectedAgentForDialogueView === "all" ||
            ssePayload.agent_name !== currentSelectedAgentForDialogueView) {
            return;
        }
        if (topLevelEventType === 'llm_event' && ssePayload.type === 'observation') {
            pendingLiveStationObservations.set(String(ssePayload.tick), eventData);
            return;
        }
        if (topLevelEventType === 'llm_event' && ssePayload.type === 'response') {
            const tickKey = String(ssePayload.tick);
            const pendingObservation = pendingLiveStationObservations.get(tickKey);
            if (pendingObservation) {
                removeRenderedDialogueTick(tickKey);
                addMessageToAgentBubbleLog(pendingObservation, false);
                pendingLiveStationObservations.delete(tickKey);
            }
            addMessageToAgentBubbleLog(eventData, false);
            return;
        }
        if (topLevelEventType === 'human_assist_event') {
            if (ssePayload.type === 'manual_message_human_part_sent') {
                addMessageToAgentBubbleLog({
                    event: 'manual_message_to_agent_llm',
                    data: {
                        agent_name: ssePayload.agent_name,
                        tick: ssePayload.tick || timestamp,
                        speaker: 'HumanAssistant',
                        text_content: ssePayload.text_content
                    },
                    timestamp
                }, false);
            } else if (ssePayload.type === 'manual_llm_response_received') {
                addMessageToAgentBubbleLog({
                    event: 'manual_llm_response_to_human',
                    data: {
                        agent_name: ssePayload.agent_name,
                        tick: ssePayload.tick || timestamp,
                        speaker: `${ssePayload.agent_name} (LLM)`,
                        text_content: ssePayload.text_content,
                        thinking_text: ssePayload.thinking_text,
                        token_info: ssePayload.token_info
                    },
                    timestamp
                }, false);
            }
        } else if (topLevelEventType === 'final_chat_event') {
            if (ssePayload.type === 'human_message_sent') {
                addMessageToAgentBubbleLog({
                    event: 'final_message_to_agent',
                    data: {
                        agent_name: ssePayload.agent_name,
                        tick: ssePayload.tick || timestamp,
                        speaker: 'HumanFinalChat',
                        text_content: ssePayload.human_message
                    },
                    timestamp
                }, false);
            } else if (ssePayload.type === 'agent_response_received') {
                addMessageToAgentBubbleLog({
                    event: 'final_agent_response_to_human',
                    data: {
                        agent_name: ssePayload.agent_name,
                        tick: ssePayload.tick || timestamp,
                        speaker: `${ssePayload.agent_name} (FinalResponse)`,
                        text_content: ssePayload.llm_response,
                        thinking_text: ssePayload.thinking_text,
                        token_info: ssePayload.token_info
                    },
                    timestamp
                }, false);
            }
        } else {
            addMessageToAgentBubbleLog(eventData, false);
        }
    }

    function processLiveEvent(eventData) {
        if (!eventData || typeof eventData !== 'object') return;
        const sequence = Number(eventData.stream_sequence);
        const epochChanged = Boolean(eventData.stream_epoch && eventData.stream_epoch !== liveStreamEpoch);
        if (eventData.stream_control) {
            applyLiveStreamIdentity(eventData.stream_epoch, sequence, epochChanged || Boolean(eventData.data?.cursor_reset));
            showLiveGapNotice(eventData.data?.dropped_count);
            return;
        }
        if (!epochChanged && Number.isFinite(sequence) && liveEventCursor !== null && sequence <= liveEventCursor) {
            return;
        }
        applyLiveStreamIdentity(eventData.stream_epoch, sequence, epochChanged);

        const topLevelEventType = eventData.event;
        const ssePayload = eventData.data && typeof eventData.data === 'object' ? eventData.data : {};

        addMessageToGlobalNotificationBubbleLog(eventData);

        if (currentSelectedAgentForDialogueView !== "all" &&
            ssePayload.agent_name === currentSelectedAgentForDialogueView) {
            if (isLoadingHistoryForAgent) {
                bufferedLiveEventsDuringHistory.push(eventData);
            } else {
                appendLiveEventToAgentDialogue(eventData);
            }
        }

        if (['orchestrator_status', 'orchestrator_control', 'tick_event', 'agent_management', 'human_assist_event', 'connector_status', 'connector_error', 'final_chat_event'].includes(topLevelEventType)) {
            const refreshAgents = topLevelEventType === 'agent_management' ||
                (topLevelEventType === 'tick_event' && ssePayload.type === 'end') ||
                topLevelEventType === 'connector_status' ||
                (topLevelEventType === 'orchestrator_status' && ['paused_agent_departure', 'stopped'].includes(ssePayload.status));
            scheduleLiveUiRefresh(refreshAgents);
        }
        if (topLevelEventType === 'tick_event' && ssePayload.type === 'end' && ssePayload.next_tick !== undefined) {
            if (stationTickDashboard) stationTickDashboard.textContent = ssePayload.next_tick;
        }
    }

    // Lightweight fallback used only while SSE is unavailable.
    async function fetchRecentEvents() {
        if (pollingRequestInFlight) return;
        pollingRequestInFlight = true;
        try {
            const response = await fetch(buildRecentEventsEndpoint(), {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });
            if (!response.ok) throw new Error(`Fallback request failed: ${response.status}`);
            const data = await response.json();
            if (!data.success) return;
            if (liveTransportMode === 'sse' && data.stream_epoch !== liveStreamEpoch) return;
            const epochChanged = Boolean(data.stream_epoch && data.stream_epoch !== liveStreamEpoch);
            if (epochChanged) applyLiveStreamIdentity(data.stream_epoch, 0, true);
            (Array.isArray(data.events) ? data.events : []).forEach(processLiveEvent);
            applyLiveStreamIdentity(data.stream_epoch, data.cursor, Boolean(data.cursor_reset) && epochChanged);
            showLiveGapNotice(data.dropped_count);
        } catch (error) {
            console.error("Error fetching recent events:", error);
        } finally {
            pollingRequestInFlight = false;
        }
    }

    function selectedAgentStreamParam() {
        return encodeURIComponent(currentSelectedAgentForDialogueView || 'all');
    }

    function liveEventCursorParam() {
        return liveEventCursor === null ? 'latest' : String(liveEventCursor);
    }

    function buildLiveLogStreamEndpoint() {
        return `/api/orchestrator/live_log_stream?agent_name=${selectedAgentStreamParam()}&cursor=${encodeURIComponent(liveEventCursorParam())}`;
    }

    function buildRecentEventsEndpoint() {
        return `/api/orchestrator/recent_events?agent_name=${selectedAgentStreamParam()}&cursor=${encodeURIComponent(liveEventCursorParam())}&limit=50`;
    }

    function stopPollingFallback() {
        if (pollingFallbackInterval) {
            clearInterval(pollingFallbackInterval);
            pollingFallbackInterval = null;
        }
    }

    function clearSseReconnectTimer() {
        if (!sseReconnectTimer) return;
        clearTimeout(sseReconnectTimer);
        sseReconnectTimer = null;
    }

    function scheduleSseReconnect() {
        if (sseReconnectTimer || document.visibilityState !== 'visible') return;
        sseReconnectTimer = setTimeout(() => {
            sseReconnectTimer = null;
            connectSseLogStream();
        }, 15000);
    }

    function startPollingFallback() {
        liveTransportMode = 'polling';
        if (!pollingFallbackInterval) {
            void fetchRecentEvents();
            pollingFallbackInterval = setInterval(() => {
                void fetchRecentEvents();
            }, 5000);
        }
        scheduleSseReconnect();
    }

    function connectSseLogStream() {
        clearSseReconnectTimer();
        if (sseSource) {
            const previousSource = sseSource;
            sseSource = null;
            previousSource.close();
        }
        if (operationMode !== 'api') {
            stopPollingFallback();
            liveTransportMode = 'idle';
            return;
        }
        if (document.visibilityState !== 'visible') {
            liveTransportMode = 'idle';
            return;
        }

        const sseEndpoint = buildLiveLogStreamEndpoint();
        liveTransportMode = 'connecting';
        try {
            sseSource = new EventSource(sseEndpoint);
        } catch (error) {
            console.error("Failed to create EventSource:", error);
            startPollingFallback();
            return;
        }
        const activeSource = sseSource;

        const connectionTimeout = setTimeout(() => {
            if (sseSource !== activeSource) return;
            activeSource.close();
            if (sseSource === activeSource) {
                sseSource = null;
            }
            startPollingFallback();
        }, 5000);

        sseSource.onopen = function() {
            if (sseSource !== activeSource) return;
            clearTimeout(connectionTimeout);
            clearSseReconnectTimer();
            stopPollingFallback();
            liveTransportMode = 'sse';
        };

        sseSource.onmessage = function(event) {
            if (sseSource !== activeSource) return;
            try {
                processLiveEvent(JSON.parse(event.data));
            } catch (e) {
                console.error("Error parsing SSE data: ", e, "Raw data: ", event.data);
            }
        };

        sseSource.onerror = function(error) {
            if (sseSource !== activeSource) return;
            clearTimeout(connectionTimeout);
            console.error("SSE connection error:", error);
            activeSource.close();
            if (sseSource === activeSource) sseSource = null;
            if (liveTransportMode === 'sse') {
                showDashboardStatus('Live connection interrupted. Updates will continue through the fallback connection.', 'info');
            }
            startPollingFallback();
        };
    }


    // --- Event Listeners Setup ---

    if (dashboardStatusMessages) {
        dashboardStatusMessages.addEventListener('click', hideDashboardStatus);
        dashboardStatusMessages.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === 'Escape' || event.key === ' ') {
                event.preventDefault();
                hideDashboardStatus();
            }
        });
    }

    [logModeFullButton, logModeCollapsedButton].filter(Boolean).forEach(button => {
        button.addEventListener('click', () => setLogDisplayMode(button.dataset.logMode));
    });

    [globalNotificationBubbleLog, agentSpecificBubbleLog].filter(Boolean).forEach(container => {
        container.addEventListener('click', (event) => {
            const toggle = event.target.closest('.log-message-toggle');
            if (!toggle) return;
            event.preventDefault();
            event.stopPropagation();
            const bubble = toggle.closest('.log-message-collapsible');
            if (!bubble) return;
            setLogMessageExpanded(bubble, !bubble.classList.contains('log-message-expanded'));
        });
    });

    if (toggleAgentLogButton) {
        toggleAgentLogButton.addEventListener('click', () => {
            setCurrentLogSourceCollapsed('agent', !isCurrentLogSourceCollapsed('agent'));
        });
    }

    if (toggleStationLogButton) {
        toggleStationLogButton.addEventListener('click', () => {
            setCurrentLogSourceCollapsed('station', !isCurrentLogSourceCollapsed('station'));
        });
    }
    
    if (startLoopButton) startLoopButton.addEventListener('click', async () => { 
        showDashboardStatus('Launching station...', 'info');
        try { const data = await fetchApi('/api/orchestrator/start_loop', 'POST'); 
            showDashboardStatus(data.message, data.success ? 'success' : 'error'); 
            getOrchestratorStatus(); 
        } catch (error) {}
    });
    if (pauseOrchestratorButton) pauseOrchestratorButton.addEventListener('click', async () => {
        const isCancelAction = pauseOrchestratorButton.textContent === 'Cancel Pause';
        if (isCancelAction) {
            showDashboardStatus('Cancelling pause request...', 'info');
            try {
                const data = await fetchApi('/api/orchestrator/cancel_pause', 'POST');
                showDashboardStatus(data.message, data.success ? 'success' : 'info');
            } catch (error) {}
        } else {
            showDashboardStatus('Requesting orchestrator pause...', 'info');
            try { 
                const data = await fetchApi('/api/orchestrator/pause', 'POST'); 
                showDashboardStatus(data.message, data.success ? 'success' : 'info'); 
            } catch (error) {}
        }
    });
    if (resumeOrchestratorButton) resumeOrchestratorButton.addEventListener('click', async () => { 
        showDashboardStatus('Resuming orchestrator...', 'info');
        try { const data = await fetchApi('/api/orchestrator/resume', 'POST'); showDashboardStatus(data.message, data.success ? 'success' : 'error'); getOrchestratorStatus(); } catch (error) {}
    });
    if (stopOrchestratorButton) stopOrchestratorButton.addEventListener('click', async () => { 
        if (!confirm("Are you sure you want to stop the orchestrator?")) return;
        showDashboardStatus('Stopping orchestrator...', 'info');
        try { const data = await fetchApi('/api/orchestrator/stop', 'POST'); showDashboardStatus(data.message, data.success ? 'success' : 'error'); getOrchestratorStatus(); } catch (error) {}
    });

    if (agentSelectorDashboard) agentSelectorDashboard.addEventListener('change', () => {
        if (isRefreshingAgentSelector || agentSelectorDashboard.value === currentSelectedAgentForDialogueView) return;
        handleAgentDialogueViewChange();
        connectSseLogStream();
    });
    if (agentSelectorDashboard) agentSelectorDashboard.addEventListener('change', updateCopySystemPromptButtonState);

    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState !== 'visible') {
            clearSseReconnectTimer();
            stopPollingFallback();
            if (sseSource) {
                const hiddenSource = sseSource;
                sseSource = null;
                hiddenSource.close();
            }
            liveTransportMode = 'idle';
            return;
        }
        updateStationStatistics();
        getOrchestratorStatus();
        connectSseLogStream();
        void refreshMultistartPreviewDialogue();
    });

    if (copySystemPromptButton) {
        copySystemPromptButton.addEventListener('click', async () => {
            const selectedAgent = agentSelectorDashboard ? agentSelectorDashboard.value : null;
            if (!selectedAgent || selectedAgent === 'all') {
                showDashboardStatus('Select a specific agent to copy its system prompt.', 'error');
                updateCopySystemPromptButtonState();
                return;
            }

            try {
                const result = await fetchApi(`/api/agent/${encodeURIComponent(selectedAgent)}/system_prompt`, 'GET');
                const systemPrompt = String(result.system_prompt || '');
                if (!systemPrompt.trim()) {
                    showDashboardStatus(`No system prompt available for ${selectedAgent}.`, 'info');
                    return;
                }
                const promptLabel = selectedAgent === 'Reviewer' ? 'Reviewer Prompt' : `System Prompt for ${selectedAgent}`;
                const labeledSystemPrompt = `${promptLabel}:\n\n${systemPrompt}`;
                await copyTextToClipboard(labeledSystemPrompt, `${promptLabel} copied.`);
            } catch (error) {
                // fetchApi already surfaced the error
            }
        });
    }
    
    if (clearLogButton) clearLogButton.addEventListener('click', () => {
        if (currentSelectedAgentForDialogueView === "all") {
            if(globalNotificationBubbleLog) globalNotificationBubbleLog.innerHTML = `<div class="chat-bubble chat-bubble-system initial-log-message">Global notifications cleared by user. New messages will appear here.</div>`;
        } else {
            if(agentSpecificBubbleLog) agentSpecificBubbleLog.innerHTML = `<div class="chat-bubble chat-bubble-system initial-log-message">Log for ${currentSelectedAgentForDialogueView} cleared by user.</div>`;
            renderedAgentDialogueFingerprints.clear();
            pendingLiveStationObservations.clear();
            if (fullDialogueHistoryCache[currentSelectedAgentForDialogueView]) {
                fullDialogueHistoryCache[currentSelectedAgentForDialogueView] = [];
            }
        }
    });
    
    if (loadFullHistoryButton) loadFullHistoryButton.addEventListener('click', () => {
        if (currentSelectedAgentForDialogueView && currentSelectedAgentForDialogueView !== "all") {
            loadFullHistoryInBackground();
        }
    });

    if (historyWindowSelector) {
        historyWindowSelector.value = historyWindowMode;
        historyWindowSelector.addEventListener('change', () => {
            if (isMultistartPreview) return;
            const nextMode = historyWindowSelector.value;
            if (!['recent', 'earliest'].includes(nextMode)) return;
            historyWindowMode = nextMode;
            localStorage.setItem(HISTORY_WINDOW_STORAGE_KEY, historyWindowMode);
            if (currentSelectedAgentForDialogueView && currentSelectedAgentForDialogueView !== "all") {
                handleAgentDialogueViewChange(false);
            }
        });
    }
    
    if (createApiAgentModalButton) createApiAgentModalButton.addEventListener('click', () => {
        const restoreDraft = consumeBackdropDraftFlag(createApiAgentModal);
        if(createApiAgentModal) createApiAgentModal.style.display = "block";
        const modelNameInput = document.getElementById('api-model-name');
        if(modelNameInput) modelNameInput.focus();

        // Populate presets
        if (!restoreDraft && apiModelPreset && typeof MODEL_PRESETS !== 'undefined') {
            apiModelPreset.innerHTML = '<option value="">-- Select a Preset --</option>';
            MODEL_PRESETS.forEach((preset, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = preset.display_name;
                apiModelPreset.appendChild(option);
            });
        }

        // Update OpenAI params visibility when modal opens
        if (updateOpenAIParamsVisibility) {
            updateOpenAIParamsVisibility();
        }
    });
    
    if (apiModelPreset) {
        apiModelPreset.addEventListener('change', (e) => {
            const selectedIndex = e.target.value;
            if (selectedIndex && typeof MODEL_PRESETS !== 'undefined' && MODEL_PRESETS[selectedIndex]) {
                const preset = MODEL_PRESETS[selectedIndex];
                const form = createApiAgentForm;
                if(form) {
                    form.querySelector('[name="model_provider_class"]').value = preset.model_provider_class || '';
                    form.querySelector('[name="model_name"]').value = preset.model_name || '';
                    form.querySelector('[name="initial_tokens_max"]').value = preset.initial_tokens_max || '';
                    form.querySelector('[name="llm_system_prompt"]').value =
                        preset.role_definition || preset.llm_system_prompt || '';

                    // Update OpenAI params visibility after preset changes provider
                    if (updateOpenAIParamsVisibility) {
                        updateOpenAIParamsVisibility();
                    }
                }
            }
        });
    }
    
    if (closeApiModalButton) closeApiModalButton.addEventListener('click', () => { hideModalWithoutReset(createApiAgentModal, true); });
    if (apiAgentTypeSelect) apiAgentTypeSelect.addEventListener('change', (e) => {
        if(apiRecursiveFieldsDiv) apiRecursiveFieldsDiv.classList.toggle('hidden', e.target.value !== "Recursive Agent");
    });

    // Show/hide OpenAI-specific parameters based on selected provider
    const apiModelProviderSelect = document.getElementById('api-model-provider-class');
    const openaiSpecificParams = document.getElementById('openai-specific-params');
    let updateOpenAIParamsVisibility = null; // Declare in outer scope

    if (apiModelProviderSelect && openaiSpecificParams) {
        // Function to update visibility
        updateOpenAIParamsVisibility = () => {
            const isOpenAI = apiModelProviderSelect.value === 'OpenAI';
            openaiSpecificParams.classList.toggle('hidden', !isOpenAI);
        };

        // Trigger on change
        apiModelProviderSelect.addEventListener('change', updateOpenAIParamsVisibility);

        // Also trigger on initial load
        updateOpenAIParamsVisibility();
    }
    if (createApiAgentForm) createApiAgentForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(createApiAgentForm);
        const data = Object.fromEntries(formData.entries());
        const agentCountRaw = data.agent_count;
        let agentCount = parseInt(agentCountRaw, 10);
        if (!Number.isInteger(agentCount) || agentCount < 1) {
            agentCount = 1;
        }
        delete data.agent_count;
        if (data.generation) data.generation = parseInt(data.generation, 10); else delete data.generation;
        if (data.llm_temperature) data.llm_temperature = parseFloat(data.llm_temperature); else delete data.llm_temperature;
        if (data.llm_max_tokens) data.llm_max_tokens = parseInt(data.llm_max_tokens, 10); else delete data.llm_max_tokens;
        if (!data.agent_name) delete data.agent_name;
        if (!data.llm_system_prompt) delete data.llm_system_prompt;
        if (!data.initial_tokens_max) delete data.initial_tokens_max; else data.initial_tokens_max = parseInt(data.initial_tokens_max);
        if (!data.internal_note) delete data.internal_note;
        if (!data.assigned_ancestor) delete data.assigned_ancestor;

        // Build custom_api_params for provider-specific settings
        const customApiParams = {};
        if (data.model_provider_class === 'OpenAI') {
            const verbosity = document.getElementById('api-openai-verbosity')?.value;
            if (verbosity) {
                customApiParams.verbosity = verbosity;
            }
        }
        if (Object.keys(customApiParams).length > 0) {
            data.llm_custom_api_params = customApiParams;
        }

        if (data.agent_type !== "Recursive Agent") { delete data.lineage; delete data.generation; }
        else {
            if (!data.agent_name && (!data.lineage || (data.generation === null || data.generation === undefined || isNaN(data.generation)))) {
                showDashboardStatus("For Recursive type with auto-name, Lineage and Generation are required.", "error"); return;
            }
            if (agentCount > 1 && !data.agent_name) {
                showDashboardStatus("For multiple recursive agents, please provide an Agent Name base.", "error"); return;
            }
        }

        const agentLabel = agentCount === 1 ? "agent" : "agents";
        showDashboardStatus(`Creating ${agentCount} ${agentLabel}...`, 'info');
        try {
            let successCount = 0;
            let failureCount = 0;
            let lastError = null;
            for (let i = 0; i < agentCount; i += 1) {
                const payload = { ...data };
                if (agentCount > 1 && data.agent_name) {
                    payload.agent_name = `${data.agent_name}-${i + 1}`;
                }
                const result = await fetchApi('/api/orchestrator/add_agent', 'POST', payload);
                if (result.success) {
                    successCount += 1;
                } else {
                    failureCount += 1;
                    lastError = result.message;
                }
            }

            const summaryParts = [`${successCount} ${successCount === 1 ? 'agent' : 'agents'} created`];
            if (failureCount > 0) summaryParts.push(`${failureCount} failed`);
            let summaryMessage = `${summaryParts.join(', ')}.`;
            if (lastError && failureCount > 0) {
                summaryMessage += ` Last error: ${lastError}`;
            }
            showDashboardStatus(summaryMessage, failureCount > 0 ? 'error' : 'success');

            if (successCount > 0) {
                if (failureCount === 0) {
                    if (createApiAgentModal) createApiAgentModal.style.display = "none";
                    createApiAgentForm.reset();
                    if (apiRecursiveFieldsDiv) apiRecursiveFieldsDiv.classList.add('hidden');
                }
                await fetchAgentsForDashboard();
                await getOrchestratorStatus();
            }
        } catch (error) { /* Handled by fetchApi */ }
    });

    function hideModalWithoutReset(modal, preserveDraft = false) {
        if (!modal) return;
        if (preserveDraft) modal.dataset.reopenWithDraft = 'true';
        modal.style.display = "none";
    }

    function consumeBackdropDraftFlag(modal) {
        if (!modal || modal.dataset.reopenWithDraft !== 'true') return false;
        delete modal.dataset.reopenWithDraft;
        return true;
    }

    document.querySelectorAll('.modal').forEach((modal) => {
        modal.addEventListener('click', (event) => {
            if (event.target === modal) {
                hideModalWithoutReset(modal, true);
            }
        });
    });

    if(endApiAgentSessionButton) endApiAgentSessionButton.addEventListener('click', async () => {
        const agentToEnd = agentSelectorDashboard ? agentSelectorDashboard.value : null;
        if (!agentToEnd || agentToEnd === "all") { showDashboardStatus("Please select an agent to end their session.", "error"); return; }
        
        if (!confirm(`Are you sure you want to request to end the session for agent '${agentToEnd}'? The session will terminate at the start of their next turn.`)) return;
        
        showDashboardStatus(`Requesting to end session for ${agentToEnd}...`, 'info');
        endApiAgentSessionButton.disabled = true; // Disable button immediately
        try {
            const result = await fetchApi('/api/orchestrator/end_agent', 'POST', { agent_name: agentToEnd });
            showDashboardStatus(result.message, result.success ? 'success' : 'error');
            if (result.success) {
                endApiAgentSessionButton.textContent = "End Requested"; // Provide visual feedback
            } else {
                endApiAgentSessionButton.disabled = false; // Re-enable on failure
            }
        } catch (error) {
            endApiAgentSessionButton.disabled = false; // Re-enable on failure
        }
    });

    if (openSendSystemMessageModalButton) {
        openSendSystemMessageModalButton.addEventListener('click', () => {
            const restoreDraft = consumeBackdropDraftFlag(sendSystemMessageModal);
            const previousSelectedAgents = systemMessageAgentSelector
                ? Array.from(systemMessageAgentSelector.selectedOptions).map(option => option.value)
                : [];
            // Check if orchestrator is at least prepared, or if active agents can be fetched otherwise
            // For now, relies on orchestratorState.turn_order which is populated by getOrchestratorStatus
            if (!orchestratorState.is_prepared && !orchestratorState.is_running && orchestratorState.turn_order.length === 0) {
                 showDashboardStatus("Orchestrator should be at least prepared, or have active agents, to populate target list.", "info");
                 // Consider allowing opening even if orchestratorState is not fully ready,
                 // if there's another way to get 'active' agents or if user can manually input.
                 // For now, we rely on orchestratorState.turn_order.
            }
            if (systemMessageAgentSelector) {
                systemMessageAgentSelector.innerHTML = ''; // Clear previous options
                const activeAgents = orchestratorState.turn_order || []; // Get from cached orchestrator status

                if (activeAgents.length === 0) {
                    const option = document.createElement('option');
                    option.textContent = "No active agents in turn order.";
                    option.disabled = true;
                    systemMessageAgentSelector.appendChild(option);
                    if (confirmSendSystemMessageButton) confirmSendSystemMessageButton.disabled = true;
                } else {
                    activeAgents.forEach(agentName => {
                        const option = document.createElement('option');
                        const agentInfo = fullAgentListCache.find(agent => agent.name === agentName);
                        option.value = agentName;
                        option.textContent = agentInfo ? formatAgentDisplayName(agentInfo) : agentName;
                        if (restoreDraft && previousSelectedAgents.includes(agentName)) {
                            option.selected = true;
                        }
                        systemMessageAgentSelector.appendChild(option);
                    });
                    if (confirmSendSystemMessageButton) confirmSendSystemMessageButton.disabled = false;
                }
            }
            if (!restoreDraft && systemMessageContent) systemMessageContent.value = "";
            if (!restoreDraft && systemMessageArchitectCheckbox) systemMessageArchitectCheckbox.checked = false;
            if (sendSystemMessageModal) sendSystemMessageModal.style.display = "block";
            if (systemMessageContent) systemMessageContent.focus();
        });
    }

    if (closeSendSystemMessageModalButton) {
        closeSendSystemMessageModalButton.addEventListener('click', () => {
            hideModalWithoutReset(sendSystemMessageModal, true);
        });
    }

    if (confirmSendSystemMessageButton) {
        confirmSendSystemMessageButton.addEventListener('click', async () => {
            const selectedAgentOptions = systemMessageAgentSelector ? Array.from(systemMessageAgentSelector.selectedOptions) : [];
            const targetAgents = selectedAgentOptions.map(option => option.value);
            const message = systemMessageContent ? systemMessageContent.value.trim() : "";
            const messageToSend = systemMessageArchitectCheckbox && systemMessageArchitectCheckbox.checked
                ? `**Architect Message**\n${message}`
                : message;

            if (targetAgents.length === 0) {
                showDashboardStatus("Please select at least one target agent.", "error");
                return;
            }
            if (!message) {
                showDashboardStatus("System message content cannot be empty.", "error");
                return;
            }

            showDashboardStatus(`Sending system message to ${targetAgents.length} agent(s)...`, 'info', 0); // 0 for persistent
            confirmSendSystemMessageButton.disabled = true;
            try {
                const result = await fetchApi('/api/station/send_system_message', 'POST', {
                    target_agents: targetAgents,
                    message_content: messageToSend
                });
                showDashboardStatus(result.message, result.success ? 'success' : 'error');
                if (result.success) {
                    if (systemMessageContent) systemMessageContent.value = "";
                    if (systemMessageArchitectCheckbox) systemMessageArchitectCheckbox.checked = false;
                    if (sendSystemMessageModal) sendSystemMessageModal.style.display = "none";
                }
            } catch (error) {
                // fetchApi already shows error via its own catch block if it throws
                // If fetchApi doesn't throw but API returns non-ok, it's handled above
            } finally {
                confirmSendSystemMessageButton.disabled = false;
            }
        });
    }

    // --- Chat ---
    if (openArchivePapersModalButton) {
        openArchivePapersModalButton.addEventListener('click', () => {
            void openArchivePapersModal();
        });
    }

    if (closeArchivePapersModalButton) {
        closeArchivePapersModalButton.addEventListener('click', () => {
            hideModalWithoutReset(archivePapersModal);
        });
    }

    if (openWebArchiveSurveysModalButton) {
        openWebArchiveSurveysModalButton.addEventListener('click', () => {
            void openWebArchiveSurveysModal();
        });
    }

    if (closeWebArchiveSurveysModalButton) {
        closeWebArchiveSurveysModalButton.addEventListener('click', () => {
            hideModalWithoutReset(webArchiveSurveysModal);
        });
    }

    if (openWebArchiveSurveyRequestButton) {
        openWebArchiveSurveyRequestButton.addEventListener('click', () => {
            void openWebArchiveSurveyRequestModal();
        });
    }

    if (closeWebArchiveSurveyRequestModalButton) {
        closeWebArchiveSurveyRequestModalButton.addEventListener('click', () => {
            closeWebArchiveSurveyRequestModal(true);
        });
    }

    if (cancelWebArchiveSurveyRequestButton) {
        cancelWebArchiveSurveyRequestButton.addEventListener('click', () => {
            closeWebArchiveSurveyRequestModal(true);
        });
    }

    if (copyAllArchiveAbstractsButton) {
        copyAllArchiveAbstractsButton.addEventListener('click', () => {
            if (!archiveAllAbstractsMarkdown.trim()) {
                showDashboardStatus('No archive abstracts to copy.', 'info');
                return;
            }
            void copyTextToClipboard(archiveAllAbstractsMarkdown, 'All archive abstracts copied to clipboard.');
        });
    }

    if (backToArchivePapersListButton) {
        backToArchivePapersListButton.addEventListener('click', () => {
            showArchiveListView();
        });
    }

    if (copyArchivePaperMarkdownButton) {
        copyArchivePaperMarkdownButton.addEventListener('click', () => {
            if (!archiveSelectedPaper || !archiveSelectedPaper.paper_markdown) {
                showDashboardStatus('No archive paper is currently open.', 'error');
                return;
            }
            void copyTextToClipboard(String(archiveSelectedPaper.paper_markdown), 'Archive paper markdown copied to clipboard.');
        });
    }

    if (archiveRowCountSelect) {
        archiveRowCountSelect.addEventListener('change', () => {
            const nextCount = Number(archiveRowCountSelect.value);
            archiveRowCount = [25, 50, 100, 200].includes(nextCount) ? nextCount : 100;
            archivePage = 1;
            renderArchivePapersTable();
        });
    }

    if (archivePreviousPageButton) {
        archivePreviousPageButton.addEventListener('click', () => {
            if (archivePage <= 1) return;
            archivePage -= 1;
            renderArchivePapersTable();
        });
    }

    if (archiveNextPageButton) {
        archiveNextPageButton.addEventListener('click', () => {
            const totalPages = Math.max(1, Math.ceil(archivePapersCache.length / archiveRowCount));
            if (archivePage >= totalPages) return;
            archivePage += 1;
            renderArchivePapersTable();
        });
    }

    if (clearWebArchiveSurveyPromptButton) {
        clearWebArchiveSurveyPromptButton.addEventListener('click', () => {
            if (webArchiveSurveyPrompt) {
                webArchiveSurveyPrompt.value = '';
                webArchiveSurveyPrompt.focus();
            }
        });
    }

    if (submitWebArchiveSurveyButton) {
        submitWebArchiveSurveyButton.addEventListener('click', () => {
            void submitWebArchiveSurvey();
        });
    }

    if (refreshWebArchiveSurveysButton) {
        refreshWebArchiveSurveysButton.addEventListener('click', () => {
            void loadWebArchiveSurveys();
        });
    }

    if (webArchiveSurveyRowCountSelect) {
        webArchiveSurveyRowCountSelect.addEventListener('change', () => {
            const nextCount = Number(webArchiveSurveyRowCountSelect.value);
            webArchiveSurveyRowCount = [25, 50, 100, 200].includes(nextCount) ? nextCount : 100;
            webArchiveSurveyPage = 1;
            renderWebArchiveSurveysTable();
        });
    }

    if (webArchiveSurveyPreviousPageButton) {
        webArchiveSurveyPreviousPageButton.addEventListener('click', () => {
            if (webArchiveSurveyPage <= 1) return;
            webArchiveSurveyPage -= 1;
            renderWebArchiveSurveysTable();
        });
    }

    if (webArchiveSurveyNextPageButton) {
        webArchiveSurveyNextPageButton.addEventListener('click', () => {
            const totalPages = Math.max(1, Math.ceil(webArchiveSurveys.length / webArchiveSurveyRowCount));
            if (webArchiveSurveyPage >= totalPages) return;
            webArchiveSurveyPage += 1;
            renderWebArchiveSurveysTable();
        });
    }

    if (copyWebArchiveSurveyMarkdownButton) {
        copyWebArchiveSurveyMarkdownButton.addEventListener('click', () => {
            const markdown = webArchiveSelectedSurvey ? String(webArchiveSelectedSurvey.report_markdown || '') : '';
            if (!markdown.trim()) {
                showDashboardStatus('No finished survey report is currently open.', 'error');
                return;
            }
            void copyTextToClipboard(markdown, 'Survey report Markdown copied to clipboard.');
        });
    }

    if (removeWebArchiveSurveyButton) {
        removeWebArchiveSurveyButton.addEventListener('click', () => {
            const surveyId = webArchiveSelectedSurvey ? Number(webArchiveSelectedSurvey.id) : NaN;
            if (Number.isFinite(surveyId)) void removeWebArchiveSurvey(surveyId);
        });
    }

    if (closeWebArchiveSurveyReportButton) {
        closeWebArchiveSurveyReportButton.addEventListener('click', () => {
            showWebArchiveSurveysListView();
        });
    }

    document.querySelectorAll('.archive-sort-button').forEach(button => {
        button.addEventListener('click', () => {
            const key = button.getAttribute('data-sort-key');
            if (!key) return;
            if (archiveSortState.key === key) {
                archiveSortState.direction = archiveSortState.direction === 'asc' ? 'desc' : 'asc';
            } else {
                archiveSortState = {
                    key,
                    direction: key === 'title' || key === 'author' ? 'asc' : 'desc'
                };
            }
            archivePage = 1;
            renderArchivePapersTable();
        });
    });

    if (openQuestionRoomModalButton) {
        openQuestionRoomModalButton.addEventListener('click', () => {
            void openQuestionRoomModal();
        });
    }

    if (closeQuestionRoomModalButton) {
        closeQuestionRoomModalButton.addEventListener('click', () => {
            hideModalWithoutReset(questionRoomModal);
        });
    }

    if (backToQuestionRoomListButton) {
        backToQuestionRoomListButton.addEventListener('click', () => {
            showQuestionRoomListView();
        });
    }

    if (questionPreviousPageButton) {
        questionPreviousPageButton.addEventListener('click', () => {
            if (questionListState.page <= 1) return;
            questionListState.page -= 1;
            void loadQuestionRoomPage();
        });
    }

    if (questionNextPageButton) {
        questionNextPageButton.addEventListener('click', () => {
            if (questionListState.page >= questionListState.totalPages) return;
            questionListState.page += 1;
            void loadQuestionRoomPage();
        });
    }

    if (questionPageSizeSelect) {
        questionPageSizeSelect.addEventListener('change', () => {
            const nextSize = Number(questionPageSizeSelect.value);
            questionListState.pageSize = [25, 50, 100].includes(nextSize) ? nextSize : 100;
            questionListState.page = 1;
            void loadQuestionRoomPage();
        });
    }

    document.querySelectorAll('.question-sort-button').forEach(button => {
        button.addEventListener('click', () => {
            const key = button.getAttribute('data-sort-key');
            if (!key) return;
            if (questionListState.sortBy === key) {
                questionListState.sortDirection = questionListState.sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                questionListState.sortBy = key;
                questionListState.sortDirection = ['title', 'author', 'status'].includes(key) ? 'asc' : 'desc';
            }
            questionListState.page = 1;
            void loadQuestionRoomPage();
        });
    });

    if (openTemporalChatModalButton) {
        openTemporalChatModalButton.addEventListener('click', async () => {
            if (!temporalChatAgentSelector) return;
            const restoreDraft = consumeBackdropDraftFlag(temporalChatModal);
            const previousTemporalAgent = temporalChatAgentSelector.value;
            temporalChatAgentSelector.innerHTML = '';

            const selectableAgents = (fullAgentListCache || [])
                .filter(a => a && a.name && !String(a.status || '').startsWith("Ascended"))
                .sort((a, b) => {
                    const aBranched = !!a.temporal_chat_exists;
                    const bBranched = !!b.temporal_chat_exists;
                    if (aBranched !== bBranched) return aBranched ? -1 : 1;
                    if (aBranched && bBranched) {
                        const aUpdated = String(a.temporal_chat_updated_at || '');
                        const bUpdated = String(b.temporal_chat_updated_at || '');
                        if (aUpdated !== bUpdated) return bUpdated.localeCompare(aUpdated);
                    }
                    return String(a.name || '').localeCompare(String(b.name || ''));
                });

            if (selectableAgents.length === 0) {
                const option = document.createElement('option');
                option.value = "";
                option.textContent = "No agents available.";
                option.disabled = true;
                temporalChatAgentSelector.appendChild(option);
            } else {
                selectableAgents.forEach(agent => {
                    const option = document.createElement('option');
                    option.value = agent.name;
                    option.textContent = formatAgentDisplayName(agent, { includeBranchMarker: true });
                    temporalChatAgentSelector.appendChild(option);
                });

                // Default selection: current dashboard selection if valid, else first.
                const selectedAgentName = agentSelectorDashboard ? agentSelectorDashboard.value : null;
                if (restoreDraft && previousTemporalAgent && selectableAgents.some(agent => agent.name === previousTemporalAgent)) {
                    temporalChatAgentSelector.value = previousTemporalAgent;
                } else if (selectedAgentName && selectedAgentName !== "all") {
                    const selectedAgentData = fullAgentListCache.find(a => a.name === selectedAgentName);
                    const isAscended = selectedAgentData && String(selectedAgentData.status || '').startsWith("Ascended");
                    if (selectedAgentData && !isAscended) {
                        temporalChatAgentSelector.value = selectedAgentName;
                    }
                }
            }

            if (!restoreDraft && temporalChatInput) temporalChatInput.value = "";
            if (!restoreDraft && temporalChatBranchTickInput) temporalChatBranchTickInput.value = "";
            if (temporalChatModal) temporalChatModal.style.display = "block";
            _renderTemporalChatTranscript(temporalChatAgentSelector.value);
            if (temporalChatAgentSelector.value) {
                try {
                    await _loadTemporalChatState(temporalChatAgentSelector.value);
                    _renderTemporalChatTranscript(temporalChatAgentSelector.value);
                } catch (e) {
                    showDashboardStatus(`Failed to load chat: ${e.message || e}`, "error");
                }
            }
            if (temporalChatInput) temporalChatInput.focus();
        });
    }

    if (closeTemporalChatModalButton) {
        closeTemporalChatModalButton.addEventListener('click', () => {
            _closeTemporalChatModal();
        });
    }
    if (temporalChatAgentSelector) {
        temporalChatAgentSelector.addEventListener('change', async () => {
            _renderTemporalChatTranscript(temporalChatAgentSelector.value);
            if (!temporalChatAgentSelector.value) return;
            try {
                await _loadTemporalChatState(temporalChatAgentSelector.value);
                _renderTemporalChatTranscript(temporalChatAgentSelector.value);
            } catch (e) {
                showDashboardStatus(`Failed to load chat: ${e.message || e}`, "error");
            }
        });
    }

    if (branchTemporalChatButton) {
        branchTemporalChatButton.addEventListener('click', async () => {
            const agentName = temporalChatAgentSelector ? temporalChatAgentSelector.value : null;
            if (!agentName) return;
            if (_isTemporalChatProcessing(agentName) || (_getTemporalChatQueue(agentName).length > 0)) {
                showDashboardStatus("Please wait until all queued messages finish before branching chat.", "info");
                return;
            }
            const parsedBranchTick = _parseTemporalChatBranchTickInput();
            if (!parsedBranchTick.ok) {
                showDashboardStatus(parsedBranchTick.error, "error");
                return;
            }
            const existingMessages = _getTemporalChatMessages(agentName);
            if (existingMessages.length > 0 && !confirm("Branch again? This will clear the current chat for this agent.")) {
                return;
            }

            temporalChatBranchingByAgent.set(agentName, true);
            temporalChatProcessingByAgent.set(agentName, true);
            _updateTemporalChatSendButtonState(agentName);
            try {
                const payload = {};
                if (parsedBranchTick.value !== null) payload.base_tick = parsedBranchTick.value;
                const result = await fetchApi(`/api/agent/${encodeURIComponent(agentName)}/temporal_chat/refresh`, 'POST', payload);
                if (!result || !result.success || !result.chat) {
                    throw new Error((result && (result.error || result.message)) || 'Failed to branch chat.');
                }
                _applyTemporalChatState(agentName, result.chat);
                temporalChatQueueByAgent.set(agentName, []);
                _renderTemporalChatTranscript(agentName);
                if (temporalChatInput) temporalChatInput.value = "";
                const baseTick = result.chat && typeof result.chat.base_tick !== 'undefined' ? result.chat.base_tick : 'current';
                showDashboardStatus(`Chat branched from tick ${baseTick}.`, "success");
            } catch (e) {
                showDashboardStatus(`Failed to branch chat: ${e.message || e}`, "error");
            } finally {
                temporalChatBranchingByAgent.set(agentName, false);
                temporalChatProcessingByAgent.set(agentName, false);
                _updateTemporalChatSendButtonState(agentName);
            }
        });
    }

    if (copyTemporalChatButton) {
        copyTemporalChatButton.addEventListener('click', async () => {
            const agentName = temporalChatAgentSelector ? temporalChatAgentSelector.value : null;
            if (!agentName) {
                showDashboardStatus("Please select a target agent to copy the dialogue.", "error");
                return;
            }
            const text = _formatTemporalChatForCopy(agentName, true);
            if (!text.trim()) {
                showDashboardStatus("No dialogue to copy.", "info");
                return;
            }
            try {
                await navigator.clipboard.writeText(text);
                showDashboardStatus("Chat copied to clipboard.", "success");
            } catch (e) {
                showDashboardStatus("Failed to copy to clipboard.", "error");
            }
        });
    }

    if (copyTemporalChatWithoutThinkingButton) {
        copyTemporalChatWithoutThinkingButton.addEventListener('click', async () => {
            const agentName = temporalChatAgentSelector ? temporalChatAgentSelector.value : null;
            if (!agentName) {
                showDashboardStatus("Please select a target agent to copy the dialogue.", "error");
                return;
            }
            const text = _formatTemporalChatForCopy(agentName, false);
            if (!text.trim()) {
                showDashboardStatus("No dialogue to copy.", "info");
                return;
            }
            try {
                await navigator.clipboard.writeText(text);
                showDashboardStatus("Chat copied without thinking.", "success");
            } catch (e) {
                showDashboardStatus("Failed to copy to clipboard.", "error");
            }
        });
    }

    if (confirmTemporalChatSendButton) {
        confirmTemporalChatSendButton.addEventListener('click', async () => {
            if (!temporalChatAgentSelector) return;
            const agentName = temporalChatAgentSelector.value;
            const msg = temporalChatInput ? temporalChatInput.value.trim() : "";

            if (!agentName) {
                showDashboardStatus("Please select a target agent.", "error");
                return;
            }
            if (!msg) {
                showDashboardStatus("Message cannot be empty.", "error");
                return;
            }

            const q = _getTemporalChatQueue(agentName);

            // Queue immediately; append user message only when it is actually sent, for Q1 A1 Q2 A2 ordering.
            q.push(msg);
            if (temporalChatInput) temporalChatInput.value = "";
            _updateTemporalChatSendButtonState(agentName);

            // If already processing, we're done (message is queued).
            if (_isTemporalChatProcessing(agentName)) return;

            temporalChatProcessingByAgent.set(agentName, true);
            _updateTemporalChatSendButtonState(agentName);
            try {
                while (q.length > 0) {
                    const nextMsg = q.shift();
                    if (!nextMsg) continue;

                    _getTemporalChatMessages(agentName).push({ role: 'user', content: nextMsg });
                    _renderTemporalChatTranscript(agentName);

                    const result = await fetchApi(`/api/agent/${encodeURIComponent(agentName)}/temporal_chat`, 'POST', {
                        user_message: nextMsg
                    });

                    if (result && result.success) {
                        if (result.chat) {
                            _applyTemporalChatState(agentName, result.chat);
                        } else {
                            _getTemporalChatMessages(agentName).push({ role: 'assistant', content: String(result.agent_response || "") });
                        }
                    } else {
                        if (result && result.chat) _applyTemporalChatState(agentName, result.chat);
                        _getTemporalChatMessages(agentName).push({ role: 'assistant', content: String((result && (result.error || result.message)) || "Chat failed.") });
                        q.length = 0;
                    }
                    _renderTemporalChatTranscript(agentName);
                }
            } catch (error) {
                _getTemporalChatMessages(agentName).push({ role: 'assistant', content: `Error: ${error.message || 'Chat request failed.'}` });
                _renderTemporalChatTranscript(agentName);
                q.length = 0;
            } finally {
                temporalChatProcessingByAgent.set(agentName, false);
                _updateTemporalChatSendButtonState(agentName);
                updateOrchestratorControlButtons(orchestratorState);
            }
        });
    }

    if (openSpeakCommonRoomModalButton) {
        openSpeakCommonRoomModalButton.addEventListener('click', () => {
            const restoreDraft = consumeBackdropDraftFlag(speakCommonRoomModal);
            // This tool can be used regardless of orchestrator state, as it's a direct room interaction.
            // However, ensure station_instance is available on backend.
            if (!restoreDraft && commonRoomSpeakerName) commonRoomSpeakerName.value = "";
            if (!restoreDraft && commonRoomMessageContent) commonRoomMessageContent.value = "";
            if (speakCommonRoomModal) speakCommonRoomModal.style.display = "block";
            if (commonRoomSpeakerName) commonRoomSpeakerName.focus();
        });
    }
    if (closeSpeakCommonRoomModalButton) {
        closeSpeakCommonRoomModalButton.addEventListener('click', () => {
            hideModalWithoutReset(speakCommonRoomModal, true);
        });
    }

    if (confirmSpeakCommonRoomButton) {
        confirmSpeakCommonRoomButton.addEventListener('click', async () => {
            const speakerName = commonRoomSpeakerName ? commonRoomSpeakerName.value.trim() : "";
            const message = commonRoomMessageContent ? commonRoomMessageContent.value.trim() : "";

            if (!speakerName) {
                showDashboardStatus("Speaker Name cannot be empty.", "error");
                return;
            }
            if (!message) {
                showDashboardStatus("Message Content cannot be empty.", "error");
                return;
            }

            showDashboardStatus(`Submitting message from '${speakerName}' to Common Room...`, 'info', 0); // Persistent
            confirmSpeakCommonRoomButton.disabled = true;
            try {
                const result = await fetchApi('/api/room/common/speak', 'POST', {
                    speaker_name: speakerName,
                    message_content: message
                });
                showDashboardStatus(result.message, result.success ? 'success' : 'error');
                if (result.success) {
                    if (commonRoomSpeakerName) commonRoomSpeakerName.value = "";
                    if (commonRoomMessageContent) commonRoomMessageContent.value = "";
                    if (speakCommonRoomModal) speakCommonRoomModal.style.display = "none";
                }
            } catch (error) {
                // fetchApi already shows error
            } finally {
                confirmSpeakCommonRoomButton.disabled = false;
            }
        });
    }    

    function directMessageAgentStatusStartsWith(agentData, prefix) {
        return Boolean(agentData && String(agentData.status || "").startsWith(prefix));
    }

    function buildDirectMessageTargetOptions() {
        const options = [{ name: "Reviewer", label: "Reviewer" }];
        const added = new Set(["Reviewer"]);
        const turnOrder = Array.isArray(orchestratorState.turn_order) ? orchestratorState.turn_order : [];

        const addAgentOption = (agent) => {
            if (!agent || !agent.name || added.has(agent.name)) return;
            if (directMessageAgentStatusStartsWith(agent, "Ascended")) return;
            added.add(agent.name);
            options.push({
                name: agent.name,
                label: formatAgentDisplayName(agent),
            });
        };

        turnOrder.forEach(agentName => {
            const agent = fullAgentListCache.find(a => a.name === agentName);
            addAgentOption(agent);
        });

        const otherAgents = (fullAgentListCache || [])
            .filter(agent => agent && agent.name && !added.has(agent.name))
            .sort((a, b) => {
                const aEnded = directMessageAgentStatusStartsWith(a, "Session Ended");
                const bEnded = directMessageAgentStatusStartsWith(b, "Session Ended");
                if (aEnded !== bEnded) return aEnded ? 1 : -1;
                return String(a.name || "").localeCompare(String(b.name || ""));
            });
        otherAgents.forEach(addAgentOption);

        return options;
    }

    function populateDirectMessageAgentSelector(preferredAgentName) {
        if (!directMessageAgentSelector) return null;
        const options = buildDirectMessageTargetOptions();
        directMessageAgentSelector.innerHTML = "";

        if (options.length === 0) {
            const option = document.createElement('option');
            option.textContent = "No agents available.";
            option.disabled = true;
            directMessageAgentSelector.appendChild(option);
            if (confirmSendDirectMessageButton) confirmSendDirectMessageButton.disabled = true;
            return null;
        }

        options.forEach(target => {
            const option = document.createElement('option');
            option.value = target.name;
            option.textContent = target.label;
            directMessageAgentSelector.appendChild(option);
        });

        const preferredTarget = options.find(target => target.name === preferredAgentName);
        directMessageAgentSelector.value = preferredTarget ? preferredTarget.name : options[0].name;
        return directMessageAgentSelector.value;
    }

    function updateDirectMessageModalForTarget(agentName, clearDraft = false) {
        const isReviewer = agentName === "Reviewer";
        const agentData = isReviewer ? null : fullAgentListCache.find(a => a.name === agentName);
        const isEnded = directMessageAgentStatusStartsWith(agentData, "Session Ended");
        const isAscended = directMessageAgentStatusStartsWith(agentData, "Ascended");
        const isValidTarget = isReviewer || (!!agentData && !isAscended);

        if (clearDraft) {
            if (directMessageInput) directMessageInput.value = "";
            if (directMessageResponseArea) directMessageResponseArea.classList.add('hidden');
            if (directMessageLlmResponseContent) directMessageLlmResponseContent.innerHTML = "";
        }

        if (!isValidTarget) {
            if (directMessageModalTitle) directMessageModalTitle.textContent = "Intervene With Agent";
            if (directMessageModalDescription) directMessageModalDescription.textContent = "Selected agent is unavailable for intervention.";
            if (confirmSendDirectMessageButton) confirmSendDirectMessageButton.disabled = true;
            return false;
        }

        if (isReviewer) {
            if (directMessageModalTitle) directMessageModalTitle.textContent = "Intervene With Reviewer";
            if (directMessageModalDescription) directMessageModalDescription.textContent = "Your intervention below will be sent to the Reviewer system. Responses will appear in the Reviewer log.";
        } else if (isEnded) {
            if (directMessageModalTitle) directMessageModalTitle.textContent = "Intervene With Ended Agent";
            if (directMessageModalDescription) directMessageModalDescription.textContent = "This agent's session has ended, so this will not disrupt station workflow. Your message will be logged.";
        } else {
            if (directMessageModalTitle) directMessageModalTitle.textContent = "Intervene With Agent";
            if (directMessageModalDescription) directMessageModalDescription.textContent = "Your message below will be sent directly to the agent. This is highly discouraged because it can disrupt the agent's normal station workflow. Ended agents are usually fine.";
        }

        if (confirmSendDirectMessageButton) confirmSendDirectMessageButton.disabled = isDirectMessageInProgress;
        return true;
    }

    if (openDirectMessageModalButton) {
        openDirectMessageModalButton.addEventListener('click', () => {
            const restoreDraft = consumeBackdropDraftFlag(directMessageModal);
            const previousAgentName = directMessageAgentSelector ? directMessageAgentSelector.value : null;
            const selectedDashboardAgent = agentSelectorDashboard ? agentSelectorDashboard.value : null;
            const preferredAgent = restoreDraft
                ? previousAgentName
                : (selectedDashboardAgent && selectedDashboardAgent !== "all" ? selectedDashboardAgent : null);
            const selectedAgent = populateDirectMessageAgentSelector(preferredAgent);

            if (!selectedAgent) {
                showDashboardStatus("No agents are available for intervention.", "error");
                return;
            }

            updateDirectMessageModalForTarget(selectedAgent, !restoreDraft || previousAgentName !== selectedAgent);
            if (directMessageModal) directMessageModal.style.display = "block";
            if (directMessageInput) directMessageInput.focus();
        });
    }
    if (directMessageAgentSelector) {
        directMessageAgentSelector.addEventListener('change', () => {
            updateDirectMessageModalForTarget(directMessageAgentSelector.value, true);
        });
    }
    if (closeDirectMessageModalButton) closeDirectMessageModalButton.addEventListener('click', () => { hideModalWithoutReset(directMessageModal, true); });

    if (confirmSendDirectMessageButton) {
        confirmSendDirectMessageButton.addEventListener('click', async () => {
            const agentToMessage = directMessageAgentSelector ? directMessageAgentSelector.value : null;
            const messageText = directMessageInput ? directMessageInput.value.trim() : "";
            const isReviewer = agentToMessage === "Reviewer";

            if (!agentToMessage) {
                showDashboardStatus("Please select a target agent.", "error"); return;
            }
            if (!messageText) { 
                showDashboardStatus("Intervention cannot be empty.", "error"); return;
            }

            const agentData = isReviewer ? null : fullAgentListCache.find(a => a.name === agentToMessage);
            if (!isReviewer && !agentData) {
                showDashboardStatus("Selected agent not found.", "error"); return;
            }

            const isEnded = directMessageAgentStatusStartsWith(agentData, "Session Ended");
            const isAscended = directMessageAgentStatusStartsWith(agentData, "Ascended");

            if (isAscended) {
                showDashboardStatus("Cannot intervene with ascended agents.", "error"); return;
            }

            showDashboardStatus(`Sending intervention to ${agentToMessage}...`, 'info');
            confirmSendDirectMessageButton.disabled = true;
            if (openDirectMessageModalButton) openDirectMessageModalButton.disabled = true;
            isDirectMessageInProgress = true;
            
            // Close modal immediately when user clicks send
            if (directMessageModal) directMessageModal.style.display = "none";
            
            try {
                let result;
                if (isReviewer) {
                    result = await fetchApi('/api/reviewer/manual_message', 'POST', {
                        message_text: messageText
                    });

                    if (result.success) {
                        showDashboardStatus(`Intervention sent. Reviewer replied. See log.`, 'success');
                        if (directMessageInput) directMessageInput.value = "";
                    } else {
                        showDashboardStatus(result.error || "Failed to send intervention to reviewer.", "error");
                    }
                } else if (isEnded) {
                    // Send to ended agent using final chat API
                    result = await fetchApi(`/api/agent/${encodeURIComponent(agentToMessage)}/final_chat`, 'POST', {
                        human_message: messageText 
                    });
                    
                    if (result.success) {
                        showDashboardStatus(`Intervention sent. Agent ${agentToMessage} replied.`, 'success');
                        if (directMessageInput) directMessageInput.value = "";
                    } else {
                        showDashboardStatus(result.error || "Failed to message ended agent.", "error");
                    }
                } else {
                    // Send to living agent using manual message API
                    result = await fetchApi('/api/orchestrator/manual_message', 'POST', {
                        agent_name: agentToMessage, 
                        message_text: messageText, 
                        end_chat_after_send: false  // Never auto-end chat (deprecated functionality)
                    });
                    
                    if (result.success) {
                        showDashboardStatus(`Intervention sent. Agent ${agentToMessage} replied. See log.`, 'success');
                        if (directMessageInput) directMessageInput.value = "";
                    } else {
                        showDashboardStatus(result.error || "Failed to send intervention.", "error");
                    }
                }
                
            } catch (error) { 
                /* Handled by fetchApi - modal stays open on network errors so user can retry */ 
            } finally { 
                confirmSendDirectMessageButton.disabled = false;
                isDirectMessageInProgress = false;
                if (openDirectMessageModalButton) {
                    openDirectMessageModalButton.disabled = buildDirectMessageTargetOptions().length === 0;
                }
            }
        });
    }
    
    function populateResolveRequestDetails(request) {
        if (!request) return;
        if (requestIdDisplay) requestIdDisplay.textContent = request.request_id;
        if (requestTickDisplay) requestTickDisplay.textContent = request.tick;
        if (requestAgentDisplay) requestAgentDisplay.textContent = request.agent_name;
        if (requestModelDisplay) requestModelDisplay.textContent = request.agent_model;
        if (requestTitleDisplay) requestTitleDisplay.textContent = request.title;
        if (requestContentDisplay) requestContentDisplay.textContent = request.content;
        if (resolveRequestModal) {
            resolveRequestModal.dataset.requestId = request.request_id;
        }
    }

    function updateRequestSelectorOptions(requests) {
        resolveRequestCache = new Map();
        if (!requestSelector || !requestSelectorWrapper) {
            return;
        }

        requestSelector.innerHTML = '';
        if (!requests || requests.length <= 1) {
            requestSelectorWrapper.classList.add('hidden');
            return;
        }

        requestSelectorWrapper.classList.remove('hidden');
        requests.forEach((req) => {
            const option = document.createElement('option');
            option.value = String(req.request_id);
            option.textContent = `#${req.request_id} — ${req.title || 'Untitled'}`;
            requestSelector.appendChild(option);
            resolveRequestCache.set(String(req.request_id), req);
        });
    }

    if (requestSelector) {
        requestSelector.addEventListener('change', () => {
            const selectedId = String(requestSelector.value);
            const request = resolveRequestCache.get(selectedId);
            populateResolveRequestDetails(request);
        });
    }

    if(resolveHumanInterventionButton) resolveHumanInterventionButton.addEventListener('click', async () => {
        // Open modal to resolve human request for the selected agent
        const selectedAgent = agentSelectorDashboard ? agentSelectorDashboard.value : null;
        if (!selectedAgent || selectedAgent === "all") {
            showDashboardStatus("Please select a specific agent to resolve their request.", "error");
            return;
        }
        if (consumeBackdropDraftFlag(resolveRequestModal) && resolveRequestModal?.dataset.agentName === selectedAgent) {
            resolveRequestModal.style.display = "block";
            if (resolveResponseInput) resolveResponseInput.focus();
            return;
        }

        // Fetch request details
        showDashboardStatus(`Fetching request details for ${selectedAgent}...`, 'info');
        try {
            const response = await fetchApi(`/api/orchestrator/get_human_request?agent_name=${encodeURIComponent(selectedAgent)}`);
            if (!response.success) {
                showDashboardStatus(response.error || "Failed to fetch request details", "error");
                return;
            }

            // Populate modal with request details
            const requests = response.requests || (response.request ? [response.request] : []);
            if (!requests.length) {
                showDashboardStatus("No pending requests found for this agent.", "error");
                return;
            }

            requests.sort((a, b) => (b.request_id || 0) - (a.request_id || 0));
            updateRequestSelectorOptions(requests);
            const defaultRequest = requests[0];
            if (requestSelector && requestSelectorWrapper && !requestSelectorWrapper.classList.contains('hidden')) {
                requestSelector.value = String(defaultRequest.request_id);
            }
            populateResolveRequestDetails(defaultRequest);
            if (resolveResponseInput) resolveResponseInput.value = '';
            if (resolveRequestStatus) {
                resolveRequestStatus.classList.add('hidden');
                resolveRequestStatus.textContent = '';
            }

            // Store agent name for later
            if (resolveRequestModal) {
                resolveRequestModal.dataset.agentName = selectedAgent;
                resolveRequestModal.style.display = "block";
            }

            showDashboardStatus("Request details loaded", 'success');
        } catch (error) {
            showDashboardStatus("Failed to fetch request details", 'error');
        }
    });

    // Modal handlers for resolve request modal
    if (closeResolveRequestModalButton) {
        closeResolveRequestModalButton.addEventListener('click', () => {
            hideModalWithoutReset(resolveRequestModal, true);
        });
    }

    if (confirmResolveRequestButton) {
        confirmResolveRequestButton.addEventListener('click', async () => {
            const agentName = resolveRequestModal?.dataset.agentName;
            const requestId = resolveRequestModal?.dataset.requestId;
            if (!agentName) {
                if (resolveRequestStatus) {
                    resolveRequestStatus.textContent = "Error: Agent name not found";
                    resolveRequestStatus.classList.remove('hidden');
                }
                return;
            }

            // Get the optional response text
            const responseText = resolveResponseInput ? resolveResponseInput.value.trim() : '';

            // Show status
            if (resolveRequestStatus) {
                resolveRequestStatus.textContent = "Resolving request...";
                resolveRequestStatus.classList.remove('hidden');
            }

            try {
                const result = await fetchApi('/api/orchestrator/resolve_human_intervention', 'POST', {
                    agent_name: agentName,
                    request_id: requestId,
                    response_text: responseText || null,  // Send null if empty
                    reason: responseText ? "Human provided response" : "Intervention resolved by UI action."
                });

                if (resolveRequestStatus) {
                    resolveRequestStatus.textContent = result.message;
                    resolveRequestStatus.classList.remove('hidden');
                }

                if (result.success) {
                    showDashboardStatus(result.message, 'success');
                    await getOrchestratorStatus();
                    // Close modal after a short delay
                    setTimeout(() => {
                        if (resolveRequestModal) resolveRequestModal.style.display = "none";
                    }, 2000);
                } else {
                    showDashboardStatus(result.message || "Failed to resolve request", 'error');
                }
            } catch (error) {
                if (resolveRequestStatus) {
                    resolveRequestStatus.textContent = "Error resolving request";
                    resolveRequestStatus.classList.remove('hidden');
                }
            }
        });
    }

    
    function initializeDashboard() {
        applyLogDisplayMode();
        if (operationMode === 'api' || isMultistartPreview) {
            getStationVersion(); // Load version first
            loadStationConfig(); // Load station config and update top bar
            const orchestratorStatusPromise = getOrchestratorStatus();
            updateStationStatistics(); // Initial load of statistics
            renderTimeSinceLastTick();
            orchestratorStatusPromise.then(fetchAgentsForDashboard).then(() => {
                if (window.location.hash) {
                    const agentNameFromHash = window.location.hash.substring(1);
                    if (agentSelectorDashboard && Array.from(agentSelectorDashboard.options).some(opt => opt.value === agentNameFromHash)) {
                        agentSelectorDashboard.value = agentNameFromHash;
                    }
                }
                handleAgentDialogueViewChange();
                if (operationMode === 'api') {
                    connectSseLogStream();
                } else if (globalNotificationBubbleLog) {
                    addMessageToGlobalNotificationBubbleLog({
                        event: 'system_message',
                        data: {message: 'Read-only Seed 1 preview. Select an agent to inspect its durable dialogue history.'},
                        timestamp: Date.now() / 1000,
                    });
                }
            });
            applyMultistartPreviewReadOnlyState();
        } else {
            if(globalNotificationBubbleLog) addMessageToGlobalNotificationBubbleLog({event: "system_message", data: {message: "Orchestrator live log inactive in Manual Mode."}, timestamp: Date.now()/1000});
            const apiControls = [startLoopButton, pauseOrchestratorButton, resumeOrchestratorButton, stopOrchestratorButton, createApiAgentModalButton, endApiAgentSessionButton, openDirectMessageModalButton, resolveHumanInterventionButton, openResearchTaskSpecModalButton];
            apiControls.forEach(btn => { if(btn) btn.disabled = true; });
            showDashboardStatus("Station is in Manual Mode. Use the Manual Interface tab for agent interaction.", "info", 0);
        }

        setInterval(async () => {
        if (document.visibilityState === 'visible' && (operationMode === 'api' || isMultistartPreview)) {
            const orchestratorStatusData = await getOrchestratorStatus();
            updateStationStatistics(); // Periodic update of statistics
            // Always fetch agents to keep the list updated
            fetchAgentsForDashboard(orchestratorStatusData);
            }
        }, 7000); 

        setInterval(() => {
            renderTimeSinceLastTick();
        }, 1000);

        if (isMultistartPreview) {
            setInterval(() => {
                void refreshMultistartPreviewDialogue();
            }, 3 * 60 * 1000);
        }
    }

    function setTaskSpecDirty(dirty) {
        taskSpecState.dirty = !!dirty;
        if (taskSpecDirtyIndicator) taskSpecDirtyIndicator.classList.toggle('hidden', !taskSpecState.dirty);
        if (saveTaskSpecButton) saveTaskSpecButton.disabled = isMultistartPreview || !taskSpecState.dirty;
    }

    function renderTaskSpecPreview() {
        if (!taskSpecPreviewPane) return;
        const rawMarkdown = taskSpecRawEditor ? taskSpecRawEditor.value : taskSpecState.rawMarkdown;
        taskSpecPreviewPane.innerHTML = renderMarkdownForDashboard(rawMarkdown || '_No Research Task specification is available._');
        taskSpecPreviewPane.scrollTop = 0;
    }

    function showTaskSpecMode(mode) {
        const nextMode = !isMultistartPreview && mode === 'edit' ? 'edit' : 'preview';
        taskSpecState.mode = nextMode;
        const previewActive = nextMode === 'preview';
        if (taskSpecPreviewPane) taskSpecPreviewPane.classList.toggle('hidden', !previewActive);
        if (taskSpecEditPane) taskSpecEditPane.classList.toggle('hidden', previewActive);
        if (taskSpecPreviewTab) {
            taskSpecPreviewTab.classList.toggle('is-active', previewActive);
            taskSpecPreviewTab.setAttribute('aria-selected', previewActive ? 'true' : 'false');
        }
        if (taskSpecEditTab) {
            taskSpecEditTab.classList.toggle('is-active', !previewActive);
            taskSpecEditTab.setAttribute('aria-selected', previewActive ? 'false' : 'true');
        }
        if (previewActive) {
            renderTaskSpecPreview();
        } else if (taskSpecRawEditor) {
            taskSpecRawEditor.focus();
        }
    }

    function applyTaskSpecSnapshot(snapshot) {
        taskSpecState.rawMarkdown = String(snapshot.raw_markdown || '');
        taskSpecState.savedRawMarkdown = taskSpecState.rawMarkdown;
        taskSpecState.revision = String(snapshot.revision || '');
        taskSpecState.relativePath = String(snapshot.relative_path || 'rooms/research/research_task.md');
        taskSpecState.loaded = true;
        if (taskSpecRawEditor) taskSpecRawEditor.value = taskSpecState.rawMarkdown;
        if (taskSpecPath) taskSpecPath.textContent = `Path: ${taskSpecState.relativePath}`;
        if (taskSpecRevision) taskSpecRevision.textContent = `Revision: ${taskSpecState.revision.slice(0, 12) || 'unknown'}`;
        setTaskSpecDirty(false);
        showTaskSpecMode(taskSpecState.mode);
    }

    async function loadResearchTaskSpec() {
        showDashboardStatus('Loading Research Task specification...', 'info', 0);
        if (reloadTaskSpecButton) reloadTaskSpecButton.disabled = true;
        if (saveTaskSpecButton) saveTaskSpecButton.disabled = true;
        try {
            const snapshot = await fetchApi('/api/station/research_task_spec', 'GET');
            applyTaskSpecSnapshot(snapshot);
            showDashboardStatus('Research Task specification loaded.', 'success');
            return true;
        } catch (error) {
            return false;
        } finally {
            if (reloadTaskSpecButton) reloadTaskSpecButton.disabled = false;
            if (saveTaskSpecButton) saveTaskSpecButton.disabled = isMultistartPreview || !taskSpecState.dirty;
        }
    }

    async function openResearchTaskSpecModal() {
        if (!researchTaskSpecModal) return;
        const preserveDraft = consumeBackdropDraftFlag(researchTaskSpecModal);
        researchTaskSpecModal.style.display = 'block';
        if (preserveDraft && taskSpecState.loaded && taskSpecState.dirty) {
            showTaskSpecMode(taskSpecState.mode);
            return;
        }
        await loadResearchTaskSpec();
    }

    async function saveResearchTaskSpec() {
        if (!taskSpecRawEditor || !taskSpecState.loaded || !taskSpecState.revision) {
            showDashboardStatus('Load the Research Task specification before saving.', 'error');
            return false;
        }
        if (!taskSpecState.dirty) return true;
        if (!confirm('Save these changes to the active Research Task specification? New task reads and newly launched coder sessions will use the updated text immediately.')) {
            return false;
        }

        if (saveTaskSpecButton) saveTaskSpecButton.disabled = true;
        if (reloadTaskSpecButton) reloadTaskSpecButton.disabled = true;
        showDashboardStatus('Saving Research Task specification...', 'info', 0);
        try {
            const snapshot = await fetchApi('/api/station/research_task_spec', 'PUT', {
                raw_markdown: taskSpecRawEditor.value,
                expected_revision: taskSpecState.revision,
            });
            applyTaskSpecSnapshot(snapshot);
            showDashboardStatus(snapshot.message || 'Research Task specification saved.', 'success');
            return true;
        } catch (error) {
            return false;
        } finally {
            if (reloadTaskSpecButton) reloadTaskSpecButton.disabled = false;
            if (saveTaskSpecButton) saveTaskSpecButton.disabled = isMultistartPreview || !taskSpecState.dirty;
        }
    }

    if (openResearchTaskSpecModalButton) {
        openResearchTaskSpecModalButton.addEventListener('click', openResearchTaskSpecModal);
    }

    if (closeResearchTaskSpecModalButton) {
        closeResearchTaskSpecModalButton.addEventListener('click', () => {
            hideModalWithoutReset(researchTaskSpecModal, true);
        });
    }

    if (taskSpecPreviewTab) {
        taskSpecPreviewTab.addEventListener('click', () => showTaskSpecMode('preview'));
    }

    if (taskSpecEditTab) {
        taskSpecEditTab.addEventListener('click', () => showTaskSpecMode('edit'));
    }

    if (taskSpecRawEditor) {
        taskSpecRawEditor.addEventListener('input', () => {
            taskSpecState.rawMarkdown = taskSpecRawEditor.value;
            setTaskSpecDirty(taskSpecState.rawMarkdown !== taskSpecState.savedRawMarkdown);
        });
        taskSpecRawEditor.addEventListener('keydown', (event) => {
            if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
                event.preventDefault();
                saveResearchTaskSpec();
            }
        });
    }

    if (reloadTaskSpecButton) {
        reloadTaskSpecButton.addEventListener('click', async () => {
            if (taskSpecState.dirty && !confirm('Discard the unsaved task specification changes and reload from disk?')) {
                return;
            }
            await loadResearchTaskSpec();
        });
    }

    if (saveTaskSpecButton) {
        saveTaskSpecButton.addEventListener('click', saveResearchTaskSpec);
    }

    // --- Station Config Functionality ---
    async function loadStationConfig() {
        try {
            const data = await fetchApi('/api/station/config');
            if (data.success && data.config) {
                const config = data.config;
                
                // Update current value indicators
                if (currentStationStatus) currentStationStatus.textContent = `Current: ${config.station_status || '(empty)'}`;
                if (currentStationName) currentStationName.textContent = `Current: ${config.station_name || '(empty)'}`;
                if (currentStationDescription) currentStationDescription.textContent = `Current: ${config.station_description || '(empty)'}`;
                // Set read-only field
                if (updateStationId) updateStationId.value = config.station_id || 'Unknown';
                
                // Clear input fields so placeholders show (users must type to change)
                if (updateStationStatus) updateStationStatus.value = '';
                if (updateStationName) updateStationName.value = '';
                if (updateStationDescription) updateStationDescription.value = '';
                
                // Update station name in top bar
                updateTopBarStationName(config.station_name || '');
                
                return true;
            } else {
                showDashboardStatus('Failed to load station config', 'error');
                return false;
            }
        } catch (error) {
            console.error('Error loading station config:', error);
            showDashboardStatus('Error loading station config', 'error');
            return false;
        }
    }

    function updateTopBarStationName(stationName) {
        const headerRightDiv = document.getElementById('header-stats-block');
        if (!headerRightDiv) return;

        // Use cached references and get current values before DOM manipulation
        const currentVersion = stationVersionDashboard ? stationVersionDashboard.textContent : "N/A";
        const currentTick = stationTickDashboard ? stationTickDashboard.textContent : "N/A";
        const currentStatus = cachedStationStatus || "Unknown";

        if (stationName && stationName.trim()) {
            // Update page title with station name
            document.title = stationName;

            // Show version with station name and status
            headerRightDiv.innerHTML =
                `Station: <span class="font-semibold text-cyan-400">${escapeHtml(stationName)}</span> | ` +
                `Version: <span id="station-version-dashboard" class="font-semibold">${currentVersion}</span> | ` +
                `Station Tick: <span id="station-tick-dashboard" class="font-semibold">${currentTick}</span> | ` +
                `Status: <span id="station-status-dashboard" class="font-semibold">${currentStatus}</span>`;
        } else {
            // Update page title to default
            document.title = "Station";

            // Show version only with status
            headerRightDiv.innerHTML =
                `Station Version: <span id="station-version-dashboard" class="font-semibold">${currentVersion}</span> | ` +
                `Station Tick: <span id="station-tick-dashboard" class="font-semibold">${currentTick}</span> | ` +
                `Status: <span id="station-status-dashboard" class="font-semibold">${currentStatus}</span>`;
        }

        // Important: Re-cache element references after DOM manipulation
        stationTickDashboard = document.getElementById('station-tick-dashboard');
        stationVersionDashboard = document.getElementById('station-version-dashboard');
        stationStatusDashboard = document.getElementById('station-status-dashboard');
    }

    function updateStationStatusInHeader(newStatus) {
        // Update the current status variable
        cachedStationStatus = newStatus;

        // Update the DOM element if it exists
        if (stationStatusDashboard) {
            stationStatusDashboard.textContent = newStatus;
        } else {
            // If element doesn't exist yet, rebuild the header
            const headerRightDiv = document.getElementById('header-stats-block');
            if (headerRightDiv) {
                const stationNameMatch = headerRightDiv.textContent.match(/Station:\s*([^|]+)/);
                const stationName = stationNameMatch ? stationNameMatch[1].trim() : "";
                updateTopBarStationName(stationName);
            }
        }
    }

    async function updateStationConfig(formData) {
        try {
            const result = await fetchApi('/api/station/config', 'PUT', formData);
            if (result.success) {
                showDashboardStatus(result.message, 'success');
                // Update top bar with new station name (only if name was provided)
                if (formData.station_name !== undefined) {
                    updateTopBarStationName(formData.station_name);
                }
                // Update status in header if it was changed
                if (formData.station_status !== undefined) {
                    updateStationStatusInHeader(formData.station_status);
                }
                return true;
            } else {
                showDashboardStatus(result.message || 'Failed to update station config', 'error');
                return false;
            }
        } catch (error) {
            console.error('Error updating station config:', error);
            showDashboardStatus('Error updating station config', 'error');
            return false;
        }
    }

    // Event Listeners for Station Config
    if (updateStationConfigButton) {
        updateStationConfigButton.addEventListener('click', async () => {
            if (consumeBackdropDraftFlag(updateStationConfigModal)) {
                if (updateStationConfigModal) updateStationConfigModal.style.display = 'block';
                if (updateStationName) updateStationName.focus();
                return;
            }
            showDashboardStatus('Loading station configuration...', 'info');
            const loaded = await loadStationConfig();
            if (loaded && updateStationConfigModal) {
                updateStationConfigModal.style.display = 'block';
                if (updateStationName) updateStationName.focus();
            }
        });
    }

    if (closeUpdateStationConfigModalButton) {
        closeUpdateStationConfigModalButton.addEventListener('click', () => {
            hideModalWithoutReset(updateStationConfigModal, true);
        });
    }

    if (updateStationConfigForm) {
        updateStationConfigForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            
            // Get values and check if they're non-empty
            const statusValue = updateStationStatus ? updateStationStatus.value.trim() : '';
            const nameValue = updateStationName ? updateStationName.value.trim() : '';
            const descriptionValue = updateStationDescription ? updateStationDescription.value.trim() : '';
            
            // Validate that at least one field is non-blank
            if (!statusValue && !nameValue && !descriptionValue) {
                showDashboardStatus('Please fill in at least one field to update.', 'error');
                return;
            }
            
            // Build form data with only non-empty values
            const formData = {};
            if (statusValue) formData.station_status = statusValue;
            if (nameValue) formData.station_name = nameValue;
            if (descriptionValue) formData.station_description = descriptionValue;

            showDashboardStatus('Updating station configuration...', 'info');
            confirmUpdateStationConfigButton.disabled = true;
            
            const success = await updateStationConfig(formData);
            if (success && updateStationConfigModal) {
                updateStationConfigModal.style.display = 'none';
            }
            
            confirmUpdateStationConfigButton.disabled = false;
        });
    }

    // --- Runtime API Config Functionality ---
    let apiRuntimeConfigCache = null;

    function getApiRuntimeTargets(config) {
        const targets = [{ value: 'station_proxy', label: 'Station Proxy' }];
        const providers = Array.isArray(config?.providers) ? config.providers : [];
        providers.forEach((provider) => {
            targets.push({ value: `provider:${provider.id}`, label: `${provider.label || provider.id}` });
        });
        if (config?.codex?.available) {
            targets.push({ value: 'codex', label: 'Codex' });
        }
        if (config?.external_counter?.available) {
            targets.push({ value: 'external_counter', label: 'External Counter' });
        }
        return targets;
    }

    function renderApiRuntimeTargetOptions(config, preferredValue = null) {
        if (!apiRuntimeTargetSelect) return;
        const targets = getApiRuntimeTargets(config);
        const currentValue = preferredValue || apiRuntimeTargetSelect.value || targets[0]?.value || 'station_proxy';
        apiRuntimeTargetSelect.innerHTML = targets.map((target) => (
            `<option value="${escapeHtml(target.value)}">${escapeHtml(target.label)}</option>`
        )).join('');
        const hasCurrent = targets.some((target) => target.value === currentValue);
        apiRuntimeTargetSelect.value = hasCurrent ? currentValue : (targets[0]?.value || 'station_proxy');
    }

    function apiRuntimeEnvHint(envName, fallbackText = '') {
        if (envName) {
            return `<div class="text-xs text-slate-500 mb-1">OS env: <span class="font-mono">${escapeHtml(envName)}</span></div>`;
        }
        if (fallbackText) {
            return `<div class="text-xs text-slate-500 mb-1">${escapeHtml(fallbackText)}</div>`;
        }
        return '';
    }

    function apiRuntimeTextInput(id, label, value, placeholder = '', envName = '', helpText = '') {
        return `
            <div>
                <label for="${escapeHtml(id)}" class="block text-sm font-medium text-slate-400 mb-1">${escapeHtml(label)}</label>
                ${apiRuntimeEnvHint(envName)}
                <input type="text" id="${escapeHtml(id)}" value="${escapeHtml(value || '')}" placeholder="${escapeHtml(placeholder || '')}" class="api-runtime-input w-full p-2.5 bg-slate-700 border border-slate-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-slate-200">
                ${helpText ? `<div class="text-xs text-slate-500 mt-1">${escapeHtml(helpText)}</div>` : ''}
            </div>
        `;
    }

    function apiRuntimeEndpointTextField(options = {}) {
        const inputId = options.id || '';
        const inputClass = options.className || '';
        const labelFor = inputId ? ` for="${escapeHtml(inputId)}"` : '';
        const idAttr = inputId ? ` id="${escapeHtml(inputId)}"` : '';
        return `
            <div>
                <label${labelFor} class="block text-sm font-medium text-slate-400 mb-1">${escapeHtml(options.label || '')}</label>
                ${apiRuntimeEnvHint(options.envName || '')}
                <input type="text"${idAttr} value="${escapeHtml(options.value || '')}" placeholder="${escapeHtml(options.placeholder || '')}" class="${escapeHtml(inputClass)} api-runtime-input w-full p-2.5 bg-slate-700 border border-slate-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-slate-200">
                ${options.helpText ? `<div class="text-xs text-slate-500 mt-1">${escapeHtml(options.helpText)}</div>` : ''}
            </div>
        `;
    }

    function apiRuntimeApiKeyField(options = {}) {
        const currentKey = options.currentKey || {};
        const masked = currentKey?.present ? currentKey.masked : 'Not set';
        const keyEnv = options.envName || currentKey?.env || '';
        const inputId = options.inputId || '';
        const inputClass = options.inputClass || '';
        const labelFor = inputId ? ` for="${escapeHtml(inputId)}"` : '';
        const idAttr = inputId ? ` id="${escapeHtml(inputId)}"` : '';
        const currentKeyMarkup = `<span class="font-mono text-slate-300">${escapeHtml(masked)}</span>`;
        const helpText = options.newRow
            ? `Current: ${currentKeyMarkup}. New backup rows require a key.`
            : `Current: ${currentKeyMarkup}. Leave blank to keep this key.`;
        return `
            <div>
                <label${labelFor} class="block text-sm font-medium text-slate-400 mb-1">API Key</label>
                ${apiRuntimeEnvHint(keyEnv)}
                <input type="password"${idAttr} autocomplete="off" class="${escapeHtml(inputClass)} api-runtime-input w-full p-2.5 bg-slate-700 border border-slate-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-slate-200">
                <div class="text-xs text-slate-500 mt-1">${helpText}</div>
            </div>
        `;
    }

    function apiRuntimeProxyInputPair(prefix, item) {
        return `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                ${apiRuntimeTextInput(`${prefix}-http-proxy`, 'HTTP Proxy', item.http_proxy || '', 'Blank uses Station Proxy', item.http_proxy_env || '')}
                ${apiRuntimeTextInput(`${prefix}-https-proxy`, 'HTTPS Proxy', item.https_proxy || '', 'Blank uses Station Proxy', item.https_proxy_env || '')}
            </div>
        `;
    }

    function apiRuntimeEndpointFields(options = {}) {
        const baseUrl = options.baseUrl || '';
        const baseUrlEnv = options.baseUrlEnv || '';
        const baseUrlInput = options.baseUrlInput || {};
        const proxyItem = options.proxyItem || {};
        const proxyInputPrefix = options.proxyClassPrefix || options.proxyPrefix || 'api-runtime';
        const useProxyClasses = Boolean(options.proxyClassPrefix);
        return `
            ${apiRuntimeApiKeyField({
                currentKey: options.currentKey,
                envName: options.keyEnv || '',
                inputId: options.keyInputId || '',
                inputClass: options.keyInputClass || '',
                newRow: Boolean(options.newRow),
            })}
            ${apiRuntimeEndpointTextField({
                id: baseUrlInput.id || '',
                className: baseUrlInput.className || '',
                label: 'Base URL',
                value: baseUrl,
                placeholder: options.baseUrlPlaceholder || 'Blank uses provider default',
                envName: baseUrlEnv,
                helpText: options.baseUrlHelp || '',
            })}
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                ${apiRuntimeEndpointTextField({
                    id: useProxyClasses ? '' : `${proxyInputPrefix}-http-proxy`,
                    className: useProxyClasses ? `${proxyInputPrefix}-http-proxy` : '',
                    label: 'HTTP Proxy',
                    value: proxyItem.http_proxy || '',
                    placeholder: 'Blank uses Station Proxy',
                    envName: proxyItem.http_proxy_env || '',
                })}
                ${apiRuntimeEndpointTextField({
                    id: useProxyClasses ? '' : `${proxyInputPrefix}-https-proxy`,
                    className: useProxyClasses ? `${proxyInputPrefix}-https-proxy` : '',
                    label: 'HTTPS Proxy',
                    value: proxyItem.https_proxy || '',
                    placeholder: 'Blank uses Station Proxy',
                    envName: proxyItem.https_proxy_env || '',
                })}
            </div>
        `;
    }

    function findApiRuntimeProvider(providerId) {
        const providers = Array.isArray(apiRuntimeConfigCache?.providers) ? apiRuntimeConfigCache.providers : [];
        return providers.find((provider) => provider.id === providerId) || null;
    }

    function apiRuntimeBackupRow(entry = {}) {
        const existingIndex = entry.index || '';
        return `
            <div class="api-runtime-backup-row api-runtime-panel rounded-md border border-slate-700 bg-slate-900/40 p-3 space-y-3" data-existing-index="${escapeHtml(existingIndex)}">
                <div class="flex items-center justify-between gap-3">
                    <div class="text-sm text-slate-400">Backup Endpoint</div>
                    <button type="button" class="api-runtime-remove-backup text-xs bg-slate-700 hover:bg-slate-600 py-1 px-3 rounded-md">Remove</button>
                </div>
                ${apiRuntimeEndpointFields({
                    currentKey: entry.api_key,
                    keyInputClass: 'api-runtime-backup-key',
                    newRow: !existingIndex,
                    baseUrl: entry.base_url || '',
                    baseUrlInput: { className: 'api-runtime-backup-base-url' },
                    proxyItem: entry,
                    proxyClassPrefix: 'api-runtime-backup',
                })}
            </div>
        `;
    }

    function apiRuntimeBackupBlock(provider) {
        const entries = Array.isArray(provider.backup_endpoints) ? provider.backup_endpoints : [];
        const rows = entries.map((entry) => apiRuntimeBackupRow(entry)).join('');
        return `
            <div class="api-runtime-panel rounded-md border border-slate-700 bg-slate-900/40 p-3 space-y-3">
                <div class="flex items-center justify-between gap-3">
                    <div>
                        <div class="text-sm text-slate-400">Backup Endpoints</div>
                        <div class="text-xs text-slate-500">Used after provider call failures. Blank Base URL uses provider default; blank proxy uses Station Proxy.</div>
                    </div>
                    <button type="button" id="api-runtime-add-backup" class="text-xs bg-slate-700 hover:bg-slate-600 py-1 px-3 rounded-md">Add Backup</button>
                </div>
                <div id="api-runtime-backup-list" class="space-y-3">${rows}</div>
            </div>
        `;
    }

    function attachApiRuntimeBackupHandlers() {
        const addButton = document.getElementById('api-runtime-add-backup');
        const list = document.getElementById('api-runtime-backup-list');
        if (addButton && list) {
            addButton.addEventListener('click', () => {
                list.insertAdjacentHTML('beforeend', apiRuntimeBackupRow({}));
            });
            list.addEventListener('click', (event) => {
                const button = event.target.closest?.('.api-runtime-remove-backup');
                if (!button) return;
                const row = button.closest('.api-runtime-backup-row');
                if (row) row.remove();
            });
        }
    }

    function renderApiRuntimeStationProxy(config) {
        const proxy = config?.station_proxy || {};
        return `
            ${apiRuntimeProxyInputPair('api-runtime', proxy)}
            <div class="text-xs text-slate-500">Leave proxy fields blank to use no Station proxy.</div>
        `;
    }

    function apiRuntimeBackupEnvText(provider) {
        if (!provider.backup_api_key_env) return '';
        return `Add BACKUP_ prefix to the OS env names above to initialize backup settings; use ';' as delimiter.`;
    }

    function renderApiRuntimeDefaultProvider(provider) {
        const fallbackText = provider.fallback_configured
            ? `${provider.backup_endpoint_count || 0} backup endpoint(s) configured.`
            : 'No backup endpoints configured.';
        const backupEnvText = apiRuntimeBackupEnvText(provider);
        return `
            <div class="space-y-3">
                ${apiRuntimeEndpointFields({
                    currentKey: provider.api_key,
                    keyEnv: provider.api_key_env || '',
                    keyInputId: 'api-runtime-new-key',
                    baseUrl: provider.base_url || '',
                    baseUrlEnv: provider.base_url_env || '',
                    baseUrlInput: { id: 'api-runtime-base-url' },
                    baseUrlPlaceholder: provider.default_base_url || 'Blank uses provider default',
                    baseUrlHelp: 'Leave blank to clear this custom Base URL and use the official/default Base URL.',
                    proxyItem: provider,
                    proxyPrefix: 'api-runtime',
                })}
            </div>
            <div class="text-xs text-slate-500">${escapeHtml(fallbackText)} ${escapeHtml(backupEnvText)}</div>
            ${apiRuntimeBackupBlock(provider)}
        `;
    }

    function renderApiRuntimeCodex(config) {
        const codex = config?.codex || {};
        return `
            <div class="text-xs text-slate-500">If the API key and Base URL are left blank, Codex will use the default options from <span class="font-mono">.codex</span>.</div>
            ${apiRuntimeEndpointFields({
                currentKey: codex.api_key,
                keyEnv: codex.api_key_env || '',
                keyInputId: 'api-runtime-new-key',
                baseUrl: codex.base_url || '',
                baseUrlEnv: codex.base_url_env || '',
                baseUrlInput: { id: 'api-runtime-base-url' },
                baseUrlPlaceholder: codex.default_base_url || 'Blank uses provider default',
                proxyItem: codex,
                proxyPrefix: 'api-runtime-codex',
            })}
        `;
    }

    function renderApiRuntimeExternalCounter(config) {
        const external = config?.external_counter || {};
        return `
            ${apiRuntimeEndpointFields({
                currentKey: external.api_key,
                keyEnv: external.api_key_env || '',
                keyInputId: 'api-runtime-new-key',
                baseUrl: external.base_url || '',
                baseUrlEnv: external.base_url_env || '',
                baseUrlInput: { id: 'api-runtime-base-url' },
                baseUrlPlaceholder: external.default_base_url || 'Blank uses provider default',
                baseUrlHelp: 'Leave blank to clear this custom Base URL and use the official/default Base URL.',
                proxyItem: external,
                proxyPrefix: 'api-runtime',
            })}
        `;
    }

    function renderApiRuntimeConfigBody() {
        if (!apiRuntimeConfigBody || !apiRuntimeTargetSelect || !apiRuntimeConfigCache) return;
        const target = apiRuntimeTargetSelect.value;
        if (target === 'station_proxy') {
            apiRuntimeConfigBody.innerHTML = renderApiRuntimeStationProxy(apiRuntimeConfigCache);
            return;
        }
        if (target.startsWith('provider:')) {
            const providerId = target.split(':')[1];
            const provider = findApiRuntimeProvider(providerId);
            if (!provider) {
                apiRuntimeConfigBody.innerHTML = `<div class="text-sm text-rose-300">Provider not found.</div>`;
                return;
            }
            apiRuntimeConfigBody.innerHTML = renderApiRuntimeDefaultProvider(provider);
            attachApiRuntimeBackupHandlers();
            return;
        }
        if (target === 'codex') {
            apiRuntimeConfigBody.innerHTML = renderApiRuntimeCodex(apiRuntimeConfigCache);
            return;
        }
        if (target === 'external_counter') {
            apiRuntimeConfigBody.innerHTML = renderApiRuntimeExternalCounter(apiRuntimeConfigCache);
            return;
        }
        apiRuntimeConfigBody.innerHTML = '';
    }

    async function loadApiRuntimeConfig(preferredTarget = null) {
        const data = await fetchApi('/api/station/api_runtime_config');
        if (!data.success || !data.config) {
            showDashboardStatus(data.error || 'Failed to load API settings', 'error');
            return false;
        }
        apiRuntimeConfigCache = data.config;
        renderApiRuntimeTargetOptions(apiRuntimeConfigCache, preferredTarget);
        renderApiRuntimeConfigBody();
        return true;
    }

    function readApiRuntimeKeyPayload(payload) {
        const newKeyInput = document.getElementById('api-runtime-new-key');
        const newKey = newKeyInput ? newKeyInput.value.trim() : '';
        if (newKey) payload.api_key = newKey;
    }

    function validateApiRuntimeField(value, label) {
        if (value.includes(';')) {
            throw new Error(`${label} cannot contain semicolons. Add another backup row instead.`);
        }
        return value;
    }

    function readApiRuntimeBackupPayload() {
        const rows = Array.from(document.querySelectorAll('.api-runtime-backup-row'));
        return rows.map((row, index) => {
            const existingIndexRaw = row.dataset.existingIndex || '';
            const existingIndex = existingIndexRaw ? parseInt(existingIndexRaw, 10) : 0;
            const apiKey = validateApiRuntimeField(row.querySelector('.api-runtime-backup-key')?.value.trim() || '', `Backup #${index + 1} API key`);
            const baseUrl = validateApiRuntimeField(row.querySelector('.api-runtime-backup-base-url')?.value.trim() || '', `Backup #${index + 1} Base URL`);
            const httpProxy = validateApiRuntimeField(row.querySelector('.api-runtime-backup-http-proxy')?.value.trim() || '', `Backup #${index + 1} HTTP Proxy`);
            const httpsProxy = validateApiRuntimeField(row.querySelector('.api-runtime-backup-https-proxy')?.value.trim() || '', `Backup #${index + 1} HTTPS Proxy`);
            if (!apiKey && !existingIndex) {
                throw new Error(`Backup #${index + 1} API key is required.`);
            }
            return {
                existing_index: existingIndex || undefined,
                api_key: apiKey,
                base_url: baseUrl,
                http_proxy: httpProxy,
                https_proxy: httpsProxy,
            };
        });
    }

    function buildApiRuntimePayload() {
        const target = apiRuntimeTargetSelect ? apiRuntimeTargetSelect.value : 'station_proxy';
        if (target === 'station_proxy') {
            return {
                target: 'station_proxy',
                http_proxy: document.getElementById('api-runtime-http-proxy')?.value || '',
                https_proxy: document.getElementById('api-runtime-https-proxy')?.value || '',
            };
        }
        if (target.startsWith('provider:')) {
            const providerId = target.split(':')[1];
            const provider = findApiRuntimeProvider(providerId);
            const payload = {
                target: 'provider',
                provider: providerId,
                base_url: document.getElementById('api-runtime-base-url')?.value || '',
                http_proxy: document.getElementById('api-runtime-http-proxy')?.value || '',
                https_proxy: document.getElementById('api-runtime-https-proxy')?.value || '',
            };
            readApiRuntimeKeyPayload(payload);
            payload.backup_endpoints = readApiRuntimeBackupPayload();
            return payload;
        }
        if (target === 'codex') {
            const payload = {
                target: 'codex',
                base_url: document.getElementById('api-runtime-base-url')?.value || '',
                http_proxy: document.getElementById('api-runtime-codex-http-proxy')?.value || '',
                https_proxy: document.getElementById('api-runtime-codex-https-proxy')?.value || '',
            };
            readApiRuntimeKeyPayload(payload);
            return payload;
        }
        if (target === 'external_counter') {
            const payload = {
                target: 'external_counter',
                base_url: document.getElementById('api-runtime-base-url')?.value || '',
                http_proxy: document.getElementById('api-runtime-http-proxy')?.value || '',
                https_proxy: document.getElementById('api-runtime-https-proxy')?.value || '',
            };
            readApiRuntimeKeyPayload(payload);
            return payload;
        }
        throw new Error('Unknown API settings target.');
    }

    if (openApiRuntimeConfigModalButton) {
        openApiRuntimeConfigModalButton.addEventListener('click', async () => {
            if (consumeBackdropDraftFlag(apiRuntimeConfigModal)) {
                if (apiRuntimeConfigModal) apiRuntimeConfigModal.style.display = 'block';
                if (apiRuntimeTargetSelect) apiRuntimeTargetSelect.focus();
                return;
            }
            showDashboardStatus('Loading API settings...', 'info');
            const loaded = await loadApiRuntimeConfig();
            if (loaded && apiRuntimeConfigModal) {
                apiRuntimeConfigModal.style.display = 'block';
                if (apiRuntimeTargetSelect) apiRuntimeTargetSelect.focus();
            }
        });
    }

    if (apiRuntimeTargetSelect) {
        apiRuntimeTargetSelect.addEventListener('change', renderApiRuntimeConfigBody);
    }

    if (closeApiRuntimeConfigModalButton) {
        closeApiRuntimeConfigModalButton.addEventListener('click', () => {
            hideModalWithoutReset(apiRuntimeConfigModal, true);
        });
    }

    if (apiRuntimeConfigForm) {
        apiRuntimeConfigForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const selectedTarget = apiRuntimeTargetSelect ? apiRuntimeTargetSelect.value : null;
            let payload;
            try {
                payload = buildApiRuntimePayload();
            } catch (error) {
                showDashboardStatus(error.message || 'Invalid API settings form.', 'error');
                return;
            }
            if (saveApiRuntimeConfigButton) saveApiRuntimeConfigButton.disabled = true;
            showDashboardStatus('Updating API settings...', 'info');
            try {
                const result = await fetchApi('/api/station/api_runtime_config', 'PUT', payload);
                if (result.success && result.config) {
                    apiRuntimeConfigCache = result.config;
                    renderApiRuntimeTargetOptions(apiRuntimeConfigCache, selectedTarget);
                    renderApiRuntimeConfigBody();
                    showDashboardStatus(result.message || 'API settings updated.', 'success');
                } else {
                    showDashboardStatus(result.error || result.message || 'Failed to update API settings.', 'error');
                }
            } catch (error) {
                // fetchApi already reports the error.
            } finally {
                if (saveApiRuntimeConfigButton) saveApiRuntimeConfigButton.disabled = false;
            }
        });
    }

    // --- Backup Functionality ---
    const createBackupButton = document.getElementById('create-backup-button');
    
    function createBackup() {
        if (!createBackupButton) return;
        
        showDashboardStatus('Creating backup...', 'info');
        createBackupButton.disabled = true;
        createBackupButton.classList.remove('bg-orange-600', 'hover:bg-orange-500');
        createBackupButton.classList.add('bg-gray-600');
        
        const basePath = window.STATION_CONFIG && window.STATION_CONFIG.apiBasePath || '';
        fetch(basePath + '/api/backup/create', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showDashboardStatus('Backup created successfully', 'success');
            } else {
                showDashboardStatus(`Backup failed: ${data.message}`, 'error');
            }
        })
        .catch(error => {
            console.error('Backup error:', error);
            showDashboardStatus('Backup failed: Network error', 'error');
        })
        .finally(() => {
            createBackupButton.disabled = false;
            createBackupButton.classList.remove('bg-gray-600');
            createBackupButton.classList.add('bg-orange-600', 'hover:bg-orange-500');
        });
    }
    
    if (createBackupButton) {
        createBackupButton.addEventListener('click', createBackup);
        // Set initial style
        createBackupButton.classList.add('bg-orange-600', 'hover:bg-orange-500');
    }

    initializeDashboard();
});
