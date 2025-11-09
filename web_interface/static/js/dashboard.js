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
    // --- DOM Elements ---
    let stationTickDashboard = document.getElementById('station-tick-dashboard');
    let stationVersionDashboard = document.getElementById('station-version-dashboard');
    let stationStatusDashboard = null; // Will be created dynamically in header
    let cachedStationStatus = "Unknown"; // Track current station status string value
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
    const cancelCreateApiAgentButton = document.getElementById('cancel-create-api-agent-button');
    
    const manualInteractionSidebarSection = document.getElementById('manual-interaction-sidebar-section');
    const manualInteractionTitle = document.getElementById('manual-interaction-title');
    const manualInteractionSelectedAgentDisplay = document.getElementById('manual-interaction-selected-agent-display');
    
    const openDirectMessageModalButton = document.getElementById('open-direct-message-modal-button');
    const resolveHumanInterventionButton = document.getElementById('resolve-human-intervention-button');

    const directMessageModal = document.getElementById('direct-message-modal');
    const closeDirectMessageModalButton = document.getElementById('close-direct-message-modal-button');
    const cancelDirectMessageButton = document.getElementById('cancel-direct-message-button');
    const directMessageModalTitle = document.getElementById('direct-message-modal-title');
    const directMessageModalAgentName = document.getElementById('direct-message-modal-agent-name');
    const directMessageModalDescription = document.getElementById('direct-message-modal-description');
    const directMessageInput = document.getElementById('direct-message-input');
    const confirmSendDirectMessageButton = document.getElementById('confirm-send-direct-message-button');
    const directMessageResponseArea = document.getElementById('direct-message-response-area');
    const directMessageLlmResponseContent = document.getElementById('direct-message-llm-response-content');

    // Human Request Resolution Modal Elements
    const resolveRequestModal = document.getElementById('resolve-request-modal');
    const closeResolveRequestModalButton = document.getElementById('close-resolve-request-modal-button');
    const cancelResolveRequestButton = document.getElementById('cancel-resolve-request-button');
    const confirmResolveRequestButton = document.getElementById('confirm-resolve-request-button');
    const requestIdDisplay = document.getElementById('request-id-display');
    const requestTickDisplay = document.getElementById('request-tick-display');
    const requestAgentDisplay = document.getElementById('request-agent-display');
    const requestModelDisplay = document.getElementById('request-model-display');
    const requestTitleDisplay = document.getElementById('request-title-display');
    const requestContentDisplay = document.getElementById('request-content-display');
    const resolveResponseInput = document.getElementById('resolve-response-input');
    const resolveRequestStatus = document.getElementById('resolve-request-status');

    const openSendSystemMessageModalButton = document.getElementById('open-send-system-message-modal-button');
    const sendSystemMessageModal = document.getElementById('send-system-message-modal');
    const closeSendSystemMessageModalButton = document.getElementById('close-send-system-message-modal-button');
    const cancelSendSystemMessageButton = document.getElementById('cancel-send-system-message-button');
    const systemMessageAgentSelector = document.getElementById('system-message-agent-selector');
    const systemMessageContent = document.getElementById('system-message-content');
    const confirmSendSystemMessageButton = document.getElementById('confirm-send-system-message-button');

    const openSpeakCommonRoomModalButton = document.getElementById('open-speak-common-room-modal-button');
    const speakCommonRoomModal = document.getElementById('speak-common-room-modal');
    const closeSpeakCommonRoomModalButton = document.getElementById('close-speak-common-room-modal-button');
    const cancelSpeakCommonRoomButton = document.getElementById('cancel-speak-common-room-button');
    const commonRoomSpeakerName = document.getElementById('common-room-speaker-name');
    const commonRoomMessageContent = document.getElementById('common-room-message-content');
    const confirmSpeakCommonRoomButton = document.getElementById('confirm-speak-common-room-button');

    const updateStationConfigButton = document.getElementById('update-station-config-button');
    const updateStationConfigModal = document.getElementById('update-station-config-modal');
    const closeUpdateStationConfigModalButton = document.getElementById('close-update-station-config-modal-button');
    const cancelUpdateStationConfigButton = document.getElementById('cancel-update-station-config-button');
    const updateStationConfigForm = document.getElementById('update-station-config-form');
    const updateStationId = document.getElementById('update-station-id');
    const updateStationStatus = document.getElementById('update-station-status');
    const updateStationName = document.getElementById('update-station-name');
    const updateStationDescription = document.getElementById('update-station-description');
    const confirmUpdateStationConfigButton = document.getElementById('confirm-update-station-config-button');
    const currentStationStatus = document.getElementById('current-station-status');
    const currentStationName = document.getElementById('current-station-name');
    const currentStationDescription = document.getElementById('current-station-description');

    const globalNotificationBubbleLog = document.getElementById('global-notification-bubble-log');
    const agentSpecificBubbleLog = document.getElementById('agent-specific-bubble-log');
    const logViewTitle = document.getElementById('log-view-title');

    const clearLogButton = document.getElementById('clear-log-button');
    const loadFullHistoryButton = document.getElementById('load-full-history-button');
    const dashboardStatusMessages = document.getElementById('dashboard-status-messages');

    const createApiAgentModal = document.getElementById('create-api-agent-modal');
    const closeApiModalButton = document.getElementById('close-api-modal-button');
    const createApiAgentForm = document.getElementById('create-api-agent-form');
    const apiAgentTypeSelect = document.getElementById('api-agent-type');
    const apiRecursiveFieldsDiv = document.getElementById('api-recursive-fields');
    const apiModelPreset = document.getElementById('api-model-preset');

    let sseSource = null;
    let currentSelectedAgentForDialogueView = "all"; 
    let isLoadingHistoryForAgent = null;
    let orchestratorState = { is_prepared: false, is_running: false, is_paused: false, agents_awaiting_human: [], turn_order: [] };
    let fullAgentListCache = [];
    let fullDialogueHistoryCache = {};
    let isDirectMessageInProgress = false; // Track if a direct message is currently being sent 

    // --- Helper Functions ---
    function showDashboardStatus(message, type = 'info', duration = 4000) {
        if (!dashboardStatusMessages) { console.error("dashboardStatusMessages element not found"); return; }
        dashboardStatusMessages.textContent = message;
        dashboardStatusMessages.className = 'mt-auto p-3 rounded-md text-sm min-h-[50px] border transition-all duration-300 opacity-100'; 
        const typeClasses = {
            error: ['bg-red-700', 'text-red-100', 'border-red-600'],
            success: ['bg-emerald-700', 'text-emerald-100', 'border-emerald-600'],
            info: ['bg-sky-700', 'text-sky-100', 'border-sky-600']
        };
        dashboardStatusMessages.classList.add(...(typeClasses[type] || typeClasses.info));
        if (duration > 0) {
            setTimeout(() => { 
                if (dashboardStatusMessages) dashboardStatusMessages.classList.add('opacity-0'); 
            }, duration);
        }
    }

    function renderMarkdownForDashboard(markdownText) {
        if (window.marked && typeof window.marked.parse === 'function') {
            try { 
                return marked.parse(String(markdownText || ""), { gfm: true, breaks: true, pedantic: false, smartypants: false }); 
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

    function formatAgentDisplayName(agent) {
        const modelName = agent.model_name || "Unknown Model";
        
        // Handle ended/ascended agents
        let suffix = '';
        if (agent.status.startsWith("Session Ended")) {
            suffix = '; ended';
        } else if (agent.status.startsWith("Ascended")) {
            suffix = '; ascended';
        }
        
        return `${agent.name} (${modelName}${suffix})`;
    }

    function createLogBubble(bubbleContentHtml, rawTextForCopy, bubbleStyle) {
        const bubbleWrapper = document.createElement('div');
        bubbleWrapper.classList.add('chat-bubble', 'text-sm', ...bubbleStyle.split(' '));
        bubbleWrapper.innerHTML = bubbleContentHtml;

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
        const timeStr = new Date(timestamp * 1000).toLocaleTimeString();
        let titlePrefix = `<strong>[${eventType.replace(/_/g, ' ').toUpperCase()} @ ${timeStr}]</strong>`;
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
            const bubbleWrapper = createLogBubble(bubbleContentHtml, rawTextForCopy, bubbleStyle);
            
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
    function addMessageToAgentBubbleLog(eventData, isHistorical = false) {
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
        
        const agentDialogueEventTypes = [
            'observation', 'submission', 'internal_prompt', 'internal_response', 
            'llm_event', 
            'submission_outcome', 'internal_action_event', 'internal_outcome', 'internal_completion',
            'manual_message_to_agent_llm', 'manual_llm_response_to_human', // These are effective types after mapping in SSE
            'final_message_to_agent', 'final_agent_response_to_human', // These are effective types after mapping in SSE
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
            isRelevantDialogueEvent = ['observation', 'submission', 'internal_prompt', 'internal_response', 'response'].includes(data.type);
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
        const timeStr = new Date(timestamp * 1000).toLocaleTimeString();
        let titlePrefix = `<strong>[${eventType.replace(/_/g, ' ').toUpperCase()} @ ${timeStr}]</strong>`; 
        let bubbleStyle = 'chat-bubble-system text-xs';

        let thinkingHtml = "";
        const thinkingContent = data.thinking_text || data.historical_thinking_text; 
        if (thinkingContent && thinkingContent.trim() !== "") {
            thinkingHtml = `
                <div class="thinking-block border-l-4 border-sky-700 bg-slate-800/50 p-2 mb-2 text-slate-400 text-xs italic rounded">
                    <strong class="text-sky-500 block mb-1">Thinking:</strong>
                    <div class="whitespace-pre-wrap">${escapeHtml(thinkingContent)}</div>
                </div>
            `;
             rawTextForCopy = `Thinking:\n${thinkingContent}\n\nResponse:\n${(data.text_content || data.content || "")}`;
        }
        

        switch (eventType) {
            case 'system_message': 
                const systemMessageContent = String(data.message || data.content || "");
                titlePrefix = `<strong>[SYSTEM @ ${timeStr}] (Agent: ${escapeHtml(data.agent_name || currentSelectedAgentForDialogueView)})</strong>`;
                bubbleContentHtml = `${titlePrefix} <div class="mt-1 text-sm markdown-content-host">${renderMarkdownForDashboard(systemMessageContent)}</div>`;
                rawTextForCopy = systemMessageContent;
                break;
            case 'agent_event': 
                 if (['actions_executed', 'submit_error'].includes(data.type)) {
                    bubbleStyle = 'chat-bubble-system text-purple-300';
                    let agentEventHeader = `<strong>[AGENT EVENT @ ${timeStr}] (Agent: ${escapeHtml(data.agent_name)}, Tick ${escapeHtml(String(data.tick))})</strong> - ${escapeHtml(data.type.replace(/_/g, ' '))}`;
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
            case 'manual_message_to_agent_llm':
            case 'manual_llm_response_to_human':
            case 'final_message_to_agent':
            case 'final_agent_response_to_human':
            case 'historical_log':
                let dialogueMarkdownContent = String(data.text_content || data.content || "N/A (Content missing for Markdown rendering)");
                if (!thinkingHtml) { 
                    rawTextForCopy = dialogueMarkdownContent;
                }

                let speaker = data.speaker || "Unknown";
                let direction = data.direction; 

                if (isHistorical || !direction) { 
                    if (['observation', 'internal_prompt', 'llm_event', 'historical_log'].includes(eventType) && (data.type === 'internal_prompt' || speaker === 'Station')) {
                        speaker = data.speaker || "Station"; direction = 'to_llm';
                    } else if (['submission', 'internal_response', 'llm_event', 'historical_log'].includes(eventType) && (data.type === 'internal_response' || data.type === 'response')) {
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
                }

                let headerText = `<strong>[${escapeHtml(speaker)} @ ${timeStr}] (Tick ${escapeHtml(String(data.tick || 'N/A'))})</strong>`;
                if (eventType === 'llm_event' && data.type && data.type !== 'response' && 
                    (data.type === 'internal_prompt' || data.type === 'internal_response')) {
                     headerText += ` - ${escapeHtml(data.type.replace(/_/g, ' '))}`;
                }
                
                bubbleContentHtml = `${headerText}${thinkingHtml}<div class="mt-1 text-sm markdown-content-host">${renderMarkdownForDashboard(dialogueMarkdownContent)}</div>`;
                
                bubbleStyle = (direction === 'to_llm') ? 'chat-bubble-station' : 'chat-bubble-agent';
                break;

            case 'submission_outcome': 
                 bubbleStyle = 'chat-bubble-system';
                 titlePrefix = `<strong>[SUBMISSION OUTCOME @ ${timeStr}] (Agent: ${escapeHtml(data.agent_name || 'N/A')}, Tick ${escapeHtml(String(data.tick || 'N/A'))})</strong>`;
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
                 titlePrefix = `<strong>[INTERNAL ACTION @ ${timeStr}] (Agent: ${escapeHtml(data.agent_name)}, Tick ${escapeHtml(String(data.tick))})</strong>`;
                 let internalStatusMsg = `Status: ${escapeHtml(data.status.toUpperCase())}`;
                 if(data.handler) internalStatusMsg += ` | Handler: ${escapeHtml(data.handler)}. `;
                 
                 let internalContentDetailMarkdown = "";
                 let liveInternalThinkingHtml = "";
                 if (!isHistorical && data.thinking_text && data.thinking_text.trim() !== "") {
                    liveInternalThinkingHtml = `
                        <div class="thinking-block border-l-4 border-sky-700 bg-slate-800/50 p-2 mb-2 text-slate-400 text-xs italic rounded">
                            <strong class="text-sky-500 block mb-1">Thinking (Internal):</strong>
                            <div class="whitespace-pre-wrap">${escapeHtml(data.thinking_text)}</div>
                        </div>
                    `;
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
                let internalOutcomeHeader = `<strong>[STATION (Internal) @ ${timeStr}] (Agent: ${escapeHtml(data.agent_name)}, Tick ${escapeHtml(String(data.tick))})</strong>`;
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
                titlePrefix = `<strong>[ERROR @ ${timeStr}] (Agent: ${escapeHtml(data.agent_name || currentSelectedAgentForDialogueView)})</strong>`;
                bubbleContentHtml = `${titlePrefix} <div class="mt-1 text-sm markdown-content-host">${renderMarkdownForDashboard(errorMessageContent)}</div>`;
                rawTextForCopy = `ERROR: ${errorMessageContent}`;
                break;
            default: 
                console.warn(`Unhandled eventType in addMessageToAgentBubbleLog: ${eventType}`, eventData);
                return; 
        }
        
        const bubbleWrapper = createLogBubble(bubbleContentHtml, rawTextForCopy, bubbleStyle);
        
        agentSpecificBubbleLog.appendChild(bubbleWrapper);

        if (isHistorical && agentSpecificBubbleLog.lastChild === bubbleWrapper) {
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

    function updateOrchestratorControlButtons(status) {
        if (!status) { console.warn("updateOrchestratorControlButtons called with null status"); return; }
        orchestratorState = status; 

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
                endApiAgentSessionButton.textContent = "End Selected Agent";
            }
        }
        
        const selectedAgentName = agentSelectorDashboard ? agentSelectorDashboard.value : "all";
        const selectedAgentData = fullAgentListCache.find(a => a.name === selectedAgentName)

        // Update Direct Message button (unified for living and ended agents)
        if (openDirectMessageModalButton) {
            const isSpecificAgentSelected = selectedAgentName && selectedAgentName !== "all" && selectedAgentData;
            if (isSpecificAgentSelected && !isDirectMessageInProgress) {
                const isAscended = selectedAgentData.status.startsWith("Ascended");
                const canSendDirectMessage = !isAscended; // Can message any agent except ascended
                openDirectMessageModalButton.disabled = !canSendDirectMessage;
            } else {
                openDirectMessageModalButton.disabled = true;
            }
        }

        // Update Resolve Request button visibility (only show when selected agent has pending request)
        if (resolveHumanInterventionButton) {
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

    async function updateStationStatistics() {
        try {
            const data = await fetchApi('/api/station/statistics');
            if (data.success && data.statistics) {
                const stats = data.statistics;
                
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
                } else if (detailsElement) {
                    detailsElement.classList.add('hidden');
                }
                
                // Update running experiments
                const runningCount = stats.running_experiments_count || 0;
                const runningExperiments = stats.running_experiments || [];
                const countElementRunning = document.getElementById('running-experiments-count');
                const detailsElementRunning = document.getElementById('running-experiments-details');
                
                if (countElementRunning) {
                    countElementRunning.textContent = runningCount > 0 ? runningCount : '0';
                }
                
                if (detailsElementRunning && runningCount > 0) {
                    let detailsHtml = '';
                    runningExperiments.forEach((exp, index) => {
                        if (index > 0) detailsHtml += '<br>';
                        const elapsedMin = Math.floor(exp.elapsed_seconds / 60);
                        const elapsedSec = exp.elapsed_seconds % 60;
                        detailsHtml += `ID ${exp.evaluation_id}: ${exp.title} (${exp.agent_name}, ${elapsedMin}m${elapsedSec}s)`;
                    });
                    detailsElementRunning.innerHTML = detailsHtml;
                    detailsElementRunning.classList.remove('hidden');
                } else if (detailsElementRunning) {
                    detailsElementRunning.classList.add('hidden');
                }
                
                // Update top research submission
                const topSubmission = stats.top_research_submission;
                const scoreElement = document.getElementById('top-research-score');
                const detailsElementResearch = document.getElementById('top-research-details');
                
                // Check for both undefined and null to handle no-score mode
                if (scoreElement && topSubmission && topSubmission !== null) {
                    scoreElement.textContent = `Score: ${topSubmission.score}`;
                    if (detailsElementResearch) {
                        detailsElementResearch.innerHTML = 
                            `ID: ${topSubmission.evaluation_id}<br>` +
                            `Title: ${topSubmission.title}<br>` +
                            `Agent: ${topSubmission.agent_name}<br>` +
                            `Task: ${topSubmission.task_id}<br>` +
                            `Tick: ${topSubmission.submitted_tick}`;
                        detailsElementResearch.classList.remove('hidden');
                    }
                } else if (scoreElement) {
                    scoreElement.textContent = '-';
                    if (detailsElementResearch) {
                        detailsElementResearch.classList.add('hidden');
                    }
                }
            }
        } catch (error) {
            console.error('Error fetching station statistics:', error);
        }
    }

    async function getOrchestratorStatus() {
        try {
            const data = await fetchApi('/api/orchestrator/status');
            if (data.success && data.status) {
                const status = data.status;
                if(orchestratorStatusDot) orchestratorStatusDot.className = 'status-dot'; 
                if (status.is_running) {
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
                if (status.is_waiting) {
                    reasonText = Object.values(status.waiting_reasons || {}).join(', ');
                } else if (status.is_paused) {
                    reasonText = status.pause_reason || "Manual Pause";
                }
                if(orchestratorPauseReasonDisplay) orchestratorPauseReasonDisplay.textContent = reasonText;
                
                // Update next agent display
                if(orchestratorNextAgentDisplay) {
                    if (status.is_running && !status.is_paused && !status.is_waiting && status.next_agent_to_act !== "N/A") {
                        orchestratorNextAgentDisplay.textContent = `Next: ${status.next_agent_to_act} (Index: ${status.next_agent_index})`;
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
            } else {
                showDashboardStatus(data.error || 'Failed to get orchestrator status.', 'error');
                if(orchestratorStatusDisplay) orchestratorStatusDisplay.textContent = "Error";
                if(orchestratorStatusDot) orchestratorStatusDot.className = 'status-dot status-error';
            }
        } catch (error) { 
            if(orchestratorStatusDisplay) orchestratorStatusDisplay.textContent = "Offline";
            if(orchestratorStatusDot) orchestratorStatusDot.className = 'status-dot status-error';
        }
    }

    async function fetchAgentsForDashboard() {
         try {
            const data = await fetchApi('/api/agents');
            const orchestratorStatusData = await fetchApi('/api/orchestrator/status');
            
            if (data.success && agentSelectorDashboard) {
                fullAgentListCache = data.agents;
                const currentTurnOrder = orchestratorStatusData.success ? orchestratorStatusData.status.turn_order : [];
                
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
            }
        } catch (error) { console.error("Error fetching and populating agents:", error); }
    }

    async function handleAgentDialogueViewChange(loadFullHistory = false) {
        if (!agentSelectorDashboard || !globalNotificationBubbleLog || !agentSpecificBubbleLog || !logViewTitle) {
            console.error("One or more essential log view DOM elements are missing.");
            return;
        }
        
        const newSelectedAgent = agentSelectorDashboard.value;
        const wasLoadingAgent = isLoadingHistoryForAgent;
        isLoadingHistoryForAgent = (newSelectedAgent !== "all") ? newSelectedAgent : null; 
        currentSelectedAgentForDialogueView = newSelectedAgent;
        let pendingHistoricalThinking = null;

        if (loadFullHistoryButton) {
            loadFullHistoryButton.classList.add('hidden');
        }

        if (currentSelectedAgentForDialogueView === "all") {
            globalNotificationBubbleLog.style.display = "flex";
            agentSpecificBubbleLog.style.display = "none";
            logViewTitle.textContent = "Global Notifications";
            isLoadingHistoryForAgent = null; 
        } else {
            globalNotificationBubbleLog.style.display = "none";
            agentSpecificBubbleLog.style.display = "flex"; 
            logViewTitle.textContent = `Dialogue: ${currentSelectedAgentForDialogueView}`;
            
            if (newSelectedAgent !== wasLoadingAgent) {
                agentSpecificBubbleLog.innerHTML = ''; 
                const initialMessageDiv = document.createElement('div');
                initialMessageDiv.classList.add('chat-bubble', 'chat-bubble-system', 'initial-log-message'); 
                initialMessageDiv.textContent = `Loading history for ${currentSelectedAgentForDialogueView}...`;
                agentSpecificBubbleLog.appendChild(initialMessageDiv);
            }

            showDashboardStatus(`Loading history for ${currentSelectedAgentForDialogueView}...`, 'info', 0); 
            try {
                const apiUrl = `/api/agent_dialogue_history/${currentSelectedAgentForDialogueView}${loadFullHistory ? '?full=true' : ''}`;
                const apiData = await fetchApi(apiUrl);
                
                if (currentSelectedAgentForDialogueView !== newSelectedAgent) {
                    isLoadingHistoryForAgent = null; 
                    return;
                }

                if (agentSpecificBubbleLog.firstChild && agentSpecificBubbleLog.firstChild.classList.contains('initial-log-message')) {
                    agentSpecificBubbleLog.innerHTML = ''; 
                }

                let historyToDisplay = [];
                
                if (apiData.success && apiData.history) {
                    fullDialogueHistoryCache[currentSelectedAgentForDialogueView] = apiData.history;
                    historyToDisplay = apiData.history;
                    
                    if (apiData.history.length === 0) {
                        addMessageToAgentBubbleLog({event: "system_message", data: {message: `No dialogue history found for ${currentSelectedAgentForDialogueView}. New live messages for this agent will be appended.`}, timestamp: Date.now()/1000}, true);
                    } else {
                        if (apiData.is_truncated) {
                             if (loadFullHistoryButton) {
                                loadFullHistoryButton.classList.remove('hidden');
                            }
                            addMessageToAgentBubbleLog({
                                event: "system_message", 
                                data: { message: `Showing recent dialogue history. Click 'Load Full History' to see all entries.` }, 
                                timestamp: Date.now()/1000
                            }, true);
                        }
                        
                        historyToDisplay.forEach(logEntry => {
                            const entryType = logEntry.type || 'historical_log';
                            
                            if (entryType === 'thinking_block' || entryType === 'thinking_block_internal' || entryType === 'manual_llm_thinking_for_human') {
                                pendingHistoricalThinking = String(logEntry.content || "");
                                return;
                            }

                            const sseLikeEventData = { 
                                event: entryType, 
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
                                    historical_thinking_text: pendingHistoricalThinking,
                                    ...(logEntry.data && typeof logEntry.data === 'object' ? logEntry.data : {})
                                },
                                timestamp: logEntry.timestamp || (logEntry.tick !== undefined ? (Date.now()/1000 - (200000 - logEntry.tick*1000 - (logEntry.internal_step || 0)*10 )) : Date.now()/1000) 
                            };
                            
                            pendingHistoricalThinking = null; 

                            if(logEntry.content && !sseLikeEventData.data.text_content) {
                                sseLikeEventData.data.text_content = logEntry.content;
                            }
                            if(logEntry.next_prompt && (entryType === 'internal_outcome') && !sseLikeEventData.data.text_content) {
                                sseLikeEventData.data.text_content = logEntry.next_prompt;
                            }
                            addMessageToAgentBubbleLog(sseLikeEventData, true);
                        });

                        if (pendingHistoricalThinking) {
                             addMessageToAgentBubbleLog({
                                event: "thinking_block",
                                data: {
                                    agent_name: currentSelectedAgentForDialogueView,
                                    tick: "N/A",
                                    speaker: "AgentLLM (Orphaned Thinking)",
                                    content: pendingHistoricalThinking,
                                    historical_thinking_text: pendingHistoricalThinking
                                },
                                timestamp: Date.now()/1000
                            }, true);
                            pendingHistoricalThinking = null;
                        }
                    }
                    showDashboardStatus(`History loaded for ${currentSelectedAgentForDialogueView} (${historyToDisplay.length} entries).`, 'success');
                } else {
                    showDashboardStatus(apiData.error || `Failed to load history for ${currentSelectedAgentForDialogueView}.`, 'error');
                     addMessageToAgentBubbleLog({event: "error", data: {message: `Failed to load history for ${currentSelectedAgentForDialogueView}. ${apiData.error || ''}`}, timestamp: Date.now()/1000}, true);
                }
            } catch (error) { 
                console.error(`Error in handleAgentDialogueViewChange for ${newSelectedAgent}:`, error);
            }
            finally {
                if (isLoadingHistoryForAgent === newSelectedAgent) {
                    isLoadingHistoryForAgent = null;
                }
            }
        }
        getOrchestratorStatus(); 
    }
    
    // Fetch recent events for polling mode (fallback when SSE fails)
    async function fetchRecentEvents() {
        try {
            const data = await fetchApi('/api/orchestrator/recent_events');
            if (data.success && data.events) {
                // Process each event EXACTLY like SSE onmessage would
                data.events.forEach(eventData => {
                    const { event: topLevelEventType, data: ssePayload, timestamp } = eventData;
                    
                    // Debug: Log what types of events we're getting
                    console.log(`DASHBOARD.JS DEBUG: Processing event type: ${topLevelEventType}`, eventData);
                    
                    // Always log to global notifications (EXACT COPY from SSE)
                    addMessageToGlobalNotificationBubbleLog(eventData); 

                    // Conditional logging to agent-specific bubble log (EXACT COPY from SSE)
                    if (currentSelectedAgentForDialogueView !== "all" && 
                        ssePayload.agent_name === currentSelectedAgentForDialogueView &&
                        !isLoadingHistoryForAgent) { 
                        
                        // --- EXACT COPY from SSE: Handle new SSE event types for manual/final chat ---
                        if (topLevelEventType === 'human_assist_event') {
                            if (ssePayload.type === 'manual_message_human_part_sent') {
                                const humanMessageEvent = {
                                    event: 'manual_message_to_agent_llm', // Effective type for addMessageToAgentBubbleLog
                                    data: {
                                        agent_name: ssePayload.agent_name,
                                        tick: ssePayload.tick || timestamp,
                                        speaker: 'HumanAssistant', 
                                        text_content: ssePayload.text_content, // This is the human_message
                                        // No thinking_text or token_info for the human's sent message
                                    },
                                    timestamp: timestamp
                                };
                                addMessageToAgentBubbleLog(humanMessageEvent, false);
                            } else if (ssePayload.type === 'manual_llm_response_received') {
                                const agentResponseEvent = {
                                    event: 'manual_llm_response_to_human', // Effective type
                                    data: {
                                        agent_name: ssePayload.agent_name,
                                        tick: ssePayload.tick || timestamp, 
                                        speaker: `${ssePayload.agent_name} (LLM)`, 
                                        text_content: ssePayload.text_content, // LLM's response
                                        thinking_text: ssePayload.thinking_text, // LLM's thinking
                                        token_info: ssePayload.token_info
                                    },
                                    timestamp: timestamp
                                };
                                addMessageToAgentBubbleLog(agentResponseEvent, false);
                            }
                            // Other human_assist_event types might be notifications, handled by global log
                        } else if (topLevelEventType === 'final_chat_event') {
                            if (ssePayload.type === 'human_message_sent') {
                                const humanFinalMessageEvent = {
                                    event: 'final_message_to_agent', // Effective type
                                    data: {
                                        agent_name: ssePayload.agent_name,
                                        tick: ssePayload.tick || timestamp,
                                        speaker: 'HumanFinalChat',
                                        text_content: ssePayload.human_message // Content from payload
                                    },
                                    timestamp: timestamp
                                };
                                addMessageToAgentBubbleLog(humanFinalMessageEvent, false);
                            } else if (ssePayload.type === 'agent_response_received') {
                                const agentFinalResponseEvent = {
                                    event: 'final_agent_response_to_human', // Effective type
                                    data: {
                                        agent_name: ssePayload.agent_name,
                                        tick: ssePayload.tick || timestamp,
                                        speaker: `${ssePayload.agent_name} (FinalResponse)`,
                                        text_content: ssePayload.llm_response, // Content from payload
                                        thinking_text: ssePayload.thinking_text, // Thinking from payload
                                        token_info: ssePayload.token_info
                                    },
                                    timestamp: timestamp
                                };
                                addMessageToAgentBubbleLog(agentFinalResponseEvent, false);
                            }
                            // Errors in final_chat_event are handled by global log or if status indicates error
                        } else {
                            // For other existing events like 'llm_event', 'observation', etc. (EXACT COPY from SSE)
                            console.log(`DASHBOARD.JS DEBUG: Adding agent-specific event for ${ssePayload.agent_name}:`, eventData);
                            addMessageToAgentBubbleLog(eventData, false);
                        }
                        // --- END EXACT COPY ---
                    }
                });
                
                console.log(`DASHBOARD.JS DEBUG: Processed ${data.events.length} events in polling mode`);
                console.log(`DASHBOARD.JS DEBUG: Current selected agent: ${currentSelectedAgentForDialogueView}`);
            }
        } catch (error) {
            console.error("DASHBOARD.JS DEBUG: Error fetching recent events:", error);
        }
    }
    
    function connectSseLogStream() {
        if (sseSource) { sseSource.close(); }

        const sseEndpoint = '/api/orchestrator/live_log_stream';
        console.log("DASHBOARD.JS DEBUG: Attempting to connect to SSE stream:", sseEndpoint);
        
        try {
            sseSource = new EventSource(sseEndpoint);
        } catch (error) {
            console.error("DASHBOARD.JS DEBUG: Failed to create EventSource:", error);
            addMessageToGlobalNotificationBubbleLog({
                event: "stream_status",
                data: { message: "Failed to create SSE connection" },
                timestamp: Date.now() / 1000
            });
            return;
        }
        
        // Set a timeout for SSE connection
        const connectionTimeout = setTimeout(() => {
            console.log("DASHBOARD.JS DEBUG: SSE connection timeout, falling back to polling");
            if (sseSource) {
                sseSource.close();
                sseSource = null;
            }
            addMessageToGlobalNotificationBubbleLog({
                event: "stream_status",
                data: { message: "SSE connection timeout, switching to polling mode" },
                timestamp: Date.now() / 1000
            });
            setInterval(() => {
                getOrchestratorStatus();
            }, 3000);
        }, 5000); // 5 second timeout
        
        sseSource.onopen = function(event) {
            clearTimeout(connectionTimeout); // Cancel timeout if connection succeeds
            console.log("DASHBOARD.JS DEBUG: SSE connection opened");
            addMessageToGlobalNotificationBubbleLog({
                event: "stream_status", 
                data: {message: "SSE live log stream connected."}, 
                timestamp: Date.now()/1000
            });
        };

        sseSource.onmessage = function(event) {
            try {
                const eventData = JSON.parse(event.data); 
                const { event: topLevelEventType, data: ssePayload, timestamp } = eventData; 

                // Always log to global notifications
                addMessageToGlobalNotificationBubbleLog(eventData); 

                // Conditional logging to agent-specific bubble log
                if (currentSelectedAgentForDialogueView !== "all" && 
                    ssePayload.agent_name === currentSelectedAgentForDialogueView &&
                    !isLoadingHistoryForAgent) { 
                    
                    // --- MODIFICATION START: Handle new SSE event types for manual/final chat ---
                    if (topLevelEventType === 'human_assist_event') {
                        if (ssePayload.type === 'manual_message_human_part_sent') {
                            const humanMessageEvent = {
                                event: 'manual_message_to_agent_llm', // Effective type for addMessageToAgentBubbleLog
                                data: {
                                    agent_name: ssePayload.agent_name,
                                    tick: ssePayload.tick || timestamp,
                                    speaker: 'HumanAssistant', 
                                    text_content: ssePayload.text_content, // This is the human_message
                                    // No thinking_text or token_info for the human's sent message
                                },
                                timestamp: timestamp
                            };
                            addMessageToAgentBubbleLog(humanMessageEvent, false);
                        } else if (ssePayload.type === 'manual_llm_response_received') {
                            const agentResponseEvent = {
                                event: 'manual_llm_response_to_human', // Effective type
                                data: {
                                    agent_name: ssePayload.agent_name,
                                    tick: ssePayload.tick || timestamp, 
                                    speaker: `${ssePayload.agent_name} (LLM)`, 
                                    text_content: ssePayload.text_content, // LLM's response
                                    thinking_text: ssePayload.thinking_text, // LLM's thinking
                                    token_info: ssePayload.token_info
                                },
                                timestamp: timestamp
                            };
                            addMessageToAgentBubbleLog(agentResponseEvent, false);
                        }
                        // Other human_assist_event types might be notifications, handled by global log
                    } else if (topLevelEventType === 'final_chat_event') {
                        if (ssePayload.type === 'human_message_sent') {
                            const humanFinalMessageEvent = {
                                event: 'final_message_to_agent', // Effective type
                                data: {
                                    agent_name: ssePayload.agent_name,
                                    tick: ssePayload.tick || timestamp,
                                    speaker: 'HumanFinalChat',
                                    text_content: ssePayload.human_message // Content from payload
                                },
                                timestamp: timestamp
                            };
                            addMessageToAgentBubbleLog(humanFinalMessageEvent, false);
                        } else if (ssePayload.type === 'agent_response_received') {
                            const agentFinalResponseEvent = {
                                event: 'final_agent_response_to_human', // Effective type
                                data: {
                                    agent_name: ssePayload.agent_name,
                                    tick: ssePayload.tick || timestamp,
                                    speaker: `${ssePayload.agent_name} (FinalResponse)`,
                                    text_content: ssePayload.llm_response, // Content from payload
                                    thinking_text: ssePayload.thinking_text, // Thinking from payload
                                    token_info: ssePayload.token_info
                                },
                                timestamp: timestamp
                            };
                            addMessageToAgentBubbleLog(agentFinalResponseEvent, false);
                        }
                        // Errors in final_chat_event are handled by global log or if status indicates error
                    } else {
                        // For other existing events like 'llm_event', 'observation', etc.
                        addMessageToAgentBubbleLog(eventData, false);
                    }
                    // --- MODIFICATION END ---
                }

                // UI updates for orchestrator status, agent list, etc. (as before)
                if (['orchestrator_status', 'orchestrator_control', 'tick_event', 'agent_management', 'human_assist_event', 'connector_status', 'connector_error', 'final_chat_event'].includes(topLevelEventType)) {
                    getOrchestratorStatus(); 
                    updateStationStatistics(); // Update statistics alongside orchestrator status
                    if (topLevelEventType === 'agent_management' || 
                        (topLevelEventType === 'tick_event' && ssePayload.type === 'end') || 
                        topLevelEventType === 'connector_status' ||
                        (topLevelEventType === 'orchestrator_status' && (ssePayload.status === 'paused_agent_departure' || ssePayload.status === 'stopped'))) {
                        fetchAgentsForDashboard(); 
                    }
                }
                if (topLevelEventType === 'tick_event' && ssePayload.type === 'end' && ssePayload.next_tick !== undefined){
                    if (stationTickDashboard) stationTickDashboard.textContent = ssePayload.next_tick;
                }
            } catch (e) {
                console.error("Error parsing SSE data: ", e, "Raw data: ", event.data);
                const errorLogEntry = {event: "error", data: {message: "Malformed SSE event: " + event.data}, timestamp: Date.now()/1000 };
                addMessageToGlobalNotificationBubbleLog(errorLogEntry);
            }
        }; // End of sseSource.onmessage

        sseSource.onerror = function(error) {
            console.error("DASHBOARD.JS DEBUG: SSE connection error:", error);
            const errorMessage = "Live log stream connection error. Falling back to polling mode.";

            const errorData = {event: "error", data: {message: errorMessage}, timestamp: Date.now()/1000};
            addMessageToGlobalNotificationBubbleLog(errorData);
            if (currentSelectedAgentForDialogueView !== "all") {
                addMessageToAgentBubbleLog(errorData, false);
            }

            const statusMessage = 'Log stream disconnected. Switched to polling mode.';
            showDashboardStatus(statusMessage, 'info');
            
            if(sseSource) { sseSource.close(); sseSource = null; }
            
            // Fallback to polling for ANY SSE error.
            console.log("DASHBOARD.JS DEBUG: Falling back to polling mode due to SSE error.");
            setInterval(() => {
                getOrchestratorStatus();
                fetchRecentEvents(); 
            }, 3000);
        };
    }


    // --- Event Listeners Setup ---
    
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

    if (agentSelectorDashboard) agentSelectorDashboard.addEventListener('change', () => handleAgentDialogueViewChange());
    
    if (clearLogButton) clearLogButton.addEventListener('click', () => {
        if (currentSelectedAgentForDialogueView === "all") {
            if(globalNotificationBubbleLog) globalNotificationBubbleLog.innerHTML = `<div class="chat-bubble chat-bubble-system initial-log-message">Global notifications cleared by user. New messages will appear here.</div>`;
        } else {
            if(agentSpecificBubbleLog) agentSpecificBubbleLog.innerHTML = `<div class="chat-bubble chat-bubble-system initial-log-message">Log for ${currentSelectedAgentForDialogueView} cleared by user.</div>`;
            if (fullDialogueHistoryCache[currentSelectedAgentForDialogueView]) {
                fullDialogueHistoryCache[currentSelectedAgentForDialogueView] = [];
            }
        }
    });
    
    if (loadFullHistoryButton) loadFullHistoryButton.addEventListener('click', () => {
        if (currentSelectedAgentForDialogueView && currentSelectedAgentForDialogueView !== "all") {
            showDashboardStatus("Loading full dialogue history...", 'info');
            handleAgentDialogueViewChange(true); // Pass true to load full history
        }
    });
    
    if (createApiAgentModalButton) createApiAgentModalButton.addEventListener('click', () => {
        if(createApiAgentModal) createApiAgentModal.style.display = "block";
        const modelNameInput = document.getElementById('api-model-name');
        if(modelNameInput) modelNameInput.focus();

        // Populate presets
        if (apiModelPreset && typeof MODEL_PRESETS !== 'undefined') {
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
                    form.querySelector('[name="llm_system_prompt"]').value = preset.llm_system_prompt || '';

                    // Update OpenAI params visibility after preset changes provider
                    if (updateOpenAIParamsVisibility) {
                        updateOpenAIParamsVisibility();
                    }
                }
            }
        });
    }
    
    if (closeApiModalButton) closeApiModalButton.addEventListener('click', () => { if(createApiAgentModal) createApiAgentModal.style.display = "none"; });
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
        else { if (!data.agent_name && (!data.lineage || (data.generation === null || data.generation === undefined || isNaN(data.generation)))) {
                 showDashboardStatus("For Recursive type with auto-name, Lineage and Generation are required.", "error"); return; }}
        
        showDashboardStatus('Creating API agent...', 'info');
        try {
            const result = await fetchApi('/api/orchestrator/add_agent', 'POST', data); 
            showDashboardStatus(result.message, result.success ? 'success' : 'error');
            if (result.success) {
                if(createApiAgentModal) createApiAgentModal.style.display = "none";
                createApiAgentForm.reset();
                if(apiRecursiveFieldsDiv) apiRecursiveFieldsDiv.classList.add('hidden');
                await fetchAgentsForDashboard(); 
                await getOrchestratorStatus(); 
            }
        } catch (error) { /* Handled by fetchApi */ }
    });

    // Helper function to check if modal has unsaved content
    function hasUnsavedContent(modalId) {
        if (modalId === 'createApiAgentModal') {
            const form = document.getElementById('create-api-agent-form');
            if (!form) return false;
            const formData = new FormData(form);
            // Check if any required fields have content
            return formData.get('model_name') || formData.get('agent_name') || 
                   formData.get('llm_system_prompt') || formData.get('internal_note');
        }
        if (modalId === 'directMessageModal') {
            const messageInput = document.getElementById('direct-message-input');
            return messageInput && messageInput.value.trim() !== '';
        }
        if (modalId === 'sendSystemMessageModal') {
            const messageContent = document.getElementById('system-message-content');
            return messageContent && messageContent.value.trim() !== '';
        }
        if (modalId === 'speakCommonRoomModal') {
            const speakerName = document.getElementById('common-room-speaker-name');
            const messageContent = document.getElementById('common-room-message-content');
            return (speakerName && speakerName.value.trim() !== '') || 
                   (messageContent && messageContent.value.trim() !== '');
        }
        if (modalId === 'updateStationConfigModal') {
            const stationStatus = document.getElementById('update-station-status');
            const stationName = document.getElementById('update-station-name');
            const stationDescription = document.getElementById('update-station-description');
            return (stationStatus && stationStatus.value.trim() !== '') ||
                   (stationName && stationName.value.trim() !== '') || 
                   (stationDescription && stationDescription.value.trim() !== '');
        }
        return false;
    }

    function closeModalSafely(modal, modalId) {
        if (hasUnsavedContent(modalId)) {
            if (confirm("You have unsaved changes. Are you sure you want to close this modal? Your changes will be lost.")) {
                modal.style.display = "none";
            }
        } else {
            modal.style.display = "none";
        }
    }

    window.onclick = function(event) {
        if (createApiAgentModal && event.target == createApiAgentModal) { 
            closeModalSafely(createApiAgentModal, 'createApiAgentModal');
        }
        if (directMessageModal && event.target == directMessageModal) { 
            closeModalSafely(directMessageModal, 'directMessageModal');
        }
        if (sendSystemMessageModal && event.target == sendSystemMessageModal) { 
            closeModalSafely(sendSystemMessageModal, 'sendSystemMessageModal');
        }        
        if (speakCommonRoomModal && event.target == speakCommonRoomModal) { 
            closeModalSafely(speakCommonRoomModal, 'speakCommonRoomModal');
        }
        if (updateStationConfigModal && event.target == updateStationConfigModal) { 
            closeModalSafely(updateStationConfigModal, 'updateStationConfigModal');
        }
    };

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
                        option.value = agentName;
                        option.textContent = agentName;
                        systemMessageAgentSelector.appendChild(option);
                    });
                    if (confirmSendSystemMessageButton) confirmSendSystemMessageButton.disabled = false;
                }
            }
            if (systemMessageContent) systemMessageContent.value = "";
            if (sendSystemMessageModal) sendSystemMessageModal.style.display = "block";
            if (systemMessageContent) systemMessageContent.focus();
        });
    }

    if (closeSendSystemMessageModalButton) {
        closeSendSystemMessageModalButton.addEventListener('click', () => {
            if (sendSystemMessageModal) sendSystemMessageModal.style.display = "none";
        });
    }
    if (cancelSendSystemMessageButton) {
        cancelSendSystemMessageButton.addEventListener('click', () => {
            if (sendSystemMessageModal) sendSystemMessageModal.style.display = "none";
        });
    }

    if (confirmSendSystemMessageButton) {
        confirmSendSystemMessageButton.addEventListener('click', async () => {
            const selectedAgentOptions = systemMessageAgentSelector ? Array.from(systemMessageAgentSelector.selectedOptions) : [];
            const targetAgents = selectedAgentOptions.map(option => option.value);
            const message = systemMessageContent ? systemMessageContent.value.trim() : "";

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
                    message_content: message
                });
                showDashboardStatus(result.message, result.success ? 'success' : 'error');
                if (result.success) {
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

    if (openSpeakCommonRoomModalButton) {
        openSpeakCommonRoomModalButton.addEventListener('click', () => {
            // This tool can be used regardless of orchestrator state, as it's a direct room interaction.
            // However, ensure station_instance is available on backend.
            if (commonRoomSpeakerName) commonRoomSpeakerName.value = "";
            if (commonRoomMessageContent) commonRoomMessageContent.value = "";
            if (speakCommonRoomModal) speakCommonRoomModal.style.display = "block";
            if (commonRoomSpeakerName) commonRoomSpeakerName.focus();
        });
    }
    if (closeSpeakCommonRoomModalButton) {
        closeSpeakCommonRoomModalButton.addEventListener('click', () => {
            if (speakCommonRoomModal) speakCommonRoomModal.style.display = "none";
        });
    }
    if (cancelSpeakCommonRoomButton) {
        cancelSpeakCommonRoomButton.addEventListener('click', () => {
            if (speakCommonRoomModal) speakCommonRoomModal.style.display = "none";
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
                    if (speakCommonRoomModal) speakCommonRoomModal.style.display = "none";
                }
            } catch (error) {
                // fetchApi already shows error
            } finally {
                confirmSpeakCommonRoomButton.disabled = false;
            }
        });
    }    

    if (openDirectMessageModalButton) {
        openDirectMessageModalButton.addEventListener('click', () => {
            const selectedAgent = agentSelectorDashboard.value;
            if (!selectedAgent || selectedAgent === "all") {
                showDashboardStatus("Please select a specific agent to message.", "error"); return;
            }
            
            
            const agentData = fullAgentListCache.find(a => a.name === selectedAgent);
            if (!agentData) {
                showDashboardStatus("Selected agent not found.", "error"); return;
            }
            
            const isEnded = agentData.status.startsWith("Session Ended");
            const isAscended = agentData.status.startsWith("Ascended");
            
            if (isAscended) {
                showDashboardStatus("Cannot send messages to ascended agents.", "error"); return;
            }
            
            // Set modal content based on agent status
            if (directMessageModalAgentName) directMessageModalAgentName.textContent = selectedAgent;
            if (directMessageInput) directMessageInput.value = "";
            if (directMessageResponseArea) directMessageResponseArea.classList.add('hidden');
            if (directMessageLlmResponseContent) directMessageLlmResponseContent.innerHTML = "";
            
            if (isEnded) {
                if (directMessageModalTitle) directMessageModalTitle.textContent = "Send Message to Ended Agent";
                if (directMessageModalDescription) directMessageModalDescription.textContent = "This agent's session has ended. Your message will be logged.";
            } else {
                if (directMessageModalTitle) directMessageModalTitle.textContent = "Send Message to Agent";
                if (directMessageModalDescription) directMessageModalDescription.textContent = "Your message will be sent directly to the agent. System will wait for safe timing.";
            }
            
            if (directMessageModal) directMessageModal.style.display = "block";
            if (directMessageInput) directMessageInput.focus();
        });
    }
    if (closeDirectMessageModalButton) closeDirectMessageModalButton.addEventListener('click', () => { if (directMessageModal) directMessageModal.style.display = "none"; });
    if (cancelDirectMessageButton) cancelDirectMessageButton.addEventListener('click', () => { if (directMessageModal) directMessageModal.style.display = "none"; });

    if (confirmSendDirectMessageButton) {
        confirmSendDirectMessageButton.addEventListener('click', async () => {
            const agentToMessage = directMessageModalAgentName ? directMessageModalAgentName.textContent : null;
            const messageText = directMessageInput ? directMessageInput.value.trim() : "";

            if (!agentToMessage || agentToMessage === "N/A") { 
                showDashboardStatus("Error: Agent name missing in modal.", "error"); return; 
            }
            if (!messageText) { 
                showDashboardStatus("Message cannot be empty.", "error"); return; 
            }

            const agentData = fullAgentListCache.find(a => a.name === agentToMessage);
            if (!agentData) {
                showDashboardStatus("Selected agent not found.", "error"); return;
            }

            const isEnded = agentData.status.startsWith("Session Ended");
            const isAscended = agentData.status.startsWith("Ascended");

            if (isAscended) {
                showDashboardStatus("Cannot send messages to ascended agents.", "error"); return;
            }

            showDashboardStatus(`Sending message to ${agentToMessage}...`, 'info');
            confirmSendDirectMessageButton.disabled = true;
            if (openDirectMessageModalButton) openDirectMessageModalButton.disabled = true;
            isDirectMessageInProgress = true;
            
            // Close modal immediately when user clicks send
            if (directMessageModal) directMessageModal.style.display = "none";
            
            try {
                let result;
                if (isEnded) {
                    // Send to ended agent using final chat API
                    result = await fetchApi(`/api/agent/${agentToMessage}/final_chat`, 'POST', { 
                        human_message: messageText 
                    });
                    
                    if (result.success) {
                        showDashboardStatus(`Message sent. Agent ${agentToMessage} replied.`, 'success');
                    } else {
                        showDashboardStatus(result.error || "Failed to send final message.", "error");
                    }
                } else {
                    // Send to living agent using manual message API
                    result = await fetchApi('/api/orchestrator/manual_message', 'POST', {
                        agent_name: agentToMessage, 
                        message_text: messageText, 
                        end_chat_after_send: false  // Never auto-end chat (deprecated functionality)
                    });
                    
                    if (result.success) {
                        showDashboardStatus(`Message sent. Agent ${agentToMessage} replied. See log.`, 'success');
                    } else {
                        showDashboardStatus(result.error || "Failed to send manual message.", "error");
                    }
                }
                
            } catch (error) { 
                /* Handled by fetchApi - modal stays open on network errors so user can retry */ 
            } finally { 
                confirmSendDirectMessageButton.disabled = false;
                isDirectMessageInProgress = false;
                // Re-enable the main Direct Message button, but respect normal enabling rules
                if (openDirectMessageModalButton) {
                    const agentData = fullAgentListCache.find(a => a.name === agentToMessage);
                    const isAscended = agentData && agentData.status.startsWith("Ascended");
                    openDirectMessageModalButton.disabled = isAscended;
                }
            }
        });
    }
    
    if (cancelCreateApiAgentButton) {
        cancelCreateApiAgentButton.addEventListener('click', () => {
            if (createApiAgentModal) createApiAgentModal.style.display = "none";
        });
    }

    if(resolveHumanInterventionButton) resolveHumanInterventionButton.addEventListener('click', async () => {
        // Open modal to resolve human request for the selected agent
        const selectedAgent = agentSelectorDashboard ? agentSelectorDashboard.value : null;
        if (!selectedAgent || selectedAgent === "all") {
            showDashboardStatus("Please select a specific agent to resolve their request.", "error");
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
            const request = response.request;
            if (requestIdDisplay) requestIdDisplay.textContent = request.request_id;
            if (requestTickDisplay) requestTickDisplay.textContent = request.tick;
            if (requestAgentDisplay) requestAgentDisplay.textContent = request.agent_name;
            if (requestModelDisplay) requestModelDisplay.textContent = request.agent_model;
            if (requestTitleDisplay) requestTitleDisplay.textContent = request.title;
            if (requestContentDisplay) requestContentDisplay.textContent = request.content;
            if (resolveResponseInput) resolveResponseInput.value = '';
            if (resolveRequestStatus) {
                resolveRequestStatus.classList.add('hidden');
                resolveRequestStatus.textContent = '';
            }

            // Store agent name for later
            if (resolveRequestModal) {
                resolveRequestModal.dataset.agentName = selectedAgent;
                resolveRequestModal.dataset.requestId = request.request_id;
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
            if (resolveRequestModal) resolveRequestModal.style.display = "none";
        });
    }

    if (cancelResolveRequestButton) {
        cancelResolveRequestButton.addEventListener('click', () => {
            if (resolveRequestModal) resolveRequestModal.style.display = "none";
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
        const operationMode = document.body.dataset.operationMode;
        if (operationMode === 'api') {
            getStationVersion(); // Load version first
            loadStationConfig(); // Load station config and update top bar
            getOrchestratorStatus();
            updateStationStatistics(); // Initial load of statistics
            fetchAgentsForDashboard().then(() => {
                if (window.location.hash) {
                    const agentNameFromHash = window.location.hash.substring(1);
                    if (agentSelectorDashboard && Array.from(agentSelectorDashboard.options).some(opt => opt.value === agentNameFromHash)) {
                        agentSelectorDashboard.value = agentNameFromHash;
                    }
                }
                handleAgentDialogueViewChange();
            });
            connectSseLogStream();
        } else {
            if(globalNotificationBubbleLog) addMessageToGlobalNotificationBubbleLog({event: "system_message", data: {message: "Orchestrator live log inactive in Manual Mode."}, timestamp: Date.now()/1000});
            const apiControls = [startLoopButton, pauseOrchestratorButton, resumeOrchestratorButton, stopOrchestratorButton, createApiAgentModalButton, endApiAgentSessionButton, openDirectMessageModalButton, resolveHumanInterventionButton];
            apiControls.forEach(btn => { if(btn) btn.disabled = true; });
            showDashboardStatus("Station is in Manual Mode. Use the Manual Interface tab for agent interaction.", "info", 0);
        }

        setInterval(() => {
        if (document.visibilityState === 'visible' && operationMode === 'api') { 
            getOrchestratorStatus();
            updateStationStatistics(); // Periodic update of statistics
            // Always fetch agents to keep the list updated
            fetchAgentsForDashboard();
            }
        }, 7000); 
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
        const headerRightDiv = document.querySelector('header .container .text-lg');
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
            const headerRightDiv = document.querySelector('header .container .text-lg');
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
            if (updateStationConfigModal) updateStationConfigModal.style.display = 'none';
        });
    }

    if (cancelUpdateStationConfigButton) {
        cancelUpdateStationConfigButton.addEventListener('click', () => {
            if (updateStationConfigModal) updateStationConfigModal.style.display = 'none';
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
