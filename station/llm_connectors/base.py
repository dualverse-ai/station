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
import abc
import copy
import json
from typing import Dict, Any, Optional, List, Tuple, Set

from station import file_io_utils
from station import constants
from station import agent as agent_module


# --- Custom LLM Connector Exceptions ---
class LLMConnectorError(Exception):
    """Base class for connector-related errors."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception


class LLMTransientAPIError(LLMConnectorError):
    """Indicates a potentially temporary API error (e.g., 50x, rate limits) that might be retried."""
    pass


class LLMPermanentAPIError(LLMConnectorError):
    """Indicates a more permanent API error (e.g., auth, invalid request, model not found)."""
    pass


class LLMSafetyBlockError(LLMConnectorError):
    """Indicates the response was blocked due to safety filters."""
    def __init__(self, message: str, block_reason: Optional[str] = None, prompt_feedback: Any = None, original_exception: Optional[Exception] = None):
        super().__init__(message, original_exception)
        self.block_reason = block_reason
        self.prompt_feedback = prompt_feedback


class LLMContextOverflowError(LLMConnectorError):
    """Indicates the input exceeds the model's context window limit, requiring agent session termination."""
    pass


class BaseLLMConnector(abc.ABC):
    """
    Abstract base class for LLM connectors.
    Each instance is designed to handle a continuous, stateful chat session for a single agent,
    with persistent history.
    """
    def __init__(self,
                 model_name: str,
                 agent_name: str,
                 agent_data_path: str, # Path to agent's specific data directory for history
                 api_key: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 temperature: float = 1.0,
                 max_output_tokens: Optional[int] = None,
                 max_retries: int = constants.LLM_MAX_RETRIES, # Default from constants
                 retry_delay_seconds: int = constants.LLM_RETRY_DELAY_SECONDS): # Default from constants
        self.model_name = model_name
        self.agent_name = agent_name
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        self.agent_data_path = agent_data_path 
        self.history_file_path = os.path.join(self.agent_data_path, "llm_chat_history.yamll")

        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        # When False, the connector must not write anything to station_data (history files, agent YAML, etc).
        # Temporal chat uses this mode to avoid leaving any records.
        self.persist_to_disk: bool = True
        
        # Set proxy environment variables if configured in constants
        if constants.LLM_HTTP_PROXY:
            os.environ['http_proxy'] = constants.LLM_HTTP_PROXY
            # Also set grpc_proxy for gRPC-based clients
            if 'grpc_proxy' not in os.environ:
                os.environ['grpc_proxy'] = constants.LLM_HTTP_PROXY
        if constants.LLM_HTTPS_PROXY:
            os.environ['https_proxy'] = constants.LLM_HTTPS_PROXY
            # Set grpc_proxy if not already set
            if 'grpc_proxy' not in os.environ:
                os.environ['grpc_proxy'] = constants.LLM_HTTPS_PROXY
        
        # Load pruning blocks and store a copy to detect changes
        self.agent_prune_blocks: List[Dict[str, Any]] = self._load_prune_blocks_from_agent_data()
        self._last_known_prune_blocks: List[Dict[str, Any]] = copy.deepcopy(self.agent_prune_blocks)
        self._last_known_system_prompt: Optional[str] = self.system_prompt
        self._debug_station_id: Optional[str] = None

    def _debug_api_enabled(self) -> bool:
        raw_value = str(os.getenv("DEBUG_API", "")).strip().lower()
        return raw_value in {"1", "true", "yes", "on"}

    def _get_debug_api_dir(self) -> str:
        if self._debug_station_id is None:
            station_id = "unknown_station"
            try:
                station_config_path = os.path.join(
                    os.getcwd(),
                    constants.BASE_STATION_DATA_PATH,
                    constants.STATION_CONFIG_FILENAME,
                )
                station_config = file_io_utils.load_yaml(station_config_path)
                if isinstance(station_config, dict):
                    candidate = station_config.get(constants.STATION_ID_KEY)
                    if isinstance(candidate, str) and candidate.strip():
                        station_id = candidate.strip()
            except Exception as e:
                print(f"Warning ({self.agent_name}): Failed to resolve station_id for DEBUG_API path: {e}")
            self._debug_station_id = station_id
        return os.path.join(os.getcwd(), "tests", self._debug_station_id)

    def _write_debug_api_snapshot(self, filename: str, payload: Dict[str, Any]) -> None:
        if not self._debug_api_enabled():
            return
        try:
            snapshot_dir = self._get_debug_api_dir()
            file_io_utils.ensure_dir_exists(snapshot_dir)
            path = os.path.join(snapshot_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning ({self.agent_name}): Failed to write DEBUG_API snapshot '{filename}': {e}")


    def _load_prune_blocks_from_agent_data(self) -> List[Dict[str, Any]]:
        """Loads pruning blocks from agent data for summary handling."""
        try:
            agent_full_data = agent_module.load_agent_data(self.agent_name, include_ended=True, include_ascended=True)
            if agent_full_data:
                return agent_full_data.get(constants.AGENT_PRUNED_DIALOGUE_TICKS_KEY, [])
            return []
        except Exception as e:
            print(f"Error ({self.agent_name}): Failed to load prune blocks: {e}")
            return []

    def _bypass_agent_data_system_prompt_reload(self) -> bool:
        """
        Return True when the connector should keep its constructor-provided system prompt.

        Most Station agents reload their runtime system prompt from agent YAML.
        System services such as the archive reviewer are different: they use an
        explicit connector-level system prompt plus separate task/context user
        messages, so they should keep the constructor-provided prompt.
        """
        return self.agent_name == "AutoArchiveEvaluator"

    def _load_system_prompt_from_agent_data(self) -> Optional[str]:
        """Loads current system prompt from public-branch agent data."""
        try:
            if self._bypass_agent_data_system_prompt_reload():
                return self._last_known_system_prompt
            agent_full_data = agent_module.load_agent_data(self.agent_name, include_ended=True, include_ascended=True)
            if agent_full_data is None:
                return self._last_known_system_prompt
            return agent_full_data.get(constants.AGENT_LLM_SYSTEM_PROMPT_KEY, self._last_known_system_prompt)
        except Exception as e:
            print(f"Error ({self.agent_name}): Failed to load system prompt: {e}")
            return self._last_known_system_prompt

    @abc.abstractmethod
    def _load_history_from_file(self) -> List[Dict[str, Any]]:
        """Loads chat history... {'tick': int, 'role': str, 'text_content': str}"""
        pass

    @abc.abstractmethod
    def _append_turn_to_history_file(
        self,
        tick: int,
        role: str,
        text: str,
        thinking_text: Optional[str] = None,
        token_info: Optional[Dict[str, Optional[int]]] = None,
        api_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Appends a single turn with optional thinking text and token info."""
        pass

    @abc.abstractmethod
    def _initialize_chat_session(self) -> None:
        """
        Initializes the persistent chat session, loading existing history.
        """
        pass

    def _filter_and_prune_history(self,
                                   raw_history_entries: List[Dict[str, Any]]
                                   ) -> List[Dict[str, Any]]:
        """
        Filters history based on pruning blocks and inserts summary replacements.
        Input entries: List of {'tick': int, 'role': str, 'text_content': str}
        Output entries: List of {'role': str, 'text_content': str} with summary replacements
        """
        if not raw_history_entries:
            return []

        # Parse prune blocks into ranges with summaries
        pruned_ranges = []  # [(start_tick, end_tick, summary), ...]
        for block in self.agent_prune_blocks:
            ticks_input = block.get(constants.PRUNE_TICKS_KEY)
            summary = block.get(constants.PRUNE_SUMMARY_KEY, "")

            if ticks_input is not None:
                block_ticks = self._parse_ticks_for_filtering(ticks_input)
                if block_ticks:
                    start_tick, end_tick = min(block_ticks), max(block_ticks)
                    pruned_ranges.append((start_tick, end_tick, summary))

        # Get protected ticks
        protected_ticks = self._get_protected_ticks(raw_history_entries)

        # Filter out entries within pruned ranges (except protected ticks)
        filtered_entries = []
        for entry in raw_history_entries:
            tick = entry.get('tick')
            role = entry.get('role')
            text_content = entry.get('text_content', '')

            if tick is None or role is None:
                print(f"Warning ({self.agent_name}): Skipping history entry with missing tick or role: {entry}")
                continue

            # Always include protected ticks
            if tick in protected_ticks:
                preserved_entry = dict(entry)
                preserved_entry['tick'] = tick
                preserved_entry['role'] = role
                preserved_entry['text_content'] = text_content
                filtered_entries.append(preserved_entry)
                continue

            # Check if this tick is in any pruned range
            is_pruned = any(start <= tick <= end for start, end, _ in pruned_ranges)
            if not is_pruned:
                preserved_entry = dict(entry)
                preserved_entry['tick'] = tick
                preserved_entry['role'] = role
                preserved_entry['text_content'] = text_content
                filtered_entries.append(preserved_entry)

        # Insert summary replacements at chronological positions
        final_entries = []
        current_entry_index = 0

        for start_tick, end_tick, summary in sorted(pruned_ranges):
            # Add all entries before this pruned range
            while (current_entry_index < len(filtered_entries) and
                   filtered_entries[current_entry_index].get('tick', 0) < start_tick):
                entry = filtered_entries[current_entry_index]
                out_entry = dict(entry)
                out_entry.pop('tick', None)
                final_entries.append(out_entry)
                current_entry_index += 1

            # Insert summary replacement only if non-empty summary
            # Empty summary = complete removal (skip entirely, like original behavior)
            if summary.strip():
                if start_tick == end_tick:
                    system_msg = f"System: Pruned Tick {start_tick}"
                else:
                    system_msg = f"System: Pruned Ticks {start_tick}-{end_tick}"

                final_entries.append({'role': 'user', 'text_content': system_msg})
                final_entries.append({'role': 'model', 'text_content': f"Summary: {summary}"})

        # Add remaining entries after all pruned ranges
        while current_entry_index < len(filtered_entries):
            entry = filtered_entries[current_entry_index]
            out_entry = dict(entry)
            out_entry.pop('tick', None)
            final_entries.append(out_entry)
            current_entry_index += 1

        print(f"Before pruning, raw history length: {len(raw_history_entries)}, after pruning: {len(final_entries)} for {self.agent_name}.")
        return final_entries

    def _calculate_retry_delay(self, attempt_number: int) -> int:
        """
        Calculate incremental retry delay based on attempt number.
        Pattern: 60, 60, 120, 120, 240, 240, 480, 480, 960, 960, ...
        - Each delay repeats twice
        - After 2 repetitions, double the delay
        - Maximum of 4 doublings (max delay = base_delay * 16)

        Args:
            attempt_number: Current attempt number (1-indexed)

        Returns:
            Delay in seconds
        """
        if attempt_number < 1:
            return self.retry_delay_seconds

        # Calculate doubling level: 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, ...
        # Max doubling level is 4
        doubling_level = min((attempt_number - 1) // 2, 4)

        # Calculate delay: base * 2^level
        delay = self.retry_delay_seconds * (2 ** doubling_level)

        return delay

    def _compact_persisted_value(self, value: Any) -> Any:
        """Recursively drop None and empty containers before writing metadata to disk."""
        if isinstance(value, dict):
            compacted: Dict[str, Any] = {}
            for key, item in value.items():
                compacted_item = self._compact_persisted_value(item)
                if compacted_item is None:
                    continue
                if isinstance(compacted_item, (dict, list)) and not compacted_item:
                    continue
                compacted[str(key)] = compacted_item
            return compacted

        if isinstance(value, list):
            compacted_list = []
            for item in value:
                compacted_item = self._compact_persisted_value(item)
                if compacted_item is None:
                    continue
                if isinstance(compacted_item, (dict, list)) and not compacted_item:
                    continue
                compacted_list.append(compacted_item)
            return compacted_list

        return value

    def _prepare_api_metadata_for_persistence(self, metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not metadata:
            return None
        compacted = self._compact_persisted_value(metadata)
        if isinstance(compacted, dict) and compacted:
            return compacted
        return None

    def _sanitize_api_return_payload(
        self,
        value: Any,
        drop_keys: Optional[Set[str]] = None,
        max_string_length: int = 2000,
    ) -> Any:
        """
        Remove generated text fields from raw API payloads while preserving structured metadata.
        """
        effective_drop_keys = drop_keys or {
            "arguments",
            "content",
            "delta",
            "output_text",
            "reasoning_content",
            "summary",
            "text",
            "thinking",
            "thinking_content",
        }

        if isinstance(value, dict):
            sanitized: Dict[str, Any] = {}
            for key, item in value.items():
                if str(key) in effective_drop_keys:
                    continue
                sanitized[str(key)] = self._sanitize_api_return_payload(
                    item,
                    drop_keys=effective_drop_keys,
                    max_string_length=max_string_length,
                )
            return sanitized

        if isinstance(value, list):
            return [
                self._sanitize_api_return_payload(
                    item,
                    drop_keys=effective_drop_keys,
                    max_string_length=max_string_length,
                )
                for item in value
            ]

        if isinstance(value, tuple):
            return [
                self._sanitize_api_return_payload(
                    item,
                    drop_keys=effective_drop_keys,
                    max_string_length=max_string_length,
                )
                for item in value
            ]

        if isinstance(value, str) and len(value) > max_string_length:
            return f"{value[:max_string_length]}...[truncated]"

        return value

    def _parse_ticks_for_filtering(self, ticks_input) -> set:
        """Parse ticks input for filtering (simplified version without error handling)."""
        if ticks_input is None:
            return set()

        try:
            if isinstance(ticks_input, int):
                return {ticks_input}

            if isinstance(ticks_input, str):
                if '-' in ticks_input and ',' not in ticks_input:
                    # Simple range like "3-6"
                    parts = ticks_input.split('-')
                    if len(parts) == 2:
                        start, end = int(parts[0]), int(parts[1])
                        return set(range(start, end + 1))
                elif ',' in ticks_input:
                    # Comma-separated like "3,4,5,6"
                    ticks = [int(x.strip()) for x in ticks_input.split(',') if x.strip().isdigit()]
                    return set(ticks)
                else:
                    # Single number as string
                    return {int(ticks_input)}

            return set()

        except (ValueError, TypeError):
            return set()

    def _contains_protected_keywords(self, text: str) -> bool:
        """
        Check if the text contains any keywords that should prevent pruning.
        """
        if not text:
            return False
        
        # Check if text contains any of the protected keywords
        for keyword in constants.NOT_PRUNABLE_KEYWORDS:
            if keyword in text:
                return True
        return False

    def _get_protected_ticks(self, raw_history_entries: List[Dict[str, Any]]) -> set:
        """
        Identify ticks that contain protected keywords in any station response.
        Returns a set of tick numbers that should not be pruned.
        """
        protected_ticks = set()
        
        for entry in raw_history_entries:
            tick = entry.get('tick')
            role = entry.get('role')
            text_content = entry.get('text_content', '')
            
            # Check if this is a station response (user role) with protected keywords
            if role == 'user' and tick is not None and self._contains_protected_keywords(text_content):
                protected_ticks.add(tick)
        
        return protected_ticks

    @abc.abstractmethod
    def _send_message_implementation(self, user_prompt: str, current_tick: int, attempt_number: int = 0) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        pass

    def _handle_system_prompt_update(self) -> None:
        """Hook for connectors that need to rebuild internal config when system prompt changes."""
        return None

    def _before_send_message(self, current_tick: int) -> None:
        """Hook for connectors that need to adjust client state before a send."""
        return None

    def _handle_send_error(self, error: Exception, current_tick: int) -> bool:
        """
        Hook for connectors that want to react to send errors.

        Returns True to retry immediately without the normal backoff sleep.
        """
        return False

    def sync_state(self) -> None:
        """
        Synchronize connector state with agent data on disk.

        Checks if pruning blocks have changed and re-initializes the chat session
        or system prompt if needed, then recounts tokens. This method is idempotent - calling it
        multiple times with unchanged state has no effect (no wasted computation).

        Called before generating observations and before sending messages to ensure
        token counts are accurate.
        """
        current_prune_blocks_on_disk = self._load_prune_blocks_from_agent_data()
        current_system_prompt = self._load_system_prompt_from_agent_data()

        # Idempotent check - if pruning blocks unchanged, this is a no-op
        prune_changed = current_prune_blocks_on_disk != self._last_known_prune_blocks
        prompt_changed = current_system_prompt != self._last_known_system_prompt

        if prune_changed or prompt_changed:
            if prune_changed:
                print(f"Info ({self.agent_name}): Pruning blocks changed. Re-initializing chat session.")
                self.agent_prune_blocks = copy.deepcopy(current_prune_blocks_on_disk)
                self._last_known_prune_blocks = copy.deepcopy(current_prune_blocks_on_disk)
            if prompt_changed:
                print(f"Info ({self.agent_name}): System prompt changed. Re-initializing chat session.")
                self.system_prompt = current_system_prompt
                self._last_known_system_prompt = current_system_prompt
                self._handle_system_prompt_update()
            try:
                self._initialize_chat_session() # Re-initialize with new pruning rules

                # Count tokens after re-initialization and update agent's budget
                print(f"Info ({self.agent_name}): Counting tokens after re-initialization.")
                new_token_count = self.get_current_total_session_tokens()
                if new_token_count is not None:
                    if self.persist_to_disk:
                        # Update agent's token budget in the data file
                        agent_data = agent_module.load_agent_data(self.agent_name)
                        if agent_data:
                            agent_data[constants.AGENT_TOKEN_BUDGET_CURRENT_KEY] = new_token_count
                            agent_module.save_agent_data(self.agent_name, agent_data)
                            print(f"Info ({self.agent_name}): Token budget updated to {new_token_count} after pruning.")
                        else:
                            print(f"Warning ({self.agent_name}): Could not load agent data to update token count after pruning.")
                else:
                    print(f"Warning ({self.agent_name}): Could not count tokens after pruning re-initialization.")

            except Exception as e_reinit:
                print(f"Error ({self.agent_name}): Failed to re-initialize chat session after pruning update: {e_reinit}.")
                # Note: We don't raise here to allow the caller to proceed, but state may be stale

    def send_message(self, user_prompt: str, current_tick: int) -> Tuple[str, Dict[str, Optional[int]]]:
        # Synchronize state (pruning blocks, token recount) before sending
        # This is idempotent - if sync_state() was already called earlier (e.g., before request_status),
        # this will be a no-op with no wasted computation
        self.sync_state()
        self._before_send_message(current_tick)

        # --- Original send_message retry logic ---
        last_exception: Optional[Exception] = None
        last_empty_response: Optional[Tuple[str, Optional[str], Dict[str, Optional[int]]]] = None
        current_attempt = 0
        while current_attempt <= self.max_retries:
            try:
                llm_response, thinking_response, token_info = self._send_message_implementation(user_prompt, current_tick, attempt_number=current_attempt)

                # Validate that the response is not empty
                if not llm_response or not llm_response.strip():
                    # Save the empty response in case all retries are exhausted
                    last_empty_response = (llm_response, thinking_response, token_info)

                    # Only retry if we haven't exhausted all attempts
                    if current_attempt < self.max_retries:
                        raise LLMTransientAPIError(
                            f"Empty response received from LLM for {self.agent_name}. "
                            f"This may indicate a model error or all content was filtered.",
                            original_exception=None
                        )
                    else:
                        # All retries exhausted, accept the empty response
                        print(f"Warning ({self.agent_name}): Empty response received after {self.max_retries} retries. Accepting empty response.")
                        return llm_response, thinking_response, token_info

                return llm_response, thinking_response, token_info
            except LLMContextOverflowError as e:
                # Context overflow should not be retried - the context won't get smaller with retries!
                print(f"LLMConnector ({self.agent_name}): Context overflow detected, not retrying: {e}")
                raise
            except Exception as e:
                last_exception = e
                current_attempt += 1
                retry_immediately = False
                try:
                    retry_immediately = self._handle_send_error(e, current_tick)
                except Exception as hook_error:
                    print(f"Warning ({self.agent_name}): send error hook failed: {hook_error}")

                if current_attempt > self.max_retries:
                    print(f"LLMConnector ({self.agent_name}): Max retries ({self.max_retries}) exhausted. Last error: {e}")
                    raise
                if retry_immediately:
                    print(f"LLMConnector ({self.agent_name}): Retrying immediately after transient error handling.")
                    continue
                
                # Print detailed error information for debugging
                error_details = str(e)
                raw_error_info = ""
                
                # Extract additional error details if available
                if hasattr(e, 'original_exception') and e.original_exception:
                    raw_error_info = f" | Raw API Error: {e.original_exception}"
                elif hasattr(e, '__dict__'):
                    # Print all available attributes for debugging
                    error_attrs = {k: v for k, v in e.__dict__.items() if not k.startswith('_')}
                    if error_attrs:
                        raw_error_info = f" | Error Attributes: {error_attrs}"
                
                # Calculate incremental retry delay
                retry_delay = self._calculate_retry_delay(current_attempt)

                print(f"LLMConnector ({self.agent_name}): API error (Attempt {current_attempt}/{self.max_retries}): {error_details}{raw_error_info}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
        
        if last_exception: 
            raise last_exception 
        raise LLMConnectorError(f"LLMConnector ({self.agent_name}): send_message failed unexpectedly after retry logic.")

    @abc.abstractmethod
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Returns the current chat history from the active session in a simple list format.
        Each item: {'role': 'user'/'model', 'text': 'message content'}
        """
        pass
    
    @abc.abstractmethod
    def get_current_total_session_tokens(self) -> Optional[int]:
        """
        Calculates and returns the total number of tokens for the current,
        possibly pruned, chat session history as understood by the LLM.
        This should reflect the actual history that would be used for context.
        """
        pass

    def force_refresh_and_get_current_session_tokens(self) -> Optional[int]:
        """
        Forces a refresh of the pruning blocks, re-initializes the chat session
        if pruning info has changed, and then returns the current total session tokens.
        """
        current_prune_blocks_on_disk = self._load_prune_blocks_from_agent_data()

        # Check if pruning blocks actually changed to avoid unnecessary re-initialization
        if current_prune_blocks_on_disk != self._last_known_prune_blocks:
            print(f"Info ({self.agent_name}): Pruning blocks changed (detected by force_refresh). Re-initializing chat session.")
            self.agent_prune_blocks = copy.deepcopy(current_prune_blocks_on_disk)
            self._last_known_prune_blocks = copy.deepcopy(current_prune_blocks_on_disk)
            try:
                self._initialize_chat_session()
            except Exception as e_reinit:
                print(f"Error ({self.agent_name}): Failed to re-initialize chat session during force_refresh: {e_reinit}. Token count may be inaccurate.")
                return None # Indicate failure to get accurate count
        else:
            print(f"Info ({self.agent_name}): Pruning blocks unchanged. Proceeding to get current token count.")

        return self.get_current_total_session_tokens()

    def reload_session_from_disk(self) -> None:
        """
        Rebuild provider-specific in-memory chat state from the canonical history file.
        """
        self.agent_prune_blocks = self._load_prune_blocks_from_agent_data()
        self._last_known_prune_blocks = copy.deepcopy(self.agent_prune_blocks)
        self.system_prompt = self._load_system_prompt_from_agent_data()
        self._last_known_system_prompt = self.system_prompt
        self._handle_system_prompt_update()
        self._initialize_chat_session()

    def end_session_and_cleanup(self) -> None:
        """Optional: Perform any cleanup."""
        print(f"LLMConnector for {self.agent_name}: Session ending. History saved to {self.history_file_path}")
        pass
