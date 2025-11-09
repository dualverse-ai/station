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
from typing import Dict, Any, Optional, List, Tuple

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

    @abc.abstractmethod
    def _load_history_from_file(self) -> List[Dict[str, Any]]:
        """Loads chat history... {'tick': int, 'role': str, 'text_content': str}"""
        pass

    @abc.abstractmethod
    def _append_turn_to_history_file(self, tick: int, role: str, text: str, thinking_text: Optional[str] = None, token_info: Optional[Dict[str, Optional[int]]] = None) -> None:
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
                filtered_entries.append({'tick': tick, 'role': role, 'text_content': text_content})
                continue

            # Check if this tick is in any pruned range
            is_pruned = any(start <= tick <= end for start, end, _ in pruned_ranges)
            if not is_pruned:
                filtered_entries.append({'tick': tick, 'role': role, 'text_content': text_content})

        # Insert summary replacements at chronological positions
        final_entries = []
        current_entry_index = 0

        for start_tick, end_tick, summary in sorted(pruned_ranges):
            # Add all entries before this pruned range
            while (current_entry_index < len(filtered_entries) and
                   filtered_entries[current_entry_index].get('tick', 0) < start_tick):
                entry = filtered_entries[current_entry_index]
                final_entries.append({'role': entry['role'], 'text_content': entry['text_content']})
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
            final_entries.append({'role': entry['role'], 'text_content': entry['text_content']})
            current_entry_index += 1

        print(f"Before pruning, raw history length: {len(raw_history_entries)}, after pruning: {len(final_entries)} for {self.agent_name}.")
        return final_entries

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

    def send_message(self, user_prompt: str, current_tick: int) -> Tuple[str, Dict[str, Optional[int]]]:
        # --- Check for pruning updates before sending ---
        current_prune_blocks_on_disk = self._load_prune_blocks_from_agent_data()

        if current_prune_blocks_on_disk != self._last_known_prune_blocks:
            print(f"Info ({self.agent_name}): Pruning blocks changed. Re-initializing chat session.")
            self.agent_prune_blocks = copy.deepcopy(current_prune_blocks_on_disk)
            self._last_known_prune_blocks = copy.deepcopy(current_prune_blocks_on_disk)
            try:
                self._initialize_chat_session() # Re-initialize with new pruning rules
            except Exception as e_reinit:
                # If re-initialization fails, we should probably not proceed with the send_message call
                # or at least be aware that the history might be stale.
                print(f"Error ({self.agent_name}): Failed to re-initialize chat session after pruning update: {e_reinit}. Message may use stale history.")
                # Depending on desired behavior, could raise an error here or return an error tuple.
                # For now, let it proceed, but this is a point of potential failure.
                # Construct a default error token_info
                error_token_info: Dict[str, Optional[int]] = {
                    'total_tokens_in_session': None, 'last_exchange_prompt_tokens': None,
                    'last_exchange_completion_tokens': None, 'last_exchange_cached_tokens': None,
                    'last_exchange_thoughts_tokens': None
                }
                return f"SYSTEM_ERROR: Failed to update chat session with latest pruning rules. Error: {e_reinit}", None, error_token_info

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
                if current_attempt > self.max_retries:
                    print(f"LLMConnector ({self.agent_name}): Max retries ({self.max_retries}) exhausted. Last error: {e}")
                    raise 
                
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
                
                print(f"LLMConnector ({self.agent_name}): API error (Attempt {current_attempt}/{self.max_retries}): {error_details}{raw_error_info}. Retrying in {self.retry_delay_seconds}s...")
                time.sleep(self.retry_delay_seconds)
        
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

    def end_session_and_cleanup(self) -> None:
        """Optional: Perform any cleanup."""
        print(f"LLMConnector for {self.agent_name}: Session ending. History saved to {self.history_file_path}")
        pass