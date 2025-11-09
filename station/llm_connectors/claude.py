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
from typing import Dict, Any, Optional, List, Tuple

import anthropic
from anthropic import Anthropic, APIError, RateLimitError, AuthenticationError, BadRequestError, APIConnectionError, APITimeoutError, InternalServerError

from station import file_io_utils
from station import constants
from .base import (
    BaseLLMConnector,
    LLMConnectorError,
    LLMTransientAPIError,
    LLMPermanentAPIError,
    LLMSafetyBlockError,
    LLMContextOverflowError
)


class ClaudeConnector(BaseLLMConnector):
    def __init__(self,
                 model_name: str,
                 agent_name: str,
                 agent_data_path: str,
                 api_key: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 temperature: float = 1.0,
                 max_output_tokens: Optional[int] = 32000,
                 max_retries: int = constants.LLM_MAX_RETRIES,
                 retry_delay_seconds: int = constants.LLM_RETRY_DELAY_SECONDS):

        super().__init__(model_name, agent_name, agent_data_path, 
                         api_key, system_prompt, temperature, max_output_tokens,
                         max_retries, retry_delay_seconds)

        try:
            self.client = anthropic.Anthropic(api_key=self.api_key) 
        except Exception as e:
            raise LLMPermanentAPIError(f"Failed to initialize Anthropic client for {self.agent_name}: {e}", original_exception=e) 

        # Unified beta headers for all API calls
        self.api_headers = {
            "anthropic-beta": "extended-cache-ttl-2025-04-11"
        }

        # self.history_messages will store history in Anthropic's format AFTER pruning
        self.history_messages: List[Dict[str, Any]] = [] 
        self._initialize_chat_session()

        print(f"ClaudeConnector for '{self.agent_name}' initialized with model: '{self.model_name}', temp: {self.temperature}, max_tokens: {self.max_output_tokens}.")

    def _load_history_from_file(self) -> List[Dict[str, Any]]:
        history_for_filtering: List[Dict[str, Any]] = []
        if not os.path.exists(self.history_file_path):
            return history_for_filtering
        try:
            disk_entries = file_io_utils.load_yaml_lines(self.history_file_path)
            for entry in disk_entries:
                if isinstance(entry, dict) and \
                   "tick" in entry and "role" in entry and "parts" in entry and \
                   isinstance(entry["parts"], list) and entry["parts"]:
                    text_content = "".join(part.get("text", "") for part in entry["parts"] if isinstance(part, dict))
                    # Load thinking_content
                    thinking_content = entry.get("thinking_content") 
                    history_for_filtering.append({
                        "tick": entry["tick"],
                        "role": entry["role"], 
                        "text_content": text_content,
                        "thinking_content": thinking_content
                    })
                else:
                     print(f"Warning ({self.agent_name}): Malformed history entry in {self.history_file_path} for Claude, skipping: {entry}")
        except Exception as e:
            print(f"Error loading raw chat history for Claude from {self.history_file_path} for {self.agent_name}: {e}.")
        return history_for_filtering


    def _append_turn_to_history_file(self, tick: int, role: str, text: str, thinking_text: Optional[str] = None, token_info: Optional[Dict[str, Optional[int]]] = None) -> None:
        if not text.strip() and not (thinking_text and thinking_text.strip()): return # Don't save if both are empty
        try:
            turn_data = {'tick': tick, 'role': role, 'parts': [{'text': text}]}
            if thinking_text:
                turn_data['thinking_content'] = thinking_text
            # Only add token_info for model responses (not user prompts) and if it's provided
            if role == 'model' and token_info:
                turn_data['token_info'] = token_info
            file_io_utils.append_yaml_line(turn_data, self.history_file_path)
        except Exception as e:
            print(f"Error appending turn to history file {self.history_file_path} for Claude {self.agent_name}: {e}")

    def _initialize_chat_session(self) -> None:
        raw_history_with_ticks = self._load_history_from_file() 
        processed_history_entries = self._filter_and_prune_history(raw_history_with_ticks)

        claude_ready_history: List[Dict[str, Any]] = []
        for entry in processed_history_entries:
            claude_role = "user" if entry['role'] == "user" else "assistant" 
            # Skip entries with empty text content to avoid API errors
            if entry.get('text_content', '').strip():
                # Thinking blocks are not part of Claude's message history API payload
                claude_ready_history.append({"role": claude_role, "content": entry['text_content']})
        
        self.history_messages = claude_ready_history
        print(f"Info ({self.agent_name}): Claude history_messages initialized/re-initialized. Length: {len(self.history_messages)}")

    def _send_message_implementation(self, user_prompt: str, current_tick: int, attempt_number: int = 0) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        token_info: Dict[str, Optional[int]] = {
            'total_tokens_in_session': None, 
            'last_exchange_prompt_tokens': None,
            'last_exchange_completion_tokens': None,
            'cache_creation_input_tokens': None,
        }
        extracted_thinking_text: Optional[str] = None
        llm_text_response_parts: List[str] = []
        
        if not user_prompt.strip():
            raise LLMConnectorError("User prompt cannot be empty for Claude _send_message_implementation.")

        api_messages_payload: List[Dict[str, Any]] = []
        
        # Add historical messages without cache control (they'll be cached incrementally)
        for msg in self.history_messages:
            # self.history_messages contains {"role": "user/assistant", "content": "text_string"}
            # Claude API expects content to be a list of blocks.
            # Skip messages with empty content to avoid API errors
            content_str = str(msg.get("content", ""))
            if content_str.strip():  # Only add non-empty messages
                content_block = [{"type": "text", "text": content_str}] 
                
                api_messages_payload.append({
                    "role": msg["role"], # This should be "user" or "assistant"
                    "content": content_block
                })
        
        # Note: Claude API requires alternating user/assistant messages
        # If filtering empty messages causes consecutive user messages, the API will handle it
        
        # Current user prompt WITH cache control for incremental conversation caching
        api_messages_payload.append({
            "role": "user", 
            "content": [{"type": "text", "text": user_prompt, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
        })
        
        # --- ADDED BACK: Token counting and effective_max_tokens adjustment ---
        effective_max_tokens = int(self.max_output_tokens) if self.max_output_tokens is not None and self.max_output_tokens > 0 else 32000 # Default for Claude
        
        # Calculate current history tokens before adding the new user_prompt for this specific calculation
        # self.history_messages is already pruned and in Claude's format
        current_history_tokens_for_calc = 0
        if self.history_messages: # Only count if there's history
            try:
                # Use the most reliable count_tokens method available
                if hasattr(self.client, 'beta') and hasattr(self.client.beta, 'messages') and hasattr(self.client.beta.messages, 'count_tokens'):
                    count_response = self.client.beta.messages.count_tokens(
                        model=self.model_name, 
                        messages=self.history_messages,
                        extra_headers=self.api_headers
                    )
                elif hasattr(self.client, 'count_tokens'): # Fallback, less accurate for message lists
                    combined_text = " ".join([m.get('content', '') for m in self.history_messages if isinstance(m.get('content'), str)])
                    count_response = self.client.count_tokens(text=combined_text)
                else: # Should not happen if client initialized
                    count_response = None

                if count_response:
                    if hasattr(count_response, 'input_tokens'):
                        current_history_tokens_for_calc = count_response.input_tokens
                    elif hasattr(count_response, 'count'):
                        current_history_tokens_for_calc = count_response.count
            except Exception as e_count:
                print(f"Warning ({self.agent_name}): Could not count tokens for Claude history pre-adjustment: {e_count}")

        estimated_input_tokens_for_call = 0
        try:
            if hasattr(self.client, 'beta') and hasattr(self.client.beta, 'messages') and hasattr(self.client.beta.messages, 'count_tokens'):
                count_resp_payload = self.client.beta.messages.count_tokens(
                    model=self.model_name, 
                    messages=api_messages_payload,
                    extra_headers=self.api_headers
                )
                if hasattr(count_resp_payload, 'input_tokens'):
                    estimated_input_tokens_for_call = count_resp_payload.input_tokens
                elif hasattr(count_resp_payload, 'count'):
                    estimated_input_tokens_for_call = count_resp_payload.count
            # else: could do a rough string concat and count if no better method
        except Exception as e_payload_count:
             print(f"Warning ({self.agent_name}): Could not count tokens for Claude api_messages_payload: {e_payload_count}")
             # Fallback: use previous history count + rough estimate for user_prompt
             estimated_input_tokens_for_call = current_history_tokens_for_calc + len(user_prompt.split()) # Very rough

        # Claude's (and many models') context window limit (e.g., 200k) is for INPUT + OUTPUT.
        # So, max_tokens for output should be context_limit - input_tokens.
        MODEL_CONTEXT_WINDOW_LIMIT = 200000 # Example for Claude models
        
        if estimated_input_tokens_for_call + effective_max_tokens > MODEL_CONTEXT_WINDOW_LIMIT:
            original_max_tokens = effective_max_tokens
            effective_max_tokens = MODEL_CONTEXT_WINDOW_LIMIT - estimated_input_tokens_for_call
            effective_max_tokens = int(0.95 * effective_max_tokens) # Add a small buffer
            effective_max_tokens = max(1, effective_max_tokens) # Ensure at least 1 token can be generated
            print(f"Info ({self.agent_name}): Adjusted effective_max_tokens from {original_max_tokens} to {effective_max_tokens} due to context window limit (input: {estimated_input_tokens_for_call}).")
        # --- END ADDED BACK ---

        try:
            # Convert system prompt to proper format for Anthropic API with cache control
            system_messages = []
            if self.system_prompt:
                system_messages = [{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
            
            # Calculate thinking budget with Claude's minimum requirement
            thinking_budget = min(10000, int(effective_max_tokens * 0.5))
            thinking_config = None
            if thinking_budget >= 1024:  # Claude's minimum requirement
                thinking_config = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
            else:
                print(f"Warning ({self.agent_name}): Effective max tokens ({effective_max_tokens}) too low for thinking mode. Disabling thinking for this request.")

            stream_kwargs = {
                "model": self.model_name,
                "max_tokens": effective_max_tokens,
                "temperature": self.temperature,
                "system": system_messages,
                "messages": api_messages_payload,
                "extra_headers": self.api_headers
            }
            
            if thinking_config:
                stream_kwargs["thinking"] = thinking_config

            with self.client.messages.stream(**stream_kwargs) as stream:
                llm_text_response_parts: List[str] = []
                for text_delta in stream.text_stream:
                    llm_text_response_parts.append(text_delta)
                
                llm_text_response = "".join(llm_text_response_parts)
                
                # Ensure we always have a non-empty response to avoid API errors
                if not llm_text_response.strip():
                    llm_text_response = "[No response generated]"
                    
                final_message_snapshot = stream.get_final_message()
                if final_message_snapshot:
                    # Extract thinking from the final message snapshot (only if thinking was enabled)
                    if thinking_config:
                        for block in final_message_snapshot.content:
                            if block.type == 'thinking' and hasattr(block, 'thinking'):
                                extracted_thinking_text = block.thinking
                                break # Assuming one thinking block for now
                    
                    if final_message_snapshot.usage: 
                        token_info['last_exchange_prompt_tokens'] = final_message_snapshot.usage.input_tokens
                        token_info['last_exchange_completion_tokens'] = final_message_snapshot.usage.output_tokens                        
                        token_info['last_exchange_cached_tokens'] = final_message_snapshot.usage.cache_read_input_tokens # type: ignore
                        token_info['cache_creation_input_tokens'] = final_message_snapshot.usage.cache_creation_input_tokens # type: ignore


            self.history_messages.append({"role": "user", "content": user_prompt})
            # Only append assistant response if it's not empty
            if llm_text_response.strip() or (extracted_thinking_text and extracted_thinking_text.strip()):
                self.history_messages.append({"role": "assistant", "content": llm_text_response}) 

            # Append to persistent file history, now including thinking_text and token_info
            self._append_turn_to_history_file(current_tick, 'user', user_prompt, None, None)
            self._append_turn_to_history_file(current_tick, 'model', llm_text_response, extracted_thinking_text, token_info)
            
            token_info['total_tokens_in_session'] = self.get_current_total_session_tokens() 

            return llm_text_response, extracted_thinking_text, token_info

        except anthropic.RateLimitError as e: 
            print(f"DEBUG - Raw Claude RateLimitError for {self.agent_name}: {self._get_error_debug_info(e)}")
            raise LLMTransientAPIError(f"Anthropic API rate limit for {self.agent_name}: {getattr(e, 'message', str(e))}", original_exception=e)
        except anthropic.AuthenticationError as e: 
            print(f"DEBUG - Raw Claude AuthenticationError for {self.agent_name}: {self._get_error_debug_info(e)}")
            raise LLMPermanentAPIError(f"Anthropic API authentication error for {self.agent_name}: {getattr(e, 'message', str(e))}", original_exception=e)
        except anthropic.APIConnectionError as e: 
            print(f"DEBUG - Raw Claude APIConnectionError for {self.agent_name}: {self._get_error_debug_info(e)}")
            raise LLMTransientAPIError(f"Anthropic API connection error for {self.agent_name}: {getattr(e, 'message', str(e))}", original_exception=e)
        except anthropic.APITimeoutError as e: 
            print(f"DEBUG - Raw Claude APITimeoutError for {self.agent_name}: {self._get_error_debug_info(e)}")
            raise LLMTransientAPIError(f"Anthropic API request timed out for {self.agent_name}: {getattr(e, 'message', str(e))}", original_exception=e)
        except anthropic.InternalServerError as e: 
            print(f"DEBUG - Raw Claude InternalServerError for {self.agent_name}: {self._get_error_debug_info(e)}")
            raise LLMTransientAPIError(f"Anthropic API internal server error for {self.agent_name}: {getattr(e, 'message', str(e))}", original_exception=e)
        except anthropic.BadRequestError as e: 
            print(f"DEBUG - Raw Claude BadRequestError for {self.agent_name}: {self._get_error_debug_info(e)}")
            
            # Check for context overflow first - this should terminate the agent session
            if self._is_context_overflow_error(e):
                print(f"CRITICAL ({self.agent_name}): Context window overflow detected in Claude API")
                raise LLMContextOverflowError(f"Context window overflow for {self.agent_name}: {str(e)}", original_exception=e)
            
            error_message = f"Anthropic API Bad Request for {self.agent_name}: {getattr(e, 'message', str(e))}" 
            if hasattr(e, 'body') and e.body and isinstance(e.body, dict) and 'error' in e.body and isinstance(e.body['error'], dict): 
                 err_details = e.body['error'] 
                 err_type = err_details.get('type') 
                 err_msg_detail_api = err_details.get('message') 
                 error_message = f"Anthropic API Bad Request for {self.agent_name} (Type: {err_type}): {err_msg_detail_api or str(e)}" 
                 if err_type == 'overloaded_error': 
                     raise LLMTransientAPIError(error_message, original_exception=e) 
            raise LLMPermanentAPIError(error_message, original_exception=e) 
        except anthropic.APIError as e: 
            print(f"DEBUG - Raw Claude APIError for {self.agent_name}: {self._get_error_debug_info(e)}")
            status_code = getattr(e, 'status_code', None) 
            
            # Check for overloaded_error in the error body (can appear in APIError too, not just BadRequestError)
            if hasattr(e, 'body') and e.body and isinstance(e.body, dict) and 'error' in e.body and isinstance(e.body['error'], dict):
                err_details = e.body['error']
                err_type = err_details.get('type')
                if err_type == 'overloaded_error':
                    err_msg = f"Anthropic API overloaded error for {self.agent_name}: {err_details.get('message', 'Overloaded')}"
                    raise LLMTransientAPIError(err_msg, original_exception=e)
            
            err_msg = f"Anthropic API error (status: {status_code}) for {self.agent_name}: {getattr(e, 'message', str(e))}" 
            if status_code and status_code >= 500: 
                raise LLMTransientAPIError(err_msg, original_exception=e) 
            else: 
                raise LLMPermanentAPIError(err_msg, original_exception=e) 
        except Exception as e: 
            print(f"DEBUG - Raw Claude Exception for {self.agent_name}: {self._get_error_debug_info(e)}")
            raise LLMConnectorError(f"Unexpected error in Claude _send_message_implementation for {self.agent_name}: {str(e)}", original_exception=e) 
 
    def _get_error_debug_info(self, e: Exception) -> str:
        """Helper method to extract detailed error information for debugging"""
        error_info = f"type={type(e).__name__}, str='{str(e)}'"
        
        # Common attributes for Anthropic API errors
        for attr in ['status_code', 'message', 'body', 'response', 'request_id']:
            if hasattr(e, attr):
                value = getattr(e, attr)
                error_info += f", {attr}={repr(value)}"
        
        # Any other attributes
        if hasattr(e, '__dict__'):
            extra_attrs = {k: v for k, v in e.__dict__.items() 
                          if k not in ['status_code', 'message', 'body', 'response', 'request_id'] 
                          and not k.startswith('_')}
            if extra_attrs:
                error_info += f", extra_attrs={extra_attrs}"
        
        return error_info

    def _is_context_overflow_error(self, error: Exception) -> bool:
        """Check if the error indicates context window overflow."""
        error_str = str(error)
        
        # Check for Claude-specific context overflow message patterns
        # Based on production log: "prompt is too long: 200082 tokens > 200000 maximum"
        if "prompt is too long:" in error_str and "tokens >" in error_str and "maximum" in error_str:
            return True
        
        # Check error body for Claude BadRequestError
        if hasattr(error, 'body') and error.body and isinstance(error.body, dict):
            if 'error' in error.body and isinstance(error.body['error'], dict):
                error_details = error.body['error']
                message = error_details.get('message', '')
                if "prompt is too long:" in message and "tokens >" in message and "maximum" in message:
                    return True
        
        return False

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Converts current (pruned) self.history_messages to generic format."""
        simple_history: List[Dict[str, str]] = []
        # self.history_messages is already pruned and in Claude's format
        for message in self.history_messages: 
            role = "user" if message.get("role") == "user" else "model" # Convert "assistant" to "model"
            text_content = message.get("content", "")
            if not isinstance(text_content, str): text_content = str(text_content) 
            simple_history.append({"role": role, "text": text_content})
        return simple_history

    def get_current_total_session_tokens(self) -> Optional[int]:
        """Calculates total tokens for the current, pruned history for Claude."""
        if not self.history_messages: # self.history_messages is already pruned
            return 0
        try:
            if hasattr(self.client, 'beta') and hasattr(self.client.beta, 'messages') and hasattr(self.client.beta.messages, 'count_tokens'):
                 count_response = self.client.beta.messages.count_tokens(
                     model=self.model_name, 
                     messages=self.history_messages,
                     extra_headers=self.api_headers
                 )
            elif hasattr(self.client, 'count_tokens'): 
                 combined_text = " ".join([m.get('content', '') for m in self.history_messages if isinstance(m.get('content'), str)])
                 count_response = self.client.count_tokens(text=combined_text) 
            else:
                print(f"Warning ({self.agent_name}): count_tokens method not found on Anthropic client.")
                return None

            if hasattr(count_response, 'input_tokens'): 
                return count_response.input_tokens
            elif hasattr(count_response, 'count'): 
                return count_response.count
            else:
                print(f"Warning ({self.agent_name}): Could not determine token count from Claude count_tokens response: {count_response}")
                return None
        except Exception as e:
            print(f"Warning ({self.agent_name}): Exception counting total session tokens for Claude: {e}")
            return None