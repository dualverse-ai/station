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

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

try:
    import tiktoken
except ImportError:
    tiktoken = None

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


class OpenAIConnector(BaseLLMConnector):
    def __init__(self,
                 model_name: str,
                 agent_name: str,
                 agent_data_path: str,
                 api_key: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 temperature: float = 1.0,
                 max_output_tokens: Optional[int] = None,
                 max_retries: int = constants.LLM_MAX_RETRIES,
                 retry_delay_seconds: int = constants.LLM_RETRY_DELAY_SECONDS,
                 custom_api_params: Optional[Dict[str, Any]] = None):

        # Initialize attributes needed by BaseLLMConnector before super().__init__
        self.client: Optional[OpenAI] = None
        self.chat_history: List[Dict[str, str]] = []
        self.tiktoken_encoder = None

        # Store custom API params and extract OpenAI-specific parameters
        self.custom_api_params = custom_api_params or {}
        self.verbosity = self.custom_api_params.get('verbosity')

        super().__init__(model_name, agent_name, agent_data_path,
                         api_key, system_prompt, temperature, max_output_tokens,
                         max_retries, retry_delay_seconds)

        effective_api_key = self.api_key 
        if not effective_api_key: 
            effective_api_key = os.getenv("OPENAI_API_KEY")
            if not effective_api_key:
                 raise ValueError(f"OpenAI API key not provided for {agent_name} and OPENAI_API_KEY env variable not set.")
            self.api_key = effective_api_key 

        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise LLMPermanentAPIError(f"Error creating OpenAI client for {agent_name}: {e}.", original_exception=e)
        
        # Initialize tiktoken encoder for local token counting
        self._initialize_tiktoken_encoder()
        
        # Set high max_output_tokens by default if not specified
        if self.max_output_tokens is None:
            if self._is_reasoning_model(self.model_name):
                # For reasoning models, use a high default to allow for reasoning + output
                self.max_output_tokens = 50000  # High default for reasoning models
            else:
                # For regular models, use a reasonable high default
                self.max_output_tokens = 32000   # High default for regular models
        
        self._initialize_chat_session()

        reasoning_note = " (high effort reasoning with summary)" if self._is_reasoning_model(self.model_name) else ""
        tiktoken_note = " with tiktoken" if self.tiktoken_encoder else " (no tiktoken)"
        verbosity_note = f", verbosity: {self.verbosity}" if self.verbosity else ""
        print(f"OpenAIConnector for '{self.agent_name}' initialized with model: '{self.model_name}', temp: {self.temperature}, max_tokens: {self.max_output_tokens}{reasoning_note}{tiktoken_note}{verbosity_note}.")

    def _load_history_from_file(self) -> List[Dict[str, Any]]:
        """Loads history from file, converts to {'tick', 'role', 'text_content', 'thinking_content'}."""
        history_for_filtering: List[Dict[str, Any]] = []
        if os.path.exists(self.history_file_path):
            try:
                disk_entries = file_io_utils.load_yaml_lines(self.history_file_path)
                for entry in disk_entries:
                    if isinstance(entry, dict) and "tick" in entry and "role" in entry:
                        # Support both old format (content) and new format (parts)
                        text_content = None
                        if "content" in entry:
                            # Old format
                            text_content = entry["content"]
                        elif "parts" in entry and isinstance(entry["parts"], list) and len(entry["parts"]) > 0:
                            # New Gemini format
                            text_content = entry["parts"][0].get("text", "")
                        
                        if text_content is not None:
                            history_entry = {
                                "tick": entry["tick"],
                                "role": entry["role"], 
                                "text_content": text_content
                            }
                            # Include thinking_content if present
                            if "thinking_content" in entry:
                                history_entry["thinking_content"] = entry["thinking_content"]
                            history_for_filtering.append(history_entry)
                        else:
                            print(f"Warning ({self.agent_name}): Entry missing content/parts in {self.history_file_path}, skipping: {entry}")
                    else:
                        print(f"Warning ({self.agent_name}): Malformed history entry in {self.history_file_path}, skipping: {entry}")
            except Exception as e:
                print(f"Error loading raw chat history from {self.history_file_path} for {self.agent_name}: {e}.")
        return history_for_filtering

    def _append_turn_to_history_file(self, tick: int, role: str, text: str, thinking_text: Optional[str] = None, token_info: Optional[Dict[str, Optional[int]]] = None) -> None:
        if not text and not thinking_text: # Don't save if both are empty
            return
        try:
            # Convert assistant to model for unified Gemini format
            if role == 'assistant':
                role = 'model'
            # Use Gemini format with parts
            turn_data = {'tick': tick, 'role': role, 'parts': [{'text': text}]}
            if thinking_text:
                turn_data['thinking_content'] = thinking_text
            # Only add token_info for model responses (not user prompts) and if it's provided
            if role == 'model' and token_info:
                turn_data['token_info'] = token_info
            file_io_utils.append_yaml_line(turn_data, self.history_file_path)
        except Exception as e:
            print(f"Error appending turn to history file {self.history_file_path} for {self.agent_name}: {e}")

    def _initialize_chat_session(self) -> None:
        if not self.client:
            raise ConnectionError(f"OpenAI client not initialized for {self.agent_name}.")

        raw_history_with_ticks = self._load_history_from_file()
        # self.agent_pruned_ticks_info is used by _filter_and_prune_history
        processed_history_entries = self._filter_and_prune_history(raw_history_with_ticks)

        self.chat_history = []
        
        # Add system message if provided
        if self.system_prompt:
            self.chat_history.append({
                "role": "system",
                "content": self.system_prompt
            })

        # Convert processed history to OpenAI format
        for entry in processed_history_entries:
            role = entry['role']
            # OpenAI uses 'assistant' instead of 'model'
            if role == 'model':
                role = 'assistant'
            elif role not in ['user', 'assistant', 'system']:
                print(f"Warning ({self.agent_name}): Invalid role '{role}' in processed history, defaulting to 'user'. Entry: {entry}")
                role = 'user'

            self.chat_history.append({
                "role": role,
                "content": entry['text_content']
            })
        
        print(f"Info ({self.agent_name}): OpenAI chat session initialized/re-initialized. History length: {len(self.chat_history)}")
        
    def _send_message_implementation(self, user_prompt: str, current_tick: int, attempt_number: int = 0) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        token_info: Dict[str, Optional[int]] = {
            'total_tokens_in_session': None,
            'last_exchange_prompt_tokens': None,
            'last_exchange_completion_tokens': None,
            'last_exchange_cached_tokens': None,
            'last_exchange_thoughts_tokens': None
        }

        if not self.client:
             err_msg = f"SYSTEM_ERROR: OpenAI client for {self.agent_name} is not available in _send_message_implementation."
             print(f"Error ({self.agent_name}): {err_msg}")
             return err_msg, None, token_info
        
        # Determine if this is a reasoning model
        is_reasoning_model = self._is_reasoning_model(self.model_name)
        
        if is_reasoning_model:
            return self._send_message_with_responses_api(user_prompt, current_tick, token_info, attempt_number)
        else:
            return self._send_message_with_chat_api(user_prompt, current_tick, token_info, attempt_number)
    
    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if the model is a reasoning model (o-series or gpt-5)."""
        reasoning_models = ["o1", "o3", "o4", "o4-mini", "o3-mini", "gpt-5", "gpt5"]
        return any(model_name.startswith(prefix) for prefix in reasoning_models)
    
    def _is_context_overflow_error(self, error: Exception) -> bool:
        """Check if the error indicates context window overflow."""
        # Most reliable: check error code first
        if hasattr(error, 'body') and error.body and isinstance(error.body, dict):
            body_code = error.body.get('code', '')
            if body_code == 'context_length_exceeded':
                return True
            
            # Check exact message from error body
            body_message = error.body.get('message', '')
            if body_message == 'Your input exceeds the context window of this model. Please adjust your input and try again.':
                return True
        
        # Fallback: check if full error string contains the exact OpenAI context overflow message
        error_str = str(error)
        return 'Your input exceeds the context window of this model. Please adjust your input and try again.' in error_str
    
    def _initialize_tiktoken_encoder(self):
        """Initialize tiktoken encoder for the specific model."""
        if tiktoken is None:
            print(f"Warning ({self.agent_name}): tiktoken not available. Install with: pip install tiktoken")
            return
        
        try:
            # Try to get model-specific encoder first
            self.tiktoken_encoder = tiktoken.encoding_for_model(self.model_name)
            print(f"Info ({self.agent_name}): Using tiktoken encoder for model '{self.model_name}'")
        except Exception as e:
            # Fall back to a common encoder if model-specific one isn't available
            try:
                # Check specifically for o-series models vs GPT-5
                if self.model_name.startswith(("o1", "o3", "o4")):
                    # o-series models use o200k_base encoding
                    self.tiktoken_encoder = tiktoken.get_encoding("o200k_base")
                    print(f"Info ({self.agent_name}): Using o200k_base encoder for o-series model '{self.model_name}'")
                else:
                    # GPT-5 and most other OpenAI models use cl100k_base
                    self.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
                    print(f"Info ({self.agent_name}): Using cl100k_base encoder for model '{self.model_name}'")
            except Exception as fallback_e:
                print(f"Warning ({self.agent_name}): Failed to initialize tiktoken encoder: {e}, fallback: {fallback_e}")
                self.tiktoken_encoder = None
    
    def _count_tokens_with_tiktoken(self, messages: List[Dict[str, str]]) -> Optional[int]:
        """Count tokens using tiktoken for the given messages."""
        if self.tiktoken_encoder is None:
            print(f"Debug ({self.agent_name}): tiktoken_encoder is None - cannot count tokens")
            return None

        try:
            # Count tokens for the messages format used by OpenAI API
            total_tokens = 0

            for i, message in enumerate(messages):
                # Each message has some overhead tokens
                total_tokens += 4  # Base tokens per message

                # Add tokens for role
                if "role" in message:
                    total_tokens += len(self.tiktoken_encoder.encode(message["role"]))
                else:
                    print(f"Debug ({self.agent_name}): Message {i} missing 'role' field: {message}")

                # Add tokens for content
                if "content" in message and message["content"]:
                    total_tokens += len(self.tiktoken_encoder.encode(message["content"]))
                elif "content" not in message:
                    print(f"Debug ({self.agent_name}): Message {i} missing 'content' field: {message}")
                elif message["content"] is None:
                    print(f"Debug ({self.agent_name}): Message {i} has None content: {message}")
                else:
                    print(f"Debug ({self.agent_name}): Message {i} has empty content: {message}")

            # Add some overhead for the API call structure
            total_tokens += 2  # priming tokens

            return total_tokens
        except Exception as e:
            print(f"Warning ({self.agent_name}): Error counting tokens with tiktoken: {type(e).__name__}: {e}")
            print(f"Debug ({self.agent_name}): Messages structure when error occurred: {len(messages)} messages")
            if messages and len(messages) > 0:
                print(f"Debug ({self.agent_name}): First message keys: {list(messages[0].keys()) if isinstance(messages[0], dict) else 'not a dict'}")
            import traceback
            traceback.print_exc()
            return None
    
    def _send_message_with_responses_api(self, user_prompt: str, current_tick: int, token_info: Dict[str, Optional[int]], attempt_number: int = 0) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        """Handle reasoning models using the Responses API."""
        # Use streaming for retry attempts to avoid timeout issues, or if forced via constant
        if attempt_number > 0 or constants.OPENAI_FORCE_STREAMING:
            if attempt_number > 0:
                print(f"Info ({self.agent_name}): Using streaming for retry attempt {attempt_number} to avoid timeout issues")
            else:
                print(f"Info ({self.agent_name}): Using streaming mode (forced via OPENAI_FORCE_STREAMING)")
            return self._send_message_with_responses_api_stream(user_prompt, current_tick, token_info)
        
        try:
            # Build input for Responses API
            input_messages = []
            for msg in self.chat_history:
                input_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add the current user message
            input_messages.append({
                "role": "user", 
                "content": user_prompt
            })
            
            # Prepare parameters for Responses API
            api_params = {
                "model": self.model_name,
                "input": input_messages,
                "reasoning": {
                    "effort": "high",   # Use high effort by default for GPT-5 and other reasoning models
                    "summary": "detailed"  # Request detailed summary
                },
                "store": True  # Store the response for potential future retrieval
            }

            # Add verbosity if specified (OpenAI only, for Responses API)
            if self.verbosity and self.verbosity in ["low", "medium", "high"]:
                api_params["text"] = {"verbosity": self.verbosity}

            if self.max_output_tokens:
                api_params["max_output_tokens"] = self.max_output_tokens

            response = self.client.responses.create(**api_params)
            
            # Log the response ID for checking on OpenAI website
            if hasattr(response, 'id'):
                print(f"INFO ({self.agent_name}): OpenAI Response ID: {response.id} (can be viewed at https://platform.openai.com/)")

            # GPT-5 might return empty output_text if it's only thinking
            llm_text_response = response.output_text if response.output_text else ""
            thinking_text = None
            
            # Extract reasoning summary if available
            if hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    # Check if this is a reasoning item with a summary
                    if hasattr(output_item, 'type') and output_item.type == 'reasoning':
                        if hasattr(output_item, 'summary') and output_item.summary:
                            # Extract summary text - it's a list of Summary objects
                            if isinstance(output_item.summary, list):
                                summary_parts = []
                                for item in output_item.summary:
                                    # Handle Summary objects with text attribute
                                    if hasattr(item, 'text'):
                                        summary_parts.append(item.text)
                                    # Handle dict representation
                                    elif isinstance(item, dict) and 'text' in item:
                                        summary_parts.append(item['text'])
                                    # Handle string directly
                                    elif isinstance(item, str):
                                        summary_parts.append(item)
                                if summary_parts:
                                    thinking_text = "\n\n".join(summary_parts)
                                elif isinstance(output_item.summary, str):
                                    thinking_text = output_item.summary
                                
                                if thinking_text:
                                    print(f"Info ({self.agent_name}): Extracted reasoning summary ({len(thinking_text)} chars)")
                        break  # Stop after finding the first reasoning item
            
            # Check if we got anything meaningful
            if not llm_text_response and not thinking_text:
                raise LLMSafetyBlockError(
                    f"LLM response generation failed for {self.agent_name}. No output text or reasoning.",
                    block_reason="empty_response"
                )
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": user_prompt})
            self.chat_history.append({"role": "assistant", "content": llm_text_response})
            
            self._append_turn_to_history_file(current_tick, 'user', user_prompt, None, None) 
            self._append_turn_to_history_file(current_tick, 'model', llm_text_response, thinking_text, token_info)

            # Extract token usage information for reasoning models
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                token_info['last_exchange_prompt_tokens'] = getattr(usage, 'input_tokens', None)
                token_info['last_exchange_completion_tokens'] = getattr(usage, 'output_tokens', None)
                token_info['total_tokens_in_session'] = getattr(usage, 'total_tokens', None)
                
                # Handle cached tokens
                if hasattr(usage, 'input_tokens_details') and usage.input_tokens_details:
                    token_info['last_exchange_cached_tokens'] = getattr(usage.input_tokens_details, 'cached_tokens', None)
                else:
                    token_info['last_exchange_cached_tokens'] = None
                
                # Handle reasoning tokens (equivalent to thinking tokens)
                if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
                    token_info['last_exchange_thoughts_tokens'] = getattr(usage.output_tokens_details, 'reasoning_tokens', None)
                else:
                    token_info['last_exchange_thoughts_tokens'] = None
            
            # Use tiktoken for total session tokens if API didn't provide it
            if token_info['total_tokens_in_session'] is None:
                tiktoken_count = self._count_tokens_with_tiktoken(self.chat_history)
                if tiktoken_count is not None:
                    token_info['total_tokens_in_session'] = tiktoken_count
            
            return llm_text_response, thinking_text, token_info
            
        except Exception as e:
            # Check for context overflow first - this should terminate the agent session
            if self._is_context_overflow_error(e):
                print(f"CRITICAL ({self.agent_name}): Context window overflow detected in Responses API")
                raise LLMContextOverflowError(f"Context window overflow for {self.agent_name}: {str(e)}", original_exception=e)
            
            # Fall back to chat API if Responses API fails, but reasoning models may not work with Chat API
            print(f"Warning ({self.agent_name}): Responses API failed for reasoning model '{self.model_name}': {e}")
            if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                # This is likely a reasoning model that doesn't support Chat Completions API
                raise LLMPermanentAPIError(f"Reasoning model '{self.model_name}' requires Responses API, but it failed: {e}", original_exception=e)
            else:
                print(f"Attempting fallback to Chat API for reasoning model...")
                return self._send_message_with_chat_api(user_prompt, current_tick, token_info, attempt_number)
    
    def _send_message_with_chat_api(self, user_prompt: str, current_tick: int, token_info: Dict[str, Optional[int]], attempt_number: int = 0) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        """Handle regular models using the Chat Completions API."""
        # Check if this is a reasoning model being forced to use Chat API
        if self._is_reasoning_model(self.model_name):
            raise LLMPermanentAPIError(f"Reasoning model '{self.model_name}' cannot use Chat Completions API. Please use a regular model like 'gpt-4o' or ensure Responses API is working.")
        
        # Use streaming for retry attempts to avoid timeout issues, or if forced via constant
        if attempt_number > 0 or constants.OPENAI_FORCE_STREAMING:
            if attempt_number > 0:
                print(f"Info ({self.agent_name}): Using streaming for retry attempt {attempt_number} to avoid timeout issues")
            else:
                print(f"Info ({self.agent_name}): Using streaming mode (forced via OPENAI_FORCE_STREAMING)")
            return self._send_message_with_chat_api_stream(user_prompt, current_tick, token_info)
        
        # Track if we've already saved to history file
        history_saved = False
        
        # Add user message to history
        self.chat_history.append({
            "role": "user",
            "content": user_prompt
        })

        try:
            # Prepare parameters for API call
            api_params = {
                "model": self.model_name,
                "messages": self.chat_history,
                "temperature": self.temperature
            }
            
            if self.max_output_tokens:
                # Use max_completion_tokens for reasoning models, max_tokens for regular models
                if self._is_reasoning_model(self.model_name):
                    api_params["max_completion_tokens"] = self.max_output_tokens
                else:
                    api_params["max_tokens"] = self.max_output_tokens

            response: ChatCompletion = self.client.chat.completions.create(**api_params)

            if not response.choices:
                raise LLMSafetyBlockError(
                    f"LLM response generation failed for {self.agent_name}. No choices returned.",
                    block_reason="no_choices"
                )

            choice = response.choices[0]
            if not choice.message or not choice.message.content:
                raise LLMSafetyBlockError(
                    f"LLM response generation failed for {self.agent_name}. Empty message content.",
                    block_reason="empty_content"
                )

            llm_text_response = choice.message.content
            thinking_text = None  # Regular models don't have thinking mode
            
            # Add assistant response to history
            self.chat_history.append({
                "role": "assistant",
                "content": llm_text_response
            })
            
            if not history_saved:
                self._append_turn_to_history_file(current_tick, 'user', user_prompt, None, None) 
                self._append_turn_to_history_file(current_tick, 'model', llm_text_response, thinking_text, token_info)
                history_saved = True

            # Extract token usage information
            if response.usage:
                token_info['last_exchange_prompt_tokens'] = response.usage.prompt_tokens
                token_info['last_exchange_completion_tokens'] = response.usage.completion_tokens
                token_info['total_tokens_in_session'] = response.usage.total_tokens
                # Regular chat API doesn't provide cached tokens or thinking tokens
                token_info['last_exchange_cached_tokens'] = None
                token_info['last_exchange_thoughts_tokens'] = None
            
            # Use tiktoken for total session tokens if API didn't provide it
            if token_info['total_tokens_in_session'] is None:
                tiktoken_count = self._count_tokens_with_tiktoken(self.chat_history)
                if tiktoken_count is not None:
                    token_info['total_tokens_in_session'] = tiktoken_count
            
            return llm_text_response, thinking_text, token_info

        except openai.APIConnectionError as e:
            print(f"DEBUG - OpenAI Connection Error for {self.agent_name}: {str(e)}")
            raise LLMTransientAPIError(f"OpenAI API Connection Error for {self.agent_name}: {str(e)}", original_exception=e)
        except openai.RateLimitError as e:
            print(f"DEBUG - OpenAI Rate Limit Error for {self.agent_name}: {str(e)}")
            raise LLMTransientAPIError(f"OpenAI API Rate Limit Error for {self.agent_name}: {str(e)}", original_exception=e)
        except openai.APIStatusError as e:
            print(f"DEBUG - OpenAI API Status Error for {self.agent_name}: status={e.status_code}, message={str(e)}")
            
            # Check for context overflow first - this should terminate the agent session
            if self._is_context_overflow_error(e):
                print(f"CRITICAL ({self.agent_name}): Context window overflow detected in Chat API")
                raise LLMContextOverflowError(f"Context window overflow for {self.agent_name}: {str(e)}", original_exception=e)
            
            # Handle specific parameter issues
            if e.status_code == 400 and "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                print(f"INFO - Retrying with max_completion_tokens for model {self.model_name}")
                # Retry with max_completion_tokens instead of max_tokens
                try:
                    api_params_retry = api_params.copy()
                    if "max_tokens" in api_params_retry:
                        api_params_retry["max_completion_tokens"] = api_params_retry.pop("max_tokens")
                    
                    response: ChatCompletion = self.client.chat.completions.create(**api_params_retry)
                    
                    if not response.choices:
                        raise LLMSafetyBlockError(f"LLM response generation failed for {self.agent_name}. No choices returned.", block_reason="no_choices")

                    choice = response.choices[0]
                    if not choice.message or not choice.message.content:
                        raise LLMSafetyBlockError(f"LLM response generation failed for {self.agent_name}. Empty message content.", block_reason="empty_content")

                    llm_text_response = choice.message.content
                    thinking_text = None  # Regular models don't have thinking mode
                    
                    # Remove the user message that was added before the first attempt
                    if self.chat_history and self.chat_history[-1]["role"] == "user":
                        self.chat_history.pop()
                    
                    # Now add the assistant response to history
                    self.chat_history.append({"role": "assistant", "content": llm_text_response})
                    
                    # Save to history file if not already saved
                    if not history_saved:
                        self._append_turn_to_history_file(current_tick, 'user', user_prompt, None, None) 
                        self._append_turn_to_history_file(current_tick, 'model', llm_text_response, thinking_text, token_info)
                        history_saved = True

                    # Extract token usage information
                    if response.usage:
                        token_info['last_exchange_prompt_tokens'] = response.usage.prompt_tokens
                        token_info['last_exchange_completion_tokens'] = response.usage.completion_tokens
                        token_info['total_tokens_in_session'] = response.usage.total_tokens
                        token_info['last_exchange_cached_tokens'] = None
                        token_info['last_exchange_thoughts_tokens'] = None
                    
                    # Use tiktoken for total session tokens if API didn't provide it
                    if token_info['total_tokens_in_session'] is None:
                        tiktoken_count = self._count_tokens_with_tiktoken(self.chat_history)
                        if tiktoken_count is not None:
                            token_info['total_tokens_in_session'] = tiktoken_count
                    
                    return llm_text_response, thinking_text, token_info
                    
                except Exception as retry_e:
                    print(f"DEBUG - Retry with max_completion_tokens also failed for {self.agent_name}: {retry_e}")
                    # Fall through to original error handling
            
            if e.status_code >= 500:
                raise LLMTransientAPIError(f"OpenAI API Server Error for {self.agent_name}: {str(e)}", original_exception=e)
            else:
                raise LLMPermanentAPIError(f"OpenAI API Client Error for {self.agent_name}: {str(e)}", original_exception=e)
        except Exception as e:
            print(f"DEBUG - Unexpected OpenAI Error for {self.agent_name}: type={type(e).__name__}, str={str(e)}")
            if hasattr(e, '__dict__'):
                error_attrs = {k: v for k, v in e.__dict__.items() if not k.startswith('_')}
                if error_attrs:
                    print(f"DEBUG - Exception Attributes: {error_attrs}")
            
            import traceback; traceback.print_exc()
            raise LLMConnectorError(f"Unexpected OpenAI API call failure for {self.agent_name}. Details: {str(e)}", original_exception=e)
    
    def _send_message_with_chat_api_stream(self, user_prompt: str, current_tick: int, token_info: Dict[str, Optional[int]]) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        """Handle regular models using the Chat Completions API with streaming."""
        # Track if we've already saved to history file
        history_saved = False
        
        # Add user message to history
        self.chat_history.append({
            "role": "user",
            "content": user_prompt
        })

        try:
            # Prepare parameters for streaming API call
            api_params = {
                "model": self.model_name,
                "messages": self.chat_history,
                "temperature": self.temperature,
                "stream": True  # Enable streaming
            }
            
            if self.max_output_tokens:
                api_params["max_tokens"] = self.max_output_tokens

            # Stream the response
            stream = self.client.chat.completions.create(**api_params)
            
            # Collect response chunks
            llm_text_response_parts = []
            final_usage = None
            
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        llm_text_response_parts.append(delta.content)
                
                # Usage metadata is typically in the last chunk
                if hasattr(chunk, 'usage') and chunk.usage:
                    final_usage = chunk.usage
            
            # Combine all response parts
            llm_text_response = "".join(llm_text_response_parts)
            thinking_text = None  # Regular models don't have thinking mode
            
            if not llm_text_response:
                raise LLMSafetyBlockError(
                    f"LLM response generation failed for {self.agent_name}. Empty response from streaming.",
                    block_reason="empty_stream_response"
                )
            
            # Add assistant response to history
            self.chat_history.append({
                "role": "assistant",
                "content": llm_text_response
            })
            
            if not history_saved:
                self._append_turn_to_history_file(current_tick, 'user', user_prompt, None, None) 
                self._append_turn_to_history_file(current_tick, 'model', llm_text_response, thinking_text, token_info)
                history_saved = True

            # Extract token usage information
            if final_usage:
                token_info['last_exchange_prompt_tokens'] = getattr(final_usage, 'prompt_tokens', None)
                token_info['last_exchange_completion_tokens'] = getattr(final_usage, 'completion_tokens', None)
                token_info['total_tokens_in_session'] = getattr(final_usage, 'total_tokens', None)
                # Streaming chat API doesn't provide cached tokens or thinking tokens
                token_info['last_exchange_cached_tokens'] = None
                token_info['last_exchange_thoughts_tokens'] = None
            
            # Use tiktoken for total session tokens if API didn't provide it
            if token_info['total_tokens_in_session'] is None:
                tiktoken_count = self._count_tokens_with_tiktoken(self.chat_history)
                if tiktoken_count is not None:
                    token_info['total_tokens_in_session'] = tiktoken_count
            
            return llm_text_response, thinking_text, token_info

        except openai.APIConnectionError as e:
            # Remove the user message we added since the request failed
            if self.chat_history and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()
            print(f"DEBUG - OpenAI Connection Error (streaming) for {self.agent_name}: {str(e)}")
            raise LLMTransientAPIError(f"OpenAI API Connection Error for {self.agent_name}: {str(e)}", original_exception=e)
        except openai.RateLimitError as e:
            # Remove the user message we added since the request failed
            if self.chat_history and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()
            print(f"DEBUG - OpenAI Rate Limit Error (streaming) for {self.agent_name}: {str(e)}")
            raise LLMTransientAPIError(f"OpenAI API Rate Limit Error for {self.agent_name}: {str(e)}", original_exception=e)
        except openai.APIStatusError as e:
            # Remove the user message we added since the request failed
            if self.chat_history and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()
            print(f"DEBUG - OpenAI API Status Error (streaming) for {self.agent_name}: status={e.status_code}, message={str(e)}")
            
            # Check for context overflow first - this should terminate the agent session
            if self._is_context_overflow_error(e):
                print(f"CRITICAL ({self.agent_name}): Context window overflow detected in Chat API streaming")
                raise LLMContextOverflowError(f"Context window overflow for {self.agent_name}: {str(e)}", original_exception=e)
            
            if e.status_code >= 500:
                raise LLMTransientAPIError(f"OpenAI API Server Error for {self.agent_name}: {str(e)}", original_exception=e)
            else:
                raise LLMPermanentAPIError(f"OpenAI API Client Error for {self.agent_name}: {str(e)}", original_exception=e)
        except Exception as e:
            # Remove the user message we added since the request failed
            if self.chat_history and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()
            print(f"DEBUG - Unexpected OpenAI Error (streaming) for {self.agent_name}: type={type(e).__name__}, str={str(e)}")
            import traceback; traceback.print_exc()
            raise LLMConnectorError(f"Unexpected OpenAI API streaming failure for {self.agent_name}. Details: {str(e)}", original_exception=e)
    
    def _send_message_with_responses_api_stream(self, user_prompt: str, current_tick: int, token_info: Dict[str, Optional[int]]) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        """Handle reasoning models using the Responses API with streaming."""
        try:
            # Build input for Responses API
            input_messages = []
            for msg in self.chat_history:
                input_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add the current user message
            input_messages.append({
                "role": "user", 
                "content": user_prompt
            })
            
            # Prepare parameters for streaming Responses API
            api_params = {
                "model": self.model_name,
                "input": input_messages,
                "reasoning": {
                    "effort": "high",   # Use high effort by default for GPT-5 and other reasoning models
                    "summary": "detailed"  # Request detailed summary
                },
                "store": True,  # Store the response for potential future retrieval
                "stream": True  # Enable streaming
            }

            # Add verbosity if specified (OpenAI only, for Responses API)
            if self.verbosity and self.verbosity in ["low", "medium", "high"]:
                api_params["text"] = {"verbosity": self.verbosity}

            if self.max_output_tokens:
                api_params["max_output_tokens"] = self.max_output_tokens

            # Stream the response
            stream = self.client.responses.create(**api_params)
            
            # Variables to collect streaming data
            llm_text_response_parts = []
            thinking_summary_parts = []
            response_id = None
            final_usage = None
            
            # Process streaming events
            for event in stream:
                # Capture response ID from first event
                if hasattr(event, 'response') and hasattr(event.response, 'id') and not response_id:
                    response_id = event.response.id
                    print(f"INFO ({self.agent_name}): OpenAI Response ID (streaming): {response_id}")
                
                # Check event type
                if hasattr(event, 'type'):
                    # Collect output text deltas - delta is directly on event
                    if event.type == 'response.output_text.delta':
                        if hasattr(event, 'delta'):
                            llm_text_response_parts.append(event.delta)
                    
                    # Collect reasoning summary deltas
                    elif event.type == 'response.reasoning_summary_text.delta':
                        if hasattr(event, 'delta'):
                            thinking_summary_parts.append(event.delta)
                    
                    # Extract final usage from completed event
                    elif event.type == 'response.completed':
                        if hasattr(event, 'response') and hasattr(event.response, 'usage'):
                            final_usage = event.response.usage
                
                # Handle usage metadata from events
                if hasattr(event, 'usage') and event.usage:
                    final_usage = event.usage
            
            # Combine collected parts
            llm_text_response = "".join(llm_text_response_parts)
            thinking_text = "".join(thinking_summary_parts) if thinking_summary_parts else None
            
            # Check if we got anything meaningful
            if not llm_text_response and not thinking_text:
                raise LLMSafetyBlockError(
                    f"LLM response generation failed for {self.agent_name}. No output text or reasoning from streaming.",
                    block_reason="empty_stream_response"
                )
            
            if thinking_text:
                print(f"Info ({self.agent_name}): Extracted reasoning summary from stream ({len(thinking_text)} chars)")
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": user_prompt})
            self.chat_history.append({"role": "assistant", "content": llm_text_response})
            
            self._append_turn_to_history_file(current_tick, 'user', user_prompt, None, None) 
            self._append_turn_to_history_file(current_tick, 'model', llm_text_response, thinking_text, token_info)

            # Extract token usage information
            if final_usage:
                token_info['last_exchange_prompt_tokens'] = getattr(final_usage, 'input_tokens', None)
                token_info['last_exchange_completion_tokens'] = getattr(final_usage, 'output_tokens', None)
                token_info['total_tokens_in_session'] = getattr(final_usage, 'total_tokens', None)
                
                # Handle cached tokens
                if hasattr(final_usage, 'input_tokens_details') and final_usage.input_tokens_details:
                    token_info['last_exchange_cached_tokens'] = getattr(final_usage.input_tokens_details, 'cached_tokens', None)
                else:
                    token_info['last_exchange_cached_tokens'] = None
                
                # Handle reasoning tokens
                if hasattr(final_usage, 'output_tokens_details') and final_usage.output_tokens_details:
                    token_info['last_exchange_thoughts_tokens'] = getattr(final_usage.output_tokens_details, 'reasoning_tokens', None)
                else:
                    token_info['last_exchange_thoughts_tokens'] = None
            
            # Use tiktoken for total session tokens if API didn't provide it
            if token_info['total_tokens_in_session'] is None:
                tiktoken_count = self._count_tokens_with_tiktoken(self.chat_history)
                if tiktoken_count is not None:
                    token_info['total_tokens_in_session'] = tiktoken_count
            
            return llm_text_response, thinking_text, token_info
            
        except openai.APIConnectionError as e:
            print(f"DEBUG - OpenAI Connection Error (Responses streaming) for {self.agent_name}: {str(e)}")
            raise LLMTransientAPIError(f"OpenAI API Connection Error for {self.agent_name}: {str(e)}", original_exception=e)
        except Exception as e:
            # Check for context overflow first - this should terminate the agent session
            if self._is_context_overflow_error(e):
                print(f"CRITICAL ({self.agent_name}): Context window overflow detected in Responses API streaming")
                raise LLMContextOverflowError(f"Context window overflow for {self.agent_name}: {str(e)}", original_exception=e)
            
            print(f"WARNING ({self.agent_name}): Responses API streaming failed: {e}")
            # Some reasoning models might not support streaming, fall back to non-streaming
            print(f"INFO ({self.agent_name}): Falling back to non-streaming Responses API")
            # Remove the attempt_number to prevent infinite recursion
            return self._send_message_with_responses_api(user_prompt, current_tick, token_info, attempt_number=0)

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Returns the current (pruned) chat history from the active session."""
        # If no chat session, reconstruct from file (similar to Gemini)
        if not self.chat_history:
            print(f"Warning ({self.agent_name}): get_chat_history called but no active chat session. Attempting to reconstruct from file.")
            raw_history_with_ticks = self._load_history_from_file()
            processed_history_entries = self._filter_and_prune_history(raw_history_with_ticks)
            return [{'role': entry['role'], 
                     'text': entry['text_content'], 
                     'thinking': entry.get('thinking_content')} 
                    for entry in processed_history_entries]
        
        # For active chat session, return simple format (like Gemini does)
        # The OpenAI SDK doesn't store thinking content in the chat history
        simple_history: List[Dict[str, str]] = []
        for message in self.chat_history:
            simple_history.append({
                "role": message["role"],
                "text": message["content"],
                "thinking": None  # Thinking content not available in active session
            })
        return simple_history
    
    def get_current_total_session_tokens(self) -> Optional[int]:
        """Calculates total tokens based on the current chat session history using tiktoken."""
        if not self.chat_history:
            return 0

        # Use tiktoken for local token counting (fast and free)
        tiktoken_count = self._count_tokens_with_tiktoken(self.chat_history)
        if tiktoken_count is not None:
            return tiktoken_count

        # If tiktoken fails, return None instead of making expensive API calls
        print(f"Warning ({self.agent_name}): Unable to count tokens with tiktoken. Token count unavailable.")
        return None