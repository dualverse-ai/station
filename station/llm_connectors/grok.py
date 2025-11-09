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
import requests
from typing import Dict, Any, Optional, List, Tuple

from xai_sdk import Client
from xai_sdk.chat import user, system, assistant

from station import file_io_utils
from station import constants
from .base import (
    BaseLLMConnector,
    LLMConnectorError,
    LLMTransientAPIError,
    LLMPermanentAPIError,
    LLMContextOverflowError
)

class GrokConnector(BaseLLMConnector):
    def __init__(self,
                 model_name: str,
                 agent_name: str,
                 agent_data_path: str,
                 api_key: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 temperature: float = 1.0,
                 max_output_tokens: Optional[int] = None,
                 max_retries: int = constants.LLM_MAX_RETRIES,
                 retry_delay_seconds: int = constants.LLM_RETRY_DELAY_SECONDS):

        super().__init__(model_name, agent_name, agent_data_path,
                         api_key, system_prompt, temperature, max_output_tokens,
                         max_retries, retry_delay_seconds)

        if not self.api_key:
            self.api_key = os.getenv("XAI_API_KEY")

        if not self.api_key:
            raise LLMPermanentAPIError("XAI_API_KEY not provided or set in environment.")

        try:
            self.client = Client(api_key=self.api_key)
        except Exception as e:
            raise LLMPermanentAPIError(f"Failed to initialize XAI client for {self.agent_name}: {e}", original_exception=e)

        self.history_messages: List[Dict[str, Any]] = []
        self._initialize_chat_session()
        print(f"GrokConnector for '{self.agent_name}' initialized with model: '{self.model_name}', temp: {self.temperature}, max_tokens: {self.max_output_tokens}.")

    def _load_history_from_file(self) -> List[Dict[str, Any]]:
        history_for_filtering: List[Dict[str, Any]] = []
        if not os.path.exists(self.history_file_path):
            return history_for_filtering
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
                        history_for_filtering.append({
                            "tick": entry["tick"],
                            "role": entry["role"],
                            "text_content": text_content,
                            "thinking_content": entry.get("thinking_content")
                        })
                    else:
                        print(f"Warning ({self.agent_name}): Entry missing content/parts in {self.history_file_path} for Grok, skipping: {entry}")
                else:
                    print(f"Warning ({self.agent_name}): Malformed history entry in {self.history_file_path} for Grok, skipping: {entry}")
        except Exception as e:
            print(f"Error loading raw chat history for Grok from {self.history_file_path} for {self.agent_name}: {e}.")
        return history_for_filtering

    def _append_turn_to_history_file(self, tick: int, role: str, text: str, thinking_text: Optional[str] = None, token_info: Optional[Dict[str, Optional[int]]] = None) -> None:
        if not text.strip() and not (thinking_text and thinking_text.strip()):
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
            print(f"Error appending turn to history file {self.history_file_path} for Grok {self.agent_name}: {e}")

    def _initialize_chat_session(self) -> None:
        raw_history_with_ticks = self._load_history_from_file()
        processed_history_entries = self._filter_and_prune_history(raw_history_with_ticks)

        grok_ready_history: List[Dict[str, Any]] = []
        if self.system_prompt:
            grok_ready_history.append(system(self.system_prompt))

        for entry in processed_history_entries:
            role = entry['role']
            if role == 'user':
                grok_ready_history.append(user(entry['text_content']))
            elif role == 'model' or role == 'assistant':
                grok_ready_history.append(assistant(entry['text_content']))

        self.history_messages = grok_ready_history
        print(f"Info ({self.agent_name}): Grok history_messages initialized/re-initialized. Length: {len(self.history_messages)}")

    def _get_error_debug_info(self, e: Exception) -> str:
        """Helper method to extract detailed error information for debugging"""
        error_info = f"type={type(e).__name__}, str='{str(e)}'"
        
        # Common attributes for requests exceptions
        if isinstance(e, requests.exceptions.RequestException):
            if e.response is not None:
                error_info += f", status_code={e.response.status_code}, response_text='{e.response.text}'"
        
        # Any other attributes
        if hasattr(e, '__dict__'):
            extra_attrs = {k: v for k, v in e.__dict__.items() if not k.startswith('_')}
            if extra_attrs:
                error_info += f", extra_attrs={extra_attrs}"
        
        return error_info

    def _send_message_implementation(self, user_prompt: str, current_tick: int, attempt_number: int = 0) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        token_info: Dict[str, Optional[int]] = {
            'total_tokens_in_session': None,
            'last_exchange_prompt_tokens': None,
            'last_exchange_completion_tokens': None,
            'last_exchange_cached_tokens': None,
            'last_exchange_thoughts_tokens': None
        }

        messages = self.history_messages + [user(user_prompt)]

        try:
            # The xai-sdk does not use requests directly in a way that exposes the response object easily.
            # We will rely on the exceptions raised by the SDK and inspect them.
            # For now, we assume the SDK handles retries for transient server errors internally.
            # The SDK's error handling is not well-documented, so this is based on observed behavior.

            chat = self.client.chat.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens
            )
            response = chat.sample()

            llm_text_response = response.content
            thinking_text = response.reasoning_content
            
            # Log when response is empty or contains the N/A message
            if not llm_text_response or llm_text_response.strip() == "" or "N/A (Content missing for Markdown rendering)" in llm_text_response:
                error_message = f"Empty or N/A response from Grok for {self.agent_name}"
                print(f"WARNING: {error_message}")
                print(f"  Full response object: {response}")
                print(f"  Response content: '{llm_text_response}'")
                print(f"  Response reasoning: '{thinking_text}'")
                if hasattr(response, '__dict__'):
                    print(f"  Response attributes: {response.__dict__}")
                # raise LLMTransientAPIError(error_message)

            if response.usage:
                token_info['last_exchange_prompt_tokens'] = response.usage.prompt_tokens
                token_info['last_exchange_completion_tokens'] = response.usage.completion_tokens
                token_info['total_tokens_in_session'] = response.usage.total_tokens
                if hasattr(response.usage, 'cached_prompt_text_tokens'):
                    token_info['last_exchange_cached_tokens'] = response.usage.cached_prompt_text_tokens
                if hasattr(response.usage, 'reasoning_tokens'):
                    token_info['last_exchange_thoughts_tokens'] = response.usage.reasoning_tokens

            self.history_messages.append(user(user_prompt))
            self.history_messages.append(assistant(llm_text_response))

            self._append_turn_to_history_file(current_tick, 'user', user_prompt, None, None)
            self._append_turn_to_history_file(current_tick, 'model', llm_text_response, thinking_text, token_info)

            return llm_text_response, thinking_text, token_info

        except Exception as e:
            # Log the raw error first for debugging
            print(f"DEBUG - Raw Grok Exception for {self.agent_name}: {self._get_error_debug_info(e)}")

            # Heuristic-based error classification
            error_str = str(e).lower()
            
            if "rate limit" in error_str or "429" in error_str:
                raise LLMTransientAPIError(f"Grok API rate limit for {self.agent_name}: {e}", original_exception=e)
            
            # Check for specific Grok context overflow error pattern
            if "maximum prompt length is" in error_str and "but the request contains" in error_str and "tokens" in error_str:
                print(f"CRITICAL ({self.agent_name}): Context window overflow detected in Grok API")
                raise LLMContextOverflowError(f"Context window overflow for {self.agent_name}: {str(e)}", original_exception=e)
            
            if "authentication" in error_str or "401" in error_str or "403" in error_str:
                raise LLMPermanentAPIError(f"Grok API authentication error for {self.agent_name}: {e}", original_exception=e)

            if "invalid request" in error_str or "bad request" in error_str or "400" in error_str:
                 raise LLMPermanentAPIError(f"Grok API Bad Request for {self.agent_name}: {e}", original_exception=e)

            # Check for server-side errors (5xx)
            # This is tricky without direct access to status code, relying on string representation
            if any(code in error_str for code in ["500", "502", "503", "504", "server error"]):
                raise LLMTransientAPIError(f"Grok API server error for {self.agent_name}: {e}", original_exception=e)
            
            # Fallback for any other exception
            raise LLMConnectorError(f"Unexpected error in Grok _send_message_implementation for {self.agent_name}: {str(e)}", original_exception=e)

    def get_chat_history(self) -> List[Dict[str, str]]:
        simple_history: List[Dict[str, str]] = []
        for message in self.history_messages:
            role = "user" if message.role == "user" else "model"
            text_content = ""
            if message.content and len(message.content) > 0:
                text_content = message.content[0].text
            simple_history.append({"role": role, "text": text_content})
        return simple_history

    def get_current_total_session_tokens(self) -> Optional[int]:
        if not self.history_messages:
            return 0

        full_text = ""
        for message in self.history_messages:
            if message.content and len(message.content) > 0:
                text_content = message.content[0].text
                full_text += text_content + "\n"

        if not full_text.strip():
            return 0

        try:
            api_url = "https://api.x.ai/v1/tokenize-text"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "text": full_text,
                "model": self.model_name
            }
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            
            token_data = response.json()
            num_tokens = len(token_data.get("token_ids", []))
            return num_tokens

        except requests.exceptions.RequestException as e:
            print(f"Warning ({self.agent_name}): Failed to call tokenizer endpoint: {e}")
            return None
        except Exception as e:
            print(f"Warning ({self.agent_name}): Exception counting total session tokens for Grok: {e}")
            return None
