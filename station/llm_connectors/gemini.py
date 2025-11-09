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

# Use the import style from the provided Google examples
from google import genai
from google.genai import types as google_genai_types
from google.genai import errors as google_genai_errors

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


class GoogleGeminiConnector(BaseLLMConnector):
    def __init__(self,
                 model_name: str,
                 agent_name: str,
                 agent_data_path: str, 
                 api_key: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 temperature: float = 2.0, 
                 max_output_tokens: Optional[int] = None,
                 max_retries: int = constants.LLM_MAX_RETRIES,
                 retry_delay_seconds: int = constants.LLM_RETRY_DELAY_SECONDS):
        
        # Initialize attributes needed by BaseLLMConnector before super().__init__
        # if _initialize_chat_session in super() needs them.
        # In this revised plan, _initialize_chat_session is called at the end of this __init__.
        self.client: Optional[genai.Client] = None
        self.chat_session: Optional[genai.Chat] = None
        self.generation_config: Optional[google_genai_types.GenerateContentConfig] = None

        super().__init__(model_name, agent_name, agent_data_path,
                         api_key, system_prompt, temperature, max_output_tokens,
                         max_retries, retry_delay_seconds)

        effective_api_key = self.api_key 
        if not effective_api_key: 
            effective_api_key = os.getenv("GOOGLE_API_KEY")
            if not effective_api_key:
                 raise ValueError(f"Google API key not provided for {agent_name} and GOOGLE_API_KEY env variable not set.")
            self.api_key = effective_api_key 

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise LLMPermanentAPIError(f"Error creating genai.Client for {agent_name}: {e}.", original_exception=e)
        
        valid_safety_settings = []
        for cat_name in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]:
            if hasattr(google_genai_types.HarmCategory, cat_name):
                valid_safety_settings.append(google_genai_types.SafetySetting(
                    category=getattr(google_genai_types.HarmCategory, cat_name),
                    threshold=google_genai_types.HarmBlockThreshold.BLOCK_NONE
                ))
        
        self.generation_config = google_genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens, 
            safety_settings=valid_safety_settings,
            system_instruction=self.system_prompt,
            thinking_config=google_genai_types.ThinkingConfig(thinking_budget=24576, include_thoughts=True)
        )
        
        self._initialize_chat_session()

        print(f"GoogleGeminiConnector for '{self.agent_name}' initialized with model: '{self.model_name}', temp: {self.temperature}.")

    def _load_history_from_file(self) -> List[Dict[str, Any]]:
        """Loads history from file, converts to {'tick', 'role', 'text_content'}."""
        history_for_filtering: List[Dict[str, Any]] = []
        if os.path.exists(self.history_file_path):
            try:
                disk_entries = file_io_utils.load_yaml_lines(self.history_file_path)
                for entry in disk_entries:
                    if isinstance(entry, dict) and \
                       "tick" in entry and "role" in entry and "parts" in entry and \
                       isinstance(entry["parts"], list) and entry["parts"]:
                        text_content = "".join(part.get("text", "") for part in entry["parts"] if isinstance(part, dict))
                        thinking_content = entry.get("thinking_content") 
                        history_for_filtering.append({
                            "tick": entry["tick"],
                            "role": entry["role"], 
                            "text_content": text_content,
                            "thinking_content": thinking_content
                        })
                    else:
                        print(f"Warning ({self.agent_name}): Malformed history entry in {self.history_file_path}, skipping: {entry}")
            except Exception as e:
                print(f"Error loading raw chat history from {self.history_file_path} for {self.agent_name}: {e}.")
        return history_for_filtering

    def _append_turn_to_history_file(self, tick: int, role: str, text: str, thinking_text: Optional[str] = None, token_info: Optional[Dict[str, Optional[int]]] = None) -> None:
        if not text and not thinking_text: # Don't save if both are empty
            return
        try:
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
            raise ConnectionError(f"genai.Client not initialized for {self.agent_name}.")

        raw_history_with_ticks = self._load_history_from_file()
        # self.agent_pruned_ticks_info is used by _filter_and_prune_history
        processed_history_entries = self._filter_and_prune_history(raw_history_with_ticks)

        sdk_history_for_init: List[google_genai_types.ContentDict] = []
        for entry in processed_history_entries:
            sdk_role = entry['role']
            if sdk_role not in ['user', 'model']:
                print(f"Warning ({self.agent_name}): Invalid role '{sdk_role}' in processed history, defaulting to 'user'. Entry: {entry}")
                sdk_role = 'user'

            sdk_history_for_init.append(google_genai_types.ContentDict({
                'role': sdk_role, 
                'parts': [{'text': entry['text_content']}]
            }))
        
        try:
            # The system_instruction is part of self.generation_config which is used in send_message.
            # For chats.create, only the turn history is typically passed.
            self.chat_session = self.client.chats.create(
                model=self.model_name,
                history=sdk_history_for_init if sdk_history_for_init else None,
            )
            print(f"Info ({self.agent_name}): Gemini ChatSession initialized/re-initialized. History length for SDK: {len(sdk_history_for_init)}")
        except Exception as e:
            print(f"Error ({self.agent_name}): starting/loading chat session with model '{self.model_name}': {e}")
            self.chat_session = None
            raise
        
    def _send_message_implementation(self, user_prompt: str, current_tick: int, attempt_number: int = 0) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        token_info: Dict[str, Optional[int]] = {
            'total_tokens_in_session': None,
            'last_exchange_prompt_tokens': None,
            'last_exchange_completion_tokens': None,
            'last_exchange_cached_tokens': None,
            'last_exchange_thoughts_tokens': None
        }
        thinking_text_parts: List[str] = [] # Initialize list for thinking parts
        llm_text_response_parts: List[str] = [] # Initialize list for response parts

        if not self.chat_session: # Should have been initialized or re-initialized by send_message
             err_msg = f"SYSTEM_ERROR: Chat session for {self.agent_name} is not available in _send_message_implementation."
             print(f"Error ({self.agent_name}): {err_msg}")
             return err_msg, None, token_info
        
        try:
            # Use one-off generation for first attempt, streaming for retries
            if attempt_number == 0:
                # First attempt: use regular send_message (one-off generation)
                api_response = self.chat_session.send_message(
                    user_prompt, 
                    config=self.generation_config 
                )

                if not api_response.candidates:
                    block_reason_detail = "Unknown (no candidates)"
                    pb_feedback = getattr(api_response, 'prompt_feedback', None)
                    if pb_feedback and pb_feedback.block_reason:
                        block_reason_detail = f"Reason: {pb_feedback.block_reason.name}."
                    raise LLMSafetyBlockError(
                        f"LLM response generation failed for {self.agent_name}. {block_reason_detail}",
                        block_reason=pb_feedback.block_reason.name if pb_feedback and pb_feedback.block_reason else None,
                        prompt_feedback=pb_feedback
                    )

                candidate = api_response.candidates[0]

                # --- MODIFICATION START: Segregate text based on part.thought ---
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text: # Process only if there's text
                            if hasattr(part, 'thought') and part.thought: # Check if 'thought' attribute is present and truthy
                                thinking_text_parts.append(part.text)
                            else:
                                llm_text_response_parts.append(part.text)
                
                llm_text_response = "".join(llm_text_response_parts)
                thinking_text = "\n".join(thinking_text_parts) if thinking_text_parts else None
                # --- MODIFICATION END ---
                
                # Get usage metadata
                final_usage_metadata = api_response.usage_metadata if hasattr(api_response, 'usage_metadata') else None
                
            else:
                # Retry attempts: use streaming to avoid timeout issues
                print(f"Info ({self.agent_name}): Using streaming for retry attempt {attempt_number} to avoid timeout issues")
                
                # For tracking usage metadata across chunks
                final_usage_metadata = None
                
                stream_response = self.chat_session.send_message_stream(
                    user_prompt, 
                    config=self.generation_config 
                )
                
                # Variable to track if we got any candidates
                got_candidates = False
                prompt_feedback = None
                
                # Collect all chunks
                for chunk in stream_response:
                    # Save prompt feedback from first chunk if available
                    if not prompt_feedback and hasattr(chunk, 'prompt_feedback'):
                        prompt_feedback = chunk.prompt_feedback
                    
                    # Check if chunk has candidates
                    if chunk.candidates:
                        got_candidates = True
                        candidate = chunk.candidates[0]
                        
                        # Process content parts in the chunk
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text: # Process only if there's text
                                    if hasattr(part, 'thought') and part.thought: # Check if 'thought' attribute is present and truthy
                                        thinking_text_parts.append(part.text)
                                    else:
                                        llm_text_response_parts.append(part.text)
                    
                    # Save usage metadata from the latest chunk (usually last chunk has complete metadata)
                    if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                        final_usage_metadata = chunk.usage_metadata
                
                # Check if we got any valid response
                if not got_candidates:
                    block_reason_detail = "Unknown (no candidates)"
                    if prompt_feedback and prompt_feedback.block_reason:
                        block_reason_detail = f"Reason: {prompt_feedback.block_reason.name}."
                    raise LLMSafetyBlockError(
                        f"LLM response generation failed for {self.agent_name}. {block_reason_detail}",
                        block_reason=prompt_feedback.block_reason.name if prompt_feedback and prompt_feedback.block_reason else None,
                        prompt_feedback=prompt_feedback
                    )
                
                # Combine all collected parts
                llm_text_response = "".join(llm_text_response_parts)
                thinking_text = "\n".join(thinking_text_parts) if thinking_text_parts else None
            
            self._append_turn_to_history_file(current_tick, 'user', user_prompt, None, None) 
            self._append_turn_to_history_file(current_tick, 'model', llm_text_response, thinking_text, token_info)

            # Process usage metadata if available
            if final_usage_metadata:
                token_info['last_exchange_prompt_tokens'] = getattr(final_usage_metadata, 'prompt_token_count', None)
                token_info['last_exchange_completion_tokens'] = getattr(final_usage_metadata, 'candidates_token_count', None)
                token_info['last_exchange_cached_tokens'] = getattr(final_usage_metadata, 'cached_content_token_count', None)
                token_info['last_exchange_thoughts_tokens'] = getattr(final_usage_metadata, 'thoughts_token_count', None)
                token_info['total_tokens_in_session'] = getattr(final_usage_metadata, 'total_token_count', None)
            
            if token_info['total_tokens_in_session'] is None and self.client and self.chat_session:
                try:
                    print(f"Warning ({self.agent_name}): total_token_count not in usage_metadata. Recounting session tokens manually.")
                    current_sdk_history = self.chat_session.get_history()
                    count_response = self.client.models.count_tokens(
                        model=self.model_name,
                        contents=current_sdk_history
                    )
                    token_info['total_tokens_in_session'] = count_response.total_tokens
                except Exception as count_e:
                    print(f"Warning ({self.agent_name}): Could not count total session tokens after send_message: {count_e}")
            
            return llm_text_response, thinking_text, token_info

        except google_genai_errors.ServerError as e:
            # Log detailed error information for debugging
            error_details = f"message='{getattr(e, 'message', str(e))}'"
            if hasattr(e, 'status_code'):
                error_details += f", status_code={e.status_code}"
            if hasattr(e, 'details'):
                error_details += f", details='{e.details}'"
            if hasattr(e, '__dict__'):
                extra_attrs = {k: v for k, v in e.__dict__.items() if k not in ['message', 'status_code', 'details'] and not k.startswith('_')}
                if extra_attrs:
                    error_details += f", extra_attrs={extra_attrs}"
            
            print(f"DEBUG - Raw Gemini ServerError for {self.agent_name}: {error_details}")
            raise LLMTransientAPIError(f"Gemini API Server Error for {self.agent_name}: {getattr(e, 'message', str(e))}", original_exception=e)
        except LLMSafetyBlockError:
            raise
        except google_genai_errors.ClientError as e:
            # Check for specific context overflow error pattern
            if (hasattr(e, 'status') and e.status == 'INVALID_ARGUMENT' and 
                hasattr(e, 'message') and 'input token count exceeds the maximum number of tokens allowed' in str(e.message)):
                print(f"CRITICAL ({self.agent_name}): Context window overflow detected in Gemini API")
                raise LLMContextOverflowError(f"Context window overflow for {self.agent_name}: {str(e)}", original_exception=e)
            
            # Handle other client errors normally (rate limits, auth, etc.)
            if hasattr(e, 'status') and e.status == 'RESOURCE_EXHAUSTED':
                raise LLMTransientAPIError(f"Gemini API quota/rate limit error for {self.agent_name}: {getattr(e, 'message', str(e))}", original_exception=e)
            else:
                raise LLMPermanentAPIError(f"Gemini API client error for {self.agent_name}: {getattr(e, 'message', str(e))}", original_exception=e)
        except Exception as e:
            # Log detailed error information for debugging
            print(f"DEBUG - Raw Exception for {self.agent_name}: type={type(e).__name__}, str={str(e)}")
            if hasattr(e, '__dict__'):
                error_attrs = {k: v for k, v in e.__dict__.items() if not k.startswith('_')}
                if error_attrs:
                    print(f"DEBUG - Exception Attributes: {error_attrs}")
            
            import traceback; traceback.print_exc()
            raise LLMConnectorError(f"Unexpected LLM API call failure for {self.agent_name}. Details: {str(e)}", original_exception=e)

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Returns the current (pruned) chat history from the active session."""
        if not self.chat_session:
            print(f"Warning ({self.agent_name}): get_chat_history called but no active chat session. Attempting to reconstruct from file (may be slow or incomplete if init failed).")
            raw_history_with_ticks = self._load_history_from_file() 
            processed_history_entries = self._filter_and_prune_history(raw_history_with_ticks)
            return [{'role': entry['role'], 
                     'text': entry['text_content'], 
                     'thinking': entry.get('thinking_content')} 
                    for entry in processed_history_entries]

        simple_history: List[Dict[str, str]] = []
        try:
            sdk_chat_history = self.chat_session.get_history() 
            for message_content in sdk_chat_history: 
                role = getattr(message_content, "role", "unknown")
                text = "".join(getattr(part,"text","") for part in getattr(message_content, "parts", []) if hasattr(part, "text"))
                simple_history.append({"role": role, "text": text, "thinking": None})
        except Exception as e:
            print(f"Error ({self.agent_name}): converting SDK history to simple format: {e}")
        return simple_history
    
    def get_current_total_session_tokens(self) -> Optional[int]:
        """Calculates total tokens based on the current, possibly pruned, chat session history."""
        if not self.client: return None 

        history_for_count_sdk_format: List[google_genai_types.ContentDict] = []
        if self.chat_session:
            try:
                history_for_count_sdk_format = self.chat_session.get_history()
            except Exception as e:
                print(f"Error ({self.agent_name}): getting history from active session for token count: {e}. Will attempt to load, prune, and convert from file.")
                raw_history_with_ticks = self._load_history_from_file()
                processed_history_entries = self._filter_and_prune_history(raw_history_with_ticks)
                for entry in processed_history_entries:
                    sdk_role = entry['role']
                    if sdk_role not in ['user', 'model']: sdk_role = 'user'
                    history_for_count_sdk_format.append(google_genai_types.ContentDict({
                        'role': sdk_role,
                        'parts': [{'text': entry['text_content']}]
                    }))
        else: 
            print(f"Warning ({self.agent_name}): No active chat session for get_current_total_session_tokens. Loading/pruning from file.")
            raw_history_with_ticks = self._load_history_from_file()
            processed_history_entries = self._filter_and_prune_history(raw_history_with_ticks)
            for entry in processed_history_entries:
                sdk_role = entry['role']
                if sdk_role not in ['user', 'model']: sdk_role = 'user'
                history_for_count_sdk_format.append(google_genai_types.ContentDict({
                    'role': sdk_role,
                    'parts': [{'text': entry['text_content']}]
                }))
            
        if not history_for_count_sdk_format: return 0

        try:
            count_response = self.client.models.count_tokens(
                model=self.model_name,
                contents=history_for_count_sdk_format
            )
            return count_response.total_tokens
        except Exception as e:
            print(f"Warning ({self.agent_name}): Could not count total session tokens in get_current_total_session_tokens: {e}")
            return None