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
import json
import re
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
                 max_output_tokens: Optional[int] = None,
                 max_retries: int = constants.LLM_MAX_RETRIES,
                 retry_delay_seconds: int = constants.LLM_RETRY_DELAY_SECONDS,
                 custom_api_params: Optional[Dict[str, Any]] = None):

        self.custom_api_params = custom_api_params or {}
        self.active_endpoint_name: Optional[str] = None
        self.base_url = str(
            self.custom_api_params.get("base_url")
            or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        ).rstrip("/")
        super().__init__(model_name, agent_name, agent_data_path, 
                         api_key, system_prompt, temperature, max_output_tokens,
                         max_retries, retry_delay_seconds)

        self._configure_client()

        # Unified beta headers for all API calls
        self.api_headers = self._build_api_headers()

        # self.history_messages will store history in Anthropic's format AFTER pruning
        self.history_messages: List[Dict[str, Any]] = []
        self.last_known_total_session_tokens: Optional[int] = None
        self._initialize_chat_session()

        print(f"ClaudeConnector for '{self.agent_name}' initialized with model: '{self.model_name}', temp: {self.temperature}, max_tokens: {self.max_output_tokens}.")

    def _build_api_headers(self) -> Dict[str, str]:
        if self._should_skip_token_counting():
            return {}
        return {
            "anthropic-beta": "extended-cache-ttl-2025-04-11"
        }

    def _build_cache_control(self) -> Dict[str, str]:
        if self._should_skip_token_counting():
            return {"type": "ephemeral", "ttl": "5m"}
        return {"type": "ephemeral", "ttl": "1h"}

    def _configure_client(self) -> None:
        effective_api_key = (
            self.api_key
            or self.custom_api_params.get("api_key")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        if not effective_api_key:
            raise ValueError(f"Anthropic API key not provided for {self.agent_name} and ANTHROPIC_API_KEY env variable not set.")
        self.api_key = effective_api_key
        try:
            self.client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            raise LLMPermanentAPIError(f"Failed to initialize Anthropic client for {self.agent_name}: {e}", original_exception=e)
        self._skip_token_counting = self._should_skip_token_counting()
        self.api_headers = self._build_api_headers()
        print(f"Info ({self.agent_name}): Claude client configured (base_url={self.base_url}).")

    def _should_skip_token_counting(self) -> bool:
        official_base_url = "https://api.anthropic.com"
        return self.base_url.rstrip("/") != official_base_url

    def _get_claude_model_version(self) -> Optional[Tuple[int, int]]:
        """Extract Claude major/minor version from model names such as claude-opus-4-6 or claude-3-5-haiku-20241022."""
        parts = self.model_name.split("-")
        if not parts or parts[0] != "claude":
            return None

        numeric_parts = [part for part in parts[1:] if part.isdigit()]
        if not numeric_parts:
            return None

        major = int(numeric_parts[0])
        minor = 0

        if len(numeric_parts) >= 2 and len(numeric_parts[1]) <= 2:
            minor = int(numeric_parts[1])

        return (major, minor)

    def _get_default_max_output_tokens(self) -> int:
        version = self._get_claude_model_version()
        model_name_lower = self.model_name.lower()
        if version == (4, 6) and "opus" in model_name_lower:
            return 128000
        return 64000

    def _build_manual_thinking_config(self, effective_max_tokens: Optional[int]) -> Optional[Dict[str, Any]]:
        if effective_max_tokens is None:
            return None
        thinking_budget = min(10000, int(effective_max_tokens * 0.5))
        if thinking_budget < 1024:
            print(f"Warning ({self.agent_name}): Effective max tokens ({effective_max_tokens}) too low for thinking mode. Disabling thinking for this request.")
            return None
        return {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }

    def _build_preferred_thinking_config(self, effective_max_tokens: Optional[int]) -> Optional[Dict[str, Any]]:
        version = self._get_claude_model_version()
        if version is not None and version >= (4, 6):
            return {"type": "adaptive"}
        return self._build_manual_thinking_config(effective_max_tokens)

    def _build_output_config(self) -> Optional[Dict[str, Any]]:
        version = self._get_claude_model_version()
        if version == (4, 6) and "opus" in self.model_name:
            return {"effort": "max"}
        if version is not None and version >= (4, 5):
            return {"effort": "high"}
        return None

    def _is_adaptive_thinking_unsupported_error(self, error: Exception) -> bool:
        error_text = str(error).lower()
        indicators = [
            "unexpected keyword argument",
            "unknown parameter",
            "extra inputs are not permitted",
            "invalid thinking type",
            "adaptive",
            "effort",
            "unsupported",
            "not enabled for this channel",
        ]
        return any(indicator in error_text for indicator in indicators)

    def _is_output_config_unsupported_error(self, error: Exception) -> bool:
        error_text = str(error).lower()
        return "output_config" in error_text and any(indicator in error_text for indicator in [
            "unexpected keyword argument",
            "unknown parameter",
            "extra inputs are not permitted",
            "unsupported",
        ])

    def _stream_with_output_config_fallback(self, stream_kwargs: Dict[str, Any]):
        try:
            return self.client.messages.stream(**stream_kwargs)
        except Exception as stream_error:
            if "output_config" in stream_kwargs and self._is_output_config_unsupported_error(stream_error):
                print(f"Info ({self.agent_name}): output_config rejected; retrying Claude request without output_config.")
                fallback_kwargs = dict(stream_kwargs)
                fallback_kwargs.pop("output_config", None)
                return self.client.messages.stream(**fallback_kwargs)
            raise

    def _usage_to_jsonable(self, usage: Any) -> Dict[str, Any]:
        fields = [
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "server_tool_use",
        ]
        out: Dict[str, Any] = {}
        for field in fields:
            if hasattr(usage, field):
                out[field] = getattr(usage, field)
        return out

    def _to_jsonable(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        if hasattr(value, "model_dump"):
            try:
                return self._to_jsonable(value.model_dump())
            except Exception:
                pass
        if hasattr(value, "to_dict"):
            try:
                return self._to_jsonable(value.to_dict())
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                return self._to_jsonable(vars(value))
            except Exception:
                pass
        return repr(value)

    def _summarize_final_message_blocks(self, final_message_snapshot: Any) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        if not final_message_snapshot or not getattr(final_message_snapshot, "content", None):
            return summaries

        for block in final_message_snapshot.content:
            block_type = getattr(block, "type", None)
            summary: Dict[str, Any] = {"type": block_type}
            if block_type == "thinking":
                thinking_text = getattr(block, "thinking", "") or ""
                summary["thinking_len"] = len(thinking_text)
            elif block_type == "text":
                text = getattr(block, "text", "") or ""
                summary["text_len"] = len(text)
            elif block_type == "tool_use":
                summary["name"] = getattr(block, "name", None)
            summaries.append(summary)

        return summaries

    def _dump_send_payload_snapshot(
        self,
        user_prompt: str,
        current_tick: int,
        attempt_number: int,
        stream_kwargs: Dict[str, Any],
    ) -> None:
        if not self._debug_api_enabled():
            return
        safe_agent_name = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in self.agent_name)
        ts_ms = int(time.time() * 1000)
        filename = f"claude_send_{safe_agent_name}_tick{current_tick}_attempt{attempt_number}_{ts_ms}.json"
        snapshot = {
            "agent_name": self.agent_name,
            "tick": current_tick,
            "attempt_number": attempt_number,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "history_messages": self._to_jsonable(self.history_messages),
            "stream_kwargs": self._to_jsonable(stream_kwargs),
        }
        self._write_debug_api_snapshot(filename, snapshot)

    def _dump_response_snapshot(self, current_tick: int, attempt_number: int, payload: Any) -> None:
        if not self._debug_api_enabled():
            return
        safe_agent_name = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in self.agent_name)
        ts_ms = int(time.time() * 1000)
        filename = f"claude_response_{safe_agent_name}_tick{current_tick}_attempt{attempt_number}_{ts_ms}.json"
        snapshot = {
            "agent_name": self.agent_name,
            "tick": current_tick,
            "attempt_number": attempt_number,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "response": self._to_jsonable(payload),
        }
        self._write_debug_api_snapshot(filename, snapshot)

    def _rough_token_estimate_for_text(self, text: str) -> int:
        text = str(text or "")
        if not text:
            return 0
        word_estimate = len(text.split())
        char_estimate = max(1, len(text) // 4)
        return max(word_estimate, char_estimate)

    def _estimate_tokens_without_api_counting(self, user_prompt: str) -> int:
        history_estimate = 0
        for msg in getattr(self, "history_messages", []) or []:
            history_estimate += self._rough_token_estimate_for_text(msg.get("content", ""))

        system_estimate = self._rough_token_estimate_for_text(getattr(self, "system_prompt", ""))
        prompt_estimate = self._rough_token_estimate_for_text(user_prompt)
        return history_estimate + system_estimate + prompt_estimate

    def _estimate_total_session_tokens_from_usage(self, usage: Any) -> Optional[int]:
        if usage is None:
            return None
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        cache_read_input_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        if input_tokens is None or output_tokens is None:
            return None
        return int(input_tokens) + int(cache_read_input_tokens) + int(output_tokens)

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


    def _append_turn_to_history_file(
        self,
        tick: int,
        role: str,
        text: str,
        thinking_text: Optional[str] = None,
        token_info: Optional[Dict[str, Optional[int]]] = None,
        api_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not getattr(self, "persist_to_disk", True):
            return
        if not text.strip() and not (thinking_text and thinking_text.strip()): return # Don't save if both are empty
        try:
            turn_data = {'tick': tick, 'role': role, 'parts': [{'text': text}]}
            if thinking_text:
                turn_data['thinking_content'] = thinking_text
            # Only add token_info for model responses (not user prompts) and if it's provided
            if role == 'model' and token_info:
                turn_data['token_info'] = token_info
            if role == 'model':
                persisted_api_metadata = self._prepare_api_metadata_for_persistence(api_metadata)
                if persisted_api_metadata:
                    turn_data['api_metadata'] = persisted_api_metadata
            file_io_utils.append_yaml_line(turn_data, self.history_file_path)
        except Exception as e:
            print(f"Error appending turn to history file {self.history_file_path} for Claude {self.agent_name}: {e}")

    def _build_claude_api_metadata(
        self,
        final_message_snapshot: Any,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        usage_obj = getattr(final_message_snapshot, "usage", None) if final_message_snapshot is not None else None
        metadata: Dict[str, Any] = {
            "provider": "claude",
            "streaming": True,
            "model_name": self.model_name,
            "endpoint_name": self.active_endpoint_name,
            "base_url": self.base_url,
            "message_id": getattr(final_message_snapshot, "id", None),
            "request_id": getattr(final_message_snapshot, "request_id", None),
            "stop_reason": getattr(final_message_snapshot, "stop_reason", None),
            "stop_sequence": getattr(final_message_snapshot, "stop_sequence", None),
            "usage_raw": self._sanitize_api_return_payload(self._usage_to_jsonable(usage_obj)) if usage_obj is not None else None,
            "raw_return": self._sanitize_api_return_payload(self._to_jsonable(final_message_snapshot)) if final_message_snapshot is not None else None,
        }
        if final_message_snapshot is not None:
            metadata["content_blocks"] = self._summarize_final_message_blocks(final_message_snapshot)
        if extra_metadata:
            metadata.update(extra_metadata)
        return self._prepare_api_metadata_for_persistence(metadata)

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
        final_message_snapshot: Any = None

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
            "content": [{"type": "text", "text": user_prompt, "cache_control": self._build_cache_control()}]
        })

        # --- ADDED BACK: Token counting and effective_max_tokens adjustment ---
        effective_max_tokens = int(self.max_output_tokens) if self.max_output_tokens is not None and self.max_output_tokens > 0 else self._get_default_max_output_tokens()

        # Calculate current history tokens before adding the new user_prompt for this specific calculation
        # self.history_messages is already pruned and in Claude's format
        current_history_tokens_for_calc = 0
        if not self._skip_token_counting and self.history_messages: # Only count if there's history
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
        if self._skip_token_counting:
            estimated_input_tokens_for_call = self._estimate_tokens_without_api_counting(user_prompt)
        else:
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

        if effective_max_tokens is not None and estimated_input_tokens_for_call + effective_max_tokens > MODEL_CONTEXT_WINDOW_LIMIT:
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
                system_messages = [{"type": "text", "text": self.system_prompt, "cache_control": self._build_cache_control()}]

            thinking_config = self._build_preferred_thinking_config(effective_max_tokens)
            manual_thinking_config = self._build_manual_thinking_config(effective_max_tokens)
            output_config = self._build_output_config()

            stream_kwargs = {
                "model": self.model_name,
                "temperature": self.temperature,
                "system": system_messages,
                "messages": api_messages_payload,
                "extra_headers": self.api_headers
            }
            if effective_max_tokens is not None:
                stream_kwargs["max_tokens"] = effective_max_tokens

            if thinking_config:
                stream_kwargs["thinking"] = thinking_config
            if output_config:
                stream_kwargs["output_config"] = output_config
            self._dump_send_payload_snapshot(user_prompt, current_tick, attempt_number, stream_kwargs)

            try:
                stream_context = self._stream_with_output_config_fallback(stream_kwargs)
            except Exception as stream_error:
                if (
                    thinking_config
                    and thinking_config.get("type") == "adaptive"
                    and manual_thinking_config
                    and self._is_adaptive_thinking_unsupported_error(stream_error)
                ):
                    print(f"Info ({self.agent_name}): Adaptive thinking rejected; retrying with manual thinking mode.")
                    stream_kwargs["thinking"] = manual_thinking_config
                    thinking_config = manual_thinking_config
                    stream_context = self._stream_with_output_config_fallback(stream_kwargs)
                else:
                    raise

            with stream_context as stream:
                llm_text_response_parts: List[str] = []
                for text_delta in stream.text_stream:
                    llm_text_response_parts.append(text_delta)

                llm_text_response = "".join(llm_text_response_parts)

                final_message_snapshot = stream.get_final_message()
                self._dump_response_snapshot(current_tick, attempt_number, final_message_snapshot)
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
                        estimated_total_tokens = self._estimate_total_session_tokens_from_usage(final_message_snapshot.usage)
                        if estimated_total_tokens is not None:
                            token_info['total_tokens_in_session'] = estimated_total_tokens
                            self.last_known_total_session_tokens = estimated_total_tokens
                        if token_info['total_tokens_in_session'] is None:
                            usage_payload = self._usage_to_jsonable(final_message_snapshot.usage)
                            print(f"Info ({self.agent_name}): Claude usage payload: {json.dumps(usage_payload, ensure_ascii=True)}")

                    if not llm_text_response.strip():
                        stop_reason = getattr(final_message_snapshot, "stop_reason", None)
                        content_summary = self._summarize_final_message_blocks(final_message_snapshot)
                        print(
                            f"Warning ({self.agent_name}): Claude returned no text content. "
                            f"stop_reason={stop_reason!r}, content_blocks={json.dumps(content_summary, ensure_ascii=True)}"
                        )

                # Ensure we always have a non-empty response to avoid API errors
                if not llm_text_response.strip():
                    llm_text_response = "[No response generated]"


            self.history_messages.append({"role": "user", "content": user_prompt})
            # Only append assistant response if it's not empty
            if llm_text_response.strip() or (extracted_thinking_text and extracted_thinking_text.strip()):
                self.history_messages.append({"role": "assistant", "content": llm_text_response})

            recounted_total = self.get_current_total_session_tokens()
            if recounted_total is not None:
                token_info['total_tokens_in_session'] = recounted_total

            api_metadata = self._build_claude_api_metadata(
                final_message_snapshot,
                extra_metadata={
                    "thinking_enabled": thinking_config is not None,
                    "output_config": output_config,
                },
            )
            self._append_turn_to_history_file(current_tick, 'user', user_prompt, None, None)
            self._append_turn_to_history_file(current_tick, 'model', llm_text_response, extracted_thinking_text, token_info, api_metadata)

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

            if self._is_provider_instability_bad_request(e):
                error_message = (
                    f"Anthropic API provider instability for {self.agent_name}: "
                    f"{getattr(e, 'message', str(e))}"
                )
                raise LLMTransientAPIError(error_message, original_exception=e)

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

        if self._has_explicit_context_overflow_signal(error_str):
            return True

        # Check error body for Claude BadRequestError
        if hasattr(error, 'body') and error.body and isinstance(error.body, dict):
            if 'error' in error.body and isinstance(error.body['error'], dict):
                error_details = error.body['error']
                message = error_details.get('message', '')
                if self._has_explicit_context_overflow_signal(str(message)):
                    return True

        return False

    def _has_explicit_context_overflow_signal(self, message: str) -> bool:
        """Only treat explicit token-limit or context-limit signals as true overflow."""
        normalized = str(message or "").lower()
        explicit_patterns = [
            r"prompt is too long:\s*\d+\s+tokens\s*>\s*\d+\s+maximum",
            r"context window overflow",
            r"maximum context length",
            r"maximum context window",
            r"exceeds?(?: the)? maximum context",
            r"exceeded(?: the)? maximum context",
            r"input is too long",
            r"context is too long",
            r"context too long",
            r"超出.{0,8}(?:最大|模型)?上下文",
            r"超出.{0,8}(?:最大|模型)?token",
            r"超过.{0,8}(?:最大|模型)?上下文",
            r"超过.{0,8}(?:最大|模型)?token",
            r"token[s]?\s*>\s*\d+\s+maximum",
        ]
        return any(re.search(pattern, normalized) for pattern in explicit_patterns)

    def _is_provider_instability_bad_request(self, error: Exception) -> bool:
        """
        Detect proxy/provider instability where a frontend returns a synthetic 400 with
        generic compact/clear guidance instead of a real request validation failure.
        """
        if not hasattr(error, 'body') or not error.body or not isinstance(error.body, dict):
            return False

        error_details = error.body.get('error')
        if not isinstance(error_details, dict):
            return False

        err_type = str(error_details.get('type', '') or '').strip().lower()
        message = str(error_details.get('message', '') or '')
        normalized = message.lower()

        if self._has_explicit_context_overflow_signal(message):
            return False

        instability_markers = [
            "compact or clear",
            "compact 或者 clear",
            "压缩或清空对话",
            "claude code 客户端",
            "claude code client",
            "参数似乎不正确",
        ]
        has_marker = any(marker.lower() in normalized for marker in instability_markers)
        return err_type in {"<nil>", "nil", ""} and has_marker

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
        if self._skip_token_counting:
            return self.last_known_total_session_tokens
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
                return self.last_known_total_session_tokens
        except Exception as e:
            print(f"Warning ({self.agent_name}): Exception counting total session tokens for Claude: {e}")
            return self.last_known_total_session_tokens
