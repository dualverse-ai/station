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
import time
import hashlib
from typing import Dict, Any, Optional, List, Tuple

import anthropic
from anthropic import Anthropic, APIError, RateLimitError, AuthenticationError, BadRequestError, APIConnectionError, APITimeoutError, InternalServerError

from station import file_io_utils
from station import constants
from station import runtime_api_config
from .base import (
    BaseLLMConnector,
    LLMConnectorError,
    LLMTransientAPIError,
    LLMPermanentAPIError,
    LLMSafetyBlockError,
    LLMContextOverflowError
)


CLAUDE_OPUS_5_REFUSAL_FALLBACK_MODEL = "claude-opus-4-8"


class _ClaudeRawStreamMessage:
    def __init__(
        self,
        *,
        message_id: Optional[str],
        request_id: Optional[str],
        stop_reason: Optional[str],
        stop_sequence: Optional[str],
        usage: Dict[str, Any],
        content: List[Dict[str, Any]],
    ) -> None:
        self.id = message_id
        self.request_id = request_id
        self.stop_reason = stop_reason
        self.stop_sequence = stop_sequence
        self.usage = usage
        self.content = content


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
        self.api_runtime_provider_id = "claude"
        self.api_runtime_env_names = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL")
        self._api_runtime_config_snapshot = runtime_api_config.get_config_snapshot([
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
        ], provider_id="claude")
        snapshot_env = self._api_runtime_config_snapshot.get("env", {})
        self.active_endpoint_name: Optional[str] = None
        self.base_url = (snapshot_env.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        super().__init__(model_name, agent_name, agent_data_path, 
                         api_key, system_prompt, temperature, max_output_tokens,
                         max_retries, retry_delay_seconds)

        self._configure_client_for_active_endpoint()

        # Unified beta headers for all API calls
        self.api_headers = self._build_api_headers()

        # self.history_messages will store history in Anthropic's format AFTER pruning
        self.history_messages: List[Dict[str, Any]] = [] 
        self.last_known_total_session_tokens: Optional[int] = None
        self._initialize_chat_session()

        print(f"ClaudeConnector for '{self.agent_name}' initialized with model: '{self.model_name}', temp: {self.temperature}, max_tokens: {self.max_output_tokens}.")

    def _resolve_cache_ttl(self) -> Optional[str]:
        custom_api_params = getattr(self, "custom_api_params", {}) or {}
        raw_ttl = custom_api_params.get("claude_cache_ttl")
        if raw_ttl is not None:
            ttl = str(raw_ttl).strip().lower()
            if ttl in {"", "none", "off", "false", "disabled"}:
                return None
            return ttl
        if self._should_skip_token_counting():
            return "5m"
        return "1h"

    def _build_api_headers(self) -> Dict[str, str]:
        if self._resolve_cache_ttl() != "1h":
            return {}
        return {
            "anthropic-beta": "extended-cache-ttl-2025-04-11"
        }

    def _build_request_cache_control(self) -> Optional[Dict[str, str]]:
        ttl = self._resolve_cache_ttl()
        if ttl is None:
            return None
        return {"type": "ephemeral", "ttl": ttl}

    def _load_station_id_for_cache(self) -> str:
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
                    return candidate.strip()
        except Exception as e:
            self._log("WARNING", f"Failed to resolve station_id for Claude cache metadata: {e}")
        station_data_path = os.path.abspath(os.path.join(os.getcwd(), constants.BASE_STATION_DATA_PATH))
        return f"path:{station_data_path}"

    def _build_stable_agent_cache_id(self) -> str:
        station_id = self._load_station_id_for_cache()
        raw_key = f"{station_id}:{self.agent_name}"
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]
        return f"station-agent-{digest}"

    def _build_request_metadata(self) -> Optional[Dict[str, str]]:
        if self._resolve_cache_ttl() is None:
            return None
        metadata_id = self._build_stable_agent_cache_id()
        metadata = {"user_id": metadata_id}
        if self._should_skip_token_counting():
            metadata["session_id"] = metadata_id
        return metadata

    def _build_text_block(self, text: str) -> Dict[str, Any]:
        return {"type": "text", "text": text}

    def _resolve_endpoint_settings(self, snapshot: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], str, str]:
        if snapshot is None:
            snapshot = runtime_api_config.get_config_snapshot([
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
            ], provider_id="claude")
        snapshot_env = snapshot.get("env", {})
        effective_api_key = self.api_key if self._explicit_api_key else snapshot_env.get("ANTHROPIC_API_KEY")
        if not effective_api_key:
            raise ValueError(f"Anthropic API key not provided for {self.agent_name} and ANTHROPIC_API_KEY env variable not set.")
        base_url = str(snapshot_env.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        endpoint = snapshot.get("provider_endpoint") if isinstance(snapshot, dict) else None
        endpoint_name = endpoint.get("name") if isinstance(endpoint, dict) else None
        return endpoint_name, effective_api_key, base_url

    def _configure_client_for_active_endpoint(self, snapshot: Optional[Dict[str, Any]] = None) -> None:
        if snapshot is None:
            snapshot = runtime_api_config.get_config_snapshot([
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
            ], provider_id="claude")
        self._apply_runtime_proxy_snapshot(snapshot)
        endpoint_name, resolved_api_key, resolved_base_url = self._resolve_endpoint_settings(snapshot)
        self.api_key = resolved_api_key
        self.base_url = resolved_base_url
        self.active_endpoint_name = endpoint_name
        self._api_runtime_config_snapshot = snapshot
        self.api_runtime_config_generation = int(snapshot.get("generation", 0))
        try:
            self.client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            raise LLMPermanentAPIError(f"Failed to initialize Anthropic client for {self.agent_name}: {e}", original_exception=e)
        self._skip_token_counting = self._should_skip_token_counting()
        self.api_headers = self._build_api_headers()
        endpoint_label = self.active_endpoint_name or "default"
        print(
            f"Info ({self.agent_name}): Claude client configured for endpoint '{endpoint_label}' "
            f"(base_url={self.base_url})."
        )

    def _refresh_runtime_api_config_if_changed(self) -> bool:
        if runtime_api_config.get_generation() == self.api_runtime_config_generation:
            return False
        self._configure_client_for_active_endpoint()
        self._initialize_chat_session()
        return True

    def _apply_provider_runtime_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._configure_client_for_active_endpoint(snapshot)
        self._initialize_chat_session()

    def _run_provider_base_recovery_probe(self, snapshot: Dict[str, Any]) -> bool:
        self._apply_runtime_proxy_snapshot(snapshot)
        _endpoint_name, api_key, base_url = self._resolve_endpoint_settings(snapshot)
        client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        client.messages.create(
            model=self.model_name,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with hi."}],
        )
        return True

    def _should_skip_token_counting(self) -> bool:
        official_base_url = "https://api.anthropic.com"
        return self.base_url.rstrip("/") != official_base_url

    def _can_count_current_session_tokens_authoritatively(self) -> bool:
        return not self._skip_token_counting

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

    def _should_fallback_after_refusal(self, final_message: Any) -> bool:
        model_name = str(self.model_name or "").strip().lower()
        stop_reason = str(self._read_value(final_message, "stop_reason", "") or "").strip().lower()
        return (
            (model_name == "claude-opus-5" or model_name.startswith("claude-opus-5-"))
            and stop_reason == "refusal"
        )

    def _is_output_config_unsupported_error(self, error: Exception) -> bool:
        error_text = str(error).lower()
        return "output_config" in error_text and any(indicator in error_text for indicator in [
            "unexpected keyword argument",
            "unknown parameter",
            "extra inputs are not permitted",
            "unsupported",
        ])

    def _create_raw_stream_with_output_config_fallback(self, request_kwargs: Dict[str, Any]):
        stream_request_kwargs = dict(request_kwargs)
        stream_request_kwargs["stream"] = True
        try:
            return self.client.messages.create(**stream_request_kwargs)
        except Exception as stream_error:
            if "output_config" in stream_request_kwargs and self._is_output_config_unsupported_error(stream_error):
                print(f"Info ({self.agent_name}): output_config rejected; retrying Claude stream request without output_config.")
                fallback_kwargs = dict(stream_request_kwargs)
                fallback_kwargs.pop("output_config", None)
                return self.client.messages.create(**fallback_kwargs)
            raise

    def _read_value(self, value: Any, field: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(field, default)
        return getattr(value, field, default)

    def _merge_usage_dict(self, target: Dict[str, Any], usage: Any) -> None:
        if not usage:
            return
        for key, value in self._usage_to_jsonable(usage).items():
            if value is not None:
                target[key] = value

    def _consume_raw_stream(
        self,
        request_kwargs: Dict[str, Any],
        thinking_enabled: bool,
    ) -> Tuple[str, Optional[str], _ClaudeRawStreamMessage]:
        raw_stream = self._create_raw_stream_with_output_config_fallback(request_kwargs)
        text_parts: List[str] = []
        thinking_parts: List[str] = []
        content_blocks: List[Dict[str, Any]] = []
        usage: Dict[str, Any] = {}
        message_id: Optional[str] = None
        request_id: Optional[str] = None
        stop_reason: Optional[str] = None
        stop_sequence: Optional[str] = None

        for event in raw_stream:
            event_type = self._read_value(event, "type")
            if event_type == "message_start":
                message = self._read_value(event, "message")
                message_id = self._read_value(message, "id")
                request_id = self._read_value(message, "request_id")
                self._merge_usage_dict(usage, self._read_value(message, "usage"))
            elif event_type == "content_block_start":
                index = self._read_value(event, "index", len(content_blocks))
                block = self._to_jsonable(self._read_value(event, "content_block", {}))
                if not isinstance(block, dict):
                    block = {"type": self._read_value(block, "type"), "raw": repr(block)}
                block_type = block.get("type")
                if block_type == "text":
                    block["text"] = block.get("text") or ""
                elif block_type == "thinking":
                    block["thinking"] = block.get("thinking") or ""
                while len(content_blocks) <= int(index):
                    content_blocks.append({"type": "unknown"})
                content_blocks[int(index)] = block
            elif event_type == "content_block_delta":
                index = int(self._read_value(event, "index", 0) or 0)
                delta = self._read_value(event, "delta", {})
                delta_type = self._read_value(delta, "type")
                while len(content_blocks) <= index:
                    content_blocks.append({"type": "unknown"})
                block = content_blocks[index]
                if delta_type == "text_delta":
                    text = self._block_text(delta, "text")
                    text_parts.append(text)
                    block["type"] = "text"
                    block["text"] = str(block.get("text") or "") + text
                elif thinking_enabled and delta_type == "thinking_delta":
                    thinking = self._block_text(delta, "thinking")
                    thinking_parts.append(thinking)
                    block["type"] = "thinking"
                    block["thinking"] = str(block.get("thinking") or "") + thinking
                elif thinking_enabled and delta_type == "signature_delta":
                    block["signature"] = self._block_text(delta, "signature")
            elif event_type == "message_delta":
                delta = self._read_value(event, "delta", {})
                stop_reason = self._read_value(delta, "stop_reason", stop_reason)
                stop_sequence = self._read_value(delta, "stop_sequence", stop_sequence)
                self._merge_usage_dict(usage, self._read_value(event, "usage"))

        llm_text_response = "".join(text_parts)
        thinking_text = "\n".join(part for part in thinking_parts if part) or None
        final_message = _ClaudeRawStreamMessage(
            message_id=message_id,
            request_id=request_id,
            stop_reason=stop_reason,
            stop_sequence=stop_sequence,
            usage=usage,
            content=content_blocks,
        )
        return llm_text_response, thinking_text, final_message

    def _iter_content_blocks(self, message: Any) -> List[Any]:
        content = self._read_value(message, "content")
        if not content:
            return []
        if isinstance(content, list):
            return content
        return [content]

    def _block_text(self, block: Any, field: str) -> str:
        value = self._read_value(block, field, "") or ""
        return value if isinstance(value, str) else str(value)

    def _extract_message_text_and_thinking(
        self,
        message: Any,
        thinking_enabled: bool,
    ) -> Tuple[str, Optional[str]]:
        text_parts: List[str] = []
        thinking_text: Optional[str] = None
        for block in self._iter_content_blocks(message):
            block_type = self._read_value(block, "type")
            if block_type == "text":
                text_parts.append(self._block_text(block, "text"))
            elif thinking_enabled and block_type == "thinking" and thinking_text is None:
                thinking_text = self._block_text(block, "thinking")
        return "".join(text_parts), thinking_text

    def _usage_to_jsonable(self, usage: Any) -> Dict[str, Any]:
        if isinstance(usage, dict):
            return {str(k): self._to_jsonable(v) for k, v in usage.items()}
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
        if not final_message_snapshot or not self._read_value(final_message_snapshot, "content"):
            return summaries

        for block in self._iter_content_blocks(final_message_snapshot):
            block_type = self._read_value(block, "type")
            summary: Dict[str, Any] = {"type": block_type}
            if block_type == "thinking":
                thinking_text = self._block_text(block, "thinking")
                summary["thinking_len"] = len(thinking_text)
            elif block_type == "text":
                text = self._block_text(block, "text")
                summary["text_len"] = len(text)
            elif block_type == "tool_use":
                summary["name"] = self._read_value(block, "name")
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

    def _estimate_total_session_tokens_from_usage(self, usage: Any) -> Optional[int]:
        if usage is None:
            return None
        input_tokens = self._read_value(usage, "input_tokens")
        output_tokens = self._read_value(usage, "output_tokens")
        cache_read_input_tokens = self._read_value(usage, "cache_read_input_tokens", 0) or 0
        cache_creation_input_tokens = self._read_value(usage, "cache_creation_input_tokens", 0) or 0
        if input_tokens is None or output_tokens is None:
            return None
        return (
            int(input_tokens)
            + int(cache_read_input_tokens)
            + int(cache_creation_input_tokens)
            + int(output_tokens)
        )

    def _raise_for_empty_text_response(
        self,
        final_message_snapshot: Any,
        token_info: Dict[str, Optional[int]],
    ) -> None:
        stop_reason = self._read_value(final_message_snapshot, "stop_reason") if final_message_snapshot is not None else None
        content_summary = self._summarize_final_message_blocks(final_message_snapshot)
        total_tokens = token_info.get("total_tokens_in_session")
        detail = (
            f"Claude returned no text content for {self.agent_name}. "
            f"stop_reason={stop_reason!r}, content_blocks={json.dumps(content_summary, ensure_ascii=True)}, "
            f"total_tokens_in_session={total_tokens!r}."
        )
        raise LLMTransientAPIError(
            f"{detail} This may indicate a provider/model empty completion or filtered content."
        )

    def _raise_for_missing_stop_reason(self, final_message_snapshot: Any) -> None:
        stop_reason = (
            self._read_value(final_message_snapshot, "stop_reason")
            if final_message_snapshot is not None
            else None
        )
        if stop_reason is not None and str(stop_reason).strip():
            return

        content_summary = self._summarize_final_message_blocks(final_message_snapshot)
        raise LLMTransientAPIError(
            f"Claude stream ended without a stop_reason for {self.agent_name}. "
            f"content_blocks={json.dumps(content_summary, ensure_ascii=True)}. "
            "The response may be truncated and will be retried."
        )

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
                    history_entry = {
                        "tick": entry["tick"],
                        "role": entry["role"], 
                        "text_content": text_content,
                        "thinking_content": thinking_content
                    }
                    if entry["role"] == "model" and isinstance(entry.get("token_info"), dict):
                        history_entry["token_info"] = entry.get("token_info")
                    history_for_filtering.append(history_entry)
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
        usage_obj = self._read_value(final_message_snapshot, "usage") if final_message_snapshot is not None else None
        metadata: Dict[str, Any] = {
            "provider": "claude",
            "streaming": True,
            "model_name": self.model_name,
            "endpoint_name": self.active_endpoint_name,
            "base_url": self.base_url,
            "message_id": self._read_value(final_message_snapshot, "id"),
            "request_id": self._read_value(final_message_snapshot, "request_id"),
            "stop_reason": self._read_value(final_message_snapshot, "stop_reason"),
            "stop_sequence": self._read_value(final_message_snapshot, "stop_sequence"),
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
        last_active_total_tokens: Optional[int] = None
        for entry in processed_history_entries:
            claude_role = "user" if entry['role'] == "user" else "assistant" 
            # Skip entries with empty text content to avoid API errors
            if entry.get('text_content', '').strip():
                # Thinking blocks are not part of Claude's message history API payload
                claude_ready_history.append({"role": claude_role, "content": entry['text_content']})
                if entry.get("role") == "model" and isinstance(entry.get("token_info"), dict):
                    total_tokens = entry["token_info"].get("total_tokens_in_session")
                    if isinstance(total_tokens, (int, float)):
                        last_active_total_tokens = int(total_tokens)
        
        self.history_messages = claude_ready_history
        self.last_known_total_session_tokens = last_active_total_tokens
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
        
        # Current user prompt is plain; prompt caching is controlled at the request level.
        api_messages_payload.append({
            "role": "user", 
            "content": [self._build_text_block(user_prompt)]
        })
        
        effective_max_tokens = int(self.max_output_tokens) if self.max_output_tokens is not None and self.max_output_tokens > 0 else self._get_default_max_output_tokens()

        try:
            # Convert system prompt to Anthropic's block format.
            system_messages = []
            if self.system_prompt:
                system_messages = [self._build_text_block(self.system_prompt)]
            
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
            request_cache_control = self._build_request_cache_control()
            if request_cache_control is not None:
                stream_kwargs["cache_control"] = request_cache_control
                request_metadata = self._build_request_metadata()
                if request_metadata is not None:
                    stream_kwargs["metadata"] = request_metadata
            if effective_max_tokens is not None:
                stream_kwargs["max_tokens"] = effective_max_tokens
            
            if thinking_config:
                stream_kwargs["thinking"] = thinking_config
            if output_config:
                stream_kwargs["output_config"] = output_config
            self._dump_send_payload_snapshot(user_prompt, current_tick, attempt_number, stream_kwargs)

            try:
                llm_text_response, extracted_thinking_text, final_message_snapshot = self._consume_raw_stream(
                    stream_kwargs,
                    thinking_enabled=thinking_config is not None,
                )
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
                    llm_text_response, extracted_thinking_text, final_message_snapshot = self._consume_raw_stream(
                        stream_kwargs,
                        thinking_enabled=thinking_config is not None,
                    )
                else:
                    raise

            response_model_name = self.model_name
            refusal_fallback_model = None
            if self._should_fallback_after_refusal(final_message_snapshot):
                refusal_fallback_model = CLAUDE_OPUS_5_REFUSAL_FALLBACK_MODEL
                print(
                    f"Warning ({self.agent_name}): Claude Opus 5 returned stop_reason='refusal'; "
                    f"retrying this turn with {refusal_fallback_model}."
                )
                fallback_kwargs = dict(stream_kwargs)
                fallback_kwargs["model"] = refusal_fallback_model
                llm_text_response, extracted_thinking_text, final_message_snapshot = self._consume_raw_stream(
                    fallback_kwargs,
                    thinking_enabled=thinking_config is not None,
                )
                response_model_name = refusal_fallback_model

            self._dump_response_snapshot(current_tick, attempt_number, final_message_snapshot)
            if final_message_snapshot:
                if thinking_config and extracted_thinking_text is None:
                    _response_text, extracted_thinking_text = self._extract_message_text_and_thinking(
                        final_message_snapshot,
                        thinking_enabled=True,
                    )

                usage_obj = self._read_value(final_message_snapshot, "usage")
                if usage_obj:
                    token_info['last_exchange_prompt_tokens'] = self._read_value(usage_obj, "input_tokens")
                    token_info['last_exchange_completion_tokens'] = self._read_value(usage_obj, "output_tokens")
                    token_info['last_exchange_cached_tokens'] = self._read_value(usage_obj, "cache_read_input_tokens") # type: ignore
                    token_info['cache_creation_input_tokens'] = self._read_value(usage_obj, "cache_creation_input_tokens") # type: ignore
                    estimated_total_tokens = self._estimate_total_session_tokens_from_usage(usage_obj)
                    if estimated_total_tokens is not None:
                        token_info['total_tokens_in_session'] = estimated_total_tokens
                        self.last_known_total_session_tokens = estimated_total_tokens
                    if token_info['total_tokens_in_session'] is None:
                        usage_payload = self._usage_to_jsonable(usage_obj)
                        print(f"Info ({self.agent_name}): Claude usage payload: {json.dumps(usage_payload, ensure_ascii=True)}")

                if not llm_text_response.strip():
                    stop_reason = self._read_value(final_message_snapshot, "stop_reason")
                    content_summary = self._summarize_final_message_blocks(final_message_snapshot)
                    print(
                        f"Warning ({self.agent_name}): Claude returned no text content. "
                        f"stop_reason={stop_reason!r}, content_blocks={json.dumps(content_summary, ensure_ascii=True)}"
                    )

            self._raise_for_missing_stop_reason(final_message_snapshot)

            if not llm_text_response.strip():
                self._raise_for_empty_text_response(final_message_snapshot, token_info)


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
                    "model_name": response_model_name,
                    "thinking_enabled": thinking_config is not None,
                    "output_config": output_config,
                },
            )
            if refusal_fallback_model is not None and api_metadata is not None:
                api_metadata["configured_model_name"] = self.model_name
                api_metadata["refusal_fallback_model"] = refusal_fallback_model
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
        except LLMConnectorError:
            raise
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
