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
import time
import base64
import enum
import re
import yaml
from typing import Dict, Any, Optional, List, Tuple

# Use the import style from the provided Google examples
from google import genai
from google.genai import types as google_genai_types
from google.genai import errors as google_genai_errors

from station import file_io_utils
from station import constants
from station import runtime_api_config
from .base import (
    BaseLLMConnector,
    LLMConnectorError,
    LLMCorruptedThoughtSignatureError,
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
                 retry_delay_seconds: int = constants.LLM_RETRY_DELAY_SECONDS,
                 custom_api_params: Optional[Dict[str, Any]] = None):
        
        # Initialize attributes needed by BaseLLMConnector before super().__init__
        # if _initialize_chat_session in super() needs them.
        # In this revised plan, _initialize_chat_session is called at the end of this __init__.
        self.client: Optional[genai.Client] = None
        self.chat_session: Optional[genai.Chat] = None
        self.generation_config: Optional[google_genai_types.GenerateContentConfig] = None
        self.safety_settings: List[google_genai_types.SafetySetting] = []
        self.custom_api_params = custom_api_params or {}
        self.api_runtime_provider_id = "gemini"
        self.api_runtime_env_names = ("GOOGLE_API_KEY", "GOOGLE_GEMINI_BASE_URL")
        self._api_runtime_config_snapshot = runtime_api_config.get_config_snapshot([
            "GOOGLE_API_KEY",
            "GOOGLE_GEMINI_BASE_URL",
        ], provider_id="gemini")
        self.active_endpoint_name: Optional[str] = None
        self.active_base_url: Optional[str] = None

        super().__init__(model_name, agent_name, agent_data_path,
                         api_key, system_prompt, temperature, max_output_tokens,
                         max_retries, retry_delay_seconds)
        
        for cat_name in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]:
            if hasattr(google_genai_types.HarmCategory, cat_name):
                self.safety_settings.append(google_genai_types.SafetySetting(
                    category=getattr(google_genai_types.HarmCategory, cat_name),
                    threshold=google_genai_types.HarmBlockThreshold.BLOCK_NONE
                ))
        
        self.generation_config = self._build_generation_config()
        self._configure_client_for_active_endpoint()
        
        self._initialize_chat_session()

        print(f"GoogleGeminiConnector for '{self.agent_name}' initialized with model: '{self.model_name}', temp: {self.temperature}.")

    def _normalize_base_url(self, base_url: Optional[str]) -> Optional[str]:
        if not base_url:
            return None
        return str(base_url).rstrip("/")

    def _is_official_base_url(self, base_url: Optional[str]) -> bool:
        if not base_url:
            return True
        normalized = str(base_url).rstrip("/")
        return normalized.startswith("https://generativelanguage.googleapis.com")

    def _resolve_endpoint_settings(self, snapshot: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], str, Optional[str]]:
        if snapshot is None:
            snapshot = runtime_api_config.get_config_snapshot([
                "GOOGLE_API_KEY",
                "GOOGLE_GEMINI_BASE_URL",
            ], provider_id="gemini")
        snapshot_env = snapshot.get("env", {})
        effective_api_key = self.api_key if self._explicit_api_key else snapshot_env.get("GOOGLE_API_KEY")
        if not effective_api_key:
            raise ValueError(f"Google API key not provided for {self.agent_name} and GOOGLE_API_KEY env variable not set.")
        base_url = self._normalize_base_url(snapshot_env.get("GOOGLE_GEMINI_BASE_URL"))
        endpoint = snapshot.get("provider_endpoint") if isinstance(snapshot, dict) else None
        endpoint_name = endpoint.get("name") if isinstance(endpoint, dict) else None
        return endpoint_name, effective_api_key, base_url

    def _build_http_options(self, base_url: Optional[str]) -> Optional[google_genai_types.HttpOptions]:
        http_options_kwargs: Dict[str, Any] = {}
        if base_url:
            http_options_kwargs["baseUrl"] = base_url
        elif os.environ.get("GOOGLE_GEMINI_BASE_URL"):
            # The Google SDK treats GOOGLE_GEMINI_BASE_URL as its own implicit
            # default. A blank Station backup endpoint must bypass that env
            # override and use the official Gemini API instead.
            http_options_kwargs["baseUrl"] = runtime_api_config.PROVIDER_SPECS["gemini"].default_base_url
        if constants.GEMINI_TIMEOUT is not None:
            timeout_ms = int(constants.GEMINI_TIMEOUT * 1000)
            http_options_kwargs["timeout"] = timeout_ms
        return google_genai_types.HttpOptions(**http_options_kwargs) if http_options_kwargs else None

    def _create_genai_client(
        self,
        api_key: str,
        http_options: Optional[google_genai_types.HttpOptions],
    ) -> genai.Client:
        """
        Create the Gemini SDK client without the SDK's duplicate env-var warning.

        Station resolves and passes one explicit key. If both GOOGLE_API_KEY and
        GEMINI_API_KEY are present in the process environment, the SDK prints a
        warning during client construction even though Station is not relying on
        the ambiguous env lookup.
        """
        hidden_gemini_key = None
        if os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
            hidden_gemini_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            return genai.Client(api_key=api_key, http_options=http_options)
        finally:
            if hidden_gemini_key is not None:
                os.environ["GEMINI_API_KEY"] = hidden_gemini_key

    def _configure_client_for_active_endpoint(self, snapshot: Optional[Dict[str, Any]] = None) -> None:
        if snapshot is None:
            snapshot = runtime_api_config.get_config_snapshot([
                "GOOGLE_API_KEY",
                "GOOGLE_GEMINI_BASE_URL",
            ], provider_id="gemini")
        self._apply_runtime_proxy_snapshot(snapshot)
        endpoint_name, resolved_api_key, resolved_base_url = self._resolve_endpoint_settings(snapshot)
        self.api_key = resolved_api_key
        self.active_endpoint_name = endpoint_name
        self.active_base_url = resolved_base_url
        self._api_runtime_config_snapshot = snapshot
        self.api_runtime_config_generation = int(snapshot.get("generation", 0))

        try:
            http_options = self._build_http_options(self.active_base_url)
            self.client = self._create_genai_client(self.api_key, http_options)
            endpoint_label = self.active_endpoint_name or "default"
            timeout_suffix = (
                f", timeout={constants.GEMINI_TIMEOUT}s"
                if constants.GEMINI_TIMEOUT is not None
                else ""
            )
            self._log(
                "INFO",
                f"Gemini client endpoint={endpoint_label} "
                f"base_url={self.active_base_url or 'official_default'}{timeout_suffix}.",
            )
        except Exception as e:
            raise LLMPermanentAPIError(f"Error creating genai.Client for {self.agent_name}: {e}.", original_exception=e)

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
        client = self._create_genai_client(api_key, self._build_http_options(base_url))
        client.models.generate_content(model=self.model_name, contents="Reply with hi.")
        return True

    def _can_count_current_session_tokens_authoritatively(self) -> bool:
        return self._is_official_base_url(self.active_base_url)

    def _to_jsonable(self, value: Any) -> Any:
        """Best-effort conversion for SDK objects into JSON-serializable data."""
        if value is None:
            return None
        if isinstance(value, enum.Enum):
            return value.value
        if isinstance(value, (bytes, bytearray)):
            return base64.b64encode(bytes(value)).decode("ascii")
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]

        # Prefer modern model serialization methods if available.
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

    def _format_gemini_error_details(self, error: Exception) -> str:
        parts = [
            f"type={type(error).__name__}",
            f"repr={repr(error)}",
            f"str={str(error)!r}",
        ]
        for attr_name in ("code", "status", "message", "details", "status_code"):
            if hasattr(error, attr_name):
                try:
                    parts.append(f"{attr_name}={getattr(error, attr_name)!r}")
                except Exception as attr_error:
                    parts.append(f"{attr_name}=<unreadable: {attr_error!r}>")

        response = getattr(error, "response", None)
        if response is not None:
            response_parts = []
            for attr_name in ("status_code", "reason_phrase", "text"):
                if hasattr(response, attr_name):
                    try:
                        response_parts.append(f"{attr_name}={getattr(response, attr_name)!r}")
                    except Exception as attr_error:
                        response_parts.append(f"{attr_name}=<unreadable: {attr_error!r}>")
            if response_parts:
                parts.append("response={" + ", ".join(response_parts) + "}")

        return ", ".join(parts)

    def _format_original_exception_for_log(self, error: Exception) -> str:
        """Compact Gemini SDK error fields for routine retry logs."""
        fields = [f"type={type(error).__name__}"]
        for attr_name in ("code", "status", "status_code"):
            if hasattr(error, attr_name):
                try:
                    value = getattr(error, attr_name)
                except Exception as attr_error:
                    fields.append(f"{attr_name}=<unreadable: {attr_error!r}>")
                    continue
                if value is not None:
                    fields.append(f"{attr_name}={value!r}")

        raw_message = getattr(error, "message", None) or str(error)
        if raw_message:
            fields.append(f"message={self._compact_log_value(raw_message, 700)!r}")
            request_id_matches = re.findall(r"(?:request id|cch_session_id):\s*([A-Za-z0-9_.:-]+)", str(raw_message))
            if request_id_matches:
                fields.append(f"request_id={request_id_matches[-1]!r}")

        details = getattr(error, "details", None)
        if isinstance(details, dict):
            error_body = details.get("error")
            if isinstance(error_body, dict):
                for source_key, output_key in (
                    ("type", "provider_type"),
                    ("code", "provider_code"),
                    ("param", "provider_param"),
                ):
                    value = error_body.get(source_key)
                    if value:
                        fields.append(f"{output_key}={value!r}")

        response = getattr(error, "response", None)
        if response is not None:
            response_fields = []
            for attr_name in ("status_code", "reason_phrase"):
                if hasattr(response, attr_name):
                    try:
                        value = getattr(response, attr_name)
                    except Exception as attr_error:
                        response_fields.append(f"{attr_name}=<unreadable: {attr_error!r}>")
                        continue
                    if value is not None:
                        response_fields.append(f"{attr_name}={value!r}")
            if response_fields:
                fields.append("response={" + ", ".join(response_fields) + "}")

        return ", ".join(fields)

    def _is_corrupted_thought_signature_error(self, error_details: str) -> bool:
        normalized = str(error_details).lower().replace("_", " ").replace("-", " ")
        return "corrupted thought signature" in normalized

    def _dump_send_payload_snapshot(self, user_prompt: str, current_tick: int, attempt_number: int, mode: str) -> None:
        """
        Persist outbound Gemini send payload snapshot for debugging.
        One file per send attempt in tmp/debug_api.
        """
        if not self._debug_api_enabled():
            return
        try:
            safe_agent_name = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in self.agent_name)
            ts_ms = int(time.time() * 1000)
            filename = f"gemini_send_{safe_agent_name}_tick{current_tick}_attempt{attempt_number}_{mode}_{ts_ms}.json"

            history_dump: Any = None
            history_error: Optional[str] = None
            try:
                history_dump = self.chat_session.get_history() if self.chat_session else None
            except Exception as e_hist:
                history_error = str(e_hist)

            snapshot = {
                "agent_name": self.agent_name,
                "tick": current_tick,
                "attempt_number": attempt_number,
                "mode": mode,
                "model_name": self.model_name,
                "generation_config": self._to_jsonable(self.generation_config),
                "system_prompt": self.system_prompt,
                "user_prompt": user_prompt,
                "history_before_send": self._to_jsonable(history_dump),
                "history_error": history_error,
            }
            self._write_debug_api_snapshot(filename, snapshot)
        except Exception as e:
            self._log("WARNING", f"Failed to write Gemini send snapshot: {e}")

    def _dump_response_snapshot(self, current_tick: int, attempt_number: int, mode: str, payload: Any) -> None:
        if not self._debug_api_enabled():
            return
        safe_agent_name = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in self.agent_name)
        ts_ms = int(time.time() * 1000)
        filename = f"gemini_response_{safe_agent_name}_tick{current_tick}_attempt{attempt_number}_{mode}_{ts_ms}.json"
        snapshot = {
            "agent_name": self.agent_name,
            "tick": current_tick,
            "attempt_number": attempt_number,
            "mode": mode,
            "model_name": self.model_name,
            "response": self._to_jsonable(payload),
        }
        self._write_debug_api_snapshot(filename, snapshot)

    def _build_generation_config(self) -> google_genai_types.GenerateContentConfig:
        config_kwargs = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "safety_settings": self.safety_settings,
            "system_instruction": self.system_prompt,
        }
        thinking_config = self._build_thinking_config()
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        return google_genai_types.GenerateContentConfig(**config_kwargs)

    def _build_thinking_config(self) -> Optional[google_genai_types.ThinkingConfig]:
        """Return the right thinking config for the model family."""
        model_prefix = (self.model_name or "").lower()
        if model_prefix.startswith("models/"):
            model_prefix = model_prefix[len("models/"):]
        if model_prefix.startswith("gemini-2.0"):
            return None
        if model_prefix.startswith("gemini-2.5"):
            return google_genai_types.ThinkingConfig(thinking_budget=24576, include_thoughts=True)
        return google_genai_types.ThinkingConfig(include_thoughts=True, thinking_level="high")

    def _normalize_thought_signature_for_storage(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            return base64.b64encode(bytes(value)).decode("ascii")
        if isinstance(value, str):
            return value
        try:
            return str(value)
        except Exception:
            return None

    def _extract_part_thought_signature(self, part: Any) -> Optional[str]:
        for key in ("thought_signature", "thoughtSignature"):
            value = None
            if isinstance(part, dict):
                value = part.get(key)
            else:
                value = getattr(part, key, None)
            normalized = self._normalize_thought_signature_for_storage(value)
            if normalized is not None:
                return normalized
        return None

    def _encode_varint(self, value: int) -> bytes:
        encoded = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                encoded.append(byte | 0x80)
            else:
                encoded.append(byte)
                return bytes(encoded)

    def _read_varint(self, data: bytes, offset: int) -> Tuple[int, int]:
        value = 0
        shift = 0
        position = offset
        while position < len(data):
            byte = data[position]
            value |= (byte & 0x7F) << shift
            position += 1
            if not byte & 0x80:
                return value, position
            shift += 7
            if shift > 70:
                raise ValueError("varint_too_long")
        raise ValueError("truncated_varint")

    def _skip_protobuf_field(self, data: bytes, offset: int, wire_type: int) -> int:
        if wire_type == 0:
            _value, next_offset = self._read_varint(data, offset)
            return next_offset
        if wire_type == 1:
            next_offset = offset + 8
            if next_offset > len(data):
                raise ValueError("truncated_fixed64")
            return next_offset
        if wire_type == 2:
            payload_length, next_offset = self._read_varint(data, offset)
            end_offset = next_offset + payload_length
            if end_offset > len(data):
                raise ValueError(
                    f"truncated_length_delimited_field_length={payload_length}_offset={next_offset}"
                )
            return end_offset
        if wire_type == 5:
            next_offset = offset + 4
            if next_offset > len(data):
                raise ValueError("truncated_fixed32")
            return next_offset
        raise ValueError(f"unsupported_wire_type={wire_type}")

    def _parse_protobuf_envelope_fields(self, data: bytes) -> List[Tuple[int, int]]:
        fields: List[Tuple[int, int]] = []
        offset = 0
        while offset < len(data):
            tag, payload_offset = self._read_varint(data, offset)
            field_number = tag >> 3
            wire_type = tag & 0x07
            if field_number == 0:
                raise ValueError(f"invalid_field_number_0_at_offset={offset}")
            end_offset = self._skip_protobuf_field(data, payload_offset, wire_type)
            fields.append((field_number, wire_type))
            offset = end_offset
        return fields

    def _repair_raw_payload_thought_signature(self, decoded: bytes) -> str:
        wrapped = b"\x0a" + self._encode_varint(len(decoded)) + decoded
        return base64.b64encode(wrapped).decode("ascii")

    def _thought_signature_diagnostics(self, value: Any) -> Optional[Dict[str, Any]]:
        signature = self._normalize_thought_signature_for_storage(value)
        if signature is None:
            return None

        diagnostics: Dict[str, Any] = {
            "signature": signature,
            "signature_b64_length": len(signature),
            "decoded_length": None,
            "first_bytes_hex": "",
            "valid_protobuf_envelope": False,
            "invalid_signature": False,
            "wrong_leading_byte": False,
            "repairable_raw_payload": False,
            "repaired_signature": None,
            "protobuf_fields": [],
            "reason": None,
        }
        try:
            decoded = base64.b64decode(signature, validate=True)
        except Exception as exc:
            diagnostics["invalid_signature"] = True
            diagnostics["reason"] = f"base64_decode_error={type(exc).__name__}: {exc}"
            return diagnostics

        diagnostics["decoded_length"] = len(decoded)
        diagnostics["first_bytes_hex"] = " ".join(f"{byte:02x}" for byte in decoded[:16])
        if not decoded:
            diagnostics["invalid_signature"] = True
            diagnostics["reason"] = "empty_decoded_signature"
            return diagnostics

        try:
            fields = self._parse_protobuf_envelope_fields(decoded)
            diagnostics["valid_protobuf_envelope"] = True
            diagnostics["protobuf_fields"] = fields[:8]
            diagnostics["reason"] = None
        except Exception as exc:
            diagnostics["invalid_signature"] = True
            diagnostics["wrong_leading_byte"] = decoded[0] != 0x0A
            diagnostics["reason"] = f"invalid_protobuf_envelope={type(exc).__name__}: {exc}"
            if decoded[0] == 0x01:
                diagnostics["repairable_raw_payload"] = True
                diagnostics["repaired_signature"] = self._repair_raw_payload_thought_signature(decoded)
        return diagnostics

    def _format_thought_signature_diagnostics(self, diagnostics: Dict[str, Any]) -> str:
        parts = [
            f"signature_b64_length={diagnostics.get('signature_b64_length')}",
            f"decoded_length={diagnostics.get('decoded_length')}",
            f"first_bytes={diagnostics.get('first_bytes_hex') or '<none>'}",
        ]
        if diagnostics.get("protobuf_fields"):
            parts.append(f"protobuf_fields={diagnostics.get('protobuf_fields')}")
        if diagnostics.get("repairable_raw_payload"):
            parts.append("repairable_raw_payload=True")
        reason = diagnostics.get("reason")
        if reason:
            parts.append(f"reason={reason}")
        if "processed_index" in diagnostics:
            parts.append(f"processed_index={diagnostics.get('processed_index')}")
        if "history_index" in diagnostics:
            parts.append(f"history_index={diagnostics.get('history_index')}")
        if "tick" in diagnostics:
            parts.append(f"tick={diagnostics.get('tick')}")
        return ", ".join(parts)

    def _find_invalid_thought_signatures(
        self,
        processed_history_entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        bad_signatures: List[Dict[str, Any]] = []
        for processed_index, entry in enumerate(processed_history_entries):
            if entry.get("role") != "model":
                continue
            diagnostics = self._thought_signature_diagnostics(entry.get("thought_signature"))
            if not diagnostics or not diagnostics.get("invalid_signature"):
                continue
            diagnostics["processed_index"] = processed_index
            bad_signatures.append(diagnostics)
            self._log(
                "WARNING",
                "Gemini invalid thought_signature detected in active pruned history: "
                f"{self._format_thought_signature_diagnostics(diagnostics)}"
            )

        if not bad_signatures:
            self._log(
                "WARNING",
                "Gemini reported a corrupted thought signature, "
                "but no active invalid thought_signature was found by protobuf-envelope parsing."
            )
        else:
            self._log(
                "INFO",
                f"Gemini found {len(bad_signatures)} invalid "
                "thought_signature(s) in active pruned history."
            )
        return bad_signatures

    def _find_last_wrong_leading_thought_signature(
        self,
        processed_history_entries: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        bad_signatures = self._find_invalid_thought_signatures(processed_history_entries)
        return bad_signatures[-1] if bad_signatures else None

    def _find_invalid_thought_signatures_in_active_history(self) -> List[Dict[str, Any]]:
        raw_history_with_ticks = self._load_history_from_file()
        processed_history_entries = self._filter_and_prune_history(raw_history_with_ticks)
        return self._find_invalid_thought_signatures(processed_history_entries)

    def _find_last_wrong_leading_thought_signature_in_active_history(self) -> Optional[Dict[str, Any]]:
        bad_signatures = self._find_invalid_thought_signatures_in_active_history()
        return bad_signatures[-1] if bad_signatures else None

    def _remove_persistent_thought_signature(self, bad_signature: Dict[str, Any]) -> bool:
        return self._remove_persistent_thought_signatures([bad_signature]) > 0

    def _replace_persistent_thought_signatures(self, repaired_signatures: List[Dict[str, Any]]) -> int:
        replacements: Dict[str, str] = {}
        for diagnostics in repaired_signatures:
            signature = diagnostics.get("signature")
            repaired_signature = diagnostics.get("repaired_signature")
            if signature and repaired_signature:
                replacements[signature] = repaired_signature
        if not replacements:
            self._log(
                "WARNING",
                "Cannot repair persisted Gemini thought_signature; "
                "no replacement signatures were provided."
            )
            return 0
        if not os.path.exists(self.history_file_path):
            self._log(
                "WARNING",
                "Cannot repair persisted Gemini thought_signature; "
                f"history file does not exist: {self.history_file_path}"
            )
            return 0

        entries = file_io_utils.load_yaml_lines(self.history_file_path)
        replacement_count = 0
        for history_index, entry in enumerate(entries):
            if not isinstance(entry, dict) or entry.get("role") != "model":
                continue
            parts = entry.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                for key in ("thought_signature", "thoughtSignature"):
                    current_signature = part.get(key)
                    repaired_signature = replacements.get(current_signature)
                    if not repaired_signature:
                        continue
                    diagnostics = self._thought_signature_diagnostics(current_signature) or {
                        "signature": current_signature,
                    }
                    diagnostics["history_index"] = history_index
                    diagnostics["tick"] = entry.get("tick")
                    repaired_diagnostics = self._thought_signature_diagnostics(repaired_signature) or {}
                    self._log(
                        "INFO",
                        f"Repairing persisted Gemini thought_signature in {self.history_file_path}: "
                        f"{self._format_thought_signature_diagnostics(diagnostics)}; "
                        f"repaired_first_bytes={repaired_diagnostics.get('first_bytes_hex') or '<none>'}"
                    )
                    part[key] = repaired_signature
                    replacement_count += 1

        if replacement_count:
            self._save_history_entries(entries)
        else:
            self._log(
                "WARNING",
                "Gemini repair retry succeeded, but no matching "
                "persisted thought_signature was found to repair."
            )
        return replacement_count

    def _remove_persistent_thought_signatures(self, bad_signatures: List[Dict[str, Any]]) -> int:
        target_signatures = {
            diagnostics.get("signature")
            for diagnostics in bad_signatures
            if diagnostics.get("signature")
        }
        if not target_signatures:
            self._log(
                "WARNING",
                "Cannot remove invalid thought_signature "
                "from persistent history because no target signatures were provided."
            )
            return 0
        if not os.path.exists(self.history_file_path):
            self._log(
                "WARNING",
                "Cannot remove invalid thought_signature; "
                f"history file does not exist: {self.history_file_path}"
            )
            return 0

        entries = file_io_utils.load_yaml_lines(self.history_file_path)
        removal_count = 0
        for history_index, entry in enumerate(entries):
            entry = entries[history_index]
            if not isinstance(entry, dict) or entry.get("role") != "model":
                continue
            parts = entry.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                for key in ("thought_signature", "thoughtSignature"):
                    if part.get(key) not in target_signatures:
                        continue
                    diagnostics = self._thought_signature_diagnostics(part.get(key)) or {}
                    diagnostics["history_index"] = history_index
                    diagnostics["tick"] = entry.get("tick")
                    self._log(
                        "INFO",
                        f"Removing persisted invalid Gemini thought_signature from {self.history_file_path}: "
                        f"{self._format_thought_signature_diagnostics(diagnostics)}"
                    )
                    del part[key]
                    removal_count += 1

        if removal_count:
            self._save_history_entries(entries)
        else:
            self._log(
                "WARNING",
                "Cleaned Gemini retry succeeded, but no target "
                "invalid thought_signature was found in persistent history."
            )
        return removal_count

    def _save_history_entries(self, entries: List[Dict[str, Any]]) -> None:
        documents: List[str] = []
        for index, entry in enumerate(entries):
            if index > 0:
                documents.append("---\n")
            documents.append(
                yaml.safe_dump(
                    entry,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                    width=1000,
                )
            )
        file_io_utils.save_text("".join(documents), self.history_file_path)

    def _run_corrupted_signature_cleanup_retry(
        self,
        user_prompt: str,
        current_tick: int,
        attempt_number: int,
        *,
        repair_signatures: Optional[List[Dict[str, Any]]] = None,
        omit_signatures: Optional[List[Dict[str, Any]]] = None,
        label: str,
    ) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        previous_retry_state = getattr(self, "_gemini_bad_signature_retry_active", False)
        previous_retry_stage = getattr(self, "_gemini_bad_signature_retry_stage", None)
        self._gemini_bad_signature_retry_active = True
        self._gemini_bad_signature_retry_stage = label
        try:
            self._initialize_chat_session(
                repair_thought_signatures=repair_signatures,
                omit_thought_signatures=omit_signatures,
            )
            return self._send_message_implementation(
                user_prompt,
                current_tick,
                attempt_number=attempt_number,
            )
        finally:
            self._gemini_bad_signature_retry_active = previous_retry_state
            self._gemini_bad_signature_retry_stage = previous_retry_stage

    def _is_corrupted_signature_exception(self, error: Exception) -> bool:
        if isinstance(error, LLMCorruptedThoughtSignatureError):
            return True
        original_exception = getattr(error, "original_exception", None)
        if isinstance(original_exception, Exception) and self._is_corrupted_signature_exception(original_exception):
            return True
        if isinstance(error, google_genai_errors.ClientError):
            return self._is_corrupted_thought_signature_error(
                self._format_gemini_error_details(error)
            )
        return self._is_corrupted_thought_signature_error(str(error))

    def _persist_omitted_signature_cleanup_after_non_corrupted_error(
        self,
        bad_signatures: List[Dict[str, Any]],
    ) -> None:
        try:
            removal_count = self._remove_persistent_thought_signatures(bad_signatures)
        except Exception as cleanup_error:
            self._log(
                "WARNING",
                "Failed to persist Gemini thought_signature cleanup after "
                f"non-corrupted omission retry error: {cleanup_error}",
            )
            return

        if not removal_count:
            return
        self._log(
            "INFO",
            f"Persisted invalid Gemini thought_signature removal deleted {removal_count} "
            f"entr{'y' if removal_count == 1 else 'ies'} after omission retry reached a "
            "non-corrupted error."
        )
        if getattr(self, "persist_to_disk", True):
            try:
                self._initialize_chat_session()
            except Exception as reinit_error:
                self._log(
                    "WARNING",
                    "Failed to reinitialize Gemini chat after persisting invalid "
                    f"thought_signature removal for non-corrupted retry error: {reinit_error}",
                )

    def _retry_after_repairing_or_omitting_invalid_thought_signatures(
        self,
        user_prompt: str,
        current_tick: int,
        attempt_number: int,
        original_exception: Exception,
        error_details: str,
        bad_signatures: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        self._log(
            "WARNING",
            "Gemini corrupted thought signature detected. "
            "Searching active pruned history for invalid protobuf-envelope signatures."
        )
        if bad_signatures is None:
            bad_signatures = self._find_invalid_thought_signatures_in_active_history()
        if not bad_signatures:
            raise LLMCorruptedThoughtSignatureError(
                f"Gemini corrupted thought signature for {self.agent_name}; no repairable "
                f"or removable invalid signature found: {error_details}",
                original_exception=original_exception,
            )

        repairable_signatures = [
            diagnostics for diagnostics in bad_signatures
            if diagnostics.get("repairable_raw_payload") and diagnostics.get("repaired_signature")
        ]
        if repairable_signatures:
            self._log(
                "INFO",
                f"Gemini will retry once after repairing "
                f"{len(repairable_signatures)} invalid thought_signature(s)."
            )
            try:
                retry_result = self._run_corrupted_signature_cleanup_retry(
                    user_prompt,
                    current_tick,
                    attempt_number,
                    repair_signatures=repairable_signatures,
                    label="repair",
                )
            except Exception as repair_error:
                self._log(
                    "WARNING",
                    "Gemini retry after repairing invalid "
                    f"thought_signature(s) failed; will retry after omitting invalid "
                    f"signature(s). Repair error: {type(repair_error).__name__}: {repair_error}"
                )
            else:
                self._log(
                    "INFO",
                    f"Gemini retry succeeded after repairing "
                    f"{len(repairable_signatures)} invalid thought_signature(s)."
                )
                replacement_count = self._replace_persistent_thought_signatures(repairable_signatures)
                if replacement_count:
                    self._log(
                        "INFO",
                        f"Persisted Gemini thought_signature repair "
                        f"updated {replacement_count} entr{'y' if replacement_count == 1 else 'ies'}."
                    )
                    if getattr(self, "persist_to_disk", True):
                        try:
                            self._initialize_chat_session()
                        except Exception as reinit_error:
                            self._log(
                                "WARNING",
                                "Failed to reinitialize Gemini chat "
                                f"after repairing persisted thought_signature(s): {reinit_error}"
                            )
                return retry_result
        else:
            self._log(
                "INFO",
                "No repairable raw-payload Gemini "
                "thought_signature was found; proceeding to omission retry."
            )

        self._log(
            "INFO",
            f"Gemini will retry once after omitting "
            f"{len(bad_signatures)} invalid thought_signature(s)."
        )
        try:
            retry_result = self._run_corrupted_signature_cleanup_retry(
                user_prompt,
                current_tick,
                attempt_number,
                omit_signatures=bad_signatures,
                label="omit",
            )
        except Exception as removal_retry_error:
            if self._is_corrupted_signature_exception(removal_retry_error):
                self._log(
                    "ERROR",
                    "Gemini retry after omitting invalid "
                    "thought_signature(s) still failed with corrupted signature. "
                    f"Station will pause. Removed candidates={len(bad_signatures)}. "
                    f"Retry error: {type(removal_retry_error).__name__}: {removal_retry_error}"
                )
                raise LLMCorruptedThoughtSignatureError(
                    f"Gemini corrupted thought signature for {self.agent_name} after repair "
                    f"and removal retries failed: {type(removal_retry_error).__name__}: "
                    f"{removal_retry_error}",
                    original_exception=removal_retry_error,
                )

            self._log(
                "WARNING",
                "Gemini retry after omitting invalid thought_signature(s) reached a "
                "non-corrupted error; returning to normal retry handling. "
                f"Removed candidates={len(bad_signatures)}. "
                f"Retry error: {type(removal_retry_error).__name__}: {removal_retry_error}"
            )
            self._persist_omitted_signature_cleanup_after_non_corrupted_error(bad_signatures)
            if isinstance(removal_retry_error, LLMTransientAPIError):
                raise
            raise LLMTransientAPIError(
                f"Gemini retry after omitting invalid thought_signature(s) for "
                f"{self.agent_name} reached a non-corrupted error; returning to "
                f"normal retry handling. Error: {type(removal_retry_error).__name__}: "
                f"{removal_retry_error}",
                original_exception=removal_retry_error,
            )

        self._log(
            "INFO",
            f"Gemini retry succeeded after omitting "
            f"{len(bad_signatures)} invalid thought_signature(s)."
        )
        removal_count = self._remove_persistent_thought_signatures(bad_signatures)
        if removal_count:
            self._log(
                "INFO",
                f"Persisted invalid Gemini "
                f"thought_signature removal deleted {removal_count} entr"
                f"{'y' if removal_count == 1 else 'ies'} after successful retry."
            )
            if getattr(self, "persist_to_disk", True):
                try:
                    self._initialize_chat_session()
                except Exception as reinit_error:
                    self._log(
                        "WARNING",
                        "Failed to reinitialize Gemini chat "
                        f"after removing persisted invalid thought_signature(s): "
                        f"{reinit_error}"
                    )
        else:
            self._log(
                "WARNING",
                "Gemini retry succeeded, but persistent "
                "invalid thought_signature removal did not modify history."
            )
        return retry_result

    def _retry_after_omitting_wrong_thought_signature(
        self,
        user_prompt: str,
        current_tick: int,
        attempt_number: int,
        original_exception: Exception,
        error_details: str,
        bad_signatures: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        return self._retry_after_repairing_or_omitting_invalid_thought_signatures(
            user_prompt,
            current_tick,
            attempt_number,
            original_exception,
            error_details,
            bad_signatures=bad_signatures,
        )

    def _handle_system_prompt_update(self) -> None:
        if not self.generation_config:
            return
        self.generation_config = self._build_generation_config()

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
                        thought_signature = None
                        for part in entry["parts"]:
                            if isinstance(part, dict):
                                thought_signature = self._extract_part_thought_signature(part)
                                if thought_signature is not None:
                                    break
                        history_for_filtering.append({
                            "tick": entry["tick"],
                            "role": entry["role"], 
                            "text_content": text_content,
                            "thinking_content": thinking_content,
                            "thought_signature": thought_signature,
                        })
                    else:
                        self._log(
                            "WARNING",
                            f"Malformed history entry in {self.history_file_path}, skipping: {entry}",
                        )
            except Exception as e:
                self._log(
                    "ERROR",
                    f"Error loading raw chat history from {self.history_file_path}: {e}.",
                )
        return history_for_filtering

    def _append_turn_to_history_file(
        self,
        tick: int,
        role: str,
        text: str,
        thinking_text: Optional[str] = None,
        token_info: Optional[Dict[str, Optional[int]]] = None,
        api_metadata: Optional[Dict[str, Any]] = None,
        thought_signature: Optional[str] = None,
    ) -> None:
        if not getattr(self, "persist_to_disk", True):
            return
        if not text and not thinking_text: # Don't save if both are empty
            return
        try:
            part_data: Dict[str, Any] = {'text': text}
            if role == 'model' and thought_signature is not None:
                part_data['thought_signature'] = thought_signature
            turn_data = {'tick': tick, 'role': role, 'parts': [part_data]}
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
            self._log(
                "ERROR",
                f"Error appending turn to history file {self.history_file_path}: {e}",
            )

    def _build_gemini_api_metadata(
        self,
        raw_response: Any,
        usage_metadata: Any,
        streaming: bool,
        prompt_feedback: Any = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        raw_response_jsonable = self._to_jsonable(raw_response) if raw_response is not None else None
        request_id = None
        if isinstance(raw_response_jsonable, dict):
            sdk_http_response = raw_response_jsonable.get("sdk_http_response")
            if isinstance(sdk_http_response, dict):
                headers = sdk_http_response.get("headers")
                if isinstance(headers, dict):
                    request_id = (
                        headers.get("x-modelverse-request-id")
                        or headers.get("x-request-id")
                        or headers.get("request-id")
                    )
        metadata: Dict[str, Any] = {
            "provider": "gemini",
            "streaming": streaming,
            "model_name": self.model_name,
            "endpoint_name": self.active_endpoint_name,
            "base_url": self.active_base_url,
            "response_id": getattr(raw_response, "response_id", None),
            "request_id": request_id,
            "model_version": getattr(raw_response, "model_version", None),
            "prompt_feedback": self._sanitize_api_return_payload(self._to_jsonable(prompt_feedback)) if prompt_feedback is not None else None,
            "usage_raw": self._sanitize_api_return_payload(self._to_jsonable(usage_metadata)) if usage_metadata is not None else None,
            "raw_return": self._sanitize_api_return_payload(raw_response_jsonable) if raw_response_jsonable is not None else None,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return self._prepare_api_metadata_for_persistence(metadata)

    def _initialize_chat_session(
        self,
        omit_wrong_thought_signature: Optional[Dict[str, Any]] = None,
        omit_thought_signatures: Optional[List[Dict[str, Any]]] = None,
        repair_thought_signatures: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not self.client:
            raise ConnectionError(f"genai.Client not initialized for {self.agent_name}.")

        raw_history_with_ticks = self._load_history_from_file()
        # self.agent_pruned_ticks_info is used by _filter_and_prune_history
        processed_history_entries = self._filter_and_prune_history(raw_history_with_ticks)

        omit_targets: List[Dict[str, Any]] = list(omit_thought_signatures or [])
        if omit_wrong_thought_signature is not None:
            omit_targets.append(omit_wrong_thought_signature)
        omit_lookup = {
            (target.get("processed_index"), target.get("signature"))
            for target in omit_targets
            if target.get("signature") is not None
        }
        repair_lookup = {
            (target.get("processed_index"), target.get("signature")): target
            for target in (repair_thought_signatures or [])
            if target.get("signature") is not None and target.get("repaired_signature") is not None
        }
        omitted_targets = set()
        repaired_targets = set()

        sdk_history_for_init: List[google_genai_types.ContentDict] = []
        for processed_index, entry in enumerate(processed_history_entries):
            sdk_role = entry['role']
            if sdk_role not in ['user', 'model']:
                self._log(
                    "WARNING",
                    f"Invalid role '{sdk_role}' in processed history, defaulting to 'user'. Entry: {entry}",
                )
                sdk_role = 'user'

            part_for_init: Dict[str, Any] = {'text': entry['text_content']}
            if sdk_role == 'model':
                thought_signature = self._normalize_thought_signature_for_storage(entry.get('thought_signature'))
                if thought_signature is not None:
                    lookup_key = (processed_index, thought_signature)
                    repair_target = repair_lookup.get(lookup_key)
                    if repair_target is not None:
                        repaired_signature = repair_target.get("repaired_signature")
                        part_for_init['thought_signature'] = repaired_signature
                        repaired_targets.add(lookup_key)
                        repaired_diagnostics = self._thought_signature_diagnostics(repaired_signature) or {}
                        self._log(
                            "INFO",
                            "Repaired invalid Gemini thought_signature for retry SDK history: "
                            f"{self._format_thought_signature_diagnostics(repair_target)}; "
                            f"repaired_first_bytes={repaired_diagnostics.get('first_bytes_hex') or '<none>'}"
                        )
                    elif lookup_key in omit_lookup:
                        omitted_targets.add(lookup_key)
                        self._log(
                            "INFO",
                            "Omitted invalid Gemini thought_signature from retry SDK history: "
                            f"processed_index={processed_index}, "
                            f"{self._format_thought_signature_diagnostics(self._thought_signature_diagnostics(thought_signature) or {})}"
                        )
                    else:
                        part_for_init['thought_signature'] = thought_signature

            sdk_history_for_init.append(google_genai_types.ContentDict({
                'role': sdk_role, 
                'parts': [part_for_init]
            }))
        missing_omissions = omit_lookup - omitted_targets
        if missing_omissions:
            self._log(
                "WARNING",
                f"Requested omission of {len(missing_omissions)} "
                "Gemini thought_signature(s), but they were not found while rebuilding SDK history."
            )
        missing_repairs = set(repair_lookup.keys()) - repaired_targets
        if missing_repairs:
            self._log(
                "WARNING",
                f"Requested repair of {len(missing_repairs)} "
                "Gemini thought_signature(s), but they were not found while rebuilding SDK history."
            )
        
        try:
            # The system_instruction is part of self.generation_config which is used in send_message.
            # For chats.create, only the turn history is typically passed.
            self.chat_session = self.client.chats.create(
                model=self.model_name,
                history=sdk_history_for_init if sdk_history_for_init else None,
            )
            if self._debug_api_enabled():
                self._log(
                    "DEBUG",
                    f"Gemini chat session initialized sdk_history_entries={len(sdk_history_for_init)}.",
                )
        except Exception as e:
            self._log(
                "ERROR",
                f"Gemini chat session initialization failed model={self.model_name!r}: "
                f"{self._format_retry_error_for_log(e)}",
            )
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
        model_turn_thought_signature: Optional[str] = None

        if not self.chat_session: # Should have been initialized or re-initialized by send_message
             err_msg = f"SYSTEM_ERROR: Chat session for {self.agent_name} is not available in _send_message_implementation."
             self._log("ERROR", err_msg)
             return err_msg, None, token_info
        
        try:
            # Use one-off generation for first attempt, streaming for retries
            if attempt_number == 0:
                self._dump_send_payload_snapshot(user_prompt, current_tick, attempt_number, mode="non_stream")
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
                        if model_turn_thought_signature is None:
                            model_turn_thought_signature = self._extract_part_thought_signature(part)
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
                self._dump_response_snapshot(current_tick, attempt_number, "non_stream", api_response)
                
            else:
                # Retry attempts: use streaming to avoid timeout issues
                self._log(
                    "INFO",
                    f"Gemini send mode=stream attempt={attempt_number} "
                    "reason=retry_timeout_avoidance.",
                )
                self._dump_send_payload_snapshot(user_prompt, current_tick, attempt_number, mode="stream")
                
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
                                if model_turn_thought_signature is None:
                                    model_turn_thought_signature = self._extract_part_thought_signature(part)
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
                self._dump_response_snapshot(
                    current_tick,
                    attempt_number,
                    "stream",
                    {
                        "prompt_feedback": self._to_jsonable(prompt_feedback),
                        "usage_metadata": self._to_jsonable(final_usage_metadata),
                        "thinking_text_parts": thinking_text_parts,
                        "llm_text_response_parts": llm_text_response_parts,
                    },
                )
            
            # Process usage metadata if available
            if final_usage_metadata:
                token_info['last_exchange_prompt_tokens'] = getattr(final_usage_metadata, 'prompt_token_count', None)
                token_info['last_exchange_completion_tokens'] = getattr(final_usage_metadata, 'candidates_token_count', None)
                token_info['last_exchange_cached_tokens'] = getattr(final_usage_metadata, 'cached_content_token_count', None)
                token_info['last_exchange_thoughts_tokens'] = getattr(final_usage_metadata, 'thoughts_token_count', None)
                token_info['total_tokens_in_session'] = getattr(final_usage_metadata, 'total_token_count', None)
            
            if token_info['total_tokens_in_session'] is None and self.client and self.chat_session:
                try:
                    self._log(
                        "WARNING",
                        "total_token_count not in usage_metadata. Recounting session tokens manually.",
                    )
                    current_sdk_history = self.chat_session.get_history()
                    count_response = self.client.models.count_tokens(
                        model=self.model_name,
                        contents=current_sdk_history
                    )
                    token_info['total_tokens_in_session'] = count_response.total_tokens
                except Exception as count_e:
                    self._log(
                        "WARNING",
                        f"Could not count total session tokens after send_message: {count_e}",
                    )

            prompt_feedback = getattr(api_response, 'prompt_feedback', None) if attempt_number == 0 else prompt_feedback
            raw_response_for_metadata = api_response if attempt_number == 0 else {
                "prompt_feedback": self._to_jsonable(prompt_feedback),
                "usage_metadata": self._to_jsonable(final_usage_metadata),
                "streaming_retry_attempt": attempt_number,
            }
            api_metadata = self._build_gemini_api_metadata(
                raw_response=raw_response_for_metadata,
                usage_metadata=final_usage_metadata,
                streaming=attempt_number > 0,
                prompt_feedback=prompt_feedback,
                extra_metadata={
                    "thought_signature_present": model_turn_thought_signature is not None,
                },
            )
            self._last_model_turn_thought_signature = model_turn_thought_signature

            self._append_turn_to_history_file(current_tick, 'user', user_prompt, None, None)
            self._append_turn_to_history_file(
                current_tick,
                'model',
                llm_text_response,
                thinking_text,
                token_info,
                api_metadata,
                thought_signature=model_turn_thought_signature
            )

            # Streaming can leave highly fragmented SDK-side comprehensive history.
            # Rebuild from canonical YAML history after a successful streaming send.
            if attempt_number > 0:
                if getattr(self, "persist_to_disk", True):
                    try:
                        self._initialize_chat_session()
                    except Exception as reinit_err:
                        self._log(
                            "WARNING",
                            f"Failed to reinitialize chat after streaming send: {reinit_err}",
                        )
                else:
                    self._needs_reload_after_staged_history_flush = True
            
            return llm_text_response, thinking_text, token_info

        except google_genai_errors.ServerError as e:
            # Log detailed error information for debugging
            error_details = self._format_gemini_error_details(e)
            mode = "stream" if attempt_number > 0 else "non_stream"
            self._log(
                "ERROR",
                f"Gemini send failed mode={mode} attempt={attempt_number} "
                f"error={self._format_original_exception_for_log(e)}.",
            )
            raise LLMTransientAPIError(f"Gemini API Server Error for {self.agent_name}: {error_details}", original_exception=e)
        except LLMSafetyBlockError:
            raise
        except google_genai_errors.ClientError as e:
            error_details = self._format_gemini_error_details(e)
            mode = "stream" if attempt_number > 0 else "non_stream"
            self._log(
                "ERROR",
                f"Gemini send failed mode={mode} attempt={attempt_number} "
                f"error={self._format_original_exception_for_log(e)}.",
            )
            is_corrupted_signature_error = self._is_corrupted_thought_signature_error(error_details)
            if is_corrupted_signature_error:
                if getattr(self, "_gemini_bad_signature_retry_active", False):
                    retry_stage = getattr(self, "_gemini_bad_signature_retry_stage", None)
                    if retry_stage == "repair":
                        self._log(
                            "WARNING",
                            "Gemini repair retry still failed with "
                            f"corrupted thought signature. Removal fallback will run next. "
                            f"Error: {error_details}"
                        )
                    else:
                        self._log(
                            "ERROR",
                            "Gemini cleaned retry still failed with "
                            f"corrupted thought signature. Station will pause. Error: {error_details}"
                        )
                    raise LLMCorruptedThoughtSignatureError(
                        f"Gemini corrupted thought signature for {self.agent_name} after cleaned retry: {error_details}",
                        original_exception=e,
                    )
                return self._retry_after_omitting_wrong_thought_signature(
                    user_prompt,
                    current_tick,
                    attempt_number,
                    original_exception=e,
                    error_details=error_details,
                )
            # Check for specific context overflow error pattern
            if (hasattr(e, 'status') and e.status == 'INVALID_ARGUMENT' and 
                hasattr(e, 'message') and 'input token count exceeds the maximum number of tokens allowed' in str(e.message)):
                self._log("CRITICAL", "Context window overflow detected in Gemini API")
                raise LLMContextOverflowError(f"Context window overflow for {self.agent_name}: {error_details}", original_exception=e)
            
            # Handle other client errors normally (rate limits, auth, etc.)
            if hasattr(e, 'status') and e.status == 'RESOURCE_EXHAUSTED':
                raise LLMTransientAPIError(f"Gemini API quota/rate limit error for {self.agent_name}: {error_details}", original_exception=e)
            else:
                raise LLMPermanentAPIError(f"Gemini API client error for {self.agent_name}: {error_details}", original_exception=e)
        except Exception as e:
            # Log detailed error information for debugging
            self._log("ERROR", f"Unexpected Gemini exception: type={type(e).__name__}, str={str(e)}")
            if hasattr(e, '__dict__'):
                error_attrs = {k: v for k, v in e.__dict__.items() if not k.startswith('_')}
                if error_attrs:
                    self._log("ERROR", f"Unexpected Gemini exception attributes: {error_attrs}")
            
            import traceback; traceback.print_exc()
            raise LLMConnectorError(f"Unexpected LLM API call failure for {self.agent_name}. Details: {str(e)}", original_exception=e)

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Returns the current (pruned) chat history from the active session."""
        if not self.chat_session:
            self._log(
                "WARNING",
                "get_chat_history called but no active chat session. "
                "Attempting to reconstruct from file (may be slow or incomplete if init failed).",
            )
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
            self._log("ERROR", f"converting SDK history to simple format: {e}")
        return simple_history
    
    def get_current_total_session_tokens(self) -> Optional[int]:
        """Calculates total tokens based on the current, possibly pruned, chat session history."""
        if not self.client: return None

        history_for_count_sdk_format: List[google_genai_types.ContentDict] = []
        if self.chat_session:
            try:
                history_for_count_sdk_format = self.chat_session.get_history()
            except Exception as e:
                self._log(
                    "ERROR",
                    f"getting history from active session for token count: {e}. "
                    "Will attempt to load, prune, and convert from file.",
                )
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
            self._log(
                "WARNING",
                "No active chat session for get_current_total_session_tokens. Loading/pruning from file.",
            )
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

        if not self._is_official_base_url(self.active_base_url):
            self._log(
                "INFO",
                "Third-party Gemini provider detected. Token count unavailable without completion fallback.",
            )
            return None

        try:
            count_response = self.client.models.count_tokens(
                model=self.model_name,
                contents=history_for_count_sdk_format,
            )
            token_count = count_response.total_tokens
            if token_count is None:
                self._log("WARNING", "count_tokens returned None. Token count unavailable.")
                return None
            return token_count
        except Exception as e:
            self._log("WARNING", f"count_tokens failed: {e}. Token count unavailable.")
            return None
