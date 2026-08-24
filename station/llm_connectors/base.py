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
from station import runtime_api_config
from station.system_messages import build_station_level_system_prompt


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


class LLMCorruptedThoughtSignatureError(LLMPermanentAPIError):
    """Indicates Gemini rejected persisted thought signatures in the submitted history."""
    pass


class LLMSafetyBlockError(LLMConnectorError):
    """Indicates the response was blocked due to safety filters."""
    def __init__(self, message: str, block_reason: Optional[str] = None, prompt_feedback: Any = None, original_exception: Optional[Exception] = None):
        super().__init__(message, original_exception)
        self.block_reason = block_reason
        self.prompt_feedback = prompt_feedback


class LLMContextOverflowError(LLMConnectorError):
    """Indicates the input exceeds the model's context window limit."""
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
        self._explicit_api_key = api_key is not None
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        self.agent_data_path = agent_data_path 
        self.history_file_path = os.path.join(self.agent_data_path, "llm_chat_history.yamll")

        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        # When False, the connector must not write anything to station_data (history files, agent YAML, etc).
        # Parallel staged sends use this mode while they prepare isolated output.
        self.persist_to_disk: bool = True
        self._needs_reload_after_staged_history_flush: bool = False
        self._last_api_metadata: Optional[Dict[str, Any]] = None
        runtime_proxy_snapshot = getattr(self, "_api_runtime_config_snapshot", None)
        if not isinstance(runtime_proxy_snapshot, dict):
            runtime_proxy_snapshot = runtime_api_config.get_station_proxy_snapshot()
        self.api_runtime_config_generation: int = int(runtime_proxy_snapshot.get("generation", 0))
        
        self._apply_runtime_proxy_snapshot(runtime_proxy_snapshot)
        
        # Load context filters and store copies to detect changes.
        self.agent_prune_blocks: List[Dict[str, Any]] = self._load_prune_blocks_from_agent_data()
        self._last_known_prune_blocks: List[Dict[str, Any]] = copy.deepcopy(self.agent_prune_blocks)
        self.context_history_start_tick: Optional[int] = self._load_context_history_start_tick()
        self._last_known_context_history_start_tick: Optional[int] = self.context_history_start_tick
        self._last_known_system_prompt: Optional[str] = self.system_prompt
        self._debug_station_id: Optional[str] = None
        if not hasattr(self, "api_runtime_provider_id"):
            self.api_runtime_provider_id: Optional[str] = None
        if not hasattr(self, "api_runtime_env_names"):
            self.api_runtime_env_names: Tuple[str, ...] = ()

    def _debug_api_enabled(self) -> bool:
        raw_value = str(os.getenv("DEBUG_API", "")).strip().lower()
        return raw_value in {"1", "true", "yes", "on"}

    def _apply_runtime_proxy_snapshot(self, runtime_proxy_snapshot: Dict[str, Any]) -> None:
        """Apply the effective runtime proxy for this connector before client creation."""
        http_proxy = runtime_proxy_snapshot.get("http_proxy")
        https_proxy = runtime_proxy_snapshot.get("https_proxy")
        for env_name in ("http_proxy", "HTTP_PROXY"):
            if http_proxy:
                os.environ[env_name] = http_proxy
            else:
                os.environ.pop(env_name, None)
        for env_name in ("https_proxy", "HTTPS_PROXY"):
            if https_proxy:
                os.environ[env_name] = https_proxy
            else:
                os.environ.pop(env_name, None)
        grpc_proxy = https_proxy or http_proxy
        if grpc_proxy:
            os.environ["grpc_proxy"] = grpc_proxy
        else:
            os.environ.pop("grpc_proxy", None)

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
                self._log("WARNING", f"Failed to resolve station_id for DEBUG_API path: {e}")
            self._debug_station_id = station_id
        return os.path.join(os.getcwd(), "tmp", "debug_api", self._debug_station_id)

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
            self._log("WARNING", f"Failed to write DEBUG_API snapshot '{filename}': {e}")


    def _load_prune_blocks_from_agent_data(self) -> List[Dict[str, Any]]:
        """Load system-service pruning blocks from agent data."""
        try:
            agent_full_data = agent_module.load_agent_data(self.agent_name, include_ended=True, include_ascended=True)
            if agent_full_data:
                return agent_full_data.get(constants.AGENT_PRUNED_DIALOGUE_TICKS_KEY, [])
            return []
        except Exception as e:
            self._log("ERROR", f"Failed to load prune blocks: {e}")
            return []

    def _load_context_history_start_tick(self) -> Optional[int]:
        try:
            agent_full_data = agent_module.load_agent_data(self.agent_name, include_ended=True, include_ascended=True)
            if not agent_full_data:
                return None
            anchors = [
                int(event[constants.CONTEXT_COMPACTION_ANCHOR_TICK_KEY])
                for event in agent_module.get_context_compaction_events(agent_full_data)
                if event.get(constants.CONTEXT_COMPACTION_ANCHOR_TICK_KEY) is not None
            ]
            return max(anchors) if anchors else None
        except Exception as e:
            self._log("ERROR", f"Failed to load context compaction anchor: {e}")
            return getattr(self, "_last_known_context_history_start_tick", None)

    def _bypass_agent_data_system_prompt_reload(self) -> bool:
        """
        Return True when the connector should keep its constructor-provided system prompt.

        Most Station agents derive their runtime system prompt from agent YAML via
        build_station_level_system_prompt(...). System services such as the archive
        reviewer are different: they use an explicit connector-level system prompt
        plus separate task/context user messages, and should not be wrapped with the
        Station-wide prefix/Codex prompt.
        """
        return self.agent_name == "AutoArchiveEvaluator"

    def _load_system_prompt_from_agent_data(self) -> Optional[str]:
        """Loads current system prompt from agent data."""
        try:
            if self._bypass_agent_data_system_prompt_reload():
                return self._last_known_system_prompt
            agent_full_data = agent_module.load_agent_data(self.agent_name, include_ended=True, include_ascended=True)
            if agent_full_data is None:
                return self._last_known_system_prompt
            raw_prompt = agent_module.get_agent_role_definition(agent_full_data)
            return build_station_level_system_prompt(self.agent_name, raw_prompt)
        except Exception as e:
            self._log("ERROR", f"Failed to load system prompt: {e}")
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
        Applies context compaction anchors and system-service pruning blocks.
        Input entries: List of {'tick': int, 'role': str, 'text_content': str}
        Output entries: List of {'role': str, 'text_content': str} with summary replacements
        """
        if not raw_history_entries:
            return []

        context_history_start_tick = getattr(self, "context_history_start_tick", None)
        if context_history_start_tick is not None:
            start_tick = context_history_start_tick
            anchored_entries = []
            for entry in raw_history_entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    entry_tick = int(entry.get("tick"))
                except (TypeError, ValueError):
                    continue
                if entry_tick >= start_tick:
                    normalized_entry = dict(entry)
                    normalized_entry["tick"] = entry_tick
                    anchored_entries.append(normalized_entry)
            raw_history_entries = anchored_entries
            if not raw_history_entries:
                return []

        # Parse system-service prune blocks into ranges with summaries.
        pruned_ranges = []  # [(start_tick, end_tick, summary), ...]
        for block in getattr(self, "agent_prune_blocks", []):
            ticks_input = block.get(constants.PRUNE_TICKS_KEY)
            summary = block.get(constants.PRUNE_SUMMARY_KEY, "")

            if ticks_input is not None:
                block_ticks = self._parse_ticks_for_filtering(ticks_input)
                if block_ticks:
                    start_tick, end_tick = min(block_ticks), max(block_ticks)
                    pruned_ranges.append((start_tick, end_tick, summary))

        protected_ticks = self._get_protected_ticks(raw_history_entries)

        # Filter out entries within system-service compacted ranges.
        filtered_entries = []
        for entry in raw_history_entries:
            try:
                tick = int(entry.get('tick'))
            except (TypeError, ValueError):
                tick = None
            role = entry.get('role')
            text_content = entry.get('text_content', '')

            if tick is None or role is None:
                self._log("WARNING", f"Skipping history entry with missing tick or role: {entry}")
                continue

            # Always include protected ticks
            if tick in protected_ticks:
                preserved_entry = dict(entry)
                preserved_entry['tick'] = tick
                preserved_entry['role'] = role
                preserved_entry['text_content'] = text_content
                filtered_entries.append(preserved_entry)
                continue

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
            stripped_summary = summary.strip()
            if stripped_summary:
                if start_tick == end_tick:
                    tick_label = f"Tick {start_tick}"
                    verb = "was"
                else:
                    tick_label = f"Ticks {start_tick}-{end_tick}"
                    verb = "were"

                final_entries.append({
                    'role': 'user',
                    'text_content': (
                        f"{tick_label} {verb} compacted by the Station.\n"
                        "Summary:\n"
                        f"{stripped_summary}"
                    ),
                })
                final_entries.append({'role': 'model', 'text_content': "Dialogue compacted."})

        # Add remaining entries after all pruned ranges
        while current_entry_index < len(filtered_entries):
            entry = filtered_entries[current_entry_index]
            out_entry = dict(entry)
            out_entry.pop('tick', None)
            final_entries.append(out_entry)
            current_entry_index += 1

        if self._debug_api_enabled():
            self._log(
                "DEBUG",
                f"Context filtering raw_entries={len(raw_history_entries)} "
                f"active_entries={len(final_entries)}",
            )
        return final_entries

    def _log(self, level: str, message: str) -> None:
        print(f"LLMConnector {level.upper()} ({self.agent_name}): {message}")

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

    def _compact_log_value(self, value: Any, max_length: Optional[int] = 500) -> str:
        text = " ".join(str(value).split())
        if max_length is not None and len(text) > max_length:
            return f"{text[:max_length]}...[truncated]"
        return text

    def _format_exception_for_log(
        self,
        error: Exception,
        *,
        include_raw: bool = True,
        max_message_length: Optional[int] = 500,
    ) -> str:
        """
        Build a useful one-line error summary for retry logs.

        Connector exceptions often wrap SDK exceptions in ``original_exception``;
        include both layers so the retry log says what actually failed.
        """
        parts = [
            f"{type(error).__name__}: {self._compact_log_value(error, max_message_length)}",
        ]
        original_exception = getattr(error, "original_exception", None)
        if include_raw and original_exception is not None:
            formatter = getattr(self, "_format_original_exception_for_log", None)
            if callable(formatter) and max_message_length is not None:
                parts.append("Raw exception: " + formatter(original_exception))
                return " | ".join(parts)

            raw_parts = [
                f"type={type(original_exception).__name__}",
                f"repr={self._compact_log_value(repr(original_exception), max_message_length)!r}",
                f"str={self._compact_log_value(str(original_exception), max_message_length)!r}",
            ]
            for attr_name in ("code", "status", "message", "details", "status_code", "body"):
                if hasattr(original_exception, attr_name):
                    try:
                        raw_parts.append(
                            f"{attr_name}="
                            f"{self._compact_log_value(getattr(original_exception, attr_name), max_message_length)!r}"
                        )
                    except Exception as attr_error:
                        raw_parts.append(f"{attr_name}=<unreadable: {attr_error!r}>")

            response = getattr(original_exception, "response", None)
            if response is not None:
                response_parts = []
                for attr_name in ("status_code", "reason_phrase", "text"):
                    if hasattr(response, attr_name):
                        try:
                            response_parts.append(
                                f"{attr_name}="
                                f"{self._compact_log_value(getattr(response, attr_name), max_message_length)!r}"
                            )
                        except Exception as attr_error:
                            response_parts.append(f"{attr_name}=<unreadable: {attr_error!r}>")
                if response_parts:
                    raw_parts.append("response={" + ", ".join(response_parts) + "}")

            parts.append("Raw exception: " + ", ".join(raw_parts))
        elif hasattr(error, "__dict__"):
            error_attrs = {
                k: v
                for k, v in error.__dict__.items()
                if not k.startswith("_") and k != "original_exception" and v is not None
            }
            if error_attrs:
                parts.append(f"Error attributes: {error_attrs}")
        return " | ".join(parts)

    def _format_retry_error_for_log(self, error: Exception) -> str:
        original_exception = getattr(error, "original_exception", None)
        if original_exception is not None:
            formatter = getattr(self, "_format_original_exception_for_log", None)
            if callable(formatter):
                return (
                    f"{type(error).__name__}; raw="
                    f"{formatter(original_exception)}"
                )
        return self._format_exception_for_log(
            error,
            include_raw=True,
            max_message_length=700,
        )

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
            self._last_api_metadata = None
            return None
        compacted = self._compact_persisted_value(metadata)
        if isinstance(compacted, dict) and compacted:
            self._last_api_metadata = compacted
            return compacted
        self._last_api_metadata = None
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

    def _get_protected_ticks(self, raw_history_entries: List[Dict[str, Any]]) -> set:
        """
        Identify ticks that must remain visible to system-service pruning.
        Normal agent protected context is stored in agent YAML and rendered by
        the compaction anchor prompt, not preserved by raw dialogue tick.
        """
        return set()

    @abc.abstractmethod
    def _send_message_implementation(self, user_prompt: str, current_tick: int, attempt_number: int = 0) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        pass

    def _handle_system_prompt_update(self) -> None:
        """Hook for connectors that need to rebuild internal config when system prompt changes."""
        return None

    def _before_send_message(self, current_tick: int) -> None:
        """Hook for connectors that need to adjust client state before a send."""
        return None

    def _apply_provider_runtime_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Hook for connectors that can rebuild their client from a provider endpoint snapshot."""
        return None

    def _run_provider_base_recovery_probe(self, snapshot: Dict[str, Any]) -> bool:
        """Hook for connectors to probe an earlier endpoint with their current model."""
        return False

    def _provider_fallback_enabled(self) -> bool:
        if not getattr(self, "api_runtime_provider_id", None):
            return False
        if getattr(self, "_explicit_api_key", False):
            return False
        endpoint = runtime_api_config.get_provider_default_endpoint(self.api_runtime_provider_id)
        return bool(endpoint.get("configured"))

    def _current_provider_endpoint_index(self) -> Optional[int]:
        snapshot = getattr(self, "_api_runtime_config_snapshot", None)
        if not isinstance(snapshot, dict):
            return None
        endpoint = snapshot.get("provider_endpoint")
        if not isinstance(endpoint, dict):
            return None
        try:
            return int(endpoint.get("index", 0))
        except Exception:
            return None

    def _provider_base_url_from_snapshot(self, snapshot: Dict[str, Any]) -> Optional[str]:
        snapshot_env = snapshot.get("env") if isinstance(snapshot, dict) else {}
        if not isinstance(snapshot_env, dict):
            return None
        for env_name in tuple(getattr(self, "api_runtime_env_names", ()) or ()):
            if str(env_name).endswith("_BASE_URL"):
                return snapshot_env.get(env_name)
        return None

    def _provider_snapshot_matches_current(self, snapshot: Dict[str, Any]) -> bool:
        endpoint = snapshot.get("provider_endpoint") if isinstance(snapshot, dict) else None
        if not isinstance(endpoint, dict):
            return runtime_api_config.get_generation() == self.api_runtime_config_generation
        try:
            snapshot_index = int(endpoint.get("index", 0))
        except Exception:
            snapshot_index = None
        return (
            int(snapshot.get("generation", 0)) == self.api_runtime_config_generation
            and snapshot_index == self._current_provider_endpoint_index()
        )

    def _prepare_provider_runtime_before_send(self, current_tick: int) -> None:
        if not self._provider_fallback_enabled():
            return

        provider_id = str(self.api_runtime_provider_id)
        env_names = tuple(getattr(self, "api_runtime_env_names", ()) or ())
        current_endpoint_index = self._current_provider_endpoint_index()
        probe_snapshots = runtime_api_config.claim_provider_recovery_probes(provider_id, env_names)
        if probe_snapshots:
            probe_results = []
            for probe_snapshot in probe_snapshots:
                endpoint = probe_snapshot.get("provider_endpoint") or {}
                try:
                    probe_endpoint_index = int(endpoint.get("index", 0))
                except Exception:
                    probe_endpoint_index = 0
                success = False
                try:
                    success = self._run_provider_base_recovery_probe(probe_snapshot)
                except Exception as probe_error:
                    self._log(
                        "INFO",
                        f"Provider recovery probe for {provider_id} endpoint "
                        f"{endpoint.get('name', probe_endpoint_index)} "
                        f"(index={probe_endpoint_index}) failed: "
                        f"{self._format_retry_error_for_log(probe_error)}",
                    )
                probe_results.append((probe_endpoint_index, success))
                if success:
                    self._log(
                        "INFO",
                        f"Provider recovery probe for {provider_id} endpoint "
                        f"{endpoint.get('name', probe_endpoint_index)} "
                        f"(index={probe_endpoint_index}, "
                        f"base_url={self._provider_base_url_from_snapshot(probe_snapshot) or 'provider_default'}) "
                        "succeeded."
                    )
                    break
            runtime_api_config.complete_provider_recovery_probes(provider_id, probe_results)

        default_snapshot = runtime_api_config.get_config_snapshot(env_names, provider_id=provider_id)
        if not self._provider_snapshot_matches_current(default_snapshot):
            endpoint = default_snapshot.get("provider_endpoint") or {}
            try:
                next_endpoint_index = int(endpoint.get("index", 0))
            except Exception:
                next_endpoint_index = None
            if current_endpoint_index is not None and next_endpoint_index != current_endpoint_index:
                self._log(
                    "INFO",
                    f"Switching {provider_id} default endpoint to "
                    f"{endpoint.get('name', endpoint.get('index'))} "
                    f"(index={endpoint.get('index')}, "
                    f"base_url={self._provider_base_url_from_snapshot(default_snapshot) or 'provider_default'})."
                )
            self._apply_provider_runtime_snapshot(default_snapshot)

    def _refresh_runtime_api_config_if_changed(self) -> bool:
        """
        Refresh provider clients after dashboard runtime API/proxy updates.

        Returns True when the connector refreshed internal client state. Existing
        in-flight API calls keep their old client; retry attempts call this hook
        before starting the next attempt.
        """
        return False

    def _handle_send_error(self, error: Exception, current_tick: int) -> bool:
        """
        Hook for connectors that want to react to send errors.

        Returns True to retry immediately without the normal backoff sleep.
        """
        return False

    def _handle_provider_send_error(self, error: Exception, current_tick: int) -> bool:
        if not self._provider_fallback_enabled():
            return False
        provider_id = str(self.api_runtime_provider_id)
        env_names = tuple(getattr(self, "api_runtime_env_names", ()) or ())
        current_endpoint_index = self._current_provider_endpoint_index()
        retry_state = getattr(self, "_provider_send_retry_state", None)
        if not isinstance(retry_state, dict):
            retry_state = {
                "endpoint_index": current_endpoint_index,
                "failure_streak": 0,
                "cycle_count": 0,
            }
            self._provider_send_retry_state = retry_state
        if "cycle_count" not in retry_state:
            retry_state["cycle_count"] = 0
        if retry_state.get("endpoint_index") != current_endpoint_index:
            retry_state["endpoint_index"] = current_endpoint_index
            retry_state["failure_streak"] = 0
        decision = runtime_api_config.advance_provider_fallback_after_failure(
            provider_id,
            current_endpoint_index,
            int(retry_state.get("failure_streak", 0)),
            env_names,
        )
        retry_state["failure_streak"] = int(decision.get("failure_streak", 0))
        if not decision.get("handled"):
            return False

        if decision.get("retry_same_endpoint"):
            self._log(
                "INFO",
                f"{provider_id} endpoint index={current_endpoint_index} failed "
                f"({self._format_retry_error_for_log(error)}); retrying same endpoint once before switching."
            )
            return True

        retry_snapshot = decision.get("retry_snapshot")
        endpoint = retry_snapshot.get("provider_endpoint") or {}
        self._log(
            "INFO",
            f"{provider_id} endpoint index={current_endpoint_index} failed "
            f"({self._format_retry_error_for_log(error)}); switching request to endpoint "
            f"{endpoint.get('name', endpoint.get('index'))} "
            f"(index={endpoint.get('index')}, "
            f"base_url={self._provider_base_url_from_snapshot(retry_snapshot) or 'provider_default'})."
        )
        self._apply_provider_runtime_snapshot(retry_snapshot)
        retry_state["endpoint_index"] = self._current_provider_endpoint_index()
        if decision.get("cycle_wrapped"):
            retry_state["cycle_count"] = int(retry_state.get("cycle_count", 0)) + 1
            retry_delay = self._calculate_retry_delay(int(retry_state["cycle_count"]))
            endpoint_count = endpoint.get("endpoint_count", "?")
            self._log(
                "INFO",
                f"Completed one {provider_id} provider fallback loop "
                f"({endpoint_count} endpoint(s)); waiting {retry_delay}s before starting "
                f"fallback loop #{int(retry_state['cycle_count']) + 1}."
            )
            if retry_delay > 0:
                time.sleep(retry_delay)
        return True

    def _record_provider_success(self) -> None:
        if not self._provider_fallback_enabled():
            return
        runtime_api_config.record_provider_success(
            str(self.api_runtime_provider_id),
            self._current_provider_endpoint_index(),
        )

    def sync_state(self) -> None:
        """
        Synchronize connector state with agent data on disk.

        Checks if context filters or the system prompt changed and re-initializes
        the chat session if needed, then recounts tokens. This method is idempotent - calling it
        multiple times with unchanged state has no effect (no wasted computation).

        Called before generating observations and before sending messages to ensure
        token counts are accurate.
        """
        if getattr(self, "_skip_agent_data_sync", False):
            return

        current_prune_blocks_on_disk = self._load_prune_blocks_from_agent_data()
        current_system_prompt = self._load_system_prompt_from_agent_data()
        current_context_start_tick = self._load_context_history_start_tick()

        # Idempotent check - if context filters are unchanged, this is a no-op.
        prune_changed = current_prune_blocks_on_disk != self._last_known_prune_blocks
        prompt_changed = current_system_prompt != self._last_known_system_prompt
        context_start_changed = current_context_start_tick != self._last_known_context_history_start_tick

        if prune_changed or prompt_changed or context_start_changed:
            if prune_changed:
                self._log("INFO", "Service pruning blocks changed. Re-initializing chat session.")
                self.agent_prune_blocks = copy.deepcopy(current_prune_blocks_on_disk)
                self._last_known_prune_blocks = copy.deepcopy(current_prune_blocks_on_disk)
            if context_start_changed:
                self._log("INFO", "Context compaction anchor changed. Re-initializing chat session.")
                self.context_history_start_tick = current_context_start_tick
                self._last_known_context_history_start_tick = current_context_start_tick
            if prompt_changed:
                self._log("INFO", "System prompt changed. Re-initializing chat session.")
                self.system_prompt = current_system_prompt
                self._last_known_system_prompt = current_system_prompt
                self._handle_system_prompt_update()
            try:
                self._initialize_chat_session()

                # Count tokens after re-initialization and update agent's budget.
                # If pruning changed but this provider cannot recount the rebuilt
                # session, do not persist a stale last-known count.
                can_count_authoritatively = self._can_count_current_session_tokens_authoritatively()
                if (prune_changed or context_start_changed) and not can_count_authoritatively:
                    self._mark_agent_token_budget_stale(
                        constants.TOKEN_BUDGET_STALE_REASON_CONTEXT_COMPACTED
                        if context_start_changed
                        else constants.TOKEN_BUDGET_STALE_REASON_PROVIDER_COUNT_UNAVAILABLE_AFTER_CONTEXT_FILTER
                    )
                    self._log("WARNING", "Token count unavailable after context rebuild; marking token budget stale.")
                    return

                self._log("INFO", "Counting tokens after re-initialization.")
                new_token_count = self.get_current_total_session_tokens()
                if new_token_count is not None:
                    if self.persist_to_disk:
                        # Update agent's token budget in the data file
                        agent_data = agent_module.load_agent_data(self.agent_name)
                        if agent_data:
                            agent_data[constants.AGENT_TOKEN_BUDGET_CURRENT_KEY] = new_token_count
                            agent_data.pop(constants.AGENT_TOKEN_BUDGET_CURRENT_STALE_KEY, None)
                            agent_data.pop(constants.AGENT_TOKEN_BUDGET_STALE_REASON_KEY, None)
                            agent_module.save_agent_data(self.agent_name, agent_data)
                            self._log("INFO", f"Token budget updated to {new_token_count} after context rebuild.")
                        else:
                            self._log("WARNING", "Could not load agent data to update token count after context rebuild.")
                else:
                    if context_start_changed:
                        self._mark_agent_token_budget_stale(
                            constants.TOKEN_BUDGET_STALE_REASON_CONTEXT_COMPACTED
                        )
                    elif prune_changed:
                        self._mark_agent_token_budget_stale(
                            constants.TOKEN_BUDGET_STALE_REASON_PROVIDER_COUNT_UNAVAILABLE_AFTER_CONTEXT_FILTER
                        )
                    self._log("WARNING", "Could not count tokens after context re-initialization.")

            except Exception as e_reinit:
                self._log("ERROR", f"Failed to re-initialize chat session after context filter update: {e_reinit}.")
                # Note: We don't raise here to allow the caller to proceed, but state may be stale

    def send_message(self, user_prompt: str, current_tick: int) -> Tuple[str, Dict[str, Optional[int]]]:
        # Synchronize state (context filters, token recount) before sending.
        # This is idempotent - if sync_state() was already called earlier (e.g., before request_status),
        # this will be a no-op with no wasted computation
        self.sync_state()
        self._prepare_provider_runtime_before_send(current_tick)
        self._before_send_message(current_tick)

        # --- Original send_message retry logic ---
        last_exception: Optional[Exception] = None
        last_empty_response: Optional[Tuple[str, Optional[str], Dict[str, Optional[int]]]] = None
        current_attempt = 0
        self._provider_send_retry_state = {}
        context_overflow_failures = 0
        context_overflow_max_attempts = max(
            1,
            int(getattr(constants, "LLM_CONTEXT_OVERFLOW_MAX_ATTEMPTS", 3)),
        )
        while current_attempt <= self.max_retries or context_overflow_failures < context_overflow_max_attempts:
            try:
                try:
                    if self._refresh_runtime_api_config_if_changed():
                        self._log(
                            "INFO",
                            f"Runtime API config refreshed before attempt {current_attempt + 1}."
                        )
                        self._before_send_message(current_tick)
                except Exception as refresh_error:
                    self._log(
                        "WARNING",
                        f"Runtime API config refresh failed: "
                        f"{self._format_retry_error_for_log(refresh_error)}",
                    )

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
                        self._log(
                            "WARNING",
                            f"Empty response received after {self.max_retries} retries. "
                            "Accepting empty response.",
                        )
                        return llm_response, thinking_response, token_info

                self._record_provider_success()
                return llm_response, thinking_response, token_info
            except LLMContextOverflowError as e:
                last_exception = e
                current_attempt += 1
                context_overflow_failures += 1
                if context_overflow_failures >= context_overflow_max_attempts:
                    self._log(
                        "ERROR",
                        f"Context overflow persisted after "
                        f"{context_overflow_failures} attempt(s). Pausing caller for manual review."
                    )
                    raise
                if self._handle_provider_send_error(e, current_tick):
                    continue

                retry_delay = self._calculate_retry_delay(context_overflow_failures)
                self._log(
                    "WARNING",
                    f"Context overflow detected "
                    f"(attempt {context_overflow_failures}/{context_overflow_max_attempts}): {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                continue
            except LLMCorruptedThoughtSignatureError:
                self._log(
                    "ERROR",
                    "Corrupted thought signature detected. "
                    "Pausing caller without provider fallback or retry."
                )
                raise
            except Exception as e:
                last_exception = e
                current_attempt += 1
                retry_immediately = False
                handled_by_provider = False
                try:
                    retry_immediately = self._handle_provider_send_error(e, current_tick)
                    handled_by_provider = retry_immediately
                    if not retry_immediately:
                        retry_immediately = self._handle_send_error(e, current_tick)
                except Exception as hook_error:
                    self._log(
                        "WARNING",
                        f"Send error hook failed: {self._format_retry_error_for_log(hook_error)}",
                    )

                if current_attempt > self.max_retries:
                    self._log(
                        "ERROR",
                        f"Max retries ({self.max_retries}) exhausted. Full error: "
                        f"{self._format_exception_for_log(e, include_raw=True, max_message_length=None)}",
                    )
                    raise
                if retry_immediately:
                    if not handled_by_provider:
                        self._log(
                            "INFO",
                            f"Retrying immediately after handled error "
                            f"(next_attempt={current_attempt + 1}/{self.max_retries}, "
                            f"error={self._format_retry_error_for_log(e)}).",
                        )
                    continue
                
                # Calculate incremental retry delay
                retry_delay = self._calculate_retry_delay(current_attempt)

                self._log(
                    "WARNING",
                    f"API error (attempt {current_attempt}/{self.max_retries}): "
                    f"{self._format_retry_error_for_log(e)}. Retrying in {retry_delay}s...",
                )
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
        Calculates and returns the total number of tokens for the current
        connector-visible chat session history as understood by the LLM.
        This should reflect the actual history that would be used for context.
        """
        pass

    def _can_count_current_session_tokens_authoritatively(self) -> bool:
        """
        Return False when get_current_total_session_tokens() would only return
        stale provider usage instead of recounting the rebuilt session.
        """
        return True

    def _mark_agent_token_budget_stale(self, reason: str) -> None:
        if not self.persist_to_disk:
            return
        agent_data = agent_module.load_agent_data(self.agent_name)
        if not agent_data:
            self._log("WARNING", "Could not load agent data to mark token budget stale.")
            return
        agent_data[constants.AGENT_TOKEN_BUDGET_CURRENT_STALE_KEY] = True
        agent_data[constants.AGENT_TOKEN_BUDGET_STALE_REASON_KEY] = reason
        agent_module.save_agent_data(self.agent_name, agent_data)

    def reload_session_from_disk(self) -> None:
        """
        Rebuild provider-specific in-memory chat state from the canonical history file.
        """
        self.agent_prune_blocks = self._load_prune_blocks_from_agent_data()
        self._last_known_prune_blocks = copy.deepcopy(self.agent_prune_blocks)
        self.context_history_start_tick = self._load_context_history_start_tick()
        self._last_known_context_history_start_tick = self.context_history_start_tick
        self.system_prompt = self._load_system_prompt_from_agent_data()
        self._last_known_system_prompt = self.system_prompt
        self._handle_system_prompt_update()
        self._initialize_chat_session()

    def end_session_and_cleanup(self) -> None:
        """Optional: Perform any cleanup."""
        print(f"LLMConnector for {self.agent_name}: Session ending. History saved to {self.history_file_path}")
        pass
