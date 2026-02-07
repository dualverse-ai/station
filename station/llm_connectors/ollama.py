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

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

from station import constants
from .base import (
    LLMConnectorError,
    LLMContextOverflowError,
    LLMPermanentAPIError,
    LLMSafetyBlockError,
    LLMTransientAPIError,
)
from .openai import OpenAIConnector


class OllamaConnector(OpenAIConnector):
    """
    OpenAI-compatible connector specialized for Ollama endpoints.
    Keeps Ollama-specific tool-calling behavior isolated from OpenAIConnector.
    """

    def __init__(
        self,
        model_name: str,
        agent_name: str,
        agent_data_path: str,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_output_tokens: Optional[int] = None,
        max_retries: int = constants.LLM_MAX_RETRIES,
        retry_delay_seconds: int = constants.LLM_RETRY_DELAY_SECONDS,
        custom_api_params: Optional[Dict[str, Any]] = None,
    ):
        params = dict(custom_api_params or {})
        base_url = (
            params.get("base_url")
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "http://127.0.0.1:11434/v1"
        )
        params["base_url"] = base_url.rstrip("/")
        self._enable_station_action_tools = bool(params.pop("enable_station_action_tools", False))

        super().__init__(
            model_name=model_name,
            agent_name=agent_name,
            agent_data_path=agent_data_path,
            api_key=api_key or os.getenv("OLLAMA_API_KEY") or "ollama",
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            custom_api_params=params,
        )

        # Rebind client to Ollama endpoint without modifying OpenAIConnector core.
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=params["base_url"])
            print(f"OllamaConnector for '{agent_name}' using base_url: {params['base_url']}")
        except Exception as e:
            raise LLMPermanentAPIError(f"Error creating Ollama client for {agent_name}: {e}.", original_exception=e)

    _SUPPORTED_ACTIONS = sorted(
        {
            value
            for name, value in vars(constants).items()
            if name.startswith("ACTION_") and isinstance(value, str)
        },
        key=len,
        reverse=True,
    )
    _GENERIC_ACTION_PATTERN = re.compile(
        r"\b(" + "|".join(re.escape(action) for action in _SUPPORTED_ACTIONS) + r")\b(?:\s+([^\n;`]+))?",
        re.IGNORECASE,
    )
    _EXECUTE_ACTION_PATTERN = re.compile(r"/execute_action\{[^}]+\}", re.IGNORECASE)

    def _tool_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "assistant",
                    "description": "Emit a Station execute action command.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "params": {"type": "string"},
                        },
                        "required": ["action"],
                    },
                },
            }
        ]

    def _normalize_action(self, action: str, params: str) -> str:
        a = (action or "").strip().lower()
        p = (params or "").strip()
        alias = {"navigate": "goto", "go_to": "goto", "open": "goto"}
        a = alias.get(a, a)
        cmd = a if not p else f"{a} {p}"
        return f"/execute_action{{{cmd}}}"

    def _extract_action_from_text(self, text: str) -> Optional[str]:
        if not text:
            return None
        explicit = self._EXECUTE_ACTION_PATTERN.findall(text)
        if explicit:
            return "\n".join(x.strip() for x in explicit)
        m = self._GENERIC_ACTION_PATTERN.search(text)
        if not m:
            return None
        action = m.group(1) or ""
        params = (m.group(2) or "").strip()
        return self._normalize_action(action, params)

    def _tool_call_to_execute_action(self, tool_call: Any) -> Optional[str]:
        try:
            func = getattr(tool_call, "function", None)
            if not func:
                return None
            func_name = getattr(func, "name", None)
            args_raw = getattr(func, "arguments", None)
            if not args_raw:
                return None
            payload = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            if not isinstance(payload, dict):
                return None

            if isinstance(func_name, str) and func_name.strip() == "assistant":
                action = str(payload.get("action") or "").strip()
                params = str(payload.get("params") or "").strip()
                if action:
                    return self._normalize_action(action, params)

            if isinstance(func_name, str) and func_name.strip() == "container.exec":
                cmd = payload.get("cmd")
                cmd_text = " ".join(str(x) for x in cmd) if isinstance(cmd, list) else str(cmd or "")
                mapped = self._extract_action_from_text(cmd_text)
                if mapped:
                    return mapped
                return "/execute_action{help lobby}"
        except Exception:
            return None
        return None

    def _extract_message_text_from_choice(self, choice: Any) -> Optional[str]:
        message = getattr(choice, "message", None)
        if not message:
            return None

        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            commands: List[str] = []
            for tc in tool_calls:
                cmd = self._tool_call_to_execute_action(tc)
                if cmd:
                    commands.append(cmd)
            if commands:
                synthesized = "\n".join(commands)
                print(
                    f"Info ({self.agent_name}): synthesized response from tool_calls "
                    f"({len(commands)} command(s))."
                )
                return synthesized
        return None

    def _send_message_with_chat_api(
        self,
        user_prompt: str,
        current_tick: int,
        token_info: Dict[str, Optional[int]],
        attempt_number: int = 0,
    ) -> Tuple[str, Optional[str], Dict[str, Optional[int]]]:
        # Ollama path uses non-streaming to keep tool-call behavior stable.
        history_saved = False
        self.chat_history.append({"role": "user", "content": user_prompt})

        try:
            api_params = {
                "model": self.model_name,
                "messages": self.chat_history,
                "temperature": self.temperature,
            }
            if self._enable_station_action_tools:
                api_params["tools"] = self._tool_schema()
                api_params["tool_choice"] = "auto"
            if self.max_output_tokens:
                api_params["max_tokens"] = self.max_output_tokens

            response: ChatCompletion = self.client.chat.completions.create(**api_params)

            if not response.choices:
                raise LLMSafetyBlockError(
                    f"LLM response generation failed for {self.agent_name}. No choices returned.",
                    block_reason="no_choices",
                )

            llm_text_response = self._extract_message_text_from_choice(response.choices[0])
            if not llm_text_response:
                raw_response = None
                try:
                    raw_response = response.model_dump(exclude_none=True)
                except Exception as dump_err:
                    raw_response = f"dump_error={dump_err}; repr={response!r}"
                print(f"Debug ({self.agent_name}): empty message content from model '{self.model_name}'. Raw response: {json.dumps(raw_response)[:2000]}")
                raise LLMSafetyBlockError(
                    f"LLM response generation failed for {self.agent_name}. Empty message content.",
                    block_reason="empty_content",
                )

            thinking_text = None
            self.chat_history.append({"role": "assistant", "content": llm_text_response})

            if not history_saved:
                self._append_turn_to_history_file(current_tick, "user", user_prompt, None, None)
                self._append_turn_to_history_file(current_tick, "model", llm_text_response, thinking_text, token_info)
                history_saved = True

            if response.usage:
                token_info["last_exchange_prompt_tokens"] = response.usage.prompt_tokens
                token_info["last_exchange_completion_tokens"] = response.usage.completion_tokens
                token_info["total_tokens_in_session"] = response.usage.total_tokens
                token_info["last_exchange_cached_tokens"] = None
                token_info["last_exchange_thoughts_tokens"] = None

            if token_info["total_tokens_in_session"] is None:
                tiktoken_count = self._count_tokens_with_tiktoken(self.chat_history)
                if tiktoken_count is not None:
                    token_info["total_tokens_in_session"] = tiktoken_count

            return llm_text_response, thinking_text, token_info

        except openai.APIConnectionError as e:
            self._rollback_last_user_turn()
            raise LLMTransientAPIError(f"OpenAI API Connection Error for {self.agent_name}: {str(e)}", original_exception=e)
        except openai.RateLimitError as e:
            self._rollback_last_user_turn()
            raise LLMTransientAPIError(f"OpenAI API Rate Limit Error for {self.agent_name}: {str(e)}", original_exception=e)
        except openai.APIStatusError as e:
            self._rollback_last_user_turn()
            if self._is_context_overflow_error(e):
                raise LLMContextOverflowError(f"Context window overflow for {self.agent_name}: {str(e)}", original_exception=e)
            if e.status_code >= 500:
                raise LLMTransientAPIError(f"OpenAI API Server Error for {self.agent_name}: {str(e)}", original_exception=e)
            raise LLMPermanentAPIError(f"OpenAI API Client Error for {self.agent_name}: {str(e)}", original_exception=e)
        except Exception as e:
            self._rollback_last_user_turn()
            raise LLMConnectorError(f"Unexpected OpenAI API call failure for {self.agent_name}. Details: {str(e)}", original_exception=e)

    def _rollback_last_user_turn(self) -> None:
        if self.chat_history and self.chat_history[-1].get("role") == "user":
            self.chat_history.pop()
