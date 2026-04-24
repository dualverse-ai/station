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
from typing import Any, Dict, Optional

from openai import OpenAI

from station import constants
from .base import LLMPermanentAPIError
from .openai import OpenAIConnector


class OllamaConnector(OpenAIConnector):
    """
    OpenAI-compatible connector specialized for Ollama endpoints.
    Uses OpenAIConnector behavior and only customizes endpoint/auth defaults.
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
        params.setdefault("prompt_cache_retention", None)

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
