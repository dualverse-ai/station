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

from typing import Any, Optional
from .base import BaseLLMConnector
from .gemini import GoogleGeminiConnector
from .claude import ClaudeConnector
from .openai import OpenAIConnector
from .grok import GrokConnector


def create_llm_connector(
    model_class_name: str,
    model_name: str,
    agent_name: str,
    agent_data_path: str,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    max_output_tokens: Optional[int] = None,
    custom_api_params: Optional[dict] = None,
    **kwargs: Any
) -> Optional[BaseLLMConnector]:
    """
    Factory function to create LLM connectors based on model class name.

    Args:
        model_class_name: The type of LLM connector ("gemini", "claude", "openai", "grok")
        model_name: The specific model to use (e.g. "gpt-4o", "o4-mini", "claude-3-5-sonnet", "grok-4")
        agent_name: Name of the agent using this connector
        agent_data_path: Path to agent's data directory for history storage
        api_key: API key for the service (optional, can use env vars)
        system_prompt: System prompt for the model
        temperature: Temperature setting for generation
        max_output_tokens: Maximum tokens to generate
        custom_api_params: Provider-specific custom API parameters dict
        **kwargs: Additional arguments passed to specific connectors

    Returns:
        BaseLLMConnector instance or None if model class not supported
    """
    normalized_class_name = model_class_name.lower()

    if normalized_class_name == "gemini":
        return GoogleGeminiConnector(
            model_name=model_name,
            agent_name=agent_name,
            agent_data_path=agent_data_path,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    elif normalized_class_name == "claude":
        return ClaudeConnector(
            model_name=model_name,
            agent_name=agent_name,
            agent_data_path=agent_data_path,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    elif normalized_class_name == "openai":
        return OpenAIConnector(
            model_name=model_name,
            agent_name=agent_name,
            agent_data_path=agent_data_path,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            custom_api_params=custom_api_params,
        )
    elif normalized_class_name == "grok":
        return GrokConnector(
            model_name=model_name,
            agent_name=agent_name,
            agent_data_path=agent_data_path,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    else:
        print(f"Warning: LLM connector for model class '{model_class_name}' is not supported.")
        return None