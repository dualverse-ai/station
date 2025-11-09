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

"""
LLM Connectors Package

This package provides a modular system for connecting to different LLM APIs.
Each connector inherits from BaseLLMConnector and implements the specific API protocol.

Usage:
    from station.llm_connectors import create_llm_connector
    
    connector = create_llm_connector(
        model_class_name="gemini",  # or "claude", "openai"
        model_name="gemini-1.5-flash-latest",
        agent_name="MyAgent",
        agent_data_path="/path/to/agent/data"
    )
"""

from .base import (
    BaseLLMConnector,
    LLMConnectorError,
    LLMTransientAPIError,
    LLMPermanentAPIError,
    LLMSafetyBlockError,
    LLMContextOverflowError
)
from .factory import create_llm_connector
from .gemini import GoogleGeminiConnector
from .claude import ClaudeConnector
from .openai import OpenAIConnector

__all__ = [
    # Base classes and exceptions
    'BaseLLMConnector',
    'LLMConnectorError',
    'LLMTransientAPIError', 
    'LLMPermanentAPIError',
    'LLMSafetyBlockError',
    'LLMContextOverflowError',
    
    # Factory function
    'create_llm_connector',
    
    # Specific connectors
    'GoogleGeminiConnector',
    'ClaudeConnector',
    'OpenAIConnector',
]