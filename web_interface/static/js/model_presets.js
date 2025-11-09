/*
 * Copyright 2025 DualverseAI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// web_interface/static/js/model_presets.js
const MODEL_PRESETS = [
    {
        "display_name": "Gemini 2.5 Pro (300k)",
        "model_provider_class": "Gemini",
        "model_name": "gemini-2.5-pro",
        "initial_tokens_max": 300000,
        "llm_system_prompt": "",
    },
    {
        "display_name": "Gemini 2.5 Flash (600k)",
        "model_provider_class": "Gemini",
        "model_name": "gemini-2.5-flash",
        "initial_tokens_max": 600000,
        "llm_system_prompt": "",
    },
    {
        "display_name": "Claude Opus 4.1 (200k)",
        "model_provider_class": "Claude",
        "model_name": "claude-opus-4-1-20250805",
        "initial_tokens_max": 200000,
        "llm_system_prompt": ""
    },
    {
        "display_name": "Grok 4 (200k)",
        "model_provider_class": "Grok",
        "model_name": "grok-4",
        "initial_tokens_max": 200000,
        "llm_system_prompt": ""
    },
    {
        "display_name": "GPT-5 (270k)",
        "model_provider_class": "OpenAI",
        "model_name": "gpt-5",
        "initial_tokens_max": 270000,
        "llm_system_prompt": ""
    },
];
