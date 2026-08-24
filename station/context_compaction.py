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

"""Helpers for agent context compaction and restart anchors."""

from __future__ import annotations

from typing import Any, Dict, List

from station import constants


def normalize_summary_response(response_text: Any) -> str:
    """Treat the whole model response as the compaction summary."""
    if response_text is None:
        return ""
    return str(response_text).strip()


def format_protected_messages(items: List[Dict[str, Any]]) -> str:
    """Render protected context items for the first prompt after compaction."""
    if not items:
        return "None"

    blocks: List[str] = []
    for index, item in enumerate(items, start=1):
        tick = item.get(constants.PROTECTED_CONTEXT_TICK_KEY, "unknown")
        title = str(
            item.get(constants.PROTECTED_CONTEXT_TITLE_KEY)
            or item.get(constants.PROTECTED_CONTEXT_KIND_KEY)
            or "Protected Message"
        ).strip()
        content = str(item.get(constants.PROTECTED_CONTEXT_CONTENT_KEY) or "").strip()
        if not content:
            continue
        header = f"## Message {index} (Station Tick {tick})"
        if title:
            header += f": {title}"
        blocks.append(f"{header}\n\n{content}")

    return "\n\n---\n\n".join(blocks) if blocks else "None"


def build_compaction_anchor_intro(
    *,
    agent_name: str,
    protected_items: List[Dict[str, Any]],
    summary: str,
) -> str:
    protected_messages = format_protected_messages(protected_items)
    return constants.CONTEXT_COMPACTION_INTRO_TEMPLATE.format(
        agent_name=agent_name,
        protected_messages=protected_messages,
        summary=str(summary or "").strip(),
    ).strip()
