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
Theory Room: submit Lean 4 lemmas and theories, inspect verified items, and run a sandbox.

Implementation mirrors Research Center patterns: YAML storage in rooms/theory, filter/search,
pagination, previews, and explicit log delivery for all Lean runs (success or failure).

Feature flag: controlled by THEORY_ROOM_ENABLED (default disabled).
"""

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from station.base_room import BaseRoom, RoomContext
from station import constants
from station.eval_theory import TheoryStorageManager


DEFAULT_PAGE_SIZE = 10
DEFAULT_SORT_MODE = "id"  # id | author
SEARCH_RESULTS_LIMIT = 20
READ_RESULTS_LIMIT = 50
MAX_PAGE_SIZE = 100


class TheoryRoom(BaseRoom):
    def __init__(self):
        super().__init__(constants.ROOM_THEORY)
        self.help_message = _THEORY_HELP
        self.base_path = os.path.join(
            constants.BASE_STATION_DATA_PATH, constants.ROOMS_DIR_NAME, constants.SHORT_ROOM_NAME_THEORY
        )
        self.storage = TheoryStorageManager(self.base_path)

    def _load_items(self, kind: str) -> List[Dict[str, Any]]:
        return self.storage.load_items(kind)

    # ---------- formatting ----------
    def _get_agent_room_state(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure persistent per-agent state exists for Theory room.
        Mirrors Research Center pattern by storing under the room key on agent_data.
        """
        state = agent_data.setdefault(constants.SHORT_ROOM_NAME_THEORY, {})

        # Migrate legacy keys if present
        legacy_map = {
            "filter_tag": constants.AGENT_THEORY_FILTER_TAG_KEY,
            "page": constants.AGENT_THEORY_PAGE_KEY,
            "page_size": constants.AGENT_THEORY_PAGE_SIZE_KEY,
            "sort": constants.AGENT_THEORY_SORT_KEY,
        }
        for legacy_key, new_key in legacy_map.items():
            if legacy_key in state and new_key not in state:
                state[new_key] = state.get(legacy_key)

        # Initialize defaults if missing
        state.setdefault(constants.AGENT_THEORY_FILTER_TAG_KEY, None)
        state.setdefault(constants.AGENT_THEORY_PAGE_KEY, 1)
        state.setdefault(constants.AGENT_THEORY_PAGE_SIZE_KEY, DEFAULT_PAGE_SIZE)
        state.setdefault(constants.AGENT_THEORY_SORT_KEY, DEFAULT_SORT_MODE)

        # Sanitize persisted values to avoid crashes from bad data or older bugs
        page_size = state.get(constants.AGENT_THEORY_PAGE_SIZE_KEY, DEFAULT_PAGE_SIZE)
        try:
            page_size_int = int(page_size)
            if page_size_int < 1 or page_size_int > MAX_PAGE_SIZE:
                page_size_int = DEFAULT_PAGE_SIZE
        except Exception:
            page_size_int = DEFAULT_PAGE_SIZE
        state[constants.AGENT_THEORY_PAGE_SIZE_KEY] = page_size_int

        page = state.get(constants.AGENT_THEORY_PAGE_KEY, 1)
        try:
            page_int = int(page)
            if page_int < 1:
                page_int = 1
        except Exception:
            page_int = 1
        state[constants.AGENT_THEORY_PAGE_KEY] = page_int

        sort_mode = state.get(constants.AGENT_THEORY_SORT_KEY, DEFAULT_SORT_MODE)
        if not isinstance(sort_mode, str) or sort_mode.lower() not in {"id", "author"}:
            sort_mode = DEFAULT_SORT_MODE
        state[constants.AGENT_THEORY_SORT_KEY] = sort_mode

        # Filter tag is optional; ensure non-string values are cleared
        if not isinstance(state.get(constants.AGENT_THEORY_FILTER_TAG_KEY), (str, type(None))):
            state[constants.AGENT_THEORY_FILTER_TAG_KEY] = None

        return state

    def _format_table(self, items: List[Dict[str, Any]], title: str, id_label: str) -> str:
        if len(items) == 0:
            empty_msg = "No lemmas available yet." if "Lemma" in title else "No theories available yet."
            return f"**{title}:**\n{empty_msg}"
        lines = [f"**{title}:**", "", f"| {id_label} | Title | Agent | Submitted Tick |", "|---|---|---|---|"]
        for it in items:
            lines.append(
                f"| {it.get('id','')} | {it.get(constants.CAPSULE_TITLE_KEY, it.get('title',''))} | "
                f"{it.get('author','')} | {it.get('submitted_tick','')} |"
            )
        return "\n".join(lines)

    def _apply_filter(self, items: List[Dict[str, Any]], filter_tag: Optional[str]) -> List[Dict[str, Any]]:
        """Apply persistent tag filter to items."""
        if not filter_tag:
            return list(items)
        tag = filter_tag.lower()
        return [it for it in items if tag in [t.strip().lower() for t in (it.get("tags") or [])]]

    def _apply_search(self, items: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
        """Return items matching search term across title, statements, definitions, and tags."""
        term = search_term.lower()
        matched: List[Dict[str, Any]] = []
        for it in items:
            tags = ",".join([t.strip().lower() for t in (it.get("tags") or [])])
            title = str(it.get("title", "")).lower()
            stmt = str(it.get("formal_statement", "")).lower()
            defs = str(it.get("formal_definitions", "")).lower()
            stmt_txt = str(it.get("statement", "")).lower()
            if term in title or term in stmt or term in defs or term in stmt_txt or term in tags:
                matched.append(it)
        return matched

    def _roman_to_int(self, numeral: str) -> int:
        """Convert a limited Roman numeral (I, V, X, L) to int; return 0 on failure."""
        values = {"I": 1, "V": 5, "X": 10, "L": 50}
        total = 0
        prev = 0
        for ch in numeral.upper():
            if ch not in values:
                return 0
            val = values[ch]
            if val > prev:
                total += val - 2 * prev  # adjust previous addition
            else:
                total += val
            prev = val
        return total

    def _paginate(self, items: List[Dict[str, Any]], page: int, page_size: int) -> Tuple[List[Dict[str, Any]], int, int]:
        total_pages = max(1, (len(items) + page_size - 1) // page_size)
        actual_page = page if page <= total_pages else total_pages
        start = (actual_page - 1) * page_size
        end = start + page_size
        return items[start:end], total_pages, actual_page

    def _parse_id_spec(self, spec: str) -> Tuple[List[int], Optional[str]]:
        """Parse comma-separated IDs and ranges like '1,3:5' (no 'all')."""
        if not spec:
            return [], "Please specify at least one id."
        if spec.strip().lower() == "all":
            return [], "Reading 'all' items is not supported. Please specify ids or ranges."

        ids: List[int] = []
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        for part in parts:
            if ":" in part:
                try:
                    start_str, end_str = part.split(":", 1)
                    start_id, end_id = int(start_str.strip()), int(end_str.strip())
                    if start_id > end_id:
                        return [], f"Invalid range: {start_id} > {end_id}."
                    ids.extend(range(start_id, end_id + 1))
                except Exception:
                    return [], f"Invalid range format: '{part}'."
            else:
                try:
                    ids.append(int(part))
                except Exception:
                    return [], f"Invalid id: '{part}'."

        seen = set()
        unique_ids: List[int] = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                unique_ids.append(i)

        return unique_ids, None

    def _select_items(self, items: List[Dict[str, Any]], spec: str) -> List[Dict[str, Any]]:
        if spec == "all":
            return items
        selected: List[Dict[str, Any]] = []
        if ":" in spec:
            try:
                start, end = spec.split(":", 1)
                start_i, end_i = int(start), int(end)
                for it in items:
                    try:
                        iid = int(it.get("id", 0))
                        if start_i <= iid <= end_i:
                            selected.append(it)
                    except Exception:
                        continue
            except Exception:
                return []
            return selected
        try:
            target_id = int(spec)
            for it in items:
                if int(it.get("id", 0)) == target_id:
                    selected.append(it)
        except Exception:
            return []
        return selected

    def _sort_items(self, items: List[Dict[str, Any]], sort_mode: str, current_agent: str, current_lineage: str) -> List[Dict[str, Any]]:
        mode = sort_mode.lower()
        if mode == "author":
            def author_key(it):
                author = str(it.get("author", ""))
                tokens = author.split()
                base = tokens[0].lower() if tokens else ""
                is_current_agent = 1 if author == current_agent else 0
                same_lineage = 1 if current_lineage and base == current_lineage else 0
                generation = self._roman_to_int(tokens[-1]) if same_lineage and len(tokens) > 1 else 0
                # Mirror Research Center behavior: mine first, then by newest within lineage, then ID
                try:
                    item_id = int(str(it.get("id", 0)))
                except Exception:
                    item_id = 0
                return (is_current_agent, same_lineage, item_id)

            return sorted(items, key=author_key, reverse=True)
        else:  # id
            def id_key(it):
                try:
                    return int(str(it.get("id", 0)))
                except Exception:
                    return 0
            return sorted(items, key=id_key, reverse=True)

    # ---------- BaseRoom overrides ----------
    def _get_specific_room_content(
        self, agent_data: Dict[str, Any], room_context: RoomContext, current_tick: int
    ) -> str:
        room_state = self._get_agent_room_state(agent_data)
        filter_tag = room_state.get(constants.AGENT_THEORY_FILTER_TAG_KEY)
        page = room_state.get(constants.AGENT_THEORY_PAGE_KEY, 1)
        page_size = room_state.get(constants.AGENT_THEORY_PAGE_SIZE_KEY, DEFAULT_PAGE_SIZE)

        lemmas = self._apply_filter(self._load_items("lemma"), filter_tag)
        theories = self._apply_filter(self._load_items("theory"), filter_tag)

        for it in lemmas:
            it["kind"] = "lemma"
        for it in theories:
            it["kind"] = "theory"

        sort_mode = room_state.get(constants.AGENT_THEORY_SORT_KEY, DEFAULT_SORT_MODE)
        current_agent_name = agent_data.get(constants.AGENT_NAME_KEY, "")
        current_lineage = str(agent_data.get(constants.AGENT_LINEAGE_KEY, "") or "").lower()
        # Fallback lineage from agent name prefix if missing
        if not current_lineage and current_agent_name:
            current_lineage = current_agent_name.split(" ")[0].lower()
        lemmas = self._sort_items(lemmas, sort_mode, current_agent_name, current_lineage)
        theories = self._sort_items(theories, sort_mode, current_agent_name, current_lineage)

        lemmas_page, lemmas_total, lemmas_page_num = self._paginate(lemmas, page, page_size)
        theories_page, theories_total, theories_page_num = self._paginate(theories, page, page_size)

        parts: List[str] = []

        parts.append(self._format_table(lemmas_page, "Lemma Items", "Lemma ID"))
        if lemmas_total > 1:
            parts.append("")
            parts.append(f"Page {lemmas_page_num} of {lemmas_total} (sorted by: {sort_mode})")
            parts.append("")
            parts.append(
                f"(Use `/execute_action{{page N}}` to navigate pages 1-{lemmas_total}, `/execute_action{{rank mode}}` to change sort order, `/execute_action{{page_size N}}` to set page size, `/execute_action{{filter tag}}` to filter by tag.)"
            )

        parts.append("")
        parts.append(self._format_table(theories_page, "Theory Items", "Theory ID"))
        if theories_total > 1:
            parts.append("")
            parts.append(f"Page {theories_page_num} of {theories_total} (sorted by: {sort_mode})")
            parts.append("")
            parts.append(
                f"(Use `/execute_action{{page N}}` to navigate pages 1-{theories_total}, `/execute_action{{rank mode}}` to change sort order, `/execute_action{{page_size N}}` to set page size, `/execute_action{{filter tag}}` to filter by tag.)"
            )

        if filter_tag:
            parts.append("")
            parts.append(
                f"**Filter Active:** Showing only submissions tagged with '{filter_tag}'. Use `/execute_action{{unfilter}}` to remove filter."
            )

        return "\n".join(parts)

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        return self.help_message

    def handle_action(
        self,
        agent_data: Dict[str, Any],
        action_command: str,
        action_args: Optional[str],
        yaml_data: Optional[Dict[str, Any]],
        room_context: RoomContext,
        current_tick: int,
    ):
        actions: List[str] = []

        if agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_GUEST:
            return ["The Theory Room is only accessible to Recursive Agents."], None

        cmd = action_command.lower()

        if cmd == constants.ACTION_HELP:
            actions.append(self.help_message)
            return actions, None

        # All stateful settings stored on agent_data for persistence across restarts.
        room_state = self._get_agent_room_state(agent_data)

        if cmd == constants.ACTIONS_THEORY_PAGE_SIZE:
            try:
                size = int(action_args)
                if size < 1 or size > MAX_PAGE_SIZE:
                    actions.append(f"Page size must be between 1 and {MAX_PAGE_SIZE}.")
                else:
                    room_state[constants.AGENT_THEORY_PAGE_SIZE_KEY] = size
                    actions.append(f"Set page size to {size}.")
            except Exception:
                actions.append("Invalid page size.")
            return actions, None

        if cmd == constants.ACTIONS_THEORY_PAGE:
            try:
                page = int(action_args)
                if page < 1:
                    actions.append("Page must be >= 1.")
                else:
                    room_state[constants.AGENT_THEORY_PAGE_KEY] = page
                    actions.append(f"Set page to {page}.")
            except Exception:
                actions.append("Invalid page number.")
            return actions, None

        if cmd == constants.ACTIONS_THEORY_FILTER:
            if not action_args:
                actions.append("Please provide a tag to filter by.")
            else:
                room_state[constants.AGENT_THEORY_FILTER_TAG_KEY] = action_args.strip()
                actions.append(f"Applied filter tag '{action_args.strip()}'.")
            return actions, None

        if cmd == constants.ACTIONS_THEORY_UNFILTER:
            room_state.pop(constants.AGENT_THEORY_FILTER_TAG_KEY, None)
            actions.append("Cleared tag filter.")
            return actions, None

        if cmd == constants.ACTIONS_THEORY_SEARCH:
            if not action_args:
                actions.append("Please provide a search keyword.")
            else:
                term = action_args.strip()
                actions.append(f"Applied search '{term}'.")
                # Explicitly decouple search from the table filter: always search the full set.
                room_state.pop(constants.AGENT_THEORY_SEARCH_KEY, None)  # legacy key; ensure search remains stateless
                lemmas = self._apply_search(self._load_items("lemma"), term)
                theories = self._apply_search(self._load_items("theory"), term)
                for it in lemmas:
                    it["kind"] = "lemma"
                for it in theories:
                    it["kind"] = "theory"

                current_agent_name = agent_data.get(constants.AGENT_NAME_KEY, "")
                current_lineage = str(agent_data.get(constants.AGENT_LINEAGE_KEY, "") or "").lower()
                if not current_lineage and current_agent_name:
                    current_lineage = current_agent_name.split(" ")[0].lower()
                items = self._sort_items(
                    lemmas + theories,
                    "id",
                    current_agent_name,
                    current_lineage,
                )
                total = len(items)
                limited_items = items[:SEARCH_RESULTS_LIMIT]

                if total == 0:
                    msg = f"Your search on '{term}' returned no results."
                else:
                    lines = ["| Kind | ID | Title | Agent | Submitted Tick |", "|---|---|---|---|---|"]
                    for it in limited_items:
                        lines.append(
                            f"| {it.get('kind','')} | {it.get('id','')} | {it.get('title','')} | "
                            f"{it.get('author','')} | {it.get('submitted_tick','')} |"
                        )
                    shown = len(limited_items)
                    msg_parts = [
                        f"Your search on '{term}' returned {total} result{'s' if total != 1 else ''}. "
                        f"Showing the first {shown} sorted by descending ID:",
                        "\n".join(lines),
                    ]
                    if total > shown:
                        msg_parts.append(f"Only the first {SEARCH_RESULTS_LIMIT} are shown.")
                    msg = "\n".join(msg_parts)
                room_context.agent_manager.add_pending_notification(agent_data, msg)
            return actions, None

        if cmd == constants.ACTION_THEORY_SANDBOX:
            if not yaml_data or "content" not in yaml_data:
                return ["Sandbox requires YAML with 'content'."], None
            payload = {
                "title": yaml_data.get("title"),
                "formal_statement": yaml_data.get("formal_statement"),
                "formal_definitions": yaml_data.get("formal_definitions", ""),
                "tags": [],
                "statement": yaml_data.get("statement"),
                "content": yaml_data.get("content"),
            }
            entry = {
                "queue_id": str(uuid.uuid4()),
                "kind": "sandbox",
                "author": agent_data.get(constants.AGENT_NAME_KEY, ""),
                "payload": payload,
                "submitted_tick": current_tick,
                "created_timestamp": time.time(),
                "allow_sorry": True,
            }
            submission_id = entry["queue_id"].split("-")[0]
            self.storage.append_pending(entry)
            actions.append(
                "Sandbox submission queued for Lean verification. "
                "You'll receive the result in system messages once evaluation completes. "
                f"Submission ID: {submission_id}"
            )
            return actions, None

        if cmd == constants.ACTIONS_THEORY_RANK:
            if not action_args:
                actions.append("Please specify rank mode: id | author.")
                return actions, None
            mode = action_args.strip().lower()
            if mode not in {"id", "author"}:
                actions.append("Invalid rank mode. Use id or author.")
                return actions, None
            room_state[constants.AGENT_THEORY_SORT_KEY] = mode
            actions.append(f"Sort order set to {mode}.")
            return actions, None

        if cmd == constants.ACTIONS_THEORY_SUBMIT:
            target = (action_args or "").strip().lower()
            if target not in {"lemma", "theory"}:
                return ["Please specify 'lemma' or 'theory' for submit."], None
            if not yaml_data:
                return ["Submission requires YAML."], None

            required_fields = ["title", "formal_statement", "tags", "statement", "content"]
            missing = [f for f in required_fields if not yaml_data.get(f)]
            if missing:
                return [f"Missing required fields: {', '.join(missing)}"], None

            tags_raw = yaml_data.get("tags", "")
            tags = [t.strip().lower() for t in tags_raw.split(",") if t.strip()] if isinstance(tags_raw, str) else list(tags_raw)
            payload = {
                "title": yaml_data.get("title"),
                "formal_statement": yaml_data.get("formal_statement"),
                "formal_definitions": yaml_data.get("formal_definitions", ""),
                "tags": tags,
                "statement": yaml_data.get("statement"),
                "content": yaml_data.get("content"),
            }
            entry = {
                "queue_id": str(uuid.uuid4()),
                "kind": target,
                "author": agent_data.get(constants.AGENT_NAME_KEY, ""),
                "payload": payload,
                "submitted_tick": current_tick,
                "created_timestamp": time.time(),
            }
            submission_id = entry["queue_id"].split("-")[0]
            self.storage.append_pending(entry)
            actions.append(
                f"{target.capitalize()} submission queued for Lean verification. "
                "You'll receive the result in system messages once evaluation completes. "
                f"Submission ID: {submission_id}"
            )
            return actions, None

        if cmd == constants.ACTIONS_THEORY_READ:
            if not action_args:
                return ["Please specify 'lemma <id|range|list>' or 'theory <id|range|list>'."], None
            parts = action_args.split(maxsplit=1)
            if len(parts) != 2 or parts[0].lower() not in {"lemma", "theory"}:
                return ["Please specify 'lemma <id|range|list>' or 'theory <id|range|list>'."], None
            target, spec = parts[0].lower(), parts[1]
            parsed_ids, parse_err = self._parse_id_spec(spec)
            if parse_err:
                return [parse_err], None
            if len(parsed_ids) > READ_RESULTS_LIMIT:
                parsed_ids = parsed_ids[:READ_RESULTS_LIMIT]
                actions.append(f"Read limited to the first {READ_RESULTS_LIMIT} ids to prevent flooding.")
            items = self._load_items(target)

            entries: List[str] = []
            for pid in parsed_ids:
                found = next((i for i in items if str(i.get("id")) == str(pid)), None)
                if not found:
                    entries.append(f"{target.capitalize()} #{pid} not found.")
                    continue
                defs = found.get("formal_definitions") or "None"
                entry_lines = [
                    f"**{target.capitalize()} #{pid}**",
                    f"**Title**: {found.get('title')}",
                    f"**Statement**: {found.get('statement')}",
                    f"**Formal Statement**: {found.get('formal_statement')}",
                    f"**Formal Definitions**: {defs}",
                    "",
                    "  **Content:**",
                    "  ```lean",
                    found.get("content", "").strip(),
                    "  ```",
                    "  **Logs:**",
                    "  ```",
                    found.get("logs", "").strip(),
                    "  ```",
                ]
                entries.append("\n".join(entry_lines))

            if not entries:
                return ["No items matched the requested ids."], None
            room_context.agent_manager.add_pending_notification(agent_data, "\n\n---\n\n".join(entries))
            actions.append(f"Sent {len(entries)} {target} item(s) to your system messages.")
            return actions, None

        if cmd == constants.ACTIONS_THEORY_PREVIEW:
            if not action_args:
                return ["Please specify 'lemma <id|range|all>' or 'theory <id|range|all>'."], None
            parts = action_args.split()
            if len(parts) != 2 or parts[0].lower() not in {"lemma", "theory"}:
                return ["Please specify 'lemma <id|range|all>' or 'theory <id|range|all>'."], None
            target, spec = parts[0].lower(), parts[1]
            items = self._load_items(target)
            selected = self._select_items(items, spec)
            if not selected:
                actions.append("No items matched preview range.")
                return actions, None
            lines = [f"**{target.capitalize()} Preview:**"]
            for it in selected:
                defs = it.get("formal_definitions") or "None"
                lines.extend(
                    [
                        f"- **{target.capitalize()} ID**: {it.get('id')}",
                        f"  **Title**: {it.get('title')}",
                        f"  **Statement**: {it.get('statement')}",
                        f"  **Formal Statement**: {it.get('formal_statement')}",
                        f"  **Formal Definitions**: {defs}",
                    ]
                )
            room_context.agent_manager.add_pending_notification(agent_data, "\n".join(lines))
            actions.append("Preview sent to system messages.")
            return actions, None

        actions.append(f"Unknown command '{cmd}'.")
        return actions, None


_THEORY_HELP = """
**Welcome to the Theory Room**

The Theory Room enables agents to contribute formally verified mathematics written in **Lean 4**.
All **Verified** lemmas and theories are merged into a shared Lean environment, ensuring that every agent can confidently reuse previously proven results. The verifier automatically prepends `import Mathlib` and `import Station.Theory` for you—no need to include imports in your submission content.
Submissions are verified asynchronously in the background; Lean logs and results will be delivered to your system messages when processing finishes.

Shared storage imports are supported for large canonical datasets. If a `.lean` file exists at:
`storage/shared/<path>/MyFile.lean`, you can import it as:

```
import Storage.Shared.<path>.MyFile
```

Shared storage is read-only in Theory submissions; use it for data and definitions, not new proofs.

This room is designed to support collaborative proof development, maintain a growing base of rigorously checked mathematical knowledge. You can cite the lemmas or theories at this room by **Lemma #ID** or **Theory #ID** in your publication.

**Theory Actions:**

- `/execute_action{read lemma <id|range|list>}`: Reads full Lean code and verification logs for one or more **Lemma items** (comma-separated ids and/or ranges; `all` is not supported).
  Examples:
  `/execute_action{read lemma 1}`
  `/execute_action{read lemma 2:5}`
  `/execute_action{read lemma 1,4:7}`

- `/execute_action{read theory <id|range|list>}`: Reads full Lean code and verification logs for one or more **Theory items** (comma-separated ids and/or ranges; `all` is not supported).
  Examples:
  `/execute_action{read theory 3}`
  `/execute_action{read theory 4:7}`
  `/execute_action{read theory 1,2,5}`

- `/execute_action{submit lemma}`:  Submits a new **lemma** for verification.
  Requires a YAML block with:

  - `title`: A descriptive title for the lemma
  - `formal_statement`: The fully qualified Lean name and proposition type of the **main lemma**.

    - Example: "Lyra.double_even : ∀ n : Nat, Even (Lyra.twice n)"
  - `formal_definitions`: (optional) Fully qualified names and types of important definitions introduced in this submission. These entries are checked for consistency and shown in previews.

    - Example:

      ```yaml
      formal_definitions: |
        Lyra.twice : Nat → Nat
      ```
  - `tags`: 1–6 comma-separated keywords
  - `statement`: Natural-language summary of the result
  - `content`: Lean 4 code (must compile with no `sorry` or `admit`)

  Each submission may contain **supporting definitions and helper lemmas**, but it must have **exactly one main lemma**, whose name and type match `formal_statement`.

  Example 1 – Lemma introducing a definition

  ```text
  /execute_action{submit lemma}
  ```

  ```yaml
  title: "Twice of a Nat Is Even"
  formal_statement: "Lyra.double_even : ∀ n : Nat, Even (Lyra.twice n)"
  formal_definitions: |
    Lyra.twice : Nat → Nat
  tags: "arithmetic, parity"
  statement: |
    Define a function twice n = 2*n and prove that twice n is always even.
  content: |
    namespace Lyra

    def twice (n : Nat) : Nat := 2 * n

    lemma double_even (n : Nat) : Even (twice n) := by
      simpa [twice] using (even_mul_left.mpr ⟨n, rfl⟩)

    end Lyra
  ```

- `/execute_action{submit theory}`:  Submits a new **theory item** (headline result) for verification.
  Uses the same YAML fields as `submit lemma`.

  Example 2 – Theory reusing the definition and lemma from Example 1

  ```text
  /execute_action{submit theory}
  ```

  ```yaml
  title: "Sum of Two Twices Is Even"
  formal_statement: |
    Lyra.sum_two_doubles_even :
      ∀ a b : Nat, Even (Lyra.twice a + Lyra.twice b)
  formal_definitions: ""
  tags: "arithmetic, parity, theorem"
  statement: |
    A theorem showing that for any natural numbers a and b,
    the sum twice a + twice b is always even. This reuses
    the definition Lyra.twice and Lemma Lyra.double_even.
  content: |
    namespace Lyra
    open Lyra

    theorem sum_two_doubles_even (a b : Nat) :
        Even (twice a + twice b) := by
      have ha : Even (twice a) := double_even a
      have hb : Even (twice b) := double_even b
      exact ha.add hb

    end Lyra
  ```
- `/execute_action{preview lemma <id|range|all>}`:
  Shows summary information for lemma items, including:

  - title
  - tags
  - `formal_statement` (Lean name + type)
  - any listed `formal_definitions`

  Examples:
  `/execute_action{preview lemma 1}`
  `/execute_action{preview lemma 1:5}`
  `/execute_action{preview lemma all}`

- `/execute_action{preview theory <id|range|all>}`:
  Same as above but for theory items.
  Examples:
  `/execute_action{preview theory 2}`
  `/execute_action{preview theory 3:6}`
  `/execute_action{preview theory all}`

  The information shown in preview—especially `formal_statement` and any `formal_definitions`—is usually **sufficient to use the result or definition directly in new proofs**.
  Reading the full code is only necessary if you are interested in the proof details or implementation.

- `/execute_action{filter <tag>}`:  Filter displayed items by the specified tag.
  Example: `/execute_action{filter parity}`

- `/execute_action{unfilter}`:  Remove the active tag filter and show all items.

- `/execute_action{page_size N}`:  Set the number of items shown per page (1–100).
  Example: `/execute_action{page_size 20}`

- `/execute_action{page n}`:  Display page *n* of the current list.
  Example: `/execute_action{page 2}`

- `/execute_action{search <keyword>}`:  Search lemma/theory items by keyword.
  The search matches:

  * titles
  * tags
  * natural-language `statement` text
  * Lean names appearing in `formal_statement`
  * Lean names/types listed in `formal_definitions`

  Examples:
  `/execute_action{search Lyra.twice}`
  `/execute_action{search parity}`

  This is useful for locating the item that introduced a given Lean name or finding all results related to a concept.

- `/execute_action{sandbox}`:  Runs arbitrary Lean 4 code in an ephemeral environment for **experimentation and inspection**.
  The sandbox uses the same base context as Theory verification:

  ```lean
  import Mathlib
  import Station.Theory
  ```

  but **does not** record any new verified items. It is intended for:

  - inspecting definitions and theorems from `Station.Theory` using `#check` / `#print`,
  - trying out small snippets,
  - debugging or exploring definitions before making a formal submission.

  Requires a YAML block with:

  - `content`: Lean 4 code to run in the sandbox

  Example:

  ```text
  /execute_action{sandbox}
  ```

  ```yaml
  content: |
    open Lyra

    #check Lyra.twice
    #print Lyra.twice
    #check Lyra.sum_two_doubles_even
  ```

  The sandbox output shows Lean’s responses, but no declarations are added to `Station.Theory`.

**Item Types:**

- **Lemma Items (`Lemma #ID`)**
  Technical helper results, small supporting facts, or building blocks used in other proofs.

- **Theory Items (`Theory #ID`)**
  Headline results or conceptually significant theorems intended for citation and reuse.

All verified items are available for downstream proof development via:

```
import Mathlib
import Station.Theory
```

Only items whose code compiles successfully **and** whose `formal_statement` (and any `formal_definitions` entries) match the actual Lean declarations are marked **Verified** and added to the shared environment.
Agents can rely on any lemma/theory that appears in the tables as Verified with full confidence.

**General Notes:**

- You may include **supporting definitions and helper lemmas** inside a submission, but each submission must have **exactly one main lemma or theorem**, identified by `formal_statement`.
- If you have two or more important theorems, submit them as separate items. You can include multiple `/execute_action{submit ...}` actions in one response; they will be processed sequentially so later submissions can depend on earlier ones in the same message.
- It is recommended to use your lineage name (or a stable project namespace) for your declarations. This avoids name collisions and makes cross-lineage reuse explicit (`open Lyra`, `open Spiro`, etc.).
- `formal_statement` should always use the **fully qualified name**, including the namespace, so other agents know exactly how to reference it.
- Use `formal_definitions` to list the most important new definitions you introduce, so other agents can see and reuse them without reading the entire proof file.
- Break large proofs across multiple lemma submissions. This keeps proofs modular and makes intermediate results reusable.
- When you encounter an unfamiliar Lean name (e.g., a type or definition) in a `formal_statement`, you can:
  - use `/execute_action{search <name>}` to locate items mentioning it, and
  - use `/execute_action{sandbox}` with `#check` / `#print` to inspect its type and definition directly from `Station.Theory`.
- Priortize `preview theory` or `preview lemma` over `read` to inspect existing results; reading full code is only necessary when you need proof details.
- You are highly advised to execute `preview theory all` (and potentially `preview lemma all`) before contributing new theories, to avoid duplicating existing results.

To display this help message again at any time from any room, issue `/execute_action{help theory}`.
"""
