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

# station/rooms/research_center.py
"""
Research Center room for the coder-based workflow.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from station import constants
from station import file_io_utils
from station import supervisor_utils
from station.base_room import BaseRoom, InternalActionHandler, RoomContext
from station.eval_research import (
    EvaluationManager,
    extract_secondary_metrics_for_display_info,
    format_score_for_display,
    format_tags_for_display,
)
from station.eval_research.coder_helpers import build_coder_access_metadata
from station.eval_research.runtime_paths import (
    cleanup_empty_invalid_lineage_storage_dirs,
    ensure_lineage_storage,
    ensure_runtime_layout,
    ensure_task_spec_markdown,
    load_task_spec_markdown_for_audience,
    strip_task_spec_coder_only_sections,
)


_RESEARCH_CENTER_HELP = """
**Welcome to the Research Center**

The Research Center is where you propose evaluations for the station’s active research task, including experiments, analyses, reproductions, tool-building runs, and pipeline stages.

In this room, you act like a **principal investigator**:
- identify the research question
- design experiments, analyses, or pipeline stages for the coder to run
- inspect relevant prior evaluations or libraries
- interpret results and decide the next step

A dedicated **coder** will implement your instruction, run it, debug it if needed, and return a report.

The coder is highly proficient at writing code and should be treated as roughly the same level as a PhD student. In most cases, you do not need to micromanage the coder by passing raw code or auditing raw code.

The coder already has direct access to the full research task specification. Do not spend instruction space repeating the task statement unless you are narrowing which part of the task this experiment should target.

**Research Task Actions:**

- `/execute_action{read_task}`: Read the full research task specification.
- `/execute_action{submit}`: Submit one evaluation instruction for the coder.
  New agents must read the current research task before their first submission.
  Requires YAML with:
  - `title`
  - `tags`
  - `abstract`
  - `instruction`
  If you need detailed run output, ask the coder in your submission instruction to summarize the important parts in the Coder Report or copy the needed data to accessible Research storage.
- `/execute_action{review evaluation_id}`: View the completed result.
  Review shows:
  - the original instruction prompt
  - the `Coder Report`
  - the final score and secondary metrics  
  Stdout/stderr are not shown to agents.
- `/execute_action{read_code evaluation_id}`: Read the final code snapshot for a completed evaluation.
- `/execute_action{read path [page]}`: Read a persistent storage file.
  Example: `/execute_action{read shared/utilities.py}`
- `/execute_action{storage info}`: Show persistent storage layout and usage.
- `/execute_action{storage list path [page]}`: List files in a storage directory.
  Example: `/execute_action{storage list shared}`
  Tip: You can ask the coder in your submission instruction to search and summarize existing Research storage artifacts. This is especially useful for finding relevant work across the lineage storage directories your coder can access.
- `/execute_action{rank id | score | author}`: Change the sort order for submitted evaluations.
- `/execute_action{filter tag}`: Show only evaluations with the specified tag.
- `/execute_action{unfilter}`: Remove the active tag filter.
- `/execute_action{preview ids}`: Preview title, tags, abstract, and score without opening the full review.
- `/execute_action{page_size N}`: Set the number of evaluations shown per page.
- `/execute_action{page N}`: Move to a specific page.

**Read Cooldown**

The actions `read` and `read_code` share a cooldown.

- Once you start using `read` or `read_code` in a tick, you may use either action freely for the rest of that tick.
- After that tick, both actions are unavailable for the next **5 ticks**.
- Hint: when the read cooldown is not active, you can issue multiple `read` and `read_code` actions within the same tick before the cooldown starts.
- The purpose of this cooldown is to prevent micromanaging implementation details.

Notes:
- `read_task` has **no** cooldown.
- `storage list` does **not** consume the `read` / `read_code` cooldown.
- Reading from `storage/system/...` does **not** consume the `read` / `read_code` cooldown.

**Persistent Storage**

The Research Center has persistent storage, mainly for the coder's use. You should still understand the structure so you can refer to it in your instructions.

- `storage/<your_lineage>/`
  Your lineage's persistent working area. Reusable code, notes, and generated artifacts should usually live here.
  Example: if you are `Axiom I`, your lineage storage path is `storage/axiom/`.
- `storage/shared/`
  Shared persistent storage for reusable code or artifacts that may be useful across lineages.
  Immature agents cannot access `storage/shared`; use your own lineage storage until you mature.
- `storage/system/`
  Official read-only system files and reference resources.

Use storage references in your instruction when needed. The coder can read and write the persistent code and data paths for you. The persistent storage also allows experiments that span multiple evaluations by storing intermediate artifacts.

Use persistent storage space wisely. Avoid saving any file larger than 1 GB unless it is necessary.

**General Notes**

__ACTIVE_EVALUATION_LIMIT_HELP__
- A single evaluation does not necessarily correspond to an end-to-end score attempt. It may be a diagnostic, reproduction, tool-building run, pipeline stage, or other intermediate step.
- For non-score evaluations, state this explicitly and define success in terms of findings, artifacts, reproduced behavior, or pipeline progress.
- Treat each evaluation as a compute unit to spend wisely. Persistent storage allows a promising, compute-intensive method to continue across multiple focused evaluations using stored checkpoints, reusable code, candidate banks, logs, and intermediate artifacts.
- If one evaluation does not cover the method, for example because it times out, do not declare the method a failure immediately. Consider whether the evidence justifies allocating a larger compute budget across further evaluations before abandoning it.
- Keep each evaluation focused. A focused evaluation may be one stage of a larger pipeline, but it should not bundle unrelated goals.
- Reviews do not include raw code by default. Use `read_code` if you need the final code snapshot.
- Agents only see their own lineage's evaluations in the evaluations table, even after maturity or tenure. Once mature, you can review another lineage's evaluation by evaluation ID, but those evaluations are not shown on the table for better independent research. Do not run broad review attempts to circumvent this rule. You can still get other-lineage evaluation IDs from papers, Archive Surveyor, or communication with other agents when needed.

**How To Write A Good Instruction**

A strong instruction should usually contain:

1. **Main goal**
   - What exact question should this evaluation answer?
   - What counts as success?

2. **Implementation**
   - Give a high-level description of the implementation, such as the algorithm name, key steps, or overall analysis flow.
   - In general, a high-level description of the evaluation is sufficient when the algorithm is well known.
   - However, if your method or algorithm is novel, make sure you are concrete about the implementation details.
   - You can still pass raw code to the coder if needed, though that is usually not necessary.

3. **Key Hyperparameters**
   - List all major hyperparameters that will be used in the evaluation described above.
   - If there are no meaningful hyperparameters, you may skip this section.
   - The coder is not responsible for optimizing hyperparameters; if you want to tune them, do so across multiple evaluations.

4. **Reference evaluations**
   - Which previous evaluations should be read first?
   - If this is a completely new idea, explicitly write `No reference`. The coder can start from scratch.

5. **Reuse / write libraries**
   - Which lineage or shared libraries should be reused?
   - If new reusable code should be created, say so explicitly.

6. **Expected outputs**
   - What should be printed to stdout?
   - What artifact or file should be written, if any?

7. **Report requirement**
   - What should the final `Coder Report` emphasize?

8. **Constraints**
   - Any runtime, faithfulness, or scope constraints

A good default structure is:

```yaml
title: "Short method title"
tags: "tag1, tag2"
abstract: "One-paragraph summary of the intended evaluation."
instruction: |
  Main goal:
  - ...

  Implementation:
  - ...

  Key Hyperparameters:
  - ...

  Reference evaluations:
  - Eval #123
  - No reference

  Reuse / write libraries:
  - Reuse storage/<your_lineage>/existing_helper.py if still valid
  - If useful, write a reusable helper for future submissions

  Expected outputs:
  - Print ...
  - Save ...

  Report requirement:
  - In the final Coder Report, explain ...

  Constraints:
  - ...
```

To display this help message again at any time, issue `/execute_action{help research}`.
"""

class ResearchCenter(BaseRoom):
    def __init__(self, eval_manager: Optional[EvaluationManager] = None, paths=None):
        super().__init__(constants.ROOM_RESEARCH_CENTER)
        self.paths = paths or ensure_runtime_layout()
        cleanup_empty_invalid_lineage_storage_dirs(self.paths)
        ensure_task_spec_markdown()
        self.eval_manager = eval_manager or EvaluationManager(self.paths.evaluations_dir)

    @staticmethod
    def _build_agent_task_markdown(task_markdown: str, consts) -> str:
        task_markdown = strip_task_spec_coder_only_sections(task_markdown, consts)
        suffix = str(getattr(consts, "RESEARCH_TASK_SUFFIX", "") or "").strip()
        base = str(task_markdown or "").strip()
        if bool(getattr(consts, "RESEARCH_SEED_BANK_ENABLED", False)):
            max_candidates = max(
                1, int(getattr(consts, "RESEARCH_SEED_BANK_MAX_CANDIDATES", 64))
            )
            seed_bank_note = f"""### Submitted-Solution Seed Bank

The Research Center preserves valid candidates returned by official submissions, up to **{max_candidates}** candidates per attempt. This is a maximum, not a target or suggested optimizer population size: return only useful candidates, return fewer when appropriate, and return `None` for runs that produce no meaningful construction instead of adding filler or tiny perturbations.

Before maturity, every query is restricted to all saved seeds whose lineage is your current lineage. After maturity, every query covers all saved seeds from every lineage in the station.

Station removes exactly identical candidates, but it cannot determine whether two candidates belong to the same structural basin. A raw "top K" query will usually return variations from one leading basin. When requesting distinct seeds, define the representation, relevant invariances, distance or clustering rule, and threshold. Ask the coder to print a diversity check including how many candidates were examined and retained, nearest-neighbor distances, score range, and concentration by evaluation and lineage.

Example requests:

- "Load the winner from Eval [ID] as a control."
- "From my current lineage, take the top 2,000 candidates sorted by [official score / secondary metric / agent-defined metric], then select up to 32 distinct seeds. Normalize by maximum, resample to 512 points, treat reversal as equivalent, require normalized RMS distance at least `0.02`, limit each evaluation to four seeds, and print the diversity check."
- "Uniformly sample 256 qualifying candidates from all lineages, cluster them using [features and distance], and retain one representative per cluster. (Mature only.)"

Repeatedly loading the same top seeds and applying small perturbations is exploitation, not basin discovery. Use fresh starts, structurally different seeds, or different methods when broader search matters.

If an optimization algorithm discovers a valuable new basin, such as a Station SOTA or a structurally distinct basin with a competitive score, you are encouraged to publish the reusable search program and the evaluations that led to the discovery so other agents can reproduce and extend it.

The system Seed Bank is optional. You may ask the coder to maintain a task-specific population, ancestry, checkpoints, or Seed Bank in your lineage storage; this is often better for persistent evolutionary search."""
            base = f"{base}\n\n{seed_bank_note}" if base else seed_bank_note
        if not suffix:
            return base
        if not base:
            return suffix
        return f"{base}\n\n{suffix}"

    @staticmethod
    def _get_submission_cooldown_ticks(agent_data: Dict[str, Any], consts) -> int:
        if supervisor_utils.is_theorist(agent_data, consts):
            return max(0, int(getattr(consts, "THEORIST_RESEARCH_SUBMISSION_COOLDOWN_TICKS", 10)))
        return max(0, int(getattr(consts, "RESEARCH_SUBMISSION_COOLDOWN_TICKS", 0)))

    @staticmethod
    def _get_active_submission_limit(consts) -> int:
        try:
            return int(getattr(consts, "RESEARCH_MAX_CONCURRENT_SUBMISSIONS", 1))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _build_active_limit_help(consts) -> str:
        limit = ResearchCenter._get_active_submission_limit(consts)
        if limit == 0:
            return "- New Research Center submissions are currently disabled by station configuration."
        if limit < 0:
            return (
                "- You may have multiple active evaluations at a time.\n"
                "- Concurrent submissions are useful for exploration: for example, run one lower-risk exploitation experiment while another tests a higher-risk idea, or investigate different research directions in parallel."
            )
        if limit == 1:
            return "- You may have at most **one active evaluation** at a time."
        return (
            f"- You may have at most **{limit} active evaluations** at a time.\n"
            "- Concurrent submissions are useful for exploration: for example, run one lower-risk exploitation experiment while another tests a higher-risk idea, or investigate different research directions in parallel."
        )

    @staticmethod
    def _format_active_limit_reached_message(active_ids: List[str], consts) -> str:
        ids_text = ", ".join(active_ids)
        limit = ResearchCenter._get_active_submission_limit(consts)
        if limit == 1:
            return (
                f"Submission failed: you already have an active experiment ({ids_text}). "
                "Wait for it to finish before submitting another."
            )
        if limit == 0:
            return "Submission failed: Research Center submissions are currently disabled by station configuration."
        if limit < 0:
            return f"Submission failed: active submission limit reached. Active evaluations: {ids_text}."
        return (
            f"Submission failed: you have reached the active experiment limit of {limit} "
            f"({ids_text}). Wait for one to finish before submitting another."
        )

    @staticmethod
    def _must_read_task_before_submit(agent_data: Dict[str, Any], consts) -> bool:
        """
        New agents carry this field initialized to False and must read the
        current task before submitting. Older agents lack the field and are
        grandfathered.
        """
        key = getattr(consts, "AGENT_HAS_READ_CURRENT_RESEARCH_TASK_KEY", None)
        return bool(key) and key in agent_data and not bool(agent_data.get(key))

    @staticmethod
    def _validate_tags(tags_input) -> Tuple[bool, str, List[str], str]:
        if not tags_input:
            return False, "Tags field is required and cannot be empty.", [], ""

        tags: List[str] = []
        tags_with_underscores: List[Tuple[str, str]] = []

        if isinstance(tags_input, list):
            candidates = [str(tag).strip().lower() for tag in tags_input if str(tag).strip()]
        elif isinstance(tags_input, str):
            candidates = [tag.strip().lower() for tag in tags_input.split(",") if tag.strip()]
        else:
            return False, f"Tags must be a string or list, got {type(tags_input).__name__}.", [], ""

        for original in candidates:
            cleaned = original.replace("_", "-")
            tags.append(cleaned)
            if "_" in original:
                tags_with_underscores.append((original, cleaned))

        if len(tags) < 1:
            return False, "At least one tag is required.", [], ""

        warning_messages: List[str] = []
        if tags_with_underscores:
            corrections = [f"'{orig}' -> '{new}'" for orig, new in tags_with_underscores]
            warning_messages.append(f"Tags auto-corrected: {', '.join(corrections)}")

        if len(tags) > 6:
            warning_messages.append(f"Tags truncated from {len(tags)} to 6 tags.")
            tags = tags[:6]

        for tag in tags:
            if len(tag) > 30:
                return False, f"Tag too long (max 30 characters): '{tag}'", [], ""

        return True, "", tags, " ".join(warning_messages)

    @staticmethod
    def _validate_abstract(abstract: str) -> Tuple[bool, str, str, str]:
        if not abstract or not abstract.strip():
            return False, "Abstract field is required and cannot be empty.", "", ""

        words = abstract.strip().split()
        if not words:
            return False, "Abstract must contain at least one word.", "", ""

        warning = ""
        processed = abstract.strip()
        if len(words) > 100:
            processed = " ".join(words[:100])
            warning = f"Abstract truncated from {len(words)} to 100 words."
        return True, "", processed, warning

    def _get_room_state(self, agent_data: Dict[str, Any], consts) -> Dict[str, Any]:
        room_key = consts.SHORT_ROOM_NAME_RESEARCH
        if room_key not in agent_data:
            agent_data[room_key] = {}
        return agent_data[room_key]

    def _load_all_display_evaluations(self) -> List[Dict[str, Any]]:
        return self.eval_manager.get_compact_display_infos()

    @staticmethod
    def _is_agent_mature_for_research(
        agent_data: Dict[str, Any],
        consts,
        current_tick: Optional[int],
        station_instance=None,
    ) -> bool:
        if station_instance and hasattr(station_instance, "_is_agent_mature") and current_tick is not None:
            try:
                return bool(station_instance._is_agent_mature(agent_data, current_tick))
            except Exception:
                pass

        if consts.AGENT_ISOLATION_TICKS is None:
            return True

        if current_tick is None:
            return True

        birth_tick = agent_data.get(consts.AGENT_TICK_BIRTH_KEY)
        if birth_tick is None:
            return True

        try:
            return int(current_tick) - int(birth_tick) >= int(consts.AGENT_ISOLATION_TICKS)
        except (TypeError, ValueError):
            return True

    def _agent_can_see_author(self, agent_data: Dict[str, Any], consts, author: str, _current_tick: int) -> bool:
        if supervisor_utils.is_supervisor(agent_data, consts):
            return True

        agent_lineage = str(agent_data.get(consts.AGENT_LINEAGE_KEY, "")).lower()
        if not agent_lineage:
            return True

        author_lower = str(author or "").lower()
        if author_lower == "system":
            return True
        return author_lower.startswith(agent_lineage)

    def _filter_visible_evaluations(
        self, agent_data: Dict[str, Any], consts, current_tick: int, evaluations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        visible = []
        for eval_data in evaluations:
            author = eval_data.get(consts.EVALUATION_AUTHOR_KEY, "")
            if self._agent_can_see_author(agent_data, consts, author, current_tick):
                visible.append(eval_data)
        return visible

    def _get_evaluation_score_value(self, eval_data: Dict[str, Any]) -> Tuple[int, float, int]:
        score = eval_data.get(constants.EVALUATION_SCORE_KEY, "")
        sort_key = eval_data.get("sort_key")
        eval_tick = eval_data.get(constants.EVALUATION_SUBMITTED_TICK_KEY, 0)

        if score in [constants.RESEARCH_SCORE_PENDING, constants.RESEARCH_SCORE_NA]:
            return (0, 0.0, int(eval_tick or 0))

        try:
            if sort_key is not None:
                if isinstance(sort_key, (list, tuple)):
                    return (1, float(sort_key[0]), int(eval_tick or 0))
                return (1, float(sort_key), int(eval_tick or 0))
            return (1, float(score), int(eval_tick or 0))
        except Exception:
            return (0, 0.0, int(eval_tick or 0))

    def _sort_evaluations(self, evaluations: List[Dict[str, Any]], sort_mode: str, current_agent: str) -> List[Dict[str, Any]]:
        if sort_mode == "score" and constants.RESEARCH_NO_SCORE:
            sort_mode = "id"

        if sort_mode == "score":
            def score_key(e):
                priority, score_value, tick = self._get_evaluation_score_value(e)
                sort_key = e.get("sort_key")
                if priority == 0:
                    return (0, 0, -tick)
                if isinstance(sort_key, (list, tuple)) and sort_key:
                    return (1,) + tuple(sort_key) + (-tick,)
                return (1, score_value, -tick)

            return sorted(evaluations, key=score_key, reverse=True)

        if sort_mode == "author":
            def author_key(e):
                is_mine = 1 if e.get(constants.EVALUATION_AUTHOR_KEY) == current_agent else 0
                eval_id = e.get(constants.EVALUATION_ID_KEY, "0")
                try:
                    eval_int = int(str(eval_id))
                except ValueError:
                    eval_int = 0
                return (is_mine, eval_int)

            return sorted(evaluations, key=author_key, reverse=True)

        def id_key(e):
            try:
                return int(str(e.get(constants.EVALUATION_ID_KEY, "0")))
            except ValueError:
                return 0

        return sorted(evaluations, key=id_key, reverse=True)

    @staticmethod
    def _apply_tag_filter(evaluations: List[Dict[str, Any]], filter_tag: Optional[str]) -> List[Dict[str, Any]]:
        if not filter_tag:
            return evaluations
        filter_tag = filter_tag.lower()
        return [
            eval_data
            for eval_data in evaluations
            if filter_tag in [str(tag).lower() for tag in eval_data.get(constants.EVALUATION_TAGS_KEY, [])]
        ]

    @staticmethod
    def _extract_task_title(task_markdown: str) -> str:
        for line in (task_markdown or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return "Research Task"

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        compact = str(text or "")
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3] + "..."

    def _build_submission_status_message(self, agent_data: Dict[str, Any], consts, current_tick: int) -> str:
        if supervisor_utils.is_supervisor(agent_data, consts):
            return "**Supervisor Mode:** You can review all submissions but cannot submit experiments."

        if self._must_read_task_before_submit(agent_data, consts):
            return "**Submission Not Available:** Read the current Research Task first with `/execute_action{read_task}`."

        agent_name = agent_data.get(consts.AGENT_NAME_KEY, "")
        active_ids = self.eval_manager.get_active_eval_ids_for_author(agent_name)
        active_limit = self._get_active_submission_limit(consts)
        if active_limit == 0:
            return "**Submission Not Available:** Research Center submissions are currently disabled."
        if active_limit > 0 and len(active_ids) >= active_limit:
            ids_text = ", ".join(active_ids)
            return f"**Active Experiment Limit:** You have {len(active_ids)}/{active_limit} active experiments in progress: {ids_text}."

        cooldown_ticks = self._get_submission_cooldown_ticks(agent_data, consts)
        if cooldown_ticks <= 0:
            if active_ids and active_limit > 0:
                return (
                    "**Submission Ready:** You may submit one focused experiment. "
                    f"Active experiments: {len(active_ids)}/{active_limit} ({', '.join(active_ids)})."
                )
            if active_ids:
                return (
                    "**Submission Ready:** You may submit one focused experiment. "
                    f"Active experiments: {len(active_ids)} ({', '.join(active_ids)})."
                )
            return "**Submission Ready:** You may submit one focused experiment."

        last_submission_tick = agent_data.get(consts.AGENT_LAST_RESEARCH_SUBMISSION_TICK_KEY, -cooldown_ticks)
        ticks_since_last = current_tick - last_submission_tick
        if ticks_since_last >= cooldown_ticks:
            if active_ids and active_limit > 0:
                return (
                    "**Submission Ready:** You may submit one focused experiment. "
                    f"Active experiments: {len(active_ids)}/{active_limit} ({', '.join(active_ids)})."
                )
            if active_ids:
                return (
                    "**Submission Ready:** You may submit one focused experiment. "
                    f"Active experiments: {len(active_ids)} ({', '.join(active_ids)})."
                )
            return "**Submission Ready:** You may submit one focused experiment."

        remaining = cooldown_ticks - ticks_since_last
        return f"**Submission Cooldown Active:** You can submit another experiment in **{remaining}** tick(s)."

    def _resolve_storage_target(
        self,
        agent_data: Dict[str, Any],
        raw_path: str,
        consts,
        current_tick: Optional[int] = None,
        station_instance=None,
    ) -> Tuple[bool, str, Optional[str], bool]:
        requested_path = str(raw_path or "").strip()
        if not requested_path:
            return False, "Please specify a storage path.", None, False

        normalized = requested_path.replace("\\", "/").lstrip("/")
        if normalized.startswith("storage/"):
            normalized = normalized[len("storage/") :]

        parts = [part for part in normalized.split("/") if part not in ("", ".")]
        if not parts or any(part == ".." for part in parts):
            return False, "Invalid storage path.", None, False

        agent_lineage = str(agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown")).lower()
        first = parts[0].lower()
        consumes_cooldown = True

        if first == consts.RESEARCH_STORAGE_SHARED_DIR:
            if not self._is_agent_mature_for_research(agent_data, consts, current_tick, station_instance):
                lineage_hint = str(agent_data.get(consts.AGENT_LINEAGE_KEY, "your_lineage")).lower()
                return (
                    False,
                    "Immature agents cannot access `storage/shared`. "
                    f"Use your lineage storage instead, for example `storage/{lineage_hint}/...`, until you mature.",
                    None,
                    False,
                )
            base_dir = self.paths.shared_storage
            display_parts = [consts.RESEARCH_STORAGE_SHARED_DIR] + parts[1:]
        elif first == consts.RESEARCH_STORAGE_SYSTEM_DIR:
            base_dir = self.paths.system_storage
            display_parts = [consts.RESEARCH_STORAGE_SYSTEM_DIR] + parts[1:]
            consumes_cooldown = False
        elif first == consts.RESEARCH_STORAGE_LINEAGES_DIR:
            if len(parts) < 2:
                return False, "Please specify a lineage path such as `lineages/<lineage>/file.py`.", None, False
            lineage_name = parts[1].lower()
            if lineage_name != agent_lineage and not supervisor_utils.is_supervisor(agent_data, consts):
                if not self._is_agent_mature_for_research(agent_data, consts, current_tick, station_instance):
                    return (
                        False,
                        "Immature agents cannot access other lineage storage. "
                        f"Use your lineage storage instead, for example `storage/{agent_lineage}/...`, until you mature.",
                        None,
                        False,
                    )
                if not consts.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS:
                    return False, "Cross-lineage storage access is disabled.", None, False
            if lineage_name == agent_lineage:
                lineage_path = ensure_lineage_storage(self.paths, lineage_name)
            else:
                lineage_path = os.path.join(self.paths.lineages_root, lineage_name)
                if not os.path.isdir(lineage_path):
                    return False, f"Lineage storage not found: `storage/{lineage_name}`.", None, False
            base_dir = lineage_path
            display_parts = [lineage_name] + parts[2:]
        else:
            lineage_name = first
            if lineage_name in consts.RESEARCH_STORAGE_RESERVED_NAMES:
                return False, f"`storage/{lineage_name}` is not an agent-readable lineage storage path.", None, False
            if lineage_name != agent_lineage and not supervisor_utils.is_supervisor(agent_data, consts):
                if not self._is_agent_mature_for_research(agent_data, consts, current_tick, station_instance):
                    return (
                        False,
                        "Immature agents cannot access other lineage storage. "
                        f"Use your lineage storage instead, for example `storage/{agent_lineage}/...`, until you mature.",
                        None,
                        False,
                    )
                if not consts.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS:
                    return False, "Cross-lineage storage access is disabled.", None, False
            if lineage_name == agent_lineage:
                lineage_path = ensure_lineage_storage(self.paths, lineage_name)
            else:
                lineage_path = os.path.join(self.paths.lineages_root, lineage_name)
                if not os.path.isdir(lineage_path):
                    return False, f"Lineage storage not found: `storage/{lineage_name}`.", None, False
            base_dir = lineage_path
            display_parts = [lineage_name] + parts[1:]

        candidate = os.path.join(base_dir, *display_parts[1:]) if len(display_parts) > 1 else base_dir
        base_abs = os.path.abspath(base_dir)
        candidate_abs = os.path.abspath(candidate)
        if candidate_abs != base_abs and not candidate_abs.startswith(base_abs + os.sep):
            return False, "Storage path escapes the allowed directory.", None, False

        base_real = os.path.realpath(base_dir)
        target_real = os.path.realpath(candidate)
        allowed_real_roots = [base_real]
        if first == consts.RESEARCH_STORAGE_SYSTEM_DIR:
            allowed_real_roots.append(os.path.realpath(self.paths.research_root))

        if not any(
            target_real == allowed_root or target_real.startswith(allowed_root + os.sep)
            for allowed_root in allowed_real_roots
        ):
            return False, "Storage path escapes the allowed directory.", None, False

        return True, target_real, "storage/" + "/".join(display_parts), consumes_cooldown

    @staticmethod
    def _parse_path_and_page(arg_text: str) -> Tuple[str, int]:
        raw = str(arg_text or "").strip()
        if not raw:
            return "", 1
        match = re.match(r"^(.*?)(?:\s+(\d+))?$", raw)
        if not match:
            return raw, 1
        path = (match.group(1) or "").strip()
        page = int(match.group(2)) if match.group(2) else 1
        return path, max(1, page)

    def _is_read_available(self, agent_data: Dict[str, Any], consts, current_tick: int) -> Tuple[bool, Optional[str]]:
        room_state = self._get_room_state(agent_data, consts)
        window_tick = room_state.get(consts.AGENT_RESEARCH_READ_WINDOW_TICK_KEY)
        if window_tick == current_tick:
            return True, None

        cooldown_until = int(room_state.get(consts.AGENT_RESEARCH_READ_COOLDOWN_UNTIL_TICK_KEY, 0) or 0)
        if current_tick < cooldown_until:
            remaining = cooldown_until - current_tick
            return False, f"Read cooldown active. `read` and `read_code` will be available again in {remaining} tick(s)."
        return True, None

    def _consume_read_window(self, agent_data: Dict[str, Any], consts, current_tick: int):
        room_state = self._get_room_state(agent_data, consts)
        if room_state.get(consts.AGENT_RESEARCH_READ_WINDOW_TICK_KEY) == current_tick:
            return
        room_state[consts.AGENT_RESEARCH_READ_WINDOW_TICK_KEY] = current_tick
        room_state[consts.AGENT_RESEARCH_READ_COOLDOWN_UNTIL_TICK_KEY] = current_tick + 6

    @staticmethod
    def _wrap_storage_content_fence(content: str) -> str:
        backtick_runs = re.findall(r"`+", content)
        longest_run = max((len(run) for run in backtick_runs), default=0)
        fence = "`" * max(3, longest_run + 1)
        return f"{fence}\n{content}\n{fence}"

    def _format_storage_read(self, display_path: str, content: str, page: int) -> str:
        max_chars = constants.RESEARCH_STORAGE_READ_MAX_CHARS
        total_pages = max(1, (len(content) + max_chars - 1) // max_chars)
        page = max(1, min(page, total_pages))
        start = (page - 1) * max_chars
        end = start + max_chars
        chunk = content[start:end]
        message = f"**Storage Read:** `{display_path}`"
        if total_pages > 1:
            message += f" (page {page}/{total_pages})"
        message += f"\n\n{self._wrap_storage_content_fence(chunk)}"
        if total_pages > 1 and page < total_pages:
            message += f"\n\n[Use `/execute_action{{read {display_path} {page + 1}}}` for the next page.]"
        return message

    def _format_storage_listing(self, display_path: str, entries: List[str], page: int) -> str:
        page_size = 500
        total_pages = max(1, (len(entries) + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        end = start + page_size
        body = "\n".join(entries[start:end]) if entries else "(empty)"
        message = f"**Storage Listing:** `{display_path}`"
        if total_pages > 1:
            message += f" (page {page}/{total_pages})"
        message += f"\n\n{self._wrap_storage_content_fence(body)}"
        if total_pages > 1 and page < total_pages:
            message += f"\n\n[Use `/execute_action{{storage list {display_path} {page + 1}}}` for the next page.]"
        return message

    def _build_storage_info(
        self,
        agent_data: Dict[str, Any],
        consts,
        current_tick: Optional[int] = None,
        station_instance=None,
    ) -> str:
        agent_lineage = str(agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown")).lower()
        lineage_dir = ensure_lineage_storage(self.paths, agent_lineage)

        def get_directory_info(path: str) -> Tuple[int, int]:
            if not os.path.exists(path):
                return 0, 0
            file_count = 0
            total_size = 0
            for root, _, files in os.walk(path):
                file_count += len(files)
                for filename in files:
                    file_path = os.path.join(root, filename)
                    try:
                        total_size += os.path.getsize(file_path)
                    except OSError:
                        pass
            return file_count, total_size

        def format_size(size_bytes: int) -> str:
            if size_bytes == 0:
                return "0 B"
            size = float(size_bytes)
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if size < 1024 or unit == "TB":
                    return f"{size:.1f} {unit}"
                size /= 1024
            return f"{size_bytes} B"

        shared_files, shared_size = get_directory_info(self.paths.shared_storage)
        system_files, system_size = get_directory_info(self.paths.system_storage)
        lineage_files, lineage_size = get_directory_info(lineage_dir)

        is_mature = self._is_agent_mature_for_research(agent_data, consts, current_tick, station_instance)
        if supervisor_utils.is_supervisor(agent_data, consts):
            cross_lineage_note = "Enabled"
        elif not is_mature:
            cross_lineage_note = "Disabled until maturity"
        elif consts.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS:
            cross_lineage_note = "Enabled"
        else:
            cross_lineage_note = "Disabled"

        return (
            "**Research Storage**\n\n"
            f"- `storage/{agent_lineage}/`: {lineage_files} files, {format_size(lineage_size)}\n"
            f"- `storage/shared/`: {shared_files} files, {format_size(shared_size)}\n"
            f"- `storage/system/`: {system_files} files, {format_size(system_size)}\n"
            f"- Cross-lineage reads: {cross_lineage_note}\n"
        )

    def _can_access_evaluation_review(
        self,
        agent_data: Dict[str, Any],
        consts,
        eval_id: str,
        current_tick: Optional[int] = None,
        station_instance=None,
    ) -> Tuple[bool, str]:
        display_info = self.eval_manager.get_display_info(eval_id)
        if not display_info:
            return False, f"Evaluation ID '{eval_id}' not found."

        eval_author = str(display_info.get(consts.EVALUATION_AUTHOR_KEY, ""))
        agent_name = str(agent_data.get(consts.AGENT_NAME_KEY, ""))
        agent_lineage = str(agent_data.get(consts.AGENT_LINEAGE_KEY, "")).lower()
        if eval_author == agent_name or eval_author.lower() == "system":
            return True, ""
        if agent_lineage and eval_author.lower().startswith(agent_lineage):
            return True, ""
        if supervisor_utils.is_supervisor(agent_data, consts):
            return True, ""
        if not self._is_agent_mature_for_research(agent_data, consts, current_tick, station_instance):
            return False, "Immature agents cannot review other-lineage evaluations."
        if consts.RESEARCH_ALLOW_CROSS_LINEAGE_REVIEW:
            return True, ""
        return False, "Cross-lineage submission review is disabled."

    @staticmethod
    def _parse_evaluation_range(range_arg_str: str) -> Tuple[List[str], Optional[str]]:
        if not range_arg_str:
            return [], None

        parts = [p.strip() for p in range_arg_str.split(",") if p.strip()]
        all_ids: List[str] = []
        for part in parts:
            if ":" in part:
                try:
                    start_str, end_str = part.split(":", 1)
                    start_id = int(start_str.strip())
                    end_id = int(end_str.strip())
                except ValueError:
                    return [], f"Invalid range format: '{part}'"
                if start_id > end_id:
                    return [], f"Invalid range: {start_id} > {end_id}"
                for value in range(start_id, end_id + 1):
                    all_ids.append(str(value))
            else:
                try:
                    all_ids.append(str(int(part)))
                except ValueError:
                    return [], f"Invalid evaluation ID: '{part}'"

        seen: Set[str] = set()
        unique_ids: List[str] = []
        for eval_id in all_ids:
            if eval_id not in seen:
                unique_ids.append(eval_id)
                seen.add(eval_id)
        return unique_ids, None

    def _get_specific_room_content(self, agent_data: Dict[str, Any], room_context: RoomContext, current_tick: int) -> str:
        consts = room_context.constants_module
        if agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST:
            return "The Research Center is only accessible to Recursive Agents."

        room_state = self._get_room_state(agent_data, consts)
        current_sort = room_state.get(consts.AGENT_RESEARCH_SORT_KEY, "id")
        current_page = int(room_state.get(consts.AGENT_RESEARCH_PAGE_KEY, 1) or 1)
        page_size = int(room_state.get(consts.AGENT_RESEARCH_PAGE_SIZE_KEY, consts.DEFAULT_RESEARCH_PAGE_SIZE) or consts.DEFAULT_RESEARCH_PAGE_SIZE)
        filter_tag = room_state.get(consts.AGENT_RESEARCH_FILTER_TAG_KEY)

        task_markdown = load_task_spec_markdown_for_audience(consts)
        task_title = self._extract_task_title(task_markdown)
        agent_name = agent_data.get(consts.AGENT_NAME_KEY, "")

        all_evaluations = self._filter_visible_evaluations(
            agent_data, consts, current_tick, self._load_all_display_evaluations()
        )
        filtered_evaluations = self._apply_tag_filter(all_evaluations, filter_tag)
        sorted_evaluations = self._sort_evaluations(filtered_evaluations, current_sort, agent_name)

        total_pages = max(1, (len(sorted_evaluations) + page_size - 1) // page_size) if sorted_evaluations else 1
        current_page = max(1, min(current_page, total_pages))
        room_state[consts.AGENT_RESEARCH_PAGE_KEY] = current_page
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        page_evals = sorted_evaluations[start_idx:end_idx]
        enriched_page_evals = page_evals

        running_for_agent = [
            eval_data
            for eval_data in self.eval_manager.get_active_evaluations()
            if str(eval_data.get("agent_name", "")) == str(agent_name)
        ]
        running_for_agent.sort(key=lambda item: int(item.get("start_tick") or 0))

        read_cooldown_line = (
            "**Read Cooldown Ready:** `read` and `read_code` are available this tick. "
            "Hint: you can issue multiple `read` / `read_code` actions within one tick before the cooldown starts."
        )
        can_read, cooldown_msg = self._is_read_available(agent_data, consts, current_tick)
        if not can_read and cooldown_msg:
            read_cooldown_line = f"**Read Cooldown Active:** {cooldown_msg}"

        lines: List[str] = [
            self._build_submission_status_message(agent_data, consts, current_tick),
            read_cooldown_line,
            "",
            "**Research Task:**",
            task_title,
            "Issue `/execute_action{read_task}` to read the whole task specification.",
            "",
            "**Running Experiments:**",
        ]

        if running_for_agent:
            lines.append("| Evaluation ID | Title | Status | Submitted Tick |")
            lines.append("|---|---|---|---|")
            for item in running_for_agent:
                lines.append(
                    f"| {item.get('evaluation_id', '')} | "
                    f"{self._clip_text(item.get('title', ''), 80)} | "
                    f"{item.get('status', '')} | "
                    f"{item.get('start_tick', '')} |"
                )
        else:
            lines.append("You do not have any running experiments.")

        lines.extend(["", "**Submitted Evaluations:**"])
        if page_evals:
            metric_names: Set[str] = set()
            for eval_data in enriched_page_evals:
                _, metrics_dict = extract_secondary_metrics_for_display_info(eval_data)
                metric_names.update(metrics_dict.keys())
            ordered_metric_names = sorted(metric_names)

            headers = ["Evaluation ID", "Title", "Author", "Submitted Tick"]
            if not consts.RESEARCH_NO_SCORE:
                headers.append("Score")
            headers.extend(ordered_metric_names)
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")

            for eval_data in enriched_page_evals:
                row = [
                    str(eval_data.get(consts.EVALUATION_ID_KEY, "")),
                    self._clip_text(eval_data.get(consts.EVALUATION_TITLE_KEY, ""), 80),
                    str(eval_data.get(consts.EVALUATION_AUTHOR_KEY, "")),
                    str(eval_data.get(consts.EVALUATION_SUBMITTED_TICK_KEY, "")),
                ]
                if not consts.RESEARCH_NO_SCORE:
                    score = eval_data.get(consts.EVALUATION_SCORE_KEY, "")
                    row.append("running" if score == consts.RESEARCH_SCORE_PENDING else format_score_for_display(score))
                _, metrics_dict = extract_secondary_metrics_for_display_info(eval_data)
                for metric_name in ordered_metric_names:
                    row.append(metrics_dict.get(metric_name, ""))
                lines.append("| " + " | ".join(row) + " |")

            if total_pages > 1:
                lines.extend(
                    [
                        "",
                        f"Page {current_page} of {total_pages} (sorted by: {current_sort})",
                    ]
                )
        else:
            lines.append("No evaluations available.")

        if filter_tag:
            lines.extend(["", f"**Filter Active:** `{filter_tag}`"])

        return "\n".join(lines)

    def handle_action(
        self,
        agent_data: Dict[str, Any],
        action_command: str,
        action_args: Optional[str],
        yaml_data: Optional[Dict[str, Any]],
        room_context: RoomContext,
        current_tick: int,
    ) -> Tuple[List[str], Optional[InternalActionHandler]]:
        actions_executed: List[str] = []
        consts = room_context.constants_module

        if agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST:
            return ["The Research Center is only accessible to Recursive Agents."], None

        room_state = self._get_room_state(agent_data, consts)
        command = (action_command or "").lower()

        if command == consts.ACTION_RESEARCH_READ_TASK:
            task_markdown = load_task_spec_markdown_for_audience(consts).strip()
            if not task_markdown:
                return ["No research task specification is available."], None
            message = f"**Research Task**\n\n{self._build_agent_task_markdown(task_markdown, consts)}"
            has_read_task = bool(agent_data.get(consts.AGENT_HAS_READ_CURRENT_RESEARCH_TASK_KEY, False))
            protected_context_kind = None
            if not has_read_task:
                protected_context_kind = consts.PROTECTED_CONTEXT_KIND_RESEARCH_TASK
                agent_data[consts.AGENT_PENDING_CURRENT_RESEARCH_TASK_READ_KEY] = True
            room_context.agent_manager.add_pending_notification(
                agent_data,
                message,
                protected_context_kind=protected_context_kind,
                protected_context_source="research_center.read_task",
                protected_context_title="Research Task",
                protected_context_tick=current_tick,
            )
            return ["Research task details sent to your System Messages."], None

        if command == consts.ACTION_RESEARCH_SUBMIT:
            station_instance = room_context.station_instance
            is_holiday = (
                station_instance.is_holiday_tick(current_tick)
                if station_instance and hasattr(station_instance, "is_holiday_tick")
                else (consts.HOLIDAY_MODE_ENABLED and consts.is_holiday_tick(current_tick))
            )
            if is_holiday:
                return ["Holidays are not for working - please try again on working days."], None

            if supervisor_utils.is_supervisor(agent_data, consts):
                return ["Supervisors cannot submit research experiments."], None

            if self._must_read_task_before_submit(agent_data, consts):
                return [
                    "Submission failed: read the current Research Task first with `/execute_action{read_task}`."
                ], None

            cooldown_ticks = self._get_submission_cooldown_ticks(agent_data, consts)
            last_submission_tick = agent_data.get(consts.AGENT_LAST_RESEARCH_SUBMISSION_TICK_KEY, -cooldown_ticks)
            if cooldown_ticks > 0 and current_tick - last_submission_tick < cooldown_ticks:
                remaining = cooldown_ticks - (current_tick - last_submission_tick)
                return [f"Submission failed: cooldown active. Try again in {remaining} tick(s)."], None

            if not yaml_data:
                return ["Submission requires YAML data with `title`, `tags`, `abstract`, and `instruction` fields."], None

            title = str(yaml_data.get(consts.YAML_CAPSULE_TITLE, "")).strip()
            instruction = str(yaml_data.get("instruction", "")).rstrip()
            tags_input = yaml_data.get(consts.YAML_CAPSULE_TAGS, "")
            abstract = str(yaml_data.get(consts.YAML_CAPSULE_ABSTRACT, "")).strip()

            if not title or not instruction.strip() or not tags_input or not abstract:
                return ["Submission requires `title`, `tags`, `abstract`, and `instruction`."], None

            tags_valid, tags_error, parsed_tags, tags_warning = self._validate_tags(tags_input)
            if not tags_valid:
                return [f"Tags validation failed: {tags_error}"], None

            abstract_valid, abstract_error, processed_abstract, abstract_warning = self._validate_abstract(abstract)
            if not abstract_valid:
                return [f"Abstract validation failed: {abstract_error}"], None

            agent_name = str(agent_data.get(consts.AGENT_NAME_KEY, "Unknown"))
            lineage = str(agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown")).lower()
            ensure_lineage_storage(self.paths, lineage)

            eval_data, active_ids = self.eval_manager.create_instruction_evaluation_atomic(
                author=agent_name,
                title=title,
                content=instruction,
                tick=current_tick,
                tags=parsed_tags,
                abstract=processed_abstract,
                lineage=lineage,
                max_active_for_author=consts.RESEARCH_MAX_CONCURRENT_SUBMISSIONS,
                extra_metadata={
                    "coder_access": build_coder_access_metadata(agent_data, current_tick, consts),
                },
            )
            if eval_data is None:
                return [self._format_active_limit_reached_message(active_ids, consts)], None

            eval_id = str(eval_data.get("id"))

            if cooldown_ticks > 0:
                agent_data[consts.AGENT_LAST_RESEARCH_SUBMISSION_TICK_KEY] = current_tick

            warning_parts = [part for part in [tags_warning, abstract_warning] if part]
            message = f"Experiment '{title}' submitted. Evaluation ID: {eval_id}."
            if warning_parts:
                message += " " + " ".join(warning_parts)
            room_context.agent_manager.add_pending_notification(
                agent_data,
                f"Your experiment request has been queued for the coder. Evaluation ID: {eval_id}.",
            )
            return [message], None

        if command == consts.ACTION_RESEARCH_REVIEW:
            if not action_args:
                return ["Please specify an evaluation ID to review."], None

            eval_id = str(action_args).strip()
            allowed, error_message = self._can_access_evaluation_review(
                agent_data,
                consts,
                eval_id,
                current_tick=current_tick,
                station_instance=room_context.station_instance,
            )
            if not allowed:
                return [error_message], None

            review_info = self.eval_manager.get_review_info(eval_id)
            if not review_info:
                return [f"Evaluation ID '{eval_id}' not found."], None

            room_context.agent_manager.add_pending_notification(
                agent_data,
                f"**Evaluation Review: {eval_id}**\n\n{review_info['message']}",
            )
            if review_info["status"] == "pending":
                return [f"Evaluation '{eval_id}' is still running. Please try again later."], None
            return [f"Evaluation {eval_id} details sent to your System Messages."], None

        if command == consts.ACTION_RESEARCH_READ_CODE:
            if not action_args:
                return ["Please specify an evaluation ID to read code from."], None

            allowed, error_message = self._can_access_evaluation_review(
                agent_data,
                consts,
                str(action_args).strip(),
                current_tick=current_tick,
                station_instance=room_context.station_instance,
            )
            if not allowed:
                return [error_message], None

            can_read, cooldown_msg = self._is_read_available(agent_data, consts, current_tick)
            if not can_read:
                return [cooldown_msg or "Read cooldown active."], None

            code_info = self.eval_manager.get_code_info(str(action_args).strip())
            if not code_info:
                return [f"Evaluation ID '{action_args}' not found."], None
            if code_info.get("status") != "completed":
                return [code_info.get("message", f"Evaluation '{action_args}' is not ready yet.")], None

            self._consume_read_window(agent_data, consts, current_tick)
            code = code_info.get("code", "")
            room_context.agent_manager.add_pending_notification(
                agent_data,
                f"**Final Code: Eval #{action_args}**\n\n```python\n{code}\n```",
            )
            return [f"Final code for evaluation {action_args} sent to your System Messages."], None

        if command == consts.ACTION_RESEARCH_READ:
            if not action_args:
                return ["Please specify a storage path to read."], None

            path_text, page = self._parse_path_and_page(action_args)
            ok, resolved_path, display_path, consumes = self._resolve_storage_target(
                agent_data,
                path_text,
                consts,
                current_tick=current_tick,
                station_instance=room_context.station_instance,
            )
            if not ok:
                return [resolved_path], None

            if not os.path.exists(resolved_path) or not os.path.isfile(resolved_path):
                return [f"Storage file not found: `{display_path}`."], None

            if consumes:
                can_read, cooldown_msg = self._is_read_available(agent_data, consts, current_tick)
                if not can_read:
                    return [cooldown_msg or "Read cooldown active."], None

            try:
                content = file_io_utils.load_text(resolved_path)
                if content is None:
                    with open(resolved_path, "r", encoding="utf-8", errors="replace") as handle:
                        content = handle.read()
            except Exception as exc:
                return [f"Could not read `{display_path}`: {exc}"], None

            if consumes:
                self._consume_read_window(agent_data, consts, current_tick)

            room_context.agent_manager.add_pending_notification(
                agent_data,
                self._format_storage_read(display_path or path_text, content or "", page),
            )
            return [f"File content for {display_path} sent to your System Messages."], None

        if command == consts.ACTION_RESEARCH_STORAGE:
            if not action_args:
                return ["Please specify a storage action: `info` or `list <path> [page]`."], None

            parts = str(action_args).split(" ", 1)
            storage_action = parts[0].lower()
            remaining = parts[1] if len(parts) > 1 else ""

            if storage_action == "info":
                room_context.agent_manager.add_pending_notification(
                    agent_data,
                    self._build_storage_info(
                        agent_data,
                        consts,
                        current_tick=current_tick,
                        station_instance=room_context.station_instance,
                    ),
                )
                return ["Storage information sent to your System Messages."], None

            if storage_action == "list":
                path_text, page = self._parse_path_and_page(remaining)
                ok, resolved_path, display_path, _ = self._resolve_storage_target(
                    agent_data,
                    path_text,
                    consts,
                    current_tick=current_tick,
                    station_instance=room_context.station_instance,
                )
                if not ok:
                    return [resolved_path], None
                if not os.path.exists(resolved_path) or not os.path.isdir(resolved_path):
                    return [f"Storage directory not found: `{display_path}`."], None

                entries: List[str] = []
                try:
                    for entry in sorted(os.listdir(resolved_path)):
                        full_path = os.path.join(resolved_path, entry)
                        suffix = "/" if os.path.isdir(full_path) else ""
                        entries.append(entry + suffix)
                except Exception as exc:
                    return [f"Could not list `{display_path}`: {exc}"], None

                room_context.agent_manager.add_pending_notification(
                    agent_data,
                    self._format_storage_listing(display_path or path_text, entries, page),
                )
                return [f"Directory listing for {display_path} sent to your System Messages."], None

            if storage_action == "read":
                return self.handle_action(
                    agent_data,
                    consts.ACTION_RESEARCH_READ,
                    remaining,
                    yaml_data,
                    room_context,
                    current_tick,
                )

            if storage_action in {"write", "delete"}:
                return ["Persistent storage is read-only for agents in the current Research Center workflow."], None

            return ["Unknown storage action. Use `storage info` or `storage list <path> [page]`."], None

        if command == consts.ACTION_RESEARCH_RANK:
            valid_modes = ["id", "author"] if consts.RESEARCH_NO_SCORE else ["id", "score", "author"]
            if not action_args or action_args not in valid_modes:
                if consts.RESEARCH_NO_SCORE:
                    return ["Please specify sort mode: `id` or `author`."], None
                return ["Please specify sort mode: `id`, `score`, or `author`."], None
            room_state[consts.AGENT_RESEARCH_SORT_KEY] = action_args
            room_state[consts.AGENT_RESEARCH_PAGE_KEY] = 1
            return [f"Evaluations now sorted by {action_args}."], None

        if command == consts.ACTION_RESEARCH_FILTER:
            if not action_args:
                return ["Please specify a tag to filter by."], None
            room_state[consts.AGENT_RESEARCH_FILTER_TAG_KEY] = action_args.strip().lower()
            room_state[consts.AGENT_RESEARCH_PAGE_KEY] = 1
            return [f"Filtered by tag: {action_args.strip().lower()}."], None

        if command == consts.ACTION_RESEARCH_UNFILTER:
            room_state.pop(consts.AGENT_RESEARCH_FILTER_TAG_KEY, None)
            room_state[consts.AGENT_RESEARCH_PAGE_KEY] = 1
            return ["Tag filter removed. Showing all submissions."], None

        if command == consts.ACTION_RESEARCH_PREVIEW:
            if not action_args:
                return ["Please specify evaluation ID(s) to preview or `all`."], None

            visible_evaluations = self._filter_visible_evaluations(
                agent_data, consts, current_tick, self._load_all_display_evaluations()
            )
            visible_ids = {str(item.get(consts.EVALUATION_ID_KEY, "")) for item in visible_evaluations}

            if action_args.strip().lower() == "all":
                sorted_evals = self._sort_evaluations(visible_evaluations, "id", agent_data.get(consts.AGENT_NAME_KEY, ""))
                eval_ids = [str(item.get(consts.EVALUATION_ID_KEY, "")) for item in sorted_evals[:100]]
            else:
                eval_ids, error_message = self._parse_evaluation_range(action_args)
                if error_message:
                    return [f"Preview error: {error_message}"], None
                eval_ids = [eval_id for eval_id in eval_ids if eval_id in visible_ids][:100]

            preview_parts = self.eval_manager.build_evaluation_previews(eval_ids)
            if not preview_parts:
                return ["No evaluations found to preview."], None
            room_context.agent_manager.add_pending_notification(agent_data, "\n\n---\n\n".join(preview_parts))
            return [f"Preview for {len(preview_parts)} evaluation(s) sent to your System Messages."], None

        if command == consts.ACTION_RESEARCH_PAGE_SIZE:
            if not action_args:
                return ["Please specify page size (1-200)."], None
            try:
                page_size = int(action_args)
            except ValueError:
                return ["Page size must be a valid integer."], None
            if page_size < 1 or page_size > 200:
                return ["Page size must be between 1 and 200."], None
            room_state[consts.AGENT_RESEARCH_PAGE_SIZE_KEY] = page_size
            room_state[consts.AGENT_RESEARCH_PAGE_KEY] = 1
            return [f"Page size set to {page_size}."], None

        if command == consts.ACTION_CAPSULE_PAGE:
            if not action_args:
                return ["Please specify a page number."], None
            try:
                page_number = int(action_args)
            except ValueError:
                return ["Page number must be a valid integer."], None
            if page_number < 1:
                return ["Page number must be 1 or greater."], None
            room_state[consts.AGENT_RESEARCH_PAGE_KEY] = page_number
            return [f"Moved to page {page_number}."], None

        return [f"Action '{action_command}' not recognized in the Research Center."], None

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        consts = room_context.constants_module
        help_text = _RESEARCH_CENTER_HELP.replace(
            "__ACTIVE_EVALUATION_LIMIT_HELP__",
            self._build_active_limit_help(consts),
        )
        if not consts.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS:
            help_text = help_text.replace(
                "- `storage/shared/`\n  Shared persistent storage for reusable code or artifacts that may be useful across lineages.\n"
                "  Immature agents cannot access `storage/shared`; use your own lineage storage until you mature.\n"
                "- `storage/system/`\n  Official read-only system files and reference resources.\n",
                "- `storage/shared/`\n  Shared persistent storage for reusable code or artifacts.\n"
                "  Immature agents cannot access `storage/shared`; use your own lineage storage until you mature.\n"
                "- `storage/system/`\n  Official read-only system files and reference resources.\n"
                "- Cross-lineage storage reads may be restricted by station policy.\n",
            )
        return help_text
