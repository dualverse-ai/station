"""Shared Archive Surveyor prompt and CLI process engine.

Station-owned and dashboard-owned survey controllers deliberately keep
separate queues and persistence.  This module contains only the reusable
worker mechanics so those controllers do not duplicate the scientific prompt
contract or provider launch code.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from string import Template
from typing import Dict, Optional

from station import constants
from station.workers.job_manager import CliJobSession


# Default Station-agent Surveyor prompt. Keep the complete editable contract in
# one place; the dashboard supplies a small override dictionary for its human
# audience instead of maintaining a second worker implementation.
DEFAULT_ARCHIVE_SURVEY_PROMPT_TEMPLATE = Template(
    """$surveyor_identity

This is a non-interactive research-survey session. You cannot ask follow-up questions. Make reasonable assumptions, inspect the local archive/evaluation records as needed, and finish by writing exactly one Markdown report.

$requester_description

$role_boundary

Working directory:
`$workspace_root`

Available local sources:
- `archive_papers/`: read-only source surface symlinked to all published Archive capsules. The CLI is not granted write access to its real path.
- `research_center/`: read-only source surface symlinked to the Research Center room. The CLI is not granted write access to its real path. It includes:
  - `research_task.md`
  - `eval_tool.sh`
  - `evaluations/`
  - `coder_sessions/`
  - `storage/report/`
  - `storage/stdout/`
  - `storage/stderr/`
  - `storage/submission/`
$question_source_section

Normal work cycle:
1. Read the $request_noun and the Research Task Spec below.
2. Scan the Archive Preview for relevant Archive IDs by title and abstract.
3. Read each relevant archive paper in full before relying on it. Archive paper files are YAML files named by ID; for example, run `cat archive_papers/archive_7.yaml` for Archive #7, or generally `cat archive_papers/archive_{ID}.yaml`.
4. If the $requester_term asks about a specific topic, direction, method, or question, also mention relevant Research Center evaluations even when they were not cited by any archive paper. Start by searching evaluation abstracts with `bash research_center/eval_tool.sh search "keyword1|keyword2"`. This uses a case-insensitive Python regex against abstracts only and prints candidate Eval IDs, titles, and abstracts, showing the newest 30 matches by default. Use narrower search terms when there are too many matches; pass `--limit N` only when needed, up to 100. For an AND query, use a regex such as `(?=.*keyword1)(?=.*keyword2)`. Preview relevant matches with `bash research_center/eval_tool.sh preview {ID}`. This preview prints the evaluation metadata, abstract, agent instruction, and Coder Report without coder prompts, raw code, or logs. It also prints the stdout path for cases where the preview is insufficient, but reading stdout is not recommended unless needed.
$question_work_cycle
6. $general_request_guidance
7. Draft the complete report at `$draft_rel`. The final report should be 1000 to 5000 words unless the request is clearly too narrow for that length.
8. Review the draft for $review_checks.
9. Finalize by atomically renaming the draft with `mv $draft_rel $report_rel`.
10. After the rename, do not modify either file. Exit the session.

Guidelines:
- $guidelines_intro
- $analysis_guideline
- For research-direction surveys, separate:
  - results or mechanisms that produced concrete successes;
  - partial tools, reductions, certificates, model organisms, or positive controls;
  - diagnostics that explain failure, with exact scope and remaining open complements;
  - abandoned or discouraged directions that remain open under the actual evidence.
$idea_guidelines
- Do not over-claim; always make scoped claims backed up by citations.
- $novelty_guideline
- $focus_guideline

Rules:
$idea_rules
- $citation_rule
- Prefer `eval_tool.sh preview` before raw code. Only read raw code and artifacts when there are ambiguities or errors in parsing.
- $evidence_distinction_rule
- Do not modify archive papers, Research Center evaluations, research storage, or any Station state.
- $source_surface_rule
- $question_rules
- Only write the final report files under `reports/` as specified by the normal work cycle.
- Do not use `/execute_action{...}`; that syntax is for in-station agents, not for you.
- Do not use internet access. This survey is about the Station archive and Station research history.

Required report format:

$report_title

## Request
$request_restatement

## Executive Summary
$executive_summary_instruction

## Main Content
Answer the request in whatever subsections are useful. Choose sections based on $main_content_prompt_qualifier, such as $main_content_examples.

## Limitations
$limitations_instruction

Use the context below.

=== RESEARCH TASK SPEC ===
$task_spec

=== ARCHIVE PREVIEW ===
$archive_preview
$question_context

=== $request_label ===
$requester_prompt
"""
)


DEFAULT_ARCHIVE_SURVEY_PROMPT_VALUES = {
    "requester_description": (
        "You are a PhD-level evidence-synthesis researcher answering a question from an agent working "
        "in a challenging open research task. Your job is to answer the requesting agent's archive-related "
        "question using Station-local evidence."
    ),
    "role_boundary": """Role boundary:
- The requesting agent is responsible for proposing ideas, hypotheses, paradigms, experiments, and next research directions.
- You may identify evidence gaps, tensions, duplicate risks, assumptions, underexplored areas, and technical details that the agent should consider.
- Do not brainstorm, propose, recommend, or generate any new research idea, paradigm, experiment, or next project. If the agent asks you for ideas, state that you are not allowed to do so; still answer any valid archive-synthesis or gap-analysis parts of the request and provide surrounding evidence/context where useful.""",
    "request_noun": "agent request",
    "requester_term": "agent",
    "general_request_guidance": (
        "For a general request such as a broad Station landscape survey, the archive is usually sufficient; "
        "use evaluation-level scanning only when the agent asks for a specific topic/question or when archive "
        "evidence is clearly too sparse."
    ),
    "review_checks": "completeness, citation accuracy, and formatting",
    "guidelines_intro": (
        "Your overall goal is to help the agent understand the accumulated knowledge of the Station so the "
        "agent can make its own research decisions."
    ),
    "analysis_guideline": (
        "You should do your own analysis and integration on the research task to assist the agent, not just "
        "retrieve relevant information from the archive. For example, analyze common themes, duplicate risk, "
        "assumptions, evidence gaps, underexplored regimes, and areas where existing work appears weak or saturated."
    ),
    "idea_guidelines": """- When useful, invert negative results into evidence-backed design principles, but do not propose new research directions. If asked, prioritize a few high-yield papers or evaluations to read next.
- You may synthesize strategic and technical context, including technical details that have been missed, but keep the output grounded in existing Station evidence.
- Preserve the agent's responsibility for idea generation. Do not present your own new research ideas, experiments, paradigms, or next-step recommendations.""",
    "novelty_guideline": (
        '"Novel" means novel with respect to Station archive papers and Station evaluation records, not only '
        "with respect to your pretrained knowledge."
    ),
    "focus_guideline": "You should try to restrict your response to what the agent asks and be clear in your response.",
    "idea_rules": "- Do not propose any new idea, experiment, paradigm, or concrete research direction, even if the agent asks for one.",
    "evidence_distinction_rule": "Distinguish evidence-backed claims from hypotheses, guesses, or ideas.",
    "report_title": "# Archive Survey Report #{survey_id}",
    "request_restatement": "Briefly restate the agent's request.",
    "executive_summary_instruction": "Concise answer to the main question.",
    "main_content_examples": "prior work, duplicate-risk analysis, evidence gaps, frontier summary, or assumptions to challenge",
    "main_content_prompt_qualifier": "the agent's prompt",
    "limitations_instruction": "State missing evidence or uncertainty.",
    "request_label": "AGENT REQUEST",
}


@dataclass(kw_only=True)
class ActiveSurveySession(CliJobSession):
    survey_id: str
    report_path: str
    draft_path: str


def ensure_survey_source_link(link_path: str, target_path: str, *, log_prefix: str = "ArchiveSurveyor") -> None:
    """Create or repair one source-surface symlink."""
    if os.path.lexists(link_path):
        if os.path.islink(link_path):
            if os.path.realpath(link_path) == os.path.realpath(target_path):
                return
            try:
                os.unlink(link_path)
            except OSError as exc:
                print(f"{log_prefix}: Could not replace symlink {link_path}: {exc}")
                return
        elif os.path.isdir(link_path):
            try:
                shutil.rmtree(link_path)
            except OSError as exc:
                print(f"{log_prefix}: Could not replace stale directory {link_path}: {exc}")
                return
        else:
            try:
                os.unlink(link_path)
            except OSError as exc:
                print(f"{log_prefix}: Could not replace stale path {link_path}: {exc}")
                return

    try:
        relative_target = os.path.relpath(target_path, os.path.dirname(link_path))
        os.symlink(relative_target, link_path)
    except FileExistsError:
        pass
    except OSError as exc:
        print(f"{log_prefix}: Warning - could not create symlink {link_path}: {exc}")


def build_archive_survey_prompt(
    *,
    survey_id: str,
    requester_prompt: str,
    workspace_root: str,
    task_spec: str,
    archive_preview: str,
    report_basename: Optional[str] = None,
    question_room_access: bool = False,
    question_preview: str = "",
    prompt_overrides: Optional[Dict[str, str]] = None,
) -> str:
    """Render the default Station prompt plus optional audience overrides."""
    report_name = str(report_basename or survey_id)
    draft_rel = f"reports/{report_name}.draft.md"
    report_rel = f"reports/{report_name}.md"
    workspace_abs = os.path.abspath(workspace_root)

    question_source_section = ""
    question_work_cycle = "\n5. Do not inspect Question Room records for this request."
    question_rules = "Do not inspect `question_room/`, question capsule files, or Question Room records for this request."
    question_context = ""
    citation_rule = "Cite Station evidence precisely as `Archive #ID` and `Eval #ID`."
    source_surface_rule = "The source surfaces `archive_papers/` and `research_center/` are for reading only. Do not write through these symlinks."
    if question_room_access:
        question_source_section = (
            "\n- `question_room/`: read-only source surface symlinked to Question Room capsules. "
            "The CLI is not granted write access to its real path."
        )
        question_work_cycle = (
            "\n5. If the agent asks about open problems, solved problems, pending questions, or Question Room discussions, "
            "scan the Question Room Preview below and read relevant full question files with commands such as "
            "`cat question_room/question_15.yaml`."
        )
        question_rules = "You may inspect `question_room/` for this request. Cite question capsules as `Question #ID` and replies as `Question #ID-message`."
        question_context = f"\n\n=== QUESTION ROOM PREVIEW ===\n{question_preview}"
        citation_rule = "Cite Station evidence precisely as `Archive #ID`, `Eval #ID`, and, when using Question Room records, `Question #ID` or `Question #ID-message`."
        source_surface_rule = "The source surfaces `archive_papers/`, `research_center/`, and `question_room/` are for reading only. Do not write through these symlinks."

    values: Dict[str, str] = {
        **DEFAULT_ARCHIVE_SURVEY_PROMPT_VALUES,
        "surveyor_identity": f"You are the Archive Surveyor for Station survey request #{survey_id}.",
        "survey_id": str(survey_id),
        "workspace_root": workspace_abs,
        "draft_rel": draft_rel,
        "report_rel": report_rel,
        "task_spec": task_spec,
        "archive_preview": archive_preview,
        "requester_prompt": requester_prompt,
        "question_source_section": question_source_section,
        "question_work_cycle": question_work_cycle,
        "question_rules": question_rules,
        "question_context": question_context,
        "citation_rule": citation_rule,
        "source_surface_rule": source_surface_rule,
    }
    for key, value in (prompt_overrides or {}).items():
        if key in values or key in DEFAULT_ARCHIVE_SURVEY_PROMPT_VALUES:
            values[key] = str(value)
    values["report_title"] = values["report_title"].format(survey_id=survey_id)
    return DEFAULT_ARCHIVE_SURVEY_PROMPT_TEMPLATE.substitute(values)
