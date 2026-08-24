"""Coder prompt access policy for Research Center submissions."""

from __future__ import annotations

from typing import Any, Dict, Optional


PHASE_IMMATURE = "immature"
PHASE_MATURE = "mature"


def build_coder_access_metadata(agent_data: Dict[str, Any], current_tick: int, consts) -> Dict[str, Any]:
    """Snapshot submit-time access phase for a Research Center coder session.

    If phase cannot be determined, default to mature as a permissive fallback.
    """

    isolation_ticks = getattr(consts, "AGENT_ISOLATION_TICKS", None)
    birth_tick = agent_data.get(getattr(consts, "AGENT_TICK_BIRTH_KEY", "tick_birth"))
    phase = PHASE_MATURE
    age_ticks: Optional[int] = None
    reason = "mature_by_default"

    if isolation_ticks is None:
        reason = "isolation_disabled"
    elif birth_tick is None:
        reason = "missing_birth_tick"
    else:
        try:
            age_ticks = int(current_tick) - int(birth_tick)
            if age_ticks < int(isolation_ticks):
                phase = PHASE_IMMATURE
                reason = "age_below_isolation_threshold"
            else:
                reason = "age_at_or_above_isolation_threshold"
        except (TypeError, ValueError):
            phase = PHASE_MATURE
            age_ticks = None
            reason = "phase_unknown_default_mature"

    metadata = {
        "phase": phase,
        "submitted_tick": int(current_tick) if isinstance(current_tick, int) else current_tick,
        "agent_birth_tick": birth_tick,
        "agent_age_ticks": age_ticks,
        "maturity_threshold_ticks": isolation_ticks,
        "reason": reason,
    }

    if getattr(consts, "EXTERNAL_COUNTER_ENABLED", False):
        tenure_threshold = getattr(consts, "MIN_AGENT_AGE_BEFORE_LEAVE", None)
        agent_status = agent_data.get(getattr(consts, "AGENT_STATUS_KEY", "status"))
        recursive_status = getattr(consts, "AGENT_STATUS_RECURSIVE", "Recursive Agent")
        tenure_age: Optional[int] = None
        tenure_threshold_value: Optional[int] = None
        try:
            if birth_tick is not None:
                tenure_age = int(current_tick) - int(birth_tick)
            if tenure_threshold is not None:
                tenure_threshold_value = int(tenure_threshold)
        except (TypeError, ValueError):
            tenure_age = None
            tenure_threshold_value = None
        is_tenured = bool(
            tenure_age is not None
            and tenure_threshold_value is not None
            and tenure_threshold_value > 0
            and agent_status == recursive_status
            and tenure_age >= tenure_threshold_value
        )
        metadata.update({
            "tenure_threshold_ticks": tenure_threshold,
            "is_tenured": is_tenured,
            "internet_access_allowed": is_tenured,
        })

    return metadata


def get_coder_access_phase(eval_data: Dict[str, Any]) -> str:
    access = eval_data.get("coder_access") or {}
    phase = str(access.get("phase") or PHASE_MATURE).strip().lower()
    if phase not in {PHASE_IMMATURE, PHASE_MATURE}:
        return PHASE_MATURE
    return phase


def is_coder_internet_allowed(eval_data: Dict[str, Any], consts=None) -> bool:
    if consts is not None and not getattr(consts, "EXTERNAL_COUNTER_ENABLED", False):
        return False
    access = eval_data.get("coder_access") or {}
    return access.get("internet_access_allowed") is True


def render_coder_internet_policy(eval_data: Dict[str, Any], consts=None) -> str:
    if not is_coder_internet_allowed(eval_data, consts):
        return ""

    return """Internet access
- This coder session has internet access because the submitting agent was tenured at submission time.
- Access external websites only when the agent's instruction explicitly requests an external lookup, external data download, or brief web survey. If the instruction does not request this, do not access external websites.
- Keep all internet activity within the scope requested by the agent. Do not browse opportunistically or expand the experiment's scope."""


def render_coder_access_policy(
    eval_data: Dict[str, Any],
    lineage: str,
    eval_id: str,
    consts=None,
) -> str:
    phase = get_coder_access_phase(eval_data)
    lineage = str(lineage or "unknown").lower()
    eval_id = str(eval_id)
    seed_bank_enabled = bool(getattr(consts, "RESEARCH_SEED_BANK_ENABLED", False)) if consts else False

    header = """Research access and filesystem policy
This policy is based on the submitting agent's phase at submission time. It is the only authoritative access policy for this coder session. If any later instruction, example, or historical note conflicts with this policy, follow this policy.
- Do not read or write anything outside this Research Center workspace; access authorized remote storage only through its listed storage paths or symlinks."""
    artifact_access = (
        f"- This evaluation's artifacts: `storage/submission/{eval_id}.py`, "
        f"`storage/stdout/{eval_id}.log`, `storage/stderr/{eval_id}.log`, "
        f"and `storage/report/{eval_id}.md`"
    )
    read_write_paths = [
        f"- `storage/{lineage}/...`",
        f"- `storage/tmp/{lineage}/...`",
    ]
    read_only_paths = [
        "- The active research task specification",
        "- `evaluators/...`",
        "- `storage/system/...`",
    ]

    if phase == PHASE_IMMATURE:
        read_write_paths.append(artifact_access)
        if seed_bank_enabled:
            read_only_paths.append(
                "- `storage/shared/seed_bank/...` (read-only; normally through "
                "`storage/system/seed_bank.py`; direct NPZ access only as permitted "
                "by the Seed Bank instructions)"
            )
        read_only_paths.extend(
            [
                f"- `evaluations/{eval_id}.yaml`",
                f"- Evaluation records whose top-level YAML field `lineage` is exactly `{lineage}`",
                "- System-authored evaluation records, identified by top-level `author: System` or `lineage: system`",
            ]
        )
        forbidden_paths = [
            (
                "- Other paths under `storage/shared/...`"
                if seed_bank_enabled
                else "- `storage/shared/...`"
            ),
            "- Other lineage storage, including `storage/<other_lineage>/...` and `storage/lineages/<other_lineage>/...`",
            "- Non-system evaluations from other lineages",
            "- Prior submissions, reports, stdout, stderr, or code from another lineage, even if they have high scores or seem useful",
        ]
        phase_sections = [
            """Metadata-only rule for evaluation files:
- Before reading any non-current evaluation record, inspect only the top-level `lineage` and `author` fields.
- If it is not same-lineage and not system-authored, do not read further.""",
            "Forbidden:\n" + "\n".join(forbidden_paths),
            "If you accidentally read off-limit content, do not quote it, summarize it, copy it, use it, preserve it in code, or mention conclusions derived from it. Treat it as unavailable and continue from allowed information only.",
        ]
    else:
        shared_write_access = (
            "- `storage/shared/...` except `storage/shared/seed_bank/...`"
            if seed_bank_enabled
            else "- `storage/shared/...`"
        )
        read_write_paths.extend([shared_write_access, artifact_access])
        do_not_modify_paths = ["- `storage/system/...`"]
        if seed_bank_enabled:
            read_only_paths.append(
                "- `storage/shared/seed_bank/...` (read-only; through "
                "`storage/system/seed_bank.py` or direct NPZ access when needed)"
            )
            do_not_modify_paths.append("- `storage/shared/seed_bank/...`")
        read_only_paths.extend(
            [
                "- Other lineage storage, including `storage/<lineage_name>/...` and `storage/lineages/<lineage_name>/...`",
                "- Evaluation records under `evaluations/...`",
            ]
        )
        do_not_modify_paths.extend(
            [
                "- Other lineage storage",
                "- `evaluations/...`",
                "- `evaluators/...`",
                "- Task spec files",
            ]
        )
        phase_sections = ["Do not modify:\n" + "\n".join(do_not_modify_paths)]

    sections = [
        header,
        f"Phase: {phase}",
        "Allowed read/write paths:\n" + "\n".join(read_write_paths),
        "Allowed read-only paths:\n" + "\n".join(read_only_paths),
        *phase_sections,
    ]
    return "\n\n".join(sections)


def render_import_guidance(phase: str, lineage: str) -> str:
    if phase == PHASE_IMMATURE:
        return f"""Importing lineage libraries
- Import helpers only from paths allowed by the Research access and filesystem policy above.
- For this lineage:
  ```python
  import sys
  sys.path.insert(0, "storage/{lineage}")
  ```
- Helper imports from this lineage's storage are supported when you use that `sys.path.insert(...)` entry.
- Do not import from `storage/shared` or another lineage while this evaluation is in the immature access phase."""

    return f"""Importing lineage and shared libraries
- Import helpers only from paths allowed by the Research access and filesystem policy above.
- For this lineage:
  ```python
  import sys
  sys.path.insert(0, "storage/{lineage}")
  ```
- Because this mature-phase policy allows `storage/shared`, you may also use:
  ```python
  sys.path.insert(0, "storage/shared")
  ```
- Helper imports from lineage and shared storage are supported when you use those `sys.path.insert(...)` entries.
- Shared helpers may be read and updated when the agent explicitly asks for shared reusable work."""


def render_workflow_inspection_step(phase: str) -> str:
    if phase == PHASE_IMMATURE:
        return (
            "4. Inspect only context allowed by the Research access and filesystem policy above. "
            "If a prior eval ID is referenced, apply that policy before reading the record."
        )
    return (
        "4. Inspect only context allowed by the Research access and filesystem policy above. "
        "If a prior eval ID is referenced, apply that policy before reading the record."
    )
