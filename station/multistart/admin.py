from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

from station import constants
from station.eval_archive.archive_preview import build_archive_preview
from station.multistart import state
from station.workers.cli import (
    apply_codex_proxy_overrides,
    get_cli_worker_backend,
)
from station.workers.job_manager import (
    CliJobLaunchSpec,
    CliJobManager,
    CliJobState,
)


ADMIN_REPORTS_DIR_NAME = "reports"
ADMIN_SELECTION_REPORT_FILENAME = "selection_report.md"
ADMIN_GUIDANCE_REPORT_FILENAME = "guidance_report.md"
ADMIN_SELECTED_FILENAME = "selected.txt"
ADMIN_TRANSCRIPT_FILENAME = "transcript.jsonl"
ADMIN_STDERR_FILENAME = "codex_stderr.txt"
ADMIN_LAST_MESSAGE_FILENAME = "last_message.txt"
ADMIN_SESSIONS_DIR_NAME = "sessions"
ADMIN_SELECTION_STATE_KEY = "admin_selection"
ADMIN_GUIDANCE_WORD_MIN = 500
ADMIN_GUIDANCE_WORD_MAX = 2000


class AdminSelectionAttemptsExhausted(RuntimeError):
    pass


def _format_score(score: Any) -> str:
    if score is None or score == "":
        return "-"
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return f"{float(score):.8g}"
    return str(score)


def _admin_model_name(job_dir: Path) -> str:
    """Resolve the admin model from the preserved pre-branch station config."""

    config = state.load_yaml_mapping(job_dir / state.ORIGIN_DIR_NAME / "constant_config.yaml")
    configured = config.get("MULTISTART_ADMIN_MODEL_NAME")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    return str(constants.MULTISTART_ADMIN_MODEL_NAME).strip()


def _word_count(text: str) -> int:
    return len([part for part in text.split() if part])


def _reports_dir(admin_dir: Path) -> Path:
    return admin_dir / ADMIN_REPORTS_DIR_NAME


def report_paths(admin_dir: Path) -> dict[str, Path]:
    reports = _reports_dir(admin_dir)
    return {
        "selection": reports / ADMIN_SELECTION_REPORT_FILENAME,
        "guidance": reports / ADMIN_GUIDANCE_REPORT_FILENAME,
        "selected": reports / ADMIN_SELECTED_FILENAME,
    }


def read_selected_seed(admin_dir: Path, total_seeds: int) -> int | None:
    selected_path = report_paths(admin_dir)["selected"]
    try:
        raw = selected_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw.isdigit():
        return None
    selected = int(raw)
    return selected if 1 <= selected <= total_seeds else None


def _is_stagnation_mode(mode: Any) -> bool:
    return str(mode or "").strip().lower() == "stagnation"


def reports_complete(admin_dir: Path, total_seeds: int, mode: Any = None) -> bool:
    paths = report_paths(admin_dir)
    for path in paths.values():
        if not path.is_file():
            return False
        try:
            if not path.read_text(encoding="utf-8").strip():
                return False
        except OSError:
            return False
    if read_selected_seed(admin_dir, total_seeds) is None:
        return False
    try:
        guidance_words = _word_count(paths["guidance"].read_text(encoding="utf-8"))
    except OSError:
        return False
    return ADMIN_GUIDANCE_WORD_MIN <= guidance_words <= ADMIN_GUIDANCE_WORD_MAX


def _seed_table(job_dir: Path, total_seeds: int) -> str:
    headers = ["Seed", "Data dir", "Station name", "Tick", "Top eval", "Top tick", "Top score", "Best-score agent"]
    rows: list[list[str]] = []
    for seed in range(1, total_seeds + 1):
        seed_dir = state.branch_dir(job_dir, seed)
        config = state.read_station_config(seed_dir)
        rows.append(
            [
                str(seed),
                seed_dir.name,
                str(config.get("station_name") or "-"),
                str(config.get("current_tick", "-")),
                str(config.get("top_evaluation_id") or "-"),
                str(config.get("top_tick") or "-"),
                _format_score(config.get("top_score")),
                str(config.get("top_agent_name") or "-"),
            ]
        )
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    lines = [
        " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))),
        " | ".join("-" * widths[index] for index in range(len(headers))),
    ]
    lines.extend(" | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) for row in rows)
    return "\n".join(lines)


def _task_spec(job_dir: Path) -> str:
    task_path = state.branch_dir(job_dir, 1) / "rooms" / "research" / "research_task.md"
    try:
        return task_path.read_text(encoding="utf-8")
    except OSError:
        return "(Task specification could not be read.)"


def _pre_branch_archive_section(mode: str, job_dir: Path, branch_tick: int) -> str:
    if not _is_stagnation_mode(mode):
        return ""
    archive_preview = build_archive_preview(state.origin_dir(job_dir) / "capsules" / "archive")
    return f"""
## Pre-Branch Archive Context

For this stagnation multistart, the archive below is from the shared station state before branching at tick {branch_tick}. Use it when judging novelty and duplicate risk.

In your evaluation of which seed to select, review the relevant archive before deciding. A promising lane should be discounted if it largely overlaps with existing archive work, even if the interviewed agent was unaware of that overlap. Scan all abstracts below. If any archive paper is relevant to a branch's main lane, read it in full before relying on the branch's novelty claim:

`cat origin_station_data/capsules/archive/archive_{{ID}}.yaml`

Archive abstracts:

{archive_preview}
"""


def _admin_prompt(job_state: dict[str, Any], job_dir: Path, python_bin: str) -> str:
    total_seeds = int(job_state.get("seed_count") or 0)
    seed_numbers = ", ".join(str(seed) for seed in range(1, total_seeds + 1))
    mode = str(job_state.get("mode") or "multistart")
    branch_tick = int(job_state.get("branch_tick") or 0)
    context = (
        "Each branch starts from the same initial station state."
        if mode == "init"
        else (
            f"Each branch starts from the same station state immediately before stagnation lane assignment at tick {branch_tick}. "
            f"For this stagnation multistart, all branches were branched from tick {branch_tick}, so anything before tick {branch_tick} "
            "is identical across branch data. You can still read pre-branch material for context, but it should not affect "
            "cross-branch comparison because it is shared."
        )
    )
    origin_location = (
        "- `origin_station_data`: the shared station state before branching, including the pre-branch archive."
        if _is_stagnation_mode(mode)
        else ""
    )
    read_only_sentence = (
        "Treat every `station_data_s*` and `origin_station_data` directory as read-only. "
        "Do not modify branch data or the origin snapshot."
        if _is_stagnation_mode(mode)
        else "Treat every `station_data_s*` directory as read-only. Do not modify branch data."
    )
    stagnation_selection_note = (
        "When estimating each branch's future probability, account for overlap with pre-branch archive work. "
        "If a branch is mainly rediscovering an existing archive result, diagnostic, or failed basin, discount it unless it adds "
        "a concrete new technical ingredient, stronger evidence, or a materially different route."
        if _is_stagnation_mode(mode)
        else ""
    )
    return f"""# Station Administrator Branch Selection

You are the station administrator responsible for controlling and monitoring the station. Multiple independent Station branches were spawned on the same Research Center task. Due to resource limits, only one branch can continue running.

Mode: {mode}
Branch tick: {branch_tick}

{context}

## Task Specification

The task specification is identical across all branches, so read it here and do not re-read copies from other branches.

<task_spec>
{_task_spec(job_dir).rstrip()}
</task_spec>
{_pre_branch_archive_section(mode, job_dir, branch_tick)}

## Branch Locations

All branch data is available through symlinks in your current workspace:

- `station_data_s1`, `station_data_s2`, ...
- Available branch numbers: {seed_numbers}
{origin_location}

{read_only_sentence} Your writable workspace is the admin directory itself, including `reports/`.

## Current Best Scores

{_seed_table(job_dir, total_seeds)}

## Existing Per-Station Helpers

Each branch has its own Research Center helper scripts under `station_data_s*/rooms/research/`.

- To preview a prior evaluation, run from that branch's research directory:
  `bash eval_tool.sh preview <ID>`
- The preview shows evaluation metadata, abstract, the agent instruction, Coder Report, and stdout path without printing raw code or full logs.
- If the preview, report, score, or status conflict or remain uncertain, inspect the raw evaluation YAML and stdout—and code or a read-only diagnostic if needed—rather than arbitrarily trusting one source or discarding the result.
- Do not run `submit_eval.sh`; you are not submitting new research evaluations.

## Rules

- No web access. Do not try to access external websites, package indexes, remote repositories, APIs, or network resources in any way.
- Do not install packages.
- Do not modify `station_data_s*` or any files in the source checkout.
- You may write notes, scripts, and local diagnostic outputs inside this admin workspace.
- Each branch's agent research journeys are summarized in `station_data_s*/interview.yamll`.
- You must read every `station_data_s*/interview.yamll` in full, without truncation.
- Agent raw dialogue is under `station_data_s*/dialogue_logs/`; read it only if interviews and evaluation previews leave an important uncertainty.
- Evaluation records are under `station_data_s*/rooms/research/evaluations/`.
- Research artifacts are under `station_data_s*/rooms/research/storage/`.
- Python executable: `{python_bin}`
- You may use `{python_bin}` for local testing or probing if needed. Local tests are advisory only. Do not run official Research Center attempts.

## Required Outputs

### Selection Report

To select the best branch, use this criterion: which branch has the largest probability of solving the task, or fulfilling the task goal, if continued to run?

When scores are identical or close enough that the scoreboard alone is not decisive, compare branches by meaningful output and future potential. Consider near-misses, new basins with similar score but better follow-up potential, material advances in understanding, and whether the agent interviews describe a credible plan. Extrapolate from the interviews, Coder Reports, evaluation artifacts, and branch plans. Run local diagnostic scripts to evaluate artifact potential if needed.

These are common selection biases that can lead to choosing the wrong branch. Guard against them explicitly:

- **Clean-narrative bias:** Do not favor an interview merely because it presents a coherent story, clean closure, or polished plan. Messy, unresolved work may contain a more promising breakthrough direction.
- **Local-rigor bias:** Do not favor rigorous but incremental, already-closed, or frontier-distant work over high-risk exploratory work solely because the former is easier to validate. Rigor matters, but rigor applied to a low-value direction may add little.
- **Auditability bias:** Do not equate ease of auditing with scientific promise. When a potentially important result is difficult to audit, investigate it carefully using the available reports, artifacts, code, and read-only diagnostics. Judge the result by its underlying scientific potential after auditing it, rather than dismissing it because verification requires more effort.
- **Literature-familiarity bias:** Do not favor standard or literature-established approaches merely because they are familiar. These are open research problems, so conventional methods may already be near their limits. Give genuinely novel mechanisms appropriate additional weight, while still testing their correctness and feasibility.

{stagnation_selection_note}

The Selection Report must contain a table comparing branches on score, active agent progress, understanding, diversity, novelty, and final estimated probability.

Store the Selection Report in `reports/selection_report.md`.

Store the final chosen branch number in `reports/selected.txt`. It should contain a single number, for example `1` for choosing `station_data_s1`.

The Selection Report is to be read by a human admin for regular monitoring. It should be around 1,000 to 2,000 words. Include a brief note on any local diagnostic you ran, or explicitly justify why no local diagnostic was needed.

### Guidance Report

After you select the branch, lessons and results from other archived stations should not be wasted. Write a post titled "Guidance from previous stations" that summarizes transferable lessons and results from unselected stations. It will be posted to the selected branch, where agents will read it.

Store the Guidance Report in `reports/guidance_report.md`.

The Guidance Report should:

- Be concise but complete: {ADMIN_GUIDANCE_WORD_MIN} to {ADMIN_GUIDANCE_WORD_MAX} words.
- Jump directly into the main content, without an introductory phrase.
- Do not give instructions; write in a tone similar to a literature survey.
- Balance high-level lessons with low-level results. Do not be overly abstract or overly detailed about implementation.
- Contain no negative lessons; state only positive, constructive lessons.
- Avoid overclaiming; use precise scope and appropriate qualifications where needed.
- Contain no branch references. Agents cannot see unselected branch data, so do not refer to branch numbers, unavailable agent names, paths, or artifacts from unselected branches.
- Include a section titled `Important Work From Previous Stations Not Observed in Current Station`. It should contain all high-potential lanes from unselected branches that should not be lost. Include enough technical detail for agents in the selected branch to reproduce each idea independently. When judging what is high-potential, take care not to fall into the four selection biases described above.

## Finalization

Before submitting and finalizing, read both reports in full—the Guidance Report and the Selection Report—and revise them twice to ensure they follow every instruction and contain no material errors or omissions.

After writing all required files, run:

```bash
./submit.sh
```
"""


def _validator_source(mode: Any = None) -> str:
    return f"""from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
reports = root / "{ADMIN_REPORTS_DIR_NAME}"
selection = reports / "{ADMIN_SELECTION_REPORT_FILENAME}"
guidance = reports / "{ADMIN_GUIDANCE_REPORT_FILENAME}"
selected = reports / "{ADMIN_SELECTED_FILENAME}"

errors: list[str] = []
for path in (selection, guidance, selected):
    if not path.is_file():
        errors.append(f"missing required file: {{path.relative_to(root)}}")
    elif not path.read_text(encoding="utf-8", errors="replace").strip():
        errors.append(f"empty required file: {{path.relative_to(root)}}")

selected_text = selected.read_text(encoding="utf-8", errors="replace").strip() if selected.is_file() else ""
if not re.fullmatch(r"\\d+", selected_text):
    errors.append("reports/selected.txt must contain exactly one integer branch number")
else:
    branch_dir = root / f"station_data_s{{int(selected_text)}}"
    if not branch_dir.is_dir():
        errors.append(f"selected branch directory does not exist: {{branch_dir.name}}")

if guidance.is_file():
    words = re.findall(r"\\b\\S+\\b", guidance.read_text(encoding="utf-8", errors="replace"))
    if len(words) < {ADMIN_GUIDANCE_WORD_MIN} or len(words) > {ADMIN_GUIDANCE_WORD_MAX}:
        errors.append(f"reports/guidance_report.md is {{len(words)}} words; expected {ADMIN_GUIDANCE_WORD_MIN} to {ADMIN_GUIDANCE_WORD_MAX}")

if selection.is_file():
    text = selection.read_text(encoding="utf-8", errors="replace")
    if "|" not in text:
        errors.append("reports/selection_report.md should include a comparison table")

if errors:
    print("Submission validation failed:")
    for error in errors:
        print(f"- {{error}}")
    sys.exit(1)

print(f"Submission validation passed. Selected branch: {{selected_text}}")
"""


def _detect_python_executable() -> str:
    candidates = [
        os.environ.get("PYTHON"),
        "/home/ubuntu/miniconda3/envs/station/bin/python",
        "/home/ubuntu/miniconda3/bin/python",
        shutil.which("python3"),
        shutil.which("python"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable or "python3"


def _detect_codex_executable() -> str | None:
    candidates = [
        os.environ.get("CODEX_BIN_PATH"),
        os.environ.get("CODEX_BIN"),
        shutil.which("codex"),
        shutil.which("ccodex"),
        str(Path.home() / "codex-standalone" / "bin" / "codex-configurable"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, parsed)


def _nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(0, parsed)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _selection_state_payload(job_dir: Path, fallback: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = state.load_job_state(job_dir) or dict(fallback)
    raw_selection = payload.get(ADMIN_SELECTION_STATE_KEY)
    selection = dict(raw_selection) if isinstance(raw_selection, dict) else {}
    selection.setdefault("backend", "codex")
    selection.setdefault("spawn_count", 0)
    selection.setdefault("resume_count", 0)
    selection.setdefault(
        "max_spawns",
        _positive_int(getattr(constants, "RESEARCH_CODER_MAX_SPAWNS", 3), 3),
    )
    selection.setdefault(
        "max_resumes",
        _nonnegative_int(getattr(constants, "RESEARCH_CODER_MAX_RESUMES", 14), 14),
    )
    selection.setdefault("sessions", [])

    admin_dir = state.admin_dir(job_dir)
    legacy_transcript = admin_dir / ADMIN_TRANSCRIPT_FILENAME
    if not raw_selection and legacy_transcript.is_file():
        backend_runner = get_cli_worker_backend("codex")
        resume_token = backend_runner.extract_resume_token(str(legacy_transcript))
        pending_resume = bool(resume_token)
        resume_delay = AdminSelectionManager.get_resume_backoff_delay_seconds(0) if pending_resume else 0
        selection.update(
            {
                "spawn_count": 1,
                "resume_count": 0,
                "resume_token": resume_token,
                "status": "pending_resume" if pending_resume else "pending_spawn",
                "resume_delay_seconds": resume_delay,
                "next_resume_timestamp": time.time() + resume_delay if pending_resume else None,
                "last_error": "Legacy admin selection attempt exited without valid reports.",
                "legacy_workspace_imported": True,
                "sessions": [
                    {
                        "session_id": "legacy_spawn_1",
                        "launch_mode": "fresh",
                        "spawn_count": 1,
                        "resume_count": 0,
                        "transcript_path": ADMIN_TRANSCRIPT_FILENAME,
                        "stderr_path": ADMIN_STDERR_FILENAME,
                    }
                ],
            }
        )

    # Matching Coder/Surveyor restart recovery: an interrupted live launch or
    # an explicitly restarted blocked job starts with a fresh retry budget.
    selection_status = str(selection.get("status") or "").lower()
    if selection_status in {"blocked", "running"}:
        selection["restart_recovery_count"] = int(selection.get("restart_recovery_count") or 0) + 1
        selection.update(
            {
                "status": "pending_spawn",
                "spawn_count": 0,
                "resume_count": 0,
                "resume_token": None,
                "next_resume_timestamp": None,
                "resume_delay_seconds": 0,
            }
        )

    if not selection.get("status"):
        selection["status"] = "pending_spawn"
    payload[ADMIN_SELECTION_STATE_KEY] = selection
    state.save_job_state(job_dir, payload)
    return payload, selection


def _save_selection_state(job_dir: Path, selection: dict[str, Any]) -> None:
    payload = state.load_job_state(job_dir)
    payload[ADMIN_SELECTION_STATE_KEY] = selection
    state.save_job_state(job_dir, payload)


def _ensure_workspace(job_state: dict[str, Any], job_dir: Path) -> Path:
    admin_dir = state.admin_dir(job_dir)
    if not admin_dir.exists():
        return prepare_workspace(job_state, job_dir)
    required = [
        admin_dir / "prompt.md",
        admin_dir / "validate_submission.py",
        admin_dir / "submit.sh",
        _reports_dir(admin_dir),
    ]
    if all(path.exists() for path in required):
        return admin_dir
    raise RuntimeError(f"existing admin selection workspace is incomplete: {admin_dir}")


def _resume_prompt() -> str:
    return """Resume the same Station administrator branch-selection session.

The prior Codex turn was interrupted by a transient backend/provider failure.
Continue from the existing admin workspace and reports rather than restarting the analysis.
Finish all required reports and run `./submit.sh` before exiting.
"""


def _fresh_respawn_prefix(selection: dict[str, Any]) -> str:
    if int(selection.get("spawn_count") or 0) <= 1:
        return ""
    return f"""Previous administrator selection attempts exited without valid final reports.

Their artifacts are preserved under `{ADMIN_SESSIONS_DIR_NAME}/`, and partial files may exist under `reports/`.
Inspect and reuse valid prior work, but independently verify the final comparison and selection.
Last recorded failure: {selection.get('last_error') or 'report missing'}

"""


class AdminSelectionManager(CliJobManager):
    JOB_ID = "admin"
    fresh_attempt_after_resume_exhaustion = True

    def __init__(
        self,
        *,
        job_state: dict[str, Any],
        job_dir: Path,
        admin_dir: Path,
        executable: str,
        selection: dict[str, Any],
    ) -> None:
        super().__init__()
        self.job_dir = job_dir
        self.admin_dir = admin_dir
        self.executable = executable
        self.selection = selection
        self.total_seeds = int(job_state.get("seed_count") or 0)
        self.mode = str(job_state.get("mode") or "multistart")
        self.selected_seed: Optional[int] = None

    @classmethod
    def _resume_backoff_schedule(cls) -> Any:
        return getattr(constants, "RESEARCH_CODER_RESUME_BACKOFF_SECONDS", [])

    def _refresh_selection(self) -> dict[str, Any]:
        payload = state.load_job_state(self.job_dir) or {}
        raw = payload.get(ADMIN_SELECTION_STATE_KEY)
        if isinstance(raw, dict):
            self.selection = dict(raw)
        return self.selection

    def _persist_selection(self) -> None:
        _save_selection_state(self.job_dir, self.selection)

    def _load_cli_job_state(self, job_id: str) -> CliJobState:
        selection = self._refresh_selection()
        status_text = str(selection.get("status") or "pending_spawn").lower()
        return CliJobState(
            backend="codex",
            spawn_count=int(selection.get("spawn_count") or 0),
            resume_count=int(selection.get("resume_count") or 0),
            max_spawns=_positive_int(
                selection.get("max_spawns"),
                _positive_int(getattr(constants, "RESEARCH_CODER_MAX_SPAWNS", 3), 3),
            ),
            max_resumes=_nonnegative_int(
                selection.get("max_resumes"),
                _nonnegative_int(getattr(constants, "RESEARCH_CODER_MAX_RESUMES", 14), 14),
            ),
            resume_token=str(selection.get("resume_token") or "").strip() or None,
            next_resume_timestamp=selection.get("next_resume_timestamp"),
            fresh_launch_eligible=status_text == "pending_spawn",
            resume_launch_eligible=status_text == "pending_resume",
        )

    def _format_cli_job_session_id(self, job_id, state_snapshot, decision):
        return (
            f"codex_spawn_{decision.spawn_count}_resume_{decision.resume_count}_"
            f"{uuid.uuid4().hex[:8]}"
        )

    def _claim_cli_job_launch(self, job_id, session_id, decision):
        self.selection.update(
            {
                "status": "running",
                "spawn_count": decision.spawn_count,
                "resume_count": decision.resume_count,
                "resume_token": decision.resume_token,
                "next_resume_timestamp": None,
                "resume_delay_seconds": 0,
                "launch_mode": decision.mode,
                "started_at": time.time(),
                "active_pid": None,
            }
        )
        self._persist_selection()
        return dict(self.selection)

    def _build_cli_job_launch_spec(self, job_id, session_id, decision, claimed):
        session_dir = self.admin_dir / ADMIN_SESSIONS_DIR_NAME / session_id
        prompt_text = (
            _resume_prompt()
            if decision.is_resume
            else _fresh_respawn_prefix(claimed) + _read_text(self.admin_dir / "prompt.md")
        )
        launch_env = os.environ.copy()
        apply_codex_proxy_overrides(launch_env)
        return CliJobLaunchSpec(
            executable=self.executable,
            run_dir=str(session_dir),
            backend="codex",
            model_name=_admin_model_name(self.job_dir),
            workspace_root=str(self.admin_dir),
            storage_root=str(self.admin_dir),
            prompt=prompt_text,
            env=launch_env,
        )

    def _mark_cli_job_pid(self, job_id: str, pid: int) -> None:
        self.selection["active_pid"] = pid
        self._persist_selection()

    def _on_cli_job_process_exited(self, job_id, session, returncode):
        record = {
            "session_id": session.session_id,
            "launch_mode": self.selection.get("launch_mode"),
            "spawn_count": self.selection.get("spawn_count"),
            "resume_count": self.selection.get("resume_count"),
            "transcript_path": os.path.relpath(session.transcript_path, self.admin_dir),
            "stderr_path": os.path.relpath(session.stderr_path, self.admin_dir),
            "last_message_path": (
                os.path.relpath(session.last_message_path, self.admin_dir)
                if session.last_message_path
                else None
            ),
            "started_at": self.selection.get("started_at"),
            "completed_at": time.time(),
            "returncode": returncode,
        }
        sessions = self.selection.setdefault("sessions", [])
        if isinstance(sessions, list):
            sessions.append(record)
        self.selection["last_session_id"] = session.session_id
        self.selection["last_returncode"] = returncode
        self.selection["active_pid"] = None
        self._persist_selection()

    def _cli_job_completion_ready(self, job_id, session):
        if not reports_complete(self.admin_dir, self.total_seeds, self.mode):
            return False
        submit_result = subprocess.run([str(self.admin_dir / "submit.sh")], cwd=self.admin_dir, check=False)
        return submit_result.returncode == 0 and reports_complete(self.admin_dir, self.total_seeds, self.mode)

    def _on_cli_job_completed(self, job_id, session, returncode):
        selected = read_selected_seed(self.admin_dir, self.total_seeds)
        if selected is None:
            raise RuntimeError("admin selected branch is invalid")
        self.selection.update(
            {
                "status": "completed",
                "completed_at": time.time(),
                "selected_seed": selected,
                "active_pid": None,
            }
        )
        self._persist_selection()
        self.selected_seed = selected

    def _cli_job_missing_report_reason(self, job_id, session):
        return f"Admin selection attempt exited without valid reports (return code {session.process.returncode})."

    def _schedule_cli_job_resume(self, job_id, session, failure):
        self.selection.update(
            {
                "status": "pending_resume",
                "resume_token": failure.resume_token,
                "resume_count": failure.resume_count,
                "resume_delay_seconds": failure.delay_seconds,
                "next_resume_timestamp": failure.next_resume_timestamp,
                "last_error": failure.reason,
                "failure_category": failure.category,
            }
        )
        self._persist_selection()
        state.append_job_log(
            self.job_dir,
            "admin selection scheduled same-session resume "
            f"{failure.resume_count + 1}/{self._load_cli_job_state(job_id).max_resumes} "
            f"after transient failure ({failure.reason})",
        )

    def _schedule_cli_job_fresh_attempt(self, job_id, session, failure):
        state_snapshot = self._load_cli_job_state(job_id)
        self.selection.update(
            {
                "status": "pending_spawn",
                "resume_count": 0,
                "resume_token": None,
                "resume_delay_seconds": 0,
                "next_resume_timestamp": None,
                "last_error": failure.reason,
                "failure_category": failure.category,
                "active_pid": None,
            }
        )
        self._persist_selection()
        state.append_job_log(
            self.job_dir,
            f"admin selection scheduled fresh spawn {state_snapshot.spawn_count + 1}/{state_snapshot.max_spawns}",
        )

    def _on_cli_job_attempts_exhausted(
        self,
        job_id,
        session,
        failure,
    ):
        state_snapshot = self._load_cli_job_state(job_id)
        reason = (
            "Multistart administrator selection exited without valid reports after "
            f"{state_snapshot.spawn_count} total fresh spawns and "
            f"{state_snapshot.resume_count} same-session resume attempts on the final spawn. "
            "Manual intervention is required."
        )
        self.selection.update(
            {
                "status": "blocked",
                "last_error": reason,
                "blocked_at": time.time(),
                "resume_token": failure.resume_token or None,
                "next_resume_timestamp": None,
                "active_pid": None,
            }
        )
        self._persist_selection()
        raise AdminSelectionAttemptsExhausted(reason)

    def _cli_job_idle_timeout_reason(self, job_id, session, idle_seconds, timeout_seconds):
        return (
            "Codex CLI transcript for multistart Admin selection did not grow for "
            f"{idle_seconds} seconds, exceeding the configured CLI worker transcript idle timeout "
            f"of {timeout_seconds} seconds."
        )

    def _on_cli_job_idle_timeout(self, job_id, session, reason, idle_seconds, timeout_seconds):
        self.selection["last_error"] = reason
        self.selection["failure_category"] = "codex_transcript_idle_timeout"
        self._persist_selection()

    def run(self) -> int:
        while True:
            if self.selected_seed is not None:
                return self.selected_seed
            if reports_complete(self.admin_dir, self.total_seeds, self.mode):
                selected = read_selected_seed(self.admin_dir, self.total_seeds)
                if selected is None:
                    raise RuntimeError("admin reports complete but selected branch is invalid")
                self.selection.update(
                    {"status": "completed", "completed_at": time.time(), "selected_seed": selected}
                )
                self._persist_selection()
                return selected

            launched = self.launch_cli_job(self.JOB_ID)
            if launched or self.active_sessions:
                while self.active_sessions:
                    self.poll_cli_jobs()
                    if self.active_sessions:
                        time.sleep(0.5)
                continue

            state_snapshot = self._load_cli_job_state(self.JOB_ID)
            remaining = self.get_resume_backoff_remaining_seconds(state_snapshot)
            if remaining > 0:
                time.sleep(min(remaining, 1.0))
                continue


def prepare_workspace(job_state: dict[str, Any], job_dir: Path) -> Path:
    admin_dir = state.admin_dir(job_dir)
    if admin_dir.exists():
        shutil.rmtree(admin_dir)
    admin_dir.mkdir(parents=True, exist_ok=True)
    _reports_dir(admin_dir).mkdir(parents=True, exist_ok=True)

    mode = str(job_state.get("mode") or "multistart")
    if _is_stagnation_mode(mode):
        origin_target = state.origin_dir(job_dir)
        if not origin_target.is_dir():
            raise RuntimeError(f"stagnation admin workspace requires origin station data: {origin_target}")
        relative_origin = os.path.relpath(origin_target, admin_dir)
        (admin_dir / origin_target.name).symlink_to(relative_origin, target_is_directory=True)

    total_seeds = int(job_state.get("seed_count") or 0)
    for seed in range(1, total_seeds + 1):
        target = state.branch_dir(job_dir, seed)
        relative_target = os.path.relpath(target, admin_dir)
        (admin_dir / target.name).symlink_to(relative_target, target_is_directory=True)

    python_bin = _detect_python_executable()
    (admin_dir / "prompt.md").write_text(_admin_prompt(job_state, job_dir, python_bin), encoding="utf-8")
    (admin_dir / "validate_submission.py").write_text(_validator_source(mode), encoding="utf-8")
    submit_path = admin_dir / "submit.sh"
    submit_path.write_text(
        f'#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n"{python_bin}" validate_submission.py\n',
        encoding="utf-8",
    )
    submit_path.chmod(submit_path.stat().st_mode | 0o111)
    return admin_dir


def run_selection(job_state: dict[str, Any], job_dir: Path) -> int:
    total_seeds = int(job_state.get("seed_count") or 0)
    mode = str(job_state.get("mode") or "multistart")
    admin_dir = state.admin_dir(job_dir)
    if reports_complete(admin_dir, total_seeds, mode):
        selected = read_selected_seed(admin_dir, total_seeds)
        if selected is None:
            raise RuntimeError("admin reports complete but selected branch is invalid")
        return selected

    executable = _detect_codex_executable()
    if not executable:
        raise RuntimeError("Codex executable not found. Set CODEX_BIN_PATH or add codex/ccodex to PATH.")

    admin_dir = _ensure_workspace(job_state, job_dir)
    _payload, selection = _selection_state_payload(job_dir, job_state)
    manager = AdminSelectionManager(
        job_state=job_state,
        job_dir=job_dir,
        admin_dir=admin_dir,
        executable=executable,
        selection=selection,
    )
    return manager.run()


def guidance_announcement(guidance_report: str) -> str:
    preamble = (
        "**System Message**\n\n"
        "To aid general understanding of the problem, we include a summary report from previous stations below. "
        "Please note:\n\n"
        "- This is not an instruction to pivot; continue focusing on your current work and incorporate these lessons only if you judge them useful.\n"
        "- If your current research direction is exhausted now or in future ticks, you may choose to pivot to these ideas; treat them as backup options if needed.\n"
        "- The lessons are mostly adapted from station interviews, so some understanding may be refined further and should not be treated as final doctrine.\n"
        "- This report is not citable in Archive papers; it is intended as general guidance only."
    )
    return f"{preamble}\n\n{guidance_report.strip()}\n\nSystem Admin\n"
