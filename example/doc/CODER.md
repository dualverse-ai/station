# Research Center Coder System

## Scope

This document describes the **required Research Center coder design** and the implementation contract the code should follow.

- It applies to the **Research Center only**.
- The **Theory Room** is separate and is intentionally out of scope here.
- This document is the source of truth for Research Center coder lifecycle semantics.
- If code behavior disagrees with this document, the code should be treated as wrong and brought back into alignment.

## Current Behavior Summary

The Research Center is now an instruction-to-code workflow:

- Agents submit an **instruction prompt** instead of raw Python.
- A room-owned **coder** session implements the instruction, runs the official attempt, debugs if needed, and writes a final `Coder Report`.
- The active research task is a **single markdown task spec** at `research_task.md`.
- Coder-only instructions may be embedded in `research_task.md` between `__CODER_ONLY_BEGIN__` and `__CODER_ONLY_END__`; those sections are hidden from agent-facing reads and review surfaces but remain visible to the coder prompt.
- Evaluation records are stored as **one YAML file per evaluation** at `evaluations/{id}.yaml`.
- Agents can have **at most one active experiment** in the Research Center at a time.
- Each evaluation allows **at most one active official attempt at a time** and up to **5 attempts total** by default.
- The global number of live coder workflows is limited by `RESEARCH_EVAL_MAX_PARALLEL_WORKERS`.
- Agents do not write Research Center storage directly. They interact through instructions, review, and optional read actions.

The system is optimized for **faithful implementation** of the agent instruction, not autonomous score chasing.

## Core Lifecycle Contract

The Research Center should use only three top-level evaluation states:

- `queued`
- `running`
- terminal states such as `completed`, `partial`, `failed`, `blocked`

Everything else should be treated as a substate under `running`, not as a separate top-level scheduling state.

This is the intended invariant:

- `RESEARCH_EVAL_MAX_PARALLEL_WORKERS` limits live coder workflows
- this includes retryable-infra `pending_resume` evaluations even before the replacement process is relaunched
- `attempt_queued` and `attempt_running` evaluations still consume coder capacity while the coder workflow is live
- attempt execution is controlled by CPU / GPU / central resource coordination, not by the coder launch limit
- dashboard display status should distinguish a queued official attempt (`attempt_queued`) from one that has started on resources (`attempt_running`)
- frontend running count should reflect active top-level `running` evaluations, not only live PIDs and not only coder launch capacity
- restart and manual recovery should move unfinished `running` evaluations back to `queued`
- the central queue launcher should then relaunch from `queued` subject to the coder limit

Recommended running substates are:

- `coder_running`
- `attempt_running`
- `pending_resume`
- `resuming`

But those are all still `running` for:

- active-experiment and restart control
- frontend counting
- submit gating
- restart / recovery behavior

## Agent Interface

### Submission Model

Agents submit one focused experiment at a time with YAML fields:

```yaml
title: "Short method title"
tags: "tag1, tag2"
abstract: "Short summary of the intended experiment."
instruction: |
  Clear implementation instructions for the coder.
```

Each submission corresponds to **one experiment**.

- Newly created agents must read the current research task with `read_task` before their first `submit`. Older agents without the new read-tracking field are grandfathered.
- Do not ask the coder to run multiple unrelated experiments in one submission.
- Do not ask for parameter sweeps such as “try 5 variants”.
- If you want multiple experiments, submit them as multiple evaluations.

### Room Actions

The active Research Center actions are:

- `read_task`
- `submit`
- `review <evaluation_id>`
- `read_code <evaluation_id>`
- `read <path> [page]`
- `storage info`
- `storage list <path> [page]`
- `rank id|score|author`
- `filter <tag>`
- `unfilter`
- `preview <ids>`
- `page_size <n>`
- `page <n>`

### Cooldown Rules

`read_task`:

- Reads the active research task specification.
- Has **no cooldown**.

`read` and `read_code`:

- Share one cooldown.
- The first accepted `read` or `read_code` in a tick opens a read window for that tick.
- During that tick, the agent may issue **multiple** `read` and `read_code` actions.
- After that tick ends, both actions go on cooldown for **5 ticks**.

Exceptions:

- `storage list` does **not** consume the cooldown.
- Reading from `storage/system/...` does **not** consume the cooldown.

### Agent-Facing Review vs Notification

This is an intentional distinction in the current code.

`review <evaluation_id>` includes:

1. Title / ID / tags / abstract
2. Original instruction prompt
3. Final score and secondary metrics
4. `Coder Report`
5. Final stdout from the last attempt

Completion notification includes:

1. Completion line
2. Final score and secondary metrics
3. `Coder Report`
4. Final stdout from the last attempt

Notification does **not** repeat the abstract or instruction prompt.

Long stdout is truncated for visible display using `RESEARCH_EVAL_LOG_MAX_CHARS` and wrapped in fenced code blocks.

### Agent Storage Access

Agents have read-only access to Research Center storage surfaces.

- Allowed: `read`, `storage info`, `storage list`, `read_code`
- Not allowed: direct write / edit / delete operations

The expected storage layout is:

- `storage/<your_lineage>/`
- `storage/shared/`
- `storage/system/`

Agents should refer to these paths in instructions when useful, but the coder is the main storage user.

## Coder Runtime

### Backends

The Research Center coder uses the shared CLI worker backend layer. In
agent-facing text, "coder" still means this Research Center workflow. The
current backend names remain:

- `codex`
- `claude`

Configured by:

- `RESEARCH_CODER_BACKEND`
- `RESEARCH_CODER_MODEL_NAME`
- `RESEARCH_CODER_TIMEOUT_SECONDS`
- `RESEARCH_CODER_MAX_ATTEMPTS`
- `RESEARCH_CODER_MAX_SPAWNS`
- `RESEARCH_CODER_MAX_RESUMES`
- `RESEARCH_CODER_RESUME_BACKOFF_SECONDS`
- `CLI_WORKER_TRANSCRIPT_IDLE_TIMEOUT_SECONDS`

Default coder timeout is **6 hours**.

### Backend Launch Behavior

Codex backend:

- Uses `codex exec`
- Runs in `--sandbox workspace-write`
- Uses `--json`
- Writes the raw backend stream to `coder_sessions/{session_id}/transcript.jsonl`
- Writes the final assistant message to `coder_sessions/{session_id}/last_message.txt`
- Is terminated by the shared CLI worker watchdog if `transcript.jsonl`
  does not grow for `CLI_WORKER_TRANSCRIPT_IDLE_TIMEOUT_SECONDS` seconds
  (default 30 minutes). This transcript-growth check is Codex-specific.

Claude backend:

- Uses the Claude CLI in streaming JSON mode
- Writes the raw backend stream to `coder_sessions/{session_id}/transcript.jsonl`
- Is not subject to the Codex transcript-growth watchdog.

Executable resolution:

1. `CODEX_BIN_PATH` or `CLAUDE_BIN_PATH`
2. `PATH`
3. common NVM bin directories

`deploy.sh` now detects and persists both `CODEX_BIN_PATH` and `CLAUDE_BIN_PATH` into `.env` when missing.

### Running Substates

Within the single top-level `running` state, the intended meanings are:

- `coder_running`
  - the coder process is alive and actively reading, writing, testing, or preparing an official attempt

- `attempt_running`
  - the coder process is still alive
  - an official attempt has already been submitted
  - this includes both:
    - waiting for central resources
    - actual evaluator execution
  - dashboard display should show `attempt_queued` while the latest official attempt is queued, and `attempt_running` once the evaluator has started it
  - it consumes coder capacity while the coder workflow is still live; a new queued coder may only enter the pool after an existing coder workflow releases its slot

- `pending_resume`
  - the coder process exited because of a retryable infrastructure failure
  - the evaluation remains top-level `running`
  - it still counts against the coder launch limit
  - the auto evaluator should relaunch the same backend session after the configured resume backoff and transition it to `resuming`
  - the default resume backoff is 5 minutes, 10 minutes, 20 minutes, 40 minutes, then 60 minutes for all later resumes

- `resuming`
  - the relaunched coder session is continuing the same interrupted backend session rather than starting a fresh one
  - this still counts as the same active coder work and should not be pushed back to `queued`

Rare edge case:

- if the coder has already exited after writing the report, but the latest official attempt has not fully settled yet
  - keep the evaluation top-level `running` until finalization is ready
  - do **not** create a separate scheduler state for this
  - do **not** count it as an active coder slot, because the coder process is already gone

The top-level scheduler and frontend should not invent additional active states beyond `queued` and `running`.

### Prompt Contents

Each evaluation launches a fresh coder context. The live prompt includes:

- non-interactive instruction that the coder must finish with a report
- active research task specification from `research_task.md`
- lineage `CODER.md`
- most recent 5 completed evaluations from the same lineage, or an explicit “none” message
- the agent instruction
- working directory
- detected station Python executable
- official submission path and log paths
- disposable lineage scratch path `storage/tmp/<lineage>`
- submit-time Research access and filesystem policy
- direct note that `/execute_action{...}` is for in-station agents, not for the coder

### Coder Access Rules

The coder prompt snapshots the submitting agent's Research access phase at submission time. This phase should not be recomputed later when the coder launches, because queued work must keep the information boundary that existed when the agent submitted it.

The generated prompt has one authoritative section named `Research access and filesystem policy`. Later operational sections should not restate or broaden permissions.

If phase cannot be determined, the prompt defaults to the mature phase.

Immature-phase access model:

- Read/write: `storage/<lineage>`, `storage/tmp/<lineage>`
- Read-only: `storage/system`, `evaluators/`, task spec files
- Evaluation records may be read only when their top-level YAML field `lineage` matches the submitting lineage, or when the record is system-authored (`author: System` or `lineage: system`)
- Before reading any non-current evaluation record, the coder should check only top-level `lineage` and `author` metadata
- Forbidden: `storage/shared`, other lineage storage, and non-system evaluations from other lineages
- If the coder accidentally reads off-limit content, it must ignore it and not quote, summarize, copy, or use it

Mature-phase access model:

- Read/write: `storage/<lineage>`, `storage/tmp/<lineage>`, `storage/shared`
- Read-only: `storage/system`, other lineage storage, `evaluations/`, `evaluators/`, task spec files
- System-authored evaluation records remain available

The coder is instructed to:

- use normal shell/read/write tools
- use `storage/<lineage>/data/` as the default output location when the agent asks to emit a file without giving a path
- prefer reusable lineage libraries when creating code useful for future submissions
- use `storage/tmp/<lineage>/eval_<id>/` for disposable per-evaluation work such as temporary test scripts, probes, generated intermediates, caches, and mutable workspaces
- use `storage/tmp/<lineage>/eval_<id>/sage/` for Sage/CAS caches and mutable workspaces
- avoid putting disposable work, Sage/CAS cache, or mutable workspace directories under persistent lineage storage
- import helpers only from paths allowed by the active phase. Immature coders should not import from `storage/shared`; mature coders may import and update shared helpers when the agent asks for shared reusable work.
- avoid `__file__`-relative path resolution for `storage/...`, because the evaluator copies the submission into a temporary sandbox file such as `run.py` before import

### Official Attempt Flow

The coder may use the detected station Python for local probing, but an official attempt must be started only through:

```bash
bash submit_eval.sh <eval_id>
```

Current behavior:

- stdout log: `storage/stdout/{id}.log`
- stderr log: `storage/stderr/{id}.log`
- final report: `storage/report/{id}.md`
- submission file: `storage/submission/{id}.py`
- evaluation YAML stores metadata and artifact paths; large stdout/stderr,
  report, and submission contents are not embedded in the YAML
- `submit_eval.sh` invokes `_internal/submit_eval_cli_snapshot.py`, a
  startup-generated helper that registers attempts through `EvaluationManager`
  so the evaluation YAML is updated atomically and the live manager can refresh
  its in-memory indexes from the changed file
- `eval_tool.sh search REGEX` invokes `_internal/eval_tool_cli_snapshot.py`,
  a startup-generated read-only helper that searches the SQLite evaluation
  index abstracts only and prints matching Eval IDs, titles, and abstracts
- `eval_tool.sh preview <eval_id>` uses the same helper to print metadata,
  abstract, the agent instruction, coder prompt snapshot, and Coder Report
  without raw code or logs

The system clears stdout/stderr only when an official attempt is accepted.

Rejected concurrent submissions do **not** wipe the active logs.

Current prompt policy also makes these points explicit:

- direct Python execution is for lightweight debugging, testing, and probing only
- computationally intensive work should go through the official submit path
- while an attempt is active, the coder should poll stdout/stderr using `sleep {time}` with intervals of 30s, 60s, 120s, 240s, 480s, then 600s for all later polls
- when a new attempt starts, the coder should first run `sleep 30`, then inspect both logs, then continue with the next interval in that sequence until completion
- if an agent asks for multiple experiments in one submission, the coder should execute only the first experiment
- in that case, the coder should state clearly in the report that the Research Center allows only one experiment per submission
- multiple official attempts are for debugging, not cross-attempt optimization
- if the implementation is faithful and there is no exception, a poor result is usually still a valid stopping point for returning control to the agent

Intended official attempt lifecycle:

1. evaluation starts in `queued`
2. coder launches and evaluation becomes top-level `running`
3. substate may be `coder_running`
4. coder submits an official attempt
5. top-level state remains `running`
6. substate may be `attempt_running`, including resource wait time
7. coder inspects the result, debugs if needed, and either resubmits or finishes
8. evaluation exits `running` only when it reaches a terminal state

Retryable infra interruption edge case:

1. the coder process exits without a report
2. the evaluation stays top-level `running`
3. substate becomes `pending_resume`
4. the scheduler waits until `coder.next_resume_timestamp` if a resume backoff was scheduled
5. the scheduler relaunches the interrupted backend session
6. substate becomes `resuming`

### Coder Report

The coder must always write `storage/report/{id}.md` before finishing.

Required sections in the current prompt:

- `Summary`
- `Final Result`
- `Files Changed`
- `Implementation Details`
- `Faithfulness And Confidence`
- `Major Deviations`
- `Miscellaneous`
- `Final Status`

### Transcript and Debug Artifacts

Transcript/debug saving is enabled by default via:

- `RESEARCH_CODER_DEBUG_DUMP_ENABLED = True`

Current artifacts:

- raw backend transcript:
  - `coder_sessions/{session_id}/transcript.jsonl`
- backend stderr:
  - `coder_sessions/{session_id}/stderr.txt`
- final assistant message for Codex:
  - `coder_sessions/{session_id}/last_message.txt`
- prompt snapshot:
  - `coder_sessions/{session_id}/prompt.txt`
- fresh respawns after a no-report crash/timeout:
  - the fresh respawn prompt lists prior `coder_sessions/{old_session_id}/...` folders for the same evaluation
  - prior session folders are not copied or symlinked; they remain in place under `coder_sessions/`
  - the coder is told to inspect these artifacts before restarting from scratch
- expanded debug dump under the Research Center room root:
  - `tmp/{eval_id}/dialogue.log`

## Runtime Layout

The active Research Center task template format is:

- `research_task.md`
- `baseline.yamll`
- `evaluators/evaluator.py`

## Migrating Old Task Templates

Older Research Center task bundles used the previous layout:

- `research_tasks.yaml`
- `pending_evaluations.yamll`
- `evaluators/task_1_evaluator.py`
- duplicated evaluator copies under `storage/system/`

The new layout is:

- `research_task.md`
- `baseline.yamll`
- `evaluators/evaluator.py`

Current repo status:

- all bundled research task templates under `example/research_*` and `example_private/research_*` have already been migrated to this layout
- this section is kept as guidance only for future cases where a user asks to migrate a newly added or externally copied old-format task bundle

### Mechanical Migration For New Bundles

The repository includes a helper script:

- `scripts/migrate/migrate_research_templates.py`

Use it when a user asks to migrate a newly introduced old-format research task bundle.

The migration performs these file-level changes:

- `research_tasks.yaml` -> `research_task.md`
- `pending_evaluations.yamll` -> `baseline.yamll`
- `evaluators/task_1_evaluator.py` -> `evaluators/evaluator.py`
- removes duplicated `storage/system/evaluator.py` copies from task templates
- removes checked-in legacy `storage/system/task_1_evaluator.py` symlinks or files from task templates

The script does not leave task templates depending on a checked-in system evaluator copy. Instead, the runtime now creates this symlink automatically at startup if missing:

- `storage/system/evaluator.py`

It points to:

- `evaluators/evaluator.py`

Migration note:

- do **not** keep `storage/system/task_1_evaluator.py` in migrated task bundles
- after the evaluator rename, that legacy path is usually a broken symlink in templates
- the runtime removes the live legacy path and recreates only `storage/system/evaluator.py`
- template bundles should therefore keep the evaluator only at `evaluators/evaluator.py`
- task specs should not mention the legacy evaluator filename or path; agent-facing references should use `storage/system/evaluator.py`

### Baseline Format

`baseline.yamll` is a persistent template input, not a consumable queue file.

Current behavior:

- it stays on disk after baseline seeding
- it may contain one or more baseline entries
- baselines are seeded and run directly by the evaluator at startup
- the baseline report is system-generated rather than coder-generated

### Manual Cleanup After Mechanical Migration

For the bundled task templates currently in this repo, the main migration pass is already complete.

Use this checklist only when migrating a newly added old-format task bundle. The migration script handles file layout and simple content rewrites, but it does not fully rewrite agent-facing wording.

The wording that typically still needs updating is:

- old raw-code submission phrasing such as “you should submit Python code”
- old task-id or multi-task phrasing
- old headings such as `Research Task 1`; the new single-task format should just say `Research Task`
- old action syntax like `read 1` or `submit 1`
- old concurrency wording such as more than one active experiment per agent
- agent-facing wording that should now say “your coder should submit ...”
- any stale references to direct agent coding, direct storage editing, or debugger behavior

Runtime-created or runtime-managed paths:

- `evaluations/{id}.yaml`
- `run_requests/{id}_attempt_{n}.yaml`
- `coder_sessions/{session_id}/...`
- `submit_eval.sh`
- `eval_tool.sh`
- `_internal/submit_eval_cli_snapshot.py`
- `_internal/eval_tool_cli_snapshot.py`

Persistent storage layout:

- `storage/shared/`
- `storage/system/`
- `storage/lineages/<lineage>/`
- lineage alias: `storage/<lineage>`
- `storage/submission/`
- `storage/stdout/`
- `storage/stderr/`
- `storage/report/`
- `storage/tmp/`

Important storage detail:

- physical lineage storage lives under `storage/lineages/<lineage>`
- coder-facing compatibility aliases such as `storage/<lineage>` are created automatically
- `RESEARCH_STORAGE_BASE_PATH` is still respected, so storage can be moved to shared disk and symlinked back into the room

The evaluator symlink is created automatically at startup if missing:

- `storage/system/evaluator.py`

It points to:

- `evaluators/evaluator.py`

## Evaluation Model

The active evaluation format is a flat YAML record per evaluation.

At a high level it stores:

- metadata: id, author, lineage, title, tags, abstract, instruction
- coder state
- attempt history
- final result snapshot
- notification state

The old Research Center version-chain model is no longer the active path.

### State Contract

The intended top-level evaluation state machine is:

- `queued`
- `running`
- `completed`
- `partial`
- `failed`
- `blocked`

All active coder-managed work must be represented as `running`.

The intended running substates are:

- `coder_running`
- `attempt_running`
- `resuming`

But these are subordinate labels only. For active-experiment gating, frontend counts, and restart recovery, the only top-level distinction that matters is:

- `queued`
- `running`
- terminal

### Top Submission

Top-submission tracking is computed from final scored evaluations.

The current `top_submission` payload includes:

- `evaluation_id`
- `title`
- `score`
- `agent_name`
- `submitted_tick`
- `tags`
- `abstract`
- `sort_key`

Tie-breaking currently favors the **earlier** submission when scores are equal.

## Baseline Behavior

System baselines are defined in:

- `baseline.yamll`

Current behavior:

- baselines are seeded at Research Center startup
- they are run directly by the evaluator without launching a coder
- their review/report uses a system-generated report that embeds the raw baseline code
- `baseline.yamll` remains on disk after seeding
- reseeding is prevented because the startup code skips any baseline whose evaluation ID already exists

Important current limitation:

- if a system baseline is interrupted mid-run, it is **not** automatically requeued on restart
- instruction-driven evaluations **are** requeued on restart

This is the current code behavior and is intentionally documented here.

## Restart and Shutdown Semantics

### Startup Recovery

On station startup, the Research Center recovery path should:

- find unfinished instruction-driven evaluations in top-level `running`
- move them back to `queued`
- clear stale live-process metadata
- mark any in-flight attempt as abandoned if needed
- remove stale run-request files for those evaluations
- kill matching coder processes for this station only
- kill matching sandbox wrapper processes for this station only
- let the normal central queue launcher relaunch them subject to the configured coder launch limit

System baselines are excluded from this restart requeue path.

### Shutdown Recovery

`stop.sh --force` and manual restart recovery should follow the same simplification:

- terminate active coder processes for this station
- leave unfinished evaluations recoverable as `queued`
- do not preserve a separate active scheduler state across restart

This means:

- active coder sessions are terminated
- their instruction prompts are placed back in queue
- the station can resume those evaluations after restart

Same-session backend resume is different:

- if the station process is still alive and a coder is resuming the same interrupted Codex session
- that work should remain top-level `running`
- it should not be rewritten to `queued`

### Station Scoping

Process cleanup is scoped to the current station instance.

The cleanup logic checks:

- station research root
- coder session directory
- sandbox wrapper environment variable `STATION_RESEARCH_ROOT`

This is intended to avoid killing coder or wrapper processes belonging to a different station on the same machine.

### Tick Boundary Behavior

The station still pauses tick advancement while there are active Research Center evaluations.

That behavior now works through the new evaluation manager state rather than the old debugger/version-chain model.

Under the intended semantics, “active Research Center evaluations” here means top-level `running`.

## Compatibility Surfaces

The current implementation preserves the public evaluation interfaces needed by other subsystems.

Main consumers:

- `station/eval_archive/auto_evaluator.py`
- `station/stagnation_protocol.py`
- `station/supervisor_utils.py`
- `station/station.py`
- `web_interface/static/js/dashboard.js`

The main integration contract is now centered on `EvaluationManager`, especially:

- `get_evaluation_display_info()`
- `get_evaluation_review_info()`
- `build_evaluation_previews()`
- `get_top_submission()`
- `get_evaluation_statistics()`
- `get_submission_payload()`

These interfaces are what Archive, stagnation tracking, supervisor views, and top-submission displays should read.

## Current Code Structure

Main files and responsibilities:

- `station/rooms/research_center.py`
  - room help text
  - agent action parsing
  - cooldown enforcement
  - evaluation list/review/read-code display

- `station/eval_research/evaluation_manager.py`
  - YAML evaluation storage
  - review / notification formatting
  - top-submission tracking
  - archive-facing submission payloads

- `station/eval_research/auto_evaluator.py`
  - evaluation loop
  - baseline seeding
  - coder scheduling
  - official attempt execution
  - notification dispatch

- `station/eval_research/coder_manager.py`
  - coder prompt construction
  - backend launch
  - transcript/debug capture
  - report detection
  - respawn up to `RESEARCH_CODER_MAX_SPAWNS`
  - orchestrator pause if no report is produced after the final spawn

- `station/workers/cli.py`
  - generic backend-specific CLI launch definitions for Codex and Claude,
    shared by the coder, surveyor, and other specialized local workers

- `station/eval_research/runtime_paths.py`
  - runtime directory layout
  - storage alias/symlink handling
  - evaluator symlink creation
  - runtime `submit_eval.sh` and `eval_tool.sh` installation
  - station Python detection

- `station/eval_research/submit_eval_cli.py`
  - source implementation mirrored into the startup-generated submit snapshot

- `station/eval_research/restart_evaluations.py`
  - startup/shutdown requeue logic for instruction-driven evaluations

- `station/eval_research/executor_sandbox.py`
  - official code execution wrapper
  - station-scoped wrapper environment tagging

- `scripts/restart_eval.py`
  - manual recovery CLI for restart / shutdown requeue behavior

- `deploy.sh`
  - detects CLI executables and persists `CODEX_BIN_PATH` / `CLAUDE_BIN_PATH`

## Current Help-Message Rules

The live Research Center help text currently tells agents that:

- the coder already has direct access to the task spec
- the coder is highly proficient and usually should not be micromanaged with raw-code handoff or raw-code auditing
- each submission is one experiment
- they should usually give a high-level description of the implementation, such as the algorithm name, key steps, or overall analysis flow
- they should include a separate `Key Hyperparameters` section when the experiment has meaningful hyperparameters
- that section should list the major hyperparameters to be used, and may be skipped when there are no meaningful hyperparameters
- the coder is not responsible for hyperparameter optimization across attempts, so tuning should happen across multiple submissions
- they are encouraged not to pass raw code to the coder and should trust the coder to write code
- if the method or algorithm is novel, they should be concrete about the implementation details
- they may explicitly write `No reference` when there is no useful prior evaluation
- `read_task` has no cooldown
- `storage list` does not consume the read cooldown
- multiple `read` / `read_code` actions may be used in one tick before cooldown starts

## Migration Scope Today

There is no standing repo-wide migration backlog for the bundled research task templates.

Current status:

- bundled task templates under both `example_private/research_*` and `example/research_*` already use the current single-task Research Center layout
- migration guidance in this document now primarily applies when a user asks to migrate a new old-format bundle into the repo
- task-specific editorial improvements may still happen later, but those are no longer part of the template migration project
- `example_private/research_epoch_hadamard` remains a clean reference for post-migration wording/style

Out of scope for template migration guidance:

- `example/station_*`
- `example_private/station_*`
