# Research Center Coder System

## Scope

This document describes the **required Research Center coder design** and the implementation contract the code should follow.

- It applies to the **Research Center only**.
- This document is the source of truth for Research Center coder lifecycle semantics.
- If code behavior disagrees with this document, the code should be treated as wrong and brought back into alignment.

## Current Behavior Summary

The Research Center is now an instruction-to-code workflow:

- Agents submit an **instruction prompt** instead of raw Python.
- A room-owned **coder** session implements the instruction, runs the official attempt, debugs if needed, and writes a final `Coder Report`.
- The active research task is a **single markdown task spec** at `research_task.md`.
- Coder-only instructions may be embedded in `research_task.md` between `__CODER_ONLY_BEGIN__` and `__CODER_ONLY_END__`; those sections are hidden from agent-facing reads and review surfaces but remain visible to the coder prompt.
- Evaluation records are stored as **one YAML file per evaluation** at `evaluations/{id}.yaml`.
- Agents can have up to `RESEARCH_MAX_CONCURRENT_SUBMISSIONS` active experiments in the Research Center at a time.
- Each evaluation allows **at most one active official attempt at a time** and up to **5 attempts total** by default. If a non-final independent audit fails after that budget is exhausted, the same evaluation receives one additional official attempt for the required repair; the budget is not reset or shared across evaluations.
- The global number of live coder workflows is limited by `RESEARCH_EVAL_MAX_PARALLEL_WORKERS`.
- Agents do not write Research Center storage directly. They interact through instructions, review, and optional read actions.

After a coder writes its report, a fresh independent auditor reviews the submitted code,
official result, logs, and report before finalization. The auditor writes
`storage/audit/{id}.md` and submits `bash submit_audit.sh {id} pass|fail`.
Minor wording or scope qualifications are a pass and are appended to the Coder Report.
A material scientific error is a fail; the report is appended and the coder is relaunched
for repair. `RESEARCH_CODER_AUDIT_MAX_ROUNDS` defaults to two total audits: the initial audit
and, after one coder repair, one final re-audit. If the final audit still finds a material
error, the evaluation finalizes as `partial`. Auditor and coder infrastructure spawn budgets
are enforced by the same shared `CliJobManager`: both use
`RESEARCH_CODER_MAX_SPAWNS`, `RESEARCH_CODER_MAX_RESUMES`, and
`RESEARCH_CODER_RESUME_BACKOFF_SECONDS`. Their persisted counters remain independent per stage
and reset between scientific rounds.
Auditor infrastructure crashes do not consume audit rounds; persisted audit state allows
restart to relaunch only the unfinished stage.
The final configured auditor is told that no further coder repair is available and that its
report will go directly to the submitting agent. Whether passing or failing, its report is
limited to at most two concise paragraphs. Completion notifications include the base Coder
Report and only the latest auditor report; complete prior audit history remains in the on-disk
artifacts.
When the coder backend supports conversation resume, an audit repair resumes the completed
coder thread with a new prompt containing the full auditor report. If resume is unavailable,
the repair starts a fresh coder session with the same report embedded in the full prompt.
Auditing is controlled by `RESEARCH_CODER_AUDIT_ENABLED` and defaults to `True`; setting it to
`False` restores direct post-coder finalization and does not install or poll auditor runtime files.
Each auditor process has an independent `RESEARCH_CODER_AUDIT_TIMEOUT_SECONDS` deadline,
which defaults to 1800 seconds.

The system is optimized for **faithful implementation** of the agent instruction, not autonomous score chasing.

## Dashboard Task Specification Editor

The API-mode dashboard exposes **Research Task Spec** under **More Tools**. It reads and updates the active `research_task.md` file directly:

- **Preview** renders the task with the dashboard's shared Markdown and KaTeX renderer.
- **Edit Raw** exposes the complete Markdown source, including coder-only marker sections.
- Saves use the repository atomic text writer and an expected content revision, so a stale browser draft cannot silently overwrite a newer edit.
- The source remains Markdown rather than the legacy `research_tasks.yaml` format.
- A save affects subsequent `read_task` calls and newly launched coder sessions. It does not mutate agent YAML or automatically reset existing agents' read-tracking flag; operators should tell active agents to run `read_task` again after a material task change.

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
- multistart branches namespace CPU / GPU ownership as `<station_id>:s<seed>` while continuing to share the central coordination files
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

### Evaluation Visibility

The submitted-evaluations table is lineage-local at every age, including mature and tenured agents. Supervisors are exempt and can see all lineages in the table.

Direct review remains separate from table visibility. When mature, an agent can review another lineage's evaluation by evaluation ID. Agents should not run broad review attempts to reconstruct hidden tables; other-lineage evaluation IDs should come from papers, Archive Surveyor, or communication when relevant.

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
5. Notice that stdout/stderr are hidden

Completion notification includes:

1. Completion line
2. Final score and secondary metrics
3. `Coder Report`
4. Notice that stdout/stderr are hidden

Notification does **not** repeat the abstract or instruction prompt.

Agents do not see stdout/stderr in review or completion notifications. If
important information may appear in stdout/stderr, agents should ask for it in
the submission instruction and the coder should summarize it in the Coder Report
or copy the needed data to accessible Research storage.

### Agent Storage Access

Agents have read-only access to Research Center storage surfaces.

- Allowed: `read`, `storage info`, `storage list`, `read_code`
- Not allowed: direct write / edit / delete operations

The expected storage layout is:

- `storage/<your_lineage>/`
- `storage/shared/`
- `storage/system/`

For compatibility with older stations, the official attempt sandbox also exposes existing non-reserved top-level storage directories such as `storage/axioma/` when cross-lineage storage access is enabled.

Immature agents can read their own lineage storage and `storage/system/...`, but cannot read `storage/shared` or other lineage storage until maturity. Once mature, other-lineage storage access is still controlled by station policy.

Agents should refer to these paths in instructions when useful, but the coder is the main storage user.
The agent-facing Research Center help asks agents to use persistent storage
wisely and avoid saving any file larger than 1 GB unless it is necessary.

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

Subprocess launch, transcript polling, retryable-infrastructure classification,
same-thread resume, backoff, transcript-idle termination, and fresh-attempt
budgeting are owned by the inherited `CliJobManager` in
`station/workers/job_manager.py`. Research Coder supplies Research-specific
state persistence, prompt construction, report finalization, and pause hooks.

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

Dedicated Codex API overrides are opt-in and must be complete. When both
`CODEX_API_KEY` and `CODEX_BASE_URL` are set, every fresh and resumed Codex
worker receives an explicit `alt` model provider whose base URL is the exact
`CODEX_BASE_URL`, whose credential source is `CODEX_API_KEY`, and whose wire API
is `responses`. `CODEX_BASE_URL` must therefore be the complete API base, such
as `https://provider.example/v1`. Setting only one of the two variables is an
error. The shared launcher removes inherited OpenAI and alternate Codex auth
variables from that child so the dedicated key cannot silently fall back to a
different credential. When neither dedicated variable is set, no provider
override is added and the pre-existing Codex launch/auth behavior is preserved.

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
  - the default resume backoff is 10s, 20s, 40s, 60s, 120s, then nine 300s retries

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

When the External Counter is enabled, the same submit-time snapshot grants internet access to coders requested by tenured agents. Coders requested by immature or mature non-tenured agents remain offline. The permission is preserved across queue delay and session resume while the External Counter remains enabled; disabling the feature is a global network-access kill switch. An internet-enabled coder is instructed to access external websites only when the submitting agent's instruction explicitly requests an external lookup, external data download, or brief web survey; otherwise it must not use external websites.

For Codex coder and auditor sessions, internet-enabled launches retain the
`workspace-write` sandbox while adding native live web search and unrestricted
command networking. This exception is available only from the submit-time
tenured-agent snapshot while the External Counter remains enabled. Offline
launches explicitly set `web_search="disabled"` and route command networking
through Codex's network proxy, whose allowlist contains `api.openai.com`, an
explicit `CODEX_BASE_URL` or effective `OPENAI_BASE_URL` host, and every valid
provider `base_url` in the active `$CODEX_HOME/config.toml`. This provider-only
connectivity lets the CLI reach configured model services without granting
access to other websites.
The same policy is applied to fresh and resumed Codex sessions. Internet
permission applies to the coder workflow, not to the official evaluator
sandbox.

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

On stations that manage a GPU pool, the coder may mark an official attempt as CPU-only when the submitted script does not need GPU resources:

```bash
bash submit_eval.sh <eval_id> --cpu-only
```

Local coder probes may have no GPU/CUDA access even when official attempts can receive a station-managed GPU. The coder must not infer that the official attempt lacks GPU access from local `torch.cuda.is_available() == False`, empty local JAX CUDA devices, or a failed local `nvidia-smi` check. If the agent instruction requests GPU/CUDA/JAX/PyTorch GPU use, GPU-scale search, or says results are only meaningful on GPU, the coder must not pass `--cpu-only`; it should start the official attempt with normal `bash submit_eval.sh <eval_id>` and let the scheduler allocate the GPU.

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
- `submit_audit.sh` invokes `_internal/submit_audit_cli_snapshot.py`, a frozen
  artifact-only verdict helper refreshed at station startup, so a running auditor
  is isolated from later source-tree or git changes
- `local_probe.sh` invokes `_internal/local_probe_snapshot.py`, a frozen helper
  refreshed at station startup. It applies the live
  `RESEARCH_EVAL_MEMORY_LIMIT` to local probe commands and accepts an optional
  coder-selected `--timeout SECONDS`; omitting the option applies no timeout.
- `eval_tool.sh search REGEX` invokes `_internal/eval_tool_cli_snapshot.py`,
  a startup-generated read-only helper that searches the SQLite evaluation
  index abstracts only and prints matching Eval IDs, titles, and abstracts;
  it shows the newest 30 matches by default and accepts `--limit N` up to 100
- `eval_tool.sh preview <eval_id>` uses the same helper to print metadata,
  abstract, the agent instruction, and Coder Report without the coder prompt,
  raw code, or logs; it also prints the stdout path for cases where the preview
  is insufficient, but stdout inspection is not recommended unless needed

The system clears stdout/stderr only when an official attempt is accepted.

Rejected concurrent submissions do **not** wipe the active logs.

Current prompt policy also makes these points explicit:

- direct Python execution is for lightweight debugging, testing, and probing only
- computationally intensive work should go through the official submit path
- the coder prompt displays the live official-attempt memory limit and directs
  potentially memory-heavy local probes through `local_probe.sh`
- when station-managed GPUs are active and a submitted script does not need GPU resources, and the agent did not request GPU access, the coder should add `--cpu-only` to the official submit command so the scheduler does not reserve a GPU and the sandbox hides CUDA devices
- when station-managed GPUs are active and the agent requests GPU access or GPU-scale computation, local GPU probe failures are not authoritative; the coder should use normal `bash submit_eval.sh <eval_id>` and let the scheduler allocate the GPU
- while an attempt is active, the coder should poll stdout/stderr using `sleep {time}` with intervals of 30s, 60s, 120s, 240s, 480s, then 600s for all later polls
- when a new attempt starts, the coder should first run `sleep 30`, then inspect both logs, then continue with the next interval in that sequence until completion
- if an agent asks for multiple experiments in one submission, the coder should execute only the first experiment
- in that case, the coder should state clearly in the report that the Research Center allows only one experiment per submission
- multiple official attempts are for debugging, not cross-attempt optimization
- `ATTEMPT_STATUS: completed` is the official run-completion signal
- a primary score of `n.a.` is allowed for diagnostic or non-scorable work and is not by itself a debugging failure
- if the implementation is faithful and the attempt completed without exception or material mismatch with the agent instruction, a poor or non-scorable result is usually still a valid stopping point for returning control to the agent
- when the main scientific goal was not achieved but the attempt completed faithfully, the coder should proactively run a bounded lightweight diagnosis of the final result and artifacts when useful, without starting a new official attempt, changing the submitted result, or turning it into a separate research experiment
- for optimization or search runs, the post-experiment diagnosis should state whether the search appeared converged, non-converged, or inconclusive, and whether continuing the same search process in a later evaluation appears promising; this assessment should cite concrete trace or termination evidence and should not expand into broader research recommendations
- retry should be reserved for exceptions or material mismatch with the agent instruction, not for multi-attempt score optimization

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

Failure classification is universal for shared CLI workers. If a Research
coder or auditor, Archive Surveyor, multistart administrator, or another shared
CLI worker exits without producing its required completion artifact, the job
manager treats that exit as retryable infrastructure failure regardless of the
provider's error wording. A completed required artifact still takes precedence
and finalizes normally. Retries remain bounded by the configured resume and
fresh-spawn budgets. When a same-session resume budget is exhausted, every
shared worker, including the Research coder and auditor, starts a fresh session
when its fresh-spawn budget permits. Only exhaustion of all fresh sessions
requires manual intervention.

### Coder Report

The coder must always write `storage/report/{id}.md` before finishing.

For computation, the coder uses the maximum budget allowed by the task within
the single official attempt unless the agent explicitly specifies a different
budget. An explicit agent budget must be respected; unavoidable deviations
must be announced to the agent and recorded as requested-versus-actual compute.

Required sections in the current prompt:

- `Summary`
- `Final Result`
- `Files Changed`
- `Implementation Details`
- `Faithfulness And Confidence`
- `Major Deviations`
- `Post-Experiment Diagnosis`
- `Miscellaneous`
- `Final Status`

For optimization or search runs, `Post-Experiment Diagnosis` must classify the
observed convergence evidence as appeared converged, non-converged, or
inconclusive, and give a narrowly scoped outlook on whether continuing the same
search process in a later evaluation appears promising. The coder should cite
evidence such as late-run gains, termination reason, budget limits, or variation
across runs. If the requested computation timed out or was cut short, the coder
must estimate the additional compute likely needed to finish it, or explain why
no reliable estimate is possible. `Final Result` must state total compute
invested and whether the execution was faithful to the requested budget and
coverage. Broader research recommendations remain the agent's responsibility.

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

## Baseline Format

`baseline.yamll` is a persistent template input, not a consumable queue file.

Current behavior:

- it stays on disk after baseline seeding
- it may contain one or more baseline entries
- baselines are seeded and run directly by the evaluator at startup
- the baseline report is system-generated rather than coder-generated

## Generated And Persistent Paths

Runtime-created or runtime-managed paths:

- `evaluations/{id}.yaml`
- `run_requests/{id}_attempt_{n}.yaml`
- `coder_sessions/{session_id}/...`
- `submit_eval.sh`
- `eval_tool.sh`
- `_internal/submit_eval_cli_snapshot.py`
- `_internal/submit_audit_cli_snapshot.py`
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
- `RESEARCH_STORAGE_BASE_PATH` moves storage to a UUID allocation on shared disk and symlinks it back into the room
- the process environment or checkout `.env` may override the YAML value with the same variable name; the environment wins
- multistart uses the same UUID allocator, promotes the selected allocation as live storage, and removes only marked obsolete allocations after verified backup
- when the Seed Bank is enabled, its frozen `storage/system/seed_bank.py` client is a regular file inside the resolved UUID allocation; each multistart branch therefore has its own client and Seed Bank data, and selected-branch promotion does not leave a link to the old branch root
- if a remote filesystem permits create/rename but rejects `chmod`, startup replaces a copied immutable `storage/system` tree through the writable allocation root and retains the original tiny tree as a hidden recovery backup; subsequent starts and selected-branch promotion reuse the replacement

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

### Breakthrough Detection

Canonical Research Center breakthrough detection lives in
`station/eval_research/breakthroughs.py`.

The detector reads successful evaluation rows from the Research SQLite index and
does not scan evaluation YAML during normal operation. It always includes the
legacy `global` breakthrough track from `final.sort_key` or `final.primary_score`.
Tasks may additionally persist `final.progress_records`, produced by the
evaluator `get_progress_records()` hook, to define independent breakthrough
tracks such as dimensions, datasets, or theorem families.

The stagnation protocol, lineage evolution, and `scripts/breakthroughs.py`
should use this canonical detector rather than reconstructing SOTA history
independently.

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

- fail station startup if the required auto evaluator cannot start

- find unfinished instruction-driven evaluations in top-level `running`
- also reopen no-report terminal instruction evaluations in `failed`, `blocked`, or `partial` when `final` is still empty
- move them back to `queued`
- clear stale live-process metadata
- reset stale spawn/resume counters so a recovered evaluation restarts cleanly
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
- `get_breakthrough_events()`
- `get_latest_breakthrough_summary()`
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
  - Research evaluation state and report hooks for the shared CLI job manager
  - transcript/debug capture and deferred attempt/report finalization
  - orchestrator pause policy after resume or fresh-spawn exhaustion

- `station/workers/cli.py`
  - backend-specific Codex and Claude command construction

- `station/workers/job_manager.py`
  - shared subprocess launch and polling loop
  - transcript watchdog and transient failure classification
  - same-thread resume/backoff and fresh-attempt accounting
  - shutdown interruption primitives used by Coder, Surveyor, and Admin

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
- agents can ask the coder to search and summarize existing Research storage artifacts across accessible lineage storage
- multiple `read` / `read_code` actions may be used in one tick before cooldown starts
