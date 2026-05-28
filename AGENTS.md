# AGENT.md

This file gives Claude Code and other coding agents the current operating guidance for this repository.

## First Principles

1. **Read the relevant docs before touching a subsystem.** The `example/doc/` folder contains subsystem notes. Use these as maintenance references, but verify against code when they disagree.
2. **Check runtime overrides.** Defaults live in `station/constants.py`, but `station_data/constant_config.yaml` overrides them at import time. Always inspect both when behavior depends on configuration.
3. **Verify names and signatures before calling or editing.** Use `rg` to find definitions and call sites. Do not infer parameter names, constants, action names, or YAML fields from memory.
4. **Treat `station_data/` as live station state.** Reading is allowed for debugging. Ask the user before modifying real station data unless the request explicitly asks for that modification.
5. **Keep disposable local probes in `/tmp`.** Use `tests/test_*.py`, `tests/debug_*.py`, or `tests/analysis_*.py` only for files that should be visible to git. Use `/tmp` directly for generated API/debug snapshots, Sage/CAS probes, and other throwaway local test output.
6. **Use repository file helpers for persistent station data.** Use `station/file_io_utils.py` for atomic YAML/text writes and safe directory creation.

## Coding Agent Checklist

Before making code changes:

1. Run `git status --short` and identify unrelated user changes.
2. Read this file, then read the subsystem doc listed in the Documentation Map.
3. Inspect `station_data/constant_config.yaml` when configuration affects the issue.
4. Use `rg` to find the live implementation, constants, tests, and call sites.
5. Decide which files are code, docs, tests, or live station state before editing.

While changing code:

1. Prefer small, localized edits that preserve existing YAML schema compatibility.
2. Use `file_io_utils` for station persistence and keep in-memory indexes synchronized.
3. Update help text when changing agent-facing actions, YAML fields, or room behavior.
4. Add or adjust tests for behavior, restart/recovery paths, and notifications when relevant.
5. Do not change `.env`, API keys, production logs, backups, or live `station_data/` unless explicitly requested.

Before finishing:

1. Run the most relevant tests from the Verification Matrix below.
2. Re-run `git status --short`.
3. Report changed files, tests run, and any untested risk.

## Current Project Snapshot

- Package/version: `station` version `1.5.0` in `setup.py`, `station/__init__.py`, `README.md`, and `CITATION.cff`.
- Main app: `python -m web_interface.app`.
- Default data root: `./station_data`.
- Default research task layout: `station_data/rooms/research/`.
- Test style: the existing tests are mostly `unittest` files under `tests/`; run specific files with `python -m unittest tests.test_name`.
- Core supported LLM connectors: Gemini, Claude, OpenAI, and Grok via `station/llm_connectors/`.
- Current model presets are in `station/llm_connectors/model_presets.yaml`.

## Setup And Run

```bash
conda create -y -n station python=3.11
conda activate station
pip install -e .
python -m web_interface.app
```

Useful optional setup:

```bash
sudo apt install ripgrep
bash scripts/setup_theory.sh
```

Use the Theory setup only when working with `THEORY_ROOM_ENABLED` or Lean verification.

## Secrets And Local State

Sensitive or machine-local files can exist in this repo checkout. Treat these as read-only unless the user explicitly asks otherwise:

- `.env`
- `station_data/`
- `backup/`
- `deployment/`
- `worker_monitor.log`
- `example_private/`

API keys are read from environment variables such as `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `XAI_API_KEY`. Research coder backends can also use `CODEX_BIN_PATH`, `CLAUDE_BIN_PATH`, `CODEX_API_KEY`, and `CLAUDE_CODE_API_KEY`. Do not print, copy, or rewrite secrets.

For live-station debugging, `deployment/error.log` is often useful, but it accumulates across station restarts. Do not run a blind global search across the whole file. First locate the most recent `Station initialized.` line, then inspect or search from that point onward.

## Documentation Map

- `example/doc/SYNC_MODES.md`: Sequential and parallel orchestrator sync modes, parallel tick state, staged LLM history, internal actions, Research Center fast-lane submit, recovery semantics, dashboard status, and tests. Read this before changing `SYNC_MODE`, `station_runner.py` tick execution, `station/sync/`, connector history persistence, or parallel Research submit behavior.
- `example/doc/API_BACKUP.md`: Provider-level API backup fallback configuration and runtime behavior. Read this before changing `station/runtime_api_config.py`, provider connector runtime endpoint handling, or dashboard API backup settings.
- `example/doc/CAPSULE_INDEX.md`: SQLite station index for capsule metadata and Research Center evaluation aggregates, rebuild behavior, tmp DB path handling, no-silent-YAML-fallback read contract, and selected-agent dashboard stream payloads. Read this before changing `station/capsule_index.py`, `station/eval_research/evaluation_index.py`, `station/index_paths.py`, capsule room list rendering, Research Center list/stat/top-score paths, or dashboard stream sanitization.
- `example/doc/CODER.md`: Current source of truth for the Research Center coder workflow. Read this before changing Research Center scheduling, coder launch, runtime paths, attempt lifecycle, restart semantics, or agent-facing research behavior.
- `example/doc/THEORY.md`: Theory Room, Lean setup, storage, evaluation queue, and tests.
- `example/doc/REVIEWER.md`: Archive reviewer system and auto-pruning.
- `example/doc/ARCHIVE_SURVEYOR.md`: Archive Surveyor action, Codex/Claude survey sessions, report finalization, mail delivery, and recovery.
- `example/doc/ASCENSION.md`: Guest ascension, lineage inheritance, and lineage-evolution selection.
- `example/doc/STAGNATION.md`: Stagnation state model, breakthrough detection, and status transitions.
- `example/doc/META_REFLECTION.md`: Meta-reflection configuration and behavior.
- `example/doc/EVALUATION_VERSIONS.md`: Historical evaluation versioning notes.
- `example/doc/CODEX_BUILD.md`: Isolated standalone Codex build notes.
- `example/doc/RESEARCH_TASK.md`: Older research-task authoring guide. Some content still describes the previous `research_tasks.yaml` and `task_{id}_evaluator.py` flow. For current Research Center behavior, prefer `CODER.md` plus the code in `station/rooms/research_center.py` and `station/eval_research/`.

Source-of-truth rule:

- Current code wins over docs when they conflict.
- `example/doc/CODER.md` wins over `example/doc/RESEARCH_TASK.md` for current Research Center lifecycle behavior.
- `station/constants.py` plus `station_data/constant_config.yaml` wins over copied constant values in this file.
- Room help text in `station/rooms/*.py` is part of the user-facing contract and should be updated with behavior changes.

Archive reviewer prompt rule:

- Distinguish the reviewer **system prompt** from the reviewer **initial context prompt**. They are different layers.
- The reviewer system prompt is the explicit connector-level constant `ARCHIVE_REVIEWER_SYSTEM_PROMPT`.
- The reviewer initial context prompt is `EVAL_ARCHIVE_INITIAL_PROMPT`, rendered with live task/archive/Codex data and sent as the first chat message.
- `AutoArchiveEvaluator` is a system service, not a normal agent prompt flow. It must **not** be wrapped by `build_station_level_system_prompt(...)` or the station Codex prefix unless the reviewer architecture is intentionally redesigned.
- If you expose or debug “Copy System Prompt” for `Reviewer`, return the actual runtime connector system prompt, not the initial context prompt and not the station-wrapped agent prompt.

## Architecture

The Station is an open-world, room-based multi-agent research environment. Agents move among rooms, persist memory in capsules and YAML files, submit experiments, publish papers, and interact with other agents.

Core modules:

- `station/station.py`: central `Station` object, room registration, status updates, orchestration hooks, background evaluator startup, navigation rules, and station config.
- `station/station_runner.py`: orchestrator loop, turn order, connector initialization, ticking, pause/wait/resume behavior, backups, and runtime coordination.
- `station/sync/`: parallel tick runner, persistent parallel tick state, crash recovery helpers, and dashboard progress summaries.
- `station/agent.py`: agent YAML lifecycle, creation, ascension, token state, notifications, locations, and pruning state.
- `station/action_parser.py`: `/execute_action{...}` parsing and YAML extraction.
- `station/base_room.py`: room base class, room context, help rendering, and action handling interfaces.
- `station/capsule.py`: capsule persistence and shared capsule operations.
- `station/capsule_index.py`: rebuildable SQLite read model for capsule metadata, tags, recipients, and active message IDs.
- `station/eval_research/evaluation_index.py`: rebuildable SQLite read model for Research Center evaluation metadata, active queues, and top submission.
- `station/index_paths.py`: shared SQLite index path resolution, including station-specific tmp override handling.
- `station/file_io_utils.py`: atomic file I/O and safe directory helpers.
- `station/constants.py`: defaults for constants, action keywords, room names, configuration, prompt templates, and override loading.

Data flow:

1. Web UI or orchestrator triggers an agent turn.
2. LLM output is parsed by `ActionParser`.
3. `Station` routes actions to the current room or handles universal navigation/help actions.
4. Rooms update agent YAML, capsule files, evaluation records, notifications, or station config through helper modules.
5. Background evaluators update queued work and notify agents.

## Data Layout

Important runtime paths under `station_data/`:

- `station_config.yaml`: station tick, status, turn order, station metadata, top submission fields, and status history.
- `constant_config.yaml`: runtime overrides for `station/constants.py`.
- `agents/`: agent YAML files and model/role/state fields.
- `capsules/private/`, `capsules/public/`, `capsules/mail/`, `capsules/archive/`: capsule stores.
- `rooms/research/`: active Research Center task, evaluator, evaluation files, coder sessions, run requests, and storage.
- `rooms/theory/`: Theory Room queue, verified lemma/theory files, index, and Lean environment files when enabled.
- `rooms/archive/`: archive evaluation queues, reviewer/surveyor artifacts, and evaluation records.
- `rooms/external/`: External Counter pending reports and generated reports when enabled.
- `dialogue_logs/`: per-agent dialogue logs.

Do not assume these directories all exist. Optional rooms create their own data directories when enabled or first used.

## LLM And Model Configuration

LLM connectors are created through `station/llm_connectors/factory.py` and configured by agent YAML fields:

- `model_provider_class`
- `model_name`
- `llm_temperature`
- `llm_max_tokens`
- `llm_custom_api_params`

Provider implementations live in `station/llm_connectors/gemini.py`, `claude.py`, `openai.py`, and `grok.py`. Dashboard presets come from `station/llm_connectors/model_presets.yaml`.

Provider fallback support lives in `station/runtime_api_config.py` and is shared across agents in memory. Backups are read from semicolon-delimited OS env vars such as `BACKUP_OPENAI_API_KEY`, `BACKUP_OPENAI_BASE_URL`, `BACKUP_OPENAI_HTTP_PROXY`, and `BACKUP_OPENAI_HTTPS_PROXY`. Per-agent `llm_custom_api_params` remain for provider-specific request options, not endpoint ownership.

Proxy-related defaults are in `LLM_HTTP_PROXY` and `LLM_HTTPS_PROXY`, but environment variables such as `HTTP_PROXY` and `HTTPS_PROXY` can override them at import time. Be careful when debugging connector behavior because imported constants may already include environment overrides.

## Rooms

Always use `constants.ROOM_*` and `constants.SHORT_ROOM_NAME_*` rather than string literals.

Current room modules:

- `Lobby`: `station/rooms/lobby.py`
- `Reflection Chamber`: `station/rooms/reflect.py`
- `Private Memory Room`: `station/rooms/private_memory.py`
- `Public Memory Room`: `station/rooms/public_memory.py`
- `Archive Room`: `station/rooms/archive.py`
- `Mail Room`: `station/rooms/mail.py`
- `Common Room`: `station/rooms/common.py`
- `Administrative Counter`: `station/rooms/admin.py`
- `Misc Room`: `station/rooms/misc.py`
- `Token Management Room`: `station/rooms/token_management.py`, gated by `TOKEN_MANAGEMENT_ROOM_ENABLED`
- `Research Center`: `station/rooms/research_center.py`, gated by `RESEARCH_CENTER_ENABLED`
- `External Counter`: `station/rooms/external_counter.py`, gated by `EXTERNAL_COUNTER_ENABLED`
- `Theory Room`: `station/rooms/theory.py`, gated by `THEORY_ROOM_ENABLED`
- `Maze`: `station/rooms/maze.py`, gated by `MAZE_ENABLED`
- `Exit`: `station/rooms/exit.py`

Navigation uses short names in `/execute_action{goto <short_name>}`. The mappings are in `ROOM_NAME_TO_SHORT_MAP` and `SHORT_ROOM_NAME_TO_FULL_MAP`.

## Configuration

The override system is at the bottom of `station/constants.py`. On import, `_load_config_overrides()` loads `station_data/constant_config.yaml` and replaces matching globals. Unknown keys are ignored unless verbose loading is enabled.

Current local override snapshot at the time this guide was updated:

```yaml
# station_data/constant_config.yaml
RESEARCH_EVAL_MAX_PARALLEL_WORKERS: 8
RESEARCH_EVAL_TIMEOUT: 900
RESEARCH_SCORE_DISPLAY_PRECISION: 4
RESEARCH_EVAL_USE_DIFF_GPU: false
RESEARCH_EVAL_CPU_NUM: 20
REFLECTION_META_INTERVAL: 25
```

Important defaults to know:

- `BASE_STATION_DATA_PATH = "./station_data"`
- `SYNC_MODE = "parallel"`
- `PARALLEL_RESEARCH_FAST_LANE_ENABLED = True`
- `PARALLEL_RESEARCH_SUBMISSION_TIMEOUT_SECONDS = 0.0`
- `PARALLEL_ARCHIVE_SURVEY_FAST_LANE_ENABLED = True`
- `PARALLEL_ARCHIVE_SURVEY_SUBMISSION_TIMEOUT_SECONDS = 0.0`
- `WEB_AUTH_ENABLED = True`
- `AUTO_EVAL_RESEARCH = True`
- `RESEARCH_EVAL_USE_PYTHON_SANDBOX = True`
- `RESEARCH_EVAL_MAX_TICK = 2`
- `RESEARCH_EVAL_MAX_PARALLEL_WORKERS = 4`
- `RESEARCH_SUBMISSION_COOLDOWN_TICKS = 0`
- `THEORIST_RESEARCH_SUBMISSION_COOLDOWN_TICKS = 10`
- `RESEARCH_MAX_CONCURRENT_SUBMISSIONS = 1`
- `RESEARCH_CODER_BACKEND = "codex"`
- `RESEARCH_CODER_TIMEOUT_SECONDS = 21600`
- `RESEARCH_CODER_MAX_ATTEMPTS = 5`
- `RESEARCH_CODER_MAX_SPAWNS = 3`
- `RESEARCH_EVAL_USE_DIFF_GPU = False`
- `RESEARCH_EVAL_CPU_NUM = None`
- `EVAL_ARCHIVE_MODE = "auto"`
- `STAGNATION_ENABLED = True`
- `STAGNATION_THRESHOLD_TICKS = 320`
- `RANDOM_PROMPT_FREQUENCY = 4`
- `BACKUP_FREQUENCY_TICKS = 10`
- `AGENT_MAX_LIFE = 300`
- `AGENT_ISOLATION_TICKS = 30`
- `EXTERNAL_COUNTER_ENABLED = False`
- `THEORY_ROOM_ENABLED = False`

Do not rely on this snapshot alone. Re-check `constants.py` and `station_data/constant_config.yaml` before changing behavior.

## Research Center

The current Research Center is an instruction-to-coder workflow, not the older direct raw-code, multi-task `task_id` workflow.

Current behavior:

- Agents submit one experiment instruction with YAML fields `title`, `tags`, `abstract`, and `instruction`.
- A room-owned coder session implements the instruction, runs official attempts, debugs if needed, and writes a final Coder Report.
- The active research task is a single markdown spec at `station_data/rooms/research/research_task.md`.
- The active evaluator is loaded from `station_data/rooms/research/evaluators/evaluator.py`.
- Evaluation records are one YAML file per evaluation under `station_data/rooms/research/evaluations/{id}.yaml`.
- Runtime directories and storage are normalized by `station/eval_research/runtime_paths.py`.
- Agents have at most one active research experiment by default.
- Research Center storage is read-only to agents, but writable to coder sessions where allowed.

State invariants:

- Top-level active states should be `queued` or `running`.
- Terminal states include `completed`, `failed`, `blocked`, and `partial`; `success` normalizes to `completed`.
- Coder substates such as `coder_running`, `attempt_running`, `pending_resume`, and `resuming` live under top-level `running`.
- `RESEARCH_EVAL_MAX_PARALLEL_WORKERS` limits live coder workflows.
- `attempt_queued` and `attempt_running` evaluations still consume coder capacity while the coder workflow is live; official attempt execution is queued and constrained by CPU/GPU resource coordination.
- Restart and resume helpers must safely requeue unfinished running work without clobbering terminal results.

Agent-facing Research Center actions include:

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
- `preview <ids|all>`
- `page_size <n>`
- `page <n>`

Storage surfaces:

- `storage/<lineage>/`: lineage working area, physically under `storage/lineages/<lineage>` with a compatibility alias.
- `storage/shared/`: shared persistent storage.
- `storage/system/`: official read-only task resources.
- `storage/submission/`, `storage/stdout/`, `storage/stderr/`, and `storage/report/`: runtime artifacts.
- `submit_eval.sh`: generated at station startup. It invokes `station_data/rooms/research/_internal/submit_eval_cli_snapshot.py`, a frozen artifact-only submit CLI refreshed on station restart.
- `eval_tool.sh`: generated at station startup. It invokes `station_data/rooms/research/_internal/eval_tool_cli_snapshot.py`, a read-only helper with `search REGEX` for abstract-only evaluation search and `preview <eval_id>` for metadata, abstract, instruction, coder prompt, and Coder Report without raw code or logs.

Research Center modules:

- `station/rooms/research_center.py`: room UI, action handling, submission validation, read cooldowns, storage reads, review/code display, and visibility rules.
- `station/eval_research/evaluation_manager.py`: evaluation YAML schema, active/queued indexes, top submission tracking, notifications, display/review/code payloads.
- `station/eval_research/auto_evaluator.py`: queue processing and evaluator/coder orchestration.
- `station/eval_research/submission_service.py`: single-writer fast-lane Research Center submit service used by parallel sync mode.
- `station/sync/fast_lane_service.py`: generic single-writer service base for parallel fast-lane actions.
- `station/eval_research/coder_manager.py`: coder prompt construction, backend sessions, attempts, reports, and finalization.
- `station/workers/cli.py`: generic Codex and Claude CLI command construction shared by the coder, surveyor, and specialized local workers.
- `station/eval_research/submit_eval_cli.py` and `submit_eval.sh`: official attempt submission path.
- `station/eval_research/executor_sandbox.py`: Python sandbox execution.
- `station/eval_research/task_registry.py`: loads the single active evaluator class from `evaluator.py`.
- `station/eval_research/base_evaluator.py`: `ResearchTaskEvaluator` interface.
- `station/eval_research/gpu_coordinator.py` and `cpu_coordinator.py`: optional resource allocation.
- `station/eval_research/restart_evaluations.py`: restart/requeue/recovery helpers.

When debugging a specific evaluation `#id`, inspect live station data first before inferring behavior from the code. Start with `station_data/rooms/research/evaluations/{id}.yaml`, especially `status`, `coder.session_id`, `attempts`, `current_attempt`, `final`, and artifact paths. If `coder.session_id` is set, then inspect `station_data/rooms/research/coder_sessions/<session_id>/prompt.txt`, `transcript.jsonl`, `stderr.txt`, and `last_message.txt` when present. Also check the live attempt artifacts at `station_data/rooms/research/storage/submission/{id}.py`, `station_data/rooms/research/storage/stdout/{id}.log`, `station_data/rooms/research/storage/stderr/{id}.log`, and `station_data/rooms/research/storage/report/{id}.md`.

Before editing this system, read `example/doc/CODER.md` and run the relevant tests:

```bash
python -m unittest tests.test_research_center_interfaces
python -m unittest tests.test_research_coder_runtime
python -m unittest tests.test_research_restart_semantics
```

## Theory Room

The Theory Room is disabled by default and is controlled by `THEORY_ROOM_ENABLED`. It supports asynchronous Lean 4 verification for lemma/theory submissions and sandbox checks.

Key files:

- `station/rooms/theory.py`: room actions and display.
- `station/eval_theory/auto_evaluator.py`: background Lean queue.
- `station/eval_theory/lean_runner.py`: Lean invocation.
- `station/eval_theory/storage.py`: YAML/index storage.
- `station/eval_theory/debugger.py`: optional theory debugger integration.
- `scripts/setup_theory.sh`: builds the repo-specific Lean/Mathlib cache and writes environment values.
- `example/doc/THEORY.md`: setup and maintenance notes.

Theory actions include `submit lemma`, `submit theory`, `sandbox`, `read`, `preview`, `filter`, `unfilter`, `search`, `rank`, `page`, and `page_size`.

## Archive, Reviewer, And Surveyor

Archive papers use capsule persistence plus optional automated evaluation.

Important settings:

- `EVAL_ARCHIVE_MODE = "auto"` enables the reviewer.
- `AUTO_EVAL_ARCHIVE_MODEL_CLASS` and `AUTO_EVAL_ARCHIVE_MODEL_NAME` configure the reviewer model.
- `ARCHIVE_EVALUATION_PASS_THRESHOLD = 6` controls publication acceptance.
- `ARCHIVE_SURVEY_ENABLED = True` enables the Archive Room `survey` action and background Surveyor.

Key files:

- `station/rooms/archive.py`
- `station/eval_archive/auto_evaluator.py`
- `station/eval_archive/surveyor.py`
- `example/doc/ARCHIVE_SURVEYOR.md`
- `example/doc/REVIEWER.md`

## External Counter

The External Counter is disabled by default via `EXTERNAL_COUNTER_ENABLED = False`. It lets mature agents request external literature reports generated by `AutoExternalReporter`.

Key files:

- `station/rooms/external_counter.py`
- `station/eval_external/auto_external_reporter.py`

Settings include `AUTO_EVAL_EXTERNAL_REPORT`, `EXTERNAL_REPORT_MODEL_NAME`, `EXTERNAL_REPORT_MAX_PARALLEL_WORKERS`, `EXTERNAL_REPORT_TIMEOUT_SECONDS`, `EXTERNAL_REPORT_MAX_TOOL_CALLS`, and `EXTERNAL_MAX_CONCURRENT_REQUESTS`.

## Agent Lifecycle

Agent statuses:

- `Guest Agent`: starts with limited privileges and a 100k token ceiling.
- `Recursive Agent`: full station privileges after ascension, default 1M token budget.

Current lifecycle features:

- Guests can ascend immediately via `ascend_inherit` or `ascend_new`.
- Guest role definitions are initialized in `station/agent.py:create_guest_agent()`. If a caller passes an explicit `role_definition` string, use it, including an explicit empty string. If no explicit role definition is passed, sample from the fresh-guest role pool: entries from `station_data/init_role_def.yaml` plus `next_role_definition` values left by departed non-supervisor, non-theorist agents. The YAML may intentionally include an empty string entry for "no role".
- `station_data/init_agents.yaml` auto-spawn names model presets. Blank `role_definition: ""` values in `station/llm_connectors/model_presets.yaml` are normalized to no explicit role, so init-agent auto-spawn samples from the fresh-guest role pool. Non-empty preset role definitions are explicit overrides.
- The web dashboard create-agent form treats an untouched/blank role field as no explicit role: frontend JS omits it, and `web_interface/app.py` also normalizes blank strings to `None` before calling the orchestrator.
- Auto-respawn after a non-ascension exit creates a fresh guest with no explicit role override, so it samples from the fresh-guest role pool. It does not inherit the departed agent's current `role_definition` or deterministically take that agent's `next_role_definition`; that `next_role_definition` is only one candidate in the shared pool.
- On `ascend_inherit`, an ancestor's `next_role_definition` has priority over the guest's role definition, including explicit empty strings. On `ascend_new`, the guest's role definition carries over.
- Lineage selection can be fitness-based via `LINEAGE_SELECTION_MODE = "evolution"`.
- Private memory capsules are inherited across lineage generations.
- Agents have life limits controlled by `AGENT_MAX_LIFE` and `AGENT_LIFE_WARNING_THRESHOLD`.
- Immature agents are restricted by `AGENT_ISOLATION_TICKS`.
- Supervisors and theorists are role-based variants controlled by `AGENT_ROLE_KEY`, `ROLE_SUPERVISOR`, and `ROLE_THEORIST`; their runtime role text overrides stored `role_definition` in `build_station_level_system_prompt(...)`.
- Supervisors and theorists do not get the Exit Room descendant-role prompt, and their stored `next_role_definition` values are not used in the fresh-guest role sampling pool.
- Auto-respawn is controlled by `AUTO_RESPAWN`.
- Exit flow is handled by the Exit room and may require minimum archive/age conditions.

Read `example/doc/ASCENSION.md` before changing ascension or lineage evolution.

## Station Status And Orchestrator State

Station status is centrally managed through:

```python
station.update_station_status(new_status, current_tick)
```

This keeps `station_config.yaml` and `status_history` consistent. Do not directly mutate station status in config unless you are intentionally bypassing the API.

Orchestrator states:

- `Running`: normal turn processing.
- `Paused`: manual resume required.
- `Waiting`: station is blocked at a tick boundary until pending work reaches a safe state.

Background work can continue while the station runs. The Research Center, Theory Room, Archive evaluator, Archive Surveyor, and External Counter each have independent background evaluator/reporting loops when enabled.

## Stagnation, Backups, Tips, And System Messages

- Stagnation logic lives in `station/stagnation_protocol.py`; read `example/doc/STAGNATION.md` before changing it.
- Backups use content-addressable storage in `station/backup_utils.py` and are triggered every `BACKUP_FREQUENCY_TICKS` unless disabled.
- Restore helper: `bash scripts/restore.sh {station_id} {tick}`.
- Random tips are loaded from `station_data/random_prompts.yaml` and controlled by `RANDOM_PROMPT_FREQUENCY`.
- System message rendering/truncation lives in `station/system_messages.py`.
- Pending notifications must be preserved safely while agents are responding.

## Actions And YAML Parsing

When adding or changing actions requiring YAML:

1. Add the action keyword to `ACTIONS_EXPECTING_YAML` in `station/constants.py`.
2. Define field constants when the field is shared or likely to be reused.
3. Update the room help text.
4. Add or update tests for parser and room behavior.
5. Verify no other room reuses the same command in an incompatible way.

Common YAML-backed actions include capsule `create`, `reply`, `update`, `forward`; Archive `survey`; Research Center `submit`; Theory `submit`, `sandbox`, `finish`; Administrative Counter `request_human`; Reflection `reflect`; Common Room `speak` and `invite`; and ascension actions.

Agent-facing text that mentions an action should use the exact parser command. For example, the Research Center currently uses `read_task`, not the older `read task_id` pattern.

## Constants That Are Easy To Get Wrong

Use these exact names:

- `BASE_STATION_DATA_PATH`, not `STATION_DATA_DIR`
- `ROOMS_DIR_NAME`, not `ROOMS_DATA_DIR`
- `RESEARCH_EVALUATIONS_SUBDIR_NAME`, not `RESEARCH_EVALUATIONS_DIR`
- `RESEARCH_TASK_SPEC_FILENAME`, currently `research_task.md`
- `RESEARCH_BASELINE_FILENAME`, currently `baseline.yamll`
- `PENDING_THEORY_EVALUATIONS_FILENAME`
- `PENDING_EXTERNAL_REPORTS_FILENAME`
- `ARCHIVE_EVALUATIONS_SUBDIR_NAME`

When in doubt, search `station/constants.py`.

## Development Patterns

Adding a room:

1. Create `station/rooms/<room>.py` inheriting `BaseRoom`.
2. Add full and short room constants in `station/constants.py`.
3. Add the room to `ROOM_NAME_TO_SHORT_MAP`.
4. Instantiate it in `Station.__init__`, behind a feature flag if optional.
5. Add navigation/access checks in `station/station.py` if needed.
6. Add help text and tests.

Changing agent behavior:

1. Check `station/agent.py`, `station/station.py`, and `station/station_runner.py`.
2. Check LLM connector behavior in `station/llm_connectors/`.
3. Preserve agent YAML compatibility with older fields when possible.
4. Do not clear or overwrite pending notifications unintentionally.

Changing persistence:

1. Use `file_io_utils` atomic helpers.
2. Preserve YAML/YAMLL format compatibility.
3. Rebuild or update in-memory indexes after external edits.
4. Avoid full rescans in hot paths unless tests or profiling support it.

Changing evaluator behavior:

1. Identify whether the flow is Research, Archive, Theory, or External.
2. Read the relevant `example/doc/*.md`.
3. Check max-tick wait behavior and notification behavior.
4. Check restart/requeue semantics.
5. Add tests for terminal states, notification delivery, and index updates.

Changing the web UI:

1. Check `web_interface/app.py`, `web_interface/static/js/dashboard.js`, and `web_interface/templates/dashboard.html`.
2. Confirm API payload fields match backend route handlers.
3. Preserve auth behavior controlled by `WEB_AUTH_ENABLED`.
4. Avoid embedding secrets or station_data content into client-side code.
5. Do **not** import `web_interface.app` from tests, helper scripts, or utility modules. Importing it initializes the live Station and Orchestrator at module import time and can mutate `station_data/`. Move side-effect-free helpers into separate modules such as `web_interface/input_utils.py` and test those modules directly.

## Version Updates

When releasing a new version, update all current version references:

1. Update all version references:
   - `setup.py`: `version='X.Y.Z'`
   - `station/__init__.py`: `__version__ = 'X.Y.Z'`
   - `README.md`: `<strong>Version X.Y.Z</strong>`
   - `CITATION.cff`: `version: "X.Y.Z"`

2. Use this git workflow for a beta-to-main release:
   - Commit version changes to the `beta` branch.
   - Squash-merge `beta` into `main`: `git switch main && git merge --squash beta`.
   - Commit the squash on `main` with the release message.
   - Push `main`: `git push origin main`.
   - Tag the release: `git tag vX.Y.Z && git push origin vX.Y.Z`.
   - Hard-reset `beta` to `main`: `git switch beta && git reset --hard main && git push origin beta --force-with-lease`.

Current version is `1.5.0`.

## Verification Matrix

Use the narrowest meaningful test set first:

- When running tests, prefer the Python executable from the default conda environment `station`. Check common locations such as `/home/ubuntu/miniconda3/envs/station/bin/python` to identify the correct interpreter. If you truly cannot find the `station` environment Python, fall back to `python` or `python3`.
- Do not import `web_interface.app` in tests; it initializes live station runtime state at import time. Test route-independent helpers through side-effect-free modules instead.

- Research Center room/display/index behavior: `python -m unittest tests.test_research_center_interfaces`
- Research coder runtime/backends/prompts: `python -m unittest tests.test_research_coder_runtime`
- Research restart/requeue semantics: `python -m unittest tests.test_research_restart_semantics`
- Archive notification safety: `python -m unittest tests.test_archive_notification_atomic`
- Stagnation protocol: `python -m unittest tests.test_stagnation_protocol`
- Agent role definition initialization, ascension inheritance, and supervisor/theorist prompt overrides: `python -m unittest tests.test_agent_role_definitions`
- Task-specific baseline checks: `python -m unittest tests.test_research_epoch_ramsey_baseline`

For parser-only changes, also consider:

```bash
python station/action_parser.py
```

For sync-mode or parallel-runner changes:

```bash
python -m unittest tests.test_parallel_research_sync
```

For import/syntax sanity after broad edits:

```bash
python -m compileall station web_interface
```

## Useful Commands

```bash
rg "def function_name" station
rg "CONSTANT_NAME" station/constants.py
python -m unittest tests.test_research_center_interfaces
python -m unittest tests.test_research_coder_runtime
python -m unittest tests.test_research_restart_semantics
python -m unittest tests.test_stagnation_protocol
python -m unittest tests.test_archive_notification_atomic
```

Use `git status --short` before and after edits. The worktree may contain user changes; do not revert unrelated files.
