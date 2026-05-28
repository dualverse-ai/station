# Station Sync Modes

## Scope

This document describes the Station orchestrator sync modes:

- `sequential`: the original one-agent-at-a-time tick model, still available by override.
- `parallel`: the default tick model where agent LLM calls run concurrently while station mutations remain deterministic.

Read this before changing:

- `station/station_runner.py`
- `station/sync/parallel_runner.py`
- `station/sync/parallel_state.py`
- `station/sync/parallel_status.py`
- `station/sync/fast_lane_service.py`
- `station/eval_research/submission_service.py`
- `station/eval_archive/surveyor.py`
- connector persistence/reload behavior that affects staged LLM turns

Current code is the source of truth if it disagrees with this document.

## Configuration

Defaults live in `station/constants.py` and can be overridden by `station_data/constant_config.yaml`:

```python
SYNC_MODE_SEQUENTIAL = "sequential"
SYNC_MODE_PARALLEL = "parallel"
SYNC_MODE = SYNC_MODE_PARALLEL
PARALLEL_TICK_STATE_DIR_NAME = "sync"
PARALLEL_RESEARCH_FAST_LANE_ENABLED = True
PARALLEL_RESEARCH_SUBMISSION_TIMEOUT_SECONDS = 0.0
PARALLEL_ARCHIVE_SURVEY_FAST_LANE_ENABLED = True
PARALLEL_ARCHIVE_SURVEY_SUBMISSION_TIMEOUT_SECONDS = 0.0
```

`SYNC_MODE = "parallel"` is the default. Use `SYNC_MODE: sequential` in `station_data/constant_config.yaml` to preserve the original one-agent-at-a-time behavior for a specific deployment.

The `*_SUBMISSION_TIMEOUT_SECONDS = 0.0` fast-lane timeout means the LLM
worker waits indefinitely for the single-writer service to accept or reject the
action. This avoids orphan provisional submissions.

## Sequential Mode

Sequential mode is implemented by `Orchestrator._run_single_tick_sequential()` in `station/station_runner.py`.

One tick proceeds agent by agent:

1. Load the current turn order and resume at `current_agent_index_in_turn_order`.
2. For the current agent, prepare the station response with `station.request_status(...)`.
3. Send that response to the agent's connector.
4. Immediately pass the LLM response to `station.submit_response(...)`.
5. If the response starts an internal action, run that internal action loop immediately before moving to the next agent.
6. Persist the next agent index after pauses/errors so restart can resume inside the tick.
7. When all agents are processed, reset next index to `0`, clear `current_tick_processed_agents`, and finish the tick boundary.

Important sequential semantics:

- Later agents in the same tick can observe mutations made by earlier agents.
- Mail, common-room speech, capsules, room state, and notifications can therefore be same-tick visible depending on turn order.
- LLM connector history is persisted as part of `connector.send_message(...)` before station action mutation.
- Crash recovery can resume from the saved next-agent index instead of retrying the whole tick.

This mode is simpler and remains the safest baseline when debugging station behavior.

## Parallel Mode Goals

Parallel mode is implemented by `ParallelTickRunner` in `station/sync/parallel_runner.py` and selected by `Orchestrator.run_single_tick()`.

The main invariant is:

> Every eligible agent receives a station response prepared from the same tick-boundary state, before any LLM response from that tick mutates normal station state.

Consequences:

- All initial agent LLM calls for a tick run concurrently.
- Normal station action commits are centralized and deterministic.
- Mail, common-room speech, public memory, archive effects, and most other room mutations are intentionally lagged by one tick for other agents.
- The result does not depend on provider latency, retry timing, or the order in which LLM responses return.

## Parallel Tick Lifecycle

At the start of each parallel tick:

1. `ParallelTickRunner.run_single_tick()` calls `_cleanup_stale_parallel_state()`.
2. The runner reloads turn order and resets `current_agent_index_in_turn_order` to `0`.
3. `ParallelTickState.begin_tick(...)` creates `station_data/sync/current_parallel_tick.yaml`.
4. Tick-boundary housekeeping that is not caused by current LLM responses may run, such as moving agents out of Research Center on holidays.

Observation preparation:

1. `_prepare_observations(...)` iterates over the turn order.
2. It performs life-limit, maturity, session-end, connector `sync_state()`, and `station.request_status(...)` checks.
3. It writes each prepared station response to `station_data/sync/parallel_ticks/.../observation.md`.
4. It marks `observation_prepared` in the parallel tick ledger.

No current-tick LLM response processing happens until all station responses are prepared.

Initial LLM response collection:

1. `_collect_initial_llm_responses(...)` starts a worker thread for each prepared agent.
2. Each worker calls `_send_llm_staged(..., persist_to_disk=False)`.
3. The connector request uses the normal connector configuration, model params, routing, thinking settings, and debug hooks.
4. The returned user/model turn is stored as a `StagedLLMTurn` instead of being immediately written to agent history.
5. The raw agent response is written to `station_data/sync/parallel_ticks/.../response.md`.
6. The ledger marks `response_received`.

Initial response commit:

1. After all initial LLM workers complete, the runner iterates in original turn-order order.
2. For each agent, it flushes the staged LLM turn to disk before station mutation.
3. It marks `history_flushed`.
4. It calls `station.submit_response(...)`.
5. It marks `actions_committed`.

Flushing history before mutation is required for actions such as ascension, where `submit_response(...)` can move the current identity's history to a new recursive identity.

Tick finish:

1. Internal action handlers, if any, are run.
2. Token budgets are updated from latest connector token info.
3. The saved next-agent index is reset to `0`.
4. The usual tick-boundary work runs through the existing orchestrator finish path.
5. `ParallelTickState.mark_completed(...)` clears `current_parallel_tick.yaml`.

## Internal Actions In Parallel Mode

Internal actions are not globally sequential, but each agent's internal action conversation is sequential.

For each internal handler:

1. `_run_internal_handlers_parallel(...)` starts one worker per handler.
2. `_handle_internal_action_loop_parallel(...)` sends the current internal prompt to the agent.
3. The agent's internal response is staged.
4. If the connector context is not an override context, the staged internal LLM turn is flushed to history before handler mutation.
5. `handler_wrapper.step(response_text)` runs under `orchestrator._parallel_action_commit_lock`.
6. Delta updates are applied while still in the serialized mutation path.
7. The next internal prompt, if any, is sent by the same worker for the same agent.

This means:

- Network/API time for different agents overlaps.
- Mutating handler steps are serialized.
- One agent's multi-step internal action remains ordered.
- Room handlers should not spawn their own unsynchronized station-data mutations outside `handler.step(...)`.

Connector override contexts are created and finalized through:

- `_prepare_internal_action_connector_context(...)`
- `_finalize_internal_action_connector_context(...)`

Do not bypass these helpers for rooms that use temporary connector state, pruning, or special prompt contexts.

## Fast-Lane Actions

Research `submit` and Archive Room `survey` are fast-lane station mutations in parallel mode.

Reason: coder/evaluator launch and Surveyor launch should start as soon as an
agent response containing a valid request is received, not only after every
other agent finishes its provider call.

Implementation:

- `station/sync/fast_lane_service.py` owns the shared single-writer worker base.
- `station/eval_research/submission_service.py` owns `ResearchSubmissionService`.
- `station/eval_archive/surveyor.py` owns `ArchiveSurveySubmissionService`.
- Each service is a single-writer background thread.
- `ParallelTickRunner._precommit_fast_lane_actions(...)` parses the LLM response immediately after it arrives.
- The runner simulates in-response navigation before deciding whether a `submit` action is really in the Research Center or a `survey` action is really in the Archive Room.
- If a Research submit is valid, the service calls `EvaluationManager.create_instruction_evaluation_atomic(...)`.
- If an Archive survey is valid, the service calls `queue_archive_survey_request(...)`.
- The new evaluation or survey receives `parallel_commit_status: provisional` and `parallel_tick` metadata.
- The service wakes the existing auto evaluator/surveyor immediately.
- Later, the centralized `station.submit_response(...)` receives the precommitted result and must not allocate a duplicate evaluation or survey request.

The Research service validates:

- recursive agent only
- not holiday
- not supervisor
- submission cooldown
- required YAML fields
- tag and abstract validation
- per-author active experiment limit

The Archive Survey service validates:

- survey feature enabled
- recursive mature agent only
- required YAML `prompt`
- per-author active survey limit

If a response contains ascension, fast-lane commit is restricted so unrelated actions are not precommitted from a turn whose identity may change.

## Persistent Parallel State

The active ledger is:

```text
station_data/sync/current_parallel_tick.yaml
```

Per-agent observation/response snapshots are stored under:

```text
station_data/sync/parallel_ticks/tick_{tick}_{run_id}/{safe_agent_name}/
```

The ledger tracks:

- `tick`
- `run_id`
- `baseline_research_max_eval_id`
- `turn_order`
- per-agent `observation_prepared`
- per-agent `response_received`
- per-agent `history_flushed`
- per-agent `actions_committed`
- `fast_lane_evaluations`
- `fast_lane_surveys`

This is intentionally a small recovery ledger, not a full replay journal.

## Crash Recovery

Parallel mode favors simple restart semantics over intricate mid-tick replay.

On orchestrator startup and at parallel tick start, `ParallelTickState.cleanup_stale_run(...)` checks for an unfinished `current_parallel_tick.yaml`.

If one exists:

1. Provisional research evaluations from that run are rolled back.
2. Provisional Archive Surveyor requests from that run are rolled back.
3. Associated Research Center coder/evaluator processes are requeued/terminated through the existing restart helpers where possible.
4. Associated Archive Surveyor subprocesses are terminated when their PID is recorded.
5. Staged LLM history entries are removed for agents with `history_flushed: true` but without `actions_committed: true`.
6. agent `waiting_station_response` flags are cleared.
7. `next_agent_index` is reset to `0`.
8. The stale ledger is marked recovered and removed.
9. The tick is retried from a clean tick boundary.

Recovery deliberately does not try to undo arbitrary room mutations after `actions_committed: true`. If a crash occurs after an agent's actions commit, those committed effects remain.

This is why the runner only uses rollback for:

- provisional fast-lane research submissions
- provisional fast-lane Archive Surveyor requests
- staged LLM turns whose station actions did not commit
- transient waiting flags

## Prompt And History Invariants

Parallel mode must preserve the same connector-visible history shape as sequential mode.

Required invariants:

- The exact station response sent to each agent is the prepared observation snapshot for that tick.
- Initial LLM turns are flushed before `station.submit_response(...)`.
- Internal LLM turns are flushed before `handler.step(...)` unless an explicit override connector context persists directly.
- Connector reload should happen only when required by the connector, not unconditionally after every staged turn.
- Normal agents must still pick up role/system prompt changes from their YAML when reloaded.
- `AutoArchiveEvaluator` uses its explicit reviewer connector system prompt and must not be station-wrapped by normal agent prompt reload.
- Supervisor assignment, token-management pruning, ascension, and reviewer flows should be tested whenever connector reload or history staging changes.

## Concurrency Rules

Allowed in parallel:

- LLM provider calls for different agents.
- Per-agent internal-action LLM calls.
- Background Research Center coder/evaluator work after a submission is accepted.
- Background Archive Surveyor work after a survey request is accepted.

Serialized or single-writer:

- Normal station response commit through `station.submit_response(...)`.
- Internal `handler.step(...)` mutation through `_parallel_action_commit_lock`.
- Research submit allocation through `ResearchSubmissionService`.
- Archive survey allocation through `ArchiveSurveySubmissionService`.
- Evaluation ID allocation through `EvaluationManager.create_instruction_evaluation_atomic(...)`.
- Parallel state writes through the runner's state lock.

When adding a new action or room behavior:

- If it mutates station data, keep it in the centralized commit path unless there is a strong reason for a fast lane.
- If it needs a fast lane, give it a single-writer service and explicit recovery metadata.
- Use `station/file_io_utils.py` for persistent state.
- Add tests for crash recovery and duplicate prevention.

## Dashboard Status

The web API returns:

- `sync_mode`
- `parallel_tick_status`

`station/sync/parallel_status.py` builds a read-only summary from the active ledger. The dashboard uses it only when `sync_mode == "parallel"`.

The compact frontend line reports the current bottleneck, for example:

- `Waiting for Axiom I, Eidos I (3/5 done)`
- `Committing 2 response(s) (3/5 done)`
- `Committed 5/5 responses`

Sequential mode keeps the original `Next: <agent> (Index: <n>)` display.

## Debugging

For live station debugging:

1. Check `station_data/constant_config.yaml` to confirm `SYNC_MODE`.
2. Check `/api/orchestrator/status` or the dashboard for sync mode and active parallel progress.
3. If a parallel tick is in flight, inspect `station_data/sync/current_parallel_tick.yaml`.
4. For exact prepared prompts/responses, inspect the matching `station_data/sync/parallel_ticks/...` files.
5. For Research Center submissions, inspect `station_data/rooms/research/evaluations/{id}.yaml`, especially `parallel_commit_status` and `parallel_tick`.
6. For Archive surveys, inspect `station_data/rooms/archive/surveyor/requests/survey_{id}.yaml`, especially `parallel_commit_status` and `parallel_tick`.
7. If `DEBUG_API` is enabled, raw connector request snapshots remain connector-level debug output under `tmp/debug_api/<station_id>/`; parallel mode should not change model params, thinking config, routing, or provider request construction.

Do not run broad searches across large production logs. Locate the latest `Station initialized.` entry first, then inspect from there.

## Verification

Primary parallel-mode regression test:

```bash
python -m unittest tests.test_parallel_research_sync
```

For syntax/import sanity after sync changes:

```bash
python -m compileall station web_interface tests
```

Also run Research Center tests when touching fast-lane submission, evaluation IDs, restart behavior, or coder launch:

```bash
python -m unittest tests.test_research_center_interfaces
python -m unittest tests.test_research_coder_runtime
python -m unittest tests.test_research_restart_semantics
```

Run Archive Surveyor tests when touching survey fast-lane submission, recovery,
or delivery:

```bash
python -m unittest tests.test_archive_surveyor
```

High-risk areas that need targeted tests:

- ascension during a parallel tick
- internal actions that mutate room or agent state
- token-management pruning
- supervisor assignment and role prompt reload
- archive reviewer prompt reload
- crash after history flush but before action commit
- crash after provisional research submit
- crash after provisional archive survey
- duplicate prevention for precommitted Research Center submissions
- duplicate prevention for precommitted Archive Surveyor requests
