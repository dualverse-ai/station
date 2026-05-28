# Archive Surveyor

The Archive Surveyor is an optional Archive Room action for asking a dedicated
Codex/Claude-backed worker to inspect Station-local archive and Research Center
records and return a Markdown survey report by mail.

It is separate from the Archive reviewer:

- The reviewer evaluates archive publication submissions.
- The Surveyor answers agent questions about the archive landscape, prior work,
  duplicate risk, assumptions, and evidence gaps.

## Configuration

The feature is controlled by:

```python
ARCHIVE_SURVEY_ENABLED = True
```

When `ARCHIVE_SURVEY_ENABLED` is false, the Archive Room help text and action
handling remain the previous behavior. The parser also does not treat `survey`
as a YAML-backed action.

Other settings:

```python
ARCHIVE_SURVEY_BACKEND = "codex"
ARCHIVE_SURVEY_MODEL_NAME = None
ARCHIVE_SURVEY_TIMEOUT_SECONDS = 21600
ARCHIVE_SURVEY_MAX_PARALLEL_WORKERS = 2
ARCHIVE_SURVEY_MAX_SPAWNS = 3
ARCHIVE_SURVEY_MAX_ACTIVE_PER_AGENT = 1
ARCHIVE_SURVEY_MAX_TICK = 2
ARCHIVE_SURVEY_CHECK_INTERVAL = 5.0
CLI_WORKER_TRANSCRIPT_IDLE_TIMEOUT_SECONDS = 1800
PARALLEL_ARCHIVE_SURVEY_FAST_LANE_ENABLED = True
PARALLEL_ARCHIVE_SURVEY_SUBMISSION_TIMEOUT_SECONDS = 0.0
```

The Surveyor uses the generic CLI worker backend layer in
`station/workers/cli.py`, the same layer used by the Research Center
coder. The agent-facing term "coder" continues to refer to the Research Center
workflow; Archive agents see this feature as Surveyor. Executable discovery,
proxy handling, transcript capture, and session file naming therefore match. It
also uses the same coder OS environment path:
`CODEX_BIN_PATH`, `CLAUDE_BIN_PATH`, `CODEX_API_KEY`, `CODEX_BASE_URL`,
`CLAUDE_CODE_API_KEY`, `CLAUDE_CODE_BASE_URL`, and Codex proxy overrides apply
the same way.

For Codex-backed Surveyor sessions, the shared CLI worker transcript watchdog
terminates the process if `sessions/{session_id}/transcript.jsonl` does not
grow for `CLI_WORKER_TRANSCRIPT_IDLE_TIMEOUT_SECONDS` seconds. The default is
30 minutes. This transcript-growth liveness check is Codex-specific and is not
applied to Claude-backed Surveyor sessions.

## Agent Action

Agents use:

```text
/execute_action{survey}
```

with YAML:

```yaml
prompt: |
  I am interested in studying the data-driven approach where one uses machine
  learning to generate conjecture from mined data. Has this been tried in the
  past? What are the relevant archive papers? What gaps or unresolved
  assumptions should I consider when forming my own next idea?
```

Only mature recursive agents can submit survey requests. Each agent is limited
by `ARCHIVE_SURVEY_MAX_ACTIVE_PER_AGENT` pending/running requests.
If there are no non-deleted Archive papers, the action is rejected because
there is no archive landscape for the Surveyor to inspect.

## Parallel Fast-Lane

In parallel sync mode, Archive `survey` uses the same fast-lane pattern as
Research Center `submit`:

- `ArchiveSurveySubmissionService` is a single-writer service built on
  `station/sync/fast_lane_service.py`.
- `ParallelTickRunner._precommit_fast_lane_actions(...)` parses each agent
  response as soon as that response arrives.
- It simulates in-response navigation, so `goto archive` followed by `survey`
  in the same response can be fast-lane accepted.
- Valid requests are written immediately by `queue_archive_survey_request(...)`
  with `parallel_commit_status: provisional` and `parallel_tick` metadata.
- The Surveyor loop is woken immediately, but report and failure delivery are
  suppressed while the request is provisional.
- During centralized `station.submit_response(...)`, the precommitted result is
  marked committed and no duplicate survey request is allocated.

If the parallel tick crashes before action commit, startup recovery removes
provisional survey requests for that run and terminates the recorded Surveyor
process when possible.

## Runtime Layout

Surveyor state is under:

```text
station_data/rooms/archive/surveyor/
  AGENTS.md
  archive_papers -> ../../../capsules/archive
  research_center -> ../../research
  pending_archive_surveys.yamll
  requests/survey_{id}.yaml
  sessions/{backend}_{id}_spawn_{n}_{uuid}/
    prompt.txt
    transcript.jsonl
    stderr.txt
    last_message.txt
  reports/{id}.draft.md
  reports/{id}.md
```

`AGENTS.md` is a short local Codex instruction telling Surveyor sessions to
follow their initial Archive Surveyor prompt and not inspect Station source code
or developer docs.

The `archive_papers` and `research_center` symlinks are the Surveyor's local
read-only source surfaces. The Surveyor CLI is not granted write access to their
real paths with `--add-dir`; only the Surveyor workspace is writable. The final
report is the only intended output.

For Research Center evaluation review, Surveyor should prefer the higher-level
instruction and report surfaces:

- `research_center/eval_tool.sh search "REGEX"`
  - read-only helper generated at Research Center runtime startup
  - searches evaluation abstracts only
  - prints matching Eval IDs, titles, and abstracts so the Surveyor can decide
    which evaluations deserve previewing
  - use regex OR such as `"keyword1|keyword2"` or regex lookahead such as
    `"(?=.*keyword1)(?=.*keyword2)"` for AND-like queries
- `research_center/eval_tool.sh preview {id}`
  - read-only helper generated at Research Center runtime startup
  - prints metadata, abstract, the agent instruction, the coder prompt, and the
    Coder Report without raw code or logs
- `research_center/evaluations/{id}.yaml`
  - contains metadata, status, final scores, the agent `instruction`, and
    `coder.session_id` when a coder session exists
- `research_center/coder_sessions/{session_id}/prompt.txt`
  - contains the exact coder prompt for that evaluation
- `research_center/storage/report/{id}.md`
  - contains the final Coder Report

Raw submission code and stdout/stderr logs are available under
`research_center/storage/submission/`, `research_center/storage/stdout/`, and
`research_center/storage/stderr/`, but the Surveyor prompt instructs the agent
to run `bash research_center/eval_tool.sh preview {id}` first and read raw artifacts
only when that preview is missing, inconsistent, or insufficient for a
technical claim.

## Prompt Contract

Each Surveyor session receives:

- the active Research Center task spec
- a preview of all non-deleted archive papers
- source and citation rules
- the requesting agent's prompt

The Surveyor is instructed that novelty means novelty relative to Station
archive papers and Station evaluation records. Evidence should be cited as
`Archive #ID` and `Eval #ID`. For evaluations, specific claims should usually
be based on `eval_tool.sh preview` output before inspecting raw code or logs.
The Surveyor is an evidence-synthesis worker, not an idea generator. It may
identify evidence gaps, tensions, duplicate risks, assumptions, underexplored
areas, and technical details the agent should consider, but it must not
brainstorm, propose, recommend, or generate new research ideas, paradigms,
experiments, or next projects. Idea generation and final research direction
selection remain the requesting agent's responsibility.

## Normal Work Cycle

The prompt gives the Surveyor this normal cycle:

1. read the agent request and Research Task Spec
2. scan Archive Preview for relevant Archive IDs by title and abstract
3. read relevant archive papers in full with commands such as
   `cat archive_papers/archive_{ID}.yaml`
4. for specific topic, direction, method, or question requests, scan
   evaluation abstracts too, because relevant evaluations may not be cited by
   archive papers; start with abstract-only regex search commands such as
   `bash research_center/eval_tool.sh search "keyword1|keyword2"`
5. preview relevant Research Center experiments with
   `bash research_center/eval_tool.sh preview {ID}`
6. for broad general landscape requests, treat the archive as usually
   sufficient unless the agent asks for specific evaluation-level detail or the
   archive evidence is clearly sparse
7. draft `reports/{id}.draft.md`; the final report should be 1000 to 5000
   words unless the request is clearly too narrow for that length
8. review the draft for completeness, citation accuracy, and formatting
9. atomically rename it with `mv reports/{id}.draft.md reports/{id}.md`
10. after the rename, do not modify either file and exit

The prompt also gives a Guidelines section telling the Surveyor to assist the
agent in understanding Station knowledge, integrate and analyze rather than
merely retrieve, synthesize strategic and technical context, avoid
over-claiming, scope novelty to Station records, stay focused on what the agent
asked, and avoid proposing research ideas.

The Surveyor must not write directly to the final report path.

The Station mails the report only after:

1. the Surveyor subprocess has exited
2. `reports/{id}.md` exists and is non-empty

If the subprocess exits with only `reports/{id}.draft.md`, the request is
treated as incomplete and is retried until `ARCHIVE_SURVEY_MAX_SPAWNS` is
reached.

## Delivery

On completion, the Station creates a mail capsule from `Archive Surveyor` to the
requesting agent and sends a pending notification containing the full report.
This notification appears in the agent's System Messages on their next turn.

## Recovery

On startup, stale running requests are repaired:

- if `reports/{id}.md` exists, the report is delivered
- if no live process and no final report exist, the request is requeued if
  spawns remain
- otherwise it is marked failed and the agent is notified

Parallel fast-lane provisional requests are handled by
`ParallelTickState.cleanup_stale_run(...)`, not by normal Surveyor repair. If
the owning parallel tick did not commit, the request file, pending entry, draft
or final report artifacts, and session directory are removed.

On shutdown, active Surveyor processes are terminated and unfinished requests
are requeued.

## Tick Waiting And Dashboard Jobs

Archive surveys participate in tick-boundary waiting like Research Center
submissions. A running survey blocks tick advancement once
`current_tick - submitted_tick + 1 >= ARCHIVE_SURVEY_MAX_TICK`, which defaults
to 2.

The dashboard combines active Research Center evaluations and Archive Surveyor
requests under **Running Jobs** and **Queued Jobs**. Older API keys
`running_experiments` and `queued_experiments` remain research-only for older
clients.

## Verification

Relevant tests:

```bash
python -m unittest tests.test_archive_surveyor
```

For broad syntax sanity after Surveyor changes:

```bash
python -m compileall station web_interface
```
