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
ARCHIVE_SURVEY_MODEL_NAME = "gpt-5.6-sol"
ARCHIVE_SURVEY_TIMEOUT_SECONDS = 21600
ARCHIVE_SURVEY_MAX_PARALLEL_WORKERS = 2
ARCHIVE_SURVEY_MAX_SPAWNS = 3
ARCHIVE_SURVEY_MAX_RESUMES = 14
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

When both dedicated Codex variables are present, the shared backend configures
an explicit `alt` provider using the exact `CODEX_BASE_URL`, reads its
credential only from `CODEX_API_KEY`, and uses the Responses wire API. The URL
must be a complete API base such as `https://provider.example/v1`; setting only
one variable is an error. When neither is set, the shared backend adds no model
provider override and preserves the previous Codex launch/auth behavior.

The complete subprocess lifecycle is inherited from
`station/workers/job_manager.py`, which is also used by Research Coder and
multistart Admin. That manager owns polling, transcript-idle detection,
transient failure classification, same-thread resume/backoff, fresh-attempt
accounting, and process interruption. Surveyor supplies only request YAML,
prompt/workspace, report delivery, requeue, and pause hooks.

Codex-backed Surveyor launches never receive general web access. Fresh and
resumed sessions explicitly disable native web search and constrain command
networking to `api.openai.com`, an explicit `CODEX_BASE_URL` or effective
`OPENAI_BASE_URL` host, and every valid model-provider `base_url` in the active
`$CODEX_HOME/config.toml`. The shared Codex backend applies the same
provider-only policy to dashboard
Surveyor and multistart Admin sessions. Only a Research coder or auditor with a
tenured-agent submit-time access snapshot may take the External Counter web
exception.

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
Requests are allowed even when there are no non-deleted Archive papers; in that
case the Surveyor can still inspect available Research Center evidence and, for
eligible requesters, Question Room information.

## Fast-Lane Submission

Archive `survey` uses the same fast-lane pattern as
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
  question_room -> ../../../capsules/question
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

The `archive_papers`, `research_center`, and `question_room` symlinks are the
Surveyor's local read-only source surfaces. The Surveyor CLI is not granted
write access to their real paths with `--add-dir`; only the Surveyor workspace
is writable. The final report is the only intended output. Question Room access
is conditional: requests from tenured recursive agents and Supervisors include a
Question Room prompt section and preview; other requests explicitly instruct the
Surveyor not to inspect Question Room records.

For Research Center evaluation review, Surveyor should prefer the higher-level
instruction and report surfaces:

- `research_center/eval_tool.sh search "REGEX"`
  - read-only helper generated at Research Center runtime startup
  - searches evaluation abstracts only
  - prints matching Eval IDs, titles, and abstracts so the Surveyor can decide
    which evaluations deserve previewing
  - shows the newest 30 matches by default; use narrower terms when there are
    too many matches, or pass `--limit N` up to 100 when broader recall is
    necessary
  - use regex OR such as `"keyword1|keyword2"` or regex lookahead such as
    `"(?=.*keyword1)(?=.*keyword2)"` for AND-like queries
- `research_center/eval_tool.sh preview {id}`
  - read-only helper generated at Research Center runtime startup
  - prints metadata, abstract, the agent instruction, and Coder Report without
    the coder prompt, raw code, or logs
  - prints the stdout path for cases where the preview is insufficient, but
    stdout inspection is not recommended unless needed
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
- for eligible requesters, a preview of all non-deleted Question Room problems
- source and citation rules
- the requesting agent's prompt

The Surveyor is instructed that novelty means novelty relative to Station
archive papers and Station evaluation records. Evidence should be cited as
`Archive #ID` and `Eval #ID`; when Question Room access is authorized, Question
Room evidence should be cited as `Question #ID` or `Question #ID-message`. For
evaluations, specific claims should usually be based on `eval_tool.sh preview`
output before inspecting raw code or logs.
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
6. for eligible requesters asking about open problems, solved problems, pending
   questions, or Question Room discussions, scan the Question Room preview and
   read relevant full question YAML files such as
   `cat question_room/question_15.yaml`
7. for broad general landscape requests, treat the archive as usually
   sufficient unless the agent asks for specific evaluation-level detail or the
   archive evidence is clearly sparse
8. draft `reports/{id}.draft.md`; the final report should be 1000 to 5000
   words unless the request is clearly too narrow for that length
9. review the draft for completeness, citation accuracy, and formatting
10. atomically rename it with `mv reports/{id}.draft.md reports/{id}.md`
11. after the rename, do not modify either file and exit

The prompt also gives a Guidelines section telling the Surveyor to assist the
agent in understanding Station knowledge, integrate and analyze rather than
merely retrieve, synthesize strategic and technical context, avoid
over-claiming, scope novelty to Station records, stay focused on what the agent
asked, and avoid proposing research ideas. For research-direction surveys, the
Surveyor should separate concrete successes, partial tools or controls,
diagnostics with exact scope and open complements, and abandoned or discouraged
directions that remain open under the actual evidence. Where useful, it should
invert negative results into evidence-backed design principles without
proposing new research directions.

The Surveyor must not write directly to the final report path.

The Station mails the report only after:

1. the Surveyor subprocess has exited
2. `reports/{id}.md` exists and is non-empty

If the subprocess exits with only `reports/{id}.draft.md`, the request is
treated as incomplete. When the shared manager detects a retryable
infrastructure/provider failure and the backend exposes a resume token
(currently Codex), the Surveyor retries the same backend session through the
backend's resume command, using the Research Coder's configured resume
backoff schedule, for up to `ARCHIVE_SURVEY_MAX_RESUMES` resumes. A
non-transient incomplete exit, missing resume token, unsupported backend, or
exhausted resume budget falls back to a fresh Surveyor spawn. Fresh spawns follow
`ARCHIVE_SURVEY_MAX_SPAWNS` (the same default as the Research Coder). If the
final fresh spawn still produces no report, the request is marked `blocked`
and the station is paused for manual intervention. The requester is not sent a
terminal failure notification. On station restart, unfinished `running` and
new `blocked` requests without a final report are reset to a fresh `queued`
request with spawn/resume counters cleared, matching Research Coder restart
recovery. Historical `failed` requests created by older Surveyor versions are
left untouched and are never reopened automatically. Graceful Surveyor
shutdown uses the same fresh-requeue semantics rather than preserving a
backend resume across restart.

## Dashboard Surveyor

The dashboard exposes **Surveyor** as a standalone Station Tool under the
**Research** subsection of **More Tools**, immediately above **Research Task
Spec**. Archive Papers remains a separate single-purpose tool. Dashboard
Surveyor requests are not Archive Room actions and do not queue work through
`AutoArchiveSurveyor`. They always have read-only access to Archive papers,
Research Center evidence, and Question Room threads.

Dashboard survey state is isolated under:

```text
station_data/web_interface/archive_surveyor/
  question_room -> ../../capsules/question
  requests/web_survey_{id}.yaml
  reports/web_{id}.draft.md
  reports/web_{id}.md
  sessions/web_{id}_.../
  sources/{id}/station_index.sqlite3  # transient while the job is active
  index/web_archive_surveys.sqlite3
  .worker.lock
```

The dashboard service has its own SQLite list/queue index, worker lease, and
worker concurrency limit. Its jobs do not participate in Station tick waiting,
agent mail, pending notifications, fast-lane submission, Running Jobs, or the
Station SSE stream. The `.worker.lock` prevents duplicate workers when more
than one web process serves the dashboard.

Despite that separate persistence and UI contract, `WebArchiveSurveyService`
inherits the same `CliJobManager` as the Station Surveyor, Research Coder, and
multistart Admin. Codex transient failures therefore use the same resume token
extraction, same-session resume budget, backoff schedule, fresh-spawn fallback,
transcript watchdog, and process interruption loop. Only completion delivery
and terminal policy remain dashboard-specific.

Production safe-stop is the exception to that runtime isolation: normal
`stop.sh` and therefore normal `start.sh` wait for Research Center jobs, normal
agent Archive Surveyor requests, and web Surveyor requests before stopping
Gunicorn. `--force` bypasses that drain, but the shutdown hook terminates and
persistently requeues active web requests. Gunicorn startup eagerly starts the
web Surveyor worker, so queued requests resume and dead-process running records
are recovered even if no user opens the Surveyor tool after restart.

At submission time the service stores the dashboard request, the Research Task
Spec, the Archive preview, a bounded Question Room preview read from the SQLite
capsule index, and the source Station tick. Every request can search the full
Archive; the frontend does not require or expose paper selection. Each worker
receives a private SQLite snapshot for Research evaluation search, so a long
survey never queries or writes the live Station index. Archive, Research, and
Question Room source surfaces remain read-only. The per-job SQLite snapshot is
removed after completion or terminal failure; the request and report remain
persistent.

The dashboard prompt keeps the same evidence-search and report-finalization
cycle as the Station Surveyor, but changes the audience contract:

- write for an external expert who may not know Station terminology
- define Station-specific and newly introduced terms on first use
- avoid internal workflow language when ordinary scientific language is clear
- common field terminology need not be explained at an elementary level
- reports are self-contained for relevant technical details because the reader
  is not expected to open cited papers, evaluations, or Question Room threads
- brainstorming and new research proposals are allowed when requested, but
  must be separated from Station-supported conclusions

The Surveyor tool opens on the queued/running/completed/failed report table. A
prominent **New Request** button opens a separate request modal. The frontend
loads the list once and keeps it in browser memory; it does not poll. Submitting
or removing a request updates that cached list directly, and the server is
queried again only when the user presses **Refresh** or reloads the page. The
frontend renders finished reports with the shared Markdown/KaTeX renderer,
supports copying raw report Markdown, and allows non-running requests and
reports to be removed.

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
python -m unittest tests.test_archive_dashboard_payload
python -m unittest tests.test_dashboard_math_rendering
```

For broad syntax sanity after Surveyor changes:

```bash
python -m compileall station web_interface
```
