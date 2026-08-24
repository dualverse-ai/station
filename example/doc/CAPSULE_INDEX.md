# Station SQLite Index

## Scope

Capsule YAML files and Research Center evaluation YAML files remain the
authoritative station state. The SQLite database is a derived read model for
list/search/render/statistics paths that previously rescanned every YAML file in
hot loops.

The capsule index covers:

- public memory capsules
- private memory capsules by lineage
- mail capsules
- archive capsules
- question capsules

It stores capsule metadata, tags, recipients, active message IDs, active message
counts, deletion state, and archive reviewer score. It does not store full
capsule message content.

The Research Center index covers:

- compact evaluation display rows
- active/queued/running evaluation summaries
- author and lineage recent-attempt summaries
- pending notification IDs
- top research submission metadata

It stores only evaluation metadata needed for room rendering, dashboard
statistics, scheduling, and top-score display. It does not store submission code,
stdout, stderr, Coder Reports, or full review payloads.

The archive reviewer evaluation index covers:

- compact archive evaluation log rows
- accepted high-quality paper counts by author lineage for lineage evolution

It stores only the archive evaluation ID, agent name, parsed lineage, result,
score, and file metadata. It does not store archive submission content, reviewer
prompts, reviewer responses, or thinking text.

## Runtime Files

Default database path:

```text
station_data/index/station_index.sqlite3
```

Override path:

```bash
STATION_INDEX_DB_PATH=/path/to/station_index.sqlite3
```

Override directory:

```bash
STATION_INDEX_DB_DIR=/tmp/station-index
```

The default is local to `station_data/`. If `station_data/` is on network
storage and SQLite locking becomes unreliable, put `STATION_INDEX_DB_DIR` on a
local disk and rebuild from YAML at startup. If `STATION_INDEX_DB_PATH` points
inside `/tmp`, Station prefixes the database filename with the station ID, or a
stable station-data path hash when no station ID exists, so multiple station
instances do not share one tmp database accidentally.

## Startup And Rebuild

Station initializes the capsule index and Research Center evaluation index before
their hot paths are used. If the DB file does not exist, or if a schema version
does not match, Station rebuilds the affected index from YAML.

Manual rebuild:

```bash
python -m web_interface.app --rebuild-db
```

Production startup rebuild:

```bash
./start.sh --rebuild-db
```

Equivalent environment flag:

```bash
STATION_REBUILD_DB=1 python -m web_interface.app
```

Station startup logs `CapsuleIndex:` and `ResearchIndex:` lines to confirm
whether each index is ready or being rebuilt. Under `./start.sh`, the
application-side lines are captured in `deployment/error.log` by Gunicorn.

Rebuild scans YAML and writes the derived SQLite tables. Normal room rendering,
dashboard statistics, top-score display, and Research Center queue checks do not
fall back to full YAML directory scans on DB errors; index errors should be
fixed by rebuilding or by correcting the DB path/permissions.

## Write Contract

Capsule write operations still save YAML first through `file_io_utils`. After a
successful YAML write, `station/capsule.py` updates the SQLite index for that
capsule.

Research evaluation write operations still save YAML first through
`EvaluationManager`. After a successful YAML write, `station/eval_research/evaluation_index.py`
updates the SQLite index for that evaluation. Research evaluation writes should
go through `EvaluationManager`; direct writes to `evaluations/*.yaml` require an
explicit DB rebuild before aggregate views are correct.

Archive reviewer evaluation writes still save YAML logs first through
`AutoArchiveEvaluator`. After a successful YAML write,
`station/eval_archive/evaluation_index.py` updates the SQLite index for that
archive evaluation. Direct writes to `rooms/archive/evaluations/*.yaml` require
an explicit DB rebuild before lineage-evolution archive-paper aggregates are
correct.

If an index update fails, the error is printed with a `CapsuleIndex:`,
`ResearchIndex:`, or `ArchiveEvalIndex:` prefix and the caller receives an error
instead of silently using the old full-directory YAML scan path. YAML remains
the source of truth for recovery.

## Read Contract

Capsule metadata reads and room list/search/page rendering go through
`station/capsule_index.py`.

Research Center list/stat/top-score/scheduler aggregate reads go through
`station/eval_research/evaluation_index.py`.

Research Center restart and manual-resume recovery candidate discovery also goes
through `station/eval_research/evaluation_index.py`. Recovery should query the
index for unfinished, inactive-coder, and no-report terminal instruction
evaluations, then read exact YAML records only for those candidate IDs before
mutating them. It should not scan `evaluations/*.yaml` to discover candidates.

Archive reviewer aggregate reads for lineage evolution go through
`station/eval_archive/evaluation_index.py`.

Allowed YAML reads outside the index:

- full capsule reads for `read`, `reply`, `update`, and `delete`
- explicit capsule index rebuilds inside `station/capsule_index.py`
- exact-ID Research evaluation reads for `review`, `read_code`, final payloads,
  submit validation, and update mutators
- explicit Research index rebuilds inside `station/eval_research/evaluation_index.py`
- station debugging and explicit maintenance tools

The dashboard Question Room activity table uses bounded, server-side pagination
and sorting over capsule metadata in SQLite. Opening a question reads only that
question's exact YAML record so the full active thread and accepted-solution
marker can be displayed; list loads never scan the Question Room YAML directory.
The supporting sort indexes are created in SQLite without changing the capsule
schema version or triggering a YAML rebuild.

Dashboard Archive Survey requests are intentionally not stored in this
database. They use the separate
`station_data/web_interface/archive_surveyor/index/web_archive_surveys.sqlite3`
queue/list index, so their lifecycle cannot affect Station capsule, Research,
Archive reviewer, tick-wait, or scheduler queries.

## Dashboard Stream Payloads

Live dashboard stream and polling events are sanitized by
`web_interface/stream_utils.py`. The selected agent's dialogue view receives full
prompt/response content; other agents' events keep metadata and omit message
bodies. Persistent dialogue logs still retain the full observation, response,
and thinking text.

The transport uses the bounded in-memory broadcast buffer in
`web_interface/live_event_broker.py`, not a destructive queue. Each browser has
an independent sequence cursor, so multiple dashboards receive the same live
events and the SSE and polling fallback paths cannot consume or duplicate one
another's messages. A newly loaded dashboard starts at the latest sequence and
does not replay transient events accumulated while it was absent. Short
disconnects can replay only a small bounded window; older transient events are
skipped while persistent agent dialogue remains available through the normal
history endpoint. The buffer itself is bounded, so an absent dashboard cannot
cause unbounded memory growth.

The default agent dialogue view requests the latest 50 unique ticks rather
than the full append-only dialogue log. Recent-window reads locate complete
YAML documents from the end of the file and parse the selected window once.
The dashboard renders that response in small browser-frame chunks, preserving
live events until the historical batch has finished, and retries one timed-out
request. Production Nginx compresses ordinary JSON responses but leaves the SSE
stream uncompressed; Gunicorn thread capacity is configurable through
`GUNICORN_THREADS` (default `8`) so live streams do not occupy every request
thread.
