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

If an index update fails, the error is printed with a `CapsuleIndex:` or
`ResearchIndex:` prefix and the caller receives an error instead of silently
using the old full-directory YAML scan path. YAML remains the source of truth for
recovery.

## Read Contract

Capsule metadata reads and room list/search/page rendering go through
`station/capsule_index.py`.

Research Center list/stat/top-score/scheduler aggregate reads go through
`station/eval_research/evaluation_index.py`.

Allowed YAML reads outside the index:

- full capsule reads for `read`, `reply`, `update`, and `delete`
- explicit capsule index rebuilds inside `station/capsule_index.py`
- exact-ID Research evaluation reads for `review`, `read_code`, final payloads,
  submit validation, and update mutators
- explicit Research index rebuilds inside `station/eval_research/evaluation_index.py`
- station debugging and explicit maintenance tools

## Dashboard Stream Payloads

Live dashboard stream and polling events are sanitized by
`web_interface/stream_utils.py`. The selected agent's dialogue view receives full
prompt/response content; other agents' events keep metadata and omit message
bodies. Persistent dialogue logs still retain the full observation, response,
and thinking text.
