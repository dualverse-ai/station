# Multistart Controller

This document describes the repo-local multistart controller that replaced the
public Station Tools `station multi_init` command.

## Goals

- Run multiple initialization seeds in parallel from the normal station init/start path.
- Run multiple stagnation-recovery branches in parallel just before stagnation lane assignment.
- Keep the implementation mostly outside normal Station operation.
- Preserve unselected seed data, selected branch data, admin reports, and controller logs in the final selected `station_data` backup.
- Make crashes and manual intervention recoverable from durable state.
- When both multistart modes are disabled, keep `./start.sh` behavior exactly as it is today.

## Constants

Users should only need these six constants, usually through
`station_data/constant_config.yaml`:

```python
MULTISTART_INIT_MAX_PARALLEL = 4
MULTISTART_INIT_SEEDS = 8
MULTISTART_INIT_ROLL_TICKS = 40

MULTISTART_STAGNATION_MAX_PARALLEL = 4
MULTISTART_STAGNATION_SEEDS = 8
MULTISTART_STAGNATION_ROLL_TICKS = 40
```

For either mode, `SEEDS <= 1` disables that multistart mode. `MAX_PARALLEL` is
clamped to at least `1` and at most the enabled seed count.

`station init` disables both modes by writing zero seed and roll-tick values by
default. Pass `station init --multistart` to preserve the selected station
template and research task's values, including the defaults shown above.

## Shared Research Storage Integration

`RESEARCH_STORAGE_BASE_PATH` is the single storage-root setting for both normal
Station operation and multistart. It may be configured in
`station_data/constant_config.yaml`, or through the process environment or the
checkout's `.env`. The environment value wins over YAML:

```bash
RESEARCH_STORAGE_BASE_PATH=/mnt/stephen/tmp
```

When the effective value is unset, Research storage and multistart copying keep
their original local behavior. If a station still has an external storage
symlink from an earlier configuration, multistart materializes a private local
copy for each seed rather than sharing that target. When the value is set,
normal Station startup moves a
regular `rooms/research/storage` tree into a UUID directory beneath the base and
symlinks it back into the room. Multistart uses the same allocator and gives
every seed an isolated UUID directory:

```text
<base>/<seed-1-uuid>
<base>/<seed-2-uuid>
...
```

Each allocation has a sidecar ownership marker next to its UUID directory, and
the job records exact seed targets in `research_storage_allocations.yaml`. An
origin that is already remotely managed is recorded for later cleanup; a local
origin remains local and is not copied remotely before seed fanout.
Legacy unmarked storage targets remain readable but are never automatically
deleted. Seeds never share one writable storage tree.

CPU and GPU coordination remains global through the configured shared
coordination files, but each branch uses `<station_id>:s<seed>` as its resource
owner ID. This prevents branches copied from the same station UUID from
clearing or reusing one another's live allocations, including when their
evaluation IDs are equal. Restarting a seed retains the same owner ID so it can
clean only that seed's stale allocations.

When the Research Seed Bank is enabled, every allocation contains its own
`shared/seed_bank/` data and frozen regular `system/seed_bank.py` client. The
runtime installer replaces the older branch-local snapshot symlink form during
startup. Keeping the client inside the allocation prevents remote branch copies
and selected-branch promotion from retaining a path to the old data root.
On filesystems that reject `chmod`, a copied immutable `system/` directory is
swapped through the writable allocation root and its original contents are kept
as a hidden recovery backup; the replacement is reused after finalization.

The controller checks local and storage-base capacity independently and refuses
to start if the configured base is absent, read-only, or would exceed the
normal 95% disk safety limit. The selected seed's UUID allocation is promoted
directly to live Station storage without copying it locally. After the
finalization backup is verified and archived branch folders are pruned, the
controller removes marked unselected-seed allocations and the marked obsolete
origin allocation while retaining the selected live UUID.

Finalization and explicit active-job archives dereference managed origin and
branch storage links so restores remain self-contained. On active-job restore,
regular archived storage directories are reconciled into fresh UUID allocations
under the currently effective base before branch work resumes. A verified
explicit active-job archive removes all marked allocations only after its ZIP
passes integrity verification; if cleanup fails, local job metadata is retained
for a safe retry.

## Runtime Layout

The controller has a stable repo-local root while it is running:

```text
station_multistart/
  controller.pid
  controller.sock
  controller.log
  current_job.yaml
  pending_init.yaml
  pending_stagnation.yaml
```

Each multistart operation gets one job id. The temporary per-job working
directory is repo-local and ignored by git:

```text
station_multistart/<branch_tick>_<job_id>/
  state.yaml
  job.log
  origin_station_data/
  station_data_s1/
  station_data_s2/
  ...
  admin/
    reports/
    sessions/
```

The directory name starts with the branch tick for easy sorting and inspection.
`state.yaml` should include at least:

- `job_id`
- `mode`: `init` or `stagnation`
- `status`
- `station_name`
- `origin_station_id`
- `branch_tick`
- `seed_count`
- `max_parallel`
- `roll_ticks`
- branch statuses, pids, start/end times, current ticks, and failure notes
- selected seed, once known

No API keys, proxy secrets, provider endpoint secrets, or frontend runtime API
payloads should be written into `state.yaml`, logs, or archived branch
metadata.

`station_multistart/` is ignored by git. If both multistart modes are disabled,
`./start.sh` should not create this directory.

Job and current-job state are persisted with status `creating` before live
`station_data` is moved. The waiting frontend is started before stagnation
fanout, all enabled seed trees are copied concurrently, and the page reports
pending, active, completed, and failed copy counts. If creation is interrupted,
the durable `creating` state resumes only seed copies that do not already have a
complete branch tree and verified Research storage allocation.

## Data Root Override

Branch workers do not run dashboard services. They run the Station headlessly
with a process-level data root override, for example:

```text
STATION_BASE_DATA_PATH=station_multistart/<branch_tick>_<job_id>/station_data_s3
```

The Station reads this before loading `constant_config.yaml`, so each branch
uses its own `station_data_sN` while sharing the same source checkout.

Research coder, Archive Surveyor, and generated Research helper subprocesses
also receive `STATION_BASE_DATA_PATH` explicitly. This prevents child processes
from drifting back to repo-local `./station_data` when they import Station
modules.

Branch workers should not write normal repo `backup/<station_id>` while rolling.
Automatic backups are disabled in branch workers. The authoritative backup is
created after finalization, once all branch data and admin materials have been
moved under the selected live `station_data`.

Before the finalization backup, rebuildable/transient runtime state is stripped
from archived branch data: SQLite station indexes, station sync state, Research
Center `storage/tmp`, and Research Center `storage/shared/tmp`. This mirrors
normal backup omission rules and prevents disposable workspaces from bloating
the multistart backup.

Before large branch fanout copies, the controller estimates projected disk
usage on the target filesystem. If branch creation would raise usage above 95%,
it leaves live `station_data` in place and records a durable pending request.
The controller rechecks the request on its normal poll loop and automatically
creates the job once enough disk space is available. Init multistart uses
`pending_init.yaml`; stagnation multistart keeps its existing
`pending_stagnation.yaml` request.
With `RESEARCH_STORAGE_BASE_PATH` enabled on a separate filesystem, the local
estimate excludes branch Research storage copies and a second preflight checks
the storage-base filesystem for the per-seed allocations.
Finalization must not require another full-copy-sized reserve: the selected
branch is moved into live `station_data`, and the remaining job folder is moved
under `station_data/multistart/<job>/` by rename. The selected branch itself is
not duplicated in that archive because it is the live station data; the archive
records this with `station_data_sN.installed.yaml` and preserves the selected
interview under `multistart/<job>/interviews/`.

## Runtime API Updates

Frontend runtime API updates are in-memory station state today, and may include
API keys. They should not be persisted by the multistart controller.

Use a repo-local Unix domain socket instead of a TCP port:

```text
station_multistart/controller.sock
```

When the frontend route updates runtime API config, it should:

1. update the main station process as it does today;
2. try to connect to `station_multistart/controller.sock`;
3. if the controller is running, send the same update payload over the socket;
4. if the socket is missing, skip controller notification without changing
   normal station behavior.

The controller applies the latest runtime API update to its own environment and
newly launched branch workers inherit that environment. Already-running branch
workers keep the config they inherited at launch. This avoids child worker
control channels and is acceptable unless API changes during active multistart
runs become common.

If the controller restarts, `./start.sh` relaunches it under the current
environment and hooks. Runtime API updates that only existed in memory must be
reapplied through the frontend if they are still needed.

For init multistart, there is normally no frontend session yet. Runtime API
configuration therefore comes from the `start.sh` environment and local hooks.

## Start Path

`./start.sh` launches or resumes a repo-local multistart controller after
environment/conda setup and before normal service startup.

If both enabled seed counts are `<= 1`, the hook exits immediately and
`./start.sh` continues exactly as it did before multistart. It should not create
`station_multistart/`.

If init multistart is enabled and no finalized init job exists for the current
fresh `station_data`, the controller starts or resumes the init job in the
background using a detached Python process. The live `station_data` is moved
into the job folder as `origin_station_data`. `./start.sh` starts the lightweight
multistart frontend, prints normal service information, and exits. The controller
finalizes the selected seed and starts the normal station afterward.

If init branch creation is temporarily blocked by the disk-space guard, the
startup hook routes the web process to multistart waiting mode. The controller,
not the startup hook, owns the pending request and retries it until branch
creation can proceed.

For ordinary station runs, the controller remains available as a background
monitor for stagnation requests, but normal dashboard startup continues.

`stop.sh` stops the controller when it is running. A normal stop asks the
controller to stop gracefully through `controller.sock` and waits for active
branch workers to reach their ordinary stop point. `./start.sh` uses this same
normal stop path before restarting services, so a normal restart during
multistart can wait for active branches. `stop.sh --force` kills repo-scoped
multistart process groups, including controller, branch workers, and orphaned
coder/sandbox descendants whose command line or cwd is under
`station_multistart/`. Stopping must preserve the job folder so `./start.sh`
can resume it later.

## Frontend During Multistart

When any multistart job is active, the normal URL serves the familiar Station
dashboard as a read-only preview of seed 1. A persistent banner states that
multistart is running, that seed 1 may not be selected, and links to the full
branch monitor at `/multistart`. Seed 1 remains fixed for the lifetime of the
job; the preview does not silently switch to another running or successful
branch.

The preview reads seed 1 in place from its existing `station_data_s1` root. It
must not copy or move that branch into live `station_data`, construct a second
`Station` or `Orchestrator`, start background services, or change the global
data root per request. Lightweight configuration, agent, dialogue, and task
reads use explicit paths. Research and bounded capsule list views use the branch's
existing SQLite index in query-only mode. If the index is unavailable, the
preview returns an empty/unavailable view rather than performing an expensive
YAML tree scan or rebuilding the index.

All Station mutations are locked in preview mode at the HTTP layer as well as
in the UI. Agent lifecycle controls, messages, chats, Station configuration,
task edits, surveys, backups, and shutdown are unavailable. This prevents
concurrent writers and avoids giving seed 1 a human-assisted selection
advantage. The separate multistart monitor retains job-level pause/resume
controls.

Branch workers continue to run without dashboard services. The waiting web
process owns the read-only preview, so no extra per-branch web processes or
ports are launched. Durable dialogue history is available, while the ordinary
in-memory live event stream is intentionally absent. While a specific agent is
selected, the preview polls only that agent's recent durable dialogue window
every three minutes and atomically refreshes it when new Station observations or
agent responses appear. Other routine dashboard state uses small periodic read
models instead.

The `/multistart` monitor exposes branch rolling pause/resume controls. These
controls write a durable job-level control flag through `controller.sock`. A pause stops
new queued branches from launching and asks already-running branch workers to
pause at the next safe tick boundary. It does not interrupt an in-flight LLM
call, coder session, interview, quiescence wait, completed branch, or failed
branch. Resume clears the control flag and only resumes branches that have not
reached their target tick.

## Init Multistart Flow

1. `station init --multistart` or another multistart-enabled init path creates
   the initial task `station_data`.
2. `./start.sh` sees `MULTISTART_INIT_SEEDS > 1`.
3. The controller creates `station_multistart/<branch_tick>_<job_id>/`.
4. It moves or keeps the initialized base data as `origin_station_data`.
5. It creates `station_data_s1`, `station_data_s2`, ... from the initialized
   base data.
6. It runs up to `MULTISTART_INIT_MAX_PARALLEL` branch workers at a time.
7. Each branch runs headlessly for `MULTISTART_INIT_ROLL_TICKS`.
8. Each completed branch waits for coder/evaluator work to finish, then runs
   incognito interviews with active Recursive Agents using the best available
   headless Orchestrator/temporal-chat path. Interviews are written to
   `interview.yamll` and ask agents to summarize progress since the branch
   tick.
9. Each completed branch is paused/stopped with no normal repo backup.
10. Admin selection runs over all `station_data_sN` folders. The admin writes
    `admin/reports/selection_report.md`, `admin/reports/guidance_report.md`,
    and `admin/reports/selected.txt`.
11. Finalization installs the selected branch as live `station_data`, moves all
   branch data and admin material into `station_data/multistart/<branch_tick>_<job_id>/`,
   stops the lightweight multistart frontend, creates one manual backup, prunes bulky
   archived `station_data_s*` folders after that backup is verified, removes
   the temporary job directory, and starts the normal station.

## Stagnation Multistart Flow

Stagnation branching must happen before lane messages are assigned.

1. At tick end, after ordinary tick-boundary waiting and before normal
   stagnation lane assignment, the station detects that a stagnation escalation
   would occur.
2. If `MULTISTART_STAGNATION_SEEDS <= 1`, existing stagnation behavior runs
   unchanged.
3. If stagnation multistart is enabled, the station writes a durable branch
   request, pauses, and does not assign lanes in the main `station_data`.
4. The controller stops the normal station service and replaces it with the
   lightweight multistart frontend and read-only seed-1 preview.
5. The controller waits until all active coder/evaluator work is complete. The
   branch source must not be copied merely because the orchestrator is paused.
6. The controller records the branch tick and creates
   `station_multistart/<branch_tick>_<job_id>/`.
7. The quiescent main `station_data` is moved to `origin_station_data`, leaving
   no live canonical `station_data` while multistart is running.
8. `origin_station_data` is copied into each `station_data_sN`.
9. Branch workers run headlessly. In branch workers only, the stagnation
   deferral hook is disabled, so lane assignment happens independently inside
   each branch.
10. The lane-recipient shuffle must be sampled separately per branch, not before
   copying. This ensures different branches can explore different lane
   assignments.
11. Each branch runs for `MULTISTART_STAGNATION_ROLL_TICKS` after assignment.
12. Each completed branch waits for coder/evaluator work to finish, then runs
    incognito interviews with active Recursive Agents. Interviews are written
    to `interview.yamll` and ask agents to summarize progress since the
    stagnation branch tick.
13. Admin selection chooses one branch to continue. For stagnation jobs only,
    the admin workspace includes an `origin_station_data` symlink and the
    prompt includes pre-branch archive abstracts from that origin snapshot.
    The admin should discount candidate lanes that mostly duplicate existing
    archive work. In both init and stagnation modes, the 500-to-2,000-word
    guidance report must include all high-potential lanes from unselected
    branches that are not represented in the selected branch, with enough
    technical detail for agents to reproduce each idea independently.
14. Finalization replaces live `station_data` with the selected branch, moves
    all branch data/admin logs into
    `station_data/multistart/<branch_tick>_<job_id>/`, stops the lightweight
    multistart frontend, removes archived `origin_station_data` before the finalization
    backup, creates a manual backup, prunes bulky archived `station_data_s*`
    folders after that backup is verified, removes temporary job storage, and
    restarts the normal station. From the user's perspective the normal writable
    station becomes a read-only seed-1 preview, then returns at the selected branch's
    completion tick.

If a stray live `station_data` exists during finalization with no
`station_config.yaml`, or with `station_config.yaml` reporting
`current_tick <= 1`, the controller treats it as an early startup/waiting-page
placeholder and deletes it instead of archiving `_unexpected_live_station_data`.
A live station with a real config past tick 1 still blocks finalization.

## Finalization Layout

After selection and before backup-pruning, audit data lives inside the selected
station:

```text
station_data/
  multistart/
    <branch_tick>_<job_id>/
      state.yaml
      job.log
      admin/
      station_data_s1/
      station_data_s2/
      ...
```

Before the manual finalization backup is created, this audit folder contains
all candidate branches, the selected branch copy, and the exact selection
context. `origin_station_data/` is removed before this manual backup because it
is already covered by the pre-branch backup. The finalization backup writes a
job-specific snapshot named `tick_<tick>_multistart_<branch_tick>_<job_id>.json`
so later same-tick periodic backups cannot replace the manifest needed for
branch recovery. After the finalization backup succeeds and its snapshot is
verified to contain every archived `station_data_s*` folder, the controller
prunes bulky archived branch folders from live `station_data`. It keeps
`branch_archive_manifest.yaml`, per-branch `station_data_s*.pruned.yaml`,
preserved branch interviews, and all admin reports in place.

After pruning, the live audit folder is:

```text
station_data/
  multistart/
    <branch_tick>_<job_id>/
      state.yaml
      branch_archive_manifest.yaml
      station_data_s1.pruned.yaml
      station_data_s2.pruned.yaml
      interviews/
      job.log
      admin/
```

The full archived branch folders can be recovered from the recorded backup
snapshot:

```bash
bash scripts/restore.sh --multistart-job <branch_tick>_<job_id> <station_id> <tick>
```

To restore the entire selected Station from the exact finalization snapshot,
including when a later ordinary backup exists at the same tick number, use:

```bash
bash scripts/restore.sh --multistart-snapshot <branch_tick>_<job_id> <station_id> <tick>
```

An active, unfinished multistart job can also be closed with `station archive`.
It is stored through the normal content-addressed backup object pool rather
than copying every seed directory into the archive independently. Restoring
without a tick compares the archived active job's maximum recorded branch tick
with the latest ordinary backup tick:

```bash
bash scripts/restore.sh <station_id>
```

The Station Tools wrapper accepts the same UUID as well as the portable zip
itself:

```bash
station restore <station_id>
station restore /path/to/any_station_archive.zip
```

The active multistart job is restored to `station_multistart/` only when its
tick is strictly newer. Otherwise the latest ordinary snapshot is restored to
`station_data/`. Supplying an explicit tick always selects an ordinary backup.
After an active-job restore, run `./start.sh -s` to resume it.

This explicit close-down path does not enable mid-run automatic backups.
Branch workers continue to disable automatic backups, and ordinary multistart
execution still creates its authoritative backup only during finalization.
Active-job archive manifests preserve file and directory symlinks, omit stale
process/runtime state, and do not overwrite earlier manifests or backup
objects.

For multistart restores, `scripts/restore.sh` prefers the job-specific snapshot
`tick_<tick>_multistart_<branch_tick>_<job_id>.json` when it exists, and falls
back to `tick_<tick>.json` for older backups. The default output directory is
`multistart_<branch_tick>_<job_id>`. Only after the manual backup succeeds
should the controller remove the temporary
`station_multistart/<branch_tick>_<job_id>/` directory.

## Crash And Intervention Semantics

The controller should be idempotent:

- Re-running `./start.sh` resumes an incomplete job.
- Re-running `./start.sh -s` during a multistart job is the intended recovery
  path after transient API/runtime/coder issues: it restarts the controller if
  needed, keeps the lightweight multistart frontend active, and retries incomplete branch
  workers from durable job state.
- A branch, interview, or administrator failure halts the durable job but does
  not terminate the controller. The controller keeps its Unix socket available
  so the frontend Resume action can retry the unfinished work in place.
- The controller IPC listener recreates its socket if the socket path or listener
  disappears unexpectedly. If the controller process itself is missing or
  unresponsive, the frontend Resume request serializes recovery, starts a new
  controller from durable state, and retries the Resume request automatically.
- Existing completed branch directories are not rerun.
- A branch is not `completed` until its roll ticks have finished, active
  coder/evaluator work has completed, and incognito interviews have completed.
- Dead branch pids are detected from `state.yaml` and process checks.
- Failed branches are marked with logs preserved and may be retried on
  controller resume. Persistent failures remain visible in `state.yaml` and
  branch logs for manual intervention.
- Admin selection is automatic through Codex when a Codex executable is
  available. It inherits the same `CliJobManager` subprocess loop used by the
  Research Coder and Archive Surveyor. Admin launches explicitly use
  `MULTISTART_ADMIN_MODEL_NAME` (default `gpt-5.6-sol`) rather than inheriting
  the model selected in the user's Codex config. During active-job recovery,
  the admin resolves this override from the preserved
  `origin_station_data/constant_config.yaml`. Retryable provider/backend failures
  resume the same Codex thread with the Research Coder's configured resume backoff and
  `RESEARCH_CODER_MAX_RESUMES` budget. If that budget is exhausted or no resume
  token is available, selection falls back to a fresh Codex spawn, up to
  `RESEARCH_CODER_MAX_SPAWNS` total fresh spawns. Each launch keeps its prompt,
  transcript, stderr, and final-message artifacts under `admin/sessions/`, and
  counters are persisted under `admin_selection` in `state.yaml` so controller
  restarts can recover the selection stage. Older in-place
  `admin/transcript.jsonl` sessions are imported and resumed when possible.
  The shared Codex backend also applies the dedicated `CODEX_API_KEY` plus
  `CODEX_BASE_URL` `alt` provider override to both fresh and resumed Admin
  sessions when both variables are set.
- If the final admin spawn and resume budget produce no valid reports, the job
  is marked failed/paused for manual intervention. An explicit controller
  restart clears the exhausted counters and starts a fresh admin retry budget,
  matching Coder/Surveyor restart recovery, while preserving all prior admin
  artifacts for inspection.
- Finalization is resumable: archive/copy all material first, replace live
  `station_data`, post guidance once, remove archived `origin_station_data`,
  create the manual backup once, clear `current_job.yaml`, then delete temp.
- During the final handoff, the old preview process keeps its statistics drain
  endpoint available with zero active jobs after `current_job.yaml` is cleared.
  This lets the normal safe-stop path replace the preview without requiring a
  forced restart.
- Runtime API updates are held in memory and are not part of crash recovery.
- Stagnation branch copying waits for active coder/evaluator work to complete,
  not merely for the orchestrator pause flag.

The controller should never remove temporary job-folder `origin_station_data`
until the selected branch and audit folder have been fully prepared. The
archived copy under live `station_data/multistart/<job>/origin_station_data` is
removed before the finalization backup because the source state was already
captured by the pre-branch backup.

## Implementation Boundaries

Most logic should live outside the core station:

- multistart controller, branch queue, admin selection, finalization, and
  restart behavior live in a new repo-local controller module.
- normal station code only needs:
  - a data-root env override,
  - six constants,
  - a small stagnation pre-assignment deferral hook,
  - branch-worker bypass for that hook,
  - a static waiting-page mode for active multistart jobs.

The public `station multi_init` command and its legacy Station Tools
implementation are removed. Normal users control multistart through constants
and `./start.sh`.

## Future Frontend Extension

The fixed seed-1 preview intentionally avoids arbitrary branch switching.
However, the controller and frontend should remain structured so a future
richer read-only UI does not require a material rewrite.

Keep these boundaries from the start:

- The controller owns job state and exposes read-only job status through a
  narrow local interface.
- The branch monitor and seed-1 preview get their data through explicit
  controller/read-model interfaces.
- Branches are represented by explicit descriptors in `state.yaml`: seed
  number, data root, status, pid, current tick, top score, task-defined top
  sort key, and log path.
- Branch workers do not depend on the main frontend process.
- The normal Station backend remains single-station; do not add request-level
  global data-root switching.

A future Level 3 UI can then add read-only seed selection by either:

- reading branch summaries through the controller and showing read-only
  previews; or
- launching one read-only dashboard service per branch behind a controller
  proxy.

Interactive branch controls remain out of scope because they would change
selection fairness and require stronger worker IPC ownership.

## Verification

Focused tests should cover:

- disabled constants leave `./start.sh` behavior unchanged and create no job
  folder.
- init multistart queues N seeds with max-parallel K.
- stagnation multistart defers before lane assignment.
- active multistart removes live `station_data`, serves the read-only seed-1
  dashboard at the normal URL, and serves the branch monitor at `/multistart`.
- the seed-1 preview performs no branch copy, creates no Station/Orchestrator,
  uses query-only SQLite reads, and rejects Station mutation requests.
- stagnation copy waits for active coder/evaluator completion.
- each stagnation branch samples lane assignment independently.
- completed branches run incognito interviews before admin selection.
- branch workers do not write normal repo backups.
- finalization replaces live `station_data`, archives all branches under
  `station_data/multistart/<branch_tick>_<job_id>/`, creates a manual backup,
  prunes bulky branch folders only after backup verification, and removes temp
  job storage only after backup success.
- shared-base Research storage gives every seed an isolated marked UUID target,
  tracks an already-managed origin, survives branch reset/resume and active-job restore, is archived
  as ordinary file content, promotes the selected UUID as live storage, and
  removes only obsolete owned allocations after verified backup/pruning.
- an unavailable or read-only storage base blocks before live `station_data` is
  moved and leaves a retryable pending request.
- interrupted controller jobs resume without rerunning completed branches or
  deleting original paused data.
- branch creation exposes waiting-page copy progress, fans out every enabled
  seed concurrently, preserves all per-seed allocation manifest entries, and
  resumes interrupted copies from state written before live data is moved.
