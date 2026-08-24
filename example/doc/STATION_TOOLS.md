# Station Tools

Developer notes for the multi-station CLI in `station_tools/`.

## Scope

The `station` console command is a multi-checkout operations tool. It replaces local one-off scripts such as update/resume/API-refresh/archive/init wrappers while keeping command-specific behavior in shared Python modules.

Commands:

- `station list`
- `station monitor`
- `station update`
- `station resume`
- `station refresh-api`
- `station archive`
- `station restore`
- `station init`

User-facing installation and basic usage are documented in `station_tools/README.md`.

## Configuration

Default config path:

```text
~/.config/station-tools/station_tools.toml
```

If that file is absent, station discovery defaults to:

```toml
station_patterns = [
  "~/station",
  "~/station_*",
]
```

Do not add repo-local `station_tools.toml` behavior. Multiple station checkouts can install the package, so config must live outside individual checkouts to avoid ambiguity.

Local machine hooks belong in the user config, not in committed code or docs. Hooks are intentionally generic and run private local commands for tasks such as proxy setup or API profile setup.

Any Station Tools command that starts or restarts a station with `start.sh` must run the configured startup hook in the same shell immediately before `start.sh`, following the `station update` pattern. In practice, this means using the configured environment, allowing shell setup such as `source ...` or alias/function definitions to take effect, running `update.before_start` or the command-specific startup hook, and then invoking `./start.sh -s` or `./start.sh --name ...` without switching shells between the hook and start command.

## Shared Modules

- `station_tools/config.py`: user config loading and hook environment expansion.
- `station_tools/selectors.py`: station glob discovery, suffix parsing, duplicate removal, and path selection. Use this instead of reimplementing `1`, `2`, `abc`, `station_abc`, or explicit-path parsing.
- `station_tools/repo.py`: station metadata, `.env` parsing, nginx port parsing, and PID/process helpers.
- `station_tools/frontend_api.py`: local dashboard/frontend API discovery and JSON requests. Use this for API access from Station Tools commands instead of duplicating auth, port fallback, HTTPS handling, or proxy bypass logic.
- `station_tools/hooks.py`: optional local shell hook execution.

## Frontend API Access

Use `station_tools.frontend_api.find_endpoint(repo, probe_path, timeout=...)` to discover a reachable local dashboard endpoint.

The helper checks station `.env` plus OS environment overrides for:

- `FLASK_AUTH_USERNAME`
- `FLASK_AUTH_PASSWORD`
- `FLASK_PORT`
- `NGINX_HTTP_PORT`
- `NGINX_HTTPS_PORT`

It tries HTTPS first, then HTTP/nginx when configured, then Flask. HTTPS uses an unverified local SSL context because deployment certificates are local/self-signed. Requests bypass proxy settings through `urllib.request.ProxyHandler({})`.

Use `station_tools.frontend_api.request_json(...)` for subsequent API calls once an endpoint is found.

Current API-using commands:

- `station resume`: probes `/api/orchestrator/status`, then posts `/api/orchestrator/resume`.
- `station refresh-api`: probes and updates `/api/station/api_runtime_config`.
- `station monitor`: delegates to `scripts/monitor_station.py`, which has its own status-fetching logic.
  When a pending init or stagnation multistart is blocked by the disk-space guard,
  the status column reads the durable pending request and reports the estimated
  additional GiB needed instead of showing the orchestrator's generic
  "waiting for controller branch selection" pause reason.
  During multistart it ranks branch top submissions by the task-defined
  `top_sort_key` persisted in each branch station config, falling back to the
  default `(top_score,)` ordering for older configs.
  The `SINCE BT` column reads the persisted `stagnation_counter` from station
  config. Top score and breakthrough age do not require evaluation-YAML scans
  or Research SQLite reads.
  The monitor header includes simple disk usage as percent and used/total GiB
  for the filesystem(s) containing discovered station folders. With color
  enabled, disk percentages are green below 70%, yellow from 70%, and red from
  90%.

## Command Notes

`station update`:

- Selects station repos with git and `start.sh`.
- Runs one tmux window per selected station.
- Active stations are the same checkouts shown in `station monitor`: repos with `station_data/station_config.yaml` or an active repo-local multistart job under `station_multistart/`.
- For active stations, first stops and verifies the old station/controller, then runs git pull, optional retry hook, copies template prompt files and `codex.md`, runs the optional start hook, and finally runs `./start.sh -s`.
- Prompt files and `codex.md` are copied from the active station's persisted `station_template_source`. Existing
  stations without that `station_config.yaml` field fall back to `example/station/default`.
- Legacy flat persisted sources such as `example/station_default` and `example/station_gpt-5-5` are normalized
  to their nested `example/station/<name>` paths before runner creation.
- During an active multistart job, template files are refreshed in `origin_station_data/` and every existing
  `station_data_s*/` branch root instead of creating a stray top-level `station_data/` directory.
- With `--force`, passes `--force` to active-station `start.sh` restarts, which makes `start.sh` pass `--force` through to `stop.sh`.
- Every normal `stop.sh` path verifies that the checkout has no surviving multistart controller or branch-worker process; `start.sh` aborts instead of restarting when that verification fails.
- For inactive repos, runs the git update only and skips prompt copying, the start hook, and `./start.sh`.
- Preserves the old status-table/wait-mode behavior at the command level.

`station archive`:

- This is the destructive close-down command.
- Stops selected stations, removes live `station_data`, zips `backup/<station_id>`, and removes the unzipped backup directory unless `--keep-backup-dir` or `KEEP_BACKUP_DIR=1` is used.
- For ordinary stations whose Research storage is a managed remote allocation, the command reads the station-owned symlink target before removing `station_data` and removes that remote allocation only after the archive zip passes verification. It refuses to delete remote storage whose allocation marker does not match the station.
- Active multistart jobs are a special case because live `station_data` is
  normally absent. The command reads identity and progress from the active job,
  force-stops the controller/branches, and stores `station_multistart/` as a
  content-addressed manifest under `backup/<station_id>/multistart_archives/`.
  Objects are shared with ordinary backup snapshots and across all branches, so
  repeated origin/seed files are stored once. Runtime sockets, PID files,
  rebuildable indexes, sync state, and Research temporary directories are
  omitted. File and directory symlinks remain symlinks in the manifest and are
  recreated on restore. Existing object files and earlier active-job manifests
  are never overwritten. The raw `station_multistart/` tree is removed only
  after the zip has been created and verified; every failure before that point
  leaves the original tree in place.
- Active-multistart zips follow the ordinary naming convention with only an
  `_ms` suffix before `.zip`. UUID lookup inspects the embedded archive metadata,
  so the filename does not need to contain the UUID.
- This special snapshot is created only by the explicit `station archive`
  command. It does not alter ordinary automatic backup behavior: branch
  workers retain `STATION_MULTISTART_BRANCH=1`, automatic backups remain
  disabled while branches roll, and the controller's normal finalization
  backup remains the only automatic backup of a multistart run.

`station restore`:

- Runs the target checkout's `scripts/restore.sh` and accepts either a station
  UUID/prefix or a complete archive zip filename/path.
- Operates on the current Station checkout, falling back to the checkout that
  installed the command when invoked elsewhere.
- Zip extraction is path-validated, uses a temporary directory, and refuses to
  overwrite an existing `backup/<station_id>` directory.
- A zip passed directly as the restore source is deleted only after extraction
  and the complete restore succeed. It is retained on cancellation or failure.
- Existing restore targets containing intentionally read-only directories are
  made owner-accessible before removal; directory symlinks are not followed.
- For portable active-multistart archives, UUID lookup searches both `backup/`
  and the checkout root for the copied zip. Passing the zip path directly works
  for any ordinary or multistart Station archive filename.
- With no explicit tick, restore compares the newest active-multistart manifest
  tick against the newest ordinary snapshot tick. Active multistart wins only
  when strictly newer; ties restore the ordinary snapshot.
- Honors `YES=1` for noninteractive confirmation.

`station init`:

- Initializes one station checkout from a station template and a grouped Research task bundle.
- Defaults to the current Station checkout, so `station init book` is enough
  when run at its root. The station display name is optional and defaults from
  the task leaf. Use `--station-id <id-or-path>` to select another checkout;
  the legacy positional `station init <id> <task> [name]` form remains valid.
- Research tasks live at `example*/research_<group>/<task>/`, where group is
  `epoch`, `alpha_evolve`, or `misc`. A bare task name scans every group and
  both public/private roots. `group/task` restricts the group, and a canonical
  path such as `example/research_epoch/book` selects one exact bundle.
- If a task query has multiple matches, interactive use asks the user to choose.
  Noninteractive use fails with the candidate paths instead of guessing.
- Uses `--station-template <name-or-source>` to select a template. Bare names search
  `example_private/station/` before `example/station/`; explicit sources are
  restricted to those roots. The default is `example/station/default`.
- The public `example/station/gpt-5-5` template routes the Archive Reviewer, Research coder, Archive Surveyor,
  and compulsory meta-reflection sessions to GPT-5.5, replaces GPT-5.6 roster entries with GPT-5.5, sets the
  compulsory meta-reflection interval to 25 ticks and supervisor assignment cooldown to 200 ticks, and disables
  the independent Research coder audit by default.
- Persists the canonical source in `station_config.yaml` as `station_template_source` for later updates.
- Merges station-template and research-task `constant_config.yaml` mappings, with research-task values taking
  precedence on duplicate keys. A task-level `constant_config.yaml` is optional;
  when absent, the task contributes no configuration overrides.
- Writes the requested station name into the initial `station_config.yaml` before startup, so init multistart jobs and their waiting page inherit the display name.
- Supports `--post-copy-cmd '<bash command>'` after station/task files are copied and before the init hook/start step.
- Supports `--test` for quick test stations by forwarding `--test` to
  `start.sh`. Startup then disables init and stagnation multistart, writes
  `init_agents.yaml` with only `GPT-5.5` and `Gemini 3.1 Pro`, and writes
  `PAUSE_AFTER_TICK_END: 20` to `constant_config.yaml`.
- Disables both init and stagnation multistart in `constant_config.yaml` by
  default and forwards `--no-multistart` to `start.sh`.
- Supports `--multistart` as an explicit opt-in that preserves the selected
  template and task's `MULTISTART_*` settings. `--multistart`, `--test`, and
  `--no-spawn` are mutually exclusive. The old `--no-multistart` option remains
  accepted as a hidden compatibility alias for the new default.
- Supports `--no-start` to finish template/task copying, configuration merging,
  station naming, and `--post-copy-cmd` without running the
  `init.before_start` hook or `start.sh`. The prepared checkout can be started
  later with `./start.sh`.
- Supports `--no-spawn` to remove the template's `init_agents.yaml` roster
  while still starting the dashboard. Create agents manually in the dashboard,
  then launch the station.
- Runs the configured `init.before_start` hook before starting the station.

For other checkout-targeted commands, omitted targets similarly select the
current Station checkout when the command is run at its root. This applies to
`station update`, `station resume`, `station refresh-api`, and
`station archive`. Outside a Station checkout, an omitted target retains the
configured multi-station discovery behavior.

Multistart:

- The old public `station multi_init` command has been replaced by the repo-local multistart controller launched by `./start.sh`.
- Station Tools commands that start a checkout should continue to call `./start.sh -s`; the local start script handles controller launch, static waiting mode, and resume.
- `station init` writes disabled values for all six `MULTISTART_*` constants by
  default. Pass `--multistart` to preserve the selected template and task's
  values instead. The six constants are documented in `example/doc/MULTISTART.md`.
- `RESEARCH_STORAGE_BASE_PATH` is shared by normal Research storage and
  multistart UUID allocations. It may come from `constant_config.yaml`, the
  checkout `.env`, or inherited process environment; the environment value
  takes precedence over YAML.

## Tests

Focused tests:

```bash
python -m unittest tests.test_station_tools
```

For command changes that touch frontend API behavior, prefer isolated unit tests around helpers. Do not import `web_interface.app` in tests; it initializes live Station state.
