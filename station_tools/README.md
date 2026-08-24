# Station Tools

Multi-station command line tools for managing several Station checkouts from one command.

## Install

From a checkout:

```bash
pip install -e .
```

This installs the console script:

```bash
station help
```

The command is available anywhere when the Python environment's `bin` directory is on `PATH`.

For a wrapper-based install, symlink `scripts/station` into a directory on `PATH`.

## Local Config

Optional local settings live in:

```text
~/.config/station-tools/station_tools.toml
```

Each machine can use this file to choose its own station paths and hooks.

If no config exists, discovery defaults to `~/station` and `~/station_*`.

Example:

```toml
station_patterns = [
  "~/station",
  "~/station_*",
]

[env]
PATH = "/path/to/local/bin:${PATH}"

[hooks.update]
before_git_pull = "your-proxy-enable-command"
git_pull_retry = "your-proxy-disable-command"
before_start = "your-api-profile-command"

[hooks.init]
before_start = "your-api-profile-command"
```

Hooks are optional. They are intended for local setup such as private API profile commands or proxy setup.

## Commands

```bash
station list
station monitor
station update
station update 1,2,3
station update 1,2,3 --force
station resume
station resume 3
station refresh-api
station refresh-api 1,2
station archive
station archive abc
station restore abc
station restore /path/to/any_station_archive.zip
station init book
station init epoch/book "Station Name"
station init example/research_epoch/book "Station Name"
station init book "Station Name" --station-template gpt-5-5
station init task_name "Station Name" --station-template example_private/station/custom
station init task_name "Station Name" --test
station init task_name "Station Name" --no-spawn
station init task_name "Station Name" --multistart
station init task_name "Station Name" --no-start
station init kissing_margin "Station Name" --post-copy-cmd 'python example/research_alpha_evolve/kissing_margin/replace_d.py 12'
```

When run from a Station checkout, omitted selectors default to that checkout for
`init`, `update`, `resume`, `refresh-api`, and `archive`. Explicit Station
suffixes remain supported: `station_3` can be selected as `3`, and
`station_abc` as `abc`. `station init` uses `--station-id <id-or-path>` for an
explicit checkout and still accepts the legacy positional ID form. If its
display name is omitted, it derives one from the task name.

`station archive` is the destructive close-down command: it stops the station, removes live `station_data`, zips `backup/<station_id>`, and removes the unzipped backup directory unless `--keep-backup-dir` is used. After the zip passes verification, it also removes any station-owned managed remote Research storage allocation; marker mismatches are retained for safety. During active multistart it archives the unfinished job with content-addressed deduplication instead.

`station restore` accepts either a station UUID/prefix or any complete Station archive zip filename/path. It operates on the current checkout, falling back to the checkout that installed the command when invoked elsewhere. With no tick, an active multistart archive is restored only when its recorded branch tick is strictly newer than the latest ordinary snapshot; otherwise the ordinary snapshot is restored. A zip passed directly is deleted only after the complete restore succeeds and is retained on cancellation or failure.

For the portable multistart workflow, copy the zip into the checkout root or `backup/` and run `station restore <uuid>`. Alternatively, pass the zip filename/path directly; this works for ordinary Station archives too.

`station update --force` forwards `--force` to restarted stations' `start.sh`, so their pre-start `stop.sh` call bypasses pause and experiment-drain checks.

The old public `station multi_init` command has been replaced by the repo-local
multistart controller launched by `./start.sh`. `station init` disables both
initialization and stagnation multistart by default. Use `--multistart` to keep
the selected template's `MULTISTART_*` settings. See `example/doc/MULTISTART.md`.

`station init --test` forwards `--test` to `start.sh`, which applies quick-test overrides to the new `station_data`: only `GPT-5.5` and `Gemini 3.1 Pro` are spawned, and `constant_config.yaml` makes the station pause after tick 20 completes. Multistart remains disabled.

`station init --no-spawn` starts the dashboard without creating the template's
default agents. It removes `init_agents.yaml` so you can create agents manually
and launch the station from the dashboard. `--no-spawn`, `--test`, and
`--multistart` are mutually exclusive.

`station init --no-start` copies and configures `station_data`, including
`--post-copy-cmd`, but skips the `init.before_start` hook and does not run
`start.sh`. Start the prepared checkout later with `./start.sh`.

`--post-copy-cmd` runs from the repo root after `station_data` is copied and before the init hook/start step.

Research task templates are grouped under `research_epoch`,
`research_alpha_evolve`, and `research_misc` in both `example/` and
`example_private/`. A bare task name scans every group and root. Use
`epoch/book` to restrict the group or a canonical path such as
`example/research_epoch/book` to select one exact template. If multiple tasks
have the same leaf name, interactive use asks which candidate to use;
noninteractive use prints the candidates and exits.

`--station-template` selects the station template copied before the research task is installed. A bare name is
looked up in `example_private/station/` first and then `example/station/`; an explicit source must be under one of those two
directories. The default is `example/station/default`. The canonical source is saved as
`station_template_source` in `station_data/station_config.yaml`, and `station update` uses it when refreshing
prompt YAML files and `codex.md`. Existing stations without the field continue to use `example/station/default`.
Legacy flat sources such as `example/station_default` and `example/station_gpt-5-5` are normalized to the
corresponding nested template paths during update.
The public `gpt-5-5` template also disables the independent Research coder audit by default.

When both the station template and research task provide `constant_config.yaml`, their YAML mappings are merged.
Research task values take precedence for duplicate keys.
A research task may omit `constant_config.yaml` when it needs no overrides.
