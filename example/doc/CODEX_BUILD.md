# Codex Standalone Build Guide

This guide is for building a standalone Codex binary on a remote machine without affecting any existing Codex installation on that machine.

Goal:

- Keep the machine's existing `codex` command untouched.
- Build a separate standalone Codex binary in a dedicated directory.
- Add a `ccodex` command that points at the standalone configurable binary.
- Patch Codex so both timeout caps are configurable from `~/.codex/config.toml`.
- Set both timeout overrides to `1 hour` in the standard config file.
- Publish the verified binary to a shared binary directory for other station checkouts/machines.
- Point every station `.env` at the published `ccodex` binary through `CODEX_BIN_PATH`.
- Produce a binary that you can manually copy or invoke by full path.

This guide assumes Linux on `x86_64`, ideally Ubuntu/Debian. If the remote machine is different, the dependency package names may differ slightly.

## Important Constraints

Do not:

- replace the existing `codex` executable
- modify the existing global npm-installed Codex
- change shell `PATH` unless explicitly requested later

Do:

- clone source into a separate build directory
- build a standalone binary such as `~/codex-standalone/bin/codex-configurable`
- create a `ccodex` shortcut in a user-local bin directory already on `PATH`
- use the normal `~/.codex/config.toml` for the timeout overrides
- verify that binary by running it directly via full path

## What To Change

Do not hardcode `1 hour` into the Rust constants anymore.

The standard patch is:

- add a new `foreground_terminal_max_timeout` config key
- keep `background_terminal_max_timeout` as the background polling key
- keep the source defaults at the stock values: `30000` foreground, `300000` background
- clamp `exec_command` and non-empty `write_stdin` against the configurable foreground cap
- clamp empty `write_stdin` against the configurable background cap

Then set these values in the normal Codex config file:

```toml
foreground_terminal_max_timeout = 3600000
background_terminal_max_timeout = 3600000
```

That is now the recommended setup. The source keeps the stock defaults, and the standalone binary reads the 1-hour overrides from the standard `~/.codex/config.toml`.

## Recommended Directory Layout

Use something like:

- source checkout: `~/codex-build/src`
- standalone binary output: `~/codex-standalone/bin/codex-configurable`
- staged standalone binary output while replacing an existing install: `~/codex-standalone/bin/codex-configurable_tmp`
- shortcut on `PATH`: `~/.local/bin/ccodex`, `~/.cargo/bin/ccodex`, or `/usr/local/bin/ccodex`
- staged shortcut while replacing an existing install: `ccodex_tmp`

This keeps the build isolated from the machine's current Codex executable while still using the normal Codex config file.

When replacing an existing `ccodex`, stage the new command as `ccodex_tmp` first. Only remove or replace the old `ccodex` after `ccodex_tmp --version` or another direct launch succeeds.

## Standard Agent Commands

Going forward, users should use only two high-level prompts for Codex standalone binary maintenance:

```text
Please update ccodex. [Optional: Shared bin path is: /path/to/shared/bin/root]
```

```text
Please refresh ccodex. [Optional: Shared bin path is: /path/to/shared/bin/root]
```

If no shared bin path is supplied, use:

```bash
SHARED_BIN_ROOT="/mnt/stephen/template"
```

The shared binary artifact path should be a new dated directory under that root:

```bash
STAMP="$(date +%Y_%m_%d)"
PUBLISH_DIR="$SHARED_BIN_ROOT/${STAMP}_ccodex"
PUBLISHED_CODEX_BIN="$PUBLISH_DIR/ccodex.bin"
```

If the dated directory already exists, create a unique suffix such as `${STAMP}_ccodex_1`, `${STAMP}_ccodex_2`, and so on. The final file name inside the directory should stay `ccodex.bin`.

When updating station `.env` files, set this exact key to the published binary path:

```bash
CODEX_BIN_PATH=/mnt/stephen/template/2026_04_25_ccodex/ccodex.bin
```

Do not print `.env` contents while doing this. Report only file paths updated and counts.

### Update Command

For:

```text
Please update ccodex. [Optional: Shared bin path is: ...]
```

Do this:

1. Read this guide and inspect the current machine as in Step 1.
2. Update the source checkout from upstream:
   - Prefer `git -C ~/codex-build/src pull --ff-only` if the checkout is clean.
   - If local timeout-patch edits block the pull, treat `~/codex-build/src` as an isolated build checkout: use a fresh temporary checkout such as `~/codex-build/src_tmp`, patch that checkout, and build from it.
   - Do not touch the machine's existing `codex` command or any npm-installed launcher.
3. Apply or reapply the configurable-timeout patch from Step 6 against the current source layout.
4. Ensure `~/.codex/config.toml` contains the 1-hour foreground/background timeout overrides from Step 7.
5. Run the focused tests from Step 6 when practical:

```bash
cargo test -p codex-core load_config_loads_terminal_timeout_overrides
cargo test -p codex-core unified_exec_timeouts
```

6. Build the release binary with:

```bash
cargo build --release -p codex-cli
```

7. Install the new local binary through the staged path:
   - copy to `~/codex-standalone/bin/codex-configurable_tmp`
   - create/run `ccodex_tmp`
   - only after it runs, replace `~/codex-standalone/bin/codex-configurable` and `ccodex`
8. Publish the verified binary to the shared path:

```bash
mkdir -p "$PUBLISH_DIR"
install -m 755 "$HOME/codex-standalone/bin/codex-configurable" "$PUBLISHED_CODEX_BIN"
"$PUBLISHED_CODEX_BIN" --version
```

9. Scan all station checkouts, including `~/station*`, and update every existing `.env` to contain the new `CODEX_BIN_PATH=$PUBLISHED_CODEX_BIN`.

### Refresh Command

For:

```text
Please refresh ccodex. [Optional: Shared bin path is: ...]
```

Do this:

1. Locate the latest shared binary under the shared bin root:

```bash
PUBLISHED_CODEX_BIN="$(
  find "$SHARED_BIN_ROOT" -mindepth 2 -maxdepth 2 -type f -name ccodex.bin -printf '%T@ %p\n' |
  sort -nr |
  head -n 1 |
  cut -d' ' -f2-
)"
test -n "$PUBLISHED_CODEX_BIN"
```

2. Copy it into the local standalone staging path and run it:

```bash
mkdir -p "$HOME/codex-standalone/bin"
install -m 755 "$PUBLISHED_CODEX_BIN" "$HOME/codex-standalone/bin/codex-configurable_tmp"
"$HOME/codex-standalone/bin/codex-configurable_tmp" --version
```

3. Install or replace local `ccodex` only after the staged binary runs:
   - create/run `ccodex_tmp`
   - move `codex-configurable_tmp` to `codex-configurable`
   - recreate `ccodex` to point at `codex-configurable`
4. Ensure `~/.codex/config.toml` contains the 1-hour foreground/background timeout overrides from Step 7.
5. Scan all station checkouts, including `~/station*`, and update every existing `.env` to contain the new `CODEX_BIN_PATH=$PUBLISHED_CODEX_BIN`.

### Station `.env` Update Helper

Use this helper after either `update` or `refresh`. It updates only `CODEX_BIN_PATH` and does not print secret values:

```bash
find "$HOME" -maxdepth 2 -type f -path "$HOME/station*/.env" -print0 |
while IFS= read -r -d '' env_file; do
  sed -i '/^[[:space:]]*CODEX_BIN_PATH[[:space:]]*=/d' "$env_file"
  printf '\nCODEX_BIN_PATH=%s\n' "$PUBLISHED_CODEX_BIN" >> "$env_file"
  printf 'updated %s\n' "$env_file"
done
```

If a station checkout has no `.env`, do not create one unless the user explicitly asks.

## Special Note: If The Binary Is Already Built

If you already have a working `codex-configurable` binary from another machine with the same setup, you do not need to rebuild from source.

Assumptions:

- same OS and CPU architecture
- compatible runtime environment and libraries
- the prebuilt binary has already been copied onto the target machine

For this prebuilt-binary path, `cargo` does not need to be installed. `~/.cargo/bin` is only a directory name; it is not proof that Rust is installed, and you can use `~/.local/bin` instead if that directory is already on `PATH`.

Example: if the binary is already present at `/mnt/stephen/tmp/codex-configurable`, use:

```bash
mkdir -p "$HOME/codex-standalone/bin" "$HOME/.codex"

if printf '%s\n' ":$PATH:" | grep -q ":$HOME/.local/bin:"; then
  BIN_DIR="$HOME/.local/bin"
elif printf '%s\n' ":$PATH:" | grep -q ":$HOME/.cargo/bin:"; then
  BIN_DIR="$HOME/.cargo/bin"
elif printf '%s\n' ":$PATH:" | grep -q ":/usr/local/bin:" && sudo -n test -w /usr/local/bin 2>/dev/null; then
  BIN_DIR="/usr/local/bin"
else
  BIN_DIR="$HOME/.cargo/bin"
  echo "Note: $BIN_DIR is not on PATH in this shell."
fi

mkdir -p "$BIN_DIR" 2>/dev/null || sudo -n mkdir -p "$BIN_DIR"

install -m 755 /mnt/stephen/tmp/codex-configurable "$HOME/codex-standalone/bin/codex-configurable_tmp"

touch "$HOME/.codex/config.toml"
tmp="$(mktemp)"
awk '
BEGIN {
  fg = "foreground_terminal_max_timeout = 3600000"
  bg = "background_terminal_max_timeout = 3600000"
  inserted = 0
}
/^[[:space:]]*foreground_terminal_max_timeout[[:space:]]*=/ { next }
/^[[:space:]]*background_terminal_max_timeout[[:space:]]*=/ { next }
!inserted && /^[[:space:]]*\[/ {
  print fg
  print bg
  print ""
  inserted = 1
}
{ print }
END {
  if (!inserted) {
    if (NR > 0) print ""
    print fg
    print bg
  }
}
' "$HOME/.codex/config.toml" > "$tmp" && mv "$tmp" "$HOME/.codex/config.toml"

if [ -w "$BIN_DIR" ]; then
  ln -sfn "$HOME/codex-standalone/bin/codex-configurable_tmp" "$BIN_DIR/ccodex_tmp"
else
  sudo -n ln -sfn "$HOME/codex-standalone/bin/codex-configurable_tmp" "$BIN_DIR/ccodex_tmp"
fi

"$BIN_DIR/ccodex_tmp" --version

rm -f "$HOME/codex-standalone/bin/codex-configurable"
mv "$HOME/codex-standalone/bin/codex-configurable_tmp" "$HOME/codex-standalone/bin/codex-configurable"

if [ -w "$BIN_DIR" ]; then
  rm -f "$BIN_DIR/ccodex"
  ln -sfn "$HOME/codex-standalone/bin/codex-configurable" "$BIN_DIR/ccodex"
  rm -f "$BIN_DIR/ccodex_tmp"
else
  sudo -n rm -f "$BIN_DIR/ccodex"
  sudo -n ln -sfn "$HOME/codex-standalone/bin/codex-configurable" "$BIN_DIR/ccodex"
  sudo -n rm -f "$BIN_DIR/ccodex_tmp"
fi

command -v ccodex || true
readlink -f "$BIN_DIR/ccodex"
"$BIN_DIR/ccodex" --version
```

After that, start it with:

```bash
"$HOME/codex-standalone/bin/codex-configurable"
```

If `command -v ccodex` prints nothing, the shortcut directory is not on `PATH` in that shell. The binary still works; run `~/codex-standalone/bin/codex-configurable` directly, or use the explicit shortcut path such as `~/.local/bin/ccodex` or `~/.cargo/bin/ccodex`.

On many Ubuntu setups, `~/.local/bin` is added to `PATH` by `~/.profile` for login shells, not by `~/.bashrc`. That means `source ~/.bashrc` alone may still leave `ccodex` unresolved. In that case, either:

- run `source ~/.profile`
- start a fresh login shell with `exec bash -l`
- or add `export PATH="$HOME/.local/bin:$PATH"` to `~/.bashrc` if you want non-login interactive shells to find `ccodex`

## Step 1: Inspect The Current Machine

Run:

```bash
uname -a
arch
codex --version || true
which codex || true
readlink -f "$(which codex)" 2>/dev/null || true
```

This is only for information. Do not modify the existing Codex install.

## Step 2: Install Build Dependencies

For Ubuntu/Debian, start with:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  pkg-config \
  libssl-dev \
  libcap-dev \
  bubblewrap \
  curl \
  git \
  clang \
  cmake
```

Possible optional extras:

```bash
sudo apt install -y \
  libxkbcommon-dev \
  libwayland-dev \
  libx11-dev
```

If the machine is not Ubuntu/Debian, adapt package names accordingly:

- `pkg-config`
- OpenSSL development headers
- `libcap` development headers
- `bubblewrap`
- compiler toolchain

## Step 3: Install Rust

If Rust is missing:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
rustup component add rustfmt clippy
cargo --version
rustc --version
```

If Rust already exists, still run:

```bash
. "$HOME/.cargo/env"
```

If `rustup` exists but `cargo` still errors because no default toolchain is configured, repair it with:

```bash
. "$HOME/.cargo/env"
rustup set profile minimal
rustup toolchain install stable
rustup default stable
rustup component add rustfmt clippy
```

## Step 4: Create Standalone Build Directories

```bash
mkdir -p ~/codex-build
mkdir -p ~/codex-standalone/bin
```

## Step 5: Clone Codex Source

```bash
git clone https://github.com/openai/codex.git ~/codex-build/src
cd ~/codex-build/src/codex-rs
```

If you want to match the installed version more closely, inspect tags:

```bash
git fetch --tags
git tag | tail -n 50
```

If you know the desired tag, check it out before patching.

If you do not need full history or tags, a shallow clone is usually enough for a standalone build and is much faster:

```bash
git clone --depth 1 https://github.com/openai/codex.git ~/codex-build/src
cd ~/codex-build/src/codex-rs
```

## Step 6: Patch In Configurable Timeouts

Do not patch the constants to `1 hour` directly.

Instead, patch Codex so both caps are configurable and the source defaults stay at the stock values.

Apply these edits:

1. In `config/src/config_toml.rs`, add a new optional config key:

```rust
/// Maximum poll window for foreground terminal output (`exec_command` and
/// non-empty `write_stdin`), in milliseconds.
/// Default: `30000` (30 seconds).
pub foreground_terminal_max_timeout: Option<u64>,
```

2. In `core/src/config/mod.rs`:

- add `pub foreground_terminal_max_timeout: u64,` to `Config`
- load it with:

```rust
let foreground_terminal_max_timeout = cfg
    .foreground_terminal_max_timeout
    .unwrap_or(DEFAULT_MAX_YIELD_TIME_MS)
    .max(MIN_YIELD_TIME_MS);
```

- keep `background_terminal_max_timeout` loading from config as the background cap

3. In `core/src/unified_exec/mod.rs`:

- keep or rename the source defaults so the stock values remain:

```rust
pub(crate) const DEFAULT_MAX_YIELD_TIME_MS: u64 = 30_000;
pub(crate) const DEFAULT_MAX_BACKGROUND_TERMINAL_TIMEOUT_MS: u64 = 300_000;
```

Recent Codex checkouts may still call the foreground constant `MAX_YIELD_TIME_MS`. In that case, rename it to `DEFAULT_MAX_YIELD_TIME_MS` and update call sites instead of changing the numeric value.

- make `UnifiedExecProcessManager::new(...)` accept both a foreground max and a background max
- store both values in the manager

4. In `core/src/unified_exec/process_manager.rs`:

- clamp the initial `exec_command` wait against the configured foreground max
- clamp non-empty `write_stdin` against the configured foreground max
- clamp empty `write_stdin` against the configured background max
- if there is a helper like `clamp_yield_time(yield_time_ms)`, make it accept the configured foreground max instead of hardcoding `MAX_YIELD_TIME_MS`

5. Construct the manager with both configured values:

```rust
UnifiedExecProcessManager::new(
    config.foreground_terminal_max_timeout,
    config.background_terminal_max_timeout,
)
```

In older source this was in `core/src/codex.rs`. In current source it may be in `core/src/session/session.rs`, and tests or helpers may have additional `UnifiedExecProcessManager::new(...)` call sites under `core/src/session/tests.rs`. Use `rg "UnifiedExecProcessManager::new" core/src` and update every compile-time call site.

Optional but recommended:

- add or update a config-loading test in `core/src/config/config_tests.rs`
- run the focused tests:

```bash
cargo test -p codex-core load_config_loads_terminal_timeout_overrides
cargo test -p codex-core unified_exec_timeouts
```

Quick verification after editing:

```bash
rg -n "foreground_terminal_max_timeout|background_terminal_max_timeout|DEFAULT_MAX_YIELD_TIME_MS|UnifiedExecProcessManager::new" \
  config/src/config_toml.rs \
  core/src/config/mod.rs \
  core/src/session/session.rs \
  core/src/session/tests.rs \
  core/src/unified_exec/mod.rs \
  core/src/unified_exec/process_manager.rs \
  core/src/config/config_tests.rs
```

## Step 7: Set The Standard Codex Config

Use the normal Codex config file.

In `~/.codex/config.toml`, ensure these lines exist:

```toml
foreground_terminal_max_timeout = 3600000
background_terminal_max_timeout = 3600000
```

If the file already exists, replace any existing values for those keys rather than duplicating them.

Do not append these keys blindly to the end of the file if `config.toml` already contains TOML tables such as `[projects."..."]`. Insert or replace them before the first section header:

```bash
touch "$HOME/.codex/config.toml"
tmp="$(mktemp)"
awk '
BEGIN {
  fg = "foreground_terminal_max_timeout = 3600000"
  bg = "background_terminal_max_timeout = 3600000"
  inserted = 0
}
/^[[:space:]]*foreground_terminal_max_timeout[[:space:]]*=/ { next }
/^[[:space:]]*background_terminal_max_timeout[[:space:]]*=/ { next }
!inserted && /^[[:space:]]*\[/ {
  print fg
  print bg
  print ""
  inserted = 1
}
{ print }
END {
  if (!inserted) {
    if (NR > 0) print ""
    print fg
    print bg
  }
}
' "$HOME/.codex/config.toml" > "$tmp" && mv "$tmp" "$HOME/.codex/config.toml"
```

Validate without printing the whole config file:

```bash
python3 - <<'PY'
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

path = Path.home() / ".codex" / "config.toml"
data = tomllib.loads(path.read_text())
assert data.get("foreground_terminal_max_timeout") == 3600000
assert data.get("background_terminal_max_timeout") == 3600000
print("toml-ok")
PY
```

If the system `python3` is older and has no `tomllib` or `tomli`, use a Python 3.11+ interpreter such as the project conda environment when available.

## Step 8: Build The Full Release Binary

Use the full release build first:

```bash
cargo build --release -p codex-cli
```

Note:

- this repo may include `rust-toolchain.toml`
- if so, `cargo build` may first try to download that pinned compiler version before compilation starts

If successful, copy it to a staged standalone output path first:

```bash
install -m 755 target/release/codex ~/codex-standalone/bin/codex-configurable_tmp
```

Verify it exists:

```bash
ls -lh ~/codex-standalone/bin/codex-configurable_tmp
~/codex-standalone/bin/codex-configurable_tmp --version
```

Important:

- Do not overwrite `which codex`
- Do not change `PATH`
- Do not replace the npm-installed launcher

## Step 9: Create Or Replace A `ccodex` Shortcut Safely

For a prebuilt standalone binary, `cargo` does not need to be installed just to host the `ccodex` symlink. Pick a bin directory that is already on `PATH` when possible:

```bash
if printf '%s\n' ":$PATH:" | grep -q ":$HOME/.local/bin:"; then
  BIN_DIR="$HOME/.local/bin"
elif printf '%s\n' ":$PATH:" | grep -q ":$HOME/.cargo/bin:"; then
  BIN_DIR="$HOME/.cargo/bin"
elif printf '%s\n' ":$PATH:" | grep -q ":/usr/local/bin:" && sudo -n test -w /usr/local/bin 2>/dev/null; then
  BIN_DIR="/usr/local/bin"
else
  BIN_DIR="$HOME/.cargo/bin"
  echo "Note: $BIN_DIR is not on PATH in this shell."
fi
```

Stage the shortcut as `ccodex_tmp` and run it before touching any existing `ccodex`:

```bash
mkdir -p "$BIN_DIR" 2>/dev/null || sudo -n mkdir -p "$BIN_DIR"

if [ -w "$BIN_DIR" ]; then
  ln -sfn "$HOME/codex-standalone/bin/codex-configurable_tmp" "$BIN_DIR/ccodex_tmp"
else
  sudo -n ln -sfn "$HOME/codex-standalone/bin/codex-configurable_tmp" "$BIN_DIR/ccodex_tmp"
fi

"$BIN_DIR/ccodex_tmp" --version
```

Only after `ccodex_tmp` runs, replace the final binary and command:

```bash
rm -f "$HOME/codex-standalone/bin/codex-configurable"
mv "$HOME/codex-standalone/bin/codex-configurable_tmp" "$HOME/codex-standalone/bin/codex-configurable"

if [ -w "$BIN_DIR" ]; then
  rm -f "$BIN_DIR/ccodex"
  ln -sfn "$HOME/codex-standalone/bin/codex-configurable" "$BIN_DIR/ccodex"
  rm -f "$BIN_DIR/ccodex_tmp"
else
  sudo -n rm -f "$BIN_DIR/ccodex"
  sudo -n ln -sfn "$HOME/codex-standalone/bin/codex-configurable" "$BIN_DIR/ccodex"
  sudo -n rm -f "$BIN_DIR/ccodex_tmp"
fi

command -v ccodex || true
"$BIN_DIR/ccodex" --version
```

Because `ccodex_tmp` above is a symlink to the temporary binary, recreate the final `ccodex` symlink after the binary is moved. If your staged command is a real wrapper script or executable rather than a symlink, moving `ccodex_tmp` to `ccodex` is fine.

If `command -v ccodex` prints nothing, use the full shortcut path or the standalone binary directly.

If you chose `~/.local/bin`, remember that `source ~/.bashrc` may still not expose it if your machine adds that directory in `~/.profile` instead. Use `source ~/.profile` or start a fresh login shell to verify.

## Step 10: Test The Standalone Binary

After editing `~/.codex/config.toml`, restart using the standalone binary by full path:

```bash
~/codex-standalone/bin/codex-configurable
```

Then ask that Codex session to run the timing experiment:

```bash
date -Is; sleep 360; date -Is
```

Expected:

- the command should return after about 6 minutes
- it should not yield at about 30 seconds

After that, test a longer wait:

```bash
date -Is; sleep 3700; date -Is
```

Expected:

- it should remain blocked until completion if under 1 hour
- if over 1 hour, it should yield around 1 hour, not 30 seconds

## Common Failures And Fixes

### 1. `cargo: command not found`

Install Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
```

### 2. OpenSSL not found

Typical error text:

- `Could not find openssl via pkg-config`
- `openssl-sys could not find directory of OpenSSL installation`

Fix on Ubuntu:

```bash
sudo apt install -y pkg-config libssl-dev
```

If OpenSSL is in a custom prefix, set:

```bash
export OPENSSL_DIR=/path/to/prefix
export OPENSSL_LIB_DIR=/path/to/prefix/lib
export OPENSSL_INCLUDE_DIR=/path/to/prefix/include
cargo build --release -p codex-cli
```

Example with Conda:

```bash
export OPENSSL_DIR=$HOME/miniconda3/envs/station
export OPENSSL_LIB_DIR=$HOME/miniconda3/envs/station/lib
export OPENSSL_INCLUDE_DIR=$HOME/miniconda3/envs/station/include
cargo build --release -p codex-cli
```

This should be treated as a fallback, not the first choice.

### 3. `pkg-config` missing

Typical error:

- `The pkg-config command could not be found`

Fix:

```bash
sudo apt install -y pkg-config
```

### 4. Vendored bubblewrap / `libcap` build failure

Typical error:

- `failed to compile vendored bubblewrap`
- `libcap not available via pkg-config`

Fix:

```bash
sudo apt install -y libcap-dev pkg-config bubblewrap
```

This is the correct fix if you want a full normal Codex build.

Do not use:

```bash
CODEX_SKIP_VENDORED_BWRAP=1
```

unless you are explicitly making a temporary test-only build. That workaround can break sandboxed execution at runtime.

### 5. `bwrap` missing at runtime

Install:

```bash
sudo apt install -y bubblewrap
```

### 6. Release build is too slow

This repo's release profile can be very heavy.

On one real build of this guide:

- the full release build completed successfully, but took about 36 minutes
- the final `rustc --crate-name codex` step stayed silent for about 20 minutes
- that long silent phase was still active work, not a hang

On another current `x86_64` Ubuntu build from a shallow checkout:

- installing missing build deps needed `libssl-dev`, `libcap-dev`, `clang`, and `cmake`
- the pinned `1.93.0` Rust toolchain downloaded automatically after `rustup` installed stable
- the focused config test took about 5 minutes on the first compile because it fetched and compiled dependencies
- the full release build completed in about 27 minutes after the test cache was warm
- `codex-app-server` emitted a dead-code warning, but the release build still succeeded

If you only need to verify behavior quickly, you can use:

```bash
cargo build -p codex-cli
```

and then copy:

```bash
cp target/debug/codex ~/codex-standalone/bin/codex-configurable
```

But for the final standalone binary you requested, prefer:

```bash
cargo build --release -p codex-cli
```

If you need to confirm the build is still alive during the silent final step, inspect the process list:

```bash
ps -eo pid,ppid,pcpu,pmem,etime,cmd | rg 'cargo build --release -p codex-cli|rustc .*codex'
```

`cargo fmt` can print warnings like `can't set imports_granularity = Item` on stable Rust. Those warnings are from unstable rustfmt options and are not build failures.

### 7. Cargo fetch / git dependency failures

Try:

```bash
cargo fetch
cargo build --release -p codex-cli
```

If DNS or GitHub access is broken, fix the machine/network first.

### 8. Binary builds, but shows version `0.0.0`

This can happen in local source builds and is not by itself a failure.

The important checks are:

- does the binary launch?
- does the timing experiment behave correctly?

### 9. Existing Codex should remain untouched

To verify:

```bash
which codex
readlink -f "$(which codex)"
ls -lh ~/codex-standalone/bin/codex-configurable
command -v ccodex || true
readlink -f ~/.local/bin/ccodex 2>/dev/null || true
readlink -f ~/.cargo/bin/ccodex 2>/dev/null || true
```

You should run the standalone binary directly by path:

```bash
~/codex-standalone/bin/codex-configurable
```

That avoids touching the machine's existing Codex agent or launchers. The binary is separate even though it reads the normal `~/.codex/config.toml`.

### 10. `rustup` exists, but `cargo` says no default toolchain is configured

Typical error:

- `rustup could not choose a version of cargo to run`
- `no default is configured`

Fix:

```bash
. "$HOME/.cargo/env"
rustup set profile minimal
rustup toolchain install stable
rustup default stable
rustup component add rustfmt clippy
cargo --version
rustc --version
```

This happened on a real build after an incomplete Rust install and was fixed by explicitly reinstalling and selecting the default toolchain.

### 11. Pinned Rust toolchain download fails even though Rust is installed

This repo may pin a toolchain in `rust-toolchain.toml`. A real build hit this failure:

- `could not download file from 'https://static.rust-lang.org/dist/channel-rust-1.93.0.toml.sha256'`
- `tls handshake eof`

Preferred fix:

```bash
. "$HOME/.cargo/env"
rustup toolchain install 1.93.0
cargo +1.93.0 build --release -p codex-cli
```

If that still fails because `rustup` cannot fetch the pinned toolchain, and you already have a newer stable compiler installed, a fallback that worked on one real build was:

```bash
cargo +stable build --release -p codex-cli
```

Treat that as a fallback, not the first choice. The preferred path is still to build with the pinned toolchain when the network allows it.

### 12. `apt update` is extremely slow because of unrelated third-party repositories

This can look like the machine is hung even when the Codex build dependencies themselves are normal.

Observed symptom on a real build:

- `apt update` spent most of its time waiting on an unrelated external repository, not Ubuntu itself

What to do:

- let `apt update` finish if it is still making progress
- if the slowdown is caused by an unrelated third-party source, temporarily disable that source, install the build prerequisites, then re-enable it if needed later

This is a machine hygiene issue, not a Codex-specific failure.

## Minimal Clean Command Sequence

If the machine is a normal Ubuntu box and you want the cleanest full standalone build:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  pkg-config \
  libssl-dev \
  libcap-dev \
  bubblewrap \
  curl \
  git \
  clang \
  cmake

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
rustup component add rustfmt clippy

mkdir -p ~/codex-build
mkdir -p ~/codex-standalone/bin

mkdir -p ~/.codex

git clone https://github.com/openai/codex.git ~/codex-build/src
cd ~/codex-build/src/codex-rs

# apply the configurable-timeout source edits from Step 6
# then ensure ~/.codex/config.toml contains:
# foreground_terminal_max_timeout = 3600000
# background_terminal_max_timeout = 3600000

cargo test -p codex-core load_config_loads_terminal_timeout_overrides
cargo test -p codex-core unified_exec_timeouts
cargo build --release -p codex-cli

install -m 755 target/release/codex "$HOME/codex-standalone/bin/codex-configurable_tmp"

if printf '%s\n' ":$PATH:" | grep -q ":$HOME/.local/bin:"; then
  BIN_DIR="$HOME/.local/bin"
elif printf '%s\n' ":$PATH:" | grep -q ":$HOME/.cargo/bin:"; then
  BIN_DIR="$HOME/.cargo/bin"
elif printf '%s\n' ":$PATH:" | grep -q ":/usr/local/bin:" && sudo -n test -w /usr/local/bin 2>/dev/null; then
  BIN_DIR="/usr/local/bin"
else
  BIN_DIR="$HOME/.cargo/bin"
  echo "Note: $BIN_DIR is not on PATH in this shell."
fi

mkdir -p "$BIN_DIR" 2>/dev/null || sudo -n mkdir -p "$BIN_DIR"

if [ -w "$BIN_DIR" ]; then
  ln -sfn "$HOME/codex-standalone/bin/codex-configurable_tmp" "$BIN_DIR/ccodex_tmp"
else
  sudo -n ln -sfn "$HOME/codex-standalone/bin/codex-configurable_tmp" "$BIN_DIR/ccodex_tmp"
fi

"$BIN_DIR/ccodex_tmp" --version

rm -f "$HOME/codex-standalone/bin/codex-configurable"
mv "$HOME/codex-standalone/bin/codex-configurable_tmp" "$HOME/codex-standalone/bin/codex-configurable"

if [ -w "$BIN_DIR" ]; then
  rm -f "$BIN_DIR/ccodex"
  ln -sfn "$HOME/codex-standalone/bin/codex-configurable" "$BIN_DIR/ccodex"
  rm -f "$BIN_DIR/ccodex_tmp"
else
  sudo -n rm -f "$BIN_DIR/ccodex"
  sudo -n ln -sfn "$HOME/codex-standalone/bin/codex-configurable" "$BIN_DIR/ccodex"
  sudo -n rm -f "$BIN_DIR/ccodex_tmp"
fi

~/codex-standalone/bin/codex-configurable --version
"$BIN_DIR/ccodex" --version
command -v ccodex || true
```

## Final Verification

Start:

```bash
~/codex-standalone/bin/codex-configurable
```

Then run the 6-minute experiment:

```bash
date -Is; sleep 360; date -Is
```

If it returns after about 6 minutes instead of about 30 seconds, the patch is working.
