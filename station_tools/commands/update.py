from __future__ import annotations

import os
import shlex
import subprocess
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

from station_tools.config import ToolsConfig, build_hook_env
from station_tools.selectors import select_repos, targets_or_current
from station_tools.station_templates import configured_station_template_source


def add_parser(subparsers) -> None:
    parser: ArgumentParser = subparsers.add_parser("update", help="Git pull and restart selected station checkouts in tmux")
    parser.add_argument("targets", nargs="*", help="Station ids, suffixes, names, or paths")
    parser.add_argument("--wait-seconds", type=int, default=int(os.environ.get("WAIT_SECONDS", "7200")))
    parser.add_argument("--wait-mode", choices=("exit", "launch"), default=os.environ.get("WAIT_MODE", "exit"))
    parser.add_argument("--poll-seconds", type=int, default=int(os.environ.get("POLL_SECONDS", "2")))
    parser.add_argument("--status-last-width", type=int, default=int(os.environ.get("STATUS_LAST_WIDTH", "100")))
    parser.add_argument("--git-pull-timeout", type=int, default=int(os.environ.get("GIT_PULL_TIMEOUT_SECONDS", "300")))
    parser.add_argument("--keep-tmux", action="store_true", default=os.environ.get("KEEP_TMUX", "0") == "1")
    parser.add_argument("--no-clear", action="store_true", default=os.environ.get("NO_CLEAR", "0") == "1")
    parser.add_argument("--no-hooks", action="store_true", help="Skip configured local hooks")
    parser.add_argument("--force", action="store_true", help="Pass --force to start.sh when restarting active stations")
    parser.set_defaults(func=run)


def _quote(value: str) -> str:
    return shlex.quote(value)


def _hook(config: ToolsConfig, disabled: bool, scope: str, name: str) -> str:
    if disabled:
        return ""
    scoped = config.hooks.get(scope, {})
    return scoped.get(name) or config.hooks.get("global", {}).get(name) or ""


def _env_exports(config: ToolsConfig) -> str:
    env = build_hook_env(config)
    lines = []
    for key, value in config.env.items():
        resolved = env.get(key, str(value))
        lines.append(f"export {key}={_quote(resolved)}")
    return "\n".join(lines)


def _make_runner(
    repo: Path,
    name: str,
    runner: Path,
    status: Path,
    log: Path,
    config: ToolsConfig,
    args: Namespace,
    start_after_update: bool,
) -> None:
    before_git_pull = _hook(config, args.no_hooks, "update", "before_git_pull")
    git_pull_retry = _hook(config, args.no_hooks, "update", "git_pull_retry")
    before_start = _hook(config, args.no_hooks, "update", "before_start")
    env_exports = _env_exports(config)
    station_template_source = configured_station_template_source(repo)
    runner.write_text(
        f"""#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s expand_aliases || true

repo={_quote(str(repo))}
name={_quote(name)}
status_file={_quote(str(status))}
log_file={_quote(str(log))}
started=0
git_pull_timeout={int(args.git_pull_timeout)}
before_git_pull={_quote(before_git_pull)}
git_pull_retry={_quote(git_pull_retry)}
before_start={_quote(before_start)}
start_after_update={1 if start_after_update else 0}
force_start={1 if getattr(args, "force", False) else 0}
station_template_source={_quote(station_template_source)}

{env_exports}

write_status() {{
  printf '%s\\n' "$1" >"$status_file"
}}

run_hook() {{
  local hook_name=$1
  local hook_command=$2
  [[ -n "$hook_command" ]] || return 0
  printf 'running update.%s hook\\n' "$hook_name"
  eval "$hook_command"
}}

run_git_pull_once() {{
  if [[ "$git_pull_timeout" -gt 0 ]]; then
    timeout "$git_pull_timeout" git pull
  else
    git pull
  fi
}}

git_pull_with_retry() {{
  local rc retry_rc
  set +e
  run_hook before_git_pull "$before_git_pull"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    run_git_pull_once
    rc=$?
  fi
  set -e
  if [[ $rc -eq 0 ]]; then
    return 0
  fi
  printf 'git pull failed rc=%s; running retry hook and retrying once\\n' "$rc"
  set +e
  run_hook git_pull_retry "$git_pull_retry"
  run_git_pull_once
  retry_rc=$?
  set -e
  if [[ $retry_rc -ne 0 ]]; then
    printf 'git pull retry failed rc=%s; user intervention required for %s\\n' "$retry_rc" "$repo"
    write_status "NEEDS_INTERVENTION:$retry_rc:$(date -Is)"
    exit 0
  fi
}}

on_exit() {{
  local rc=$?
  if [[ $rc -ne 0 && $started -eq 0 ]]; then
    write_status "FAILED:$rc:$(date -Is)"
  fi
  return $rc
}}

trap on_exit EXIT
exec > >(awk -v station="$name" '{{ print "[" strftime("%Y-%m-%dT%H:%M:%S%z") "] [" station "] " $0; fflush() }}' | tee -a "$log_file") 2>&1

write_status "RUNNING:$(date -Is)"
printf 'refresh started in %s\\n' "$repo"
cd "$repo"
if [[ "$start_after_update" -eq 1 ]]; then
  stop_args=()
  if [[ "$force_start" -eq 1 ]]; then
    stop_args+=(--force)
  fi
  printf 'stopping existing station before git pull: ./stop.sh %s\\n' "${{stop_args[*]}}"
  ./stop.sh "${{stop_args[@]}}"
fi
git_pull_with_retry
if [[ "$start_after_update" -ne 1 ]]; then
  write_status "UPDATED:$(date -Is)"
  printf 'git update complete; station is not active, skipping start.sh\\n'
  exit 0
fi

template_dir="$repo/$station_template_source"
if [[ ! -d "$template_dir" ]]; then
  printf 'configured station template is missing after git pull: %s\n' "$template_dir" >&2
  exit 1
fi
python -m station_tools.station_templates "$repo" "$station_template_source"
run_hook before_start "$before_start"

start_args=(-s)
if [[ "$force_start" -eq 1 ]]; then
  start_args+=(--force)
fi

started=1
write_status "STARTED:$(date -Is)"
printf 'refresh complete; launching ./start.sh %s\\n' "${{start_args[*]}}"

set +e
./start.sh "${{start_args[@]}}"
rc=$?
set -e
write_status "SERVER_EXITED:$rc:$(date -Is)"
printf './start.sh %s exited with rc=%s\\n' "${{start_args[*]}}" "$rc"
exit "$rc"
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)


def _status_label(status: str) -> str:
    if status.startswith("PENDING:"):
        return "pending"
    if status.startswith("RUNNING:"):
        return "refreshing"
    if status.startswith("STARTED:"):
        return "start-running"
    if status.startswith("UPDATED:"):
        return "git-updated"
    if status.startswith("SERVER_EXITED:0:"):
        return "exited-0"
    if status.startswith("SERVER_EXITED:"):
        return "exited-" + status.split(":", 2)[1]
    if status.startswith("FAILED:"):
        return "failed-" + status.split(":", 2)[1]
    if status.startswith("NEEDS_INTERVENTION:"):
        return "needs-user"
    return status


def _latest_log_line(log: Path, width: int) -> str:
    if not log.exists() or log.stat().st_size == 0:
        return "(no output yet)"
    line = log.read_text(encoding="utf-8", errors="replace").splitlines()[-1]
    if "] " in line:
        line = line.rsplit("] ", 1)[-1]
    return line if len(line) <= width else line[: max(0, width - 3)] + "..."


def _render_status_table(
    repos: tuple[Path, ...],
    status_files: dict[Path, Path],
    log_files: dict[Path, Path],
    session: str,
    run_dir: Path,
    args: Namespace,
    deadline: float,
) -> None:
    if os.isatty(1) and not args.no_clear:
        print("\033[H\033[2J", end="")
    remaining = max(0, int(deadline - time.time()))
    print(f"station update status  session={session}  mode={args.wait_mode}  remaining={remaining}s")
    print(f"run files: {run_dir}\n")
    print(f"{'REPO':<16} {'STATE':<15} LAST OUTPUT")
    print(f"{'----':<16} {'-----':<15} -----------")
    for repo in repos:
        status = status_files[repo].read_text(encoding="utf-8").strip()
        print(f"{repo.name:<16.16} {_status_label(status):<15.15} {_latest_log_line(log_files[repo], args.status_last_width)}")


def _tmux_available() -> bool:
    return subprocess.run(["bash", "-lc", "command -v tmux"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _station_is_active_for_update(repo: Path) -> bool:
    return (
        (repo / "station_data" / "station_config.yaml").is_file()
        or (repo / "station_multistart" / "current_job.yaml").is_file()
        or (repo / "station_multistart" / "current_job").is_file()
    )


def run(args: Namespace, config: ToolsConfig) -> int:
    if args.poll_seconds < 1:
        print("--poll-seconds must be >= 1")
        return 2
    if not _tmux_available():
        print("error: missing required command: tmux")
        return 1

    selection = select_repos(
        targets_or_current(args.targets),
        config.station_patterns,
        require_git=True,
        require_start=True,
    )
    if not selection.repos:
        print("no valid station repos selected")
        return 1

    session_prefix = os.environ.get("STATION_SESSION_PREFIX", "station_run")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = f"{session_prefix}_{stamp}"
    run_dir = Path("/tmp") / f"update_station_{os.environ.get('USER', 'user')}_{stamp}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    status_files: dict[Path, Path] = {}
    log_files: dict[Path, Path] = {}

    print(f"Starting {len(selection.repos)} station repo(s) in tmux session: {session}")
    print(f"Run files: {run_dir}\n")

    first = True
    for repo in selection.repos:
        safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in repo.name)
        runner = run_dir / f"{safe}.runner.sh"
        status = run_dir / f"{safe}.status"
        log = run_dir / f"{safe}.log"
        status.write_text(f"PENDING:{datetime.now().isoformat()}\n", encoding="utf-8")
        start_after_update = _station_is_active_for_update(repo)
        _make_runner(repo, repo.name, runner, status, log, config, args, start_after_update)
        status_files[repo] = status
        log_files[repo] = log

        if args.keep_tmux:
            shell_cmd = f"bash -i {_quote(str(runner))}; rc=$?; printf '\\n[%s] runner exited with rc=%s\\n' \"$(date -Is)\" \"$rc\"; exec \"${{SHELL:-/bin/bash}}\" -i"
        else:
            shell_cmd = f"bash -i {_quote(str(runner))}"
        if first:
            subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", safe, shell_cmd], check=True)
            first = False
        else:
            subprocess.run(["tmux", "new-window", "-d", "-t", f"{session}:", "-n", safe, shell_cmd], check=True)
        mode = "restart" if start_after_update else "git-only"
        print(f"  {repo.name:<20} -> {repo} ({mode})")

    if selection.skipped:
        print("\nSkipped paths that do not look like station repos:")
        for item in selection.skipped:
            print(f"  {item}")

    deadline = time.time() + args.wait_seconds
    print(f"\nWaiting up to {args.wait_seconds}s for mode={args.wait_mode}...")
    while True:
        _render_status_table(selection.repos, status_files, log_files, session, run_dir, args, deadline)
        all_done = True
        for repo in selection.repos:
            status = status_files[repo].read_text(encoding="utf-8").strip()
            if args.wait_mode == "launch":
                done = status.startswith(("STARTED:", "UPDATED:", "SERVER_EXITED:", "FAILED:", "NEEDS_INTERVENTION:"))
            else:
                done = status.startswith(("UPDATED:", "SERVER_EXITED:", "FAILED:", "NEEDS_INTERVENTION:"))
            if not done:
                all_done = False
        if all_done or time.time() >= deadline:
            break
        time.sleep(args.poll_seconds)

    _render_status_table(selection.repos, status_files, log_files, session, run_dir, args, deadline)
    failed = []
    still_running = []
    for repo in selection.repos:
        status = status_files[repo].read_text(encoding="utf-8").strip()
        if status.startswith(("SERVER_EXITED:0:", "STARTED:")) and args.wait_mode == "launch":
            continue
        if status.startswith("UPDATED:"):
            continue
        if status.startswith("SERVER_EXITED:0:"):
            continue
        if status.startswith(("FAILED:", "NEEDS_INTERVENTION:", "SERVER_EXITED:")):
            failed.append(f"{repo.name} status={status} log={log_files[repo]}")
        else:
            still_running.append(f"{repo.name} status={status} log={log_files[repo]}")

    print("\nSummary")
    if failed:
        print("Failed:")
        for item in failed:
            print(f"  {item}")
    if still_running:
        print("Still running:")
        for item in still_running:
            print(f"  {item}")
    if subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0:
        print(f"\nAttach to tmux:\n  tmux attach -t {session}")
    print(f"Logs:\n  {run_dir}")
    return 1 if failed or still_running else 0
