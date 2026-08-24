from __future__ import annotations

import shutil
import subprocess
import shlex
import sys
from argparse import SUPPRESS, ArgumentParser, Namespace, RawDescriptionHelpFormatter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml
from station import constants, file_io_utils, startup_overrides, station_config

from station_tools.config import ToolsConfig
from station_tools.config import build_hook_env
from station_tools.hooks import HookRunner
from station_tools.repo import load_yaml_mapping, station_service_running
from station_tools.selectors import token_to_path
from station_tools.station_templates import (
    DEFAULT_STATION_TEMPLATE_SOURCE,
    STATION_TEMPLATE_SOURCE_KEY,
    resolve_station_template,
)


def add_parser(subparsers) -> None:
    parser: ArgumentParser = subparsers.add_parser(
        "init",
        help="Initialize one station checkout with a research task",
        usage="station init [options] TASK [STATION_NAME]",
        formatter_class=RawDescriptionHelpFormatter,
        description=(
            "Initialize a Station checkout with a grouped Research task.\n\n"
            "Run from a Station root to omit the checkout id. A bare task name scans\n"
            "epoch, alpha_evolve, and misc under both example/ and example_private/.\n"
            "Use group/task to restrict the group or a canonical path to select one\n"
            "exact template. Interactive use asks when multiple tasks match;\n"
            "noninteractive use prints the candidates and exits."
        ),
        epilog=(
            "Examples:\n"
            "  station init book\n"
            "  station init epoch/book \"Book Problem\"\n"
            "  station init example/research_epoch/book \"Book Problem\"\n"
            "  station init --station-id 3 book \"Book Problem\"\n"
            "  station init 3 book \"Book Problem\"  # legacy positional id"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without copying files or starting")
    parser.add_argument("--no-hooks", action="store_true", help="Skip configured local hooks")
    parser.add_argument("--post-copy-cmd", default="", help="Bash command to run after copying station_data and before start")
    startup_mode = parser.add_mutually_exclusive_group()
    startup_mode.add_argument(
        "--test",
        action="store_true",
        help="Forward start.sh --test for quick-test startup overrides",
    )
    startup_mode.add_argument(
        "--no-spawn",
        action="store_true",
        help="Start the dashboard without automatically spawning the template agent roster",
    )
    startup_mode.add_argument(
        "--multistart",
        action="store_true",
        help="Enable the template's init and stagnation multistart settings",
    )
    parser.add_argument(
        "--no-multistart",
        dest="no_multistart",
        action="store_true",
        help=SUPPRESS,
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Initialize station_data without running the init.before_start hook or start.sh",
    )
    parser.add_argument(
        "--station-template",
        default="",
        help="Station template name or source under example*/station/ (default: default)",
    )
    parser.add_argument(
        "--station-id",
        dest="station_selector",
        default="",
        help="Explicit station checkout id, suffix, name, or path (default: current directory)",
    )
    parser.add_argument(
        "init_args",
        nargs="+",
        metavar="TASK [STATION_NAME]",
        help="Task query and optional station display name; legacy 'ID TASK [NAME]' is also accepted",
    )
    parser.set_defaults(func=run)


_TASK_GROUPS = ("epoch", "alpha_evolve", "misc")
_TASK_ROOTS = ("example", "example_private")


@dataclass(frozen=True)
class ResearchTaskCandidate:
    path: Path
    source: str
    group: str
    name: str


@dataclass(frozen=True)
class InitRequest:
    repo: Path
    task_name: str
    station_name: str


def _default_station_name(task_name: str) -> str:
    leaf = PurePosixPath(task_name.replace("\\", "/")).name
    return leaf.replace("_", " ").replace("-", " ").title()


def _parse_init_request(args: Namespace) -> InitRequest:
    values = list(getattr(args, "init_args", ()) or ())
    selector = str(getattr(args, "station_selector", "") or "").strip()

    # Preserve compatibility for callers and older command lines using the
    # former positional ``ID TASK NAME`` shape.
    if not values and hasattr(args, "task_name"):
        legacy_selector = str(getattr(args, "station_id", "") or "").strip()
        values = [str(args.task_name)]
        legacy_name = str(getattr(args, "station_name", "") or "").strip()
        if legacy_name:
            values.append(legacy_name)
        selector = selector or legacy_selector
    elif values and values[0].isdigit() and not selector:
        selector = values.pop(0)

    if not values:
        raise ValueError("a research task name is required")
    if len(values) > 2:
        raise ValueError(
            "too many init arguments; use TASK [STATION_NAME] or legacy ID TASK [STATION_NAME]"
        )

    task_name = values[0].strip()
    if not task_name:
        raise ValueError("a research task name is required")
    station_name = values[1].strip() if len(values) == 2 else _default_station_name(task_name)
    if not station_name:
        raise ValueError("station name cannot be empty")

    repo = token_to_path(selector) if selector else Path.cwd()
    return InitRequest(repo=repo, task_name=task_name, station_name=station_name)


def _iter_task_candidates(repo: Path):
    for root_name in _TASK_ROOTS:
        for group in _TASK_GROUPS:
            group_dir = repo / root_name / f"research_{group}"
            if not group_dir.is_dir():
                continue
            for task_dir in sorted(group_dir.iterdir(), key=lambda path: path.name):
                if not task_dir.is_dir():
                    continue
                yield ResearchTaskCandidate(
                    path=task_dir,
                    source=task_dir.relative_to(repo).as_posix(),
                    group=group,
                    name=task_dir.name,
                )


def _find_task_candidates(repo: Path, task_name: str) -> list[ResearchTaskCandidate]:
    query = task_name.strip().replace("\\", "/")
    path = PurePosixPath(query)
    if not query or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("task name must be a task, group/task, or canonical template path")

    candidates = list(_iter_task_candidates(repo))
    if len(path.parts) == 1:
        exact = [candidate for candidate in candidates if candidate.name == path.parts[0]]
        if exact:
            return exact

        # Compatibility with the old flat Epoch names accepted by station init,
        # for example ``epoch_book`` -> ``epoch/book``.
        for group in _TASK_GROUPS:
            prefix = f"{group}_"
            if path.parts[0].startswith(prefix):
                legacy_name = path.parts[0].removeprefix(prefix)
                return [
                    candidate
                    for candidate in candidates
                    if candidate.group == group and candidate.name == legacy_name
                ]
        return []

    if len(path.parts) == 2:
        group = path.parts[0].removeprefix("research_")
        if group not in _TASK_GROUPS:
            raise ValueError(f"unknown research task group: {path.parts[0]}")
        return [
            candidate
            for candidate in candidates
            if candidate.group == group and candidate.name == path.parts[1]
        ]

    if (
        len(path.parts) == 3
        and path.parts[0] in _TASK_ROOTS
        and path.parts[1].startswith("research_")
        and path.parts[1].removeprefix("research_") in _TASK_GROUPS
    ):
        source = path.as_posix()
        return [candidate for candidate in candidates if candidate.source == source]

    raise ValueError("task name must be a task, group/task, or canonical template path")


def _choose_task_candidate(candidates: list[ResearchTaskCandidate]) -> ResearchTaskCandidate | None:
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    print("Multiple research tasks match:")
    for index, candidate in enumerate(candidates, start=1):
        print(f"  {index}. {candidate.source}")
    if not sys.stdin.isatty():
        print("error: task name is ambiguous; use group/task or a canonical template path")
        return None

    while True:
        try:
            reply = input(f"Choose task [1-{len(candidates)}] (or q to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None
        if reply.lower() == "q":
            return None
        if reply.isdigit() and 1 <= int(reply) <= len(candidates):
            return candidates[int(reply) - 1]
        print("Invalid selection.")


def _station_data_summary(station_data: Path) -> None:
    config = load_yaml_mapping(station_data / "station_config.yaml")
    print(f"  path:         {station_data}")
    if not config:
        print("  status:       station_config.yaml missing or unreadable")
    else:
        print(f"  station name: {config.get('station_name') or '<missing>'}")
        print(f"  station id:   {config.get('station_id') or '<missing>'}")
        print(f"  tick:         {config.get('current_tick', '<missing>')}")
        print(f"  top score:    {config.get('top_score', '<missing>')}")
    try:
        count = len(list(station_data.iterdir()))
    except OSError as exc:
        print(f"  entries:      could not list: {exc}")
    else:
        print(f"  entries:      {count} top-level item(s)")


def _remove_existing_station_data(repo: Path, station_data: Path) -> bool:
    is_running = station_service_running(repo)
    print("Existing station_data found. It must be removed before initialization:")
    _station_data_summary(station_data)
    if is_running:
        print("\nStation service appears to be running.")
        print(f"This will run:\n  cd {repo}\n  ./stop.sh --force\n  sudo rm -rf {station_data}")
    else:
        print(f"\nThis will run: sudo rm -rf {station_data}")
    reply = input("Remove existing station_data and continue? [y/N]: ")
    if reply not in {"y", "Y"}:
        print("Cancelled; existing station_data left untouched.")
        return False
    if is_running and (repo / "stop.sh").exists():
        subprocess.run([str(repo / "stop.sh"), "--force"], cwd=repo, check=False)
    return subprocess.run(["sudo", "rm", "-rf", str(station_data)], check=False).returncode == 0


def _load_yaml_mapping_for_init(path: Path, *, required: bool) -> dict:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"missing required YAML file: {path}")
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return data


def copy_station_template_and_task(
    station_data: Path,
    template_dir: Path,
    task_dir: Path,
) -> None:
    template_config = _load_yaml_mapping_for_init(template_dir / "constant_config.yaml", required=False)
    task_config = _load_yaml_mapping_for_init(task_dir / "constant_config.yaml", required=False)

    shutil.copytree(template_dir, station_data)
    rooms_dir = station_data / "rooms"
    rooms_dir.mkdir(parents=True, exist_ok=True)
    research_dest = rooms_dir / "research"
    if research_dest.exists():
        shutil.rmtree(research_dest)
    shutil.copytree(task_dir / "research", research_dest, symlinks=False)

    merged_config = {**template_config, **task_config}
    file_io_utils.save_yaml(merged_config, str(station_data / "constant_config.yaml"), sort_keys=False)


def save_yaml_mapping(path: Path, data) -> None:
    file_io_utils.save_yaml(data, str(path), sort_keys=False)


def write_initial_station_config(
    station_data: Path,
    station_name: str,
    repo: Path,
    station_template_source: str = DEFAULT_STATION_TEMPLATE_SOURCE,
) -> None:
    config_path = station_data / "station_config.yaml"
    existing = load_yaml_mapping(config_path)
    config, _ = station_config.apply_station_config_defaults(
        existing,
        station_name=station_name,
        git_commit=station_config.current_git_commit(repo),
    )
    config[STATION_TEMPLATE_SOURCE_KEY] = station_template_source
    save_yaml_mapping(config_path, config)


def disable_initial_agent_spawn(station_data: Path) -> None:
    """Remove the template roster so agents can be created manually."""
    file_io_utils.delete_file(str(station_data / constants.INIT_AGENTS_FILENAME))


def configure_multistart_for_init(station_data: Path, *, enabled: bool) -> None:
    """Disable both multistart modes unless the user explicitly opts in."""
    if not enabled:
        startup_overrides.apply_no_multistart(station_data)


def _quote(value: str) -> str:
    return shlex.quote(value)


def _run_start_with_hook(
    repo: Path,
    station_name: str,
    config: ToolsConfig,
    no_hooks: bool,
    *,
    test: bool = False,
    no_multistart: bool = False,
) -> int:
    hook_command = HookRunner(config, disabled=no_hooks).command_for("init", "before_start") or ""
    start_args = ["--name", station_name]
    if test:
        start_args.append("--test")
    if no_multistart:
        start_args.append("--no-multistart")
    start_command = "./start.sh " + " ".join(_quote(arg) for arg in start_args)
    script = [
        "set -Eeuo pipefail",
        "shopt -s expand_aliases || true",
        f"hook_command={_quote(hook_command)}",
        'if [[ -n "$hook_command" ]]; then',
        "  printf 'running init.before_start hook\\n'",
        '  eval "$hook_command"',
        "fi",
        f"exec {start_command}",
    ]
    return subprocess.run(
        ["bash", "-lc", "\n".join(script)],
        cwd=repo,
        env=build_hook_env(config),
        check=False,
    ).returncode


def run_post_copy_command(repo: Path, command: str) -> int:
    if not command or not command.strip():
        return 0
    print("running init.post_copy_cmd")
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=repo,
        check=False,
    ).returncode


def run(args: Namespace, config: ToolsConfig) -> int:
    if getattr(args, "multistart", False) and getattr(args, "no_multistart", False):
        print("error: --multistart and --no-multistart cannot be used together")
        return 2
    try:
        request = _parse_init_request(args)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    repo = request.repo
    if not repo.is_dir():
        print(f"error: station repo does not exist: {repo}")
        return 1
    if not (repo / "start.sh").is_file():
        print(f"error: missing start.sh in {repo}")
        return 1
    try:
        template_dir, template_source = resolve_station_template(repo, getattr(args, "station_template", ""))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    try:
        task_candidates = _find_task_candidates(repo, request.task_name)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    selected_task = _choose_task_candidate(task_candidates)
    if selected_task is None:
        if not task_candidates:
            print(f"error: task not found: {request.task_name}")
            print("searched task groups:")
            for root_name in _TASK_ROOTS:
                for group in _TASK_GROUPS:
                    print(f"  {repo / root_name / ('research_' + group)}")
        elif sys.stdin.isatty():
            print("Cancelled; no task selected.")
        return 1
    task_dir = selected_task.path
    if not (task_dir / "research").is_dir():
        print(f"error: task is missing research directory: {task_dir / 'research'}")
        return 1
    try:
        _load_yaml_mapping_for_init(template_dir / "constant_config.yaml", required=False)
        _load_yaml_mapping_for_init(task_dir / "constant_config.yaml", required=False)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    station_data = repo / "station_data"
    print("Initializing station:")
    print(f"  repo:         {repo}")
    print(f"  template:     {template_source}")
    print(f"  task:         {task_dir}")
    print(f"  station_data: {station_data}")
    print(f"  station name: {request.station_name}")
    if getattr(args, "no_spawn", False):
        print("  init agents:  manual (--no-spawn)")
    print(
        "  multistart:   "
        + ("enabled (--multistart)" if getattr(args, "multistart", False) else "disabled (default)")
    )

    if args.dry_run:
        if station_data.exists():
            print("\nExisting station_data summary:")
            _station_data_summary(station_data)
            print(f"  service:      {'running' if station_service_running(repo) else 'not running'}")
        print("Dry run only; no files copied and start.sh not run.")
        return 0

    if station_data.exists() and not _remove_existing_station_data(repo, station_data):
        return 1

    copy_station_template_and_task(station_data, template_dir, task_dir)
    write_initial_station_config(station_data, request.station_name, repo, template_source)
    if run_post_copy_command(repo, getattr(args, "post_copy_cmd", "")) != 0:
        return 1
    post_copy_callback = getattr(args, "post_copy_callback", None)
    if callable(post_copy_callback):
        post_copy_callback(repo, station_data)
    multistart_enabled = getattr(args, "multistart", False)
    configure_multistart_for_init(station_data, enabled=multistart_enabled)
    if not multistart_enabled:
        print("Initialization and stagnation multistart disabled (default).")
    if getattr(args, "no_spawn", False):
        disable_initial_agent_spawn(station_data)
        print("Automatic init-agent spawning disabled; create and launch agents from the dashboard.")
    if getattr(args, "no_start", False):
        print("Station initialized; skipped init.before_start hook and start.sh (--no-start).")
        return 0
    return _run_start_with_hook(
        repo,
        request.station_name,
        config,
        args.no_hooks,
        test=getattr(args, "test", False),
        no_multistart=not multistart_enabled,
    )
