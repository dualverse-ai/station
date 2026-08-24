from __future__ import annotations

import importlib.util
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from station_tools.config import ToolsConfig


def add_parser(subparsers) -> None:
    parser: ArgumentParser = subparsers.add_parser("monitor", help="Monitor station status and CPU/GPU allocation")
    parser.add_argument("targets", nargs="*", help="Optional station ids, suffixes, names, or paths")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal between refreshes")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.set_defaults(func=run)


def run(args: Namespace, config: ToolsConfig) -> int:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "monitor_station.py"
    spec = importlib.util.spec_from_file_location("station_tools_monitor_station", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load monitor implementation: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monitor_main = module.main

    argv: list[str] = []
    if args.once:
        argv.append("--once")
    if args.no_clear:
        argv.append("--no-clear")
    if args.no_color:
        argv.append("--no-color")
    argv.extend(["--interval", str(args.interval)])

    if args.targets:
        from station_tools.selectors import select_repos

        selection = select_repos(args.targets, config.station_patterns)
        for repo in selection.repos:
            argv.extend(["--station-glob", str(repo)])
    else:
        for pattern in config.station_patterns:
            argv.extend(["--station-glob", pattern])
    return int(monitor_main(argv))
