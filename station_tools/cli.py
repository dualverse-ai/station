"""Entrypoint for the multi-station CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import CONFIG_FILENAME, load_config
from .commands import archive, init, list as list_command, restore
from .commands import monitor, refresh_api, resume, update


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="station",
        description="Manage and monitor multiple Station checkouts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Config path (default: ~/.config/station-tools/{CONFIG_FILENAME}; absent config uses ~/station and ~/station_*)",
    )
    subparsers = parser.add_subparsers(dest="command")
    list_command.add_parser(subparsers)
    monitor.add_parser(subparsers)
    update.add_parser(subparsers)
    resume.add_parser(subparsers)
    refresh_api.add_parser(subparsers)
    archive.add_parser(subparsers)
    restore.add_parser(subparsers)
    init.add_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv if argv is not None else sys.argv[1:])
    if raw_args and raw_args[0] == "help":
        raw_args = ["--help"]
    if len(raw_args) >= 2 and raw_args[1] == "help":
        raw_args = [raw_args[0], "--help", *raw_args[2:]]

    parser = build_parser()
    args = parser.parse_args(raw_args)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    config = load_config(args.config)
    try:
        return int(args.func(args, config))
    except KeyboardInterrupt:
        print()
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
