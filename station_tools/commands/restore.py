from __future__ import annotations

import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path

from station_tools.config import ToolsConfig


def add_parser(subparsers) -> None:
    parser: ArgumentParser = subparsers.add_parser(
        "restore",
        help="Restore an archived station by UUID or zip filename",
        description=(
            "Run scripts/restore.sh in a target Station checkout. The source may "
            "be a station UUID/prefix or a complete station archive zip path."
        ),
    )
    parser.add_argument("source", help="Station UUID/prefix or archive zip filename/path")
    parser.add_argument("tick", nargs="?", help="Optional ordinary backup tick")
    parser.add_argument("-o", "--output", help="Forward an explicit restore output directory")
    parser.set_defaults(func=run)


def _resolve_repo() -> Path:
    current = Path.cwd()
    if (current / "scripts" / "restore.sh").is_file():
        return current.resolve()
    return Path(__file__).resolve().parents[2]


def run(args: Namespace, _config: ToolsConfig) -> int:
    repo = _resolve_repo()
    script = repo / "scripts" / "restore.sh"
    if not script.is_file():
        raise ValueError(f"target is not a Station checkout with scripts/restore.sh: {repo}")

    source = str(args.source)
    source_path = Path(source).expanduser()
    if source_path.is_file():
        source = str(source_path.resolve())

    command = ["bash", str(script)]
    if getattr(args, "output", None):
        command.extend(["--output", str(args.output)])
    command.append(source)
    if getattr(args, "tick", None) is not None:
        command.append(str(args.tick))
    return subprocess.run(command, cwd=repo, check=False).returncode
