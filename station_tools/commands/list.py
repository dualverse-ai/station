from __future__ import annotations

from argparse import ArgumentParser, Namespace
from shutil import get_terminal_size
from typing import Any

from station_tools.config import ToolsConfig
from station_tools.repo import read_station_metadata
from station_tools.selectors import display_path, select_repos, suffix_for_repo


def add_parser(subparsers) -> None:
    parser: ArgumentParser = subparsers.add_parser("list", help="List configured station checkouts")
    parser.add_argument("targets", nargs="*", help="Optional station ids, suffixes, names, or paths")
    parser.set_defaults(func=run)


def _truncate(value: object, width: int) -> str:
    text = str(value or "")
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _format_score(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "-"
    if abs(score) >= 1000:
        return f"{score:.3g}"
    if abs(score) >= 100:
        return f"{score:.2f}"
    if abs(score) >= 10:
        return f"{score:.3f}"
    return f"{score:.6g}"


def run(args: Namespace, config: ToolsConfig) -> int:
    selection = select_repos(args.targets, config.station_patterns)
    if not selection.repos:
        print("no station repos found")
        return 1

    rows = []
    for repo in selection.repos:
        meta = read_station_metadata(repo)
        rows.append(
            {
                "id": suffix_for_repo(repo),
                "station": meta.station_name,
                "tick": "-" if meta.current_tick is None else str(meta.current_tick),
                "score": _format_score(meta.top_score),
                "folder": display_path(repo),
            }
        )

    term_width = get_terminal_size((120, 30)).columns
    id_width = min(8, max(2, max(len(row["id"]) for row in rows), len("ID")))
    tick_width = max(5, max(len(row["tick"]) for row in rows), len("TICK"))
    score_width = max(9, max(len(row["score"]) for row in rows), len("TOP SCORE"))
    folder_width = min(22, max(len("FOLDER"), max(len(row["folder"]) for row in rows)))
    fixed_width = id_width + tick_width + score_width + folder_width + 8
    station_width = max(18, min(48, term_width - fixed_width))

    print("Station list")
    print(
        f"{'ID':<{id_width}}  "
        f"{'STATION':<{station_width}} "
        f"{'TICK':>{tick_width}}  "
        f"{'TOP SCORE':>{score_width}}  "
        f"{'FOLDER':<{folder_width}}"
    )
    for row in rows:
        print(
            f"{_truncate(row['id'], id_width):<{id_width}}  "
            f"{_truncate(row['station'], station_width):<{station_width}} "
            f"{row['tick']:>{tick_width}}  "
            f"{row['score']:>{score_width}}  "
            f"{_truncate(row['folder'], folder_width):<{folder_width}}"
        )

    if selection.skipped:
        print("\nSkipped invalid paths:")
        for path in selection.skipped:
            print(f"  {path}")
    return 0
