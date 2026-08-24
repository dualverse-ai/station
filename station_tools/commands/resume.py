from __future__ import annotations

from argparse import ArgumentParser, Namespace

from station_tools.config import ToolsConfig
from station_tools.frontend_api import find_endpoint, request_json
from station_tools.repo import read_station_metadata, station_service_running
from station_tools.selectors import select_repos, targets_or_current


def add_parser(subparsers) -> None:
    parser: ArgumentParser = subparsers.add_parser("resume", help="Resume paused/waiting stations through the dashboard API")
    parser.add_argument("targets", nargs="*", help="Station ids, suffixes, names, or paths")
    parser.add_argument("--status-only", action="store_true", help="Report resumable stations without sending resume")
    parser.add_argument("--timeout", type=float, default=10.0, help="API timeout in seconds")
    parser.set_defaults(func=run)


def _message(response: dict) -> str:
    return str(response.get("message") or response.get("error") or response)


def run(args: Namespace, config: ToolsConfig) -> int:
    selection = select_repos(targets_or_current(args.targets), config.station_patterns)
    if not selection.repos:
        print("no valid station repos selected")
        return 1

    resumed: list[str] = []
    would_resume: list[str] = []
    reasons: list[str] = []
    already_active: list[str] = []
    not_running: list[str] = []
    unreachable: list[str] = []
    failed: list[str] = []

    for repo in selection.repos:
        meta = read_station_metadata(repo)
        label = f"{meta.station_name} ({repo.name})"
        found = find_endpoint(repo, "/api/orchestrator/status", timeout=args.timeout)
        if not found:
            if station_service_running(repo):
                unreachable.append(f"{label} API not reachable on local ports")
            else:
                not_running.append(f"{label} station service not running")
            continue

        endpoint, response = found
        status = response.get("status") if isinstance(response.get("status"), dict) else {}
        is_running = status.get("is_running") is True
        is_paused = status.get("is_paused") is True
        is_waiting = status.get("is_waiting") is True
        current_tick = status.get("current_tick", "")
        pause_reason = str(status.get("pause_reason") or "").strip()
        multistart = status.get("multistart") if isinstance(status.get("multistart"), dict) else None

        if multistart:
            stage = str(multistart.get("stage") or multistart.get("status") or "multistart")
            if args.status_only:
                would_resume.append(f"{label} multistart stage={stage}")
                continue
            try:
                resume_response = request_json(endpoint, "/api/multistart/resume", method="POST", timeout=args.timeout)
            except Exception as exc:
                failed.append(f"{label} multistart resume request failed: {exc}")
                continue
            resumed.append(f"{label} multistart stage={stage} {_message(resume_response)}")
            continue

        if not is_running:
            not_running.append(f"{label} tick={current_tick or '?'}")
            continue
        if not is_paused and not is_waiting:
            already_active.append(f"{label} running but not paused/waiting tick={current_tick or '?'}")
            continue
        if args.status_only:
            would_resume.append(f"{label} tick={current_tick or '?'} paused={str(is_paused).lower()} waiting={str(is_waiting).lower()}")
            if pause_reason:
                reasons.append(f"{label} current reason: {pause_reason}")
            continue
        try:
            resume_response = request_json(endpoint, "/api/orchestrator/resume", method="POST", timeout=args.timeout)
        except Exception as exc:
            failed.append(f"{label} resume request failed: {exc}")
            continue
        resumed.append(f"{label} tick={current_tick or '?'} {_message(resume_response)}")
        if pause_reason:
            reasons.append(f"{label} previous reason: {pause_reason}")

    print("station resume summary")
    for title, items in [
        ("Would resume", would_resume),
        ("Resumed", resumed),
        ("Previous pause/wait reasons", reasons),
        ("Already running, not paused or waiting", already_active),
        ("Not running, cannot be resumed", not_running),
        ("API unreachable", unreachable),
        ("Skipped invalid paths", list(selection.skipped)),
        ("Failed", failed),
    ]:
        if items:
            print(f"{title}:")
            for item in items:
                print(f"  {item}")
    return 1 if unreachable or failed else 0
