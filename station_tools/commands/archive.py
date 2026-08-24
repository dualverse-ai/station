from __future__ import annotations

import re
import shutil
import subprocess
import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

from station import research_storage
from station_tools.config import ToolsConfig
from station_tools.multistart_backup import (
    active_multistart_info,
    create_active_multistart_archive,
    remove_active_multistart_research_storage_allocations,
)
from station_tools.repo import read_station_metadata
from station_tools.selectors import select_repos, targets_or_current


@dataclass(frozen=True)
class ArchiveTarget:
    repo: Path
    station_id: str
    station_name: str
    tick: int
    score: str
    zip_path: Path
    multistart_info: dict | None = None


def add_parser(subparsers) -> None:
    parser: ArgumentParser = subparsers.add_parser(
        "archive",
        help="Stop selected stations and zip their backup directory",
        description=(
            "Stop selected stations and zip backup/<station_id>. Active multistart jobs "
            "are stored as deduplicated content-addressed archives; ordinary stations "
            "remove live station_data. The unzipped backup directory is removed unless "
            "--keep-backup-dir is used."
        ),
    )
    parser.add_argument("targets", nargs="*", help="Station ids, suffixes, names, or paths")
    parser.add_argument("-y", "--yes", action="store_true", default=os.environ.get("YES", "0") == "1", help="Skip confirmation prompt")
    parser.add_argument(
        "--keep-backup-dir",
        action="store_true",
        default=os.environ.get("KEEP_BACKUP_DIR", "0") == "1",
        help="Keep backup/<station_id> after creating the zip",
    )
    parser.set_defaults(func=run)


def _sanitize_name(value: str) -> str:
    name = re.sub(r"^\[[^]]+\]", "", value)
    name = name.replace(" ", "_").replace("/", "_").replace(":", "-")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name.strip("_")


def _score_slug(value: object) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return f"{score:.6f}".replace(".", "-")


def _latest_snapshot_tick(snapshots_dir: Path) -> int | None:
    latest = 0
    for path in snapshots_dir.glob("tick_*.json"):
        raw_tick = path.stem.removeprefix("tick_")
        if raw_tick.isdigit():
            latest = max(latest, int(raw_tick))
    return latest or None


def _stop_station(repo: Path) -> bool:
    for script in ("stop.sh", "stop"):
        path = repo / script
        if path.exists() and path.stat().st_mode & 0o111:
            return subprocess.run([str(path), "--force"], cwd=repo, check=False).returncode == 0
    return False


def _sudo_rm_rf(path: Path) -> bool:
    if not path.exists():
        return True
    return subprocess.run(["sudo", "rm", "-rf", str(path)], check=False).returncode == 0


def _managed_research_storage_allocation(repo: Path, station_id: str) -> Path | None:
    """Return the station-owned remote Research storage target, when configured."""
    data_root = repo / "station_data"
    storage_path = data_root / research_storage.RESEARCH_STORAGE_RELATIVE_PATH
    if not storage_path.is_symlink():
        return None
    try:
        allocation = storage_path.resolve(strict=False)
    except OSError as exc:
        raise RuntimeError(f"could not resolve Research storage link {storage_path}: {exc}") from exc
    if research_storage.path_is_within(allocation, data_root):
        return None
    if not research_storage.marker_matches(
        allocation,
        station_id=station_id,
        kinds={"live"},
    ):
        raise RuntimeError(
            f"refusing to remove unowned Research storage target {allocation}: "
            "allocation marker does not match this station"
        )
    return allocation


def _remove_managed_research_storage_allocation(
    allocation: Path,
    station_id: str,
    repo: Path,
) -> tuple[bool, str]:
    marker_path = research_storage.allocation_marker_path(allocation)
    if not allocation.exists() and not marker_path.exists():
        return True, "already missing"
    if not research_storage.marker_matches(
        allocation,
        station_id=station_id,
        kinds={"live"},
    ):
        return False, f"allocation marker no longer matches this station: {allocation}"
    try:
        research_storage.remove_tree_allow_read_only(
            allocation,
            sudo_fallback=True,
            cwd=repo,
        )
        research_storage.remove_allocation_marker(allocation)
    except OSError as exc:
        return False, f"{allocation}: {exc}"
    return True, str(allocation)


def run(args: Namespace, config: ToolsConfig) -> int:
    if shutil.which("zip") is None:
        print("error: missing required command: zip")
        return 1
    if shutil.which("sudo") is None:
        print("error: missing required command: sudo")
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

    rows: list[ArchiveTarget] = []
    invalid: list[str] = []
    for repo in selection.repos:
        multistart_info = active_multistart_info(repo)
        if multistart_info is not None:
            station_id = str(multistart_info["station_id"])
            station_name = str(multistart_info.get("station_name") or repo.name)
            tick = int(multistart_info.get("station_tick") or 0)
            safe_name = _sanitize_name(station_name) or station_id
            score = _score_slug(multistart_info.get("top_score"))
            zip_path = repo / "backup" / f"{safe_name}_tick_{tick}_score_{score}_ms.zip"
            if zip_path.exists():
                invalid.append(f"{repo} zip already exists: {zip_path}")
                continue
            rows.append(
                ArchiveTarget(
                    repo=repo,
                    station_id=station_id,
                    station_name=station_name,
                    tick=tick,
                    score=f"{score}_ms",
                    zip_path=zip_path,
                    multistart_info=multistart_info,
                )
            )
            continue

        meta = read_station_metadata(repo)
        if not meta.station_id:
            invalid.append(f"{repo} missing station_id")
            continue
        backup_dir = repo / "backup" / meta.station_id
        snapshots_dir = backup_dir / "snapshots"
        if not backup_dir.is_dir():
            invalid.append(f"{repo} missing backup dir {backup_dir}")
            continue
        if not snapshots_dir.is_dir():
            invalid.append(f"{repo} missing snapshots dir {snapshots_dir}")
            continue
        tick = _latest_snapshot_tick(snapshots_dir)
        if tick is None:
            invalid.append(f"{repo} has no tick_*.json snapshots in {snapshots_dir}")
            continue
        safe_name = _sanitize_name(meta.station_name) or meta.station_id
        score = _score_slug(meta.top_score)
        zip_path = repo / "backup" / f"{safe_name}_tick_{tick}_score_{score}.zip"
        if zip_path.exists():
            invalid.append(f"{repo} zip already exists: {zip_path}")
            continue
        rows.append(
            ArchiveTarget(
                repo=repo,
                station_id=meta.station_id,
                station_name=meta.station_name,
                tick=tick,
                score=score,
                zip_path=zip_path,
            )
        )

    if invalid:
        print("Skipped invalid selections:")
        for item in invalid:
            print(f"  {item}")
    if selection.skipped:
        print("Skipped paths that do not look like station repos:")
        for item in selection.skipped:
            print(f"  {item}")
    if not rows:
        print("no archivable station repos selected")
        return 1

    print("\nThe following station(s) will be stopped, archived, and zipped:")
    print(f"\n{'REPO':<16} {'STATION_ID':<36} {'TICK':<8} {'TYPE/SCORE':<14} ZIP")
    print(f"{'----':<16} {'----------':<36} {'----':<8} {'----------':<14} ---")
    for row in rows:
        print(f"{row.repo.name:<16.16} {row.station_id:<36} {row.tick:<8} {row.score:<14} {row.zip_path}")

    if not args.yes:
        reply = input("\nConfirm? Type Y to proceed: ")
        if reply not in {"Y", "y"}:
            print("Cancelled.")
            return 1

    completed: list[str] = []
    failed: list[str] = []
    for row in rows:
        repo = row.repo
        station_id = row.station_id
        zip_path = row.zip_path
        managed_storage_allocation: Path | None = None
        print(f"\n==> {repo}")
        if not _stop_station(repo):
            failed.append(f"{repo.name} stop script failed or missing")
            continue
        backup_dir = repo / "backup" / station_id
        if row.multistart_info is not None:
            try:
                refreshed_info = active_multistart_info(repo) or row.multistart_info
                manifest_path = create_active_multistart_archive(repo, refreshed_info, backup_dir)
                print(f"Archived active multistart job: {manifest_path}")
            except Exception as exc:
                failed.append(f"{repo.name} multistart archive failed: {exc}")
                continue
        else:
            try:
                managed_storage_allocation = _managed_research_storage_allocation(repo, station_id)
            except RuntimeError as exc:
                failed.append(f"{repo.name} Research storage cleanup preflight failed: {exc}")
                continue
            if not _sudo_rm_rf(repo / "station_data"):
                failed.append(f"{repo.name} failed to remove station_data")
                continue
        if not backup_dir.is_dir():
            failed.append(f"{repo.name} missing backup dir after stop: {backup_dir}")
            continue
        result = subprocess.run(["zip", "-r", zip_path.name, station_id], cwd=repo / "backup", check=False)
        if result.returncode != 0:
            failed.append(f"{repo.name} zip failed: {zip_path}")
            continue
        verify = subprocess.run(["zip", "-T", zip_path.name], cwd=repo / "backup", check=False)
        if verify.returncode != 0:
            failed.append(f"{repo.name} zip verification failed: {zip_path}")
            continue
        if row.multistart_info is not None:
            cleanup = remove_active_multistart_research_storage_allocations(refreshed_info)
            if not cleanup.get("success"):
                failed.append(
                    f"{repo.name} archived Research storage allocation cleanup failed; "
                    f"local multistart metadata retained: {cleanup.get('reason') or cleanup}"
                )
                continue
            if not _sudo_rm_rf(repo / "station_multistart"):
                failed.append(f"{repo.name} failed to remove archived station_multistart")
                continue
            if not _sudo_rm_rf(repo / "station_data"):
                failed.append(f"{repo.name} failed to remove station_data placeholder")
                continue
        elif managed_storage_allocation is not None:
            cleanup_ok, cleanup_detail = _remove_managed_research_storage_allocation(
                managed_storage_allocation,
                station_id,
                repo,
            )
            if not cleanup_ok:
                failed.append(
                    f"{repo.name} archived Research storage allocation cleanup failed: "
                    f"{cleanup_detail}"
                )
                continue
            print(f"Removed managed Research storage allocation: {cleanup_detail}")
        if not args.keep_backup_dir and not _sudo_rm_rf(backup_dir):
            failed.append(f"{repo.name} failed to remove backup/{station_id}")
            continue
        completed.append(f"{repo.name} -> {zip_path}")

    print("\nSummary")
    if completed:
        print("Completed:")
        for item in completed:
            print(f"  {item}")
    if failed:
        print("Failed:")
        for item in failed:
            print(f"  {item}")
        return 1
    return 0
