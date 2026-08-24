#!/usr/bin/env python3
"""Manually reserve Station research CPU/GPU coordinator slots."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fcntl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from station import constants  # noqa: E402

DEFAULT_POLL_SECONDS = 30.0
MANUAL_STATION_PREFIX = "manual"


class ReservationError(RuntimeError):
    pass


def _now_str(timestamp: Optional[float] = None) -> str:
    return datetime.fromtimestamp(timestamp or time.time()).strftime("%Y-%m-%d %H:%M:%S")


def _parse_int_list(value: Any, *, empty_means: Optional[List[int]] = None) -> List[int]:
    if value is None:
        return list(empty_means or [])
    if isinstance(value, (list, tuple)):
        if not value:
            return list(empty_means or [])
        return [int(item) for item in value]
    if isinstance(value, str):
        if not value.strip():
            return list(empty_means or [])
        items: List[int] = []
        for chunk in [part.strip() for part in value.split(",") if part.strip()]:
            if "-" in chunk:
                start_str, end_str = chunk.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                if end < start:
                    start, end = end, start
                items.extend(range(start, end + 1))
            else:
                items.append(int(chunk))
        deduped: List[int] = []
        seen = set()
        for item in items:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped
    return [int(value)]


def _detect_gpu_ids() -> List[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    gpu_ids: List[int] = []
    for line in result.stdout.splitlines():
        match = re.search(r"\d+", line)
        if match:
            gpu_ids.append(int(match.group(0)))
    return gpu_ids


def default_gpu_pool() -> List[int]:
    return _parse_int_list(constants.RESEARCH_EVAL_AVAILABLE_GPUS, empty_means=_detect_gpu_ids())


def default_cpu_pool() -> List[int]:
    return _parse_int_list(
        constants.RESEARCH_EVAL_AVAILABLE_CPUS,
        empty_means=list(range(os.cpu_count() or 0)),
    )


def _initial_data() -> Dict[str, Any]:
    now = time.time()
    return {
        "allocations": {},
        "last_updated": now,
        "last_updated_str": _now_str(now),
    }


def _ensure_coord_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(json.dumps(_initial_data(), indent=2), encoding="utf-8")


@contextmanager
def _locked_coord_files(paths: Sequence[Path]):
    ordered_paths = sorted({path for path in paths}, key=lambda item: str(item))
    handles = []
    try:
        for path in ordered_paths:
            _ensure_coord_file(path)
            handle = path.open("r+", encoding="utf-8")
            fcntl.flock(handle, fcntl.LOCK_EX)
            handles.append((path, handle))
        yield handles
    finally:
        for _path, handle in reversed(handles):
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()


def _read_locked(handle) -> Dict[str, Any]:
    handle.seek(0)
    content = handle.read()
    if not content.strip():
        return _initial_data()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ReservationError(f"Invalid coordination JSON in {handle.name}: {exc}") from exc
    data.setdefault("allocations", {})
    return data


def _write_locked(handle, data: Dict[str, Any]) -> None:
    now = time.time()
    data["last_updated"] = now
    data["last_updated_str"] = _now_str(now)
    handle.seek(0)
    json.dump(data, handle, indent=2)
    handle.truncate()
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def _allocation_is_expired(info: Dict[str, Any], now: float) -> bool:
    expires_at = info.get("expires_at")
    if expires_at is None:
        return False
    try:
        return now > float(expires_at)
    except (TypeError, ValueError):
        return False


def _drop_expired_allocations(data: Dict[str, Any]) -> int:
    now = time.time()
    allocations = data.get("allocations", {})
    kept = {
        key: info
        for key, info in allocations.items()
        if not _allocation_is_expired(info, now)
    }
    removed = len(allocations) - len(kept)
    if removed:
        data["allocations"] = kept
    return removed


def _used_ids(data: Dict[str, Any], field: str) -> set[int]:
    used: set[int] = set()
    for info in data.get("allocations", {}).values():
        used.update(int(item) for item in info.get(field, []))
    return used


def _available_ids(data: Dict[str, Any], pool: Sequence[int], field: str) -> List[int]:
    used = _used_ids(data, field)
    return [item for item in pool if item not in used]


def _manual_station_id(owner: str) -> str:
    safe_owner = re.sub(r"[^A-Za-z0-9_.-]+", "_", owner.strip()) or "user"
    return f"{MANUAL_STATION_PREFIX}-{safe_owner}"


def _reservation_record(
    *,
    resource_field: str,
    ids: Sequence[int],
    station_id: str,
    reservation_id: str,
    duration_seconds: float,
    note: str,
) -> Dict[str, Any]:
    now = time.time()
    expires_at = now + duration_seconds
    record = {
        resource_field: list(ids),
        "station_id": station_id,
        "eval_id": reservation_id,
        "start_time": now,
        "start_time_str": _now_str(now),
        "timeout_seconds": duration_seconds,
        "expires_at": expires_at,
        "expires_at_str": _now_str(expires_at),
        "manual_reservation": True,
        "note": note,
    }
    return record


def reserve_once(
    *,
    reservation_id: str,
    station_id: str,
    gpu_count: int,
    cpu_count: int,
    gpu_pool: Sequence[int],
    cpu_pool: Sequence[int],
    gpu_file: Path,
    cpu_file: Path,
    duration_seconds: float,
    note: str,
) -> Tuple[bool, Dict[str, Any]]:
    coord_specs: List[Tuple[str, int, Sequence[int], Path, str]] = []
    if gpu_count:
        coord_specs.append(("gpu", gpu_count, gpu_pool, gpu_file, "gpus"))
    if cpu_count:
        coord_specs.append(("cpu", cpu_count, cpu_pool, cpu_file, "cpus"))
    if not coord_specs:
        raise ReservationError("Nothing to reserve. Request at least one GPU or CPU.")

    with _locked_coord_files([spec[3] for spec in coord_specs]) as locked:
        handle_by_path = {path: handle for path, handle in locked}
        data_by_kind: Dict[str, Dict[str, Any]] = {}
        allocated_by_kind: Dict[str, List[int]] = {}
        unavailable: Dict[str, Dict[str, Any]] = {}

        for kind, count, pool, path, field in coord_specs:
            data = _read_locked(handle_by_path[path])
            _drop_expired_allocations(data)
            available = _available_ids(data, pool, field)
            if len(available) < count:
                unavailable[kind] = {
                    "need": count,
                    "available_count": len(available),
                    "available": available,
                }
            else:
                allocated_by_kind[kind] = available[:count]
            data_by_kind[kind] = data

        if unavailable:
            return False, {"unavailable": unavailable}

        allocation_key = f"{station_id}:{reservation_id}"
        for kind, _count, _pool, path, field in coord_specs:
            data = data_by_kind[kind]
            data.setdefault("allocations", {})[allocation_key] = _reservation_record(
                resource_field=field,
                ids=allocated_by_kind[kind],
                station_id=station_id,
                reservation_id=reservation_id,
                duration_seconds=duration_seconds,
                note=note,
            )
            _write_locked(handle_by_path[path], data)

        return True, {
            "id": reservation_id,
            "station_id": station_id,
            "gpus": allocated_by_kind.get("gpu", []),
            "cpus": allocated_by_kind.get("cpu", []),
            "expires_at_str": _now_str(time.time() + duration_seconds),
        }


def reserve_with_wait(args: argparse.Namespace) -> int:
    gpu_count = int(args.gpus)
    cpu_count = int(args.cpus)
    if gpu_count < 0 or cpu_count < 0:
        raise ReservationError("GPU and CPU counts must be non-negative.")

    gpu_pool = _parse_int_list(args.gpu_ids) if args.gpu_ids else default_gpu_pool()
    cpu_pool = _parse_int_list(args.cpu_ids) if args.cpu_ids else default_cpu_pool()
    if gpu_count and len(gpu_pool) < gpu_count:
        raise ReservationError(f"Requested {gpu_count} GPU(s), but only these GPU IDs are configured/detected: {gpu_pool}")
    if cpu_count and len(cpu_pool) < cpu_count:
        raise ReservationError(f"Requested {cpu_count} CPU(s), but only {len(cpu_pool)} CPU IDs are configured.")

    duration_seconds = float(args.seconds)
    reservation_id = args.id or f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    station_id = _manual_station_id(args.owner)
    note = args.note or "manual reservation"

    first_wait = True
    while True:
        ok, result = reserve_once(
            reservation_id=reservation_id,
            station_id=station_id,
            gpu_count=gpu_count,
            cpu_count=cpu_count,
            gpu_pool=gpu_pool,
            cpu_pool=cpu_pool,
            gpu_file=Path(args.gpu_file),
            cpu_file=Path(args.cpu_file),
            duration_seconds=duration_seconds,
            note=note,
        )
        if ok:
            print(f"Reserved id={result['id']} station_id={result['station_id']}")
            if gpu_count:
                print(f"  GPUs: {result['gpus']}")
            if cpu_count:
                print(f"  CPUs: {result['cpus']}")
            print(f"  Expires: {result['expires_at_str']}")
            print("Release command:")
            print(f"  {build_release_command(args, result['id'])}")
            return 0

        if args.no_wait:
            print(_format_unavailable(result["unavailable"]), file=sys.stderr)
            return 1
        if first_wait:
            print(_format_unavailable(result["unavailable"]))
            first_wait = False
        time.sleep(float(args.poll))


def _format_unavailable(unavailable: Dict[str, Dict[str, Any]]) -> str:
    pieces = []
    for kind, info in sorted(unavailable.items()):
        pieces.append(
            f"{kind}: need {info['need']}, available {info['available_count']} ({info['available']})"
        )
    return "Waiting for resources: " + "; ".join(pieces)


def build_release_command(args: argparse.Namespace, reservation_id: str) -> str:
    parts = [
        "scripts/reserve_resources.py",
        "release",
        "--id",
        reservation_id,
    ]
    if str(args.owner) != getpass.getuser():
        parts.extend(["--owner", str(args.owner)])
    if str(args.gpu_file) != str(constants.RESEARCH_EVAL_GPU_COORD_FILE):
        parts.extend(["--gpu-file", str(args.gpu_file)])
    if str(args.cpu_file) != str(constants.RESEARCH_EVAL_CPU_COORD_FILE):
        parts.extend(["--cpu-file", str(args.cpu_file)])
    return " ".join(shlex.quote(part) for part in parts)


def _iter_manual_allocations(data: Dict[str, Any], station_id: Optional[str]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for key, info in data.get("allocations", {}).items():
        info_station_id = str(info.get("station_id", ""))
        if station_id:
            if info_station_id != station_id:
                continue
        elif not info_station_id.startswith(f"{MANUAL_STATION_PREFIX}-"):
            continue
        yield key, info


def release(args: argparse.Namespace) -> int:
    station_id = _manual_station_id(args.owner)
    if not args.id and not args.all:
        raise ReservationError("release requires --id ID or --all.")

    removed: List[str] = []
    paths = [Path(args.gpu_file), Path(args.cpu_file)]
    with _locked_coord_files(paths) as locked:
        for path, handle in locked:
            data = _read_locked(handle)
            allocations = data.get("allocations", {})
            kept = {}
            changed = False
            for key, info in allocations.items():
                matches_owner = info.get("station_id") == station_id
                matches_id = args.all or str(info.get("eval_id")) == str(args.id)
                if matches_owner and matches_id:
                    removed.append(f"{path}:{key}")
                    changed = True
                else:
                    kept[key] = info
            if changed:
                data["allocations"] = kept
                _write_locked(handle, data)

    if removed:
        print("Released:")
        for item in removed:
            print(f"  {item}")
        return 0
    print("No matching manual reservation found.")
    return 1


def list_allocations(args: argparse.Namespace) -> int:
    station_id = _manual_station_id(args.owner) if not args.all else None
    paths = [(Path(args.gpu_file), "gpus"), (Path(args.cpu_file), "cpus")]
    found = False
    with _locked_coord_files([path for path, _field in paths]) as locked:
        handle_by_path = {path: handle for path, handle in locked}
        for path, field in paths:
            data = _read_locked(handle_by_path[path])
            print(path)
            for key, info in _iter_manual_allocations(data, station_id):
                found = True
                print(
                    f"  {key}: {field}={info.get(field, [])} "
                    f"expires={info.get('expires_at_str', 'unknown')} "
                    f"note={info.get('note', '')}"
                )
    if not found:
        print("No manual reservations found.")
    return 0


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reserve or release Station research CPU/GPU slots in the /tmp coordination files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  scripts/reserve_resources.py\n"
            "  scripts/reserve_resources.py reserve --gpus 2 --cpus 20 --hours 12\n"
            "  scripts/reserve_resources.py reserve --gpus 0 --cpus 10 --id cpu-work\n"
            "  scripts/reserve_resources.py release --id cpu-work\n"
            "  scripts/reserve_resources.py list\n"
        ),
    )
    parser.add_argument("action", nargs="?", choices=["reserve", "release", "list", "help"], default="reserve")
    parser.add_argument("--gpus", type=positive_int, default=1, help="GPU count to reserve; default: 1")
    parser.add_argument("--cpus", type=positive_int, default=0, help="CPU count to reserve; default: 0")
    parser.add_argument("--days", type=positive_float, default=7.0, help="reservation duration in days; default: 7")
    parser.add_argument("--hours", type=positive_float, help="reservation duration in hours")
    parser.add_argument("--seconds", type=positive_float, help=argparse.SUPPRESS)
    parser.add_argument("--id", help="reservation id to create or release")
    parser.add_argument("--owner", default=getpass.getuser(), help="manual owner namespace; default: current user")
    parser.add_argument("--note", help="human-readable note stored in the coordination file")
    parser.add_argument("--no-wait", action="store_true", help="fail instead of waiting when resources are unavailable")
    parser.add_argument("--poll", type=positive_float, default=DEFAULT_POLL_SECONDS, help="wait poll seconds; default: 30")
    parser.add_argument("--gpu-ids", help="GPU pool override, e.g. 0,1,2 or 0-3")
    parser.add_argument("--cpu-ids", help="CPU pool override, e.g. 0-15,32")
    parser.add_argument("--gpu-file", default=constants.RESEARCH_EVAL_GPU_COORD_FILE)
    parser.add_argument("--cpu-file", default=constants.RESEARCH_EVAL_CPU_COORD_FILE)
    parser.add_argument("--all", action="store_true", help="release/list all reservations in this owner namespace")
    return parser


def normalize_duration(args: argparse.Namespace) -> None:
    if args.seconds is not None:
        return
    if args.hours is not None:
        args.seconds = args.hours * 60 * 60
        return
    args.seconds = args.days * 24 * 60 * 60


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    normalize_duration(args)
    try:
        if args.action == "reserve":
            return reserve_with_wait(args)
        if args.action == "release":
            return release(args)
        if args.action == "list":
            return list_allocations(args)
        if args.action == "help":
            parser.print_help()
            return 0
        parser.error(f"unknown action {args.action!r}")
        return 2
    except ReservationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
