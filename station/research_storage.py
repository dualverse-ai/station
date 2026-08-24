from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from station import file_io_utils


BASE_PATH_ENV = "RESEARCH_STORAGE_BASE_PATH"
ALLOCATION_MARKER_SUFFIX = ".station_research_storage.yaml"
JOB_MANIFEST_FILENAME = "research_storage_allocations.yaml"
RESEARCH_STORAGE_RELATIVE_PATH = Path("rooms", "research", "storage")


def configured_base_path(
    config_value: object = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    values = os.environ if env is None else env
    raw_value = values.get(BASE_PATH_ENV) if BASE_PATH_ENV in values else config_value
    raw = str(raw_value or "").strip()
    if not raw:
        return None
    base = Path(raw).expanduser()
    if not base.is_absolute():
        raise ValueError(f"{BASE_PATH_ENV} must be an absolute path: {raw}")
    return base.resolve(strict=False)


def research_storage_path(data_root: Path) -> Path:
    return Path(data_root) / RESEARCH_STORAGE_RELATIVE_PATH


def resolved_research_storage_path(data_root: Path) -> Path | None:
    storage = research_storage_path(data_root)
    if not storage.exists():
        return None
    resolved = storage.resolve()
    return resolved if resolved.is_dir() else None


def new_allocation_path(base_path: Path) -> Path:
    return Path(base_path) / str(uuid.uuid4())


def allocation_marker_path(allocation_path: Path) -> Path:
    allocation = Path(allocation_path)
    return allocation.parent / f".{allocation.name}{ALLOCATION_MARKER_SUFFIX}"


def load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def save_mapping(path: Path, value: dict[str, Any]) -> None:
    file_io_utils.save_yaml(value, str(path))


def read_allocation_marker(allocation_path: Path) -> dict[str, Any]:
    return load_mapping(allocation_marker_path(allocation_path))


def write_allocation_marker(allocation_path: Path, value: dict[str, Any]) -> None:
    payload = dict(value)
    payload.setdefault("format_version", 1)
    payload.setdefault("storage_id", Path(allocation_path).name)
    save_mapping(allocation_marker_path(allocation_path), payload)


def remove_allocation_marker(allocation_path: Path) -> None:
    allocation_marker_path(allocation_path).unlink(missing_ok=True)


def path_is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve(strict=False).relative_to(Path(root).resolve(strict=False))
        return True
    except ValueError:
        return False


def remove_tree_allow_read_only(
    target: Path,
    *,
    sudo_fallback: bool = False,
    cwd: Path | None = None,
) -> None:
    """Remove a tree after making read-only directories owner-accessible.

    Directory symlinks are unlinked rather than traversed. Making directories
    writable top-down is required before ``shutil.rmtree`` can unlink files
    from Research ``storage/system`` trees on NFS.
    """
    target = Path(target).expanduser()
    try:
        target_mode = target.lstat().st_mode
    except FileNotFoundError:
        return
    try:
        if not stat.S_ISDIR(target_mode):
            target.unlink()
            return

        pending = [target]
        while pending:
            directory = pending.pop()
            try:
                mode = directory.lstat().st_mode
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(mode):
                continue
            # Some NFS servers reject fchmodat(..., AT_SYMLINK_NOFOLLOW) even
            # for real directories. lstat above excluded directory symlinks.
            os.chmod(directory, mode | stat.S_IRWXU)
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))

        shutil.rmtree(target)
    except OSError as exc:
        if not sudo_fallback:
            raise
        result = subprocess.run(
            ["sudo", "rm", "-rf", "--", str(target)],
            cwd=cwd,
            check=False,
        )
        if result.returncode != 0 or os.path.lexists(target):
            raise OSError(
                f"sudo cleanup failed for {target} (exit {result.returncode})"
            ) from exc


def marker_matches(
    allocation_path: Path,
    *,
    station_id: object | None = None,
    job_id: object | None = None,
    seed: int | None = None,
    kinds: set[str] | None = None,
) -> bool:
    marker = read_allocation_marker(allocation_path)
    if not marker:
        return False
    if str(marker.get("storage_id") or "") != Path(allocation_path).name:
        return False
    if station_id is not None and str(marker.get("station_id") or "") != str(station_id):
        return False
    if job_id is not None and str(marker.get("job_id") or "") != str(job_id):
        return False
    if seed is not None:
        try:
            if int(marker.get("seed")) != int(seed):
                return False
        except (TypeError, ValueError):
            return False
    if kinds is not None and str(marker.get("kind") or "") not in kinds:
        return False
    return True


def is_multistart_research_storage_path(path: Path) -> bool:
    path = Path(path)
    if tuple(path.parts[-3:]) != tuple(RESEARCH_STORAGE_RELATIVE_PATH.parts):
        return False
    if len(path.parents) < 3:
        return False
    data_root = path.parents[2]
    if data_root.name != "origin_station_data" and not re.fullmatch(
        r"station_data_s\d+", data_root.name
    ):
        return False
    return (data_root.parent / "state.yaml").is_file()


def should_follow_research_storage_symlink(
    path: Path,
    live_storage_root: Path | None = None,
) -> bool:
    path = Path(path)
    if live_storage_root is not None:
        try:
            if path.absolute() == Path(live_storage_root).absolute():
                return True
        except OSError:
            pass
    return is_multistart_research_storage_path(path)


def remove_job_allocations(
    job_path: Path,
    *,
    preserve_selected: bool,
    include_origin: bool,
    remove_tree: Callable[[Path], None] | None = None,
    sudo_fallback: bool = False,
    cwd: Path | None = None,
    station_id: object | None = None,
    job_id: object | None = None,
) -> dict[str, Any]:
    """Remove only marked allocations owned by one multistart job."""
    job_path = Path(job_path)
    manifest = load_mapping(job_path / JOB_MANIFEST_FILENAME)
    if not manifest:
        return {"success": True, "configured": False, "removed": []}
    base_raw = str(manifest.get("base_path") or "").strip()
    if not base_raw:
        return {"success": False, "reason": "Research storage allocation manifest is incomplete"}
    storage_base = Path(base_raw)
    station_id = str(station_id or manifest.get("station_id") or "station")
    job_id = str(job_id or manifest.get("job_id") or job_path.name)
    selected_seed = _safe_int(manifest.get("selected_seed")) if preserve_selected else None
    seeds = manifest.get("seeds") if isinstance(manifest.get("seeds"), dict) else {}

    candidates: list[tuple[Path, int | None, bool]] = []
    for seed_raw, seed_info in seeds.items():
        if not isinstance(seed_info, dict):
            continue
        seed = _safe_int(seed_raw)
        target_raw = str(seed_info.get("target") or "").strip()
        if seed is not None and seed != selected_seed and target_raw:
            candidates.append((Path(target_raw), seed, False))
    origin = manifest.get("origin") if isinstance(manifest.get("origin"), dict) else {}
    origin_raw = str(origin.get("target") or "").strip()
    selected_info = seeds.get(str(selected_seed)) if selected_seed is not None else None
    selected_raw = str(selected_info.get("target") or "").strip() if isinstance(selected_info, dict) else ""
    if include_origin and origin.get("owned") and origin_raw and origin_raw != selected_raw:
        candidates.append((Path(origin_raw), None, True))

    removed: list[str] = []
    missing: list[str] = []
    failures: list[str] = []
    seen: set[Path] = set()
    for target, seed, is_origin in candidates:
        resolved = target.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path_is_within(target, storage_base):
            failures.append(f"unsafe allocation path: {target}")
            continue
        marker = read_allocation_marker(target)
        marker_valid = (
            str(marker.get("station_id") or "") == station_id
            and str(marker.get("storage_id") or "") == target.name
        )
        if is_origin:
            marker_valid = marker_valid and str(marker.get("kind") or "") == "live"
        else:
            seed_marker = (
                str(marker.get("kind") or "") == "multistart_seed"
                and str(marker.get("job_id") or "") == job_id
                and str(marker.get("seed") or "") == str(seed)
            )
            promoted_marker = (
                str(marker.get("kind") or "") == "live"
                and str(marker.get("promoted_from_job_id") or "") == job_id
                and str(marker.get("promoted_from_seed") or "") == str(seed)
            )
            marker_valid = marker_valid and (seed_marker or promoted_marker)
        if not target.exists() and not allocation_marker_path(target).exists():
            missing.append(str(target))
            continue
        if not marker_valid:
            failures.append(f"allocation marker mismatch: {target}")
            continue
        try:
            if target.exists() or target.is_symlink():
                if remove_tree is None:
                    remove_tree_allow_read_only(
                        target,
                        sudo_fallback=sudo_fallback,
                        cwd=cwd,
                    )
                else:
                    remove_tree(target)
            remove_allocation_marker(target)
            removed.append(str(target))
        except OSError as exc:
            failures.append(f"{target}: {exc}")
    return {
        "success": not failures,
        "configured": True,
        "removed": removed,
        "already_missing": missing,
        "failures": failures,
    }


def relocate_storage_symlink(
    storage_path: Path,
    storage_base: Path,
    *,
    marker_payload: Mapping[str, Any],
    remove_tree: Callable[[Path], None],
) -> Path:
    """Move an existing linked live allocation to a newly configured base."""
    storage_path = Path(storage_path)
    storage_base = Path(storage_base)
    if not storage_path.is_symlink():
        raise ValueError(f"Research storage is not a symlink: {storage_path}")
    source = storage_path.resolve()
    if not source.is_dir():
        raise RuntimeError(f"Research storage link target is missing: {storage_path} -> {source}")
    if path_is_within(source, storage_base):
        return source

    storage_base.mkdir(parents=True, exist_ok=True)
    target = new_allocation_path(storage_base)
    temporary = storage_base / f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary_link = storage_path.with_name(f".{storage_path.name}.{uuid.uuid4().hex}.tmp")
    switched = False
    try:
        shutil.copytree(source, temporary, symlinks=True)
        os.replace(temporary, target)
        write_allocation_marker(target, dict(marker_payload))
        temporary_link.symlink_to(target, target_is_directory=True)
        os.replace(temporary_link, storage_path)
        switched = True
    finally:
        if temporary.exists():
            remove_tree(temporary)
        temporary_link.unlink(missing_ok=True)
        if not switched:
            if target.exists():
                remove_tree(target)
            remove_allocation_marker(target)

    station_id = marker_payload.get("station_id")
    if marker_matches(source, station_id=station_id, kinds={"live"}):
        try:
            remove_tree(source)
            remove_allocation_marker(source)
        except OSError:
            pass
    return target


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
