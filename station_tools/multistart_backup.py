"""Content-addressed archive/restore helpers for active multistart jobs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import yaml

from station import research_storage


ACTIVE_ARCHIVE_DIRNAME = "multistart_archives"
ACTIVE_ARCHIVE_FORMAT_VERSION = 1
_TEXT_SUFFIXES = {
    ".cfg", ".conf", ".csv", ".json", ".jsonl", ".log", ".md", ".py",
    ".sh", ".txt", ".yaml", ".yamll", ".yml",
}
_EPHEMERAL_NAMES = {"controller.pid", "controller.sock"}
_TERMINAL_BRANCH_STATUSES = {"completed", "complete", "cancelled", "canceled"}


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _save_mapping(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def _safe_repo_path(repo: Path, value: Any, root: Path) -> Path | None:
    if not value:
        return None
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = repo / candidate
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _current_job(repo: Path) -> tuple[Path, dict[str, Any]] | None:
    root = repo / "station_multistart"
    for filename in ("current_job.yaml", "current_job"):
        current_path = root / filename
        if not current_path.is_file():
            continue
        current = _load_mapping(current_path)
        job_value = current.get("job_dir")
        job_path = _safe_repo_path(repo, job_value, root) if job_value else None
        if job_path is None:
            branch_tick = current.get("branch_tick")
            job_id = current.get("job_id")
            if branch_tick is not None and job_id:
                job_path = root / f"{int(branch_tick)}_{job_id}"
        if job_path is not None and job_path.is_dir():
            return job_path, current
    return None


def active_multistart_info(repo: Path) -> dict[str, Any] | None:
    """Return validated metadata for the current active multistart job."""
    root = repo / "station_multistart"
    found = _current_job(repo)
    if found is None:
        return None
    job_path, current = found
    state = _load_mapping(job_path / "state.yaml")
    status = str(state.get("status") or current.get("status") or "").strip().lower()
    if status in {"complete", "completed", "cancelled", "canceled"}:
        return None

    station_id = str(state.get("origin_station_id") or current.get("origin_station_id") or "").strip()
    station_name = str(state.get("station_name") or current.get("station_name") or "").strip()
    origin_config = _load_mapping(job_path / "origin_station_data" / "station_config.yaml")
    station_id = station_id or str(origin_config.get("station_id") or "").strip()
    station_name = station_name or str(origin_config.get("station_name") or "").strip()
    if not station_id:
        return None

    ticks: list[int] = []
    scores: list[float] = []
    branch_tick = _coerce_int(state.get("branch_tick") or current.get("branch_tick"))
    if branch_tick is not None:
        ticks.append(branch_tick)
    branches = state.get("branches")
    if isinstance(branches, list):
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            branch_current_tick = _coerce_int(branch.get("current_tick"))
            data_root = _safe_repo_path(repo, branch.get("data_root"), job_path)
            if data_root is None:
                seed = _coerce_int(branch.get("seed"))
                if seed is not None:
                    data_root = job_path / f"station_data_s{seed}"
            branch_config = _load_mapping(data_root / "station_config.yaml") if data_root else {}
            branch_current_tick = _coerce_int(branch_config.get("current_tick")) or branch_current_tick
            try:
                branch_score = float(branch_config.get("top_score"))
            except (TypeError, ValueError):
                branch_score = None
            if branch_score is not None and math.isfinite(branch_score):
                scores.append(branch_score)
            if branch_current_tick is not None:
                ticks.append(branch_current_tick)
    origin_tick = _coerce_int(origin_config.get("current_tick"))
    if origin_tick is not None:
        ticks.append(origin_tick)

    return {
        "repo": str(repo.resolve()),
        "root": str(root.resolve()),
        "job_path": str(job_path.resolve()),
        "job_id": str(state.get("job_id") or current.get("job_id") or job_path.name),
        "job_dir_name": job_path.name,
        "station_id": station_id,
        "station_name": station_name or repo.name,
        "station_tick": max(ticks) if ticks else 0,
        "top_score": max(scores) if scores else None,
    }


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_token(value: Any, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "")).strip("._")
    return token or fallback


def _is_transient_dir(path: Path, root: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    if path.name == "index":
        return True
    if path.name == "sync" and (path.parent / "station_config.yaml").is_file():
        return True
    return relative.endswith("rooms/research/storage/tmp") or relative.endswith("rooms/research/storage/shared/tmp")


def _iter_archive_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        current = Path(dirpath)
        kept_dirnames: list[str] = []
        for name in dirnames:
            path = current / name
            try:
                info = path.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(info.st_mode):
                if research_storage.should_follow_research_storage_symlink(path):
                    try:
                        target_real = path.resolve()
                        current_real = current.resolve()
                    except OSError:
                        target_real = None
                        current_real = None
                    if (
                        target_real is not None
                        and current_real is not None
                        and not research_storage.path_is_within(current_real, target_real)
                    ):
                        kept_dirnames.append(name)
                        continue
                yield path.relative_to(root).as_posix(), path, "symlink", info
                continue
            if not _is_transient_dir(path, root):
                kept_dirnames.append(name)
        dirnames[:] = kept_dirnames
        for filename in filenames:
            path = current / filename
            if filename in _EPHEMERAL_NAMES or path.suffix.lower() == ".pid":
                continue
            relative = path.relative_to(root).as_posix()
            try:
                info = path.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(info.st_mode):
                yield relative, path, "symlink", info
            elif stat.S_ISREG(info.st_mode):
                yield relative, path, "file", info


def _store_object(path: Path, objects_dir: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    file_hash = digest.hexdigest()
    object_path = objects_dir / file_hash[:2] / file_hash[2:]
    if object_path.exists():
        existing_digest = hashlib.sha256()
        with object_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                existing_digest.update(chunk)
        if existing_digest.hexdigest() != file_hash:
            raise RuntimeError(f"existing backup object has a hash collision/corruption: {object_path}")
    else:
        object_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = object_path.with_name(f".{object_path.name}.{uuid.uuid4().hex}.tmp")
        shutil.copy2(path, temporary)
        os.replace(temporary, object_path)
    return file_hash, size


def create_active_multistart_archive(repo: Path, info: dict[str, Any], backup_dir: Path) -> Path:
    root = Path(str(info["root"]))
    if not root.is_dir():
        raise FileNotFoundError(root)
    objects_dir = backup_dir / "objects"
    manifest_dir = backup_dir / ACTIVE_ARCHIVE_DIRNAME
    objects_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, Any]] = []
    symlinks: list[dict[str, Any]] = []
    for relative, path, kind, stat_result in _iter_archive_files(root):
        if kind == "symlink":
            symlinks.append({
                "path": relative,
                "target": os.readlink(path),
                "mode": stat_result.st_mode,
                "mtime": stat_result.st_mtime,
            })
            continue
        file_hash, size = _store_object(path, objects_dir)
        files.append({
            "path": relative,
            "hash": file_hash,
            "size": size,
            "mode": stat_result.st_mode,
            "mtime": stat_result.st_mtime,
        })

    manifest = {
        "format_version": ACTIVE_ARCHIVE_FORMAT_VERSION,
        "archive_type": "active_multistart",
        "archived_at": time.time(),
        "source_repo": str(repo.resolve()),
        "source_root": str(root),
        "restore_root": "station_multistart",
        "job_id": info["job_id"],
        "job_dir_name": info.get("job_dir_name") or Path(str(info["job_path"])).name,
        "station_id": info["station_id"],
        "station_name": info.get("station_name") or repo.name,
        "station_tick": int(info.get("station_tick") or 0),
        "files": files,
        "symlinks": symlinks,
    }
    manifest_base = f"active_{_safe_token(info['job_id'], 'job')}"
    manifest_path = manifest_dir / f"{manifest_base}.json"
    if manifest_path.exists():
        manifest_path = manifest_dir / f"{manifest_base}_{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest_path


def remove_active_multistart_research_storage_allocations(info: dict[str, Any]) -> dict[str, Any]:
    job_path = Path(str(info.get("job_path") or ""))
    return research_storage.remove_job_allocations(
        job_path,
        preserve_selected=False,
        include_origin=True,
        sudo_fallback=True,
        cwd=Path(str(info.get("repo") or ".")),
    )


def _manifest_sort_key(path: Path, manifest: dict[str, Any]) -> tuple[int, float, str]:
    try:
        archived_at = float(manifest.get("archived_at") or 0)
    except (TypeError, ValueError):
        archived_at = 0.0
    return (_coerce_int(manifest.get("station_tick")) or 0, archived_at, path.name)


def latest_active_multistart_manifest(backup_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in (backup_dir / ACTIVE_ARCHIVE_DIRNAME).glob("active_*.json"):
        manifest = _load_json(path)
        if manifest.get("archive_type") == "active_multistart":
            candidates.append((path, manifest))
    if not candidates:
        return None
    return max(candidates, key=lambda item: _manifest_sort_key(item[0], item[1]))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def archive_station_id(zip_path: Path) -> str:
    """Read the station ID represented by a portable Station archive zip."""
    with zipfile.ZipFile(zip_path) as archive:
        top_levels: set[str] = set()
        for member in archive.infolist():
            normalized = member.filename.replace("\\", "/")
            parts = Path(normalized).parts
            if parts:
                top_levels.add(parts[0])
        if len(top_levels) != 1:
            return ""
        top_level = next(iter(top_levels))
        manifest_names = [
            name for name in archive.namelist()
            if name.startswith(f"{top_level}/{ACTIVE_ARCHIVE_DIRNAME}/active_") and name.endswith(".json")
        ]
        for name in manifest_names:
            manifest = _load_json_bytes(archive.read(name))
            station_id = str(manifest.get("station_id") or "").strip()
            if station_id:
                return station_id
        return top_level


def _load_json_bytes(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def find_archive_zips(search_roots: list[Path], station_prefix: str) -> list[Path]:
    """Find portable archive zips by inspecting their embedded station ID."""
    matches: list[Path] = []
    seen: set[Path] = set()
    prefix = str(station_prefix or "").strip()
    for root in search_roots:
        if not root.is_dir():
            continue
        for zip_path in root.glob("*.zip"):
            resolved = zip_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                station_id = archive_station_id(resolved)
            except (OSError, zipfile.BadZipFile):
                continue
            if station_id.startswith(prefix):
                matches.append(resolved)
    return sorted(matches)


def _restore_path(root: Path, relative: str) -> Path:
    destination = (root / relative).resolve()
    destination.relative_to(root.resolve())
    return destination


def restore_active_multistart_archive(manifest_path: Path, target_dir: Path) -> None:
    manifest = _load_json(manifest_path)
    if manifest.get("archive_type") != "active_multistart":
        raise ValueError(f"not an active multistart manifest: {manifest_path}")
    if target_dir.exists():
        raise FileExistsError(target_dir)
    objects_dir = manifest_path.parent.parent / "objects"
    old_root = str(manifest.get("source_root") or "")
    new_root = str(target_dir.resolve())
    old_repo = str(manifest.get("source_repo") or "")
    new_repo = str(target_dir.resolve().parent)
    replacements = [(old_root, new_root), (old_repo, new_repo)]
    target_dir.mkdir(parents=True)
    for item in manifest.get("files", []):
        relative = str(item.get("path") or "").replace("\\", "/").lstrip("/")
        file_hash = str(item.get("hash") or "")
        if not relative or len(file_hash) != 64:
            continue
        source = objects_dir / file_hash[:2] / file_hash[2:]
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = _restore_path(target_dir, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if item.get("mode") is not None:
            os.chmod(destination, int(item["mode"]))
    for item in manifest.get("symlinks", []):
        relative = str(item.get("path") or "").replace("\\", "/").lstrip("/")
        target = str(item.get("target") or "")
        if not relative or not target:
            continue
        for old, new in replacements:
            if old and old != new:
                target = target.replace(old, new)
        destination = _restore_path(target_dir, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, destination)

    if any(old and old != new for old, new in replacements):
        for path in target_dir.rglob("*"):
            if not path.is_file() or (path.suffix.lower() not in _TEXT_SUFFIXES and path.name != "current_job"):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            updated = content
            for old, new in replacements:
                if old and old != new:
                    updated = updated.replace(old, new)
            if updated != content:
                path.write_text(updated, encoding="utf-8")

    # PIDs from the archived controller/branches must never be trusted after restore.
    job_path = target_dir / str(manifest.get("job_dir_name") or "")
    if not job_path.is_dir():
        matches = list(target_dir.glob("*/state.yaml"))
        job_path = matches[0].parent if matches else job_path
    state_path = job_path / "state.yaml"
    state = _load_mapping(state_path)
    if state:
        branches = state.get("branches")
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, dict) and str(branch.get("status") or "").lower() not in _TERMINAL_BRANCH_STATUSES:
                    branch["pid"] = None
        state.pop("shutdown_requested", None)
        state.pop("shutdown_requested_at", None)
        _save_mapping(state_path, state)


def cli_active_info(backup_dir: Path) -> str:
    found = latest_active_multistart_manifest(backup_dir)
    if found is None:
        return ""
    path, manifest = found
    return f"{int(manifest.get('station_tick') or 0)}\t{path}"


def extract_station_archive_zip(zip_path: Path, backup_root: Path) -> Path:
    """Safely extract one station archive zip without overwriting a backup."""
    zip_path = zip_path.expanduser().resolve()
    backup_root = backup_root.expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    backup_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        top_levels: set[str] = set()
        for member in members:
            normalized = member.filename.replace("\\", "/")
            parts = Path(normalized).parts
            if not parts or normalized.startswith("/") or ".." in parts:
                raise ValueError(f"unsafe archive member: {member.filename}")
            top_levels.add(parts[0])
        if len(top_levels) != 1:
            raise ValueError(f"station archive must contain exactly one top-level backup directory: {zip_path}")
        station_id = next(iter(top_levels))
        target = backup_root / station_id
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing backup directory: {target}")

        temporary_root = backup_root / f".extract_{station_id}_{uuid.uuid4().hex}"
        temporary_root.mkdir()
        try:
            archive.extractall(temporary_root)
            extracted = temporary_root / station_id
            if not extracted.is_dir():
                raise ValueError(f"archive did not create expected station backup directory: {station_id}")
            if not (extracted / "snapshots").is_dir() and not (extracted / ACTIVE_ARCHIVE_DIRNAME).is_dir():
                raise ValueError(f"zip does not contain a recognized Station backup: {zip_path}")
            os.replace(extracted, target)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
    return target


remove_restore_target = research_storage.remove_tree_allow_read_only
