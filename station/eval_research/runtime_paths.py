"""
Runtime path helpers for the Research Center.
"""

import errno
import os
import shutil
import stat
import uuid
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from filelock import FileLock

from station import constants
from station import file_io_utils
from station import research_storage


_CHMOD_BEST_EFFORT_ERRNOS = {
    errno.EACCES,
    errno.EPERM,
    errno.EROFS,
}
for _errno_name in ("ENOTSUP", "EOPNOTSUPP"):
    _errno_value = getattr(errno, _errno_name, None)
    if _errno_value is not None:
        _CHMOD_BEST_EFFORT_ERRNOS.add(_errno_value)


SUBMIT_EVAL_CLI_SNAPSHOT_FILENAME = "submit_eval_cli_snapshot.py"
SUBMIT_AUDIT_CLI_SNAPSHOT_FILENAME = "submit_audit_cli_snapshot.py"
EVAL_TOOL_CLI_SNAPSHOT_FILENAME = "eval_tool_cli_snapshot.py"
LOCAL_PROBE_SNAPSHOT_FILENAME = "local_probe_snapshot.py"
OLD_PREVIEW_EVAL_CLI_SNAPSHOT_FILENAME = "preview_eval_cli_snapshot.py"
SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME = "_internal"
LINEAGE_ALIAS_MIGRATION_LOCK_FILENAME = "lineage_alias_migration.lock"
LINEAGE_ALIAS_CONFLICT_SUFFIX = ".pre_alias_merge"
IMMUTABLE_SYSTEM_STORAGE_BACKUP_PREFIX = ".system_read_only_legacy"


@dataclass(frozen=True)
class ResearchRuntimePaths:
    research_root: str
    debug_tmp_root: str
    storage_root: str
    storage_real_root: str
    shared_storage: str
    system_storage: str
    lineages_root: str
    architect_storage: str
    tmp_storage: str
    submissions_dir: str
    stdout_dir: str
    stderr_dir: str
    reports_dir: str
    audit_dir: str
    internal_root: str
    evaluations_dir: str
    evaluators_dir: str
    run_requests_dir: str
    coder_sessions_dir: str
    task_spec_path: str
    baseline_path: str
    submit_script_path: str
    eval_tool_script_path: str
    local_probe_script_path: str
    evaluation_index_path: str


def get_research_root(consts_module=constants) -> str:
    return os.path.join(
        consts_module.BASE_STATION_DATA_PATH,
        consts_module.ROOMS_DIR_NAME,
        consts_module.SHORT_ROOM_NAME_RESEARCH,
    )


def _chmod_best_effort(path: str, mode: int) -> bool:
    try:
        os.chmod(path, mode)
        return True
    except FileNotFoundError:
        return True
    except NotImplementedError:
        return False
    except OSError as exc:
        if exc.errno in _CHMOD_BEST_EFFORT_ERRNOS:
            return False
        raise


def _set_read_only_tree(dir_path: str) -> bool:
    if not os.path.exists(dir_path):
        return True
    permissions_applied = True
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.islink(file_path) and not os.path.exists(file_path):
                continue
            permissions_applied = _chmod_best_effort(file_path, 0o444) and permissions_applied
        for dirname in dirs:
            dir_child = os.path.join(root, dirname)
            if os.path.islink(dir_child) and not os.path.exists(dir_child):
                continue
            permissions_applied = _chmod_best_effort(dir_child, 0o555) and permissions_applied
    permissions_applied = _chmod_best_effort(dir_path, 0o555) and permissions_applied
    return permissions_applied


def _set_writable_tree(dir_path: str) -> bool:
    if not os.path.exists(dir_path):
        return True
    permissions_applied = True
    for root, dirs, files in os.walk(dir_path):
        permissions_applied = _chmod_best_effort(root, 0o755) and permissions_applied
        for dirname in dirs:
            dir_child = os.path.join(root, dirname)
            if os.path.islink(dir_child) and not os.path.exists(dir_child):
                continue
            permissions_applied = _chmod_best_effort(dir_child, 0o755) and permissions_applied
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.islink(file_path) and not os.path.exists(file_path):
                continue
            permissions_applied = _chmod_best_effort(file_path, 0o644) and permissions_applied
    return permissions_applied


def _copy_tree_with_writable_creation_modes(source: Path, destination: Path) -> None:
    """Copy a tree without chmod/copystat calls.

    Some shared filesystems permit creating files and directories but reject
    chmod.  New entries therefore need owner-write permission in their creation
    mode rather than through a later permission repair.
    """
    source_mode = stat.S_IMODE(source.stat().st_mode)
    destination.mkdir(mode=source_mode | stat.S_IRWXU)
    with os.scandir(source) as entries:
        for entry in entries:
            source_child = Path(entry.path)
            destination_child = destination / entry.name
            if entry.is_symlink():
                os.symlink(os.readlink(source_child), destination_child)
            elif entry.is_dir(follow_symlinks=False):
                _copy_tree_with_writable_creation_modes(source_child, destination_child)
            elif entry.is_file(follow_symlinks=False):
                child_mode = stat.S_IMODE(entry.stat(follow_symlinks=False).st_mode)
                descriptor = os.open(
                    destination_child,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    child_mode | stat.S_IWUSR,
                )
                try:
                    with open(source_child, "rb") as source_handle, os.fdopen(descriptor, "wb") as destination_handle:
                        descriptor = -1
                        shutil.copyfileobj(source_handle, destination_handle)
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
            else:
                raise RuntimeError(f"unsupported storage/system entry: {source_child}")


def _replace_immutable_system_storage(system_storage: str) -> bool:
    """Replace an immutable copied system tree through its writable parent.

    Multistart can copy a local ``0555 storage/system`` directory into an NFS
    allocation whose server permits create/rename but rejects chmod.  The old
    tree cannot be updated in place, so build a writable copy beside it and swap
    directory entries atomically.  Keep the old, tiny tree as a hidden recovery
    backup because that same NFS policy may also prevent deleting its children.
    """
    system_path = Path(system_storage)
    if not system_path.is_dir() or system_path.is_symlink():
        return False
    parent = system_path.parent
    temporary = parent / f".system_writable.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    backup = parent / IMMUTABLE_SYSTEM_STORAGE_BACKUP_PREFIX
    if backup.exists() or backup.is_symlink():
        backup = parent / f"{IMMUTABLE_SYSTEM_STORAGE_BACKUP_PREFIX}.{uuid.uuid4().hex}"
    moved_original = False
    try:
        _copy_tree_with_writable_creation_modes(system_path, temporary)
        os.replace(system_path, backup)
        moved_original = True
        os.replace(temporary, system_path)
        print(
            "Research Center: Replaced immutable storage/system through its writable "
            f"allocation parent; recovery backup kept at {backup}"
        )
        return True
    except OSError as exc:
        if moved_original and not system_path.exists() and backup.exists():
            os.replace(backup, system_path)
        if exc.errno in _CHMOD_BEST_EFFORT_ERRNOS:
            return False
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _station_id_for_storage(consts_module=constants) -> str:
    config_path = os.path.join(
        consts_module.BASE_STATION_DATA_PATH,
        consts_module.STATION_CONFIG_FILENAME,
    )
    try:
        config = file_io_utils.load_yaml(config_path)
    except Exception:
        config = {}
    if isinstance(config, dict) and config.get(consts_module.STATION_ID_KEY):
        return str(config[consts_module.STATION_ID_KEY])
    return "station"


def _migrate_storage_to_base_path(local_storage_path: str, consts_module=constants) -> str:
    base_path = research_storage.configured_base_path(
        getattr(consts_module, "RESEARCH_STORAGE_BASE_PATH", None)
    )
    if base_path is None:
        return local_storage_path
    station_id = _station_id_for_storage(consts_module)
    if os.path.islink(local_storage_path):
        target = research_storage.relocate_storage_symlink(
            Path(local_storage_path),
            base_path,
            marker_payload={
                "kind": "live",
                "station_id": station_id,
                "created_by": "research_runtime_relocation",
            },
            remove_tree=research_storage.remove_tree_allow_read_only,
        )
        print(f"Research Center: Storage link now uses configured base: {local_storage_path} -> {target}")
        return str(target)
    base_path.mkdir(parents=True, exist_ok=True)
    shared_base_path = str(research_storage.new_allocation_path(base_path))
    print(f"Research Center: Starting storage migration to: {shared_base_path}")
    os.makedirs(shared_base_path, exist_ok=True)
    research_storage.write_allocation_marker(
        Path(shared_base_path),
        {
            "kind": "live",
            "station_id": station_id,
            "created_by": "research_runtime_migration",
        },
    )

    if os.path.exists(local_storage_path) and os.path.isdir(local_storage_path):
        for item in os.listdir(local_storage_path):
            src = os.path.join(local_storage_path, item)
            dst = os.path.join(shared_base_path, item)
            if os.path.exists(dst):
                print(f"Research Center: Skipping existing shared storage item {item}")
                continue
            try:
                if os.path.isdir(src) and item == consts_module.RESEARCH_STORAGE_SYSTEM_DIR:
                    _set_writable_tree(src)
                shutil.move(src, dst)
                if item == consts_module.RESEARCH_STORAGE_SYSTEM_DIR:
                    _set_read_only_tree(dst)
            except Exception as exc:
                print(f"Research Center: Warning - Could not move {item}: {exc}")

        try:
            os.rmdir(local_storage_path)
        except OSError:
            backup_path = local_storage_path + ".old"
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path, ignore_errors=True)
            os.rename(local_storage_path, backup_path)
            print(f"Research Center: Renamed old storage directory to {backup_path}")

    os.symlink(shared_base_path, local_storage_path)
    print(f"Research Center: Storage migration completed. Symlink: {local_storage_path} -> {shared_base_path}")
    return shared_base_path


def build_runtime_paths(consts_module=constants) -> ResearchRuntimePaths:
    research_root = get_research_root(consts_module)
    storage_root = os.path.join(research_root, consts_module.RESEARCH_STORAGE_DIR)
    storage_real_root = os.path.realpath(storage_root) if os.path.islink(storage_root) else storage_root

    return ResearchRuntimePaths(
        research_root=research_root,
        debug_tmp_root=os.path.join(research_root, consts_module.RESEARCH_CODER_DEBUG_DIR_NAME),
        storage_root=storage_root,
        storage_real_root=storage_real_root,
        shared_storage=os.path.join(storage_real_root, consts_module.RESEARCH_STORAGE_SHARED_DIR),
        system_storage=os.path.join(storage_real_root, consts_module.RESEARCH_STORAGE_SYSTEM_DIR),
        lineages_root=os.path.join(storage_real_root, consts_module.RESEARCH_STORAGE_LINEAGES_DIR),
        architect_storage=os.path.join(storage_real_root, "architect"),
        tmp_storage=os.path.join(storage_real_root, "tmp"),
        submissions_dir=os.path.join(storage_real_root, "submission"),
        stdout_dir=os.path.join(storage_real_root, "stdout"),
        stderr_dir=os.path.join(storage_real_root, "stderr"),
        reports_dir=os.path.join(storage_real_root, "report"),
        audit_dir=os.path.join(storage_real_root, getattr(consts_module, "RESEARCH_AUDIT_SUBDIR_NAME", "audit")),
        internal_root=os.path.join(research_root, consts_module.RESEARCH_INTERNAL_DIR),
        evaluations_dir=os.path.join(research_root, consts_module.RESEARCH_EVALUATIONS_SUBDIR_NAME),
        evaluators_dir=os.path.join(research_root, "evaluators"),
        run_requests_dir=os.path.join(research_root, consts_module.RESEARCH_RUN_REQUESTS_SUBDIR_NAME),
        coder_sessions_dir=os.path.join(research_root, consts_module.RESEARCH_CODER_SESSIONS_SUBDIR_NAME),
        task_spec_path=os.path.join(research_root, consts_module.RESEARCH_TASK_SPEC_FILENAME),
        baseline_path=os.path.join(research_root, consts_module.RESEARCH_BASELINE_FILENAME),
        submit_script_path=os.path.join(research_root, "submit_eval.sh"),
        eval_tool_script_path=os.path.join(research_root, "eval_tool.sh"),
        local_probe_script_path=os.path.join(research_root, "local_probe.sh"),
        evaluation_index_path=os.path.join(research_root, consts_module.RESEARCH_EVALUATIONS_SUBDIR_NAME, consts_module.RESEARCH_EVALUATION_INDEX_FILENAME),
    )


def parse_memory_limit_bytes(memory_limit: Any) -> Optional[int]:
    if not memory_limit:
        return None
    memory_str = str(memory_limit).strip().lower()
    if not memory_str:
        return None
    multipliers = {
        "gb": 1024 ** 3,
        "g": 1024 ** 3,
        "mb": 1024 ** 2,
        "m": 1024 ** 2,
        "kb": 1024,
        "k": 1024,
        "b": 1,
    }
    for suffix, multiplier in multipliers.items():
        if memory_str.endswith(suffix):
            return int(float(memory_str[: -len(suffix)]) * multiplier)
    return int(memory_str)


def _ensure_dir_list(dir_paths: list[str]):
    for dir_path in dir_paths:
        file_io_utils.ensure_dir_exists(dir_path)


def _install_lineage_alias(alias_path: str, physical_path: str):
    relative_target = os.path.relpath(physical_path, os.path.dirname(alias_path))
    temporary_alias = f"{alias_path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        os.symlink(relative_target, temporary_alias)
        os.replace(temporary_alias, alias_path)
    finally:
        if os.path.lexists(temporary_alias):
            os.unlink(temporary_alias)


def _next_lineage_conflict_path(path: str) -> str:
    candidate = path + LINEAGE_ALIAS_CONFLICT_SUFFIX
    index = 2
    while os.path.lexists(candidate):
        candidate = f"{path}{LINEAGE_ALIAS_CONFLICT_SUFFIX}.{index}"
        index += 1
    return candidate


def _merge_directory_contents(source_dir: str, destination_dir: str) -> int:
    renamed_conflicts = 0
    for name in sorted(os.listdir(source_dir)):
        source = os.path.join(source_dir, name)
        destination = os.path.join(destination_dir, name)
        if (
            not os.path.islink(source)
            and os.path.isdir(source)
            and not os.path.islink(destination)
            and os.path.isdir(destination)
        ):
            renamed_conflicts += _merge_directory_contents(source, destination)
            os.rmdir(source)
            continue
        if os.path.lexists(destination):
            os.replace(destination, _next_lineage_conflict_path(destination))
            renamed_conflicts += 1
        os.replace(source, destination)
    return renamed_conflicts


def _merge_legacy_lineage_directory(legacy_path: str, physical_path: str) -> int:
    renamed_conflicts = _merge_directory_contents(legacy_path, physical_path)
    os.rmdir(legacy_path)
    _install_lineage_alias(legacy_path, physical_path)
    return renamed_conflicts


def ensure_lineage_storage(paths: ResearchRuntimePaths, lineage_name: str) -> str:
    lineage_name = (lineage_name or "unknown").lower()
    physical_path = os.path.join(paths.lineages_root, lineage_name)
    alias_path = os.path.join(paths.storage_real_root, lineage_name)

    file_io_utils.ensure_dir_exists(physical_path)
    data_path = os.path.join(physical_path, "data")
    file_io_utils.ensure_dir_exists(data_path)

    if os.path.islink(alias_path) and os.path.realpath(alias_path) == os.path.realpath(physical_path):
        return physical_path

    file_io_utils.ensure_dir_exists(paths.internal_root)
    migration_lock_path = os.path.join(paths.internal_root, LINEAGE_ALIAS_MIGRATION_LOCK_FILENAME)
    with FileLock(migration_lock_path):
        if os.path.islink(alias_path):
            if os.path.realpath(alias_path) == os.path.realpath(physical_path):
                return physical_path
            raise RuntimeError(
                f"Lineage alias {alias_path} points to {os.path.realpath(alias_path)}, expected {physical_path}"
            )
        if os.path.lexists(alias_path):
            if not os.path.isdir(alias_path):
                raise RuntimeError(
                    f"Lineage alias path is not a directory or symlink: {alias_path}"
                )
            conflict_count = _merge_legacy_lineage_directory(alias_path, physical_path)
            print(
                f"Research Center: Migrated legacy lineage storage {alias_path} -> {physical_path} "
                f"({conflict_count} existing canonical entries received a {LINEAGE_ALIAS_CONFLICT_SUFFIX} suffix)"
            )
        else:
            _install_lineage_alias(alias_path, physical_path)

        if not os.path.islink(alias_path) or os.path.realpath(alias_path) != os.path.realpath(physical_path):
            raise RuntimeError(
                f"Could not establish lineage alias {alias_path} -> {physical_path}"
            )

    return physical_path


def _is_invalid_lineage_storage_name(lineage_name: str) -> bool:
    normalized = str(lineage_name or "").strip().lower()
    return normalized in constants.RESEARCH_STORAGE_RESERVED_NAMES or not normalized.isalpha()


def _directory_contains_file_or_symlink(dir_path: str) -> bool:
    try:
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_symlink() or entry.is_file(follow_symlinks=False):
                    return True
                if entry.is_dir(follow_symlinks=False) and _directory_contains_file_or_symlink(entry.path):
                    return True
    except OSError:
        return True
    return False


def cleanup_empty_invalid_lineage_storage_dirs(paths: ResearchRuntimePaths) -> list[str]:
    """Remove empty invalid lineage stubs under storage/lineages only."""
    removed: list[str] = []
    if not os.path.isdir(paths.lineages_root):
        return removed

    for lineage_name in sorted(os.listdir(paths.lineages_root)):
        lineage_path = os.path.join(paths.lineages_root, lineage_name)
        if os.path.islink(lineage_path) or not os.path.isdir(lineage_path):
            continue
        if not _is_invalid_lineage_storage_name(lineage_name):
            continue
        if _directory_contains_file_or_symlink(lineage_path):
            continue
        try:
            shutil.rmtree(lineage_path)
            removed.append(lineage_name)
        except OSError as exc:
            print(f"Research Center: Warning - Could not remove empty invalid lineage storage {lineage_path}: {exc}")
            continue

        alias_path = os.path.join(paths.storage_real_root, lineage_name)
        if os.path.islink(alias_path):
            try:
                target = os.path.realpath(alias_path)
                if target == os.path.realpath(lineage_path):
                    os.unlink(alias_path)
            except OSError as exc:
                print(f"Research Center: Warning - Could not remove invalid lineage alias {alias_path}: {exc}")

    if removed:
        print(f"Research Center: Removed empty invalid lineage storage directories: {', '.join(removed)}")
    return removed


def sync_lineage_aliases(paths: ResearchRuntimePaths):
    if not os.path.isdir(paths.lineages_root):
        return

    reserved_names = {
        constants.RESEARCH_STORAGE_SHARED_DIR,
        constants.RESEARCH_STORAGE_SYSTEM_DIR,
        constants.RESEARCH_STORAGE_LINEAGES_DIR,
        "architect",
        "tmp",
        "submission",
        "stdout",
        "stderr",
        "report",
    }

    for lineage_name in os.listdir(paths.lineages_root):
        if lineage_name in reserved_names:
            continue
        lineage_path = os.path.join(paths.lineages_root, lineage_name)
        if os.path.isdir(lineage_path):
            ensure_lineage_storage(paths, lineage_name)


def ensure_runtime_layout(consts_module=constants) -> ResearchRuntimePaths:
    research_root = get_research_root(consts_module)
    file_io_utils.ensure_dir_exists(research_root)

    local_storage_path = os.path.join(research_root, consts_module.RESEARCH_STORAGE_DIR)
    storage_real_root = local_storage_path
    if research_storage.configured_base_path(
        getattr(consts_module, "RESEARCH_STORAGE_BASE_PATH", None)
    ):
        storage_real_root = _migrate_storage_to_base_path(local_storage_path, consts_module)

    file_io_utils.ensure_dir_exists(storage_real_root)
    paths = build_runtime_paths(consts_module)

    runtime_dirs = [
        paths.debug_tmp_root,
        paths.shared_storage,
        paths.system_storage,
        paths.lineages_root,
        paths.architect_storage,
        paths.tmp_storage,
        paths.submissions_dir,
        paths.stdout_dir,
        paths.stderr_dir,
        paths.reports_dir,
        paths.internal_root,
        paths.evaluations_dir,
        paths.run_requests_dir,
        paths.coder_sessions_dir,
    ]
    if bool(getattr(consts_module, "RESEARCH_CODER_AUDIT_ENABLED", True)):
        runtime_dirs.append(paths.audit_dir)
    _ensure_dir_list(runtime_dirs)

    system_storage_writable = _set_writable_tree(paths.system_storage)
    if not system_storage_writable and os.access(paths.system_storage, os.W_OK | os.X_OK):
        system_storage_writable = True
    if not system_storage_writable:
        system_storage_writable = _replace_immutable_system_storage(paths.system_storage)
    sync_lineage_aliases(paths)
    ensure_evaluator_symlinks(paths)
    if bool(getattr(consts_module, "RESEARCH_SEED_BANK_ENABLED", False)):
        from station.eval_research.seed_bank import ensure_seed_bank_layout, install_seed_bank_client

        ensure_seed_bank_layout(paths, consts_module)
        install_seed_bank_client(
            paths,
            consts_module,
            snapshot_dirname=SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME,
        )
    system_storage_read_only = _set_read_only_tree(paths.system_storage)
    if not system_storage_writable or not system_storage_read_only:
        print(
            "Research Center: Warning - Could not fully update permissions for "
            f"{paths.system_storage}; continuing with existing filesystem permissions."
        )
    ensure_submit_script(paths)
    ensure_eval_tool_script(paths)
    ensure_local_probe_script(paths, consts_module)
    if bool(getattr(consts_module, "RESEARCH_CODER_AUDIT_ENABLED", True)):
        ensure_audit_script(paths)
    return paths


def ensure_submit_runtime_layout(consts_module=constants) -> ResearchRuntimePaths:
    """
    Ensure only the minimal writable runtime needed for queueing official attempts.

    This intentionally does not mutate `storage/system` or other read-only task assets.
    Full bootstrap belongs to station / evaluator startup through `ensure_runtime_layout()`.
    """
    research_root = get_research_root(consts_module)
    file_io_utils.ensure_dir_exists(research_root)

    local_storage_path = os.path.join(research_root, consts_module.RESEARCH_STORAGE_DIR)
    file_io_utils.ensure_dir_exists(local_storage_path)
    paths = build_runtime_paths(consts_module)

    _ensure_dir_list([
        paths.storage_real_root,
        paths.tmp_storage,
        paths.submissions_dir,
        paths.stdout_dir,
        paths.stderr_dir,
        paths.reports_dir,
        paths.evaluations_dir,
        paths.run_requests_dir,
        paths.coder_sessions_dir,
    ])
    return paths


def strip_task_spec_coder_only_sections(task_spec_markdown: str, consts_module=constants) -> str:
    begin_marker = str(getattr(consts_module, "RESEARCH_TASK_CODER_ONLY_BEGIN_MARKER", "__CODER_ONLY_BEGIN__"))
    end_marker = str(getattr(consts_module, "RESEARCH_TASK_CODER_ONLY_END_MARKER", "__CODER_ONLY_END__"))
    if not task_spec_markdown or begin_marker not in task_spec_markdown or end_marker not in task_spec_markdown:
        return task_spec_markdown

    pattern = re.compile(
        re.escape(begin_marker) + r".*?" + re.escape(end_marker),
        flags=re.DOTALL,
    )
    return pattern.sub("", task_spec_markdown)


def load_task_spec_markdown_for_audience(consts_module=constants, *, include_coder_only: bool = False) -> str:
    paths = build_runtime_paths(consts_module)
    if file_io_utils.file_exists(paths.task_spec_path):
        task_spec_markdown = file_io_utils.load_text(paths.task_spec_path) or ""
        if include_coder_only:
            return task_spec_markdown
        return strip_task_spec_coder_only_sections(task_spec_markdown, consts_module)
    return ""


def load_task_spec_markdown(consts_module=constants, *, include_coder_only: bool = False) -> str:
    return load_task_spec_markdown_for_audience(consts_module, include_coder_only=include_coder_only)


def ensure_task_spec_markdown(consts_module=constants):
    paths = ensure_runtime_layout(consts_module)
    if file_io_utils.file_exists(paths.task_spec_path):
        return


def ensure_evaluator_symlinks(paths: Optional[ResearchRuntimePaths] = None):
    if paths is None:
        paths = build_runtime_paths(constants)

    evaluator_source = os.path.join(paths.evaluators_dir, "evaluator.py")
    if not file_io_utils.file_exists(evaluator_source):
        return

    desired_links = [os.path.join(paths.system_storage, "evaluator.py")]

    for link_path in desired_links:
        if os.path.lexists(link_path):
            if os.path.islink(link_path) and os.path.realpath(link_path) == os.path.realpath(evaluator_source):
                continue
            if os.path.isfile(link_path):
                try:
                    os.remove(link_path)
                except OSError:
                    continue
            elif os.path.islink(link_path):
                try:
                    os.unlink(link_path)
                except OSError:
                    continue
            else:
                continue
        try:
            relative_target = os.path.relpath(evaluator_source, os.path.dirname(link_path))
            os.symlink(relative_target, link_path)
        except FileExistsError:
            continue
        except OSError as exc:
            print(f"Research Center: Warning - Could not create evaluator symlink {link_path}: {exc}")


def _submit_eval_snapshot_source() -> str:
    template = r'''#!/usr/bin/env python3
"""Frozen Research Center submit CLI generated at station startup."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

STATION_REPO_ROOT = __STATION_REPO_ROOT__
if STATION_REPO_ROOT and STATION_REPO_ROOT not in sys.path:
    sys.path.insert(0, STATION_REPO_ROOT)

from station.eval_research.evaluation_manager import EvaluationManager


RESEARCH_STORAGE_DIR = __RESEARCH_STORAGE_DIR__
RESEARCH_EVALUATIONS_SUBDIR_NAME = __RESEARCH_EVALUATIONS_SUBDIR_NAME__
RESEARCH_RUN_REQUESTS_SUBDIR_NAME = __RESEARCH_RUN_REQUESTS_SUBDIR_NAME__
RESEARCH_SCORE_NA = __RESEARCH_SCORE_NA__
RESEARCH_CODER_MAX_ATTEMPTS = __RESEARCH_CODER_MAX_ATTEMPTS__
GPU_MANAGEMENT_ENABLED = __GPU_MANAGEMENT_ENABLED__
TERMINAL_STATUSES = {"completed", "failed", "blocked", "partial"}
ACTIVE_ATTEMPT_STATUSES = {"queued", "running"}
ARTIFACT_SPECS = {
    "submission": ("submission", ".py"),
    "stdout": ("stdout", ".log"),
    "stderr": ("stderr", ".log"),
    "report": ("report", ".md"),
}


def _research_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _artifact_rel_path(eval_id: str, key: str) -> str:
    subdir, extension = ARTIFACT_SPECS[key]
    return os.path.join(RESEARCH_STORAGE_DIR, subdir, f"{eval_id}{extension}")


def _artifact_abs_path(research_root: Path, eval_id: str, key: str, eval_data: Dict[str, Any] | None = None) -> Path:
    rel_path = ""
    artifacts = eval_data.get("artifacts") if isinstance(eval_data, dict) else None
    if isinstance(artifacts, dict):
        rel_value = artifacts.get(key)
        if isinstance(rel_value, str) and rel_value.strip():
            rel_path = rel_value.strip()
    if not rel_path:
        rel_path = _artifact_rel_path(eval_id, key)
    path = Path(rel_path)
    return path if path.is_absolute() else research_root / path


def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _atomic_write_yaml(path: Path, data: Dict[str, Any]):
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    _atomic_write_text(path, text)


def _attempt_footer(
    primary_score: Any,
    secondary_metrics: Any,
    safe_to_resubmit: bool,
    attempt_status: str = "completed",
) -> list[str]:
    normalized_status = str(attempt_status or "completed").strip().lower()
    if normalized_status == "completed":
        status_message = (
            "The submission has run to completion. You may now submit the next attempt or write the final Coder Report."
        )
    else:
        status_message = (
            f"The attempt has settled with status '{normalized_status}'. Inspect the rejection or failure details "
            "before deciding whether to retry or finalize."
        )
    return [
        "ATTEMPT_COMPLETE",
        f"ATTEMPT_STATUS: {normalized_status}",
        status_message,
        f"PRIMARY_SCORE: {primary_score}",
        f"SECONDARY_METRICS: {secondary_metrics}",
        f"SAFE_TO_RESUBMIT: {'true' if safe_to_resubmit else 'false'}",
    ]


def _attempt_queue_banner(eval_id: str, attempt_number: int) -> list[str]:
    return [
        "ATTEMPT_QUEUED",
        f"Official evaluation attempt {attempt_number} for evaluation {eval_id} has been queued.",
        "The attempt is waiting for an evaluator worker and any required CPU/GPU resources.",
        "This is normal. Keep polling this log and wait here while the station scheduler dispatches the official run.",
        "Do not treat the queued state by itself as a failure, and do not bypass the official submit path just because the run has not started yet.",
        "The evaluator may run outside your sandbox's PID namespace, so do not use `ps` or `pgrep` to decide whether it has ended; wait patiently for `ATTEMPT_COMPLETE`.",
    ]


def _write_status(research_root: Path, eval_id: str, eval_data: Dict[str, Any] | None, lines: list[str]):
    text = "\n".join(lines).rstrip() + "\n"
    stdout_path = _artifact_abs_path(research_root, eval_id, "stdout", eval_data)
    stderr_path = _artifact_abs_path(research_root, eval_id, "stderr", eval_data)
    _atomic_write_text(stdout_path, text)
    if not stderr_path.exists():
        _atomic_write_text(stderr_path, "")


def _print_rejection(lines: list[str]):
    text = "\n".join(lines).rstrip()
    if text:
        print(text)


def _latest_attempt_is_active(eval_data: Dict[str, Any]) -> bool:
    attempts = eval_data.get("attempts") or []
    if not attempts:
        return False
    latest = attempts[-1]
    return str(latest.get("status")) in ACTIVE_ATTEMPT_STATUSES


def submit_eval(eval_id: str, *, cpu_only: bool = False) -> int:
    eval_id = str(eval_id)
    research_root = _research_root()
    evaluations_dir = research_root / RESEARCH_EVALUATIONS_SUBDIR_NAME
    submission_path = research_root / RESEARCH_STORAGE_DIR / "submission" / f"{eval_id}.py"
    run_requests_dir = research_root / RESEARCH_RUN_REQUESTS_SUBDIR_NAME
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    (research_root / RESEARCH_STORAGE_DIR / "stdout").mkdir(parents=True, exist_ok=True)
    (research_root / RESEARCH_STORAGE_DIR / "stderr").mkdir(parents=True, exist_ok=True)
    run_requests_dir.mkdir(parents=True, exist_ok=True)

    eval_manager = EvaluationManager(str(evaluations_dir), preload=False)
    eval_data = eval_manager.get_evaluation(eval_id)
    effective_cpu_only = bool(cpu_only and GPU_MANAGEMENT_ENABLED)
    if not eval_data:
        _write_status(
            research_root,
            eval_id,
            None,
            [f"Submission rejected: evaluation {eval_id} not found."]
            + _attempt_footer(RESEARCH_SCORE_NA, "{}", False, attempt_status="rejected"),
        )
        return 1

    if not submission_path.exists():
        _write_status(
            research_root,
            eval_id,
            eval_data,
            [f"Submission rejected: submission file not found at {submission_path}."]
            + _attempt_footer(RESEARCH_SCORE_NA, "{}", False, attempt_status="rejected"),
        )
        return 1

    submission_text = submission_path.read_text(encoding="utf-8")
    if not submission_text.strip():
        _write_status(
            research_root,
            eval_id,
            eval_data,
            [f"Submission rejected: submission file {submission_path} is empty."]
            + _attempt_footer(RESEARCH_SCORE_NA, "{}", False, attempt_status="rejected"),
        )
        return 1

    if eval_data.get("status") in TERMINAL_STATUSES:
        _write_status(
            research_root,
            eval_id,
            eval_data,
            [f"Submission rejected: evaluation {eval_id} is already in terminal status '{eval_data.get('status')}'."]
            + _attempt_footer(RESEARCH_SCORE_NA, "{}", False, attempt_status="rejected"),
        )
        return 1
    max_attempts = int((eval_data.get("coder") or {}).get("max_attempts", RESEARCH_CODER_MAX_ATTEMPTS))
    attempts = eval_data.get("attempts") or []
    if len(attempts) >= max_attempts:
        _write_status(
            research_root,
            eval_id,
            eval_data,
            [f"Submission rejected: maximum attempts reached ({max_attempts})."]
            + _attempt_footer(RESEARCH_SCORE_NA, "{}", False, attempt_status="rejected"),
        )
        return 1
    if _latest_attempt_is_active(eval_data):
        _print_rejection(
            [
                f"Submission rejected: an attempt is already active for evaluation {eval_id}.",
                "Keep polling the current stdout/stderr logs for the active attempt; they were not modified.",
            ]
        )
        return 1
    attempt_number = eval_manager.register_attempt(eval_id, str(submission_path), cpu_only=effective_cpu_only)
    if attempt_number is None:
        _write_status(
            research_root,
            eval_id,
            eval_data,
            [f"Submission rejected: could not register attempt for evaluation {eval_id}."]
            + _attempt_footer(RESEARCH_SCORE_NA, "{}", False, attempt_status="rejected"),
        )
        return 1

    queue_banner = _attempt_queue_banner(eval_id, int(attempt_number))
    if effective_cpu_only:
        queue_banner.append("CPU-only mode requested: the scheduler will not reserve a GPU for this attempt.")
    _write_status(research_root, eval_id, eval_data, queue_banner)
    run_request = {
        "eval_id": eval_id,
        "attempt": int(attempt_number),
        "created_timestamp": time.time(),
    }
    if effective_cpu_only:
        run_request["cpu_only"] = True
    _atomic_write_yaml(
        run_requests_dir / f"{eval_id}_attempt_{attempt_number}.yaml",
        run_request,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Queue a Research Center evaluation attempt.")
    parser.add_argument("eval_id", help="Evaluation ID to submit")
    if GPU_MANAGEMENT_ENABLED:
        parser.add_argument(
            "--cpu-only",
            action="store_true",
            help="Queue this attempt without reserving a station-managed GPU.",
        )
    args = parser.parse_args(argv)
    return submit_eval(args.eval_id, cpu_only=getattr(args, "cpu_only", False))


if __name__ == "__main__":
    sys.exit(main())
'''
    replacements = {
        "__RESEARCH_STORAGE_DIR__": repr(constants.RESEARCH_STORAGE_DIR),
        "__RESEARCH_EVALUATIONS_SUBDIR_NAME__": repr(constants.RESEARCH_EVALUATIONS_SUBDIR_NAME),
        "__RESEARCH_RUN_REQUESTS_SUBDIR_NAME__": repr(constants.RESEARCH_RUN_REQUESTS_SUBDIR_NAME),
        "__RESEARCH_SCORE_NA__": repr(constants.RESEARCH_SCORE_NA),
        "__RESEARCH_CODER_MAX_ATTEMPTS__": repr(constants.RESEARCH_CODER_MAX_ATTEMPTS),
        "__GPU_MANAGEMENT_ENABLED__": repr(
            constants.RESEARCH_EVAL_GPU_NUM is not None or bool(constants.RESEARCH_EVAL_USE_DIFF_GPU)
        ),
        "__STATION_REPO_ROOT__": repr(str(Path(__file__).resolve().parents[2])),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def _submit_script_wrapper_source(python_executable: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export STATION_BASE_DATA_PATH="$(cd "$SCRIPT_DIR/../.." && pwd)"
exec "{python_executable}" "$SCRIPT_DIR/{SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME}/{SUBMIT_EVAL_CLI_SNAPSHOT_FILENAME}" "$@"
"""


def _eval_tool_snapshot_source() -> str:
    template = r'''#!/usr/bin/env python3
"""Frozen Research Center evaluation tool generated at station startup."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

STATION_REPO_ROOT = __STATION_REPO_ROOT__
if STATION_REPO_ROOT and STATION_REPO_ROOT not in sys.path:
    sys.path.insert(0, STATION_REPO_ROOT)


RESEARCH_STORAGE_DIR = __RESEARCH_STORAGE_DIR__
RESEARCH_EVALUATIONS_SUBDIR_NAME = __RESEARCH_EVALUATIONS_SUBDIR_NAME__
RESEARCH_EVALUATION_FILE_EXTENSION = __RESEARCH_EVALUATION_FILE_EXTENSION__
EVALUATION_DETAILS_KEY = __EVALUATION_DETAILS_KEY__
EVALUATION_ID_KEY = __EVALUATION_ID_KEY__
EVALUATION_TITLE_KEY = __EVALUATION_TITLE_KEY__
EVALUATION_ABSTRACT_KEY = __EVALUATION_ABSTRACT_KEY__
SEARCH_DEFAULT_LIMIT = 30
SEARCH_MAX_LIMIT = 100
ARTIFACT_SPECS = {
    "submission": ("submission", ".py"),
    "stdout": ("stdout", ".log"),
    "stderr": ("stderr", ".log"),
    "report": ("report", ".md"),
}


def _research_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _one_line(value: Any, limit: int = 700) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _format_tags(tags: Any) -> str:
    if isinstance(tags, list):
        return ", ".join(str(tag) for tag in tags) if tags else "(none)"
    if tags:
        return str(tags)
    return "(none)"


def _evaluations_dir(research_root: Path) -> Path:
    return research_root / RESEARCH_EVALUATIONS_SUBDIR_NAME


def _search_abstracts_from_index(research_root: Path, pattern: str, limit: int) -> tuple[int, list[Dict[str, Any]]]:
    try:
        from station.eval_research import evaluation_index

        return evaluation_index.search_abstracts(str(_evaluations_dir(research_root)), pattern, limit)
    except Exception as exc:
        raise RuntimeError(
            "Research SQLite index is unavailable for eval_tool search; restart the station with --rebuild-db."
        ) from exc


def _artifact_rel_path(eval_id: str, key: str) -> str:
    subdir, extension = ARTIFACT_SPECS[key]
    return os.path.join(RESEARCH_STORAGE_DIR, subdir, f"{eval_id}{extension}")


def _artifact_abs_path(research_root: Path, eval_id: str, key: str, eval_data: Dict[str, Any]) -> Path:
    rel_path = ""
    artifacts = eval_data.get("artifacts")
    if isinstance(artifacts, dict):
        rel_value = artifacts.get(key)
        if isinstance(rel_value, str) and rel_value.strip():
            rel_path = rel_value.strip()
    if not rel_path:
        rel_path = _artifact_rel_path(eval_id, key)
    path = Path(rel_path)
    return path if path.is_absolute() else research_root / path


def _display_path(research_root: Path, path: Optional[Path]) -> str:
    if path is None:
        return "(none)"
    try:
        return str(path.relative_to(research_root))
    except ValueError:
        return str(path)


def _final_score_summary(eval_data: Dict[str, Any]) -> str:
    final = eval_data.get("final")
    if not isinstance(final, dict) or not final:
        return "N/A"
    score = final.get("primary_score", "N/A")
    details = final.get(EVALUATION_DETAILS_KEY)
    if isinstance(details, str) and details:
        return f"{score} ({details})"
    legacy_details = final.get("details")
    if isinstance(legacy_details, str) and legacy_details:
        return f"{score} ({legacy_details})"
    return str(score)


def _secondary_metrics_summary(eval_data: Dict[str, Any]) -> str:
    final = eval_data.get("final")
    details = final.get(EVALUATION_DETAILS_KEY) if isinstance(final, dict) else None
    if not isinstance(details, dict):
        return "N/A"
    metrics = []
    for key, value in details.items():
        if key == "Message":
            continue
        if isinstance(value, (list, tuple)) and len(value) == 2:
            metrics.append(f"{key}={value[0]}")
        else:
            metrics.append(f"{key}={value}")
    return "; ".join(metrics) if metrics else "N/A"


def _print_section(title: str, body: str, missing_message: str = "_Not available._"):
    print(f"\n## {title}\n")
    text = body.strip()
    print(text if text else missing_message)


def search_evals(pattern: str, limit: int) -> int:
    research_root = _research_root()
    if limit < 0:
        print("Search limit must be non-negative.", file=sys.stderr)
        return 2
    effective_limit = min(limit, SEARCH_MAX_LIMIT)
    try:
        regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    except re.error as exc:
        print(f"Invalid regex: {exc}", file=sys.stderr)
        return 2

    try:
        total_matches, shown = _search_abstracts_from_index(research_root, pattern, effective_limit)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("# Research Evaluation Abstract Search")
    print(f"\nRegex: `{pattern}`")
    print("Scope: evaluation abstracts only")
    print(f"Matches: {total_matches}")
    if limit > SEARCH_MAX_LIMIT:
        print(f"Requested limit {limit} exceeds the maximum; showing at most {SEARCH_MAX_LIMIT}.")
    print(f"Showing: first {len(shown)} of {total_matches} newest matches")
    if total_matches > len(shown):
        print(f"Use a narrower search term if needed, or pass `--limit N` with N up to {SEARCH_MAX_LIMIT}.")
    print("\nPreview a relevant match with `bash eval_tool.sh preview {ID}` from the Research Center directory, or `bash research_center/eval_tool.sh preview {ID}` from the Surveyor workspace.")

    if not shown:
        print("\nNo matching evaluation abstracts found.")
        return 0

    for search_fields in shown:
        eval_id = str(search_fields.get(EVALUATION_ID_KEY) or "").strip()
        title = search_fields.get(EVALUATION_TITLE_KEY) or "(untitled)"
        abstract = _one_line(search_fields.get(EVALUATION_ABSTRACT_KEY) or "_No abstract recorded._")
        print(f"\n## Eval #{eval_id}: {title}\n")
        print(abstract)
    return 0


def preview_eval(eval_id: str) -> int:
    eval_id = str(eval_id)
    research_root = _research_root()
    eval_path = research_root / RESEARCH_EVALUATIONS_SUBDIR_NAME / f"{eval_id}{RESEARCH_EVALUATION_FILE_EXTENSION}"
    eval_data = _load_yaml(eval_path)
    if not eval_data:
        print(f"Evaluation {eval_id} not found at {_display_path(research_root, eval_path)}.", file=sys.stderr)
        return 1

    report_path = _artifact_abs_path(research_root, eval_id, "report", eval_data)
    stdout_path = _artifact_abs_path(research_root, eval_id, "stdout", eval_data)
    abstract = str(eval_data.get("abstract") or "").strip()
    instruction = str(eval_data.get("instruction") or "").strip()
    report_text = _read_text(report_path)

    print(f"# Research Evaluation Preview #{eval_id}")
    print("\nThis preview intentionally includes evaluation metadata, abstract, the agent instruction, and Coder Report. It does not print the coder prompt, raw submission code, stdout, or stderr logs.")
    print("\n## Metadata\n")
    print(f"- Eval: #{eval_id}")
    print(f"- Title: {eval_data.get('title') or '(untitled)'}")
    print(f"- Author: {eval_data.get('author') or 'Unknown'}")
    print(f"- Lineage: {eval_data.get('lineage') or 'unknown'}")
    print(f"- Tags: {_format_tags(eval_data.get('tags'))}")
    print(f"- Status: {eval_data.get('status') or 'unknown'}")
    print(f"- Submitted tick: {eval_data.get('submitted_tick', 'N/A')}")
    print(f"- Completed tick: {eval_data.get('completed_tick', 'N/A')}")
    print(f"- Final score: {_final_score_summary(eval_data)}")
    print(f"- Secondary metrics: {_secondary_metrics_summary(eval_data)}")
    print(f"- Evaluation YAML: {_display_path(research_root, eval_path)}")
    print("- Agent instruction: shown below")
    print(f"- Coder Report: {_display_path(research_root, report_path)}")

    _print_section("Abstract", abstract, "_No abstract recorded._")
    _print_section("Agent Instruction", instruction, "_No instruction recorded._")
    _print_section(
        "Coder Report",
        report_text,
        f"_No Coder Report found at `{_display_path(research_root, report_path)}` yet._",
    )

    print("\n## Follow-Up Paths\n")
    print(f"- Evaluation YAML: `{_display_path(research_root, eval_path)}`")
    print(f"- Coder Report: `{_display_path(research_root, report_path)}`")
    print(f"- Stdout log: `{_display_path(research_root, stdout_path)}`")
    print("- Reading stdout is not recommended unless the preview is insufficient for the needed technical claim.")
    print("- Raw submission code and stderr are under `storage/submission/` and `storage/stderr/`; inspect them only when the preview and stdout are insufficient.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search Research Center evaluation abstracts or preview evaluation details."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser(
        "search",
        help="Search evaluation abstracts with a case-insensitive Python regex.",
    )
    search_parser.add_argument(
        "regex",
        nargs="+",
        help='Regex to match against abstracts. Multiple tokens are joined with spaces; use "a|b" for OR or "(?=.*a)(?=.*b)" for AND.',
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=SEARCH_DEFAULT_LIMIT,
        help=f"Maximum number of newest matches to print. Default: {SEARCH_DEFAULT_LIMIT}; maximum: {SEARCH_MAX_LIMIT}.",
    )

    preview_parser = subparsers.add_parser(
        "preview",
        help="Preview metadata, abstract, instruction, and Coder Report for one evaluation.",
    )
    preview_parser.add_argument("eval_id", help="Evaluation ID to preview")

    args = parser.parse_args(argv)
    if args.command == "search":
        return search_evals(" ".join(args.regex), args.limit)
    if args.command == "preview":
        return preview_eval(args.eval_id)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
'''
    replacements = {
        "__RESEARCH_STORAGE_DIR__": repr(constants.RESEARCH_STORAGE_DIR),
        "__RESEARCH_EVALUATIONS_SUBDIR_NAME__": repr(constants.RESEARCH_EVALUATIONS_SUBDIR_NAME),
        "__RESEARCH_EVALUATION_FILE_EXTENSION__": repr(constants.RESEARCH_EVALUATION_FILE_EXTENSION),
        "__EVALUATION_DETAILS_KEY__": repr(constants.EVALUATION_DETAILS_KEY),
        "__EVALUATION_ID_KEY__": repr(constants.EVALUATION_ID_KEY),
        "__EVALUATION_TITLE_KEY__": repr(constants.EVALUATION_TITLE_KEY),
        "__EVALUATION_ABSTRACT_KEY__": repr(constants.EVALUATION_ABSTRACT_KEY),
        "__STATION_REPO_ROOT__": repr(str(Path(__file__).resolve().parents[2])),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def _eval_tool_script_wrapper_source(python_executable: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export STATION_BASE_DATA_PATH="$(cd "$SCRIPT_DIR/../.." && pwd)"
exec "{python_executable}" "$SCRIPT_DIR/{SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME}/{EVAL_TOOL_CLI_SNAPSHOT_FILENAME}" "$@"
"""


def _local_probe_snapshot_source(memory_limit: Any) -> str:
    memory_limit_bytes = parse_memory_limit_bytes(memory_limit)
    template = r'''#!/usr/bin/env python3
"""Frozen Research Center local-probe wrapper generated at station startup."""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import time


MEMORY_LIMIT = __MEMORY_LIMIT__
MEMORY_LIMIT_BYTES = __MEMORY_LIMIT_BYTES__
POLL_SECONDS = 0.25
TERMINATE_GRACE_SECONDS = 5.0


def _usage() -> str:
    return (
        "Usage: bash local_probe.sh [--timeout SECONDS] -- <command> [args...]\n"
        "The timeout is optional; omitting it applies no wall-clock timeout."
    )


def _process_group_rss_bytes(process_group_id: int) -> int:
    page_size = os.sysconf("SC_PAGE_SIZE")
    total = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        proc_path = os.path.join("/proc", entry)
        try:
            with open(os.path.join(proc_path, "stat"), "r", encoding="utf-8", errors="replace") as handle:
                stat_text = handle.read()
            close_paren = stat_text.rfind(")")
            if close_paren < 0:
                continue
            fields = stat_text[close_paren + 2 :].split()
            if len(fields) < 3 or int(fields[2]) != int(process_group_id):
                continue
            with open(os.path.join(proc_path, "statm"), "r", encoding="utf-8", errors="replace") as handle:
                statm = handle.read().split()
            if len(statm) >= 2:
                total += int(statm[1]) * page_size
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
            continue
    return total


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(process.pid):
            return
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _parse_args(argv: list[str]) -> tuple[float | None, list[str]]:
    args = list(argv)
    timeout = None
    if args[:1] == ["--timeout"]:
        if len(args) < 2:
            raise ValueError("--timeout requires a positive number of seconds")
        try:
            timeout = float(args[1])
        except ValueError as exc:
            raise ValueError("--timeout requires a positive number of seconds") from exc
        if timeout <= 0:
            raise ValueError("--timeout requires a positive number of seconds")
        args = args[2:]
    if args[:1] != ["--"]:
        raise ValueError("expected -- before the command")
    command = args[1:]
    if not command:
        raise ValueError("no command supplied")
    return timeout, command


def main(argv: list[str] | None = None) -> int:
    try:
        timeout, command = _parse_args(sys.argv[1:] if argv is None else argv)
    except ValueError as exc:
        print("local_probe.sh: %s" % exc, file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    if MEMORY_LIMIT_BYTES is None:
        print(
            "local_probe.sh: RESEARCH_EVAL_MEMORY_LIMIT is not configured; "
            "refusing to run an unbounded local probe. Use the official submit path.",
            file=sys.stderr,
        )
        return 2

    def _set_memory_limit() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))

    try:
        process = subprocess.Popen(
            command,
            start_new_session=True,
            preexec_fn=_set_memory_limit,
        )
    except OSError as exc:
        print("local_probe.sh: failed to start command: %s" % exc, file=sys.stderr)
        return 127

    started_at = time.monotonic()
    while process.poll() is None:
        rss_bytes = _process_group_rss_bytes(process.pid)
        if rss_bytes > MEMORY_LIMIT_BYTES:
            print(
                "local_probe.sh: memory limit exceeded: rss=%d bytes limit=%s"
                % (rss_bytes, MEMORY_LIMIT),
                file=sys.stderr,
            )
            _terminate_process_group(process)
            return 137
        if timeout is not None and time.monotonic() - started_at >= timeout:
            print(
                "local_probe.sh: timeout exceeded after %.3f seconds" % timeout,
                file=sys.stderr,
            )
            _terminate_process_group(process)
            return 124
        time.sleep(POLL_SECONDS)
    return_code = int(process.returncode or 0)
    return 128 - return_code if return_code < 0 else return_code


if __name__ == "__main__":
    sys.exit(main())
'''
    replacements = {
        "__MEMORY_LIMIT__": repr(memory_limit),
        "__MEMORY_LIMIT_BYTES__": repr(memory_limit_bytes),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def _local_probe_script_wrapper_source(python_executable: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
exec "{python_executable}" "$SCRIPT_DIR/{SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME}/{LOCAL_PROBE_SNAPSHOT_FILENAME}" "$@"
"""


def detect_station_python_executable(env: Optional[Dict[str, str]] = None) -> str:
    env = dict(env or os.environ)
    from .evaluation_helpers import resolve_conda_env

    executable = resolve_conda_env(constants.RESEARCH_EVAL_PYTHON_CONDA_ENV, env)
    if executable:
        return executable
    return "python3"


def ensure_submit_script(paths: Optional[ResearchRuntimePaths] = None):
    if paths is None:
        paths = build_runtime_paths(constants)
    python_executable = detect_station_python_executable()
    snapshot_dir = os.path.join(paths.research_root, SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME)
    file_io_utils.ensure_dir_exists(snapshot_dir)
    snapshot_path = os.path.join(snapshot_dir, SUBMIT_EVAL_CLI_SNAPSHOT_FILENAME)
    snapshot_content = _submit_eval_snapshot_source()
    current_snapshot = file_io_utils.load_text(snapshot_path) if file_io_utils.file_exists(snapshot_path) else None
    if current_snapshot != snapshot_content:
        file_io_utils.save_text(snapshot_content, snapshot_path)
    script_content = _submit_script_wrapper_source(python_executable)
    current_content = file_io_utils.load_text(paths.submit_script_path) if file_io_utils.file_exists(paths.submit_script_path) else None
    if current_content == script_content:
        return
    file_io_utils.save_text(script_content, paths.submit_script_path)
    current_mode = os.stat(paths.submit_script_path).st_mode
    os.chmod(paths.submit_script_path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _audit_snapshot_source() -> str:
    source_path = os.path.join(os.path.dirname(__file__), "submit_audit_cli.py")
    return file_io_utils.load_text(source_path)


def _audit_script_wrapper_source(python_executable: str) -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export STATION_RESEARCH_ROOT="$SCRIPT_DIR"
exec "{python_executable}" "$SCRIPT_DIR/{SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME}/{SUBMIT_AUDIT_CLI_SNAPSHOT_FILENAME}" "$@"
'''


def ensure_audit_script(paths: Optional[ResearchRuntimePaths] = None):
    """Install the small, artifact-only auditor verdict command."""
    if paths is None:
        paths = build_runtime_paths(constants)
    python_executable = detect_station_python_executable()
    snapshot_dir = os.path.join(paths.research_root, SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME)
    file_io_utils.ensure_dir_exists(snapshot_dir)
    snapshot_path = os.path.join(snapshot_dir, SUBMIT_AUDIT_CLI_SNAPSHOT_FILENAME)
    snapshot_content = _audit_snapshot_source()
    current_snapshot = file_io_utils.load_text(snapshot_path) if file_io_utils.file_exists(snapshot_path) else None
    if current_snapshot != snapshot_content:
        file_io_utils.save_text(snapshot_content, snapshot_path)
    script_path = os.path.join(paths.research_root, getattr(constants, "RESEARCH_AUDIT_SCRIPT_FILENAME", "submit_audit.sh"))
    content = _audit_script_wrapper_source(python_executable)
    current = file_io_utils.load_text(script_path) if file_io_utils.file_exists(script_path) else None
    if current != content:
        file_io_utils.save_text(content, script_path)
    mode = os.stat(script_path).st_mode
    os.chmod(script_path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def ensure_eval_tool_script(paths: Optional[ResearchRuntimePaths] = None):
    if paths is None:
        paths = build_runtime_paths(constants)
    python_executable = detect_station_python_executable()
    snapshot_dir = os.path.join(paths.research_root, SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME)
    file_io_utils.ensure_dir_exists(snapshot_dir)
    snapshot_path = os.path.join(snapshot_dir, EVAL_TOOL_CLI_SNAPSHOT_FILENAME)
    snapshot_content = _eval_tool_snapshot_source()
    current_snapshot = file_io_utils.load_text(snapshot_path) if file_io_utils.file_exists(snapshot_path) else None
    if current_snapshot != snapshot_content:
        file_io_utils.save_text(snapshot_content, snapshot_path)
    script_content = _eval_tool_script_wrapper_source(python_executable)
    current_content = file_io_utils.load_text(paths.eval_tool_script_path) if file_io_utils.file_exists(paths.eval_tool_script_path) else None
    if current_content == script_content:
        _remove_old_preview_eval_tool(paths, snapshot_dir)
        return
    file_io_utils.save_text(script_content, paths.eval_tool_script_path)
    current_mode = os.stat(paths.eval_tool_script_path).st_mode
    os.chmod(paths.eval_tool_script_path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    _remove_old_preview_eval_tool(paths, snapshot_dir)


def ensure_local_probe_script(
    paths: Optional[ResearchRuntimePaths] = None,
    consts_module=constants,
):
    if paths is None:
        paths = build_runtime_paths(consts_module)
    python_executable = detect_station_python_executable()
    snapshot_dir = os.path.join(paths.research_root, SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME)
    file_io_utils.ensure_dir_exists(snapshot_dir)
    snapshot_path = os.path.join(snapshot_dir, LOCAL_PROBE_SNAPSHOT_FILENAME)
    snapshot_content = _local_probe_snapshot_source(consts_module.RESEARCH_EVAL_MEMORY_LIMIT)
    current_snapshot = file_io_utils.load_text(snapshot_path) if file_io_utils.file_exists(snapshot_path) else None
    if current_snapshot != snapshot_content:
        file_io_utils.save_text(snapshot_content, snapshot_path)
    script_content = _local_probe_script_wrapper_source(python_executable)
    current_content = (
        file_io_utils.load_text(paths.local_probe_script_path)
        if file_io_utils.file_exists(paths.local_probe_script_path)
        else None
    )
    if current_content != script_content:
        file_io_utils.save_text(script_content, paths.local_probe_script_path)
    current_mode = os.stat(paths.local_probe_script_path).st_mode
    os.chmod(paths.local_probe_script_path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _remove_old_preview_eval_tool(paths: ResearchRuntimePaths, snapshot_dir: str):
    old_script_path = os.path.join(paths.research_root, "preview_eval.sh")
    old_snapshot_path = os.path.join(snapshot_dir, OLD_PREVIEW_EVAL_CLI_SNAPSHOT_FILENAME)
    for path in (old_script_path, old_snapshot_path):
        if file_io_utils.file_exists(path):
            file_io_utils.delete_file(path)


def load_yaml_file(file_path: str) -> Optional[Dict[str, Any]]:
    if not file_io_utils.file_exists(file_path):
        return None
    data = file_io_utils.load_yaml(file_path)
    return data if isinstance(data, dict) else None


def load_baseline_definitions(consts_module=constants) -> list[Dict[str, Any]]:
    paths = build_runtime_paths(consts_module)
    if file_io_utils.file_exists(paths.baseline_path):
        try:
            with open(paths.baseline_path, "r", encoding="utf-8") as handle:
                docs = [doc for doc in yaml.safe_load_all(handle) if isinstance(doc, dict)]
            return docs
        except Exception:
            return []
    return []
