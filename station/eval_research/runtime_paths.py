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

from station import constants
from station import file_io_utils


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
EVAL_TOOL_CLI_SNAPSHOT_FILENAME = "eval_tool_cli_snapshot.py"
OLD_PREVIEW_EVAL_CLI_SNAPSHOT_FILENAME = "preview_eval_cli_snapshot.py"
SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME = "_internal"


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
    internal_root: str
    evaluations_dir: str
    evaluators_dir: str
    run_requests_dir: str
    coder_sessions_dir: str
    task_spec_path: str
    baseline_path: str
    submit_script_path: str
    eval_tool_script_path: str
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


def _migrate_storage_to_base_path(local_storage_path: str) -> str:
    if os.path.islink(local_storage_path):
        return os.path.realpath(local_storage_path)

    storage_uuid = str(uuid.uuid4())
    shared_base_path = os.path.join(constants.RESEARCH_STORAGE_BASE_PATH, storage_uuid)
    print(f"Research Center: Starting storage migration to: {shared_base_path}")
    os.makedirs(shared_base_path, exist_ok=True)

    if os.path.exists(local_storage_path) and os.path.isdir(local_storage_path):
        for item in os.listdir(local_storage_path):
            src = os.path.join(local_storage_path, item)
            dst = os.path.join(shared_base_path, item)
            if os.path.exists(dst):
                print(f"Research Center: Skipping existing shared storage item {item}")
                continue
            try:
                if os.path.isdir(src) and item == constants.RESEARCH_STORAGE_SYSTEM_DIR:
                    _set_writable_tree(src)
                shutil.move(src, dst)
                if item == constants.RESEARCH_STORAGE_SYSTEM_DIR:
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
        internal_root=os.path.join(research_root, consts_module.RESEARCH_INTERNAL_DIR),
        evaluations_dir=os.path.join(research_root, consts_module.RESEARCH_EVALUATIONS_SUBDIR_NAME),
        evaluators_dir=os.path.join(research_root, "evaluators"),
        run_requests_dir=os.path.join(research_root, consts_module.RESEARCH_RUN_REQUESTS_SUBDIR_NAME),
        coder_sessions_dir=os.path.join(research_root, consts_module.RESEARCH_CODER_SESSIONS_SUBDIR_NAME),
        task_spec_path=os.path.join(research_root, consts_module.RESEARCH_TASK_SPEC_FILENAME),
        baseline_path=os.path.join(research_root, consts_module.RESEARCH_BASELINE_FILENAME),
        submit_script_path=os.path.join(research_root, "submit_eval.sh"),
        eval_tool_script_path=os.path.join(research_root, "eval_tool.sh"),
        evaluation_index_path=os.path.join(research_root, consts_module.RESEARCH_EVALUATIONS_SUBDIR_NAME, consts_module.RESEARCH_EVALUATION_INDEX_FILENAME),
    )


def _ensure_dir_list(dir_paths: list[str]):
    for dir_path in dir_paths:
        file_io_utils.ensure_dir_exists(dir_path)


def ensure_lineage_storage(paths: ResearchRuntimePaths, lineage_name: str) -> str:
    lineage_name = (lineage_name or "unknown").lower()
    physical_path = os.path.join(paths.lineages_root, lineage_name)
    alias_path = os.path.join(paths.storage_real_root, lineage_name)

    file_io_utils.ensure_dir_exists(physical_path)
    data_path = os.path.join(physical_path, "data")
    file_io_utils.ensure_dir_exists(data_path)

    if os.path.lexists(alias_path):
        if os.path.islink(alias_path):
            try:
                target = os.path.realpath(alias_path)
                if target == os.path.realpath(physical_path):
                    return physical_path
            except OSError:
                pass
        if os.path.isdir(alias_path) and os.path.realpath(alias_path) == os.path.realpath(physical_path):
            return physical_path
        # Reserved directories should never be replaced.
        return physical_path

    try:
        relative_target = os.path.relpath(physical_path, os.path.dirname(alias_path))
        os.symlink(relative_target, alias_path)
    except FileExistsError:
        pass
    except OSError as exc:
        print(f"Research Center: Warning - Could not create lineage alias {alias_path}: {exc}")

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
    if consts_module.RESEARCH_STORAGE_BASE_PATH:
        storage_real_root = _migrate_storage_to_base_path(local_storage_path)

    file_io_utils.ensure_dir_exists(storage_real_root)
    paths = build_runtime_paths(consts_module)

    _ensure_dir_list([
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
    ])

    system_storage_writable = _set_writable_tree(paths.system_storage)
    sync_lineage_aliases(paths)
    ensure_evaluator_symlinks(paths)
    system_storage_read_only = _set_read_only_tree(paths.system_storage)
    if not system_storage_writable or not system_storage_read_only:
        print(
            "Research Center: Warning - Could not fully update permissions for "
            f"{paths.system_storage}; continuing with existing filesystem permissions."
        )
    ensure_submit_script(paths)
    ensure_eval_tool_script(paths)
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

    legacy_link = os.path.join(paths.system_storage, "task_1_evaluator.py")
    if os.path.lexists(legacy_link):
        try:
            if os.path.isdir(legacy_link) and not os.path.islink(legacy_link):
                pass
            else:
                os.unlink(legacy_link)
        except OSError:
            pass

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


def _attempt_footer(primary_score: Any, secondary_metrics: Any, safe_to_resubmit: bool) -> list[str]:
    return [
        "ATTEMPT_COMPLETE",
        "The submission has run to completion. You may now submit the next attempt or write the final Coder Report.",
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


def submit_eval(eval_id: str) -> int:
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
    if not eval_data:
        _write_status(
            research_root,
            eval_id,
            None,
            [f"Submission rejected: evaluation {eval_id} not found."]
            + _attempt_footer(RESEARCH_SCORE_NA, "{}", False),
        )
        return 1

    if not submission_path.exists():
        _write_status(
            research_root,
            eval_id,
            eval_data,
            [f"Submission rejected: submission file not found at {submission_path}."]
            + _attempt_footer(RESEARCH_SCORE_NA, "{}", False),
        )
        return 1

    submission_text = submission_path.read_text(encoding="utf-8")
    if not submission_text.strip():
        _write_status(
            research_root,
            eval_id,
            eval_data,
            [f"Submission rejected: submission file {submission_path} is empty."]
            + _attempt_footer(RESEARCH_SCORE_NA, "{}", False),
        )
        return 1

    if eval_data.get("status") in TERMINAL_STATUSES:
        _write_status(
            research_root,
            eval_id,
            eval_data,
            [f"Submission rejected: evaluation {eval_id} is already in terminal status '{eval_data.get('status')}'."]
            + _attempt_footer(RESEARCH_SCORE_NA, "{}", False),
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
            + _attempt_footer(RESEARCH_SCORE_NA, "{}", False),
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
    attempt_number = eval_manager.register_attempt(eval_id, str(submission_path))
    if attempt_number is None:
        _write_status(
            research_root,
            eval_id,
            eval_data,
            [f"Submission rejected: could not register attempt for evaluation {eval_id}."]
            + _attempt_footer(RESEARCH_SCORE_NA, "{}", False),
        )
        return 1

    _write_status(research_root, eval_id, eval_data, _attempt_queue_banner(eval_id, int(attempt_number)))
    _atomic_write_yaml(
        run_requests_dir / f"{eval_id}_attempt_{attempt_number}.yaml",
        {
            "eval_id": eval_id,
            "attempt": int(attempt_number),
            "created_timestamp": time.time(),
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Queue a Research Center evaluation attempt.")
    parser.add_argument("eval_id", help="Evaluation ID to submit")
    args = parser.parse_args(argv)
    return submit_eval(args.eval_id)


if __name__ == "__main__":
    sys.exit(main())
'''
    replacements = {
        "__RESEARCH_STORAGE_DIR__": repr(constants.RESEARCH_STORAGE_DIR),
        "__RESEARCH_EVALUATIONS_SUBDIR_NAME__": repr(constants.RESEARCH_EVALUATIONS_SUBDIR_NAME),
        "__RESEARCH_RUN_REQUESTS_SUBDIR_NAME__": repr(constants.RESEARCH_RUN_REQUESTS_SUBDIR_NAME),
        "__RESEARCH_SCORE_NA__": repr(constants.RESEARCH_SCORE_NA),
        "__RESEARCH_CODER_MAX_ATTEMPTS__": repr(constants.RESEARCH_CODER_MAX_ATTEMPTS),
        "__STATION_REPO_ROOT__": repr(str(Path(__file__).resolve().parents[2])),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def _submit_script_wrapper_source(python_executable: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
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
RESEARCH_CODER_SESSIONS_SUBDIR_NAME = __RESEARCH_CODER_SESSIONS_SUBDIR_NAME__
EVALUATION_DETAILS_KEY = __EVALUATION_DETAILS_KEY__
EVALUATION_ID_KEY = __EVALUATION_ID_KEY__
EVALUATION_TITLE_KEY = __EVALUATION_TITLE_KEY__
EVALUATION_ABSTRACT_KEY = __EVALUATION_ABSTRACT_KEY__
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


def _session_prompt_path(research_root: Path, eval_id: str, eval_data: Dict[str, Any]) -> tuple[Optional[Path], str]:
    coder = eval_data.get("coder") if isinstance(eval_data.get("coder"), dict) else {}
    raw_session_id = str(coder.get("session_id") or "").strip()
    sessions_dir = research_root / RESEARCH_CODER_SESSIONS_SUBDIR_NAME
    if raw_session_id:
        session_id = Path(raw_session_id).name
        return sessions_dir / session_id / "prompt.txt", session_id

    candidates = []
    if sessions_dir.exists():
        for path in sessions_dir.iterdir():
            if path.is_dir() and f"_{eval_id}_spawn_" in path.name:
                candidates.append(path)
    if not candidates:
        return None, ""
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] / "prompt.txt", candidates[0].name


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
    try:
        regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    except re.error as exc:
        print(f"Invalid regex: {exc}", file=sys.stderr)
        return 2

    try:
        total_matches, shown = _search_abstracts_from_index(research_root, pattern, limit)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("# Research Evaluation Abstract Search")
    print(f"\nRegex: `{pattern}`")
    print("Scope: evaluation abstracts only")
    print(f"Matches: {total_matches}")
    if limit >= 0 and total_matches > len(shown):
        print(f"Showing: first {len(shown)} newest matches")
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

    prompt_path, session_id = _session_prompt_path(research_root, eval_id, eval_data)
    report_path = _artifact_abs_path(research_root, eval_id, "report", eval_data)
    abstract = str(eval_data.get("abstract") or "").strip()
    instruction = str(eval_data.get("instruction") or "").strip()
    prompt_text = _read_text(prompt_path) if prompt_path else ""
    report_text = _read_text(report_path)

    print(f"# Research Evaluation Preview #{eval_id}")
    print("\nThis preview intentionally includes the evaluation instruction, coder prompt, and Coder Report. It does not print raw submission code, stdout, or stderr logs.")
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
    print(f"- Coder session: {session_id or '(none recorded)'}")
    print(f"- Coder prompt: {_display_path(research_root, prompt_path)}")
    print(f"- Coder Report: {_display_path(research_root, report_path)}")

    _print_section("Abstract", abstract, "_No abstract recorded._")
    _print_section("Agent Instruction", instruction, "_No instruction recorded._")
    if prompt_path is None:
        _print_section(
            "Coder Prompt",
            "",
            "_No coder session is recorded. This may be a direct/system-baseline evaluation or a queued evaluation that has not launched._",
        )
    else:
        _print_section(
            "Coder Prompt",
            prompt_text,
            f"_Prompt file not found or empty at `{_display_path(research_root, prompt_path)}`._",
        )
    _print_section(
        "Coder Report",
        report_text,
        f"_No Coder Report found at `{_display_path(research_root, report_path)}` yet._",
    )

    print("\n## Follow-Up Paths\n")
    print(f"- Evaluation YAML: `{_display_path(research_root, eval_path)}`")
    if prompt_path is not None:
        print(f"- Coder prompt: `{_display_path(research_root, prompt_path)}`")
    print(f"- Coder Report: `{_display_path(research_root, report_path)}`")
    print(f"- Raw submission, stdout, and stderr are under `storage/submission/`, `storage/stdout/`, and `storage/stderr/`; inspect them only when this preview is insufficient for the requested technical claim.")
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
        default=50,
        help="Maximum number of newest matches to print. Default: 50.",
    )

    preview_parser = subparsers.add_parser(
        "preview",
        help="Preview metadata, abstract, instruction, coder prompt, and Coder Report for one evaluation.",
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
        "__RESEARCH_CODER_SESSIONS_SUBDIR_NAME__": repr(constants.RESEARCH_CODER_SESSIONS_SUBDIR_NAME),
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
exec "{python_executable}" "$SCRIPT_DIR/{SUBMIT_EVAL_CLI_SNAPSHOT_DIRNAME}/{EVAL_TOOL_CLI_SNAPSHOT_FILENAME}" "$@"
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
