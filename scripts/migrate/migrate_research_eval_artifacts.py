#!/usr/bin/env python3
"""Move Research Center evaluation blobs out of YAML metadata files.

Run this while the station is stopped. By default the script writes missing
artifacts, creates YAML backups, and strips inline blob fields from evaluation
YAML. Use --dry-run to inspect planned changes without writing.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from station import constants
from station import file_io_utils
from station.eval_research.evaluation_manager import EvaluationManager
from station.eval_research.runtime_paths import build_runtime_paths


BLOB_FIELDS = ("submission_snapshot", "stdout", "stdout_visible", "stderr", "coder_report")
ARTIFACT_SPECS = {
    "submission": ("submission", ".py", "submission_snapshot"),
    "stdout": ("stdout", ".log", "stdout"),
    "stderr": ("stderr", ".log", "stderr"),
    "report": ("report", ".md", "coder_report"),
}


def _artifact_rel_path(eval_id: str, key: str) -> str:
    subdir, extension, _ = ARTIFACT_SPECS[key]
    return os.path.join(constants.RESEARCH_STORAGE_DIR, subdir, f"{eval_id}{extension}")


def _artifact_abs_path(research_root: Path, eval_id: str, key: str) -> Path:
    return research_root / _artifact_rel_path(eval_id, key)


def needs_migration(*, data_root: str) -> bool:
    constants.BASE_STATION_DATA_PATH = data_root
    paths = build_runtime_paths(constants)
    evaluations_dir = Path(paths.evaluations_dir)
    if not evaluations_dir.is_dir():
        return False

    return EvaluationManager(str(evaluations_dir), preload=False).needs_artifact_migration()


def _write_artifact_if_needed(path: Path, content: str, *, apply: bool) -> Tuple[bool, bool, bool]:
    if path.exists():
        existing = file_io_utils.load_text(str(path)) or ""
        if existing != content and existing.rstrip("\n") != content.rstrip("\n"):
            return False, True, True
        return False, True, False

    if not content:
        return False, False, False

    if apply:
        file_io_utils.save_text(content, str(path))
    return True, False, False


def _strip_blob_fields(record: Dict[str, Any]) -> int:
    removed = 0
    for key in BLOB_FIELDS:
        value = record.pop(key, None)
        if isinstance(value, str):
            removed += len(value)
    return removed


def _latest_attempt_with_content(data: Dict[str, Any], field: str) -> Dict[str, Any]:
    attempts = [
        attempt
        for attempt in data.get("attempts") or []
        if isinstance(attempt, dict) and field in attempt and str(attempt.get(field) or "")
    ]
    if not attempts:
        return {}

    def key(attempt: Dict[str, Any]) -> int:
        try:
            return int(attempt.get("attempt", 0) or 0)
        except (TypeError, ValueError):
            return 0

    return max(attempts, key=key)


def _write_selected_artifact(
    artifact_path: Path,
    content: str,
    *,
    apply: bool,
    wrote_count: int,
    existing_count: int,
    mismatch_count: int,
    mismatch_paths: list[str],
) -> Tuple[int, int, int]:
    wrote, existed, mismatched = _write_artifact_if_needed(artifact_path, content, apply=apply)
    if mismatched and len(mismatch_paths) < 20:
        mismatch_paths.append(str(artifact_path))
    return wrote_count + int(wrote), existing_count + int(existed and not mismatched), mismatch_count + int(mismatched)


def migrate(*, apply: bool, data_root: str) -> int:
    constants.BASE_STATION_DATA_PATH = data_root
    paths = build_runtime_paths(constants)
    research_root = Path(paths.research_root)
    evaluations_dir = Path(paths.evaluations_dir)
    if not evaluations_dir.is_dir():
        raise SystemExit(f"Evaluations directory not found: {evaluations_dir}")

    eval_manager = EvaluationManager(str(evaluations_dir), preload=False)
    eval_ids = eval_manager.get_artifact_migration_eval_ids()
    all_eval_count = len(eval_manager.get_all_evaluation_ids())
    backup_dir = evaluations_dir / ".artifact_migration_backup" / time.strftime("%Y%m%d_%H%M%S")
    changed_files = 0
    stripped_bytes = 0
    would_write_artifacts = 0
    existing_artifacts = 0
    mismatched_artifacts = 0
    mismatch_paths: list[str] = []

    for eval_id in eval_ids:
        eval_path = evaluations_dir / f"{eval_id}{constants.RESEARCH_EVALUATION_FILE_EXTENSION}"
        data = eval_manager.get_evaluation(eval_id)
        if not data:
            continue

        final = data.get("final") if isinstance(data.get("final"), dict) else {}
        original_artifacts = dict(data.get("artifacts") or {})
        original_final_artifacts = dict(final.get("artifacts") or {}) if final else {}
        artifacts = dict(data.get("artifacts") or {})
        final_artifacts = dict(final.get("artifacts") or {}) if final else {}

        for key, (_, _, inline_field) in ARTIFACT_SPECS.items():
            rel_path = _artifact_rel_path(eval_id, key)
            artifacts[key] = rel_path
            artifact_path = _artifact_abs_path(research_root, eval_id, key)

            final_has_inline_field = bool(final and inline_field in final)
            final_inline_content = str(final.get(inline_field) or "") if final_has_inline_field else ""
            if final and final_has_inline_field:
                if final_inline_content or inline_field != "submission_snapshot":
                    final_artifacts[key] = rel_path
                    would_write_artifacts, existing_artifacts, mismatched_artifacts = _write_selected_artifact(
                        artifact_path,
                        final_inline_content,
                        apply=apply,
                        wrote_count=would_write_artifacts,
                        existing_count=existing_artifacts,
                        mismatch_count=mismatched_artifacts,
                        mismatch_paths=mismatch_paths,
                    )
                continue

            if final and key in final_artifacts:
                if artifact_path.exists():
                    existing_artifacts += 1
                continue

            attempt = _latest_attempt_with_content(data, inline_field)
            if attempt:
                would_write_artifacts, existing_artifacts, mismatched_artifacts = _write_selected_artifact(
                    artifact_path,
                    str(attempt.get(inline_field) or ""),
                    apply=apply,
                    wrote_count=would_write_artifacts,
                    existing_count=existing_artifacts,
                    mismatch_count=mismatched_artifacts,
                    mismatch_paths=mismatch_paths,
                )
            elif artifact_path.exists():
                existing_artifacts += 1

        removed = 0
        attempts = data.get("attempts") or []
        if isinstance(attempts, list):
            for attempt in attempts:
                if isinstance(attempt, dict):
                    removed += _strip_blob_fields(attempt)
        if final:
            removed += _strip_blob_fields(final)
            final["artifacts"] = final_artifacts
            data["final"] = final
        data["artifacts"] = artifacts
        yaml_changed = bool(removed) or artifacts != original_artifacts
        if final:
            yaml_changed = yaml_changed or final_artifacts != original_final_artifacts

        if removed:
            changed_files += 1
            stripped_bytes += removed
        if yaml_changed:
            if apply:
                file_io_utils.ensure_dir_exists(str(backup_dir))
                file_io_utils.save_text(eval_path.read_text(encoding="utf-8"), str(backup_dir / eval_path.name))
                replacement = copy.deepcopy(data)

                def replace_record(record: Dict[str, Any]):
                    record.clear()
                    record.update(copy.deepcopy(replacement))

                if eval_manager.update_evaluation(eval_id, replace_record) is None:
                    raise RuntimeError(f"Could not update evaluation through EvaluationManager: {eval_id}")

    mode = "APPLY" if apply else "DRY-RUN"
    print(f"{mode}: evaluations={all_eval_count}")
    print(f"{mode}: migration_candidates={len(eval_ids)}")
    print(f"{mode}: files_with_inline_blobs={changed_files}")
    print(f"{mode}: stripped_inline_bytes={stripped_bytes}")
    print(f"{mode}: artifacts_existing_and_matching={existing_artifacts}")
    print(f"{mode}: artifacts_existing_with_inline_mismatch={mismatched_artifacts}")
    print(f"{mode}: artifacts_to_write={would_write_artifacts}")
    for path in mismatch_paths:
        print(f"{mode}: kept_existing_artifact_after_inline_mismatch={path}")
    if apply:
        print(f"APPLY: backups={backup_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=constants.BASE_STATION_DATA_PATH)
    parser.add_argument("--apply", action="store_true", help="Deprecated no-op; apply is now the default.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect planned changes without modifying files.")
    parser.add_argument("--check", action="store_true", help="Print yes if any evaluation still needs migration.")
    args = parser.parse_args()
    if args.check:
        print("yes" if needs_migration(data_root=str(args.data_root)) else "no")
        return 0
    if args.dry_run:
        return migrate(apply=False, data_root=str(args.data_root))

    print("Preflight dry-run before applying migration...")
    preflight_result = migrate(apply=False, data_root=str(args.data_root))
    if preflight_result != 0:
        return preflight_result
    print("Preflight complete. Applying migration...")
    return migrate(apply=True, data_root=str(args.data_root))


if __name__ == "__main__":
    raise SystemExit(main())
