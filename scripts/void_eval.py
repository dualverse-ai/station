#!/usr/bin/env python3
"""
Void Research Center evaluations by marking them failed and notifying authors.

Usage:
    python scripts/void_eval.py 3-6,12 "Non-compliant training methods"
    python scripts/void_eval.py 15 "Use of prohibited libraries"
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Set

station_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if station_path not in sys.path:
    sys.path.insert(0, station_path)

from station import constants
from station.agent import add_pending_notification_atomic
from station.eval_research.evaluation_manager import EvaluationManager


def parse_eval_ids(eval_spec: str) -> Set[str]:
    """Parse an evaluation ID/range specification into a set of IDs."""
    eval_ids = set()
    for part in eval_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            try:
                start_num = int(start.strip())
                end_num = int(end.strip())
            except ValueError:
                print(f"Warning: Invalid range '{part}', skipping")
                continue
            for value in range(start_num, end_num + 1):
                eval_ids.add(str(value))
            continue

        try:
            eval_ids.add(str(int(part)))
        except ValueError:
            print(f"Warning: Invalid ID '{part}', skipping")
    return eval_ids


def _default_evaluations_dir() -> str:
    return str(
        Path(constants.BASE_STATION_DATA_PATH)
        / constants.ROOMS_DIR_NAME
        / constants.SHORT_ROOM_NAME_RESEARCH
        / constants.RESEARCH_EVALUATIONS_SUBDIR_NAME
    )


def _latest_attempt(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    attempts = record.get("attempts") or []
    if not isinstance(attempts, list) or not attempts:
        return None
    latest = attempts[-1]
    return latest if isinstance(latest, dict) else None


def _void_final_record(record: dict[str, Any], eval_id: str, void_details: str) -> dict[str, Any]:
    latest = _latest_attempt(record) or {}
    existing_final = record.get("final") if isinstance(record.get("final"), dict) else {}
    artifacts = {}
    if isinstance(existing_final.get("artifacts"), dict):
        artifacts = copy.deepcopy(existing_final.get("artifacts"))
    elif isinstance(record.get("artifacts"), dict):
        artifacts = copy.deepcopy(record.get("artifacts"))

    final_record = {
        "status": "failed",
        "attempt": latest.get("attempt"),
        "primary_score": constants.RESEARCH_SCORE_NA,
        constants.EVALUATION_DETAILS_KEY: void_details,
        "sort_key": None,
        "error": void_details,
    }
    if artifacts:
        final_record["artifacts"] = artifacts
    return final_record


def _apply_void(record: dict[str, Any], eval_id: str, reason: Optional[str], notification_message: str, timestamp: float):
    void_details = f"Manually terminated and voided.{f' Reason: {reason}.' if reason else ''}"

    latest = _latest_attempt(record)
    if latest is not None and str(latest.get("status", "")).strip().lower() in {"queued", "running"}:
        latest["status"] = "failed"
        latest["completed_timestamp"] = latest.get("completed_timestamp") or timestamp
        latest["primary_score"] = constants.RESEARCH_SCORE_NA
        latest[constants.EVALUATION_DETAILS_KEY] = void_details
        latest["error"] = void_details
        latest["sort_key"] = None

    record["status"] = "failed"
    record["final"] = _void_final_record(record, eval_id, void_details)

    coder = record.setdefault("coder", {})
    if isinstance(coder, dict):
        coder["active"] = False
        coder["active_pid"] = None
        coder["status"] = "failed"
        coder["failure_category"] = "manual_void"
        coder["completed_timestamp"] = timestamp
        coder["last_error"] = void_details

    notification = record.setdefault("notification", {})
    if isinstance(notification, dict):
        notification.update(
            {
                "sent": True,
                "sent_timestamp": timestamp,
                "message": notification_message,
            }
        )


def void_evaluation(
    eval_manager: EvaluationManager,
    eval_id: str,
    reason: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """Void one current-schema evaluation through EvaluationManager."""
    eval_id = str(eval_id)
    eval_data = eval_manager.get_evaluation(eval_id)
    if not isinstance(eval_data, dict):
        print(f"Evaluation {eval_id} not found")
        return False

    author = str(eval_data.get("author") or "").strip()
    title = str(eval_data.get("title") or "Untitled Submission")
    if not author:
        print(f"Evaluation {eval_id}: no author found")
        return False
    if author.lower() == "system":
        print(f"Evaluation {eval_id}: skipping System evaluation")
        return True

    reason_text = f" Reason: {reason}." if reason else ""
    notification_message = (
        f"Your research submission '{title}' (ID: {eval_id}) evaluation has been "
        f"manually terminated and voided.{reason_text}"
    )

    if dry_run:
        print(f"[DRY RUN] Would void evaluation {eval_id}: '{title}' by {author}")
        if reason:
            print(f"  Reason: {reason}")
        return True

    timestamp = time.time()

    def mutator(record: dict[str, Any]):
        _apply_void(record, eval_id, reason, notification_message, timestamp)

    updated = eval_manager.update_evaluation(eval_id, mutator)
    if updated is None:
        print(f"Evaluation {eval_id}: failed to update through EvaluationManager")
        return False

    notification_success = add_pending_notification_atomic(author, notification_message)
    if not notification_success:
        print(f"Warning: failed to send notification to {author}")

    print(f"Voided evaluation {eval_id}: '{title}' by {author}")
    if reason:
        print(f"  Reason: {reason}")
    return True


def _sort_eval_ids(eval_ids: Set[str]) -> list[str]:
    return sorted(eval_ids, key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Void current Research Center evaluations")
    parser.add_argument("eval_ids", help='Evaluation IDs to void, e.g. "3-5" or "3,4,5,9-12"')
    parser.add_argument("reason", nargs="?", default=None, help="Optional reason for voiding")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument(
        "--dir",
        default=_default_evaluations_dir(),
        help="Directory containing current Research Center .yaml evaluation files",
    )
    args = parser.parse_args()

    eval_ids = parse_eval_ids(args.eval_ids)
    if not eval_ids:
        print("No valid evaluation IDs provided")
        return 1

    sorted_ids = _sort_eval_ids(eval_ids)
    print(f"Will void evaluations: {sorted_ids}")
    if args.reason:
        print(f"Reason: {args.reason}")

    eval_manager = EvaluationManager(str(Path(args.dir)), preload=False)
    success_count = 0
    for eval_id in sorted_ids:
        if void_evaluation(eval_manager, eval_id, args.reason, args.dry_run):
            success_count += 1

    if args.dry_run:
        print(f"\n[DRY RUN] Would void {success_count}/{len(eval_ids)} evaluations")
    else:
        print(f"\nSuccessfully voided {success_count}/{len(eval_ids)} evaluations")
        if success_count > 0:
            print("Agents will receive termination notifications.")
    return 0 if success_count == len(eval_ids) else 1


if __name__ == "__main__":
    sys.exit(main())
