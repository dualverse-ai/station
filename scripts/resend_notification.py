#!/usr/bin/env python3
"""Resend current Research Center completion notifications to agents."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Set

station_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if station_path not in sys.path:
    sys.path.insert(0, station_path)

from station import constants
from station.agent import add_pending_notification_atomic
from station.eval_research.evaluation_manager import EvaluationManager


def parse_eval_ids(eval_spec: str) -> Set[str]:
    eval_ids: set[str] = set()
    for part in str(eval_spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start_num = int(start.strip())
            end_num = int(end.strip())
            for eval_id in range(start_num, end_num + 1):
                eval_ids.add(str(eval_id))
        else:
            eval_ids.add(str(int(part)))
    return eval_ids


def _default_evaluations_dir() -> str:
    return str(
        Path(constants.BASE_STATION_DATA_PATH)
        / constants.ROOMS_DIR_NAME
        / constants.SHORT_ROOM_NAME_RESEARCH
        / constants.RESEARCH_EVALUATIONS_SUBDIR_NAME
    )


def _sort_eval_ids(eval_ids: Set[str]) -> list[str]:
    return sorted(eval_ids, key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)))


def resend_notification(eval_manager: EvaluationManager, eval_id: str, dry_run: bool = False) -> str:
    eval_id = str(eval_id)
    eval_data = eval_manager.get_evaluation(eval_id)
    if not isinstance(eval_data, dict):
        return f"Evaluation {eval_id}: not found"

    author = str(eval_data.get("author") or "").strip()
    if not author:
        return f"Evaluation {eval_id}: no author recorded"
    if author.lower() == "system":
        return f"Evaluation {eval_id}: skipping System evaluation"

    if not eval_data.get("final"):
        return f"Evaluation {eval_id}: no final result recorded"

    message = eval_manager.get_notification_message(eval_id)
    if not message:
        return f"Evaluation {eval_id}: could not build current notification message"

    if dry_run:
        print(f"\n--- DRY RUN: Notification for {author} (Eval {eval_id}) ---")
        print(message[:500] + "..." if len(message) > 500 else message)
        print("--- END ---\n")
        return f"Evaluation {eval_id}: [DRY RUN] would send notification to {author}"

    if not add_pending_notification_atomic(author, message):
        return f"Evaluation {eval_id}: failed to add notification for {author}"
    return f"Evaluation {eval_id}: resent notification to {author}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_ids", help="Evaluation IDs to resend, supports ranges and comma lists")
    parser.add_argument("--dry-run", action="store_true", help="Show messages without adding notifications")
    parser.add_argument(
        "--dir",
        default=_default_evaluations_dir(),
        help="Directory containing current Research Center .yaml evaluation files",
    )
    args = parser.parse_args()

    try:
        eval_ids = _sort_eval_ids(parse_eval_ids(args.eval_ids))
    except ValueError as exc:
        print(f"Error parsing evaluation IDs: {exc}")
        return 1

    eval_manager = EvaluationManager(str(Path(args.dir)), preload=False)
    print(f"Resending notifications for evaluations: {eval_ids}")
    success_count = 0
    for eval_id in eval_ids:
        result = resend_notification(eval_manager, eval_id, dry_run=args.dry_run)
        print(result)
        if "resent notification" in result or "[DRY RUN]" in result or "skipping System" in result:
            success_count += 1

    print(f"\nProcessed {len(eval_ids)} evaluations")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(eval_ids) - success_count}")
    return 0 if success_count == len(eval_ids) else 1


if __name__ == "__main__":
    sys.exit(main())
