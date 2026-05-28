#!/usr/bin/env python3
"""Set current Research Center evaluation scores to 'n.a.'."""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from station import constants
from station.eval_research.evaluation_manager import EvaluationManager


SCORE_KEYS = {"score", "latest_score", "primary_score"}


def _set_scores(obj: Any) -> int:
    """Recursively set score fields to the configured Research n.a. sentinel."""
    changes = 0
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in SCORE_KEYS and obj[key] != constants.RESEARCH_SCORE_NA:
                obj[key] = constants.RESEARCH_SCORE_NA
                changes += 1
            elif key == "sort_key" and obj[key] is not None:
                obj[key] = None
                changes += 1
            changes += _set_scores(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            changes += _set_scores(item)
    return changes


def _rewrite_score_line(text: str) -> str:
    return re.sub(r"(\*\*Score:\*\*\s*)(.+)", lambda match: f"{match.group(1)}{constants.RESEARCH_SCORE_NA}", text)


def process_evaluation(manager: EvaluationManager, eval_id: str, rewrite_messages: bool) -> tuple[bool, int]:
    data = manager.get_evaluation(str(eval_id))
    if not isinstance(data, dict):
        return False, 0

    updated_data = copy.deepcopy(data)
    changes = _set_scores(updated_data)

    if rewrite_messages:
        notif = updated_data.get("notification")
        if isinstance(notif, dict):
            msg = notif.get("message")
            if isinstance(msg, str):
                new_msg = _rewrite_score_line(msg)
                if new_msg != msg:
                    notif["message"] = new_msg
                    changes += 1

    if changes <= 0:
        return False, 0

    replacement = copy.deepcopy(updated_data)

    def replace_record(record: dict[str, Any]):
        record.clear()
        record.update(copy.deepcopy(replacement))

    if manager.update_evaluation(str(eval_id), replace_record) is None:
        raise RuntimeError(f"Could not update evaluation through EvaluationManager: {eval_id}")
    return True, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default=str(
            Path(constants.BASE_STATION_DATA_PATH)
            / constants.ROOMS_DIR_NAME
            / constants.SHORT_ROOM_NAME_RESEARCH
            / constants.RESEARCH_EVALUATIONS_SUBDIR_NAME
        ),
        help="Directory containing current Research Center .yaml evaluation files",
    )
    parser.add_argument(
        "--eval-ids",
        default=None,
        help="Optional comma/range list of evaluation IDs to update, e.g. '3-5,8'",
    )
    parser.add_argument(
        "--no-rewrite-messages",
        action="store_true",
        help="Do not rewrite '**Score:** ...' in notification messages",
    )
    args = parser.parse_args()

    manager = EvaluationManager(str(Path(args.dir)), preload=not bool(args.eval_ids))
    if args.eval_ids:
        eval_ids = _parse_eval_ids(args.eval_ids)
    else:
        eval_ids = manager.get_all_evaluation_ids()
    updated_files = 0
    total_changes = 0

    for eval_id in eval_ids:
        updated, changes = process_evaluation(manager, eval_id, rewrite_messages=not args.no_rewrite_messages)
        if updated:
            updated_files += 1
            total_changes += changes

    print(f"scanned={len(eval_ids)} updated={updated_files} field_changes={total_changes}")


def _parse_eval_ids(eval_spec: str) -> list[str]:
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
    return sorted(eval_ids, key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)))


if __name__ == "__main__":
    main()
