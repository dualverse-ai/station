#!/usr/bin/env python3
"""
Requeue Research Center evaluations at the instruction-prompt level.

Usage:
    python scripts/restart_eval.py --restart-stuck
    python scripts/restart_eval.py --shutdown-requeue-active
    python scripts/restart_eval.py 3-5
"""

import argparse
import os
import sys
from typing import List, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from station.eval_research import requeue_instruction_evaluations, restart_stuck_evaluations


def parse_eval_ids(eval_spec: str) -> List[str]:
    eval_ids: Set[int] = set()
    for part in eval_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            try:
                start = int(start_raw.strip())
                end = int(end_raw.strip())
            except ValueError:
                print(f"Warning: Invalid range '{part}', skipping")
                continue
            if end < start:
                start, end = end, start
            for value in range(start, end + 1):
                eval_ids.add(value)
            continue
        try:
            eval_ids.add(int(part))
        except ValueError:
            print(f"Warning: Invalid ID '{part}', skipping")
    return [str(value) for value in sorted(eval_ids)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Requeue Research Center evaluations")
    parser.add_argument("eval_ids", nargs="?", default=None, help='Evaluation IDs to requeue, e.g. "3-5" or "3,4,9"')
    parser.add_argument("--restart-stuck", action="store_true", help="Requeue all active instruction-level evaluations after a restart")
    parser.add_argument(
        "--shutdown-requeue-active",
        action="store_true",
        help="Terminate matching active coder processes for this station and requeue active instruction prompts",
    )
    args = parser.parse_args()

    if not args.eval_ids and not args.restart_stuck and not args.shutdown_requeue_active:
        args.restart_stuck = True
        print("No mode specified. Defaulting to --restart-stuck.")

    if args.eval_ids:
        eval_ids = parse_eval_ids(args.eval_ids)
        if not eval_ids:
            print("No valid evaluation IDs provided.")
            return 1
        count = requeue_instruction_evaluations(
            eval_ids=eval_ids,
            reason="Manually requeued via scripts/restart_eval.py.",
            kill_running_coders=True,
            force_reopen_terminal=True,
        )
        print(f"Requeued {count} evaluation(s): {', '.join(eval_ids)}")
        return 0

    if args.shutdown_requeue_active:
        count = requeue_instruction_evaluations(
            reason="Recovered during station shutdown: instruction prompt requeued.",
            kill_running_coders=True,
        )
        print(f"Shutdown requeued {count} active instruction-level evaluation(s).")
        return 0

    count = restart_stuck_evaluations()
    print(f"Restart recovery requeued {count} active instruction-level evaluation(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
