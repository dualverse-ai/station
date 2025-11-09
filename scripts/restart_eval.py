#!/usr/bin/env python3
"""
Restart evaluations by moving them from completed back to pending.

Usage:
    python restart_eval.py 3-5        # Restart evaluations 3, 4, 5
    python restart_eval.py --restart-stuck # Restart all evaluations with unsent notifications
"""

import os
import sys
import argparse
from typing import Set

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from station.eval_research import restart_stuck_evaluations


def parse_eval_ids(eval_spec: str) -> Set[int]:
    """Parse evaluation ID specification into a set of integer IDs."""
    eval_ids = set()

    # Split by comma
    parts = eval_spec.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range like "3-5"
            start, end = part.split('-')
            try:
                start_num = int(start.strip())
                end_num = int(end.strip())
                for i in range(start_num, end_num + 1):
                    eval_ids.add(i)
            except ValueError:
                print(f"Warning: Invalid range '{part}', skipping")
        else:
            # Single ID
            try:
                eval_id = int(part)
                eval_ids.add(eval_id)
            except ValueError:
                print(f"Warning: Invalid ID '{part}', skipping")

    return eval_ids


def main():
    parser = argparse.ArgumentParser(description='Restart research evaluations')
    parser.add_argument('eval_ids', nargs='?', default=None, help='Evaluation IDs to restart (e.g., "3-5" or "3,4,5,9-12")')
    parser.add_argument('--restart-stuck', action='store_true', help='Automatically find and restart evaluations with unsent notifications')
    parser.add_argument('--no-clean-claude', action='store_true', help='Do NOT clean Claude workspace and history (default: clean)')
    parser.add_argument('--keep-original', action='store_true', help='Keep original evaluation files (default: remove)')
    parser.add_argument('--no-clear-notifications', action='store_true', help='Do NOT clear agent notifications (default: clear)')

    args = parser.parse_args()

    # If no arguments are provided, default to restarting stuck evaluations
    if not args.eval_ids and not args.restart_stuck:
        args.restart_stuck = True
        print("No specific evaluations provided. Defaulting to --restart-stuck.")

    # Determine evaluation IDs
    eval_ids_list = None
    if args.eval_ids:
        eval_ids_set = parse_eval_ids(args.eval_ids)
        if not eval_ids_set:
            print("No valid evaluation IDs provided")
            return 1
        eval_ids_list = sorted(eval_ids_set)
        print(f"Will restart evaluations: {eval_ids_list}")
    else:
        print("Scanning for stuck evaluations (notification.sent=false)...")

    # Call the station function
    count = restart_stuck_evaluations(
        eval_ids=eval_ids_list,
        clean_claude=not args.no_clean_claude,
        keep_original=args.keep_original,
        clear_notifications=not args.no_clear_notifications
    )

    if count == 0:
        if eval_ids_list:
            print("No evaluations were restarted")
        else:
            print("No stuck evaluations found.")
        return 0

    print(f"\n✓ Successfully restarted {count} evaluation(s)")
    print("\nNext steps:")
    print("1. The auto evaluator will pick up these evaluations in the next cycle")
    print("2. Monitor the logs to see them being processed")
    if not args.no_clear_notifications:
        print("3. Agent notifications have been cleared for restarted evaluations")

    return 0


if __name__ == '__main__':
    sys.exit(main())