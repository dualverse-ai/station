#!/usr/bin/env python3
"""
Void evaluations by marking them as failed and sending termination notifications.

Usage:
    python scripts/void_eval.py 3-6,12 "Non-compliant training methods"
    python scripts/void_eval.py 15 "Use of prohibited libraries"
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Set

# Add station package to path for imports
station_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if station_path not in sys.path:
    sys.path.insert(0, station_path)

from station import constants
from station.agent import add_pending_notification_atomic


def parse_eval_ids(eval_spec: str) -> Set[str]:
    """Parse evaluation ID specification into a set of IDs."""
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
                    eval_ids.add(str(i))
            except ValueError:
                print(f"Warning: Invalid range '{part}', skipping")
        else:
            # Single ID
            try:
                eval_id = str(int(part))  # Validate it's a number
                eval_ids.add(eval_id)
            except ValueError:
                print(f"Warning: Invalid ID '{part}', skipping")
    
    return eval_ids


def void_evaluation(eval_id: str, reason: str = None, dry_run: bool = False) -> bool:
    """
    Void a single evaluation by updating JSON and sending notification.
    
    Args:
        eval_id: Evaluation ID to void
        reason: Optional reason for voiding
        dry_run: If True, show what would be done without making changes
        
    Returns:
        True if successful, False otherwise
    """
    # Build paths
    evaluations_dir = os.path.join(
        constants.BASE_STATION_DATA_PATH,
        constants.ROOMS_DIR_NAME,
        constants.SHORT_ROOM_NAME_RESEARCH,
        constants.RESEARCH_EVALUATIONS_SUBDIR_NAME
    )
    eval_file = os.path.join(evaluations_dir, f'evaluation_{eval_id}.json')
    
    if not os.path.exists(eval_file):
        print(f"✗ Evaluation {eval_id} not found: {eval_file}")
        return False
    
    try:
        # Read evaluation JSON
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        
        author = eval_data.get('author')
        title = eval_data.get('title', 'Untitled Submission')
        
        if not author:
            print(f"✗ No author found for evaluation {eval_id}")
            return False
        
        # Skip System author
        if author.lower() == "system":
            print(f"⚠ Skipping System evaluation {eval_id}")
            return True
        
        if dry_run:
            print(f"[DRY RUN] Would void evaluation {eval_id}: '{title}' by {author}")
            if reason:
                print(f"  Reason: {reason}")
            return True
        
        # Update evaluation JSON - mark as failed
        void_details = f'Manually terminated and voided.{f" Reason: {reason}." if reason else ""}'
        timestamp = time.time()
        
        # Check if this is original submission or has versions
        if 'versions' in eval_data and eval_data['versions']:
            # Has versions - update the latest version and current state
            current_state = eval_data.get('current_state', {})
            latest_version = current_state.get('latest_version', 'original')
            
            if latest_version == 'original':
                # Update original submission
                if 'original_submission' not in eval_data:
                    eval_data['original_submission'] = {}
                if 'evaluation_result' not in eval_data['original_submission']:
                    eval_data['original_submission']['evaluation_result'] = {}
                
                eval_data['original_submission']['evaluation_result'].update({
                    'success': False,
                    'score': 'n.a.',
                    'status': 'failed',
                    'error': void_details,
                    'logs': f"MANUALLY VOIDED: {void_details}",
                    'evaluation_details': void_details,
                    'evaluation_timestamp': timestamp,
                    'evaluation_datetime': datetime.fromtimestamp(timestamp).isoformat(),
                    'sort_key': [float('-inf')]
                })
            else:
                # Update the latest version
                if latest_version in eval_data['versions']:
                    if 'evaluation_result' not in eval_data['versions'][latest_version]:
                        eval_data['versions'][latest_version]['evaluation_result'] = {}
                    
                    eval_data['versions'][latest_version]['evaluation_result'].update({
                        'success': False,
                        'score': 'n.a.',
                        'status': 'failed',
                        'error': void_details,
                        'logs': f"MANUALLY VOIDED: {void_details}",
                        'evaluation_details': void_details,
                        'evaluation_timestamp': timestamp,
                        'evaluation_datetime': datetime.fromtimestamp(timestamp).isoformat(),
                        'sort_key': [float('-inf')]
                    })
            
            # Update current state
            eval_data['current_state'].update({
                'latest_score': 'n.a.',
                'latest_status': 'failed',
                'claude_active': False,
                'all_evaluations_complete': True
            })
        else:
            # Simple evaluation - update original_submission
            if 'original_submission' not in eval_data:
                eval_data['original_submission'] = {}
            if 'evaluation_result' not in eval_data['original_submission']:
                eval_data['original_submission']['evaluation_result'] = {}
                
            eval_data['original_submission']['evaluation_result'].update({
                'success': False,
                'score': 'n.a.',
                'status': 'failed',
                'error': void_details,
                'logs': f"MANUALLY VOIDED: {void_details}",
                'evaluation_details': void_details,
                'evaluation_timestamp': timestamp,
                'evaluation_datetime': datetime.fromtimestamp(timestamp).isoformat(),
                'sort_key': [float('-inf')]
            })
            
            # Update current state
            if 'current_state' not in eval_data:
                eval_data['current_state'] = {}
            eval_data['current_state'].update({
                'latest_version': 'original',
                'latest_score': 'n.a.',
                'latest_status': 'failed',
                'claude_active': False,
                'all_evaluations_complete': True
            })
        
        # Mark notification as sent since we're manually sending it
        reason_text = f" Reason: {reason}." if reason else ""
        notification_message = (
            f"Your research submission '{title}' (ID: {eval_id}) evaluation has been "
            f"manually terminated and voided.{reason_text}"
        )
        
        eval_data['notification'] = {
            'sent': True,
            'sent_timestamp': timestamp,
            'version_notified': eval_data.get('current_state', {}).get('latest_version', 'original'),
            'message': notification_message
        }
        
        # Write back to file
        with open(eval_file, 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        # Send system notification to agent
        notification_success = add_pending_notification_atomic(author, notification_message)
        if not notification_success:
            print(f"⚠ Warning: Failed to send notification to {author}")
        
        print(f"✓ Voided evaluation {eval_id}: '{title}' by {author}")
        if reason:
            print(f"  Reason: {reason}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing evaluation {eval_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Void research evaluations')
    parser.add_argument('eval_ids', help='Evaluation IDs to void (e.g., "3-5" or "3,4,5,9-12")')
    parser.add_argument('reason', nargs='?', default=None, help='Optional reason for voiding')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    # Parse evaluation IDs
    eval_ids = parse_eval_ids(args.eval_ids)
    
    if not eval_ids:
        print("No valid evaluation IDs provided")
        return 1
    
    print(f"Will void evaluations: {sorted(eval_ids, key=int)}")
    if args.reason:
        print(f"Reason: {args.reason}")
    
    # Process each evaluation
    success_count = 0
    for eval_id in sorted(eval_ids, key=int):
        if void_evaluation(eval_id, args.reason, args.dry_run):
            success_count += 1
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would void {success_count}/{len(eval_ids)} evaluations")
    else:
        print(f"\n✓ Successfully voided {success_count}/{len(eval_ids)} evaluations")
        if success_count > 0:
            print("\nNext steps:")
            print("1. Agents will receive termination notifications")
            print("2. Evaluation results are marked as 'n.a.' in the research counter")
    
    return 0 if success_count == len(eval_ids) else 1


if __name__ == '__main__':
    sys.exit(main())