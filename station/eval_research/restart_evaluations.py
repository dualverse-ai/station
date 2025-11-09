# Copyright 2025 DualverseAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Restart evaluations by moving them from completed back to pending.
This module provides functionality to restart stuck or failed evaluations.
"""

import os
import json
import yaml
import shutil
import glob
from typing import List, Set, Dict, Any, Optional

from station import constants
from station import file_io_utils


def find_stuck_evaluations(evaluations_dir: str) -> Set[int]:
    """Scan for completed evaluations where notification.sent is false."""
    stuck_ids = set()

    # Check if evaluations directory exists
    if not os.path.exists(evaluations_dir):
        return stuck_ids

    for filename in os.listdir(evaluations_dir):
        if not filename.startswith('evaluation_') or not filename.endswith('.json'):
            continue

        eval_file = os.path.join(evaluations_dir, filename)
        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)

            # A stuck evaluation is one where the final notification has not been sent.
            notification_status = eval_data.get('notification', {})
            if notification_status.get('sent') is False:
                stuck_ids.add(int(eval_data['id']))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"RestartEvaluations: Warning - Could not parse or check {filename}: {e}")
            continue
    return stuck_ids


def extract_original_submission(eval_data: dict) -> dict:
    """Extract the minimal pending evaluation entry from completed evaluation."""
    # Get the original submission data
    original = eval_data.get('original_submission', {})
    content = original.get('content', '')

    # Create pending entry with required fields
    pending_entry = {
        'author': eval_data.get('author'),
        'id': eval_data.get('id'),
        'logs': '',  # Reset logs
        'research_task_id': eval_data.get('research_task_id'),
        'score': 'pending',  # Reset to pending
        'status': 'pending',  # Add status field
        'submitted_tick': eval_data.get('submitted_tick'),
        'title': eval_data.get('title'),
        'content': content
    }

    # Add optional fields if they exist
    if eval_data.get('no_debugger'):
        pending_entry['no_debugger'] = eval_data['no_debugger']
    if eval_data.get('cpu_only'):
        pending_entry['cpu_only'] = eval_data['cpu_only']

    # Preserve tags and abstract from original evaluation
    if eval_data.get('tags'):
        pending_entry['tags'] = eval_data['tags']
    if eval_data.get('abstract'):
        pending_entry['abstract'] = eval_data['abstract']

    return pending_entry


def clear_agent_notifications(eval_id: int, author: str) -> bool:
    """Clear notifications for a specific evaluation ID from the agent's pending notifications."""
    if author.lower() == "system":
        # Skip System author
        return True

    agent_file = os.path.join(constants.BASE_STATION_DATA_PATH, 'agents', f'{author}.yaml')

    if not os.path.exists(agent_file):
        print(f"RestartEvaluations: Warning - Agent file not found for {author}")
        return False

    try:
        # Read agent data
        agent_data = file_io_utils.load_yaml(agent_file)

        # Check for pending notifications
        notifications = agent_data.get('notifications_pending', [])
        if not notifications:
            return True

        # Filter out notifications for this evaluation
        filtered_notifications = []
        removed_count = 0

        for notif in notifications:
            # Check if this notification mentions the evaluation ID
            if f"(ID: {eval_id})" in notif:
                removed_count += 1
            else:
                filtered_notifications.append(notif)

        if removed_count > 0:
            # Update agent data with filtered notifications
            agent_data['notifications_pending'] = filtered_notifications

            # Write back to file
            file_io_utils.atomic_write_yaml(agent_file, agent_data)

            print(f"RestartEvaluations: Removed {removed_count} notification(s) for eval {eval_id} from {author}")

        return True

    except Exception as e:
        print(f"RestartEvaluations: Error clearing notifications for {author}: {e}")
        return False


def restart_stuck_evaluations(
    eval_ids: Optional[List[int]] = None,
    clean_claude: bool = True,
    keep_original: bool = False,
    clear_notifications: bool = True
) -> int:
    """
    Restart research evaluations by moving them from completed back to pending.

    Args:
        eval_ids: List of evaluation IDs to restart. If None, automatically finds stuck evaluations.
        clean_claude: Whether to clean Claude Code workspace and history directories
        keep_original: Whether to keep original evaluation files (default: remove them)
        clear_notifications: Whether to clear agent notifications for restarted evaluations

    Returns:
        Number of evaluations restarted
    """
    # Paths
    research_room_path = os.path.join(
        constants.BASE_STATION_DATA_PATH,
        constants.ROOMS_DIR_NAME,
        constants.SHORT_ROOM_NAME_RESEARCH
    )
    evaluations_dir = os.path.join(research_room_path, constants.RESEARCH_EVALUATIONS_SUBDIR_NAME)
    pending_file = os.path.join(research_room_path, constants.PENDING_RESEARCH_EVALUATIONS_FILENAME)
    claude_workspaces_dir = os.path.join(research_room_path, 'claude_workspaces')

    # Determine which evaluations to restart
    if eval_ids is None:
        # Auto-detect stuck evaluations
        stuck_ids = find_stuck_evaluations(evaluations_dir)
        if not stuck_ids:
            return 0
        eval_ids_set = stuck_ids
    else:
        eval_ids_set = set(eval_ids)

    if not eval_ids_set:
        return 0

    # Collect pending entries
    pending_entries = []

    for eval_id in sorted(eval_ids_set):
        eval_file = os.path.join(evaluations_dir, f'evaluation_{eval_id}.json')

        if not os.path.exists(eval_file):
            print(f"RestartEvaluations: Evaluation {eval_id} not found, skipping")
            continue

        try:
            # Read completed evaluation
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)

            # Extract original submission
            pending_entry = extract_original_submission(eval_data)
            pending_entries.append(pending_entry)

            # Clear notifications if requested
            if clear_notifications:
                clear_agent_notifications(eval_id, eval_data.get('author', ''))

        except Exception as e:
            print(f"RestartEvaluations: Error processing evaluation {eval_id}: {e}")
            continue

    if not pending_entries:
        return 0

    # Append to pending evaluations file
    try:
        with open(pending_file, 'a', encoding='utf-8') as f:
            for entry in pending_entries:
                f.write("---\n")
                yaml.dump(entry, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception as e:
        print(f"RestartEvaluations: Error writing to pending file: {e}")
        return 0

    # Clean Claude workspaces if requested
    if clean_claude:
        for eval_id in sorted(eval_ids_set):
            # Clean main workspace
            workspace_path = os.path.join(claude_workspaces_dir, f'eval_{eval_id}')
            if os.path.exists(workspace_path):
                try:
                    shutil.rmtree(workspace_path)
                except Exception as e:
                    print(f"RestartEvaluations: Error cleaning workspace {workspace_path}: {e}")

            # Clean history directories
            history_pattern = os.path.join(claude_workspaces_dir, 'history', f'eval_{eval_id}_*')
            history_dirs = glob.glob(history_pattern)
            for hist_dir in history_dirs:
                try:
                    shutil.rmtree(hist_dir)
                except Exception as e:
                    print(f"RestartEvaluations: Error cleaning history {hist_dir}: {e}")

    # Remove original evaluation files if requested
    if not keep_original:
        for eval_id in sorted(eval_ids_set):
            # Remove evaluation JSON file
            eval_file = os.path.join(evaluations_dir, f'evaluation_{eval_id}.json')
            if os.path.exists(eval_file):
                try:
                    os.remove(eval_file)
                except Exception as e:
                    print(f"RestartEvaluations: Error removing {eval_file}: {e}")

            # Remove lock file
            lock_file = f"{eval_file}.lock"
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except Exception as e:
                    print(f"RestartEvaluations: Error removing {lock_file}: {e}")

    return len(pending_entries)
