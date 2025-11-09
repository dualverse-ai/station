#!/usr/bin/env python3
"""
Resend notifications for research evaluations to agents.

Usage:
    python resend_notification.py 222
    python resend_notification.py 222,234,236
    python resend_notification.py 234-236
    python resend_notification.py 222,234-236,240
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add station to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from station import constants
from station import file_io_utils
from station.agent import Agent


def parse_eval_ids(eval_ids_str):
    """Parse evaluation ID string into list of IDs.
    
    Examples:
        '222' -> [222]
        '222,234,236' -> [222, 234, 236]
        '234-236' -> [234, 235, 236]
        '222,234-236,240' -> [222, 234, 235, 236, 240]
    """
    eval_ids = []
    
    # Split by comma
    parts = eval_ids_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range
            start, end = part.split('-')
            start = int(start.strip())
            end = int(end.strip())
            eval_ids.extend(range(start, end + 1))
        else:
            # Single ID
            eval_ids.append(int(part))
    
    # Remove duplicates and sort
    eval_ids = sorted(list(set(eval_ids)))
    
    return eval_ids


def load_evaluation(eval_id):
    """Load evaluation data from JSON file."""
    evaluations_dir = os.path.join(
        constants.BASE_STATION_DATA_PATH,
        constants.ROOMS_SUBDIR,
        constants.RESEARCH_SHORT_NAME,
        constants.EVALUATIONS_SUBDIR
    )
    
    eval_path = os.path.join(evaluations_dir, f"evaluation_{eval_id}.json")
    
    if not os.path.exists(eval_path):
        return None
    
    try:
        with open(eval_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading evaluation {eval_id}: {e}")
        return None


def format_notification_message(eval_data):
    """Format notification message based on evaluation data."""
    eval_id = eval_data.get("id")
    title = eval_data.get("title", "Unknown")
    author = eval_data.get("author", "Unknown")
    
    # Check notification status
    notification = eval_data.get("notification", {})
    if not notification.get("sent", False):
        return None, "Notification not sent yet"
    
    # Find the latest evaluation result
    latest_version = eval_data.get("current_state", {}).get("latest_version", "original")
    
    if latest_version == "original":
        eval_result = eval_data.get("original_submission", {}).get("evaluation_result", {})
    else:
        version_data = eval_data.get("versions", {}).get(latest_version, {})
        eval_result = version_data.get("evaluation_result", {})
    
    status = eval_result.get("status", "unknown")
    score = eval_result.get("score", "n.a.")
    success = eval_result.get("success", False)
    error = eval_result.get("error")
    logs = eval_result.get("logs", "")
    details = eval_result.get("evaluation_details", "")
    
    # Truncate logs if needed
    truncated_logs = logs
    if len(logs) > constants.RESEARCH_EVAL_LOG_MAX_CHARS:
        truncated_logs = logs[:constants.RESEARCH_EVAL_LOG_MAX_CHARS] + f"\n\n[... truncated after {constants.RESEARCH_EVAL_LOG_MAX_CHARS:,} characters]"
    
    # Escape backslashes
    truncated_logs = truncated_logs.replace('\\', '\\\\')
    
    if success:
        message = f"Your research submission '{title}' (ID: {eval_id}) has been evaluated.\n\n" \
                 f"**Score:** {score}\n" \
                 f"**Evaluation Details:** {details}\n\n" \
                 f"**Full Execution Log:**\n```\n{truncated_logs}\n```"
    else:
        # For import errors, provide more helpful guidance
        error_guidance = ""
        if error and ("Import failed:" in error or "IMPORT_ERROR:" in truncated_logs):
            error_guidance = "\n\n**Import Error Diagnosis:**\n"
            
            # Check for common import error patterns
            import re
            if "No module named" in error or "No module named" in truncated_logs:
                # Try to extract the missing module name
                match = re.search(r"No module named '([^']+)'", (error or "") + " " + truncated_logs)
                if match:
                    module_match = match.group(1)
                    error_guidance += f"- Missing module: `{module_match}`\n"
                    error_guidance += f"- Check that the file `{module_match}.py` exists in your storage\n"
                    error_guidance += f"- Verify the file path in your sys.path.append() statement\n"
            
            error_guidance += "\n**Common Solutions:**\n"
            error_guidance += "1. Ensure all imported .py files exist in your storage directories\n"
            error_guidance += "2. Use `/execute_action{storage list intended_path}` to verify your files\n"
            error_guidance += "3. Check for typos in module names and import statements\n"
        
        error_msg = error or details or "Unknown error"
        message = f"Your research submission '{title}' (ID: {eval_id}) evaluation failed.\n\n" \
                 f"**Error Summary:** {error_msg}\n" \
                 f"{error_guidance}\n" \
                 f"**Full Execution Log:**\n```\n{truncated_logs}\n```"
    
    return message, None


def resend_notification(eval_id, dry_run=False):
    """Resend notification for a single evaluation."""
    # Load evaluation data
    eval_data = load_evaluation(eval_id)
    
    if not eval_data:
        return f"Evaluation {eval_id} not found"
    
    author = eval_data.get("author", "Unknown")
    
    # Skip System author
    if author == "System":
        return f"Evaluation {eval_id}: Skipping System author (baseline evaluation)"
    
    # Format notification message
    message, error = format_notification_message(eval_data)
    
    if error:
        return f"Evaluation {eval_id}: {error}"
    
    if not message:
        return f"Evaluation {eval_id}: Could not format notification message"
    
    # Check if agent exists
    agent_file = os.path.join(
        constants.BASE_STATION_DATA_PATH,
        constants.AGENTS_SUBDIR,
        f"{author}.yaml"
    )
    
    if not os.path.exists(agent_file):
        return f"Evaluation {eval_id}: Agent '{author}' not found"
    
    if dry_run:
        print(f"\n--- DRY RUN: Notification for {author} (Eval {eval_id}) ---")
        print(message[:500] + "..." if len(message) > 500 else message)
        print("--- END ---\n")
        return f"Evaluation {eval_id}: [DRY RUN] Would send notification to {author}"
    
    # Load agent data and add notification
    try:
        agent_data = file_io_utils.load_yaml(agent_file)
        
        # Get pending notifications list
        pending_notifications = agent_data.get(constants.AGENT_PENDING_NOTIFICATIONS_KEY, [])
        if not isinstance(pending_notifications, list):
            pending_notifications = []
        
        # Add notification
        pending_notifications.append(message)
        agent_data[constants.AGENT_PENDING_NOTIFICATIONS_KEY] = pending_notifications
        
        # Save back
        file_io_utils.save_yaml(agent_data, agent_file)
        
        return f"Evaluation {eval_id}: Successfully resent notification to {author}"
        
    except Exception as e:
        return f"Evaluation {eval_id}: Failed to add notification for {author}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Resend notifications for research evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python resend_notification.py 222
    python resend_notification.py 222,234,236
    python resend_notification.py 234-236
    python resend_notification.py 222,234-236,240
    python resend_notification.py --dry-run 234-236
        """
    )
    
    parser.add_argument(
        "eval_ids",
        help="Evaluation IDs to resend notifications for (supports ranges and lists)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent without actually sending"
    )
    
    args = parser.parse_args()
    
    # Parse evaluation IDs
    try:
        eval_ids = parse_eval_ids(args.eval_ids)
    except ValueError as e:
        print(f"Error parsing evaluation IDs: {e}")
        sys.exit(1)
    
    print(f"Resending notifications for evaluations: {eval_ids}")
    if args.dry_run:
        print("DRY RUN MODE - No notifications will actually be sent\n")
    
    # Process each evaluation
    results = []
    for eval_id in eval_ids:
        result = resend_notification(eval_id, dry_run=args.dry_run)
        results.append(result)
        print(result)
    
    # Summary
    print(f"\nProcessed {len(eval_ids)} evaluations")
    success_count = sum(1 for r in results if "Successfully" in r or "[DRY RUN]" in r)
    print(f"Success: {success_count}, Failed/Skipped: {len(eval_ids) - success_count}")


if __name__ == "__main__":
    main()