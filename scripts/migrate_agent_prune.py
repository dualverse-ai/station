#!/usr/bin/env python3
"""
Script to update agent prune data from old format to new format.

Old format (dict): {3: "prune_response", 4: "prune_response"}
New format (list): [{"ticks": "3", "summary": ""}, {"ticks": "4", "summary": ""}]

Usage:
    python scripts/update_agent_prune_data.py [station_data_path]

If station_data_path is not provided, defaults to station_data/ in project root.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from station import constants
from station import file_io_utils


def convert_prune_data(raw_prune_data):
    """
    Convert pruned dialogue ticks data from old format to new format.

    Args:
        raw_prune_data: Either dict (old format) or list (new format)

    Returns:
        List of prune blocks in new format, or None if already in new format
    """
    if not raw_prune_data:
        return None

    # Already in new format (list of blocks)
    if isinstance(raw_prune_data, list):
        return None

    # Old format (dict mapping tick -> "prune_response")
    if isinstance(raw_prune_data, dict):
        normalized_blocks = []
        for tick in sorted(raw_prune_data.keys()):
            # Convert each tick to a block with empty summary (complete removal)
            normalized_blocks.append({
                constants.PRUNE_TICKS_KEY: str(tick),
                constants.PRUNE_SUMMARY_KEY: ""
            })
        return normalized_blocks

    # Unknown format
    print(f"Warning: Unknown prune data format: {type(raw_prune_data)}")
    return None


def update_agent_file(agent_file_path):
    """
    Update a single agent file to convert prune data to new format.

    Args:
        agent_file_path: Path to agent YAML file

    Returns:
        True if updated, False if no update needed or failed
    """
    try:
        # Load agent data
        agent_data = file_io_utils.load_yaml(agent_file_path)
        if not agent_data:
            print(f"  Skipping {agent_file_path} (empty or invalid)")
            return False

        # Check if pruned_dialogue_ticks exists
        if constants.AGENT_PRUNED_DIALOGUE_TICKS_KEY not in agent_data:
            return False

        raw_prune_data = agent_data[constants.AGENT_PRUNED_DIALOGUE_TICKS_KEY]

        # Convert if needed
        converted_data = convert_prune_data(raw_prune_data)
        if converted_data is None:
            # Already in new format or empty
            return False

        # Update agent data
        agent_data[constants.AGENT_PRUNED_DIALOGUE_TICKS_KEY] = converted_data

        # Save back to file
        file_io_utils.save_yaml(agent_data, agent_file_path)

        return True

    except Exception as e:
        print(f"  Error processing {agent_file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update agent prune data from old format to new format"
    )
    parser.add_argument(
        "station_data_path",
        nargs="?",
        default="station_data",
        help="Path to station_data directory (default: station_data/)"
    )

    args = parser.parse_args()

    # Get absolute path
    station_data_path = Path(args.station_data_path).resolve()
    agents_dir = station_data_path / "agents"

    if not agents_dir.exists():
        print(f"Error: Agents directory not found at {agents_dir}")
        sys.exit(1)

    print(f"Scanning agents in: {agents_dir}")
    print()

    # Find all agent YAML files
    agent_files = sorted(agents_dir.glob("*.yaml"))

    if not agent_files:
        print("No agent files found.")
        return

    updated_count = 0
    skipped_count = 0

    for agent_file in agent_files:
        agent_name = agent_file.stem
        if update_agent_file(agent_file):
            print(f"✓ Updated: {agent_name}")
            updated_count += 1
        else:
            skipped_count += 1

    print()
    print(f"Summary:")
    print(f"  Total agents: {len(agent_files)}")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped (already new format or no prune data): {skipped_count}")


if __name__ == "__main__":
    main()
