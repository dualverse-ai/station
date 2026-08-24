#!/usr/bin/env python3
"""
Generate preview of capsules in the same format as the Station's {preview all} action.
Supports archive (default), public, and private capsules.
"""
import os
import sys
import yaml
import argparse
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from station import constants
from station import file_io_utils


def get_archive_capsules(station_data_path: str) -> List[Dict[str, Any]]:
    """Load all archive capsules from the station data."""
    capsules_dir = os.path.join(station_data_path, 'capsules', 'archive')

    if not os.path.exists(capsules_dir):
        print(f"Archive capsules directory not found: {capsules_dir}")
        return []

    # Get all archive capsule files (YAML format) and sort by ID
    capsule_files = []
    for filename in os.listdir(capsules_dir):
        if filename.startswith('archive_') and filename.endswith('.yaml'):
            try:
                # Extract ID from filename (archive_N.yaml)
                capsule_id = int(filename.split('_')[1].split('.')[0])
                capsule_files.append((capsule_id, filename))
            except (IndexError, ValueError):
                print(f"Skipping file with unexpected name format: {filename}")

    # Sort by capsule ID
    capsule_files.sort(key=lambda x: x[0])

    capsules = []
    for capsule_id, filename in capsule_files:
        filepath = os.path.join(capsules_dir, filename)
        try:
            data = file_io_utils.load_yaml(filepath)
            if data:
                capsules.append(data)
        except Exception as e:
            print(f"Error reading capsule {filename}: {e}")

    return capsules


def get_public_capsules(station_data_path: str) -> List[Dict[str, Any]]:
    """Load all public capsules from the station data."""
    capsules_dir = os.path.join(station_data_path, 'capsules', 'public')

    if not os.path.exists(capsules_dir):
        print(f"Public capsules directory not found: {capsules_dir}")
        return []

    # Get all public capsule files (YAML format) and sort by ID
    capsule_files = []
    for filename in os.listdir(capsules_dir):
        if filename.startswith('public_') and filename.endswith('.yaml'):
            try:
                # Extract ID from filename (public_N.yaml)
                capsule_id = int(filename.split('_')[1].split('.')[0])
                capsule_files.append((capsule_id, filename))
            except (IndexError, ValueError):
                print(f"Skipping file with unexpected name format: {filename}")

    # Sort by capsule ID
    capsule_files.sort(key=lambda x: x[0])

    capsules = []
    for capsule_id, filename in capsule_files:
        filepath = os.path.join(capsules_dir, filename)
        try:
            data = file_io_utils.load_yaml(filepath)
            if data:
                capsules.append(data)
        except Exception as e:
            print(f"Error reading capsule {filename}: {e}")

    return capsules


def get_private_capsules(station_data_path: str, lineage_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load all private capsules from a specific lineage or all lineages if not specified."""
    private_base_dir = os.path.join(station_data_path, 'capsules', 'private')

    if not os.path.exists(private_base_dir):
        print(f"Private capsules directory not found: {private_base_dir}")
        return []

    # Determine which lineages to process
    if lineage_name:
        # Normalize lineage name: capitalize first letter, lowercase rest
        normalized_lineage = lineage_name[0].upper() + lineage_name[1:].lower() if lineage_name else lineage_name
        lineage_dirs = [f'lineage_{normalized_lineage}']
    else:
        # Get all lineage directories
        lineage_dirs = [d for d in os.listdir(private_base_dir)
                       if os.path.isdir(os.path.join(private_base_dir, d)) and d.startswith('lineage_')]

    all_capsules = []

    for lineage_dir in lineage_dirs:
        capsules_dir = os.path.join(private_base_dir, lineage_dir)

        if not os.path.exists(capsules_dir):
            if lineage_name:  # Only print error if user specified a lineage
                print(f"Private capsules directory for lineage '{lineage_name}' not found: {capsules_dir}")
            continue

        # Get all private capsule files (YAML format)
        capsule_files = []
        for filename in os.listdir(capsules_dir):
            if filename.endswith('.yaml') and '_private_' in filename:
                try:
                    # Extract ID from filename (Lineage_private_N.yaml)
                    parts = filename.replace('.yaml', '').split('_private_')
                    if len(parts) == 2:
                        capsule_id = int(parts[1])
                        capsule_files.append((capsule_id, filename))
                except (IndexError, ValueError):
                    print(f"Skipping file with unexpected name format: {filename}")

        # Load capsules
        for capsule_id, filename in capsule_files:
            filepath = os.path.join(capsules_dir, filename)
            try:
                data = file_io_utils.load_yaml(filepath)
                if data:
                    all_capsules.append(data)
            except Exception as e:
                print(f"Error reading capsule {filename}: {e}")

    # Sort all capsules by created_at_tick
    all_capsules.sort(key=lambda x: x.get(constants.CAPSULE_CREATED_AT_TICK_KEY, 0))

    return all_capsules


def format_capsule_preview(capsule: Dict[str, Any]) -> str:
    """Format a single capsule preview in the Station's format."""
    capsule_id = capsule.get(constants.CAPSULE_ID_KEY, "Unknown")
    title = capsule.get(constants.CAPSULE_TITLE_KEY, "Untitled")
    author = capsule.get(constants.CAPSULE_AUTHOR_NAME_KEY, "Unknown")
    created_tick = capsule.get(constants.CAPSULE_CREATED_AT_TICK_KEY, "N/A")
    abstract = capsule.get(constants.CAPSULE_ABSTRACT_KEY)

    preview_str = f"**Preview for Capsule #{capsule_id}: {title}**\n"
    preview_str += f"Author: {author}, Created at Tick: {created_tick}\n"

    if abstract:
        preview_str += f"Abstract: {abstract}"
    else:
        # If no abstract, just show the title (already in header)
        preview_str += f"Title: {title}"

    return preview_str


def generate_preview(station_data_path: str, capsule_type: str, lineage_name: Optional[str] = None, output_file: Optional[str] = None):
    """Generate preview of capsules based on type."""

    # Load capsules based on type
    if capsule_type == 'archive':
        capsules = get_archive_capsules(station_data_path)
        type_label = "archive"
    elif capsule_type == 'public':
        capsules = get_public_capsules(station_data_path)
        type_label = "public"
    elif capsule_type == 'private':
        capsules = get_private_capsules(station_data_path, lineage_name)
        if lineage_name:
            # Normalize for display
            normalized_lineage = lineage_name[0].upper() + lineage_name[1:].lower()
            type_label = f"private (lineage: {normalized_lineage})"
        else:
            type_label = "private (all lineages)"
    else:
        print(f"Error: Unknown capsule type '{capsule_type}'")
        sys.exit(1)

    if not capsules:
        print(f"No {type_label} capsules found.")
        return

    print(f"Found {len(capsules)} {type_label} capsules.\n")

    # Generate previews
    preview_parts = []
    for capsule in capsules:
        preview = format_capsule_preview(capsule)
        preview_parts.append(preview)

    # Join with separator
    full_preview = "\n\n---\n\n".join(preview_parts)

    # Output to console
    print(full_preview)

    # Save to file if requested
    if output_file:
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_preview)
            print(f"\n\nPreview saved to: {output_file}")
        except Exception as e:
            print(f"\nError saving to file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate preview of capsules in Station format (archive, public, or private)'
    )
    parser.add_argument(
        'station_data_path',
        nargs='?',
        default=None,
        help='Path to station_data directory (defaults to ../station_data from script location)'
    )
    parser.add_argument(
        '--public',
        action='store_true',
        help='Preview all public capsules (default: archive)'
    )
    parser.add_argument(
        '--private',
        nargs='?',
        const='',
        metavar='LINEAGE',
        help='Preview private capsules. Specify lineage name (e.g., --private Axiom) or omit for all lineages (--private). Case-insensitive.'
    )
    parser.add_argument(
        '-o', '--output',
        help='Optional output file path to save the preview'
    )

    args = parser.parse_args()

    # Determine capsule type
    if args.public and args.private is not None:
        print("Error: Cannot specify both --public and --private")
        sys.exit(1)

    if args.public:
        capsule_type = 'public'
        lineage_name = None
    elif args.private is not None:
        capsule_type = 'private'
        # Empty string means show all lineages, otherwise use the specified lineage
        lineage_name = args.private if args.private else None
    else:
        # Default to archive
        capsule_type = 'archive'
        lineage_name = None

    # Determine station_data path
    if args.station_data_path:
        station_data_path = args.station_data_path
    else:
        # Default path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        station_data_path = os.path.join(script_dir, '..', 'station_data')

    # Convert to absolute path and verify it exists
    station_data_path = os.path.abspath(station_data_path)

    if not os.path.exists(station_data_path):
        print(f"Error: station_data path does not exist: {station_data_path}")
        sys.exit(1)

    print(f"Using station_data path: {station_data_path}")

    # Generate preview
    generate_preview(station_data_path, capsule_type, lineage_name, args.output)


if __name__ == "__main__":
    main()
