#!/usr/bin/env python3
"""
Extract code from research evaluation submissions and save to files.

Usage: python scripts/eval_to_file.py 3-5 --station_data_path station_data_tmp --output_path tests
"""

import argparse
import json
import os
import sys
import uuid
import yaml
from pathlib import Path


def parse_submission_ids(ids_arg):
    """Parse submission IDs from argument like '3-5' or '3,4,5'."""
    ids = []
    
    if '-' in ids_arg and ',' not in ids_arg:
        # Range format like '3-5'
        start, end = map(int, ids_arg.split('-'))
        ids = list(range(start, end + 1))
    else:
        # Comma-separated format like '3,4,5'
        ids = [int(x.strip()) for x in ids_arg.split(',')]
    
    return ids


def get_station_id(station_data_path):
    """Extract station ID from station_config.yaml, fallback to random UUID."""
    config_path = os.path.join(station_data_path, 'station_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            station_id = config.get('station_id')
            if station_id:
                return station_id
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not read station config from {config_path}: {e}")
    
    # Fallback to random UUID
    fallback_id = str(uuid.uuid4())
    print(f"Using fallback station ID: {fallback_id}")
    return fallback_id


def load_evaluation(evaluation_path):
    """Load and parse evaluation JSON file."""
    try:
        with open(evaluation_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {evaluation_path}: {e}")
        return None


def get_latest_code(evaluation_data):
    """Extract the most recent code version from evaluation data."""
    # Check if there are any versions (corrected code)
    versions = evaluation_data.get('versions', {})
    
    if versions:
        # Get the highest version number
        version_keys = list(versions.keys())
        # Sort by version number if they're like 'v2', 'v3', etc.
        try:
            version_keys.sort(key=lambda x: int(x[1:]) if x.startswith('v') else 0)
        except:
            # If sorting fails, just use the last one
            pass
        
        latest_version = version_keys[-1]
        return versions[latest_version]['content']
    
    # Fall back to original submission
    original = evaluation_data.get('original_submission', {})
    return original.get('content', '')


def save_code_to_file(code, submission_id, output_dir):
    """Save code to a file named submission_N.py."""
    filename = f"submission_{submission_id}.py"
    output_path = os.path.join(output_dir, filename)
    
    try:
        with open(output_path, 'w') as f:
            f.write(code)
        print(f"Saved submission {submission_id} to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving submission {submission_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract code from research evaluation submissions"
    )
    parser.add_argument(
        'submission_ids', 
        help="Submission IDs to extract (e.g., '3-5' or '3,4,5')"
    )
    parser.add_argument(
        '--station_data_path', 
        default='station_data',
        help="Path to station_data directory (default: station_data)"
    )
    parser.add_argument(
        '--output_path', 
        default='tests',
        help="Output directory for saved files (default: tests)"
    )
    
    args = parser.parse_args()
    
    # Parse submission IDs
    try:
        submission_ids = parse_submission_ids(args.submission_ids)
    except ValueError as e:
        print(f"Error parsing submission IDs: {e}")
        return 1
    
    # Get station ID and create station-specific output directory
    station_id = get_station_id(args.station_data_path)
    station_output_dir = os.path.join(args.output_path, station_id)
    os.makedirs(station_output_dir, exist_ok=True)
    print(f"Saving submissions to: {station_output_dir}")
    
    # Process each submission
    success_count = 0
    for submission_id in submission_ids:
        evaluation_path = os.path.join(
            args.station_data_path,
            'rooms/research/evaluations',
            f'evaluation_{submission_id}.json'
        )
        
        print(f"Processing submission {submission_id}...")
        
        # Load evaluation data
        evaluation_data = load_evaluation(evaluation_path)
        if not evaluation_data:
            continue
            
        # Extract latest code
        code = get_latest_code(evaluation_data)
        if not code:
            print(f"No code found for submission {submission_id}")
            continue
            
        # Save to file
        if save_code_to_file(code, submission_id, station_output_dir):
            success_count += 1
    
    print(f"\nSuccessfully extracted {success_count}/{len(submission_ids)} submissions")
    print(f"Files saved to: {station_output_dir}")
    return 0 if success_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())