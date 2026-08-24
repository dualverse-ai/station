#!/usr/bin/env python3
"""
Clean large files from incremental backup system.

This script removes large files from the backup while preserving the backup structure.
It updates the manifest files to reflect the removed files.

Usage:
    python clean_backup.py <backup_path> <file_pattern> [--min-size <MB>] [--dry-run]

Examples:
    # Clean all .npz files from a backup (dry run)
    python clean_backup.py /workspace/station/backup/sokoban_2_push_tick_1880 "rooms/research/storage/*.npz" --dry-run
    
    # Clean .npz files larger than 10MB 
    python clean_backup.py /workspace/station/backup/sokoban_2_push_tick_1880 "rooms/research/storage/*.npz" --min-size 10
    
    # Clean all .pkl files regardless of size
    python clean_backup.py /workspace/station/backup/station_id_here "**/*.pkl"
"""

import os
import sys
import json
import argparse
import fnmatch
from pathlib import Path
from typing import List, Set, Tuple
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean large files from incremental backup system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "backup_path",
        help="Path to the backup directory (e.g., /workspace/station/backup/station_id)"
    )
    
    parser.add_argument(
        "file_pattern",
        help="File pattern to clean (e.g., 'rooms/research/storage/*.npz' or '**/*.pkl')"
    )
    
    parser.add_argument(
        "--min-size",
        type=float,
        default=0,
        help="Minimum file size in MB to clean (default: 0, no limit)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without actually removing files"
    )
    
    return parser.parse_args()


def find_matching_files(manifest: dict, pattern: str, min_size_bytes: int) -> List[Tuple[str, str, int]]:
    """
    Find files in manifest that match the pattern and size criteria.
    
    Returns:
        List of tuples (relative_path, hash, size)
    """
    matching_files = []
    
    for file_info in manifest.get("files", []):
        relative_path = file_info["path"]
        file_size = file_info["size"]
        file_hash = file_info["hash"]
        
        # Check pattern match
        if fnmatch.fnmatch(relative_path, pattern):
            # Check size criteria
            if file_size >= min_size_bytes:
                matching_files.append((relative_path, file_hash, file_size))
    
    return matching_files


def get_object_path(objects_dir: str, file_hash: str) -> str:
    """Get the object path for a given file hash."""
    return os.path.join(objects_dir, file_hash[:2], file_hash[2:])


def is_hash_referenced_elsewhere(snapshots_dir: str, file_hash: str, exclude_manifests: Set[str]) -> bool:
    """Check if a hash is referenced in any manifest other than the ones being cleaned."""
    for manifest_file in os.listdir(snapshots_dir):
        if manifest_file.endswith(".json") and manifest_file not in exclude_manifests:
            manifest_path = os.path.join(snapshots_dir, manifest_file)
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    
                for file_info in manifest.get("files", []):
                    if file_info["hash"] == file_hash:
                        return True
            except Exception as e:
                print(f"Warning: Could not read manifest {manifest_file}: {e}")
    
    return False


def clean_backup(backup_path: str, file_pattern: str, min_size_mb: float, dry_run: bool):
    """Clean large files from the backup."""
    # Validate backup path
    if not os.path.exists(backup_path):
        print(f"Error: Backup path does not exist: {backup_path}")
        return 1
    
    objects_dir = os.path.join(backup_path, "objects")
    snapshots_dir = os.path.join(backup_path, "snapshots")
    
    if not os.path.exists(objects_dir) or not os.path.exists(snapshots_dir):
        print(f"Error: Invalid backup structure. Expected 'objects' and 'snapshots' directories.")
        return 1
    
    min_size_bytes = int(min_size_mb * 1024 * 1024)
    
    # Track all files to remove and hashes that might be cleaned
    files_to_remove = []
    hashes_to_check = set()
    modified_manifests = set()
    
    # Process all manifest files
    for manifest_file in sorted(os.listdir(snapshots_dir)):
        if not manifest_file.endswith(".json"):
            continue
            
        manifest_path = os.path.join(snapshots_dir, manifest_file)
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Find matching files in this manifest
            matching_files = find_matching_files(manifest, file_pattern, min_size_bytes)
            
            if matching_files:
                print(f"\nManifest: {manifest_file}")
                print(f"  Found {len(matching_files)} matching files:")
                
                for rel_path, file_hash, size in matching_files:
                    size_mb = size / (1024 * 1024)
                    print(f"    - {rel_path} ({size_mb:.2f} MB)")
                    files_to_remove.append((manifest_file, rel_path, file_hash, size))
                    hashes_to_check.add(file_hash)
                    modified_manifests.add(manifest_file)
                
        except Exception as e:
            print(f"Error reading manifest {manifest_file}: {e}")
            continue
    
    if not files_to_remove:
        print(f"\nNo files found matching pattern '{file_pattern}' with size >= {min_size_mb} MB")
        return 0
    
    # Determine which object files can be safely removed
    objects_to_remove = set()
    for file_hash in hashes_to_check:
        if not is_hash_referenced_elsewhere(snapshots_dir, file_hash, modified_manifests):
            objects_to_remove.add(file_hash)
    
    # Calculate total size
    total_size = sum(size for _, _, _, size in files_to_remove)
    total_size_mb = total_size / (1024 * 1024)
    
    # Summary
    print(f"\n{'DRY RUN ' if dry_run else ''}Summary:")
    print(f"  - Files to remove from manifests: {len(files_to_remove)}")
    print(f"  - Object files to delete: {len(objects_to_remove)}")
    print(f"  - Total size to free: {total_size_mb:.2f} MB")
    print(f"  - Manifests to update: {len(modified_manifests)}")
    
    if dry_run:
        print("\nDry run complete. No files were modified.")
        return 0
    
    # Confirm action
    response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
    if response != "yes":
        print("Cleanup cancelled.")
        return 0
    
    # Perform cleanup
    print("\nCleaning backup...")
    
    # Update manifests
    for manifest_file in modified_manifests:
        manifest_path = os.path.join(snapshots_dir, manifest_file)
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Filter out removed files
            original_count = len(manifest["files"])
            removed_paths = {rel_path for mf, rel_path, _, _ in files_to_remove if mf == manifest_file}
            manifest["files"] = [f for f in manifest["files"] if f["path"] not in removed_paths]
            
            # Update manifest metadata
            manifest["cleaned"] = True
            manifest["clean_timestamp"] = datetime.now().isoformat()
            manifest["files_removed"] = len(removed_paths)
            
            # Save updated manifest
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print(f"  Updated {manifest_file}: removed {len(removed_paths)} entries")
            
        except Exception as e:
            print(f"  Error updating manifest {manifest_file}: {e}")
    
    # Remove orphaned objects
    removed_objects = 0
    failed_removals = 0
    
    for file_hash in objects_to_remove:
        obj_path = get_object_path(objects_dir, file_hash)
        
        if os.path.exists(obj_path):
            try:
                os.remove(obj_path)
                removed_objects += 1
            except Exception as e:
                print(f"  Error removing object {file_hash[:8]}...: {e}")
                failed_removals += 1
    
    print(f"\nCleanup complete:")
    print(f"  - Manifests updated: {len(modified_manifests)}")
    print(f"  - Objects removed: {removed_objects}")
    if failed_removals > 0:
        print(f"  - Failed removals: {failed_removals}")
    
    return 0


def main():
    """Main entry point."""
    args = parse_args()
    
    return clean_backup(
        args.backup_path,
        args.file_pattern,
        args.min_size,
        args.dry_run
    )


if __name__ == "__main__":
    sys.exit(main())