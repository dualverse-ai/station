# Copyright 2025 Dualverse AI
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

# station/backup_utils.py
"""
Backup utilities for the Station.
Handles periodic backups and manual backup creation using incremental backup system.
"""
import os
import shutil
import uuid
import time
import json
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

from station import constants
from station import file_io_utils


@dataclass
class FileSnapshot:
    """Represents a file in the backup system"""
    path: str
    hash: str
    size: int
    mode: int
    mtime: float


def _ensure_station_id(station_config_path: str) -> str:
    """
    Ensure station has a unique ID in station_config.yaml.
    Generate and save one if it doesn't exist.
    
    Returns:
        str: The station's unique ID
    """
    try:
        # Try to load existing config
        if file_io_utils.file_exists(station_config_path):
            config_data = file_io_utils.load_yaml(station_config_path)
            if isinstance(config_data, dict) and constants.STATION_ID_KEY in config_data:
                station_id = config_data[constants.STATION_ID_KEY]
                if station_id and isinstance(station_id, str):
                    return station_id
        else:
            config_data = {}
        
        # Generate new station ID if not found
        new_station_id = str(uuid.uuid4())
        config_data[constants.STATION_ID_KEY] = new_station_id
        
        # Save updated config
        file_io_utils.save_yaml(config_data, station_config_path)
        print(f"Generated new station ID: {new_station_id}")
        
        return new_station_id
        
    except Exception as e:
        print(f"Error ensuring station ID: {e}")
        # Return a fallback ID based on timestamp
        fallback_id = f"station_{int(time.time())}"
        print(f"Using fallback station ID: {fallback_id}")
        return fallback_id


def _compute_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of file contents"""
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def _store_file_object(file_path: str, file_hash: str, objects_dir: str) -> bool:
    """Store file content in objects directory if not already present"""
    # Use first 2 chars as directory (like git)
    obj_subdir = os.path.join(objects_dir, file_hash[:2])
    obj_path = os.path.join(obj_subdir, file_hash[2:])
    
    # If object already exists, no need to copy
    if os.path.exists(obj_path):
        return True
    
    # Create subdirectory and copy file
    os.makedirs(obj_subdir, exist_ok=True)
    shutil.copy2(file_path, obj_path)
    return True


def create_backup(current_tick: int, backup_type: str = "automatic", station_instance=None) -> Optional[str]:
    """
    Create an incremental backup of the station_data directory.
    
    Args:
        current_tick: Current station tick number
        backup_type: Type of backup ("automatic" or "manual")
        station_instance: Station instance to use for station ID management (optional)
        
    Returns:
        str: Path to created backup manifest
        
    Raises:
        Exception: If backup creation fails
    """
    try:
        # Get station ID using station instance if available, otherwise fallback to file method
        if station_instance and hasattr(station_instance, 'station_id'):
            station_id = station_instance.station_id
        else:
            station_config_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.STATION_CONFIG_FILENAME)
            station_id = _ensure_station_id(station_config_path)
        
        # Create backup directory structure
        backup_base = constants.BACKUP_BASE_DIR
        station_backup_dir = os.path.join(backup_base, station_id)
        objects_dir = os.path.join(station_backup_dir, "objects")
        snapshots_dir = os.path.join(station_backup_dir, "snapshots")
        
        # Ensure directories exist
        os.makedirs(objects_dir, exist_ok=True)
        os.makedirs(snapshots_dir, exist_ok=True)
        
        # Check if source directory exists
        if not os.path.exists(constants.BASE_STATION_DATA_PATH):
            raise Exception(f"Source directory {constants.BASE_STATION_DATA_PATH} does not exist")
        
        start_time = datetime.now()
        
        # Scan all files in source directory
        file_snapshots = []
        total_size = 0
        new_objects = 0
        reused_objects = 0
        
        # Track progress without printing each file
        files_processed = 0
        errors = []
        
        for root, dirs, files in os.walk(constants.BASE_STATION_DATA_PATH, followlinks=True):
            # Skip the backup directory itself if it's inside station_data
            if backup_base in root:
                continue

            # Skip claude_workspaces directory
            if "claude_workspaces" in root:
                continue

            # Skip research storage tmp directory
            if "rooms/research/storage/tmp" in root:
                continue

            # Skip research storage shared tmp directory
            if "rooms/research/storage/shared/tmp" in root:
                continue

            for filename in files:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, constants.BASE_STATION_DATA_PATH)
                
                try:
                    # Get file stats
                    stat = os.stat(file_path)
                    file_size = stat.st_size
                    
                    # Compute hash
                    file_hash = _compute_file_hash(file_path)
                    
                    # Check if object already exists
                    obj_path = os.path.join(objects_dir, file_hash[:2], file_hash[2:])
                    if os.path.exists(obj_path):
                        reused_objects += 1
                    else:
                        # Store new object
                        _store_file_object(file_path, file_hash, objects_dir)
                        new_objects += 1
                    
                    # Create snapshot entry
                    snapshot = FileSnapshot(
                        path=relative_path,
                        hash=file_hash,
                        size=file_size,
                        mode=stat.st_mode,
                        mtime=stat.st_mtime
                    )
                    file_snapshots.append(snapshot)
                    total_size += file_size
                    files_processed += 1
                    
                    # Print progress every 100 files
                    if files_processed % 100 == 0:
                        print(f"  Backup Progress: {files_processed} files processed...")
                    
                except Exception as e:
                    errors.append(f"{relative_path}: {str(e)}")
        
        # Print errors summary if any
        if errors:
            print(f"  Errors processing {len(errors)} files:")
            for err in errors[:5]:  # Show first 5 errors
                print(f"    - {err}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more errors")
        
        # Create backup manifest
        manifest = {
            "station_id": station_id,
            "tick": current_tick,
            "backup_type": backup_type,
            "timestamp": start_time.isoformat(),
            "source_dir": constants.BASE_STATION_DATA_PATH,
            "total_files": len(file_snapshots),
            "total_size": total_size,
            "new_objects": new_objects,
            "reused_objects": reused_objects,
            "files": [asdict(fs) for fs in file_snapshots]
        }
        
        # Save manifest
        manifest_path = os.path.join(snapshots_dir, f"tick_{current_tick}.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Save latest station config separately in the main backup directory
        station_config_source = os.path.join(constants.BASE_STATION_DATA_PATH, constants.STATION_CONFIG_FILENAME)
        if os.path.exists(station_config_source):
            try:
                station_config_backup_path = os.path.join(station_backup_dir, "station_config.yaml")
                shutil.copy2(station_config_source, station_config_backup_path)
                print(f"  Latest station config saved separately: {station_config_backup_path}")
            except Exception as e:
                print(f"  Warning: Could not save separate station config: {e}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Calculate space saved
        avg_file_size = total_size / len(file_snapshots) if file_snapshots else 0
        space_saved = reused_objects * avg_file_size
        
        print(f"Backup completed in {duration:.2f}s:")
        print(f"  - Total files: {len(file_snapshots)}")
        print(f"  - Total size: {total_size / (1024**3):.2f} GB")
        print(f"  - New objects stored: {new_objects}")
        print(f"  - Objects reused: {reused_objects}")
        print(f"  - Approx space saved: {space_saved / (1024**3):.2f} GB")
        
        return manifest_path
            
    except Exception as e:
        print(f"Error creating backup: {e}")
        import traceback
        traceback.print_exc()
        raise


def should_create_automatic_backup(current_tick: int) -> bool:
    """
    Check if an automatic backup should be created based on current tick.
    
    Args:
        current_tick: Current station tick number
        
    Returns:
        bool: True if backup should be created
    """
    # Check if automatic backups are enabled
    if constants.BACKUP_FREQUENCY_TICKS <= 0:
        return False
    
    # Check if current tick is a backup tick
    return current_tick > 0 and current_tick % constants.BACKUP_FREQUENCY_TICKS == 0


def restore_backup(station_id: str, tick: int, target_dir: str) -> bool:
    """
    Restore a backup to target directory.
    
    Args:
        station_id: Station ID of the backup to restore
        tick: Tick number of the backup to restore
        target_dir: Target directory to restore files to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get paths
        station_backup_dir = os.path.join(constants.BACKUP_BASE_DIR, station_id)
        objects_dir = os.path.join(station_backup_dir, "objects")
        snapshots_dir = os.path.join(station_backup_dir, "snapshots")
        manifest_path = os.path.join(snapshots_dir, f"tick_{tick}.json")
        
        if not os.path.exists(manifest_path):
            print(f"Backup not found: {manifest_path}")
            return False
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Check if target directory exists and halt if it does
        if os.path.exists(target_dir):
            print(f"Error: Target directory '{target_dir}' already exists.")
            print("Please remove or rename it before restoring.")
            print(f"Suggested command: mv {target_dir} {target_dir}_backup_$(date +%s)")
            return False
        
        os.makedirs(target_dir)
        
        # Restore files
        restored_count = 0
        missing_objects = []
        
        for i, file_info in enumerate(manifest['files']):
            relative_path = file_info['path']
            file_hash = file_info['hash']

            # Skip research storage tmp directory during restore
            if "rooms/research/storage/tmp" in relative_path:
                continue

            # Skip research storage shared tmp directory during restore
            if "rooms/research/storage/shared/tmp" in relative_path:
                continue

            # Get object path
            obj_path = os.path.join(objects_dir, file_hash[:2], file_hash[2:])

            if not os.path.exists(obj_path):
                missing_objects.append(relative_path)
                continue
            
            # Create target file path
            target_path = os.path.join(target_dir, relative_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Copy file from objects
            shutil.copy2(obj_path, target_path)
            
            # Restore permissions
            os.chmod(target_path, file_info['mode'])
            
            restored_count += 1
            
            # Print progress every 100 files
            if restored_count % 100 == 0:
                print(f"  Progress: {restored_count} files restored...")
        
        # Print summary
        print(f"Restored {restored_count}/{len(manifest['files'])} files from tick {tick}")
        
        # Print missing objects summary if any
        if missing_objects:
            print(f"  Warning: {len(missing_objects)} files could not be restored (missing objects):")
            for path in missing_objects[:5]:  # Show first 5
                print(f"    - {path}")
            if len(missing_objects) > 5:
                print(f"    ... and {len(missing_objects) - 5} more")
        return True
        
    except Exception as e:
        print(f"Error restoring backup: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_station_id() -> Optional[str]:
    """
    Get the current station's unique ID.
    
    Returns:
        str: Station ID or None if error
    """
    try:
        station_config_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.STATION_CONFIG_FILENAME)
        return _ensure_station_id(station_config_path)
    except Exception as e:
        print(f"Error getting station ID: {e}")
        return None


if __name__ == "__main__":
    """
    Unit tests for backup utilities.
    Run with: python -m station.backup_utils
    """
    print("🧪 Incremental Backup Utilities Unit Tests")
    print("=" * 50)
    
    # Test 1: Station ID generation/retrieval
    print("\n1. Testing Station ID...")
    station_id = get_station_id()
    if station_id:
        print(f"   ✓ Station ID: {station_id}")
    else:
        print("   ❌ Failed to get station ID")
    
    # Test 2: Backup frequency check
    print("\n2. Testing Backup Frequency Check...")
    test_cases = [
        (49, False),  # Not a backup tick
        (50, True),   # Backup tick
        (100, True),  # Backup tick
        (101, False), # Not a backup tick
    ]
    
    for tick, expected in test_cases:
        result = should_create_automatic_backup(tick)
        status = "✓" if result == expected else "❌"
        print(f"   {status} Tick {tick}: {result} (expected {expected})")
    
    # Test 3: Incremental backup creation (if station_data exists)
    print("\n3. Testing Incremental Backup Creation...")
    if os.path.exists(constants.BASE_STATION_DATA_PATH):
        # First backup
        manifest_path = create_backup(999, "test")
        if manifest_path and os.path.exists(manifest_path):
            print(f"   ✓ Test backup manifest created: {manifest_path}")
            
            # Check manifest content
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            if isinstance(manifest, dict) and "backup_type" in manifest:
                print(f"   ✓ Manifest contains backup type: {manifest['backup_type']}")
                print(f"   ✓ Total files: {manifest['total_files']}")
                print(f"   ✓ New objects: {manifest['new_objects']}")
                print(f"   ✓ Reused objects: {manifest['reused_objects']}")
            else:
                print("   ❌ Invalid manifest format")
            
            # Test second backup to verify object reuse
            print("\n   Testing object reuse...")
            manifest_path2 = create_backup(1000, "test2")
            if manifest_path2 and os.path.exists(manifest_path2):
                with open(manifest_path2, 'r') as f:
                    manifest2 = json.load(f)
                print(f"   ✓ Second backup created")
                print(f"   ✓ New objects: {manifest2['new_objects']}")
                print(f"   ✓ Reused objects: {manifest2['reused_objects']}")
                if manifest2['reused_objects'] > 0:
                    print("   ✓ Object deduplication working!")
                else:
                    print("   ⚠ No objects were reused (expected for identical content)")
        else:
            print("   ❌ Test backup creation failed")
    else:
        print(f"   ⚠ Station data directory not found at {constants.BASE_STATION_DATA_PATH}")
    
    # Test 4: Backup restoration
    print("\n4. Testing Backup Restoration...")
    if station_id and os.path.exists(constants.BASE_STATION_DATA_PATH):
        test_restore_dir = "/tmp/test_restore_station"
        if restore_backup(station_id, 999, test_restore_dir):
            print(f"   ✓ Backup restored to {test_restore_dir}")
            # Check if key files exist
            config_path = os.path.join(test_restore_dir, constants.STATION_CONFIG_FILENAME)
            if os.path.exists(config_path):
                print("   ✓ Station config file restored")
            else:
                print("   ❌ Station config file missing in restore")
            # Clean up
            shutil.rmtree(test_restore_dir)
        else:
            print("   ❌ Backup restoration failed")
    
    print("\n" + "=" * 50)
    print("🎉 Incremental Backup Utilities Unit Tests Complete!")