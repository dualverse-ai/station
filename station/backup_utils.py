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
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

from station import constants
from station import file_io_utils
from station import research_storage


@dataclass
class FileSnapshot:
    """Represents a file in the backup system"""
    path: str
    hash: str
    size: int
    mode: int
    mtime: float


@dataclass
class SymlinkSnapshot:
    """Represents a symbolic link in the backup system."""
    path: str
    target: str
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


def _safe_snapshot_suffix(suffix: str) -> str:
    suffix = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(suffix or "")
    ).strip("_")
    return suffix


def _snapshot_manifest_path(snapshots_dir: str, current_tick: int, snapshot_suffix: Optional[str] = None) -> str:
    suffix = _safe_snapshot_suffix(snapshot_suffix or "")
    filename = f"tick_{current_tick}_{suffix}.json" if suffix else f"tick_{current_tick}.json"
    return os.path.join(snapshots_dir, filename)


def _move_existing_snapshot_aside(manifest_path: str) -> Optional[str]:
    if not os.path.exists(manifest_path):
        return None

    directory = os.path.dirname(manifest_path)
    stem, ext = os.path.splitext(os.path.basename(manifest_path))
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    unique = uuid.uuid4().hex[:12]
    for attempt in range(1000):
        attempt_suffix = "" if attempt == 0 else f"_{attempt}"
        candidate = os.path.join(directory, f"{stem}_backup_{timestamp}_{unique}{attempt_suffix}{ext}")
        if not os.path.exists(candidate):
            shutil.move(manifest_path, candidate)
            return candidate
    raise RuntimeError(f"Could not move existing backup manifest aside: {manifest_path}")


def _should_skip_backup_dir(path: str, backup_base: str) -> bool:
    """Return True when a directory should be excluded from backup traversal."""
    normalized = os.path.normpath(path).replace("\\", "/")
    sync_dir = os.path.normpath(os.path.join(
        constants.BASE_STATION_DATA_PATH,
        constants.PARALLEL_TICK_STATE_DIR_NAME,
    )).replace("\\", "/")
    if normalized == sync_dir or normalized.startswith(f"{sync_dir}/"):
        return True

    skip_markers = (
        backup_base.replace("\\", "/"),
        "claude_workspaces",
        "rooms/research/storage/tmp",
        "rooms/research/storage/shared/tmp",
    )
    return any(marker in normalized for marker in skip_markers)


def _research_storage_root_path() -> str:
    return os.path.normpath(os.path.join(
        constants.BASE_STATION_DATA_PATH,
        constants.ROOMS_DIR_NAME,
        constants.SHORT_ROOM_NAME_RESEARCH,
        constants.RESEARCH_STORAGE_DIR,
    )).replace("\\", "/")


def _should_follow_backup_symlink_dir(path: str) -> bool:
    """Return True for symlink directories whose contents are station state."""
    live_storage_root = Path(_research_storage_root_path())
    return research_storage.should_follow_research_storage_symlink(Path(path), live_storage_root)


def create_backup(
    current_tick: int,
    backup_type: str = "automatic",
    station_instance=None,
    snapshot_suffix: Optional[str] = None,
) -> Optional[str]:
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
        symlink_snapshots = []
        total_size = 0
        new_objects = 0
        reused_objects = 0
        
        # Track progress without printing each file
        files_processed = 0
        errors = []
        
        for root, dirs, files in os.walk(constants.BASE_STATION_DATA_PATH, followlinks=True):
            # Prune skipped subdirectories before os.walk descends into them.
            kept_dirs = []
            for dirname in dirs:
                dir_path = os.path.join(root, dirname)
                if _should_skip_backup_dir(dir_path, backup_base):
                    continue
                follow_storage_link = os.path.islink(dir_path) and _should_follow_backup_symlink_dir(dir_path)
                if follow_storage_link:
                    target_real = Path(os.path.realpath(dir_path))
                    root_real = Path(os.path.realpath(root))
                    if research_storage.path_is_within(root_real, target_real):
                        follow_storage_link = False
                if os.path.islink(dir_path) and not follow_storage_link:
                    try:
                        stat = os.lstat(dir_path)
                        symlink_snapshots.append(SymlinkSnapshot(
                            path=os.path.relpath(dir_path, constants.BASE_STATION_DATA_PATH),
                            target=os.readlink(dir_path),
                            mode=stat.st_mode,
                            mtime=stat.st_mtime,
                        ))
                    except Exception as e:
                        errors.append(f"{os.path.relpath(dir_path, constants.BASE_STATION_DATA_PATH)}: {str(e)}")
                    continue
                kept_dirs.append(dirname)
            dirs[:] = kept_dirs

            if _should_skip_backup_dir(root, backup_base):
                dirs[:] = []
                continue

            for filename in files:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, constants.BASE_STATION_DATA_PATH)
                
                try:
                    if os.path.islink(file_path):
                        stat = os.lstat(file_path)
                        symlink_snapshots.append(SymlinkSnapshot(
                            path=relative_path,
                            target=os.readlink(file_path),
                            mode=stat.st_mode,
                            mtime=stat.st_mtime,
                        ))
                        continue

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
            "files": [asdict(fs) for fs in file_snapshots],
            "symlinks": [asdict(link) for link in symlink_snapshots],
        }
        
        # Save manifest
        manifest_path = _snapshot_manifest_path(snapshots_dir, current_tick, snapshot_suffix)
        moved_manifest = _move_existing_snapshot_aside(manifest_path)
        if moved_manifest:
            print(
                f"  Existing snapshot manifest moved aside before {backup_type} backup: "
                f"{moved_manifest}"
            )
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
        print(f"  - Symlinks: {len(symlink_snapshots)}")
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
    if os.environ.get("STATION_DISABLE_BACKUPS") == "1" or os.environ.get("STATION_MULTISTART_BRANCH") == "1":
        return False

    # Check if automatic backups are enabled
    if constants.BACKUP_FREQUENCY_TICKS <= 0:
        return False
    
    # Check if current tick is a backup tick
    return current_tick > 0 and current_tick % constants.BACKUP_FREQUENCY_TICKS == 0


def _restore_target_relative_path(relative_path: str, source_prefix: Optional[str]) -> Optional[str]:
    normalized_path = relative_path.replace("\\", "/").strip("/")
    if not source_prefix:
        return normalized_path

    normalized_prefix = source_prefix.replace("\\", "/").strip("/")
    if normalized_path == normalized_prefix:
        return os.path.basename(normalized_path)
    if normalized_path.startswith(f"{normalized_prefix}/"):
        return normalized_path[len(normalized_prefix) + 1:]
    return None


def _should_skip_restore_relative_path(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    if normalized.startswith(f"{constants.PARALLEL_TICK_STATE_DIR_NAME}/"):
        return True
    if "rooms/research/storage/tmp" in normalized:
        return True
    if "rooms/research/storage/shared/tmp" in normalized:
        return True
    return False


def _safe_restore_path(target_dir: str, relative_path: str) -> Optional[str]:
    target_root = os.path.abspath(target_dir)
    target_path = os.path.abspath(os.path.join(target_root, relative_path))
    if target_path != target_root and not target_path.startswith(target_root + os.sep):
        return None
    return target_path


def _multistart_snapshot_suffix_from_source_prefix(source_prefix: Optional[str]) -> Optional[str]:
    if not source_prefix:
        return None
    normalized_prefix = source_prefix.replace("\\", "/").strip("/")
    parts = normalized_prefix.split("/")
    if len(parts) >= 2 and parts[0] == "multistart" and parts[1]:
        return f"multistart_{parts[1]}"
    return None


def _resolve_restore_manifest_path(
    station_backup_dir: str,
    tick: int,
    source_prefix: Optional[str] = None,
    snapshot_suffix: Optional[str] = None,
) -> str:
    snapshots_dir = os.path.join(station_backup_dir, "snapshots")
    if snapshot_suffix:
        return _snapshot_manifest_path(snapshots_dir, tick, snapshot_suffix)
    multistart_suffix = _multistart_snapshot_suffix_from_source_prefix(source_prefix)
    if multistart_suffix:
        multistart_manifest = _snapshot_manifest_path(snapshots_dir, tick, multistart_suffix)
        if os.path.exists(multistart_manifest):
            return multistart_manifest
    return _snapshot_manifest_path(snapshots_dir, tick)


def _restore_backup_contents(
    station_id: str,
    tick: int,
    target_dir: str,
    source_prefix: Optional[str] = None,
    snapshot_suffix: Optional[str] = None,
) -> bool:
    try:
        # Get paths
        station_backup_dir = os.path.join(constants.BACKUP_BASE_DIR, station_id)
        objects_dir = os.path.join(station_backup_dir, "objects")
        manifest_path = _resolve_restore_manifest_path(
            station_backup_dir,
            tick,
            source_prefix,
            snapshot_suffix,
        )
        
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
        restored_symlinks = 0
        matched_entries = 0
        missing_objects = []
        
        for i, file_info in enumerate(manifest['files']):
            relative_path = file_info['path']
            file_hash = file_info['hash']

            target_relative_path = _restore_target_relative_path(relative_path, source_prefix)
            if target_relative_path is None:
                continue
            matched_entries += 1

            if _should_skip_restore_relative_path(relative_path):
                continue

            # Get object path
            obj_path = os.path.join(objects_dir, file_hash[:2], file_hash[2:])

            if not os.path.exists(obj_path):
                missing_objects.append(relative_path)
                continue
            
            # Create target file path
            target_path = _safe_restore_path(target_dir, target_relative_path)
            if target_path is None:
                missing_objects.append(f"{relative_path} (unsafe target path)")
                continue
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Copy file from objects
            shutil.copy2(obj_path, target_path)
            
            # Restore permissions
            os.chmod(target_path, file_info['mode'])
            
            restored_count += 1
            
            # Print progress every 100 files
            if restored_count % 100 == 0:
                print(f"  Progress: {restored_count} files restored...")

        for link_info in manifest.get('symlinks', []):
            relative_path = str(link_info.get('path') or '')
            link_target = str(link_info.get('target') or '')
            if not relative_path or not link_target:
                continue

            target_relative_path = _restore_target_relative_path(relative_path, source_prefix)
            if target_relative_path is None:
                continue
            matched_entries += 1

            if _should_skip_restore_relative_path(relative_path):
                continue

            target_path = _safe_restore_path(target_dir, target_relative_path)
            if target_path is None:
                missing_objects.append(f"{relative_path} (unsafe symlink target path)")
                continue
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            try:
                os.symlink(link_target, target_path)
                restored_symlinks += 1
            except FileExistsError:
                continue
            except OSError as exc:
                missing_objects.append(f"{relative_path} (symlink restore failed: {exc})")
        
        # Print summary
        if source_prefix and matched_entries == 0:
            print(f"No entries found under '{source_prefix}' in backup tick {tick}")
            return False

        print(f"Restored {restored_count}/{len(manifest['files'])} files from tick {tick}")
        if manifest.get('symlinks'):
            print(f"Restored {restored_symlinks}/{len(manifest.get('symlinks', []))} symlinks from tick {tick}")
        
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


def restore_backup(
    station_id: str,
    tick: int,
    target_dir: str,
    snapshot_suffix: Optional[str] = None,
) -> bool:
    """
    Restore a backup to target directory.

    Args:
        station_id: Station ID of the backup to restore
        tick: Tick number of the backup to restore
        target_dir: Target directory to restore files to
        snapshot_suffix: Optional named manifest suffix, such as
            ``multistart_501_job``. The ordinary tick manifest is used when omitted.

    Returns:
        bool: True if successful, False otherwise
    """
    return _restore_backup_contents(
        station_id,
        tick,
        target_dir,
        snapshot_suffix=snapshot_suffix,
    )


def restore_backup_subtree(station_id: str, tick: int, source_prefix: str, target_dir: str) -> bool:
    """
    Restore one subtree from a backup snapshot into target_dir.

    source_prefix is relative to station_data, for example
    "multistart/501_abcd1234".
    """
    return _restore_backup_contents(station_id, tick, target_dir, source_prefix=source_prefix)


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
