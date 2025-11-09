#!/usr/bin/env python3


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
GPU Coordinator for managing GPU allocation across multiple stations.

This module provides both file-based and in-memory GPU allocation tracking
to enable multiple stations to share GPUs without conflicts.
"""

import os
import json
import time
import fcntl
import threading
from typing import Dict, List, Optional
from datetime import datetime


class GPUCoordinator:
    """
    Manages GPU allocation using either file-based coordination (for multi-station)
    or in-memory tracking (for single station).

    File format:
    {
        "allocations": {
            "station_uuid:eval_id": {
                "gpus": [0, 1],
                "station_id": "uuid",
                "eval_id": "123",
                "start_time": 1234567890.123,
                "start_time_str": "2024-01-01 12:00:00"
            }
        },
        "last_updated": 1234567890.123,
        "last_updated_str": "2024-01-01 12:00:00"
    }
    """

    def __init__(self, coord_file_path: Optional[str] = None,
                 available_gpus: Optional[List[int]] = None,
                 station_id: Optional[str] = None):
        """
        Initialize GPU coordinator.

        Args:
            coord_file_path: Path to coordination file, None for in-memory mode
            available_gpus: List of GPU IDs available for allocation
            station_id: Unique station identifier
        """
        self.coord_file = coord_file_path
        self.total_gpus = available_gpus or []
        self.station_id = station_id or "unknown"
        self.lock_timeout = 5.0  # Hardcoded timeout for file lock acquisition

        if coord_file_path:
            # File-based coordination
            print(f"GPUCoordinator: File-based coordination at {coord_file_path} (station: {self.station_id})")
            self._ensure_coord_file_exists()
            self._cleanup_station_allocations()
        else:
            # In-memory coordination (current behavior)
            print("GPUCoordinator: Using in-memory GPU tracking (single station mode)")
            self.lock = threading.Lock()
            self.allocated: Dict[str, List[int]] = {}
            self.available = self.total_gpus.copy()

    def _ensure_coord_file_exists(self):
        """Create coordination file if it doesn't exist."""
        if not os.path.exists(self.coord_file):
            initial_data = {
                "allocations": {},
                "last_updated": time.time(),
                "last_updated_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            dir_path = os.path.dirname(self.coord_file)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(self.coord_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
            print(f"GPUCoordinator: Created new coordination file at {self.coord_file}")

    def _cleanup_station_allocations(self):
        """Clean up all allocations from this station on startup."""
        if not self.coord_file:
            return

        try:
            with open(self.coord_file, 'r+') as f:
                # Acquire exclusive lock with timeout
                self._acquire_lock(f)

                try:
                    # Handle empty file case
                    content = f.read()
                    if not content:
                        # Initialize empty file
                        data = {
                            "allocations": {},
                            "last_updated": time.time(),
                            "last_updated_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()
                        return

                    f.seek(0)
                    data = json.load(f)
                    original_count = len(data.get("allocations", {}))

                    # Remove all allocations from this station
                    cleaned_allocations = {}
                    removed_count = 0
                    for key, info in data.get("allocations", {}).items():
                        if info.get("station_id") != self.station_id:
                            cleaned_allocations[key] = info
                        else:
                            removed_count += 1

                    if removed_count > 0:
                        data["allocations"] = cleaned_allocations
                        data["last_updated"] = time.time()
                        data["last_updated_str"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()

                        print(f"GPUCoordinator: Cleaned up {removed_count} stale allocations from previous session")

                finally:
                    self._release_lock(f)

        except Exception as e:
            print(f"GPUCoordinator: Error during cleanup: {e}")

    def _acquire_lock(self, file_handle):
        """Acquire exclusive lock on file with timeout."""
        start_time = time.time()
        while True:
            try:
                fcntl.flock(file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except IOError:
                if time.time() - start_time > self.lock_timeout:
                    raise TimeoutError(f"Could not acquire lock on {self.coord_file} after {self.lock_timeout}s")
                time.sleep(0.01)

    def _release_lock(self, file_handle):
        """Release file lock."""
        fcntl.flock(file_handle, fcntl.LOCK_UN)

    def allocate(self, eval_id: str, count: int = 1) -> Optional[List[int]]:
        """
        Allocate GPUs for an evaluation.

        Args:
            eval_id: Evaluation identifier
            count: Number of GPUs to allocate

        Returns:
            List of allocated GPU IDs or None if unavailable
        """
        if self.coord_file:
            return self._allocate_file_based(eval_id, count)
        else:
            return self._allocate_memory_based(eval_id, count)

    def _allocate_memory_based(self, eval_id: str, count: int) -> Optional[List[int]]:
        """In-memory allocation (original behavior)."""
        with self.lock:
            if len(self.available) >= count:
                allocated_gpu_ids = []
                for _ in range(count):
                    gpu_id = self.available.pop(0)
                    allocated_gpu_ids.append(gpu_id)

                self.allocated[eval_id] = allocated_gpu_ids
                return allocated_gpu_ids
            else:
                return None

    def _allocate_file_based(self, eval_id: str, count: int) -> Optional[List[int]]:
        """File-based allocation for multi-station coordination."""
        allocation_key = f"{self.station_id}:{eval_id}"

        try:
            # Ensure file exists (may have been deleted)
            self._ensure_coord_file_exists()

            with open(self.coord_file, 'r+') as f:
                self._acquire_lock(f)

                try:
                    data = json.load(f)

                    # Check if already allocated (idempotent)
                    if allocation_key in data.get("allocations", {}):
                        return data["allocations"][allocation_key].get("gpus", [])

                    # Calculate available GPUs
                    used_gpus = set()
                    for info in data.get("allocations", {}).values():
                        used_gpus.update(info.get("gpus", []))

                    available = [gpu for gpu in self.total_gpus if gpu not in used_gpus]

                    if len(available) >= count:
                        # Allocate GPUs
                        allocated_gpus = available[:count]

                        # Record allocation with metadata
                        current_time = time.time()
                        data["allocations"][allocation_key] = {
                            "gpus": allocated_gpus,
                            "station_id": self.station_id,
                            "eval_id": eval_id,
                            "start_time": current_time,
                            "start_time_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        data["last_updated"] = current_time
                        data["last_updated_str"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Write back
                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()

                        print(f"GPUCoordinator: Allocated GPUs {allocated_gpus} to {eval_id}")
                        return allocated_gpus
                    else:
                        print(f"GPUCoordinator: Insufficient GPUs for {eval_id} (need {count}, available {len(available)})")
                        return None

                finally:
                    self._release_lock(f)

        except Exception as e:
            print(f"GPUCoordinator: Error allocating GPUs for {eval_id}: {e}")
            return None

    def deallocate(self, eval_id: str):
        """
        Deallocate GPUs from an evaluation.

        Args:
            eval_id: Evaluation identifier
        """
        if self.coord_file:
            self._deallocate_file_based(eval_id)
        else:
            self._deallocate_memory_based(eval_id)

    def _deallocate_memory_based(self, eval_id: str):
        """In-memory deallocation (original behavior)."""
        with self.lock:
            if eval_id in self.allocated:
                gpu_ids = self.allocated.pop(eval_id)
                self.available.extend(gpu_ids)
                self.available.sort()

    def _deallocate_file_based(self, eval_id: str):
        """File-based deallocation for multi-station coordination."""
        allocation_key = f"{self.station_id}:{eval_id}"

        try:
            # Check file exists before opening
            if not os.path.exists(self.coord_file):
                return  # Nothing to deallocate if file doesn't exist

            with open(self.coord_file, 'r+') as f:
                self._acquire_lock(f)

                try:
                    data = json.load(f)

                    if allocation_key in data.get("allocations", {}):
                        info = data["allocations"][allocation_key]
                        gpu_ids = info.get("gpus", [])
                        duration = time.time() - info.get("start_time", time.time())

                        del data["allocations"][allocation_key]
                        data["last_updated"] = time.time()
                        data["last_updated_str"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Write back
                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()

                        print(f"GPUCoordinator: Deallocated GPUs {gpu_ids} from {eval_id} (duration: {duration:.1f}s)")
                    # Silently ignore if no allocation found (may have been cleaned up already)

                finally:
                    self._release_lock(f)

        except Exception as e:
            print(f"GPUCoordinator: Error deallocating GPUs for {eval_id}: {e}")

    def get_status(self) -> Dict:
        """
        Get current GPU allocation status.

        Returns:
            Dictionary with allocation information
        """
        if self.coord_file:
            try:
                if not os.path.exists(self.coord_file):
                    return {"allocations": {}, "error": "Coordination file not found"}

                with open(self.coord_file, 'r') as f:
                    self._acquire_lock(f)
                    try:
                        data = json.load(f)
                        return data
                    finally:
                        self._release_lock(f)
            except Exception as e:
                print(f"GPUCoordinator: Error reading status: {e}")
                return {}
        else:
            with self.lock:
                return {
                    "allocations": {
                        eval_id: {"gpus": gpus, "station_id": self.station_id}
                        for eval_id, gpus in self.allocated.items()
                    },
                    "available": self.available.copy(),
                    "mode": "in-memory"
                }

    def cleanup_stale_allocations(self, max_age_seconds: float = 3600):
        """
        Clean up allocations older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds before considering allocation stale
        """
        if not self.coord_file or not os.path.exists(self.coord_file):
            return

        try:
            with open(self.coord_file, 'r+') as f:
                self._acquire_lock(f)

                try:
                    data = json.load(f)
                    current_time = time.time()
                    cleaned_allocations = {}
                    removed_count = 0

                    for key, info in data.get("allocations", {}).items():
                        age = current_time - info.get("start_time", current_time)
                        if age > max_age_seconds:
                            removed_count += 1
                            print(f"GPUCoordinator: Removing stale allocation {key} "
                                  f"(age: {age:.1f}s, GPUs: {info.get('gpus')})")
                        else:
                            cleaned_allocations[key] = info

                    if removed_count > 0:
                        data["allocations"] = cleaned_allocations
                        data["last_updated"] = current_time
                        data["last_updated_str"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()

                        print(f"GPUCoordinator: Cleaned up {removed_count} stale allocations")

                finally:
                    self._release_lock(f)

        except Exception as e:
            print(f"GPUCoordinator: Error cleaning stale allocations: {e}")