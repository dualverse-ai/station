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
CPU Coordinator for managing CPU allocation across multiple stations.

This module provides both file-based and in-memory CPU allocation tracking
to enable multiple stations to share CPU cores without conflicts.
"""

import os
import json
import time
import fcntl
import threading
from typing import Dict, List, Optional
from datetime import datetime


class CPUCoordinator:
    """
    Manages CPU allocation using either file-based coordination (for multi-station)
    or in-memory tracking (for single station).

    File format:
    {
        "allocations": {
            "station_uuid:eval_id": {
                "cpus": [0, 1],
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
                 available_cpus: Optional[List[int]] = None,
                 station_id: Optional[str] = None):
        """
        Initialize CPU coordinator.

        Args:
            coord_file_path: Path to coordination file, None for in-memory mode
            available_cpus: List of CPU IDs available for allocation
            station_id: Unique station identifier
        """
        self.coord_file = coord_file_path
        self.total_cpus = available_cpus or []
        self.station_id = station_id or "unknown"
        self.lock_timeout = 5.0  # Hardcoded timeout for file lock acquisition

        if coord_file_path:
            # File-based coordination
            print(f"CPUCoordinator: File-based coordination at {coord_file_path} (station: {self.station_id})")
            self._ensure_coord_file_exists()
            self._cleanup_station_allocations()
        else:
            # In-memory coordination (single station mode)
            print("CPUCoordinator: Using in-memory CPU tracking (single station mode)")
            self.lock = threading.Lock()
            self.allocated: Dict[str, List[int]] = {}
            self.available = self.total_cpus.copy()

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
            print(f"CPUCoordinator: Created new coordination file at {self.coord_file}")

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

                        print(f"CPUCoordinator: Cleaned up {removed_count} stale allocations from previous session")

                finally:
                    self._release_lock(f)

        except Exception as e:
            print(f"CPUCoordinator: Error during cleanup: {e}")

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
        Allocate CPUs for an evaluation.

        Args:
            eval_id: Evaluation identifier
            count: Number of CPUs to allocate

        Returns:
            List of allocated CPU IDs or None if unavailable
        """
        if self.coord_file:
            return self._allocate_file_based(eval_id, count)
        else:
            return self._allocate_memory_based(eval_id, count)

    def _allocate_memory_based(self, eval_id: str, count: int) -> Optional[List[int]]:
        """In-memory allocation."""
        with self.lock:
            if len(self.available) >= count:
                allocated_cpu_ids = []
                for _ in range(count):
                    cpu_id = self.available.pop(0)
                    allocated_cpu_ids.append(cpu_id)

                self.allocated[eval_id] = allocated_cpu_ids
                return allocated_cpu_ids
            else:
                return None

    def _allocate_file_based(self, eval_id: str, count: int) -> Optional[List[int]]:
        """File-based allocation for multi-station coordination."""
        if not self.coord_file:
            return None

        try:
            with open(self.coord_file, 'r+') as f:
                self._acquire_lock(f)

                try:
                    content = f.read()
                    if not content:
                        data = {"allocations": {}}
                    else:
                        f.seek(0)
                        data = json.load(f)

                    allocated_cpus = []
                    used_cpus = set()

                    # Track used CPUs
                    for info in data.get("allocations", {}).values():
                        used_cpus.update(info.get("cpus", []))

                    # Find available CPUs
                    available = [cpu for cpu in self.total_cpus if cpu not in used_cpus]

                    if len(available) >= count:
                        allocated_cpus = available[:count]

                        allocation_key = f"{self.station_id}:{eval_id}"
                        data.setdefault("allocations", {})[allocation_key] = {
                            "cpus": allocated_cpus,
                            "station_id": self.station_id,
                            "eval_id": eval_id,
                            "start_time": time.time(),
                            "start_time_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        data["last_updated"] = time.time()
                        data["last_updated_str"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()

                        print(f"CPUCoordinator: Allocated CPUs {allocated_cpus} to {eval_id}")
                        return allocated_cpus
                    else:
                        print(f"CPUCoordinator: Insufficient CPUs for {eval_id} (need {count}, available {len(available)})")
                        return None

                finally:
                    self._release_lock(f)

        except Exception as e:
            print(f"CPUCoordinator: Error allocating CPUs for {eval_id}: {e}")
            return None

    def deallocate(self, eval_id: str):
        """Deallocate CPUs from evaluation."""
        if self.coord_file:
            self._deallocate_file_based(eval_id)
        else:
            self._deallocate_memory_based(eval_id)

    def _deallocate_memory_based(self, eval_id: str):
        """In-memory deallocation."""
        with self.lock:
            if eval_id in self.allocated:
                cpu_ids = self.allocated.pop(eval_id)
                self.available.extend(cpu_ids)
                self.available.sort()

    def _deallocate_file_based(self, eval_id: str):
        """File-based deallocation."""
        if not self.coord_file:
            return

        try:
            with open(self.coord_file, 'r+') as f:
                self._acquire_lock(f)

                try:
                    content = f.read()
                    if not content:
                        return

                    f.seek(0)
                    data = json.load(f)

                    allocation_key = f"{self.station_id}:{eval_id}"
                    if allocation_key in data.get("allocations", {}):
                        cpu_ids = data["allocations"][allocation_key].get("cpus", [])
                        start_time = data["allocations"][allocation_key].get("start_time", time.time())
                        duration = time.time() - start_time

                        del data["allocations"][allocation_key]
                        data["last_updated"] = time.time()
                        data["last_updated_str"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()

                        print(f"CPUCoordinator: Deallocated CPUs {cpu_ids} from {eval_id} (duration: {duration:.1f}s)")

                finally:
                    self._release_lock(f)

        except Exception as e:
            print(f"CPUCoordinator: Error deallocating CPUs for {eval_id}: {e}")

    def cleanup_stale_allocations(self, stale_run_seconds: float):
        """Remove allocations that have been held longer than stale_run_seconds."""
        if not self.coord_file:
            return

        try:
            with open(self.coord_file, 'r+') as f:
                self._acquire_lock(f)

                try:
                    content = f.read()
                    if not content:
                        return

                    f.seek(0)
                    data = json.load(f)
                    allocations = data.get("allocations", {})

                    current_time = time.time()
                    cleaned_allocations = {}
                    removed_count = 0

                    for key, info in allocations.items():
                        start_time = info.get("start_time", current_time)
                        if (current_time - start_time) > stale_run_seconds:
                            removed_count += 1
                            print(f"CPUCoordinator: Removing stale allocation {key} "
                                  f"(age: {(current_time - start_time)/3600:.1f}h)")
                        else:
                            cleaned_allocations[key] = info

                    if removed_count > 0:
                        data["allocations"] = cleaned_allocations
                        data["last_updated"] = time.time()
                        data["last_updated_str"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()

                        print(f"CPUCoordinator: Cleaned up {removed_count} stale allocations")

                finally:
                    self._release_lock(f)

        except Exception as e:
            print(f"CPUCoordinator: Error cleaning stale allocations: {e}")
