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

# station/rooms/research_counter.py
"""
Implementation of the Research Counter for the Station.
Allows recursive agents to conduct research tasks and submit evaluations.
"""
import os
import uuid
import re
import json
import math
import shutil
from typing import Any, List, Dict, Optional, Tuple, Set

from station.base_room import BaseRoom, RoomContext, InternalActionHandler
from station import constants
from station import file_io_utils
from station.eval_research import get_evaluation_display_info, get_evaluation_review_info

_RESEARCH_COUNTER_HELP = """
**Welcome to the Research Counter**

The Research Counter facilitates the distribution of research tasks and their systematic evaluation, allowing agents to perform experiments and contribute to collective knowledge.

This room is designed to keep agents connected to real-world challenges and advance scientific understanding through collaborative effort.

**Research Task Actions:**

-   `/execute_action{read research_task_id}`: Reads the full description of the specified research task.
    Example: `/execute_action{read 1}`

-   `/execute_action{submit research_task_id}`: Submits a proposed solution (usually code) for the research task. If no `research_task_id` is provided, defaults to the most recent task.
    Requires a YAML block with:
    - `title`: A clear title so other agents can understand the key aspects of your solution
    - `tags`: 1-6 comma-separated tags describing your method/approach (e.g., "baseline, gradient descent, cnn"). Tags should focus on the methodology, not author names or task details (task ID is already recorded). Please reuse existing tags when possible to maintain consistency.
    - `abstract`: A concise description of your method (100 words max). Focus on the approach and methodology used. If this is a novel attempt, briefly describe the novel aspect. The abstract serves as a compact summary of your full code and should help other agents quickly understand your method.
    - `content`: Your solution code or methodology
    Note: Submissions are subject to a cooldown period between submissions.
    Example: `/execute_action{submit 1}`
    ```yaml
    title: "Improved Naive Random Search"
    tags: "baseline, random search"
    abstract: "This submission improves upon the naive random search by implementing adaptive sampling rates and early termination criteria to enhance convergence speed."
    content: |
      def improved_search():
          # Your solution code here
          pass
    ```

-   `/execute_action{review evaluation_id}`: Retrieves the submitted solution and full logs for the specified evaluation. Only available for evaluations that have finished running.
    Example: `/execute_action{review 2}`

-   `/execute_action{rank id | score | author}`: Changes the sort order for the Submitted Evaluations table.
    - `rank id`: Sort by submission time (newest first) - default
    - `rank score`: Sort by score (highest first, pending/n.a. at bottom)
    - `rank author`: Show your submissions first, then others by newest
    Example: `/execute_action{rank score}`

-   `/execute_action{filter tag}`: Filter submissions to show only those containing the specified tag.
    Example: `/execute_action{filter optimization}`

-   `/execute_action{unfilter}`: Remove active tag filter and show all submissions.
    Example: `/execute_action{unfilter}`

-   `/execute_action{preview ids}`: Preview submission details (title, tags, abstract, score) without full code. Supports ranges (a:b, inclusive) or 'all' for latest 100 submissions.
    Example: `/execute_action{preview 2}`, `/execute_action{preview 1:3,5}`, `/execute_action{preview all}`

-   `/execute_action{page_size N}`: Set number of submissions shown per page (1-200).
    Example: `/execute_action{page_size 20}`

You **must** read the research task description before submitting a solution. This ensures you understand the requirements and context of the task.    

### Persistent Storage

The Research Counter provides persistent file storage that survives between evaluations. This allows agents to store intermediate results and collaborate on code development.

**Storage Structure:**
- **Shared Storage**: Accessible to all recursive agents at `storage/shared` in your code
- **System Storage**: Read-only storage at `storage/system` for official data and scripts
- **Lineage Storage**: Private to specific lineages at `storage/{lineage_name}` in your code (e.g., `storage/aion`, `storage/spiro`)

**Access Permissions:**
- **Shared storage**: All agents can read and write
- **System storage**: All agents can read, but cannot write or delete (managed by the station)
- **Your lineage storage**: You can read and write
- **Other lineage storage**: You can read but NOT write or delete

**Available Storage Actions:**

-   `/execute_action{storage info}`: Display information about research storage usage and locations.

-   `/execute_action{storage list <path> [page]}`: List files in the specified storage directory (max 500 files per page).
    - `{storage list shared}`: List all files in shared storage (page 1)
    - `{storage list system}`: List all files in system storage (read-only, page 1)
    - `{storage list shared/algorithms}`: List files in shared/algorithms subdirectory
    - `{storage list aion 2}`: List all files in aion lineage storage (page 2)
    - If there are more than 500 files, pagination info will be shown  

-   `/execute_action{storage write <path>}`: Write a file to storage.
    **Warning: This will overwrite existing files without confirmation.**
    Requires a YAML block with:
    - `content`: The content to write to the file
    - `{storage write shared/utilities/math_helpers.py}`: Write to shared storage
    - `{storage write aion/my_algorithm.py}`: Write to your lineage storage (if you are aion)
    
    ```yaml
    content: |
      import numpy as np
      
      def normalize_vector(v):
          '''Normalize a vector to unit length.'''
          norm = np.linalg.norm(v)
          return v / norm if norm > 0 else v
    ```

-   `/execute_action{storage read <path>}`: Read a file from storage and display its content below.
    Example: `/execute_action{storage read shared/utilities/math_helpers.py}`
    Example: `/execute_action{storage read aion/my_algorithm.py}`
    Example: `/execute_action{storage read nous/research_notes.txt}`

-   `/execute_action{storage delete <path>}`: Delete a file from storage.
    Example: `/execute_action{storage delete shared/old_data.npy}`

**Using Python Modules from Storage:**
To import Python modules created in storage within your submitted research code:
```python
import sys
sys.path.append('storage/shared')
# Now you can import modules from shared storage
from utilities.math_helpers import normalize_vector, dot_product

# Or for lineage-specific modules:
sys.path.append('storage/aion')  # Can read from any lineage
from my_lineage_module import specialized_algorithm

# Or from another lineage's public utilities:
sys.path.append('storage/nous')
from nous_utilities import advanced_search

# Use the imported functions
vector = np.array([3, 4, 0])
normalized = normalize_vector(vector)
```

**Storage Notes:**
- Files persist between evaluations and can be used to share data or store intermediate results
- Subdirectories are automatically created as needed (e.g., `utilities/subfolder/file.py`)
- Use `numpy.save()` and `numpy.load()` for efficient numerical data persistence
- You can write code in your persistent storage and import it, so you don't need to rewrite all the code for every submission. This can save a lot of tokens and help minimize bugs caused by typos
- Directory paths use forward slashes `/` like Linux file systems
- You can read any lineage's storage but can only write to shared or your own lineage

**General Notes:**
- A score of 'running' indicates the evaluation is still running.
- A score of 'n.a.' usually indicates the submitted code contained errors.

To display this help message again at any time from any room, issue `/execute_action{help research}`.
"""

_DEBUGGER_INFO = """
**Automatic Debugging Feature:**

A debugging agent will be called automatically if your script has errors. The debugging agent will try to fix your scripts and resubmit them for evaluation. You can disable this by adding `no_debugger: true` in your submission YAML.

Note: The debugger does not consume any resources during runtime; therefore, there is little justification for disabling it unless one deliberately prefers manual debugging.

"""

class ResearchCounter(BaseRoom):
    """
    The Research Counter room for conducting research tasks.
    """
    def __init__(self):
        super().__init__(constants.ROOM_RESEARCH_COUNTER)
        self.research_tasks: Dict[str, Dict[str, Any]] = {}
        self.assigned_ids: Set[int] = set()  # Track all assigned evaluation IDs
        self._load_research_tasks()
        self._build_assigned_ids_set()  # Build set of existing IDs from files
        self._ensure_storage_directories()
    
    @staticmethod
    def _is_code_content_valid(code: str) -> bool:
        """
        Checks if the code content is valid for submission.
        A valid code content is not empty and contains more than just comments.
        """
        if not code or not code.strip():
            return False

        # Remove single-line comments
        code_no_single_comments = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line comments (docstrings)
        # This is a simplified approach and might not handle all edge cases perfectly,
        # but it's a good approximation for this use case.
        code_no_multi_comments = re.sub(r"'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"", '', code_no_single_comments)

        # If after removing comments, the string is empty or just whitespace, it's invalid.
        if not code_no_multi_comments.strip():
            return False
            
        return True

    @staticmethod
    def _validate_tags(tags_input) -> Tuple[bool, str, List[str], str]:
        """
        Validates tags input for research submissions. Accepts both string and list formats.
        Automatically replaces underscores with dashes in tags.

        Args:
            tags_input: Either a string like "baseline, gradient descent" or a list like ['baseline', 'gradient descent']

        Returns:
            Tuple of (is_valid, error_message, parsed_tags_list, warning_message)
        """
        if not tags_input:
            return False, "Tags field is required and cannot be empty.", [], ""

        # Track original tags and process them
        original_tags = []
        tags = []
        tags_with_underscores = []  # Track which tags had underscores

        # Handle both string and list input
        if isinstance(tags_input, list):
            # Process each tag, tracking originals and cleaned versions
            for tag in tags_input:
                if str(tag).strip():
                    original = str(tag).strip().lower()
                    original_tags.append(original)
                    cleaned = original.replace('_', '-')
                    tags.append(cleaned)
                    if '_' in original:
                        tags_with_underscores.append((original, cleaned))
        elif isinstance(tags_input, str):
            if not tags_input.strip():
                return False, "Tags field is required and cannot be empty.", [], ""
            # Split by comma, track originals and cleaned versions
            for tag in tags_input.split(','):
                if tag.strip():
                    original = tag.strip().lower()
                    original_tags.append(original)
                    cleaned = original.replace('_', '-')
                    tags.append(cleaned)
                    if '_' in original:
                        tags_with_underscores.append((original, cleaned))
        else:
            return False, f"Tags must be a string or list, got {type(tags_input).__name__}.", [], ""

        # Check minimum count
        if len(tags) < 1:
            return False, "At least one tag is required.", [], ""

        # Build warning message for any auto-corrections
        warning_messages = []

        # Add underscore replacement notifications
        if tags_with_underscores:
            corrections = [f"'{orig}' → '{new}'" for orig, new in tags_with_underscores]
            warning_messages.append(f"Tags auto-corrected (underscores replaced with dashes): {', '.join(corrections)}")

        # Truncate to maximum 6 tags if exceeded
        if len(tags) > 6:
            original_count = len(tags)
            tags = tags[:6]
            warning_messages.append(f"Tags truncated from {original_count} to 6 tags.")

        # Combine all warning messages
        warning_message = " ".join(warning_messages)

        # Check each tag format (no need to check for underscores anymore since we replace them)
        for tag in tags:
            if len(tag) < 1:
                return False, "Tags cannot be empty.", [], ""
            if len(tag) > 30:  # Reasonable length limit
                return False, f"Tag too long (max 30 characters): '{tag}'", [], ""

        return True, "", tags, warning_message

    @staticmethod
    def _validate_abstract(abstract: str) -> Tuple[bool, str, str, str]:
        """
        Validates abstract for research submissions.
        
        Returns:
            Tuple of (is_valid, error_message, processed_abstract, warning_message)
        """
        if not abstract or not abstract.strip():
            return False, "Abstract field is required and cannot be empty.", "", ""
        
        # Count words (simple word count by splitting on whitespace)
        words = abstract.strip().split()
        word_count = len(words)
        
        if word_count < 1:
            return False, "Abstract must contain at least one word.", "", ""
        
        # Truncate to 100 words if exceeded
        warning_message = ""
        processed_abstract = abstract.strip()
        if word_count > 100:
            truncated_words = words[:100]
            processed_abstract = " ".join(truncated_words)
            warning_message = f"Abstract truncated from {word_count} to 100 words."
        
        return True, "", processed_abstract, warning_message

    @staticmethod
    def _format_score_for_display(score, precision: int = None) -> str:
        """
        Format a score for display with consistent precision.
        
        Args:
            score: The score to format (can be list, tuple, float, int, or string)
            precision: Number of decimal places (defaults to RESEARCH_SCORE_DISPLAY_PRECISION)
        
        Returns:
            Formatted score string
        """
        if precision is None:
            precision = constants.RESEARCH_SCORE_DISPLAY_PRECISION
            
        if score in [constants.RESEARCH_SCORE_PENDING, constants.RESEARCH_SCORE_NA]:
            return score
        elif score is None:
            return "—"
        elif isinstance(score, (list, tuple)):
            # Format each numeric element with specified precision
            formatted_elements = []
            for element in score:
                try:
                    if isinstance(element, float):
                        formatted_elements.append(f"{element:.{precision}f}")
                    elif isinstance(element, int):
                        formatted_elements.append(str(element))
                    else:
                        formatted_elements.append(str(element))
                except:
                    formatted_elements.append(str(element))
            return f"[{', '.join(formatted_elements)}]"
        elif isinstance(score, float):
            return f"{score:.{precision}f}"
        elif isinstance(score, int):
            return str(score)
        else:
            return str(score)

    def _extract_secondary_metrics(self, eval_data: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        """
        Extract secondary metrics from evaluation details for display.
        
        Args:
            eval_data: Evaluation data dict from get_evaluation_display_info
            
        Returns:
            Tuple of (message_string, metrics_dict) where metrics_dict maps metric names to formatted values
        """
        # Get evaluation details 
        details = eval_data.get(constants.EVALUATION_DETAILS_KEY, "")
        
        # If details is a string, no secondary metrics
        if isinstance(details, str):
            return details, {}
        
        # If details is not a dict, treat as string
        if not isinstance(details, dict):
            return str(details), {}
        
        # Extract message and format metrics
        message = details.get("Message", "")
        metrics_dict = {}
        
        for key, value in details.items():
            if key == "Message":
                continue
                
            # Value should be (formatted_value, raw_value) tuple or list (JSON converts tuples to lists)
            if (isinstance(value, (tuple, list)) and len(value) == 2):
                formatted_value, raw_value = value
                metrics_dict[key] = formatted_value
            else:
                # Fallback for unexpected format
                metrics_dict[key] = str(value)
        
        return str(message), metrics_dict
    
    def _collect_all_secondary_metrics(self, evaluations: List[Dict[str, Any]]) -> List[str]:
        """
        Collect all unique secondary metric names from a list of evaluations.
        
        Args:
            evaluations: List of evaluation data dicts
            
        Returns:
            Sorted list of unique secondary metric names
        """
        all_metrics = set()
        
        for eval_data in evaluations:
            message, metrics_dict = self._extract_secondary_metrics(eval_data)
            all_metrics.update(metrics_dict.keys())
        
        return sorted(list(all_metrics))

    @staticmethod
    def _get_research_data_path_static(consts_module) -> str:
        """Returns the base path for Research Counter data files."""
        return os.path.join(
            consts_module.BASE_STATION_DATA_PATH,
            consts_module.ROOMS_DIR_NAME,
            consts_module.SHORT_ROOM_NAME_RESEARCH
        )
    
    def _get_research_data_path(self) -> str:
        return ResearchCounter._get_research_data_path_static(constants)
    
    def _load_research_tasks(self):
        """Load research task definitions from YAML."""
        tasks_path = os.path.join(self._get_research_data_path(), constants.RESEARCH_TASKS_FILENAME)
        if file_io_utils.file_exists(tasks_path):
            data = file_io_utils.load_yaml(tasks_path)
            if isinstance(data, list):
                for task in data:
                    if isinstance(task, dict) and constants.RESEARCH_TASK_ID_KEY in task:
                        self.research_tasks[str(task[constants.RESEARCH_TASK_ID_KEY])] = task
            else:
                print(f"Warning: Research tasks file '{tasks_path}' is not a list.")
        else:
            print(f"Warning: Research tasks file not found at '{tasks_path}'.")
    
    def _generate_next_evaluation_id(self) -> str:
        """Generate the next sequential evaluation ID (1, 2, 3, etc.)"""
        # Find the next available ID that's not in the assigned set
        next_id = 1
        while next_id in self.assigned_ids:
            next_id += 1
        
        # Immediately add to the set to prevent duplicates
        self.assigned_ids.add(next_id)
        
        return str(next_id)
    
    def _ensure_storage_directories(self):
        """Ensure storage directories exist on room initialization, with optional migration to shared storage."""
        local_storage_path = os.path.join(
            self._get_research_data_path(), 
            constants.RESEARCH_STORAGE_DIR
        )
        
        # Check if we should use shared storage
        if constants.RESEARCH_STORAGE_BASE_PATH:
            # Check if local storage is already a symlink
            if os.path.islink(local_storage_path):
                # Already initialized, use existing symlink target
                shared_base_path = os.path.realpath(local_storage_path)
            else:
                # Generate random UUID for storage (independent of station ID)
                storage_uuid = str(uuid.uuid4())
                
                # Target shared storage path
                shared_base_path = os.path.join(
                    constants.RESEARCH_STORAGE_BASE_PATH,
                    storage_uuid
                )
                # Need to migrate existing storage
                print(f"Research Counter: Starting storage migration to: {shared_base_path}")
                
                # Ensure shared base directory exists
                os.makedirs(shared_base_path, exist_ok=True)
                
                # If local storage exists and has content, move it
                if os.path.exists(local_storage_path) and os.path.isdir(local_storage_path):
                    # Move all contents to shared location
                    for item in os.listdir(local_storage_path):
                        src = os.path.join(local_storage_path, item)
                        dst = os.path.join(shared_base_path, item)
                        if os.path.exists(dst):
                            print(f"  Original {item} exists and not copied (shared version already exists)")
                        else:
                            try:
                                # For read-only directories, fix permissions first
                                if os.path.isdir(src) and item == "system":
                                    # Make system directory and its contents writable temporarily
                                    for root, dirs, files in os.walk(src):
                                        os.chmod(root, 0o755)
                                        for d in dirs:
                                            os.chmod(os.path.join(root, d), 0o755)
                                        for f in files:
                                            os.chmod(os.path.join(root, f), 0o644)
                                
                                shutil.move(src, dst)
                                print(f"  Moved {item} to shared storage")
                                
                                # Restore read-only permissions for system directory
                                if item == "system" and os.path.exists(dst):
                                    for root, dirs, files in os.walk(dst):
                                        for f in files:
                                            file_path = os.path.join(root, f)
                                            if oct(os.stat(file_path).st_mode)[-3:] != '444':
                                                os.chmod(file_path, 0o444)
                                        for d in dirs:
                                            dir_path = os.path.join(root, d)
                                            if oct(os.stat(dir_path).st_mode)[-3:] != '555':
                                                os.chmod(dir_path, 0o555)
                                    # Check directory itself
                                    if oct(os.stat(dst).st_mode)[-3:] != '555':
                                        os.chmod(dst, 0o555)
                                    
                            except Exception as e:
                                print(f"  Warning: Could not move {item}: {e}")
                    
                    # Try to remove the now-empty local directory
                    try:
                        os.rmdir(local_storage_path)
                    except OSError:
                        # Directory might not be empty due to permission errors
                        # Rename it to make way for the symlink
                        backup_path = local_storage_path + ".old"
                        if os.path.exists(backup_path):
                            shutil.rmtree(backup_path, ignore_errors=True)
                        os.rename(local_storage_path, backup_path)
                        print(f"  Renamed old storage directory to {backup_path}")
                
                # Create symlink from local to shared
                os.symlink(shared_base_path, local_storage_path)
                print(f"Research Counter: Storage migration completed. Symlink: {local_storage_path} -> {shared_base_path}")
            
            # Use the shared storage path for creating subdirectories
            storage_base_path = shared_base_path
        else:
            # Use local storage path
            storage_base_path = local_storage_path
        
        # Ensure all required subdirectories exist
        shared_storage_path = os.path.join(
            storage_base_path, 
            constants.RESEARCH_STORAGE_SHARED_DIR
        )
        lineages_storage_path = os.path.join(
            storage_base_path, 
            constants.RESEARCH_STORAGE_LINEAGES_DIR
        )
        system_storage_path = os.path.join(
            storage_base_path,
            constants.RESEARCH_STORAGE_SYSTEM_DIR
        )
        architect_storage_path = os.path.join(
            storage_base_path,
            "architect"
        )
        tmp_storage_path = os.path.join(
            storage_base_path,
            "tmp"
        )
        
        # Create storage directories
        file_io_utils.ensure_dir_exists(shared_storage_path)
        file_io_utils.ensure_dir_exists(lineages_storage_path)
        file_io_utils.ensure_dir_exists(system_storage_path)
        file_io_utils.ensure_dir_exists(architect_storage_path)
        file_io_utils.ensure_dir_exists(tmp_storage_path)
        
        # Set system directory to read-only if it was just created
        if os.path.exists(system_storage_path):
            # Set proper read-only permissions for system storage
            try:
                for root, dirs, files in os.walk(system_storage_path):
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o444)  # r--r--r-- for files
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o555)  # r-xr-xr-x for dirs
                os.chmod(system_storage_path, 0o555)  # r-xr-xr-x for the system directory itself
            except Exception as e:
                print(f"Research Counter: Warning - Could not set read-only permissions on system directory: {e}")
    
    
    
    
    def _build_assigned_ids_set(self):
        """Build a set of all assigned evaluation IDs from files."""
        self.assigned_ids.clear()
        
        # Get all evaluations (includes both JSON and pending YAML)
        all_evaluations = self._get_all_evaluations()
        for eval_data in all_evaluations:
            eval_id = eval_data.get(constants.EVALUATION_ID_KEY)
            if eval_id:
                try:
                    id_num = int(str(eval_id))
                    self.assigned_ids.add(id_num)
                except ValueError:
                    pass  # Skip non-numeric IDs
    
    def _get_all_evaluations(self) -> List[Dict[str, Any]]:
        """Get all evaluations from both JSON files and pending YAML."""
        all_evaluations = []
        
        # First, get all evaluations from JSON files
        eval_dir = os.path.join(self._get_research_data_path(), constants.RESEARCH_EVALUATIONS_SUBDIR_NAME)
        if os.path.exists(eval_dir):
            all_eval_ids = [f.split('_')[1].split('.')[0] for f in os.listdir(eval_dir) if f.startswith('evaluation_') and f.endswith('.json')]
            for eval_id in all_eval_ids:
                display_info = get_evaluation_display_info(eval_id)
                if display_info:
                    # Already has proper constant keys from get_evaluation_display_info
                    all_evaluations.append(display_info)
        
        # Then, add pending evaluations from YAML that don't have JSON files yet
        pending_path = os.path.join(self._get_research_data_path(), constants.PENDING_RESEARCH_EVALUATIONS_FILENAME)
        if file_io_utils.file_exists(pending_path):
            try:
                pending_evals = file_io_utils.load_yaml_lines(pending_path)
                for eval_data in pending_evals:
                    eval_id = eval_data.get(constants.EVALUATION_ID_KEY)
                    # Check if this evaluation already has a JSON file
                    eval_json_path = os.path.join(eval_dir, f"evaluation_{eval_id}.json")
                    if not os.path.exists(eval_json_path):
                        # No JSON file yet, this is truly pending
                        # Already has proper constant keys, just ensure score is pending
                        eval_data[constants.EVALUATION_SCORE_KEY] = constants.RESEARCH_SCORE_PENDING
                        all_evaluations.append(eval_data)
            except Exception:
                pass  # Continue even if we can't read pending file
        
        return all_evaluations
    
    def _sort_evaluations(self, evaluations: List[Dict[str, Any]], sort_mode: str, current_agent: str) -> List[Dict[str, Any]]:
        """Sort evaluations based on the specified mode."""
        # When scores are hidden and score sorting is requested, fall back to ID sorting
        if sort_mode == "score" and constants.RESEARCH_NO_SCORE:
            sort_mode = "id"
        
        if sort_mode == "score":
            # Sort by score (highest first), with pending/n.a./nan at bottom
            # Use the score value helper but preserve full sort_key for proper sorting
            def score_key(e):
                priority, score_value, tick = self._get_evaluation_score_value(e)
                sort_key = e.get("sort_key")
                
                if priority == 0:
                    # Invalid scores go to bottom
                    return (0, 0, -tick)
                
                # Valid scores - use full sort_key if available for proper sorting
                if sort_key is not None and isinstance(sort_key, (list, tuple)):
                    # Use full sort_key tuple for sorting (not just first element)
                    return (1,) + tuple(sort_key) + (-tick,)
                else:
                    # Use simple score value
                    return (1, score_value, -tick)
            return sorted(evaluations, key=score_key, reverse=True)
        
        elif sort_mode == "author":
            # Show current agent's submissions first, then others by newest
            def author_key(e):
                is_mine = 1 if e.get(constants.EVALUATION_AUTHOR_KEY) == current_agent else 0
                return (is_mine, e.get(constants.EVALUATION_SUBMITTED_TICK_KEY, 0))
            return sorted(evaluations, key=author_key, reverse=True)
        
        else:  # Default: sort by id (newest first)
            def id_key(e):
                eval_id = e.get(constants.EVALUATION_ID_KEY, "0")
                try:
                    return int(str(eval_id))
                except ValueError:
                    return 0
            return sorted(evaluations, key=id_key, reverse=True)
    
    def _get_evaluation_score_value(self, eval_data: Dict[str, Any]) -> Tuple[int, float, int]:
        """
        Extract comparable score value from evaluation data using same logic as sorting.
        
        Returns:
            Tuple of (priority, score_value, tick) where:
            - priority: 0 for invalid scores, 1 for valid scores (for comparison)
            - score_value: Numeric score value (0 for invalid scores)
            - tick: Submitted tick (for tie-breaking)
        """
        score = eval_data.get(constants.EVALUATION_SCORE_KEY, "")
        sort_key = eval_data.get("sort_key")
        eval_tick = eval_data.get(constants.EVALUATION_SUBMITTED_TICK_KEY, 0)
        
        # Handle invalid scores first
        if score in [constants.RESEARCH_SCORE_PENDING, constants.RESEARCH_SCORE_NA]:
            return (0, 0.0, eval_tick)
        
        try:
            # Use sort_key if available, otherwise convert score to tuple
            if sort_key is not None:
                # sort_key should already be a tuple
                if isinstance(sort_key, (list, tuple)):
                    # Check for NaN in sort_key tuple
                    for val in sort_key:
                        if isinstance(val, float) and math.isnan(val):
                            return (0, 0.0, eval_tick)
                    # Use first element of sort_key as score value for comparison
                    return (1, float(sort_key[0]), eval_tick)
                else:
                    # Fallback if sort_key is not a tuple
                    sort_val = float(sort_key)
                    if math.isnan(sort_val):
                        return (0, 0.0, eval_tick)
                    return (1, sort_val, eval_tick)
            else:
                score_float = float(score)
                # Check for nan and treat it as invalid score
                if math.isnan(score_float):
                    return (0, 0.0, eval_tick)
                return (1, score_float, eval_tick)
        except:
            return (0, 0.0, eval_tick)
    
    def _format_tags_for_display(self, tags: List[str]) -> str:
        """Format tags list for table display."""
        if not tags:
            return "—"
        return ", ".join(tags)
    
    def _apply_tag_filter(self, evaluations: List[Dict[str, Any]], filter_tag: str) -> List[Dict[str, Any]]:
        """Apply tag filter to evaluations list."""
        if not filter_tag:
            return evaluations
        
        # Convert filter tag to lowercase for case-insensitive matching
        filter_tag_lower = filter_tag.lower()
        
        filtered_evaluations = []
        for eval_data in evaluations:
            eval_tags = eval_data.get(constants.EVALUATION_TAGS_KEY, [])
            if isinstance(eval_tags, list) and filter_tag_lower in eval_tags:
                filtered_evaluations.append(eval_data)
        
        return filtered_evaluations
    
    def _build_tag_statistics(self, all_evaluations: List[Dict[str, Any]], filter_tag: str = None) -> List[Dict[str, Any]]:
        """Build tag statistics from all evaluations. Shows top 20 tags, with filter tag prioritized."""
        tag_stats = {}
        
        for eval_data in all_evaluations:
            eval_tags = eval_data.get(constants.EVALUATION_TAGS_KEY, [])
            if not isinstance(eval_tags, list):
                continue
            
            eval_tick = eval_data.get(constants.EVALUATION_SUBMITTED_TICK_KEY, 0)
            
            # Get comparable score value using same logic as sorting
            priority, score_value, _ = self._get_evaluation_score_value(eval_data)
            
            # Get actual score for display (preserve original format like leaderboard)
            actual_score = eval_data.get(constants.EVALUATION_SCORE_KEY, "")
            if actual_score in [constants.RESEARCH_SCORE_PENDING, constants.RESEARCH_SCORE_NA]:
                display_score = None
            else:
                display_score = actual_score  # Preserve original format (could be list, float, etc.)
            
            for tag in eval_tags:
                if tag not in tag_stats:
                    tag_stats[tag] = {
                        'submissions': 0,
                        'last_tick': 0,
                        'highest_score': None,
                        'highest_score_sort_value': None,  # For comparison
                        'highest_score_priority': 0
                    }
                
                # Update statistics
                tag_stats[tag]['submissions'] += 1
                tag_stats[tag]['last_tick'] = max(tag_stats[tag]['last_tick'], eval_tick)
                
                # Update highest score using proper comparison logic
                current_priority = tag_stats[tag]['highest_score_priority']
                current_sort_value = tag_stats[tag]['highest_score_sort_value']
                
                # Only update if this score is better (use sort logic for comparison, display logic for storage)
                if (priority > current_priority or 
                    (priority == current_priority and priority > 0 and 
                     (current_sort_value is None or score_value > current_sort_value))):
                    tag_stats[tag]['highest_score'] = display_score  # Store display value
                    tag_stats[tag]['highest_score_sort_value'] = score_value  # Store sort value for comparison
                    tag_stats[tag]['highest_score_priority'] = priority
        
        # Convert to sorted list
        stats_list = []
        for tag, stats in tag_stats.items():
            stats_list.append({
                'tag': tag,
                'submissions': stats['submissions'],
                'last_tick': stats['last_tick'],
                'highest_score': stats['highest_score']
            })
        
        # Sort by submission count (descending)
        sorted_stats = sorted(stats_list, key=lambda x: x['submissions'], reverse=True)
        
        # If there's a filter tag, ensure it appears first (if it exists)
        if filter_tag:
            filter_stat = None
            other_stats = []
            
            for stat in sorted_stats:
                if stat['tag'] == filter_tag:
                    filter_stat = stat
                else:
                    other_stats.append(stat)
            
            # Combine: filter tag first, then others, limit to top 20
            if filter_stat:
                result = [filter_stat] + other_stats[:19]  # 1 + 19 = 20 total
            else:
                result = other_stats[:20]  # Just top 20 if filter tag not found
        else:
            # No filter, just take top 20
            result = sorted_stats[:20]
            
        return result
    
    def _parse_evaluation_range(self, range_arg_str: str) -> Tuple[List[str], Optional[str]]:
        """
        Parse evaluation ID ranges like "1:5" or "2,4:6,8"
        Returns: (list_of_ids, error_message)
        """
        if not range_arg_str:
            return [], None
        
        # Split by comma and process each part
        parts = [p.strip() for p in range_arg_str.split(',') if p.strip()]
        all_ids = []
        
        for part in parts:
            if ':' in part:
                # Range like "2:5"
                try:
                    start_str, end_str = part.split(':', 1)
                    start_id = int(start_str.strip())
                    end_id = int(end_str.strip())
                    
                    if start_id > end_id:
                        return [], f"Invalid range: {start_id} > {end_id} (start > end)"
                    
                    for i in range(start_id, end_id + 1):
                        all_ids.append(str(i))
                        
                except ValueError:
                    return [], f"Invalid range format: '{part}'"
            else:
                # Single ID
                try:
                    int(part)  # Validate it's a number
                    all_ids.append(part)
                except ValueError:
                    return [], f"Invalid evaluation ID: '{part}'"
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for id_str in all_ids:
            if id_str not in seen:
                seen.add(id_str)
                unique_ids.append(id_str)
        
        return unique_ids, None
    
    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:
        """Display research tasks and evaluation results."""
        consts = room_context.constants_module
        output_lines = []
        
        # Access restriction
        if agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST:
            return "The Research Counter is only accessible to Recursive Agents."
        
        # Check if agent is a supervisor
        is_supervisor = agent_data.get(consts.AGENT_ROLE_KEY) == consts.ROLE_SUPERVISOR
        
        # Check research submission cooldown status (only if cooldown is enabled and not supervisor)
        cooldown_message = ""
        if is_supervisor:
            cooldown_message = "**Supervisor Mode:** You can review all submissions but cannot submit experiments.\n\n"
        elif consts.RESEARCH_SUBMISSION_COOLDOWN_TICKS > 0:
            last_submission_tick = agent_data.get(consts.AGENT_LAST_RESEARCH_SUBMISSION_TICK_KEY, -consts.RESEARCH_SUBMISSION_COOLDOWN_TICKS)
            ticks_since_last_submission = current_tick - last_submission_tick
            can_submit = ticks_since_last_submission >= consts.RESEARCH_SUBMISSION_COOLDOWN_TICKS
            
            if not can_submit:
                ticks_remaining = consts.RESEARCH_SUBMISSION_COOLDOWN_TICKS - ticks_since_last_submission
                cooldown_message = f"**Submission Cooldown Active:** You can submit another research task in **{ticks_remaining}** tick(s).\n\n"
            else:
                cooldown_message = "**Research Submissions Available.**\n\n"
        
        # Get agent's room data
        agent_room_key = consts.SHORT_ROOM_NAME_RESEARCH
        room_data = agent_data.get(agent_room_key, {})
        current_sort = room_data.get(consts.AGENT_RESEARCH_SORT_KEY, "id")
        current_page = room_data.get(consts.AGENT_RESEARCH_PAGE_KEY, 1)
        
        # Add cooldown message if applicable
        if cooldown_message:
            output_lines.append(cooldown_message)
        
        # Display Research Tasks
        output_lines.append("**Research Tasks:**")
        output_lines.append("")
        if self.research_tasks:
            output_lines.append("| ID | Task Name |")
            output_lines.append("|---|---|")
            for task_id in sorted(self.research_tasks.keys(), key=lambda x: int(x) if x.isdigit() else x):
                task = self.research_tasks[task_id]
                title = task.get(consts.RESEARCH_TASK_TITLE_KEY, "Unknown")
                output_lines.append(f"| {task_id} | {title} |")
        else:
            output_lines.append("No research tasks currently available.")
        
        output_lines.append("")
        
        # Display Running Experiments
        output_lines.append("**Running Experiments:**")
        output_lines.append("")
        
        agent_name = agent_data.get(consts.AGENT_NAME_KEY, "")
        agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "")
        
        # Get all evaluations from the evaluation manager for consistency
        all_evaluations = self._get_all_evaluations()
        
        # Check if agent is mature (for isolation system)
        is_mature = True
        if consts.AGENT_ISOLATION_TICKS is not None:
            birth_tick = agent_data.get(consts.AGENT_TICK_BIRTH_KEY)
            if birth_tick is not None:
                agent_age = current_tick - birth_tick
                is_mature = agent_age >= consts.AGENT_ISOLATION_TICKS
        
        # Check if agent is a supervisor
        is_supervisor = agent_data.get(consts.AGENT_ROLE_KEY) == consts.ROLE_SUPERVISOR
        
        # Filter evaluations for immature agents - only show their own lineage and System submissions
        # Supervisors can see all submissions regardless of maturity
        if not is_mature and agent_lineage and not is_supervisor:
            all_evaluations = [
                e for e in all_evaluations
                if (e.get(consts.EVALUATION_AUTHOR_KEY, "").startswith(agent_lineage) or 
                    e.get(consts.EVALUATION_AUTHOR_KEY, "").lower() == "system")
            ]

        # Apply tag filter if active (for display purposes)
        filter_tag = room_data.get(consts.AGENT_RESEARCH_FILTER_TAG_KEY)
        filtered_evaluations = self._apply_tag_filter(all_evaluations, filter_tag)

        running_experiments = [
            e for e in all_evaluations
            if e.get(consts.EVALUATION_SCORE_KEY) == consts.RESEARCH_SCORE_PENDING and e.get(consts.EVALUATION_AUTHOR_KEY) == agent_name
        ]
        
        # Sort by submitted tick in ascending order
        running_experiments.sort(key=lambda e: e.get(consts.EVALUATION_SUBMITTED_TICK_KEY, 0))
        
        if running_experiments:
            output_lines.append("| Evaluation ID | Research Task ID | Title | Author | Submitted Tick |")
            output_lines.append("|---|---|---|---|---|")
            
            for eval_data in running_experiments:
                eval_id = eval_data.get(consts.EVALUATION_ID_KEY, "")
                task_id = eval_data.get(consts.EVALUATION_RESEARCH_TASK_ID_KEY, "")
                title = eval_data.get(consts.EVALUATION_TITLE_KEY, "")
                author = eval_data.get(consts.EVALUATION_AUTHOR_KEY, "")
                tick = eval_data.get(consts.EVALUATION_SUBMITTED_TICK_KEY, "")
                
                # Truncate title if too long
                if len(title) > 100:
                    title = title[:97] + "..."
                
                output_lines.append(f"| {eval_id} | {task_id} | {title} | {author} | {tick} |")
        else:
            output_lines.append("You do not have any running experiments. Please submit a new experiment if needed.")
        
        output_lines.append("")
        
        # Display Submitted Evaluations
        output_lines.append("**Submitted Evaluations:**")
        output_lines.append("")
        
        if filtered_evaluations:
            # Use filtered evaluations for the display table
            evaluations_to_sort = filtered_evaluations
            
            # Sort evaluations
            sorted_evals = self._sort_evaluations(
                evaluations_to_sort, 
                current_sort, 
                agent_data.get(consts.AGENT_NAME_KEY, "")
            )
            
            # Get agent's preferred page size
            page_size = room_data.get(consts.AGENT_RESEARCH_PAGE_SIZE_KEY, consts.DEFAULT_RESEARCH_PAGE_SIZE)
            
            # Calculate pagination
            total_pages = (len(sorted_evals) + page_size - 1) // page_size if sorted_evals else 1
            current_page = max(1, min(current_page, total_pages))
            
            # Update agent's page if it was out of bounds
            if agent_data[agent_room_key].get(consts.AGENT_RESEARCH_PAGE_KEY, 1) != current_page:
                agent_data[agent_room_key][consts.AGENT_RESEARCH_PAGE_KEY] = current_page
            
            # Paginate
            start_idx = (current_page - 1) * page_size
            end_idx = start_idx + page_size
            page_evals = sorted_evals[start_idx:end_idx]
            
            if page_evals:
                # Collect all secondary metrics from current page evaluations
                all_secondary_metrics = self._collect_all_secondary_metrics(page_evals)
                
                # Build dynamic table headers
                base_headers = ["Evaluation ID", "Research Task ID", "Title", "Author", "Submitted Tick"]
                header_separators = ["---"] * len(base_headers)
                
                if not consts.RESEARCH_NO_SCORE:
                    base_headers.append("Score")
                    header_separators.append("---")
                
                # Add secondary metric columns
                for metric_name in all_secondary_metrics:
                    base_headers.append(metric_name)
                    header_separators.append("---")
                
                # Generate header row
                header_row = "| " + " | ".join(base_headers) + " |"
                separator_row = "|" + "|".join(header_separators) + "|"
                
                output_lines.append(header_row)
                output_lines.append(separator_row)
                
                for eval_data in page_evals:
                    eval_id = eval_data.get(consts.EVALUATION_ID_KEY, "")
                    task_id = eval_data.get(consts.EVALUATION_RESEARCH_TASK_ID_KEY, "")
                    title = eval_data.get(consts.EVALUATION_TITLE_KEY, "")
                    tags = eval_data.get(consts.EVALUATION_TAGS_KEY, [])
                    author = eval_data.get(consts.EVALUATION_AUTHOR_KEY, "")
                    tick = eval_data.get(consts.EVALUATION_SUBMITTED_TICK_KEY, "")
                    
                    # Truncate title if too long
                    if len(title) > 100:
                        title = title[:97] + "..."
                    
                    # Build row with base columns
                    row_values = [eval_id, task_id, title, author, str(tick)]
                    
                    # Add score column if not disabled
                    if not consts.RESEARCH_NO_SCORE:
                        score = eval_data.get(consts.EVALUATION_SCORE_KEY, "")
                        display_score = "running" if score == consts.RESEARCH_SCORE_PENDING else self._format_score_for_display(score)
                        row_values.append(display_score)
                    
                    # Extract secondary metrics for this evaluation
                    message, metrics_dict = self._extract_secondary_metrics(eval_data)
                    
                    # Add secondary metric columns (empty string if metric not present)
                    for metric_name in all_secondary_metrics:
                        metric_value = metrics_dict.get(metric_name, "")
                        row_values.append(metric_value)
                    
                    # Generate row
                    row = "| " + " | ".join(row_values) + " |"
                    output_lines.append(row)
                
                # Page info (only show if multiple pages)
                if total_pages > 1:
                    output_lines.append("")
                    output_lines.append(f"Page {current_page} of {total_pages} (sorted by: {current_sort})")
                    output_lines.append("")
                    output_lines.append(f"(Use `/execute_action{{page N}}` to navigate pages 1-{total_pages}, `/execute_action{{rank mode}}` to change sort order, `/execute_action{{page_size N}}` to set page size, `/execute_action{{filter tag}}` to filter by tag.)")
            else:
                output_lines.append("No evaluations on this page.")
        else:
            output_lines.append("No evaluations submitted yet.")
        
        # Add filter status if active
        if filter_tag:
            output_lines.append("")
            output_lines.append(f"**Filter Active:** Showing only submissions tagged with '{filter_tag}'. Use `/execute_action{{unfilter}}` to remove filter.")
        
        # Add Tag Statistics table (hidden for immature agents and when disabled)
        if is_mature and all_evaluations and consts.RESEARCH_COUNTER_SHOW_TAG_STATS:  # Use unfiltered evaluations for global statistics
            tag_statistics = self._build_tag_statistics(all_evaluations, filter_tag)
            if tag_statistics:
                output_lines.append("")
                output_lines.append("**Tag Statistics:**")
                output_lines.append("")
                
                # Score-aware table header
                if consts.RESEARCH_NO_SCORE:
                    output_lines.append("| Tag | Submissions | Last Tick |")
                    output_lines.append("|---|---|---|")
                else:
                    output_lines.append("| Tag | Submissions | Last Tick | Highest Score |")
                    output_lines.append("|---|---|---|---|")
                
                for stat in tag_statistics:
                    tag = stat['tag']
                    submissions = stat['submissions']
                    last_tick = stat['last_tick']
                    highest_score = stat['highest_score']
                    
                    if consts.RESEARCH_NO_SCORE:
                        output_lines.append(f"| {tag} | {submissions} | {last_tick} |")
                    else:
                        score_display = self._format_score_for_display(highest_score)
                        output_lines.append(f"| {tag} | {submissions} | {last_tick} | {score_display} |")
        
        # Add storage display sections based on last actions
        storage_sections = self._get_storage_display_sections(agent_data, consts)
        if storage_sections:
            output_lines.extend(storage_sections)
        
        return "\n".join(output_lines)
    
    def handle_action(self,
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str],
                      yaml_data: Optional[Dict[str, Any]],
                      room_context: RoomContext,
                      current_tick: int
                     ) -> Tuple[List[str], Optional[InternalActionHandler]]:
        
        actions_executed = []
        consts = room_context.constants_module
        
        # Access restriction
        if agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST:
            actions_executed.append("The Research Counter is only accessible to Recursive Agents.")
            return actions_executed, None
        
        agent_room_key = consts.SHORT_ROOM_NAME_RESEARCH
        if agent_room_key not in agent_data:
            agent_data[agent_room_key] = {}
        
        if action_command.lower() == consts.ACTION_RESEARCH_READ:
            # Read research task
            if not action_args:
                actions_executed.append("Please specify a research task ID to read.")
                return actions_executed, None
            
            # Reload research tasks to ensure we have the latest specifications
            self._load_research_tasks()
            
            task_id = str(action_args)
            task = self.research_tasks.get(task_id)
            
            if not task:
                actions_executed.append(f"Research task ID '{task_id}' not found.")
                return actions_executed, None
            
            # Send task content as system message
            title = task.get(consts.RESEARCH_TASK_TITLE_KEY, "Unknown Task")
            content = task.get(consts.RESEARCH_TASK_CONTENT_KEY, "No content available.")
            
            system_msg = f"**Research Task #{task_id}: {title}**\n\n{content}"
            room_context.agent_manager.add_pending_notification(agent_data, system_msg)
            actions_executed.append(f"Research task #{task_id} details sent to your System Messages.")
            
        elif action_command.lower() == consts.ACTION_RESEARCH_SUBMIT:
            # Check if it's a holiday and holiday mode is enabled
            if consts.HOLIDAY_MODE_ENABLED and consts.is_holiday_tick(current_tick):
                actions_executed.append("Holidays are not for working - please try again on working days.")
                return actions_executed, None
                
            # Check if agent is a supervisor (supervisors cannot submit)
            if agent_data.get(consts.AGENT_ROLE_KEY) == consts.ROLE_SUPERVISOR:
                actions_executed.append("Supervisors cannot submit research experiments. Your role is to review and monitor submissions.")
                return actions_executed, None
            
            # Check cooldown first (only if cooldown is enabled)
            if consts.RESEARCH_SUBMISSION_COOLDOWN_TICKS > 0:
                last_submission_tick = agent_data.get(consts.AGENT_LAST_RESEARCH_SUBMISSION_TICK_KEY, -consts.RESEARCH_SUBMISSION_COOLDOWN_TICKS)
                ticks_since_last_submission = current_tick - last_submission_tick
                
                if ticks_since_last_submission < consts.RESEARCH_SUBMISSION_COOLDOWN_TICKS:
                    ticks_remaining = consts.RESEARCH_SUBMISSION_COOLDOWN_TICKS - ticks_since_last_submission
                    actions_executed.append(f"Submission failed: Cooldown active. Try again in {ticks_remaining} tick(s).")
                    return actions_executed, None
            
            # Submit solution
            if not yaml_data:
                actions_executed.append("Submission requires YAML data with 'title', 'tags', 'abstract', and 'content' fields.")
                return actions_executed, None
            
            title = yaml_data.get(consts.YAML_CAPSULE_TITLE, "")
            content = yaml_data.get(consts.YAML_CAPSULE_CONTENT, "")
            tags_input = yaml_data.get(consts.YAML_CAPSULE_TAGS, "")
            abstract = yaml_data.get(consts.YAML_CAPSULE_ABSTRACT, "")
            cpu_only = yaml_data.get('cpu_only', False) if consts.RESEARCH_EVAL_ALLOW_CPU_ONLY else False
            
            if not title or not content or not tags_input or not abstract:
                actions_executed.append("Submission requires all fields: 'title', 'tags', 'abstract', and 'content' in YAML data.")
                return actions_executed, None
            
            # Validate tags
            tags_valid, tags_error, parsed_tags, tags_warning = self._validate_tags(tags_input)
            if not tags_valid:
                actions_executed.append(f"Tags validation failed: {tags_error}")
                return actions_executed, None
            
            # Validate abstract
            abstract_valid, abstract_error, processed_abstract, abstract_warning = self._validate_abstract(abstract)
            if not abstract_valid:
                actions_executed.append(f"Abstract validation failed: {abstract_error}")
                return actions_executed, None
            
            # Add warning messages for truncations
            warning_messages = []
            if tags_warning:
                warning_messages.append(tags_warning)
            if abstract_warning:
                warning_messages.append(abstract_warning)
            
            # Validate cpu_only if provided
            if 'cpu_only' in yaml_data:
                if not consts.RESEARCH_EVAL_ALLOW_CPU_ONLY:
                    actions_executed.append("The 'cpu_only' field is not available. This feature is disabled.")
                    return actions_executed, None
                if not isinstance(yaml_data['cpu_only'], bool):
                    actions_executed.append("The 'cpu_only' field must be a boolean value (true or false).")
                    return actions_executed, None
            
            # Check for valid code content
            if not self._is_code_content_valid(content):
                actions_executed.append("Submission failed. The submitted code appears to be incomplete or contains only comments. This is a real submission system that executes your code on a computer cluster. Please provide the full, complete code for your submission. If you wish to reuse code, you can save it to your lineage files and import it in your submission.")
                return actions_executed, None

            # Determine task ID
            if action_args:
                task_id = str(action_args)
            else:
                # Default to most recent task
                task_ids = sorted(self.research_tasks.keys(), key=lambda x: int(x) if x.isdigit() else x)
                if not task_ids:
                    actions_executed.append("No research tasks available for submission.")
                    return actions_executed, None
                task_id = task_ids[-1]
            
            task = self.research_tasks.get(task_id)
            if not task:
                actions_executed.append(f"Research task ID '{task_id}' not found.")
                return actions_executed, None
            
            # Check concurrent submission limit
            agent_name = agent_data.get(consts.AGENT_NAME_KEY, "Unknown")
            
            # Get all evaluations and filter for this agent's pending/running ones
            all_evaluations = self._get_all_evaluations()
            concurrent_eval_ids = set()
            
            for eval_data in all_evaluations:
                if eval_data.get(consts.EVALUATION_AUTHOR_KEY) == agent_name:
                    score = eval_data.get(consts.EVALUATION_SCORE_KEY, "")
                    # Count as concurrent if it's pending or running
                    if score == consts.RESEARCH_SCORE_PENDING or score == "running":
                        concurrent_eval_ids.add(eval_data.get(consts.EVALUATION_ID_KEY))

            if len(concurrent_eval_ids) >= consts.RESEARCH_MAX_CONCURRENT_SUBMISSIONS:
                # Sort the IDs for consistent display
                sorted_ids = sorted(concurrent_eval_ids, key=lambda x: int(x) if str(x).isdigit() else float('inf'))
                ids_str = ", ".join(str(id) for id in sorted_ids)
                actions_executed.append(
                    f"Submission failed: You have reached the maximum limit of {consts.RESEARCH_MAX_CONCURRENT_SUBMISSIONS} concurrent pending evaluations. "
                    f"Your current pending/running evaluations are: {ids_str}. "
                    f"Please wait for these to complete before submitting new ones."
                )
                return actions_executed, None
            
            # Generate simple sequential evaluation ID
            evaluation_id = self._generate_next_evaluation_id()
            
            # Prepare evaluation data
            pending_eval = {
                consts.EVALUATION_ID_KEY: evaluation_id,
                consts.EVALUATION_RESEARCH_TASK_ID_KEY: task_id,
                consts.EVALUATION_TITLE_KEY: title,
                consts.EVALUATION_TAGS_KEY: parsed_tags,
                consts.EVALUATION_ABSTRACT_KEY: processed_abstract,
                consts.EVALUATION_CONTENT_KEY: content,
                consts.EVALUATION_AUTHOR_KEY: agent_name,
                consts.EVALUATION_SUBMITTED_TICK_KEY: current_tick,
                consts.EVALUATION_SCORE_KEY: consts.RESEARCH_SCORE_PENDING,
                consts.EVALUATION_LOGS_KEY: "",
                consts.EVALUATION_CPU_ONLY_KEY: cpu_only
            }
            
            # Check if agent wants to disable debugging
            if yaml_data.get('no_debugger', False):
                pending_eval['no_debugger'] = True
            
            # Save to pending evaluations
            pending_path = os.path.join(
                self._get_research_data_path(),
                consts.PENDING_RESEARCH_EVALUATIONS_FILENAME
            )
            
            try:
                file_io_utils.append_yaml_line(pending_eval, pending_path)
            except Exception as e:
                actions_executed.append(f"ERROR: Failed to save evaluation: {e}")
                # Remove from assigned IDs if save failed
                self.assigned_ids.discard(int(evaluation_id))
                return actions_executed, None
            
            # Update agent's research data
            if 'submitted_evaluations' not in agent_data[agent_room_key]:
                agent_data[agent_room_key]['submitted_evaluations'] = []
                
            agent_data[agent_room_key]['submitted_evaluations'].append(evaluation_id)
            
            # Update last submission tick for cooldown tracking
            if consts.RESEARCH_SUBMISSION_COOLDOWN_TICKS > 0:
                agent_data[consts.AGENT_LAST_RESEARCH_SUBMISSION_TICK_KEY] = current_tick
            
            task_title = task.get(consts.RESEARCH_TASK_TITLE_KEY, "Unknown Task")
            success_msg = f"Evaluation '{title}' submitted for Research Task #{task_id}: {task_title}. Evaluation ID: {evaluation_id}."
            
            # Add warning messages if any truncation occurred
            if warning_messages:
                success_msg += " " + " ".join(warning_messages)
            
            actions_executed.append(success_msg)
            
            # Notify agent that results will be available later
            room_context.agent_manager.add_pending_notification(
                agent_data, 
                f"Your submission has been queued for evaluation. Results will be sent to you once the evaluation completes."
            )
            
        elif action_command.lower() == consts.ACTION_RESEARCH_REVIEW:
            # Review evaluation
            if not action_args:
                actions_executed.append("Please specify an evaluation ID to review.")
                return actions_executed, None
            
            eval_id = str(action_args)
            
            # Get review info using standalone function
            review_info = get_evaluation_review_info(eval_id)
            if not review_info:
                actions_executed.append(f"Evaluation ID '{eval_id}' not found.")
                return actions_executed, None
            
            # Check if cross-lineage review is disabled
            if not consts.RESEARCH_ALLOW_CROSS_LINEAGE_REVIEW:
                # Get the evaluation author using get_evaluation_display_info
                eval_display_info = get_evaluation_display_info(eval_id)
                if eval_display_info:
                    eval_author = eval_display_info.get(consts.EVALUATION_AUTHOR_KEY, "")
                    agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "")
                    
                    # Check if this is a cross-lineage review attempt
                    is_own_submission = eval_author == agent_data.get(consts.AGENT_NAME_KEY, "")
                    is_system_submission = eval_author.lower() == "system"
                    is_same_lineage = eval_author.startswith(agent_lineage) if agent_lineage else False
                    is_supervisor = agent_data.get(consts.AGENT_ROLE_KEY) == consts.ROLE_SUPERVISOR
                    
                    if not (is_own_submission or is_system_submission or is_same_lineage or is_supervisor):
                        actions_executed.append(
                            "Cross-lineage submission review is disabled. You can only review submissions from your own lineage. "
                            "You are encouraged to visit the Archive Room to read published papers from other lineages."
                        )
                        return actions_executed, None
            
            # Send the stored notification message as system message
            room_context.agent_manager.add_pending_notification(
                agent_data,
                f"**Evaluation Review: {eval_id}**\n\n{review_info['message']}"
            )
            
            if review_info["status"] == "pending":
                actions_executed.append(f"Evaluation '{eval_id}' is still running. Please try again later.")
            else:
                actions_executed.append(f"Evaluation {eval_id} details sent to your System Messages.")
            
            return actions_executed, None
            
        elif action_command.lower() == consts.ACTION_RESEARCH_STORAGE:
            # Storage management actions
            if not action_args:
                actions_executed.append("Please specify a storage action: 'info', 'list <path>', 'delete <path>', 'write <path>', or 'read <path>'.")
                return actions_executed, None
            
            # Parse storage action with directory-based format
            storage_parts = action_args.split(' ', 1)
            storage_action = storage_parts[0].lower()
            
            # Get the current agent's lineage
            agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown").lower()
            
            if storage_action == "info":
                # Storage information will be displayed in room content
                actions_executed.append("Storage information will be displayed below.")
                
            elif storage_action == "list":
                # List files in storage directory
                if len(storage_parts) < 2:
                    actions_executed.append("Please specify a path to list: 'list <path> [page]'.")
                    return actions_executed, None
                
                # Keep the original path parsing intact
                path = storage_parts[1]
                
                # File list will be displayed in room content
                actions_executed.append(f"File list for {path} will be displayed below.")
                
            elif storage_action == "delete":
                # Delete file from storage
                if len(storage_parts) < 2:
                    actions_executed.append("Please specify file path: 'delete <path>'.")
                    return actions_executed, None
                
                path = storage_parts[1]
                success, message = self._delete_storage_file_directory(agent_data, path, consts)
                actions_executed.append(message)
                
            elif storage_action == "write":
                # Write file to storage
                if len(storage_parts) < 2:
                    actions_executed.append("Please specify file path: 'write <path>'.")
                    return actions_executed, None
                
                path = storage_parts[1]
                
                if not yaml_data:
                    actions_executed.append("Storage write requires YAML data with 'content' field.")
                    return actions_executed, None
                
                content = yaml_data.get("content", "")
                if not content:
                    actions_executed.append("Missing required field 'content' in YAML data.")
                    return actions_executed, None

                # Check if content is bytes (e.g., from YAML !!binary tag)
                if isinstance(content, bytes):
                    actions_executed.append("Error: Binary content not supported. Storage write expects text content only. Remove '!!binary' tags from YAML.")
                    return actions_executed, None
                
                success, message = self._write_storage_file_directory(agent_data, path, content, consts)
                actions_executed.append(message)
                
            elif storage_action == "read":
                # Read file from storage
                if len(storage_parts) < 2:
                    actions_executed.append("Please specify file path: 'read <path>'.")
                    return actions_executed, None
                
                path = storage_parts[1]
                success, content = self._read_storage_file_directory(agent_data, path, consts)
                if success:
                    # File content will be displayed in room content
                    actions_executed.append(f"File content for {path} will be displayed below.")
                else:
                    actions_executed.append(content)  # content contains error message
                
            else:
                actions_executed.append("Unknown storage action. Use: 'info', 'list <path>', 'delete <path>', 'write <path>', or 'read <path>'.")
            
        elif action_command.lower() == consts.ACTION_RESEARCH_RANK:
            # Change sort order
            # Check if score sorting is disabled when RESEARCH_NO_SCORE is True
            if consts.RESEARCH_NO_SCORE and action_args == "score":
                actions_executed.append("Score sorting is not available when scores are hidden.")
                return actions_executed, None
            
            # Validate sort mode based on whether scores are hidden
            valid_modes = ["id", "author"] if consts.RESEARCH_NO_SCORE else ["id", "score", "author"]
            if not action_args or action_args not in valid_modes:
                if consts.RESEARCH_NO_SCORE:
                    actions_executed.append("Please specify sort mode: 'id' or 'author'.")
                else:
                    actions_executed.append("Please specify sort mode: 'id', 'score', or 'author'.")
                return actions_executed, None
            
            agent_data[agent_room_key][consts.AGENT_RESEARCH_SORT_KEY] = action_args
            agent_data[agent_room_key][consts.AGENT_RESEARCH_PAGE_KEY] = 1  # Reset to first page
            actions_executed.append(f"Evaluations now sorted by {action_args}.")
            
        elif action_command.lower() == consts.ACTION_CAPSULE_PAGE:
            # Navigate to specific page
            if not action_args:
                actions_executed.append("Please specify a page number.")
                return actions_executed, None
            
            try:
                page_num = int(action_args)
                if page_num < 1:
                    actions_executed.append("Page number must be 1 or greater.")
                    return actions_executed, None
                
                # Get agent info needed for filtering
                agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "")
                is_mature = True
                if consts.AGENT_ISOLATION_TICKS is not None:
                    birth_tick = agent_data.get(consts.AGENT_TICK_BIRTH_KEY)
                    if birth_tick is not None:
                        agent_age = current_tick - birth_tick
                        is_mature = agent_age >= consts.AGENT_ISOLATION_TICKS
                is_supervisor = agent_data.get(consts.AGENT_ROLE_KEY) == consts.ROLE_SUPERVISOR
                
                # Get room data
                room_data = agent_data.get(agent_room_key, {})
                
                # Calculate max pages to validate input
                all_evaluations = self._get_all_evaluations()
                # Apply same filtering logic as in display
                if not is_mature and agent_lineage and not is_supervisor:
                    all_evaluations = [
                        e for e in all_evaluations
                        if (e.get(consts.EVALUATION_AUTHOR_KEY, "").startswith(agent_lineage) or 
                            e.get(consts.EVALUATION_AUTHOR_KEY, "").lower() == "system")
                    ]
                filter_tag = room_data.get(consts.AGENT_RESEARCH_FILTER_TAG_KEY)
                filtered_evaluations = self._apply_tag_filter(all_evaluations, filter_tag)
                
                agent_page_size = room_data.get(consts.AGENT_RESEARCH_PAGE_SIZE_KEY, consts.DEFAULT_RESEARCH_PAGE_SIZE)
                total_evals = len(filtered_evaluations)
                max_pages = (total_evals + agent_page_size - 1) // agent_page_size if total_evals > 0 else 1
                
                if page_num > max_pages:
                    actions_executed.append(f"Page {page_num} does not exist. Maximum page is {max_pages}.")
                    return actions_executed, None
                
                agent_data[agent_room_key][consts.AGENT_RESEARCH_PAGE_KEY] = page_num
                actions_executed.append(f"Moved to page {page_num}.")
            except ValueError:
                actions_executed.append("Page number must be a valid integer.")
        
        elif action_command.lower() == consts.ACTION_RESEARCH_FILTER:
            # Filter by tag
            if not action_args:
                actions_executed.append("Please specify a tag to filter by.")
                return actions_executed, None
            
            filter_tag = action_args.strip().lower()  # Convert to lowercase for consistency
            agent_data[agent_room_key][consts.AGENT_RESEARCH_FILTER_TAG_KEY] = filter_tag
            agent_data[agent_room_key][consts.AGENT_RESEARCH_PAGE_KEY] = 1  # Reset to first page
            actions_executed.append(f"Filtered by tag: {filter_tag}. Use '/execute_action{{unfilter}}' to remove filter.")
            
        elif action_command.lower() == consts.ACTION_RESEARCH_UNFILTER:
            # Remove tag filter
            if consts.AGENT_RESEARCH_FILTER_TAG_KEY in agent_data[agent_room_key]:
                del agent_data[agent_room_key][consts.AGENT_RESEARCH_FILTER_TAG_KEY]
            agent_data[agent_room_key][consts.AGENT_RESEARCH_PAGE_KEY] = 1  # Reset to first page
            actions_executed.append("Tag filter removed. Showing all submissions.")
            
        elif action_command.lower() == consts.ACTION_RESEARCH_PREVIEW:
            # Preview submissions
            if not action_args:
                actions_executed.append("Please specify evaluation ID(s) to preview or 'all'.")
                return actions_executed, None
            
            # Handle "all" case
            if action_args.strip().lower() == "all":
                # Get latest 100 evaluations
                all_evaluations = self._get_all_evaluations()
                
                # Apply same filtering as display (lineage isolation)
                agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "")
                is_mature = True
                if consts.AGENT_ISOLATION_TICKS is not None:
                    birth_tick = agent_data.get(consts.AGENT_TICK_BIRTH_KEY)
                    if birth_tick is not None:
                        agent_age = current_tick - birth_tick
                        is_mature = agent_age >= consts.AGENT_ISOLATION_TICKS
                is_supervisor = agent_data.get(consts.AGENT_ROLE_KEY) == consts.ROLE_SUPERVISOR
                
                if not is_mature and agent_lineage and not is_supervisor:
                    all_evaluations = [
                        e for e in all_evaluations
                        if (e.get(consts.EVALUATION_AUTHOR_KEY, "").startswith(agent_lineage) or 
                            e.get(consts.EVALUATION_AUTHOR_KEY, "").lower() == "system")
                    ]
                
                # Sort by ID (newest first) and take latest 100
                sorted_evals = self._sort_evaluations(all_evaluations, "id", agent_data.get(consts.AGENT_NAME_KEY, ""))
                latest_evals = sorted_evals[:100]
                expanded_ids = [str(e.get(consts.EVALUATION_ID_KEY, "")) for e in latest_evals]
            else:
                # Parse ranges using our own implementation
                expanded_ids, error_msg = self._parse_evaluation_range(action_args)
                if error_msg:
                    actions_executed.append(f"Preview error: {error_msg}")
                    return actions_executed, None
                
                # Limit to 100 submissions to prevent flooding
                if len(expanded_ids) > 100:
                    expanded_ids = expanded_ids[:100]
                    actions_executed.append("Preview limited to first 100 submissions to prevent flooding.")
            
            preview_parts = []
            for eval_id in expanded_ids:
                eval_display_info = get_evaluation_display_info(eval_id)
                if not eval_display_info:
                    preview_parts.append(f"**Evaluation {eval_id}:** Not found")
                    continue
                
                # Format preview
                title = eval_display_info.get(consts.EVALUATION_TITLE_KEY, "No title")
                author = eval_display_info.get(consts.EVALUATION_AUTHOR_KEY, "Unknown")
                tags = eval_display_info.get(consts.EVALUATION_TAGS_KEY, [])
                abstract = eval_display_info.get(consts.EVALUATION_ABSTRACT_KEY, "No abstract")
                score = eval_display_info.get(consts.EVALUATION_SCORE_KEY, "n.a.")
                
                tags_display = self._format_tags_for_display(tags) if tags else "—"
                
                preview_str = f"**Evaluation {eval_id}: {title}**\n"
                preview_str += f"Author: {author}\n"
                preview_str += f"Tags: {tags_display}\n"
                preview_str += f"Abstract: {abstract}"
                
                if not consts.RESEARCH_NO_SCORE:
                    display_score = "running" if score == consts.RESEARCH_SCORE_PENDING else self._format_score_for_display(score)
                    
                    # Add secondary metrics to preview
                    if score != consts.RESEARCH_SCORE_PENDING and score != consts.RESEARCH_SCORE_NA:
                        message, metrics_dict = self._extract_secondary_metrics(eval_display_info)
                        if metrics_dict:
                            metrics_parts = [f"{k}: {v}" for k, v in metrics_dict.items()]
                            metrics_string = " | ".join(metrics_parts)
                            display_score = f"{display_score} | {metrics_string}"
                    
                    preview_str += f"\nScore: {display_score}"
                
                preview_parts.append(preview_str)
            
            if preview_parts:
                room_context.agent_manager.add_pending_notification(agent_data, "\n\n---\n\n".join(preview_parts))
                actions_executed.append(f"Preview for {len(preview_parts)} evaluation(s) sent to your System Messages.")
            else:
                actions_executed.append("No evaluations found to preview.")
                
        elif action_command.lower() == consts.ACTION_RESEARCH_PAGE_SIZE:
            # Set page size
            if not action_args:
                actions_executed.append("Please specify page size (1-200).")
                return actions_executed, None
            
            try:
                page_size = int(action_args)
                if page_size < 1 or page_size > 200:
                    actions_executed.append("Page size must be between 1 and 200.")
                    return actions_executed, None
                
                agent_data[agent_room_key][consts.AGENT_RESEARCH_PAGE_SIZE_KEY] = page_size
                agent_data[agent_room_key][consts.AGENT_RESEARCH_PAGE_KEY] = 1  # Reset to first page
                actions_executed.append(f"Page size set to {page_size}.")
            except ValueError:
                actions_executed.append("Page size must be a valid integer.")
                
        else:
            actions_executed.append(f"Action '{action_command}' not recognized in the Research Counter.")
        
        return actions_executed, None
    
    def _get_storage_info_message(self, agent_data: Dict[str, Any], consts) -> str:
        """Generate storage information message."""
        agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown").lower()
        
        shared_path = os.path.join(
            self._get_research_data_path(),
            consts.RESEARCH_STORAGE_DIR,
            consts.RESEARCH_STORAGE_SHARED_DIR
        )
        system_path = os.path.join(
            self._get_research_data_path(),
            consts.RESEARCH_STORAGE_DIR,
            consts.RESEARCH_STORAGE_SYSTEM_DIR
        )
        lineage_path = os.path.join(
            self._get_research_data_path(),
            consts.RESEARCH_STORAGE_DIR,
            consts.RESEARCH_STORAGE_LINEAGES_DIR,
            agent_lineage
        )
        
        def get_directory_info(path: str) -> Tuple[int, int]:
            """Get file count and total size in bytes."""
            if not os.path.exists(path):
                return 0, 0
            
            file_count = 0
            total_size = 0
            for root, dirs, files in os.walk(path):
                file_count += len(files)
                for file in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, file))
                    except OSError:
                        pass  # Skip files we can't access
            return file_count, total_size
        
        shared_files, shared_size = get_directory_info(shared_path)
        system_files, system_size = get_directory_info(system_path)
        lineage_files, lineage_size = get_directory_info(lineage_path)
        
        def format_size(size_bytes: int) -> str:
            """Format bytes as human-readable string."""
            if size_bytes == 0:
                return "0 B"
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024
            return f"{size_bytes:.1f} TB"
        
        # Determine other lineage access based on configuration and role
        is_supervisor = agent_data.get(consts.AGENT_ROLE_KEY) == consts.ROLE_SUPERVISOR
        if is_supervisor:
            other_lineage_access = "As a supervisor, you can read all lineage files but NOT write or delete"
        elif consts.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS:
            other_lineage_access = "You can read but NOT write or delete"
        else:
            other_lineage_access = "Access disabled (cross-lineage storage access is not allowed)"
        
        return f"""**Research Storage Information**

**Storage Locations:**
- Shared Storage: `storage/shared` (accessible to all recursive agents)
- System Storage: `storage/system` (read-only, managed by the station)
- Your Lineage Storage: `storage/{agent_lineage}` (private to your lineage)

**Current Usage:**
- Shared Storage: {shared_files} files, {format_size(shared_size)}
- System Storage: {system_files} files, {format_size(system_size)}
- Your Lineage Storage: {lineage_files} files, {format_size(lineage_size)}

**Access Permissions:**
- **Shared storage**: All agents can read and write
- **System storage**: All agents can read, but cannot write or delete
- **Your lineage storage**: You can read and write  
- **Other lineage storage**: {other_lineage_access}

**Storage Features:**
- Files persist between research evaluations
- Use `numpy.save()` and `numpy.load()` for data persistence
- Subdirectories are allowed and automatically created

**Usage in Code:**
```python
import numpy as np

# Save to shared storage
data = np.array([1, 2, 3])
np.save('storage/shared/my_data.npy', data)

# Load from shared storage
loaded_data = np.load('storage/shared/my_data.npy')

# Save to your lineage storage
np.save('storage/{agent_lineage}/my_algorithm.npy', data)

# Import from your lineage modules
sys.path.append('storage/{agent_lineage}')
from my_utilities import helper_function
```"""
    
    def _list_storage_files(self, agent_data: Dict[str, Any], storage_type: str, consts) -> List[str]:
        """List files in the specified storage type."""
        agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown").lower()
        
        if storage_type == "shared":
            storage_path = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_SHARED_DIR
            )
        else:  # lineage
            storage_path = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_LINEAGES_DIR,
                agent_lineage
            )
        
        if not os.path.exists(storage_path):
            # Ensure directory exists before listing
            file_io_utils.ensure_dir_exists(storage_path)
            return []
        
        file_list = []
        for root, dirs, files in os.walk(storage_path):
            for file in files:
                # Skip .pyc files
                if file.endswith('.pyc'):
                    continue
                    
                full_path = os.path.join(root, file)
                # Get relative path from storage directory
                rel_path = os.path.relpath(full_path, storage_path)
                
                file_list.append(f"- {rel_path}")
        
        return sorted(file_list)
    
    def _resolve_lineage_directory_case(self, lineage_name: str) -> str:
        """
        Resolve the actual directory name for a lineage (case-insensitive lookup).
        Returns the actual directory name or the lowercase input if not found.
        """
        lineages_base_path = os.path.join(
            self._get_research_data_path(),
            "storage",  # consts.RESEARCH_STORAGE_DIR
            "lineages"  # consts.RESEARCH_STORAGE_LINEAGES_DIR
        )
        
        if not os.path.exists(lineages_base_path):
            return lineage_name.lower()
        
        try:
            # List all directories in the lineages folder
            existing_dirs = [d for d in os.listdir(lineages_base_path) 
                           if os.path.isdir(os.path.join(lineages_base_path, d))]
            
            # Look for case-insensitive match
            for existing_dir in existing_dirs:
                if existing_dir.lower() == lineage_name.lower():
                    return existing_dir
            
            # If no match found, return the lowercase version
            return lineage_name.lower()
            
        except OSError:
            # If we can't read the directory, return the lowercase version
            return lineage_name.lower()
    
    def _list_storage_files_directory(self, agent_data: Dict[str, Any], path: str, consts, page: int = 1) -> Tuple[List[str], Dict[str, Any]]:
        """List files in the specified directory path with pagination.
        
        Returns:
            Tuple of (file_list, pagination_info)
            pagination_info contains: total_files, total_pages, current_page, files_per_page
        """
        agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown").lower()
        
        # Handle multiple path formats: shared, /shared, storage/shared, /storage/shared
        if path.startswith('/storage/'):
            # Remove /storage/ prefix and continue with normal parsing
            path = path[9:]  # Remove '/storage/' prefix
        elif path.startswith('storage/'):
            # Remove storage/ prefix and continue with normal parsing
            path = path[8:]  # Remove 'storage/' prefix
        elif path.startswith('/'):
            # Remove leading slash
            path = path[1:]  # Remove '/' prefix
        
        # Parse the path to determine storage location and subdirectory
        path_parts = path.split('/', 1)
        storage_name = path_parts[0].lower()
        subdir = path_parts[1] if len(path_parts) > 1 else ""
        
        # Determine the base storage path
        if storage_name == "shared":
            base_storage_path = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_SHARED_DIR
            )
        elif storage_name == "system":
            base_storage_path = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_SYSTEM_DIR
            )
        else:
            # It's a lineage name - check if cross-lineage access is allowed
            is_supervisor = agent_data.get(consts.AGENT_ROLE_KEY) == consts.ROLE_SUPERVISOR
            if not consts.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS and storage_name != agent_lineage.lower() and not is_supervisor:
                # Return empty list with error message in pagination info
                return ["Error: Cross-lineage storage access is disabled. You can only access shared storage and your own lineage storage."], {
                    "total_files": 0, "total_pages": 0, "current_page": 1, "files_per_page": 500
                }
            
            # Find the actual directory with correct case
            actual_lineage_for_directory = self._resolve_lineage_directory_case(storage_name)
            
            base_storage_path = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_LINEAGES_DIR,
                actual_lineage_for_directory
            )
        
        # Combine with subdirectory if specified
        storage_path = os.path.join(base_storage_path, subdir) if subdir else base_storage_path
        
        if not os.path.exists(storage_path):
            return [], {"total_files": 0, "total_pages": 0, "current_page": 1, "files_per_page": 500}
        
        # Collect all files first
        all_files = []
        for root, dirs, files in os.walk(storage_path):
            for file in files:
                # Skip .pyc files
                if file.endswith('.pyc'):
                    continue
                    
                full_path = os.path.join(root, file)
                # Get relative path from the requested directory
                rel_path = os.path.relpath(full_path, storage_path)
                
                all_files.append(rel_path)
        
        # Sort all files
        all_files.sort()
        
        # Pagination settings
        files_per_page = 500
        total_files = len(all_files)
        total_pages = (total_files + files_per_page - 1) // files_per_page  # Ceiling division
        
        # Ensure page is within valid range
        page = max(1, min(page, total_pages)) if total_pages > 0 else 1
        
        # Calculate slice indices
        start_idx = (page - 1) * files_per_page
        end_idx = min(start_idx + files_per_page, total_files)
        
        # Get files for current page
        page_files = all_files[start_idx:end_idx]
        
        # Format file list
        file_list = [f"- {file}" for file in page_files]
        
        pagination_info = {
            "total_files": total_files,
            "total_pages": total_pages,
            "current_page": page,
            "files_per_page": files_per_page
        }
        
        return file_list, pagination_info
    
    def _delete_storage_file(self, agent_data: Dict[str, Any], file_path: str, consts) -> Tuple[bool, str]:
        """Delete a file from storage. Returns (success, message)."""
        agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown").lower()
        
        # Determine storage type and validate path
        if file_path.startswith("shared/"):
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_SHARED_DIR
            )
            rel_path = file_path[7:]  # Remove "shared/" prefix
        elif file_path.startswith("lineage/"):
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_LINEAGES_DIR,
                agent_lineage
            )
            rel_path = file_path[8:]  # Remove "lineage/" prefix
        else:
            # Default to lineage storage if no prefix
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_LINEAGES_DIR,
                agent_lineage
            )
            rel_path = file_path
        
        # Security check: ensure the path doesn't escape the storage directory
        full_path = os.path.join(storage_base, rel_path)
        real_storage_base = os.path.realpath(storage_base)
        real_full_path = os.path.realpath(full_path)
        
        if not real_full_path.startswith(real_storage_base):
            return False, "Error: Invalid file path (path traversal detected)."
        
        # Check if file exists
        if not os.path.exists(full_path):
            return False, f"File not found: {file_path}"
        
        # Check if it's actually a file (not a directory)
        if not os.path.isfile(full_path):
            return False, f"Path is not a file: {file_path}"
        
        # Delete the file
        try:
            os.remove(full_path)
            return True, f"File deleted successfully: {file_path}"
        except OSError as e:
            return False, f"Error deleting file: {e}"
    
    def _write_storage_file(self, agent_data: Dict[str, Any], storage_type: str, file_name: str, content: str, consts) -> Tuple[bool, str]:
        """Create a file in storage. Returns (success, message)."""
        agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown").lower()
        
        # Determine storage base path
        if storage_type == "shared":
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_SHARED_DIR
            )
        else:  # lineage
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_LINEAGES_DIR,
                agent_lineage
            )
        
        # Security check: ensure the path doesn't escape the storage directory
        full_path = os.path.join(storage_base, file_name)
        real_storage_base = os.path.realpath(storage_base)
        
        try:
            # Ensure directory exists
            file_io_utils.ensure_dir_exists(storage_base)
            
            # Create subdirectories if needed
            file_dir = os.path.dirname(full_path)
            if file_dir != storage_base:
                # Validate subdirectory path
                real_file_dir = os.path.realpath(file_dir)
                if not real_file_dir.startswith(real_storage_base):
                    return False, "Error: Invalid file path (path traversal detected)."
                file_io_utils.ensure_dir_exists(file_dir)
            
            # Validate final file path
            real_full_path = os.path.realpath(full_path)
            if not real_full_path.startswith(real_storage_base):
                return False, "Error: Invalid file path (path traversal detected)."
            
            # Check if file already exists
            file_existed = os.path.exists(full_path)
            
            # Write the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Get relative path for display
            if storage_type == "shared":
                display_path = f"shared/{file_name}"
            else:
                display_path = f"lineage/{file_name}"
            
            if file_existed:
                return True, f"File overwritten successfully: {display_path}"
            else:
                return True, f"File created successfully: {display_path}"
            
        except OSError as e:
            return False, f"Error creating file: {e}"
    
    def _delete_storage_file_new(self, agent_data: Dict[str, Any], storage_type: str, file_path: str, consts) -> Tuple[bool, str]:
        """Delete a file from storage using new format. Returns (success, message)."""
        agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown").lower()
        
        # Determine storage base path
        if storage_type == "shared":
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_SHARED_DIR
            )
        else:  # lineage
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_LINEAGES_DIR,
                agent_lineage
            )
        
        # Security check: ensure the path doesn't escape the storage directory
        full_path = os.path.join(storage_base, file_path)
        real_storage_base = os.path.realpath(storage_base)
        real_full_path = os.path.realpath(full_path)
        
        if not real_full_path.startswith(real_storage_base):
            return False, "Error: Invalid file path (path traversal detected)."
        
        # Check if file exists
        if not os.path.exists(full_path):
            return False, f"File not found: {storage_type}/{file_path}"
        
        # Check if it's actually a file (not a directory)
        if not os.path.isfile(full_path):
            return False, f"Path is not a file: {storage_type}/{file_path}"
        
        # Delete the file
        try:
            os.remove(full_path)
            return True, f"File deleted successfully: {storage_type}/{file_path}"
        except OSError as e:
            return False, f"Error deleting file: {e}"
    
    def _delete_storage_file_directory(self, agent_data: Dict[str, Any], path: str, consts) -> Tuple[bool, str]:
        """Delete a file from storage using directory-based path. Returns (success, message)."""
        agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown").lower()
        
        # Handle multiple path formats: shared, /shared, storage/shared, /storage/shared
        if path.startswith('/storage/'):
            # Remove /storage/ prefix and continue with normal parsing
            path = path[9:]  # Remove '/storage/' prefix
        elif path.startswith('storage/'):
            # Remove storage/ prefix and continue with normal parsing
            path = path[8:]  # Remove 'storage/' prefix
        elif path.startswith('/'):
            # Remove leading slash
            path = path[1:]  # Remove '/' prefix
        
        # Parse the path to determine storage location and file path
        path_parts = path.split('/', 1)
        if len(path_parts) < 2:
            return False, "Error: Path must include both storage location and file name (e.g., 'shared/file.txt')."
        
        storage_name = path_parts[0].lower()
        file_path = path_parts[1]
        
        # Check write permissions
        if storage_name == "system":
            return False, "Error: The system storage is read-only and managed by the station."
        elif storage_name != "shared" and storage_name != agent_lineage.lower():
            # Supervisors also cannot delete from other lineages - they can only read
            return False, f"Error: You can only delete files from shared storage or your own lineage storage ({agent_lineage})."
        
        # Determine the base storage path
        if storage_name == "shared":
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_SHARED_DIR
            )
        else:
            # It's a lineage name - find the actual directory with correct case
            actual_lineage_for_directory = self._resolve_lineage_directory_case(storage_name)
            
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_LINEAGES_DIR,
                actual_lineage_for_directory
            )
        
        # Security check: ensure the path doesn't escape the storage directory
        full_path = os.path.join(storage_base, file_path)
        real_storage_base = os.path.realpath(storage_base)
        real_full_path = os.path.realpath(full_path)
        
        if not real_full_path.startswith(real_storage_base):
            return False, "Error: Invalid file path (path traversal detected)."
        
        # Check if file exists
        if not os.path.exists(full_path):
            return False, f"File not found: {path}"
        
        # Check if it's actually a file (not a directory)
        if not os.path.isfile(full_path):
            return False, f"Path is not a file: {path}"
        
        # Delete the file
        try:
            os.remove(full_path)
            return True, f"File deleted successfully: {path}"
        except OSError as e:
            return False, f"Error deleting file: {e}"
    
    def _read_storage_file(self, agent_data: Dict[str, Any], storage_type: str, file_path: str, consts) -> Tuple[bool, str]:
        """Read a file from storage. Returns (success, content_or_error_message)."""
        agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown").lower()
        
        # Determine storage base path
        if storage_type == "shared":
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_SHARED_DIR
            )
        else:  # lineage
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_LINEAGES_DIR,
                agent_lineage
            )
        
        # Security check: ensure the path doesn't escape the storage directory
        full_path = os.path.join(storage_base, file_path)
        real_storage_base = os.path.realpath(storage_base)
        real_full_path = os.path.realpath(full_path)
        
        if not real_full_path.startswith(real_storage_base):
            return False, "Error: Invalid file path (path traversal detected)."
        
        # Check if file exists
        if not os.path.exists(full_path):
            return False, f"File not found: {storage_type}/{file_path}"
        
        # Check if it's actually a file (not a directory)
        if not os.path.isfile(full_path):
            return False, f"Path is not a file: {storage_type}/{file_path}"
        
        # Read the file
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return True, content
        except OSError as e:
            return False, f"Error reading file: {e}"
        except UnicodeDecodeError:
            return False, f"Error: File appears to be binary and cannot be displayed as text."
    
    def _write_storage_file_directory(self, agent_data: Dict[str, Any], path: str, content: str, consts) -> Tuple[bool, str]:
        """Write a file to storage using directory-based path. Returns (success, message)."""
        agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown").lower()
        
        # Handle multiple path formats: shared, /shared, storage/shared, /storage/shared
        if path.startswith('/storage/'):
            # Remove /storage/ prefix and continue with normal parsing
            path = path[9:]  # Remove '/storage/' prefix
        elif path.startswith('storage/'):
            # Remove storage/ prefix and continue with normal parsing
            path = path[8:]  # Remove 'storage/' prefix
        elif path.startswith('/'):
            # Remove leading slash
            path = path[1:]  # Remove '/' prefix
        
        # Parse the path to determine storage location and file path
        path_parts = path.split('/', 1)
        if len(path_parts) < 2:
            return False, "Error: Path must include both storage location and file name (e.g., 'shared/file.txt')."
        
        storage_name = path_parts[0].lower()
        file_path = path_parts[1]
        
        # Check write permissions
        if storage_name == "system":
            return False, "Error: The system storage is read-only and managed by the station."
        elif storage_name != "shared" and storage_name != agent_lineage.lower():
            # Supervisors also cannot write to other lineages - they can only read
            return False, f"Error: You can only write files to shared storage or your own lineage storage ({agent_lineage})."
        
        # Determine the base storage path
        if storage_name == "shared":
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_SHARED_DIR
            )
        else:
            # It's a lineage name - ensure it matches the agent's lineage
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_LINEAGES_DIR,
                agent_lineage  # Use agent's actual lineage, not the provided name
            )
        
        # Security check: ensure the path doesn't escape the storage directory
        full_path = os.path.join(storage_base, file_path)
        real_storage_base = os.path.realpath(storage_base)
        
        try:
            # Ensure directory exists
            file_io_utils.ensure_dir_exists(storage_base)
            
            # Create subdirectories if needed
            file_dir = os.path.dirname(full_path)
            if file_dir != storage_base:
                # Validate subdirectory path
                real_file_dir = os.path.realpath(file_dir)
                if not real_file_dir.startswith(real_storage_base):
                    return False, "Error: Invalid file path (path traversal detected)."
                file_io_utils.ensure_dir_exists(file_dir)
            
            # Validate final file path
            real_full_path = os.path.realpath(full_path)
            if not real_full_path.startswith(real_storage_base):
                return False, "Error: Invalid file path (path traversal detected)."
            
            # Check if file already exists
            file_existed = os.path.exists(full_path)
            
            # Write the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Construct display path using actual storage names
            if storage_name == "shared":
                display_path = f"shared/{file_path}"
            else:
                display_path = f"{agent_lineage}/{file_path}"
            
            if file_existed:
                return True, f"File overwritten successfully: {display_path}"
            else:
                return True, f"File created successfully: {display_path}"
            
        except OSError as e:
            return False, f"Error creating file: {e}"
    
    def _read_storage_file_directory(self, agent_data: Dict[str, Any], path: str, consts) -> Tuple[bool, str]:
        """Read a file from storage using directory-based path. Returns (success, content_or_error_message)."""
        agent_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown").lower()
        
        # Handle multiple path formats: shared, /shared, storage/shared, /storage/shared
        if path.startswith('/storage/'):
            # Remove /storage/ prefix and continue with normal parsing
            path = path[9:]  # Remove '/storage/' prefix
        elif path.startswith('storage/'):
            # Remove storage/ prefix and continue with normal parsing
            path = path[8:]  # Remove 'storage/' prefix
        elif path.startswith('/'):
            # Remove leading slash
            path = path[1:]  # Remove '/' prefix
        
        # Parse the path to determine storage location and file path
        path_parts = path.split('/', 1)
        if len(path_parts) < 2:
            return False, "Error: Path must include both storage location and file name (e.g., 'shared/file.txt')."
        
        storage_name = path_parts[0].lower()
        file_path = path_parts[1]
        
        # Determine the base storage path
        if storage_name == "shared":
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_SHARED_DIR
            )
        elif storage_name == "system":
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_SYSTEM_DIR
            )
        else:
            # It's a lineage name - check if cross-lineage access is allowed
            is_supervisor = agent_data.get(consts.AGENT_ROLE_KEY) == consts.ROLE_SUPERVISOR
            if not consts.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS and storage_name != agent_lineage.lower() and not is_supervisor:
                return False, "Cross-lineage storage access is disabled. You can only access shared storage and your own lineage storage."
            
            # Find the actual directory with correct case
            actual_lineage_for_directory = self._resolve_lineage_directory_case(storage_name)
            
            storage_base = os.path.join(
                self._get_research_data_path(),
                consts.RESEARCH_STORAGE_DIR,
                consts.RESEARCH_STORAGE_LINEAGES_DIR,
                actual_lineage_for_directory
            )
        
        # Security check: ensure the path doesn't escape the storage directory
        full_path = os.path.join(storage_base, file_path)
        real_storage_base = os.path.realpath(storage_base)
        real_full_path = os.path.realpath(full_path)
        
        if not real_full_path.startswith(real_storage_base):
            return False, "Error: Invalid file path (path traversal detected)."
        
        # Check if file exists
        if not os.path.exists(full_path):
            return False, f"File not found: {path}"
        
        # Check if it's actually a file (not a directory)
        if not os.path.isfile(full_path):
            return False, f"Path is not a file: {path}"
        
        # Read the file
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return True, content
        except OSError as e:
            return False, f"Error reading file: {e}"
        except UnicodeDecodeError:
            return False, f"Error: File appears to be binary and cannot be displayed as text."
    
    def _get_storage_display_sections(self, agent_data: Dict[str, Any], consts) -> List[str]:
        """Generate storage display sections based on last actions performed."""
        # Get raw action data with backward compatibility
        last_raw_actions = agent_data.get(consts.AGENT_LAST_PARSED_ACTIONS_RAW_KEY, [])
        
        if not last_raw_actions:
            return []
        
        # Filter for storage actions in Research Counter
        research_storage_actions = [
            action for action in last_raw_actions 
            if (action.get("location") == consts.ROOM_RESEARCH_COUNTER and 
                action.get("command") == consts.ACTION_RESEARCH_STORAGE)
        ]
        
        if not research_storage_actions:
            return []
        
        output_sections = []
        
        # Process storage actions to generate display sections
        info_displayed = False
        list_actions = []
        read_actions = []
        
        for action in research_storage_actions:
            args = action.get("args", "")
            if not args:
                continue
                
            # Parse storage action arguments
            storage_parts = args.split(' ', 1)
            if len(storage_parts) < 1:
                continue
                
            storage_action = storage_parts[0].lower()
            
            if storage_action == "info" and not info_displayed:
                # Generate fresh storage info
                info_content = self._get_storage_info_message(agent_data, consts)
                output_sections.extend([
                    "",
                    "**Storage Information:**",
                    "",
                    info_content
                ])
                info_displayed = True
                
            elif storage_action == "list" and len(storage_parts) >= 2:
                # Keep the full path, parse page number separately if it exists
                full_arg = storage_parts[1]
                
                # Check if the last part is a number (page)
                parts = full_arg.rsplit(' ', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    path = parts[0]
                    page = int(parts[1])
                    if page < 1:
                        page = 1
                else:
                    # No page number provided, use the full arg as path
                    path = full_arg
                    page = 1
                    
                list_actions.append((path, page))
                
            elif storage_action == "read" and len(storage_parts) >= 2:
                path = storage_parts[1]
                read_actions.append(path)
        
        # Generate file listings section
        if list_actions:
            output_sections.extend(["", "**File Listings:**", ""])
            
            # Remove duplicates while preserving order (now with page info)
            unique_actions = []
            seen = set()
            for action in list_actions:
                if action not in seen:
                    unique_actions.append(action)
                    seen.add(action)
            
            for path, page in unique_actions:
                file_list, pagination_info = self._list_storage_files_directory(agent_data, path, consts, page)
                
                # Add path header with page info if multiple pages
                if pagination_info["total_pages"] > 1:
                    output_sections.append(f"**{path} (page {pagination_info['current_page']} of {pagination_info['total_pages']}):**")
                else:
                    output_sections.append(f"**{path}:**")
                
                if file_list:
                    output_sections.extend(file_list)
                    
                    # Add pagination info if there are multiple pages
                    if pagination_info["total_pages"] > 1:
                        output_sections.extend([
                            "",
                            f"[Total files: {pagination_info['total_files']}, Total pages: {pagination_info['total_pages']}]",
                            f"[Use /execute_action{{storage list {path} N}} to show page N (current: {pagination_info['current_page']})]"
                        ])
                else:
                    output_sections.append("No files found.")
                output_sections.append("")
        
        # Generate read contents section
        if read_actions:
            output_sections.extend(["", "**Read Contents:**", ""])
            
            for path in read_actions:
                success, content = self._read_storage_file_directory(agent_data, path, consts)
                if success:
                    # Truncate content for display (shorter than system messages)
                    display_content = content
                    if len(content) > consts.RESEARCH_STORAGE_READ_MAX_CHARS:
                        display_content = content[:consts.RESEARCH_STORAGE_READ_MAX_CHARS] + f"\n\n[... truncated after {consts.RESEARCH_STORAGE_READ_MAX_CHARS:,} characters]"
                    
                    output_sections.append(f"**{path}:**")
                    output_sections.append("```")
                    output_sections.append(display_content)
                    output_sections.append("```")
                else:
                    output_sections.append(f"**{path}:** {content}")  # content contains error message
                output_sections.append("---")
            
            # Remove trailing separator
            if output_sections and output_sections[-1] == "---":
                output_sections.pop()
        
        return output_sections

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        # Check for constant override first
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        
        # Get base help message
        help_message = _RESEARCH_COUNTER_HELP
        
        # Add CPU-only option to help if enabled
        if constants.RESEARCH_EVAL_ALLOW_CPU_ONLY:
            content_original = "    - `content`: Your solution code or methodology"
            content_replacement = ("    - `content`: Your solution code or methodology\n"
                                  "    - `cpu_only`: Optional boolean (default: false). Set to true if your script does not need GPU to conserve resources for other agents")
            help_message = help_message.replace(content_original, content_replacement)
        
        
        # Dynamically adjust the help message based on no-score setting
        if constants.RESEARCH_NO_SCORE:
            # Update rank action documentation
            rank_original = ("-   `/execute_action{rank id | score | author}`: Changes the sort order for the Submitted Evaluations table.\n"
                            "    - `rank id`: Sort by submission time (newest first) - default\n"
                            "    - `rank score`: Sort by score (highest first, pending/n.a. at bottom)\n"
                            "    - `rank author`: Show your submissions first, then others by newest\n"
                            "    Example: `/execute_action{rank score}`")
            rank_replacement = ("-   `/execute_action{rank id | author}`: Changes the sort order for the Submitted Evaluations table.\n"
                               "    - `rank id`: Sort by submission time (newest first) - default\n"
                               "    - `rank author`: Show your submissions first, then others by newest\n"
                               "    Example: `/execute_action{rank id}`")
            
            if rank_original not in help_message:
                raise ValueError(f"RESEARCH_NO_SCORE: Could not find rank action text in help message for replacement. "
                               f"Expected text not found. This likely means the help message format has changed.")
            help_message = help_message.replace(rank_original, rank_replacement)
            
            # Remove general notes about scores
            notes_original = ("**General Notes:**\n"
                            "- A score of 'running' indicates the evaluation is still running.\n"
                            "- A score of 'n.a.' usually indicates the submitted code contained errors.")
            notes_replacement = ("**General Notes:**\n"
                               "- Evaluations are processed automatically and results will be sent to you when complete.")
            
            if notes_original not in help_message:
                raise ValueError(f"RESEARCH_NO_SCORE: Could not find general notes text in help message for replacement. "
                               f"Expected text not found. This likely means the help message format has changed.")
            help_message = help_message.replace(notes_original, notes_replacement)
        
        # Dynamically adjust the help message based on cross-lineage storage access setting
        if not constants.RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS:
            # Replace the line about other lineage storage access
            help_message = help_message.replace(
                "- **Other lineage storage**: You can read but NOT write or delete",
                "- **Other lineage storage**: Access disabled (cross-lineage storage access is not allowed)"
            )
            # Also update the example that shows reading from another lineage
            help_message = help_message.replace(
                "    - `{storage list aion 2}`: List all files in aion lineage storage (page 2)",
                "    - `{storage list your_lineage}`: List files in your own lineage storage"
            )
        
        # Add Claude Code debugger information if enabled
        if constants.CLAUDE_CODE_DEBUG_ENABLED:
            lines = help_message.split('\n')
            insert_index = -2  # Insert before the last line (which is "To display this help message...")
            debugger_lines = _DEBUGGER_INFO.strip().split('\n')
            help_message = '\n'.join(lines[:insert_index] + [''] + debugger_lines + [''] + lines[insert_index:])
        
        return help_message