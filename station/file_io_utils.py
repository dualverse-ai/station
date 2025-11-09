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

# file_io_utils.py
"""
Utility functions for file and directory operations, including loading and
saving data in YAML, YAML Lines, and plain text formats, with atomic writes.
"""

import os
import json
import shutil
import yaml # Requires PyYAML: pip install PyYAML
import re # For parsing IDs from filenames
from typing import Any, List, Dict, Optional, Union

# Assuming constants.py is in the same directory or Python path
from station import constants

# --- Directory and File Management ---

def _ensure_base_dir_exists(file_path: str) -> None:
    """Ensures the base directory for the given file_path exists."""
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

def ensure_dir_exists(dir_path: str) -> None:
    """Ensures the specified directory exists. Creates it if not."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    elif not os.path.isdir(dir_path):
        raise IOError(f"Path {dir_path} exists but is not a directory.")

def file_exists(file_path: str) -> bool:
    """Checks if a file exists at the given path."""
    return os.path.exists(file_path) and os.path.isfile(file_path)

def dir_exists(dir_path: str) -> bool:
    """Checks if a directory exists at the given path."""
    return os.path.isdir(dir_path)

def delete_file(file_path: str) -> bool:
    """Deletes a file if it exists. Returns True if deleted, False otherwise."""
    if file_exists(file_path):
        try:
            os.remove(file_path)
            return True
        except OSError as e:
            raise IOError(f"Error deleting file {file_path}: {e}")
    elif os.path.exists(file_path) and not os.path.isfile(file_path):
        raise IOError(f"Path {file_path} exists but is not a file. Cannot delete.")
    return False

def list_files(dir_path: str, extension: Optional[str] = None) -> List[str]:
    """Lists filenames in a directory, optionally filtering by extension."""
    if not dir_exists(dir_path):
        # Return empty list if directory doesn't exist, common for ID generation
        return []
    
    # Retry logic for stale file handle errors
    max_retries = 3
    retry_delay = 0.5  # seconds
    
    for attempt in range(max_retries):
        try:
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            if extension:
                return [f for f in files if f.endswith(extension)]
            return files
        except OSError as e:
            # Check for stale file handle error (errno 116) or I/O error (errno 5)
            if e.errno in [5, 116] and attempt < max_retries - 1:
                error_type = "I/O error" if e.errno == 5 else "Stale file handle"
                print(f"Warning: {error_type} listing {dir_path}, retrying ({attempt + 1}/{max_retries})...")
                import time
                time.sleep(retry_delay)
                continue
            else:
                raise IOError(f"Error listing files in {dir_path}: {e}")

def list_subdirectories(dir_path: str) -> List[str]:
    """Lists subdirectory names in a given directory."""
    if not dir_exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Retry logic for stale file handle errors
    max_retries = 3
    retry_delay = 0.5  # seconds
    
    for attempt in range(max_retries):
        try:
            return [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        except OSError as e:
            # Check for stale file handle error (errno 116) or I/O error (errno 5)
            if e.errno in [5, 116] and attempt < max_retries - 1:
                error_type = "I/O error" if e.errno == 5 else "Stale file handle"
                print(f"Warning: {error_type} listing subdirs in {dir_path}, retrying ({attempt + 1}/{max_retries})...")
                import time
                time.sleep(retry_delay)
                continue
            else:
                raise IOError(f"Error listing subdirectories in {dir_path}: {e}")

# --- Data Loading ---

def load_yaml(file_path: str) -> Optional[Dict[str, Any]]: # Usually returns Dict or None
    """
    Loads data from a YAML file. 
    Returns the loaded data (usually a dict) or None if the file 
    doesn't exist, is empty, or a parsing/reading error occurs.
    Includes retry logic for stale file handle errors (common in network file systems).
    """
    if not file_exists(file_path):
        # print(f"Debug: YAML file not found at {file_path}") # Optional debug
        return None
    
    # Retry logic for stale file handle errors
    max_retries = 3
    retry_delay = 0.5  # seconds
    
    for attempt in range(max_retries):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data is None: # Handles empty YAML file
                    # print(f"Debug: YAML file {file_path} is empty.") # Optional debug
                    return None
                if not isinstance(data, dict):
                    print(f"Warning: YAML file {file_path} did not parse into a dictionary (type: {type(data)}). Returning as is, but might cause issues.")
                    # Depending on strictness, you might want to return None here too if a dict is always expected.
                    # For agent files, a dict is expected.
                    # return None 
                return data
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing YAML file {file_path}: {e}. Returning None.")
            return None
        except OSError as e:
            # Check for stale file handle error (errno 116) or I/O error (errno 5)
            # These are common in network file systems like NFS
            if e.errno in [5, 116] and attempt < max_retries - 1:
                error_type = "I/O error" if e.errno == 5 else "Stale file handle"
                print(f"Warning: {error_type} for {file_path}, retrying ({attempt + 1}/{max_retries})...")
                import time
                time.sleep(retry_delay)
                continue
            else:
                # Not a retryable error or last retry attempt
                print(f"Warning: Error reading file {file_path}: {e}. Returning None.")
                if e.errno in [5, 116]:
                    error_type = "I/O error" if e.errno == 5 else "Stale file handle"
                    print(f"DEBUG: {error_type} persisted after {max_retries} attempts")
                print(f"DEBUG: load_yaml IOError details for {file_path}:")
                import traceback
                traceback.print_exc()
                print(f"DEBUG: Current working directory: {os.getcwd()}")
                print(f"DEBUG: Absolute file path: {os.path.abspath(file_path)}")
                print(f"DEBUG: File exists check: {os.path.exists(file_path)}")
                return None
        except Exception as e: # Catch any other unexpected errors during loading
            print(f"Warning: Unexpected error loading YAML from {file_path}: {e}. Returning None.")
            return None
    
    # Should not reach here, but just in case
    return None

def load_yaml_lines(filepath: str) -> List[Dict[str, Any]]:
    """
    Loads a list of dictionaries from a YAML Lines file (.yamll),
    where each YAML document in the file is expected to be a dictionary.
    Uses yaml.safe_load_all to handle multi-line YAML documents.

    Args:
        filepath (str): The path to the .yamll file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, or an empty list if
                              the file doesn't exist or an error occurs.
    """
    entries: List[Dict[str, Any]] = []
    if not os.path.exists(filepath): # Use os.path.exists for clarity, file_exists also works
        return entries
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # yaml.safe_load_all can handle multiple YAML documents in a single file stream
            for doc_index, doc in enumerate(yaml.safe_load_all(f)):
                if isinstance(doc, dict):
                    entries.append(doc)
                elif doc is not None: # If it parsed something but not a dict
                    print(f"Warning: Skipped non-dictionary entry (document #{doc_index + 1}) in {filepath}: type was {type(doc)}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML Lines from {filepath}: {e}")
        # Depending on desired behavior, you might return partially loaded entries
        # or an empty list. Returning what was loaded so far might be okay.
    except Exception as e: # Catch other potential errors like file read issues
        print(f"Error reading or processing YAML Lines file {filepath}: {e}")
    
    return entries

def load_text(file_path: str) -> Optional[str]:
    """Loads content from a plain text file. Returns None if file doesn't exist."""
    if not file_exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")

# --- Data Saving (Atomic Writes) ---

def save_yaml(data: Any,
              file_path: str,
              default_flow_style: Optional[bool] = False,
              sort_keys: bool = False) -> None:
    """Saves data to a YAML file atomically with retry logic for stale file handles."""
    _ensure_base_dir_exists(file_path)
    base_dir = os.path.dirname(file_path) or "."
    temp_path_dir = os.path.join(base_dir, constants.TEMP_DIR_NAME)
    ensure_dir_exists(temp_path_dir)

    temp_file_name = f"{os.path.basename(file_path)}.{os.getpid()}.{id(data)}.tmp"
    temp_file_path = os.path.join(temp_path_dir, temp_file_name)

    # Custom representer for multi-line strings to use literal style
    def str_presenter(dumper, data):
        """Use literal style for multi-line strings for better readability."""
        if '\n' in data:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    # Create a custom Dumper class with our string representer
    class ReadableDumper(yaml.SafeDumper):
        pass

    ReadableDumper.add_representer(str, str_presenter)

    # Retry logic for stale file handle errors
    max_retries = 3
    retry_delay = 0.5  # seconds

    for attempt in range(max_retries):
        try:
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, Dumper=ReadableDumper,
                         default_flow_style=default_flow_style,
                         sort_keys=sort_keys,
                         allow_unicode=True,
                         width=100,  # Reasonable width for readability
                         indent=2)  # Use 2 spaces for indentation
            shutil.move(temp_file_path, file_path)
            return  # Success, exit the function
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error serializing data to YAML for {file_path}: {e}")
        except OSError as e:
            # Check for stale file handle error (errno 116) or I/O error (errno 5)
            if e.errno in [5, 116] and attempt < max_retries - 1:
                error_type = "I/O error" if e.errno == 5 else "Stale file handle"
                print(f"Warning: {error_type} during save to {file_path}, retrying ({attempt + 1}/{max_retries})...")
                import time
                time.sleep(retry_delay)
                # Clean up temp file if it exists before retry
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        pass
                continue
            else:
                # Not a retryable error or last retry attempt
                raise IOError(f"Error writing or moving file {file_path}: {e}")
        finally:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError: # Log this if logging is available
                    pass

def save_text(content: str, file_path: str) -> None:
    """Saves text content to a file atomically."""
    _ensure_base_dir_exists(file_path)
    base_dir = os.path.dirname(file_path) or "."
    temp_path_dir = os.path.join(base_dir, constants.TEMP_DIR_NAME)
    ensure_dir_exists(temp_path_dir)

    temp_file_name = f"{os.path.basename(file_path)}.{os.getpid()}.{id(content)}.tmp"
    temp_file_path = os.path.join(temp_path_dir, temp_file_name)

    try:
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        shutil.move(temp_file_path, file_path)
    except (IOError, OSError) as e:
        raise IOError(f"Error writing or moving file {file_path}: {e}")
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError: # Log this if logging is available
                pass

# --- ID Generation Utility ---

def get_next_sequential_id(dir_path: str,
                           prefix: str,
                           suffix: str = constants.YAML_EXTENSION) -> int:
    """
    Calculates the next sequential ID for files in a directory
    matching a given prefix and suffix.
    Example: prefix_1.yaml, prefix_2.yaml -> next ID is 3.
    """
    ensure_dir_exists(dir_path) # Ensure directory exists before listing
    
    existing_ids = []
    # Escape prefix for regex, as it might contain special characters
    escaped_prefix = re.escape(prefix)
    # Escape suffix for regex
    escaped_suffix = re.escape(suffix)
    
    # Regex to capture the numeric part
    # It looks for prefix, then one or more digits, then suffix, at the end of the string.
    pattern = re.compile(f"^{escaped_prefix}(\\d+){escaped_suffix}$")

    for filename in list_files(dir_path): # list_files now returns empty list if dir_path doesn't exist
        match = pattern.match(filename)
        if match:
            try:
                existing_ids.append(int(match.group(1)))
            except ValueError:
                # Filename matched pattern but numeric part isn't a valid int (shouldn't happen with \d+)
                continue # Or log a warning

    if not existing_ids:
        return 1
    return max(existing_ids) + 1

def append_yaml_line(data_dict: Dict[str, Any], filepath: str, Dumper=yaml.SafeDumper) -> None:
    """
    Appends a dictionary as a YAML formatted string to the specified file,
    ensuring it's a new YAML document typically separated by '---'.
    """
    try:
        ensure_dir_exists(os.path.dirname(filepath))
        
        file_needs_separator = False
        if os.path.exists(filepath):
            try:
                if os.path.getsize(filepath) > 0:
                    # Check if the file already ends with a newline.
                    # If not, it might indicate an incomplete previous write.
                    # For robustness, ensure a newline before potentially adding a separator.
                    with open(filepath, 'rb+') as f: # Open in binary read/append to check last char
                        f.seek(0, os.SEEK_END) # Go to the end
                        if f.tell() > 0: # If file is not empty
                            f.seek(-1, os.SEEK_CUR) # Go back one byte
                            if f.read(1) != b'\n':
                                f.write(b'\n') # Add a newline if the file doesn't end with one
                    file_needs_separator = True
            except OSError: 
                pass # File might be empty or just created
            except ValueError: # seek on empty file
                pass

        with open(filepath, 'a', encoding='utf-8') as f:
            if file_needs_separator:
                f.write("---\n")
            
            # Dump the single document.
            # default_flow_style=False encourages block style for multi-line strings.
            # PyYAML should choose a style (like '|') for multi-line strings containing '---'
            # that makes them unambiguous.
            yaml.dump(
                data_dict, 
                f, # Dump directly to the file stream
                Dumper=Dumper, 
                sort_keys=False, 
                allow_unicode=True, 
                default_flow_style=False, 
                width=1000, # Helps with formatting readability
                explicit_start=False, # We handle the '---' for multiple documents
                explicit_end=False    # We don't typically need '...'
            )
            # yaml.dump to a stream usually ensures the dumped object ends with a newline.
            
            # Force immediate flush to disk to prevent data loss and ensure subsequent reads see the data
            f.flush()
            os.fsync(f.fileno())

    except Exception as e:
        print(f"Error appending YAML line to {filepath}: {e}")
        # Consider re-raising if this is critical: raise

def append_yaml_line_(data_dict: Dict[str, Any], filepath: str, Dumper=yaml.SafeDumper) -> None:
    """
    Appends a dictionary as a YAML formatted string to the specified file,
    ensuring it's a new YAML document typically separated by '---'.
    Useful for creating .yamll (YAML Lines) files where each entry is a distinct document.
    """
    try:
        ensure_dir_exists(os.path.dirname(filepath))
        
        # default_flow_style=False encourages block style for better readability of individual entries.
        # sort_keys=False preserves insertion order from the dict.
        yaml_string = yaml.dump(data_dict, Dumper=Dumper, sort_keys=False, allow_unicode=True, default_flow_style=False)
        
        # yaml.dump with default_flow_style=False usually ends with a newline.
        # We'll manage our own newlines for consistency.
        yaml_string = yaml_string.rstrip('\n')

        file_exists_before_write = os.path.exists(filepath)
        file_is_empty = True
        if file_exists_before_write:
            try:
                if os.path.getsize(filepath) > 0:
                    file_is_empty = False
            except OSError: # Handles potential race condition if file is deleted between exists and getsize
                file_is_empty = True # Assume empty if getsize fails

        with open(filepath, 'a', encoding='utf-8') as f:
            # Add a YAML document separator '---' if the file is not empty,
            # to clearly separate this new document from previous ones.
            if not file_is_empty:
                f.write("---\n")
            
            f.write(yaml_string + "\n") # Add the YAML content and a final newline

    except Exception as e:
        print(f"Error appending YAML line to {filepath}: {e}")
        raise

# --- Example of an alternative: append_json_line (for .jsonl files) ---
def append_json_line(data_dict: Dict[str, Any], filepath: str) -> None:
    """
    Appends a dictionary as a JSON string to a new line in the specified file.
    """
    try:
        ensure_dir_exists(os.path.dirname(filepath))
        json_string = json.dumps(data_dict, ensure_ascii=False) # ensure_ascii=False for unicode
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json_string + "\n")
    except Exception as e:
        print(f"Error appending JSON line to {filepath}: {e}")
        raise