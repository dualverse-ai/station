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

import os
from typing import Any, Dict, List

from station import file_io_utils

_PRESETS_FILENAME = "model_presets.yaml"


def load_model_presets() -> List[Dict[str, Any]]:
    """Load model presets from the packaged YAML file."""
    presets_path = os.path.join(os.path.dirname(__file__), _PRESETS_FILENAME)
    data = file_io_utils.load_yaml(presets_path)
    if not data:
        return []
    if not isinstance(data, list):
        print(f"Warning: Model presets at {presets_path} are not a list. Ignoring.")
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def build_model_preset_lookup() -> Dict[str, Dict[str, Any]]:
    """Return presets indexed by display name for quick lookups."""
    presets = load_model_presets()
    lookup: Dict[str, Dict[str, Any]] = {}
    for preset in presets:
        display_name = preset.get("display_name")
        if isinstance(display_name, str) and display_name:
            lookup[display_name] = preset
    return lookup
