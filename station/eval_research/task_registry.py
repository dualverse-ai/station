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

# station/eval_research/task_registry.py
"""
Dynamic task registry for research evaluators.
"""

import importlib.util
import os
import sys
from typing import List, Optional, Type
from .base_evaluator import ResearchTaskEvaluator
from station import constants


class ResearchTaskRegistry:
    """
    Registry for the single active Research Center evaluator.
    """
    
    def __init__(self):
        self._evaluator_class: Optional[Type[ResearchTaskEvaluator]] = None
        self._load_dynamic_evaluators()
    
    def _load_dynamic_evaluators(self):
        """Load the active evaluator dynamically from station_data/rooms/research/evaluators/."""
        try:
            evaluators_dir = os.path.join(
                constants.BASE_STATION_DATA_PATH,
                constants.ROOMS_DIR_NAME,
                constants.SHORT_ROOM_NAME_RESEARCH,
                "evaluators"
            )
            
            if not os.path.exists(evaluators_dir):
                print(f"ResearchTaskRegistry: Evaluators directory not found at {evaluators_dir}")
                return

            filenames: List[str] = []
            canonical_name = "evaluator.py"
            canonical_path = os.path.join(evaluators_dir, canonical_name)
            if os.path.exists(canonical_path):
                filenames.append(canonical_name)

            for filename in filenames:
                try:
                    module_path = os.path.join(evaluators_dir, filename)
                    module_name = os.path.splitext(filename)[0]
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = module
                    spec.loader.exec_module(module)

                    loaded_any = False
                    for value in module.__dict__.values():
                        if not isinstance(value, type):
                            continue
                        if value is ResearchTaskEvaluator or not issubclass(value, ResearchTaskEvaluator):
                            continue
                        evaluator_class = value
                        if self._evaluator_class is not None:
                            loaded_any = True
                            continue
                        self.register_evaluator(evaluator_class)
                        print(f"ResearchTaskRegistry: Loaded active evaluator from {filename}")
                        loaded_any = True

                    if not loaded_any:
                        print(f"ResearchTaskRegistry: No evaluator class found in {filename}")
                except Exception as e:
                    print(f"ResearchTaskRegistry: Failed to load evaluator from {filename}: {e}")
                        
        except Exception as e:
            print(f"ResearchTaskRegistry: Error loading dynamic evaluators: {e}")
    
    def register_evaluator(self, evaluator_class: Type[ResearchTaskEvaluator]):
        """Register the active research evaluator."""
        self._evaluator_class = evaluator_class
    
    def get_evaluator(self) -> Optional[ResearchTaskEvaluator]:
        """Get an instance of the active research evaluator."""
        evaluator_class = self._evaluator_class
        if evaluator_class:
            return evaluator_class()
        return None
    
    def get_supported_tasks(self) -> List[str]:
        """Compatibility shim for the single-task Research Center."""
        return ["active"] if self._evaluator_class else []
