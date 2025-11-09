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

import os
import sys
import importlib.util
from typing import Dict, List, Optional, Type
from .base_evaluator import ResearchTaskEvaluator
from station import constants


class ResearchTaskRegistry:
    """
    Registry for managing different research task evaluators.
    Supports both built-in and dynamically loaded evaluators.
    """
    
    def __init__(self):
        self._evaluators: Dict[str, Type[ResearchTaskEvaluator]] = {}
        self._register_default_evaluators()
        self._load_dynamic_evaluators()
    
    def _register_default_evaluators(self):
        """Register built-in evaluators (temporary fallback)."""
        # Legacy fallback disabled - relying on dynamic loading only
        pass
    
    def _load_dynamic_evaluators(self):
        """Load evaluators dynamically from station_data/rooms/research/evaluators/"""
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
            
            # Look for evaluator files matching pattern: task_{id}_evaluator.py
            for filename in os.listdir(evaluators_dir):
                if filename.startswith("task_") and filename.endswith("_evaluator.py"):
                    try:
                        # Extract task ID from filename: task_1_evaluator.py -> 1
                        base_name = filename[5:]  # Remove "task_" prefix  
                        task_id = base_name.split('_evaluator.py')[0]  # Get the part before "_evaluator.py"
                        
                        # Load the module dynamically
                        module_path = os.path.join(evaluators_dir, filename)
                        spec = importlib.util.spec_from_file_location(
                            f"task_{task_id}_evaluator", module_path
                        )
                        module = importlib.util.module_from_spec(spec)
                        
                        # Add to sys.modules to allow imports within the module
                        sys.modules[spec.name] = module
                        spec.loader.exec_module(module)
                        
                        # Look for evaluator class (convention: Task{ID}Evaluator)
                        class_name = f"Task{task_id}Evaluator"
                        if hasattr(module, class_name):
                            evaluator_class = getattr(module, class_name)
                            if issubclass(evaluator_class, ResearchTaskEvaluator):
                                self.register_evaluator(task_id, evaluator_class)
                                print(f"ResearchTaskRegistry: Loaded dynamic evaluator for task {task_id}")
                            else:
                                print(f"ResearchTaskRegistry: {class_name} does not inherit from ResearchTaskEvaluator")
                        else:
                            print(f"ResearchTaskRegistry: No {class_name} class found in {filename}")
                            
                    except Exception as e:
                        print(f"ResearchTaskRegistry: Failed to load evaluator from {filename}: {e}")
                        
        except Exception as e:
            print(f"ResearchTaskRegistry: Error loading dynamic evaluators: {e}")
    
    def register_evaluator(self, task_id: str, evaluator_class: Type[ResearchTaskEvaluator]):
        """Register a new research task evaluator."""
        self._evaluators[task_id] = evaluator_class
    
    def get_evaluator(self, task_id: str) -> Optional[ResearchTaskEvaluator]:
        """Get an evaluator instance for the given task ID."""
        evaluator_class = self._evaluators.get(task_id)
        if evaluator_class:
            return evaluator_class()
        return None
    
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported task IDs."""
        return list(self._evaluators.keys())