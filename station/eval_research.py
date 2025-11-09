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

# station/eval_research.py
"""
Modular research evaluation framework for automated code execution and assessment.
This module provides a clean interface to the research evaluation subsystem.
"""

# Import from the new modular structure
from .eval_research.base_evaluator import ResearchTaskEvaluator
from .eval_research.task_registry import ResearchTaskRegistry

# Import AutoResearchEvaluator from the new modular structure
from .eval_research.auto_evaluator import AutoResearchEvaluator

# Export the main classes
__all__ = ['ResearchTaskEvaluator', 'ResearchTaskRegistry', 'AutoResearchEvaluator']