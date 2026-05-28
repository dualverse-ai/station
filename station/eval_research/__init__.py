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

# station/eval_research/__init__.py
"""
Modular research evaluation framework for automated code execution and assessment.
"""

from .base_evaluator import ResearchTaskEvaluator
from .evaluation_manager import (
    EvaluationManager,
    get_evaluation_display_info,
    get_evaluation_review_info,
    get_evaluation_code_info,
    get_evaluation_result_summary,
    get_evaluation_submission_payload,
    format_score_for_display,
    format_tags_for_display,
    extract_secondary_metrics_for_display_info,
    build_evaluation_previews,
)
from .restart_evaluations import (
    restart_stuck_evaluations,
    requeue_instruction_evaluations,
    requeue_unfinished_instruction_evaluations,
    reset_runtime_coder_counters,
)

__all__ = ['ResearchTaskEvaluator', 'ResearchTaskRegistry', 'AutoResearchEvaluator',
           'EvaluationManager', 'get_evaluation_display_info', 'get_evaluation_review_info',
           'get_evaluation_code_info',
           'get_evaluation_result_summary', 'get_evaluation_submission_payload',
           'format_score_for_display', 'format_tags_for_display',
           'extract_secondary_metrics_for_display_info', 'build_evaluation_previews',
           'restart_stuck_evaluations', 'requeue_instruction_evaluations',
           'requeue_unfinished_instruction_evaluations',
           'reset_runtime_coder_counters']


def __getattr__(name):
    if name == "ResearchTaskRegistry":
        from .task_registry import ResearchTaskRegistry

        return ResearchTaskRegistry
    if name == "AutoResearchEvaluator":
        from .auto_evaluator import AutoResearchEvaluator

        return AutoResearchEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
