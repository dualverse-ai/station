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

# station/eval_research/base_evaluator.py
"""
Abstract base class for research task evaluators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Dict, Any, List
import numpy as np


@dataclass
class SeedBatchEvaluation:
    """Task-owned per-candidate evaluation for an optional Research Seed Bank."""

    seeds: List[Any]
    scores: np.ndarray
    valid: np.ndarray
    sort_keys: List[tuple]
    details: List[Any]
    errors: List[Optional[str]]


class ResearchTaskEvaluator(ABC):
    """
    Abstract base class for research task evaluators.
    Each research task type should inherit from this class.
    """
    
    def __init__(self, *_args, **_kwargs):
        pass
    
    @abstractmethod
    def evaluate_submission(self, result, eval_id: str = None, author: str = None):
        """
        Evaluate a research submission result.
        
        Args:
            result: Result returned by the submitted algorithm (type varies by task)
            eval_id: Optional evaluation ID for saving successful configurations
            author: Optional author name for context
            
        Returns:
            Tuple of (success, score, details) or (success, score, details, sort_key):
            - success: True if evaluation passed, False if failed
            - score: Numeric score (can be float), or 0 if failed
            - details: For tasks with secondary metrics: Dict with structure:
                      {"Metric1": raw_value, "Message": "description"}
                      For tasks without secondary metrics: String with evaluation details
            - sort_key: Optional tuple for custom sorting (if not provided, score is used)
            
        Note: If get_secondary_metrics_format() is implemented, dict-based details will be 
        automatically formatted to {"Metric1": (formatted_str, raw_value), ...} by the system.
        """
        pass
    
    @abstractmethod
    def get_expected_function_name(self) -> str:
        """Return the expected function name that should be defined in submitted code."""
        pass
    
    @abstractmethod
    def get_task_description(self) -> str:
        """Return a brief description of what this task evaluates."""
        pass
    
    def get_execution_mode(self) -> str:
        """
        Return execution mode: 'function' or 'command'
        Default is 'function' for backward compatibility.
        """
        return "function"
    
    def get_submission_filename(self) -> str:
        """
        For command mode: filename to save submission as.
        Default is 'submission.py'.
        """
        return "submission.py"
    
    def get_execution_command(self) -> str:
        """
        For command mode: command to execute.
        Must be implemented by evaluators using command mode.
        """
        raise NotImplementedError("Must implement get_execution_command() for command mode")
    
    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        """
        Define secondary metrics and their display formatting.
        
        Returns:
            Dict mapping metric name to format string, e.g.:
            {"Density": ".2f", "Hit Rate": "d", "Status": None}
            Returns None if no secondary metrics are defined.
            
        Format strings follow Python format specification (without the colon):
        - ".2f" for 2 decimal places float
        - "d" for integer
        - None for no special formatting (uses str())
        """
        return None

    def evaluate_seed_batch(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> SeedBatchEvaluation:
        """Evaluate and canonicalize every member of a seed-enabled submission.

        Seed-enabled task templates must override this method. The coder returns
        solutions only; the task evaluator computes all candidate metadata.
        """
        raise NotImplementedError(
            "This task enabled RESEARCH_SEED_BANK_ENABLED but its evaluator does "
            "not implement evaluate_seed_batch()."
        )

    def evaluate_seed_batch_with_formatting(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> SeedBatchEvaluation:
        batch = self.evaluate_seed_batch(result, eval_id, author)
        if not isinstance(batch, SeedBatchEvaluation):
            raise TypeError("evaluate_seed_batch() must return SeedBatchEvaluation")
        return SeedBatchEvaluation(
            seeds=list(batch.seeds),
            scores=np.asarray(batch.scores),
            valid=np.asarray(batch.valid, dtype=bool),
            sort_keys=[tuple(key) for key in batch.sort_keys],
            details=[self._format_secondary_metrics(item) for item in batch.details],
            errors=list(batch.errors),
        )

    def get_progress_records(
        self,
        *,
        result: Any,
        success: bool,
        score: Any,
        details: Any,
        sort_key: Any,
        eval_id: str = None,
        author: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Optionally expose task-defined breakthrough tracks.

        Each returned record should contain a stable ``track`` and a ``rank_key``
        where larger is better, matching the normal sort_key convention. The
        Research Center persists normalized records under final.progress_records.
        Existing tasks can ignore this hook.
        """
        return []
    
    def _format_secondary_metrics(self, raw_details: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Tuple[str, Any]]]:
        """
        Internal method to format secondary metrics according to format specification.
        This is called automatically by the evaluation system.
        
        Args:
            raw_details: Either string (no metrics) or dict with raw metric values
            
        Returns:
            Either string (unchanged) or dict with (formatted_value, raw_value) tuples
        """
        # If no secondary metrics format defined, return as-is
        metrics_format = self.get_secondary_metrics_format()
        if not metrics_format:
            return raw_details
        
        # If details is string, return as-is (backward compatibility)
        if isinstance(raw_details, str):
            return raw_details
        
        # If details is not dict, return as-is
        if not isinstance(raw_details, dict):
            return raw_details
        
        # Format each metric according to specification
        formatted_details = {}
        
        # Ensure Message key exists (default to empty string if missing)
        formatted_details["Message"] = raw_details.get("Message", "")
        
        for key, value in raw_details.items():
            if key == "Message":
                # Already handled above
                continue
            elif key in metrics_format:
                # Apply formatting to metric values
                format_spec = metrics_format[key]
                if format_spec is None:
                    # No special formatting
                    formatted_value = str(value)
                else:
                    try:
                        formatted_value = format(value, format_spec)
                    except (ValueError, TypeError):
                        # Fallback to string if formatting fails
                        formatted_value = str(value)
                
                formatted_details[key] = (formatted_value, value)
            else:
                # Unknown metric, keep as-is
                formatted_details[key] = value
        
        return formatted_details
    
    def evaluate_submission_with_formatting(self, result, eval_id: str = None, author: str = None):
        """
        Wrapper method that calls evaluate_submission and automatically formats secondary metrics.
        This is what the auto evaluator should call instead of evaluate_submission directly.
        
        Args:
            result: Result returned by the submitted algorithm (type varies by task)
            eval_id: Optional evaluation ID for saving successful configurations
            author: Optional author name for context
            
        Returns:
            Tuple of (success, score, formatted_details) or (success, score, formatted_details, sort_key):
            The details will be automatically formatted for secondary metrics display.
        """
        # Call the task-specific evaluation method
        eval_result = self.evaluate_submission(result, eval_id, author)
        
        # Handle both 3-tuple and 4-tuple returns
        if len(eval_result) == 4:
            success, score, raw_details, sort_key = eval_result
        else:
            success, score, raw_details = eval_result
            sort_key = None
        
        # Format secondary metrics
        formatted_details = self._format_secondary_metrics(raw_details)
        
        # Return with same structure as input
        if sort_key is not None:
            return success, score, formatted_details, sort_key
        else:
            return success, score, formatted_details
