"""
Evaluator for Research Task 1: General Purpose Python Execution
"""

import os
import sys
from typing import Tuple, Optional

from station.eval_research.base_evaluator import ResearchTaskEvaluator


class Task1Evaluator(ResearchTaskEvaluator):
    """
    Evaluator for Research Task 1: General Purpose Python Execution
    Simple command-mode evaluator that runs submitted code without GPU access.
    """
    
    def __init__(self):
        super().__init__("1")
    
    def get_execution_mode(self) -> str:
        """Return 'command' to indicate this task uses command execution mode."""
        return "command"
    
    def get_execution_command(self) -> str:
        """
        Return the command to execute the submitted code.
        Sets CUDA_VISIBLE_DEVICES=-1 before running python submission.py.
        """
        return "CUDA_VISIBLE_DEVICES=-1 python submission.py"
    
    def evaluate_submission(self, result, eval_id: str = None, author: str = None) -> Tuple[bool, float, str]:
        """
        Evaluate the submission result from command execution.
        For this general-purpose task, we simply confirm successful execution.
        
        Args:
            result: The captured stdout from running the command
            eval_id: Evaluation ID
            author: Author of the submission
            
        Returns:
            Tuple of (success, score, details)
            - success: Always True if we got here (code completed without timeout/error)
            - score: Always 1.0 (no specific scoring criteria)
            - details: Brief success message (full output available in logs)
        """
        # For this general task, success just means the code ran
        # The "score" is meaningless - just set to 1.0
        score = 1.0
        
        # Keep details simple since full output is already in logs
        details = "Code executed successfully. Full output available in evaluation logs."
        
        return True, score, details
    
    def get_expected_function_name(self) -> str:
        """Not used for command mode tasks."""
        return "dummy_function"  # Required by base class but not used in command mode
    
    def get_task_description(self) -> str:
        """Return a brief description of this task."""
        return "General Purpose Python Execution"
    
    def validate_submission_code(self, content: str, author: str, agent_module) -> Tuple[bool, Optional[str]]:
        """
        Validate submitted code before execution.
        For this general-purpose task, we accept any valid Python code.
        
        Args:
            content: The submitted code content
            author: The author of the submission
            agent_module: Module for loading agent data
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if validation passes, False if violations detected
            - error_message: Description of violation if found, None otherwise
        """
        # For a general-purpose execution task, we don't impose restrictions
        # The code will be saved as main.py and executed directly
        return True, None