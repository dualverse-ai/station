"""Evaluator for general-purpose Python execution."""

from typing import Optional, Tuple

from station.eval_research.base_evaluator import ResearchTaskEvaluator


class Task1Evaluator(ResearchTaskEvaluator):
    """Accept any submission that completes successfully in command mode."""

    def get_execution_mode(self) -> str:
        return "command"

    def get_submission_filename(self) -> str:
        return "submission.py"

    def get_execution_command(self) -> str:
        return "CUDA_VISIBLE_DEVICES=-1 python submission.py"

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, float, str]:
        return (
            True,
            1.0,
            "Code executed successfully. Full output is available in the evaluation logs.",
        )

    def get_expected_function_name(self) -> str:
        return "dummy_function"

    def get_task_description(self) -> str:
        return "General Purpose Python Execution"

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        return True, None
