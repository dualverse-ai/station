# station_data/rooms/research/evaluators/evaluator.py
"""
Evaluator for Research Task 1: Flat Deformation to the Spider Algebra.
Runs in command mode and parses EVAL_JSON emitted by the runner.
"""

from __future__ import annotations

import json
import re

from station.eval_research.base_evaluator import ResearchTaskEvaluator
from station import constants


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for flat deformation to the spider algebra."""

    def __init__(self):
        super().__init__()

    def get_execution_mode(self) -> str:
        """Use command mode for this task."""
        return "command"

    def get_submission_filename(self) -> str:
        """Match the spec: submission is saved as run.py."""
        return "run.py"

    def get_execution_command(self) -> str:
        """
        Execute the system runner that imports run.construct_deformation(),
        evaluates the snippet, and prints EVAL_JSON.
        """
        return "python -u storage/system/epoch_deformations_eval_runner.py"

    def get_expected_function_name(self) -> str:
        """Not used for command mode tasks."""
        return "dummy_function"

    def get_task_description(self) -> str:
        return "Flat Deformation to the Spider Algebra"

    def get_secondary_metrics_format(self):
        return {
            "L0": "d",
            "L1": "d",
            "edim1": "d",
            "nilp1": "d",
        }

    def evaluate_submission(self, result: str, eval_id: str = None, author: str = None):
        if result is None:
            return False, constants.RESEARCH_SCORE_NA, "No output received from submission"
        output = str(result)
        payload_text = None
        for line in output.splitlines():
            if "EVAL_JSON:" in line:
                payload_text = line.split("EVAL_JSON:", 1)[1].strip()
        if not payload_text:
            return False, constants.RESEARCH_SCORE_NA, "Missing EVAL_JSON line in runner output"
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            return False, constants.RESEARCH_SCORE_NA, f"Invalid EVAL_JSON payload: {exc}"

        score = payload.get("score", constants.RESEARCH_SCORE_NA)
        details = payload.get("details", {})
        if not isinstance(details, dict):
            return False, constants.RESEARCH_SCORE_NA, "EVAL_JSON.details must be an object"
        return True, score, details
