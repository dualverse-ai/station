from __future__ import annotations

import json

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for the Epoch Book B task with +1 per valid n in 22..100."""

    def __init__(self):
        super().__init__()

    def get_execution_mode(self) -> str:
        return "command"

    def get_submission_filename(self) -> str:
        return "run.py"

    def get_execution_command(self) -> str:
        return "python -u storage/system/epoch_book_b_eval_runner.py"

    def get_expected_function_name(self) -> str:
        return "dummy_function"

    def get_task_description(self) -> str:
        return "Ramsey Lower-Bound Construction for Triangular Books, n=22..100"

    def get_secondary_metrics_format(self):
        return {
            "ValidCount": "d",
            "TotalCount": "d",
            "WorstRedExcess": "d",
            "WorstBlueExcess": "d",
            "WorstN": "d",
            "WorstViolatingPairRate": ".6f",
        }

    def evaluate_submission(self, result: str, eval_id: str = None, author: str = None):
        if result is None:
            return False, constants.RESEARCH_SCORE_NA, "No output received from submission"

        output = str(result)
        if "=== Test Mode Detected ===" in output:
            return False, constants.RESEARCH_SCORE_NA, "Test mode - no scoring"

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
