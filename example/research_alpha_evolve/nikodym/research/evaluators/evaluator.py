"""Command-mode wrapper for three-dimensional finite-field Nikodym evaluation."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator


class Task1Evaluator(ResearchTaskEvaluator):
    """Parse deterministic results emitted by the bundled Nikodym runner."""

    def get_execution_mode(self) -> str:
        return "command"

    def get_submission_filename(self) -> str:
        return "run.py"

    def get_execution_command(self) -> str:
        return "python -u storage/system/nikodym_eval_runner.py"

    def get_expected_function_name(self) -> str:
        return "dummy_function"

    def get_task_description(self) -> str:
        return "Three-dimensional finite-field Nikodym set optimization"

    def get_secondary_metrics_format(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "MeanRelativeSize": ".8f",
            "WorstRelativeSize": ".8f",
            "BestRelativeSize": ".8f",
            "WorstP": "d",
            "BestP": "d",
            "ValidCount": "d",
            "ScoredCount": "d",
        }

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def search_for_best_construction" not in content:
            return False, "Submission must define search_for_best_construction(p, d)."
        return True, None

    def evaluate_submission(
        self,
        result,
        eval_id: str = None,
        author: str = None,
    ):
        if result is None:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": "No output received from the Nikodym runner."
            }

        payload_text = None
        for line in str(result).splitlines():
            if "EVAL_JSON:" in line:
                payload_text = line.split("EVAL_JSON:", 1)[1].strip()

        if not payload_text:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": "Missing EVAL_JSON line in runner output."
            }

        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Invalid EVAL_JSON payload: {exc}"
            }

        score = payload.get("score", constants.RESEARCH_SCORE_NA)
        details = payload.get("details", {})
        if not isinstance(details, dict):
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": "EVAL_JSON.details must be an object."
            }

        sort_key = (float(score),) if isinstance(score, (int, float)) else (float("-inf"),)
        return True, score, details, sort_key
