"""Fast-lane Research Center submission service for parallel ticks."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from station import constants, supervisor_utils
from station.eval_research.coder_helpers import build_coder_access_metadata
from station.eval_research.runtime_paths import ensure_lineage_storage, ensure_runtime_layout
from station.sync.fast_lane_service import FastLaneSubmissionRequest, FastLaneSubmissionService


@dataclass
class ResearchSubmissionResult:
    accepted: bool
    messages: List[str]
    eval_id: Optional[str] = None
    notification: Optional[str] = None
    error: Optional[str] = None


class ResearchSubmissionService(FastLaneSubmissionService):
    """Single-writer service for Research submit actions received in parallel ticks."""
    service_name = "ResearchSubmissionService"

    def __init__(
        self,
        station_instance: Any,
        *,
        log_event_func: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        super().__init__(station_instance, log_event_func=log_event_func)

    def _process_request(self, request: FastLaneSubmissionRequest) -> ResearchSubmissionResult:
        consts = constants
        agent_data = copy.deepcopy(request.agent_data)
        yaml_data = copy.deepcopy(request.yaml_data)

        eval_manager = getattr(self.station, "research_eval_manager", None)
        if eval_manager is None:
            return ResearchSubmissionResult(
                accepted=False,
                messages=["Submission failed: Research Center evaluator is unavailable."],
                error="missing_eval_manager",
            )

        if agent_data.get(consts.AGENT_STATUS_KEY) == consts.AGENT_STATUS_GUEST:
            return ResearchSubmissionResult(
                accepted=False,
                messages=["The Research Center is only accessible to Recursive Agents."],
                error="guest_agent",
            )

        is_holiday = (
            self.station.is_holiday_tick(request.current_tick)
            if self.station and hasattr(self.station, "is_holiday_tick")
            else (consts.HOLIDAY_MODE_ENABLED and consts.is_holiday_tick(request.current_tick))
        )
        if is_holiday:
            return ResearchSubmissionResult(
                accepted=False,
                messages=["Holidays are not for working - please try again on working days."],
                error="holiday",
            )

        if supervisor_utils.is_supervisor(agent_data, consts):
            return ResearchSubmissionResult(
                accepted=False,
                messages=["Supervisors cannot submit research experiments."],
                error="supervisor",
            )

        from station.rooms.research_center import ResearchCenter

        if ResearchCenter._must_read_task_before_submit(agent_data, consts):
            return ResearchSubmissionResult(
                accepted=False,
                messages=["Submission failed: read the current Research Task first with `/execute_action{read_task}`."],
                error="task_not_read",
            )

        cooldown_ticks = ResearchCenter._get_submission_cooldown_ticks(agent_data, consts)
        last_submission_tick = agent_data.get(
            consts.AGENT_LAST_RESEARCH_SUBMISSION_TICK_KEY,
            -cooldown_ticks,
        )
        if cooldown_ticks > 0 and request.current_tick - last_submission_tick < cooldown_ticks:
            remaining = cooldown_ticks - (request.current_tick - last_submission_tick)
            return ResearchSubmissionResult(
                accepted=False,
                messages=[f"Submission failed: cooldown active. Try again in {remaining} tick(s)."],
                error="cooldown",
            )

        if not yaml_data:
            return ResearchSubmissionResult(
                accepted=False,
                messages=["Submission requires YAML data with `title`, `tags`, `abstract`, and `instruction` fields."],
                error="missing_yaml",
            )

        title = str(yaml_data.get(consts.YAML_CAPSULE_TITLE, "")).strip()
        instruction = str(yaml_data.get("instruction", "")).rstrip()
        tags_input = yaml_data.get(consts.YAML_CAPSULE_TAGS, "")
        abstract = str(yaml_data.get(consts.YAML_CAPSULE_ABSTRACT, "")).strip()

        if not title or not instruction.strip() or not tags_input or not abstract:
            return ResearchSubmissionResult(
                accepted=False,
                messages=["Submission requires `title`, `tags`, `abstract`, and `instruction`."],
                error="missing_fields",
            )

        tags_valid, tags_error, parsed_tags, tags_warning = ResearchCenter._validate_tags(tags_input)
        if not tags_valid:
            return ResearchSubmissionResult(
                accepted=False,
                messages=[f"Tags validation failed: {tags_error}"],
                error="tags",
            )

        abstract_valid, abstract_error, processed_abstract, abstract_warning = ResearchCenter._validate_abstract(abstract)
        if not abstract_valid:
            return ResearchSubmissionResult(
                accepted=False,
                messages=[f"Abstract validation failed: {abstract_error}"],
                error="abstract",
            )

        agent_name = str(agent_data.get(consts.AGENT_NAME_KEY, "Unknown"))
        lineage = str(agent_data.get(consts.AGENT_LINEAGE_KEY, "unknown")).lower()
        paths = ensure_runtime_layout()
        ensure_lineage_storage(paths, lineage)

        eval_data, active_ids = eval_manager.create_instruction_evaluation_atomic(
            author=agent_name,
            title=title,
            content=instruction,
            tick=request.current_tick,
            tags=parsed_tags,
            abstract=processed_abstract,
            lineage=lineage,
            max_active_for_author=consts.RESEARCH_MAX_CONCURRENT_SUBMISSIONS,
            extra_metadata={
                "coder_access": build_coder_access_metadata(agent_data, request.current_tick, consts),
                "parallel_commit_status": "provisional",
                "parallel_tick": {
                    "run_id": request.run_id,
                    "op_id": request.op_id,
                    "created_timestamp": time.time(),
                },
            },
        )

        if eval_data is None:
            return ResearchSubmissionResult(
                accepted=False,
                messages=[ResearchCenter._format_active_limit_reached_message(active_ids, consts)],
                error="active_limit",
            )

        eval_id = str(eval_data.get("id"))
        warning_parts = [part for part in [tags_warning, abstract_warning] if part]
        message = f"Experiment '{title}' submitted. Evaluation ID: {eval_id}."
        if warning_parts:
            message += " " + " ".join(warning_parts)
        notification = f"Your experiment request has been queued for the coder. Evaluation ID: {eval_id}."

        self._wake_auto_evaluator(eval_id)
        self._push_log_event(
            "parallel_research_submission_accepted",
            {"agent_name": agent_name, "eval_id": eval_id, "op_id": request.op_id},
        )

        return ResearchSubmissionResult(
            accepted=True,
            messages=[message],
            eval_id=eval_id,
            notification=notification,
        )

    def _default_timeout_seconds(self) -> float:
        return float(constants.PARALLEL_RESEARCH_SUBMISSION_TIMEOUT_SECONDS)

    def _timeout_result(self) -> ResearchSubmissionResult:
        return ResearchSubmissionResult(
            accepted=False,
            messages=["Submission failed: Research submission service timed out."],
            error="timeout",
        )

    def _empty_result(self) -> ResearchSubmissionResult:
        return ResearchSubmissionResult(
            accepted=False,
            messages=["Submission failed: Research submission service returned no result."],
            error="empty_result",
        )

    def _exception_result(self, exc: Exception, request: FastLaneSubmissionRequest) -> ResearchSubmissionResult:
        return ResearchSubmissionResult(
            accepted=False,
            messages=[f"Submission failed: internal Research submission error: {exc}"],
            error=str(exc),
        )

    def _exception_event_type(self) -> str:
        return "parallel_research_submission_error"

    def _wake_auto_evaluator(self, eval_id: str) -> None:
        evaluator = getattr(self.station, "auto_research_evaluator", None)
        if evaluator and hasattr(evaluator, "wake"):
            try:
                evaluator.wake(reason=f"parallel submission {eval_id}")
            except Exception as exc:
                self._push_log_event(
                    "parallel_research_wake_error",
                    {"eval_id": eval_id, "error": str(exc)},
                )
