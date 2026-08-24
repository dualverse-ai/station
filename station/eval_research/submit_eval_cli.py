"""
CLI entrypoint for `submit_eval.sh`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

from station import constants
from station import file_io_utils
from station.eval_research.evaluation_helpers import save_stdout_with_limit
from station.eval_research.evaluation_manager import EvaluationManager
from station.eval_research.runtime_paths import ensure_submit_runtime_layout


def _write_status(stdout_path: str, stderr_path: str, lines: list[str]):
    text = "\n".join(lines).rstrip() + "\n"
    save_stdout_with_limit(stdout_path, text)
    if not file_io_utils.file_exists(stderr_path):
        file_io_utils.save_text("", stderr_path)


def _attempt_footer(
    primary_score: Any,
    secondary_metrics: Any,
    safe_to_resubmit: bool,
    attempt_status: str = "completed",
) -> list[str]:
    normalized_status = str(attempt_status or "completed").strip().lower()
    if normalized_status == "completed":
        status_message = (
            "The submission has run to completion. You may now submit the next attempt or write the final Coder Report."
        )
    else:
        status_message = (
            f"The attempt has settled with status '{normalized_status}'. Inspect the rejection or failure details "
            "before deciding whether to retry or finalize."
        )
    return [
        "ATTEMPT_COMPLETE",
        f"ATTEMPT_STATUS: {normalized_status}",
        status_message,
        f"PRIMARY_SCORE: {primary_score}",
        f"SECONDARY_METRICS: {secondary_metrics}",
        f"SAFE_TO_RESUBMIT: {'true' if safe_to_resubmit else 'false'}",
    ]


def _attempt_queue_banner(eval_id: str, attempt_number: int) -> list[str]:
    lines = [
        "ATTEMPT_QUEUED",
        f"Official evaluation attempt {attempt_number} for evaluation {eval_id} has been queued.",
        "The attempt is waiting for an evaluator worker and any required CPU/GPU resources.",
        "This is normal. Keep polling this log and wait here while the station scheduler dispatches the official run.",
        "Do not treat the queued state by itself as a failure, and do not bypass the official submit path just because the run has not started yet.",
        "The evaluator may run outside your sandbox's PID namespace, so do not use `ps` or `pgrep` to decide whether it has ended; wait patiently for `ATTEMPT_COMPLETE`.",
    ]
    return lines


def _print_rejection(lines: list[str]):
    text = "\n".join(lines).rstrip()
    if text:
        print(text)


def _latest_attempt_is_active(eval_data: Dict[str, Any]) -> bool:
    attempts = eval_data.get("attempts") or []
    if not attempts:
        return False
    latest = attempts[-1]
    return str(latest.get("status")) in {"queued", "running"}


def _gpu_management_enabled() -> bool:
    return constants.RESEARCH_EVAL_GPU_NUM is not None or bool(constants.RESEARCH_EVAL_USE_DIFF_GPU)


def submit_eval(eval_id: str, *, cpu_only: bool = False) -> int:
    paths = ensure_submit_runtime_layout()
    eval_manager = EvaluationManager(paths.evaluations_dir, preload=False)
    eval_data = eval_manager.get_evaluation(str(eval_id))
    stdout_path = os.path.join(paths.stdout_dir, f"{eval_id}.log")
    stderr_path = os.path.join(paths.stderr_dir, f"{eval_id}.log")
    effective_cpu_only = bool(cpu_only and _gpu_management_enabled())

    if not eval_data:
        _write_status(
            stdout_path,
            stderr_path,
            [f"Submission rejected: evaluation {eval_id} not found."]
            + _attempt_footer(constants.RESEARCH_SCORE_NA, "{}", False, attempt_status="rejected"),
        )
        return 1

    if eval_data.get("status") in {"completed", "failed", "blocked", "partial"}:
        _write_status(
            stdout_path,
            stderr_path,
            [f"Submission rejected: evaluation {eval_id} is already in terminal status '{eval_data.get('status')}'."]
            + _attempt_footer(constants.RESEARCH_SCORE_NA, "{}", False, attempt_status="rejected"),
        )
        return 1

    max_attempts = int(eval_data.get("coder", {}).get("max_attempts", constants.RESEARCH_CODER_MAX_ATTEMPTS))
    attempts = eval_data.get("attempts") or []
    if len(attempts) >= max_attempts:
        _write_status(
            stdout_path,
            stderr_path,
            [f"Submission rejected: maximum attempts reached ({max_attempts})."]
            + _attempt_footer(constants.RESEARCH_SCORE_NA, "{}", False, attempt_status="rejected"),
        )
        return 1

    if _latest_attempt_is_active(eval_data):
        _print_rejection(
            [
                f"Submission rejected: an attempt is already active for evaluation {eval_id}.",
                "Keep polling the current stdout/stderr logs for the active attempt; they were not modified.",
            ]
        )
        return 1

    submission_path = os.path.join(paths.submissions_dir, f"{eval_id}.py")
    if not file_io_utils.file_exists(submission_path):
        _write_status(
            stdout_path,
            stderr_path,
            [f"Submission rejected: submission file not found at {submission_path}."]
            + _attempt_footer(constants.RESEARCH_SCORE_NA, "{}", False, attempt_status="rejected"),
        )
        return 1

    submission_text = file_io_utils.load_text(submission_path) or ""
    if not submission_text.strip():
        _write_status(
            stdout_path,
            stderr_path,
            [f"Submission rejected: submission file {submission_path} is empty."]
            + _attempt_footer(constants.RESEARCH_SCORE_NA, "{}", False, attempt_status="rejected"),
        )
        return 1

    attempt_number = eval_manager.register_attempt(str(eval_id), submission_path, cpu_only=effective_cpu_only)
    if attempt_number is None:
        _write_status(
            stdout_path,
            stderr_path,
            [f"Submission rejected: could not register attempt for evaluation {eval_id}."]
            + _attempt_footer(constants.RESEARCH_SCORE_NA, "{}", False, attempt_status="rejected"),
        )
        return 1

    queue_banner = _attempt_queue_banner(str(eval_id), int(attempt_number))
    if effective_cpu_only:
        queue_banner.append("CPU-only mode requested: the scheduler will not reserve a GPU for this attempt.")
    _write_status(stdout_path, stderr_path, queue_banner)

    run_request = {
        "eval_id": str(eval_id),
        "attempt": int(attempt_number),
        "created_timestamp": time.time(),
    }
    if effective_cpu_only:
        run_request["cpu_only"] = True
    run_request_path = os.path.join(paths.run_requests_dir, f"{eval_id}_attempt_{attempt_number}.yaml")
    file_io_utils.save_yaml(run_request, run_request_path, sort_keys=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Queue a Research Center evaluation attempt.")
    parser.add_argument("eval_id", help="Evaluation ID to submit")
    if _gpu_management_enabled():
        parser.add_argument(
            "--cpu-only",
            action="store_true",
            help="Queue this attempt without reserving a station-managed GPU.",
        )
    args = parser.parse_args(argv)
    return submit_eval(args.eval_id, cpu_only=getattr(args, "cpu_only", False))


if __name__ == "__main__":
    sys.exit(main())
