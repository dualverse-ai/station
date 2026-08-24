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

"""
Evaluation Manager for Research Center evaluations.

New evaluations are stored as one YAML file per evaluation at:
`evaluations/{eval_id}.yaml`
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import filelock

from station import constants
from station import file_io_utils
from station.eval_research.evaluation_helpers import truncate_persisted_stdout
from . import evaluation_index
from .runtime_paths import build_runtime_paths


TERMINAL_EVALUATION_STATUSES = {"completed", "success", "failed", "blocked", "partial"}
ACTIVE_EVALUATION_STATUSES = {"queued", "running"}
STDOUT_STDERR_HIDDEN_NOTICE = (
    "Stdout/stderr are not shown here. To retrieve information that was only printed in those logs, submit "
    "another instruction asking the coder to summarize it in the Coder Report or copy the relevant data to "
    "accessible Research storage."
)


def _normalize_evaluation_status(status: Optional[str]) -> Optional[str]:
    if status is None:
        return None
    normalized = str(status).strip().lower()
    if not normalized:
        return None
    if normalized == "success":
        return "completed"
    if normalized in {"running", "coder_running", "attempt_running", "waiting_for_attempt", "waiting_for_report"}:
        return "running"
    if normalized == "queued":
        return "queued"
    if normalized in {"completed", "failed", "blocked", "partial"}:
        return normalized
    return normalized


def _get_default_evaluations_dir() -> str:
    return build_runtime_paths(constants).evaluations_dir


def _get_yaml_eval_path(eval_id: str, evaluations_dir: Optional[str] = None) -> str:
    evaluations_dir = evaluations_dir or _get_default_evaluations_dir()
    return os.path.join(evaluations_dir, f"{eval_id}{constants.RESEARCH_EVALUATION_FILE_EXTENSION}")


def _load_yaml_evaluation(eval_id: str, evaluations_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    data = file_io_utils.load_yaml(_get_yaml_eval_path(eval_id, evaluations_dir))
    return data if isinstance(data, dict) else None


def _load_evaluation_record(eval_id: str, evaluations_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    return _load_yaml_evaluation(eval_id, evaluations_dir)


_ARTIFACT_SPECS = {
    "submission": ("submission", ".py"),
    "stdout": ("stdout", ".log"),
    "stderr": ("stderr", ".log"),
    "report": ("report", ".md"),
}


def _research_root_from_evaluations_dir(evaluations_dir: Optional[str] = None) -> str:
    return os.path.dirname(os.path.abspath(evaluations_dir or _get_default_evaluations_dir()))


def _default_artifact_rel_path(eval_id: str, artifact_key: str) -> str:
    subdir, extension = _ARTIFACT_SPECS[artifact_key]
    return os.path.join(constants.RESEARCH_STORAGE_DIR, subdir, f"{eval_id}{extension}")


def _default_artifact_map(eval_id: str) -> Dict[str, str]:
    return {key: _default_artifact_rel_path(str(eval_id), key) for key in _ARTIFACT_SPECS}


def _ensure_default_artifacts(eval_data: Dict[str, Any], eval_id: str) -> Dict[str, str]:
    artifacts = eval_data.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        eval_data["artifacts"] = artifacts
    for key, path in _default_artifact_map(str(eval_id)).items():
        artifacts.setdefault(key, path)
    return artifacts


def _resolve_research_path(path: str, evaluations_dir: Optional[str] = None) -> str:
    if os.path.isabs(path):
        return path
    normalized = path[2:] if path.startswith("./") else path
    if normalized.startswith(constants.RESEARCH_STORAGE_DIR + os.sep):
        return os.path.join(_research_root_from_evaluations_dir(evaluations_dir), normalized)
    return os.path.abspath(path)


def _get_eval_artifacts(eval_data: Dict[str, Any]) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    top_level = eval_data.get("artifacts")
    if isinstance(top_level, dict):
        artifacts.update({str(key): str(value) for key, value in top_level.items() if value})
    final = eval_data.get("final") or {}
    final_artifacts = final.get("artifacts")
    if isinstance(final_artifacts, dict):
        artifacts.update({str(key): str(value) for key, value in final_artifacts.items() if value})
    return artifacts


def _get_artifact_path(eval_data: Dict[str, Any], artifact_key: str, evaluations_dir: Optional[str] = None) -> str:
    eval_id = str(eval_data.get("id"))
    artifact_path = _get_eval_artifacts(eval_data).get(artifact_key) or _default_artifact_rel_path(eval_id, artifact_key)
    return _resolve_research_path(artifact_path, evaluations_dir)


def _load_eval_artifact(eval_data: Dict[str, Any], artifact_key: str, evaluations_dir: Optional[str] = None) -> str:
    path = _get_artifact_path(eval_data, artifact_key, evaluations_dir)
    return file_io_utils.load_text(path) if file_io_utils.file_exists(path) else ""


def _load_eval_final_artifact(eval_data: Dict[str, Any], artifact_key: str, evaluations_dir: Optional[str] = None) -> str:
    final = eval_data.get("final") or {}
    final_artifacts = final.get("artifacts")
    if isinstance(final_artifacts, dict):
        artifact_path = final_artifacts.get(artifact_key)
        if not artifact_path:
            return ""
        path = _resolve_research_path(str(artifact_path), evaluations_dir)
        return file_io_utils.load_text(path) if file_io_utils.file_exists(path) else ""
    return _load_eval_artifact(eval_data, artifact_key, evaluations_dir)


def _strip_attempt_footer(stdout_text: str) -> str:
    if not stdout_text:
        return stdout_text

    marker = "\nATTEMPT_COMPLETE"
    idx = stdout_text.find(marker)
    if idx >= 0:
        return stdout_text[:idx].rstrip()

    lines = stdout_text.rstrip().splitlines()
    if lines and lines[-1].strip().startswith("ATTEMPT_COMPLETE"):
        return "\n".join(lines[:-1]).rstrip()
    return stdout_text


def _truncate_visible_stdout(stdout_text: str) -> str:
    if not stdout_text:
        return stdout_text
    max_chars = getattr(constants, "RESEARCH_EVAL_LOG_MAX_CHARS", 15000)
    if len(stdout_text) > max_chars:
        return stdout_text[:max_chars] + f"\n\n[... truncated after {max_chars:,} characters]"
    return stdout_text


def _get_final_attempt(eval_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    attempts = eval_data.get("attempts") or []
    if not attempts:
        return None
    return attempts[-1]


def _cleanup_eval_tmp_storage(eval_data: Dict[str, Any], evaluations_dir: str) -> bool:
    eval_id = str(eval_data.get("id") or "")
    status = _normalize_evaluation_status(eval_data.get("status"))
    if not eval_id or status not in TERMINAL_EVALUATION_STATUSES:
        return False

    lineage = str(eval_data.get("lineage") or "unknown").lower()
    research_root = _research_root_from_evaluations_dir(evaluations_dir)
    storage_root = os.path.join(research_root, constants.RESEARCH_STORAGE_DIR)
    storage_real_root = os.path.realpath(storage_root)
    tmp_root = os.path.realpath(os.path.join(storage_real_root, "tmp"))
    lineage_tmp = os.path.realpath(os.path.join(tmp_root, lineage))
    eval_tmp = os.path.realpath(os.path.join(lineage_tmp, f"eval_{eval_id}"))

    expected_basename = f"eval_{eval_id}"
    if os.path.basename(eval_tmp) != expected_basename:
        return False
    try:
        lineage_common = os.path.commonpath([tmp_root, lineage_tmp])
        eval_common = os.path.commonpath([lineage_tmp, eval_tmp])
    except ValueError:
        return False
    if lineage_common != tmp_root or eval_common != lineage_tmp:
        return False
    if not os.path.isdir(eval_tmp):
        return False

    try:
        shutil.rmtree(eval_tmp)
        try:
            os.rmdir(lineage_tmp)
        except OSError:
            pass
        return True
    except Exception as exc:
        print(
            "EvaluationManager: failed to remove research tmp storage "
            f"eval_id={eval_id!r} path={eval_tmp!r}: {exc}"
        )
        return False


def format_tags_for_display(tags: Optional[List[str]]) -> str:
    if not tags:
        return "—"
    return ", ".join(tags)


def format_score_for_display(score: Any, precision: Optional[int] = None) -> str:
    if precision is None:
        precision = constants.RESEARCH_SCORE_DISPLAY_PRECISION

    if score in [constants.RESEARCH_SCORE_PENDING, constants.RESEARCH_SCORE_NA]:
        return str(score)
    if score is None:
        return "—"
    if isinstance(score, (list, tuple)):
        formatted = []
        for element in score:
            if isinstance(element, float):
                formatted.append(f"{element:.{precision}f}")
            else:
                formatted.append(str(element))
        return f"[{', '.join(formatted)}]"
    if isinstance(score, float):
        return f"{score:.{precision}f}"
    return str(score)


def extract_secondary_metrics_for_display_info(eval_data: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    details = eval_data.get(constants.EVALUATION_DETAILS_KEY, "")
    if isinstance(details, str):
        return details, {}
    if not isinstance(details, dict):
        return str(details), {}

    message = str(details.get("Message", ""))
    metrics_dict: Dict[str, str] = {}
    for key, value in details.items():
        if key == "Message":
            continue
        if isinstance(value, (tuple, list)) and len(value) == 2:
            metrics_dict[key] = str(value[0])
        else:
            metrics_dict[key] = str(value)
    return message, metrics_dict


def _format_secondary_metrics_for_display(evaluation_details: Any) -> Tuple[str, str]:
    message, metrics = extract_secondary_metrics_for_display_info({constants.EVALUATION_DETAILS_KEY: evaluation_details})
    if not metrics:
        return message, ""
    formatted = "\n" + "\n".join(f"**{key}:** {value}" for key, value in metrics.items())
    return message, formatted


def _build_evaluation_preview_from_display_info(eval_id: str, display_info: Dict[str, Any]) -> str:
    title = display_info.get(constants.EVALUATION_TITLE_KEY, "No title")
    author = display_info.get(constants.EVALUATION_AUTHOR_KEY, "Unknown")
    tags = display_info.get(constants.EVALUATION_TAGS_KEY, [])
    abstract = display_info.get(constants.EVALUATION_ABSTRACT_KEY, "No abstract")
    score = display_info.get(constants.EVALUATION_SCORE_KEY, constants.RESEARCH_SCORE_NA)

    preview = (
        f"**Evaluation {eval_id}: {title}**\n"
        f"Author: {author}\n"
        f"Tags: {format_tags_for_display(tags)}\n"
        f"Abstract: {abstract}"
    )
    if not constants.RESEARCH_NO_SCORE:
        display_score = "running" if score == constants.RESEARCH_SCORE_PENDING else format_score_for_display(score)
        details_message = ""
        if score not in (constants.RESEARCH_SCORE_PENDING, constants.RESEARCH_SCORE_NA):
            details_message, metrics = extract_secondary_metrics_for_display_info(display_info)
            score_parts = [display_score]
            score_parts.extend(f"{key}: {value}" for key, value in metrics.items())
            display_score = " | ".join(score_parts)
        preview += f"\nScore: {display_score}"
        if details_message:
            preview += f"\nMessage:\n{details_message}"
    return preview


def _build_new_schema_display_info(eval_data: Dict[str, Any]) -> Dict[str, Any]:
    final = eval_data.get("final") or {}
    status = _normalize_evaluation_status(eval_data.get("status")) or "queued"

    if final and status in TERMINAL_EVALUATION_STATUSES:
        score = final.get("primary_score", constants.RESEARCH_SCORE_NA)
        details = final.get(constants.EVALUATION_DETAILS_KEY, "")
        sort_key = final.get("sort_key")
    else:
        score = constants.RESEARCH_SCORE_PENDING
        details = ""
        sort_key = None

    result = {
        constants.EVALUATION_ID_KEY: str(eval_data.get("id")),
        constants.EVALUATION_AUTHOR_KEY: eval_data.get("author", "Unknown"),
        constants.EVALUATION_TITLE_KEY: eval_data.get("title", "Untitled"),
        constants.EVALUATION_TAGS_KEY: eval_data.get("tags", []),
        constants.EVALUATION_ABSTRACT_KEY: eval_data.get("abstract", ""),
        constants.EVALUATION_SCORE_KEY: score,
        constants.EVALUATION_SUBMITTED_TICK_KEY: eval_data.get("submitted_tick", 0),
        constants.EVALUATION_DETAILS_KEY: details,
        constants.EVALUATION_STATUS_KEY: status,
    }
    if sort_key is not None:
        result["sort_key"] = sort_key
    return result


def get_evaluation_display_info(eval_id: str, evaluations_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    eval_data = _load_evaluation_record(eval_id, evaluations_dir)
    if not eval_data:
        return None
    return _build_new_schema_display_info(eval_data)


def build_evaluation_previews(eval_ids: List[str], evaluations_dir: Optional[str] = None) -> List[str]:
    previews: List[str] = []
    for eval_id in eval_ids:
        display_info = get_evaluation_display_info(eval_id, evaluations_dir)
        if not display_info:
            previews.append(f"**Evaluation {eval_id}:** Not found")
            continue

        previews.append(_build_evaluation_preview_from_display_info(eval_id, display_info))
    return previews


def _build_new_schema_review_info(eval_data: Dict[str, Any], evaluations_dir: Optional[str] = None) -> Dict[str, str]:
    final = eval_data.get("final") or {}
    if not final:
        return {
            "status": "pending",
            "message": f"Evaluation '{eval_data.get('id')}' is still pending. Please try again later.",
        }

    score = final.get("primary_score", constants.RESEARCH_SCORE_NA)
    coder_report = _load_eval_final_artifact(eval_data, "report", evaluations_dir).strip()
    instruction = str(eval_data.get("instruction", "")).strip()
    details = final.get(constants.EVALUATION_DETAILS_KEY, "")
    details_message, secondary_metrics_string = _format_secondary_metrics_for_display(details)
    tags_display = format_tags_for_display(eval_data.get("tags", []))

    message = (
        f"**Research Submission Review**\n\n"
        f"**Title:** {eval_data.get('title', 'Untitled')}\n"
        f"**ID:** {eval_data.get('id')}\n"
        f"**Tags:** {tags_display}\n"
        f"**Abstract:** {eval_data.get('abstract', '')}"
    )

    message += "\n\n**Instruction Prompt:**\n"
    message += f"```\n{instruction}\n```" if instruction else "_No instruction recorded._"

    if not constants.RESEARCH_NO_SCORE:
        message += f"\n\n**Score:** {score}"
        if secondary_metrics_string:
            message += secondary_metrics_string
        message += f"\n**Evaluation Details:** {details_message}"

    message += "\n\n**Coder Report:**\n"
    message += coder_report if coder_report else "_No coder report available._"
    message += f"\n\n**Stdout/Stderr:** {STDOUT_STDERR_HIDDEN_NOTICE}"

    return {"status": "completed", "message": message}


def _build_new_schema_result_summary(eval_data: Dict[str, Any]) -> Dict[str, Any]:
    final = eval_data.get("final") or {}
    status = _normalize_evaluation_status(eval_data.get("status")) or "queued"
    score = final.get("primary_score", constants.RESEARCH_SCORE_NA)
    success = bool(
        final
        and status in TERMINAL_EVALUATION_STATUSES
        and score not in [constants.RESEARCH_SCORE_PENDING, constants.RESEARCH_SCORE_NA, None]
    )
    return {
        "evaluation_id": str(eval_data.get("id")),
        "author": eval_data.get("author", "Unknown"),
        "title": eval_data.get("title", "Untitled"),
        "submitted_tick": eval_data.get("submitted_tick", 0),
        "score": score if success else constants.RESEARCH_SCORE_NA,
        "success": success,
        "sort_key": final.get("sort_key") if success else None,
        "status": status,
    }


def get_evaluation_result_summary(eval_id: str, evaluations_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    eval_data = _load_evaluation_record(eval_id, evaluations_dir)
    if not eval_data:
        return None
    return _build_new_schema_result_summary(eval_data)


def get_evaluation_review_info(eval_id: str, evaluations_dir: Optional[str] = None) -> Optional[Dict[str, str]]:
    eval_data = _load_evaluation_record(eval_id, evaluations_dir)
    if not eval_data:
        return None
    return _build_new_schema_review_info(eval_data, evaluations_dir)


def get_evaluation_code_info(eval_id: str, evaluations_dir: Optional[str] = None) -> Optional[Dict[str, str]]:
    eval_data = _load_evaluation_record(eval_id, evaluations_dir)
    if not eval_data:
        return None

    final = eval_data.get("final") or {}
    code = _load_eval_final_artifact(eval_data, "submission", evaluations_dir) if final else ""
    if not code:
        return {
            "status": "pending",
            "message": f"Evaluation '{eval_id}' does not have a final code snapshot yet.",
        }
    return {"status": "completed", "code": code}


def _build_new_schema_submission_payload(eval_data: Dict[str, Any]) -> Dict[str, Any]:
    final = eval_data.get("final", {}) or {}
    evaluations_dir = eval_data.get("_evaluations_dir")
    stdout_text = _load_eval_final_artifact(eval_data, "stdout", evaluations_dir) if final else ""
    stderr_text = _load_eval_final_artifact(eval_data, "stderr", evaluations_dir) if final else ""
    logs_parts = []
    if stdout_text:
        logs_parts.append(f"STDOUT:\n{stdout_text}")
    if stderr_text:
        logs_parts.append(f"STDERR:\n{stderr_text}")
    return {
        "version_info": " (final)",
        "code": (_load_eval_final_artifact(eval_data, "submission", evaluations_dir) if final else "") or "Code not available",
        "score": final.get("primary_score", "N/A"),
        "status": final.get("status", eval_data.get("status", "unknown")),
        "logs": "\n\n".join(logs_parts) if logs_parts else "Logs not available",
        "details": final.get(constants.EVALUATION_DETAILS_KEY, ""),
        "error": final.get("error", ""),
        "instruction": eval_data.get("instruction", ""),
    }


def get_evaluation_submission_payload(eval_id: str, evaluations_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    eval_data = _load_evaluation_record(eval_id, evaluations_dir)
    if not eval_data:
        return None
    eval_data = copy.deepcopy(eval_data)
    eval_data["_evaluations_dir"] = evaluations_dir
    return _build_new_schema_submission_payload(eval_data)


class EvaluationManager:
    """
    Manages Research Center evaluation files and derived display state.
    """

    def __init__(self, evaluations_dir: Optional[str] = None, *, preload: bool = True):
        self.evaluations_dir = evaluations_dir or _get_default_evaluations_dir()
        self._notification_callback: Optional[Callable[[str, str], None]] = None
        self._top_submission_callback: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None
        self._lock = threading.Lock()
        self._submission_lock = threading.RLock()
        self._submission_file_lock_path = os.path.join(self.evaluations_dir, ".submission.lock")
        self._evaluation_record_cache: Dict[str, Dict[str, Any]] = {}
        self._eval_file_mtimes: Dict[str, int] = {}
        self.top_submission: Optional[Dict[str, Any]] = None
        self._preload_complete = False
        file_io_utils.ensure_dir_exists(self.evaluations_dir)
        evaluation_index.ensure_research_evaluation_index(
            self.evaluations_dir,
            rebuild=evaluation_index.should_rebuild_from_process_args(),
            log_status=preload,
        )
        evaluation_index.refresh_top_submission_from_index(self.evaluations_dir)
        self.top_submission = evaluation_index.get_top_submission(self.evaluations_dir)
        self._preload_complete = True

    def set_notification_callback(self, callback: Callable[[str, str], None]):
        self._notification_callback = callback

    def set_top_submission_callback(self, callback: Callable[[Optional[Dict[str, Any]]], None]):
        self._top_submission_callback = callback

    @contextmanager
    def _file_lock(self, eval_id: str, timeout: float = 60):
        lock_path = _get_yaml_eval_path(eval_id, self.evaluations_dir) + ".lock"
        lock = filelock.FileLock(lock_path, timeout=timeout)
        try:
            with lock:
                yield
        except filelock.Timeout:
            print(
                "EvaluationManager: failed to acquire evaluation lock "
                f"eval_id={eval_id!r} lock_path={lock_path!r} timeout={timeout}s"
            )
            raise

    def _save_evaluation(self, eval_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        old_top_submission = evaluation_index.get_top_submission(self.evaluations_dir)
        previous_dir_mtime_ns = self._get_evaluations_dir_mtime_ns()
        file_io_utils.save_yaml(data, _get_yaml_eval_path(eval_id, self.evaluations_dir), sort_keys=False)
        self._bump_evaluations_dir_mtime(previous_dir_mtime_ns)
        try:
            evaluation_index.upsert_evaluation(data, self.evaluations_dir)
        except Exception as e:
            print(f"ResearchIndex: update failed eval_id={eval_id!r}: {e}")
            raise
        self._sync_eval_record_in_memory(
            data,
            mtime_ns=self._get_eval_mtime_ns(str(eval_id)),
            emit_top_submission_callback=False,
        )
        new_top_submission = evaluation_index.get_top_submission(self.evaluations_dir)
        with self._lock:
            self.top_submission = copy.deepcopy(new_top_submission) if new_top_submission else None
        if (
            self._top_submission_callback
            and not self._top_submissions_equal(old_top_submission, new_top_submission)
        ):
            self._top_submission_callback(copy.deepcopy(new_top_submission) if new_top_submission else None)
        return data

    def _load_evaluation_yaml(self, eval_id: str) -> Optional[Dict[str, Any]]:
        return _load_yaml_evaluation(eval_id, self.evaluations_dir)

    def _load_evaluation_any(self, eval_id: str) -> Optional[Dict[str, Any]]:
        return _load_evaluation_record(eval_id, self.evaluations_dir)

    def get_evaluation(self, eval_id: str) -> Optional[Dict[str, Any]]:
        self._refresh_from_disk_if_needed()
        eval_id = str(eval_id)
        mtime_ns = self._get_eval_mtime_ns(eval_id)
        with self._lock:
            cached = self._evaluation_record_cache.get(eval_id)
            cached_mtime_ns = self._eval_file_mtimes.get(eval_id)
        if cached is not None and cached_mtime_ns == mtime_ns:
            return copy.deepcopy(cached)
        if mtime_ns is None:
            evaluation_index.delete_evaluation(eval_id, self.evaluations_dir)
            self._sync_deleted_eval_in_memory(eval_id, emit_top_submission_callback=False)
            return None

        eval_data = self._load_evaluation_any(eval_id)
        if not eval_data:
            evaluation_index.delete_evaluation(eval_id, self.evaluations_dir)
            self._sync_deleted_eval_in_memory(eval_id, emit_top_submission_callback=False)
            return None
        evaluation_index.upsert_evaluation(eval_data, self.evaluations_dir)
        self._sync_eval_record_in_memory(eval_data, mtime_ns=mtime_ns, emit_top_submission_callback=False)
        return copy.deepcopy(eval_data)

    def get_display_info(self, eval_id: str) -> Optional[Dict[str, Any]]:
        return evaluation_index.get_display_info(str(eval_id), self.evaluations_dir)

    def get_compact_display_infos(self) -> List[Dict[str, Any]]:
        return evaluation_index.list_display_infos(self.evaluations_dir)

    def get_review_info(self, eval_id: str) -> Optional[Dict[str, str]]:
        eval_data = self.get_evaluation(eval_id)
        if not eval_data:
            return None
        return _build_new_schema_review_info(eval_data, self.evaluations_dir)

    def get_code_info(self, eval_id: str) -> Optional[Dict[str, str]]:
        eval_data = self.get_evaluation(eval_id)
        if not eval_data:
            return None

        final = eval_data.get("final") or {}
        code = _load_eval_final_artifact(eval_data, "submission", self.evaluations_dir) if final else ""
        if not code:
            return {
                "status": "pending",
                "message": f"Evaluation '{eval_id}' does not have a final code snapshot yet.",
            }
        return {"status": "completed", "code": code}

    def get_result_summary(self, eval_id: str) -> Optional[Dict[str, Any]]:
        eval_data = self.get_evaluation(eval_id)
        if not eval_data:
            return None
        return _build_new_schema_result_summary(eval_data)

    def get_submission_payload(self, eval_id: str) -> Optional[Dict[str, Any]]:
        eval_data = self.get_evaluation(eval_id)
        if not eval_data:
            return None
        eval_data["_evaluations_dir"] = self.evaluations_dir
        return _build_new_schema_submission_payload(eval_data)

    def build_evaluation_previews(self, eval_ids: List[str]) -> List[str]:
        previews: List[str] = []
        for eval_id in eval_ids:
            display_info = self.get_display_info(eval_id)
            if not display_info:
                previews.append(f"**Evaluation {eval_id}:** Not found")
                continue

            previews.append(_build_evaluation_preview_from_display_info(eval_id, display_info))
        return previews

    @staticmethod
    def _sorted_eval_ids(eval_ids: List[str]) -> List[str]:
        return sorted(eval_ids, key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)))

    def _get_eval_mtime_ns(self, eval_id: str) -> Optional[int]:
        try:
            return os.stat(_get_yaml_eval_path(str(eval_id), self.evaluations_dir)).st_mtime_ns
        except (FileNotFoundError, OSError):
            return None

    def _get_evaluations_dir_mtime_ns(self) -> Optional[int]:
        try:
            return os.stat(self.evaluations_dir).st_mtime_ns
        except (FileNotFoundError, OSError):
            return None

    def _bump_evaluations_dir_mtime(self, previous_mtime_ns: Optional[int] = None):
        try:
            target_ns = time.time_ns()
            if previous_mtime_ns is not None:
                target_ns = max(target_ns, int(previous_mtime_ns) + 1)
            os.utime(self.evaluations_dir, ns=(target_ns, target_ns))
        except (FileNotFoundError, OSError):
            pass

    def _to_yaml_safe(self, value: Any) -> Any:
        if isinstance(value, tuple):
            return [self._to_yaml_safe(item) for item in value]
        if isinstance(value, list):
            return [self._to_yaml_safe(item) for item in value]
        if isinstance(value, set):
            return [self._to_yaml_safe(item) for item in self._sorted_eval_ids([str(item) for item in value])]
        if isinstance(value, dict):
            return {str(key): self._to_yaml_safe(item) for key, item in value.items()}
        return copy.deepcopy(value)

    def _top_submissions_equal(self, first: Optional[Dict[str, Any]], second: Optional[Dict[str, Any]]) -> bool:
        return self._to_yaml_safe(first) == self._to_yaml_safe(second)

    def _sync_eval_record_in_memory(
        self,
        eval_data: Dict[str, Any],
        *,
        mtime_ns: Optional[int],
        emit_top_submission_callback: bool,
    ):
        eval_id = str(eval_data.get("id", "")).strip()
        if not eval_id:
            return
        with self._lock:
            old_top_submission = copy.deepcopy(self.top_submission)
            self._evaluation_record_cache.pop(eval_id, None)
            self._evaluation_record_cache[eval_id] = copy.deepcopy(eval_data)
            if mtime_ns is not None:
                self._eval_file_mtimes[eval_id] = int(mtime_ns)
            db_top_submission = evaluation_index.get_top_submission(self.evaluations_dir)
            self.top_submission = copy.deepcopy(db_top_submission) if db_top_submission else None
            new_top_submission = copy.deepcopy(self.top_submission)

        if (
            emit_top_submission_callback
            and self._top_submission_callback
            and not self._top_submissions_equal(old_top_submission, new_top_submission)
        ):
            self._top_submission_callback(copy.deepcopy(new_top_submission) if new_top_submission else None)

    def _sync_deleted_eval_in_memory(self, eval_id: str, *, emit_top_submission_callback: bool):
        eval_id = str(eval_id)
        with self._lock:
            old_top_submission = copy.deepcopy(self.top_submission)
            self._evaluation_record_cache.pop(eval_id, None)
            self._eval_file_mtimes.pop(eval_id, None)
            db_top_submission = evaluation_index.get_top_submission(self.evaluations_dir)
            self.top_submission = copy.deepcopy(db_top_submission) if db_top_submission else None
            new_top_submission = copy.deepcopy(self.top_submission)

        if (
            emit_top_submission_callback
            and self._top_submission_callback
            and not self._top_submissions_equal(old_top_submission, new_top_submission)
        ):
            self._top_submission_callback(copy.deepcopy(new_top_submission) if new_top_submission else None)

    def _rebuild_indexes_and_top_submission(self):
        evaluation_index.rebuild_research_evaluation_index(self.evaluations_dir)
        with self._lock:
            old_top_submission = copy.deepcopy(self.top_submission)
            self._evaluation_record_cache = {}
            self._eval_file_mtimes = {}
            self.top_submission = evaluation_index.get_top_submission(self.evaluations_dir)
            self._preload_complete = True
            new_top_submission = copy.deepcopy(self.top_submission)

        if self._top_submission_callback and not self._top_submissions_equal(old_top_submission, new_top_submission):
            self._top_submission_callback(copy.deepcopy(new_top_submission) if new_top_submission else None)

    def _ensure_preloaded(self):
        evaluation_index.ensure_research_evaluation_index(self.evaluations_dir)
        with self._lock:
            self._preload_complete = True

    def _refresh_from_disk_if_needed(self):
        evaluation_index.ensure_research_evaluation_index(self.evaluations_dir)
        db_top_submission = evaluation_index.get_top_submission(self.evaluations_dir)
        with self._lock:
            old_top_submission = copy.deepcopy(self.top_submission)
            self.top_submission = copy.deepcopy(db_top_submission) if db_top_submission else None
            new_top_submission = copy.deepcopy(self.top_submission)

        if self._top_submission_callback and not self._top_submissions_equal(old_top_submission, new_top_submission):
            self._top_submission_callback(copy.deepcopy(new_top_submission) if new_top_submission else None)

    def get_all_evaluation_ids(self) -> List[str]:
        return evaluation_index.get_all_evaluation_ids(self.evaluations_dir)

    def refresh_from_disk(self):
        self._rebuild_indexes_and_top_submission()

    def get_next_evaluation_id(self) -> str:
        return evaluation_index.get_next_evaluation_id(self.evaluations_dir)

    def create_evaluation(
        self,
        eval_id: str,
        author: str,
        title: str,
        content: str,
        tick: int,
        tags: Optional[List[str]] = None,
        abstract: str = "",
        lineage: Optional[str] = None,
        backend: Optional[str] = None,
        model_name: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        eval_id = str(eval_id)

        with self._file_lock(eval_id):
            existing = self._load_evaluation_yaml(eval_id)
            if existing is not None:
                self._sync_eval_record_in_memory(
                    existing,
                    mtime_ns=self._get_eval_mtime_ns(eval_id),
                    emit_top_submission_callback=False,
                )
                return existing

            now = time.time()
            eval_data = {
                "schema_version": 2,
                "id": eval_id,
                "author": author,
                "lineage": (lineage or "unknown").lower(),
                "title": title,
                "tags": tags or [],
                "abstract": abstract or "",
                "instruction": content or "",
                "submitted_tick": tick,
                "submitted_timestamp": now,
                "status": "queued",
                "artifacts": _default_artifact_map(eval_id),
                "coder": {
                    "backend": backend or constants.RESEARCH_CODER_BACKEND,
                    "model_name": model_name if model_name is not None else constants.RESEARCH_CODER_MODEL_NAME,
                    "active": False,
                    "status": "queued",
                    "spawn_count": 0,
                    "max_spawns": constants.RESEARCH_CODER_MAX_SPAWNS,
                    "resume_count": 0,
                    "max_resumes": constants.RESEARCH_CODER_MAX_RESUMES,
                    "next_resume_timestamp": None,
                    "resume_delay_seconds": None,
                    "max_attempts": constants.RESEARCH_CODER_MAX_ATTEMPTS,
                    "session_id": None,
                    "active_pid": None,
                    "started_timestamp": None,
                    "completed_timestamp": None,
                    "exit_code": None,
                    "failure_category": None,
                    "resume_token": None,
                    "last_error": None,
                },
                "audit": {
                    "status": "not_started",
                    "active": False,
                    "spawn_count": 0,
                    "max_spawns": constants.RESEARCH_CODER_MAX_SPAWNS,
                    "resume_count": 0,
                    "max_resumes": constants.RESEARCH_CODER_MAX_RESUMES,
                    "next_resume_timestamp": None,
                    "resume_delay_seconds": None,
                    "failure_category": None,
                    "resume_token": None,
                    "session_id": None,
                    "active_pid": None,
                    "repair_round": 0,
                    "last_verdict": None,
                    "last_report_path": None,
                    "last_error": None,
                },
                "attempts": [],
                "current_attempt": None,
                "final": {},
                "notification": {
                    "sent": False,
                    "sent_timestamp": None,
                    "message": None,
                },
            }
            if extra_metadata:
                eval_data.update(copy.deepcopy(extra_metadata))
            self._save_evaluation(eval_id, eval_data)
            return eval_data

    def create_instruction_evaluation_atomic(
        self,
        *,
        author: str,
        title: str,
        content: str,
        tick: int,
        tags: Optional[List[str]] = None,
        abstract: str = "",
        lineage: Optional[str] = None,
        max_active_for_author: int = 1,
        backend: Optional[str] = None,
        model_name: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Create one instruction evaluation while holding the submission lock."""

        with self._submission_lock:
            file_io_utils.ensure_dir_exists(self.evaluations_dir)
            submission_lock = filelock.FileLock(self._submission_file_lock_path, timeout=60)
            try:
                submission_lock.acquire()
            except filelock.Timeout:
                print(
                    "EvaluationManager: failed to acquire submission lock "
                    f"lock_path={self._submission_file_lock_path!r} timeout=60s"
                )
                raise
            try:
                self._ensure_preloaded()
                self._refresh_from_disk_if_needed()
                active_ids = self.get_active_eval_ids_for_author(author)
                if max_active_for_author >= 0 and len(active_ids) >= max_active_for_author:
                    return None, active_ids

                eval_id = self.get_next_evaluation_id()
                eval_data = self.create_evaluation(
                    eval_id=eval_id,
                    author=author,
                    title=title,
                    content=content,
                    tick=tick,
                    tags=tags,
                    abstract=abstract,
                    lineage=lineage,
                    backend=backend,
                    model_name=model_name,
                    extra_metadata=extra_metadata,
                )
                return eval_data, []
            finally:
                submission_lock.release()

    def mark_parallel_submission_committed(self, eval_id: str) -> bool:
        def mutator(record: Dict[str, Any]):
            if str(record.get("parallel_commit_status") or "").strip().lower() == "provisional":
                record["parallel_commit_status"] = "committed"
                parallel_meta = record.setdefault("parallel_tick", {})
                parallel_meta["committed_timestamp"] = time.time()

        return self.update_evaluation(eval_id, mutator) is not None

    def delete_evaluation(self, eval_id: str) -> bool:
        eval_id = str(eval_id)
        old_top_submission = evaluation_index.get_top_submission(self.evaluations_dir)
        previous_dir_mtime_ns = self._get_evaluations_dir_mtime_ns()
        with self._file_lock(eval_id):
            deleted = file_io_utils.delete_file(_get_yaml_eval_path(eval_id, self.evaluations_dir))
        if deleted:
            self._bump_evaluations_dir_mtime(previous_dir_mtime_ns)
            try:
                evaluation_index.delete_evaluation(eval_id, self.evaluations_dir)
            except Exception as e:
                print(f"ResearchIndex: delete failed eval_id={eval_id!r}: {e}")
                raise
            self._sync_deleted_eval_in_memory(eval_id, emit_top_submission_callback=False)
            new_top_submission = evaluation_index.get_top_submission(self.evaluations_dir)
            if (
                self._top_submission_callback
                and not self._top_submissions_equal(old_top_submission, new_top_submission)
            ):
                self._top_submission_callback(copy.deepcopy(new_top_submission) if new_top_submission else None)
        return deleted

    def update_evaluation(self, eval_id: str, mutator: Callable[[Dict[str, Any]], None]) -> Optional[Dict[str, Any]]:
        eval_id = str(eval_id)
        with self._file_lock(eval_id):
            eval_data = self._load_evaluation_yaml(eval_id)
            if not eval_data:
                return None
            mutator(eval_data)
            self._save_evaluation(eval_id, eval_data)
            return eval_data

    def set_coder_spawned(
        self,
        eval_id: str,
        session_id: str,
        pid: Optional[int] = None,
        backend: Optional[str] = None,
        model_name: Optional[str] = None,
        preserve_started_timestamp: bool = False,
        substate: str = "coder_running",
        increment_spawn_count: bool = True,
        increment_resume_count: bool = False,
    ) -> Optional[Dict[str, Any]]:
        def mutator(eval_data: Dict[str, Any]):
            coder = eval_data.setdefault("coder", {})
            coder["backend"] = backend or coder.get("backend") or constants.RESEARCH_CODER_BACKEND
            coder["model_name"] = model_name if model_name is not None else coder.get("model_name")
            coder["active"] = True
            coder["status"] = substate
            if increment_spawn_count:
                coder["spawn_count"] = int(coder.get("spawn_count", 0)) + 1
            if increment_resume_count:
                coder["resume_count"] = int(coder.get("resume_count", 0)) + 1
            coder["session_id"] = session_id
            coder["active_pid"] = pid
            if not (preserve_started_timestamp and coder.get("started_timestamp")):
                coder["started_timestamp"] = time.time()
            coder["completed_timestamp"] = None
            coder["exit_code"] = None
            coder["failure_category"] = None
            if substate != "resuming":
                coder["resume_token"] = None
            coder["next_resume_timestamp"] = None
            coder["resume_delay_seconds"] = None
            coder["last_error"] = None
            eval_data["status"] = "running"

        return self.update_evaluation(eval_id, mutator)

    def claim_coder_launch(
        self,
        eval_id: str,
        session_id: str,
        *,
        backend: Optional[str] = None,
        model_name: Optional[str] = None,
        preserve_started_timestamp: bool = False,
        substate: str = "coder_running",
        increment_spawn_count: bool = True,
        increment_resume_count: bool = False,
    ) -> Optional[Dict[str, Any]]:
        eval_id = str(eval_id)
        with self._file_lock(eval_id):
            eval_data = self._load_evaluation_yaml(eval_id)
            if not eval_data:
                return None

            if eval_data.get("final"):
                return None

            top_level_status = _normalize_evaluation_status(eval_data.get("status")) or "queued"
            coder = eval_data.setdefault("coder", {})
            coder_active = bool(coder.get("active"))
            coder_status = str(coder.get("status", "")).strip().lower()

            if substate == "resuming":
                if top_level_status != "running":
                    return None
                if coder_status not in {"pending_resume", "resuming"}:
                    return None
                if not str(coder.get("resume_token", "")).strip():
                    return None
            else:
                if top_level_status != "queued":
                    return None

            if coder_active and not (substate == "resuming" and coder_status == "pending_resume"):
                return None

            coder["backend"] = backend or coder.get("backend") or constants.RESEARCH_CODER_BACKEND
            coder["model_name"] = model_name if model_name is not None else coder.get("model_name")
            coder["active"] = True
            coder["status"] = substate
            if increment_spawn_count:
                coder["spawn_count"] = int(coder.get("spawn_count", 0)) + 1
            if increment_resume_count:
                coder["resume_count"] = int(coder.get("resume_count", 0)) + 1
            coder["session_id"] = session_id
            coder["active_pid"] = None
            if not (preserve_started_timestamp and coder.get("started_timestamp")):
                coder["started_timestamp"] = time.time()
            coder["completed_timestamp"] = None
            coder["exit_code"] = None
            coder["failure_category"] = None
            if substate != "resuming":
                coder["resume_token"] = None
            coder["next_resume_timestamp"] = None
            coder["resume_delay_seconds"] = None
            coder["last_error"] = None
            eval_data["status"] = "running"

            self._save_evaluation(eval_id, eval_data)
            return copy.deepcopy(eval_data)

    def set_coder_exited(
        self,
        eval_id: str,
        exit_code: Optional[int] = None,
        error: Optional[str] = None,
        keep_running: bool = False,
        running_substate: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        def mutator(eval_data: Dict[str, Any]):
            coder = eval_data.setdefault("coder", {})
            coder["active_pid"] = None
            coder["completed_timestamp"] = time.time()
            coder["exit_code"] = exit_code
            coder["last_error"] = error
            if keep_running:
                coder["active"] = False
                if running_substate:
                    coder["status"] = running_substate
                if _normalize_evaluation_status(eval_data.get("status")) not in TERMINAL_EVALUATION_STATUSES:
                    eval_data["status"] = "running"
                return

            coder["active"] = False
            coder["failure_category"] = None
            coder["status"] = "queued"
            if _normalize_evaluation_status(eval_data.get("status")) not in TERMINAL_EVALUATION_STATUSES:
                eval_data["status"] = "queued"

        return self.update_evaluation(eval_id, mutator)

    def register_attempt(
        self,
        eval_id: str,
        submission_or_path: str,
        submission_path: Optional[str] = None,
        *,
        cpu_only: bool = False,
    ) -> Optional[int]:
        attempt_number: Optional[int] = None
        if submission_path is None:
            submission_path = submission_or_path
            submission_text = None
        else:
            submission_text = submission_or_path

        def mutator(eval_data: Dict[str, Any]):
            nonlocal attempt_number
            attempts = eval_data.setdefault("attempts", [])
            artifacts = _ensure_default_artifacts(eval_data, str(eval_id))
            if submission_text is not None:
                file_io_utils.save_text(submission_text, _get_artifact_path(eval_data, "submission", self.evaluations_dir))
            attempt_number = len(attempts) + 1
            attempt_record = {
                "attempt": attempt_number,
                "status": "queued",
                "requested_timestamp": time.time(),
                "started_timestamp": None,
                "completed_timestamp": None,
                "submission_path": submission_path,
                "stdout_path": artifacts.get("stdout") or _default_artifact_rel_path(str(eval_id), "stdout"),
                "stderr_path": artifacts.get("stderr") or _default_artifact_rel_path(str(eval_id), "stderr"),
                "primary_score": constants.RESEARCH_SCORE_PENDING,
                constants.EVALUATION_DETAILS_KEY: "",
                "sort_key": None,
                "error": None,
            }
            if cpu_only:
                attempt_record["cpu_only"] = True
            attempts.append(attempt_record)
            eval_data["current_attempt"] = attempt_number
            eval_data["status"] = "running"
            if (
                "instruction" in eval_data
                and not eval_data.get("submission_mode") == "direct"
                and not eval_data.get("system_baseline")
            ):
                coder = eval_data.setdefault("coder", {})
                coder["status"] = "attempt_running"

        self.update_evaluation(eval_id, mutator)
        return attempt_number

    def mark_attempt_running(self, eval_id: str, attempt_number: int, start_tick: Optional[int] = None) -> Optional[Dict[str, Any]]:
        def mutator(eval_data: Dict[str, Any]):
            for attempt in eval_data.get("attempts", []):
                if int(attempt.get("attempt", -1)) == int(attempt_number):
                    attempt["status"] = "running"
                    attempt["started_timestamp"] = time.time()
                    if start_tick is not None:
                        attempt["start_tick"] = start_tick
                    break
            eval_data["status"] = "running"
            coder = eval_data.setdefault("coder", {})
            if "instruction" in eval_data and not eval_data.get("system_baseline"):
                coder["status"] = "attempt_running"

        return self.update_evaluation(eval_id, mutator)

    def complete_attempt(
        self,
        eval_id: str,
        attempt_number: int,
        success: bool,
        score: Any,
        stdout: str,
        stderr: str,
        details: Any = "",
        error: Optional[str] = None,
        sort_key: Optional[Any] = None,
        status: Optional[str] = None,
        progress_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        persisted_stdout = truncate_persisted_stdout(stdout)
        persisted_stderr = stderr or ""

        def mutator(eval_data: Dict[str, Any]):
            artifacts = _ensure_default_artifacts(eval_data, str(eval_id))
            for attempt in eval_data.get("attempts", []):
                if int(attempt.get("attempt", -1)) != int(attempt_number):
                    continue
                file_io_utils.save_text(persisted_stdout, _get_artifact_path(eval_data, "stdout", self.evaluations_dir))
                file_io_utils.save_text(persisted_stderr, _get_artifact_path(eval_data, "stderr", self.evaluations_dir))
                attempt["completed_timestamp"] = time.time()
                attempt["status"] = status or ("completed" if success else "failed")
                attempt["primary_score"] = score
                attempt["stdout_path"] = artifacts.get("stdout") or _default_artifact_rel_path(str(eval_id), "stdout")
                attempt["stderr_path"] = artifacts.get("stderr") or _default_artifact_rel_path(str(eval_id), "stderr")
                attempt[constants.EVALUATION_DETAILS_KEY] = details
                attempt["error"] = error
                attempt["sort_key"] = sort_key
                attempt["progress_records"] = self._to_yaml_safe(progress_records or [])
                break
            current_status = _normalize_evaluation_status(eval_data.get("status"))
            if eval_data.get("final") or current_status in TERMINAL_EVALUATION_STATUSES:
                return
            coder = eval_data.get("coder", {}) or {}
            coder_active = bool(coder.get("active"))
            coder_substate = str(coder.get("status") or "").strip().lower()
            if coder_active:
                eval_data["status"] = "running"
                if coder_substate == "attempt_running":
                    coder["status"] = "coder_running"
            elif current_status == "running" and coder_substate == "attempt_running":
                # The coder already exited after writing the report, but we keep the
                # eval in top-level running until finalization is truly ready.
                eval_data["status"] = "running"
            else:
                eval_data["status"] = "queued"

        return self.update_evaluation(eval_id, mutator)

    def finalize_evaluation(
        self,
        eval_id: str,
        coder_report: str,
        final_status: Optional[str] = None,
        notification_report_mode: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        eval_id = str(eval_id)
        final_record: Optional[Dict[str, Any]] = None
        notification_message: Optional[str] = None
        author: Optional[str] = None
        cleanup_eval_data: Optional[Dict[str, Any]] = None

        with self._file_lock(eval_id):
            eval_data = self._load_evaluation_yaml(eval_id)
            if not eval_data:
                return None
            if str(eval_data.get("parallel_commit_status") or "").strip().lower() == "rolled_back":
                return eval_data

            existing_final = eval_data.get("final") or {}
            existing_final_status = _normalize_evaluation_status(existing_final.get("status"))
            existing_score = existing_final.get("primary_score", constants.RESEARCH_SCORE_NA)
            latest_attempt = _get_final_attempt(eval_data) or {}
            derived_status = _normalize_evaluation_status(final_status)
            if derived_status is None:
                latest_attempt_status = str(latest_attempt.get("status", "")).strip().lower()
                score = latest_attempt.get("primary_score", constants.RESEARCH_SCORE_NA)
                if latest_attempt_status in {"completed", "success"}:
                    derived_status = "completed"
                elif score not in (constants.RESEARCH_SCORE_PENDING, constants.RESEARCH_SCORE_NA, None):
                    derived_status = "completed"
                elif latest_attempt:
                    derived_status = "failed"
                else:
                    derived_status = "blocked"

            if (
                existing_final
                and existing_final_status in TERMINAL_EVALUATION_STATUSES
                and (
                    existing_final_status == "completed"
                    or existing_score not in (constants.RESEARCH_SCORE_PENDING, constants.RESEARCH_SCORE_NA, None)
                )
                and derived_status in {"failed", "blocked", "partial"}
            ):
                coder = eval_data.setdefault("coder", {})
                coder["active"] = False
                coder["status"] = existing_final_status
                coder["active_pid"] = None
                coder["completed_timestamp"] = time.time()
                self._save_evaluation(eval_id, eval_data)
                return eval_data

            final_record = {
                "status": derived_status,
                "attempt": latest_attempt.get("attempt"),
                "primary_score": latest_attempt.get("primary_score", constants.RESEARCH_SCORE_NA),
                constants.EVALUATION_DETAILS_KEY: latest_attempt.get(constants.EVALUATION_DETAILS_KEY, ""),
                "sort_key": self._clone_sort_key_for_persistence(latest_attempt.get("sort_key")),
                "progress_records": self._to_yaml_safe(latest_attempt.get("progress_records") or []),
                "artifacts": _default_artifact_map(eval_id),
                "error": latest_attempt.get("error"),
            }
            eval_data["final"] = final_record
            _ensure_default_artifacts(eval_data, eval_id)
            eval_data["status"] = derived_status
            coder = eval_data.setdefault("coder", {})
            coder["active"] = False
            coder["status"] = derived_status
            coder["active_pid"] = None
            coder["completed_timestamp"] = time.time()
            file_io_utils.save_text(coder_report or "", _get_artifact_path(eval_data, "report", self.evaluations_dir))
            notification = eval_data.setdefault("notification", {})
            if notification_report_mode is not None:
                notification["report_mode"] = notification_report_mode
            self._save_evaluation(eval_id, eval_data)
            cleanup_eval_data = copy.deepcopy(eval_data)

            author = eval_data.get("author")
            if self._notification_callback and not notification.get("sent"):
                notification_message = self._generate_notification_message(eval_data)

        if cleanup_eval_data:
            _cleanup_eval_tmp_storage(cleanup_eval_data, self.evaluations_dir)

        if notification_message and author:
            self._notification_callback(author, notification_message)
            with self._file_lock(eval_id):
                eval_data = self._load_evaluation_yaml(eval_id)
                if not eval_data:
                    return None
                notification = eval_data.setdefault("notification", {})
                if notification.get("sent"):
                    self._update_top_submission_if_needed(eval_data)
                    return eval_data
                notification.update(
                    {
                        "sent": True,
                        "sent_timestamp": time.time(),
                        "message": notification_message,
                    }
                )
                self._save_evaluation(eval_id, eval_data)
                self._update_top_submission_if_needed(eval_data)
                return eval_data

        if final_record:
            with self._file_lock(eval_id):
                eval_data = self._load_evaluation_yaml(eval_id)
                if eval_data:
                    self._update_top_submission_if_needed(eval_data)
                    return eval_data
        return self._load_evaluation_yaml(eval_id)

    def send_notification_if_pending(self, eval_id: str) -> bool:
        eval_id = str(eval_id)
        if not self._notification_callback:
            return False

        author: Optional[str] = None
        notification_message: Optional[str] = None
        should_send = False

        try:
            with self._file_lock(eval_id, timeout=0.0):
                eval_data = self._load_evaluation_yaml(eval_id)
                if not eval_data:
                    return False
                if str(eval_data.get("parallel_commit_status") or "").strip().lower() == "rolled_back":
                    return False
                if _normalize_evaluation_status(eval_data.get("status")) not in TERMINAL_EVALUATION_STATUSES:
                    return False
                if not (eval_data.get("final") or {}):
                    return False
                notification = eval_data.setdefault("notification", {})
                if notification.get("sent"):
                    return False

                author = eval_data.get("author")
                notification_message = self._generate_notification_message(eval_data)
                should_send = bool(author and notification_message)
        except filelock.Timeout:
            print(f"EvaluationManager: notification send skipped because evaluation lock is busy eval_id={eval_id!r}")
            return False

        if not should_send or not author or not notification_message:
            return False

        self._notification_callback(author, notification_message)
        try:
            with self._file_lock(eval_id, timeout=0.0):
                eval_data = self._load_evaluation_yaml(eval_id)
                if not eval_data:
                    return False
                notification = eval_data.setdefault("notification", {})
                if notification.get("sent"):
                    return False
                notification.update(
                    {
                        "sent": True,
                        "sent_timestamp": time.time(),
                        "message": notification_message,
                    }
                )
                self._save_evaluation(eval_id, eval_data)
                self._update_top_submission_if_needed(eval_data)
        except filelock.Timeout:
            print(f"EvaluationManager: notification mark-sent skipped because evaluation lock is busy eval_id={eval_id!r}")
            return False
        return True

    def get_notification_message(self, eval_id: str) -> Optional[str]:
        eval_data = self.get_evaluation(eval_id)
        if not eval_data:
            return None
        if _normalize_evaluation_status(eval_data.get("status")) not in TERMINAL_EVALUATION_STATUSES:
            return None
        if not (eval_data.get("final") or {}):
            return None
        return self._generate_notification_message(eval_data)

    def _generate_notification_message(
        self,
        eval_data: Dict[str, Any],
        report_override: Optional[str] = None,
    ) -> str:
        final = eval_data.get("final") or {}
        score = final.get("primary_score", constants.RESEARCH_SCORE_NA)
        details = final.get(constants.EVALUATION_DETAILS_KEY, "")
        details_message, secondary_metrics_string = _format_secondary_metrics_for_display(details)
        report = report_override or _load_eval_final_artifact(eval_data, "report", self.evaluations_dir)
        if (eval_data.get("notification") or {}).get("report_mode") == "latest_audit":
            audit_heading = re.search(
                r"\n## Independent Audit(?: Failure)?(?: \([^\n)]*\))?\s*\n",
                report,
                flags=re.IGNORECASE,
            )
            base_report = report[:audit_heading.start()].rstrip() if audit_heading else report.rstrip()
            audit_path = str((eval_data.get("audit") or {}).get("last_report_path") or "").strip()
            if audit_path:
                if not os.path.isabs(audit_path):
                    audit_path = os.path.join(os.path.dirname(self.evaluations_dir), audit_path)
                latest_audit = file_io_utils.load_text(audit_path) if file_io_utils.file_exists(audit_path) else ""
                if latest_audit.strip():
                    report = (
                        f"{base_report}\n\n## Final Independent Audit\n\n"
                        f"{latest_audit.strip()}\n"
                    )
        report = report.strip() or "_No coder report available._"

        message = f"Your research submission '{eval_data.get('title', 'Untitled')}' (ID: {eval_data.get('id')}) has completed.\n\n"

        if not constants.RESEARCH_NO_SCORE:
            message += f"**Score:** {score}"
            if secondary_metrics_string:
                message += secondary_metrics_string
            message += f"\n**Evaluation Details:** {details_message}\n\n"

        message += f"**Coder Report:**\n{report}\n\n"
        message += f"**Stdout/Stderr:** {STDOUT_STDERR_HIDDEN_NOTICE}"
        return message

    def _clone_sort_key_for_persistence(self, sort_key: Any) -> Any:
        if isinstance(sort_key, tuple):
            return tuple(self._clone_sort_key_for_persistence(item) for item in sort_key)
        if isinstance(sort_key, list):
            return [self._clone_sort_key_for_persistence(item) for item in sort_key]
        return copy.deepcopy(sort_key)

    def _update_top_submission_if_needed(self, eval_data: Dict[str, Any]):
        # Evaluation saves update the in-memory cache and emit the callback when
        # the top submission changes. This compatibility hook refreshes external
        # file changes for older call sites after notification delivery.
        self._refresh_from_disk_if_needed()

    def get_top_submission(self) -> Optional[Dict[str, Any]]:
        top_submission = evaluation_index.get_top_submission(self.evaluations_dir)
        with self._lock:
            self.top_submission = copy.deepcopy(top_submission) if top_submission else None
        return copy.deepcopy(top_submission) if top_submission else None

    def get_breakthrough_events(self) -> List[Dict[str, Any]]:
        from station.eval_research import breakthroughs

        return [event.to_dict() for event in breakthroughs.get_breakthrough_events(self.evaluations_dir)]

    def get_latest_breakthrough_summary(self) -> Dict[str, Any]:
        from station.eval_research import breakthroughs

        return breakthroughs.get_latest_breakthrough_summary(self.evaluations_dir)

    def get_active_evaluations(self) -> List[Dict[str, Any]]:
        return evaluation_index.get_active_evaluations(self.evaluations_dir)

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        stats = evaluation_index.get_evaluation_statistics(self.evaluations_dir)
        with self._lock:
            top_submission = stats.get("top_submission")
            self.top_submission = copy.deepcopy(top_submission) if top_submission else None
        return stats

    def get_active_eval_ids_for_author(self, author: str) -> List[str]:
        return evaluation_index.get_active_eval_ids_for_author(self.evaluations_dir, author)

    def get_queued_instruction_eval_ids(self) -> List[str]:
        return evaluation_index.get_queued_instruction_eval_ids(self.evaluations_dir)

    def get_running_instruction_eval_ids(self) -> List[str]:
        return evaluation_index.get_running_instruction_eval_ids(self.evaluations_dir)

    def get_resuming_instruction_eval_ids(self) -> List[str]:
        return evaluation_index.get_resuming_instruction_eval_ids(self.evaluations_dir)

    def get_unfinished_instruction_eval_ids(self) -> List[str]:
        return evaluation_index.get_unfinished_instruction_eval_ids(self.evaluations_dir)

    def get_retryable_blocked_instruction_eval_ids(self) -> List[str]:
        return evaluation_index.get_retryable_blocked_instruction_eval_ids(self.evaluations_dir)

    def get_unfinished_requeue_candidate_instruction_eval_ids(
        self,
        *,
        include_active_statuses: bool = True,
    ) -> List[str]:
        return evaluation_index.get_unfinished_requeue_candidate_instruction_eval_ids(
            self.evaluations_dir,
            include_active_statuses=include_active_statuses,
        )

    def get_no_report_terminal_requeueable_eval_ids(self) -> List[str]:
        return evaluation_index.get_no_report_terminal_requeueable_eval_ids(self.evaluations_dir)

    def get_running_evaluation_count(self) -> int:
        return self.get_active_coder_count()

    def get_active_coder_count(self) -> int:
        return evaluation_index.get_active_coder_count(self.evaluations_dir)

    def get_pending_notification_eval_ids(self) -> List[str]:
        return evaluation_index.get_pending_notification_eval_ids(self.evaluations_dir)

    def get_recent_attempt_summaries(self, author: str, limit: int = 5, exclude_eval_id: Optional[str] = None) -> List[Dict[str, Any]]:
        author_lower = str(author or "").lower()
        return evaluation_index.get_recent_attempt_summaries(
            self.evaluations_dir,
            author=author_lower,
            limit=limit,
            exclude_eval_id=exclude_eval_id,
        )

    def get_recent_lineage_attempt_summaries(
        self,
        lineage: str,
        limit: int = 5,
        exclude_eval_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        lineage_lower = str(lineage or "").lower()
        return evaluation_index.get_recent_attempt_summaries(
            self.evaluations_dir,
            lineage=lineage_lower,
            limit=limit,
            exclude_eval_id=exclude_eval_id,
        )

    def should_wait_at_tick(self, current_tick: int) -> bool:
        return evaluation_index.should_wait_at_tick(
            self.evaluations_dir,
            current_tick,
            constants.RESEARCH_EVAL_MAX_TICK,
        )


__all__ = [
    "EvaluationManager",
    "get_evaluation_display_info",
    "get_evaluation_review_info",
    "get_evaluation_code_info",
    "format_score_for_display",
    "format_tags_for_display",
    "extract_secondary_metrics_for_display_info",
    "build_evaluation_previews",
]
