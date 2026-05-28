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
Background evaluator for Theory Room submissions.

Workflow:
- TheoryRoom writes submissions to a pending yamll file.
- AutoTheoryEvaluator picks them up, runs Lean in a thread pool, and persists results.
"""

import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from station import constants
from station.eval_theory.lean_runner import LeanRunResult, build_theory_project, run_lean_submission
from station.eval_theory.storage import TheoryStorageManager
from station.eval_theory.debugger import TheoryDebugger


class AutoTheoryEvaluator:
    """Simple background worker that processes pending Theory Room submissions."""

    def __init__(
        self,
        station_instance,
        storage_manager: Optional[TheoryStorageManager] = None,
        enabled: Optional[bool] = None,
    ):
        self.station = station_instance
        self.enabled = constants.AUTO_EVAL_THEORY if enabled is None else enabled
        self.check_interval = constants.THEORY_EVAL_CHECK_INTERVAL
        self.max_workers = constants.THEORY_EVAL_MAX_PARALLEL_WORKERS

        base_path = storage_manager.base_path if storage_manager else None
        if not base_path:
            base_path = os.path.join(
                constants.BASE_STATION_DATA_PATH, constants.ROOMS_DIR_NAME, constants.SHORT_ROOM_NAME_THEORY
            )
        self.storage = storage_manager or TheoryStorageManager(base_path)
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        self.thread: Optional[threading.Thread] = None
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.is_running = False

        self._running_futures: Dict[str, Dict[str, Any]] = {}
        self._running_lock = threading.Lock()
        self._tick_limit_logged: set[str] = set()
        self._build_lock = threading.Lock()

    def _gather_theory_contents(self, new_entry: Optional[Dict[str, Any]] = None) -> List[str]:
        items: List[Dict[str, Any]] = []
        for kind in ("lemma", "theory"):
            for it in self.storage.load_items(kind):
                items.append(
                    {
                        "content": it.get("content", ""),
                        "submitted_tick": it.get("submitted_tick") or 0,
                        "kind": kind,
                        "id": it.get("id") or 0,
                        "is_new": False,
                    }
                )
        if new_entry:
            items.append(
                {
                    "content": new_entry.get("content", ""),
                    "submitted_tick": new_entry.get("submitted_tick") or 0,
                    "kind": new_entry.get("kind", ""),
                    "id": new_entry.get("id") or 0,
                    "is_new": True,
                }
            )

        items.sort(key=lambda it: (it["submitted_tick"], 1 if it["is_new"] else 0, it["kind"], it["id"]))
        return [it["content"] for it in items if it.get("content")]

    # ---------- lifecycle ----------
    def start_evaluation_loop(self) -> bool:
        if not self.enabled:
            print("AutoTheoryEvaluator: disabled by configuration.")
            return False
        if self.is_running:
            print("AutoTheoryEvaluator: already running.")
            return True

        self.is_running = True
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.thread.start()
        print(f"AutoTheoryEvaluator: started with max_workers={self.max_workers}")
        return True

    def stop_evaluation_loop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False, cancel_futures=True)
        self.thread = None
        self.thread_pool = None
        print("AutoTheoryEvaluator: stopped.")

    # ---------- public helpers ----------
    def process_pending_queue(self, blocking: bool = False) -> int:
        """Process pending evaluations. If blocking, continue until queue drains."""
        if not self.enabled:
            return 0
        if not self.thread_pool:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        total_scheduled = 0
        while True:
            scheduled = self._schedule_pending_once()
            total_scheduled += scheduled
            if not blocking or scheduled == 0:
                break
            self._wait_for_running()
        return total_scheduled

    def _schedule_pending_once(self) -> int:
        # Clean up completed futures
        with self._running_lock:
            done_ids = [qid for qid, info in self._running_futures.items() if info["future"].done()]
            for qid in done_ids:
                self._running_futures.pop(qid, None)

        pending = self.storage.load_pending()
        pending_changed = False
        for entry in pending:
            if not entry.get("queue_id"):
                entry["queue_id"] = str(uuid.uuid4())
                pending_changed = True
        if pending_changed:
            self.storage.rewrite_pending(pending)

        # Respect author ordering: one active task per author at a time.
        with self._running_lock:
            blocked_authors = {info["author"] for info in self._running_futures.values() if not info["future"].done()}

        scheduled = 0
        pending_sorted = sorted(
            pending,
            key=lambda e: (e.get("created_timestamp") or 0, str(e.get("queue_id") or "")),
        )
        for entry in pending_sorted:
            queue_id = str(entry.get("queue_id"))
            author = entry.get("author")
            with self._running_lock:
                if queue_id in self._running_futures:
                    continue
                active = len([1 for info in self._running_futures.values() if not info["future"].done()])
            if active >= self.max_workers:
                break
            if author in blocked_authors:
                continue
            self._submit_entry(entry)
            blocked_authors.add(author)
            scheduled += 1
        return scheduled

    def has_pending_or_running(self) -> bool:
        if not self.enabled:
            return False
        pending = self.storage.load_pending()
        if pending:
            return True
        with self._running_lock:
            return any(not info["future"].done() for info in self._running_futures.values())

    def should_wait_at_tick(self, current_tick: int) -> bool:
        max_allowed_ticks = constants.THEORY_EVAL_MAX_TICK

        def elapsed_ticks_for(start_tick: Optional[int]) -> Optional[int]:
            if start_tick is None:
                return None
            return current_tick - start_tick + 1

        with self._running_lock:
            for info in self._running_futures.values():
                queue_id = str(info.get("queue_id") or "")
                start_tick = info.get("start_tick")
                if start_tick is None:
                    start_tick = info.get("submitted_tick")
                elapsed_ticks = elapsed_ticks_for(start_tick)
                if elapsed_ticks is None:
                    continue
                if elapsed_ticks >= max_allowed_ticks:
                    if queue_id and queue_id not in self._tick_limit_logged:
                        print(
                            "AutoTheoryEvaluator: running evaluation reached tick limit "
                            f"(queue_id={queue_id}, start_tick={start_tick}, current_tick={current_tick}, "
                            f"elapsed_ticks={elapsed_ticks}, max_allowed_ticks={max_allowed_ticks})"
                        )
                        self._tick_limit_logged.add(queue_id)
                    return True

        try:
            pending = self.storage.load_pending()
        except Exception as e:
            print(f"AutoTheoryEvaluator: error loading pending during tick check: {e}")
            return False
        for entry in pending:
            queue_id = str(entry.get("queue_id") or "")
            submitted_tick = entry.get("submitted_tick")
            elapsed_ticks = elapsed_ticks_for(submitted_tick)
            if elapsed_ticks is None:
                continue
            if elapsed_ticks >= max_allowed_ticks:
                if queue_id and queue_id not in self._tick_limit_logged:
                    print(
                        "AutoTheoryEvaluator: pending evaluation reached tick limit "
                        f"(queue_id={queue_id}, submitted_tick={submitted_tick}, current_tick={current_tick}, "
                        f"elapsed_ticks={elapsed_ticks}, max_allowed_ticks={max_allowed_ticks})"
                    )
                    self._tick_limit_logged.add(queue_id)
                return True
        return False

    # ---------- internal ----------
    def _evaluation_loop(self):
        while self.is_running:
            try:
                self.process_pending_queue(blocking=False)
            except Exception as e:
                print(f"AutoTheoryEvaluator: error in evaluation loop: {e}")
            time.sleep(self.check_interval)

    def _submit_entry(self, entry: Dict[str, Any]) -> None:
        queue_id = str(entry.get("queue_id") or str(uuid.uuid4()))
        entry["queue_id"] = queue_id
        if not self.thread_pool:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        # Match research tick accounting: running evals are measured from the submission tick
        # (the tick carried by the queued item), not from when a background worker picks it up.
        start_tick = entry.get("submitted_tick")
        future = self.thread_pool.submit(self._evaluate_entry, entry)
        with self._running_lock:
            self._running_futures[queue_id] = {
                "future": future,
                "queue_id": queue_id,
                "submitted_tick": entry.get("submitted_tick"),
                "start_tick": start_tick,
                "author": entry.get("author"),
            }
        future.add_done_callback(lambda fut, qid=queue_id, ent=entry: self._handle_completion(qid, ent, fut))

    def _handle_completion(self, queue_id: str, entry: Dict[str, Any], future: Future) -> None:
        try:
            future.result()
        except Exception as e:
            print(f"AutoTheoryEvaluator: exception processing queue_id={queue_id}: {e}")
        finally:
            try:
                self.storage.remove_pending(queue_id)
            except Exception as e:
                print(f"AutoTheoryEvaluator: failed to prune pending for {queue_id}: {e}")
            with self._running_lock:
                self._running_futures.pop(queue_id, None)

    def _evaluate_entry(self, entry: Dict[str, Any]) -> LeanRunResult:
        kind = entry.get("kind")
        payload = entry.get("payload", {})
        author = entry.get("author", "Unknown")
        submitted_tick = entry.get("submitted_tick")

        if kind not in {"lemma", "theory", "sandbox"} or not payload:
            msg = f"Skipping invalid theory submission (kind={kind})."
            print(f"AutoTheoryEvaluator: {msg}")
            self._notify(author, msg)
            return LeanRunResult(False, msg)

        allow_sorry = entry.get("allow_sorry", False) or kind == "sandbox"
        try:
            result = run_lean_submission(
                payload.get("content", ""),
                formal_statement=payload.get("formal_statement"),
                formal_definitions=payload.get("formal_definitions", ""),
                allow_sorry=allow_sorry,
            )
        except Exception as e:
            print(f"AutoTheoryEvaluator: Lean execution crashed for queue_id={entry.get('queue_id')}: {e}")
            result = LeanRunResult(False, f"Internal evaluator error: {e}")

        if kind == "sandbox":
            status_line = "completed" if result.success else "failed"
            completion_line = self._format_submission_finished_line(entry)
            log_msg = (
                f"{completion_line}\n\n"
                f"Your sandbox submission at the Theory Room has {status_line}:\n```\n{result.logs}\n```"
            )
            self._notify(author, log_msg)
            return result

        if result.success:
            build_result = None
            if kind in {"lemma", "theory"}:
                new_entry = {
                    "content": payload.get("content", ""),
                    "submitted_tick": submitted_tick,
                    "kind": kind,
                    "id": 0,
                }
                with self._build_lock:
                    contents = self._gather_theory_contents(new_entry)
                    build_result = build_theory_project(self.repo_root, contents)
            if build_result and not build_result.success:
                prefix = f"Your {kind} cannot be verified: {payload.get('formal_statement')}"
                failure_details = (
                    f"{prefix}\n"
                    f"Lake build failed after verification:\n```\n{build_result.logs}\n```"
                )
                if self._should_run_debugger(entry):
                    self._maybe_run_debugger(entry, result, failure_details)
                else:
                    completion_line = self._format_submission_finished_line(entry)
                    log_msg = f"{completion_line}\n\n{failure_details}"
                    self._notify(author, log_msg)
                return LeanRunResult(False, build_result.logs)

            if build_result and build_result.logs:
                combined_logs = f"{result.logs}\n\n{build_result.logs}"
            else:
                combined_logs = result.logs

            item_fields = {
                "title": payload.get("title"),
                "formal_statement": payload.get("formal_statement"),
                "formal_definitions": payload.get("formal_definitions", ""),
                "tags": payload.get("tags", []),
                "statement": payload.get("statement"),
                "content": payload.get("content"),
                "author": author,
                "submitted_tick": submitted_tick,
                "logs": combined_logs,
                "status": "Verified",
            }
            item = self.storage.add_verified_item(kind, item_fields)
            new_id = item["id"]
            self.storage.append_env_code("\n" + (payload.get("content") or "") + "\n")
            prefix = f"Your {kind} is verified successfully with {kind.capitalize()} ID {new_id}: {payload.get('formal_statement')}"
            completion_line = self._format_submission_finished_line(entry)
            actions_msg = (
                f"{completion_line}\n\n{prefix}\n{kind.capitalize()} submission logs:\n```\n{combined_logs}\n```"
            )
            self._notify(author, actions_msg)
        else:
            prefix = f"Your {kind} cannot be verified: {payload.get('formal_statement')}"
            failure_details = f"{prefix}\n{kind.capitalize()} submission logs:\n```\n{result.logs}\n```"
            if self._should_run_debugger(entry):
                self._maybe_run_debugger(entry, result, failure_details)
            else:
                completion_line = self._format_submission_finished_line(entry)
                log_msg = f"{completion_line}\n\n{failure_details}"
                self._notify(author, log_msg)
        return result

    def _should_run_debugger(self, entry: Dict[str, Any]) -> bool:
        if entry.get("allow_sorry") or entry.get("from_debugger"):
            return False
        if not getattr(constants, "THEORY_DEBUGGER_ENABLED", False):
            return False
        return True

    def _maybe_run_debugger(self, entry: Dict[str, Any], failed_result: LeanRunResult, failure_message: str) -> None:
        if not self._should_run_debugger(entry):
            return
        try:
            room_instance = self.station.rooms.get(constants.ROOM_THEORY) if getattr(self.station, "rooms", None) else None
            if not room_instance:
                return
            debugger = TheoryDebugger(
                station_instance=self.station,
                room_instance=room_instance,
                storage=self.storage,
                failed_entry=entry,
                failed_logs=failed_result.logs,
            )
            outcome = debugger.run()
            report = outcome.get("report") or "No report."
            success_flag = outcome.get("success", False)
            completion_line = self._format_submission_finished_line(entry)
            if success_flag:
                final_msg = (
                    f"{completion_line}\n\n"
                    "Your submission is successful after debugger's fix:\n\n"
                    f"---\n{report}\n---\n\n"
                    "Your original submission has the following failure:\n\n"
                    f"{failure_message}"
                )
                if hasattr(debugger, "successful_item") and debugger.successful_item:
                    code_block = debugger.successful_item.get("content", "")
                    if code_block:
                        final_msg += f"\n\nUpdated submission code:\n```\n{code_block}\n```"
            else:
                final_msg = (
                    f"{completion_line}\n\n"
                    "Your submission failed despite debugger's attempt:\n\n"
                    f"---\n{report}\n---\n\n"
                    "Your original submission has the following failure:\n\n"
                    f"{failure_message}"
                )
            self._notify(entry.get("author", "Unknown"), final_msg)
        except Exception as e:
            # The debugger crashed (e.g., Lean timeout or connector failure). Fall back to
            # notifying the author about the original failure so the submission does not
            # silently disappear.
            print(f"AutoTheoryEvaluator: debugger failed for queue_id={entry.get('queue_id')}: {e}")
            try:
                fallback_msg = (
                    f"{self._format_submission_finished_line(entry)}\n\n"
                    "Your submission failed verification. Original failure:\n\n"
                    f"{failure_message}"
                )
                self._notify(entry.get("author", "Unknown"), fallback_msg)
            except Exception as notify_err:
                print(f"AutoTheoryEvaluator: failed to send fallback notification: {notify_err}")

    def _notify(self, author: str, message: str) -> None:
        # Skip when station is not available
        if not getattr(self, "station", None):
            return
        try:
            added = False
            if hasattr(self.station, "agent_module") and hasattr(self.station.agent_module, "add_pending_notification_atomic"):
                added = self.station.agent_module.add_pending_notification_atomic(author, message)
            if not added and hasattr(self.station, "agent_module") and hasattr(self.station.agent_module, "add_pending_notification"):
                # Fallback to non-atomic path if present
                agent_data = self.station.agent_module.load_agent_data(author)
                if agent_data:
                    self.station.agent_module.add_pending_notification(agent_data, message)
                    self.station.agent_module.save_agent_data(author, agent_data)
        except Exception as e:
            print(f"AutoTheoryEvaluator: failed to send notification to {author}: {e}")

    def _format_submission_finished_line(self, entry: Dict[str, Any]) -> str:
        tick = entry.get("submitted_tick")
        tick_label = str(tick) if tick is not None else "unknown"
        submission_id = self._get_submission_short_id(entry)
        kind = entry.get("kind") or "submission"
        if kind == "sandbox":
            return (
                f"Your {kind} submission at the Theory Room in Tick "
                f"{tick_label} (Submission ID: {submission_id}) has been completed."
            )
        title = (entry.get("payload") or {}).get("title") or "unknown"
        return (
            f"Your {kind} submission at the Theory Room in Tick "
            f"{tick_label} (Title: {title}; Submission ID: {submission_id}) has been completed."
        )

    def _get_submission_short_id(self, entry: Dict[str, Any]) -> str:
        queue_id = entry.get("queue_id") or ""
        if not queue_id:
            return "unknown"
        return queue_id.split("-")[0]

    def _wait_for_running(self):
        """Wait until all running futures are complete (used in blocking test mode)."""
        while True:
            with self._running_lock:
                active = [fut for fut in self._running_futures.values() if not fut["future"].done()]
            if not active:
                return
            time.sleep(0.1)
