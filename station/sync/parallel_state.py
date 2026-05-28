"""Persistent state for parallel tick execution.

The parallel runner keeps this state deliberately small. It is not a full
resume journal; it is a recovery guard that lets startup detect an incomplete
parallel tick and roll back provisional fast-lane submissions before the tick
is retried.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional
import yaml

from station import constants, file_io_utils


def build_parallel_action_op_id(tick: int, agent_name: str, action_index: int) -> str:
    safe_agent = str(agent_name).replace(":", "_")
    return f"tick:{tick}:agent:{safe_agent}:action:{action_index}"


class ParallelTickState:
    """Small YAML-backed ledger for one in-flight parallel tick."""

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path or constants.BASE_STATION_DATA_PATH
        self.root_dir = os.path.join(self.base_path, constants.PARALLEL_TICK_STATE_DIR_NAME)
        self.run_dir = os.path.join(self.root_dir, "parallel_ticks")
        self.current_state_path = os.path.join(self.root_dir, "current_parallel_tick.yaml")

    @staticmethod
    def safe_agent_name(agent_name: str) -> str:
        text = str(agent_name or "unknown")
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "unknown"

    def ensure_layout(self) -> None:
        file_io_utils.ensure_dir_exists(self.root_dir)
        file_io_utils.ensure_dir_exists(self.run_dir)

    def load_current(self) -> Optional[Dict[str, Any]]:
        data = file_io_utils.load_yaml(self.current_state_path)
        return data if isinstance(data, dict) else None

    def save_current(self, state: Dict[str, Any]) -> None:
        self.ensure_layout()
        file_io_utils.save_yaml(state, self.current_state_path, sort_keys=False)

    def clear_current(self) -> None:
        file_io_utils.delete_file(self.current_state_path)

    def begin_tick(self, tick: int, turn_order: Iterable[str], eval_manager: Any = None) -> Dict[str, Any]:
        self.ensure_layout()
        run_id = uuid.uuid4().hex
        state = {
            "schema_version": 1,
            "status": "running",
            "tick": int(tick),
            "run_id": run_id,
            "started_timestamp": time.time(),
            "baseline_research_max_eval_id": self._max_numeric_eval_id(eval_manager),
            "turn_order": list(turn_order),
            "agents": {},
            "fast_lane_evaluations": [],
            "fast_lane_surveys": [],
        }
        self.save_current(state)
        return state

    def mark_agent_observation(self, state: Dict[str, Any], agent_name: str, observation_path: str) -> None:
        agent_state = state.setdefault("agents", {}).setdefault(agent_name, {})
        agent_state.update({
            "observation_path": observation_path,
            "observation_prepared": True,
            "observation_timestamp": time.time(),
        })
        self.save_current(state)

    def mark_agent_response(self, state: Dict[str, Any], agent_name: str, response_path: str) -> None:
        agent_state = state.setdefault("agents", {}).setdefault(agent_name, {})
        agent_state.update({
            "response_path": response_path,
            "response_received": True,
            "response_timestamp": time.time(),
        })
        self.save_current(state)

    def mark_agent_history_flushed(self, state: Dict[str, Any], agent_name: str) -> None:
        agent_state = state.setdefault("agents", {}).setdefault(agent_name, {})
        agent_state.update({
            "history_flushed": True,
            "history_flush_timestamp": time.time(),
        })
        self.save_current(state)

    def mark_agent_committed(self, state: Dict[str, Any], agent_name: str) -> None:
        agent_state = state.setdefault("agents", {}).setdefault(agent_name, {})
        agent_state.update({
            "actions_committed": True,
            "commit_timestamp": time.time(),
        })
        self.save_current(state)

    def mark_internal_action_started(self, state: Dict[str, Any], agent_name: str, handler_name: str) -> None:
        internal_state = state.setdefault("internal_actions", {}).setdefault(agent_name, {})
        internal_state.update({
            "status": "running",
            "handler": handler_name,
            "started_timestamp": time.time(),
        })
        self.save_current(state)

    def mark_internal_action_completed(self, state: Dict[str, Any], agent_name: str) -> None:
        internal_state = state.setdefault("internal_actions", {}).setdefault(agent_name, {})
        internal_state.update({
            "status": "completed",
            "completed_timestamp": time.time(),
        })
        self.save_current(state)

    def mark_internal_action_failed(self, state: Dict[str, Any], agent_name: str, error: str) -> None:
        internal_state = state.setdefault("internal_actions", {}).setdefault(agent_name, {})
        internal_state.update({
            "status": "failed",
            "error": str(error),
            "completed_timestamp": time.time(),
        })
        self.save_current(state)

    def record_fast_lane_evaluation(
        self,
        state: Dict[str, Any],
        *,
        eval_id: str,
        agent_name: str,
        op_id: str,
    ) -> None:
        entries = state.setdefault("fast_lane_evaluations", [])
        entry = {
            "eval_id": str(eval_id),
            "agent_name": agent_name,
            "op_id": op_id,
            "timestamp": time.time(),
        }
        if not any(
            str(existing.get("eval_id")) == str(eval_id)
            for existing in entries
            if isinstance(existing, dict)
        ):
            entries.append(entry)
        self.save_current(state)

    def record_fast_lane_survey(
        self,
        state: Dict[str, Any],
        *,
        survey_id: str,
        agent_name: str,
        op_id: str,
    ) -> None:
        entries = state.setdefault("fast_lane_surveys", [])
        entry = {
            "survey_id": str(survey_id),
            "agent_name": agent_name,
            "op_id": op_id,
            "timestamp": time.time(),
        }
        if not any(
            str(existing.get("survey_id")) == str(survey_id)
            for existing in entries
            if isinstance(existing, dict)
        ):
            entries.append(entry)
        self.save_current(state)

    def mark_completed(self, state: Dict[str, Any]) -> None:
        state["status"] = "completed"
        state["completed_timestamp"] = time.time()
        self.save_current(state)
        self.clear_current()

    def write_agent_text(
        self,
        state: Dict[str, Any],
        *,
        agent_name: str,
        filename: str,
        content: str,
    ) -> str:
        tick = state.get("tick", "unknown")
        run_id = state.get("run_id", "unknown")
        agent_dir = os.path.join(
            self.run_dir,
            f"tick_{tick}_{run_id}",
            self.safe_agent_name(agent_name),
        )
        file_io_utils.ensure_dir_exists(agent_dir)
        path = os.path.join(agent_dir, filename)
        file_io_utils.save_text(content or "", path)
        return path

    def cleanup_stale_run(self, *, station: Any = None, eval_manager: Any = None) -> Dict[str, Any]:
        """Rollback provisional fast-lane work created by an unfinished parallel tick.

        Cleanup is intentionally conservative for non-fast-lane station effects:
        it clears stale response flags and resets the saved agent index to 0 so
        the tick is retried from a clean tick boundary. Fast-lane Research
        evaluations and Archive surveys have explicit provisional metadata and
        can be removed.
        """

        state = self.load_current()
        if not state or state.get("status") != "running":
            return {"had_stale_state": False, "rolled_back_eval_ids": [], "rolled_back_survey_ids": []}

        rolled_back = self.rollback_provisional_research(eval_manager, state)
        rolled_back_surveys = self.rollback_provisional_archive_surveys(state)
        rolled_back_history_agents: List[str] = []

        if station is not None:
            rolled_back_history_agents = self.rollback_uncommitted_history(station, state)
            self._clear_waiting_flags(station, state.get("turn_order") or [])
            try:
                station.save_next_agent_index_to_config(0)
            except Exception:
                pass

        state["status"] = "recovered"
        state["recovered_timestamp"] = time.time()
        state["rolled_back_eval_ids"] = rolled_back
        state["rolled_back_survey_ids"] = rolled_back_surveys
        state["rolled_back_history_agents"] = rolled_back_history_agents
        try:
            self.save_current(state)
        finally:
            self.clear_current()

        return {
            "had_stale_state": True,
            "rolled_back_eval_ids": rolled_back,
            "rolled_back_survey_ids": rolled_back_surveys,
            "rolled_back_history_agents": rolled_back_history_agents,
        }

    def rollback_uncommitted_history(self, station: Any, state: Optional[Dict[str, Any]]) -> List[str]:
        """Remove staged LLM turns from agents whose actions never committed."""

        tick = self._coerce_int((state or {}).get("tick"))
        if tick is None:
            return []

        rolled_back_agents: List[str] = []
        for agent_name, agent_state in ((state or {}).get("agents") or {}).items():
            if not isinstance(agent_state, dict):
                continue
            if not agent_state.get("history_flushed") or agent_state.get("actions_committed"):
                continue

            history_path = os.path.join(
                constants.BASE_STATION_DATA_PATH,
                constants.AGENTS_DIR_NAME,
                str(agent_name),
                "llm_chat_history.yamll",
            )
            try:
                entries = file_io_utils.load_yaml_lines(history_path)
                if not entries:
                    continue
                kept_entries = [
                    entry for entry in entries
                    if self._coerce_int(entry.get("tick")) != tick
                ]
                if len(kept_entries) == len(entries):
                    continue
                content = ""
                for idx, entry in enumerate(kept_entries):
                    if idx:
                        content += "---\n"
                    content += yaml.safe_dump(
                        entry,
                        sort_keys=False,
                        allow_unicode=True,
                        default_flow_style=False,
                        width=1000,
                    )
                file_io_utils.save_text(content, history_path)
                rolled_back_agents.append(str(agent_name))
            except Exception as exc:
                print(f"ParallelTickState: failed to roll back uncommitted history for {agent_name}: {exc}")

        return rolled_back_agents

    def rollback_provisional_research(self, eval_manager: Any, state: Optional[Dict[str, Any]]) -> List[str]:
        if eval_manager is None:
            return []

        baseline = self._coerce_int((state or {}).get("baseline_research_max_eval_id"))
        run_id = str((state or {}).get("run_id") or "")
        explicit_ids = {
            str(entry.get("eval_id"))
            for entry in ((state or {}).get("fast_lane_evaluations") or [])
            if isinstance(entry, dict) and entry.get("eval_id") is not None
        }

        candidates: List[str] = []
        try:
            all_ids = list(eval_manager.get_all_evaluation_ids())
        except Exception:
            all_ids = list(explicit_ids)

        for eval_id in all_ids:
            eval_id_text = str(eval_id)
            eval_data = eval_manager.get_evaluation(eval_id_text)
            if not isinstance(eval_data, dict):
                continue

            parallel_meta = eval_data.get("parallel_tick") or {}
            status = str(eval_data.get("parallel_commit_status") or "").strip().lower()
            eval_run_id = str(parallel_meta.get("run_id") or "")
            numeric_id = self._coerce_int(eval_id_text)

            should_delete = status == "provisional" and (
                eval_id_text in explicit_ids
                or (run_id and eval_run_id == run_id)
                or (baseline is not None and numeric_id is not None and numeric_id > baseline)
            )
            if should_delete:
                candidates.append(eval_id_text)

        if not candidates:
            return []

        self._terminate_research_processes_for_eval_ids(candidates, eval_manager)

        rolled_back: List[str] = []
        for eval_id in candidates:
            try:
                if hasattr(eval_manager, "delete_evaluation") and eval_manager.delete_evaluation(eval_id):
                    rolled_back.append(str(eval_id))
            except Exception as exc:
                print(f"ParallelTickState: failed to delete provisional evaluation {eval_id}: {exc}")
        return rolled_back

    def rollback_provisional_archive_surveys(self, state: Optional[Dict[str, Any]]) -> List[str]:
        run_id = str((state or {}).get("run_id") or "")
        explicit_ids = [
            str(entry.get("survey_id"))
            for entry in ((state or {}).get("fast_lane_surveys") or [])
            if isinstance(entry, dict) and entry.get("survey_id") is not None
        ]
        try:
            from station.eval_archive.surveyor import rollback_provisional_archive_surveys

            return rollback_provisional_archive_surveys(run_id=run_id, explicit_ids=explicit_ids)
        except Exception as exc:
            print(f"ParallelTickState: failed to roll back provisional archive surveys: {exc}")
            return []

    def _terminate_research_processes_for_eval_ids(self, eval_ids: List[str], eval_manager: Any) -> None:
        try:
            from station.eval_research.restart_evaluations import requeue_instruction_evaluations
            from station.eval_research.runtime_paths import ensure_submit_runtime_layout

            requeue_instruction_evaluations(
                eval_ids=eval_ids,
                reason="Rolled back incomplete parallel tick provisional submission.",
                kill_running_coders=True,
                eval_manager=eval_manager,
                paths=ensure_submit_runtime_layout(),
            )
        except Exception as exc:
            print(f"ParallelTickState: failed to terminate provisional research processes: {exc}")

    def _clear_waiting_flags(self, station: Any, turn_order: Iterable[str]) -> None:
        agent_names = list(turn_order)
        try:
            active_names = station.agent_module.get_all_active_agent_names()
            for name in active_names:
                if name not in agent_names:
                    agent_names.append(name)
        except Exception:
            pass

        for agent_name in agent_names:
            try:
                agent_data = station.agent_module.load_agent_data(
                    agent_name,
                    include_ascended=True,
                    include_ended=True,
                )
                if agent_data and agent_data.get(constants.AGENT_WAITING_STATION_RESPONSE_KEY):
                    agent_data[constants.AGENT_WAITING_STATION_RESPONSE_KEY] = False
                    station.agent_module.save_agent_data(agent_name, agent_data)
            except Exception as exc:
                print(f"ParallelTickState: failed clearing waiting flag for {agent_name}: {exc}")

    def _max_numeric_eval_id(self, eval_manager: Any) -> int:
        if eval_manager is None:
            return 0
        numeric_ids: List[int] = []
        try:
            for eval_id in eval_manager.get_all_evaluation_ids():
                value = self._coerce_int(eval_id)
                if value is not None:
                    numeric_ids.append(value)
        except Exception:
            return 0
        return max(numeric_ids) if numeric_ids else 0

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None
