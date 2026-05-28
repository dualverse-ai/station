import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from station import constants
from station import file_io_utils
from station.action_parser import ActionParser
from station.eval_theory.lean_runner import LeanRunResult, build_theory_project, run_lean_submission
from station.eval_theory.storage import TheoryStorageManager
from station.llm_connectors import create_llm_connector


class _MemoryAgentManager:
    """Lightweight agent manager for debugger interactions (no persistence)."""

    def __init__(self, base_agent_data: Dict[str, Any]):
        self.notifications: List[str] = []
        self.room_state_store: Dict[Tuple[str, str], Any] = {}
        self.agent_data = base_agent_data

    def add_pending_notification(self, agent_data: dict, message: str) -> None:
        self.notifications.append(message)
        agent_data.setdefault(constants.AGENT_NOTIFICATIONS_PENDING_KEY, []).append(message)

    def get_agent_room_state(self, agent_data: dict, room_name: str, key: str, default=None):
        return self.room_state_store.get((room_name, key), default)

    def set_agent_room_state(self, agent_data: dict, room_name: str, key: str, value):
        self.room_state_store[(room_name, key)] = value


class TheoryDebugger:
    """Runs an LLM-driven debugging loop to repair failed Theory submissions."""

    def __init__(
        self,
        *,
        station_instance,
        room_instance,
        storage: TheoryStorageManager,
        failed_entry: Dict[str, Any],
        failed_logs: str,
        llm_responder: Optional[Callable[[str], str]] = None,
    ):
        self.station = station_instance
        self.room = room_instance
        self.storage = storage
        self.failed_entry = failed_entry
        self.failed_logs = failed_logs
        self.kind = failed_entry.get("kind")
        self.author = failed_entry.get("author")
        self.payload = failed_entry.get("payload", {})
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.llm_responder = llm_responder

        # Debugger state
        self.parser = ActionParser()
        self.successful_item: Optional[Dict[str, Any]] = None
        self.successful_env_code: Optional[str] = None
        self.report: str = ""
        self.debugger_success: bool = False
        self._action_feedback: List[str] = []
        self._log_records: List[Dict[str, Any]] = []
        self.log_file_path = os.path.join(
            self.storage.base_path,
            "debugger_logs",
            f"tick_{failed_entry.get('submitted_tick', 0)}_{failed_entry.get('queue_id', 'unknown')}.yamll",
        )

        # Scratch agent data/context for room interactions
        base_agent_data = {
            constants.AGENT_NAME_KEY: self.author,
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_LINEAGE_KEY: self.failed_entry.get(constants.AGENT_LINEAGE_KEY, ""),
            constants.AGENT_GENERATION_KEY: self.failed_entry.get(constants.AGENT_GENERATION_KEY, ""),
        }
        self.agent_manager = _MemoryAgentManager(base_agent_data)
        rc = getattr(self.station, "room_context", None)
        from station.base_room import RoomContext  # local import to avoid cycle at module load

        if rc:
            self.room_context = RoomContext(
                agent_manager=self.agent_manager,
                capsule_manager=rc.capsule_manager,
                notification_manager=rc.notification_manager,
                constants_module=rc.constants_module,
                station_instance=rc.station_instance,
            )
        else:
            self.room_context = RoomContext(
                agent_manager=self.agent_manager,
                capsule_manager=None,
                notification_manager=None,
                constants_module=constants,
                station_instance=self.station,
            )

        # The debugger system prompt already embeds the Theory Room help text. Mark the
        # room help as already shown so the room observation doesn't repeat it.
        self.agent_manager.set_agent_room_state(
            self.agent_manager.agent_data,
            constants.SHORT_ROOM_NAME_THEORY,
            constants.AGENT_ROOM_STATE_FIRST_VISIT_HELP_SHOWN_KEY,
            True,
        )

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

    def run(self) -> Dict[str, Any]:
        """Main loop. Returns {'success': bool, 'report': str}."""
        if not constants.THEORY_DEBUGGER_ENABLED:
            return {"success": False, "report": "Debugger disabled."}

        connector = None if self.llm_responder else self._build_connector()
        if connector is None and not self.llm_responder:
            return {"success": False, "report": "Debugger unavailable (connector not created)."}

        system_prompt = self._build_system_prompt()
        if connector:
            connector.system_prompt = system_prompt
        self._append_log("system", system_prompt, turn=0)

        max_turns = getattr(constants, "THEORY_DEBUGGER_MAX_TURNS", 20)
        timeout_seconds = getattr(constants, "THEORY_DEBUGGER_TIMEOUT_SECONDS", 1800)
        start_ts = time.monotonic()
        for turn in range(max_turns):
            if timeout_seconds and (time.monotonic() - start_ts) >= timeout_seconds:
                break
            observation = self._build_observation()
            self._append_log("user", observation, turn=turn)
            token_info = None
            thinking_text = None
            if self.llm_responder:
                llm_text = self.llm_responder(observation)
            else:
                resp = connector.send_message(observation, current_tick=self.failed_entry.get("submitted_tick", 0))
                if isinstance(resp, tuple):
                    if len(resp) == 3:
                        llm_text, thinking_text, token_info = resp
                    elif len(resp) == 2:
                        llm_text, token_info = resp
                    else:
                        llm_text = resp[0]
                else:
                    llm_text = str(resp)
            self._append_log("assistant", llm_text, turn=turn, thinking=thinking_text, token_info=token_info)

            actions = self.parser.parse(llm_text)
            if not actions:
                self._action_feedback.append("No actions parsed. Please issue an action (e.g., submit, sandbox, finish).")
                continue

            for action in actions:
                stop = self._handle_action(action)
                if stop:
                    break
            if self.report:
                break

        if not self.report:
            self.debugger_success = False
            self.report = "Debugger ended without finish action (max turns reached)."
            self._cleanup_pending_items()
        self._flush_logs()
        return {"success": self.debugger_success, "report": self.report}

    def _build_connector(self):
        try:
            model_class = getattr(constants, "THEORY_DEBUGGER_MODEL_CLASS", "openai")
            model_name = getattr(constants, "THEORY_DEBUGGER_MODEL_NAME", "gpt-5.1")
            debug_dir = os.path.join(
                constants.BASE_STATION_DATA_PATH,
                constants.ROOMS_DIR_NAME,
                constants.SHORT_ROOM_NAME_THEORY,
                "debugger_sessions",
                str(self.failed_entry.get("queue_id", "unknown")),
            )
            os.makedirs(debug_dir, exist_ok=True)
            if not create_llm_connector:
                return None
            connector = create_llm_connector(
                model_class_name=model_class,
                model_name=model_name,
                agent_name=f"{self.author}_theory_debugger",
                agent_data_path=debug_dir,
                system_prompt=self._build_system_prompt(),
            )
            if connector and hasattr(connector, "_initialize_chat_session"):
                connector._initialize_chat_session()
            return connector
        except Exception as e:
            print(f"TheoryDebugger: failed to create connector: {e}")
            return None

    def _build_system_prompt(self) -> str:
        help_text = self.room.help_message
        failed_details = [
            "You are in Theory Room debugging mode. Fix the failed submission below without changing title or formal_statement.",
            f"Submission kind: {self.kind}",
            f"Title: {self.payload.get('title')}",
            f"Formal statement: {self.payload.get('formal_statement')}",
            f"Formal definitions: {self.payload.get('formal_definitions','')}",
            f"Tags: {self.payload.get('tags')}",
            f"Statement: {self.payload.get('statement')}",
            "Original content:",
            self.payload.get("content", ""),
            "Failure logs:",
            self.failed_logs,
            "",
            "Rules:",
            "- Do not change title or formal_statement.",
            "- You are expected to actively try to fix the proof: inspect existing lemmas/theories (read/preview), use sandbox to test Lean snippets, and iterate before giving up.",
            "- Use similar logic to fix the proof; if truly too hard after reasonable attempts, finish with success: false and summary.",
            "- Allowed actions: submit lemma/theory, sandbox, read/preview/filter/search/page/rank, revert, finish.",
            "- Use /execute_action{...} with an optional YAML block on the following lines. Example:",
            "  /execute_action{submit lemma}\n  ```yaml\n  title: ...\n  formal_statement: \"...\"\n  statement: ...\n  content: |\n    ...Lean code...\n  ```",
            "- Only one successful submission of the original kind is allowed. After one success, either revert if irrelevant or finish.",
            "- Finish action must be `/execute_action{finish}` with YAML: success: true/false, content: summary.",
            "- On finish success: true, your verified submission is kept. On success: false, all your debugging submissions are discarded.",
            "- You may take chain-of-thought steps before submitting: think out loud, sandbox, or explore existing items; you do not need to submit immediately, but you should still attempt a Lean fix.",
            "- Finish report guidance (max 800 words, written directly to the agent as 'you'):\n"
            "  - If success: briefly explain the original failure, the specific changes you made, and recommendations for the agent.\n"
            "    Example: \"I have successfully fixed the original error. Your original script used the wrong definition... I updated ... I recommend you ...\"\n"
            "  - If failure: summarize what you tried, why it failed, and suggestions (e.g., intermediate lemmas to prove first).\n"
            "- If you believe the formal statement itself is wrong (e.g., you find a counterexample or are highly confident), finish directly with that reason; give a concise argument/counterexample, do NOT propose changing the statement or adding axioms/sorry.",
        ]
        return help_text.strip() + "\n\n" + "\n".join(failed_details)

    def _build_observation(self) -> str:
        parts: List[str] = []
        if self._action_feedback:
            parts.append("Recent actions/results:\n" + "\n".join(f"- {m}" for m in self._action_feedback))
        if self.successful_item:
            parts.append(
                f"A successful {self.kind} has been created but not yet finalized. "
                "You must finish (success: true/false) or revert if it is incorrect (e.g., wrong formal statement). "
                "Further submissions of this kind are blocked."
            )
        room_output = self.room.get_room_output(self.agent_manager.agent_data, self.room_context, 0)
        parts.append("Room observation:\n" + room_output)
        self._action_feedback.clear()
        return "\n\n".join(parts)

    def _handle_action(self, action_info) -> bool:
        cmd = action_info.command
        args = action_info.args
        yaml_data = action_info.yaml_data
        if action_info.yaml_error:
            self._action_feedback.append(action_info.yaml_error)
            return False

        if cmd in {"help"}:
            self._action_feedback.append("Help not needed; refer to system prompt.")
            return False

        if cmd == "finish":
            if not yaml_data or "success" not in yaml_data or "content" not in yaml_data:
                self._action_feedback.append("Finish requires YAML with success: true/false and content.")
                return False
            success_flag = bool(yaml_data.get("success"))
            summary = str(yaml_data.get("content") or "").strip()
            if success_flag and not self.successful_item:
                self._action_feedback.append("No successful submission to finalize. Submit first or finish with success: false.")
                return False
            if success_flag:
                persisted_id = self._persist_successful_item()
                if persisted_id is None:
                    self.debugger_success = False
                    self.report = "Debugger could not finalize submission (lake build failed)."
                else:
                    self.debugger_success = True
                    self.report = summary or f"Debugger fixed submission with {self.kind} ID {persisted_id}."
            else:
                self._cleanup_pending_items()
                self.debugger_success = False
                self.report = summary or "Debugger could not fix the submission."
            return True

        if cmd == "revert":
            self._cleanup_pending_items()
            self._action_feedback.append("Reverted pending debugger submission.")
            return False

        if cmd == "sandbox":
            if not yaml_data or "content" not in yaml_data:
                self._action_feedback.append("Sandbox requires YAML with content.")
                return False
            result = run_lean_submission(
                yaml_data.get("content", ""),
                formal_statement=yaml_data.get("formal_statement"),
                formal_definitions=yaml_data.get("formal_definitions", ""),
                allow_sorry=True,
            )
            status_line = "completed" if result.success else "failed"
            self._action_feedback.append(f"Sandbox {status_line}:\n```\n{result.logs}\n```")
            return False

        if cmd == "submit":
            return self._handle_submit(args, yaml_data)

        if cmd in {"read", "preview", "filter", "unfilter", "page", "page_size", "search", "rank"}:
            return self._handle_room_action(cmd, args, yaml_data)

        self._action_feedback.append(f"Unknown or unsupported command in debugger: {cmd}")
        return False

    def _handle_room_action(self, cmd: str, args: Optional[str], yaml_data: Optional[Dict[str, Any]]) -> bool:
        actions, _ = self.room.handle_action(
            self.agent_manager.agent_data,
            cmd,
            args,
            yaml_data,
            self.room_context,
            current_tick=0,
        )
        if actions:
            self._action_feedback.extend(actions)
        if self.agent_manager.notifications:
            self._action_feedback.extend(self.agent_manager.notifications)
            self.agent_manager.notifications.clear()
        return False

    def _handle_submit(self, args: Optional[str], yaml_data: Optional[Dict[str, Any]]) -> bool:
        if self.successful_item:
            self._action_feedback.append("A successful submission already exists. Either revert it or finish.")
            return False
        target = (args or "").strip().lower()
        if target not in {"lemma", "theory"}:
            self._action_feedback.append("Please specify 'lemma' or 'theory' for submit.")
            return False
        if target != self.kind:
            self._action_feedback.append(f"Must submit the same kind as original ({self.kind}).")
            return False
        if not yaml_data:
            self._action_feedback.append("Submission requires YAML.")
            return False
        required_fields = ["title", "formal_statement", "statement", "content"]
        missing = [f for f in required_fields if not yaml_data.get(f)]
        if missing:
            self._action_feedback.append(f"Missing required fields: {', '.join(missing)}")
            return False
        if yaml_data.get("title") != self.payload.get("title") or yaml_data.get("formal_statement") != self.payload.get("formal_statement"):
            self._action_feedback.append("Title/formal_statement must match the original submission.")
            return False

        result = run_lean_submission(
            yaml_data.get("content", ""),
            formal_statement=yaml_data.get("formal_statement"),
            formal_definitions=yaml_data.get("formal_definitions", ""),
            allow_sorry=False,
        )
        if result.success:
            self.successful_item = {
                "title": yaml_data.get("title"),
                "formal_statement": yaml_data.get("formal_statement"),
                "formal_definitions": yaml_data.get("formal_definitions", ""),
                "tags": yaml_data.get("tags", []),
                "statement": yaml_data.get("statement"),
                "content": yaml_data.get("content"),
                "author": self.author,
                "submitted_tick": self.failed_entry.get("submitted_tick"),
                "logs": result.logs,
                "status": "Verified",
            }
            self.successful_env_code = "\n" + (yaml_data.get("content") or "") + "\n"
            self._action_feedback.append(
                "Submission verified. You must now finish (success: true/false). Further submissions are blocked."
            )
        else:
            self._action_feedback.append(
                f"{target.capitalize()} submission failed. Logs:\n```\n{result.logs}\n```"
            )
        return False

    def _persist_successful_item(self) -> Optional[int]:
        if not self.successful_item:
            return None
        new_entry = {
            "content": self.successful_item.get("content", ""),
            "submitted_tick": self.successful_item.get("submitted_tick"),
            "kind": self.kind,
            "id": 0,
        }
        contents = self._gather_theory_contents(new_entry)
        build_result = build_theory_project(self.repo_root, contents)
        if not build_result.success:
            self._action_feedback.append(
                f"Lake build failed after verification:\n```\n{build_result.logs}\n```"
            )
            return None
        if build_result.logs:
            self.successful_item["logs"] = f"{self.successful_item.get('logs', '')}\n\n{build_result.logs}"
        item = self.storage.add_verified_item(self.kind, self.successful_item)
        if self.successful_env_code:
            self.storage.append_env_code(self.successful_env_code)
        return item.get("id")

    def _cleanup_pending_items(self):
        self.successful_item = None
        self.successful_env_code = None

    def _append_log(self, role: str, content: str, turn: Optional[int] = None, thinking: Optional[str] = None, token_info: Optional[Dict[str, Any]] = None) -> None:
        self._log_records.append(
            {
                "role": role,
                "turn": turn if turn is not None else 0,
                "content": content,
                "thinking": thinking,
                "token_info": token_info,
            }
        )

    def _flush_logs(self) -> None:
        try:
            file_io_utils.ensure_dir_exists(os.path.dirname(self.log_file_path))
            meta = {
                "queue_id": self.failed_entry.get("queue_id"),
                "author": self.author,
                "kind": self.kind,
                "submitted_tick": self.failed_entry.get("submitted_tick"),
                "success": self.debugger_success,
                "report": self.report,
            }
            file_io_utils.append_yaml_line(meta, self.log_file_path)
            for rec in self._log_records:
                role = rec.get("role", "user")
                text = rec.get("content", "")
                thinking = rec.get("thinking")
                token_info = rec.get("token_info")
                # Map assistant to model for consistency with station logs
                if role == "assistant":
                    role = "model"
                entry = {
                    "tick": rec.get("turn", 0),
                    "role": role,
                    "parts": [{"text": text}],
                }
                if thinking:
                    entry["thinking_content"] = thinking
                if role == "model" and token_info:
                    entry["token_info"] = token_info
                file_io_utils.append_yaml_line(entry, self.log_file_path)
        except Exception as e:
            print(f"TheoryDebugger: failed to flush logs: {e}")
