import os
import tempfile
import types
import unittest
from unittest.mock import patch

from station import constants
from station import file_io_utils
from station.eval_archive.auto_evaluator import AutoArchiveEvaluator
from station.llm_connectors.base import BaseLLMConnector


class DummyConnector(BaseLLMConnector):
    def _load_history_from_file(self):
        return []

    def _append_turn_to_history_file(
        self,
        tick,
        role,
        text,
        thinking_text=None,
        token_info=None,
        api_metadata=None,
    ):
        raise NotImplementedError

    def _initialize_chat_session(self):
        return None

    def _send_message_implementation(self, user_prompt, current_tick, attempt_number=0):
        raise NotImplementedError

    def get_chat_history(self):
        return []

    def get_current_total_session_tokens(self):
        return 0


class TestReviewerSystemPrompt(unittest.TestCase):
    def test_reviewer_reply_includes_non_authoritative_disclaimer(self):
        evaluator = AutoArchiveEvaluator.__new__(AutoArchiveEvaluator)
        evaluator.station = types.SimpleNamespace(_get_current_tick=lambda: 17)

        with patch(
            "station.eval_archive.auto_evaluator.capsule_module.add_message_to_capsule",
            return_value=True,
        ) as add_reply:
            evaluator._add_reviewer_reply(
                3,
                {
                    "score": 8,
                    "comment": "Useful result.",
                    "suggestion": "Clarify the proof.",
                },
            )

        reply_data = add_reply.call_args.kwargs["message_content_from_agent"]
        content = reply_data[constants.YAML_CAPSULE_CONTENT]
        self.assertIn("should not be treated as authoritative", content)
        self.assertIn("it does not validate every claim", content)
        self.assertIn("may disagree when supported by evidence", content)

    def test_auto_archive_evaluator_keeps_explicit_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = os.path.join(tmpdir, constants.AGENTS_DIR_NAME)
            file_io_utils.ensure_dir_exists(agents_dir)
            file_io_utils.save_yaml(
                {
                    constants.AGENT_NAME_KEY: "AutoArchiveEvaluator",
                    constants.AGENT_ROLE_DEFINITION_KEY: "Reviewer role text that should be ignored.",
                },
                os.path.join(agents_dir, f"AutoArchiveEvaluator{constants.YAML_EXTENSION}"),
            )
            with open(os.path.join(tmpdir, constants.CODEX_FILENAME), "w", encoding="utf-8") as f:
                f.write("Codex text that should not be injected into the reviewer system prompt.")

            with patch.object(constants, "BASE_STATION_DATA_PATH", tmpdir):
                connector = DummyConnector(
                    model_name="dummy-model",
                    agent_name="AutoArchiveEvaluator",
                    agent_data_path=tmpdir,
                    system_prompt=constants.ARCHIVE_REVIEWER_SYSTEM_PROMPT,
                )

                loaded_prompt = connector._load_system_prompt_from_agent_data()

            self.assertEqual(loaded_prompt, constants.ARCHIVE_REVIEWER_SYSTEM_PROMPT)

    def test_normal_agent_uses_station_wrapped_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = os.path.join(tmpdir, constants.AGENTS_DIR_NAME)
            file_io_utils.ensure_dir_exists(agents_dir)
            file_io_utils.save_yaml(
                {
                    constants.AGENT_NAME_KEY: "Guest_1",
                    constants.AGENT_ROLE_DEFINITION_KEY: "A precise test role.",
                },
                os.path.join(agents_dir, f"Guest_1{constants.YAML_EXTENSION}"),
            )
            with patch.object(constants, "BASE_STATION_DATA_PATH", tmpdir):
                connector = DummyConnector(
                    model_name="dummy-model",
                    agent_name="Guest_1",
                    agent_data_path=tmpdir,
                    system_prompt="placeholder",
                )

                loaded_prompt = connector._load_system_prompt_from_agent_data()

            self.assertIn("A precise test role.", loaded_prompt)


if __name__ == "__main__":
    unittest.main()
