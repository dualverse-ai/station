import tempfile
import unittest
from types import SimpleNamespace

from station.llm_connectors.base import LLMConnectorError
from station.llm_connectors.ollama import OllamaConnector


class TestOllamaConnector(unittest.TestCase):
    def _make_connector(self) -> OllamaConnector:
        tmp = tempfile.mkdtemp()
        return OllamaConnector(
            model_name="gpt-oss:20b",
            agent_name="TestAgent",
            agent_data_path=tmp,
            custom_api_params={"base_url": "http://example.com/v1"},
            max_output_tokens=128,
        )

    def test_extracts_assistant_tool_call(self):
        connector = self._make_connector()
        tc = SimpleNamespace(
            function=SimpleNamespace(name="assistant", arguments='{"action":"goto","params":"research"}')
        )
        self.assertEqual(
            connector._tool_call_to_execute_action(tc),
            "/execute_action{goto research}",
        )

    def test_extracts_execute_action_from_container_exec(self):
        connector = self._make_connector()
        tc = SimpleNamespace(
            function=SimpleNamespace(
                name="container.exec",
                arguments='{"cmd":["bash","-lc","echo /execute_action{read 1}"]}',
            )
        )
        self.assertEqual(connector._tool_call_to_execute_action(tc), "/execute_action{read 1}")

    def test_station_action_tools_disabled_by_default(self):
        connector = self._make_connector()
        self.assertFalse(connector._enable_station_action_tools)

    def test_station_action_tools_can_be_enabled(self):
        tmp = tempfile.mkdtemp()
        connector = OllamaConnector(
            model_name="gpt-oss:20b",
            agent_name="TestAgent",
            agent_data_path=tmp,
            custom_api_params={
                "base_url": "http://example.com/v1",
                "enable_station_action_tools": True,
            },
            max_output_tokens=128,
        )
        self.assertTrue(connector._enable_station_action_tools)

    def test_failed_request_rolls_back_last_user_turn(self):
        connector = self._make_connector()
        connector.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
                )
            )
        )
        token_info = {
            "total_tokens_in_session": None,
            "last_exchange_prompt_tokens": None,
            "last_exchange_completion_tokens": None,
            "last_exchange_cached_tokens": None,
            "last_exchange_thoughts_tokens": None,
        }
        with self.assertRaises(LLMConnectorError):
            connector._send_message_with_chat_api("hello", current_tick=0, token_info=token_info)
        self.assertEqual(connector.chat_history, [])


if __name__ == "__main__":
    unittest.main()
