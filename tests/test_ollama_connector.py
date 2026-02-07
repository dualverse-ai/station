import tempfile
import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
