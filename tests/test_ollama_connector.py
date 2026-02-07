import tempfile
import unittest

from station.llm_connectors.ollama import OllamaConnector
from station.llm_connectors.openai import OpenAIConnector


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

    def test_uses_default_local_base_url(self):
        tmp = tempfile.mkdtemp()
        connector = OllamaConnector(
            model_name="gpt-oss:20b",
            agent_name="TestAgent",
            agent_data_path=tmp,
            custom_api_params={},
            max_output_tokens=128,
        )
        self.assertEqual(str(connector.client.base_url), "http://127.0.0.1:11434/v1/")

    def test_respects_custom_base_url(self):
        connector = self._make_connector()
        self.assertEqual(str(connector.client.base_url), "http://example.com/v1/")

    def test_uses_default_ollama_api_key_when_not_provided(self):
        connector = self._make_connector()
        self.assertEqual(connector.api_key, "ollama")

    def test_reuses_openai_chat_implementation(self):
        connector = self._make_connector()
        self.assertEqual(
            connector._send_message_with_chat_api.__func__,
            OpenAIConnector._send_message_with_chat_api,
        )


if __name__ == "__main__":
    unittest.main()
