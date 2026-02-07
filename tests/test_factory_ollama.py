import tempfile
import unittest

from station.llm_connectors.factory import create_llm_connector
from station.llm_connectors.ollama import OllamaConnector


class TestOllamaFactory(unittest.TestCase):
    def test_create_ollama_connector(self):
        with tempfile.TemporaryDirectory() as tmp:
            connector = create_llm_connector(
                model_class_name="ollama",
                model_name="gpt-oss:20b",
                agent_name="TestAgent",
                agent_data_path=tmp,
                custom_api_params={"base_url": "http://example.com/v1"},
            )
            self.assertIsInstance(connector, OllamaConnector)


if __name__ == "__main__":
    unittest.main()
