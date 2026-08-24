import copy
import os
import shutil
import tempfile
import threading
import unittest

from station import agent as agent_module
from station import constants
from station import file_io_utils
from station.station_runner import Orchestrator


class _FakeAgentModule:
    def __init__(self, agent_data):
        self.agent_data = copy.deepcopy(agent_data)

    def load_agent_data(self, agent_name, include_ended=False, include_ascended=False):
        if agent_name != self.agent_data.get(constants.AGENT_NAME_KEY):
            return None
        return copy.deepcopy(self.agent_data)

    def get_agent_role_definition(self, agent_data):
        return agent_data.get(constants.AGENT_ROLE_DEFINITION_KEY, "")

    latest_context_compaction_anchor_at_or_before = staticmethod(
        agent_module.latest_context_compaction_anchor_at_or_before
    )


class _FakeStation:
    def __init__(self, agent_data, current_tick):
        self.agent_module = _FakeAgentModule(agent_data)
        self.current_tick = current_tick

    def _get_current_tick(self):
        return self.current_tick


class TemporalChatBranchingTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_temporal_chat_branch_test_", dir="/tmp")
        self.old_base = constants.BASE_STATION_DATA_PATH
        constants.BASE_STATION_DATA_PATH = self.tmpdir

    def tearDown(self):
        constants.BASE_STATION_DATA_PATH = self.old_base
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _agent_data(self):
        return {
            constants.AGENT_NAME_KEY: "Agent A",
            constants.AGENT_MODEL_PROVIDER_CLASS_KEY: "openai",
            constants.AGENT_MODEL_NAME_KEY: "test-model",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_CONTEXT_COMPACTION_EVENTS_KEY: [
                {
                    constants.CONTEXT_COMPACTION_COMPACTED_AFTER_TICK_KEY: 4,
                    constants.CONTEXT_COMPACTION_ANCHOR_TICK_KEY: 5,
                    constants.CONTEXT_COMPACTION_SUMMARY_KEY: "summary after tick 4",
                },
                {
                    constants.CONTEXT_COMPACTION_COMPACTED_AFTER_TICK_KEY: 8,
                    constants.CONTEXT_COMPACTION_ANCHOR_TICK_KEY: 9,
                    constants.CONTEXT_COMPACTION_SUMMARY_KEY: "summary after tick 8",
                },
            ],
        }

    def _orchestrator(self, current_tick=10):
        orch = Orchestrator.__new__(Orchestrator)
        orch.station = _FakeStation(self._agent_data(), current_tick)
        orch._temporal_chat_lock = threading.Lock()
        return orch

    def _write_history(self):
        history_dir = os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME, "Agent A")
        file_io_utils.ensure_dir_exists(history_dir)
        history_path = os.path.join(history_dir, "llm_chat_history.yamll")
        for tick in range(1, 11):
            for role in ("user", "model"):
                file_io_utils.append_yaml_line(
                    {
                        "tick": tick,
                        "role": role,
                        "text_content": f"{role} content tick {tick}",
                    },
                    history_path,
                )

    def test_branch_after_first_compaction_loads_from_anchor(self):
        self._write_history()
        orch = self._orchestrator(current_tick=10)

        chat_state, error_msg = orch.refresh_temporal_chat("Agent A", base_tick=7)

        self.assertIsNone(error_msg)
        self.assertEqual(7, chat_state["base_tick"])
        self.assertEqual(5, chat_state["history_start_tick"])

        internal_history = file_io_utils.load_yaml_lines(orch._temporal_chat_internal_history_path("Agent A"))
        self.assertEqual([5, 5, 6, 6, 7, 7], [entry["tick"] for entry in internal_history])

        meta = file_io_utils.load_yaml(orch._temporal_chat_internal_meta_path("Agent A"))
        self.assertEqual(5, meta["model"]["history_start_tick"])
        self.assertNotIn("prune_blocks", meta["model"])

    def test_branch_before_first_compaction_keeps_early_history(self):
        self._write_history()
        orch = self._orchestrator(current_tick=10)

        chat_state, error_msg = orch.refresh_temporal_chat("Agent A", base_tick=4)

        self.assertIsNone(error_msg)
        self.assertIsNone(chat_state["history_start_tick"])
        internal_history = file_io_utils.load_yaml_lines(orch._temporal_chat_internal_history_path("Agent A"))
        self.assertEqual([1, 1, 2, 2, 3, 3, 4, 4], [entry["tick"] for entry in internal_history])

    def test_branch_after_second_compaction_loads_from_second_anchor(self):
        self._write_history()
        orch = self._orchestrator(current_tick=10)

        chat_state, error_msg = orch.refresh_temporal_chat("Agent A", base_tick=10)

        self.assertIsNone(error_msg)
        self.assertEqual(9, chat_state["history_start_tick"])
        internal_history = file_io_utils.load_yaml_lines(orch._temporal_chat_internal_history_path("Agent A"))
        self.assertEqual([9, 9, 10, 10], [entry["tick"] for entry in internal_history])

    def test_future_branch_tick_is_rejected(self):
        orch = self._orchestrator(current_tick=10)

        chat_state, error_msg = orch.refresh_temporal_chat("Agent A", base_tick=11)

        self.assertIsNone(chat_state)
        self.assertIn("future", error_msg)
        self.assertFalse(os.path.exists(orch._temporal_chat_transcript_path("Agent A")))


if __name__ == "__main__":
    unittest.main()
